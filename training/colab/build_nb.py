import json

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""# CliniFlow - Pillar 1: QLoRA fine-tune of Llama-3.2-3B-Instruct
Runtime: **Colab -> Runtime -> Change runtime type -> T4 GPU**.
Run once per config: set `CONFIG_NAME` in the config cell, then **Runtime -> Restart session and run all**. Compare configs in `results.jsonl` on Drive.""")

md("## 1. Install")
code("""
%%capture
!pip install unsloth
!pip install --upgrade --no-deps trl peft accelerate bitsandbytes
!pip install pydantic
""")

md("## 2. Mount Drive, load HF token, check data")
code("""
from google.colab import drive, userdata
drive.mount('/content/drive')
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")   # Colab Secrets (key icon) -> add HF_TOKEN, enable notebook access
DRIVE_DIR = "/content/drive/MyDrive/cliniflow"
os.makedirs(DRIVE_DIR, exist_ok=True)
# Upload train.jsonl, val.jsonl, test.jsonl to this Drive folder first
for f in ["train.jsonl", "val.jsonl"]:
    assert os.path.exists(f"{DRIVE_DIR}/{f}"), f"missing {f} in {DRIVE_DIR}"
""")

md("## 3. Config (the only cell you edit between runs)")
code("""
CONFIG_NAME = "A"   # A, B or C
CONFIGS = {
    "A": dict(r=16, alpha=32, lr=2e-4, epochs=2),   # baseline
    "B": dict(r=32, alpha=64, lr=1e-4, epochs=3),   # bigger rank, gentler LR
    "C": dict(r=16, alpha=32, lr=2e-4, epochs=3),   # same as A, more epochs
}
cfg = CONFIGS[CONFIG_NAME]
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"   # ungated mirror of meta-llama/Llama-3.2-3B-Instruct (same weights)
MAX_SEQ_LEN = 4096
SEED = 3407
HF_REPO = "YOUR_HF_USERNAME/cliniflow-soap-lora"   # <-- edit; private repo
OUT_DIR = f"{DRIVE_DIR}/adapters/config_{CONFIG_NAME}"
print(CONFIG_NAME, cfg)
""")

md("""## 4. Prompt + data formatting
`SYSTEM_PROMPT` must be **byte-identical** at serving time (FastAPI gateway, Pillar 5). It is saved next to the adapter.""")
code("""
import json
from datasets import Dataset

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Convert the doctor-patient conversation into a structured SOAP note. "
    "Respond with ONLY a JSON object with keys: subjective (string), objective (string), assessment (string), plan (string), "
    "suggested_codes (array of objects with code, system, description, confidence). "
    "Only include facts stated in the conversation. Do not invent findings."
)

def load(name):
    with open(f"{DRIVE_DIR}/{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def to_messages(r):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": r["input"]},
        {"role": "assistant", "content": json.dumps(r["output"], ensure_ascii=False)},  # compact JSON = fewer tokens
    ]}

train_raw, val_raw = load("train"), load("val")
train_ds = Dataset.from_list([to_messages(r) for r in train_raw])
val_ds   = Dataset.from_list([to_messages(r) for r in val_raw])
print(len(train_ds), len(val_ds))
""")

md("## 5. Load 4-bit base + attach LoRA")
code("""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LEN, dtype=None, load_in_4bit=True)   # dtype auto -> fp16 on T4
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

model = FastLanguageModel.get_peft_model(
    model, r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED)

def fmt(batch):
    return {"text": [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in batch["messages"]]}
train_ds = train_ds.map(fmt, batched=True)
val_ds   = val_ds.map(fmt, batched=True)

lens = [len(tokenizer(t)["input_ids"]) for t in train_ds["text"]]
print("tokens max/mean:", max(lens), sum(lens) // len(lens), "| over limit:", sum(l > MAX_SEQ_LEN for l in lens))
""")

md("## 6. Train (loss only on the assistant JSON)")
code("""
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=val_ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ_LEN, packing=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,      # effective batch 8
        per_device_eval_batch_size=1,
        num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
        lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
        optim="adamw_8bit", fp16=True, bf16=False,                          # T4 has no bf16
        logging_steps=5, eval_strategy="epoch", save_strategy="no",
        seed=SEED, report_to="none", output_dir="/content/tmp_out"))

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\\n\\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\\n\\n")

stats = trainer.train()
# Not trainer.evaluate() here: eval_strategy="epoch" already ran eval each epoch, and
# calling evaluate() again after train() crashes some transformers versions in notebooks
# (NotebookProgressCallback: "on_train_begin must be called before on_evaluate").
# The last epoch's eval metrics are already in log_history.
final_eval = {"eval_loss": [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h][-1]}
print(stats.metrics, final_eval)
""")

md("## 7. Check: does it emit schema-valid JSON on validation examples?")
code("""
from pydantic import BaseModel, ValidationError
class SuggestedCode(BaseModel):
    code: str; system: str; description: str; confidence: str
class Soap(BaseModel):
    subjective: str; objective: str; assessment: str; plan: str
    suggested_codes: list[SuggestedCode]

FastLanguageModel.for_inference(model)
N_CHECK = 20
valid = parse_ok = code_match = 0
for r in val_raw[:N_CHECK]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": r["input"]}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=ids, max_new_tokens=2000, do_sample=False, use_cache=True)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    try:
        obj = json.loads(text); parse_ok += 1
        s = Soap(**obj); valid += 1
        gold = {c["code"] for c in r["output"]["suggested_codes"]}
        code_match += bool(gold & {c.code for c in s.suggested_codes})
    except (json.JSONDecodeError, ValidationError):
        pass
print(f"json parse {parse_ok}/{N_CHECK} | schema valid {valid}/{N_CHECK} | ICD code matches gold {code_match}/{N_CHECK}")
print("last output:\\n", text[:800])
""")

md("## 8. Record results (compare configs)")
code("""
result = dict(config=CONFIG_NAME, **cfg, eval_loss=final_eval["eval_loss"], train_loss=stats.metrics["train_loss"],
              json_parse=f"{parse_ok}/{N_CHECK}", schema_valid=f"{valid}/{N_CHECK}", icd_match=f"{code_match}/{N_CHECK}",
              train_minutes=round(stats.metrics["train_runtime"] / 60, 1))
print(result)
with open(f"{DRIVE_DIR}/results.jsonl", "a") as f:
    f.write(json.dumps(result) + "\\n")
""")

md("## 9. Save adapter only (no merge)")
code("""
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)          # LoRA weights only, NOT merged
tokenizer.save_pretrained(OUT_DIR)
open(f"{OUT_DIR}/system_prompt.txt", "w").write(SYSTEM_PROMPT)
json.dump(result, open(f"{OUT_DIR}/manifest.json", "w"), indent=2)
print(os.listdir(OUT_DIR))
""")

md("## 10. Push to Hugging Face (winning config only)")
code("""
PUSH = False   # set True for the best config only
if PUSH:
    model.push_to_hub(HF_REPO, private=True, token=os.environ["HF_TOKEN"])
    tokenizer.push_to_hub(HF_REPO, private=True, token=os.environ["HF_TOKEN"])
""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 0}
import os
HERE = os.path.dirname(os.path.abspath(__file__))
json.dump(nb, open(os.path.join(HERE, "cliniflow_train.ipynb"), "w"), indent=1)
print("wrote", os.path.join(HERE, "cliniflow_train.ipynb"))
