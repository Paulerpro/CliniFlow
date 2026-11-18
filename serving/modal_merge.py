"""Merges the config A LoRA adapter into the fp16 base model, on Modal, and stores the result
in a Modal Volume for serving/modal_app_merged.py.

Why on Modal: merging a 3B model needs about 13 GB of RAM in float32. Running it there needs no
GPU, no Colab session and no 6 GB upload from a laptop; only the 49 MB adapter travels.

The merge is done in float32 (the base weights are bfloat16, which float32 holds exactly) and
cast to float16 once at the end: the same fp16 base that serving/modal_app.py applies the adapter
to at runtime. Before anything is saved it checks the result two independent ways:
  1. three sampled weights are recomputed by hand as W + (alpha/r) * B @ A and compared;
  2. the logits of the adapter-applied model and the merged model agree on a sample prompt.
If either check fails the job raises and nothing is written.

Usage (Windows: set PYTHONUTF8=1 or Modal's progress output crashes on a checkmark):
    python -m modal run serving/modal_merge.py
"""
import hashlib
import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent  # serving/
ADAPTER_LOCAL = HERE.parent / "registry" / "adapters" / "config_A"
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"  # the fp16-servable base serving applies the adapter to
VOLUME_NAME = "cliniflow-merged"
OUT = "/vol/config_A_merged"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("transformers", "peft", "safetensors", "accelerate")
    .add_local_dir(str(ADAPTER_LOCAL), remote_path="/adapter", ignore=[".cache", ".gitattributes", ".git"])
)
app = modal.App("cliniflow-merge", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

SAMPLE_TRANSCRIPT = (
    "[doctor] What brings you in today?\n\n[patient] I have had lower belly pain for six months, "
    "worse during my period.\n\n[doctor] Your blood pressure is 120/80. I will start naproxen 500 mg twice a day."
)


@app.function(cpu=4, memory=24576, timeout=60 * 40, volumes={"/vol": volume})
def merge() -> dict:
    import datetime
    import shutil

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads(pathlib.Path("/adapter/adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise RuntimeError("this merge assumes plain LoRA scaling (alpha / r)")
    scaling = cfg["lora_alpha"] / cfg["r"]
    adapter_sha = hashlib.sha256(pathlib.Path("/adapter/adapter_model.safetensors").read_bytes()).hexdigest()

    tok = AutoTokenizer.from_pretrained("/adapter")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    base.eval()

    # Weights to verify by hand: first, middle and last layer, one projection each.
    adapter = load_file("/adapter/adapter_model.safetensors")
    n_layers = base.config.num_hidden_layers
    picks = [(0, "self_attn.q_proj"), (n_layers // 2, "self_attn.v_proj"), (n_layers - 1, "mlp.down_proj")]
    expected = {}
    state = base.state_dict()
    for layer, proj in picks:
        prefix = f"base_model.model.model.layers.{layer}.{proj}"
        a, b = adapter[f"{prefix}.lora_A.weight"].float(), adapter[f"{prefix}.lora_B.weight"].float()
        name = f"model.layers.{layer}.{proj}.weight"
        expected[name] = state[name].clone() + scaling * (b @ a)
    del state

    model = PeftModel.from_pretrained(base, "/adapter")
    model.eval()
    system_prompt = pathlib.Path("/adapter/system_prompt.txt").read_text(encoding="utf-8")
    ids = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": SAMPLE_TRANSCRIPT}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
    with torch.no_grad():
        logits_adapter = model(ids).logits

    merged = model.merge_and_unload()
    merged.eval()
    with torch.no_grad():
        logits_merged = merged(ids).logits

    weight_diffs = {}
    for name, want in expected.items():
        got = merged.state_dict()[name]
        weight_diffs[name] = float((got - want).abs().max())
        if weight_diffs[name] > 1e-4:
            raise RuntimeError(f"merged weight {name} differs from W + scale*B@A by {weight_diffs[name]}")
    logit_diff = float((logits_adapter - logits_merged).abs().max())
    same_tokens = bool((logits_adapter.argmax(-1) == logits_merged.argmax(-1)).all())
    if logit_diff > 1e-2 or not same_tokens:
        raise RuntimeError(f"merged model disagrees with the adapter-applied model (max logit diff {logit_diff})")

    # One rounding to fp16, then check the stored precision against the hand-computed weights.
    merged = merged.half()
    fp16_diffs = {n: float((merged.state_dict()[n].float() - w).abs().max()) for n, w in expected.items()}
    for n, w in expected.items():
        if not torch.allclose(merged.state_dict()[n].float(), w, rtol=2e-3, atol=1e-3):
            raise RuntimeError(f"fp16 rounding error too large for {n}")

    shutil.rmtree(OUT, ignore_errors=True)
    merged.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(OUT)
    shutil.copy("/adapter/system_prompt.txt", f"{OUT}/system_prompt.txt")
    manifest = {
        "base_model": BASE_MODEL, "adapter": "config_A", "adapter_sha256": adapter_sha,
        "r": cfg["r"], "lora_alpha": cfg["lora_alpha"], "scaling": scaling,
        "merge_dtype": "float32", "stored_dtype": "float16",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "checks": {"max_weight_diff_fp32": weight_diffs, "max_weight_diff_fp16": fp16_diffs,
                   "max_logit_diff_vs_adapter": logit_diff, "argmax_tokens_identical": same_tokens,
                   "sampled_weights": list(expected)},
    }
    pathlib.Path(f"{OUT}/merge_manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    sizes = {p.name: round(p.stat().st_size / 1e9, 2) for p in pathlib.Path(OUT).glob("*.safetensors")}
    return {**manifest, "files_gb": sizes}


@app.local_entrypoint()
def main():
    print(json.dumps(merge.remote(), indent=2))
