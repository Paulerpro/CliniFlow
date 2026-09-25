import json
from datasets import load_dataset

# Load MedSynth dataset
print("Downloading Ahmad0067/MedSynth from Hugging Face...")
dataset = load_dataset("Ahmad0067/MedSynth", split="train")

output_file = "medsynth_soap.jsonl"

# Save directly as raw JSONL pairs without extra formatting
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done! Dataset saved locally to '{output_file}'")
