import json
import random

# Set a seed so the randomness is reproducible
random.seed(42)

input_file = "medsynth_soap.jsonl"
output_file = "medsynth_1k_soap.jsonl"

print(f"Reading data from {input_file}...")

# Load all rows from your existing local file
with open(input_file, "r", encoding="utf-8") as f:
    all_samples = [json.loads(line) for f in f if (line := f.strip())]

print(f"Found {len(all_samples)} total rows.")

# Extract exactly 1,000 random samples
if len(all_samples) >= 1000:
    small_dataset = random.sample(all_samples, 1000)
else:
    print("⚠️ Your input file has less than 1,000 samples. Saving all available samples.")
    small_dataset = all_samples

# Save the 1K subset to the new file
with open(output_file, "w", encoding="utf-8") as f:
    for item in small_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done! Successfully cut and saved {len(small_dataset)} samples to '{output_file}'")
