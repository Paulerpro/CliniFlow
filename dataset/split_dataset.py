"""Deterministic train/val/test split. test.jsonl is the held-out Pillar 4 eval set - never train on it."""
import json
import os
import random

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "colab")
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open("medsynth_1k_noisy.jsonl", encoding="utf-8") if l.strip()]
random.Random(42).shuffle(rows)
n_test, n_val = 50, 49
splits = {"test": rows[:n_test], "val": rows[n_test:n_test + n_val], "train": rows[n_test + n_val:]}
for name, part in splits.items():
    with open(os.path.join(OUT, f"{name}.jsonl"), "w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(part))
