"""Check the data files against data_manifest.json (row counts and SHA-256).

The MedSynth-derived data is not stored in this repository (see NOTICE), so after regenerating it
with the scripts in this folder, run this to confirm you have exactly the data the results in
evals/ were produced from.

    python dataset/verify_data.py            # check the files under the repo root
    python dataset/verify_data.py --root DIR # check another folder with the same layout
    python dataset/verify_data.py --write    # (author only) rebuild the manifest from the current files
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "data_manifest.json"
FILES = [
    "dataset/raw/medsynth_soap.jsonl",
    "dataset/raw/medsynth_1k_soap.jsonl",
    "dataset/medsynth_1k_noisy.jsonl",
    "training/colab/train.jsonl",
    "training/colab/val.jsonl",
    "training/colab/test.jsonl",
]
# The scripts write everything into dataset/ (they use the working directory), while the
# author's copy keeps the two raw files in dataset/raw/. Either place is accepted.
ALTERNATE = {
    "dataset/raw/medsynth_soap.jsonl": "dataset/medsynth_soap.jsonl",
    "dataset/raw/medsynth_1k_soap.jsonl": "dataset/medsynth_1k_soap.jsonl",
}


def locate(root, name):
    path = root / name
    if not path.exists() and name in ALTERNATE:
        path = root / ALTERNATE[name]
    return path


def describe(path):
    # Windows writes CRLF line endings and Linux/macOS write LF: hash the content, not the platform.
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return {"rows": sum(1 for line in data.decode("utf-8").splitlines() if line.strip()),
            "sha256": hashlib.sha256(data).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parent))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    if args.write:
        manifest = {name: describe(locate(root, name)) for name in FILES}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST.name} for {len(manifest)} files")
        return

    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = 0
    for name, want in expected.items():
        path = locate(root, name)
        if not path.exists():
            print(f"MISSING    {name}")
            bad += 1
            continue
        got = describe(path)
        ok = got == want
        bad += not ok
        print(f"{'OK       ' if ok else 'MISMATCH '}  {name}  ({got['rows']} rows)")
    print("all files match the manifest" if not bad else f"{bad} file(s) do not match")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
