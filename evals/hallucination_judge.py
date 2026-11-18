"""LLM-as-judge hallucination scoring for SOAP notes (OpenAI API).

For each (transcript, note) pair the judge lists claims in the note that the
transcript does not support, and lists key transcript facts the note leaves out. Gold notes are judged too, as a control: the gold notes are
elaborated beyond their transcripts, so models are compared to that baseline.
The judge is not told which model wrote a note.

The API key is read from OPENAI_API_KEY or from .env (never printed).
Results are written after every call, so an interrupted run resumes.

Usage:
    python evals/hallucination_judge.py --model <openai-model> --dry-run
    python evals/hallucination_judge.py --model <openai-model> \
        --test training/colab/test.jsonl --base evals/results_base.json \
        --finetuned evals/results_finetuned.json --out evals/hallucination_judge.json
"""
import argparse
import json
import os
import statistics as st
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hallucination_rules import note_text  # noqa: E402
from run_eval import parse_lenient  # noqa: E402

CLAIM_TYPES = [
    "fabricated_finding",     # exam finding, vital, lab or imaging result not in the transcript
    "fabricated_medication",  # drug, dose, route or frequency not in the transcript
    "fabricated_history",     # history, allergy, social or family detail not in the transcript
    "fabricated_demographic", # age, sex, ethnicity, name, occupation or residence not in the transcript
    "fabricated_assessment",  # diagnosis, differential, severity or clinical impression not stated
    "fabricated_plan_item",   # test, referral, follow-up or instruction not discussed
    "contradiction",          # note says the opposite of the transcript
]

SYSTEM_PROMPT = f"""You audit AI-written clinical notes against the doctor-patient transcript they came from.

You receive a TRANSCRIPT and a NOTE (SOAP format). Judge ONLY whether the NOTE is supported by the TRANSCRIPT.

List every specific claim in the NOTE that the TRANSCRIPT does not support. Use these types:
{json.dumps(CLAIM_TYPES)}

Rules:
- A claim is supported if the transcript states it or it follows directly from what is stated. Rewording, abbreviating and reorganising are fine.
- Generic boilerplate that adds no specific fact (e.g. "follow up as needed") is NOT a claim.
- A specific value, drug, dose, test, referral, name, date or finding that never appears in the transcript IS an unsupported claim.
- If the note contradicts the transcript, report it as "contradiction".
- Severity: "major" if a clinician acting on it could be misled (wrong/invented drug, dose, vital, diagnosis-relevant finding, contradiction); "minor" otherwise.
- Do not judge writing style, formatting, or ICD codes.

Also list the clinically important facts stated in the TRANSCRIPT that the NOTE leaves out or gets wrong (symptoms, durations, vitals, medications with doses, tests ordered, follow-up). Be strict: list each missing fact separately, at most 8, and use an empty list only if nothing important is missing.

Reply with JSON only:
{{"unsupported_claims": [{{"claim": "<short quote or paraphrase from the note>", "type": "<type>", "severity": "major|minor"}}], "missing_key_facts": ["<fact>"], "rationale": "<one sentence>"}}"""


def get_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                name, _, value = line.strip().partition("=")
                if name.strip() == "OPENAI_API_KEY":
                    return value.strip().strip('"').strip("'")
    return None


def build_messages(transcript, note):
    user = f"TRANSCRIPT:\n{transcript}\n\nNOTE:\n{note_text(note)}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def call_judge(key, model, messages, retries=3):
    body = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):
        body["temperature"] = 0  # reasoning-family models only accept the default
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return json.loads(payload["choices"][0]["message"]["content"]), payload.get("usage", {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


def collect_notes(test_path, base_path, ft_path):
    rows = [json.loads(l) for l in open(test_path, encoding="utf-8") if l.strip()]
    sources = {"gold": {}, "base": {}, "finetuned": {}}
    for i, r in enumerate(rows):
        g = r["output"]
        sources["gold"][i] = json.loads(g) if isinstance(g, str) else g
    for label, path in (("base", base_path), ("finetuned", ft_path)):
        for r in json.load(open(path, encoding="utf-8"))["results"]:
            obj = parse_lenient(r["output"]) if r["output"] else None
            if obj is not None:
                sources[label][r["index"] - 1] = obj
    return rows, sources


def summarise(per_note):
    n = len(per_note)
    claims = [c for v in per_note for c in v.get("unsupported_claims", [])]
    missing = [len(v.get("missing_key_facts", [])) for v in per_note]
    by_type = {}
    for c in claims:
        by_type[c.get("type", "other")] = by_type.get(c.get("type", "other"), 0) + 1
    return {
        "notes_judged": n,
        "mean_unsupported_claims_per_note": round(len(claims) / n, 2),
        "mean_major_claims_per_note": round(sum(c.get("severity") == "major" for c in claims) / n, 2),
        "notes_with_any_major_claim":
            sum(any(c.get("severity") == "major" for c in v.get("unsupported_claims", [])) for v in per_note),
        "notes_with_no_unsupported_claim": sum(not v.get("unsupported_claims") for v in per_note),
        "mean_missing_key_facts_per_note": round(st.mean(missing), 2),
        "claims_by_type": by_type,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default="training/colab/test.jsonl")
    ap.add_argument("--base", default="evals/results_base.json")
    ap.add_argument("--finetuned", default="evals/results_finetuned.json")
    ap.add_argument("--out", default="evals/hallucination_judge.json")
    ap.add_argument("--limit", type=int, default=None, help="notes per source (for a cheap trial)")
    ap.add_argument("--only", nargs="+", choices=["gold", "base", "finetuned"],
                    help="judge only these sources (e.g. a new results file passed as --finetuned)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    args = ap.parse_args()

    rows, sources = collect_notes(args.test, args.base, args.finetuned)
    jobs = [(label, i) for label, notes in sources.items() for i in sorted(notes)
            if (args.limit is None or i < args.limit) and (not args.only or label in args.only)]

    if args.dry_run:
        chars = sum(len(rows[i]["input"]) + len(note_text(sources[l][i])) + len(SYSTEM_PROMPT) for l, i in jobs)
        print(f"{len(jobs)} judge calls; about {chars // 4:,} input tokens (rough, 4 chars/token)")
        return

    key = get_api_key()
    if not key:
        sys.exit("OPENAI_API_KEY not found in the environment or .env")

    state = {"model": args.model, "judgments": {}, "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    if os.path.exists(args.out):
        prev = json.load(open(args.out, encoding="utf-8"))
        if prev.get("model") == args.model:
            state = prev  # resume
    done = state["judgments"]

    for n, (label, i) in enumerate(jobs, 1):
        k = f"{label}:{i + 1}"
        if k in done:
            continue
        try:
            verdict, usage = call_judge(key, args.model, build_messages(rows[i]["input"], sources[label][i]))
        except Exception as e:
            print(f"[{n}/{len(jobs)}] {k} FAILED: {e}")
            continue
        done[k] = verdict
        for u in ("prompt_tokens", "completion_tokens"):
            state["usage"][u] += usage.get(u, 0)
        print(f"[{n}/{len(jobs)}] {k} unsupported={len(verdict.get('unsupported_claims', []))} "
              f"missing={len(verdict.get('missing_key_facts', []))}")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    state["summary"] = {
        label: summarise([v for k, v in done.items() if k.startswith(label + ":")])
        for label in sources if any(k.startswith(label + ":") for k in done)
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"\ntokens used: {state['usage']}")
    for label, s in state["summary"].items():
        print(label, s)


if __name__ == "__main__":
    main()
