"""Record the demo fixtures: send the sample conversations in demo_data/transcripts/ to a running model
and save its raw replies to demo_data/samples.json, which `python -m gateway.demo` replays.

The conversations are original text written for this project (not MedSynth data), so the recordings can
be shared. Everything marked "recorded" is exactly what the model returned. The one sample marked
"scripted" has its first reply wrapped in a markdown code fence, so the gateway's retry can be shown;
that fence is added by this script, not produced by the model, and the sample says so.

    python -m gateway.record_demo                       # uses VLLM_URL and MODEL_NAME from .env
    python -m gateway.record_demo --url https://... --model cliniflow-soap
"""
import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from .config import Settings, proxy_auth_headers
from .llm import LLMError, VllmClient

DATA = Path(__file__).parent / "demo_data"
TITLES = {
    "01_diabetes_kidney": "Type 2 diabetes follow-up with kidney disease",
    "02_low_back_strain": "Low back pain after lifting",
    "03_child_asthma": "Child with an asthma flare-up",
    "04_urinary_infection": "Bladder infection",
    "05_sore_throat": "Sore throat (scripted retry demonstration)",
    "06_pelvic_pain": "Pelvic pain, suspected endometriosis",
}
SCRIPTED = {"05_sore_throat"}  # first reply is wrapped in a code fence to demonstrate the retry


async def record(args):
    settings = Settings()
    client = VllmClient(args.url or settings.vllm_url, args.model or settings.model_name, settings.max_tokens,
                        settings.request_timeout_s, headers=proxy_auth_headers())
    prompt = settings.system_prompt_path.read_text(encoding="utf-8")
    paths = sorted((DATA / "transcripts").glob("*.txt"))

    async def one(path):
        transcript = path.read_text(encoding="utf-8").strip()
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": transcript}]
        for attempt in range(3):  # a cold container can drop a connection while it loads; asking again is safe
            try:
                return path.stem, transcript, (await client.complete(messages)).strip()
            except LLMError as e:
                if attempt == 2:
                    raise
                print(f"{path.stem}: {e}; trying again ({attempt + 1}/2)", flush=True)
                await asyncio.sleep(5)

    # the first request alone wakes the container; the rest then run together
    results = [await one(paths[0])]
    results += await asyncio.gather(*(one(p) for p in paths[1:]))
    samples = []
    for sid, transcript, reply in results:
        scripted = sid in SCRIPTED
        samples.append({
            "id": sid,
            "title": TITLES[sid],
            "kind": "scripted" if scripted else "recorded",
            "transcript": transcript,
            # a retry sees the model's second reply: the fenced first reply is scripted, the second is real
            "replies": ["```json\n" + reply + "\n```", reply] if scripted else [reply],
        })
    out = {"recorded_with": {"model": args.model or settings.model_name,
                             "recorded_utc": datetime.now(UTC).isoformat(timespec="seconds")},
           "samples": samples}
    (DATA / "samples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"recorded {len(samples)} samples to {DATA / 'samples.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--model")
    asyncio.run(record(ap.parse_args()))


if __name__ == "__main__":
    main()
