"""Run the real gateway against recorded model replies, so everything can be tried with no GPU, no Modal
account and no API keys.

    python -m gateway.demo            # http://127.0.0.1:8080, interactive docs at /docs

The gateway code is the real one: strict JSON check, retry, ICD-10 repair, grounding flags, the same
responses. Only the model is replaced. For each sample conversation it answers with the reply the real
model gave when that conversation was recorded (see gateway/record_demo.py). Any other transcript gets a
503 that says so; GET /demo/samples lists the samples to paste into POST /v1/notes.

The agent tools and the review CLI work against it unchanged (GATEWAY_URL defaults to http://localhost:8080).
"""
import argparse
import json
from pathlib import Path

import uvicorn

from .app import create_app
from .config import Settings
from .icd10 import IcdTable
from .llm import LLMError
from .tracing import NoopTracer

SAMPLES_PATH = Path(__file__).parent / "demo_data" / "samples.json"


def _key(text):
    return " ".join(text.split())  # tolerate whitespace differences when pasting a transcript


class ReplayLLM:
    """A model stand-in that returns the recorded reply for each sample conversation."""

    def __init__(self, samples):
        self._by_transcript = {_key(s["transcript"]): s for s in samples}

    async def complete(self, messages):
        sample = self._by_transcript.get(_key(messages[1]["content"]))
        if sample is None:
            raise LLMError("demo mode only knows the sample conversations; GET /demo/samples lists them")
        # a retry adds the model's earlier replies to the conversation, so the nth reply answers the nth attempt
        attempt = sum(1 for m in messages if m["role"] == "assistant")
        replies = sample["replies"]
        return replies[min(attempt, len(replies) - 1)]


def create_demo_app(samples_path=SAMPLES_PATH):
    data = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    samples = data["samples"]
    app = create_app(
        settings=Settings(vllm_url="(demo: recorded replies, no model server)", model_name="cliniflow-soap"),
        llm=ReplayLLM(samples), table=IcdTable.load(), system_prompt="demo", tracer=NoopTracer())

    @app.get("/demo/samples")
    def demo_samples():
        return {
            "about": "Paste a transcript into POST /v1/notes. Samples marked 'recorded' are exactly what the "
                     "model returned; the one marked 'scripted' has its first reply wrapped in a code fence "
                     "(added by the recording script) to show the gateway's retry.",
            "recorded_with": data["recorded_with"],
            "samples": [{k: s[k] for k in ("id", "title", "kind", "transcript")} for s in samples],
        }

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    print(f"Demo gateway on http://{args.host}:{args.port}  (try /docs and /demo/samples)")
    uvicorn.run(create_demo_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
