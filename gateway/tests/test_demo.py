import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from gateway.demo import SAMPLES_PATH, ReplayLLM, create_demo_app
from gateway.llm import LLMError

SAMPLES = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))["samples"]


@pytest.fixture(scope="module")
def client():
    return TestClient(create_demo_app())


def note(client, transcript):
    return client.post("/v1/notes", json={"transcript": transcript})


def test_every_sample_goes_through_the_real_pipeline(client):
    for s in SAMPLES:
        r = note(client, s["transcript"])
        assert r.status_code == 200, (s["id"], r.json())
        assert r.json()["attempts"] == (2 if s["kind"] == "scripted" else 1)


def test_exactly_one_sample_is_scripted_and_it_is_labelled():
    assert [s["kind"] for s in SAMPLES].count("scripted") == 1
    scripted = next(s for s in SAMPLES if s["kind"] == "scripted")
    assert scripted["replies"][0].startswith("```json") and not scripted["replies"][1].startswith("```")


def test_the_samples_show_what_the_readme_promises(client):
    bodies = [note(client, s["transcript"]).json() for s in SAMPLES]
    statuses = {c["status"] for b in bodies for c in b["suggested_codes"]}
    assert "repaired" in statuses                      # a code replaced from its description
    assert {"unresolved_mismatch", "unresolved_not_found"} & statuses  # a code the gateway could not fix, kept and flagged
    assert any(b["content_review"] for b in bodies)    # something in a note the conversation never said
    assert any(b["code_review"] for b in bodies)


def test_an_unknown_transcript_gets_a_helpful_503(client):
    r = note(client, "[doctor] Hello.\n\n[patient] Hi.")
    assert r.status_code == 503
    assert "sample conversations" in r.json()["detail"]["message"]


def test_whitespace_differences_do_not_matter(client):
    s = SAMPLES[0]
    sloppy = "\n\n".join(line.strip() for line in s["transcript"].splitlines() if line.strip()) + "\n   "
    assert note(client, sloppy).status_code == 200


def test_samples_endpoint_lists_conversations_but_not_replies(client):
    body = client.get("/demo/samples").json()
    assert len(body["samples"]) == len(SAMPLES)
    assert all(set(s) == {"id", "title", "kind", "transcript"} for s in body["samples"])
    assert body["recorded_with"]["model"]


def test_health_says_it_is_a_demo(client):
    body = client.get("/health").json()
    assert "demo" in body["vllm_url"] and body["tracing"] is False


def test_each_attempt_gets_the_matching_recorded_reply():
    scripted = next(s for s in SAMPLES if s["kind"] == "scripted")
    llm = ReplayLLM(SAMPLES)
    first = [{"role": "system", "content": "x"}, {"role": "user", "content": scripted["transcript"]}]
    retry = first + [{"role": "assistant", "content": "bad"}, {"role": "user", "content": "again"}]
    assert asyncio.run(llm.complete(first)) == scripted["replies"][0]
    assert asyncio.run(llm.complete(retry)) == scripted["replies"][1]
    with pytest.raises(LLMError):
        asyncio.run(llm.complete([{"role": "system", "content": "x"}, {"role": "user", "content": "unknown"}]))
