from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.llm import LLMError

from .conftest import TRANSCRIPT, FakeLLM, make_reply


def client(table, settings, llm):
    return TestClient(create_app(settings=settings, llm=llm, table=table, system_prompt="SYSTEM"))


def test_health(table, settings):
    r = client(table, settings, FakeLLM([])).get("/health")
    assert r.status_code == 200 and r.json()["icd_codes"] == 98186


def test_note_endpoint(table, settings):
    r = client(table, settings, FakeLLM([make_reply(code="N83")])).post(
        "/v1/notes", json={"transcript": TRANSCRIPT})
    body = r.json()
    assert r.status_code == 200
    assert body["needs_review"] is True and body["code_review"] is True
    assert body["content_review"] is False and body["attempts"] == 1
    assert {w["category"] for w in body["warnings"]} == {"code"}
    assert body["suggested_codes"][0]["code"] == "N809"
    assert body["suggested_codes"][0]["original_code"] == "N83"
    assert {"subjective", "objective", "assessment", "plan", "warnings", "latency_ms"} <= set(body)


def test_empty_transcript_is_rejected(table, settings):
    r = client(table, settings, FakeLLM([])).post("/v1/notes", json={"transcript": ""})
    assert r.status_code == 422


def test_unusable_model_output_is_502(table, settings):
    r = client(table, settings, FakeLLM(["nope"] * 3)).post("/v1/notes", json={"transcript": TRANSCRIPT})
    assert r.status_code == 502
    assert r.json()["detail"]["attempts"] == 3


def test_unreachable_model_is_503(table, settings):
    class Down:
        async def complete(self, messages):
            raise LLMError("model server timed out")

    r = client(table, settings, Down()).post("/v1/notes", json={"transcript": TRANSCRIPT})
    assert r.status_code == 503 and "timed out" in r.json()["detail"]["message"]
