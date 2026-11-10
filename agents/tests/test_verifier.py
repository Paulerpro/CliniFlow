import asyncio
import json

import httpx
import pytest

from agents import config, mcp_server, signoff, verifier
from agents.store import Store

from .conftest import gateway_response

NOTE = {"subjective": "s", "objective": "o", "assessment": "a", "plan": "p"}
JUDGE_REPLY = {
    "unsupported_claims": [
        {"claim": "hypertension managed with Metformin", "type": "fabricated_medication", "severity": "major"},
        {"claim": "45-year-old male", "type": "fabricated_demographic", "severity": "weird"},
        {"claim": "", "type": "x", "severity": "major"},
    ],
    "missing_key_facts": ["no follow-up date", ""],
    "rationale": "One wrong pairing.",
}


def transport(reply=None, status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})
    return httpx.MockTransport(handler)


def verify(**kw):
    kw.setdefault("api_key", "test-key")
    return asyncio.run(verifier.verify("[doctor] hi", NOTE, **kw))


def test_result_is_cleaned():
    result = verify(transport=transport(JUDGE_REPLY))
    assert [c["claim"] for c in result["unsupported_claims"]] == [
        "hypertension managed with Metformin", "45-year-old male"]  # the empty claim is dropped
    assert result["unsupported_claims"][1]["severity"] == "minor"  # unknown severity is not "major"
    assert result["missing_key_facts"] == ["no follow-up date"]


def test_sends_transcript_and_note_without_the_key_in_the_body():
    seen = []
    verify(transport=transport(JUDGE_REPLY, seen=seen))
    body = seen[0]
    assert body["temperature"] == 0 and body["response_format"] == {"type": "json_object"}
    user = body["messages"][1]["content"]
    assert "TRANSCRIPT:\n[doctor] hi" in user and "NOTE:\ns\no\na\np" in user
    assert "test-key" not in json.dumps(body)


def test_reasoning_models_get_no_temperature():
    seen = []
    verify(model="gpt-5-mini", transport=transport(JUDGE_REPLY, seen=seen))
    assert "temperature" not in seen[0]


def test_off_without_a_key(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    with pytest.raises(verifier.VerifierError, match="OPENAI_API_KEY"):
        asyncio.run(verifier.verify("t", NOTE, api_key=""))


def test_http_error_message_has_no_key_or_url():
    with pytest.raises(verifier.VerifierError) as e:
        verify(transport=transport(status=401))
    assert str(e.value) == "the verifier model returned HTTP 401"


def test_garbage_reply_is_a_verifier_error():
    with pytest.raises(verifier.VerifierError):
        verify(transport=transport("not an object"))


# --- how a verification changes the sign-off rules ---

@pytest.fixture
def store():
    return Store(":memory:")


def queued(store):
    nid = store.create_draft("t", gateway_response())
    signoff.request_signoff(store, nid)
    return nid


def test_major_finding_makes_a_clean_note_need_a_reason(store):
    nid = queued(store)
    signoff.record_verification(store, nid, verifier._clean(JUDGE_REPLY), "m")
    with pytest.raises(signoff.SignoffError, match="verifier_major"):
        signoff.approve(store, nid, "Dr. Lee")
    assert signoff.approve(store, nid, "Dr. Lee", "Checked: the pairing is a typo, the plan is right.")["status"] == "approved"


def test_minor_only_finding_does_not_block_approval(store):
    nid = queued(store)
    minor = {"unsupported_claims": [{"claim": "45-year-old", "type": "fabricated_demographic",
                                     "severity": "minor"}], "missing_key_facts": [], "rationale": ""}
    signoff.record_verification(store, nid, minor, "m")
    assert signoff.approve(store, nid, "Dr. Lee")["status"] == "approved"


def test_latest_verification_wins(store):
    nid = queued(store)
    signoff.record_verification(store, nid, verifier._clean(JUDGE_REPLY), "m")
    clean = {"unsupported_claims": [], "missing_key_facts": [], "rationale": "fine"}
    signoff.record_verification(store, nid, clean, "m")
    assert store.get(nid)["verification"]["rationale"] == "fine"
    assert signoff.approve(store, nid, "Dr. Lee")["status"] == "approved"


def test_finished_notes_cannot_be_verified(store):
    nid = queued(store)
    signoff.approve(store, nid, "Dr. Lee")
    with pytest.raises(signoff.SignoffError, match="no longer be verified"):
        signoff.record_verification(store, nid, verifier._clean(JUDGE_REPLY), "m")


def test_verifications_are_audited(store):
    nid = queued(store)
    signoff.record_verification(store, nid, verifier._clean(JUDGE_REPLY), "m")
    row = next(r for r in store.audit(nid) if r["action"] == "verified")
    assert row["actor"] == "agent" and row["detail"]["major_claims"] == 1


# --- the MCP tool ---

def test_verify_note_tool(monkeypatch):
    monkeypatch.setattr(mcp_server, "_store", Store(":memory:"))
    nid = mcp_server.store().create_draft("[doctor] hi", gateway_response())

    async def fake_verify(transcript, note):
        assert transcript == "[doctor] hi" and note["subjective"] == "Lower abdominal pain."
        return verifier._clean(JUDGE_REPLY)
    monkeypatch.setattr(mcp_server.verifier, "verify", fake_verify)
    view = asyncio.run(mcp_server.verify_note(nid))
    assert view["needs_human_attention"] is True and view["status"] == "draft"
    assert view["verification"]["unsupported_claims"][0]["severity"] == "major"


def test_verify_note_tool_errors(monkeypatch):
    monkeypatch.setattr(mcp_server, "_store", Store(":memory:"))
    with pytest.raises(ValueError, match="no note"):
        asyncio.run(mcp_server.verify_note("missing"))
    nid = mcp_server.store().create_draft("t", gateway_response())

    async def off(transcript, note):
        raise verifier.VerifierError("the verifier is off: set OPENAI_API_KEY")
    monkeypatch.setattr(mcp_server.verifier, "verify", off)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        asyncio.run(mcp_server.verify_note(nid))
    assert mcp_server.store().get(nid)["verification"] is None


def test_search_icd10_tool(monkeypatch, table):
    monkeypatch.setattr(mcp_server, "_table", table)
    hits = mcp_server.search_icd10("endometriosis unspecified", limit=3)
    assert hits[0]["code"] == "N809" and len(hits) <= 3 and hits[0]["match_score"] == 100.0
    assert mcp_server.search_icd10("zzz qqq") == []
    assert len(mcp_server.search_icd10("diabetes", limit=999)) <= 20


def test_verify_note_is_traced_with_scores(monkeypatch):
    from gateway.tests.test_tracing import FakeClient, LangfuseTracer, scores

    monkeypatch.setattr(mcp_server, "_store", Store(":memory:"))
    client = FakeClient()
    monkeypatch.setattr(mcp_server, "_tracer", LangfuseTracer(client))
    nid = mcp_server.store().create_draft("[doctor] hi", gateway_response())

    async def fake_verify(transcript, note):
        return verifier._clean(JUDGE_REPLY)
    monkeypatch.setattr(mcp_server.verifier, "verify", fake_verify)
    asyncio.run(mcp_server.verify_note(nid))
    start = next(e[2] for e in client.log if e[0] == "start")
    assert start["name"] == "verify_note" and start["as_type"] == "generation"
    assert scores(client) == {"verifier_major_claims": (1.0, "NUMERIC"), "verifier_missing_facts": (1.0, "NUMERIC")}
