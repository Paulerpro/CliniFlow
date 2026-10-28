import asyncio

import pytest

from agents import mcp_server
from agents.store import Store

from .conftest import FLAG_WARNING, gateway_response


@pytest.fixture(autouse=True)
def fresh_store(monkeypatch):
    monkeypatch.setattr(mcp_server, "_store", Store(":memory:"))


def fake_gateway(monkeypatch, response):
    async def request_note(transcript):
        return response
    monkeypatch.setattr(mcp_server, "request_note", request_note)


def test_exactly_three_tools_and_no_way_to_approve():
    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert names == {"draft_note", "get_note", "request_signoff"}


def test_draft_returns_flags_and_never_approved_status(monkeypatch):
    fake_gateway(monkeypatch, gateway_response(code_review=True, warnings=[FLAG_WARNING]))
    view = asyncio.run(mcp_server.draft_note("[doctor] hi"))
    assert view["status"] == "draft" and view["needs_human_attention"] is True
    assert view["warnings"][0]["type"] == "code_repaired"
    assert view["note"]["subjective"] == "Lower abdominal pain."


def test_agent_workflow_stops_at_awaiting_review(monkeypatch):
    fake_gateway(monkeypatch, gateway_response())
    nid = asyncio.run(mcp_server.draft_note("t"))["note_id"]
    assert mcp_server.get_note(nid)["status"] == "draft"
    view = mcp_server.request_signoff(nid)
    assert view["status"] == "awaiting_review" and view["reviewed_by"] is None
    with pytest.raises(ValueError, match="only a draft"):
        mcp_server.request_signoff(nid)


def test_unknown_note_is_a_tool_error():
    with pytest.raises(ValueError, match="no note"):
        mcp_server.get_note("missing")


def test_gateway_failure_becomes_a_tool_error(monkeypatch):
    from agents.gateway_client import GatewayError

    async def boom(transcript):
        raise GatewayError("could not connect to the gateway")
    monkeypatch.setattr(mcp_server, "request_note", boom)
    with pytest.raises(ValueError, match="could not connect"):
        asyncio.run(mcp_server.draft_note("t"))
    assert mcp_server.store().list() == []  # nothing is stored for a failed draft
