import asyncio

import httpx
import pytest

from agents.gateway_client import GatewayError, request_note
from gateway.app import create_app
from gateway.config import Settings
from gateway.tests.conftest import TRANSCRIPT, FakeLLM, make_reply


def real_gateway(table, llm):
    app = create_app(settings=Settings(max_retries=0), llm=llm, table=table, system_prompt="S")
    return httpx.ASGITransport(app=app)


def test_talks_to_the_real_gateway_app(table):
    transport = real_gateway(table, FakeLLM([make_reply(code="N83")]))
    body = asyncio.run(request_note(TRANSCRIPT, url="http://gw", transport=transport))
    assert body["code_review"] is True and body["suggested_codes"][0]["code"] == "N809"


def test_gateway_error_message_is_passed_on(table):
    transport = real_gateway(table, FakeLLM(["not json"]))
    with pytest.raises(GatewayError, match="502.*did not return a usable note"):
        asyncio.run(request_note(TRANSCRIPT, url="http://gw", transport=transport))


def test_unreachable_gateway_names_the_address():
    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(GatewayError, match="localhost:8080.*GATEWAY_URL"):
        asyncio.run(request_note("t", url="http://localhost:8080", transport=httpx.MockTransport(handler)))
