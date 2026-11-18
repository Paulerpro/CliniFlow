import asyncio

import httpx
import pytest

from gateway.llm import LLMError, VllmClient

REPLY = {"choices": [{"message": {"content": "hello"}}]}


def client(handler):
    return VllmClient("http://model.test/", "m", transport=httpx.MockTransport(handler))


def complete(c):
    return asyncio.run(c.complete([{"role": "user", "content": "hi"}]))


def test_returns_message_content():
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), request.content
        return httpx.Response(200, json=REPLY)

    assert complete(client(handler)) == "hello"
    assert seen["url"] == "http://model.test/v1/chat/completions"
    assert b'"temperature":0' in seen["body"].replace(b" ", b"")


def test_follows_modal_303_redirect_during_cold_start():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(303, headers={"location": "/v1/chat/completions?__modal_attempt_token=SECRET"})
        return httpx.Response(200, json=REPLY)

    assert complete(client(handler)) == "hello"


def test_http_error_message_has_no_url_or_token():
    def handler(request):
        return httpx.Response(500)

    with pytest.raises(LLMError) as e:
        complete(client(handler))
    assert str(e.value) == "model server returned HTTP 500"


def test_timeout_is_reported():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(LLMError, match="timed out"):
        complete(client(handler))


def test_malformed_reply_is_an_llm_error():
    with pytest.raises(LLMError):
        complete(client(lambda request: httpx.Response(200, json={"unexpected": True})))


def test_connect_error_names_the_host_to_check():
    def handler(request):
        raise httpx.ConnectError("All connection attempts failed", request=request)

    with pytest.raises(LLMError) as e:
        complete(client(handler))
    assert "model.test" in str(e.value) and "VLLM_URL" in str(e.value)


def test_proxy_auth_headers_are_sent_when_configured():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json=REPLY)

    c = VllmClient("http://model.test", "m", transport=httpx.MockTransport(handler),
                   headers={"Modal-Key": "wk-123", "Modal-Secret": "ws-456"})
    assert complete(c) == "hello"
    assert seen["modal-key"] == "wk-123" and seen["modal-secret"] == "ws-456"


def test_no_auth_headers_by_default():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json=REPLY)

    complete(client(handler))
    assert "modal-key" not in seen and "modal-secret" not in seen


def test_rejected_credentials_do_not_leak_into_the_error():
    def handler(request):
        return httpx.Response(401)

    c = VllmClient("http://model.test", "m", transport=httpx.MockTransport(handler),
                   headers={"Modal-Key": "wk-123", "Modal-Secret": "ws-456"})
    with pytest.raises(LLMError) as e:
        complete(c)
    assert str(e.value) == "model server returned HTTP 401"
    assert "wk-123" not in str(e.value) and "ws-456" not in str(e.value)
