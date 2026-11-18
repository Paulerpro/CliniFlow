"""Client for the OpenAI-compatible vLLM server: plain HTTP, no vendor SDK."""
from typing import Protocol
from urllib.parse import urlparse

import httpx


class LLMError(Exception):
    """The model server was unreachable, timed out, or answered with an error."""


class LLMClient(Protocol):
    async def complete(self, messages: list[dict]) -> str: ...


class VllmClient:
    def __init__(self, base_url, model, max_tokens=1500, timeout_s=600.0, transport=None, headers=None):
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout_s
        self._transport = transport  # tests inject a fake transport
        self._headers = headers or {}  # e.g. Modal proxy-auth token headers

    async def complete(self, messages):
        body = {"model": self._model, "messages": messages,
                "max_tokens": self._max_tokens, "temperature": 0}  # deterministic, like the evals
        # Modal answers a request that outlives its gateway timeout (a cold start) with
        # "303 See Other" to a URL that returns the result, so redirects must be followed.
        # Error text below deliberately omits URLs: the redirect URL carries a token.
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True,
                                         transport=self._transport, headers=self._headers) as client:
                resp = await client.post(self._url, json=body)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as e:
            raise LLMError(f"model server timed out after {self._timeout}s") from e
        except httpx.ConnectError as e:
            # Host and port only: the path and query of a redirect URL can carry a token.
            raise LLMError(f"could not connect to {urlparse(self._url).netloc} ({e or 'connection refused'}); "
                           "check VLLM_URL") from e
        except httpx.HTTPStatusError as e:
            raise LLMError(f"model server returned HTTP {e.response.status_code}") from e
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"model server error: {type(e).__name__}") from e
