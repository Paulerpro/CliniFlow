"""Calls the Pillar 5 gateway to turn a transcript into a draft note."""
import httpx

from . import config


class GatewayError(Exception):
    """The gateway was unreachable or refused the request."""


async def request_note(transcript, url=None, timeout=None, transport=None):
    url = (url or config.GATEWAY_URL).rstrip("/") + "/v1/notes"
    try:
        async with httpx.AsyncClient(timeout=timeout or config.GATEWAY_TIMEOUT_S, transport=transport) as c:
            resp = await c.post(url, json={"transcript": transcript})
    except httpx.TimeoutException as e:
        raise GatewayError("the gateway timed out (a cold model container can take several minutes)") from e
    except httpx.ConnectError as e:
        raise GatewayError(f"could not connect to the gateway at {url.rsplit('/v1', 1)[0]}; "
                           "is it running? (set GATEWAY_URL)") from e
    except httpx.HTTPError as e:
        raise GatewayError(f"gateway error: {type(e).__name__}") from e
    if resp.status_code != 200:
        detail = resp.json().get("detail") if resp.headers.get("content-type", "").startswith("application/json") else None
        message = detail.get("message") if isinstance(detail, dict) else detail
        raise GatewayError(f"the gateway returned HTTP {resp.status_code}: {message or 'no detail'}")
    return resp.json()
