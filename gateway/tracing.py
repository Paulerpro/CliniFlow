"""Optional Langfuse tracing. Does nothing unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.

The pipeline talks to the small `Tracer` interface below, never to the Langfuse SDK, so:
  - without keys (or without the SDK installed) every call is a no-op,
  - a tracing failure is logged and swallowed: it can never break a request,
  - note text stays out of traces unless TRACE_CONTENT=true (the default sends lengths and
    flags only). Turn content on for synthetic data; leave it off for anything real.
"""
import logging
import os
from contextlib import contextmanager

log = logging.getLogger("cliniflow.tracing")


def describe(value):
    """What goes into a trace when content is off: sizes, never text."""
    if value is None:
        return None
    if isinstance(value, str):
        return f"<{len(value)} chars, content hidden>"
    if isinstance(value, list) and all(isinstance(m, dict) for m in value):
        return f"<{len(value)} messages, {sum(len(str(m.get('content', ''))) for m in value)} chars, content hidden>"
    return "<content hidden>"


class NoopSpan:
    def update(self, **kwargs):
        pass

    def score(self, name, value, comment=None):
        pass


class NoopTracer:
    enabled = False

    @contextmanager
    def span(self, name, input=None, as_type="span", **kwargs):
        yield NoopSpan()

    def flush(self):
        pass


class _LangfuseSpan:
    def __init__(self, observation, tracer):
        self._obs, self._tracer = observation, tracer

    def update(self, output=None, **kwargs):
        if output is not None:
            kwargs["output"] = output if self._tracer.content else describe(output)
        self._tracer.safely(self._obs.update, **kwargs)

    def score(self, name, value, comment=None):
        """A score on the whole trace: booleans become 0/1 BOOLEAN scores, numbers NUMERIC."""
        if isinstance(value, bool):
            kw = {"value": float(value), "data_type": "BOOLEAN"}
        else:
            kw = {"value": float(value), "data_type": "NUMERIC"}
        self._tracer.safely(self._obs.score_trace, name=name, comment=comment, **kw)


class LangfuseTracer:
    enabled = True

    def __init__(self, client, trace_content=False):
        self._client = client
        self.content = trace_content

    def safely(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            log.warning("tracing call failed and was ignored", exc_info=True)

    @contextmanager
    def span(self, name, input=None, as_type="span", **kwargs):
        try:
            cm = self._client.start_as_current_observation(
                name=name, as_type=as_type, input=input if self.content else describe(input), **kwargs)
            observation = cm.__enter__()
        except Exception:
            log.warning("could not start a trace span; continuing without it", exc_info=True)
            yield NoopSpan()
            return
        try:
            yield _LangfuseSpan(observation, self)
        except BaseException as e:
            self.safely(cm.__exit__, type(e), e, e.__traceback__)
            raise
        else:
            self.safely(cm.__exit__, None, None, None)

    def flush(self):
        self.safely(self._client.flush)


def make_tracer(env=None, client_cls=None):
    """A LangfuseTracer if keys are set and the SDK imports, else a NoopTracer."""
    env = os.environ if env is None else env
    if not (env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY")):
        return NoopTracer()
    try:
        if client_cls is None:
            from langfuse import Langfuse as client_cls
        host = env.get("LANGFUSE_HOST") or env.get("LANGFUSE_BASE_URL")
        client = client_cls(public_key=env["LANGFUSE_PUBLIC_KEY"], secret_key=env["LANGFUSE_SECRET_KEY"],
                            **({"host": host} if host else {}))
    except Exception:
        log.warning("Langfuse keys are set but the client could not start; tracing is off", exc_info=True)
        return NoopTracer()
    content = env.get("TRACE_CONTENT", "false").strip().lower() in ("1", "true", "yes")
    return LangfuseTracer(client, trace_content=content)
