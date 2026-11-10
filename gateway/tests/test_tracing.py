import asyncio
from contextlib import contextmanager

import pytest

from gateway.pipeline import PipelineError, generate_note
from gateway.tracing import LangfuseTracer, NoopTracer, describe, make_tracer

from .conftest import TRANSCRIPT, FakeLLM, make_reply


class FakeObservation:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def update(self, **kwargs):
        self.log.append(("update", self.name, kwargs))

    def score_trace(self, **kwargs):
        self.log.append(("score", self.name, kwargs))


class FakeClient:
    """Stands in for the Langfuse SDK client and records what it was asked to do."""

    def __init__(self, fail_on_start=False, **kwargs):
        self.log, self.kwargs, self.fail_on_start = [], kwargs, fail_on_start
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        if self.fail_on_start:
            raise RuntimeError("langfuse is down")
        self.log.append(("start", kwargs["name"], kwargs))
        try:
            yield FakeObservation(self.log, kwargs["name"])
        except BaseException as e:
            self.log.append(("error", kwargs["name"], type(e).__name__))
            raise
        finally:
            self.log.append(("end", kwargs["name"], None))

    def flush(self):
        self.flushed = True


def starts(client):
    return [(e[1], e[2]["as_type"]) for e in client.log if e[0] == "start"]


def scores(client):
    return {e[2]["name"]: (e[2]["value"], e[2]["data_type"]) for e in client.log if e[0] == "score"}


def run(llm, table, settings, tracer):
    return asyncio.run(generate_note(TRANSCRIPT, llm, table, "SYSTEM", settings, tracer))


# --- the tracer itself ---

def test_no_keys_means_no_tracing():
    assert isinstance(make_tracer(env={}), NoopTracer)
    assert isinstance(make_tracer(env={"LANGFUSE_PUBLIC_KEY": "pk"}), NoopTracer)  # secret missing


def test_keys_start_a_langfuse_tracer_with_host_and_content_switch():
    env = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_HOST": "https://eu.example",
           "TRACE_CONTENT": "true"}
    tracer = make_tracer(env=env, client_cls=FakeClient)
    assert tracer.enabled and tracer.content is True
    assert tracer._client.kwargs == {"public_key": "pk", "secret_key": "sk", "host": "https://eu.example"}
    assert make_tracer(env={**env, "TRACE_CONTENT": ""}, client_cls=FakeClient).content is False


def test_a_client_that_fails_to_start_falls_back_to_no_tracing():
    class Broken:
        def __init__(self, **kwargs):
            raise RuntimeError("bad key format")

    tracer = make_tracer(env={"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}, client_cls=Broken)
    assert isinstance(tracer, NoopTracer)


def test_content_is_hidden_unless_switched_on():
    client = FakeClient()
    tracer = LangfuseTracer(client, trace_content=False)
    with tracer.span("s", input="secret transcript text") as span:
        span.update(output="secret note text", metadata={"attempts": 1})
    start = client.log[0][2]
    update = next(e[2] for e in client.log if e[0] == "update")
    assert "secret" not in str(client.log)
    assert start["input"] == "<22 chars, content hidden>" and update["output"] == "<16 chars, content hidden>"
    assert update["metadata"] == {"attempts": 1}  # flags and counts still go through

    shown = FakeClient()
    with LangfuseTracer(shown, trace_content=True).span("s", input="secret transcript text"):
        pass
    assert shown.log[0][2]["input"] == "secret transcript text"


def test_describe_never_returns_text():
    assert "hello" not in describe("hello")
    assert "hello" not in describe([{"role": "user", "content": "hello"}])
    assert describe(None) is None


def test_exceptions_in_the_body_propagate_and_close_the_span():
    client = FakeClient()
    with pytest.raises(ValueError, match="boom"), LangfuseTracer(client).span("s"):
        raise ValueError("boom")
    assert ("error", "s", "ValueError") in client.log and ("end", "s", None) in client.log


def test_tracing_failures_never_break_the_request():
    class Exploding(FakeObservation):
        def update(self, **kwargs):
            raise RuntimeError("network down")

        def score_trace(self, **kwargs):
            raise RuntimeError("network down")

    class Client(FakeClient):
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            yield Exploding(self.log, kwargs["name"])

        def flush(self):
            raise RuntimeError("network down")

    tracer = LangfuseTracer(Client())
    with tracer.span("s") as span:
        span.update(output="x")
        span.score("n", 1)
    tracer.flush()  # none of the above raised

    down = LangfuseTracer(FakeClient(fail_on_start=True))
    with down.span("s") as span:  # the span could not start; the body still runs
        span.update(output="x")


# --- what the pipeline reports ---

def test_pipeline_emits_the_expected_spans_and_scores(table, settings):
    client = FakeClient()
    result = run(FakeLLM([make_reply(code="N83")]), table, settings, LangfuseTracer(client))
    assert starts(client) == [("generate_note", "span"), ("llm_call", "generation"),
                              ("repair_codes", "span"), ("grounding", "span")]
    assert scores(client) == {
        "needs_review": (1.0, "BOOLEAN"), "code_review": (1.0, "BOOLEAN"), "content_review": (0.0, "BOOLEAN"),
        "attempts": (1.0, "NUMERIC"), "codes_repaired": (1.0, "NUMERIC")}
    llm_call = next(e[2] for e in client.log if e[0] == "start" and e[1] == "llm_call")
    assert llm_call["model"] == settings.model_name
    assert result.code_review


def test_retries_show_up_as_separate_model_calls_with_the_problem(table, settings):
    client = FakeClient()
    run(FakeLLM(["```json\n" + make_reply() + "\n```", make_reply()]), table, settings, LangfuseTracer(client))
    calls = [e for e in client.log if e[0] == "start" and e[1] == "llm_call"]
    assert [c[2]["metadata"]["attempt"] for c in calls] == [1, 2]
    first_update = next(e[2] for e in client.log if e[0] == "update" and e[1] == "llm_call")
    assert first_update["level"] == "WARNING" and "reply_not_bare_json" in first_update["status_message"]
    assert scores(client)["attempts"] == (2.0, "NUMERIC")


def test_a_failed_request_is_marked_as_an_error(table, settings):
    client = FakeClient()
    with pytest.raises(PipelineError):
        run(FakeLLM(["nope"] * 3), table, settings, LangfuseTracer(client))
    root_update = next(e[2] for e in client.log if e[0] == "update" and e[1] == "generate_note")
    assert root_update["level"] == "ERROR"
    assert ("error", "generate_note", "PipelineError") in client.log
    assert len([e for e in client.log if e[0] == "start" and e[1] == "llm_call"]) == 3


def test_noop_tracer_changes_nothing(table, settings):
    traced = run(FakeLLM([make_reply(code="N83")]), table, settings, LangfuseTracer(FakeClient()))
    plain = run(FakeLLM([make_reply(code="N83")]), table, settings, None)
    assert traced.suggested_codes == plain.suggested_codes and traced.needs_review == plain.needs_review


def test_health_reports_whether_tracing_is_on(table, settings):
    from fastapi.testclient import TestClient

    from gateway.app import create_app
    off = TestClient(create_app(settings=settings, llm=FakeLLM([]), table=table, system_prompt="S"))
    assert off.get("/health").json()["tracing"] is False
    client = FakeClient()
    app = create_app(settings=settings, llm=FakeLLM([]), table=table, system_prompt="S",
                     tracer=LangfuseTracer(client))
    with TestClient(app) as on:  # leaving the context runs shutdown, which flushes
        assert on.get("/health").json()["tracing"] is True
    assert client.flushed
