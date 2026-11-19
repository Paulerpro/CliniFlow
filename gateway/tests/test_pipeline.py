import asyncio

import pytest

from gateway.pipeline import PipelineError, generate_note

from .conftest import TRANSCRIPT, FakeLLM, make_reply


def run(llm, table, settings):
    return asyncio.run(generate_note(TRANSCRIPT, llm, table, "SYSTEM", settings))


def test_clean_note_passes_without_review(table, settings):
    llm = FakeLLM([make_reply()])
    result = run(llm, table, settings)
    assert result.attempts == 1 and not result.needs_review
    assert result.suggested_codes[0]["status"] == "verified"
    assert llm.calls[0][0] == {"role": "system", "content": "SYSTEM"}


def test_retries_after_unusable_reply(table, settings):
    llm = FakeLLM(["```json\n" + make_reply() + "\n```", make_reply()])
    result = run(llm, table, settings)
    assert result.attempts == 2
    retry_turn = llm.calls[1][-1]
    assert retry_turn["role"] == "user" and "reply_not_bare_json" in retry_turn["content"]
    assert llm.calls[1][-2]["role"] == "assistant"  # the bad reply is shown back to the model


def test_gives_up_after_max_retries(table, settings):
    llm = FakeLLM(["nope"] * 3)
    with pytest.raises(PipelineError) as e:
        run(llm, table, settings)
    assert e.value.attempts == 3 and e.value.problems == ["reply_not_bare_json"]
    assert llm.replies == []  # exactly 1 + max_retries calls


def test_wrong_code_is_repaired_and_flagged(table, settings):
    result = run(FakeLLM([make_reply(code="N83")]), table, settings)
    code = result.suggested_codes[0]
    assert code["code"] == "N809" and code["status"] == "repaired" and code["original_code"] == "N83"
    assert result.needs_review and result.code_review and not result.content_review
    assert any(w.type == "code_repaired" and w.category == "code" for w in result.warnings)


def test_unresolvable_code_is_kept_and_flagged(table, settings):
    result = run(FakeLLM([make_reply(code="Q9999", description="zzz qqq")]), table, settings)
    assert result.suggested_codes[0]["code"] == "Q9999"
    assert result.needs_review
    assert any(w.type == "code_not_found" for w in result.warnings)


def test_invented_content_sets_needs_review(table, settings):
    reply = make_reply(plan="Start Metformin 500 mg twice daily.")
    result = run(FakeLLM([reply]), table, settings)
    assert result.needs_review and result.content_review and not result.code_review
    assert any(w.type == "ungrounded_medication" and w.category == "content" for w in result.warnings)


def test_info_warnings_alone_do_not_need_review(table, settings):
    reply = make_reply(subjective="A 45-year-old Hispanic patient with lower abdominal pain.")
    result = run(FakeLLM([reply]), table, settings)
    assert not result.needs_review and not result.code_review and not result.content_review
    assert any(w.type == "invented_demographic" for w in result.warnings)


def test_repair_onto_a_parent_code_adds_a_not_billable_note(table, settings):
    result = run(FakeLLM([make_reply(code="N78", description="ENDOMETRIOSIS")]), table, settings)
    types = {(w.type, w.severity) for w in result.warnings}
    assert ("code_repaired", "review") in types and ("code_not_billable", "info") in types
    assert result.needs_review and result.code_review
