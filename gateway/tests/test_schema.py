from gateway.schema import parse_reply

from .conftest import make_reply


def test_valid_reply():
    obj, problems = parse_reply(make_reply())
    assert problems == [] and obj["suggested_codes"][0]["code"] == "N809"


def test_markdown_fence_fails_strict():
    obj, problems = parse_reply("```json\n" + make_reply() + "\n```")
    assert obj is None and problems == ["reply_not_bare_json"]


def test_markdown_fence_accepted_when_lenient():
    obj, problems = parse_reply("```json\n" + make_reply() + "\n```", allow_lenient=True)
    assert problems == [] and obj is not None


def test_prose_is_not_json():
    assert parse_reply("Sure! Here is the note.")[1] == ["reply_not_bare_json"]


def test_json_array_is_rejected():
    assert parse_reply("[1, 2]")[1] == ["reply_not_a_json_object"]


def test_missing_and_malformed_fields():
    _, problems = parse_reply('{"subjective": "x", "plan": 5, "suggested_codes": []}')
    assert set(problems) == {"objective_not_a_string", "assessment_not_a_string", "plan_not_a_string",
                             "suggested_codes_missing_or_empty"}


def test_code_missing_a_field_is_malformed():
    reply = make_reply(suggested_codes=[{"code": "N809", "system": "ICD-10", "description": "x"}])
    assert parse_reply(reply)[1] == ["suggested_code_malformed"]
