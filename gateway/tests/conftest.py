import json

import pytest

from gateway.config import Settings
from gateway.icd10 import IcdTable


@pytest.fixture(scope="session")
def table():
    return IcdTable.load()


@pytest.fixture
def settings():
    return Settings(max_retries=2)


TRANSCRIPT = (
    "[doctor] What brings you in?\n"
    "[patient] Lower belly pain for months, worse with my period.\n"
    "[doctor] Your blood pressure is 120/80 and heart rate is 76. I will start naproxen 500 mg twice a day.\n"
)


def make_reply(code="N809", description="ENDOMETRIOSIS, UNSPECIFIED", **overrides):
    note = {
        "subjective": "Lower abdominal pain for months, worse with menses.",
        "objective": "BP 120/80, HR 76.",
        "assessment": "Likely endometriosis.",
        "plan": "Naproxen 500 mg twice daily.",
        "suggested_codes": [
            {"code": code, "system": "ICD-10", "description": description, "confidence": "medium"}
        ],
    }
    note.update(overrides)
    return json.dumps(note)


class FakeLLM:
    """Returns the queued replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages):
        self.calls.append(messages)
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def no_real_tracing(monkeypatch):
    """Keys in .env must never make a test run send traces to a real Langfuse project."""
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
