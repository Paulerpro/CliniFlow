import pytest

from agents.store import Store


def gateway_response(code_review=False, content_review=False, warnings=None):
    """A response shaped like gateway POST /v1/notes."""
    return {
        "subjective": "Lower abdominal pain.", "objective": "BP 120/80.",
        "assessment": "Likely endometriosis.", "plan": "Naproxen 500 mg.",
        "suggested_codes": [{"code": "N809", "system": "ICD-10", "description": "Endometriosis, unspecified",
                             "confidence": "high", "status": "verified", "original_code": None}],
        "needs_review": code_review or content_review,
        "code_review": code_review, "content_review": content_review,
        "warnings": warnings or [], "attempts": 1, "model": "m", "latency_ms": 5,
    }


FLAG_WARNING = {"type": "code_repaired", "severity": "review", "category": "code",
                "detail": "model gave N83; replaced with N809 from its description"}


@pytest.fixture
def store():
    return Store(":memory:")


@pytest.fixture(scope="session")
def table():
    from gateway.icd10 import IcdTable
    return IcdTable.load()
