"""FastAPI gateway: POST a transcript, get back a validated, repaired, flagged SOAP note.

Run:
    VLLM_URL=https://<modal-endpoint> uvicorn gateway.app:app --port 8080
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings, proxy_auth_headers
from .icd10 import IcdTable
from .llm import LLMClient, LLMError, VllmClient
from .pipeline import PipelineError, generate_note
from .tracing import make_tracer


class NoteRequest(BaseModel):
    transcript: str = Field(min_length=1, max_length=40000)


class CodeOut(BaseModel):
    code: str
    system: str
    description: str
    confidence: str
    status: str  # verified | verified_non_billable | repaired | unresolved_mismatch | unresolved_not_found
    original_code: str | None = None  # set only when the code was repaired


class WarningOut(BaseModel):
    type: str
    severity: str  # "review" sets the review flags; "info" is recorded only
    category: str  # "code" (ICD-10) or "content" (note vs. transcript)
    detail: str


class NoteResponse(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    suggested_codes: list[CodeOut]
    needs_review: bool  # code_review or content_review
    code_review: bool  # an ICD-10 code was repaired, did not match, or does not exist
    content_review: bool  # the note states a drug, dose, value or doctor the transcript does not
    warnings: list[WarningOut]
    attempts: int
    model: str
    latency_ms: int


def create_app(settings: Settings | None = None, llm: LLMClient | None = None,
               table: IcdTable | None = None, system_prompt: str | None = None, tracer=None) -> FastAPI:
    settings = settings or Settings()
    if system_prompt is None:
        system_prompt = settings.system_prompt_path.read_text(encoding="utf-8")
    table = table or IcdTable.load()
    llm = llm or VllmClient(settings.vllm_url, settings.model_name,
                            settings.max_tokens, settings.request_timeout_s,
                            headers=proxy_auth_headers())

    tracer = tracer or make_tracer()  # a no-op unless Langfuse keys are set

    @asynccontextmanager
    async def lifespan(app):
        yield
        tracer.flush()  # send any buffered traces before the process exits

    app = FastAPI(title="CliniFlow gateway", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "icd_codes": len(table), "model": settings.model_name,
                "vllm_url": settings.vllm_url, "tracing": tracer.enabled}  # shown so a wrong target is obvious

    @app.post("/v1/notes", response_model=NoteResponse)
    async def create_note(req: NoteRequest):
        start = time.perf_counter()
        try:
            result = await generate_note(req.transcript, llm, table, system_prompt, settings, tracer)
        except PipelineError as e:
            raise HTTPException(status_code=502, detail={
                "message": "the model did not return a usable note",
                "attempts": e.attempts, "problems": e.problems})
        except LLMError as e:
            raise HTTPException(status_code=503, detail={"message": str(e)})
        return NoteResponse(
            **result.note,
            suggested_codes=result.suggested_codes,
            needs_review=result.needs_review,
            code_review=result.code_review,
            content_review=result.content_review,
            warnings=[WarningOut(type=w.type, severity=w.severity, category=w.category,
                                  detail=w.detail)
                      for w in result.warnings],
            attempts=result.attempts,
            model=settings.model_name,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )

    return app


def _default_app():
    # Built lazily so importing this module (tests, tooling) does not read the
    # system prompt or load the ICD table.
    return create_app()


def __getattr__(name):
    if name == "app":
        return _default_app()
    raise AttributeError(name)
