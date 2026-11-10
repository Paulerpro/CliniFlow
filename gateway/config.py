"""Gateway settings, read from environment variables (same style as serving/config.py)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only these prefixes are read from .env, so unrelated secrets in the file (AWS, OpenAI)
# never enter the gateway process.
ENV_PREFIXES = ("VLLM_", "MODEL_", "SYSTEM_PROMPT_", "MAX_", "ALLOW_", "REPAIR_", "REQUEST_", "LANGFUSE_", "TRACE_")


def load_dotenv(path=ROOT / ".env", prefixes=ENV_PREFIXES):
    """Fill os.environ from KEY=VALUE lines. Real environment variables win."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().partition("=")
        key = key.strip()
        if sep and not key.startswith("#") and key.startswith(prefixes):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


load_dotenv()


def _flag(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # OpenAI-compatible vLLM server (the Modal endpoint, or http://localhost:8000).
    vllm_url: str = field(default_factory=lambda: os.environ.get("VLLM_URL", "http://localhost:8000"))
    model_name: str = field(default_factory=lambda: os.environ.get("MODEL_NAME", "clinical-json-extractor"))
    # Must be the exact prompt the adapter was trained with (saved beside the adapter).
    system_prompt_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "SYSTEM_PROMPT_PATH", ROOT / "registry" / "adapters" / "config_A" / "system_prompt.txt")))
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("MAX_TOKENS", "1500")))
    # A cold Modal container can take minutes to answer the first request.
    request_timeout_s: float = field(default_factory=lambda: float(os.environ.get("REQUEST_TIMEOUT_S", "600")))
    # Re-asks after an unusable reply (total attempts = 1 + max_retries).
    max_retries: int = field(default_factory=lambda: int(os.environ.get("MAX_RETRIES", "2")))
    # Strict by default: a reply wrapped in a markdown fence counts as a format failure.
    allow_lenient_parse: bool = field(default_factory=lambda: _flag("ALLOW_LENIENT_PARSE", False))
    # Minimum description similarity (0 to 100) to replace a wrong code with a table match.
    repair_threshold: float = field(default_factory=lambda: float(os.environ.get("REPAIR_THRESHOLD", "90")))
