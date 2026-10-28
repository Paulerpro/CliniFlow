"""Settings for the sign-off layer, from environment variables or the repo's .env."""
import os
from pathlib import Path

from gateway.config import ROOT, load_dotenv

load_dotenv(prefixes=("GATEWAY_", "AGENTS_"))

# The gateway (Pillar 5) that produces draft notes.
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")
# A cold model container can take minutes to answer the first draft.
GATEWAY_TIMEOUT_S = float(os.environ.get("GATEWAY_TIMEOUT_S", "600"))
# Local SQLite file holding drafts and the audit log. Gitignored: it contains transcripts.
DB_PATH = Path(os.environ.get("AGENTS_DB", ROOT / "agents" / "data" / "signoff.db"))
# An approval or rejection of a flagged note must carry a reason at least this long.
MIN_REASON_CHARS = 10
