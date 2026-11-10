"""ICD-10-CM lookup and code repair.

The table (data/icd10cm_fy2026.tsv.gz) is the CDC/NCHS FY2026 order file: 98,186
entries, of which 74,719 are billable. Parent ("header") codes such as N508 are
real codes but not billable. The dataset's own gold notes use some of them, so
they are accepted with a warning and never "repaired".
"""
import gzip
import re
from dataclasses import dataclass
from pathlib import Path

from rapidfuzz import fuzz, process

DATA_PATH = Path(__file__).parent / "data" / "icd10cm_fy2026.tsv.gz"
VERIFIED_SIMILARITY = 85.0  # table description vs. the model's description


def normalize_code(code):
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", str(text).upper())).strip()


@dataclass(frozen=True)
class IcdEntry:
    code: str
    billable: bool
    description: str


@dataclass(frozen=True)
class DescriptionMatch:
    entry: IcdEntry
    score: float
    ambiguous: bool  # several entries tied; the one nearest the model's code was chosen


@dataclass(frozen=True)
class ResolvedCode:
    code: str
    description: str
    # verified | verified_non_billable | repaired | unresolved_mismatch | unresolved_not_found
    status: str
    original_code: str
    score: float | None = None
    ambiguous: bool = False


class IcdTable:
    def __init__(self, entries):
        self._entries = list(entries)
        self._by_code = {e.code: e for e in self._entries}
        self._norm_desc = [normalize_text(e.description) for e in self._entries]

    @classmethod
    def load(cls, path=DATA_PATH):
        entries = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            next(f)  # header row
            for line in f:
                code, billable, description = line.rstrip("\n").split("\t", 2)
                entries.append(IcdEntry(code, billable == "1", description))
        return cls(entries)

    def __len__(self):
        return len(self._entries)

    def lookup(self, code):
        return self._by_code.get(normalize_code(code))

    def match_description(self, description, near_code=None, cutoff=0.0):
        """Best table entry for a description. Ties go to the code nearest near_code."""
        query = normalize_text(description)
        if not query:
            return None
        hits = process.extract(query, self._norm_desc, scorer=fuzz.ratio, limit=10, score_cutoff=cutoff)
        if not hits:
            return None
        best = hits[0][1]
        tied = [self._entries[i] for _, score, i in hits if score >= best - 1e-9]
        if len(tied) > 1 and near_code:
            near = normalize_code(near_code)
            tied.sort(key=lambda e: -_common_prefix(e.code, near))
        return DescriptionMatch(tied[0], round(best, 1), ambiguous=len(tied) > 1)

    def search(self, query, limit=5, billable_only=True, min_score=70.0):
        """Free-text search over descriptions: [(entry, score 0-100)], best first.

        Works best with clinical diagnosis wording ("type 2 diabetes with kidney disease");
        plain symptom words rank poorly. Scores below min_score are dropped as noise.
        """
        q = normalize_text(query)
        if not q:
            return []
        hits = process.extract(q, self._norm_desc, scorer=fuzz.WRatio, limit=limit * 6 + 20)
        ranked = [(self._entries[i], score) for _, score, i in hits
                  if (self._entries[i].billable or not billable_only) and score >= min_score]
        # equal scores: prefer the shorter (more general) description
        ranked.sort(key=lambda t: (-t[1], len(t[0].description)))
        return [(e, round(sc, 1)) for e, sc in ranked[:limit]]

    def description_similarity(self, entry, description):
        return fuzz.ratio(normalize_text(entry.description), normalize_text(description))


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def resolve_code(table, code, description, threshold=90.0):
    """Check one suggested code against the table; repair it by description if it is wrong.

    - code exists and its description agrees  -> verified (non-billable parents are flagged)
    - otherwise a close description match     -> repaired (the code is replaced)
    - otherwise                               -> unresolved; the model's code is kept
    """
    original = str(code)
    entry = table.lookup(code)
    if entry:
        sim = table.description_similarity(entry, description)
        if sim >= VERIFIED_SIMILARITY:
            status = "verified" if entry.billable else "verified_non_billable"
            return ResolvedCode(entry.code, entry.description, status, original, sim)
    match = table.match_description(description, near_code=code, cutoff=threshold)
    if match and (entry is None or match.entry.code != entry.code):
        return ResolvedCode(match.entry.code, match.entry.description, "repaired", original,
                            match.score, match.ambiguous)
    if entry:
        return ResolvedCode(entry.code, str(description), "unresolved_mismatch", original)
    return ResolvedCode(original, str(description), "unresolved_not_found", original)
