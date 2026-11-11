"""Drug-name lookup with OCR-error tolerant fuzzy matching."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

try:
    from rapidfuzz import fuzz, process
except ImportError:  # falls back to difflib
    fuzz = process = None
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Ordinary words that appear as brand names in big drug datasets ("Take", "Best 10") or
# on every label. They must never be reported as a medicine.
BLOCKLIST = set("""take from with light water other daily every once away only pack active best
children days times three four nine tablet tablets capsule capsules syrup store keep after before
food dose doses date expiry batch month year strip bottle contains ingredients manufactured
marketed product each have this that your used adult adults child hours time warning directions
dosage storage below protect reach prescription physician doctor medicine price inclusive taxes
label usual drops cream gel lotion spray powder solution injection ointment external internal
shake well discard exceed consult
""".split())


class DrugDB:
    def __init__(self, path: str | Path | None = None):
        if path is None:
            full = DATA_DIR / "drugs.csv"
            path = full if full.exists() else DATA_DIR / "drugs_sample.csv"
        self.path = Path(path)
        self.names: set[str] = set()      # every known name (exact match)
        self.generics: set[str] = set()   # generic names only (fuzzy match allowed)
        self._by_initial: dict[str, list[str]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        with open(self.path, newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            col = "name" if "name" in (reader.fieldnames or []) else reader.fieldnames[0]
            for row in reader:
                n = (row.get(col) or "").strip().lower()
                if len(n) < 3 or n in BLOCKLIST or n in self.names:
                    continue
                self.names.add(n)
                # Brands are matched exactly only (too many look-alike words to fuzz).
                if (row.get("kind") or "generic").strip().lower() != "brand":
                    self.generics.add(n)
                    self._by_initial[n[0]].append(n)

    def __len__(self) -> int:
        return len(self.names)

    def match(self, token: str, cutoff: float = 0.86) -> tuple[str, float] | None:
        """Best fuzzy match against generic names (same first letter, similar length)."""
        token = token.lower()
        if token in self.names:
            return token, 1.0
        cands = [c for c in self._by_initial.get(token[0], [])
                 if abs(len(c) - len(token)) <= 3]
        if not cands:
            return None
        if process is not None:
            hit = process.extractOne(token, cands, scorer=fuzz.ratio,
                                     score_cutoff=cutoff * 100)
            return (hit[0], hit[1] / 100.0) if hit else None
        best, best_s = None, 0.0
        for c in cands:
            s = SequenceMatcher(None, token, c).ratio()
            if s > best_s:
                best, best_s = c, s
        return (best, best_s) if best_s >= cutoff else None
