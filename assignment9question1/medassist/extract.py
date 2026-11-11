"""Turns raw OCR text into structured medicine-label information."""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import asdict, dataclass, field

from .drugdb import DrugDB

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}

STOPWORDS = {
    "tablet", "tablets", "capsule", "capsules", "syrup", "store", "keep", "away",
    "children", "reach", "every", "hours", "daily", "twice", "once", "before",
    "after", "meals", "water", "directions", "warning", "batch", "expiry", "date",
    "manufactured", "marketed", "limited", "pharma", "pharmaceuticals", "each",
    "contains", "dosage", "storage", "below", "protect", "light", "moisture",
    "prescription", "physician", "doctor", "medicine", "price", "inclusive",
    "taxes", "tablets", "strip", "bottle", "label", "usual", "adult", "adults",
}

FORMS = ["tablet", "capsule", "syrup", "suspension", "injection", "ointment",
         "cream", "drops", "gel", "inhaler", "lotion", "sachet", "lozenge", "spray",
         "powder", "solution"]

STRENGTH_RE = re.compile(
    r"(?<![\w.])(\d{1,4}(?:[.,]\d{1,3})?)\s?(mg|mcg|µg|ug|g|ml|iu|meq|%)(?![a-z])",
    re.I)

INSTR_RE = re.compile(
    r"\b(take|taking|apply|swallow|chew|dissolve|inject|inhale|instill|use as|"
    r"once|twice|thrice|daily|every\s+\d+|\d+\s*times|od|bd|bid|tid|tds|qid|qds|"
    r"hs|prn|before\s+(?:meals?|food)|after\s+(?:meals?|food)|with\s+food|"
    r"at\s+bed\s?time|as\s+directed|as\s+prescribed)\b", re.I)

WARN_RE = re.compile(
    r"\b(warning|caution|do not|don't|not to be|keep out|children|pregnan|"
    r"breast|allerg|drowsi|alcohol|overdose|exceed|side effects?|consult|"
    r"contraindicat|for external use|shake well|discard)\b", re.I)

STORE_RE = re.compile(
    r"\b(store|storage|keep in|protect from|refrigerat|below\s*\d+|"
    r"cool|dry place|away from (?:light|heat|moisture))\b", re.I)

DATE_LABEL_RE = re.compile(
    r"(?P<label>EXP(?:IRY)?\.?(?:\s*DATE)?|USE\s*BEFORE|BEST\s*BEFORE|"
    r"MFG\.?(?:\s*DATE)?|MFD\.?|MANUFACTURED(?:\s*ON)?|MANUF\.?\s*DATE)"
    r"\s*[:.\-]*\s*(?P<chunk>[0-9A-Za-z/\-. ]{4,16})", re.I)

BATCH_RE = re.compile(
    r"(?:BATCH\s*(?:NO\.?|NUMBER|#)?|B\.?\s*NO\.?|LOT\s*(?:NO\.?|#)?)\s*[:.\-]*\s*"
    r"(?P<b>[A-Z0-9][A-Z0-9\-/]{2,14})", re.I)

MRP_RE = re.compile(r"(?:MRP|PRICE)\s*[:.\-]*\s*(?:Rs\.?|INR|₹|\$)?\s*(\d+(?:\.\d{1,2})?)",
                    re.I)


# --------------------------------------------------------------------------- dates
def _year(y: str) -> int:
    n = int(y)
    return 2000 + n if n < 100 else n


def parse_date(chunk: str) -> dt.date | None:
    """Parse the usual label formats. Month-only dates resolve to the month's last day."""
    s = chunk.strip()
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})\b", s)
    if m:  # dd/mm/yyyy (Indian/UK convention)
        d, mo, y = int(m[1]), int(m[2]), _year(m[3])
        if 1 <= mo <= 12 and 1 <= d <= 31:
            try:
                return dt.date(y, mo, d)
            except ValueError:
                pass
    m = re.search(r"\b(\d{4})\s*[/\-.]\s*(\d{1,2})(?:\s*[/\-.]\s*(\d{1,2}))?\b", s)
    if m and 1 <= int(m[2]) <= 12:
        y, mo = int(m[1]), int(m[2])
        d = int(m[3]) if m[3] else calendar.monthrange(y, mo)[1]
        try:
            return dt.date(y, mo, d)
        except ValueError:
            pass
    m = re.search(r"\b([A-Za-z]{3})[a-z]*\.?[\s\-/,]*(\d{4}|\d{2})\b", s)
    if m and m[1].lower() in MONTHS:
        y, mo = _year(m[2]), MONTHS[m[1].lower()]
        return dt.date(y, mo, calendar.monthrange(y, mo)[1])
    m = re.search(r"\b(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})\b", s)
    if m and 1 <= int(m[1]) <= 12:
        y, mo = _year(m[2]), int(m[1])
        return dt.date(y, mo, calendar.monthrange(y, mo)[1])
    return None


# --------------------------------------------------------------------------- result
@dataclass
class LabelInfo:
    raw_text: str = ""
    ocr_confidence: float = 0.0
    drug_names: list[str] = field(default_factory=list)
    name_confidence: str = "none"          # high / low / none
    strengths: list[str] = field(default_factory=list)
    form: str | None = None
    instructions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    storage: list[str] = field(default_factory=list)
    expiry_date: str | None = None         # ISO date (last day of month if month-only)
    manufacture_date: str | None = None
    expiry_status: str = "unknown"         # expired / expiring_soon / valid / unknown
    days_to_expiry: int | None = None
    batch: str | None = None
    mrp: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- human / speech friendly summary
    def summary(self) -> str:
        parts = []
        if self.drug_names:
            name = ", ".join(n.title() for n in self.drug_names[:3])
            strength = f" {self.strengths[0]}" if self.strengths else ""
            form = f" {self.form}" if self.form else ""
            hedge = "" if self.name_confidence == "high" else " (please double-check this name)"
            parts.append(f"This looks like {name}{strength}{form}{hedge}.")
        else:
            parts.append("I could not confidently identify the medicine name.")
        if self.instructions:
            parts.append("Directions printed on the label: " + " ".join(self.instructions[:2]))
        if self.expiry_date:
            msg = {"expired": "WARNING: this medicine appears to be EXPIRED. Do not use it.",
                   "expiring_soon": f"It expires in {self.days_to_expiry} days.",
                   "valid": "It has not expired yet."}.get(self.expiry_status, "")
            parts.append(f"Expiry date: {self.expiry_date}. {msg}")
        else:
            parts.append("I could not find an expiry date.")
        if self.warnings:
            parts.append("Label warnings: " + " ".join(self.warnings[:2]))
        parts.append("This is an automated reading. Confirm with your doctor or pharmacist "
                     "before taking any medicine.")
        return " ".join(parts)


# --------------------------------------------------------------------------- core
def _clean_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines() if ln.strip()]


def find_drugs(text: str, db: DrugDB) -> tuple[list[str], str]:
    tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", text)
    found: dict[str, float] = {}
    n = len(tokens)
    for i in range(n):
        for k in (3, 2, 1):
            if i + k > n:
                continue
            cand = " ".join(tokens[i:i + k]).lower()
            if k == 1 and (len(cand) < 4 or cand in STOPWORDS):
                continue
            if cand in db.names:
                found[cand] = max(found.get(cand, 0), 1.0)
            elif k == 1 and len(cand) >= 6:
                hit = db.match(cand)
                if hit:
                    found[hit[0]] = max(found.get(hit[0], 0), hit[1])
    # drop names fully contained in a longer found name ("acid" in "folic acid")
    names = [a for a in found if not any(a != b and a in b for b in found)]
    names.sort(key=lambda x: (-found[x], text.lower().find(x.split()[0])))
    names = names[:5]
    if names:
        return names, ("high" if found[names[0]] >= 0.93 else "low")

    # Fallback heuristic: the longest ALL-CAPS word near a strength is usually the brand.
    caps = [w for w in re.findall(r"\b[A-Z][A-Z\-]{4,}\b", text) if w.lower() not in STOPWORDS]
    if caps:
        return [max(caps, key=len).lower()], "low"
    return [], "none"


def extract(text: str, db: DrugDB | None = None, today: dt.date | None = None,
            ocr_confidence: float = 0.0) -> LabelInfo:
    db = db or DrugDB()
    today = today or dt.date.today()
    info = LabelInfo(raw_text=text, ocr_confidence=round(ocr_confidence, 1))
    lines = _clean_lines(text)
    flat = "\n".join(lines)

    info.drug_names, info.name_confidence = find_drugs(flat, db)

    seen = []
    for m in STRENGTH_RE.finditer(flat):
        s = f"{m[1].replace(',', '.')} {m[2].lower().replace('µg', 'mcg')}"
        if s not in seen:
            seen.append(s)
    info.strengths = seen[:4]

    low = flat.lower()
    for f in FORMS:
        if re.search(rf"\b{f}s?\b", low):
            info.form = f
            break

    info.instructions = [l for l in lines if INSTR_RE.search(l)][:4]
    info.warnings = [l for l in lines if WARN_RE.search(l)][:5]
    info.storage = [l for l in lines if STORE_RE.search(l)][:3]

    for m in DATE_LABEL_RE.finditer(flat):
        label = m["label"].upper()
        d = parse_date(m["chunk"])
        if not d:
            continue
        if label.startswith(("MFG", "MFD", "MANUF")):
            info.manufacture_date = info.manufacture_date or d.isoformat()
        elif info.expiry_date is None:
            info.expiry_date = d.isoformat()

    if info.expiry_date:
        exp = dt.date.fromisoformat(info.expiry_date)
        info.days_to_expiry = (exp - today).days
        info.expiry_status = ("expired" if info.days_to_expiry < 0
                              else "expiring_soon" if info.days_to_expiry <= 60
                              else "valid")

    b = BATCH_RE.search(flat)
    info.batch = b["b"].upper() if b else None
    p = MRP_RE.search(flat)
    info.mrp = p[1] if p else None
    return info
