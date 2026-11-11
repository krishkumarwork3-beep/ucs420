"""High-level 'cognitive assistant': see (OCR) -> understand (NLP) -> answer / speak."""
from __future__ import annotations

import re

from .drugdb import DrugDB
from .extract import LabelInfo, extract

DISCLAIMER = ("Automated reading - verify with a doctor or pharmacist before taking "
              "any medicine.")


class MedicineLabelAssistant:
    def __init__(self, ocr_backend: str = "auto", drug_db_path: str | None = None,
                 gpu: bool | None = None):
        self.db = DrugDB(drug_db_path)
        self._ocr_backend = ocr_backend
        self._gpu = gpu
        self._ocr = None
        self.info: LabelInfo | None = None

    @property
    def ocr(self):
        if self._ocr is None:  # lazy: importing OCR libs is slow
            from .ocr import OCREngine
            self._ocr = OCREngine(self._ocr_backend, gpu=self._gpu)
        return self._ocr

    # ---- perception + understanding
    def read_image(self, src):
        result, variants = self.ocr.read(src)
        self.info = extract(result.text, self.db, ocr_confidence=result.confidence)
        return self.info, result, variants

    def read_text(self, text: str) -> LabelInfo:
        """Use when OCR text was already corrected by the user (or for testing)."""
        self.info = extract(text, self.db)
        return self.info

    # ---- conversational layer
    def ask(self, question: str) -> str:
        if self.info is None:
            return "Please scan a medicine label first."
        i, q = self.info, question.lower()

        def has(*words):
            return any(re.search(rf"\b{w}", q) for w in words)

        if has("expir", "valid", "expire", "date", "good to use", "still safe"):
            if not i.expiry_date:
                return "I couldn't find an expiry date on the label."
            return (f"Expiry date: {i.expiry_date}. "
                    + {"expired": "It is EXPIRED - do not use it.",
                       "expiring_soon": f"It expires in {i.days_to_expiry} days.",
                       "valid": "It is still within date."}.get(i.expiry_status, ""))
        if has("dose", "dosage", "how", "take", "instruction", "when", "times"):
            return " ".join(i.instructions) if i.instructions else \
                "No dosing instructions were readable. Please follow your prescription."
        if has("warn", "side", "caution", "safe", "child", "pregnan", "alcohol", "avoid"):
            return " ".join(i.warnings) if i.warnings else \
                "No warnings were readable on the label."
        if has("store", "storage", "keep", "fridge"):
            return " ".join(i.storage) if i.storage else "No storage advice found."
        if has("batch", "lot"):
            return f"Batch number: {i.batch}" if i.batch else "No batch number found."
        if has("price", "mrp", "cost"):
            return f"MRP: {i.mrp}" if i.mrp else "No price found."
        if has("name", "what", "which", "medicine", "drug", "strength", "mg"):
            if i.drug_names:
                st = f" ({', '.join(i.strengths)})" if i.strengths else ""
                return f"{', '.join(n.title() for n in i.drug_names)}{st}"
            return "I could not identify the medicine name."
        return i.summary()
