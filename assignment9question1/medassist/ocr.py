"""OCR layer: EasyOCR or Tesseract, multi-variant, picks the best reading."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocess import make_variants


@dataclass
class OCRResult:
    text: str
    confidence: float          # 0-100
    backend: str
    variant: str = ""


def _score(text: str, conf: float) -> float:
    # Prefer high confidence, but penalise tiny outputs (usually garbage).
    return conf * min(1.0, len(text.strip()) / 60.0)


class OCREngine:
    def __init__(self, backend: str = "auto", languages: tuple[str, ...] = ("en",),
                 gpu: bool | None = None):
        self.languages = languages
        self.gpu = self._resolve_gpu(gpu)
        self._easy = None
        self.backend = self._pick_backend(backend)

    @staticmethod
    def _resolve_gpu(gpu: bool | None) -> bool:
        """None = use the GPU if PyTorch can see one. Only EasyOCR uses it (Tesseract is CPU-only)."""
        if gpu is not None:
            return gpu
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---- backend selection -------------------------------------------------
    def _pick_backend(self, backend: str) -> str:
        if backend in ("easyocr", "auto"):
            try:
                import easyocr  # noqa: F401
                return "easyocr"
            except Exception:
                if backend == "easyocr":
                    raise
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return "tesseract"
        except Exception as exc:
            raise RuntimeError(
                "No OCR backend available. Install Tesseract (and pytesseract) "
                "or EasyOCR. See README.") from exc

    # ---- individual engines -----------------------------------------------
    def _run_tesseract(self, img: np.ndarray, psm: int = 6) -> tuple[str, float]:
        import pytesseract
        data = pytesseract.image_to_data(
            img, config=f"--oem 3 --psm {psm}", output_type=pytesseract.Output.DICT)
        lines: dict[tuple, list[str]] = {}
        confs: list[float] = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            try:
                c = float(data["conf"][i])
            except ValueError:
                c = -1
            if not word or c < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            confs.append(c)
        text = "\n".join(" ".join(w) for _, w in sorted(lines.items()))
        return text, (sum(confs) / len(confs) if confs else 0.0)

    def _run_easyocr(self, img: np.ndarray) -> tuple[str, float]:
        if self._easy is None:
            import easyocr
            self._easy = easyocr.Reader(list(self.languages), gpu=self.gpu, verbose=False)
        results = self._easy.readtext(img)
        if not results:
            return "", 0.0
        items = []
        for box, txt, conf in results:
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append({"y": sum(ys) / 4, "x": min(xs),
                          "h": max(ys) - min(ys), "t": txt, "c": conf})
        items.sort(key=lambda d: d["y"])
        lines, cur = [], [items[0]]
        for it in items[1:]:
            if abs(it["y"] - cur[-1]["y"]) < 0.6 * max(it["h"], cur[-1]["h"], 1):
                cur.append(it)
            else:
                lines.append(cur)
                cur = [it]
        lines.append(cur)
        text = "\n".join(" ".join(d["t"] for d in sorted(l, key=lambda d: d["x"]))
                         for l in lines)
        return text, 100.0 * sum(d["c"] for d in items) / len(items)

    def _run(self, img: np.ndarray) -> tuple[str, float]:
        if self.backend == "easyocr":
            return self._run_easyocr(img)
        return self._run_tesseract(img)

    # ---- public API --------------------------------------------------------
    def read(self, src, try_rotations: bool = True) -> tuple[OCRResult, dict]:
        """Returns (best OCRResult, dict of preprocessed variants for display)."""
        variants = make_variants(src)
        best: OCRResult | None = None
        names = ["clahe", "adaptive", "inverted"] if self.backend == "tesseract" \
            else ["gray", "clahe"]
        for name in names:
            text, conf = self._run(variants[name])
            if best is None or _score(text, conf) > _score(best.text, best.confidence):
                best = OCRResult(text, conf, self.backend, name)

        # Bottle photographed sideways / upside-down? Try rotations if result is poor.
        if try_rotations and (best is None or _score(best.text, best.confidence) < 35):
            base = variants["clahe"]
            for code, label in ((cv2.ROTATE_90_CLOCKWISE, "rot90"),
                                (cv2.ROTATE_180, "rot180"),
                                (cv2.ROTATE_90_COUNTERCLOCKWISE, "rot270")):
                text, conf = self._run(cv2.rotate(base, code))
                if _score(text, conf) > _score(best.text, best.confidence):
                    best = OCRResult(text, conf, self.backend, label)
        return best, variants
