"""Evaluate OCR + extraction on a Roboflow COCO export where each box class is a brand name.

    python scripts/evaluate_coco.py path/to/medicine_v1i_coco --split test
    python scripts/evaluate_coco.py path/to/medicine_v1i_coco --split valid --crop --backend easyocr

Two metrics per image (truth = brand names labelled in that image):
  OCR hit      - is the brand name found (fuzzy, >= --threshold) anywhere in the raw OCR text?
                 This isolates the OCR quality from the drug database.
  End-to-end   - does the assistant's extracted drug_names contain the brand? The drug list is
                 built from this dataset's own class names, so it is an optimistic upper bound.
Writes <split>_results.csv so you can inspect every failure.
"""
import argparse
import csv
import json
import re
import sys
import tempfile
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402

from medassist.assistant import MedicineLabelAssistant  # noqa: E402

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def partial_score(name: str, text: str) -> float:
    """Best fuzzy match (0-1) of `name` against any window of `text`."""
    n, t = norm(name), norm(text)
    if not n or not t:
        return 0.0
    if n in t:
        return 1.0
    if fuzz is not None:
        return fuzz.partial_ratio(n, t) / 100.0
    best = 0.0
    for i in range(max(1, len(t) - len(n) + 1)):
        best = max(best, SequenceMatcher(None, n, t[i:i + len(n)]).ratio())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--crop", action="store_true", help="OCR only the labelled box (+10%% margin)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    split = Path(a.root) / a.split
    coco = json.load(open(split / "_annotations.coco.json"))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    used = {x["category_id"] for x in coco["annotations"]}
    names = sorted({cats[i] for i in used})

    truth: dict[int, list] = {}
    for ann in coco["annotations"]:
        truth.setdefault(ann["image_id"], []).append((cats[ann["category_id"]], ann["bbox"]))

    # drug list = this dataset's class names (fuzzy-matchable)
    tmp = Path(tempfile.mkdtemp()) / "eval_drugs.csv"
    tmp.write_text("name,kind\n" + "\n".join(f"{norm(n)},generic" for n in names))
    bot = MedicineLabelAssistant(a.backend, str(tmp))

    rows, ocr_hit, e2e_hit, total = [], 0, 0, 0
    images = coco["images"][: a.limit or None]
    out = Path(f"{a.split}_results{'_crop' if a.crop else ''}_{a.backend}.csv")
    fields = ["image", "truth", "ocr_score", "ocr_hit", "end_to_end", "extracted",
              "expiry", "ocr_conf", "ocr_text"]
    fh = open(out, "w", newline="", encoding="utf-8")   # written row by row: safe if interrupted
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()
    for k, im in enumerate(images, 1):
        t = truth.get(im["id"], [])
        if not t:
            continue
        img = cv2.imread(str(split / im["file_name"]))
        if img is None:
            continue
        if a.crop:
            _, (x, y, w, h) = t[0]
            m = int(0.1 * max(w, h))
            img = img[max(0, int(y) - m):int(y + h) + m, max(0, int(x) - m):int(x + w) + m]
        info, res, _ = bot.read_image(img)
        for brand in sorted({b for b, _ in t}):
            score = partial_score(brand, res.text)
            hit = score >= a.threshold
            e2e = norm(brand) in {norm(d) for d in info.drug_names}
            total += 1
            ocr_hit += hit
            e2e_hit += e2e
            row = ({"image": im["file_name"], "truth": brand, "ocr_score": round(score, 2),
                         "ocr_hit": hit, "end_to_end": e2e,
                         "extracted": "|".join(info.drug_names), "expiry": info.expiry_date or "",
                         "ocr_conf": round(res.confidence), "ocr_text": res.text.replace("\n", " / ")[:160]})
            writer.writerow(row)
            fh.flush()
        print(f"  {k}/{len(images)} {im['file_name'][:30]} hits so far {ocr_hit}/{total}", flush=True)
    fh.close()
    print(f"\nsplit={a.split} crop={a.crop} backend={bot.ocr.backend} labels={total}")
    print(f"OCR hit rate (brand found in raw text): {ocr_hit / total:.1%}")
    print(f"End-to-end name accuracy             : {e2e_hit / total:.1%}")
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
