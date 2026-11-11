"""Evaluate on a labelled image folder.

ground-truth CSV columns: image, drug_name, expiry [, text]
  expiry in ISO (YYYY-MM-DD, or YYYY-MM-last-day), text = optional full transcription
    python scripts/evaluate.py images/ truth.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medassist.assistant import MedicineLabelAssistant  # noqa: E402


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main(folder, truth_csv):
    bot = MedicineLabelAssistant()
    n = name_ok = exp_ok = exp_n = 0
    cer_sum = cer_n = 0
    for row in csv.DictReader(open(truth_csv, newline="", encoding="utf-8")):
        info, res, _ = bot.read_image(str(Path(folder) / row["image"]))
        n += 1
        if row.get("drug_name") and row["drug_name"].lower() in [d.lower() for d in info.drug_names]:
            name_ok += 1
        if row.get("expiry"):
            exp_n += 1
            exp_ok += info.expiry_date == row["expiry"]
        if row.get("text"):
            ref = " ".join(row["text"].split()).lower()
            hyp = " ".join(res.text.split()).lower()
            cer_sum += levenshtein(ref, hyp) / max(len(ref), 1)
            cer_n += 1
    print(f"images: {n}")
    print(f"drug-name accuracy : {name_ok / max(n, 1):.1%}")
    if exp_n:
        print(f"expiry accuracy    : {exp_ok / exp_n:.1%}")
    if cer_n:
        print(f"mean char error rate: {cer_sum / cer_n:.1%}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
