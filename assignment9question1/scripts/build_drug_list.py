"""Build data/drugs.csv from the Kaggle 'A-Z Medicine Dataset of India' CSV.

    python scripts/build_drug_list.py A_Z_medicines_dataset_of_India.csv
Adds (a) the first word of each brand name ("Augmentin 625 Duo Tablet" -> augmentin)
and (b) generic names from the composition columns ("Amoxycillin  (500mg)" -> amoxycillin).
Use --merge to keep what is already in data/drugs.csv.
"""
import argparse
import csv
import re
from pathlib import Path

WORD = re.compile(r"[a-z][a-z\-]{3,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--comp-cols", nargs="*", default=["short_composition1", "short_composition2"])
    ap.add_argument("--out", default="data/drugs.csv")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--min-count", type=int, default=1,
                    help="keep brand words seen at least N times (raise to prune noise)")
    a = ap.parse_args()

    generics: set[str] = set()
    brands: set[str] = set()
    if a.merge and Path(a.out).exists():
        with open(a.out, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                (brands if r.get("kind") == "brand" else generics).add(r["name"])
    counts: dict[str, int] = {}
    with open(a.csv_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            m = WORD.match((row.get(a.name_col) or "").strip().lower())
            if m:
                counts[m.group(0)] = counts.get(m.group(0), 0) + 1
            for c in a.comp_cols:
                comp = re.sub(r"\(.*?\)", " ", (row.get(c) or "").lower())  # drop strengths
                for w in WORD.findall(comp):
                    generics.add(w)
    brands |= {w for w, n in counts.items() if n >= a.min_count} - generics
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "kind"])
        for n in sorted(generics):
            w.writerow([n, "generic"])
        for n in sorted(brands):
            w.writerow([n, "brand"])
    print(f"wrote {len(generics)} generic + {len(brands)} brand names to {a.out}")


if __name__ == "__main__":
    main()
