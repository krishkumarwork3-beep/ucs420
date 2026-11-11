"""Build data/drugs.csv from the free openFDA drug-label API (needs internet).

    python scripts/fetch_openfda_names.py
"""
import csv
import requests

URL = "https://api.fda.gov/drug/label.json"
FIELDS = ["openfda.generic_name.exact", "openfda.brand_name.exact"]


def main(out="data/drugs.csv"):
    names = set()
    for field in FIELDS:
        r = requests.get(URL, params={"count": field, "limit": 1000}, timeout=60)
        r.raise_for_status()
        for item in r.json().get("results", []):
            n = item["term"].lower().strip()
            if 3 <= len(n) <= 60:
                names.add(n)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name"])
        for n in sorted(names):
            w.writerow([n])
    print(f"wrote {len(names)} names to {out}")


if __name__ == "__main__":
    main()
