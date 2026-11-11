"""Convert the COCO export in datasets/ to the YOLO layout Ultralytics trains on.

    python scripts/coco_to_yolo.py datasets/medicine_v1i_coco_plus --out datasets/yolo --single-class

--single-class : one class "medicine_pack" (recommended: finds ANY pack, then OCR reads it)
default        : one class per brand (closed set of ~18 brands; duplicates like PAROL/parol merged)
"""
import argparse
import json
import shutil
from pathlib import Path

SPLITS = ("train", "valid", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("coco_root")
    ap.add_argument("--out", default="datasets/yolo")
    ap.add_argument("--single-class", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.coco_root), Path(a.out)

    data = {sp: json.load(open(root / sp / "_annotations.coco.json")) for sp in SPLITS
            if (root / sp / "_annotations.coco.json").exists()}
    if a.single_class:
        names = ["medicine_pack"]
    else:  # classes actually used, merged case-insensitively
        used = set()
        for d in data.values():
            cat = {c["id"]: c["name"] for c in d["categories"]}
            used |= {cat[x["category_id"]].lower() for x in d["annotations"]}
        names = sorted(used)
    idx = {n: i for i, n in enumerate(names)}

    stats = {}
    for sp, d in data.items():
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        cat = {c["id"]: c["name"].lower() for c in d["categories"]}
        by_img = {}
        for x in d["annotations"]:
            by_img.setdefault(x["image_id"], []).append(x)
        n_box = 0
        for im in d["images"]:
            shutil.copy(root / sp / im["file_name"], out / "images" / sp / im["file_name"])
            W, H = im["width"], im["height"]
            lines = []
            for x in by_img.get(im["id"], []):
                bx, by, bw, bh = x["bbox"]
                bx, by = max(bx, 0), max(by, 0)
                bw, bh = min(bw, W - bx), min(bh, H - by)
                if bw <= 1 or bh <= 1:
                    continue
                c = 0 if a.single_class else idx[cat[x["category_id"]]]
                lines.append(f"{c} {(bx + bw / 2) / W:.6f} {(by + bh / 2) / H:.6f} "
                             f"{bw / W:.6f} {bh / H:.6f}")
            n_box += len(lines)
            (out / "labels" / sp / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        stats[sp] = (len(d["images"]), n_box)

    yaml = [f"path: {out.resolve().as_posix()}", "train: images/train", "val: images/valid"]
    if "test" in data:
        yaml.append("test: images/test")
    yaml.append("names:")
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"classes ({len(names)}): {names}")
    for sp, (ni, nb) in stats.items():
        print(f"{sp}: {ni} images, {nb} boxes")
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
