"""Detect the medicine pack with the trained YOLO model, crop it, then OCR + extract.

    python scripts/detect_and_read.py photo.jpg --weights runs/detect/train/weights/best.pt --device 0
Falls back to the whole image if nothing is detected.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2  # noqa: E402

from medassist.assistant import MedicineLabelAssistant  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="0", help="0 = first GPU, cpu = CPU")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO
    img = cv2.imread(a.image)
    res = YOLO(a.weights).predict(img, conf=a.conf, device=a.device, verbose=False)[0]
    crop = img
    if len(res.boxes):
        best = res.boxes[int(res.boxes.conf.argmax())]
        x1, y1, x2, y2 = best.xyxy[0].tolist()
        mx, my = (x2 - x1) * a.margin, (y2 - y1) * a.margin
        h, w = img.shape[:2]
        crop = img[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))]
        print(f"[detector] pack found, conf {float(best.conf):.2f}")
    else:
        print("[detector] nothing found - reading the whole image")

    gpu = a.device != "cpu"
    bot = MedicineLabelAssistant(a.backend, gpu=gpu)
    info, ocr, _ = bot.read_image(crop)
    print(f"[OCR] {ocr.backend}, confidence {ocr.confidence:.0f}%\n")
    print(json.dumps(info.to_dict(), indent=2) if a.json else info.summary())


if __name__ == "__main__":
    main()
