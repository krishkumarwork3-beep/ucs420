"""Generate a synthetic bottle-label image so you can test without any dataset."""
from PIL import Image, ImageDraw, ImageFont

LINES = [
    "DOLO 650", "Paracetamol Tablets IP 650 mg",
    "Take 1 tablet every 6 hours after food.", "Do not exceed 4 tablets in 24 hours.",
    "Batch No: DL2406A", "MFG: 03/2025", "EXP: 02/2027", "MRP Rs. 30.91",
    "Store below 30C. Protect from light and moisture.",
    "Keep out of reach of children.",
]


def main(out="samples/demo_label.png"):
    img = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 54)
        small = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        big = small = ImageFont.load_default()
    y = 30
    for i, ln in enumerate(LINES):
        d.text((40, y), ln, fill="black", font=big if i == 0 else small)
        y += 85 if i == 0 else 60
    img.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
