"""Image preprocessing tuned for photos of bottle / strip / box labels."""
from __future__ import annotations

import cv2
import numpy as np


def load_image(src) -> np.ndarray:
    """Accept a path, bytes, PIL image or ndarray and return a BGR ndarray."""
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (bytes, bytearray)):
        arr = np.frombuffer(src, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(src, str):
        img = cv2.imread(src, cv2.IMREAD_COLOR)
    else:  # PIL image or file-like
        from PIL import Image
        pil = src if hasattr(src, "convert") else Image.open(src)
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not read the image.")
    return img


def resize_for_ocr(img: np.ndarray, target_width: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    if w < target_width:  # upscale small photos - OCR likes ~30px tall characters
        scale = target_width / w
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if w > 2400:
        scale = 2400 / w
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def deskew(gray: np.ndarray) -> np.ndarray:
    """Straighten slightly rotated text using the minimum-area rectangle of ink pixels."""
    inv = cv2.bitwise_not(gray)
    _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 200:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 20:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def make_variants(src) -> dict[str, np.ndarray]:
    """Return several cleaned-up versions of the image; the OCR layer keeps the best."""
    img = resize_for_ocr(load_image(src))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(clahe, None, h=12,
                                        templateWindowSize=7, searchWindowSize=21)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 11)
    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # White-on-dark labels (common on amber bottles): also try the inverted image.
    inverted = cv2.bitwise_not(clahe)
    return {"gray": gray, "clahe": clahe, "adaptive": adaptive,
            "otsu": otsu, "inverted": inverted}
