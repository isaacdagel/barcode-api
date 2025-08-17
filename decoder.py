#!/usr/bin/env python3
"""
read_barcode.py — robust single-barcode reader for slab images (PCGS/NGC/etc.)

Usage:
    python read_barcode.py /path/to/image.jpg

Output:
    Prints decoded barcode text to stdout. Exit 0 on success, 1 on failure.

Deps:
    pip install pillow
    # One (or both) of the following:
    pip install pyzbar     # requires system zbar (mac: brew install zbar; ubuntu: sudo apt-get install libzbar0)
    pip install opencv-contrib-python
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# ----------------- decoders -----------------

def _decode_with_pyzbar(pil_img: Image.Image) -> List[str]:
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except Exception:
        return []
    hits = []
    for r in zbar_decode(pil_img.convert("L")):
        s = r.data.decode("utf-8", "replace")
        if s:
            hits.append(s)
    return hits

def _decode_with_cv(bgr: np.ndarray) -> List[str]:
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        return []
    det = cv2.barcode_BarcodeDetector()
    try:
        result = det.detectAndDecode(bgr)
    except Exception:
        return []
    ok, infos = False, []
    if isinstance(result, tuple):
        if len(result) == 4:
            ok, infos, _types, _pts = result
        elif len(result) == 3:
            infos, _types, _pts = result
            ok = bool(infos and any(infos))
    if ok and isinstance(infos, (list, tuple)):
        return [s for s in infos if s]
    return []

def _try_all_decoders(bgr: np.ndarray) -> List[str]:
    out: List[str] = []
    # try normal + inverted
    for img in (bgr, 255 - bgr):
        # OpenCV
        out += _decode_with_cv(img)
        # pyzbar
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out += _decode_with_pyzbar(pil)
    return out

def _uniq_sort(vals: List[str]) -> List[str]:
    return sorted(set(vals), key=lambda s: (-len(s), s))

# ----------------- slab-specific regioning -----------------

def _top_label_region(bgr: np.ndarray) -> np.ndarray:
    """Return top ~55% (covers label area on most slabs)."""
    H = bgr.shape[0]
    return bgr[: int(H * 0.55)].copy()

def _barcode_strip_roi(label_bgr: np.ndarray) -> np.ndarray:
    """
    Find the horizontal strip with the most vertical edges (where 1D bars live).
    Returns a tight horizontal slice of the label.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.convertScaleAbs(gx)
    row_energy = gx.mean(axis=1)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1,1), (1, 9), 0).ravel()

    # pick the best row window
    H = label_bgr.shape[0]
    win = max(24, H // 12)  # ~8–10% label height
    center = int(np.argmax(row_energy))
    y0 = max(0, center - win//2)
    y1 = min(H, y0 + win)
    return label_bgr[y0:y1].copy()

def _split_into_barcode_blobs(strip_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Split the strip into one or two blobs (PCGS often has 2 barcodes).
    Returns up to 3 tight crops.
    """
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)
    # strong binarization to emphasize bars
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = 255 - thr  # make bars white for morphology
    # connect vertical strokes into wider components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    fc = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(fc) == 3:
        _img, contours, _hier = fc
    else:
        contours, _hier = fc

    boxes: List[Tuple[int,int,int,int]] = []
    H, W = strip_bgr.shape[:2]
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        aspect = w / max(1.0, h)
        if h > H*0.35 and w > W*0.08 and aspect > 2.0:  # wide-ish, tall enough
            boxes.append((x,y,w,h))

    if not boxes:
        # fall back to left/right halves to be safe
        mid = W // 2
        boxes = [(0, 0, mid, H), (mid, 0, W - mid, H)]

    # left-to-right
    boxes.sort(key=lambda b: b[0])

    crops: List[np.ndarray] = []
    for (x,y,w,h) in boxes[:3]:
        pad = 8
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        crop = strip_bgr[y0:y1, x0:x1].copy()
        # upscale strongly; thin bars benefit a lot
        if max(crop.shape[:2]) < 1800:
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crops.append(crop)
    return crops

# ----------------- main API -----------------

def read_single_barcode(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 0) Quick whole-frame tries (sometimes enough)
    rotations = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)
    for rot in rotations:
        arr = bgr if rot is None else cv2.rotate(bgr, rot)
        hits = _try_all_decoders(arr)
        if hits:
            return _uniq_sort(hits)[0]

    # 1) Focus on the top label, find the barcode strip
    label = _top_label_region(bgr)
    strip = _barcode_strip_roi(label)

    # 2) Split into barcode blobs (handles PCGS double-bar layout)
    crops = _split_into_barcode_blobs(strip)

    # 3) Decode each crop with several preprocess variants
    for crop in crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants = [
            crop,
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 35, 11),
        ]
        for v in variants:
            for rot in rotations:
                vv = v if rot is None else cv2.rotate(v, rot)
                hits = _try_all_decoders(vv)
                if hits:
                    return _uniq_sort(hits)[0]

    raise ValueError("No decodable barcode found.")

# ----------------- CLI -----------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Read the single barcode in an image and print its text.")
    ap.add_argument("image", help="Path to the image file.")
    args = ap.parse_args()
    try:
        print(read_single_barcode(args.image))
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
