#!/usr/bin/env python3
"""
read_barcode.py — robust single-barcode reader for slab images (PCGS/NGC/etc.)

Assumptions:

* Slab image is already in the correct orientation (no rotation needed).
* Image is in color.
* Barcode is always roughly between 25% and 35% of the full image height.
  (Example you gave: H≈2027, barcode ≈ 565–655 → 27.9–32.3%.)

Usage:
    python read_barcode.py /path/to/image.jpg

Output:
    Prints decoded barcode text to stdout. Exit 0 on success, 1 on failure.

Deps:
    pip install pillow
    pip install pyzbar         # plus system zbar
    pip install opencv-python  # or opencv-contrib-python
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# ----------------- tunable constants -----------------

# The vertical band (fraction of full height) where the barcode lives.
# Based on your coordinates (~0.28–0.32), this band is centered at 0.30
# with a bit of padding on each side.
BAND_Y0_FRAC = 0.22
BAND_Y1_FRAC = 0.38

# Minimum length we consider a “real” barcode.
# All of your examples are long (>= 12–16 chars), so 8 is a safe cutoff.
MIN_BARCODE_LEN = 8

# ----------------- decoders -----------------


def _decode_with_pyzbar_gray(gray: np.ndarray) -> List[str]:
    """
    Run pyzbar on several contrast variants of a grayscale image.
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        from PIL import ImageOps, ImageFilter
    except Exception:
        return []

    hits: List[str] = []

    pil_base = Image.fromarray(gray)

    variants: List[Image.Image] = [pil_base]
    try:
        variants.append(ImageOps.autocontrast(pil_base, cutoff=2))
        variants.append(ImageOps.equalize(pil_base))
        variants.append(pil_base.filter(ImageFilter.GaussianBlur(radius=0.5)))
        variants.append(ImageOps.invert(pil_base))
    except Exception:
        pass

    for im in variants:
        try:
            for r in zbar_decode(im):
                s = r.data.decode("utf-8", "replace")
                if s:
                    hits.append(s)
        except Exception:
            continue

    return hits


def _decode_with_cv(bgr: np.ndarray) -> List[str]:
    """
    OpenCV barcode detector (if available in this build).
    """
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


def _normalize_size(
    bgr: np.ndarray, min_side: int = 700, max_side: int = 2600
) -> np.ndarray:
    """
    Resize image so the longest side is within [min_side, max_side].
    """
    H, W = bgr.shape[:2]
    longest = max(H, W)
    scale = 1.0
    if longest < min_side:
        scale = float(min_side) / float(longest)
    elif longest > max_side:
        scale = float(max_side) / float(longest)
    if np.isclose(scale, 1.0):
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _decode_candidate(bgr: np.ndarray) -> List[str]:
    """
    Run all decoders on a single candidate ROI, returning all raw hits.
    """
    roi = _normalize_size(bgr)
    hits: List[str] = []

    # OpenCV detector first
    hits += _decode_with_cv(roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # pyzbar on gray + Otsu-thresholded gray
    hits += _decode_with_pyzbar_gray(gray)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    hits += _decode_with_pyzbar_gray(thr)

    return hits


def _uniq_sort(vals: List[str]) -> List[str]:
    """Unique + sort by length (longest first), then lexicographically."""
    return sorted(set(vals), key=lambda s: (-len(s), s))


def _best_barcode(hits: List[str]) -> str:
    """
    Choose best barcode from raw hits:
        * strip whitespace
        * drop anything shorter than MIN_BARCODE_LEN
        * prefer longest, then lexicographically
    """
    cleaned = [h.strip() for h in hits if h and len(h.strip()) >= MIN_BARCODE_LEN]
    if not cleaned:
        return ""
    return _uniq_sort(cleaned)[0]


# ----------------- region selection -----------------


def _barcode_band(bgr: np.ndarray) -> np.ndarray:
    """
    Extract the vertical band (fraction of full height) where the barcode lives.

    We *do not* try to isolate the label; we rely on fixed fractions instead,
    since your slabs have very consistent layout.
    """
    H = bgr.shape[0]
    y0 = int(H * BAND_Y0_FRAC)
    y1 = int(H * BAND_Y1_FRAC)
    y0 = max(0, min(H, y0))
    y1 = max(y0 + 1, min(H, y1))
    return bgr[y0:y1, :].copy()


def _split_into_barcode_blobs(band_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Within the band, split into one or more blobs that look like barcodes.

    Returns:
        * the entire band (first), and then
        * up to two tighter crops found via morphology.
    """
    crops: List[np.ndarray] = []

    H, W = band_bgr.shape[:2]
    if H == 0 or W == 0:
        return crops

    # 0) Always include whole band
    crops.append(band_bgr.copy())

    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = 255 - thr  # make bars white for morphology

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    fc = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(fc) == 3:
        _img, contours, _hier = fc
    else:
        contours, _hier = fc

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1.0, h)
        # wide-ish and reasonably tall
        if h > H * 0.35 and w > W * 0.12 and aspect > 2.5:
            boxes.append((x, y, w, h))

    if not boxes:
        # fall back to left/right halves
        mid = W // 2
        boxes = [(0, 0, mid, H), (mid, 0, W - mid, H)]

    boxes.sort(key=lambda b: b[0])  # left-to-right

    for (x, y, w, h) in boxes[:2]:
        pad_x = max(8, int(0.02 * W))
        pad_y = max(8, int(0.10 * H))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = band_bgr[y0:y1, x0:x1].copy()
        crops.append(crop)

    return crops


# ----------------- main API -----------------


def read_single_barcode(image_path: str) -> str:
    """
    Read the single barcode embedded in a slab image.

    Raises:
        FileNotFoundError
        ValueError if the image cannot be read or no barcode is decodable.
    """
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(image_path)

    bgr = cv2.imread(str(p))
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # 1) Extract the vertical barcode band (fixed fraction of height)
    band = _barcode_band(bgr)

    # 2) Generate a small set of candidate crops
    candidates: List[np.ndarray] = []
    candidates.append(band)
    candidates.extend(_split_into_barcode_blobs(band))

    # Also add a slightly thicker band (for extra vertical margin)
    H = bgr.shape[0]
    pad_y = int(H * 0.03)
    y0 = max(0, int(H * BAND_Y0_FRAC) - pad_y)
    y1 = min(H, int(H * BAND_Y1_FRAC) + pad_y)
    thick_band = bgr[y0:y1, :].copy()
    candidates.append(thick_band)

    # Deduplicate candidates by shape + central pixel
    uniq_candidates: List[np.ndarray] = []
    seen_keys = set()
    for roi in candidates:
        if roi.size == 0:
            continue
        Hc, Wc = roi.shape[:2]
        key = (Hc, Wc, tuple(roi[Hc // 2, Wc // 2].tolist()))
        if key not in seen_keys:
            seen_keys.add(key)
            uniq_candidates.append(roi)

    # 3) Decode each candidate; accept only barcodes >= MIN_BARCODE_LEN.
    for roi in uniq_candidates:
        hits = _decode_candidate(roi)
        best = _best_barcode(hits)
        if best:
            return best

    raise ValueError("No decodable barcode found.")


# ----------------- CLI -----------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read the single barcode in an image and print its text."
    )
    ap.add_argument("image", help="Path to the image file.")
    args = ap.parse_args()
    try:
        text = read_single_barcode(args.image)
        print(text)
        # If you prefer JSON output instead, replace with:
        # import json; print(json.dumps({"text": text}))
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
