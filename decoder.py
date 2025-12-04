#!/usr/bin/env python3
"""
read_barcode.py — barcode reader for slab images (PCGS/NGC/etc.)

Assumptions:

* Slab image is already in the correct orientation (no rotation needed).
* Image is in color.
* Barcode is always roughly between 26% and 36% of the full image height.

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

# Vertical band (fraction of full height) where the barcode lives.
# Based on your sample (565–655 / 2027 ≈ 0.28–0.32), this brackets that region.
BAND_Y0_FRAC = 0.22
BAND_Y1_FRAC = 0.36

# Minimum length we consider a “real” barcode (ignore junk like "1").
MIN_BARCODE_LEN = 8

# Target minimum ROI size before decoding.
MIN_ROI_HEIGHT = 200
MIN_ROI_WIDTH = 800
MAX_ROI_LONG_SIDE = 2200  # keep ROIs reasonably sized for DO

# ----------------- decoders -----------------


def _decode_with_pyzbar(gray: np.ndarray) -> List[str]:
    """
    Run pyzbar on a single grayscale image.

    IMPORTANT: we explicitly restrict which symbologies zbar tries,
    to avoid buggy DataBar paths and reduce work.
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol
    except Exception:
        return []

    # Restrict to common linear symbologies; NO DataBar.
    SYMBOLS = [
        ZBarSymbol.CODE128,
        ZBarSymbol.I25,
    ]
    # If you know it's always Code128, you can tighten to:
    # SYMBOLS = [ZBarSymbol.CODE128]

    pil_img = Image.fromarray(gray)
    hits: List[str] = []
    try:
        for r in zbar_decode(pil_img, symbols=SYMBOLS):
            s = r.data.decode("utf-8", "replace")
            if s:
                hits.append(s)
    except Exception:
        pass
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


# ----------------- helpers -----------------


def _prepare_roi(bgr: np.ndarray) -> np.ndarray:
    """
    Normalize ROI size with a focus on vertical and horizontal resolution.

    * Ensure height >= MIN_ROI_HEIGHT and width >= MIN_ROI_WIDTH.
    * Ensure longest side does not exceed MAX_ROI_LONG_SIDE.
    """
    H, W = bgr.shape[:2]
    longest = max(H, W)

    scale_h = float(MIN_ROI_HEIGHT) / float(H) if H < MIN_ROI_HEIGHT else 1.0
    scale_w = float(MIN_ROI_WIDTH) / float(W) if W < MIN_ROI_WIDTH else 1.0
    scale = max(scale_h, scale_w, 1.0)

    if longest * scale > MAX_ROI_LONG_SIDE:
        scale = float(MAX_ROI_LONG_SIDE) / float(longest)

    if np.isclose(scale, 1.0):
        return bgr

    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


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
        * up to two tighter crops found via morphology.

    (Full band is handled separately by the caller.)
    """
    crops: List[np.ndarray] = []

    H, W = band_bgr.shape[:2]
    if H == 0 or W == 0:
        return crops

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
        if h > H * 0.35 and w > W * 0.10 and aspect > 2.0:
            boxes.append((x, y, w, h))

    if not boxes:
        return crops

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)  # largest first

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


# ----------------- decoding per ROI -----------------


def _decode_roi(bgr: np.ndarray) -> str:
    """
    Try to decode a single ROI. Returns best barcode text or "".
    """
    roi = _prepare_roi(bgr)

    # 1) Try OpenCV detector once on the color ROI.
    cv_hits = _decode_with_cv(roi)
    best_cv = _best_barcode(cv_hits)
    if best_cv:
        return best_cv

    # 2) Build a small set of grayscale variants and run pyzbar on each.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    variants: List[np.ndarray] = [gray]

    # Equalized gray
    try:
        eq = cv2.equalizeHist(gray)
    except Exception:
        eq = gray
    variants.append(eq)

    # Otsu on equalized
    try:
        thr = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except Exception:
        thr = eq
    variants.append(thr)

    # Inverted Otsu
    variants.append(255 - thr)

    all_hits: List[str] = []
    for g in variants:
        hits = _decode_with_pyzbar(g)
        if hits:
            all_hits.extend(hits)
            best = _best_barcode(all_hits)
            if best:
                return best

    return ""


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

    # 1) Extract the vertical barcode band
    band = _barcode_band(bgr)
    if band.size == 0:
        raise ValueError("Barcode band is empty.")

    # 2) Build a small list of candidate ROIs:
    #    - full band
    #    - up to two morphology-based blobs inside the band
    #    - slightly thicker band for extra vertical margin
    candidates: List[np.ndarray] = []

    # full band
    candidates.append(band)

    # blobs
    candidates.extend(_split_into_barcode_blobs(band))

    # slightly thicker band
    H = bgr.shape[0]
    pad_y = int(H * 0.03)
    yy0 = max(0, int(H * BAND_Y0_FRAC) - pad_y)
    yy1 = min(H, int(H * BAND_Y1_FRAC) + pad_y)
    thick_band = bgr[yy0:yy1, :].copy()
    candidates.append(thick_band)

    # Deduplicate candidates by shape + central pixel (cheap heuristic)
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

    # 3) Try decoding each candidate in order.
    for roi in uniq_candidates:
        text = _decode_roi(roi)
        if text:
            return text

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
        # If you prefer JSON output:
        # import json; print(json.dumps({"text": text}))
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
