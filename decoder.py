#!/usr/bin/env python3
"""
read_barcode.py — robust single-barcode reader for slab images (PCGS/NGC/etc.)

Assumptions:

* Slab image is already in the correct orientation (no rotation needed).
* Image is in color.
* Barcode is always in the top ~40% of the image (label region).

Strategy:

1. Take the top ~40% of the image as the label area.
2. Within that label, compute vertical gradient energy per row and pick
   the top few peak rows (where 1D barcodes tend to live).
3. Around each peak row, extract a horizontal strip (~10–12% of label height).
4. For each strip:
   - Try decoding the whole strip.
   - Use morphology to find wide "barcode-like" blobs inside the strip and
     try decoding up to two tight blobs.
5. Decoding uses:
   - OpenCV's BarcodeDetector (if available).
   - pyzbar on a few contrast/threshold variants of the ROI.
   - Only linear symbologies (no DataBar) and a minimum length filter
     to avoid junk reads like "1".

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

# Label region: top fraction of the image height.
LABEL_TOP_FRAC = 0.40

# Horizontal strips: how many candidate peak rows in the label to use.
MAX_STRIPS = 3
# Minimum vertical distance between two chosen peak rows, as fraction of label height
PEAK_SEPARATION_FRAC = 0.15

# Horizontal strip height as fraction of label height.
STRIP_HEIGHT_FRAC = 0.12  # ~12% of label height

# Minimum length we consider a “real” barcode (ignore tiny junk like "1").
MIN_BARCODE_LEN = 8

# Target minimum ROI size before decoding.
MIN_ROI_HEIGHT = 200
MIN_ROI_WIDTH = 800
MAX_ROI_LONG_SIDE = 2200  # keep ROIs reasonably sized

# ----------------- decoders -----------------


def _decode_with_pyzbar(gray: np.ndarray) -> List[str]:
    """
    Run pyzbar on a single grayscale image.

    IMPORTANT: we explicitly restrict which symbologies zbar tries
    to avoid buggy DataBar paths and reduce work.
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol
    except Exception:
        return []

    # Restrict to common linear symbologies; NO DataBar.
    SYMBOLS = [
        ZBarSymbol.CODE128,
        ZBarSymbol.CODE39,
        ZBarSymbol.CODE93,
        ZBarSymbol.I25,     # Interleaved 2 of 5
        ZBarSymbol.EAN13,
        ZBarSymbol.EAN8,
        ZBarSymbol.UPCA,
        ZBarSymbol.UPCE,
        ZBarSymbol.CODABAR,
    ]
    # If you know these slabs are always Code128, you can tighten to:
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


# ----------------- label & strip selection -----------------


def _top_label_region(bgr: np.ndarray) -> np.ndarray:
    """
    Return the top LABEL_TOP_FRAC of the image height as the label region.
    """
    H = bgr.shape[0]
    return bgr[: int(H * LABEL_TOP_FRAC)].copy()


def _find_strip_centers(label_bgr: np.ndarray) -> List[int]:
    """
    Find up to MAX_STRIPS row indices (within the label) where vertical
    edge energy is highest, enforcing separation so we don't pick the same
    area repeatedly.

    Returns list of row indices (relative to label_bgr).
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.convertScaleAbs(gx)
    row_energy = gx.mean(axis=1)
    # smooth the 1D signal a bit
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 9), 0).ravel()

    H = label_bgr.shape[0]
    indices = np.argsort(row_energy)[::-1]  # highest energy first

    centers: List[int] = []
    min_sep = int(H * PEAK_SEPARATION_FRAC)

    for idx in indices:
        if len(centers) >= MAX_STRIPS:
            break
        if any(abs(idx - c) < min_sep for c in centers):
            continue
        centers.append(int(idx))

    centers.sort()
    return centers


def _extract_strip(label_bgr: np.ndarray, center_row: int) -> np.ndarray:
    """
    Extract a horizontal strip around center_row within the label.
    """
    H = label_bgr.shape[0]
    win = max(24, int(H * STRIP_HEIGHT_FRAC))
    y0 = max(0, center_row - win // 2)
    y1 = min(H, y0 + win)
    return label_bgr[y0:y1].copy()


def _split_strip_into_blobs(strip_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Within a strip, find up to two tight "barcode-like" blobs via morphology.
    """
    crops: List[np.ndarray] = []

    H, W = strip_bgr.shape[:2]
    if H == 0 or W == 0:
        return crops

    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)

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

    # Largest areas first
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    for (x, y, w, h) in boxes[:2]:
        pad_x = max(8, int(0.02 * W))
        pad_y = max(8, int(0.10 * H))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = strip_bgr[y0:y1, x0:x1].copy()
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

    # 1) Restrict to the label region (top ~40%).
    label = _top_label_region(bgr)
    if label.size == 0:
        raise ValueError("Label region is empty.")

    # 2) Find up to MAX_STRIPS candidate strip centers by vertical edge energy.
    centers = _find_strip_centers(label)
    if not centers:
        raise ValueError("Could not find any candidate barcode strips.")

    # 3) Build candidate ROIs:
    #    - the strips themselves
    #    - up to two morphology-based blobs inside each strip
    candidates: List[np.ndarray] = []
    for c in centers:
        strip = _extract_strip(label, c)
        if strip.size == 0:
            continue
        candidates.append(strip)
        candidates.extend(_split_strip_into_blobs(strip))

    # As a final fallback, add the whole label (just in case our strips miss).
    candidates.append(label)

    # 4) Deduplicate candidates by shape + central pixel (cheap heuristic)
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

    # 5) Try decoding each candidate in order.
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
