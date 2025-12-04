#!/usr/bin/env python3
"""
read_barcode.py — robust single-barcode reader for slab images (PCGS/NGC/etc.)

Assumptions for the slab images this script is tuned for:

* The slab image is already in the correct orientation (no rotation needed).
* The image is in color (BGR/RGB, not grayscale).
* The barcode is always located in the top ~40% of the slab image.

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
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


# ----------------- decoders -----------------


def _decode_with_pyzbar_gray(gray: np.ndarray) -> List[str]:
    """
    Run pyzbar on several contrast variants of a grayscale image.
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        from PIL import ImageOps, ImageFilter
    except Exception:
        # pyzbar or its native zbar dependency is missing
        return []

    hits: List[str] = []

    pil_base = Image.fromarray(gray)

    variants: List[Image.Image] = [pil_base]
    try:
        # These are inexpensive and often help on low-contrast labels.
        variants.append(ImageOps.autocontrast(pil_base, cutoff=2))
        variants.append(ImageOps.equalize(pil_base))
        variants.append(pil_base.filter(ImageFilter.GaussianBlur(radius=0.5)))
    except Exception:
        pass

    # Also try inverted grayscale (white bars on black vs black on white)
    try:
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


def _normalize_size(bgr: np.ndarray, min_side: int = 700, max_side: int = 2000) -> np.ndarray:
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
    Run all decoders on a single candidate ROI.

    This is kept reasonably cheap: one resize, OpenCV once, and a small
    number of pyzbar passes.
    """
    roi = _normalize_size(bgr)

    hits: List[str] = []

    # OpenCV's detector first (if present; it's fast).
    hits += _decode_with_cv(roi)
    if hits:
        return hits

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # pyzbar on gray & simple thresholded versions
    hits += _decode_with_pyzbar_gray(gray)
    if hits:
        return hits

    # Otsu binary
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    hits += _decode_with_pyzbar_gray(thr)

    return hits


def _uniq_sort(vals: List[str]) -> List[str]:
    """Unique + sort by length (longest first), then lexicographically."""
    return sorted(set(vals), key=lambda s: (-len(s), s))


# ----------------- slab-specific regioning -----------------


def _top_label_region(bgr: np.ndarray) -> np.ndarray:
    """
    Return top ~40% (covers label area on slabs this script targets).
    """
    H = bgr.shape[0]
    return bgr[: int(H * 0.40)].copy()


def _barcode_strip_roi(label_bgr: np.ndarray) -> np.ndarray:
    """
    Find a horizontal strip in the label with the strongest vertical edges.

    Heuristic:
        * Compute vertical gradients (Sobel X) on the label.
        * For each row, measure mean gradient magnitude.
        * Smooth that 1-D signal and pick the row with maximum energy.
        * Return a band of height ~10% of the label centered there.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.convertScaleAbs(gx)
    row_energy = gx.mean(axis=1)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 9), 0).ravel()

    H = label_bgr.shape[0]
    win = max(20, H // 10)  # ~10% of label height
    center = int(np.argmax(row_energy))
    y0 = max(0, center - win // 2)
    y1 = min(H, y0 + win)
    return label_bgr[y0:y1].copy()


def _barcode_crops_by_morphology(label_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Scan the label for a dense cluster of vertical edges that looks like a
    1D barcode band. Returns up to 2 crops.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)

    # emphasize vertical edges
    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradX = cv2.convertScaleAbs(gradX)
    gradX = cv2.GaussianBlur(gradX, (9, 9), 0)

    _t, thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # connect vertical bars into a single wide component
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(
        closed, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)), iterations=1
    )

    fc = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(fc) == 3:
        _img, contours, _hier = fc
    else:
        contours, _hier = fc

    H, W = gray.shape
    boxes: List[Tuple[int, int, int, int]] = []

    # relaxed thresholds so we don't miss subtle bands (like NCS labels)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(max(h, 1))
        area = w * h
        if aspect > 2.0 and area > 1000 and w > W * 0.10 and h > H * 0.06:
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    # Sort by area (largest first).
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    crops: List[np.ndarray] = []
    for (x, y, w, h) in boxes[:2]:
        pad_x = max(8, int(0.02 * W))
        pad_y = max(8, int(0.05 * H))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = label_bgr[y0:y1, x0:x1].copy()
        crops.append(crop)

    return crops


def _split_into_barcode_blobs(strip_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Within the strip, split into one or more blobs.

    Some slabs (PCGS) have two barcodes; others have one.
    Returns:
        * the entire strip (first), and then
        * up to two tighter crops found via morphology.

    Kept small to avoid excessive decode work.
    """
    crops: List[np.ndarray] = []

    H, W = strip_bgr.shape[:2]

    # Always include whole strip
    crops.append(strip_bgr.copy())

    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)

    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = 255 - thr  # make bars white for morphology

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
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
        if h > H * 0.30 and w > W * 0.06 and aspect > 1.8:
            boxes.append((x, y, w, h))

    if not boxes:
        # fall back to left/right halves
        mid = W // 2
        boxes = [(0, 0, mid, H), (mid, 0, W - mid, H)]

    boxes.sort(key=lambda b: b[0])  # left-to-right

    for (x, y, w, h) in boxes[:2]:
        pad_x = max(8, int(0.02 * W))
        pad_y = max(8, int(0.05 * H))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = strip_bgr[y0:y1, x0:x1].copy()
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

    candidates: List[np.ndarray] = []

    # 1) Label region (top 40% of slab)
    label = _top_label_region(bgr)
    candidates.append(label)

    # 2) Bottom band of the label (where barcodes usually live)
    lh = label.shape[0]
    band_y0 = int(lh * 0.45)
    band = label[band_y0:, :].copy()
    candidates.append(band)

    # 3) Morphological “barcode-like” crops from the whole label
    candidates.extend(_barcode_crops_by_morphology(label))

    # 4) Vertical-edge strip from label, then its internal blobs
    strip = _barcode_strip_roi(label)
    candidates.append(strip)
    candidates.extend(_split_into_barcode_blobs(strip))

    # To avoid redundant work, dedupe by shape + hash of a small patch
    uniq_candidates: List[np.ndarray] = []
    seen_keys = set()
    for roi in candidates:
        if roi.size == 0:
            continue
        H, W = roi.shape[:2]
        key = (H, W, tuple(roi[H // 2, W // 2].tolist()))
        if key not in seen_keys:
            seen_keys.add(key)
            uniq_candidates.append(roi)

    # Decode each candidate in order of "most promising" first:
    # band, morph crops, strip blobs, then full label last.
    # (The above construction already roughly reflects that order.)
    for roi in uniq_candidates:
        hits = _decode_candidate(roi)
        if hits:
            return _uniq_sort(hits)[0]

    raise ValueError("No decodable barcode found.")


# ----------------- CLI -----------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read the single barcode in an image and print its text."
    )
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
