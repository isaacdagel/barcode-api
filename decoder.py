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


def _decode_with_pyzbar(pil_img: Image.Image) -> List[str]:
    """
    Run pyzbar on several contrast/denoise variants of a grayscale image.

    Using multiple simple variants is cheap but helps a lot on low-contrast,
    finely-printed labels such as NGC/NCS slabs.
    """
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        from PIL import ImageOps, ImageFilter
    except Exception:
        # pyzbar or its native zbar dependency is missing
        return []

    hits: List[str] = []

    base = pil_img.convert("L")

    variants: List[Image.Image] = [base]
    try:
        # autocontrast and histogram equalization often make the bars "pop"
        variants.append(ImageOps.autocontrast(base, cutoff=2))
        variants.append(ImageOps.equalize(base))
        # a tiny blur can suppress small speckle noise
        variants.append(base.filter(ImageFilter.GaussianBlur(radius=0.5)))
    except Exception:
        # If any of the above fails, we still have at least `base`.
        pass

    for im in variants:
        try:
            for r in zbar_decode(im):
                s = r.data.decode("utf-8", "replace")
                if s:
                    hits.append(s)
        except Exception:
            # Some builds of zbar can be finicky with certain formats; just skip.
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


def _try_all_decoders(bgr: np.ndarray) -> List[str]:
    """
    Try OpenCV + pyzbar on a single ROI.

    This function is intentionally *lightweight* because it will be called
    multiple times on different candidate regions.

    Strategy:
        * Normalize size so the longest side is roughly 700–2000 px.
        * Run OpenCV's BarcodeDetector (if present).
        * Run pyzbar on normal + inverted grayscale variants.
    """
    H, W = bgr.shape[:2]
    longest = max(H, W)
    scale = 1.0
    if longest < 700:
        scale = 700.0 / float(longest)
    elif longest > 2000:
        scale = 2000.0 / float(longest)
    if not np.isclose(scale, 1.0):
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    hits: List[str] = []

    # OpenCV (if available)
    hits += _decode_with_cv(bgr)

    # pyzbar on grayscale + inverted grayscale
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    hits += _decode_with_pyzbar(pil)

    try:
        from PIL import ImageOps
        hits += _decode_with_pyzbar(ImageOps.invert(pil))
    except Exception:
        pass

    return hits


def _uniq_sort(vals: List[str]) -> List[str]:
    """Unique + sort by length (longest first), then lexicographically."""
    return sorted(set(vals), key=lambda s: (-len(s), s))


# ----------------- slab-specific regioning -----------------


def _top_label_region(bgr: np.ndarray) -> np.ndarray:
    """
    Return top ~40% (covers label area on the slabs this script targets).

    The caller has guaranteed that the barcode is in this band.
    """
    H = bgr.shape[0]
    return bgr[: int(H * 0.40)].copy()


def _barcode_strip_roi(label_bgr: np.ndarray) -> np.ndarray:
    """
    Find a horizontal strip likely to contain the barcode.

    Heuristic:
        * Compute vertical gradients (Sobel X) on the label.
        * For each row, measure mean gradient magnitude.
        * Smooth that 1-D signal and pick the row with maximum energy.
        * Return a band of height ~10–15% of the label, centered there.
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


def _split_into_barcode_blobs(strip_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Split the strip into one or more blobs.

    Some slabs (PCGS) have two barcodes; others (NGC/NCS) have one.
    This routine returns:
        * The whole strip (first), and then
        * up to two tighter crops found via morphology.

    The number of returned crops is small by design to keep decoding fast.
    """
    crops: List[np.ndarray] = []

    H, W = strip_bgr.shape[:2]

    # Always include the whole strip as a candidate
    crops.append(strip_bgr.copy())

    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY)

    # strong binarization to emphasize bars
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = 255 - thr  # make bars white for morphology

    # connect vertical strokes into components
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
        # fall back to left/right halves to be safe
        mid = W // 2
        boxes = [(0, 0, mid, H), (mid, 0, W - mid, H)]

    boxes.sort(key=lambda b: b[0])  # left-to-right

    for (x, y, w, h) in boxes[:2]:  # at most 2 additional crops
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
    Top-level helper: read the single barcode embedded in a slab image.

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

    # 1) Restrict work to the label region (top 40% of the slab).
    label = _top_label_region(bgr)

    # 2) Quick attempt on the entire label.
    hits = _try_all_decoders(label)
    if hits:
        return _uniq_sort(hits)[0]

    # 3) Focus on the horizontal barcode strip inside the label.
    strip = _barcode_strip_roi(label)

    # 4) Split the strip into a small number of candidate blobs and test each.
    crops = _split_into_barcode_blobs(strip)

    for crop in crops:
        hits = _try_all_decoders(crop)
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
