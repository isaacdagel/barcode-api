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
    Run pyzbar on several contrast/denoise variants of the image.

    This helps a lot on low-contrast, finely-printed labels like NGC/NCS.
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
    Try both OpenCV & pyzbar on normal and inverted images,
    with a light size normalization.
    """
    out: List[str] = []

    H, W = bgr.shape[:2]
    longest = max(H, W)
    scale = 1.0
    if longest < 700:
        scale = 700.0 / float(longest)
    elif longest > 2600:
        scale = 2600.0 / float(longest)
    if not np.isclose(scale, 1.0):
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    for img in (bgr, 255 - bgr):  # normal + inverted
        # OpenCV (if available)
        out += _decode_with_cv(img)

        # pyzbar
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out += _decode_with_pyzbar(pil)

    return out


def _uniq_sort(vals: List[str]) -> List[str]:
    """Unique + sort by length (longest first), then lexicographically."""
    return sorted(set(vals), key=lambda s: (-len(s), s))


# ----------------- slab-specific regioning -----------------


def _top_label_region(bgr: np.ndarray) -> np.ndarray:
    """Return top ~55% (covers label area on most slabs)."""
    H = bgr.shape[0]
    return bgr[: int(H * 0.55)].copy()


def _barcode_strip_roi(label_bgr: np.ndarray) -> np.ndarray:
    """
    Original heuristic: find the horizontal strip with the most vertical edges.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gx = cv2.convertScaleAbs(gx)
    row_energy = gx.mean(axis=1)
    row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (1, 9), 0).ravel()

    H = label_bgr.shape[0]
    win = max(24, H // 12)  # ~8–10% label height
    center = int(np.argmax(row_energy))
    y0 = max(0, center - win // 2)
    y1 = min(H, y0 + win)
    return label_bgr[y0:y1].copy()


def _candidate_label_strips(label_bgr: np.ndarray) -> List[np.ndarray]:
    """
    For tricky slabs (e.g., NGC/NCS), use multiple horizontal strips
    instead of relying on a single Sobel-based one.
    """
    H = label_bgr.shape[0]
    strips: List[np.ndarray] = []

    # 1) Original “max vertical edges” strip
    strips.append(_barcode_strip_roi(label_bgr))

    # 2) Central band
    y0 = int(H * 0.30)
    y1 = int(H * 0.70)
    strips.append(label_bgr[y0:y1].copy())

    # 3) Lower band of the label (barcodes often sit here)
    y0 = int(H * 0.45)
    y1 = int(H * 0.95)
    strips.append(label_bgr[y0:y1].copy())

    # Ensure uniqueness by shape
    uniq: List[np.ndarray] = []
    seen_shapes = set()
    for s in strips:
        key = s.shape
        if key not in seen_shapes:
            seen_shapes.add(key)
            uniq.append(s)

    return uniq


def _split_into_barcode_blobs(strip_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Split the strip into one or more blobs (PCGS often has 2 barcodes).

    Always returns at least one crop. The first crop is the whole strip,
    so decoders still get a chance even if contour detection fails.
    """
    crops: List[np.ndarray] = []

    H, W = strip_bgr.shape[:2]

    # Always include the whole strip as a candidate
    base_strip = strip_bgr.copy()
    if max(H, W) < 1800:
        base_strip = cv2.resize(
            base_strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
        )
    crops.append(base_strip)

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

    # left-to-right
    boxes.sort(key=lambda b: b[0])

    for (x, y, w, h) in boxes[:3]:
        # generous padding to preserve quiet zones
        pad_x = max(8, int(0.02 * W))
        pad_y = max(8, int(0.05 * H))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = strip_bgr[y0:y1, x0:x1].copy()

        if max(crop.shape[:2]) < 1800:
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crops.append(crop)

    return crops


# -------- NEW: full-label morphological barcode detector ---------


def _barcode_crops_by_morphology(label_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Scan the entire label for a dense cluster of vertical edges that
    looks like a 1D barcode. This is especially helpful for NGC/NCS slabs
    where the barcode is a small band at the very bottom of the label.
    """
    gray = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2GRAY)

    # emphasize vertical edges
    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradX = cv2.convertScaleAbs(gradX)
    gradX = cv2.GaussianBlur(gradX, (9, 9), 0)

    # binarize based on edge strength
    _t, thresh = cv2.threshold(
        gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # connect vertical strokes into one big component
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # slightly dilate vertically so the box covers the whole bar height
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

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(max(h, 1))
        area = w * h
        # wide, reasonably tall strip somewhere in the label
        if (
            aspect > 2.5
            and area > 2000
            and w > W * 0.15
            and h > H * 0.08
        ):
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    # Sort by area (largest first) – the main barcode band wins.
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    crops: List[np.ndarray] = []
    for (x, y, w, h) in boxes[:3]:
        pad_x = max(8, int(0.02 * W))
        pad_y = max(8, int(0.05 * H))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)
        crop = label_bgr[y0:y1, x0:x1].copy()
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

    rotations = (
        None,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_180,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
    )

    # 0) Quick whole-frame tries (sometimes enough)
    for rot in rotations:
        arr = bgr if rot is None else cv2.rotate(bgr, rot)
        hits = _try_all_decoders(arr)
        if hits:
            return _uniq_sort(hits)[0]

    # 1) Focus on the top label
    label = _top_label_region(bgr)

    # 1a) NEW: full-label morphological barcode detector
    morph_crops = _barcode_crops_by_morphology(label)
    for crop in morph_crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        variants = [
            crop,
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1],
            cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                35,
                11,
            ),
        ]
        for v in variants:
            for rot in rotations:
                vv = v if rot is None else cv2.rotate(v, rot)
                hits = _try_all_decoders(vv)
                if hits:
                    return _uniq_sort(hits)[0]

    # 1b) Fallback: strip-based logic (original + improved)
    strips = _candidate_label_strips(label)

    for strip in strips:
        # First try decoding the whole strip as-is
        for rot in rotations:
            arr = strip if rot is None else cv2.rotate(strip, rot)
            hits = _try_all_decoders(arr)
            if hits:
                return _uniq_sort(hits)[0]

        # Then split into barcode blobs
        crops = _split_into_barcode_blobs(strip)

        # Decode each crop with several preprocess variants
        for crop in crops:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            variants = [
                crop,
                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )[1],
                cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    35,
                    11,
                ),
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
