#!/usr/bin/env python3
"""
read_barcode.py — robust single-barcode reader for slab images (PCGS/NGC/etc.)

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
BAND_Y0_FRAC = 0.22
BAND_Y1_FRAC = 0.36

# Minimum length we consider a “real” barcode (ignore junk like "1").
MIN_BARCODE_LEN = 8

# Target minimum ROI size before decoding.
MIN_ROI_HEIGHT = 220
MIN_ROI_WIDTH = 900
MAX_ROI_LONG_SIDE = 2800

# Micro-rotations (degrees) to tolerate small skew.
MICRO_ROTATIONS = [0.0, -4.0, 4.0, -8.0, 8.0]


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


# ----------------- image helpers -----------------


def _rotate_bound(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image by angle (in degrees) keeping entire image in view.
    """
    if abs(angle_deg) < 1e-3:
        return image

    (h, w) = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    # compute the new bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2.0) - center[0]
    M[1, 2] += (nH / 2.0) - center[1]

    return cv2.warpAffine(
        image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


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


def _decode_single_roi(bgr: np.ndarray) -> List[str]:
    """
    Decode one upright ROI (no extra rotations) and return all hits.
    """
    roi = _prepare_roi(bgr)
    hits: List[str] = []

    # OpenCV detector first (cheap)
    hits += _decode_with_cv(roi)

    gray_base = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Base grayscale variants: raw, equalized, CLAHE
    variants: List[np.ndarray] = [gray_base]

    try:
        eq = cv2.equalizeHist(gray_base)
        variants.append(eq)
    except Exception:
        eq = gray_base

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray_base)
        variants.append(clahe_img)
    except Exception:
        pass

    all_gray_variants: List[np.ndarray] = []
    for g in variants:
        all_gray_variants.append(g)

        # Otsu threshold
        try:
            thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            all_gray_variants.append(thr)
        except Exception:
            pass

        # Adaptive threshold
        try:
            ga = cv2.adaptiveThreshold(
                g,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                7,
            )
            all_gray_variants.append(ga)
        except Exception:
            pass

    for g in all_gray_variants:
        hits += _decode_with_pyzbar_gray(g)

    return hits


def _decode_candidate(bgr: np.ndarray) -> List[str]:
    """
    Run decoders on a ROI plus a few small-angle rotations, return all hits.
    """
    hits: List[str] = []
    for ang in MICRO_ROTATIONS:
        rot = _rotate_bound(bgr, ang)
        hits += _decode_single_roi(rot)
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
    """
    H = bgr.shape[0]
    y0 = int(H * BAND_Y0_FRAC)
    y1 = int(H * BAND_Y1_FRAC)
    y0 = max(0, min(H, y0))
    y1 = max(y0 + 1, min(H, y1))
    return bgr[y0:y1, :].copy()


def _generate_band_crops(band_bgr: np.ndarray) -> List[np.ndarray]:
    """
    From the barcode band, generate a small set of horizontally-focused crops:
        * full band
        * up to two morphological "blob" crops
        * several overlapping horizontal windows (sliding)
    """
    crops: List[np.ndarray] = []

    H, W = band_bgr.shape[:2]
    if H == 0 or W == 0:
        return crops

    # 0) Full band
    crops.append(band_bgr.copy())

    # 1) Morphological blobs looking for wide vertical-edge regions
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
        if h > H * 0.35 and w > W * 0.10 and aspect > 2.0:
            boxes.append((x, y, w, h))

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

    # 2) Sliding horizontal windows across the band
    #    This helps isolate the barcode away from label text on the sides.
    win_frac = 0.55  # window width as fraction of band width
    step_frac = 0.25  # step as fraction of band width
    win_w = int(W * win_frac)
    win_w = max(win_w, int(W * 0.35))  # don't get *too* narrow
    step = max(1, int(W * step_frac))

    for x0 in range(0, max(1, W - win_w + 1), step):
        x1 = x0 + win_w
        if x1 > W:
            x1 = W
            x0 = max(0, W - win_w)
        crop = band_bgr[:, x0:x1].copy()
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

    # 1) Extract the vertical barcode band
    band = _barcode_band(bgr)

    # 2) Generate candidate crops from that band
    candidates: List[np.ndarray] = _generate_band_crops(band)

    # 3) Add a slightly thicker band for extra vertical margin
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

    # 4) Decode each candidate; accept only barcodes >= MIN_BARCODE_LEN.
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
        # If you prefer JSON output:
        # import json; print(json.dumps({"text": text}))
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
