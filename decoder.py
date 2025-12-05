#!/usr/bin/env python3
"""
read_barcode.py — Enhanced barcode reader for slab images (PCGS/NGC/etc.)

Improvements:
- Expanded preprocessing variants (adaptive threshold, bilateral filter, CLAHE, etc.)
- Multiple scale factors for decoding
- Small rotation tolerance
- Relaxed blob detection parameters
- Sharpening filter
- Full-image fallback with priority search
- Maintains high reliability with broader coverage

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
from typing import List, Tuple, Set

import cv2
import numpy as np
from PIL import Image

# ----------------- tunable constants -----------------

# Primary search band (fraction of full height) where barcode is most likely
# Adjusted to cover both upper label area and middle band
PRIMARY_Y0_FRAC = 0.08
PRIMARY_Y1_FRAC = 0.42

# Secondary search bands to check before full-image fallback
SECONDARY_BANDS = [
    (0.40, 0.55),  # Lower region
    (0.55, 0.70),  # Even lower
    (0.02, 0.15),  # Very top edge (some NCS labels)
    (0.70, 0.85),  # Bottom area
]

# Minimum length we consider a "real" barcode (ignore junk like "1").
MIN_BARCODE_LEN = 8

# Target minimum ROI size before decoding.
MIN_ROI_HEIGHT = 200
MIN_ROI_WIDTH = 800
MAX_ROI_LONG_SIDE = 2200

# Scale factors to try
SCALE_FACTORS = [1.0, 1.5, 2.0, 0.75, 2.5]

# Small rotation angles to try (degrees)
ROTATION_ANGLES = [0, -2, 2, -5, 5, -1, 1, -3, 3]

# For full-image fallback: use fewer variants to maintain speed
SCALE_FACTORS_FULLIMG = [1.0, 1.5, 2.0]
ROTATION_ANGLES_FULLIMG = [0, -2, 2]

# ----------------- decoders -----------------


def _decode_with_pyzbar(gray: np.ndarray) -> List[str]:
    """
    Run pyzbar on a single grayscale image.
    Restricts symbologies to avoid buggy DataBar paths.
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


# ----------------- preprocessing -----------------


def _sharpen_image(gray: np.ndarray) -> np.ndarray:
    """Sharpen for better edge definition."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(gray, -1, kernel)


def _generate_preprocessing_variants(gray: np.ndarray, thorough: bool = True) -> List[np.ndarray]:
    """
    Generate preprocessing variants for difficult barcodes.
    
    Args:
        gray: Input grayscale image
        thorough: If True, generate all variants. If False, use faster subset.
    """
    variants: List[np.ndarray] = []
    
    # Original gray
    variants.append(gray)
    
    # Sharpened
    try:
        sharpened = _sharpen_image(gray)
        variants.append(sharpened)
    except Exception:
        sharpened = gray
    
    # Equalized histogram
    try:
        eq = cv2.equalizeHist(gray)
        variants.append(eq)
    except Exception:
        eq = gray
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(clahe_img)
    except Exception:
        clahe_img = gray
    
    if thorough:
        # Bilateral filter (reduces noise while preserving edges)
        try:
            bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
            variants.append(bilateral)
        except Exception:
            bilateral = gray
        
        # Adaptive thresholding (better for uneven lighting)
        try:
            adaptive_mean = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            variants.append(adaptive_mean)
            variants.append(255 - adaptive_mean)
            
            adaptive_gauss = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            variants.append(adaptive_gauss)
            variants.append(255 - adaptive_gauss)
        except Exception:
            pass
        
        # Morphological gradient (emphasizes bar edges)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            variants.append(gradient)
        except Exception:
            pass
        
        # Otsu thresholding on multiple inputs
        for img in [gray, eq, bilateral, clahe_img, sharpened]:
            try:
                _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variants.append(otsu)
                variants.append(255 - otsu)
            except Exception:
                pass
    else:
        # Fast subset for full-image fallback
        try:
            _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(otsu)
            variants.append(255 - otsu)
        except Exception:
            pass
    
    return variants


# ----------------- helpers -----------------


def _prepare_roi(bgr: np.ndarray) -> np.ndarray:
    """
    Normalize ROI size with a focus on vertical and horizontal resolution.
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
    Choose best barcode from raw hits.
    Prefers longer barcodes (slab IDs are typically 13-16 chars).
    """
    cleaned = [h.strip() for h in hits if h and len(h.strip()) >= MIN_BARCODE_LEN]
    if not cleaned:
        return ""
    
    # Prefer barcodes >= 13 characters (typical slab ID length)
    long_codes = [c for c in cleaned if len(c) >= 13]
    if long_codes:
        return _uniq_sort(long_codes)[0]
    
    return _uniq_sort(cleaned)[0]


# ----------------- region selection -----------------


def _extract_band(bgr: np.ndarray, y0_frac: float, y1_frac: float) -> np.ndarray:
    """
    Extract a horizontal band from the image.
    """
    H = bgr.shape[0]
    y0 = int(H * y0_frac)
    y1 = int(H * y1_frac)
    y0 = max(0, min(H, y0))
    y1 = max(y0 + 1, min(H, y1))
    return bgr[y0:y1, :].copy()


def _split_into_barcode_blobs(band_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Within the band, split into one or more blobs that look like barcodes.
    Uses multiple morphology kernel sizes for better detection.
    """
    crops: List[np.ndarray] = []
    H, W = band_bgr.shape[:2]
    if H == 0 or W == 0:
        return crops

    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thr = 255 - thr

    # Try multiple kernel sizes
    all_boxes: List[Tuple[int, int, int, int]] = []
    for ksize in [(25, 3), (20, 3), (30, 3), (15, 2), (35, 4)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

        fc = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(fc) == 3:
            _img, contours, _hier = fc
        else:
            contours, _hier = fc

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(1.0, h)
            # Relaxed constraints
            if h > H * 0.25 and w > W * 0.08 and aspect > 1.5:
                all_boxes.append((x, y, w, h))

    if not all_boxes:
        return crops

    # Deduplicate boxes that are very similar
    unique_boxes: List[Tuple[int, int, int, int]] = []
    seen: Set[Tuple[int, int, int, int]] = set()
    for box in all_boxes:
        # Round to nearest 10 pixels for deduplication
        key = tuple(round(v / 10) * 10 for v in box)
        if key not in seen:
            seen.add(key)
            unique_boxes.append(box)

    unique_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

    for (x, y, w, h) in unique_boxes[:3]:  # Take top 3
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


def _try_decode_single_roi(bgr: np.ndarray, thorough: bool = True) -> str:
    """
    Try to decode a single ROI at its current size/rotation.
    
    Args:
        bgr: BGR image ROI
        thorough: If True, use all preprocessing. If False, use faster subset.
    """
    roi = _prepare_roi(bgr)

    # 1) Try OpenCV detector on color ROI
    cv_hits = _decode_with_cv(roi)
    best_cv = _best_barcode(cv_hits)
    if best_cv:
        return best_cv

    # 2) Generate preprocessing variants
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variants = _generate_preprocessing_variants(gray, thorough=thorough)

    all_hits: List[str] = []
    for g in variants:
        hits = _decode_with_pyzbar(g)
        if hits:
            all_hits.extend(hits)
            best = _best_barcode(all_hits)
            if best:
                return best

    return ""


def _try_with_rotation(bgr: np.ndarray, angles: List[int], thorough: bool = True) -> str:
    """
    Try decoding with small rotation angles.
    """
    H, W = bgr.shape[:2]
    center = (W // 2, H // 2)

    for angle in angles:
        if angle != 0:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(bgr, M, (W, H),
                                     borderMode=cv2.BORDER_REPLICATE)
        else:
            rotated = bgr

        result = _try_decode_single_roi(rotated, thorough=thorough)
        if result:
            return result

    return ""


def _decode_roi(bgr: np.ndarray, scales: List[float], angles: List[int], thorough: bool = True) -> str:
    """
    Try to decode a single ROI with multiple scales and rotations.
    
    Args:
        bgr: BGR image ROI
        scales: Scale factors to try
        angles: Rotation angles to try
        thorough: If True, use all preprocessing. If False, use faster subset.
    """
    # Try multiple scale factors
    for scale in scales:
        if scale != 1.0:
            scaled = cv2.resize(bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
        else:
            scaled = bgr

        # For each scale, try rotations
        result = _try_with_rotation(scaled, angles, thorough=thorough)
        if result:
            return result

    return ""


# ----------------- main API -----------------


def read_single_barcode(image_path: str) -> str:
    """
    Read the single barcode embedded in a slab image.
    
    Strategy:
    1. Search primary band (20-42%) with full processing
    2. Search secondary bands with full processing
    3. Fall back to full image with reduced processing
    
    This maintains high reliability while covering the entire image.

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

    H = bgr.shape[0]

    # ========== Phase 1: Primary band (highest probability) ==========
    band = _extract_band(bgr, PRIMARY_Y0_FRAC, PRIMARY_Y1_FRAC)
    if band.size > 0:
        candidates: List[np.ndarray] = []
        
        # Full band
        candidates.append(band)
        
        # Morphology-based blobs
        candidates.extend(_split_into_barcode_blobs(band))
        
        # Slightly thicker band
        pad_y = int(H * 0.03)
        yy0 = max(0, int(H * PRIMARY_Y0_FRAC) - pad_y)
        yy1 = min(H, int(H * PRIMARY_Y1_FRAC) + pad_y)
        thick_band = bgr[yy0:yy1, :].copy()
        candidates.append(thick_band)
        
        # Even thicker band
        pad_y_large = int(H * 0.05)
        yy0_large = max(0, int(H * PRIMARY_Y0_FRAC) - pad_y_large)
        yy1_large = min(H, int(H * PRIMARY_Y1_FRAC) + pad_y_large)
        thick_band_large = bgr[yy0_large:yy1_large, :].copy()
        candidates.append(thick_band_large)
        
        # Deduplicate candidates
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
        
        # Collect ALL barcodes from all candidates, then pick best
        all_phase1_hits: List[str] = []
        for roi in uniq_candidates:
            text = _decode_roi(roi, SCALE_FACTORS, ROTATION_ANGLES, thorough=True)
            if text:
                all_phase1_hits.append(text)
        
        # Choose the best barcode from all Phase 1 results
        if all_phase1_hits:
            best = _best_barcode(all_phase1_hits)
            if best:
                return best

    # ========== Phase 2: Secondary bands ==========
    for y0_frac, y1_frac in SECONDARY_BANDS:
        band = _extract_band(bgr, y0_frac, y1_frac)
        if band.size > 0:
            # Try full band and blobs with full processing
            candidates = [band]
            candidates.extend(_split_into_barcode_blobs(band))
            
            for roi in candidates:
                if roi.size == 0:
                    continue
                text = _decode_roi(roi, SCALE_FACTORS, ROTATION_ANGLES, thorough=True)
                if text:
                    return text

    # ========== Phase 3: Full image fallback (reduced processing) ==========
    # Use fewer scales/rotations and faster preprocessing to maintain speed
    text = _decode_roi(bgr, SCALE_FACTORS_FULLIMG, ROTATION_ANGLES_FULLIMG, thorough=False)
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
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
