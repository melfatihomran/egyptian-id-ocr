"""
OpenCV preprocessing pipeline for Egyptian National ID card images.

Stages:
  1. detect_card_contour  - find the card's quadrilateral in a real-world photo
  2. perspective_correct  - warp the detected quad to a flat top-down view
  3. deskew               - fine rotation correction using text-line angle
  4. denoise_and_binarize - grayscale, denoise, adaptive threshold

Each stage is exposed individually (useful for the README's before/after
images and for unit testing) as well as via the single `preprocess()`
convenience function that runs the full pipeline.
"""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class PreprocessResult:
    original: np.ndarray
    contour_overlay: np.ndarray       # original image with detected quad drawn
    warped: np.ndarray                # perspective-corrected, still color
    deskewed: np.ndarray              # after fine rotation correction
    binarized: np.ndarray             # final grayscale/binarized output
    card_quad: np.ndarray | None      # the 4 detected corners, or None if not found
    used_fallback: bool               # True if contour detection failed and we
                                       # fell back to using the full image


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as TL, TR, BR, BL.

    Previously used an independent sum/diff heuristic per corner (smallest
    x+y -> TL, largest x+y -> BR, etc). That breaks down for elongated or
    steeply-angled quads (common on "heavy" severity samples shot at a
    raking angle): two different heuristics can both point to the *same*
    physical corner, so e.g. TR and BR end up assigned the same point and
    the other corner is dropped entirely - collapsing the quad and silently
    producing a garbage (near-zero-area or duplicated-point) result instead
    of an error. This was found via IoU evaluation against ground truth
    (scripts/evaluate_preprocessing.py) - several "confident" detections on
    heavy samples had near-zero IoU despite the underlying contour actually
    being a reasonable match.

    Fix: sort points by angle around their centroid first. This always
    produces a valid non-self-intersecting cyclic order (it's a bijection -
    each point gets a distinct angle slot), then we just need to pick a
    consistent starting corner and rotation direction, which is safe to do
    after the cyclic order is already guaranteed valid.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    cyclic = pts[np.argsort(angles)]

    # Ensure clockwise order in image coordinates (y increases downward):
    # TL -> TR -> BR -> BL should go right, then down, then left, then up.
    signed_area = sum(
        cyclic[i][0] * cyclic[(i + 1) % 4][1] - cyclic[(i + 1) % 4][0] * cyclic[i][1]
        for i in range(4)
    )
    if signed_area < 0:
        cyclic = cyclic[::-1]

    # Rotate so the corner closest to the top-left (smallest x+y) comes first.
    start = int(np.argmin(cyclic.sum(axis=1)))
    ordered = np.roll(cyclic, -start, axis=0)
    return ordered


def _blob_candidates(signal: np.ndarray, image_area: int, w: int, h: int):
    """
    Runs Otsu threshold + largest-connected-blob extraction on a single
    grayscale-like signal. Returns a list of (area, extent, quad) tuples for
    every polarity that passes basic plausibility checks. `extent` is
    contour_area / minAreaRect_area: close to 1.0 for a solid quadrilateral
    blob, much lower (~0.5) for a triangular/wedge-shaped blob - which is
    exactly the shape produced by the gradient-split failure mode described
    below, so it doubles as a cheap "does this actually look like a card"
    sanity check.
    """
    blurred = cv2.GaussianBlur(signal, (7, 7), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = []
    for candidate_mask in (otsu, cv2.bitwise_not(otsu)):
        mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if not (image_area * 0.10 <= area <= image_area * 0.80):
            continue
        x, y, bw, bh = cv2.boundingRect(largest)
        margin = 2
        touches_all_edges = (x <= margin and y <= margin and
                              x + bw >= w - margin and y + bh >= h - margin)
        if touches_all_edges:
            continue

        rect = cv2.minAreaRect(largest)
        rect_area = rect[1][0] * rect[1][1]
        extent = area / rect_area if rect_area > 0 else 0.0

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            quad = cv2.boxPoints(rect).astype(np.float32)
        candidates.append((area, extent, quad))
    return candidates


def detect_card_contour(img_bgr: np.ndarray, debug=False):
    """
    Finds the card's quadrilateral in the image.
    Returns ordered (TL,TR,BR,BL) float32 array, or None if not confidently found.

    Primary strategy: Otsu threshold on a brightness/saturation signal to
    isolate the card as one solid connected blob against the background,
    then take its largest contour. Using max(grayscale, saturation) rather
    than grayscale alone matters because a colored (e.g. navy) header can
    have almost the same luminance as a dark background while being far
    more saturated - grayscale-only thresholding would merge it into the
    background and crop it off.

    Known failure mode this guards against: under strong directional
    lighting (the "heavy" degradation tier adds a synthetic light gradient
    across the card), a single global Otsu threshold can end up splitting
    the CARD ITSELF along its own brightness gradient instead of along the
    true card/background boundary - producing a wedge-shaped blob that
    covers only part of the card but still passes naive area/edge-touching
    checks. This was caught via IoU evaluation against ground-truth corners
    (see scripts/evaluate_preprocessing.py) showing a cluster of confident-
    looking but wrong detections concentrated in heavy-severity samples.

    Fix: compute candidates from both the raw signal AND a CLAHE-equalized
    version (which flattens broad lighting gradients while preserving the
    sharp card/background edge), then among all candidates prefer ones with
    high "extent" (contour area / its own minAreaRect area) - a true card
    blob is solidly rectangular (extent ~0.9-1.0), while a gradient-split
    wedge is triangular (extent ~0.5) - and take the largest-area one that
    passes that shape check.

    Falls back to a combined grayscale+saturation Canny edge approach if no
    blob candidate is found at all (e.g. very low global contrast between
    card and background - a separate, harder failure mode not fixed by the
    above, currently a documented limitation rather than solved).
    """
    h, w = img_bgr.shape[:2]
    image_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    combined_raw = np.maximum(gray, (sat.astype(np.float32) * 0.7).astype(np.uint8))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    combined_eq = np.maximum(gray_eq, (sat.astype(np.float32) * 0.7).astype(np.uint8))

    candidates = _blob_candidates(combined_raw, image_area, w, h)
    candidates += _blob_candidates(combined_eq, image_area, w, h)

    if candidates:
        EXTENT_THRESHOLD = 0.75
        shaped_ok = [c for c in candidates if c[1] >= EXTENT_THRESHOLD]
        pool = shaped_ok if shaped_ok else candidates
        pool.sort(key=lambda c: c[0], reverse=True)  # largest area first
        return _order_quad_points(pool[0][2])

    # --- Fallback: Canny edges on the combined signal + separate saturation pass ---
    blurred = cv2.GaussianBlur(combined_raw, (7, 7), 0)
    edges_gray = cv2.Canny(blurred, 50, 150)
    sat_blur = cv2.GaussianBlur(sat, (5, 5), 0)
    edges_sat = cv2.Canny(sat_blur, 50, 150)
    edges = cv2.bitwise_or(edges_gray, edges_sat)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    in_range = [c for c in contours
                if image_area * 0.10 <= cv2.contourArea(c) <= image_area * 0.80]
    if not in_range:
        return None
    largest = max(in_range, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) == 4 and cv2.isContourConvex(approx):
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(largest)
        quad = cv2.boxPoints(rect).astype(np.float32)
    return _order_quad_points(quad)


def perspective_correct(img_bgr: np.ndarray, quad: np.ndarray,
                         output_size=(1011, 638)) -> np.ndarray:
    """Warp the quad region to a flat top-down rectangle of output_size."""
    out_w, out_h = output_size
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h))
    return warped


def estimate_skew_angle(img_bgr: np.ndarray) -> float:
    """
    Estimate residual rotation angle (degrees) using the minAreaRect of
    foreground (non-background) pixels after binarization. Works as a
    coarse correction for cards that are still slightly tilted after
    perspective correction (e.g. perspective transform assumed straight
    edges but card was also rotated in-plane).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect angle convention varies by OpenCV version; normalize
    if angle < -45:
        angle = 90 + angle
    # Only trust small corrections - this is a fine-tune step, not a
    # replacement for perspective correction
    if abs(angle) > 15:
        return 0.0
    return angle


def deskew(img_bgr: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.3:
        return img_bgr
    h, w = img_bgr.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), borderValue=(255, 255, 255),
                           flags=cv2.INTER_CUBIC)


def denoise_and_binarize(img_bgr: np.ndarray) -> np.ndarray:
    """
    Grayscale -> denoise -> adaptive threshold, so text stands out clearly
    from the card's background security pattern for OCR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
    binarized = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=15,
    )
    return binarized


def preprocess(img_bgr: np.ndarray) -> PreprocessResult:
    quad = detect_card_contour(img_bgr)
    used_fallback = quad is None

    overlay = img_bgr.copy()
    if quad is not None:
        cv2.polylines(overlay, [quad.astype(np.int32)], True, (0, 255, 0), 4)
        warped = perspective_correct(img_bgr, quad)
    else:
        # No confident quad found - proceed with the original image so the
        # pipeline degrades gracefully instead of crashing.
        warped = img_bgr.copy()

    angle = estimate_skew_angle(warped)
    deskewed = deskew(warped, angle)
    binarized = denoise_and_binarize(deskewed)

    return PreprocessResult(
        original=img_bgr,
        contour_overlay=overlay,
        warped=warped,
        deskewed=deskewed,
        binarized=binarized,
        card_quad=quad,
        used_fallback=used_fallback,
    )


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1] if len(sys.argv) > 1 else "/tmp/degraded_sample.png")
    result = preprocess(img)
    cv2.imwrite("/tmp/pp_1_contour.png", result.contour_overlay)
    cv2.imwrite("/tmp/pp_2_warped.png", result.warped)
    cv2.imwrite("/tmp/pp_3_deskewed.png", result.deskewed)
    cv2.imwrite("/tmp/pp_4_binarized.png", result.binarized)
    print("quad found:", result.card_quad is not None, "fallback used:", result.used_fallback)
