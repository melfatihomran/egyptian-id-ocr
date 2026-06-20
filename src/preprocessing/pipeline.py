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
    """Order 4 points as TL, TR, BR, BL based on sum/diff heuristic."""
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(s)]       # top-left: smallest x+y
    ordered[2] = pts[np.argmax(s)]       # bottom-right: largest x+y
    ordered[1] = pts[np.argmin(diff)]    # top-right: smallest y-x
    ordered[3] = pts[np.argmax(diff)]    # bottom-left: largest y-x
    return ordered


def detect_card_contour(img_bgr: np.ndarray, debug=False):
    """
    Finds the card's quadrilateral in the image.
    Returns ordered (TL,TR,BR,BL) float32 array, or None if not confidently found.

    Primary strategy: Otsu threshold on grayscale to isolate the card as one
    solid bright connected blob against a darker background, then take its
    largest contour. This is more robust than pure Canny edge detection for
    cards with a colored (e.g. navy) header that has low luminance contrast
    against the background but still differs enough in overall brightness
    from a typical desk/table backdrop - edge detection alone was found to
    inconsistently miss that header's outer boundary in testing.

    Falls back to a combined grayscale+saturation Canny edge approach if the
    Otsu-based blob doesn't yield a plausible 4-sided shape (e.g. very low
    global contrast, or background brighter than the card).
    """
    h, w = img_bgr.shape[:2]
    image_area = h * w

    def _quad_from_contour(c):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(c)
        return cv2.boxPoints(rect).astype(np.float32)

    # --- Strategy 1: Otsu threshold + largest connected blob ---
    # Use max(grayscale, saturation) per pixel rather than grayscale alone:
    # a colored (e.g. navy) header can have almost the same luminance as a
    # dark background while being far more saturated, so grayscale-only
    # thresholding would merge it into the background and crop it off.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    # normalize saturation onto a comparable scale to grayscale brightness,
    # then take the per-pixel max so either a brightness OR color edge lifts
    # a region out of the background class
    combined = np.maximum(gray, (sat.astype(np.float32) * 0.7).astype(np.uint8))
    blurred = cv2.GaussianBlur(combined, (7, 7), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The card could be the bright OR dark region depending on background;
    # pick whichever polarity yields a centrally-located, plausible-area blob.
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
        # Sanity check: blob shouldn't touch all four image edges (that's
        # almost certainly the whole background, not the card)
        x, y, bw, bh = cv2.boundingRect(largest)
        margin = 2
        touches_all_edges = (x <= margin and y <= margin and
                              x + bw >= w - margin and y + bh >= h - margin)
        if touches_all_edges:
            continue
        quad = _quad_from_contour(largest)
        return _order_quad_points(quad)

    # --- Strategy 2 (fallback): Canny edges on the combined signal + separate saturation pass ---
    edges_gray = cv2.Canny(blurred, 50, 150)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    edges_sat = cv2.Canny(sat, 50, 150)
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
    quad = _quad_from_contour(largest)
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
