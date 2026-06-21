"""
Text-detection wrapper around PaddleOCR's detection model.

This module is detection-only (det=True, rec=False) - finding WHERE text is,
not reading it yet (that's src/ocr/recognition.py, Day 3). Two stages are
kept deliberately separate: it makes each piece independently testable and
mirrors how PaddleOCR itself is structured internally.

NOTE ON LOCAL TESTING: this was written and reviewed for correctness against
the installed paddleocr==2.8.1 source (see PaddleOCR.ocr() in paddleocr.py
for the exact return shapes this module relies on), but the actual model
weights could not be downloaded and run in the sandbox this was written in -
they're hosted on Baidu's servers, which that environment's network
allowlist doesn't include. Run this on your own machine, where there's no
such restriction, and report back what you see (numbers, errors, or just
paste a couple of the annotated debug images this script writes to /tmp).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

_ocr_instance = None  # lazy singleton - loading the model is expensive


def _get_ocr():
    """Lazily creates and caches a single PaddleOCR instance (detection-only,
    Arabic recognition model would still get loaded too since PaddleOCR
    loads det+rec+cls together at init, but we just won't call rec here)."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR
        _ocr_instance = PaddleOCR(use_angle_cls=False, lang="ar", show_log=False)
    return _ocr_instance


def detect_text_regions(image_bgr: np.ndarray) -> list:
    """
    Runs PaddleOCR detection on a BGR image (e.g. the deskewed, NOT
    binarized, output of src.preprocessing.pipeline.preprocess() - the
    detection model was trained on natural photos, so feeding it an
    already-binarized black/white image is likely to lose texture cues it
    expects; binarization is more useful for the recognition step or for
    classical engines like Tesseract. Worth A/B testing both once this runs
    on your machine.).

    Returns a list of quads, each a (4,2) float array of [x,y] corner
    points in (TL,TR,BR,BL)-ish order as PaddleOCR's detector outputs them
    (top-to-bottom reading order is NOT guaranteed by the model itself, so
    we sort by box vertical center after detection - see sort_top_to_bottom
    below). Empty list if nothing detected.
    """
    ocr = _get_ocr()
    result = ocr.ocr(image_bgr, det=True, rec=False, cls=False)
    # result is a list with one entry per input "page" - we passed a single
    # image, so result == [boxes_for_that_image] or [None] if nothing found
    boxes = result[0] if result and result[0] else []
    quads = [np.array(box, dtype=np.float32) for box in boxes]
    return sort_top_to_bottom(quads)


def sort_top_to_bottom(quads: list) -> list:
    """Sort detected boxes by vertical center - on this card layout, fields
    are stacked vertically (name, address, ID), so this gives a reading
    order good enough to pair detections with expected fields positionally,
    without needing the recognition step to already know what's what."""
    return sorted(quads, key=lambda q: q[:, 1].mean())


def draw_debug_boxes(image_bgr: np.ndarray, quads: list, color=(0, 255, 0)) -> np.ndarray:
    """Returns a copy of image_bgr with detected boxes drawn on it, numbered
    in the sorted (top-to-bottom) order, for visual sanity-checking."""
    out = image_bgr.copy()
    for i, quad in enumerate(quads):
        pts = quad.astype(np.int32)
        cv2.polylines(out, [pts], True, color, 2)
        label_pos = tuple(pts[0])
        cv2.putText(out, str(i), label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    """
    Usage:
        python -m src.ocr.detection <path_to_corrected_card_image>

    Run this on an already-preprocessed image (output of
    src.preprocessing.pipeline.preprocess(), the "deskewed" stage - NOT the
    raw degraded photo). Writes an annotated debug image to /tmp and prints
    each detected box's coordinates and reading-order index, so you can
    paste the output back for review.
    """
    if len(sys.argv) != 2:
        print("usage: python -m src.ocr.detection <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)

    quads = detect_text_regions(img)
    print(f"Found {len(quads)} text region(s):")
    for i, q in enumerate(quads):
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        print(f"  [{i}] bbox=({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")

    debug_img = draw_debug_boxes(img, quads)
    out_path = "/tmp/detection_debug.png"
    cv2.imwrite(out_path, debug_img)
    print(f"\nAnnotated image written to {out_path}")
