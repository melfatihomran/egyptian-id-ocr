"""
Evaluates PaddleOCR text detection against ground-truth field bounding
boxes (full_name, address, national_id), the same way
scripts/evaluate_preprocessing.py evaluated card-corner detection: real IoU
numbers instead of eyeballing a few samples.

How matching works: for each ground-truth field box, we look at every
detected box and take the best IoU against it (a detector might split one
field into multiple boxes, or merge fields, so "best matching detection"
is more meaningful here than a strict 1-to-1 assignment).

Usage:
  python -m scripts.evaluate_detection
  python -m scripts.evaluate_detection --pass-iou 0.5 --n 20
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.pipeline import preprocess
from src.ocr.detection import detect_text_regions

ROOT = Path(__file__).resolve().parent.parent
DEGRADED_DIR = ROOT / "data" / "synthetic_degraded"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
REPORT_PATH = ROOT / "data" / "detection_eval_report.json"

FIELDS = ("full_name", "address", "national_id")
# Detection boxes for short text are reasonably loose-fitting (font
# ascenders/descenders, anti-aliasing) - 0.5 IoU is a fair "found it" bar
# for this stage, stricter than the 0.85 used for the card-corner check.
PASS_IOU_DEFAULT = 0.5


def box_iou(box_a, quad_b: np.ndarray) -> float:
    """IoU between an axis-aligned [x0,y0,x1,y1] box and a detected quad."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0 = quad_b[:, 0].min(), quad_b[:, 1].min()
    bx1, by1 = quad_b[:, 0].max(), quad_b[:, 1].max()

    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(pass_iou: float = PASS_IOU_DEFAULT, n: int = None):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if n:
        manifest = manifest[:n]

    per_severity = defaultdict(lambda: defaultdict(list))  # severity -> field -> [iou,...]
    failures = []
    skipped_no_field_bboxes = 0

    for entry in manifest:
        if "field_bboxes" not in entry:
            skipped_no_field_bboxes += 1
            continue
        sample_id = entry["sample_id"]
        severity = entry["severity"]
        img_path = DEGRADED_DIR / f"{sample_id}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        result = preprocess(img)
        # Feed the deskewed (not yet binarized) image to detection - see
        # rationale in src/ocr/detection.py's docstring.
        detected = detect_text_regions(result.deskewed)

        for field in FIELDS:
            gt_box = entry["field_bboxes"][field]
            best_iou = max((box_iou(gt_box, q) for q in detected), default=0.0)
            per_severity[severity][field].append(best_iou)
            if best_iou < pass_iou:
                failures.append({"sample_id": sample_id, "severity": severity,
                                  "field": field, "best_iou": round(best_iou, 4)})

    if skipped_no_field_bboxes:
        print(f"(skipped {skipped_no_field_bboxes} sample(s) generated before "
              f"field_bboxes existed - regenerate the dataset to include them)\n")

    report = {"pass_iou_threshold": pass_iou, "by_severity": {}, "failures": failures}
    print(f"Text detection accuracy vs ground-truth field boxes "
          f"(pass threshold: IoU >= {pass_iou})\n")
    print(f"{'Severity':<10} {'Field':<14} {'N':>4} {'Pass':>6} {'Pass Rate':>10} {'Mean IoU':>10}")
    print("-" * 60)

    for severity in ("light", "medium", "heavy"):
        if severity not in per_severity:
            continue
        report["by_severity"][severity] = {}
        for field in FIELDS:
            ious = per_severity[severity][field]
            if not ious:
                continue
            n_s = len(ious)
            passed = sum(1 for x in ious if x >= pass_iou)
            mean_iou = sum(ious) / n_s
            report["by_severity"][severity][field] = {
                "n": n_s, "passed": passed,
                "pass_rate": round(passed / n_s, 4),
                "mean_iou": round(mean_iou, 4),
            }
            print(f"{severity:<10} {field:<14} {n_s:>4} {passed:>6} "
                  f"{passed/n_s:>9.1%} {mean_iou:>10.4f}")
        print("-" * 60)

    if failures:
        print(f"\n{len(failures)} field(s) below threshold (showing up to 15):")
        for f in failures[:15]:
            print(f"  {f['sample_id']} [{f['severity']}] {f['field']}: IoU={f['best_iou']}")

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report written to {REPORT_PATH.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-iou", type=float, default=PASS_IOU_DEFAULT)
    parser.add_argument("--n", type=int, default=None,
                         help="limit to first N manifest entries (useful for a quick smoke test)")
    args = parser.parse_args()
    evaluate(args.pass_iou, args.n)
