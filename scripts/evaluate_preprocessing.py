"""
Evaluates the preprocessing pipeline's card-detection accuracy against
ground-truth corner coordinates, bucketed by degradation severity.

This replaces an earlier informal eyeball check with a reproducible metric:
for each sample, we run detect_card_contour() and compute Intersection-over-
Union (IoU) between the predicted quadrilateral and the ground-truth quad
recorded at dataset-generation time. A sample "passes" if IoU >= PASS_IOU.

Usage:
  python -m scripts.evaluate_preprocessing
  python -m scripts.evaluate_preprocessing --pass-iou 0.80
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing.pipeline import detect_card_contour

ROOT = Path(__file__).resolve().parent.parent
DEGRADED_DIR = ROOT / "data" / "synthetic_degraded"
MANIFEST_PATH = ROOT / "data" / "manifest.json"
REPORT_PATH = ROOT / "data" / "preprocessing_eval_report.json"

PASS_IOU_DEFAULT = 0.85


def quad_iou(quad_a: np.ndarray, quad_b: np.ndarray, img_shape) -> float:
    """Pixel-mask IoU between two quadrilaterals (order-independent)."""
    h, w = img_shape[:2]
    mask_a = np.zeros((h, w), dtype=np.uint8)
    mask_b = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_a, [quad_a.astype(np.int32)], 1)
    cv2.fillPoly(mask_b, [quad_b.astype(np.int32)], 1)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def evaluate(pass_iou: float = PASS_IOU_DEFAULT):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    per_severity = defaultdict(list)  # severity -> list of (iou, fallback, sample_id)
    failures = []

    for entry in manifest:
        sample_id = entry["sample_id"]
        severity = entry["severity"]
        img_path = DEGRADED_DIR / f"{sample_id}.png"
        if not img_path.exists():
            continue  # image wasn't generated locally (gitignored) - skip

        img = cv2.imread(str(img_path))
        gt_quad = np.array(entry["card_corners_in_degraded_image"], dtype=np.float32)

        pred_quad = detect_card_contour(img)
        fallback = pred_quad is None

        if fallback:
            iou = 0.0
        else:
            iou = quad_iou(pred_quad, gt_quad, img.shape)

        per_severity[severity].append((iou, fallback, sample_id))
        if iou < pass_iou:
            failures.append({
                "sample_id": sample_id,
                "severity": severity,
                "iou": round(iou, 4),
                "fallback_used": fallback,
            })

    report = {"pass_iou_threshold": pass_iou, "by_severity": {}, "failures": failures}

    print(f"\nPreprocessing contour-detection accuracy (pass threshold: IoU >= {pass_iou})\n")
    print(f"{'Severity':<10} {'N':>4} {'Pass':>6} {'Pass Rate':>10} {'Mean IoU':>10} {'Fallbacks':>10}")
    print("-" * 56)

    total_n, total_pass = 0, 0
    for severity in ("light", "medium", "heavy"):
        rows = per_severity.get(severity, [])
        if not rows:
            continue
        n = len(rows)
        ious = [r[0] for r in rows]
        passed = sum(1 for iou in ious if iou >= pass_iou)
        fallbacks = sum(1 for r in rows if r[1])
        mean_iou = sum(ious) / n
        total_n += n
        total_pass += passed

        report["by_severity"][severity] = {
            "n": n,
            "passed": passed,
            "pass_rate": round(passed / n, 4),
            "mean_iou": round(mean_iou, 4),
            "fallbacks_used": fallbacks,
        }
        print(f"{severity:<10} {n:>4} {passed:>6} {passed/n:>9.1%} {mean_iou:>10.4f} {fallbacks:>10}")

    print("-" * 56)
    if total_n:
        print(f"{'overall':<10} {total_n:>4} {total_pass:>6} {total_pass/total_n:>9.1%}")

    if failures:
        print(f"\n{len(failures)} sample(s) below threshold:")
        for f in failures:
            print(f"  {f['sample_id']} ({f['severity']}): IoU={f['iou']}, "
                  f"fallback={f['fallback_used']}")

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report written to {REPORT_PATH.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-iou", type=float, default=PASS_IOU_DEFAULT)
    args = parser.parse_args()
    evaluate(args.pass_iou)
