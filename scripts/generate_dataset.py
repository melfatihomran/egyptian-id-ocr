"""
Driver script: generates a synthetic dataset of N Egyptian ID cards.

For each sample, writes:
  data/synthetic_clean/{id}.png       - clean rendered card
  data/synthetic_degraded/{id}.png    - degraded ("real photo") version
  data/ground_truth/{id}.json         - ground truth fields + corner coords

Usage:
  python -m scripts.generate_dataset --n 200 --seed 42
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.generator.id_card_renderer import generate_id_card
from src.generator.degrade import degrade_card_image

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "synthetic_clean"
DEGRADED_DIR = ROOT / "data" / "synthetic_degraded"
GT_DIR = ROOT / "data" / "ground_truth"


def generate_dataset(n: int, seed: int, severities=("light", "medium", "heavy")):
    rng = random.Random(seed)
    for d in (CLEAN_DIR, DEGRADED_DIR, GT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    for i in range(n):
        sample_id = f"sample_{i:04d}"
        img, gt = generate_id_card(rng)

        severity = severities[i % len(severities)]
        degraded, corners = degrade_card_image(np.array(img), rng, severity=severity)

        clean_path = CLEAN_DIR / f"{sample_id}.png"
        degraded_path = DEGRADED_DIR / f"{sample_id}.png"
        gt_path = GT_DIR / f"{sample_id}.json"

        img.save(clean_path)
        import cv2
        cv2.imwrite(str(degraded_path), degraded)

        gt_full = dict(gt)
        gt_full["sample_id"] = sample_id
        gt_full["severity"] = severity
        gt_full["card_corners_in_degraded_image"] = corners.tolist()  # TL,TR,BR,BL
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_full, f, ensure_ascii=False, indent=2)

        manifest.append(gt_full)

        if (i + 1) % 25 == 0 or i == n - 1:
            print(f"  generated {i + 1}/{n}")

    manifest_path = ROOT / "data" / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. Wrote manifest with {len(manifest)} entries to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.n, args.seed)
