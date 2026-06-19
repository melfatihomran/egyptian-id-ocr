"""
Takes a clean synthetic ID card image and degrades it to simulate a
real-world photo: perspective skew, rotation, blur, noise, uneven lighting,
and background clutter (the card photographed on a table/surface).

This is what the preprocessing pipeline (src/preprocessing) is meant to undo.
"""
import random
import numpy as np
import cv2


def _add_background_canvas(card_bgr: np.ndarray, rng: random.Random, margin_ratio=0.35):
    """Place the card on a larger neutral/textured canvas, as if photographed
    on a desk/table, so perspective transform has real corners to detect."""
    h, w = card_bgr.shape[:2]
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)
    canvas_w, canvas_h = w + 2 * margin_x, h + 2 * margin_y

    # Textured desk-like background (random base color + noise)
    base_color = rng.choice([(60, 60, 60), (90, 80, 70), (40, 50, 55), (110, 105, 95)])
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = base_color
    noise = rng.randint(5, 18)
    canvas = canvas.astype(np.int16)
    canvas += np.random.randint(-noise, noise, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    canvas[margin_y:margin_y + h, margin_x:margin_x + w] = card_bgr
    card_corners = np.array([
        [margin_x, margin_y],
        [margin_x + w, margin_y],
        [margin_x + w, margin_y + h],
        [margin_x, margin_y + h],
    ], dtype=np.float32)
    return canvas, card_corners


def _apply_perspective_warp(canvas: np.ndarray, card_corners: np.ndarray,
                             rng: random.Random, strength=0.12):
    h, w = canvas.shape[:2]
    jitter = lambda: rng.uniform(-strength, strength)
    src = card_corners.copy()
    dst = src.copy()
    for i in range(4):
        dst[i][0] += jitter() * w
        dst[i][1] += jitter() * h
    dst[:, 0] = np.clip(dst[:, 0], 0, w - 1)
    dst[:, 1] = np.clip(dst[:, 1], 0, h - 1)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, M, (w, h), borderValue=(50, 50, 50))
    return warped, dst


def _apply_rotation(img: np.ndarray, corners: np.ndarray, rng: random.Random, max_deg=8):
    h, w = img.shape[:2]
    angle = rng.uniform(-max_deg, max_deg)
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=(50, 50, 50))
    ones = np.ones((4, 1))
    pts = np.hstack([corners, ones])
    new_corners = (M @ pts.T).T
    return rotated, new_corners, angle


def _apply_lighting(img: np.ndarray, rng: random.Random):
    h, w = img.shape[:2]
    # Random gradient lighting (simulates uneven light/shadow across the card)
    gx, gy = rng.uniform(-1, 1), rng.uniform(-1, 1)
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    gradient = gx * xx + gy * yy
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    brightness_variation = rng.uniform(0.55, 0.45)  # range of darkening
    factor = 1.0 - brightness_variation * gradient
    factor = factor[:, :, None]
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_noise_blur(img: np.ndarray, rng: random.Random):
    out = img.copy()
    if rng.random() < 0.7:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.6:
        noise = np.random.normal(0, rng.uniform(4, 14), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        # JPEG compression artifacts
        quality = rng.randint(35, 70)
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


def degrade_card_image(card_rgb: np.ndarray, rng: random.Random, severity: str = "medium"):
    """
    card_rgb: clean rendered card as RGB numpy array (e.g. np.array(PIL image))
    severity: "light" | "medium" | "heavy" - controls how aggressive the
              degradation is, useful for building a graded test set.
    Returns (degraded_bgr_image, corners_in_degraded_image) where corners
    are the 4 card corners in the OUTPUT image, in order TL, TR, BR, BL -
    this is the ground truth the perspective-correction step should recover.
    """
    card_bgr = cv2.cvtColor(card_rgb, cv2.COLOR_RGB2BGR)

    severity_params = {
        "light":  dict(margin=0.25, warp=0.05, rot=3),
        "medium": dict(margin=0.35, warp=0.12, rot=8),
        "heavy":  dict(margin=0.45, warp=0.20, rot=15),
    }
    p = severity_params[severity]

    canvas, corners = _add_background_canvas(card_bgr, rng, margin_ratio=p["margin"])
    canvas, corners = _apply_perspective_warp(canvas, corners, rng, strength=p["warp"])
    canvas, corners, angle = _apply_rotation(canvas, corners, rng, max_deg=p["rot"])
    canvas = _apply_lighting(canvas, rng)
    canvas = _apply_noise_blur(canvas, rng)

    return canvas, corners.astype(np.float32)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.generator.id_card_renderer import generate_id_card

    rng = random.Random(99)
    img, gt = generate_id_card(rng)
    degraded, corners = degrade_card_image(np.array(img), rng, severity="medium")
    cv2.imwrite("/tmp/degraded_sample.png", degraded)
    print("corners:", corners.tolist())
    print("saved /tmp/degraded_sample.png")
