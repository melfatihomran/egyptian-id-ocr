"""
Renders a synthetic Egyptian National ID card (front side) as an image,
with a known ground-truth dict for the rendered fields.

This is a stylized approximation of the real card's layout (blue/tan
gradient background, decorative guilloché-style pattern, photo box,
labeled Arabic text fields) - NOT a reproduction of the actual government
template/security design. Built purely for OCR pipeline testing.
"""
import random
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import arabic_reshaper
from bidi.algorithm import get_display

from src.generator.national_id import random_national_id, GOVERNORATE_CODES
from src.generator.fake_data import random_full_name, random_address

FONT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "fonts" / "NotoNaskhArabic-Regular.ttf"

CARD_W, CARD_H = 1011, 638  # ~ CR80 card aspect ratio at 300dpi-ish scale

# Color palette echoing the real card's blue/tan tones (approximate, not exact)
BG_TOP = (190, 213, 230)
BG_BOTTOM = (222, 207, 173)
ACCENT_BLUE = (30, 70, 120)
TEXT_DARK = (20, 20, 20)


def _arabic(text: str) -> str:
    """Reshape + apply bidi algorithm so PIL renders Arabic correctly."""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


def _font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_PATH), size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def _draw_gradient_background(draw_size):
    """Simple vertical gradient background to approximate the card's look."""
    w, h = draw_size
    base = Image.new("RGB", (w, h), BG_TOP)
    top = np.array(BG_TOP, dtype=np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.float32)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(h - 1, 1)
        arr[y, :, :] = (top * (1 - t) + bottom * t).astype(np.uint8)
    return Image.fromarray(arr)


def _add_guilloche_pattern(img: Image.Image, rng: random.Random):
    """Add faint repeating wave-line pattern reminiscent of security backgrounds."""
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    for i in range(0, h + 100, 14):
        points = []
        phase = rng.uniform(0, math.pi)
        for x in range(0, w, 6):
            y = i + 10 * math.sin((x / 40.0) + phase)
            points.append((x, y))
        draw.line(points, fill=(255, 255, 255, 40), width=1)
    return img


def _draw_photo_placeholder(img: Image.Image, box, rng: random.Random):
    """Draw a simple grey head-and-shoulders silhouette placeholder."""
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(235, 235, 235), outline=(120, 120, 120), width=2)

    cx = (x0 + x1) // 2
    box_w, box_h = (x1 - x0), (y1 - y0)

    # Head (smaller, higher up)
    head_r = int(box_w * 0.16)
    head_cy = y0 + int(box_h * 0.30)
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 fill=(175, 175, 175))

    # Shoulders/torso as a trapezoid (wider at bottom), not a second circle
    shoulder_top_w = int(box_w * 0.45)
    shoulder_bottom_w = int(box_w * 0.85)
    torso_top_y = head_cy + int(head_r * 0.9)
    torso_bottom_y = y1 - 4
    draw.polygon([
        (cx - shoulder_top_w // 2, torso_top_y),
        (cx + shoulder_top_w // 2, torso_top_y),
        (cx + shoulder_bottom_w // 2, torso_bottom_y),
        (cx - shoulder_bottom_w // 2, torso_bottom_y),
    ], fill=(175, 175, 175))


def generate_id_card(rng: random.Random):
    """
    Returns (PIL.Image card, dict ground_truth)
    """
    id_info = random_national_id(rng)
    full_name = random_full_name(rng, id_info["is_male"])
    address = random_address(rng)

    img = _draw_gradient_background((CARD_W, CARD_H))
    img = _add_guilloche_pattern(img, rng)
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, CARD_W, 90], fill=ACCENT_BLUE)
    header_font = _font(34, "Bold")
    header_text = _arabic("جمهورية مصر العربية")
    bbox = draw.textbbox((0, 0), header_text, font=header_font)
    tw = bbox[2] - bbox[0]
    draw.text(((CARD_W - tw) / 2, 22), header_text, font=header_font, fill="white")

    sub_font = _font(22, "Medium")
    sub_text = _arabic("بطاقة تحقيق الشخصية")
    bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
    tw = bbox[2] - bbox[0]
    draw.text(((CARD_W - tw) / 2, 95), sub_text, font=sub_font, fill=ACCENT_BLUE)

    # Photo placeholder (right side, since Arabic layout reads right->left)
    photo_box = (CARD_W - 230, 150, CARD_W - 40, 430)
    _draw_photo_placeholder(img, photo_box, rng)

    # Text fields (left-aligned block, right-to-left text)
    label_font = _font(18, "Medium")
    value_font = _font(26, "Regular")
    field_x_right = CARD_W - 260  # right edge for text block (left of photo)

    def draw_field(y, label, value, value_font_override=None):
        vf = value_font_override or value_font
        label_txt = _arabic(label)
        lbbox = draw.textbbox((0, 0), label_txt, font=label_font)
        lw = lbbox[2] - lbbox[0]
        draw.text((field_x_right - lw, y), label_txt, font=label_font, fill=(80, 80, 80))

        value_txt = _arabic(value)
        vbbox = draw.textbbox((0, 0), value_txt, font=vf)
        vw = vbbox[2] - vbbox[0]
        value_y = y + 24
        draw.text((field_x_right - vw, value_y), value_txt, font=vf, fill=TEXT_DARK)
        # Absolute-image-coordinate bbox of the VALUE text only (not the
        # label) - this is the ground truth a detector should localize.
        value_bbox = [
            field_x_right - vw, value_y + vbbox[1],
            field_x_right, value_y + vbbox[3],
        ]
        next_y = y + 24 + (vbbox[3] - vbbox[1]) + 18
        return next_y, value_bbox

    y = 165
    y, name_bbox = draw_field(y, "الاسم", full_name)
    y, address_bbox = draw_field(y, "العنوان", address)

    # National ID number - rendered LTR as a digit string (IDs are written
    # left-to-right even on an otherwise RTL card), using Eastern Arabic
    # numerals as on the real card, with the bold ID font style.
    nid_label_font = _font(18, "Medium")
    nid_value_font = _font(30, "Bold")
    nid_label_txt = _arabic("الرقم القومي")
    lbbox = draw.textbbox((0, 0), nid_label_txt, font=nid_label_font)
    lw = lbbox[2] - lbbox[0]
    draw.text((field_x_right - lw, y + 10), nid_label_txt, font=nid_label_font, fill=(80, 80, 80))

    from src.generator.national_id import western_to_eastern_arabic_numerals
    eastern_nid = western_to_eastern_arabic_numerals(id_info["national_id"])
    nbbox = draw.textbbox((0, 0), eastern_nid, font=nid_value_font)
    nw = nbbox[2] - nbbox[0]
    nid_y = y + 38
    draw.text((field_x_right - nw, nid_y), eastern_nid, font=nid_value_font, fill=ACCENT_BLUE)
    national_id_bbox = [
        field_x_right - nw, nid_y + nbbox[1],
        field_x_right, nid_y + nbbox[3],
    ]

    # subtle box around the ID number, like the real card
    pad = 8
    draw.rectangle([field_x_right - nw - pad, nid_y - pad,
                    field_x_right + pad, nid_y + (nbbox[3] - nbbox[1]) + pad],
                   outline=ACCENT_BLUE, width=2)

    # Decorative bottom strip
    draw.rectangle([0, CARD_H - 18, CARD_W, CARD_H], fill=ACCENT_BLUE)

    ground_truth = {
        "full_name": full_name,
        "address": address,
        "national_id": id_info["national_id"],
        "national_id_eastern": eastern_nid,
        "governorate_code": id_info["governorate_code"],
        "governorate_name": id_info["governorate_name"],
        "is_male": id_info["is_male"],
        "birth_year": id_info["birth_year"],
        "birth_month": id_info["birth_month"],
        "birth_day": id_info["birth_day"],
        # Bounding boxes [x0,y0,x1,y1] of each VALUE field in clean-card
        # pixel space. perspective_correct() warps to output_size=(1011,638)
        # - the same as CARD_W,CARD_H here - so a correctly-preprocessed
        # degraded image lands back in this exact coordinate space, making
        # these directly usable as detection-accuracy ground truth without
        # any extra transform.
        "field_bboxes": {
            "full_name": [round(v, 1) for v in name_bbox],
            "address": [round(v, 1) for v in address_bbox],
            "national_id": [round(v, 1) for v in national_id_bbox],
        },
    }
    return img, ground_truth


if __name__ == "__main__":
    rng = random.Random(123)
    img, gt = generate_id_card(rng)
    img.save("/tmp/sample_id_card.png")
    print(json.dumps(gt, ensure_ascii=False, indent=2))
