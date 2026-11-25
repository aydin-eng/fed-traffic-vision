"""Comprehensive 12-recipe class-aware traffic camera augmentation engine (train split only).

Designed for multi-source CCTV & street vehicle datasets (UA-DETRAC + MIO-TCD + COCO 2017):
- Applies deterministic geometric + adverse-weather + CCTV degradation augmentations
  ONLY to training images while keeping validation and test splits untouched.
- Uses multi-tier class-aware multipliers so rare classes (motorcycle, bus, truck)
  are boosted without re-inflating dominant car-only frames.
- Supports in-place augmentation, non-destructive --out-dir cloning, --dry-run planning,
  --preview-out visual grid rendering, and --report-json thesis metrics export.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.common import (
    CLASS_NAMES,
    PAD_GRAY,
    cv2_imdecode_bytes,
    cv2_imread,
    cv2_imwrite,
    read_yolo_labels,
    write_data_yaml,
    write_metadata_csv,
    write_yolo_labels,
    xyxy_to_yolo,
)

TIER1_IDS = {1, 2}  # motorcycle, bus
TIER2_IDS = {3}     # truck

PREVIEW_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (37, 99, 235),   # car - blue
    1: (220, 38, 38),   # motorcycle - red
    2: (234, 88, 12),   # bus - orange
    3: (5, 150, 105),   # truck - emerald
}


def _copy_boxes(boxes: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])] for b in boxes]


def aug_hflip(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    flipped = cv2.flip(img, 1)
    out = [
        [int(b[0]), max(0.0, min(1.0, 1.0 - float(b[1]))), float(b[2]), float(b[3]), float(b[4])]
        for b in boxes
    ]
    return flipped, out


def aug_night_lowlight(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    gamma = 1.75
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    dark = cv2.LUT(img, lut).astype(np.float32)
    # Slight cool night tint + mild sensor noise
    dark[:, :, 0] *= 1.06
    dark[:, :, 2] *= 0.90
    np_rng = np.random.default_rng(r.randint(0, 1_000_000))
    noise = np_rng.normal(0.0, 4.5, size=dark.shape).astype(np.float32)
    out = np.clip(dark + noise, 0, 255).astype(np.uint8)
    return out, _copy_boxes(boxes)


def aug_sun_glare(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    glare = cv2.convertScaleAbs(img, alpha=1.20, beta=22).astype(np.float32)
    h, w = img.shape[:2]
    cx = int(w * r.uniform(0.25, 0.75))
    cy = int(h * r.uniform(0.08, 0.30))
    yy, xx = np.ogrid[:h, :w]
    sigma = max(w, h) * 0.28
    halo = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    glare += (halo[:, :, None] * 55.0)
    return np.clip(glare, 0, 255).astype(np.uint8), _copy_boxes(boxes)


def aug_rain_streaks(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    h, w = img.shape[:2]
    base = cv2.GaussianBlur(img, (3, 3), 0)
    base = cv2.convertScaleAbs(base, alpha=0.90, beta=-6)
    rain_layer = np.zeros_like(base)
    num_drops = max(120, int((w * h) / 2200))
    slant = r.randint(-5, 5)
    length = r.randint(10, 18)
    for _ in range(num_drops):
        x1 = r.randint(0, w - 1)
        y1 = r.randint(0, h - 1)
        x2 = max(0, min(w - 1, x1 + slant))
        y2 = max(0, min(h - 1, y1 + length))
        shade = r.randint(190, 235)
        cv2.line(rain_layer, (x1, y1), (x2, y2), (shade, shade, shade), 1, cv2.LINE_AA)
    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    out = cv2.addWeighted(base, 0.88, rain_layer, 0.32, 0)
    return out, _copy_boxes(boxes)


def aug_fog_haze(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    haze = np.full_like(img, 205)
    out = cv2.addWeighted(blurred, 0.78, haze, 0.22, 0)
    return out, _copy_boxes(boxes)


def aug_motion_blur(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    k = 7
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel), _copy_boxes(boxes)


def aug_overpass_shadow(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    y_top = r.randint(int(h * 0.15), int(h * 0.45))
    y_bot = min(h, y_top + r.randint(int(h * 0.22), int(h * 0.40)))
    skew = r.randint(-int(w * 0.12), int(w * 0.12))
    poly = np.array([[0, y_top], [w, max(0, y_top + skew)], [w, min(h, y_bot + skew)], [0, y_bot]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    soft_mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    factor = 1.0 - (0.45 * soft_mask[:, :, None])
    out = np.clip(out * factor, 0, 255).astype(np.uint8)
    return out, _copy_boxes(boxes)


def aug_cctv_degrade(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    np_rng = np.random.default_rng(r.randint(0, 1_000_000))
    noisy = np.clip(img.astype(np.float32) + np_rng.normal(0.0, 5.5, size=img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    if ok:
        decoded = cv2_imdecode_bytes(buf.tobytes())
        if decoded is not None:
            return decoded, _copy_boxes(boxes)
    return noisy, _copy_boxes(boxes)


def aug_hsv_shift(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    gains = [1.0 + r.uniform(-0.025, 0.025), 1.0 + r.uniform(-0.35, 0.35), 1.0 + r.uniform(-0.25, 0.25)]
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * gains[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR), _copy_boxes(boxes)


def aug_safe_zoom(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
    zoom_factor: float = 1.12,
) -> Tuple[np.ndarray, List[List[float]]]:
    h, w = img.shape[:2]
    new_h = max(16, int(round(h / zoom_factor)))
    new_w = max(16, int(round(w / zoom_factor)))
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = img[y1:y1 + new_h, x1:x1 + new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    sx = w / float(new_w)
    sy = h / float(new_h)
    out_boxes: List[List[float]] = []
    for b in boxes:
        cid = int(b[0])
        xc, yc, bw, bh = float(b[1]), float(b[2]), float(b[3]), float(b[4])
        bx1 = (xc - bw / 2.0) * w
        by1 = (yc - bh / 2.0) * h
        bx2 = (xc + bw / 2.0) * w
        by2 = (yc + bh / 2.0) * h

        zx1 = (bx1 - x1) * sx
        zy1 = (by1 - y1) * sy
        zx2 = (bx2 - x1) * sx
        zy2 = (by2 - y1) * sy
        proj_area = max(1.0, (zx2 - zx1) * (zy2 - zy1))

        cx1, cy1 = max(0.0, zx1), max(0.0, zy1)
        cx2, cy2 = min(float(w), zx2), min(float(h), zy2)
        cw, ch = cx2 - cx1, cy2 - cy1
        if cw < 6.0 or ch < 6.0 or (cw * ch) < 0.35 * proj_area:
            continue
        out_boxes.append(list(xyxy_to_yolo(cid, cx1, cy1, cx2, cy2, float(w), float(h))))

    if not out_boxes:
        return img.copy(), _copy_boxes(boxes)
    return zoomed, out_boxes


def aug_mild_rotate_occ(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random.Random(42)
    h, w = img.shape[:2]
    angle = r.choice([-4.5, -3.5, 3.5, 4.5])
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    warped = cv2.warpAffine(
        img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PAD_GRAY
    )

    out_boxes: List[List[float]] = []
    pixel_boxes: List[Tuple[float, float, float, float]] = []
    for b in boxes:
        cid = int(b[0])
        xc, yc, bw, bh = float(b[1]), float(b[2]), float(b[3]), float(b[4])
        bx1, by1 = (xc - bw / 2.0) * w, (yc - bh / 2.0) * h
        bx2, by2 = (xc + bw / 2.0) * w, (yc + bh / 2.0) * h
        corners = np.array([[bx1, by1, 1.0], [bx2, by1, 1.0], [bx2, by2, 1.0], [bx1, by2, 1.0]], dtype=np.float64)
        rot = corners @ mat.T
        rx1 = max(0.0, float(np.min(rot[:, 0])))
        ry1 = max(0.0, float(np.min(rot[:, 1])))
        rx2 = min(float(w), float(np.max(rot[:, 0])))
        ry2 = min(float(h), float(np.max(rot[:, 1])))
        if (rx2 - rx1) < 6.0 or (ry2 - ry1) < 6.0:
            continue
        pixel_boxes.append((rx1, ry1, rx2, ry2))
        out_boxes.append(list(xyxy_to_yolo(cid, rx1, ry1, rx2, ry2, float(w), float(h))))

    if not out_boxes:
        return img.copy(), _copy_boxes(boxes)

    # Add 1-2 small roadside occlusion patches that never cover >35% of any vehicle box
    num_cuts = r.randint(1, 2)
    for _ in range(num_cuts):
        cw = r.randint(max(8, int(w * 0.04)), max(12, int(w * 0.08)))
        ch = r.randint(max(8, int(h * 0.04)), max(12, int(h * 0.08)))
        for _attempt in range(15):
            cx1 = r.randint(0, max(0, w - cw))
            cy1 = r.randint(0, max(0, h - ch))
            cx2, cy2 = cx1 + cw, cy1 + ch
            safe = True
            for bx1, by1, bx2, by2 in pixel_boxes:
                ix1, iy1 = max(cx1, bx1), max(cy1, by1)
                ix2, iy2 = min(cx2, bx2), min(cy2, by2)
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
                    if inter > 0.35 * b_area:
                        safe = False
                        break
            if safe:
                shade = r.randint(45, 85)
                warped[cy1:cy2, cx1:cx2] = (shade, shade + 3, shade + 5)
                break

    return warped, out_boxes


def aug_combo_night_flip(
    img: np.ndarray,
    boxes: List[List[float]],
    rng: Optional[random.Random] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    f_img, f_boxes = aug_hflip(img, boxes, rng=rng)
    return aug_night_lowlight(f_img, f_boxes, rng=rng)


RECIPE_CATALOG: List[Tuple[str, Callable[..., Tuple[np.ndarray, List[List[float]]]]]] = [
    ("hflip", aug_hflip),
    ("night_lowlight", aug_night_lowlight),
    ("sun_glare", aug_sun_glare),
    ("rain_streaks", aug_rain_streaks),
    ("fog_haze", aug_fog_haze),
    ("motion_blur", aug_motion_blur),
    ("overpass_shadow", aug_overpass_shadow),
    ("cctv_degrade", aug_cctv_degrade),
    ("hsv_shift", aug_hsv_shift),
    ("safe_zoom", aug_safe_zoom),
    ("mild_rotate_occ", aug_mild_rotate_occ),
    ("combo_night_flip", aug_combo_night_flip),
]


def load_metadata_lookup(meta_path: str) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(meta_path):
        return lookup
    with open(meta_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = row
    return lookup


def _select_recipes_for_frame(
    present_classes: set,
    rare_copies: int,
    truck_copies: int,
    car_copies: int,
    rng: random.Random,
) -> List[Tuple[str, Callable[..., Tuple[np.ndarray, List[List[float]]]]]]:
    if present_classes & TIER1_IDS:
        n_copies = rare_copies
    elif present_classes & TIER2_IDS:
        n_copies = truck_copies
    else:
        n_copies = car_copies

    if n_copies <= 0:
        return []
    pool = list(RECIPE_CATALOG[1:])
    rng.shuffle(pool)
    ordered = [RECIPE_CATALOG[0]] + pool
    return ordered[: min(n_copies, len(ordered))]


def _draw_preview_panel(img_bgr: np.ndarray, boxes: Sequence[Sequence[float]], title: str) -> Image.Image:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(rgb).resize((480, 320), Image.Resampling.LANCZOS).convert("RGBA")
    draw = ImageDraw.Draw(pil_im)
    w, h = pil_im.size
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for b in boxes:
        cid = int(b[0])
        xc, yc, bw, bh = float(b[1]), float(b[2]), float(b[3]), float(b[4])
        x1, y1 = (xc - bw / 2.0) * w, (yc - bh / 2.0) * h
        x2, y2 = (xc + bw / 2.0) * w, (yc + bh / 2.0) * h
        col = PREVIEW_COLORS.get(cid, (100, 116, 139))
        draw.rectangle([x1, y1, x2, y2], outline=col + (240,), width=2)

    draw.rectangle([0, 0, w, 22], fill=(15, 23, 42, 210))
    draw.text((8, 4), title, fill=(255, 255, 255, 255), font=font)
    return pil_im.convert("RGB")


def generate_preview_grid(dataset_dir: str, preview_out: str, seed: int = 42) -> Optional[str]:
    """Render a 3x4 visual grid showing Original + 11 augmentation recipes on a multi-class frame."""
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    best_cand = None
    best_score = (-1, -1)

    for fn in sorted(os.listdir(train_img_dir)):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")) or "_aug_" in fn:
            continue
        stem = os.path.splitext(fn)[0]
        boxes = read_yolo_labels(os.path.join(train_lbl_dir, stem + ".txt"))
        cset = {int(b[0]) for b in boxes}
        if not boxes or len(boxes) > 18:
            continue
        has_rare = int(1 in cset or 3 in cset)
        score = (has_rare, len(cset))
        if score > best_score:
            best_score = score
            best_cand = (os.path.join(train_img_dir, fn), boxes, fn)
        if has_rare and len(cset) >= 3:
            break

    if best_cand is None:
        return None

    img_path, orig_boxes, orig_name = best_cand
    orig_img = cv2_imread(img_path)
    if orig_img is None:
        return None

    panels: List[Image.Image] = [_draw_preview_panel(orig_img, orig_boxes, f"original ({orig_name[:22]})")]
    rng = random.Random(seed)
    for rname, rfn in RECIPE_CATALOG[:11]:
        aug_img, aug_boxes = rfn(orig_img.copy(), _copy_boxes(orig_boxes), rng=rng)
        panels.append(_draw_preview_panel(aug_img, aug_boxes, f"{rname} ({len(aug_boxes)} boxes)"))

    grid = Image.new("RGB", (480 * 4, 320 * 3), (255, 255, 255))
    for idx, panel in enumerate(panels):
        gx = (idx % 4) * 480
        gy = (idx // 4) * 320
        grid.paste(panel, (gx, gy))

    os.makedirs(os.path.dirname(os.path.abspath(preview_out)), exist_ok=True)
    grid.save(preview_out, quality=95)
    print(f"Preview grid saved: {preview_out} (source: {orig_name})")
    return preview_out


def _copy_static_splits(src_dir: str, dst_dir: str) -> List[Dict[str, object]]:
    """Copy val/test untouched and return existing non-train metadata rows."""
    for sp in ("val", "test"):
        s_img = os.path.join(src_dir, "images", sp)
        s_lbl = os.path.join(src_dir, "labels", sp)
        if os.path.isdir(s_img):
            shutil.copytree(s_img, os.path.join(dst_dir, "images", sp), dirs_exist_ok=True)
        if os.path.isdir(s_lbl):
            shutil.copytree(s_lbl, os.path.join(dst_dir, "labels", sp), dirs_exist_ok=True)

    write_data_yaml(dst_dir)
    meta_src = os.path.join(src_dir, "dataset_metadata.csv")
    kept_rows: List[Dict[str, object]] = []
    if os.path.exists(meta_src):
        with open(meta_src, "r", encoding="utf-8") as f:
            kept_rows = list(csv.DictReader(f))
    return kept_rows


def augment_minority_train(
    dataset_dir: str,
    rare_copies: int = 2,
    seed: int = 42,
    out_dir: Optional[str] = None,
    truck_copies: int = 1,
    car_copies: int = 0,
    dry_run: bool = False,
    preview_out: Optional[str] = None,
    report_json: Optional[str] = None,
) -> Tuple[Counter, Counter, int]:
    """Run multi-tier class-aware augmentation on train split only."""
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    if not os.path.isdir(train_img_dir) or not os.path.isdir(train_lbl_dir):
        raise FileNotFoundError(f"Train directories not found under {dataset_dir}")

    if preview_out:
        generate_preview_grid(dataset_dir, preview_out, seed=seed)

    clone_mode = bool(out_dir and os.path.abspath(out_dir) != os.path.abspath(dataset_dir))
    target_dir = out_dir if clone_mode else dataset_dir
    target_train_img = os.path.join(target_dir, "images", "train")
    target_train_lbl = os.path.join(target_dir, "labels", "train")

    existing_rows: List[Dict[str, object]] = []
    if not dry_run and clone_mode:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_train_img, exist_ok=True)
        os.makedirs(target_train_lbl, exist_ok=True)
        existing_rows = _copy_static_splits(dataset_dir, target_dir)

    meta_path = os.path.join(dataset_dir, "dataset_metadata.csv")
    meta_lookup = load_metadata_lookup(meta_path)
    rng = random.Random(seed)

    before_counts: Counter = Counter()
    after_counts: Counter = Counter()
    orig_images = 0
    created_images = 0
    new_meta_rows: List[Dict[str, object]] = []
    recipe_usage: Counter = Counter()

    image_files = sorted(f for f in os.listdir(train_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for img_name in image_files:
        stem, ext = os.path.splitext(img_name)
        src_img_path = os.path.join(train_img_dir, img_name)
        src_lbl_path = os.path.join(train_lbl_dir, stem + ".txt")
        boxes = read_yolo_labels(src_lbl_path)
        orig_images += 1
        for b in boxes:
            before_counts[int(b[0])] += 1
            after_counts[int(b[0])] += 1

        if not dry_run and clone_mode:
            shutil.copy2(src_img_path, os.path.join(target_train_img, img_name))
            shutil.copy2(src_lbl_path, os.path.join(target_train_lbl, stem + ".txt"))

        if "_aug_" in stem:
            continue

        chosen = _select_recipes_for_frame(
            {int(b[0]) for b in boxes},
            rare_copies=rare_copies,
            truck_copies=truck_copies,
            car_copies=car_copies,
            rng=rng,
        )
        if not chosen:
            continue

        if dry_run:
            for rname, _ in chosen:
                recipe_usage[rname] += 1
                created_images += 1
                for b in boxes:
                    after_counts[int(b[0])] += 1
            continue

        img = cv2_imread(src_img_path)
        if img is None:
            continue

        orig_meta = meta_lookup.get(img_name, {})
        for rname, rfn in chosen:
            aug_img, aug_boxes = rfn(img.copy(), _copy_boxes(boxes), rng=rng)
            new_img_name = f"{stem}_aug_{rname}{ext}"
            if not cv2_imwrite(os.path.join(target_train_img, new_img_name), aug_img):
                continue
            write_yolo_labels(os.path.join(target_train_lbl, f"{stem}_aug_{rname}.txt"), aug_boxes)

            cls_c = Counter(int(b[0]) for b in aug_boxes)
            after_counts.update(cls_c)
            created_images += 1
            recipe_usage[rname] += 1
            new_meta_rows.append(
                {
                    "filename": new_img_name,
                    "split": "train",
                    "source": orig_meta.get("source", "augmented"),
                    "sequence": orig_meta.get("sequence", "unknown"),
                    "weather": f"{orig_meta.get('weather', 'unknown')}_{rname}",
                    "camera_state": orig_meta.get("camera_state", "unknown"),
                    "num_car": cls_c.get(0, 0),
                    "num_motorcycle": cls_c.get(1, 0),
                    "num_bus": cls_c.get(2, 0),
                    "num_truck": cls_c.get(3, 0),
                }
            )

    if not dry_run:
        target_meta = os.path.join(target_dir, "dataset_metadata.csv")
        if clone_mode and (existing_rows or new_meta_rows):
            write_metadata_csv(target_meta, existing_rows + new_meta_rows, append=False)
        elif new_meta_rows and os.path.exists(target_meta):
            write_metadata_csv(target_meta, new_meta_rows, append=True)

    ratios_before = {
        CLASS_NAMES[c]: round(before_counts[0] / before_counts[c], 2) if before_counts[c] else None
        for c in (1, 2, 3)
    }
    ratios_after = {
        CLASS_NAMES[c]: round(after_counts[0] / after_counts[c], 2) if after_counts[c] else None
        for c in (1, 2, 3)
    }

    if report_json:
        os.makedirs(os.path.dirname(os.path.abspath(report_json)), exist_ok=True)
        payload = {
            "dry_run": dry_run,
            "original_train_images": orig_images,
            "created_augmented_images": created_images,
            "final_train_images": orig_images + created_images,
            "before_boxes": {CLASS_NAMES[k]: before_counts.get(k, 0) for k in sorted(CLASS_NAMES)},
            "after_boxes": {CLASS_NAMES[k]: after_counts.get(k, 0) for k in sorted(CLASS_NAMES)},
            "car_to_class_ratio_before": ratios_before,
            "car_to_class_ratio_after": ratios_after,
            "recipe_usage": dict(recipe_usage),
        }
        with open(report_json, "w", encoding="utf-8") as rf:
            json.dump(payload, rf, indent=2)

    print("=" * 76)
    print("  TRAFFIC-CAMERA AUGMENTATION SUMMARY (TRAIN ONLY)" + (" [DRY-RUN]" if dry_run else ""))
    print("=" * 76)
    print(f"Train images: {orig_images:,} -> {orig_images + created_images:,} (+{created_images:,} augmented)")
    print(f"{'Class':<12} | {'Before':>9} | {'After':>9} | {'Gain':>7} | {'car:cls Before':>14} | {'car:cls After':>13}")
    print("-" * 76)
    for cid in sorted(CLASS_NAMES):
        cname = CLASS_NAMES[cid]
        b_cnt, a_cnt = before_counts.get(cid, 0), after_counts.get(cid, 0)
        gain = f"{a_cnt / b_cnt:.2f}x" if b_cnt else "0.00x"
        rb = f"{ratios_before[cname]}:1" if cname in ratios_before and ratios_before[cname] else "-"
        ra = f"{ratios_after[cname]}:1" if cname in ratios_after and ratios_after[cname] else "-"
        print(f"{cname:<12} | {b_cnt:>9,} | {a_cnt:>9,} | {gain:>7} | {rb:>14} | {ra:>13}")
    print("=" * 76)
    return before_counts, after_counts, created_images


def run_self_test() -> None:
    """Validate all 12 recipes, box coordinate invariants, clone mode, and val immutability."""
    tmp = tempfile.mkdtemp(prefix="aug12_selftest_")
    try:
        src = os.path.join(tmp, "src")
        for sp in ("train", "val"):
            os.makedirs(os.path.join(src, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(src, "labels", sp), exist_ok=True)

        img = np.full((240, 320, 3), 150, dtype=np.uint8)
        boxes = [[0, 0.25, 0.50, 0.16, 0.16], [1, 0.72, 0.55, 0.10, 0.14], [3, 0.50, 0.45, 0.20, 0.22]]

        rng = random.Random(42)
        for rname, rfn in RECIPE_CATALOG:
            out_img, out_boxes = rfn(img.copy(), _copy_boxes(boxes), rng=rng)
            assert out_img.shape == img.shape, f"Shape changed in {rname}"
            assert len(out_boxes) >= 1, f"All boxes dropped in {rname}"
            for b in out_boxes:
                assert int(b[0]) in (0, 1, 2, 3)
                for v in b[1:]:
                    assert 0.0 <= float(v) <= 1.0, f"Out-of-bounds coord in {rname}: {b}"
            if rname == "hflip":
                assert abs(out_boxes[0][1] - 0.75) < 1e-5, f"hflip failed: {out_boxes[0]}"

        # Multi-tier dataset test + clone mode
        cv2_imwrite(os.path.join(src, "images", "train", "car.jpg"), img)
        write_yolo_labels(os.path.join(src, "labels", "train", "car.txt"), [[0, 0.3, 0.5, 0.1, 0.1]])
        cv2_imwrite(os.path.join(src, "images", "train", "moto.jpg"), img)
        write_yolo_labels(os.path.join(src, "labels", "train", "moto.txt"), [[1, 0.7, 0.5, 0.1, 0.1]])
        cv2_imwrite(os.path.join(src, "images", "train", "truck.jpg"), img)
        write_yolo_labels(os.path.join(src, "labels", "train", "truck.txt"), [[3, 0.5, 0.5, 0.2, 0.2]])
        cv2_imwrite(os.path.join(src, "images", "val", "val.jpg"), img)
        write_yolo_labels(os.path.join(src, "labels", "val", "val.txt"), [[2, 0.5, 0.5, 0.2, 0.2]])

        dst = os.path.join(tmp, "dst")
        preview = os.path.join(tmp, "preview.jpg")
        report = os.path.join(tmp, "report.json")
        before, after, created = augment_minority_train(
            src,
            rare_copies=2,
            truck_copies=1,
            car_copies=0,
            seed=42,
            out_dir=dst,
            preview_out=preview,
            report_json=report,
        )
        assert created == 3, f"Expected 2 (moto) + 1 (truck) + 0 (car) = 3, got {created}"
        assert before == Counter({0: 1, 1: 1, 3: 1})
        assert after[0] == 1 and after[1] == 3 and after[3] == 2
        assert len(os.listdir(os.path.join(dst, "images", "val"))) == 1
        assert os.path.exists(preview) and os.path.exists(report)
        print("PASS: augment_minority.py (12 recipes) self-test succeeded.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comprehensive 12-recipe traffic camera augmentation.")
    p.add_argument("--dataset-dir", default="unified_traffic_dataset")
    p.add_argument("--out-dir", default=None, help="Optional separate output directory (clones val/test untouched).")
    p.add_argument("--rare-copies", type=int, default=2, help="Augmented copies for frames with motorcycle or bus.")
    p.add_argument("--truck-copies", type=int, default=1, help="Augmented copies for truck frames (without moto/bus).")
    p.add_argument("--car-copies", type=int, default=0, help="Augmented copies for car-only frames.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preview-out", default=None, help="Save 3x4 visual augmentation preview grid.")
    p.add_argument("--report-json", default=None, help="Save before/after stats JSON.")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        augment_minority_train(
            args.dataset_dir,
            rare_copies=args.rare_copies,
            seed=args.seed,
            out_dir=args.out_dir,
            truck_copies=args.truck_copies,
            car_copies=args.car_copies,
            dry_run=args.dry_run,
            preview_out=args.preview_out,
            report_json=args.report_json,
        )
