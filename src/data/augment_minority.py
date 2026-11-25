"""Minority-class targeted augmentation for train split only (motorcycle, truck)."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.common import (
    CLASS_NAMES,
    cv2_imread,
    cv2_imwrite,
    read_yolo_labels,
    write_metadata_csv,
    write_yolo_labels,
)

RARE_CLASS_IDS = {1, 3}  # motorcycle, truck


def aug_hflip(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    flipped = cv2.flip(img, 1)
    out_boxes = [[int(b[0]), max(0.0, min(1.0, 1.0 - float(b[1]))), float(b[2]), float(b[3]), float(b[4])] for b in boxes]
    return flipped, out_boxes


def aug_night(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    return cv2.convertScaleAbs(img, alpha=0.68, beta=-30), [list(b) for b in boxes]


def aug_sunlight(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    return cv2.convertScaleAbs(img, alpha=1.18, beta=20), [list(b) for b in boxes]


def aug_rainfog(img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.addWeighted(blurred, 0.86, np.full_like(img, 200), 0.14, 0), [list(b) for b in boxes]


def aug_hsv(img: np.ndarray, boxes: List[List[float]], rng: Optional[random.Random] = None) -> Tuple[np.ndarray, List[List[float]]]:
    r = rng or random
    gains = [1.0 + r.uniform(-0.02, 0.02), 1.0 + r.uniform(-0.35, 0.35), 1.0 + r.uniform(-0.25, 0.25)]
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.int16)
    lut_h = ((x * gains[0]) % 180).astype(np.uint8)
    lut_s = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR), [list(b) for b in boxes]


RECIPE_CATALOG: List[Tuple[str, Callable[..., Tuple[np.ndarray, List[List[float]]]]]] = [
    ("hflip", aug_hflip),
    ("night", aug_night),
    ("rainfog", aug_rainfog),
    ("sunlight", aug_sunlight),
    ("hsv", aug_hsv),
]


def load_metadata_lookup(meta_path: str) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    if not os.path.exists(meta_path):
        return lookup
    with open(meta_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["filename"]] = row
    return lookup


def augment_minority_train(dataset_dir: str, rare_copies: int = 1, seed: int = 42) -> Tuple[Counter, Counter, int]:
    """Augment train frames containing motorcycle or truck without touching val/test."""
    train_img_dir = os.path.join(dataset_dir, "images", "train")
    train_lbl_dir = os.path.join(dataset_dir, "labels", "train")
    if not os.path.isdir(train_img_dir) or not os.path.isdir(train_lbl_dir):
        raise FileNotFoundError(f"Train directories not found under {dataset_dir}")

    meta_path = os.path.join(dataset_dir, "dataset_metadata.csv")
    meta_lookup = load_metadata_lookup(meta_path)
    rng = random.Random(seed)

    before_counts: Counter = Counter()
    after_counts: Counter = Counter()
    created_images = 0
    new_meta_rows: List[Dict[str, object]] = []

    image_files = sorted(f for f in os.listdir(train_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for img_name in image_files:
        stem, ext = os.path.splitext(img_name)
        boxes = read_yolo_labels(os.path.join(train_lbl_dir, stem + ".txt"))
        for b in boxes:
            before_counts[int(b[0])] += 1
            after_counts[int(b[0])] += 1

        if "_aug_" in stem or rare_copies <= 0:
            continue
        if not ({int(b[0]) for b in boxes} & RARE_CLASS_IDS):
            continue

        img = cv2_imread(os.path.join(train_img_dir, img_name))
        if img is None:
            continue

        pool = list(RECIPE_CATALOG[1:])
        rng.shuffle(pool)
        chosen = ([RECIPE_CATALOG[0]] + pool)[: min(rare_copies, len(RECIPE_CATALOG))]

        orig_meta = meta_lookup.get(img_name, {})
        for rname, rfn in chosen:
            aug_img, aug_boxes = aug_hsv(img.copy(), [list(b) for b in boxes], rng=rng) if rname == "hsv" else rfn(img.copy(), [list(b) for b in boxes])
            new_img_name = f"{stem}_aug_{rname}{ext}"
            if not cv2_imwrite(os.path.join(train_img_dir, new_img_name), aug_img):
                continue
            write_yolo_labels(os.path.join(train_lbl_dir, f"{stem}_aug_{rname}.txt"), aug_boxes)

            cls_c = Counter(int(b[0]) for b in aug_boxes)
            after_counts.update(cls_c)
            created_images += 1
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

    if new_meta_rows and os.path.exists(meta_path):
        write_metadata_csv(meta_path, new_meta_rows, append=True)

    print("=" * 72)
    print("  MINORITY AUGMENTATION SUMMARY (TRAIN ONLY)")
    print("=" * 72)
    print(f"New augmented train images: {created_images:,}")
    for cid in sorted(CLASS_NAMES):
        b_cnt, a_cnt = before_counts.get(cid, 0), after_counts.get(cid, 0)
        gain = f"{a_cnt / b_cnt:.2f}x" if b_cnt else "0.00x"
        print(f"  {CLASS_NAMES[cid]:<12}: {b_cnt:>7,} -> {a_cnt:>7,} ({gain})")
    print("=" * 72)
    return before_counts, after_counts, created_images


def run_self_test() -> None:
    tmp = tempfile.mkdtemp(prefix="aug_selftest_")
    try:
        for sp in ("train", "val"):
            os.makedirs(os.path.join(tmp, "images", sp), exist_ok=True)
            os.makedirs(os.path.join(tmp, "labels", sp), exist_ok=True)
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2_imwrite(os.path.join(tmp, "images", "train", "car.jpg"), dummy)
        write_yolo_labels(os.path.join(tmp, "labels", "train", "car.txt"), [[0, 0.25, 0.5, 0.1, 0.1]])
        cv2_imwrite(os.path.join(tmp, "images", "train", "moto.jpg"), dummy)
        write_yolo_labels(os.path.join(tmp, "labels", "train", "moto.txt"), [[1, 0.2, 0.5, 0.1, 0.1]])
        cv2_imwrite(os.path.join(tmp, "images", "val", "val.jpg"), dummy)
        write_yolo_labels(os.path.join(tmp, "labels", "val", "val.txt"), [[3, 0.5, 0.5, 0.2, 0.2]])

        _, after, created = augment_minority_train(tmp, rare_copies=1, seed=42)
        assert created == 1 and after[0] == 1 and after[1] == 2
        assert len(os.listdir(os.path.join(tmp, "images", "val"))) == 1
        print("PASS: augment_minority.py self-test succeeded.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="unified_traffic_dataset")
    p.add_argument("--rare-copies", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        run_self_test()
    else:
        augment_minority_train(a.dataset_dir, rare_copies=a.rare_copies, seed=a.seed)
