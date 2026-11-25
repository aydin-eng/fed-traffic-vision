"""MIO-TCD Localization extractor for the 4-class traffic vehicle schema.

Supports reading directly from MIO-TCD-Localization.tar (streaming gt_train.csv for --dry-run
or extracting only selected minority-prioritized images) as well as an extracted folder.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import shutil
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.common import (
    CLASS_NAMES,
    cv2_imdecode_bytes,
    cv2_imread,
    cv2_imwrite,
    write_data_yaml,
    write_metadata_csv,
    write_yolo_labels,
    xyxy_to_yolo,
)

MIOTCD_CLASS_MAP: Dict[str, int] = {
    "car": 0,
    "work_van": 0,
    "pickup_truck": 0,
    "motorcycle": 1,
    "bus": 2,
    "single_unit_truck": 3,
    "articulated_truck": 3,
}

MIN_BOX_PX = 6.0


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def parse_miotcd_csv_lines(lines: Iterable[str]) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Parse gt_train.csv supporting both [id, label, x1, y1, x2, y2] and [id, x1, y1, x2, y2, label]."""
    per_image: Dict[str, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    reader = csv.reader(lines)
    for row in reader:
        if len(row) < 6:
            continue
        raw_id = row[0].strip()
        if not raw_id or raw_id.lower() in ("image", "img", "filename", "id", "image_id"):
            continue
        stem = os.path.splitext(os.path.basename(raw_id))[0]

        # Auto-detect whether column 1 is class name or xmin
        if not _is_float(row[1]):
            raw_label = row[1]
            coord_strs = row[2:6]
        elif not _is_float(row[5]):
            coord_strs = row[1:5]
            raw_label = row[5]
        else:
            continue

        norm_label = raw_label.strip().lower().replace(" ", "_")
        if norm_label not in MIOTCD_CLASS_MAP:
            continue

        try:
            x1, y1, x2, y2 = (float(c) for c in coord_strs)
        except ValueError:
            continue
        if (x2 - x1) < MIN_BOX_PX or (y2 - y1) < MIN_BOX_PX:
            continue

        per_image[stem].append((MIOTCD_CLASS_MAP[norm_label], x1, y1, x2, y2))
    return per_image


def select_stratified_miotcd(
    per_image: Dict[str, List[Tuple[int, float, float, float, float]]],
    max_train: int,
    max_val: int,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Select images prioritizing motorcycle (1), truck (3), bus (2), then car (0), and split cleanly."""
    tier_motor: List[str] = []
    tier_truck: List[str] = []
    tier_bus: List[str] = []
    tier_car: List[str] = []

    for stem in sorted(per_image.keys()):
        cset = {b[0] for b in per_image[stem]}
        if 1 in cset:
            tier_motor.append(stem)
        elif 3 in cset:
            tier_truck.append(stem)
        elif 2 in cset:
            tier_bus.append(stem)
        else:
            tier_car.append(stem)

    rng = random.Random(seed)
    for tier in (tier_motor, tier_truck, tier_bus, tier_car):
        rng.shuffle(tier)

    total_budget = max_train + max_val
    ordered_pool = (tier_motor + tier_truck + tier_bus + tier_car)[:total_budget]

    # Split each tier proportionally so train and val share the same class mix with zero overlap
    train_ids: List[str] = []
    val_ids: List[str] = []
    selected_set = set(ordered_pool)

    for tier in (tier_motor, tier_truck, tier_bus, tier_car):
        chosen = [s for s in tier if s in selected_set]
        if not chosen:
            continue
        if len(chosen) >= 2 and val_ratio > 0:
            n_val = max(1, int(round(len(chosen) * val_ratio)))
            n_val = min(n_val, len(chosen) - 1)
        else:
            n_val = 0
        val_ids.extend(chosen[:n_val])
        train_ids.extend(chosen[n_val:])

    # Trim gently if rounding pushed either split slightly over budget
    return train_ids[:max_train], val_ids[:max_val]


def run_miotcd(args: argparse.Namespace) -> Tuple[Dict[str, Counter], Counter]:
    """Extract MIO-TCD Localization subset or run --dry-run analysis."""
    tf: Optional[tarfile.TarFile] = None
    tar_img_members: Dict[str, tarfile.TarInfo] = {}
    dir_img_paths: Dict[str, str] = {}

    if args.tar_path:
        tf = tarfile.open(args.tar_path, "r:*")
        gt_member = None
        for m in tf.getmembers():
            if not m.isfile():
                continue
            bname = os.path.basename(m.name)
            if bname == "gt_train.csv":
                gt_member = m
            elif not args.dry_run and bname.lower().endswith((".jpg", ".jpeg", ".png")):
                if "/train/" in f"/{m.name.lower()}" or m.name.lower().startswith("train/"):
                    tar_img_members[os.path.splitext(bname)[0]] = m
        if gt_member is None:
            raise FileNotFoundError(f"gt_train.csv not found inside {args.tar_path}")
        raw_csv = tf.extractfile(gt_member)
        assert raw_csv is not None
        with io.TextIOWrapper(raw_csv, encoding="utf-8", errors="ignore") as txt_f:
            per_image = parse_miotcd_csv_lines(txt_f)
    elif args.data_dir:
        gt_path = os.path.join(args.data_dir, "gt_train.csv")
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"gt_train.csv not found at {gt_path}")
        with open(gt_path, "r", encoding="utf-8", errors="ignore") as txt_f:
            per_image = parse_miotcd_csv_lines(txt_f)
        train_img_dir = os.path.join(args.data_dir, "train")
        if not args.dry_run and os.path.isdir(train_img_dir):
            for fname in os.listdir(train_img_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    dir_img_paths[os.path.splitext(fname)[0]] = os.path.join(train_img_dir, fname)
    else:
        raise ValueError("Provide --tar-path or --data-dir.")

    train_ids, val_ids = select_stratified_miotcd(
        per_image, max_train=args.max_train, max_val=args.max_val, val_ratio=args.val_ratio, seed=args.seed
    )
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(f"Train/val overlap detected in MIO-TCD: {len(overlap)}")

    split_box_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}
    split_img_counts: Counter = Counter()
    metadata_rows: List[Dict[str, object]] = []

    try:
        for split_name, id_list in (("train", train_ids), ("val", val_ids)):
            out_img_dir = os.path.join(args.out_dir, "images", split_name)
            out_lbl_dir = os.path.join(args.out_dir, "labels", split_name)
            if not args.dry_run:
                os.makedirs(out_img_dir, exist_ok=True)
                os.makedirs(out_lbl_dir, exist_ok=True)

            for stem in id_list:
                boxes = per_image[stem]
                if args.dry_run:
                    split_img_counts[split_name] += 1
                    for b in boxes:
                        split_box_counts[split_name][b[0]] += 1
                    continue

                img = None
                if tf is not None:
                    member = tar_img_members.get(stem)
                    if member is None:
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    img = cv2_imdecode_bytes(extracted.read())
                else:
                    img_path = dir_img_paths.get(stem)
                    if not img_path:
                        continue
                    img = cv2_imread(img_path)

                if img is None:
                    continue
                img_h, img_w = img.shape[:2]
                yolo_boxes = [
                    xyxy_to_yolo(cid, x1, y1, x2, y2, float(img_w), float(img_h))
                    for cid, x1, y1, x2, y2 in boxes
                ]
                out_fname = f"miotcd_{stem}.jpg"
                if not cv2_imwrite(os.path.join(out_img_dir, out_fname), img):
                    continue
                write_yolo_labels(os.path.join(out_lbl_dir, f"miotcd_{stem}.txt"), yolo_boxes)

                frame_cls = Counter(b[0] for b in boxes)
                split_img_counts[split_name] += 1
                split_box_counts[split_name].update(frame_cls)
                metadata_rows.append(
                    {
                        "filename": out_fname,
                        "split": split_name,
                        "source": "mio_tcd",
                        "sequence": "mio_tcd_cctv",
                        "weather": "cctv_mixed",
                        "camera_state": "stable",
                        "num_car": frame_cls.get(0, 0),
                        "num_motorcycle": frame_cls.get(1, 0),
                        "num_bus": frame_cls.get(2, 0),
                        "num_truck": frame_cls.get(3, 0),
                    }
                )
    finally:
        if tf is not None:
            tf.close()

    if not args.dry_run:
        if not os.path.exists(os.path.join(args.out_dir, "data.yaml")):
            write_data_yaml(args.out_dir)
        write_metadata_csv(os.path.join(args.out_dir, "dataset_metadata.csv"), metadata_rows, append=args.append_metadata)

    print("=" * 72)
    print("  MIO-TCD LOCALIZATION SUMMARY" + (" [DRY-RUN]" if args.dry_run else ""))
    print("=" * 72)
    print(f"Parsed annotated images: {len(per_image):,} | Train/Val overlap: {len(overlap)}")
    for sp in ("train", "val"):
        c = split_box_counts[sp]
        stats = ", ".join(f"{CLASS_NAMES[k]}={c.get(k, 0):,}" for k in sorted(CLASS_NAMES))
        print(f"[{sp.upper():<5}] imgs={split_img_counts[sp]:>5,} | {stats}")
    print("=" * 72)
    return split_box_counts, split_img_counts


def run_self_test() -> None:
    """Validate both CSV column orders, class mappings, tar extraction, and zero overlap."""
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="miotcd_selftest_")
    try:
        src_dir = os.path.join(tmp, "miotcd_raw")
        os.makedirs(os.path.join(src_dir, "train"), exist_ok=True)

        for idx in range(6):
            img = np.full((240, 320, 3), 200, dtype=np.uint8)
            cv2_imwrite(os.path.join(src_dir, "train", f"{idx:08d}.jpg"), img)

        # Mix both column orders plus ignored labels (pedestrian, bicycle, motorized_vehicle)
        csv_lines = [
            "00000000,motorcycle,10,20,50,80",
            "00000000,pedestrian,60,30,80,90",
            "00000001,15,25,120,140,single_unit_truck",
            "00000002,articulated_truck,20,20,180,150",
            "00000002,work_van,200,50,260,120",
            "00000003,bus,30,30,180,160",
            "00000003,motorized_vehicle,5,5,20,20",
            "00000004,pickup_truck,40,40,140,120",
            "00000005,car,50,50,150,140",
            "00000005,motorcycle,180,80,220,130",
        ]
        with open(os.path.join(src_dir, "gt_train.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines) + "\n")

        # Pack into a tar archive to test tar mode directly
        tar_path = os.path.join(tmp, "MIO-TCD-Localization.tar")
        with tarfile.open(tar_path, "w") as tf:
            tf.add(os.path.join(src_dir, "gt_train.csv"), arcname="MIO-TCD-Localization/gt_train.csv")
            for idx in range(6):
                tf.add(
                    os.path.join(src_dir, "train", f"{idx:08d}.jpg"),
                    arcname=f"MIO-TCD-Localization/train/{idx:08d}.jpg",
                )

        out_dir = os.path.join(tmp, "out")
        counts, imgs = run_miotcd(
            parse_args(["--tar-path", tar_path, "--out-dir", out_dir, "--max-train", "4", "--max-val", "2"])
        )
        assert imgs["train"] + imgs["val"] == 6
        assert counts["train"][1] + counts["val"][1] == 2, f"Expected 2 motorcycles: {counts}"
        assert counts["train"][3] + counts["val"][3] == 2, f"Expected 2 trucks: {counts}"
        assert counts["train"][2] + counts["val"][2] == 1, f"Expected 1 bus: {counts}"
        assert counts["train"][0] + counts["val"][0] == 3, f"Expected 3 cars (car+work_van+pickup_truck): {counts}"
        assert os.path.exists(os.path.join(out_dir, "dataset_metadata.csv"))
        print("PASS: prepare_miotcd.py self-test succeeded.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract balanced MIO-TCD Localization subset.")
    p.add_argument("--tar-path", default=None, help="Path to MIO-TCD-Localization.tar")
    p.add_argument("--data-dir", default=None, help="Extracted MIO-TCD directory containing gt_train.csv and train/")
    p.add_argument("--out-dir", default="unified_traffic_dataset")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--max-val", type=int, default=600)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append-metadata", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_miotcd(args)
