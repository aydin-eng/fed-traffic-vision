"""COCO 2017 targeted minority-vehicle subset fetcher (motorcycle, truck, bus).

Downloads only the 241 MB annotations archive, selects minority-heavy images, and fetches
only those selected JPEGs in parallel (saving ~17.5 GB vs downloading full train2017.zip).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.common import (
    CLASS_NAMES,
    cv2_imdecode_bytes,
    cv2_imwrite,
    write_data_yaml,
    write_metadata_csv,
    write_yolo_labels,
    xyxy_to_yolo,
)

COCO_ANNOT_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Official COCO 2017 category IDs -> canonical 4-class schema
COCO_CAT_MAP: Dict[int, int] = {
    3: 0,  # car -> car
    4: 1,  # motorcycle -> motorcycle
    6: 2,  # bus -> bus
    8: 3,  # truck -> truck
}

MIN_BOX_PX = 10.0


def ensure_coco_annotations(cache_dir: str) -> Dict[str, str]:
    """Download and extract annotations_trainval2017.zip if needed."""
    os.makedirs(cache_dir, exist_ok=True)
    ann_dir = os.path.join(cache_dir, "annotations")
    train_json = os.path.join(ann_dir, "instances_train2017.json")
    val_json = os.path.join(ann_dir, "instances_val2017.json")

    if not (os.path.exists(train_json) and os.path.exists(val_json)):
        zip_path = os.path.join(cache_dir, "annotations_trainval2017.zip")
        if not os.path.exists(zip_path):
            print("Downloading COCO 2017 annotations (241 MB)...")
            urllib.request.urlretrieve(COCO_ANNOT_URL, zip_path)
        print("Extracting COCO 2017 annotations...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(("instances_train2017.json", "instances_val2017.json")):
                    zf.extract(member, cache_dir)

    return {"train": train_json, "val": val_json}


def select_coco_minority_images(
    json_path: str,
    max_images: int,
    seed: int = 42,
) -> List[Tuple[dict, List[Tuple[int, float, float, float, float]]]]:
    """Parse COCO instances JSON and select images containing motorcycle, truck, or bus."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {im["id"]: im for im in data["images"]}
    boxes_by_img: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        cat_id = ann.get("category_id")
        if cat_id not in COCO_CAT_MAP:
            continue
        x, y, w, h = ann["bbox"]
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        boxes_by_img[ann["image_id"]].append((COCO_CAT_MAP[cat_id], float(x), float(y), float(x + w), float(y + h)))

    tier1_motor_truck = []
    tier2_bus = []
    for img_id in sorted(boxes_by_img.keys()):
        b_list = boxes_by_img[img_id]
        cset = {b[0] for b in b_list}
        im_info = images_by_id.get(img_id)
        if not im_info:
            continue
        if 1 in cset or 3 in cset:
            tier1_motor_truck.append((im_info, b_list))
        elif 2 in cset:
            tier2_bus.append((im_info, b_list))

    rng = random.Random(seed)
    rng.shuffle(tier1_motor_truck)
    rng.shuffle(tier2_bus)
    return (tier1_motor_truck + tier2_bus)[:max_images]


def _download_one_image(url: str, timeout: int = 15) -> Optional[bytes]:
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            continue
    return None


def process_coco_split(
    split_name: str,
    json_path: str,
    out_dir: str,
    max_images: int,
    seed: int = 42,
    local_img_root: Optional[str] = None,
) -> Tuple[int, Counter, List[Dict[str, object]]]:
    """Download (or read locally in tests) selected COCO images and write YOLO labels."""
    selected = select_coco_minority_images(json_path, max_images=max_images, seed=seed)
    out_img_dir = os.path.join(out_dir, "images", split_name)
    out_lbl_dir = os.path.join(out_dir, "labels", split_name)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    raw_bytes_map: Dict[str, bytes] = {}
    if local_img_root:
        for im_info, _ in selected:
            cand = os.path.join(local_img_root, im_info["file_name"])
            if os.path.exists(cand):
                with open(cand, "rb") as f:
                    raw_bytes_map[im_info["file_name"]] = f.read()
    else:
        with ThreadPoolExecutor(max_workers=12) as pool:
            fut_map = {
                pool.submit(
                    _download_one_image,
                    im_info.get("coco_url") or f"http://images.cocodataset.org/{split_name}2017/{im_info['file_name']}",
                ): im_info["file_name"]
                for im_info, _ in selected
            }
            for fut in as_completed(fut_map):
                fname = fut_map[fut]
                raw = fut.result()
                if raw:
                    raw_bytes_map[fname] = raw

    added = 0
    box_counts: Counter = Counter()
    metadata_rows: List[Dict[str, object]] = []

    for im_info, boxes in selected:
        fname = im_info["file_name"]
        raw = raw_bytes_map.get(fname)
        if not raw:
            continue
        img = cv2_imdecode_bytes(raw)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        yolo_boxes = [
            xyxy_to_yolo(cid, x1, y1, x2, y2, float(img_w), float(img_h))
            for cid, x1, y1, x2, y2 in boxes
        ]
        stem = os.path.splitext(fname)[0]
        out_fname = f"coco_{stem}.jpg"
        if not cv2_imwrite(os.path.join(out_img_dir, out_fname), img):
            continue
        write_yolo_labels(os.path.join(out_lbl_dir, f"coco_{stem}.txt"), yolo_boxes)

        frame_cls = Counter(b[0] for b in boxes)
        added += 1
        box_counts.update(frame_cls)
        metadata_rows.append(
            {
                "filename": out_fname,
                "split": split_name,
                "source": "coco2017",
                "sequence": "coco_street",
                "weather": "street_mixed",
                "camera_state": "mobile",
                "num_car": frame_cls.get(0, 0),
                "num_motorcycle": frame_cls.get(1, 0),
                "num_bus": frame_cls.get(2, 0),
                "num_truck": frame_cls.get(3, 0),
            }
        )

    return added, box_counts, metadata_rows


def run_coco(args: argparse.Namespace) -> None:
    """Fetch targeted COCO 2017 minority vehicle subset into unified dataset."""
    json_paths = ensure_coco_annotations(args.cache_dir)
    all_rows: List[Dict[str, object]] = []

    print("=" * 72)
    print("  COCO 2017 TARGETED VEHICLE SUBSET")
    print("=" * 72)
    for split_name, max_n in (("train", args.max_train), ("val", args.max_val)):
        if max_n <= 0:
            continue
        added, counts, rows = process_coco_split(
            split_name, json_paths[split_name], args.out_dir, max_images=max_n, seed=args.seed
        )
        all_rows.extend(rows)
        stats = ", ".join(f"{CLASS_NAMES[k]}={counts.get(k, 0):,}" for k in sorted(CLASS_NAMES))
        print(f"[{split_name.upper():<5}] imgs={added:>5,} | {stats}")

    if all_rows:
        if not os.path.exists(os.path.join(args.out_dir, "data.yaml")):
            write_data_yaml(args.out_dir)
        write_metadata_csv(os.path.join(args.out_dir, "dataset_metadata.csv"), all_rows, append=args.append_metadata)
    print("=" * 72)


def run_self_test() -> None:
    """Validate COCO category mapping, iscrowd filtering, and box normalization."""
    import numpy as np

    tmp = tempfile.mkdtemp(prefix="coco_selftest_")
    try:
        img_dir = os.path.join(tmp, "imgs")
        os.makedirs(img_dir, exist_ok=True)
        cv2_imwrite(os.path.join(img_dir, "0001.jpg"), np.full((200, 300, 3), 180, dtype=np.uint8))
        cv2_imwrite(os.path.join(img_dir, "0002.jpg"), np.full((200, 300, 3), 180, dtype=np.uint8))

        coco_payload = {
            "images": [
                {"id": 1, "file_name": "0001.jpg", "width": 300, "height": 200},
                {"id": 2, "file_name": "0002.jpg", "width": 300, "height": 200},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 4, "bbox": [20, 30, 40, 50], "iscrowd": 0},   # motorcycle -> 1
                {"id": 11, "image_id": 1, "category_id": 8, "bbox": [100, 40, 80, 60], "iscrowd": 0},  # truck -> 3
                {"id": 12, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 40], "iscrowd": 0},   # person -> skip
                {"id": 13, "image_id": 1, "category_id": 6, "bbox": [50, 50, 90, 70], "iscrowd": 1},   # crowd -> skip
                {"id": 14, "image_id": 2, "category_id": 3, "bbox": [30, 30, 50, 40], "iscrowd": 0},   # car-only -> skip!
            ],
        }
        jpath = os.path.join(tmp, "instances.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(coco_payload, f)

        out_dir = os.path.join(tmp, "out")
        added, counts, rows = process_coco_split("train", jpath, out_dir, max_images=10, local_img_root=img_dir)
        assert added == 1, f"Car-only image should be skipped; expected 1, got {added}"
        assert counts == Counter({1: 1, 3: 1}), f"Unexpected counts: {counts}"
        assert len(rows) == 1
        print("PASS: prepare_coco_subset.py self-test succeeded.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch COCO 2017 minority vehicle subset.")
    p.add_argument("--out-dir", default="unified_traffic_dataset")
    p.add_argument("--cache-dir", default="coco_cache")
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--max-val", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append-metadata", action="store_true", default=False)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_coco(args)
