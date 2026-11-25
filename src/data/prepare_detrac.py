"""UA-DETRAC balanced extractor with stratified video splits and <ignored_region> masking."""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.common import (
    cv2_imdecode_bytes,
    cv2_imread,
    cv2_imwrite,
    mask_ignored_and_restore_targets,
    write_data_yaml,
    write_metadata_csv,
    write_yolo_labels,
    xyxy_to_yolo,
)

DETRAC_VEHICLE_MAP: Dict[str, int] = {
    "car": 0,
    "van": 0,
    "bus": 2,
    "others": 3,
}

MIN_BOX_PX = 5.0


@dataclass
class PixelBox:
    cls_id: int
    left: float
    top: float
    width: float
    height: float

    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(img_w, int(round(self.left))))
        y1 = max(0, min(img_h, int(round(self.top))))
        x2 = max(0, min(img_w, int(round(self.left + self.width))))
        y2 = max(0, min(img_h, int(round(self.top + self.height))))
        return x1, y1, x2, y2


@dataclass
class SequenceData:
    name: str
    weather: str
    camera_state: str
    ignored_regions: List[Tuple[float, float, float, float]]
    frames: Dict[int, List[PixelBox]]
    class_counts: Counter


def parse_detrac_xml(xml_path: str) -> SequenceData:
    """Parse one UA-DETRAC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq_name = root.attrib.get("name", os.path.splitext(os.path.basename(xml_path))[0]).strip()

    seq_attr = root.find("sequence_attribute")
    weather = "unknown"
    camera_state = "unknown"
    if seq_attr is not None:
        weather = (
            seq_attr.attrib.get("sence_weather")
            or seq_attr.attrib.get("scene_weather")
            or "unknown"
        ).strip().lower()
        camera_state = seq_attr.attrib.get("camera_state", "unknown").strip().lower()

    ignored_regions: List[Tuple[float, float, float, float]] = []
    ir_elem = root.find("ignored_region")
    if ir_elem is not None:
        for bx in ir_elem.findall("box"):
            try:
                l = float(bx.attrib.get("left", 0.0))
                t = float(bx.attrib.get("top", 0.0))
                w = float(bx.attrib.get("width", 0.0))
                h = float(bx.attrib.get("height", 0.0))
                if w > 0 and h > 0:
                    ignored_regions.append((l, t, w, h))
            except (TypeError, ValueError):
                continue

    frames: Dict[int, List[PixelBox]] = {}
    class_counts: Counter = Counter()
    for fe in root.iter("frame"):
        n_str = fe.attrib.get("num")
        if not n_str:
            continue
        fnum = int(n_str)
        boxes: List[PixelBox] = []
        tl = fe.find("target_list")
        if tl is not None:
            for tgt in tl.findall("target"):
                bx = tgt.find("box")
                att = tgt.find("attribute")
                if bx is None or att is None:
                    continue
                vt = att.attrib.get("vehicle_type", "car").strip().lower()
                if vt not in DETRAC_VEHICLE_MAP:
                    continue
                cid = DETRAC_VEHICLE_MAP[vt]
                try:
                    l = float(bx.attrib.get("left", 0.0))
                    t = float(bx.attrib.get("top", 0.0))
                    w = float(bx.attrib.get("width", 0.0))
                    h = float(bx.attrib.get("height", 0.0))
                except (TypeError, ValueError):
                    continue
                if w < MIN_BOX_PX or h < MIN_BOX_PX:
                    continue
                boxes.append(PixelBox(cls_id=cid, left=l, top=t, width=w, height=h))
                class_counts[cid] += 1
        if boxes:
            frames[fnum] = boxes

    return SequenceData(
        name=seq_name,
        weather=weather,
        camera_state=camera_state,
        ignored_regions=ignored_regions,
        frames=frames,
        class_counts=class_counts,
    )


def stratified_video_split(
    sequences: List[SequenceData],
    val_ratio: float = 0.18,
    seed: int = 42,
) -> Tuple[List[SequenceData], List[SequenceData]]:
    """Split video sequences into train/val with zero overlap and stratified rare classes."""
    rng = random.Random(seed)
    truck_seqs: List[SequenceData] = []
    bus_seqs: List[SequenceData] = []
    car_seqs: List[SequenceData] = []

    for s in sorted(sequences, key=lambda x: x.name):
        if s.class_counts.get(3, 0) > 0:
            truck_seqs.append(s)
        elif s.class_counts.get(2, 0) > 0:
            bus_seqs.append(s)
        else:
            car_seqs.append(s)

    train_out: List[SequenceData] = []
    val_out: List[SequenceData] = []
    for bucket, min_val in ((truck_seqs, 4), (bus_seqs, 3), (car_seqs, 1)):
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        if not shuffled:
            continue
        n_val = max(min_val if len(shuffled) > min_val else 0, int(round(len(shuffled) * val_ratio)))
        n_val = min(n_val, max(0, len(shuffled) - 1))
        val_out.extend(shuffled[:n_val])
        train_out.extend(shuffled[n_val:])
    return train_out, val_out


def select_adaptive_frames(
    seq: SequenceData,
    truck_step: int,
    bus_step: int,
    car_step: int,
    max_car_only: Optional[int],
) -> List[int]:
    """Sample frames denser for truck and bus than car-only frames."""
    sorted_nums = sorted(seq.frames.keys())
    truck_frames = [n for n in sorted_nums if any(b.cls_id == 3 for b in seq.frames[n])]
    bus_frames = [
        n
        for n in sorted_nums
        if any(b.cls_id == 2 for b in seq.frames[n]) and not any(b.cls_id == 3 for b in seq.frames[n])
    ]
    car_only_frames = [n for n in sorted_nums if all(b.cls_id == 0 for b in seq.frames[n])]

    selected = set(truck_frames[:: max(1, truck_step)])
    selected.update(bus_frames[:: max(1, bus_step)])
    sampled_car = car_only_frames[:: max(1, car_step)]
    if max_car_only is not None and max_car_only >= 0:
        sampled_car = sampled_car[:max_car_only]
    selected.update(sampled_car)
    return sorted(selected)


def load_xml_sequences(annot_dir: Optional[str], annot_zip: Optional[str]) -> Tuple[List[SequenceData], List[SequenceData]]:
    """Load XMLs from an extracted directory or directly from detrac_xml_annotations.zip."""
    cleanup_dir: Optional[str] = None
    try:
        resolved_dir = annot_dir
        if (not resolved_dir or not os.path.isdir(resolved_dir)) and annot_zip:
            cleanup_dir = tempfile.mkdtemp(prefix="detrac_xml_")
            with zipfile.ZipFile(annot_zip, "r") as zf:
                zf.extractall(cleanup_dir)
            resolved_dir = cleanup_dir
        if not resolved_dir:
            raise ValueError("Provide --annot-dir or --annot-zip.")

        train_xmls = sorted(glob.glob(os.path.join(resolved_dir, "**", "train", "*.xml"), recursive=True))
        test_xmls = sorted(glob.glob(os.path.join(resolved_dir, "**", "test", "*.xml"), recursive=True))
        if not train_xmls:
            raise FileNotFoundError(f"No train XML files found under {resolved_dir}")
        return [parse_detrac_xml(p) for p in train_xmls], [parse_detrac_xml(p) for p in test_xmls]
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)


def run_detrac(args: argparse.Namespace) -> Dict[str, Counter]:
    """Extract balanced UA-DETRAC dataset or compute dry-run statistics."""
    train_all, test_seqs = load_xml_sequences(args.annot_dir, args.annot_zip)
    train_seqs, val_seqs = stratified_video_split(train_all, val_ratio=args.val_ratio, seed=args.seed)

    overlap = {s.name for s in train_seqs} & {s.name for s in val_seqs}
    if overlap:
        raise RuntimeError(f"Train/val sequence overlap detected: {overlap}")

    if args.max_sequences is not None:
        train_seqs = train_seqs[: args.max_sequences]
        val_seqs = val_seqs[: args.max_sequences]
        test_seqs = test_seqs[: args.max_sequences]

    split_configs = {
        "train": (train_seqs, args.train_truck_step, args.train_bus_step, args.train_car_step, args.max_car_only),
        "val": (val_seqs, args.val_truck_step, args.val_bus_step, args.val_car_step, args.max_val_car_only),
        "test": (test_seqs, args.test_step, args.test_step, args.test_step, None),
    }

    split_box_counts: Dict[str, Counter] = {k: Counter() for k in split_configs}
    split_img_counts: Counter = Counter()
    metadata_rows: List[Dict[str, object]] = []

    zf: Optional[zipfile.ZipFile] = None
    zip_keys: set = set()
    if not args.dry_run:
        for split in split_configs:
            os.makedirs(os.path.join(args.out_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(args.out_dir, "labels", split), exist_ok=True)
        if args.images_zip:
            zf = zipfile.ZipFile(args.images_zip, "r")
            zip_keys = set(zf.namelist())
        elif not args.images_dir:
            raise ValueError("Specify --images-zip, --images-dir, or --dry-run.")

    try:
        for split_name, (seqs, t_step, b_step, c_step, c_cap) in split_configs.items():
            for seq in seqs:
                frame_nums = select_adaptive_frames(seq, t_step, b_step, c_step, c_cap)
                if args.max_frames_per_seq is not None:
                    frame_nums = frame_nums[: args.max_frames_per_seq]
                for fnum in frame_nums:
                    boxes = seq.frames[fnum]
                    if args.dry_run:
                        split_img_counts[split_name] += 1
                        for b in boxes:
                            split_box_counts[split_name][b.cls_id] += 1
                        continue

                    img: Optional[np.ndarray] = None
                    if zf is not None:
                        cand_keys = [
                            f"DETRAC-Images/{seq.name}/img{fnum:05d}.jpg",
                            f"{seq.name}/img{fnum:05d}.jpg",
                        ]
                        matched = next((k for k in cand_keys if k in zip_keys), None)
                        if matched is None:
                            continue
                        img = cv2_imdecode_bytes(zf.read(matched))
                    else:
                        cand_paths = [
                            os.path.join(args.images_dir, seq.name, f"img{fnum:05d}.jpg"),
                            os.path.join(args.images_dir, "DETRAC-Images", seq.name, f"img{fnum:05d}.jpg"),
                        ]
                        src_path = next((p for p in cand_paths if os.path.exists(p)), None)
                        if src_path is None:
                            continue
                        img = cv2_imread(src_path)
                    if img is None:
                        continue

                    img_h, img_w = img.shape[:2]
                    if args.mask_ignored and seq.ignored_regions:
                        target_xyxy = [b.to_xyxy(img_w, img_h) for b in boxes]
                        img = mask_ignored_and_restore_targets(img, seq.ignored_regions, target_xyxy)

                    fname = f"{seq.name}_img{fnum:05d}"
                    out_img = os.path.join(args.out_dir, "images", split_name, f"{fname}.jpg")
                    out_lbl = os.path.join(args.out_dir, "labels", split_name, f"{fname}.txt")
                    if not cv2_imwrite(out_img, img):
                        continue
                    yolo_boxes = [
                        xyxy_to_yolo(b.cls_id, b.left, b.top, b.left + b.width, b.top + b.height, float(img_w), float(img_h))
                        for b in boxes
                    ]
                    write_yolo_labels(out_lbl, yolo_boxes)

                    frame_cls = Counter(b.cls_id for b in boxes)
                    split_img_counts[split_name] += 1
                    split_box_counts[split_name].update(frame_cls)
                    metadata_rows.append(
                        {
                            "filename": f"{fname}.jpg",
                            "split": split_name,
                            "source": "ua_detrac",
                            "sequence": seq.name,
                            "weather": seq.weather,
                            "camera_state": seq.camera_state,
                            "num_car": frame_cls.get(0, 0),
                            "num_motorcycle": frame_cls.get(1, 0),
                            "num_bus": frame_cls.get(2, 0),
                            "num_truck": frame_cls.get(3, 0),
                        }
                    )
    finally:
        if zf is not None:
            zf.close()

    if not args.dry_run:
        write_data_yaml(args.out_dir)
        write_metadata_csv(os.path.join(args.out_dir, "dataset_metadata.csv"), metadata_rows, append=args.append_metadata)

    print("=" * 72)
    print("  UA-DETRAC BALANCED SUMMARY" + (" [DRY-RUN]" if args.dry_run else ""))
    print("=" * 72)
    print(f"Sequences -> train: {len(train_seqs)}, val: {len(val_seqs)}, test: {len(test_seqs)} (overlap: {len(overlap)})")
    for sp in ("train", "val", "test"):
        c = split_box_counts[sp]
        car_n, bus_n, truck_n = c.get(0, 0), c.get(2, 0), c.get(3, 0)
        rt = f"{car_n / truck_n:.1f}:1" if truck_n else "inf"
        print(f"[{sp.upper():<5}] imgs={split_img_counts[sp]:>5,} | car={car_n:>6,} | bus={bus_n:>5,} | truck={truck_n:>5,} | car:truck={rt}")
    print("=" * 72)
    return split_box_counts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract class-balanced UA-DETRAC split.")
    p.add_argument("--annot-dir", default=None)
    p.add_argument("--annot-zip", default="detrac_xml_annotations.zip")
    p.add_argument("--images-zip", default=None)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--out-dir", default="unified_traffic_dataset")
    p.add_argument("--val-ratio", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-truck-step", type=int, default=2)
    p.add_argument("--train-bus-step", type=int, default=15)
    p.add_argument("--train-car-step", type=int, default=120)
    p.add_argument("--max-car-only", type=int, default=3)
    p.add_argument("--val-truck-step", type=int, default=2)
    p.add_argument("--val-bus-step", type=int, default=15)
    p.add_argument("--val-car-step", type=int, default=120)
    p.add_argument("--max-val-car-only", type=int, default=3)
    p.add_argument("--test-step", type=int, default=20)
    p.add_argument("--mask-ignored", dest="mask_ignored", action="store_true", default=True)
    p.add_argument("--no-mask-ignored", dest="mask_ignored", action="store_false")
    p.add_argument("--append-metadata", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--max-frames-per-seq", type=int, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run_detrac(parse_args())
