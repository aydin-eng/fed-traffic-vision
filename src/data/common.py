"""Shared dataset I/O, bounding-box masking, YOLO label, and FL metadata helpers."""

from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CLASS_NAMES: Dict[int, str] = {
    0: "car",
    1: "motorcycle",
    2: "bus",
    3: "truck",
}

METADATA_COLUMNS: List[str] = [
    "filename",
    "split",
    "source",
    "sequence",
    "weather",
    "camera_state",
    "num_car",
    "num_motorcycle",
    "num_bus",
    "num_truck",
]

PAD_GRAY: Tuple[int, int, int] = (114, 114, 114)


def cv2_imread(path: str) -> Optional[np.ndarray]:
    """Unicode-safe OpenCV image reader (supports non-ASCII Windows paths)."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv2_imdecode_bytes(raw: bytes) -> Optional[np.ndarray]:
    """Decode raw image bytes with OpenCV."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_imwrite(path: str, img: np.ndarray) -> bool:
    """Unicode-safe OpenCV image writer."""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True


def mask_ignored_and_restore_targets(
    img: np.ndarray,
    ignored_xywh: Iterable[Tuple[float, float, float, float]],
    target_xyxy: Iterable[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Paint ignored regions neutral gray while preserving valid target vehicle pixels."""
    regions = list(ignored_xywh)
    if not regions:
        return img
    h, w = img.shape[:2]
    masked = img.copy()
    for left, top, width, height in regions:
        x1 = max(0, min(w, int(round(left))))
        y1 = max(0, min(h, int(round(top))))
        x2 = max(0, min(w, int(round(left + width))))
        y2 = max(0, min(h, int(round(top + height))))
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = PAD_GRAY

    for tx1, ty1, tx2, ty2 in target_xyxy:
        if tx2 > tx1 and ty2 > ty1:
            masked[ty1:ty2, tx1:tx2] = img[ty1:ty2, tx1:tx2]
    return masked


def xyxy_to_yolo(
    cls_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: float,
    img_h: float,
) -> Tuple[int, float, float, float, float]:
    """Convert pixel [x1, y1, x2, y2] to clamped normalized YOLO [cls, xc, yc, w, h]."""
    cx1 = max(0.0, min(img_w, x1))
    cy1 = max(0.0, min(img_h, y1))
    cx2 = max(0.0, min(img_w, x2))
    cy2 = max(0.0, min(img_h, y2))
    bw = max(1.0, cx2 - cx1)
    bh = max(1.0, cy2 - cy1)
    xc = max(0.0, min(1.0, (cx1 + bw / 2.0) / img_w))
    yc = max(0.0, min(1.0, (cy1 + bh / 2.0) / img_h))
    wn = max(0.001, min(1.0, bw / img_w))
    hn = max(0.001, min(1.0, bh / img_h))
    return (cls_id, xc, yc, wn, hn)


def read_yolo_labels(path: str) -> List[List[float]]:
    """Read YOLO format label file into [[cls_id, x, y, w, h], ...]."""
    boxes: List[List[float]] = []
    if not os.path.exists(path):
        return boxes
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(
                    [
                        int(float(parts[0])),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4]),
                    ]
                )
    return boxes


def write_yolo_labels(path: str, boxes: Sequence[Sequence[float]]) -> None:
    """Write normalized YOLO bounding boxes to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for b in boxes:
            f.write(
                f"{int(b[0])} {float(b[1]):.6f} {float(b[2]):.6f} "
                f"{float(b[3]):.6f} {float(b[4]):.6f}\n"
            )


def write_metadata_csv(path: str, rows: Sequence[Dict[str, object]], append: bool = False) -> None:
    """Write or append dataset_metadata.csv rows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if (append and file_exists) else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        if mode == "w":
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_data_yaml(out_dir: str) -> str:
    """Write Ultralytics YOLO data.yaml for the canonical 4-class schema."""
    yaml_path = os.path.join(out_dir, "data.yaml")
    norm_path = os.path.abspath(out_dir).replace("\\", "/")
    content = (
        "# Unified Multi-Source Traffic Vehicle Detection Dataset (UA-DETRAC + MIO-TCD + COCO2017)\n"
        f"path: {norm_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 4\n"
        "names:\n"
        "  0: car\n"
        "  1: motorcycle\n"
        "  2: bus\n"
        "  3: truck\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path
