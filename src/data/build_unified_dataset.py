"""Unified 3-source dataset orchestrator: UA-DETRAC + MIO-TCD Localization + COCO 2017."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from typing import Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.augment_minority import augment_minority_train
from src.data.common import CLASS_NAMES, write_data_yaml
from src.data.prepare_coco_subset import parse_args as coco_parse_args, run_coco
from src.data.prepare_detrac import parse_args as detrac_parse_args, run_detrac
from src.data.prepare_miotcd import parse_args as miotcd_parse_args, run_miotcd


def print_unified_report(out_dir: str) -> None:
    """Summarize final unified dataset counts by split and by source domain."""
    meta_path = os.path.join(out_dir, "dataset_metadata.csv")
    if not os.path.exists(meta_path):
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n" + "=" * 78)
    print("  UNIFIED 3-SOURCE DATASET FINAL REPORT (UA-DETRAC + MIO-TCD + COCO 2017)")
    print("=" * 78)
    for sp in ("train", "val", "test"):
        sp_rows = [r for r in rows if r["split"] == sp]
        if not sp_rows:
            continue
        by_src = Counter(r["source"] for r in sp_rows)
        cls_sums = {
            CLASS_NAMES[i]: sum(int(r[f"num_{CLASS_NAMES[i]}"]) for r in sp_rows)
            for i in sorted(CLASS_NAMES)
        }
        print(f"[{sp.upper():<5}] imgs={len(sp_rows):>6,} | sources={dict(by_src)}")
        print(f"        boxes={cls_sums}")
    print("=" * 78)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Build unified 3-source traffic detection dataset.")
    p.add_argument("--out-dir", default="unified_traffic_dataset")
    p.add_argument("--detrac-annot-zip", default="detrac_xml_annotations.zip")
    p.add_argument("--detrac-images-zip", default=None)
    p.add_argument("--detrac-images-dir", default=None)
    p.add_argument("--miotcd-tar", default=None)
    p.add_argument("--miotcd-dir", default=None)
    p.add_argument("--miotcd-max-train", type=int, default=4000)
    p.add_argument("--miotcd-max-val", type=int, default=600)
    p.add_argument("--coco-cache-dir", default="coco_cache")
    p.add_argument("--coco-max-train", type=int, default=2000)
    p.add_argument("--coco-max-val", type=int, default=300)
    p.add_argument("--rare-copies", type=int, default=1, help="Train minority augmentation copies (0 to disable).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. UA-DETRAC
    if args.detrac_images_zip or args.detrac_images_dir:
        d_args = [
            "--annot-zip", args.detrac_annot_zip,
            "--out-dir", args.out_dir,
            "--seed", str(args.seed),
            "--append-metadata",
        ]
        if args.detrac_images_zip:
            d_args += ["--images-zip", args.detrac_images_zip]
        else:
            d_args += ["--images-dir", args.detrac_images_dir]
        run_detrac(detrac_parse_args(d_args))

    # 2. MIO-TCD Localization
    if args.miotcd_tar or args.miotcd_dir:
        m_args = [
            "--out-dir", args.out_dir,
            "--max-train", str(args.miotcd_max_train),
            "--max-val", str(args.miotcd_max_val),
            "--seed", str(args.seed),
            "--append-metadata",
        ]
        if args.miotcd_tar:
            m_args += ["--tar-path", args.miotcd_tar]
        else:
            m_args += ["--data-dir", args.miotcd_dir]
        run_miotcd(miotcd_parse_args(m_args))

    # 3. COCO 2017 targeted minority subset
    if args.coco_max_train > 0 or args.coco_max_val > 0:
        c_args = [
            "--out-dir", args.out_dir,
            "--cache-dir", args.coco_cache_dir,
            "--max-train", str(args.coco_max_train),
            "--max-val", str(args.coco_max_val),
            "--seed", str(args.seed),
            "--append-metadata",
        ]
        run_coco(coco_parse_args(c_args))

    # 4. Optional minority augmentation (train only)
    if args.rare_copies > 0:
        augment_minority_train(args.out_dir, rare_copies=args.rare_copies, seed=args.seed)

    write_data_yaml(args.out_dir)
    print_unified_report(args.out_dir)


if __name__ == "__main__":
    main()
