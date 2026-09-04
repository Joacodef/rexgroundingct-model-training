"""
===============================================================================
SCRIPT:         Empty-Label Loading Diagnostic (2026-09-03 audit follow-up)
LOCATION:       scratch/audit_2026_09_03/diagnose_empty_labels.py
OBJECTIVE:      Job 96800's stderr shows MONAI's RandCropByPosNegLabeld receiving labels with
                zero foreground ("Num foregrounds 0"). Masks on disk are intact and image/mask
                spatial shapes agree, so the foreground is being lost somewhere in the
                load -> transpose -> crop -> prompt-sample path. This walks a sample of the
                train split through each stage of ReXDataset.__getitem__ and reports the stage
                at which the annotated voxels disappear.
USAGE:          Submitted via bash_scripts/diagnose_empty_labels.slurm (never on the login node).
===============================================================================
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

from scripts.config import DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, TEXT_CACHE_DIR
from scripts.common.orientation import load_nifti_ras


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the empty-label loading diagnostic."""
    p = argparse.ArgumentParser(description="Locate the stage at which annotated voxels vanish")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--num_scans", type=int, default=400, help="Random sample size (0 = full split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", type=str, default="scratch/audit_2026_09_03/empty_label_report.json")
    return p.parse_args()


def main() -> None:
    """Walk sampled scans through each ReXDataset loading stage and tally where foreground is lost."""
    args = parse_args()

    with open(DATASET_JSON, "r") as fh:
        entries = json.load(fh)[args.split]
    if args.num_scans and args.num_scans < len(entries):
        random.seed(args.seed)
        entries = random.sample(entries, args.num_scans)

    tally = {
        "checked": 0,
        "empty_on_disk": 0,
        "empty_after_ras_load": 0,
        "empty_after_transpose": 0,
        "empty_after_bbox_crop": 0,
        "intact": 0,
        "text_seg_count_mismatch": 0,
    }
    offenders = []

    for i, entry in enumerate(entries, 1):
        scan_id = entry["name"].replace(".nii.gz", "")
        img_path = Path(RAW_IMAGES_DIR) / f"{scan_id}.nii.gz"
        seg_path = Path(RAW_MASKS_DIR) / f"{scan_id}.nii.gz"
        if not img_path.exists() or not seg_path.exists():
            continue

        tally["checked"] += 1
        json_px = sum(entry["pixels"].values()) if "pixels" in entry else None

        # Stage 1: canonical RAS load (the centralized spatial engine)
        seg_ras, _, _ = load_nifti_ras(seg_path)
        nz_ras = int((seg_ras > 0).sum())

        # Stage 2: transpose to (F, Z, Y, X)
        seg_data = seg_ras.transpose((0, 3, 2, 1)).astype(np.float32)
        nz_tr = int((seg_data > 0).sum())

        # Stage 3: crop to the image's non-zero bounding box
        img_ras, _, _ = load_nifti_ras(img_path)
        img_data = img_ras.transpose((2, 1, 0))[None].astype(np.float32)
        _, _, bbox = crop_to_nonzero(img_data, None)
        seg_cropped = seg_data[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        nz_crop = int((seg_cropped > 0).sum())

        # Stage 4: does the text-embedding row count match the mask channel count?
        text_path = Path(TEXT_CACHE_DIR) / f"{scan_id}.pt"
        n_text = None
        if text_path.exists():
            import torch
            n_text = int(torch.load(text_path, map_location="cpu").shape[0])
            if n_text != seg_cropped.shape[0]:
                tally["text_seg_count_mismatch"] += 1

        if json_px == 0:
            tally["empty_on_disk"] += 1
        elif nz_ras == 0:
            tally["empty_after_ras_load"] += 1
            stage = "ras_load"
        elif nz_tr == 0:
            tally["empty_after_transpose"] += 1
            stage = "transpose"
        elif nz_crop == 0:
            tally["empty_after_bbox_crop"] += 1
            stage = "bbox_crop"
        else:
            tally["intact"] += 1
            stage = None

        if nz_crop == 0 and json_px:
            offenders.append({
                "scan_id": scan_id,
                "lost_at": stage,
                "json_pixels": json_px,
                "nonzero_after_ras": nz_ras,
                "nonzero_after_transpose": nz_tr,
                "nonzero_after_crop": nz_crop,
                "seg_shape_on_disk": list(seg_ras.shape),
                "img_shape_zyx": list(img_data.shape[1:]),
                "bbox": [[int(a), int(b)] for a, b in bbox],
                "n_text_rows": n_text,
                "n_seg_channels": int(seg_cropped.shape[0]),
            })

        if i % 25 == 0:
            print(f"[{i}/{len(entries)}] intact={tally['intact']} lost={len(offenders)}", flush=True)

    checked = max(1, tally["checked"])
    print("\n" + "=" * 72)
    print("EMPTY-LABEL LOADING DIAGNOSTIC")
    print("=" * 72)
    for k, v in tally.items():
        print(f"  {k:28} {v:6}   ({100.0 * v / checked:5.1f}%)" if k != "checked" else f"  {k:28} {v:6}")
    print("=" * 72)
    if offenders:
        print(f"\nFirst {min(10, len(offenders))} scans whose annotation vanished before the cropper:")
        for o in offenders[:10]:
            print(f"  {o['scan_id']:24} lost_at={o['lost_at']:10} json_px={o['json_pixels']:>8} "
                  f"seg_disk={o['seg_shape_on_disk']} bbox={o['bbox']}")

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"tally": tally, "offenders": offenders}, fh, indent=2)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
