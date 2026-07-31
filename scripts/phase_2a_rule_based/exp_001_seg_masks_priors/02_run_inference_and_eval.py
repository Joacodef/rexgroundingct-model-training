"""
===============================================================================
SCRIPT:         02_run_inference_and_eval.py
PHASE:          Phase 2A — Statistical / Rule-Based Prior Baseline
OBJECTIVE:      Run validation inference using precomputed 3D spatial PDF 
                heatmaps, resample to canonical target CT RAS shape, threshold 
                at category cutoffs, save 4D NIfTI masks, and run automated 
                challenge metric evaluation.
USAGE:          python scripts/phase_2a_rule_based/exp_001_seg_masks_priors/02_run_inference_and_eval.py --split val --eval
===============================================================================
"""

import os
import sys
import gc
import json
import ctypes
import argparse
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import (
    DATA_DIR, DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    PREDICTIONS_DIR, LOGS_DIR
)
from scripts.phase_2a_rule_based.exp_001_seg_masks_priors.prior_engine import EmpiricalSpatialPDFBaseline


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for validation baseline inference and evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Tasks 2 & 3: Phase 2A Baseline Inference and Metric Evaluation"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--pdf_cache", type=str, default=None,
        help="Path to empirical spatial PDF npz cache (defaults to data/phase_2a/empirical_spatial_pdf_14cat.npz)"
    )
    parser.add_argument(
        "--dataset_json", type=str, default=str(DATASET_JSON),
        help="Path to dataset.json"
    )
    parser.add_argument(
        "--img_raw_dir", type=str, default=str(RAW_IMAGES_DIR),
        help="Path to raw CT images directory"
    )
    parser.add_argument(
        "--seg_raw_dir", type=str, default=str(RAW_MASKS_DIR),
        help="Path to raw CT segmentations directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--eval", action="store_true", default=True,
        help="Automatically run scripts/common/evaluate.py after inference (default: True)"
    )
    parser.add_argument(
        "--no_eval", action="store_false", dest="eval",
        help="Disable automatic evaluation"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Start index for processing entries"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for processing entries"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for Task 2 (Inference) and Task 3 (Evaluation)."""
    args = parse_args()

    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat.npz"
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_rule_based"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 2A — Tasks 2 & 3: Spatial Prior Inference & Evaluation")
    print(f"Target Split:        {args.split}")
    print(f"PDF Cache Path:      {pdf_cache_path}")
    print(f"Dataset JSON:        {args.dataset_json}")
    print(f"Output Directory:    {output_dir}")
    print("=" * 80)

    # Initialize Predictor Engine
    predictor = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        force_rebuild=False
    )

    # Load Dataset Metadata
    with open(args.dataset_json, 'r') as f:
        metadata = json.load(f)

    entries = metadata.get(args.split, [])
    if not entries:
        print(f"[ERROR] No entries found for split '{args.split}' in {args.dataset_json}")
        sys.exit(1)

    end_idx = args.end_idx if args.end_idx is not None else len(entries)
    entries = entries[args.start_idx:end_idx]

    missing_scans = 0
    generated_count = 0

    # Task 2: Validation Inference Loop
    for entry in tqdm(entries, desc=f"Phase 2A Baseline [{args.split}]"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue

        out_nii_path = output_dir / f"{scan_id}.nii.gz"
        raw_nifti_path = Path(args.img_raw_dir) / f"{scan_id}.nii.gz"
        if not raw_nifti_path.exists():
            tqdm.write(f"[WARNING] Raw image missing: {raw_nifti_path}")
            missing_scans += 1
            continue

        # 1. Read raw CT volume header & reorient to canonical RAS to get target shape (X, Y, Z)
        raw_nii = nib.load(str(raw_nifti_path))
        raw_nii_ras = nib.as_closest_canonical(raw_nii)
        target_shape_xyz = raw_nii_ras.shape # (X, Y, Z)
        original_affine = raw_nii.affine

        # 2. Extract Finding Prompts
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f.get("text", "") if isinstance(f, dict) else str(f) for f in findings]

        if not prompts:
            continue

        # 3. Generate 3D mask per finding prompt in (X, Y, Z) space
        finding_masks_xyz = []
        for prompt in prompts:
            mask_3d = predictor.generate_prediction_mask(target_shape_xyz, prompt) # shape: (X, Y, Z)
            finding_masks_xyz.append(mask_3d)

        # Stack to 4D array (F, X, Y, Z)
        pred_4d_fxyz = np.stack(finding_masks_xyz, axis=0).astype(np.uint8) # (F, X, Y, Z)

        # Save NIfTI file with original CT affine
        out_nii = nib.Nifti1Image(pred_4d_fxyz, original_affine)
        nib.save(out_nii, str(out_nii_path))
        generated_count += 1

        # Explicit memory cleanup
        del raw_nii, raw_nii_ras, finding_masks_xyz, pred_4d_fxyz, out_nii
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    print(f"\n[SUCCESS] Generated predictions for {generated_count} scans in {output_dir}")
    if missing_scans > 0:
        print(f"[WARNING] Skipped {missing_scans} missing CT files.")

    # Task 3: Automate Metric Evaluation via scripts/common/evaluate.py
    if args.eval and (generated_count > 0 or len(list(output_dir.glob("*.nii.gz"))) > 0):
        print("\n" + "=" * 80)
        exp_log_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_001_seg_masks_priors"
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        eval_json_path = exp_log_dir / f"eval_results_{args.split}.json"
        log_path = exp_log_dir / "eval.md"

        eval_script = ROOT_DIR / "scripts" / "common" / "evaluate.py"
        gt_dir = Path(args.seg_raw_dir)

        cmd = [
            sys.executable, str(eval_script),
            "--gt_dir", str(gt_dir),
            "--pred_dir", str(output_dir),
            "--split", args.split,
            "--dataset_json", args.dataset_json,
            "--output_json", str(eval_json_path)
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("\n--- Evaluation Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Evaluation Warnings/Errors ---")
            print(result.stderr)

        # Log results to markdown log file inside dedicated experiment subfolder
        with open(log_path, 'w') as f_log:
            f_log.write("# Phase 2A — Empirical Spatial Density Baseline Evaluation Log\n\n")
            f_log.write(f"- **Target Split:** {args.split}\n")
            f_log.write(f"- **PDF Cache Location:** `{pdf_cache_path}`\n")
            f_log.write(f"- **Prediction Directory:** `{output_dir}`\n")
            f_log.write(f"- **Evaluated Files:** {len(list(output_dir.glob('*.nii.gz')))}\n\n")
            f_log.write("## Quantitative Metric Evaluation Output\n\n```text\n")
            f_log.write(result.stdout)
            f_log.write("\n```\n")

        print(f"\n[INFO] Saved evaluation report to: {log_path}")


if __name__ == "__main__":
    main()
