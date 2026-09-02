"""
===============================================================================
SHARED UTILITY:   3D/4D Prediction & Segmentation Mask Statistical Profiler
LOCATION:         scripts/analysis/prediction_stats.py
OBJECTIVE:        Unified quantitative profiling tool for 3D/4D CT grounding
                  segmentation masks (VoxTell, Rule-Based Priors, Mean Teacher, 
                  GT masks). Computes volumetric (cm³), spatial alignment (mm), 
                  precision/recall, Dice, Hit Rate, and 3D connected component 
                  topology metrics (# Blobs, max blob size).
USAGE:            python scripts/analysis/prediction_stats.py \
                      --pred_dir ../data/predictions/phase_2b_voxtell_baseline \
                      --gt_dir ../data/raw/segmentations \
                      --split val \
                      --output_json logs/phase_2b_voxtell_baseline/diagnostic_stats.json
===============================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import label, center_of_mass

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    RAW_MASKS_DIR, PREDICTIONS_DIR, 
    DATASET_JSON, LOGS_DIR, CATEGORY_MAP
)
from scripts.common.orientation import load_nifti_ras


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for 3D/4D segmentation mask quantitative statistical profiling.
    """
    parser = argparse.ArgumentParser(
        description="Unified 3D/4D Segmentation Mask Quantitative Statistical Profiler"
    )
    parser.add_argument(
        "--pred_dir", type=str, default=str(PREDICTIONS_DIR / "phase_2b_voxtell_baseline"),
        help="Path to directory containing predicted 4D/3D NIfTI masks"
    )
    parser.add_argument(
        "--gt_dir", type=str, default=str(RAW_MASKS_DIR),
        help="Path to raw GT segmentations directory"
    )
    parser.add_argument(
        "--dataset_json", type=str, default=str(DATASET_JSON),
        help="Path to dataset.json"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Path to save quantitative diagnostic JSON output"
    )
    parser.add_argument(
        "--inspect_scan", type=str, default=None,
        help="Single scan ID to inspect quantitatively (e.g. --inspect_scan 10045)"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Start index for processing entries"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for processing entries"
    )
    return parser.parse_args()


def compute_mask_pair_metrics(pred_3d: np.ndarray, gt_3d: np.ndarray, spacing_mm: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Signature:
        compute_mask_pair_metrics(pred_3d: np.ndarray, gt_3d: np.ndarray, spacing_mm: tuple) -> dict

    Objective:
        Compute comprehensive volumetric (cm³), spatial alignment (mm), precision/recall,
        and 3D connected component topology metrics for a single prediction and GT 3D mask pair.

    Inputs:
        pred_3d (np.ndarray): 3D binary array of model predictions with shape (Z, Y, X).
        gt_3d (np.ndarray): 3D binary array of ground truth mask with shape (Z, Y, X).
        spacing_mm (tuple): Voxel physical spacing (dz, dy, dx) in millimeters. Default (1.0, 1.0, 1.0).

    Outputs:
        dict: Dictionary containing Dice, Precision, Recall, GT Vol (cm³), Pred Vol (cm³),
              FP Vol (cm³), FN Vol (cm³), Centroid Shift Δd (mm), and Blob topology counts.
    """
    pred_bool = (pred_3d > 0)
    gt_bool = (gt_3d > 0)

    voxel_vol_cm3 = np.prod(spacing_mm) / 1000.0  # mm³ to cm³

    tp_voxels = np.logical_and(pred_bool, gt_bool).sum()
    fp_voxels = np.logical_and(pred_bool, ~gt_bool).sum()
    fn_voxels = np.logical_and(~pred_bool, gt_bool).sum()

    gt_voxels = gt_bool.sum()
    pred_voxels = pred_bool.sum()

    gt_vol_cm3 = float(gt_voxels * voxel_vol_cm3)
    pred_vol_cm3 = float(pred_voxels * voxel_vol_cm3)
    fp_vol_cm3 = float(fp_voxels * voxel_vol_cm3)
    fn_vol_cm3 = float(fn_voxels * voxel_vol_cm3)

    # Dice Score
    union_voxels = pred_voxels + gt_voxels
    if union_voxels == 0:
        dice = 1.0
    else:
        dice = float(2.0 * tp_voxels / union_voxels)

    # Precision & Recall
    precision = float(tp_voxels / (tp_voxels + fp_voxels)) if (tp_voxels + fp_voxels) > 0 else 0.0
    recall = float(tp_voxels / (tp_voxels + fn_voxels)) if (tp_voxels + fn_voxels) > 0 else 0.0
    vol_ratio = float(pred_vol_cm3 / gt_vol_cm3) if gt_vol_cm3 > 0 else (0.0 if pred_vol_cm3 == 0 else 999.0)

    # 3D Centroid Computation (Centers of Mass)
    if gt_bool.any():
        gt_com_vox = center_of_mass(gt_bool)  # (Z, Y, X)
    else:
        gt_com_vox = None

    if pred_bool.any():
        pred_com_vox = center_of_mass(pred_bool)  # (Z, Y, X)
    else:
        pred_com_vox = None

    # Centroid Euclidean Shift Δd (mm)
    if gt_com_vox is not None and pred_com_vox is not None:
        dz = (pred_com_vox[0] - gt_com_vox[0]) * spacing_mm[0]
        dy = (pred_com_vox[1] - gt_com_vox[1]) * spacing_mm[1]
        dx = (pred_com_vox[2] - gt_com_vox[2]) * spacing_mm[2]
        centroid_err_mm = float(np.sqrt(dz**2 + dy**2 + dx**2))
    else:
        centroid_err_mm = None

    # 3D Connected Component Topology
    if gt_bool.any():
        gt_labeled, gt_num_blobs = label(gt_bool)
        gt_blob_sizes = np.bincount(gt_labeled.ravel())[1:]  # skip background
    else:
        gt_num_blobs = 0
        gt_blob_sizes = np.array([])

    if pred_bool.any():
        pred_labeled, pred_num_blobs = label(pred_bool)
        pred_blob_sizes = np.bincount(pred_labeled.ravel())[1:]
    else:
        pred_num_blobs = 0
        pred_blob_sizes = np.array([])

    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "gt_vol_cm3": gt_vol_cm3,
        "pred_vol_cm3": pred_vol_cm3,
        "fp_vol_cm3": fp_vol_cm3,
        "fn_vol_cm3": fn_vol_cm3,
        "vol_ratio": vol_ratio,
        "gt_com_vox": [float(x) for x in gt_com_vox] if gt_com_vox is not None else None,
        "pred_com_vox": [float(x) for x in pred_com_vox] if pred_com_vox is not None else None,
        "centroid_err_mm": centroid_err_mm,
        "gt_num_blobs": int(gt_num_blobs),
        "pred_num_blobs": int(pred_num_blobs),
        "gt_max_blob_vox": int(gt_blob_sizes.max()) if len(gt_blob_sizes) > 0 else 0,
        "pred_max_blob_vox": int(pred_blob_sizes.max()) if len(pred_blob_sizes) > 0 else 0
    }


def match_prompt_category(prompt_text: str) -> str:
    """
    Signature:
        match_prompt_category(prompt_text: str) -> str

    Objective:
        Identify the corresponding 14-category code (e.g., '2e') for a raw radiology finding prompt.

    Inputs:
        prompt_text (str): Free-text radiology report finding prompt string.

    Outputs:
        str: 14-category code string ('1a'..'2h'). Defaults to '2h' (Other focal) if unmatched.
    """
    cleaned = prompt_text.lower().strip()
    for code, cat_name in CATEGORY_MAP.items():
        if cat_name.lower() in cleaned:
            return code
    if "nodule" in cleaned or "mass" in cleaned:
        return "2d"
    elif "opacity" in cleaned or "consolidation" in cleaned:
        return "2c"
    elif "effusion" in cleaned:
        return "2e"
    elif "thickening" in cleaned:
        return "1d"
    elif "emphysema" in cleaned:
        return "1c"
    elif "bronch" in cleaned:
        return "1a"
    return "2h"


def main():
    """Main entry point executing full 3D/4D mask statistical profiling, error tier aggregation, and JSON report export."""
    args = parse_args()

    # Paths
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    dataset_json_path = Path(args.dataset_json)

    # Resolve experiment subfolder under logs/
    sub_phase = pred_dir.name
    if (LOGS_DIR / sub_phase).exists() or sub_phase.startswith("phase_"):
        exp_log_dir = LOGS_DIR / sub_phase / "exp_001_seg_masks_priors"
    else:
        exp_log_dir = LOGS_DIR / "common" / sub_phase
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    out_json_path = Path(args.output_json) if args.output_json else exp_log_dir / f"diagnostic_analysis_{args.split}.json"

    print("=" * 80)
    print("3D/4D Segmentation Mask Quantitative Statistical Profiler")
    print(f"Target Split:             {args.split}")
    print(f"Prediction Directory:     {pred_dir}")
    print(f"Ground Truth Directory:   {gt_dir}")
    print(f"Output Diagnostic JSON:   {out_json_path}")
    print("=" * 80)

    if not pred_dir.exists():
        print(f"[ERROR] Prediction directory does not exist: {pred_dir}")
        sys.exit(1)

    # Load metadata
    with open(dataset_json_path, 'r') as f:
        metadata = json.load(f)

    entries = metadata.get(args.split, [])
    if not entries:
        print(f"[ERROR] No entries found for split '{args.split}' in {dataset_json_path}")
        sys.exit(1)

    if args.inspect_scan:
        entries = [e for e in entries if args.inspect_scan in e.get("name", "")]
        if not entries:
            print(f"[ERROR] Specified scan ID '{args.inspect_scan}' not found in split '{args.split}'")
            sys.exit(1)

    end_idx = args.end_idx if args.end_idx is not None else len(entries)
    entries = entries[args.start_idx:end_idx]

    eval_records = []
    category_summary = {code: {"dices": [], "gt_vols": [], "pred_vols": [], "fp_vols": [], "fn_vols": [], "centroid_errs": [], "gt_blobs": [], "pred_blobs": []} for code in CATEGORY_MAP.keys()}

    missing_cases = 0

    print(f"\n[INFO] Profiling {len(entries)} scans...")
    for entry in tqdm(entries, desc="Analyzing 3D/4D Masks"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue

        gt_path = gt_dir / f"{scan_id}.nii.gz"
        pred_path = pred_dir / f"{scan_id}.nii.gz"

        if not gt_path.exists() or not pred_path.exists():
            missing_cases += 1
            continue

        try:
            gt_data, gt_nii, _ = load_nifti_ras(gt_path)
            pred_data, pred_nii, _ = load_nifti_ras(pred_path)

            spacing_mm = tuple(float(x) for x in gt_nii.header.get_zooms()[:3])
            gt_data = gt_data.astype(np.float32)
            pred_data = pred_data.astype(np.float32)
        except Exception as e:
            tqdm.write(f"[WARNING] Error reading NIfTI for {scan_id}: {e}")
            continue

        # Reorient GT / Pred dimensions to match (F, X, Y, Z)
        if gt_data.ndim == 3:
            gt_data = np.expand_dims(gt_data, axis=0)

        if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data = np.moveaxis(gt_data, -1, 0)

        if pred_data.ndim == 3:
            pred_data = np.expand_dims(pred_data, axis=0)

        if pred_data.ndim == 4 and pred_data.shape[-1] < np.min(pred_data.shape[:-1]):
            pred_data = np.moveaxis(pred_data, -1, 0)

        if gt_data.shape != pred_data.shape:
            tqdm.write(f"[WARNING] Shape mismatch for {scan_id}: GT {gt_data.shape} vs Pred {pred_data.shape}")
            continue

        num_findings = gt_data.shape[0]
        findings_meta = entry.get("findings", {})
        categories_dict = entry.get("categories", {})

        for f_idx in range(num_findings):
            # Extract prompt text
            if isinstance(findings_meta, dict):
                p_item = findings_meta.get(str(f_idx), {})
                prompt_text = p_item.get("text", "") if isinstance(p_item, dict) else str(p_item)
            elif isinstance(findings_meta, list) and f_idx < len(findings_meta):
                p_item = findings_meta[f_idx]
                prompt_text = p_item.get("text", "") if isinstance(p_item, dict) else str(p_item)
            else:
                prompt_text = ""

            # Extract category code
            cat_code = str(categories_dict.get(str(f_idx), ""))
            if not cat_code or cat_code not in CATEGORY_MAP:
                cat_code = match_prompt_category(prompt_text)

            # Compute finding-level 3D metrics
            metrics = compute_mask_pair_metrics(pred_data[f_idx], gt_data[f_idx], spacing_mm=spacing_mm)

            record = {
                "scan_id": scan_id,
                "finding_idx": f_idx,
                "category_code": cat_code,
                "category_name": CATEGORY_MAP.get(cat_code, "Unknown"),
                "prompt": prompt_text,
                "gt_path": str(gt_path),
                "pred_path": str(pred_path),
                **metrics
            }
            eval_records.append(record)

            # Accumulate category summary
            category_summary[cat_code]["dices"].append(metrics["dice"])
            category_summary[cat_code]["gt_vols"].append(metrics["gt_vol_cm3"])
            category_summary[cat_code]["pred_vols"].append(metrics["pred_vol_cm3"])
            category_summary[cat_code]["fp_vols"].append(metrics["fp_vol_cm3"])
            category_summary[cat_code]["fn_vols"].append(metrics["fn_vol_cm3"])
            category_summary[cat_code]["gt_blobs"].append(metrics["gt_num_blobs"])
            category_summary[cat_code]["pred_blobs"].append(metrics["pred_num_blobs"])
            if metrics["centroid_err_mm"] is not None:
                category_summary[cat_code]["centroid_errs"].append(metrics["centroid_err_mm"])

        del gt_data, pred_data, gt_nii, pred_nii

    if not eval_records:
        print("[ERROR] No valid finding pairs were evaluated.")
        sys.exit(1)

    # Macro & Category Statistics
    all_dices = [r["dice"] for r in eval_records]
    all_hits_01 = [1 if d >= 0.1 else 0 for d in all_dices]
    all_centroid_errs = [r["centroid_err_mm"] for r in eval_records if r["centroid_err_mm"] is not None]

    avg_dice = float(np.mean(all_dices))
    hit_rate_01 = float(np.mean(all_hits_01))
    mean_centroid_err = float(np.mean(all_centroid_errs)) if all_centroid_errs else 0.0

    # Categorized Error Tiers
    complete_misses = [r for r in eval_records if r["dice"] == 0.0]
    sub_threshold = [r for r in eval_records if 0.0 < r["dice"] < 0.1]
    hit_tier = [r for r in eval_records if 0.1 <= r["dice"] < 0.5]
    high_fidelity = [r for r in eval_records if r["dice"] >= 0.5]

    # Category Breakdown Formatting
    category_results = {}
    for code in CATEGORY_MAP.keys():
        stats = category_summary[code]
        c_dices = stats["dices"]
        c_count = len(c_dices)
        if c_count > 0:
            category_results[code] = {
                "name": CATEGORY_MAP[code],
                "count": c_count,
                "avg_dice": float(np.mean(c_dices)),
                "hit_rate_0.1": float(np.mean([1 if d >= 0.1 else 0 for d in c_dices])),
                "avg_gt_vol_cm3": float(np.mean(stats["gt_vols"])),
                "avg_pred_vol_cm3": float(np.mean(stats["pred_vols"])),
                "avg_fp_vol_cm3": float(np.mean(stats["fp_vols"])),
                "avg_fn_vol_cm3": float(np.mean(stats["fn_vols"])),
                "avg_centroid_err_mm": float(np.mean(stats["centroid_errs"])) if stats["centroid_errs"] else 0.0,
                "avg_gt_blobs": float(np.mean(stats["gt_blobs"])),
                "avg_pred_blobs": float(np.mean(stats["pred_blobs"]))
            }

    # Print Terminal Diagnostic Summary Table
    print("\n" + "=" * 105)
    print("                        3D/4D PREDICTION DIAGNOSTIC STATISTICAL PROFILER SUMMARY")
    print("=" * 105)
    print(f"Total Evaluated Findings:     {len(eval_records)} across {len(set(r['scan_id'] for r in eval_records))} scans")
    print(f"Average Dice (Primary Metric): {avg_dice:.4f}")
    print(f"Hit Rate (Dice >= 0.1):       {hit_rate_01:.4f} ({sum(all_hits_01)} / {len(eval_records)} findings)")
    print(f"Mean Centroid Shift (Δd):     {mean_centroid_err:.2f} mm")
    print("-" * 105)
    print("ERROR TIERS BREAKDOWN:")
    print(f"  - Complete Misses (Dice = 0.0):        {len(complete_misses):<4} ({len(complete_misses)/len(eval_records)*100:.1f}%)")
    print(f"  - Sub-Threshold (0.0 < Dice < 0.1):    {len(sub_threshold):<4} ({len(sub_threshold)/len(eval_records)*100:.1f}%)")
    print(f"  - Hit Rate Tier (0.1 <= Dice < 0.5):   {len(hit_tier):<4} ({len(hit_tier)/len(eval_records)*100:.1f}%)")
    print(f"  - High Fidelity Match (Dice >= 0.5):  {len(high_fidelity):<4} ({len(high_fidelity)/len(eval_records)*100:.1f}%)")
    print("=" * 105)
    print(f"{'Code':<5} {'Category Name':<30} {'Count':<6} {'Dice':<7} {'HitRate':<8} {'GT Vol':<8} {'Pred Vol':<9} {'FP Vol':<8} {'Shift(mm)':<9}")
    print("-" * 105)
    for code, res in category_results.items():
        print(f"{code:<5} {res['name']:<30} {res['count']:<6} {res['avg_dice']:.4f}  {res['hit_rate_0.1']:.4f}   {res['avg_gt_vol_cm3']:<8.2f} {res['avg_pred_vol_cm3']:<9.2f} {res['avg_fp_vol_cm3']:<8.2f} {res['avg_centroid_err_mm']:<9.1f}")
    print("=" * 105)

    # Save JSON Diagnostic Report
    report = {
        "split": args.split,
        "prediction_dir": str(pred_dir),
        "total_findings_evaluated": len(eval_records),
        "macro_metrics": {
            "average_dice": avg_dice,
            "hit_rate_0.1": hit_rate_01,
            "mean_centroid_err_mm": mean_centroid_err,
            "complete_misses_count": len(complete_misses),
            "sub_threshold_count": len(sub_threshold),
            "hit_tier_count": len(hit_tier),
            "high_fidelity_count": len(high_fidelity)
        },
        "category_breakdown": category_results,
        "all_records": eval_records
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"\n[SUCCESS] Saved diagnostic statistical analysis JSON to {out_json_path}")


if __name__ == "__main__":
    main()
