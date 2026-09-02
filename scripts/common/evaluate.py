"""
===============================================================================
SCRIPT:         Local Challenge Metric Evaluator
LOCATION:       scripts/common/evaluate.py
OBJECTIVE:      Calculates primary challenge ranking metric (Average Dice) 
                and Hit Rate (threshold 0.1) across 4D predictions (F, X, Y, Z).
USAGE:          python scripts/common/evaluate.py \
                    --gt_dir ../data/raw/segmentations \
                    --pred_dir ../data/predictions/phase_2a_rule_based \
                    --split val
===============================================================================
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    RAW_MASKS_DIR,
    RAW_IMAGES_DIR,
    PREDICTIONS_DIR,
    DATASET_JSON,
    CATEGORY_MAP
)
from scripts.common.orientation import load_nifti_ras

CATEGORY_NAMES = CATEGORY_MAP

def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Signature:
        compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float

    Objective:
        Compute the 3D/4D Dice Similarity Coefficient (DSC) between prediction and GT binary masks.

    Inputs:
        pred_mask (np.ndarray): Binary prediction mask array.
        gt_mask (np.ndarray): Binary ground truth mask array.

    Outputs:
        float: Dice coefficient value in [0.0, 1.0]. Returns 1.0 if both masks are empty.
    """
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    
    if union == 0:
        # If both masks are empty, Dice equals 1.0.
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    """Main CLI entry point for computing challenge metrics (Average Dice & Hit Rate) over 4D predictions."""
    parser = argparse.ArgumentParser(description="Evaluate 4D predictions for ReXGroundingCT")
    
    parser.add_argument("--gt_dir", type=str, default=str(RAW_MASKS_DIR), help="Directory containing raw GT masks")
    parser.add_argument("--img_dir", type=str, default=str(RAW_IMAGES_DIR), help="Directory containing raw CT images for spatial anchoring")
    parser.add_argument("--pred_dir", type=str, default=str(PREDICTIONS_DIR), help="Directory containing predicted masks")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json")
    
    parser.add_argument("--output_json", type=str, default=None, help="Path to save evaluation results (defaults to matching logs/ directory)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing dataset entries")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing dataset entries (exclusive)")
    args = parser.parse_args()

    # Intelligent default output_json resolution matching logs/ directory layout
    if not args.output_json:
        from scripts.config import LOGS_DIR
        pred_path = Path(args.pred_dir).resolve()
        sub_phase = pred_path.name
        if (LOGS_DIR / sub_phase).exists() or sub_phase.startswith("phase_"):
            log_target_dir = LOGS_DIR / sub_phase
        else:
            log_target_dir = LOGS_DIR / "common"
        log_target_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.start_idx}_{args.end_idx}" if (args.start_idx != 0 or args.end_idx is not None) else ""
        args.output_json = str(log_target_dir / f"eval_results_{args.split}{suffix}.json")

    if not all([args.gt_dir, args.pred_dir, args.dataset_json]):
        print("[ERROR] Missing required paths. Please ensure DATA_PREP_DIR, DATA_PRED_DIR, and DATASET_JSON are set in your environment variables or passed as arguments.")
        return

    print(f"Loading metadata from {args.dataset_json}")
    with open(args.dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get(args.split, [])
    if not entries:
        print(f"[ERROR] No entries found for split '{args.split}' in dataset.json")
        return

    entries = entries[args.start_idx : args.end_idx]

    all_dices = []
    hits_01 = 0
    total_findings = 0
    missing_cases = 0
    category_metrics = {}

    for entry in tqdm(entries, desc=f"Evaluating {args.split} Scans"):
        scan_id = entry["name"].replace(".nii.gz", "")
        
        # The raw GT masks are named identically to the images
        gt_path = os.path.join(args.gt_dir, f"{scan_id}.nii.gz")
        pred_path = os.path.join(args.pred_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path):
            tqdm.write(f"[WARNING] GT not found: {gt_path}. Skipping.")
            missing_cases += 1
            continue
            
        if not os.path.exists(pred_path):
            tqdm.write(f"[WARNING] Prediction not found: {pred_path}. Skipping.")
            missing_cases += 1
            continue
            
        # Load GT and Prediction 4D volumes in canonical RAS space
        gt_img, _, _ = load_nifti_ras(gt_path)
        pred_img, _, _ = load_nifti_ras(pred_path)
        
        # Expand 3D volumes to 4D (1, X, Y, Z) if single finding
        if gt_img.ndim == 3:
            gt_img = np.expand_dims(gt_img, axis=0)
        if pred_img.ndim == 3:
            pred_img = np.expand_dims(pred_img, axis=0)
                
        # Validate dimensions (F, H, W, D)
        if gt_img.ndim != 4 or pred_img.ndim != 4:
            tqdm.write(f"[ERROR] Expected 4D for {scan_id}. GT: {gt_img.shape}, Pred: {pred_img.shape}")
            continue
            
        if gt_img.shape != pred_img.shape:
            tqdm.write(f"[ERROR] Shape mismatch for {scan_id}. GT: {gt_img.shape}, Pred: {pred_img.shape}")
            continue
            
        num_gt_findings = gt_img.shape[0]
        if pred_img.shape[0] != num_gt_findings:
            tqdm.write(f"[ERROR] Findings count mismatch for {scan_id}. GT: {num_gt_findings}, Pred: {pred_img.shape[0]}")
            continue
            
        # Compute finding-level metrics
        categories_dict = entry.get("categories", {})
        for f_idx in range(num_gt_findings):
            dice = compute_dice(pred_img[f_idx], gt_img[f_idx])
            all_dices.append(dice)
            
            # Category extraction
            cat_code = str(categories_dict.get(str(f_idx), "unknown"))
            if cat_code not in category_metrics:
                category_metrics[cat_code] = {"dices": [], "hits_01": 0}
            category_metrics[cat_code]["dices"].append(dice)
            
            # Challenge specific Hit Rate threshold (Dice >= 0.1)
            if dice >= 0.1:
                hits_01 += 1
                category_metrics[cat_code]["hits_01"] += 1
            total_findings += 1

    if total_findings == 0:
        print("[ERROR] No valid findings were evaluated.")
        return

    # Aggregation
    avg_dice = np.mean(all_dices)
    hit_rate = hits_01 / total_findings

    category_breakdown = {}
    for cat_code, stats in sorted(category_metrics.items()):
        c_dices = stats["dices"]
        c_count = len(c_dices)
        category_breakdown[cat_code] = {
            "name": CATEGORY_NAMES.get(cat_code, "Unknown"),
            "count": c_count,
            "average_dice": float(np.mean(c_dices)) if c_count > 0 else 0.0,
            "hit_rate_0.1": float(stats["hits_01"] / c_count) if c_count > 0 else 0.0
        }

    results = {
        "split": args.split,
        "total_cases_evaluated": len(entries) - missing_cases,
        "total_findings_evaluated": total_findings,
        "average_dice": float(avg_dice),
        "hit_rate_0.1": float(hit_rate),
        "category_breakdown": category_breakdown
    }

    print("\n" + "="*60)
    print("          EVALUATION RESULTS")
    print("="*60)
    print(f"Average Dice (Primary Metric): {avg_dice:.4f}")
    print(f"Hit Rate (Dice >= 0.1)       : {hit_rate:.4f}")
    print("="*60)
    print(f"{'Code':<5} {'Category Name':<32} {'Count':<7} {'Dice':<8} {'HitRate':<8}")
    print("-" * 60)
    for cat_code, cat_stats in category_breakdown.items():
        print(f"{cat_code:<5} {cat_stats['name']:<32} {cat_stats['count']:<7} {cat_stats['average_dice']:.4f}   {cat_stats['hit_rate_0.1']:.4f}")
    print("="*60)
    print(f"Missing/Skipped Cases        : {missing_cases}")
    
    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results successfully saved to {args.output_json}")

if __name__ == "__main__":
    main()