"""
Local Evaluation Script for ReXGroundingCT Challenge Metrics.

This script calculates the primary ranking metric (Average Dice) and the
Hit Rate (threshold 0.1, as specified by the official MICCAI challenge rules) 
over the 4D predictions (F, H, W, D).

It serves as a lightweight local alternative to the official evaluator 
for quick local validation loops during baseline and methodology development.
"""

import os
import json
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    RAW_MASKS_DIR,
    PREDICTIONS_DIR,
    DATASET_JSON,
    DATA_DIR,
    CATEGORY_MAP
)

CATEGORY_NAMES = CATEGORY_MAP

def compute_dice(pred_mask, gt_mask):
    """Computes the Dice Coefficient between two binary masks."""
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    
    if union == 0:
        # If both masks are empty, the match is perfect (1.0).
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    parser = argparse.ArgumentParser(description="Evaluate 4D predictions for ReXGroundingCT")
    
    parser.add_argument("--gt_dir", type=str, default=str(RAW_MASKS_DIR), help="Directory containing raw GT masks")
    parser.add_argument("--pred_dir", type=str, default=str(PREDICTIONS_DIR), help="Directory containing predicted masks")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json")
    
    # Derive the default output JSON based on the predictions directory
    default_out_json = str(DATA_DIR / "eval_results.json")
    
    parser.add_argument("--output_json", type=str, default=default_out_json, help="Path to save evaluation results")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    args = parser.parse_args()

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
            
        # Load images. Type casting to handle memory more efficiently.
        gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
        pred_img = nib.load(pred_path).get_fdata(dtype=np.float32)
        
        # --- SHAPE FIX: Auto-adjust GT dimensions to match expected (F, H, W, D) ---
        # 1. Expand dims if 3D (missing finding channel because it was 1)
        if gt_img.ndim == 3:
            gt_img = np.expand_dims(gt_img, axis=-1)  # Expand to pseudo 4D at the end (X, Y, Z, 1)
            
        # 2. Check if Findings channel is at the end (X, Y, Z, F) and move to front (F, X, Y, Z)
        if gt_img.ndim == 4:
            # We assume the smallest dimension is the findings channel.
            # E.g. (X, Y, Z, F) -> F is at axis 3
            if gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
                gt_img = np.moveaxis(gt_img, -1, 0)
                
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