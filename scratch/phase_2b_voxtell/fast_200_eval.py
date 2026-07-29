import os
import json
import numpy as np
import nibabel as nib
from dotenv import load_dotenv
from multiprocessing import Pool

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def evaluate_single_scan(args):
    idx, scan_id, pred_dir, gt_dir = args
    pred_fname = f"{scan_id}.nii.gz"
    pred_path = os.path.join(pred_dir, pred_fname)
    gt_path = os.path.join(gt_dir, pred_fname)
    
    if not os.path.exists(pred_path) or not os.path.exists(gt_path):
        return []

    try:
        pred_nii = nib.load(pred_path)
        pred_data = np.asanyarray(pred_nii.dataobj)
        if pred_data.ndim == 3: pred_data = np.expand_dims(pred_data, axis=-1)
        if pred_data.ndim == 4 and pred_data.shape[-1] < np.min(pred_data.shape[:-1]):
            pred_data_f = np.moveaxis(pred_data, -1, 0)
        else:
            pred_data_f = pred_data
            
        gt_nii = nib.load(gt_path)
        gt_data = np.asanyarray(gt_nii.dataobj)
        if gt_data.ndim == 3: gt_data = np.expand_dims(gt_data, axis=-1)
        if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
            gt_data_f = np.moveaxis(gt_data, -1, 0)
        else:
            gt_data_f = gt_data
            
        results = []
        n_channels = min(pred_data_f.shape[0], gt_data_f.shape[0])
        for c in range(n_channels):
            pm = pred_data_f[c]
            gm = gt_data_f[c]
            d = compute_dice(pm, gm)
            is_empty = (np.sum(pm > 0) == 0)
            results.append((idx, c, d, is_empty, np.sum(pm > 0)))
        return results
    except Exception as e:
        return []

def main():
    load_dotenv(override=True)
    gt_dir = os.environ["SEG_RAW_DIR"]
    pred_dir = os.environ.get("DATA_PRED_DIR", "data/predictions")
    dataset_json = os.environ["DATASET_JSON"]

    with open(dataset_json) as f:
        meta = json.load(f)
        
    val_cases = meta.get("val", [])
    task_args = [(idx, entry["name"].replace(".nii.gz", ""), pred_dir, gt_dir) for idx, entry in enumerate(val_cases)]

    print(f"Evaluating {len(task_args)} validation cases across 16 CPU workers...")
    with Pool(16) as pool:
        all_results_nested = pool.map(evaluate_single_scan, task_args)
        
    all_results = [r for sublist in all_results_nested for r in sublist]
    
    dices_all = [r[2] for r in all_results]
    hits_all = [r[2] >= 0.1 for r in all_results]
    empty_all = [r[3] for r in all_results]
    
    dices_50 = [r[2] for r in all_results if r[0] < 50]
    hits_50 = [r[2] >= 0.1 for r in all_results if r[0] < 50]
    empty_50 = [r[3] for r in all_results if r[0] < 50]

    dices_150 = [r[2] for r in all_results if r[0] >= 50]
    hits_150 = [r[2] >= 0.1 for r in all_results if r[0] >= 50]
    empty_150 = [r[3] for r in all_results if r[0] >= 50]

    print("\n=========================================================================")
    print("=== FINAL 200-SCAN NORMALIZED ZERO-SHOT BASELINE EVALUATION ===")
    print("=========================================================================\n")
    
    print(f"{'Partition':<35} | {'Findings':<10} | {'Average Dice':<15} | {'Hit Rate (>=0.1)':<18} | {'Empty Preds':<12}")
    print("-" * 98)
    print(f"{'First 50 Scans (Paper Val Split)':<35} | {len(dices_50):<10} | {np.mean(dices_50):<15.4f} | {np.mean(hits_50)*100:<17.2f}% | {np.mean(empty_50)*100:<11.2f}%")
    print(f"{'Next 150 Scans (New MICCAI Split)':<35} | {len(dices_150):<10} | {np.mean(dices_150):<15.4f} | {np.mean(hits_150)*100:<17.2f}% | {np.mean(empty_150)*100:<11.2f}%")
    print(f"{'Combined 200 Scans (Normalized)':<35} | {len(dices_all):<10} | {np.mean(dices_all):<15.4f} | {np.mean(hits_all)*100:<17.2f}% | {np.mean(empty_all)*100:<11.2f}%")

if __name__ == "__main__":
    main()
