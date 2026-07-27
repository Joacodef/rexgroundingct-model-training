import os
import json
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

# Bucketing logic for size
def get_size_bucket(voxel_count):
    if voxel_count < 1000:
        return "Small (<1k)"
    elif voxel_count <= 10000:
        return "Medium (1k-10k)"
    else:
        return "Large (>10k)"

# Basic heuristic categorizer for findings text
CATEGORIES = [
    "lung", "effusion", "nodule", "fracture", "mass", "lesion",
    "atelectasis", "consolidation", "pleural", "calcification", "hernia", "cyst", "lymph"
]

def get_category(text):
    text = str(text).lower()
    for cat in CATEGORIES:
        if cat in text:
            return cat
    return "other"

def compute_dice(pred_mask, gt_mask):
    pred_bool = pred_mask > 0
    gt_bool = gt_mask > 0
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = pred_bool.sum() + gt_bool.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2. * intersection / union

def main():
    load_dotenv(override=True)
    parser = argparse.ArgumentParser(description="Evaluate 4D predictions with Bucketing")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    args = parser.parse_args()

    gt_dir = os.environ["SEG_RAW_DIR"]
    pred_dir = os.environ["DATA_PRED_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    output_json = os.path.join(os.path.dirname(pred_dir), "eval_bucketed_results.json")

    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get(args.split, [])
    
    metrics = {
        "overall": {"dices": [], "hits": 0, "total": 0},
        "by_size": {
            "Small (<1k)": {"dices": [], "hits": 0, "total": 0},
            "Medium (1k-10k)": {"dices": [], "hits": 0, "total": 0},
            "Large (>10k)": {"dices": [], "hits": 0, "total": 0},
        },
        "by_category": {}
    }

    for entry in tqdm(entries, desc="Evaluating"):
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
            
        gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
        pred_img = nib.load(pred_path).get_fdata(dtype=np.float32)
        
        if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0)
            
        if gt_img.shape != pred_img.shape: continue
            
        findings = entry.get('findings', {})
        if isinstance(findings, dict):
            finding_texts = [findings[k].get('text', '') if isinstance(findings[k], dict) else str(findings[k]) for k in sorted(findings.keys(), key=int)]
        else:
            finding_texts = [f['text'] if isinstance(f, dict) else str(f) for f in findings]
            
        for f_idx in range(gt_img.shape[0]):
            gt_mask = gt_img[f_idx]
            pred_mask = pred_img[f_idx]
            
            dice = compute_dice(pred_mask, gt_mask)
            is_hit = 1 if dice >= 0.1 else 0
            
            # Bucketing
            voxel_count = np.sum(gt_mask > 0)
            size_bucket = get_size_bucket(voxel_count)
            text = finding_texts[f_idx] if f_idx < len(finding_texts) else ""
            cat_bucket = get_category(text)
            
            # Overall
            metrics["overall"]["dices"].append(dice)
            metrics["overall"]["hits"] += is_hit
            metrics["overall"]["total"] += 1
            
            # By Size
            metrics["by_size"][size_bucket]["dices"].append(dice)
            metrics["by_size"][size_bucket]["hits"] += is_hit
            metrics["by_size"][size_bucket]["total"] += 1
            
            # By Category
            if cat_bucket not in metrics["by_category"]:
                metrics["by_category"][cat_bucket] = {"dices": [], "hits": 0, "total": 0}
            metrics["by_category"][cat_bucket]["dices"].append(dice)
            metrics["by_category"][cat_bucket]["hits"] += is_hit
            metrics["by_category"][cat_bucket]["total"] += 1

    # Summarize Results
    final_report = {"split": args.split, "overall": {}, "by_size": {}, "by_category": {}}
    
    def summarize(d):
        total = d["total"]
        if total == 0: return {"dice": 0.0, "hit_rate": 0.0, "count": 0}
        return {"dice": np.mean(d["dices"]), "hit_rate": d["hits"]/total, "count": total}

    final_report["overall"] = summarize(metrics["overall"])
    for k, v in metrics["by_size"].items():
        final_report["by_size"][k] = summarize(v)
    for k, v in metrics["by_category"].items():
        final_report["by_category"][k] = summarize(v)

    with open(output_json, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(json.dumps(final_report, indent=4))

if __name__ == "__main__":
    main()
