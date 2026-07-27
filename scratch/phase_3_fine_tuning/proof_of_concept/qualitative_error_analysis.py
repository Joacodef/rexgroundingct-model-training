import os
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv

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
    
    img_dir = os.environ["IMG_RAW_DIR"]
    gt_dir = os.environ["SEG_RAW_DIR"]
    pred_dir = os.environ["DATA_PRED_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    output_dir = os.path.join(os.path.dirname(pred_dir), "qualitative_analysis")
    os.makedirs(output_dir, exist_ok=True)

    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get("val", [])[:30]
    
    results = []

    print("Evaluating scans individually to find worst cases...")
    for entry in tqdm(entries, desc="Calculating Dice"):
        scan_id = entry["name"].replace(".nii.gz", "")
        gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{scan_id}.nii.gz")
        img_path = os.path.join(img_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue
            
        gt_img = nib.load(gt_path).get_fdata(dtype=np.float32)
        pred_img = nib.load(pred_path).get_fdata(dtype=np.float32)
        
        if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0)
            
        if pred_img.ndim == 3: pred_img = np.expand_dims(pred_img, axis=-1)
        if pred_img.ndim == 4 and pred_img.shape[-1] < np.min(pred_img.shape[:-1]):
            pred_img = np.moveaxis(pred_img, -1, 0)

        if gt_img.shape != pred_img.shape:
            continue
            
        findings = entry.get('findings', {})
        if isinstance(findings, dict):
            finding_texts = [findings[k].get('text', '') if isinstance(findings[k], dict) else str(findings[k]) for k in sorted(findings.keys(), key=int)]
        else:
            finding_texts = [f['text'] if isinstance(f, dict) else str(f) for f in findings]
            
        for f_idx in range(gt_img.shape[0]):
            gt_mask = gt_img[f_idx]
            pred_mask = pred_img[f_idx]
            gt_voxels = np.sum(gt_mask > 0)
            pred_voxels = np.sum(pred_mask > 0)
            
            if gt_voxels == 0:
                continue
                
            dice = compute_dice(pred_mask, gt_mask)
            text = finding_texts[f_idx] if f_idx < len(finding_texts) else ""
            
            results.append({
                "scan_id": scan_id,
                "f_idx": f_idx,
                "finding_text": text,
                "dice": dice,
                "gt_voxels": int(gt_voxels),
                "pred_voxels": int(pred_voxels),
                "gt_path": gt_path,
                "pred_path": pred_path,
                "img_path": img_path
            })

    # Sort results by GT size (descending) to look at medium/large findings, then by Dice (ascending)
    # We filter for GT size > 1000 so we examine clear, visible anatomical structures
    large_findings = [r for r in results if r["gt_voxels"] > 1000]
    large_findings.sort(key=lambda x: x["dice"])
    
    worst_5 = large_findings[:5]
    best_2 = sorted(results, key=lambda x: x["dice"], reverse=True)[:2]
    
    selected_cases = worst_5 + best_2
    
    print("\n--- QUALITATIVE SUMMARY OF SELECTED CASES ---")
    summary_txt = []
    for item in selected_cases:
        status = "ZERO_PREDICTION (Suppression Bias)" if item["pred_voxels"] == 0 else f"NON_ZERO_PREDICTION ({item['pred_voxels']} voxels)"
        msg = f"Scan: {item['scan_id']} | Finding #{item['f_idx']}: '{item['finding_text'][:30]}...' | GT Voxels: {item['gt_voxels']} | Pred Voxels: {item['pred_voxels']} | Dice: {item['dice']:.4f} | Status: {status}"
        print(msg)
        summary_txt.append(msg)
        
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_txt))

    print("\nGenerating 2D overlay heatmaps for visual inspection...")
    for idx, case in enumerate(selected_cases):
        scan_id = case["scan_id"]
        f_idx = case["f_idx"]
        
        gt_img = nib.load(case["gt_path"]).get_fdata(dtype=np.float32)
        pred_img = nib.load(case["pred_path"]).get_fdata(dtype=np.float32)
        
        if gt_img.ndim == 3: gt_img = np.expand_dims(gt_img, axis=-1)
        if gt_img.ndim == 4 and gt_img.shape[-1] < np.min(gt_img.shape[:-1]):
            gt_img = np.moveaxis(gt_img, -1, 0)
            
        if pred_img.ndim == 3: pred_img = np.expand_dims(pred_img, axis=-1)
        if pred_img.ndim == 4 and pred_img.shape[-1] < np.min(pred_img.shape[:-1]):
            pred_img = np.moveaxis(pred_img, -1, 0)
            
        gt_mask = gt_img[f_idx]
        pred_mask = pred_img[f_idx]
        
        # Load raw CT volume
        if os.path.exists(case["img_path"]):
            ct_img = nib.load(case["img_path"]).get_fdata(dtype=np.float32)
        else:
            ct_img = np.zeros_like(gt_mask)

        # Find axial slice with max GT mask area
        gt_slice_sums = np.sum(gt_mask > 0, axis=(0, 1))
        best_slice_idx = np.argmax(gt_slice_sums)
        
        ct_slice = ct_img[:, :, best_slice_idx] if ct_img.ndim == 3 else np.zeros_like(gt_mask[:, :, 0])
        gt_slice = gt_mask[:, :, best_slice_idx]
        pred_slice = pred_mask[:, :, best_slice_idx] if pred_mask.ndim == 3 else np.zeros_like(gt_slice)

        # Normalize CT slice for viewing
        ct_slice = np.clip(ct_slice, -1000, 1000)
        ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. CT Raw
        axes[0].imshow(np.rot90(ct_slice), cmap='gray')
        axes[0].set_title(f"CT Slice z={best_slice_idx}")
        axes[0].axis('off')
        
        # 2. GT Mask
        axes[1].imshow(np.rot90(ct_slice), cmap='gray')
        axes[1].imshow(np.rot90(gt_slice), cmap='Greens', alpha=0.5)
        axes[1].set_title(f"GT (Voxels: {case['gt_voxels']})")
        axes[1].axis('off')
        
        # 3. Pred vs GT Overlay
        axes[2].imshow(np.rot90(ct_slice), cmap='gray')
        axes[2].imshow(np.rot90(gt_slice), cmap='Greens', alpha=0.3)
        if np.sum(pred_slice) > 0:
            axes[2].imshow(np.rot90(pred_slice), cmap='Reds', alpha=0.5)
            pred_title = f"Pred (Red) vs GT (Green) | Dice: {case['dice']:.4f}"
        else:
            pred_title = f"EMPTY PREDICTION | Dice: {case['dice']:.4f}"
        axes[2].set_title(pred_title)
        axes[2].axis('off')

        plt.suptitle(f"[{'WORST' if idx < 5 else 'BEST'}] {scan_id} - '{case['finding_text'][:40]}'")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"case_{idx+1}_{scan_id}_f{f_idx}.png"), dpi=200)
        plt.close()

    print("Qualitative analysis heatmaps generated successfully in data/qualitative_analysis/")

if __name__ == "__main__":
    main()
