"""
===============================================================================
SCRIPT:         VoxTell Continuous Logit & Threshold Diagnostic Pipeline
PHASE:          Phase 2B — VoxTell Zero-Shot Baseline & Preprocessing Audit
LOCATION:       scripts/phase_2b_voxtell_baseline/exp_002_voxtell_logit_diagnostics.py
OBJECTIVE:      Profiles continuous sigmoid probability heatmaps (min, max, p95, p99, p99.9)
                and sweeps binarization thresholds (p_c in [0.01, 0.50]) to diagnose
                over-pruning failure modes in VoxTell zero-shot predictions.
USAGE:          python scripts/phase_2b_voxtell_baseline/exp_002_voxtell_logit_diagnostics.py --num_cases 5
===============================================================================
"""

import os
import gc
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables (shell environment takes precedence over .env)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv(override=False)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    DATASET_JSON,
    LOGS_DIR,
    MODEL_DIR,
    CATEGORY_MAP
)
from voxtell.inference.predictor import VoxTellPredictor
from scripts.common.orientation import load_nifti_ras, save_nifti
from scripts.common.evaluate import compute_dice


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for VoxTell logit diagnostics and threshold sweeps.
    """
    parser = argparse.ArgumentParser(description="VoxTell Continuous Logit & Threshold Diagnostic Tool")
    parser.add_argument("--num_cases", type=int, default=5, help="Number of validation cases to profile (default: 5)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json")
    parser.add_argument("--img_raw_dir", type=str, default=str(RAW_IMAGES_DIR), help="Directory containing raw images")
    parser.add_argument("--seg_raw_dir", type=str, default=str(RAW_MASKS_DIR), help="Directory containing raw GT masks")
    return parser.parse_args()


def main():
    """Main CLI entry point for continuous logit diagnostics and threshold sweeps."""
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Initializing VoxTell Predictor on device: {device}")

    predictor = VoxTellPredictor(model_dir=str(MODEL_DIR), device=device)
    
    # Offload sliding window result accumulators to CPU memory to prevent CUDA OOM
    predictor.perform_everything_on_device = False
    
    # Memory optimization: Keep text backbone on CPU to prevent CUDA OOM
    predictor.text_backbone = predictor.text_backbone.to("cpu")

    def embed_text_prompts_cpu_safe(text_prompts):
        """
        Signature:
            embed_text_prompts_cpu_safe(text_prompts: List[str] | str) -> torch.Tensor

        Objective:
            Computes text embeddings using CPU-offloaded Qwen2-0.5B text backbone to prevent
            CUDA VRAM OOM while returning final prompt tensor embeddings on GPU device.

        Args:
            text_prompts (List[str] | str): Input free-text prompt strings.

        Returns:
            torch.Tensor: Normalized prompt embeddings tensor of shape (1, N_prompts, D_embed) on predictor device.
        """
        from voxtell.utils.text_embedding import wrap_with_instruction, last_token_pool
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)
        wrapped = wrap_with_instruction(text_prompts)
        tokens = predictor.tokenizer(wrapped, padding=True, truncation=True, max_length=predictor.max_text_length, return_tensors="pt")
        with torch.no_grad():
            text_embed = predictor.text_backbone(**tokens)
            embeddings = last_token_pool(text_embed.last_hidden_state, tokens['attention_mask'])
            embeddings = embeddings.view(1, n_prompts, -1)
        return embeddings.to(predictor.device)

    predictor.embed_text_prompts = embed_text_prompts_cpu_safe

    with open(args.dataset_json, 'r') as f:
        metadata = json.load(f)

    entries = metadata.get(args.split, [])[:args.num_cases]
    print(f"Loaded {len(entries)} target validation cases for diagnostic profiling.")

    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    threshold_results = {t: [] for t in thresholds}
    category_prob_stats = {}

    for entry in tqdm(entries, desc="Diagnostic Logit Profiling"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        img_path = Path(args.img_raw_dir) / f"{scan_id}.nii.gz"
        gt_path = Path(args.seg_raw_dir) / f"{scan_id}.nii.gz"

        if not img_path.exists() or not gt_path.exists():
            continue

        # Step 1: Load CT image and GT mask in canonical NIfTI RAS physical coordinate space
        # load_nifti_ras() standardizes DICOM/NIfTI headers so axis 0=Right, 1=Anterior, 2=Superior.
        # Image shape: (X, Y, Z), GT shape: (F, X, Y, Z)
        img_ras, ras_nii, _ = load_nifti_ras(img_path)
        gt_ras, _, _ = load_nifti_ras(gt_path, ref_affine=ras_nii.affine)
        if gt_ras.ndim == 3:
            gt_ras = np.expand_dims(gt_ras, axis=0)

        # Step 2: Transpose image array from NIfTI RAS indexing (X, Y, Z) to nnUNet/VoxTell memory layout (Z, Y, X)
        # CRITICAL TECHNICAL DIRECTIVE: VoxTell's 3D Swin UNet model was pre-trained on nnUNet v2 pipelines.
        # nnUNet's NibabelIOWithReorient applies .transpose((2, 1, 0)) to place the axial depth/slice axis (Z)
        # at index 0 (depth-first C-contiguous ordering). Without this transposition, 3D convolutions receive
        # rotated sagittal cross-sections, destroying spatial feature matching and collapsing logits to near-zero.
        img_nnunet = img_ras.transpose((2, 1, 0))  # Shape: (Z, Y, X)

        findings = entry.get('findings', {})
        categories = entry.get('categories', {})

        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k]['text'] if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
            cat_codes = [str(categories.get(k, "unknown")) for k in sorted_keys]
        else:
            prompts = [f['text'] if isinstance(f, dict) else f for f in findings]
            cat_codes = [str(categories.get(str(i), "unknown")) for i in range(len(prompts))]

        # Step 3: Preprocess and run sliding window inference on nnUNet-ordered image (Z, Y, X)
        data_tensor, bbox, orig_shape = predictor.preprocess(img_nnunet)
        embeddings = predictor.embed_text_prompts(prompts)
        
        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data_tensor, embeddings).cpu()
            probs_nnunet_crop = torch.sigmoid(logits.float()).numpy()  # Output shape: (F, Z_crop, Y_crop, X_crop)

        # Step 4: Revert cropping by inserting cropped probabilities back into original 3D volume shape (F, Z_orig, Y_orig, X_orig)
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        probs_nnunet_full = np.zeros([probs_nnunet_crop.shape[0], *orig_shape], dtype=np.float32)
        probs_nnunet_full = insert_crop_into_image(probs_nnunet_full, probs_nnunet_crop, bbox)

        # Step 5: Untranspose predicted probability maps from (F, Z, Y, X) back to canonical NIfTI RAS space (F, X, Y, Z)
        # This restores exact 3D spatial alignment with ground-truth RAS segmentation mask gt_ras (F, X, Y, Z).
        probs = probs_nnunet_full.transpose((0, 3, 2, 1))  # Shape: (F, X_orig, Y_orig, Z_orig)

        num_findings = probs.shape[0]
        for f_idx in range(num_findings):
            p = probs[f_idx]
            gt = gt_ras[f_idx] > 0
            cat_code = cat_codes[f_idx]

            if cat_code not in category_prob_stats:
                category_prob_stats[cat_code] = {"max": [], "p99.9": [], "p99": [], "p95": []}

            category_prob_stats[cat_code]["max"].append(float(p.max()))
            category_prob_stats[cat_code]["p99.9"].append(float(np.percentile(p, 99.9)))
            category_prob_stats[cat_code]["p99"].append(float(np.percentile(p, 99.0)))
            category_prob_stats[cat_code]["p95"].append(float(np.percentile(p, 95.0)))

            for t in thresholds:
                pred_binary = (p > t).astype(np.uint8)
                dice = compute_dice(pred_binary, gt)
                threshold_results[t].append(dice)

        del img_ras, gt_ras, probs, data_tensor, embeddings
        gc.collect()

    print("\n" + "="*70)
    print("      VOXTELL LOGIT DIAGNOSTICS & PROBABILITY DISTRIBUTION PROFILE")
    print("="*70)
    print(f"{'Cat':<5} {'Category Name':<32} {'Count':<6} {'Max Prob':<10} {'p99.9':<10} {'p99.0':<10}")
    print("-" * 70)
    for cat_code, stats in sorted(category_prob_stats.items()):
        c_name = CATEGORY_MAP.get(cat_code, "Unknown")
        c_cnt = len(stats["max"])
        avg_max = np.mean(stats["max"]) if c_cnt > 0 else 0
        avg_p999 = np.mean(stats["p99.9"]) if c_cnt > 0 else 0
        avg_p990 = np.mean(stats["p99"]) if c_cnt > 0 else 0
        print(f"{cat_code:<5} {c_name:<32} {c_cnt:<6} {avg_max:.5f}    {avg_p999:.5f}    {avg_p990:.5f}")

    print("\n" + "="*70)
    print("      THRESHOLD SWEEP BENCHMARK (Average Dice across all findings)")
    print("="*70)
    for t in thresholds:
        avg_d = np.mean(threshold_results[t]) if threshold_results[t] else 0.0
        print(f"Threshold p_c = {t:<5.2f} -> Average Dice: {avg_d:.4f}")
    print("="*70)

    out_log_dir = LOGS_DIR / "phase_2b_voxtell_baseline" / "exp_002_voxtell_logit_diagnostics"
    out_log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_log_dir / "eval_results_diagnostics.json"

    with open(summary_path, 'w') as f:
        json.dump({
            "num_cases": len(entries),
            "threshold_sweep": {str(t): float(np.mean(threshold_results[t])) for t in thresholds},
            "category_prob_stats": {
                c: {k: float(np.mean(v)) for k, v in s.items()} for c, s in category_prob_stats.items()
            }
        }, f, indent=2)

    print(f"\n✅ Diagnostic summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
