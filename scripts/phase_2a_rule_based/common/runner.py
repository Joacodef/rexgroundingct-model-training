"""
===============================================================================
MODULE:         Phase 2A Baseline Inference & Evaluation Runner
LOCATION:       scripts/phase_2a_rule_based/common/runner.py
OBJECTIVE:      Reusable validation set inference loop and automated metric 
                evaluation runner for Phase 2A spatial prior experiments.
===============================================================================
"""

import sys
import gc
import json
import ctypes
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.common.orientation import load_nifti_ras, save_nifti


def run_prior_inference_and_eval(
    predictor,
    split: str,
    dataset_json_path: Path,
    img_raw_dir: Path,
    seg_raw_dir: Path,
    output_dir: Path,
    exp_log_dir: Path,
    pdf_cache_path: Path,
    do_eval: bool = True,
    start_idx: int = 0,
    end_idx: int = None,
) -> None:
    """
    Signature:
        run_prior_inference_and_eval(
            predictor, split: str, dataset_json_path: Path, img_raw_dir: Path,
            seg_raw_dir: Path, output_dir: Path, exp_log_dir: Path,
            pdf_cache_path: Path, do_eval: bool, start_idx: int, end_idx: int
        ) -> None

    Objective:
        Execute validation set inference using an initialized prior baseline predictor, stack predictions into 4D NIfTI masks,
        save results, and trigger automated challenge metric evaluation.

    Inputs:
        predictor: EmpiricalSpatialPDFBaseline instance.
        split (str): Dataset split ('train', 'val', 'test').
        dataset_json_path (Path): Path to dataset.json metadata.
        img_raw_dir (Path): Path to raw CT image directory.
        seg_raw_dir (Path): Path to raw CT segmentations directory.
        output_dir (Path): Path to output prediction directory.
        exp_log_dir (Path): Dedicated experiment log folder (e.g. logs/phase_2a_rule_based/exp_002...).
        pdf_cache_path (Path): Path to empirical spatial PDF .npz cache.
        do_eval (bool): Whether to run scripts/common/evaluate.py after inference. Default True.
        start_idx (int): Start index for dataset chunking. Default 0.
        end_idx (int, optional): End index for dataset chunking.

    Outputs:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_json_path, 'r') as f:
        metadata = json.load(f)

    entries = metadata.get(split, [])
    if not entries:
        print(f"[ERROR] No entries found for split '{split}' in {dataset_json_path}")
        sys.exit(1)

    end_idx = end_idx if end_idx is not None else len(entries)
    entries = entries[start_idx:end_idx]

    missing_scans = 0
    generated_count = 0

    print("=" * 80)
    print(f"Phase 2A Inference — Split: {split} [{start_idx}:{end_idx}]")
    print(f"Output Directory:    {output_dir}")
    print("=" * 80)

    # 1. Validation Inference Loop
    for entry in tqdm(entries, desc=f"Phase 2A Prior [{split}]"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue

        out_nii_path = output_dir / f"{scan_id}.nii.gz"
        if out_nii_path.exists():
            generated_count += 1
            continue

        raw_nifti_path = img_raw_dir / f"{scan_id}.nii.gz"
        if not raw_nifti_path.exists():
            tqdm.write(f"[WARNING] Raw image missing: {raw_nifti_path}")
            missing_scans += 1
            continue

        # Load raw CT image in RAS coordinate space
        raw_img_ras, raw_nii_ras, _ = load_nifti_ras(raw_nifti_path)
        target_shape_xyz = raw_img_ras.shape # (X, Y, Z)

        findings = entry.get("findings", {})
        categories_dict = entry.get("categories", {})

        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
            category_codes = [str(categories_dict.get(k, "")) for k in sorted_keys]
        else:
            prompts = [f.get("text", "") if isinstance(f, dict) else str(f) for f in findings]
            category_codes = [str(categories_dict.get(str(i), "")) for i in range(len(prompts))]

        if not prompts:
            continue

        # Generate 3D spatial prior mask per finding category
        finding_masks_xyz = []
        for i, cat_code in enumerate(category_codes):
            prompt_text = prompts[i] if i < len(prompts) else None
            mask_3d = predictor.generate_prediction_mask(
                cat_code=cat_code,
                target_shape_ras=target_shape_xyz,
                ct_img_ras=raw_img_ras,
                prompt_text=prompt_text,
            )
            finding_masks_xyz.append(mask_3d)


        # Stack into 4D tensor (F, X, Y, Z) matching ReXGroundingCT specification
        pred_4d_fxyz = np.stack(finding_masks_xyz, axis=0).astype(np.uint8)

        # Save NIfTI anchored to parent CT scan header via Centralized Spatial Engine
        save_nifti(pred_4d_fxyz, out_nii_path, parent_ct_path=raw_nifti_path)
        generated_count += 1

        del raw_img_ras, raw_nii_ras, finding_masks_xyz, pred_4d_fxyz
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    print(f"\n[SUCCESS] Generated predictions for {generated_count} scans in {output_dir}")
    if missing_scans > 0:
        print(f"[WARNING] Skipped {missing_scans} missing CT files.")

    # 2. Challenge Metric Evaluation Runner
    if do_eval and (generated_count > 0 or len(list(output_dir.glob("*.nii.gz"))) > 0):
        print("\n" + "=" * 80)
        eval_json_path = exp_log_dir / f"eval_results_{split}.json"
        log_path = exp_log_dir / "eval.md"
        eval_script = ROOT_DIR / "scripts" / "common" / "evaluate.py"

        cmd = [
            sys.executable, str(eval_script),
            "--gt_dir", str(seg_raw_dir),
            "--img_dir", str(img_raw_dir),
            "--pred_dir", str(output_dir),
            "--split", split,
            "--dataset_json", str(dataset_json_path),
            "--output_json", str(eval_json_path)
        ]

        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        print("\n--- Evaluation Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Evaluation Warnings/Errors ---")
            print(result.stderr)

        with open(log_path, 'w') as f_log:
            f_log.write(f"# Phase 2A — Baseline Evaluation Log ({exp_log_dir.name})\n\n")
            f_log.write(f"- **Target Split:** {split}\n")
            f_log.write(f"- **PDF Cache Location:** `{pdf_cache_path}`\n")
            f_log.write(f"- **Prediction Directory:** `{output_dir}`\n")
            f_log.write(f"- **Evaluated Files:** {len(list(output_dir.glob('*.nii.gz')))}\n\n")
            f_log.write("## Quantitative Metric Evaluation Output\n\n```text\n")
            f_log.write(result.stdout)
            f_log.write("\n```\n")

        print(f"\n[INFO] Saved evaluation report to: {log_path}")
