"""
===============================================================================
PHASE:         Phase 2A — Statistical / Rule-Based Prior Baseline
OBJECTIVE:     Construct a data-driven empirical 3D spatial probability density 
               baseline by accumulating and averaging ground-truth training 
               segmentation masks per pathology category, using them as 3D 
               spatial PDF predictors for validation CT scans.
INPUTS:        dataset.json, raw CT segmentations, raw CT images
OUTPUTS:       ../data/phase_2a/empirical_spatial_pdf_14cat.npz,
               ../data/predictions/phase_2a_rule_based/
USAGE:         python scripts/phase_2a_rule_based/exp_001_seg_masks_priors.py --split val --eval
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
import torch
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from scipy.ndimage import label
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

# Load environment variables
load_dotenv(override=True)

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from scripts.config import (
    DATA_DIR, DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    PREDICTIONS_DIR, LOGS_DIR, CATEGORY_MAP, REVERSE_CATEGORY_MAP
)
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from scripts.common.prompt_normalizer import clean_finding_prompt


def parse_args():
    """Parse command-line arguments for Phase 2A empirical 3D spatial PDF baseline generation and evaluation."""
    parser = argparse.ArgumentParser(
        description="Phase 2A Data-Driven Empirical Spatial Probability Density Baseline Predictor"
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
        "--max_train_scans", type=int, default=300,
        help="Maximum training scans to sample for 3D empirical PDF heatmap building (default: 300)"
    )
    parser.add_argument(
        "--force_rebuild_pdf", action="store_true", default=False,
        help="Force rebuild of 3D spatial probability density heatmaps from train set"
    )
    parser.add_argument(
        "--eval", action="store_true", default=True,
        help="Automatically run scripts/evaluate.py after inference (default: True)"
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


class EmpiricalSpatialPDFBaseline:
    """
    Data-driven 3D Empirical Spatial Probability Density Baseline:
    1. Accumulates training GT segmentations per category in canonical RAS space (192x192x192).
    2. Computes empirical probability density P_c(z, y, x) = 1/N_c * sum(Mask_i).
    3. Resamples P_c to target validation scan shape and thresholds to generate 3D/4D predictions.
    """
    CANONICAL_SHAPE = (192, 192, 192)

    def __init__(self, pdf_cache_path: Path, dataset_json_path: Path, seg_raw_dir: Path, max_train_scans: int = 300, force_rebuild: bool = False):
        """
        Signature:
            __init__(pdf_cache_path: Path, dataset_json_path: Path, seg_raw_dir: Path, max_train_scans: int, force_rebuild: bool) -> None

        Objective:
            Initialize 3D empirical PDF baseline engine, loading or generating cached 3D heatmaps.

        Inputs:
            pdf_cache_path (Path): Path to .npz file containing cached 3D probability density maps.
            dataset_json_path (Path): Path to dataset.json file.
            seg_raw_dir (Path): Path to raw GT segmentations directory.
            max_train_scans (int): Max training split scans to sample for PDF building. Default 300.
            force_rebuild (bool): Whether to force rebuilding PDF heatmaps from scratch. Default False.

        Outputs:
            None
        """
        self.pdf_cache_path = pdf_cache_path
        self.dataset_json_path = dataset_json_path
        self.seg_raw_dir = seg_raw_dir
        self.max_train_scans = max_train_scans
        self.reader = NibabelIOWithReorient()
        
        if force_rebuild or not pdf_cache_path.exists():
            print(f"[INFO] Building 3D Empirical Spatial PDF Heatmaps from Train Split (max {self.max_train_scans} scans)...")
            self.spatial_pdfs = self._build_pdf_cache()
        else:
            print(f"[INFO] Loading cached 3D Empirical Spatial PDF Heatmaps from {pdf_cache_path}...")
            self.spatial_pdfs = self._load_pdf_cache()

    def _build_pdf_cache(self) -> dict:
        """
        Signature:
            _build_pdf_cache() -> dict

        Objective:
            Accumulate and average GT 3D segmentation masks per category across the training split
            to generate 3D empirical spatial probability density heatmaps P_c(z, y, x).

        Inputs:
            None (Uses instance metadata paths and max_train_scans limit).

        Outputs:
            dict: Dictionary mapping 14 category codes ('1a'..'2h') to 3D numpy arrays of shape (192, 192, 192).
        """
        self.pdf_cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.dataset_json_path, 'r') as f:
            metadata = json.load(f)
            
        train_entries = metadata.get("train", [])
        if not train_entries:
            raise ValueError(f"No 'train' entries found in {self.dataset_json_path}")

        if self.max_train_scans and len(train_entries) > self.max_train_scans:
            train_entries = train_entries[:self.max_train_scans]

        # 3D Accumulators
        accumulators = {code: np.zeros(self.CANONICAL_SHAPE, dtype=np.float32) for code in CATEGORY_MAP.keys()}
        category_counts = {code: 0 for code in CATEGORY_MAP.keys()}

        print(f"[INFO] Processing {len(train_entries)} training scans for empirical PDF building...")
        ras_ornt = axcodes2ornt("RAS")

        for entry in tqdm(train_entries, desc="Building 3D PDF Heatmaps"):
            scan_id = entry.get("name", "").replace(".nii.gz", "")
            if not scan_id:
                continue
                
            seg_path = self.seg_raw_dir / f"{scan_id}.nii.gz"
            if not seg_path.exists():
                continue
                
            categories_dict = entry.get("categories", {})
            if not categories_dict:
                continue

            try:
                gt_nii = nib.load(str(seg_path))
                gt_ornt = io_orientation(gt_nii.affine)
                to_ras = ornt_transform(gt_ornt, ras_ornt)
                
                # Reorient to RAS space
                gt_nii_ras = gt_nii.as_reoriented(to_ras)
                gt_data = np.asanyarray(gt_nii_ras.dataobj).astype(np.float32)
            except Exception as e:
                tqdm.write(f"[WARNING] Failed to load GT mask {seg_path}: {e}")
                continue

            if gt_data.ndim == 3:
                gt_data = np.expand_dims(gt_data, axis=-1)

            if gt_data.ndim == 4:
                if gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
                    gt_data = np.transpose(gt_data, (3, 2, 1, 0)) # (F, Z, Y, X)

            num_findings = gt_data.shape[0]

            for f_idx in range(num_findings):
                cat_code = str(categories_dict.get(str(f_idx), ""))
                if not cat_code or cat_code not in CATEGORY_MAP:
                    continue

                binary_mask = (gt_data[f_idx] > 0).astype(np.float32) # shape: (Z, Y, X)
                if not binary_mask.any():
                    continue

                # PyTorch 3D Interpolation to (192, 192, 192)
                mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0) # (1, 1, Z, Y, X)
                if mask_tensor.shape[2:] != self.CANONICAL_SHAPE:
                    mask_canonical = F.interpolate(mask_tensor, size=self.CANONICAL_SHAPE, mode='nearest').squeeze(0).squeeze(0).numpy()
                else:
                    mask_canonical = binary_mask

                accumulators[cat_code] += mask_canonical
                category_counts[cat_code] += 1

            del gt_nii, gt_data
            gc.collect()

        # Normalize accumulators
        spatial_pdfs = {}
        save_dict = {}
        for code in CATEGORY_MAP.keys():
            count = category_counts[code]
            if count > 0:
                pdf_np = accumulators[code] / float(count)
                print(f"[INFO] Category '{code}' ({CATEGORY_MAP[code]}): Accumulated {count} training masks. Max PDF prob: {pdf_np.max():.4f}")
            else:
                print(f"[WARNING] Category '{code}' ({CATEGORY_MAP[code]}): 0 training masks found. Using uniform prior.")
                pdf_np = np.full(self.CANONICAL_SHAPE, 0.01, dtype=np.float32)
                
            spatial_pdfs[code] = pdf_np
            save_dict[code] = pdf_np

        # Save to npz cache
        np.savez_compressed(self.pdf_cache_path, **save_dict)
        print(f"[SUCCESS] Saved 14-category 3D Spatial PDF Heatmaps to {self.pdf_cache_path}")
        return spatial_pdfs

    def _load_pdf_cache(self) -> dict:
        """
        Signature:
            _load_pdf_cache() -> dict

        Objective:
            Load precomputed 3D spatial PDF heatmaps from .npz cache file.

        Inputs:
            None (Reads from self.pdf_cache_path).

        Outputs:
            dict: Dictionary mapping category codes to 3D float32 numpy arrays of shape (192, 192, 192).
        """
        with np.load(self.pdf_cache_path) as data:
            spatial_pdfs = {code: data[code] for code in CATEGORY_MAP.keys() if code in data}
        return spatial_pdfs

    def match_category_code(self, prompt_text: str) -> str:
        """
        Signature:
            match_category_code(prompt_text: str) -> str

        Objective:
            Match raw radiology report finding text to its corresponding 14-category code string ('1a'..'2h').

        Inputs:
            prompt_text (str): Free-text radiology report finding description.

        Outputs:
            str: 14-category code string ('1a'..'2h'). Defaults to '2h' if unmatched.
        """
        cleaned_text = clean_finding_prompt(prompt_text).lower()
        for code, cat_name in CATEGORY_MAP.items():
            if cat_name.lower() in cleaned_text:
                return code
                
        if "nodule" in cleaned_text or "mass" in cleaned_text:
            return "2d"
        elif "opacity" in cleaned_text or "consolidation" in cleaned_text:
            return "2c"
        elif "effusion" in cleaned_text:
            return "2e"
        elif "thickening" in cleaned_text:
            return "1d"
        elif "emphysema" in cleaned_text:
            return "1c"
        elif "bronch" in cleaned_text:
            return "1a"
            
        return "2h"

    def generate_prediction_mask(self, target_shape_ras: tuple, prompt_text: str) -> np.ndarray:
        """
        Signature:
            generate_prediction_mask(target_shape_ras: tuple, prompt_text: str) -> np.ndarray

        Objective:
            Resample category 3D empirical PDF heatmap P_c to target scan's RAS shape, apply
            category-specific percentile thresholding, and prune tiny 3D noise blobs (<10 voxels).

        Inputs:
            target_shape_ras (tuple): Target 3D volume shape (Z, Y, X) in canonical RAS space.
            prompt_text (str): Free-text radiology report finding prompt.

        Outputs:
            np.ndarray: 3D uint8 binary prediction mask array with shape matching target_shape_ras.
        """
        code = self.match_category_code(prompt_text)
        pdf_np = self.spatial_pdfs.get(code, np.full(self.CANONICAL_SHAPE, 0.01, dtype=np.float32))

        # PyTorch 3D Interpolation to target RAS shape
        if target_shape_ras != self.CANONICAL_SHAPE:
            pdf_tensor = torch.from_numpy(pdf_np).unsqueeze(0).unsqueeze(0)
            pdf_target = F.interpolate(pdf_tensor, size=target_shape_ras, mode='trilinear', align_corners=False).squeeze(0).squeeze(0).numpy()
        else:
            pdf_target = pdf_np.copy()

        # Category-Specific Calibrated Binarization Cutoffs (p_c)
        max_p = pdf_target.max()
        if max_p <= 0:
            return np.zeros(target_shape_ras, dtype=np.uint8)

        # Calibrated category threshold factors (p_c = factor * max_p).
        # DERIVATION: Derived from Phase 1 empirical cumulative density profiling and morphological 
        # sphericity profiles (PHASE_1_DATA_ANALYSIS_SUMMARY.md). High-concentration basal/effusion 
        # findings (max P_c in [0.15, 0.81]) use lower relative factors (0.30-0.35) to capture 
        # extended fluid boundaries, whereas highly focal/compact spherical findings (max P_c in 
        # [0.008, 0.05], e.g. Nodules 2d S=0.94) use higher factors (0.50) to isolate peak density cores.
        CATEGORY_THRESHOLD_FACTORS = {
            "1a": 0.35,  # Bronchial wall thickening (hilar/peribronchial)
            "1b": 0.40,  # Bronchiectasis (airway tree)
            "1c": 0.40,  # Emphysema (apical dominant)
            "1d": 0.40,  # Septal thickening (interstitial)
            "1e": 0.50,  # Micronodules (multi-focal clusters)
            "1f": 0.40,  # Other non-focal
            "2a": 0.50,  # Linear opacities (focal linear)
            "2b": 0.35,  # Atelectasis / consolidation (basal dependent)
            "2c": 0.35,  # Ground-glass opacity (patchy parenchymal)
            "2d": 0.50,  # Pulmonary nodules / masses (focal spherical, S=0.94)
            "2e": 0.30,  # Pleural effusion / thickening (basal dependent fluid)
            "2f": 0.40,  # Honeycombing (subpleural basal)
            "2g": 0.35,  # Pneumothorax (pleural boundary)
            "2h": 0.40,  # Other focal
        }
        factor = CATEGORY_THRESHOLD_FACTORS.get(code, 0.40)
        p_threshold = factor * max_p

        binary_mask = (pdf_target >= p_threshold).astype(np.uint8)
        
        # Component Size Cleanup
        if binary_mask.any():
            labeled, num_features = label(binary_mask)
            if num_features > 0:
                sizes = np.bincount(labeled.ravel())
                too_small = sizes < 10
                binary_mask[too_small[labeled]] = 0

        return binary_mask.astype(np.uint8)


def main():
    """Main CLI entry point for executing Phase 2A empirical 3D spatial PDF baseline inference and 4D Back-Reorientation."""
    args = parse_args()

    # Resolve Paths
    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat.npz"
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_rule_based"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase 2A — Empirical Spatial Probability Density Baseline Predictor")
    print(f"Target Split:        {args.split}")
    print(f"PDF Cache Path:      {pdf_cache_path}")
    print(f"Dataset JSON:        {args.dataset_json}")
    print(f"Output Directory:    {output_dir}")
    print("=" * 80)

    # Initialize Predictor Engine & Reorientation Reader
    predictor = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        max_train_scans=args.max_train_scans,
        force_rebuild=args.force_rebuild_pdf
    )
    reader = NibabelIOWithReorient()

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

    # Validation Inference Loop
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

        # 1. Read CT volume header & reorient to RAS to get target shape
        img_ras, img_props = reader.read_images([str(raw_nifti_path)]) # shape: (1, Z, Y, X)
        ras_volume_shape = img_ras[0].shape # (Z, Y, X)

        # 2. Extract Finding Prompts
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f.get("text", "") if isinstance(f, dict) else str(f) for f in findings]

        if not prompts:
            continue

        # 3. Generate 3D mask per finding prompt using Empirical PDF Heatmap P_c
        finding_masks_ras = []
        for prompt in prompts:
            mask_3d = predictor.generate_prediction_mask(ras_volume_shape, prompt) # shape: (Z, Y, X)
            finding_masks_ras.append(mask_3d)

        # Stack to 4D array in RAS space (Z, Y, X, F) -> transpose to (X, Y, Z, F)
        ras_4d_stack = np.stack(finding_masks_ras, axis=-1) # (Z, Y, X, F)
        pred_xyzf = np.transpose(ras_4d_stack, (2, 1, 0, 3)) # (X, Y, Z, F)

        # 4. 4D Back-Reorientation Contract Execution
        reoriented_affine = img_props['nibabel_stuff']['reoriented_affine']
        original_affine = img_props['nibabel_stuff']['original_affine']

        pred_nib_ras = nib.Nifti1Image(pred_xyzf, reoriented_affine)

        img_ornt = io_orientation(original_affine)
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)

        pred_nib_back = pred_nib_ras.as_reoriented(from_canonical)

        # Convert back-reoriented data to uint8 and transpose to (F, X, Y, Z)
        pred_back_data = np.asanyarray(pred_nib_back.dataobj).astype(np.uint8) # (X, Y, Z, F)
        pred_back_fxyz = np.transpose(pred_back_data, (3, 0, 1, 2)) # (F, X, Y, Z)

        # Save NIfTI file with original CT affine
        out_nii = nib.Nifti1Image(pred_back_fxyz, original_affine)
        nib.save(out_nii, str(out_nii_path))
        generated_count += 1

        # Explicit memory cleanup
        del img_ras, finding_masks_ras, ras_4d_stack, pred_xyzf, pred_nib_ras, pred_nib_back, pred_back_data, pred_back_fxyz, out_nii
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    print(f"\n[SUCCESS] Generated predictions for {generated_count} scans in {output_dir}")
    if missing_scans > 0:
        print(f"[WARNING] Skipped {missing_scans} missing CT files.")

    # Automate Metric Evaluation via scripts/evaluate.py
    if args.eval and (generated_count > 0 or len(list(output_dir.glob("*.nii.gz"))) > 0):
        print("\n" + "=" * 80)
        exp_log_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_001_seg_masks_priors"
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        eval_json_path = exp_log_dir / f"eval_results_{args.split}.json"
        log_path = exp_log_dir / "eval.md"

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
