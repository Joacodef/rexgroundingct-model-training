"""
===============================================================================
PHASE:         Phase 2A — Statistical / Rule-Based Prior Baseline (v2.0)
OBJECTIVE:     Construct an enhanced non-neural statistical baseline incorporating:
               1. Stat-H1: 3D Anatomical Lung Field Cavity Segmentation.
               2. Stat-H2: 3D Gaussian Spatial Density Fields & Adaptive Volume Scaling.
               3. Stat-H3: Category-Specific Pathology Constraints & Text Locators.
INPUTS:        ../data/phase_1/phase_1_priors_bundle.json, dataset.json, raw CT scans
OUTPUTS:       ../data/predictions/phase_2a_rule_based/
USAGE:         python scripts/baselines/phase_2a_rule_based_prior_baseline.py --split val --eval
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
from scipy.ndimage import label, binary_closing, binary_dilation, binary_fill_holes
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
from scripts.voxtell.prompt_normalizer import clean_finding_prompt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2A Non-Neural Statistical / Rule-Based Prior Baseline Predictor (v2.0)"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--priors_json", type=str, default=None,
        help="Path to phase_1_priors_bundle.json"
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
        "--output_dir", type=str, default=None,
        help="Output directory for predictions"
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


class StatisticalPriorPredictorV2:
    """
    Enhanced non-neural statistical baseline generator (v2.0):
    1. Lung Field Masking: Segments lung cavity (-1024 to -400 HU) to prevent false positives outside lungs.
    2. 3D Gaussian Spatial Fields: Continuous spatial density prior centered at category centroids.
    3. Adaptive Volume Scaling: Constrains predicted volume to physical lesion volume priors.
    4. Text Spatial Parsing: Anatomical locator bounds (apical, basal, left, right).
    """
    def __init__(self, priors_bundle_path: Path):
        if not priors_bundle_path.exists():
            raise FileNotFoundError(f"Priors bundle not found at: {priors_bundle_path}")
            
        with open(priors_bundle_path, 'r') as f:
            self.priors_bundle = json.load(f)
            
        self.category_priors = self.priors_bundle.get("category_profiles", {})

    def match_category_code(self, prompt_text: str) -> str:
        """Find matching 14-category code for a finding text prompt."""
        cleaned_text = clean_finding_prompt(prompt_text).lower()
        for code, cat_name in CATEGORY_MAP.items():
            if cat_name.lower() in cleaned_text:
                return code
                
        # Keyword fallbacks
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

    def extract_lung_cavity_mask(self, ras_ct_volume: np.ndarray) -> np.ndarray:
        """
        Stat-H1: Segment 3D lung parenchyma cavity.
        - Lung air/tissue resides between -1024 and -400 HU.
        - Excludes outside atmosphere air using patient bounding box heuristics.
        """
        Z, Y, X = ras_ct_volume.shape
        
        # Threshold lung air intensity
        raw_lung_air = (ras_ct_volume >= -1024) & (ras_ct_volume <= -400)
        
        # Exclude border slices where outside air dominates
        lung_mask = np.zeros_like(raw_lung_air, dtype=bool)
        
        # Internal bounding slice range (exclude outer 5% border slices)
        z_min, z_max = int(Z * 0.05), int(Z * 0.95)
        y_min, y_max = int(Y * 0.08), int(Y * 0.92)
        x_min, x_max = int(X * 0.08), int(X * 0.92)
        
        lung_mask[z_min:z_max, y_min:y_max, x_min:x_max] = raw_lung_air[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Clean connected components to isolate main lung cavities
        if lung_mask.any():
            labeled, num_features = label(lung_mask)
            if num_features > 0:
                component_sizes = np.bincount(labeled.ravel())
                component_sizes[0] = 0  # Ignore background
                
                # Keep top 4 largest air components inside interior volume (left/right lungs & branches)
                largest_indices = np.argsort(component_sizes)[-4:]
                lung_mask = np.isin(labeled, largest_indices[component_sizes[largest_indices] > 500])
                
                # Dilate slightly to include subpleural boundaries & vessels
                lung_mask = binary_dilation(lung_mask, iterations=2)
                
        return lung_mask

    def parse_spatial_locators(self, prompt_text: str, shape_ras: tuple) -> np.ndarray:
        """
        Parse directional locators from text to construct a 3D binary spatial boundary.
        shape_ras: (Z, Y, X) shape in RAS space.
        """
        text = prompt_text.lower()
        Z, Y, X = shape_ras
        spatial_mask = np.ones(shape_ras, dtype=bool)
        
        # Inferior-Superior (Z axis: 0=Inferior, Z=Superior)
        if any(w in text for w in ["apical", "upper lobe", "superior", "top", "apex"]):
            spatial_mask[: int(Z * 0.45), :, :] = False
        elif any(w in text for w in ["basal", "lower lobe", "inferior", "base"]):
            spatial_mask[int(Z * 0.55) :, :, :] = False
        elif "middle" in text:
            spatial_mask[: int(Z * 0.3), :, :] = False
            spatial_mask[int(Z * 0.7) :, :, :] = False
            
        # Right-Left (X axis: 0=Right, X=Left in standard RAS)
        if "right" in text and "left" not in text:
            spatial_mask[:, :, int(X * 0.5) :] = False
        elif "left" in text and "right" not in text:
            spatial_mask[:, :, : int(X * 0.5)] = False
            
        # Anterior-Posterior (Y axis: 0=Posterior, Y=Anterior in RAS)
        if "anterior" in text and "posterior" not in text:
            spatial_mask[:, : int(Y * 0.4), :] = False
        elif "posterior" in text and "anterior" not in text:
            spatial_mask[:, int(Y * 0.6) :, :] = False
            
        return spatial_mask

    def generate_finding_mask(self, ras_ct_volume: np.ndarray, prompt_text: str, lung_mask: np.ndarray) -> np.ndarray:
        """
        Generate 3D binary prediction mask for a single finding prompt.
        ras_ct_volume: 3D float numpy array in RAS space (Z, Y, X).
        lung_mask: 3D binary numpy array representing anatomical lung cavity.
        """
        category_code = self.match_category_code(prompt_text)
        cat_prior = self.category_priors.get(category_code, {})
        
        Z, Y, X = ras_ct_volume.shape
        
        # 1. Spatial Centroid & 3D Gaussian Probability Field
        spatial_info = cat_prior.get("spatial_prior", {})
        centroid_ras = spatial_info.get("train_centroid_ras", [0.5, 0.5, 0.5])
        if None in centroid_ras:
            centroid_ras = [0.5, 0.5, 0.5]
            
        c_x, c_y, c_z = centroid_ras[0], centroid_ras[1], centroid_ras[2]
        
        # Category physical extents (scaled relative to nominal chest dimensions)
        extents = cat_prior.get("component_topology", {}).get("physical_extent_mm", {})
        sig_x = max(extents.get("extent_x_mm", 30.0) / 250.0, 0.12)
        sig_y = max(extents.get("extent_y_mm", 30.0) / 250.0, 0.12)
        sig_z = max(extents.get("extent_z_mm", 20.0) / 250.0, 0.10)
        
        zz, yy, xx = np.ogrid[:Z, :Y, :X]
        norm_z = zz / float(Z)
        norm_y = yy / float(Y)
        norm_x = xx / float(X)
        
        # 3D Gaussian spatial probability map
        gaussian_map = np.exp(-0.5 * (
            ((norm_x - c_x) / sig_x) ** 2 +
            ((norm_y - c_y) / sig_y) ** 2 +
            ((norm_z - c_z) / sig_z) ** 2
        ))
        
        # 2. HU Intensity Windowing
        hu_info = cat_prior.get("hu_intensity_windowing", {})
        hu_min = hu_info.get("recommended_window_min", -1000.0)
        hu_max = hu_info.get("recommended_window_max", 300.0)
        
        # Pathology specific HU tuning
        if category_code == "2d": # Pulmonary Nodules (solid/subsolid soft tissue)
            hu_min, hu_max = -400.0, 200.0
        elif category_code == "1e": # Micronodules
            hu_min, hu_max = -500.0, 150.0
        elif category_code == "1c": # Emphysema (Trapped Air)
            hu_min, hu_max = -1024.0, -850.0
        elif category_code == "2e": # Pleural Effusion (Fluid)
            hu_min, hu_max = -100.0, 100.0
        elif category_code == "2g": # Pneumothorax (Air in Pleural Space)
            hu_min, hu_max = -1024.0, -900.0
            
        hu_mask = (ras_ct_volume >= hu_min) & (ras_ct_volume <= hu_max)
        
        # 3. NLP Text Spatial Locator Masking
        text_spatial_mask = self.parse_spatial_locators(prompt_text, ras_ct_volume.shape)
        
        # 4. Combine Layers: Constrain strictly inside anatomical lung cavity (Stat-H1)
        candidate_weights = gaussian_map * hu_mask * text_spatial_mask
        
        # For Pleural Effusion and Pneumothorax, allow slight extra-pulmonary peripheral extension
        if category_code not in ["2e", "2g"]:
            candidate_weights *= lung_mask
            
        # 5. Adaptive Volume Scaling & Thresholding (Stat-H2)
        # Select top probability voxels matching physical volume bounds to avoid Dice denominator explosion
        candidate_voxels = np.where(candidate_weights > 0.1)
        num_candidates = len(candidate_voxels[0])
        
        if num_candidates == 0:
            # Fallback to top spatial intensity voxels
            candidate_weights = gaussian_map * hu_mask
            candidate_voxels = np.where(candidate_weights > 0.05)
            num_candidates = len(candidate_voxels[0])
            
        if num_candidates == 0:
            return np.zeros((Z, Y, X), dtype=np.uint8)
            
        # Target volume scaling based on median component volume prior
        comp_vol_stats = cat_prior.get("component_topology", {}).get("component_vol_mm3_stats", {})
        median_vol_mm3 = comp_vol_stats.get("median", 2000.0)
        
        # Convert target volume mm3 to target voxel count (assuming ~1mm3 voxel resolution)
        target_voxels = max(int(median_vol_mm3), 50)
        
        # Focal vs Non-Focal volume cap
        if category_code in ["2d", "1e", "2a", "2h"]:
            target_voxels = min(target_voxels, 5000) # Focal lesion cap
        elif category_code in ["2b", "2c", "1a", "1b", "1d"]:
            target_voxels = min(target_voxels, 35000) # Diffuse/Parenchymal cap
        elif category_code in ["1c", "2e", "2g"]:
            target_voxels = min(target_voxels, 60000) # Large volume cap
            
        # Extract binary mask by thresholding top-N probability voxels
        if num_candidates > target_voxels:
            threshold_val = np.partition(candidate_weights[candidate_voxels], -target_voxels)[-target_voxels]
            final_mask = candidate_weights >= threshold_val
        else:
            final_mask = candidate_weights > 0.15
            
        # Morphological Cleanup
        min_size = cat_prior.get("component_topology", {}).get("recommended_min_size_voxels", 10)
        if final_mask.any():
            labeled, num_features = label(final_mask)
            if num_features > 0:
                component_sizes = np.bincount(labeled.ravel())
                too_small = component_sizes < min_size
                too_small_mask = too_small[labeled]
                final_mask[too_small_mask] = False
                
        return final_mask.astype(np.uint8)


def main():
    args = parse_args()
    
    # Resolve Paths
    priors_path = Path(args.priors_json) if args.priors_json else DATA_DIR / "phase_1" / "phase_1_priors_bundle.json"
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_rule_based"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 2A — Statistical / Rule-Based Prior Baseline Execution (v2.0)")
    print(f"Target Split:        {args.split}")
    print(f"Priors Bundle:       {priors_path}")
    print(f"Dataset JSON:        {args.dataset_json}")
    print(f"Output Directory:    {output_dir}")
    print("=" * 80)
    
    # Initialize Predictor & Reader
    predictor = StatisticalPriorPredictorV2(priors_path)
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
    
    # Process Scans Loop
    for entry in tqdm(entries, desc=f"Phase 2A Baseline v2.0 [{args.split}]"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue
            
        out_nii_path = output_dir / f"{scan_id}.nii.gz"
        if out_nii_path.exists():
            # Overwrite for v2.0 benchmark update
            pass
            
        raw_nifti_path = Path(args.img_raw_dir) / f"{scan_id}.nii.gz"
        if not raw_nifti_path.exists():
            tqdm.write(f"[WARNING] Raw image missing: {raw_nifti_path}")
            missing_scans += 1
            continue
            
        # 1. Read & Reorient CT scan volume to RAS
        img_ras, img_props = reader.read_images([str(raw_nifti_path)]) # shape: (1, Z, Y, X)
        ras_volume = img_ras[0] # 3D float volume (Z, Y, X)
        
        # 2. Extract Stat-H1 Anatomical Lung Field Cavity Mask
        lung_cavity_mask = predictor.extract_lung_cavity_mask(ras_volume)
        
        # 3. Extract Finding Prompts
        findings = entry.get("findings", {})
        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f.get("text", "") if isinstance(f, dict) else str(f) for f in findings]
            
        if not prompts:
            continue
            
        # 4. Generate 3D mask per finding prompt in RAS space using v2.0 pipeline
        finding_masks_ras = []
        for prompt in prompts:
            mask_3d = predictor.generate_finding_mask(ras_volume, prompt, lung_cavity_mask) # shape: (Z, Y, X)
            finding_masks_ras.append(mask_3d)
            
        # Stack to 4D array in RAS space (X, Y, Z, F)
        ras_4d_stack = np.stack(finding_masks_ras, axis=-1) # shape: (Z, Y, X, F)
        pred_xyzf = np.transpose(ras_4d_stack, (2, 1, 0, 3))
        
        # 5. 4D Back-Reorientation Contract Execution
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
        del img_ras, ras_volume, lung_cavity_mask, finding_masks_ras, ras_4d_stack, pred_xyzf, pred_nib_ras, pred_nib_back, pred_back_data, pred_back_fxyz, out_nii
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
        print("Invoking Official Evaluator (scripts/evaluate.py)...")
        print("=" * 80)
        
        eval_script = ROOT_DIR / "scripts" / "evaluate.py"
        gt_dir = RAW_MASKS_DIR
        
        cmd = [
            sys.executable, str(eval_script),
            "--gt_dir", str(gt_dir),
            "--pred_dir", str(output_dir),
            "--split", args.split,
            "--dataset_json", args.dataset_json
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\n--- Evaluation Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- Evaluation Warnings/Errors ---")
            print(result.stderr)
            
        # Log results to markdown log file
        log_dir = LOGS_DIR / "phase_2a_rule_based"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "exp_002_rule_based_v2_val_eval.md"
        
        with open(log_path, 'w') as f_log:
            f_log.write("# Phase 2A — Statistical Prior Baseline (v2.0) Evaluation Log\n\n")
            f_log.write(f"- **Target Split:** {args.split}\n")
            f_log.write(f"- **Prediction Directory:** `{output_dir}`\n")
            f_log.write(f"- **Evaluated Files:** {len(list(output_dir.glob('*.nii.gz')))}\n\n")
            f_log.write("## Quantitative Metric Evaluation Output\n\n```text\n")
            f_log.write(result.stdout)
            f_log.write("\n```\n")
            
        print(f"\n[INFO] Saved evaluation report to: {log_path}")


if __name__ == "__main__":
    main()
