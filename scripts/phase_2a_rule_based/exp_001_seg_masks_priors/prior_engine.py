"""
===============================================================================
MODULE:         Empirical Spatial PDF Baseline Engine
OBJECTIVE:      Core PyTorch / Nibabel processing class for building, loading, 
                and resampling 3D empirical spatial probability density heatmaps.
===============================================================================
"""

import os
import sys
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import label

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import CATEGORY_MAP
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from scripts.common.prompt_normalizer import clean_finding_prompt
from scripts.common.orientation import load_nifti_ras


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


class EmpiricalSpatialPDFBaseline:
    """
    Data-driven 3D Empirical Spatial Probability Density Baseline:
    1. Accumulates training GT segmentations per category in canonical RAS space (192x192x192).
    2. Computes empirical probability density P_c(z, y, x) = 1/N_c * sum(Mask_i).
    3. Resamples P_c to target validation scan shape and thresholds to generate 3D/4D predictions.
    """
    CANONICAL_SHAPE = (512, 512, 512)

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
                # Use centralized spatial engine which auto-repairs np.eye(4) identity bugs
                gt_data, _, _ = load_nifti_ras(seg_path, ref_affine=None)
                if gt_data.ndim == 4 and gt_data.shape[0] < np.min(gt_data.shape[1:4]):
                    # It's already (F, X, Y, Z) from load_nifti_ras if it detected 4D
                    pass
                elif gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:3]):
                    # If it came back as (X, Y, Z, F), which shouldn't happen with our robust load_nifti_ras but just in case
                    gt_data = np.moveaxis(gt_data, -1, 0)
            except Exception as e:
                tqdm.write(f"[WARNING] Failed to load GT mask {seg_path}: {e}")
                continue

            if gt_data.ndim == 3:
                gt_data = np.expand_dims(gt_data, axis=0) # (1, X, Y, Z)

            if gt_data.ndim == 4:
                if gt_data.shape[-1] < np.min(gt_data.shape[:3]):
                    gt_data = np.moveaxis(gt_data, -1, 0) # (F, X, Y, Z)

            num_findings = gt_data.shape[0]

            for f_idx in range(num_findings):
                cat_code = str(categories_dict.get(str(f_idx), ""))
                if not cat_code or cat_code not in CATEGORY_MAP:
                    continue

                binary_mask = (gt_data[f_idx] > 0).astype(np.float32) # shape: (X, Y, Z)
                if not binary_mask.any():
                    continue

                # PyTorch 3D Interpolation to Canonical Shape in (X, Y, Z) space
                # Move to GPU for massive speedup on 512-cubed volumes
                mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0).cuda() # (1, 1, X, Y, Z)
                if mask_tensor.shape[2:] != self.CANONICAL_SHAPE:
                    mask_canonical = F.interpolate(mask_tensor, size=self.CANONICAL_SHAPE, mode='nearest').squeeze(0).squeeze(0).cpu().numpy()
                else:
                    mask_canonical = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

                accumulators[cat_code] += mask_canonical
                category_counts[cat_code] += 1

            del gt_data
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

        # Specific keyword heuristics handling singular/plural & common variations
        if "linear" in cleaned_text or "band" in cleaned_text or "scar" in cleaned_text or "fibrotic band" in cleaned_text:
            return "2a"
        elif "atelectasis" in cleaned_text or "consolidation" in cleaned_text or "collapse" in cleaned_text:
            return "2b"
        elif "ground-glass" in cleaned_text or "ground glass" in cleaned_text or "ggo" in cleaned_text:
            return "2c"
        elif "nodule" in cleaned_text or "mass" in cleaned_text or "lesion" in cleaned_text:
            return "2d"
        elif "effusion" in cleaned_text or "pleural fluid" in cleaned_text or "pleural thickening" in cleaned_text:
            return "2e"
        elif "honeycomb" in cleaned_text:
            return "2f"
        elif "pneumothorax" in cleaned_text:
            return "2g"
        elif "bronchial wall" in cleaned_text or "peribronchial" in cleaned_text:
            return "1a"
        elif "bronchiectasis" in cleaned_text or "bronchiectatic" in cleaned_text:
            return "1b"
        elif "emphysema" in cleaned_text or "emphysematous" in cleaned_text or "bullous" in cleaned_text:
            return "1c"
        elif "septal" in cleaned_text or "interstitial thickening" in cleaned_text or "reticulation" in cleaned_text:
            return "1d"
        elif "micronodule" in cleaned_text or "centrilobular nodule" in cleaned_text or "tree-in-bud" in cleaned_text:
            return "1e"
        elif "thickening" in cleaned_text:
            return "1d"
        elif "opacity" in cleaned_text:
            return "2c"
            
        return "2h"

    def generate_prediction_mask(self, target_shape_ras: tuple, prompt_text: str = "", cat_code: str = None) -> np.ndarray:
        """
        Signature:
            generate_prediction_mask(target_shape_ras: tuple, prompt_text: str, cat_code: str) -> np.ndarray

        Objective:
            Resample category 3D empirical PDF heatmap P_c to target scan's RAS shape, apply
            category-specific percentile thresholding, and prune tiny 3D noise blobs (<10 voxels).

        Inputs:
            target_shape_ras (tuple): Target 3D volume shape (Z, Y, X) in canonical RAS space.
            prompt_text (str): Free-text radiology report finding prompt.
            cat_code (str): Explicit 14-category code string ('1a'..'2h'). If provided, overrides prompt matching.

        Outputs:
            np.ndarray: 3D uint8 binary prediction mask array with shape matching target_shape_ras.
        """
        if cat_code is not None and str(cat_code) in CATEGORY_MAP:
            code = str(cat_code)
        else:
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

