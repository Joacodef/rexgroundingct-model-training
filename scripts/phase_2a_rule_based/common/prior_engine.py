"""
===============================================================================
MODULE:         Empirical Spatial PDF Baseline Engine (Modular Framework)
LOCATION:       scripts/phase_2a_rule_based/common/prior_engine.py
OBJECTIVE:      Core PyTorch / Nibabel processing class for building, loading, 
                and resampling 3D empirical spatial probability density heatmaps.
                Supports both standard percentile thresholding (Exp 001) and 
                Empirical Volume Quantile Matching (Exp 002).
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
from scripts.common.orientation import load_nifti_ras


# Empirical Mean Volume Quantile Ratios (Target V_pred / V_scan).
# DERIVATION: Derived directly from Phase 1 ground-truth lesion volume statistics (PHASE_1_DATA_ANALYSIS_SUMMARY.md).
EMPIRICAL_VOLUME_QUANTILES = {
    "1a": 0.005,  # Bronchial wall thickening (~0.5% volume)
    "1b": 0.008,  # Bronchiectasis (~0.8% volume)
    "1c": 0.025,  # Emphysema (~2.5% apical volume)
    "1d": 0.012,  # Septal thickening (~1.2% interstitial volume)
    "1e": 0.003,  # Micronodules (~0.3% focal cluster volume)
    "1f": 0.010,  # Other non-focal (~1.0% volume)
    "2a": 0.002,  # Linear opacities (~0.2% linear volume)
    "2b": 0.035,  # Atelectasis / consolidation (~3.5% dependent volume)
    "2c": 0.030,  # Ground-glass opacity (~3.0% parenchymal volume)
    "2d": 0.001,  # Pulmonary nodules / masses (~0.1% compact focal volume)
    "2e": 0.045,  # Pleural effusion / thickening (~4.5% basal fluid volume)
    "2f": 0.015,  # Honeycombing (~1.5% subpleural volume)
    "2g": 0.020,  # Pneumothorax (~2.0% pleural cavity volume)
    "2h": 0.005,  # Other focal (~0.5% volume)
}


def _resample_3d_array(arr_3d: np.ndarray, target_shape: tuple[int, int, int], mode: str = "nearest") -> np.ndarray:
    """
    Signature:
        _resample_3d_array(arr_3d: np.ndarray, target_shape: tuple[int, int, int], mode: str = "nearest") -> np.ndarray

    Objective:
        Resample a 3D float32 volume array to target_shape using PyTorch 3D interpolation with GPU/CPU fallback.

    Inputs:
        arr_3d (np.ndarray): Input 3D float32 array.
        target_shape (tuple[int, int, int]): Target (X, Y, Z) grid dimensions.
        mode (str): PyTorch grid interpolation mode ('nearest' or 'trilinear').

    Outputs:
        np.ndarray: Resampled 3D float32 array.
    """
    if arr_3d.shape == target_shape:
        return arr_3d.copy()

    tensor_5d = torch.from_numpy(arr_3d).unsqueeze(0).unsqueeze(0)
    align_corners = False if mode == "trilinear" else None

    if torch.cuda.is_available():
        try:
            gpu_tensor = tensor_5d.cuda()
            resampled = F.interpolate(gpu_tensor, size=target_shape, mode=mode, align_corners=align_corners)
            out = resampled.squeeze(0).squeeze(0).cpu().numpy()
            del gpu_tensor, resampled
            return out
        except Exception:
            pass

    resampled = F.interpolate(tensor_5d, size=target_shape, mode=mode, align_corners=align_corners)
    return resampled.squeeze(0).squeeze(0).numpy()


class EmpiricalSpatialPDFBaseline:
    """
    Data-driven 3D Empirical Spatial Probability Density Baseline:
    1. Accumulates training GT segmentations per category in canonical RAS space.
    2. Computes empirical probability density P_c(z, y, x) = 1/N_c * sum(Mask_i).
    3. Resamples P_c to target validation scan shape and thresholds to generate 3D/4D predictions.
    """
    CANONICAL_SHAPE = (512, 512, 512)

    def __init__(
        self,
        pdf_cache_path: Path,
        dataset_json_path: Path,
        seg_raw_dir: Path,
        img_raw_dir: Path = None,
        max_train_scans: int = 300,
        force_rebuild: bool = False,
        threshold_mode: str = "percentile",
        min_blob_voxels: int = 10,
    ):
        """
        Signature:
            __init__(
                pdf_cache_path: Path, dataset_json_path: Path, seg_raw_dir: Path, 
                img_raw_dir: Path, max_train_scans: int, force_rebuild: bool, 
                threshold_mode: str, min_blob_voxels: int
            ) -> None

        Objective:
            Initialize 3D empirical PDF baseline engine, loading or generating cached 3D heatmaps.

        Inputs:
            pdf_cache_path (Path): Path to .npz file containing cached 3D probability density maps.
            dataset_json_path (Path): Path to dataset.json file.
            seg_raw_dir (Path): Path to raw GT segmentations directory.
            img_raw_dir (Path, optional): Path to raw CT images directory for anchoring. Defaults to RAW_IMAGES_DIR.
            max_train_scans (int): Max training split scans to sample for PDF building. Default 300.
            force_rebuild (bool): Whether to force rebuilding PDF heatmaps from scratch. Default False.
            threshold_mode (str): Binarization strategy ('percentile' for Exp 001, 'quantile' for Exp 002).
            min_blob_voxels (int): Minimum voxel threshold for 3D component noise pruning. Default 10.

        Outputs:
            None
        """
        if img_raw_dir is None:
            from scripts.config import RAW_IMAGES_DIR
            img_raw_dir = RAW_IMAGES_DIR

        self.pdf_cache_path = Path(pdf_cache_path)
        self.dataset_json_path = Path(dataset_json_path)
        self.seg_raw_dir = Path(seg_raw_dir)
        self.img_raw_dir = Path(img_raw_dir)
        self.max_train_scans = max_train_scans
        self.threshold_mode = threshold_mode
        self.min_blob_voxels = min_blob_voxels

        if force_rebuild or not self.pdf_cache_path.exists():
            print(f"[INFO] Building 3D Empirical Spatial PDF Heatmaps from Train Split (max {self.max_train_scans} scans)...")
            self.spatial_pdfs = self._build_pdf_cache()
        else:
            print(f"[INFO] Loading cached 3D Empirical Spatial PDF Heatmaps from {self.pdf_cache_path}...")
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
            dict: Dictionary mapping 14 category codes ('1a'..'2h') to 3D numpy arrays of canonical shape.
        """
        # Step 1: Metadata Initialization & Accumulator Setup
        # Create cache parent directory, load dataset metadata, limit training scans, and initialize 3D accumulators.
        self.pdf_cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.dataset_json_path, 'r') as f:
            metadata = json.load(f)

        train_entries = metadata.get("train", [])
        if not train_entries:
            raise ValueError(f"No 'train' entries found in {self.dataset_json_path}")

        if self.max_train_scans and len(train_entries) > self.max_train_scans:
            train_entries = train_entries[:self.max_train_scans]

        # 3D voxel density accumulators P_c(z, y, x) initialized to zeros for all 14 finding categories
        accumulators = {code: np.zeros(self.CANONICAL_SHAPE, dtype=np.float32) for code in CATEGORY_MAP.keys()}
        category_counts = {code: 0 for code in CATEGORY_MAP.keys()}

        print(f"[INFO] Processing {len(train_entries)} training scans for empirical PDF building...")

        # Step 2: Training Scan Iteration & GT Mask Loading
        # Iterate through training split entries, load ground truth segmentation masks, and ensure canonical RAS orientation.
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
                # Load ground truth segmentation mask in canonical RAS coordinate space
                gt_data, _, _ = load_nifti_ras(seg_path)
            except Exception as e:
                tqdm.write(f"[WARNING] Failed to load GT mask {seg_path}: {e}")
                continue

            # Ensure 4D shape convention: (F, X, Y, Z) where F is the finding channel axis
            if gt_data.ndim == 3:
                gt_data = np.expand_dims(gt_data, axis=0)

            if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:3]):
                gt_data = np.moveaxis(gt_data, -1, 0)

            num_findings = gt_data.shape[0]

            # Step 3: Finding Channel Normalization & 3D Spatial Resampling
            # Extract per-finding binary masks, resample to canonical (512, 512, 512) grid, and aggregate into 3D accumulators.
            for f_idx in range(num_findings):
                cat_code = str(categories_dict.get(str(f_idx), ""))
                if not cat_code or cat_code not in CATEGORY_MAP:
                    continue

                binary_mask = (gt_data[f_idx] > 0).astype(np.float32)
                if not binary_mask.any():
                    continue

                # Resample 3D binary mask to canonical (512, 512, 512) spatial grid via PyTorch interpolation
                mask_canonical = _resample_3d_array(binary_mask, self.CANONICAL_SHAPE, mode='nearest')
                accumulators[cat_code] += mask_canonical
                category_counts[cat_code] += 1

            # Garbage collect per-scan memory to avoid RAM accumulation
            del gt_data
            gc.collect()

        # Step 4: Density Normalization & Compressed Storage
        # Compute empirical probability density heatmaps P_c(z, y, x) = accumulators / count and write NPZ archive to disk.
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

        np.savez_compressed(self.pdf_cache_path, **save_dict)
        print(f"[SUCCESS] Saved 14-category 3D Spatial PDF Heatmaps to {self.pdf_cache_path}")
        return spatial_pdfs

    def _load_pdf_cache(self) -> dict:
        """
        Signature:
            _load_pdf_cache() -> dict

        Objective:
            Load cached 3D Empirical Spatial PDF NPZ file from disk.

        Inputs:
            None (Uses instance self.pdf_cache_path).

        Outputs:
            dict: Dictionary mapping 14 category codes to 3D numpy arrays.
        """
        if not self.pdf_cache_path.exists():
            raise FileNotFoundError(f"PDF cache file not found at {self.pdf_cache_path}")

        npz_file = np.load(self.pdf_cache_path)
        spatial_pdfs = {code: npz_file[code].astype(np.float32) for code in npz_file.files}
        return spatial_pdfs

    def generate_prediction_mask(
        self,
        cat_code: str,
        target_shape_ras: tuple,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Signature:
            generate_prediction_mask(
                cat_code: str, target_shape_ras: tuple, *args, **kwargs
            ) -> np.ndarray

        Objective:
            Generate a 3D binary segmentation mask for a target scan by resampling the 3D category PDF
            heatmap to target_shape_ras, thresholding via Percentile Scaling (Exp 001) or Quantile Matching (Exp 002),
            and applying noise component pruning.

        Inputs:
            cat_code (str): Category finding code ('1a'..'2h').
            target_shape_ras (tuple): Target 3D volume shape (X, Y, Z).

        Outputs:
            np.ndarray: 3D uint8 binary prediction mask array with shape matching target_shape_ras.
        """
        code = str(cat_code) if str(cat_code) in CATEGORY_MAP else "2h"
        pdf_np = self.spatial_pdfs.get(code, np.full(self.CANONICAL_SHAPE, 0.01, dtype=np.float32))

        pdf_target = _resample_3d_array(pdf_np, target_shape_ras, mode='trilinear')

        max_p = float(pdf_target.max())
        if max_p <= 0:
            return np.zeros(target_shape_ras, dtype=np.uint8)

        # Binarization Strategy Selection
        if self.threshold_mode == "quantile":
            # Exp 002: Empirical Volume Quantile Matching Strategy
            target_ratio = EMPIRICAL_VOLUME_QUANTILES.get(code, 0.005)
            cutoff = float(np.quantile(pdf_target, 1.0 - target_ratio))
            if cutoff <= 0:
                cutoff = 0.5 * max_p
            binary_mask = (pdf_target >= cutoff).astype(np.uint8)
        else:
            # Exp 001: Percentile Factor Scaling Strategy
            factor = self.threshold_factors.get(code, 0.40) if hasattr(self, 'threshold_factors') and isinstance(self.threshold_factors, dict) else 0.40
            p_threshold = factor * max_p
            binary_mask = (pdf_target >= p_threshold).astype(np.uint8)

        # 3D Connected Component Size Cleanup (Noise Blob Pruning)
        if isinstance(self.min_blob_voxels, dict):
            effective_min_blob = self.min_blob_voxels.get(code, 10)
        else:
            effective_min_blob = self.min_blob_voxels

        if binary_mask.any() and effective_min_blob > 0:
            labeled, num_features = label(binary_mask)
            if num_features > 0:
                sizes = np.bincount(labeled.ravel())
                too_small = sizes < effective_min_blob
                binary_mask[too_small[labeled]] = 0

        return binary_mask.astype(np.uint8)
