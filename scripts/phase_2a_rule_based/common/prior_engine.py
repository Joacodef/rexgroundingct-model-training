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
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from scripts.common.orientation import load_nifti_ras
from scripts.common.nlp_locators import generate_text_spatial_mask, parse_prompt_spatial_locators



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

# Empirical 5th-95th percentile Hounsfield Unit (HU) radiodensity bounds per category.
# DERIVATION: Derived from Phase 1 empirical attenuation profiling (PHASE_1_DATA_ANALYSIS_SUMMARY.md).
CATEGORY_HU_BOUNDS = {
    "1a": (-933.0, 69.0),   # Bronchial wall thickening (airway + peribronchial soft tissue)
    "1b": (-934.0, 86.0),   # Bronchiectasis (airway lumen + wall)
    "1c": (-992.0, 252.0),  # Emphysema (hyper-inflated trapped air)
    "1d": (-1001.0, 121.0), # Septal thickening (interstitial)
    "1e": (-998.0, 101.0),  # Micronodules (small soft tissue nodules)
    "1f": (-986.0, 98.0),   # Other non-focal
    "2a": (-992.0, 399.0),  # Linear opacities (dense linear soft tissue/calcifications)
    "2b": (-995.0, 130.0),  # Atelectasis / consolidation (dense collapsed parenchymal)
    "2c": (-995.0, 138.0),  # Ground-glass opacity (hazy parenchymal)
    "2d": (-995.0, 194.0),  # Pulmonary nodules / masses (dense solid soft tissue)
    "2e": (-1007.0, 135.0), # Pleural effusion / thickening (fluid / pleural tissue)
    "2f": (-905.0, 83.0),   # Honeycombing (subpleural fibrotic restructuring)
    "2g": (-962.0, 145.0),  # Pneumothorax (pleural air cavity)
    "2h": (-915.0, 195.0),  # Other focal
}


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
        self.reader = NibabelIOWithReorient()

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
        # Step 1: Initialize cache directory, load dataset JSON metadata, and setup 3D accumulators
        self.pdf_cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.dataset_json_path, 'r') as f:
            metadata = json.load(f)

        train_entries = metadata.get("train", [])
        if not train_entries:
            raise ValueError(f"No 'train' entries found in {self.dataset_json_path}")

        if self.max_train_scans and len(train_entries) > self.max_train_scans:
            train_entries = train_entries[:self.max_train_scans]

        accumulators = {code: np.zeros(self.CANONICAL_SHAPE, dtype=np.float32) for code in CATEGORY_MAP.keys()}
        category_counts = {code: 0 for code in CATEGORY_MAP.keys()}

        print(f"[INFO] Processing {len(train_entries)} training scans for empirical PDF building...")

        # Step 2: Iterate over training split entries and accumulate spatially aligned finding masks
        for entry in tqdm(train_entries, desc="Building 3D PDF Heatmaps"):
            # Extract unique scan identifier from entry metadata
            scan_id = entry.get("name", "").replace(".nii.gz", "")
            if not scan_id:
                continue

            # Verify existence of raw 4D ground-truth segmentation NIfTI file
            seg_path = self.seg_raw_dir / f"{scan_id}.nii.gz"
            if not seg_path.exists():
                continue

            # Ensure finding category mapping dictionary exists for this scan
            categories_dict = entry.get("categories", {})
            if not categories_dict:
                continue

            # Verify existence of parent 3D raw CT scan image (required for spatial anchoring)
            img_path = self.img_raw_dir / f"{scan_id}.nii.gz"
            if not img_path.exists():
                tqdm.write(f"[WARNING] Missing raw image for anchoring: {img_path}")
                continue

            try:
                # Step 3: Canonical RAS Spatial Anchoring — load CT image first to extract reference physical affine matrix, then load parent GT mask
                _, img_nii, _ = load_nifti_ras(img_path)
                gt_data, _, _ = load_nifti_ras(seg_path, ref_affine=img_nii.affine)

                if gt_data.ndim == 4 and gt_data.shape[0] < np.min(gt_data.shape[1:4]):
                    pass
                elif gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:3]):
                    gt_data = np.moveaxis(gt_data, -1, 0)
            except Exception as e:
                tqdm.write(f"[WARNING] Failed to load GT mask {seg_path}: {e}")
                continue

            if gt_data.ndim == 3:
                gt_data = np.expand_dims(gt_data, axis=0) # (1, X, Y, Z)

            if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:3]):
                gt_data = np.moveaxis(gt_data, -1, 0) # (F, X, Y, Z)

            num_findings = gt_data.shape[0]

            # Step 4: Extract per-finding binary mask and resample to canonical (512, 512, 512) grid via PyTorch interpolation
            for f_idx in range(num_findings):
                # Retrieve category code for finding index (e.g. '2d' for pulmonary nodules/masses)
                cat_code = str(categories_dict.get(str(f_idx), ""))
                if not cat_code or cat_code not in CATEGORY_MAP:
                    continue

                # Binarize 3D GT mask for current finding channel
                binary_mask = (gt_data[f_idx] > 0).astype(np.float32)
                if not binary_mask.any():
                    continue

                # PyTorch 3D Spatial Resampling to canonical (512, 512, 512) grid
                mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
                if mask_tensor.shape[2:] != self.CANONICAL_SHAPE:
                    # GPU CUDA interpolation with CPU memory fallback
                    if torch.cuda.is_available():
                        try:
                            mask_tensor_gpu = mask_tensor.cuda()
                            mask_canonical = F.interpolate(mask_tensor_gpu, size=self.CANONICAL_SHAPE, mode='nearest').squeeze(0).squeeze(0).cpu().numpy()
                            del mask_tensor_gpu
                        except Exception:
                            mask_canonical = F.interpolate(mask_tensor, size=self.CANONICAL_SHAPE, mode='nearest').squeeze(0).squeeze(0).numpy()
                    else:
                        mask_canonical = F.interpolate(mask_tensor, size=self.CANONICAL_SHAPE, mode='nearest').squeeze(0).squeeze(0).numpy()
                else:
                    mask_canonical = mask_tensor.squeeze(0).squeeze(0).numpy()

                # Accumulate 3D voxel frequency count and increment category scan counter
                accumulators[cat_code] += mask_canonical
                category_counts[cat_code] += 1

            # Free 4D GT volume memory and trigger garbage collection per scan loop
            del gt_data
            gc.collect()

        # Step 5: Density Normalization & Compressed NPZ Caching — compute P_c(z, y, x) = accumulators / count and write NPZ archive
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
        ct_img_ras: np.ndarray = None,
        prompt_text: str = None,
    ) -> np.ndarray:
        """
        Signature:
            generate_prediction_mask(
                cat_code: str, target_shape_ras: tuple, ct_img_ras: np.ndarray = None, prompt_text: str = None
            ) -> np.ndarray

        Objective:
            Generate a 3D binary segmentation mask for a target scan by resampling the 3D category PDF
            heatmap to target_shape_ras, applying text prompt spatial locator masking (if prompt_text is provided),
            HU radiodensity windowing (if ct_img_ras is provided), and thresholding / component filtering.

        Inputs:
            cat_code (str): Category finding code ('1a'..'2h').
            target_shape_ras (tuple): Target 3D volume shape (X, Y, Z).
            ct_img_ras (np.ndarray, optional): Raw CT HU intensity volume array matching target_shape_ras.
            prompt_text (str, optional): Free-text radiology finding description string.

        Outputs:
            np.ndarray: 3D uint8 binary prediction mask array with shape matching target_shape_ras.
        """
        code = str(cat_code) if str(cat_code) in CATEGORY_MAP else "2h"
        pdf_np = self.spatial_pdfs.get(code, np.full(self.CANONICAL_SHAPE, 0.01, dtype=np.float32))

        # 1. PyTorch 3D Spatial Resampling to target RAS volume shape
        # GPU CUDA interpolation with CPU fallback if CUDA is unavailable or OOM
        if target_shape_ras != self.CANONICAL_SHAPE:
            pdf_tensor = torch.from_numpy(pdf_np).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                try:
                    pdf_tensor_gpu = pdf_tensor.cuda()
                    pdf_target = F.interpolate(pdf_tensor_gpu, size=target_shape_ras, mode='trilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
                    del pdf_tensor_gpu
                except Exception:
                    pdf_target = F.interpolate(pdf_tensor, size=target_shape_ras, mode='trilinear', align_corners=False).squeeze(0).squeeze(0).numpy()
            else:
                pdf_target = F.interpolate(pdf_tensor, size=target_shape_ras, mode='trilinear', align_corners=False).squeeze(0).squeeze(0).numpy()
        else:
            pdf_target = pdf_np.copy()

        # 1.2 NLP Text Prompt Spatial Locator Gating (Exp 007)
        if self.threshold_mode == "text_spatial_locators" and prompt_text:
            text_roi_mask = generate_text_spatial_mask(prompt_text, target_shape_ras)
            pdf_target = pdf_target * text_roi_mask

        # 1.5 Radiodensity HU Intensity Filtering (Exp 003 / Exp 004 / Exp 005 / Exp 006 / Exp 007)
        if ct_img_ras is not None and isinstance(ct_img_ras, np.ndarray) and ct_img_ras.shape == target_shape_ras:
            if self.threshold_mode in ("body_gated_hu", "composite_rules"):
                # Exp 005 & Exp 006: Body Cavity Air Masking (HU in [-1000, 1000] HU) + Selective HU Windowing
                body_mask = (ct_img_ras >= -1000.0) & (ct_img_ras <= 1000.0)
                pdf_target[~body_mask] = 0.0

                SELECTIVE_HU_CATEGORIES = {'1a', '1b', '1c', '2f', '2g'}
                if code in SELECTIVE_HU_CATEGORIES:
                    min_hu, max_hu = CATEGORY_HU_BOUNDS.get(code, (-1000.0, 300.0))
                    invalid_hu = (ct_img_ras < min_hu) | (ct_img_ras > max_hu)
                    pdf_target[invalid_hu] = 0.0
            elif self.threshold_mode in ("selective_hu", "text_spatial_locators"):
                SELECTIVE_HU_CATEGORIES = {'1a', '1b', '1c', '2f', '2g'}
                if code in SELECTIVE_HU_CATEGORIES:
                    min_hu, max_hu = CATEGORY_HU_BOUNDS.get(code, (-1000.0, 300.0))
                    invalid_hu = (ct_img_ras < min_hu) | (ct_img_ras > max_hu)
                    pdf_target[invalid_hu] = 0.0
            elif self.threshold_mode in ("hu_windowed", "hu_quantile"):
                min_hu, max_hu = CATEGORY_HU_BOUNDS.get(code, (-1000.0, 300.0))
                invalid_hu = (ct_img_ras < min_hu) | (ct_img_ras > max_hu)
                pdf_target[invalid_hu] = 0.0

        max_p = float(pdf_target.max())
        if max_p <= 0:
            return np.zeros(target_shape_ras, dtype=np.uint8)

        # 2. Binarization Strategy Selection
        if self.threshold_mode == "composite_rules":
            # Exp 006: Composite Rules with Validation Density Scaling (1.5x target ratio)
            target_ratio = EMPIRICAL_VOLUME_QUANTILES.get(code, 0.005) * 1.5
            cutoff = float(np.quantile(pdf_target, 1.0 - target_ratio))
            if cutoff <= 0:
                cutoff = 0.5 * max_p
            binary_mask = (pdf_target >= cutoff).astype(np.uint8)
        elif self.threshold_mode in ("quantile", "hu_quantile", "selective_hu", "body_gated_hu", "text_spatial_locators"):
            # Exp 002, Exp 003, Exp 004, Exp 005 & Exp 007: Empirical Volume Quantile Matching Strategy
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

        # 3. 3D Connected Component Size Cleanup (Noise Blob Pruning)
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

