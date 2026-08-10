"""
===============================================================================
SCRIPT:         exp_009_morphological_shape_priors.py
PHASE:          Phase 2A — Statistical / Rule-Based Prior Baseline
LOCATION:       scripts/phase_2a_rule_based/exp_009_morphological_shape_priors.py
OBJECTIVE:      Single-file executable pipeline for Phase 2A Exp 009.
                1. Loads spatially-anchored 3D empirical PDF heatmaps.
                2. Applies Selective HU Radiodensity Windowing and Volume Quantile Matching.
                3. Extracts 3D connected components (blobs).
                4. Computes 3D aspect ratio Z / mean(X, Y) and approximate sphericity S.
                5. Prunes noise blobs violating category-specific 3D morphological shape profiles.
                6. Stacks 4D NIfTI masks and triggers automated challenge evaluation.
USAGE:          python scripts/phase_2a_rule_based/exp_009_morphological_shape_priors.py --split val --eval
===============================================================================
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import label, find_objects

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import (
    DATA_DIR, DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    PREDICTIONS_DIR, LOGS_DIR, CATEGORY_MAP
)
from scripts.phase_2a_rule_based.common import (
    EmpiricalSpatialPDFBaseline, run_prior_inference_and_eval,
    EMPIRICAL_VOLUME_QUANTILES, CATEGORY_HU_BOUNDS
)

# Empirical 3D Morphological Shape Profile Bounds per Category
# DERIVATION: Derived directly from Phase 1 3D morphological topology analysis (PHASE_1_DATA_ANALYSIS_SUMMARY.md).
# Format: {cat_code: (min_aspect_ratio, max_aspect_ratio, min_sphericity)}
CATEGORY_MORPHOLOGY_BOUNDS = {
    "1a": (0.1, 4.0, 0.20),  # Bronchial wall thickening (tubular/peribronchial)
    "1b": (0.1, 4.0, 0.20),  # Bronchiectasis (dilated tubular)
    "1c": (0.2, 5.0, 0.25),  # Emphysema (apical clusters)
    "1d": (0.1, 3.5, 0.15),  # Septal thickening (thin interlobular sheets)
    "1e": (0.2, 4.0, 0.30),  # Micronodules (small focal clusters)
    "1f": (0.1, 5.0, 0.15),  # Other non-focal
    "2a": (0.1, 3.5, 0.20),  # Linear opacities (linear soft tissue)
    "2b": (0.2, 5.0, 0.25),  # Atelectasis / consolidation (broad dependent)
    "2c": (0.2, 5.0, 0.25),  # Ground-glass opacity (parenchymal)
    "2d": (0.3, 3.0, 0.40),  # Pulmonary nodules / masses (compact spherical S ~ 0.94, AR ~ 0.88)
    "2e": (0.2, 5.0, 0.20),  # Pleural effusion / thickening (fluid sheet along wall)
    "2f": (0.2, 4.5, 0.20),  # Honeycombing (subpleural fibrotic)
    "2g": (0.3, 5.0, 0.30),  # Pneumothorax (pleural space)
    "2h": (0.1, 5.0, 0.15),  # Other focal
}


class MorphologicalShapeBaseline(EmpiricalSpatialPDFBaseline):
    """
    Single-experiment predictor encapsulating Exp 009 category-adaptive 3D connected
    component aspect ratio Z / mean(X,Y) and sphericity index S pruning.
    """

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
            heatmap, applying volume quantile binarization, extracting 3D connected components, and
            pruning noise blobs violating category-specific aspect ratio and sphericity bounds.

        Inputs:
            cat_code (str): Finding category code ('1a'..'2h').
            target_shape_ras (tuple): Target 3D volume shape (X, Y, Z).
            ct_img_ras (np.ndarray, optional): Raw CT HU intensity volume matching target_shape_ras.
            prompt_text (str, optional): Free-text radiology prompt.

        Outputs:
            np.ndarray: 3D uint8 binary prediction mask array.
        """
        code = str(cat_code) if str(cat_code) in CATEGORY_MAP else "2h"
        pdf_np = self.spatial_pdfs.get(code, np.full(self.CANONICAL_SHAPE, 0.01, dtype=np.float32))

        # 1. PyTorch 3D Spatial Resampling to target RAS volume shape
        if target_shape_ras != self.CANONICAL_SHAPE:
            pdf_tensor = torch.from_numpy(pdf_np).unsqueeze(0).unsqueeze(0)
            if torch.cuda.is_available():
                try:
                    pdf_tensor_gpu = pdf_tensor.cuda()
                    pdf_target = F.interpolate(
                        pdf_tensor_gpu, size=target_shape_ras, mode='trilinear', align_corners=False
                    ).squeeze(0).squeeze(0).cpu().numpy()
                    del pdf_tensor_gpu
                except Exception:
                    pdf_target = F.interpolate(
                        pdf_tensor, size=target_shape_ras, mode='trilinear', align_corners=False
                    ).squeeze(0).squeeze(0).numpy()
            else:
                pdf_target = F.interpolate(
                    pdf_tensor, size=target_shape_ras, mode='trilinear', align_corners=False
                ).squeeze(0).squeeze(0).numpy()
        else:
            pdf_target = pdf_np.copy()

        # 2. Body cavity air & Selective HU Windowing
        if ct_img_ras is not None and isinstance(ct_img_ras, np.ndarray) and ct_img_ras.shape == target_shape_ras:
            body_mask = (ct_img_ras >= -1000.0) & (ct_img_ras <= 1000.0)
            pdf_target[~body_mask] = 0.0

            SELECTIVE_HU_CATEGORIES = {'1a', '1b', '1c', '2f', '2g'}
            if code in SELECTIVE_HU_CATEGORIES:
                min_hu, max_hu = CATEGORY_HU_BOUNDS.get(code, (-1000.0, 300.0))
                invalid_hu = (ct_img_ras < min_hu) | (ct_img_ras > max_hu)
                pdf_target[invalid_hu] = 0.0

        max_p = float(pdf_target.max())
        if max_p <= 0:
            return np.zeros(target_shape_ras, dtype=np.uint8)

        # 3. Empirical Volume Quantile Matching Binarization
        target_ratio = EMPIRICAL_VOLUME_QUANTILES.get(code, 0.005)
        cutoff = float(np.quantile(pdf_target, 1.0 - target_ratio))
        if cutoff <= 0:
            cutoff = 0.5 * max_p
        binary_mask = (pdf_target >= cutoff).astype(np.uint8)

        # 4. 3D Connected Component Extraction & Morphological Shape Pruning
        effective_min_blob = self.min_blob_voxels.get(code, 10) if isinstance(self.min_blob_voxels, dict) else self.min_blob_voxels
        min_ar, max_ar, min_sphericity = CATEGORY_MORPHOLOGY_BOUNDS.get(code, (0.1, 5.0, 0.15))

        if binary_mask.any():
            labeled, num_features = label(binary_mask)
            if num_features > 0:
                objects = find_objects(labeled)
                for comp_idx, loc in enumerate(objects, start=1):
                    if loc is None:
                        continue
                    
                    # Extract 3D subvolume slice for current component
                    comp_mask = (labeled[loc] == comp_idx)
                    vol_voxels = int(comp_mask.sum())

                    # Minimum volume threshold check
                    if vol_voxels < effective_min_blob:
                        binary_mask[loc][comp_mask] = 0
                        continue

                    # Calculate 3D bounding box extents (dx, dy, dz)
                    dx = loc[0].stop - loc[0].start
                    dy = loc[1].stop - loc[1].start
                    dz = loc[2].stop - loc[2].start

                    # Aspect Ratio AR = dz / mean(dx, dy)
                    mean_xy = 0.5 * (dx + dy) + 1e-5
                    aspect_ratio = float(dz) / mean_xy

                    # Approximate surface area & Sphericity index S
                    sa_approx = 2.0 * (dx * dy + dy * dz + dz * dx) + 1e-5
                    sphericity = (np.pi ** (1.0 / 3.0)) * ((6.0 * vol_voxels) ** (2.0 / 3.0)) / sa_approx

                    # Prune blobs violating category-specific 3D aspect ratio or sphericity bounds
                    if aspect_ratio < min_ar or aspect_ratio > max_ar or sphericity < min_sphericity:
                        binary_mask[loc][comp_mask] = 0

        return binary_mask.astype(np.uint8)


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for Exp 009 pipeline.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2A Exp 009: Category-Adaptive Morphological Shape Priors Baseline Pipeline"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--pdf_cache", type=str, default=None,
        help="Path to empirical spatial PDF npz cache"
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
        "--eval", action="store_true", default=True,
        help="Automatically run scripts/common/evaluate.py after inference (default: True)"
    )
    parser.add_argument(
        "--no_eval", action="store_false", dest="eval",
        help="Disable automatic evaluation"
    )
    parser.add_argument(
        "--force_rebuild", action="store_true", default=False,
        help="Force rebuild of 3D spatial PDF heatmaps cache"
    )
    parser.add_argument(
        "--min_blob_voxels", type=int, default=10,
        help="Minimum voxel threshold for 3D component noise pruning (default: 10)"
    )
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Start index for processing entries"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for processing entries"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for Exp 009 Pipeline."""
    args = parse_args()

    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat_anchored.npz"
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_exp_009_morphological_shape"
    exp_log_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_009_morphological_shape_priors"

    predictor = MorphologicalShapeBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        img_raw_dir=Path(args.img_raw_dir),
        force_rebuild=args.force_rebuild or not pdf_cache_path.exists(),
        threshold_mode="morphological_shape",
        min_blob_voxels=args.min_blob_voxels,
    )

    run_prior_inference_and_eval(
        predictor=predictor,
        split=args.split,
        dataset_json_path=Path(args.dataset_json),
        img_raw_dir=Path(args.img_raw_dir),
        seg_raw_dir=Path(args.seg_raw_dir),
        output_dir=output_dir,
        exp_log_dir=exp_log_dir,
        pdf_cache_path=pdf_cache_path,
        do_eval=args.eval,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == "__main__":
    main()
