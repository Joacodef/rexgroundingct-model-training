"""
===============================================================================
SCRIPT:         exp_008_local_contrast_edge_priors.py
PHASE:          Phase 2A — Statistical / Rule-Based Prior Baseline
LOCATION:       scripts/phase_2a_rule_based/exp_008_local_contrast_edge_priors.py
OBJECTIVE:      Single-file executable pipeline for Phase 2A Exp 008.
                1. Loads spatially-anchored 3D empirical PDF heatmaps.
                2. Computes GPU-accelerated 3D intensity gradient magnitudes ||grad HU||
                   and local background contrast deltas (HU_voxel - mu_local_5mm).
                3. Gates spatial PDF heatmaps using local edge and contrast attenuation profiles.
                4. Applies Selective HU Radiodensity Windowing and Volume Quantile Matching.
                5. Performs 3D connected-component noise blob pruning.
                6. Stack 4D NIfTI masks and triggers automated challenge evaluation.
USAGE:          python scripts/phase_2a_rule_based/exp_008_local_contrast_edge_priors.py --split val --eval
===============================================================================
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter, label

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import (
    DATA_DIR, DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    PREDICTIONS_DIR, LOGS_DIR, CATEGORY_MAP, PHASE_2A_PDFS_DIR
)
from scripts.phase_2a_rule_based.common import (
    EmpiricalSpatialPDFBaseline, run_prior_inference_and_eval,
    EMPIRICAL_VOLUME_QUANTILES, CATEGORY_HU_BOUNDS
)


def compute_gpu_local_contrast_and_gradient(ct_img_ras: np.ndarray) -> tuple:
    """
    Signature:
        compute_gpu_local_contrast_and_gradient(ct_img_ras: np.ndarray) -> tuple

    Objective:
        Fast GPU PyTorch-accelerated calculation of 3D local background mean (mu_local)
        and 3D gradient magnitude ||grad HU||.

    Inputs:
        ct_img_ras (np.ndarray): 3D raw CT HU intensity volume matching target RAS shape.

    Outputs:
        tuple: (grad_mag: np.ndarray, delta_hu: np.ndarray)
    """
    if torch.cuda.is_available():
        try:
            t_img = torch.from_numpy(ct_img_ras).unsqueeze(0).unsqueeze(0).cuda().float()

            # 3D Local Background Mean via 3D AvgPool (kernel size 5x5x5)
            mu_local_t = F.avg_pool3d(t_img, kernel_size=5, stride=1, padding=2)
            delta_hu_t = t_img - mu_local_t

            # 3D Intensity Gradient Magnitude via Conv3d finite differences
            kx = torch.tensor([[[[-0.5, 0.0, 0.5]]]], device='cuda', dtype=torch.float32).unsqueeze(0)
            ky = torch.tensor([[[[-0.5], [0.0], [0.5]]]], device='cuda', dtype=torch.float32).unsqueeze(0)
            kz = torch.tensor([[[[[ -0.5 ]]], [[[ 0.0 ]]], [[[ 0.5 ]]]]], device='cuda', dtype=torch.float32)

            gx = F.conv3d(t_img, kx, padding=(0, 0, 1))
            gy = F.conv3d(t_img, ky, padding=(0, 1, 0))
            gz = F.conv3d(t_img, kz, padding=(1, 0, 0))

            grad_mag_t = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-6)

            grad_mag = grad_mag_t.squeeze().cpu().numpy()
            delta_hu = delta_hu_t.squeeze().cpu().numpy()

            del t_img, mu_local_t, delta_hu_t, gx, gy, gz, grad_mag_t
            torch.cuda.empty_cache()
            return grad_mag, delta_hu
        except Exception:
            pass

    grad_mag = gaussian_gradient_magnitude(ct_img_ras, sigma=1.0)
    mu_local = gaussian_filter(ct_img_ras, sigma=2.0)
    return grad_mag, ct_img_ras - mu_local


class LocalContrastEdgeBaseline(EmpiricalSpatialPDFBaseline):
    """
    Single-experiment predictor encapsulating Exp 008 local contrast delta
    (HU_voxel - mu_local_5mm) and 3D intensity gradient magnitude ||grad HU|| gating.
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
            heatmap, computing GPU-accelerated 3D intensity gradients and local contrast deltas, gating candidate voxels,
            and applying volume quantile binarization and noise blob pruning.

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

        # 2. Local Contrast Delta & 3D Gradient Edge Gating
        if ct_img_ras is not None and isinstance(ct_img_ras, np.ndarray) and ct_img_ras.shape == target_shape_ras:
            # Body cavity air masking
            body_mask = (ct_img_ras >= -1000.0) & (ct_img_ras <= 1000.0)
            pdf_target[~body_mask] = 0.0

            # Selective HU Windowing
            SELECTIVE_HU_CATEGORIES = {'1a', '1b', '1c', '2f', '2g'}
            if code in SELECTIVE_HU_CATEGORIES:
                min_hu, max_hu = CATEGORY_HU_BOUNDS.get(code, (-1000.0, 300.0))
                invalid_hu = (ct_img_ras < min_hu) | (ct_img_ras > max_hu)
                pdf_target[invalid_hu] = 0.0

            # Fast GPU-accelerated local contrast and gradient computation
            grad_mag, delta_hu = compute_gpu_local_contrast_and_gradient(ct_img_ras)

            # Category-specific local contrast / gradient gating
            if code in {'2d', '2a', '1a', '1b'}:
                edge_mask = (grad_mag >= 10.0) | (delta_hu >= -50.0)
                pdf_target[~edge_mask] *= 0.5
            elif code in {'2e', '2b', '2c'}:
                contrast_mask = (delta_hu >= -150.0) & (delta_hu <= 150.0)
                pdf_target[~contrast_mask] *= 0.7

        max_p = float(pdf_target.max())
        if max_p <= 0:
            return np.zeros(target_shape_ras, dtype=np.uint8)

        # 3. Empirical Volume Quantile Matching Binarization
        target_ratio = EMPIRICAL_VOLUME_QUANTILES.get(code, 0.005)
        cutoff = float(np.quantile(pdf_target, 1.0 - target_ratio))
        if cutoff <= 0:
            cutoff = 0.5 * max_p
        binary_mask = (pdf_target >= cutoff).astype(np.uint8)

        # 4. 3D Connected Component Noise Blob Pruning
        effective_min_blob = self.min_blob_voxels.get(code, 10) if isinstance(self.min_blob_voxels, dict) else self.min_blob_voxels
        if binary_mask.any() and effective_min_blob > 0:
            labeled, num_features = label(binary_mask)
            if num_features > 0:
                sizes = np.bincount(labeled.ravel())
                too_small = sizes < effective_min_blob
                binary_mask[too_small[labeled]] = 0

        return binary_mask.astype(np.uint8)


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for Exp 008 pipeline.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2A Exp 008: Local Contrast Delta & 3D Gradient Edge Gating Baseline Pipeline"
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
    """Main CLI entry point for Exp 008 Pipeline."""
    args = parse_args()

    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else (
        PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat_anchored.npz"
        if (PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat_anchored.npz").exists()
        else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat_anchored.npz"
    )
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_exp_008_local_contrast_edge"
    exp_log_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_008_local_contrast_edge_priors"

    predictor = LocalContrastEdgeBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        img_raw_dir=Path(args.img_raw_dir),
        force_rebuild=args.force_rebuild or not pdf_cache_path.exists(),
        threshold_mode="local_contrast_edge",
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
