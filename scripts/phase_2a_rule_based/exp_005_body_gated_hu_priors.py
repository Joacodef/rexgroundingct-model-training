"""
===============================================================================
SCRIPT:         exp_005_body_gated_hu_priors.py
PHASE:          Phase 2A — Statistical / Rule-Based Prior Baseline
LOCATION:       scripts/phase_2a_rule_based/exp_005_body_gated_hu_priors.py
OBJECTIVE:      Single-file executable pipeline for Phase 2A Exp 005.
                1. Checks/builds spatially-anchored 3D empirical PDF heatmaps (ref_affine=img_nii.affine).
                2. Applies Body Cavity Air Masking (HU in [-1000, 1000] HU) to eliminate outside-body room air false positives.
                3. Applies Selective HU Radiodensity Windowing (gating HU bounds ONLY for structural 
                   airway/air pathologies '1a', '1b', '1c', '2f', '2g', bypassing for fluid/diffuse '2e', '2c', '1d').
                4. Applies Empirical Volume Quantile Matching (binarizing at top K voxels matching Phase 1 pathology scale).
                5. Resamples predictions to target scan shapes, stacks 4D NIfTI masks (F, X, Y, Z).
                6. Runs automated challenge metric evaluation (Dice, Hit Rate @ 0.1, Centroid Error).
USAGE:          python scripts/phase_2a_rule_based/exp_005_body_gated_hu_priors.py --split val --eval
===============================================================================
"""

import sys
import argparse
from pathlib import Path

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import (
    DATA_DIR, DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    PREDICTIONS_DIR, LOGS_DIR
)
from scripts.phase_2a_rule_based.common import EmpiricalSpatialPDFBaseline, run_prior_inference_and_eval


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for Exp 005 baseline pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2A Exp 005: Body-Gated Selective HU Windowing + Quantile Spatial Prior Baseline Pipeline"
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: val)"
    )
    parser.add_argument(
        "--pdf_cache", type=str, default=None,
        help="Path to empirical spatial PDF npz cache (defaults to data/phase_2a/empirical_spatial_pdf_14cat_anchored.npz)"
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
    """Main CLI entry point for Exp 005 Pipeline."""
    args = parse_args()

    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat_anchored.npz"
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_exp_005_body_gated_hu"
    exp_log_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_005_body_gated_hu_priors"

    # Initialize Predictor Engine (Body-Gated Selective HU Windowing + Volume Quantile Matching Mode)
    predictor = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        img_raw_dir=Path(args.img_raw_dir),
        force_rebuild=args.force_rebuild or not pdf_cache_path.exists(),
        threshold_mode="body_gated_hu",
        min_blob_voxels=args.min_blob_voxels,
    )

    # Delegate to shared runner
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
