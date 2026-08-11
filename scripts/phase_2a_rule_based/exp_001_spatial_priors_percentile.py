"""
===============================================================================
SCRIPT:         exp_001_spatial_priors_percentile.py
PHASE:          Phase 2A — Statistical / Rule-Based Prior Baseline
LOCATION:       scripts/phase_2a_rule_based/exp_001_spatial_priors_percentile.py
OBJECTIVE:      Single-file executable pipeline for Phase 2A Exp 001.
                1. Checks/builds 3D empirical PDF heatmaps P_c(z, y, x) in canonical space.
                2. Applies category-calibrated percentile factor thresholding (p_c = factor * max_p).
                3. Resamples predictions to target scan shapes, stacks 4D NIfTI masks (F, X, Y, Z).
                4. Runs automated challenge metric evaluation (Dice, Hit Rate @ 0.1, Centroid Error).
USAGE:          python scripts/phase_2a_rule_based/exp_001_spatial_priors_percentile.py --split val --eval
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
    PREDICTIONS_DIR, LOGS_DIR, PHASE_2A_PDFS_DIR
)
from scripts.phase_2a_rule_based.common import EmpiricalSpatialPDFBaseline, run_prior_inference_and_eval

# Calibrated category threshold factors for Exp 001 (p_c = factor * max_p).
# DERIVATION: Derived from Phase 1 empirical cumulative density profiling and morphological sphericity.
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



def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for Exp 001 baseline pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Phase 2A Exp 001: Percentile Factor Spatial Prior Baseline Pipeline"
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
        "--start_idx", type=int, default=0, help="Start index for processing entries"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None, help="End index for processing entries"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for Exp 001 Pipeline."""
    args = parse_args()

    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else (
        PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat.npz"
        if (PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat.npz").exists()
        else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat.npz"
    )
    output_dir = Path(args.output_dir) if args.output_dir else PREDICTIONS_DIR / "phase_2a_exp_001_percentile"
    exp_log_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_001_spatial_priors_percentile"

    # Initialize Predictor Engine (Percentile Threshold Mode)
    predictor = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        img_raw_dir=Path(args.img_raw_dir),
        force_rebuild=args.force_rebuild,
        threshold_mode="percentile",
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
