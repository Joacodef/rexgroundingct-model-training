"""
===============================================================================
SCRIPT:         01_build_spatial_pdf_cache.py
PHASE:          Phase 2A — Statistical / Rule-Based Prior Baseline
OBJECTIVE:      Accumulate ground-truth training segmentations per category in 
                canonical RAS space (192x192x192) to build and cache 3D 
                empirical spatial probability density heatmaps P_c(z, y, x).
USAGE:          python scripts/phase_2a_rule_based/exp_001_seg_masks_priors/01_build_spatial_pdf_cache.py
===============================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import DATA_DIR, DATASET_JSON, RAW_MASKS_DIR
from scripts.phase_2a_rule_based.exp_001_seg_masks_priors.prior_engine import EmpiricalSpatialPDFBaseline


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for building 3D empirical spatial PDF heatmaps cache.
    """
    parser = argparse.ArgumentParser(
        description="Task 1: Build 3D Empirical Spatial Probability Density Heatmaps Cache"
    )
    parser.add_argument(
        "--pdf_cache", type=str, default=None,
        help="Output path for empirical spatial PDF npz cache (defaults to data/phase_2a/empirical_spatial_pdf_14cat.npz)"
    )
    parser.add_argument(
        "--dataset_json", type=str, default=str(DATASET_JSON),
        help="Path to dataset.json"
    )
    parser.add_argument(
        "--seg_raw_dir", type=str, default=str(RAW_MASKS_DIR),
        help="Path to raw CT segmentations directory"
    )
    parser.add_argument(
        "--max_train_scans", type=int, default=300,
        help="Maximum training scans to sample for 3D empirical PDF heatmap building (default: 300)"
    )
    parser.add_argument(
        "--force_rebuild", action="store_true", default=False,
        help="Force rebuild of 3D spatial probability density heatmaps even if cache file exists"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for Task 1: Building 3D Spatial PDF Cache."""
    args = parse_args()

    pdf_cache_path = Path(args.pdf_cache) if args.pdf_cache else DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat.npz"

    print("=" * 80)
    print("Phase 2A — Task 1: Build 3D Empirical Spatial PDF Cache")
    print(f"PDF Cache Path:      {pdf_cache_path}")
    print(f"Dataset JSON:        {args.dataset_json}")
    print(f"Max Train Scans:     {args.max_train_scans}")
    print(f"Force Rebuild:       {args.force_rebuild}")
    print("=" * 80)

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=pdf_cache_path,
        dataset_json_path=Path(args.dataset_json),
        seg_raw_dir=Path(args.seg_raw_dir),
        max_train_scans=args.max_train_scans,
        force_rebuild=args.force_rebuild or not pdf_cache_path.exists()
    )

    print(f"\n[SUCCESS] 3D Empirical Spatial PDF cache is ready at: {pdf_cache_path}")


if __name__ == "__main__":
    main()
