"""
===============================================================================
TEST SUITE:     Empirical Spatial PDF Baseline Engine Test Suite
LOCATION:       tests/test_prior_engine.py
OBJECTIVE:      Automated unit and integration test suite testing
                scripts/phase_2a_rule_based/exp_001_seg_masks_priors/prior_engine.py
                covering cache building, loading, resampling, binarization,
                component pruning, and threshold mapping.
===============================================================================
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

try:
    import pytest
except ImportError:
    pytest = None


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import CATEGORY_MAP
from scripts.phase_2a_rule_based.common.prior_engine import (
    EmpiricalSpatialPDFBaseline,
    CATEGORY_THRESHOLD_FACTORS,
)


def create_synthetic_cache(cache_path: Path, canonical_shape: tuple = (512, 512, 512)) -> dict:
    """
    Signature:
        create_synthetic_cache(cache_path: Path, canonical_shape: tuple) -> dict

    Objective:
        Helper function to create a dummy .npz PDF cache for testing loading and generation.

    Inputs:
        cache_path (Path): Destination file path for .npz cache.
        canonical_shape (tuple): Spatial 3D shape for synthetic heatmaps. Default (512, 512, 512).

    Outputs:
        dict: Synthetic dictionary mapping category codes to 3D float32 numpy arrays.
    """
    synthetic_pdfs = {}
    for code in CATEGORY_MAP.keys():
        # Create a small high-probability central gaussian blob
        grid = np.zeros((32, 32, 32), dtype=np.float32)
        grid[12:20, 12:20, 12:20] = 0.8
        # Resize to canonical shape if needed, or small shape for fast test
        synthetic_pdfs[code] = grid

    np.savez_compressed(cache_path, **synthetic_pdfs)
    return synthetic_pdfs


def test_category_threshold_factors_complete():
    """
    Signature:
        test_category_threshold_factors_complete() -> None

    Objective:
        Verify CATEGORY_THRESHOLD_FACTORS contains all 14 categories with valid float bounds [0, 1].

    Inputs:
        None

    Outputs:
        None
    """
    for code in CATEGORY_MAP.keys():
        assert code in CATEGORY_THRESHOLD_FACTORS, f"Category code '{code}' missing from CATEGORY_THRESHOLD_FACTORS"
        factor = CATEGORY_THRESHOLD_FACTORS[code]
        assert isinstance(factor, (float, int)), f"Factor for '{code}' must be float/int"
        assert 0.0 <= factor <= 1.0, f"Factor for '{code}' must be in range [0.0, 1.0], got {factor}"


def test_load_existing_pdf_cache(tmp_path: Path):
    """
    Signature:
        test_load_existing_pdf_cache(tmp_path: Path) -> None

    Objective:
        Test loading precomputed 3D spatial PDF heatmaps from an existing .npz file without rebuilding.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "synthetic_pdf_cache.npz"
    create_synthetic_cache(cache_file)

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=tmp_path / "dataset.json",
        seg_raw_dir=tmp_path / "raw_masks",
        force_rebuild=False,
    )

    assert hasattr(engine, "spatial_pdfs")
    assert len(engine.spatial_pdfs) == 14
    for code in CATEGORY_MAP.keys():
        assert code in engine.spatial_pdfs
        assert engine.spatial_pdfs[code].shape == (32, 32, 32)


def test_build_pdf_cache_from_synthetic_dataset(tmp_path: Path):
    """
    Signature:
        test_build_pdf_cache_from_synthetic_dataset(tmp_path: Path) -> None

    Objective:
        Test building PDF cache from synthetic dataset entries and ground truth NIfTI masks.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "built_pdf_cache.npz"
    seg_dir = tmp_path / "raw_masks"
    seg_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp_path / "raw_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    import os
    import scripts.config
    os.environ["IMG_RAW_DIR"] = str(img_dir)
    scripts.config.RAW_IMAGES_DIR = img_dir

    # Create dummy parent CT scan (3D volume: 32, 32, 32)
    parent_ct = np.zeros((32, 32, 32), dtype=np.float32)
    parent_nii = nib.Nifti1Image(parent_ct, np.eye(4))
    nib.save(parent_nii, str(img_dir / "scan_001.nii.gz"))

    # Create dummy GT mask (F=2, X=32, Y=32, Z=32)
    gt_mask = np.zeros((2, 32, 32, 32), dtype=np.uint8)
    gt_mask[0, 10:20, 10:20, 10:20] = 1  # Category 1a
    gt_mask[1, 5:15, 5:15, 5:15] = 1     # Category 2d

    nii = nib.Nifti1Image(gt_mask, np.eye(4))
    nib.save(nii, str(seg_dir / "scan_001.nii.gz"))

    dataset_json_content = {
        "train": [
            {
                "name": "scan_001.nii.gz",
                "categories": {"0": "1a", "1": "2d"}
            }
        ]
    }
    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json_content, f)

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=json_path,
        seg_raw_dir=seg_dir,
        max_train_scans=1,
        force_rebuild=True,
    )

    assert cache_file.exists()
    assert "1a" in engine.spatial_pdfs
    assert "2d" in engine.spatial_pdfs
    # PDF values should be normalized float32
    assert engine.spatial_pdfs["1a"].max() > 0.0


def test_build_pdf_cache_invalid_dataset_json(tmp_path: Path):
    """
    Signature:
        test_build_pdf_cache_invalid_dataset_json(tmp_path: Path) -> None

    Objective:
        Verify ValueError is raised cleanly when dataset.json lacks 'train' entries during cache building.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "invalid_cache.npz"
    json_path = tmp_path / "empty_dataset.json"
    with open(json_path, "w") as f:
        json.dump({"train": []}, f)

    raised = False
    try:
        EmpiricalSpatialPDFBaseline(
            pdf_cache_path=cache_file,
            dataset_json_path=json_path,
            seg_raw_dir=tmp_path / "raw_masks",
            force_rebuild=True,
        )
    except ValueError as e:
        raised = True
        assert "No 'train' entries found" in str(e)

    assert raised, "Expected ValueError for empty train entries was not raised"


def test_generate_prediction_mask_resampling_and_pruning(tmp_path: Path):
    """
    Signature:
        test_generate_prediction_mask_resampling_and_pruning(tmp_path: Path) -> None

    Objective:
        Test generate_prediction_mask for shape resampling, thresholding, and small component pruning.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "synthetic_pdf_cache.npz"
    create_synthetic_cache(cache_file)

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=tmp_path / "dataset.json",
        seg_raw_dir=tmp_path / "raw_masks",
        force_rebuild=False,
    )

    # Test generation to a target shape (e.g. 64, 64, 64)
    target_shape = (64, 64, 64)
    pred_mask = engine.generate_prediction_mask(target_shape_ras=target_shape, cat_code="2d")

    assert pred_mask.shape == target_shape
    assert pred_mask.dtype == np.uint8
    assert set(np.unique(pred_mask)).issubset({0, 1})
    # Central region should be 1
    assert pred_mask.sum() > 0


def test_generate_prediction_mask_noise_pruning(tmp_path: Path):
    """
    Signature:
        test_generate_prediction_mask_noise_pruning(tmp_path: Path) -> None

    Objective:
        Test component size cleanup removes small isolated blobs containing fewer than 10 voxels.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "noise_pdf_cache.npz"
    grid = np.zeros((32, 32, 32), dtype=np.float32)
    # Add a large blob (> 10 voxels)
    grid[10:15, 10:15, 10:15] = 1.0
    # Add a tiny noise blob (< 10 voxels, e.g. 2x2x2 = 8 voxels)
    grid[0:2, 0:2, 0:2] = 1.0

    np.savez_compressed(cache_file, **{"2h": grid})

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=tmp_path / "dataset.json",
        seg_raw_dir=tmp_path / "raw_masks",
        force_rebuild=False,
    )

    pred_mask = engine.generate_prediction_mask(target_shape_ras=(32, 32, 32), cat_code="2h")

    # Tiny blob at [0:2, 0:2, 0:2] should be pruned (all zero)
    assert np.all(pred_mask[0:2, 0:2, 0:2] == 0)
    # Large blob at [10:15, 10:15, 10:15] should remain preserved
    assert pred_mask[10:15, 10:15, 10:15].sum() > 0


def test_generate_prediction_mask_unknown_category_fallback(tmp_path: Path):
    """
    Signature:
        test_generate_prediction_mask_unknown_category_fallback(tmp_path: Path) -> None

    Objective:
        Test fallback behavior when an invalid category code is supplied to generate_prediction_mask.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "synthetic_pdf_cache.npz"
    create_synthetic_cache(cache_file)

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=tmp_path / "dataset.json",
        seg_raw_dir=tmp_path / "raw_masks",
        force_rebuild=False,
    )

    # Pass unknown category code
    pred_mask = engine.generate_prediction_mask(target_shape_ras=(32, 32, 32), cat_code="invalid_code")
    assert pred_mask.shape == (32, 32, 32)
    assert pred_mask.dtype == np.uint8


def test_generate_prediction_mask_all_zero_pdf(tmp_path: Path):
    """
    Signature:
        test_generate_prediction_mask_all_zero_pdf(tmp_path: Path) -> None

    Objective:
        Verify generate_prediction_mask returns zero array when max probability density is 0.

    Inputs:
        tmp_path (Path): Pytest/tempfile temporary directory fixture.

    Outputs:
        None
    """
    cache_file = tmp_path / "zero_pdf_cache.npz"
    grid = np.zeros((32, 32, 32), dtype=np.float32)
    np.savez_compressed(cache_file, **{"1c": grid})

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=tmp_path / "dataset.json",
        seg_raw_dir=tmp_path / "raw_masks",
        force_rebuild=False,
    )

    pred_mask = engine.generate_prediction_mask(target_shape_ras=(32, 32, 32), cat_code="1c")
    assert pred_mask.shape == (32, 32, 32)
    assert np.all(pred_mask == 0)


def test_generate_prediction_mask_hu_windowing(tmp_path: Path):
    """
    Signature:
        test_generate_prediction_mask_hu_windowing(tmp_path: Path) -> None

    Objective:
        Verify generate_prediction_mask zeroes out voxels outside category HU radiodensity bounds.
    """
    cache_file = tmp_path / "hu_test_cache.npz"
    grid = np.ones((32, 32, 32), dtype=np.float32)
    np.savez_compressed(cache_file, **{"1c": grid})

    engine = EmpiricalSpatialPDFBaseline(
        pdf_cache_path=cache_file,
        dataset_json_path=tmp_path / "dataset.json",
        seg_raw_dir=tmp_path / "raw_masks",
        force_rebuild=False,
        threshold_mode="hu_quantile",
    )

    # Synthetic CT image: Half valid HU (-500 HU), Half invalid HU (+1000 HU for Emphysema '1c' whose max is +252 HU)
    ct_img = np.full((32, 32, 32), -500.0, dtype=np.float32)
    ct_img[:, :, 16:] = 1000.0  # Invalid HU region for 1c

    pred_mask = engine.generate_prediction_mask(cat_code="1c", target_shape_ras=(32, 32, 32), ct_img_ras=ct_img)
    assert pred_mask.shape == (32, 32, 32)
    # The invalid HU region (Z >= 16) must be completely zeroed out
    assert np.all(pred_mask[:, :, 16:] == 0)


if __name__ == "__main__":
    print("=" * 70)
    print("      RUNNING EMPIRICAL SPATIAL PDF BASELINE ENGINE TEST SUITE")
    print("=" * 70)

    test_functions = [
        test_category_threshold_factors_complete,
        test_load_existing_pdf_cache,
        test_build_pdf_cache_from_synthetic_dataset,
        test_build_pdf_cache_invalid_dataset_json,
        test_generate_prediction_mask_resampling_and_pruning,
        test_generate_prediction_mask_noise_pruning,
        test_generate_prediction_mask_unknown_category_fallback,
        test_generate_prediction_mask_all_zero_pdf,
        test_generate_prediction_mask_hu_windowing,
    ]

    passed = 0
    failed = 0

    for test_fn in test_functions:
        test_name = test_fn.__name__
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                if test_name == "test_category_threshold_factors_complete":
                    test_fn()
                else:
                    test_fn(tmp_path)
            print(f"  [PASS] {test_name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_name}: {e}")
            failed += 1

    print("=" * 70)
    print(f"TEST SUMMARY: {passed} Passed, {failed} Failed")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)
