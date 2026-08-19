"""
===============================================================================
UNIT TEST SUITE: Phase 2A Inference & Evaluation Runner Test Suite
LOCATION:       tests/test_phase_2a_runner.py
OBJECTIVE:      Verify end-to-end execution of run_prior_inference_and_eval(),
                checking prediction output shapes, NIfTI headers, and evaluation logs.
===============================================================================
"""

import os
import sys
import tempfile
import unittest
import json
import numpy as np
import nibabel as nib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import CATEGORY_MAP
from scripts.common.orientation import save_nifti, load_nifti_ras
from scripts.phase_2a_rule_based.common.prior_engine import EmpiricalSpatialPDFBaseline
from scripts.phase_2a_rule_based.common.runner import run_prior_inference_and_eval


class TestPhase2ARunner(unittest.TestCase):
    """
    Test suite for Phase 2A runner module run_prior_inference_and_eval().
    """

    def setUp(self):
        """Set up synthetic test fixtures and temporary directories."""
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.test_dir.name)

        self.img_dir = self.tmp_path / "raw_images"
        self.seg_dir = self.tmp_path / "raw_segmentations"
        self.output_dir = self.tmp_path / "predictions"
        self.log_dir = self.tmp_path / "logs"
        self.pdf_cache_path = self.tmp_path / "pdf_cache.npz"
        self.dataset_json_path = self.tmp_path / "dataset.json"

        self.img_dir.mkdir()
        self.seg_dir.mkdir()
        self.output_dir.mkdir()
        self.log_dir.mkdir()

        # Create synthetic CT image (32, 32, 32)
        self.affine = np.eye(4)
        ct_data = np.random.randint(-1000, 1000, size=(32, 32, 32), dtype=np.int16)
        save_nifti(ct_data, self.img_dir / "test_scan_1.nii.gz", affine=self.affine)

        # Create synthetic GT segmentation (2, 32, 32, 32)
        gt_data = np.zeros((2, 32, 32, 32), dtype=np.uint8)
        gt_data[0, 10:20, 10:20, 10:20] = 1
        gt_data[1, 5:15, 5:15, 5:15] = 1
        save_nifti(gt_data, self.seg_dir / "test_scan_1.nii.gz", affine=self.affine)

        # Create dataset.json
        dataset_meta = {
            "train": [
                {
                    "name": "test_scan_1.nii.gz",
                    "findings": {"0": {"text": "nodule"}, "1": {"text": "effusion"}},
                    "categories": {"0": "2d", "1": "2e"}
                }
            ],
            "val": [
                {
                    "name": "test_scan_1.nii.gz",
                    "findings": {"0": {"text": "nodule"}, "1": {"text": "effusion"}},
                    "categories": {"0": "2d", "1": "2e"}
                }
            ]
        }
        with open(self.dataset_json_path, "w") as f:
            json.dump(dataset_meta, f)

        # Pre-build synthetic PDF cache
        synthetic_pdfs = {code: np.full((512, 512, 512), 0.05, dtype=np.float32) for code in CATEGORY_MAP.keys()}
        np.savez_compressed(self.pdf_cache_path, **synthetic_pdfs)

    def tearDown(self):
        """Clean up temporary test fixtures."""
        self.test_dir.cleanup()

    def test_run_prior_inference_and_eval_execution(self):
        """
        Verify end-to-end execution of runner on synthetic dataset.
        """
        predictor = EmpiricalSpatialPDFBaseline(
            pdf_cache_path=self.pdf_cache_path,
            dataset_json_path=self.dataset_json_path,
            seg_raw_dir=self.seg_dir,
            img_raw_dir=self.img_dir,
            threshold_mode="percentile"
        )

        run_prior_inference_and_eval(
            predictor=predictor,
            split="val",
            dataset_json_path=self.dataset_json_path,
            img_raw_dir=self.img_dir,
            seg_raw_dir=self.seg_dir,
            output_dir=self.output_dir,
            exp_log_dir=self.log_dir,
            pdf_cache_path=self.pdf_cache_path,
            do_eval=True
        )

        pred_file = self.output_dir / "test_scan_1.nii.gz"
        self.assertTrue(pred_file.exists(), f"Prediction file missing: {pred_file}")

        pred_data, pred_nii, _ = load_nifti_ras(pred_file)
        self.assertEqual(pred_data.ndim, 4, f"Prediction mask should be 4D, got ndim={pred_data.ndim}")
        self.assertEqual(pred_data.shape[0], 2, f"Prediction mask should have 2 finding channels, got shape {pred_data.shape}")

        eval_json = self.log_dir / "eval_results_val.json"
        self.assertTrue(eval_json.exists(), f"Evaluation JSON missing: {eval_json}")

        eval_md = self.log_dir / "eval.md"
        self.assertTrue(eval_md.exists(), f"Evaluation Markdown log missing: {eval_md}")


if __name__ == "__main__":
    print("=" * 70)
    print("      RUNNING PHASE 2A INFERENCE & EVALUATION RUNNER TEST SUITE")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase2ARunner)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
