import os
import sys
import unittest
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.phase_3_voxtell_training.common import resolve_num_workers, ReXDataset
from scripts.config import DATASET_MINI_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, TEXT_CACHE_DIR

class TestServerAgnosticDataLoader(unittest.TestCase):
    def test_resolve_num_workers_explicit(self):
        self.assertEqual(resolve_num_workers(4), 4)
        self.assertEqual(resolve_num_workers(0), 0)

    def test_resolve_num_workers_slurm(self):
        os.environ["SLURM_CPUS_PER_TASK"] = "16"
        try:
            self.assertEqual(resolve_num_workers(None), 14)
        finally:
            del os.environ["SLURM_CPUS_PER_TASK"]

    def test_resolve_num_workers_default(self):
        # When SLURM is not set, should return a valid integer between 1 and 8
        workers = resolve_num_workers(None)
        self.assertGreaterEqual(workers, 1)
        self.assertLessEqual(workers, 8)

    def test_dataset_initialization_streaming(self):
        if Path(DATASET_MINI_JSON).exists() and Path(RAW_IMAGES_DIR).exists():
            dataset = ReXDataset(
                dataset_json=str(DATASET_MINI_JSON),
                split="train",
                img_dir=str(RAW_IMAGES_DIR),
                seg_dir=str(RAW_MASKS_DIR),
                cache_dir=str(TEXT_CACHE_DIR),
                is_train=True,
                patch_size=192,
                use_volume_cache=False
            )
            self.assertFalse(dataset.use_volume_cache)
            self.assertGreater(len(dataset), 0)

if __name__ == "__main__":
    unittest.main()
