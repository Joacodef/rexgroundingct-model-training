"""
===============================================================================
TEST SUITE:     Prediction Post-Processing & Connected-Component Pruning
LOCATION:       tests/test_postprocess.py
OBJECTIVE:      Verify 3D connected component noise filtering and thresholding 
                logic in scripts/analysis/postprocess_predictions.py.
===============================================================================
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.analysis.postprocess_predictions import prune_small_components_3d


def test_prune_small_components_empty():
    """Test that empty mask is returned as-is with zero components."""
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    pruned = prune_small_components_3d(mask, min_voxels=10)
    assert np.all(pruned == 0)
    assert pruned.shape == (20, 20, 20)


def test_prune_small_components_disabled():
    """Test that min_voxels <= 0 disables filtering."""
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[5, 5, 5] = 1 # 1-voxel component
    pruned = prune_small_components_3d(mask, min_voxels=0)
    assert np.sum(pruned) == 1


def test_prune_small_components_filtering():
    """Test that components smaller than min_voxels are removed while larger ones are preserved."""
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    
    # Create small 2-voxel noise blob at (2, 2, 2)
    mask[2, 2, 2] = 1
    mask[2, 2, 3] = 1
    
    # Create large 3x3x3 = 27-voxel lesion at (15..17, 15..17, 15..17)
    mask[15:18, 15:18, 15:18] = 1
    
    assert np.sum(mask) == 2 + 27
    
    pruned = prune_small_components_3d(mask, min_voxels=10)
    
    # Small 2-voxel blob should be pruned, 27-voxel lesion preserved
    assert np.sum(pruned) == 27
    assert np.all(pruned[2:3, 2:3, 2:4] == 0)
    assert np.all(pruned[15:18, 15:18, 15:18] == 1)
