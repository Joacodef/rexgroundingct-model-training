"""
===============================================================================
MODULE:         SPOCO 3D Metric Embedding Post-Processing & Instance Clustering
LOCATION:       scripts/phase_4_alternative_models/common/clustering.py
OBJECTIVE:      Convert dense 3D unit-hypersphere metric embeddings into discrete
                binary segmentation masks via anchor seeds and connected-component
                Gaussian soft-mask thresholding, with candidate mask pre-filtering.
===============================================================================
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.ndimage import label, find_objects


def extract_instances_from_embeddings(
    embeddings: np.ndarray,
    delta_var: float = 0.5,
    pmaps_threshold: float = 0.5,
    sigma: Optional[float] = None,
    threshold: float = 0.5,
    min_volume_voxels: int = 10,
    max_instances: int = 15,
    candidate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Signature:
        extract_instances_from_embeddings(embeddings: np.ndarray, delta_var: float = 0.5, pmaps_threshold: float = 0.5, sigma: float | None = None, threshold: float = 0.5, min_volume_voxels: int = 10, max_instances: int = 15, candidate_mask: np.ndarray | None = None) -> np.ndarray

    Objective:
        Cluster dense continuous D-dimensional 3D metric embeddings into a binary
        or multi-instance 3D segmentation mask using iterative seed expansion on the
        unit hypersphere, optionally restricted to a candidate anatomical foreground mask.

    Inputs:
        embeddings (np.ndarray): 4D numpy array of shape (D, Z, Y, X) on unit hypersphere.
        delta_var (float): Intra-cluster pull distance margin (default 0.5).
        pmaps_threshold (float): Kernel cutoff probability at delta_var distance (default 0.5).
        sigma (float | None): Optional legacy sigma override.
        threshold (float): Soft mask binarization cutoff (default 0.5).
        min_volume_voxels (int): Minimum connected component volume in voxels (default 10).
        max_instances (int): Maximum number of object seeds to extract (default 15).
        candidate_mask (np.ndarray | None): Optional 3D boolean/binary mask of shape (Z, Y, X)
            restricting candidate seed locations (e.g. lung envelope or air pre-filter).

    Outputs:
        np.ndarray: Binary 3D mask of shape (Z, Y, X) with value 1 inside segmented instances.
    """
    D, Z, Y, X = embeddings.shape
    combined_binary_mask = np.zeros((Z, Y, X), dtype=np.uint8)

    # Resolve Gaussian variance parameter
    if sigma is not None:
        two_sigma = 2.0 * (sigma ** 2)
    else:
        two_sigma = (delta_var ** 2) / (-math.log(max(1e-7, min(1.0 - 1e-7, pmaps_threshold))))

    # Flatten spatial grid for vectorized distance computation
    flat_embeds = embeddings.reshape(D, -1).T  # (N_voxels, D)

    if candidate_mask is not None:
        unassigned = (candidate_mask.reshape(-1) > 0).astype(bool)
    else:
        unassigned = np.ones(Z * Y * X, dtype=bool)

    for _ in range(max_instances):
        remaining_indices = np.where(unassigned)[0]
        if len(remaining_indices) == 0:
            break

        # Select candidate seed from remaining candidate voxels
        seed_idx = remaining_indices[0]
        seed_vec = flat_embeds[seed_idx]  # (D,)

        # Compute dot product and Euclidean distance on unit sphere: ||e_i - s||^2 = 2 - 2 <e_i, s>
        dot_prods = np.dot(flat_embeds, seed_vec)
        dist_sq = np.maximum(2.0 - 2.0 * dot_prods, 0.0)
        soft_mask = np.exp(-dist_sq / max(1e-8, two_sigma))

        instance_binary = (soft_mask > threshold).reshape(Z, Y, X)
        labeled_components, num_features = label(instance_binary)

        added = False
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled_components == comp_id)
            comp_volume = int(np.sum(comp_mask))
            if comp_volume >= min_volume_voxels:
                combined_binary_mask[comp_mask] = 1
                unassigned[comp_mask.ravel()] = False
                added = True

        if not added:
            # Mark current seed as visited if it yielded no valid components
            unassigned[seed_idx] = False

    return combined_binary_mask
