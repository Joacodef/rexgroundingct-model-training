"""
===============================================================================
MODULE:         SPOCO 3D Metric Embedding Post-Processing & Instance Clustering
LOCATION:       scripts/phase_4_alternative_models/common/clustering.py
OBJECTIVE:      Convert dense 3D unit-hypersphere metric embeddings into discrete
                binary segmentation masks via anchor seeds and connected-component
                Gaussian soft-mask thresholding.
===============================================================================
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label, find_objects


def extract_instances_from_embeddings(
    embeddings: np.ndarray,
    sigma: float = 0.5,
    threshold: float = 0.5,
    min_volume_voxels: int = 10,
    max_instances: int = 15,
) -> np.ndarray:
    """
    Signature:
        extract_instances_from_embeddings(embeddings: np.ndarray, sigma: float = 0.5, threshold: float = 0.5, min_volume_voxels: int = 10, max_instances: int = 15) -> np.ndarray

    Objective:
        Cluster dense continuous D-dimensional 3D metric embeddings into a binary
        or multi-instance 3D segmentation mask using iterative seed expansion.

    Inputs:
        embeddings (np.ndarray): 4D numpy array of shape (D, Z, Y, X) on unit hypersphere.
        sigma (float): Gaussian bandwidth scaling factor. Default 0.5.
        threshold (float): Soft mask binarization cutoff. Default 0.5.
        min_volume_voxels (int): Minimum connected component volume in voxels. Default 10.
        max_instances (int): Maximum number of object seeds to extract. Default 15.

    Outputs:
        np.ndarray: Binary 3D mask of shape (Z, Y, X) with value 1 inside segmented instances.
    """
    D, Z, Y, X = embeddings.shape
    combined_binary_mask = np.zeros((Z, Y, X), dtype=np.uint8)

    # Flatten spatial grid for vectorized distance computation
    flat_embeds = embeddings.reshape(D, -1).T  # (N_voxels, D)
    unassigned = np.ones(Z * Y * X, dtype=bool)

    for _ in range(max_instances):
        remaining_indices = np.where(unassigned)[0]
        if len(remaining_indices) == 0:
            break

        # Select candidate seed from remaining voxels
        seed_idx = remaining_indices[0]
        seed_vec = flat_embeds[seed_idx]  # (D,)

        # Compute dot product and Euclidean distance on unit sphere
        dot_prods = np.dot(flat_embeds, seed_vec)
        dist_sq = np.maximum(2.0 - 2.0 * dot_prods, 0.0)
        soft_mask = np.exp(-dist_sq / (2.0 * (sigma ** 2)))

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

