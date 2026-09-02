"""
===============================================================================
MODULE:         SPOCO 3D Metric Embedding Post-Processing & Instance Clustering
LOCATION:       scripts/phase_4_voxtell_spoco/common/clustering.py
OBJECTIVE:      Convert dense 3D unit-hypersphere metric embeddings into discrete
                binary segmentation masks via anchor seeds and connected-component
                Gaussian soft-mask thresholding, with candidate mask pre-filtering.
===============================================================================
"""

import logging
import math
from typing import List, Optional, Tuple
import numpy as np
from scipy.ndimage import label

logger = logging.getLogger("voxtell_spoco_clustering")


def _resolve_two_sigma(delta_var: float, pmaps_threshold: float, sigma: Optional[float]) -> float:
    """
    Signature:
        _resolve_two_sigma(delta_var: float, pmaps_threshold: float, sigma: float | None) -> float

    Objective:
        Resolve the Gaussian kernel variance parameter two_sigma either from an explicit
        legacy sigma (two_sigma = 2 sigma^2) or from the calibrated form
        two_sigma = delta_var^2 / -ln(pmaps_threshold), so a voxel at embedding distance
        delta_var from the anchor evaluates to exactly pmaps_threshold.

    Inputs:
        delta_var (float): Intra-cluster pull margin.
        pmaps_threshold (float): Soft-mask probability at distance delta_var.
        sigma (float | None): Optional explicit bandwidth override.

    Outputs:
        float: two_sigma value for the Gaussian soft-mask kernel.
    """
    if sigma is not None:
        return 2.0 * (sigma ** 2)
    return (delta_var ** 2) / (-math.log(max(1e-7, min(1.0 - 1e-7, pmaps_threshold))))


def extract_instances_from_embeddings(
    embeddings: np.ndarray,
    delta_var: float = 0.5,
    pmaps_threshold: float = 0.5,
    sigma: Optional[float] = None,
    threshold: float = 0.5,
    min_volume_voxels: int = 10,
    max_instances: int = 15,
    candidate_mask: Optional[np.ndarray] = None,
    seed_coords: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """
    Signature:
        extract_instances_from_embeddings(embeddings: np.ndarray, delta_var: float = 0.5, pmaps_threshold: float = 0.5, sigma: float | None = None, threshold: float = 0.5, min_volume_voxels: int = 10, max_instances: int = 15, candidate_mask: np.ndarray | None = None, seed_coords: list[tuple[int, int, int]] | None = None) -> np.ndarray

    Objective:
        Cluster dense continuous 32D 3D metric embeddings into a binary 3D segmentation
        mask via Gaussian soft-mask seed expansion on the unit hypersphere. When
        seed_coords are provided (e.g. the argmax of the text-query logit map), each
        seed contributes the single connected component that contains it; otherwise the
        function falls back to deterministic first-unassigned-voxel seeding (which can
        seed on background and is only a last resort). Distance evaluation is restricted
        to candidate_mask voxels when given.

    Inputs:
        embeddings (np.ndarray): 4D array of shape (D, Z, Y, X) on the unit hypersphere.
        delta_var (float): Intra-cluster pull distance margin (default 0.5).
        pmaps_threshold (float): Kernel cutoff probability at delta_var distance (default 0.5).
        sigma (float | None): Optional legacy sigma override.
        threshold (float): Soft mask binarization cutoff (default 0.5).
        min_volume_voxels (int): Minimum connected component volume in voxels (default 10).
        max_instances (int): Maximum number of seeds to expand (default 15). Ignored when
            seed_coords is provided (all supplied seeds are used).
        candidate_mask (np.ndarray | None): Optional 3D boolean/binary mask of shape
            (Z, Y, X) restricting candidate voxels (e.g. lung envelope / air pre-filter).
        seed_coords (list[tuple[int, int, int]] | None): Optional explicit (z, y, x) seed
            voxels. Each yields the connected component containing it.

    Outputs:
        np.ndarray: Binary 3D mask of shape (Z, Y, X), value 1 inside segmented instances.
    """
    D, Z, Y, X = embeddings.shape
    combined_binary_mask = np.zeros((Z, Y, X), dtype=np.uint8)
    two_sigma = _resolve_two_sigma(delta_var, pmaps_threshold, sigma)

    flat_embeds = embeddings.reshape(D, -1).T  # (N_voxels, D)
    n_voxels = Z * Y * X

    if candidate_mask is not None:
        cand_flat = (candidate_mask.reshape(-1) > 0)
    else:
        cand_flat = np.ones(n_voxels, dtype=bool)
    cand_idx = np.where(cand_flat)[0]
    if cand_idx.size == 0:
        return combined_binary_mask
    cand_embeds = flat_embeds[cand_idx]  # (N_cand, D) -- the only voxels we score

    def _soft_mask_from_seed(seed_flat_idx: int) -> np.ndarray:
        """Return the (Z, Y, X) Gaussian soft mask for one seed, scored on candidate voxels only."""
        seed_vec = flat_embeds[seed_flat_idx]
        dist_sq = np.maximum(2.0 - 2.0 * (cand_embeds @ seed_vec), 0.0)
        soft_vals = np.exp(-dist_sq / max(1e-8, two_sigma))
        full = np.zeros(n_voxels, dtype=np.float32)
        full[cand_idx] = soft_vals
        return full.reshape(Z, Y, X)

    # -- Text-conditioned seeding: one connected component per supplied seed ----------
    if seed_coords:
        for (sz, sy, sx) in seed_coords:
            if not (0 <= sz < Z and 0 <= sy < Y and 0 <= sx < X):
                continue
            soft_mask = _soft_mask_from_seed(sz * Y * X + sy * X + sx)
            labeled_components, num_features = label(soft_mask > threshold)
            if num_features == 0:
                continue
            comp_id = int(labeled_components[sz, sy, sx])
            if comp_id == 0:
                continue
            comp_mask = (labeled_components == comp_id)
            if int(comp_mask.sum()) >= min_volume_voxels:
                combined_binary_mask[comp_mask] = 1
        return combined_binary_mask

    # -- Fallback: deterministic first-unassigned-voxel seeding (may seed on background)
    logger.warning(
        "extract_instances_from_embeddings called without seed_coords; falling back to "
        "first-unassigned-voxel seeding, which is not text-conditioned and can seed on background."
    )
    unassigned = cand_flat.copy()
    for _ in range(max_instances):
        remaining_indices = np.where(unassigned)[0]
        if len(remaining_indices) == 0:
            break
        seed_idx = int(remaining_indices[0])
        soft_mask = _soft_mask_from_seed(seed_idx)
        labeled_components, num_features = label(soft_mask > threshold)

        added = False
        for comp_id in range(1, num_features + 1):
            comp_mask = (labeled_components == comp_id)
            if int(comp_mask.sum()) >= min_volume_voxels:
                combined_binary_mask[comp_mask] = 1
                unassigned[comp_mask.ravel()] = False
                added = True
        if not added:
            unassigned[seed_idx] = False

    return combined_binary_mask
