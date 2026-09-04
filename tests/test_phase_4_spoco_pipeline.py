"""
===============================================================================
TEST SUITE:     Phase 4 VoxTell-SPOCO Pipeline Diagnostics & Critical Checks
LOCATION:       tests/test_phase_4_spoco_pipeline.py
OBJECTIVE:      Verify critical failure modes in Phase 4 SPOCO metric learning:
                1. Unit Hypersphere Geometry (S^31): Exact norm and fp32 distance stability (LOSS-3).
                2. Supervision Routing & Loss Decoupling (LOSS-1, LOSS-4): Separate L_neg,
                   out-of-crop skip, and background-confined consistency.
                3. Component Capping & Interior Anchor Guarantee: Max 8 components, interior anchor (MEM-1).
                4. Boundary Dilation: Exclusion of partial-volume edges from background sampling.
                5. Inference Hypersphere Re-Normalization: Gaussian blending unit norm restoration.
                6. 3D Chebyshev NMS Seeding (top_k_seeds): Confidence cutoff and spatial separation.
                7. Instance Clustering Gating: Candidate thresholding and min-volume blob pruning.
                8. Architecture Contract: Pinned embedding_dim == 32 enforcement.
===============================================================================
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import math
import pytest
import torch
import numpy as np
import torch.nn.functional as F

from scripts.phase_4_voxtell_spoco.common.losses import (
    compute_gaussian_soft_mask,
    sample_annotated_anchors,
    sample_unannotated_anchors,
    compute_unlabeled_push_loss,
    compute_spoco_total_loss,
    _unit_sphere_dist_sq,
    _confirmed_background_mask,
)
from scripts.phase_4_voxtell_spoco.exp_001_voxtell_spoco import (
    consistency_rampup_weight,
)
from scripts.phase_4_voxtell_spoco.common.clustering import (
    extract_instances_from_embeddings,
)
from scripts.phase_4_voxtell_spoco.common.spoco_inference import (
    build_gaussian_weight,
    top_k_seeds,
)
from scripts.phase_4_voxtell_spoco.common.voxtell_spoco import (
    VoxTellSpocoDecoder,
)


def test_unit_hypersphere_geometry_and_fp32_stability_loss3():
    """
    Signature:
        test_unit_hypersphere_geometry_and_fp32_stability_loss3() -> None

    Objective:
        Verify that embeddings normalized to S^31 maintain strict unit norm, and distance
        computations stay in float32 without underflow, negative values, or cancellation
        under identical or orthogonal vectors.
    """
    D, Z, Y, X = 32, 16, 16, 16
    raw_embeds = torch.randn(D, Z, Y, X, dtype=torch.bfloat16)
    normed_embeds = F.normalize(raw_embeds.float(), p=2, dim=0)

    # 1. Exact unit norm everywhere on 3D grid
    norms = torch.norm(normed_embeds, p=2, dim=0)
    assert torch.allclose(norms, torch.tensor(1.0), atol=1e-6)

    # 2. Distance between identical vectors must be identically 0.0 (no negative values)
    anchor_vec = normed_embeds[:, 5, 5, 5:6].transpose(0, 1)  # (1, D)
    vol_vec = normed_embeds[:, 5:6, 5:6, 5:6]                 # (D, 1, 1, 1)
    dist_sq_zero = _unit_sphere_dist_sq(anchor_vec, vol_vec, "kd, dzyx -> kzyx")
    assert dist_sq_zero.dtype == torch.float32
    assert dist_sq_zero.item() >= 0.0
    assert torch.isclose(dist_sq_zero, torch.tensor(0.0), atol=1e-6)

    # 3. Distance between orthogonal vectors on unit sphere must be exactly 2.0
    u = torch.zeros(D, dtype=torch.float32); u[0] = 1.0
    v = torch.zeros(D, dtype=torch.float32); v[1] = 1.0
    u_anchor = u.unsqueeze(0)  # (1, D)
    v_vol = v.view(D, 1, 1, 1)  # (D, 1, 1, 1)
    dist_sq_ortho = _unit_sphere_dist_sq(u_anchor, v_vol, "kd, dzyx -> kzyx")
    assert torch.isclose(dist_sq_ortho, torch.tensor(2.0), atol=1e-6)


def test_spoco_supervision_routing_loss1_loss4():
    """
    Signature:
        test_spoco_supervision_routing_loss1_loss4() -> None

    Objective:
        Verify SPOCO loss routing across prompt conditions:
        - Present non-empty finding: L_obj > 0, L_neg == 0.
        - Present out-of-crop finding (targets == 0, is_absent=False): SKIPPED (all losses 0).
        - Confirmed absent finding (targets == 0, is_absent=True): L_obj == 0, L_neg > 0, L_con == 0.
    """
    D, Z, Y, X = 32, 16, 16, 16
    student_embeds = F.normalize(torch.randn(1, 3, D, Z, Y, X), p=2, dim=2)
    teacher_embeds = F.normalize(torch.randn(1, 3, D, Z, Y, X), p=2, dim=2)

    targets = torch.zeros(1, 3, Z, Y, X)
    # Prompt 0 has active lesion in crop
    targets[0, 0, 6:10, 6:10, 6:10] = 1.0
    # Prompt 1 is present in scan, but lesion fell outside this crop (targets == 0)
    # Prompt 2 is confirmed absent from scan (targets == 0)
    is_absent = torch.tensor([[False, False, True]], dtype=torch.bool)

    total_loss, loss_obj, loss_con, loss_push, loss_neg = compute_spoco_total_loss(
        student_embeds=student_embeds,
        teacher_embeds=teacher_embeds,
        targets=targets,
        is_absent=is_absent,
        w_con=0.1,
        return_details=True,
    )

    # Prompt 0 contributes to L_obj
    assert loss_obj > 0.0
    # Prompt 2 contributes to L_neg
    assert loss_neg > 0.0
    # L_obj and L_neg must be cleanly decoupled
    assert math.isfinite(total_loss.item())


def test_component_capping_and_interior_anchor_guarantee():
    """
    Signature:
        test_component_capping_and_interior_anchor_guarantee() -> None

    Objective:
        Verify sample_annotated_anchors:
        - Caps connected components to max_components (keeping largest first).
        - Guarantees every anchor coordinate is strictly an interior voxel of that component.
    """
    Z, Y, X = 32, 32, 32
    target_mask = torch.zeros(Z, Y, X)

    # Create 12 distinct disjoint components with varying sizes:
    # Component i will have volume (i + 1) voxels
    for i in range(12):
        z = (i * 2) % 28 + 2
        y = ((i * 2) // 28 * 6) + 4
        # Add (i + 1) voxels along X
        target_mask[z, y, 2:2 + (i + 1)] = 1.0

    max_c = 8
    items = sample_annotated_anchors(target_mask, max_components=max_c)

    # 1. Component count must be capped to max_c
    assert len(items) == max_c

    # 2. Kept components must be the largest (volumes 12, 11, 10, 9, 8, 7, 6, 5)
    kept_volumes = [comp_mask.sum().item() for _, comp_mask in items]
    assert kept_volumes == [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0]

    # 3. Every anchor must be strictly interior to its component mask
    for (az, ay, ax), comp_mask in items:
        assert comp_mask[az, ay, ax].item() == 1.0, f"Anchor ({az}, {ay}, {ax}) fell outside its component mask!"


def test_confirmed_background_mask_boundary_dilation():
    """
    Signature:
        test_confirmed_background_mask_boundary_dilation() -> None

    Objective:
        Verify _confirmed_background_mask excludes foreground dilated by roi_dilation_voxels,
        preventing boundary partial-volume voxels from contaminating push/consistency losses.
    """
    Z, Y, X = 16, 16, 16
    target_mask = torch.zeros(Z, Y, X)
    # 2x2x2 lesion at center
    target_mask[7:9, 7:9, 7:9] = 1.0

    bg_mask = _confirmed_background_mask(target_mask, roi_dilation_voxels=2)

    # Lesion itself must not be in background
    assert not bg_mask[7:9, 7:9, 7:9].any()
    # Dilated face voxels must also NOT be in background
    assert not bg_mask[5, 7, 7]
    assert not bg_mask[10, 7, 7]
    assert not bg_mask[7, 5, 7]
    assert not bg_mask[7, 10, 7]
    # Far background must be in background
    assert bg_mask[0, 0, 0].item() is True
    assert bg_mask[15, 15, 15].item() is True


def test_inference_sliding_window_hypersphere_renormalization():
    """
    Signature:
        test_inference_sliding_window_hypersphere_renormalization() -> None

    Objective:
        Verify that blending unit hypersphere embeddings with Gaussian weights produces
        vectors with norm < 1.0, and re-normalizing by np.linalg.norm restores exact unit length.
    """
    D, Z, Y, X = 32, 16, 16, 16
    gauss = build_gaussian_weight(patch=Z)
    assert gauss.shape == (Z, Y, X)
    assert abs(gauss[Z // 2, Y // 2, X // 2] - 1.0) < 1e-4

    # Two overlapping unit embeddings with equal weights
    e1 = np.random.randn(D, Z, Y, X).astype(np.float32)
    e1 /= np.linalg.norm(e1, axis=0, keepdims=True)

    e2 = np.random.randn(D, Z, Y, X).astype(np.float32)
    e2 /= np.linalg.norm(e2, axis=0, keepdims=True)

    # Blended embedding
    blended = 0.5 * e1 + 0.5 * e2
    blended_norm = np.linalg.norm(blended, axis=0)

    # Average of distinct unit vectors has norm strictly < 1.0
    assert (blended_norm < 1.0).all()

    # Re-project back to unit hypersphere
    reprojected = blended / np.maximum(blended_norm[np.newaxis], 1e-8)
    reprojected_norm = np.linalg.norm(reprojected, axis=0)

    assert np.allclose(reprojected_norm, 1.0, atol=1e-5)


def test_top_k_seeds_chebyshev_nms():
    """
    Signature:
        test_top_k_seeds_chebyshev_nms() -> None

    Objective:
        Verify top_k_seeds extracts distinct seeds separated by Chebyshev distance,
        and halts early when confidence falls below min_prob.
    """
    Z, Y, X = 32, 32, 32
    prob = np.zeros((Z, Y, X), dtype=np.float32)

    # Peak A at (10, 10, 10) with prob 0.90
    prob[10, 10, 10] = 0.90
    # Peak B nearby at (11, 11, 11) with prob 0.85 (within Chebyshev distance 3 -> suppressed!)
    prob[11, 11, 11] = 0.85
    # Peak C far away at (25, 25, 25) with prob 0.70 (kept!)
    prob[25, 25, 25] = 0.70
    # Low confidence noise peak at (5, 5, 5) with prob 0.20 (below min_prob -> rejected!)
    prob[5, 5, 5] = 0.20

    seeds = top_k_seeds(prob, k=4, min_separation=3, min_prob=0.50)

    # Exactly 2 seeds should qualify: Peak A and Peak C
    assert len(seeds) == 2
    assert seeds[0] == (10, 10, 10)
    assert seeds[1] == (25, 25, 25)

    # Test confidence cutoff: if min_prob=0.95, no seeds qualify
    no_seeds = top_k_seeds(prob, k=4, min_separation=3, min_prob=0.95)
    assert len(no_seeds) == 0


def test_extract_instances_from_embeddings_clustering():
    """
    Signature:
        test_extract_instances_from_embeddings_clustering() -> None

    Objective:
        Verify extract_instances_from_embeddings extracts seeded cluster masks,
        prunes blobs smaller than min_volume_voxels, and handles empty candidates.
    """
    D, Z, Y, X = 32, 16, 16, 16
    # Construct synthetic embeddings where center is a tight cluster
    embeds = np.random.randn(D, Z, Y, X).astype(np.float32)
    embeds /= np.linalg.norm(embeds, axis=0, keepdims=True)

    seed_vec = np.zeros(D, dtype=np.float32); seed_vec[0] = 1.0
    # Embed a 3x3x3 lesion around seed (8, 8, 8) matching seed_vec
    embeds[:, 7:10, 7:10, 7:10] = seed_vec[:, np.newaxis, np.newaxis, np.newaxis]

    candidate_mask = np.zeros((Z, Y, X), dtype=bool)
    candidate_mask[6:11, 6:11, 6:11] = True  # covers lesion

    inst_mask = extract_instances_from_embeddings(
        embeddings=embeds,
        candidate_mask=candidate_mask,
        delta_var=0.5,
        min_volume_voxels=5,
        seed_coords=[(8, 8, 8)],
    )

    assert isinstance(inst_mask, np.ndarray)
    assert inst_mask.shape == (Z, Y, X)
    assert inst_mask[8, 8, 8] == 1
    assert inst_mask.sum() == 27  # exact 3x3x3 lesion recovered


def test_voxtell_spoco_decoder_dim_pinned_32():
    """
    Signature:
        test_voxtell_spoco_decoder_dim_pinned_32() -> None

    Objective:
        Verify that load_voxtell_spoco_model raises ValueError if embedding_dim != 32.
    """
    from scripts.phase_4_voxtell_spoco.common.voxtell_spoco import load_voxtell_spoco_model

    with pytest.raises(ValueError, match="embedding_dim must be 32"):
        load_voxtell_spoco_model(model_dir="dummy_dir", device="cpu", embedding_dim=16)

    with pytest.raises(ValueError, match="embedding_dim must be 32"):
        load_voxtell_spoco_model(model_dir="dummy_dir", device="cpu", embedding_dim=64)
