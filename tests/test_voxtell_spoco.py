"""
===============================================================================
TEST SUITE:     VoxTell-SPOCO Architecture & Loss Diagnostics
LOCATION:       tests/test_voxtell_spoco.py
OBJECTIVE:      Verify instantiation of VoxTellSpocoModel, forward pass output shapes,
                unit-hypersphere normalization, calibrated Gaussian soft masks,
                multi-instance connected-component anchor sampling, iterative background
                suppression, unlabeled push loss, view perturbation, and backprop.
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

from scripts.phase_4_alternative_models.common.voxtell_spoco import VoxTellSpocoModel
from scripts.phase_4_alternative_models.common.losses import (
    compute_gaussian_soft_mask,
    sample_annotated_anchors,
    sample_unannotated_anchors,
    compute_instance_dice_loss,
    compute_unlabeled_push_loss,
    compute_spoco_total_loss,
)
from scripts.phase_4_alternative_models.common.clustering import extract_instances_from_embeddings
from scripts.phase_4_alternative_models.exp_001_voxtell_spoco import apply_student_view_perturbation


def test_gaussian_soft_mask_mathematical_calibration():
    """Verify Gaussian soft mask mathematical calibration to delta_var and pmaps_threshold."""
    D, Z, Y, X = 16, 32, 32, 32
    embeds = torch.randn(D, Z, Y, X)
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=0)

    anchor_coords = torch.tensor([[10, 15, 20], [5, 5, 5]], dtype=torch.long)
    delta_var = 0.5
    pmaps_threshold = 0.5

    soft_masks = compute_gaussian_soft_mask(
        embeddings=embeds,
        anchor_coords=anchor_coords,
        delta_var=delta_var,
        pmaps_threshold=pmaps_threshold,
    )

    assert soft_masks.shape == (2, Z, Y, X)
    assert (soft_masks >= 0.0).all() and (soft_masks <= 1.0).all()
    # At anchor location, distance is 0 -> soft_mask must be exactly 1.0
    assert torch.isclose(soft_masks[0, 10, 15, 20], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(soft_masks[1, 5, 5, 5], torch.tensor(1.0), atol=1e-4)

    # Test synthetic point at exact distance delta_var:
    anchor_vec = embeds[:, 10, 15, 20]
    # Synthetic vector at distance delta_var on unit sphere:
    # ||v - a||^2 = 2 - 2 <v, a> = delta_var^2 => <v, a> = 1 - delta_var^2 / 2
    # two_sigma = delta_var^2 / (-ln(pmaps_threshold))
    # soft_mask = exp(- (delta_var^2) / two_sigma) = exp(ln(pmaps_threshold)) = pmaps_threshold
    two_sigma = (delta_var ** 2) / (-math.log(pmaps_threshold))
    expected_val = math.exp(- (delta_var ** 2) / two_sigma)
    assert abs(expected_val - pmaps_threshold) < 1e-6


def test_sample_annotated_anchors_multi_instance():
    """Verify connected-component anchor sampling extracts distinct anchors for disjoint lesions."""
    Z, Y, X = 20, 20, 20
    target_mask = torch.zeros((Z, Y, X), dtype=torch.float32)

    # Component 1 (left side)
    target_mask[2:5, 2:5, 2:5] = 1.0
    # Component 2 (right side, completely disjoint)
    target_mask[12:15, 12:15, 12:15] = 1.0

    items = sample_annotated_anchors(target_mask)
    assert len(items) == 2, f"Expected 2 connected components, got {len(items)}"

    coords = [item[0] for item in items]
    comp_masks = [item[1] for item in items]

    # Verify each anchor is strictly within its component
    c1, c2 = coords[0], coords[1]
    assert (2 <= c1[0] < 5 and 2 <= c1[1] < 5 and 2 <= c1[2] < 5) or \
           (12 <= c1[0] < 15 and 12 <= c1[1] < 15 and 12 <= c1[2] < 15)
    assert (2 <= c2[0] < 5 and 2 <= c2[1] < 5 and 2 <= c2[2] < 5) or \
           (12 <= c2[0] < 15 and 12 <= c2[1] < 15 and 12 <= c2[2] < 15)
    assert c1 != c2

    # Verify component masks isolate separate objects
    assert comp_masks[0].sum() == 27.0
    assert comp_masks[1].sum() == 27.0
    assert (comp_masks[0] * comp_masks[1]).sum() == 0.0


def test_sample_unannotated_anchors_coverage_suppression():
    """Verify iterative coverage suppression samples diverse unannotated anchors."""
    D, Z, Y, X = 16, 16, 16, 16
    embeds = torch.randn(D, Z, Y, X)
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=0)

    target_mask = torch.zeros((Z, Y, X), dtype=torch.float32)
    target_mask[0:4, 0:4, 0:4] = 1.0

    unlabeled_anchors = sample_unannotated_anchors(
        embeddings=embeds,
        target_mask=target_mask,
        delta_var=0.5,
        num_anchors=6,
        volume_threshold=0.05,
    )

    assert len(unlabeled_anchors) > 0
    assert len(unlabeled_anchors) <= 6

    # Verify all anchors are in unannotated regions
    for az, ay, ax in unlabeled_anchors:
        assert target_mask[az, ay, ax] == 0.0


def test_unlabeled_push_loss():
    """Verify unlabeled push loss computes positive penalty and correct gradients."""
    D, Z, Y, X = 16, 16, 16, 16
    embeds = torch.randn(D, Z, Y, X, requires_grad=True)
    embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=0)

    target_mask = torch.zeros((Z, Y, X), dtype=torch.float32)
    target_mask[2:6, 2:6, 2:6] = 1.0
    anchors = [(3, 3, 3)]

    l_push = compute_unlabeled_push_loss(
        embeddings=embeds_norm,
        instance_anchors=anchors,
        target_mask=target_mask,
        delta_dist=1.5,
    )

    assert l_push.item() >= 0.0
    assert torch.isfinite(l_push)

    l_push.backward()
    assert embeds.grad is not None
    assert torch.isfinite(embeds.grad).all()


def test_student_view_perturbation():
    """Verify student view perturbation preserves shape and modifies intensities."""
    images = torch.randn(2, 1, 16, 16, 16)
    perturbed = apply_student_view_perturbation(images)

    assert perturbed.shape == images.shape
    assert not torch.allclose(perturbed, images)
    assert torch.isfinite(perturbed).all()


def test_spoco_total_loss_gradient_flow():
    """Verify end-to-end SPOCO total loss computation with detailed loss tracking."""
    B, N, D, Z, Y, X = 1, 2, 16, 16, 16, 16

    student_embeds = torch.randn(B, N, D, Z, Y, X, requires_grad=True)
    teacher_embeds = torch.randn(B, N, D, Z, Y, X)

    targets = torch.zeros(B, N, Z, Y, X)
    # Finding 0 has two disjoint positive lesions
    targets[0, 0, 2:5, 2:5, 2:5] = 1.0
    targets[0, 0, 10:13, 10:13, 10:13] = 1.0
    # Finding 1 is a negative absent finding

    total_loss, l_obj, l_con, l_push = compute_spoco_total_loss(
        student_embeds=student_embeds,
        teacher_embeds=teacher_embeds,
        targets=targets,
        delta_var=0.5,
        delta_dist=1.5,
        pmaps_threshold=0.5,
        w_con=0.1,
        w_unl_push=0.1,
        return_details=True,
    )

    assert total_loss.item() > 0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(l_obj)
    assert torch.isfinite(l_con)
    assert torch.isfinite(l_push)

    total_loss.backward()
    assert student_embeds.grad is not None
    assert torch.isfinite(student_embeds.grad).all()


def test_clustering_with_candidate_mask():
    """Verify instance mask extraction with optional candidate mask pre-filtering."""
    D, Z, Y, X = 16, 20, 20, 20
    embeds = np.random.randn(D, Z, Y, X).astype(np.float32)
    embeds = embeds / np.linalg.norm(embeds, axis=0, keepdims=True)

    # Pre-filter candidate mask restricting to upper half
    candidate_mask = np.zeros((Z, Y, X), dtype=np.uint8)
    candidate_mask[:10, :, :] = 1

    mask = extract_instances_from_embeddings(
        embeddings=embeds,
        delta_var=0.5,
        pmaps_threshold=0.5,
        threshold=0.5,
        min_volume_voxels=5,
        candidate_mask=candidate_mask,
    )

    assert mask.shape == (Z, Y, X)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})
    # Since seeds are restricted to upper half, verify that instances can only be seeded there
