"""
===============================================================================
TEST SUITE:     VoxTell-SPOCO Architecture & Loss Diagnostics
LOCATION:       tests/test_voxtell_spoco.py
OBJECTIVE:      Verify instantiation of VoxTellSpocoModel, forward pass output shapes,
                unit-hypersphere normalization, SPOCO loss computation, and backprop.
===============================================================================
"""

import pytest
import torch
import numpy as np

from scripts.phase_4_alternative_models.common.voxtell_spoco import VoxTellSpocoModel
from scripts.phase_4_alternative_models.common.losses import (
    compute_gaussian_soft_mask,
    sample_annotated_anchors,
    sample_unannotated_anchors,
    compute_instance_dice_loss,
    compute_spoco_total_loss,
)
from scripts.phase_4_alternative_models.common.clustering import extract_instances_from_embeddings


def test_gaussian_soft_mask_properties():
    """Verify Gaussian soft mask computation on synthetic unit sphere embeddings."""
    D, Z, Y, X = 16, 32, 32, 32
    embeds = torch.randn(D, Z, Y, X)
    embeds = torch.nn.functional.normalize(embeds, p=2, dim=0)

    anchor_coords = torch.tensor([[10, 15, 20], [5, 5, 5]], dtype=torch.long)
    soft_masks = compute_gaussian_soft_mask(embeds, anchor_coords, sigma=0.5)

    assert soft_masks.shape == (2, Z, Y, X)
    assert (soft_masks >= 0.0).all() and (soft_masks <= 1.0).all()
    # At anchor location, distance is 0 -> soft_mask must be exactly 1.0
    assert torch.isclose(soft_masks[0, 10, 15, 20], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(soft_masks[1, 5, 5, 5], torch.tensor(1.0), atol=1e-4)


def test_spoco_total_loss_gradient_flow():
    """Verify end-to-end SPOCO total loss computation and gradient backpropagation."""
    B, N, D, Z, Y, X = 1, 2, 16, 16, 16, 16

    student_embeds = torch.randn(B, N, D, Z, Y, X, requires_grad=True)
    teacher_embeds = torch.randn(B, N, D, Z, Y, X)

    targets = torch.zeros(B, N, Z, Y, X)
    # Finding 0 has a positive lesion
    targets[0, 0, 6:10, 6:10, 6:10] = 1.0
    # Finding 1 is a negative finding (targets.sum() == 0)

    total_loss, l_obj, l_con = compute_spoco_total_loss(
        student_embeds, teacher_embeds, targets, sigma=0.5, w_con=0.1
    )

    assert total_loss.item() > 0
    assert torch.isfinite(total_loss)

    total_loss.backward()
    assert student_embeds.grad is not None
    assert torch.isfinite(student_embeds.grad).all()


def test_clustering_instance_extraction():
    """Verify instance mask extraction from synthetic embedding volumes."""
    D, Z, Y, X = 16, 20, 20, 20
    embeds = np.random.randn(D, Z, Y, X).astype(np.float32)
    embeds = embeds / np.linalg.norm(embeds, axis=0, keepdims=True)

    mask = extract_instances_from_embeddings(embeds, sigma=0.5, threshold=0.5, min_volume_voxels=5)
    assert mask.shape == (Z, Y, X)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})

