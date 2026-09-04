"""
===============================================================================
TEST SUITE:     Phase 3 VoxTell Fine-Tuning Diagnostics & Integrity Checks
LOCATION:       tests/test_phase_3_finetuning.py
OBJECTIVE:      Verify critical failure modes in Phase 3 fine-tuning:
                1. Supervision Routing: Present vs. absent vs. out-of-crop prompts
                   (LOSS-2, LOSS-5).
                2. Foreground Retention (BC-3): MONAI fg_union prevents channel 0 dropout.
                3. Fixed-N Prompt Collation (BC-1): Multi-finding batch stacking.
                4. PU Mean Teacher Masked Losses: ROI dilation, supervised & consistency terms.
                5. MPR / Consistency Warmup Schedules: Monotonicity and boundary constraints.
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
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from scripts.phase_3_voxtell_finetuning.exp_002_pu_mean_teacher import (
    compute_roi_mask,
    compute_roi_masked_loss,
    compute_unannotated_consistency_loss,
    get_consistency_weight,
)
from scripts.phase_3_voxtell_finetuning.exp_003_mpr_loss import (
    get_mpr_rampup_weight,
)


def test_supervision_routing_present_absent_out_of_crop():
    """
    Signature:
        test_supervision_routing_present_absent_out_of_crop() -> None

    Objective:
        Verify the critical supervision routing fix:
        - Prompt 0 (present, in crop): Receives supervised loss (BCE + Dice inside dilated ROI).
        - Prompt 1 (present, out-of-crop): Lesion fell outside random crop. MUST BE SKIPPED
          with zero loss and zero gradient, NOT penalized as confirmed background.
        - Prompt 2 (confirmed absent): Genuinely absent from scan. Receives full-volume
          BCE penalty against zeros, but zero teacher consistency loss.
    """
    B, F_dim, Z, Y, X = 1, 3, 16, 16, 16
    student_logits = torch.randn(B, F_dim, Z, Y, X, requires_grad=True)
    student_probs = torch.sigmoid(student_logits)
    teacher_probs = torch.rand(B, F_dim, Z, Y, X)

    targets = torch.zeros(B, F_dim, Z, Y, X)
    # Prompt 0 has a foreground lesion in the crop
    targets[0, 0, 6:10, 6:10, 6:10] = 1.0
    # Prompt 1 is present in the scan, but targets == 0 in this crop
    # Prompt 2 is confirmed absent from the scan, targets == 0 in this crop

    is_absent_finding = torch.tensor([[False, False, True]], dtype=torch.bool)

    has_fg = (targets.sum(dim=(2, 3, 4), keepdim=True) > 0)
    is_absent = is_absent_finding.view(B, F_dim, 1, 1, 1)
    supervise = is_absent | has_fg

    # Verify supervision flags
    assert supervise[0, 0, 0, 0, 0].item() is True    # Prompt 0: supervise
    assert supervise[0, 1, 0, 0, 0].item() is False   # Prompt 1: skip!
    assert supervise[0, 2, 0, 0, 0].item() is True    # Prompt 2: supervise

    roi_mask = compute_roi_mask(targets, kernel_size=5, padding=2)
    roi_mask = torch.where(is_absent, torch.ones_like(roi_mask), roi_mask) & supervise

    # Prompt 1 ROI must be completely empty
    assert roi_mask[0, 1].sum() == 0

    # Prompt 2 ROI must be full volume
    assert roi_mask[0, 2].sum() == Z * Y * X

    # Supervised loss
    loss_sup = compute_roi_masked_loss(student_logits.float(), targets.float(), roi_mask, pos_weight=5.0)

    # Consistency loss with valid_mask=supervise
    loss_con = compute_unannotated_consistency_loss(student_probs.float(), teacher_probs.float(), roi_mask, valid_mask=supervise)

    total_loss = loss_sup + 0.5 * loss_con
    total_loss.backward()

    # Verify gradients:
    # Prompt 1 (present, out of crop) MUST have exactly 0.0 gradient!
    assert torch.all(student_logits.grad[0, 1] == 0.0), (
        "Out-of-crop present finding received non-zero gradient! Should be skipped."
    )
    # Prompt 0 and Prompt 2 must receive active gradients
    assert torch.any(student_logits.grad[0, 0] != 0.0)
    assert torch.any(student_logits.grad[0, 2] != 0.0)


def test_monai_fg_union_preserves_channel_0_bc3():
    """
    Signature:
        test_monai_fg_union_preserves_channel_0_bc3() -> None

    Objective:
        Verify that a single-channel union mask `fg_union` prevents MONAI's
        RandCropByPosNegLabeld from discarding channel 0 of multi-channel labels.
    """
    import monai.transforms as mt

    Z, Y, X = 32, 32, 32
    patch_size = 16

    # 1-finding scan: seg has shape (1, Z, Y, X)
    seg_1 = torch.zeros(1, Z, Y, X)
    seg_1[0, 10:15, 10:15, 10:15] = 1.0  # foreground lesion
    fg_union_1 = (seg_1 > 0).any(dim=0, keepdim=True).float()

    # Pass through MONAI crop with label_key='fg_union'
    crop_fn = mt.RandCropByPosNegLabeld(
        keys=['seg', 'fg_union'],
        label_key='fg_union',
        spatial_size=[patch_size, patch_size, patch_size],
        pos=1.0,  # 100% positive crops
        neg=0.0,
        num_samples=1
    )

    data = {'seg': seg_1, 'fg_union': fg_union_1}
    cropped = crop_fn(data)[0]

    # Under fg_union, cropped seg MUST contain foreground voxels!
    assert cropped['seg'].sum() > 0, "BC-3 regression: 1-finding scan produced an empty foreground crop!"


def test_fixed_n_padding_collate_bc1():
    """
    Signature:
        test_fixed_n_padding_collate_bc1() -> None

    Objective:
        Verify that padding prompts to a fixed N = num_pos + num_neg allows default_collate
        to stack scans with different finding counts when batch_size > 1 without crashing.
    """
    num_pos = 2
    num_neg = 1
    fixed_N = num_pos + num_neg  # N = 3
    D, Z, Y, X = 8, 16, 16, 16

    # Scan A has 1 finding
    scan_a = {
        'image': torch.randn(1, Z, Y, X),
        'seg': torch.zeros(fixed_N, Z, Y, X),
        'text_embeddings': torch.randn(fixed_N, D),
        'is_absent_finding': torch.tensor([False, True, True]),  # 1 pos + 2 padded negs
    }

    # Scan B has 2 findings
    scan_b = {
        'image': torch.randn(1, Z, Y, X),
        'seg': torch.zeros(fixed_N, Z, Y, X),
        'text_embeddings': torch.randn(fixed_N, D),
        'is_absent_finding': torch.tensor([False, False, True]),  # 2 pos + 1 neg
    }

    # Collation of batch of size 2
    batch = default_collate([scan_a, scan_b])

    assert batch['image'].shape == (2, 1, Z, Y, X)
    assert batch['seg'].shape == (2, fixed_N, Z, Y, X)
    assert batch['text_embeddings'].shape == (2, fixed_N, D)
    assert batch['is_absent_finding'].shape == (2, fixed_N)


def test_compute_roi_mask_dilation():
    """
    Signature:
        test_compute_roi_mask_dilation() -> None

    Objective:
        Verify that compute_roi_mask expands binary foreground by the expected dilation kernel.
    """
    B, F_dim, Z, Y, X = 1, 1, 16, 16, 16
    targets = torch.zeros(B, F_dim, Z, Y, X)
    # Single center voxel
    targets[0, 0, 8, 8, 8] = 1.0

    # Kernel 5x5x5 with padding 2 expands 1 voxel to a 5x5x5 cube (125 voxels)
    roi_mask = compute_roi_mask(targets, kernel_size=5, padding=2)

    assert roi_mask.shape == targets.shape
    assert roi_mask.dtype == torch.bool
    assert roi_mask[0, 0, 8, 8, 8].item() is True
    # Verify exact cube expansion
    assert roi_mask[0, 0, 6:11, 6:11, 6:11].all().item() is True
    assert roi_mask.sum() == 125


def test_pu_mean_teacher_masked_loss_empty_channel():
    """
    Signature:
        test_pu_mean_teacher_masked_loss_empty_channel() -> None

    Objective:
        Verify compute_roi_masked_loss gracefully handles channels where roi_mask is empty
        without generating NaNs or corrupting active channel gradients.
    """
    B, F_dim, Z, Y, X = 1, 2, 16, 16, 16
    logits = torch.randn(B, F_dim, Z, Y, X, requires_grad=True)
    targets = torch.zeros(B, F_dim, Z, Y, X)
    targets[0, 0, 5:8, 5:8, 5:8] = 1.0

    # Channel 0 has an active ROI; Channel 1 has an empty ROI
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)
    roi_mask[0, 0, 4:9, 4:9, 4:9] = True

    loss = compute_roi_masked_loss(logits, targets, roi_mask, pos_weight=2.0)
    assert torch.isfinite(loss)
    loss.backward()

    # Active channel has gradients
    assert torch.any(logits.grad[0, 0] != 0.0)
    # Empty channel has zero gradients
    assert torch.all(logits.grad[0, 1] == 0.0)


def test_consistency_weight_schedules():
    """
    Signature:
        test_consistency_weight_schedules() -> None

    Objective:
        Verify sigmoid and exponential warmup consistency schedules:
        - Bound check: weights are non-negative and bounded by max_weight.
        - Monotonicity: weights increase monotonically across warmup epochs.
    """
    max_w = 0.5
    warmup = 15

    # 1. Sigmoid warmup (Exp 002)
    sig_weights = [get_consistency_weight(ep, max_weight=max_w, warm_up_epochs=warmup) for ep in range(1, 25)]
    for w in sig_weights:
        assert 0.0 <= w <= max_w
    for i in range(len(sig_weights) - 1):
        assert sig_weights[i] <= sig_weights[i + 1]

    # 2. Gaussian exponential warmup (Exp 003 SOUSA)
    max_ep = 15
    mpr_weights = [get_mpr_rampup_weight(ep, max_epochs=max_ep, max_weight=max_w) for ep in range(0, 25)]
    for w in mpr_weights:
        assert 0.0 <= w <= max_w
    # At epoch 0, weight should be very small (~0.0034)
    assert mpr_weights[0] < 0.01
    # At epoch >= max_ep, weight should be exactly max_w
    assert abs(mpr_weights[max_ep] - max_w) < 1e-5
    assert abs(mpr_weights[max_ep + 5] - max_w) < 1e-5
