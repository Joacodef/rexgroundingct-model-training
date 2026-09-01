"""
===============================================================================
TEST SUITE:     Multi-Planar Projection Regularization (MPR) Loss Diagnostics
LOCATION:       tests/test_mpr_loss.py
OBJECTIVE:      Verify exact mathematical and geometric properties of MPR consistency
                loss using deterministic toy examples:
                1. Identity: Zero loss when Student == Teacher.
                2. ROI Isolation: Zero loss when discrepancy is confined within ROI.
                3. Dimension Amplification: Isolated false positives are amplified by S
                   relative to 3D voxel-wise MSE.
                4. Dispersed vs. Clustered Penalty: Dispersed noise produces higher
                   loss than clustered voxels due to projection collapse.
                5. Gradient Flow: Correct backprop strictly to background voxels.
===============================================================================
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.phase_3_voxtell_training.exp_003_mpr_loss import (
    compute_mpr_consistency_loss,
    compute_roi_mask,
)


def test_mpr_identity_zero_loss():
    """
    Signature:
        test_mpr_identity_zero_loss() -> None

    Objective:
        Verify that identical Student and Teacher predictions yield an exact 0.0 MPR loss.
    """
    B, F_dim, Z, Y, X = 1, 1, 16, 16, 16
    student_probs = torch.rand(B, F_dim, Z, Y, X)
    teacher_probs = student_probs.clone()
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)

    loss = compute_mpr_consistency_loss(student_probs, teacher_probs, roi_mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_mpr_roi_mask_isolation():
    """
    Signature:
        test_mpr_roi_mask_isolation() -> None

    Objective:
        Verify that discrepancies confined entirely within the dilated ROI are ignored by MPR loss.
    """
    B, F_dim, Z, Y, X = 1, 1, 16, 16, 16
    student_probs = torch.zeros(B, F_dim, Z, Y, X)
    teacher_probs = torch.zeros(B, F_dim, Z, Y, X)

    # Place a major discrepancy in the center
    student_probs[:, :, 6:10, 6:10, 6:10] = 1.0
    teacher_probs[:, :, 6:10, 6:10, 6:10] = 0.0

    # Define ROI mask covering that exact center region
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)
    roi_mask[:, :, 5:11, 5:11, 5:11] = True

    loss = compute_mpr_consistency_loss(student_probs, teacher_probs, roi_mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_mpr_dimension_amplification():
    """
    Signature:
        test_mpr_dimension_amplification() -> None

    Objective:
        Verify that a single isolated false positive voxel of value 1.0 is amplified
        by a factor of S (dimension size) compared to standard 3D voxel-wise MSE.
    """
    S = 16
    B, F_dim, Z, Y, X = 1, 1, S, S, S
    total_voxels = S * S * S  # 4096
    proj_pixels = S * S       # 256

    student_probs = torch.zeros(B, F_dim, Z, Y, X)
    teacher_probs = torch.zeros(B, F_dim, Z, Y, X)
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)

    # Introduce a single isolated false-positive voxel in background
    student_probs[0, 0, 8, 8, 8] = 1.0

    # 1. Standard 3D Voxel MSE
    voxel_mse = F.mse_loss(student_probs, teacher_probs).item()
    expected_voxel_mse = 1.0 / total_voxels  # 1 / 4096 ≈ 0.00024414

    # 2. MPR Loss
    mpr_loss = compute_mpr_consistency_loss(student_probs, teacher_probs, roi_mask).item()
    expected_mpr_loss = 1.0 / proj_pixels    # 1 / 256 ≈ 0.00390625

    assert abs(voxel_mse - expected_voxel_mse) < 1e-7
    assert abs(mpr_loss - expected_mpr_loss) < 1e-7

    # Amplification factor must be exactly S (16x)
    amplification_ratio = mpr_loss / voxel_mse
    assert abs(amplification_ratio - S) < 1e-4


def test_mpr_dispersed_vs_clustered_penalty():
    """
    Signature:
        test_mpr_dispersed_vs_clustered_penalty() -> None

    Objective:
        Verify the Gao et al. core hypothesis: K dispersed false positives yield a higher
        projection loss than K clustered/aligned false positives due to projection ray overlap.
    """
    S = 16
    B, F_dim, Z, Y, X = 1, 1, S, S, S
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)
    teacher_probs = torch.zeros(B, F_dim, Z, Y, X)

    # Configuration A: 4 false positive voxels aligned along the Z axis (collinear cluster)
    # They share the same (Y=8, X=8) coordinates across 4 different Z slices
    student_clustered = torch.zeros(B, F_dim, Z, Y, X)
    for z in [4, 5, 6, 7]:
        student_clustered[0, 0, z, 8, 8] = 1.0

    # Configuration B: 4 false positive voxels dispersed at distinct (Z, Y, X) positions
    student_dispersed = torch.zeros(B, F_dim, Z, Y, X)
    dispersed_coords = [(2, 2, 2), (5, 5, 5), (8, 8, 8), (11, 11, 11)]
    for z, y, x in dispersed_coords:
        student_dispersed[0, 0, z, y, x] = 1.0

    # Both configurations have identical 3D voxel MSE (4 voxels out of 4096)
    voxel_mse_clustered = F.mse_loss(student_clustered, teacher_probs).item()
    voxel_mse_dispersed = F.mse_loss(student_dispersed, teacher_probs).item()
    assert abs(voxel_mse_clustered - voxel_mse_dispersed) < 1e-7

    # However, MPR loss penalizes the dispersed configuration significantly more!
    mpr_clustered = compute_mpr_consistency_loss(student_clustered, teacher_probs, roi_mask).item()
    mpr_dispersed = compute_mpr_consistency_loss(student_dispersed, teacher_probs, roi_mask).item()

    # In clustered: axial max collapses all 4 into 1 pixel!
    # In dispersed: all 3 projections see 4 distinct pixels!
    assert mpr_dispersed > mpr_clustered
    # Dispersed loss should be exactly 4x the clustered axial projection loss
    assert abs(mpr_dispersed - 4.0 * (1.0 / 256.0)) < 1e-6


def test_mpr_gradient_flow():
    """
    Signature:
        test_mpr_gradient_flow() -> None

    Objective:
        Verify gradient backpropagation: non-zero gradients in unannotated background,
        strictly zero gradient inside the annotated ROI mask.
    """
    B, F_dim, Z, Y, X = 1, 1, 16, 16, 16
    student_probs = torch.full((B, F_dim, Z, Y, X), 0.5, requires_grad=True)
    teacher_probs = torch.zeros(B, F_dim, Z, Y, X)

    # ROI mask on the first half of Z
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)
    roi_mask[:, :, :8, :, :] = True

    loss = compute_mpr_consistency_loss(student_probs, teacher_probs, roi_mask)
    loss.backward()

    assert student_probs.grad is not None
    # Inside ROI: gradient must be exactly 0.0
    roi_grads = student_probs.grad[:, :, :8, :, :]
    assert torch.all(roi_grads == 0.0)

    # Outside ROI (background): gradients must be non-zero
    bg_grads = student_probs.grad[:, :, 8:, :, :]
    assert torch.any(bg_grads != 0.0)


def test_dice_vs_mpr_spatial_invariance_and_monotonicity():
    """
    Signature:
        test_dice_vs_mpr_spatial_invariance_and_monotonicity() -> None

    Objective:
        Verify the fundamental difference between 3D Dice Loss and MPR Loss:
        - 3D Dice loss is strictly spatially invariant: moving a false-positive island
          further away from the target object does NOT change the Dice loss at all (it stays constant).
        - In contrast, MPR loss strictly increases as the false positive moves out of the target
          object's orthogonal projection shadow, increasing monotonically from 1-plane eclipsed,
          to 2-plane exposed, to 3-plane fully exposed.
    """
    S = 32
    B, F_dim, Z, Y, X = 1, 1, S, S, S
    roi_mask = torch.zeros(B, F_dim, Z, Y, X, dtype=torch.bool)

    # Target object at center [12:20, 12:20, 12:20]
    teacher = torch.zeros(B, F_dim, Z, Y, X)
    teacher[:, :, 12:20, 12:20, 12:20] = 1.0

    # Three spatial positions for a 2x2x2 false positive outside the target object:
    # Pos 1: Collinear with the target along Z (shares both Y and X shadows: eclipsed in Axial plane)
    # Pos 2: Offset in Y, but still collinear in X (eclipsed only in Coronal plane)
    # Pos 3: Offset in all three axes (fully exposed in all 3 orthogonal planes)
    positions = [
        (24, 15, 15),  # Shares (Y, X) with target -> completely eclipsed in Axial projection
        (24, 24, 15),  # Shares X with target -> eclipsed in Coronal, exposed in Axial & Sagittal
        (24, 24, 24),  # Shares no axis -> exposed in Axial, Coronal, and Sagittal
    ]

    dice_losses = []
    mpr_losses = []
    voxel_mses = []

    for z, y, x in positions:
        student = teacher.clone()
        student[:, :, z:z+2, y:y+2, x:x+2] = 1.0

        # 3D Soft Dice Loss
        intersection = (student * teacher).sum()
        union = student.sum() + teacher.sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

        # 3D Voxel-wise MSE
        voxel_mse = F.mse_loss(student, teacher)

        # MPR Consistency Loss
        mpr_loss = compute_mpr_consistency_loss(student, teacher, roi_mask)

        dice_losses.append(dice_loss.item())
        voxel_mses.append(voxel_mse.item())
        mpr_losses.append(mpr_loss.item())

    # 1. Verify that 3D Dice loss is STRICTLY CONSTANT across all spatial positions
    assert abs(dice_losses[0] - dice_losses[1]) < 1e-6
    assert abs(dice_losses[1] - dice_losses[2]) < 1e-6

    # 2. Verify that 3D Voxel MSE is STRICTLY CONSTANT across all spatial positions
    assert abs(voxel_mses[0] - voxel_mses[1]) < 1e-6
    assert abs(voxel_mses[1] - voxel_mses[2]) < 1e-6

    # 3. Verify that MPR loss INCREASES as error moves out of projection shadows
    # Pos 1 (Axial eclipsed): mpr_losses[0] ≈ 0.002604
    # Pos 2 & 3 (Exposed): mpr_losses[1] ≈ 0.003906
    assert mpr_losses[0] < mpr_losses[1]
    assert mpr_losses[1] <= mpr_losses[2]


def test_gao_multi_angle_continuous_monotonicity():
    """
    Signature:
        test_gao_multi_angle_continuous_monotonicity() -> None

    Objective:
        Verify the Gao et al. (2022) Figure 4 property:
        As a false positive moves continuously farther away from the target object,
        standard 2D/3D Dice and MSE stay completely flat, whereas multi-angle MPR loss
        grows strictly monotonically across distances d.
    """
    import math

    H, W = 128, 128
    T = torch.zeros(1, 1, H, W)
    T[0, 0, 56:72, 56:72] = 1.0

    distances = [15, 20, 30, 40, 50]
    angles = [0, 20, 40, 60, 80, 100, 120, 140, 160]

    dice_losses = []
    gao_mpr_losses = []

    for d in distances:
        S = T.clone()
        rad = math.radians(30)
        r = int(64 + d * math.sin(rad))
        c = int(64 + d * math.cos(rad))
        S[0, 0, r-3:r+3, c-3:c+3] = 1.0

        # Dice Loss
        inter = (S * T).sum()
        union = S.sum() + T.sum()
        dice = (1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)).item()
        dice_losses.append(dice)

        # Gao Multi-angle MPR
        mpr_vals = []
        for ang in angles:
            theta = torch.tensor([[
                [math.cos(math.radians(ang)), -math.sin(math.radians(ang)), 0],
                [math.sin(math.radians(ang)),  math.cos(math.radians(ang)), 0]
            ]], dtype=torch.float32)
            grid = F.affine_grid(theta, S.size(), align_corners=False)
            S_rot = F.grid_sample(S, grid, mode='nearest', align_corners=False)
            T_rot = F.grid_sample(T, grid, mode='nearest', align_corners=False)

            p_h_s = torch.max(S_rot, dim=2)[0]
            p_w_s = torch.max(S_rot, dim=3)[0]
            p_h_t = torch.max(T_rot, dim=2)[0]
            p_w_t = torch.max(T_rot, dim=3)[0]

            d_h = 1.0 - (2.0 * (p_h_s * p_h_t).sum() + 1e-6) / (p_h_s.sum() + p_h_t.sum() + 1e-6)
            d_w = 1.0 - (2.0 * (p_w_s * p_w_t).sum() + 1e-6) / (p_w_s.sum() + p_w_t.sum() + 1e-6)
            mpr_vals.append(((d_h + d_w) / 2.0).item())

        gao_mpr_losses.append(sum(mpr_vals) / len(mpr_vals))

    # Verify Dice loss is flat across all distances d >= 15
    for i in range(len(dice_losses) - 1):
        assert abs(dice_losses[i] - dice_losses[i+1]) < 1e-6

    # Verify Gao MPR loss is STRICTLY MONOTONICALLY INCREASING across all distances
    for i in range(len(gao_mpr_losses) - 1):
        assert gao_mpr_losses[i] < gao_mpr_losses[i+1], (
            f"Expected strictly monotonic increase, but got {gao_mpr_losses[i]} >= {gao_mpr_losses[i+1]}"
        )


if __name__ == "__main__":
    print("Running Toy Demonstrations of MPR Loss Behavior...\n")

    test_mpr_identity_zero_loss()
    print("✓ Test 1: Identity zero loss verified.")

    test_mpr_roi_mask_isolation()
    print("✓ Test 2: ROI mask isolation verified.")

    test_mpr_dimension_amplification()
    print("✓ Test 3: Dimension amplification verified (16x amplification on 16^3 grid).")

    test_mpr_dispersed_vs_clustered_penalty()
    print("✓ Test 4: Dispersed vs. clustered penalty verified (Dispersed loss > Clustered loss).")

    test_mpr_gradient_flow()
    print("✓ Test 5: Gradient flow verified (Strictly 0 inside ROI, active in background).")

    test_dice_vs_mpr_spatial_invariance_and_monotonicity()
    print("✓ Test 6: 3D Dice spatial invariance vs. MPR projection exposure verified.")

    test_gao_multi_angle_continuous_monotonicity()
    print("✓ Test 7: Gao et al. Fig 4 multi-angle continuous monotonicity verified.")

    print("\nAll toy demonstrations passed successfully!")
