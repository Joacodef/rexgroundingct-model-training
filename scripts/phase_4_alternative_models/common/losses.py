"""
===============================================================================
MODULE:         True 3D SPOCO Loss Functions & Anchor Sampling
LOCATION:       scripts/phase_4_alternative_models/common/losses.py
OBJECTIVE:      Implement differentiable Gaussian soft masks, instance-level Dice
                supervision on annotated anchors (L_obj), and EMA Teacher consistency
                on unannotated background anchors (L_con) for True SPOCO training.
===============================================================================
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gaussian_soft_mask(
    embeddings: torch.Tensor,
    anchor_coords: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Signature:
        compute_gaussian_soft_mask(embeddings: torch.Tensor, anchor_coords: torch.Tensor, sigma: float = 0.5) -> torch.Tensor

    Objective:
        Compute differentiable Gaussian soft segmentation masks from anchor embeddings.
        S_k(i) = exp(- ||e_i - e(a_k)||^2 / (2 * sigma^2)) = exp(- (2 - 2 <e_i, e(a_k)>) / (2 * sigma^2)).

    Inputs:
        embeddings (torch.Tensor): 3D pixel embeddings of shape (D, Z, Y, X) normalized along D.
        anchor_coords (torch.Tensor): Anchor voxel index tensor of shape (K, 3) where columns are (z, y, x).
        sigma (float): Gaussian bandwidth scaling factor. Default 0.5.

    Outputs:
        torch.Tensor: Soft mask tensor of shape (K, Z, Y, X) in range [0, 1].
    """
    D, Z, Y, X = embeddings.shape
    K = anchor_coords.shape[0]

    if K == 0:
        return torch.zeros((0, Z, Y, X), device=embeddings.device, dtype=embeddings.dtype)

    # Extract anchor embedding vectors: (K, D)
    anchor_embeds = []
    for k in range(K):
        az, ay, ax = anchor_coords[k]
        anchor_embeds.append(embeddings[:, az, ay, ax])
    anchor_embeds = torch.stack(anchor_embeds, dim=0)  # (K, D)

    # Compute Euclidean distance: ||e_i - e(a_k)||^2 = 2 - 2 * (e_i . e(a_k)) on unit sphere
    dot_prod = torch.einsum("kd, dzyx -> kzyx", anchor_embeds, embeddings)
    dist_sq = torch.clamp(2.0 - 2.0 * dot_prod, min=0.0)

    # Gaussian soft mask
    soft_masks = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))
    return soft_masks


def sample_annotated_anchors(target_mask: torch.Tensor) -> List[Tuple[int, int, int]]:
    """
    Signature:
        sample_annotated_anchors(target_mask: torch.Tensor) -> list[tuple[int, int, int]]

    Objective:
        Sample anchor voxels from annotated ground-truth lesion components.

    Inputs:
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).

    Outputs:
        list[tuple[int, int, int]]: List of (z, y, x) anchor coordinates.
    """
    positive_coords = torch.nonzero(target_mask > 0.5)
    if len(positive_coords) == 0:
        return []

    # Sample central/median coordinate of positive cluster
    med_idx = len(positive_coords) // 2
    coord = positive_coords[med_idx].tolist()
    return [tuple(coord)]


def sample_unannotated_anchors(
    target_mask: torch.Tensor,
    num_anchors: int = 5,
) -> List[Tuple[int, int, int]]:
    """
    Signature:
        sample_unannotated_anchors(target_mask: torch.Tensor, num_anchors: int = 5) -> list[tuple[int, int, int]]

    Objective:
        Sample unannotated anchor voxels from background region (1 - GT) for Teacher consistency.

    Inputs:
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).
        num_anchors (int): Number of unlabeled anchors to sample. Default 5.

    Outputs:
        list[tuple[int, int, int]]: List of (z, y, x) anchor coordinates in unannotated region.
    """
    unlabeled_coords = torch.nonzero(target_mask == 0)
    if len(unlabeled_coords) == 0:
        return []

    sampled_indices = torch.randint(0, len(unlabeled_coords), (min(num_anchors, len(unlabeled_coords)),))
    anchors = [tuple(unlabeled_coords[idx].tolist()) for idx in sampled_indices]
    return anchors


def compute_instance_dice_loss(
    soft_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Signature:
        compute_instance_dice_loss(soft_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor

    Objective:
        Compute soft Dice loss for an individual anchor soft mask against ground truth.

    Inputs:
        soft_mask (torch.Tensor): Predicted Gaussian soft mask of shape (Z, Y, X).
        target_mask (torch.Tensor): Binary ground truth mask of shape (Z, Y, X).

    Outputs:
        torch.Tensor: Scalar Dice loss tensor in range [0, 1].
    """
    intersection = (soft_mask * target_mask).sum().float()
    union = soft_mask.sum().float() + target_mask.sum().float()
    return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)


def compute_spoco_total_loss(
    student_embeds: torch.Tensor,
    teacher_embeds: torch.Tensor,
    targets: torch.Tensor,
    sigma: float = 0.5,
    w_con: float = 0.1,
    num_unlabeled_anchors: int = 5,
    negative_supervision: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Signature:
        compute_spoco_total_loss(student_embeds: torch.Tensor, teacher_embeds: torch.Tensor, targets: torch.Tensor, sigma: float = 0.5, w_con: float = 0.1, num_unlabeled_anchors: int = 5, negative_supervision: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    Objective:
        Compute full True 3D SPOCO loss: Instance-level Dice on annotated objects (L_obj)
        plus Unannotated Anchor Soft-Mask Consistency (L_con) against EMA Teacher,
        with optional null-target supervision on confirmed absent findings.

    Inputs:
        student_embeds (torch.Tensor): Student embeddings of shape (B, N, D, Z, Y, X).
        teacher_embeds (torch.Tensor): EMA Teacher embeddings of shape (B, N, D, Z, Y, X).
        targets (torch.Tensor): Ground truth target tensor of shape (B, N, Z, Y, X).
        sigma (float): Gaussian bandwidth scaling factor. Default 0.5.
        w_con (float): Consistency loss weight for unannotated anchors. Default 0.1.
        num_unlabeled_anchors (int): Number of unlabeled anchors sampled per finding. Default 5.
        negative_supervision (bool): Whether to penalize false-positive anchor masks on absent findings. Default True.

    Outputs:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (total_loss, loss_supervised_obj, loss_consistency_unlabeled).
    """
    B, N_prompts, D, Z, Y, X = student_embeds.shape
    device = student_embeds.device

    loss_obj_list = []
    loss_con_list = []

    for b in range(B):
        for n in range(N_prompts):
            s_embed = student_embeds[b, n]  # (D, Z, Y, X)
            t_embed = teacher_embeds[b, n]  # (D, Z, Y, X)
            tgt = targets[b, n]             # (Z, Y, X)

            is_negative_finding = (tgt.sum() == 0)

            # 1. Supervised Instance Soft Dice on Annotated Anchors
            if not is_negative_finding:
                annotated_anchors = sample_annotated_anchors(tgt)
                if annotated_anchors:
                    anchor_tensor = torch.tensor(annotated_anchors, device=device, dtype=torch.long)
                    soft_masks = compute_gaussian_soft_mask(s_embed, anchor_tensor, sigma=sigma)  # (K, Z, Y, X)
                    for k in range(len(annotated_anchors)):
                        l_dice = compute_instance_dice_loss(soft_masks[k], tgt)
                        loss_obj_list.append(l_dice)

                # 2. Unannotated Anchor Consistency (Student vs EMA Teacher)
                unlabeled_anchors = sample_unannotated_anchors(tgt, num_anchors=num_unlabeled_anchors)
                if unlabeled_anchors:
                    u_anchor_tensor = torch.tensor(unlabeled_anchors, device=device, dtype=torch.long)
                    s_u_soft = compute_gaussian_soft_mask(s_embed, u_anchor_tensor, sigma=sigma)
                    t_u_soft = compute_gaussian_soft_mask(t_embed, u_anchor_tensor, sigma=sigma)

                    for u in range(len(unlabeled_anchors)):
                        l_con = compute_instance_dice_loss(s_u_soft[u], t_u_soft[u].detach())
                        loss_con_list.append(l_con)
            else:
                # Confirmed absent negative finding: penalize random anchor soft masks if negative_supervision enabled
                if negative_supervision:
                    neg_anchors = sample_unannotated_anchors(tgt, num_anchors=num_unlabeled_anchors)
                    if neg_anchors:
                        neg_anchor_tensor = torch.tensor(neg_anchors, device=device, dtype=torch.long)
                        s_neg_soft = compute_gaussian_soft_mask(s_embed, neg_anchor_tensor, sigma=sigma)
                        # Penalize mass for absent structures: mean(soft_mask) -> 0
                        loss_obj_list.append(s_neg_soft.mean())

    loss_obj = torch.stack(loss_obj_list).mean() if loss_obj_list else torch.tensor(0.0, device=device, requires_grad=True)
    loss_con = torch.stack(loss_con_list).mean() if loss_con_list else torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = loss_obj + w_con * loss_con
    return total_loss, loss_obj, loss_con

