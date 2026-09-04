"""
===============================================================================
MODULE:         3D SPOCO Loss Functions & Anchor Sampling
LOCATION:       scripts/phase_4_voxtell_spoco/common/losses.py
OBJECTIVE:      Implement mathematically calibrated Gaussian soft masks, multi-anchor
                supervision on annotated lesions (L_obj) scored by default against the
                UNION of all instances of a finding (not per-instance -- ReXGroundingCT's
                official Dice/Hit-Rate metric is finding-level and carries no instance-
                matching term, see `union_target` on compute_spoco_total_loss), iterative
                coverage-suppression consistency on unannotated anchors (L_con), and
                optional unlabeled background push repulsion (L_unl_push) for SPOCO 3D
                metric learning (Wolny et al., CVPR 2022).
===============================================================================
"""

import math
from typing import List, Tuple, Any, Optional
import numpy as np
from scipy.ndimage import label, binary_dilation
import torch


def _confirmed_background_mask(target_mask: torch.Tensor, roi_dilation_voxels: int) -> torch.Tensor:
    """
    Signature:
        _confirmed_background_mask(target_mask: torch.Tensor, roi_dilation_voxels: int) -> torch.Tensor

    Objective:
        Return a bool tensor (same shape/device as `target_mask`) that is True only for
        voxels treated as confirmed background: everything outside the annotated
        foreground dilated by `roi_dilation_voxels`. The dilation band around each lesion
        is excluded so the push / unannotated-consistency terms do not fight the lesion
        boundary and partial-volume edges.

    Inputs:
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).
        roi_dilation_voxels (int): Morphological dilation radius in voxels (0 disables).

    Outputs:
        torch.Tensor: Bool "confirmed background" mask.
    """
    fg_np = (target_mask.detach().cpu().numpy() > 0.5)
    if roi_dilation_voxels and roi_dilation_voxels > 0 and fg_np.any():
        fg_np = binary_dilation(fg_np, iterations=int(roi_dilation_voxels))
    return torch.from_numpy(~fg_np).to(device=target_mask.device)


def _unit_sphere_dist_sq(
    anchor_vecs: torch.Tensor,
    embeddings: torch.Tensor,
    subscripts: str,
) -> torch.Tensor:
    """
    Signature:
        _unit_sphere_dist_sq(anchor_vecs: torch.Tensor, embeddings: torch.Tensor, subscripts: str) -> torch.Tensor

    Objective:
        Compute the squared Euclidean distance ||e_i - a_k||^2 = 2 - 2 <e_i, a_k> between anchor
        vectors and dense embeddings, both assumed L2-normalized onto the unit hypersphere S^31.
        The contraction is forced to float32 even under bfloat16 autocast: bf16 has 7 stored
        mantissa bits, so its spacing at 1.0 is 2^-7 = 0.0078125, and `2 - 2*dot` is a
        cancellation that quantizes every true squared distance below ~0.004 to exactly 0. That
        pins the soft mask at 1.0 with an identically zero gradient across the whole tight-cluster
        regime the L_obj pull term exists to sharpen.

    Inputs:
        anchor_vecs (torch.Tensor): Anchor embedding vectors, (K, D) or (D,).
        embeddings (torch.Tensor): Dense embeddings, (D, Z, Y, X) or (D, S).
        subscripts (str): einsum contraction string pairing the two operands, e.g. "kd, dzyx -> kzyx".

    Outputs:
        torch.Tensor: Non-negative float32 squared distances, shaped by `subscripts`.
    """
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        dot_prod = torch.einsum(subscripts, anchor_vecs.float(), embeddings.float())
        return torch.clamp(2.0 - 2.0 * dot_prod, min=0.0)


def compute_gaussian_soft_mask(
    embeddings: torch.Tensor,
    anchor_coords: torch.Tensor,
    delta_var: float = 0.5,
    pmaps_threshold: float = 0.5,
    sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Signature:
        compute_gaussian_soft_mask(embeddings: torch.Tensor, anchor_coords: torch.Tensor, delta_var: float = 0.5, pmaps_threshold: float = 0.5, sigma: float | None = None) -> torch.Tensor

    Objective:
        Compute differentiable Gaussian soft segmentation masks from anchor embeddings.
        S_k(i) = exp(- ||e_i - e(a_k)||^2 / two_sigma)
        where two_sigma = (delta_var^2) / (-ln(pmaps_threshold)) ensuring that points
        at distance delta_var from anchor evaluate to exactly pmaps_threshold.

    Inputs:
        embeddings (torch.Tensor): 3D pixel embeddings of shape (D, Z, Y, X) normalized along D.
        anchor_coords (torch.Tensor): Anchor voxel index tensor of shape (K, 3) where columns are (z, y, x).
        delta_var (float): Intra-cluster pull margin (default 0.5).
        pmaps_threshold (float): Soft mask probability cutoff at distance delta_var (default 0.5).
        sigma (float | None): Optional legacy sigma override. If provided, two_sigma = 2 * sigma^2.

    Outputs:
        torch.Tensor: Float32 soft mask tensor of shape (K, Z, Y, X) in range [0, 1].
    """
    D, Z, Y, X = embeddings.shape
    K = anchor_coords.shape[0]

    if K == 0:
        return torch.zeros((0, Z, Y, X), device=embeddings.device, dtype=torch.float32)

    # Resolve Gaussian variance parameter
    if sigma is not None:
        two_sigma = 2.0 * (sigma ** 2)
    else:
        two_sigma = (delta_var ** 2) / (-math.log(max(1e-7, min(1.0 - 1e-7, pmaps_threshold))))

    # Extract anchor embedding vectors: (K, D)
    anchor_embeds = []
    for k in range(K):
        az, ay, ax = anchor_coords[k]
        anchor_embeds.append(embeddings[:, az, ay, ax])
    anchor_embeds = torch.stack(anchor_embeds, dim=0)  # (K, D)

    # Euclidean distance on the unit sphere, in float32 (see _unit_sphere_dist_sq).
    dist_sq = _unit_sphere_dist_sq(anchor_embeds, embeddings, "kd, dzyx -> kzyx")

    # Gaussian soft mask (float32, so the returned mask is float32 regardless of autocast)
    soft_masks = torch.exp(-dist_sq / max(1e-8, two_sigma))
    return soft_masks


def sample_annotated_anchors(
    target_mask: torch.Tensor,
    max_components: int = 8,
) -> List[Tuple[Tuple[int, int, int], torch.Tensor]]:
    """
    Signature:
        sample_annotated_anchors(target_mask: torch.Tensor, max_components: int = 8) -> list[tuple[tuple[int, int, int], torch.Tensor]]

    Objective:
        Sample anchor voxels from annotated ground-truth lesion components using 3D
        connected-component decomposition. If a finding contains multiple disjoint lesions,
        extracts one interior anchor per component (guaranteed inside that component, unlike
        a naive union-wide median which can land in the gap between two disjoint blobs) and
        pairs it with that component's isolated mask. The isolated mask returned here is what
        `max_annotated_components` in `compute_spoco_total_loss` bounds the *count* of; it is
        NOT necessarily what each anchor's soft mask is later scored against -- under the
        caller's default `union_target=True`, every returned anchor is instead compared to the
        finding's full mask, so this cap only bounds how many redundant interior anchors are
        placed, not how many voxels receive supervision. Only the `max_components` largest
        components are kept: a single micronodule / reticular-thickening crop can hold
        dozens-to-hundreds of components, and each one adds a full (Z, Y, X) soft mask to the
        autograd tape downstream, which is a latent OOM.

    Inputs:
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).
        max_components (int): Maximum number of components to return, largest first (default 8).

    Outputs:
        list[tuple[tuple[int, int, int], torch.Tensor]]: List of tuples ((z, y, x), component_mask).
    """
    mask_np = (target_mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    if mask_np.sum() == 0:
        return []

    labeled_array, num_features = label(mask_np)
    if num_features == 0:
        return []

    # Component label ids ordered by descending voxel count (bincount index 0 == background).
    counts = np.bincount(labeled_array.ravel(), minlength=num_features + 1)
    comp_ids = (np.argsort(counts[1:])[::-1] + 1).tolist()
    if max_components is not None and len(comp_ids) > max_components:
        comp_ids = comp_ids[:max_components]

    results = []
    device = target_mask.device

    for comp_id in comp_ids:
        comp_indices = np.argwhere(labeled_array == comp_id)
        if len(comp_indices) == 0:
            continue

        # Interior anchor: the raster-order median voxel of the component. Guaranteed
        # inside the component but not a medial-axis point; multi-anchor skeleton
        # sampling for large / diffuse lesions is Phase 4 Exp 003.
        med_idx = len(comp_indices) // 2
        anchor_coord = tuple(int(x) for x in comp_indices[med_idx])

        # Isolated binary mask for this specific component
        comp_mask_tensor = torch.from_numpy(labeled_array == comp_id).to(device=device, dtype=torch.float32)
        results.append((anchor_coord, comp_mask_tensor))

    return results


def sample_unannotated_anchors(
    embeddings: torch.Tensor,
    target_mask: torch.Tensor,
    delta_var: float = 0.5,
    num_anchors: int = 8,
    volume_threshold: float = 0.05,
    roi_dilation_voxels: int = 2,
) -> List[Tuple[int, int, int]]:
    """
    Signature:
        sample_unannotated_anchors(embeddings: torch.Tensor, target_mask: torch.Tensor, delta_var: float = 0.5, num_anchors: int = 8, volume_threshold: float = 0.05, roi_dilation_voxels: int = 2) -> list[tuple[int, int, int]]

    Objective:
        Sample unannotated anchor voxels from the background region using iterative
        non-maximum coverage suppression (Wolny et al., CVPR 2022). Each sampled anchor
        suppresses its delta_var neighborhood to force subsequent anchors to explore
        distinct unannotated structures rather than redundant background air. The
        annotated foreground dilated by `roi_dilation_voxels` is excluded from the pool.

    Inputs:
        embeddings (torch.Tensor): Metric embeddings tensor of shape (D, Z, Y, X).
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).
        delta_var (float): Intra-cluster distance margin for suppression (default 0.5).
        num_anchors (int): Maximum number of unlabeled anchors to sample (default 8).
        volume_threshold (float): Stopping fraction of uncovered candidate voxels (default 0.05).
        roi_dilation_voxels (int): Dilation radius of the excluded lesion band (default 2, 0 disables).

    Outputs:
        list[tuple[int, int, int]]: List of (z, y, x) anchor coordinates in unannotated regions.
    """
    unlabeled_mask = _confirmed_background_mask(target_mask, roi_dilation_voxels)

    # Materialize the candidate coordinates ONCE. The previous form re-ran torch.nonzero over the
    # full (Z, Y, X) volume on every anchor iteration, re-allocating an (N, 3) int64 tensor
    # (~170 MB at 192^3) each time. Suppression state is now a boolean vector over this fixed
    # array, so each iteration touches only an (N,) bool and an (M,) index tensor.
    candidate_coords = torch.nonzero(unlabeled_mask, as_tuple=False)  # (N, 3)
    total_unlabeled = candidate_coords.shape[0]
    if total_unlabeled == 0:
        return []

    cz, cy, cx = candidate_coords[:, 0], candidate_coords[:, 1], candidate_coords[:, 2]
    active = torch.ones(total_unlabeled, dtype=torch.bool, device=candidate_coords.device)

    anchors = []
    delta_var_sq = delta_var ** 2

    for _ in range(num_anchors):
        active_idx = torch.nonzero(active, as_tuple=False).flatten()  # (M,)
        if active_idx.numel() < volume_threshold * total_unlabeled:
            break

        # Sample candidate anchor at random from the active unannotated pool
        rand_pos = torch.randint(0, active_idx.numel(), (1,), device=active_idx.device)
        az, ay, ax = (int(v) for v in candidate_coords[active_idx[rand_pos[0]]].tolist())
        anchors.append((az, ay, ax))

        # Vectorized neighborhood suppression on unit sphere: dist_sq < delta_var_sq
        dist_sq = _unit_sphere_dist_sq(embeddings[:, az, ay, ax], embeddings, "d, dzyx -> zyx")

        # Suppress the covered neighborhood from the subsequent candidate pool
        active &= ~(dist_sq[cz, cy, cx] < delta_var_sq)

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
    # Cast BEFORE reducing, not after: a bf16 reduction returns a bf16 scalar, so a
    # background-dominated sum of order 1e6 keeps only 8 mantissa bits and snaps to
    # multiples of ~8192. `.sum().float()` widens one operation too late.
    intersection = (soft_mask.float() * target_mask.float()).sum()
    union = soft_mask.float().sum() + target_mask.float().sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)


def compute_unlabeled_push_loss(
    embeddings: torch.Tensor,
    instance_anchors: List[Tuple[int, int, int]],
    target_mask: torch.Tensor,
    delta_dist: float = 1.5,
    max_bg_samples: int = 10000,
    roi_dilation_voxels: int = 2,
) -> torch.Tensor:
    """
    Signature:
        compute_unlabeled_push_loss(embeddings: torch.Tensor, instance_anchors: list[tuple[int, int, int]], target_mask: torch.Tensor, delta_dist: float = 1.5, max_bg_samples: int = 10000, roi_dilation_voxels: int = 2) -> torch.Tensor

    Objective:
        Compute hinge repulsion push force between annotated instance anchors and
        unannotated background voxels (Wolny et al., CVPR 2022).
        L_unl_push = (1 / K) * sum_k mean_{i in BG} max(0, delta_dist - ||e_i - a_k||)^2.

    Inputs:
        embeddings (torch.Tensor): Metric embeddings tensor of shape (D, Z, Y, X).
        instance_anchors (list[tuple[int, int, int]]): Anchor coordinates of annotated instances.
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).
        delta_dist (float): Inter-cluster push distance margin (default 1.5).
        max_bg_samples (int): Maximum background voxels to sample for the Monte Carlo
            approximation of the background expectation (default 10000).
        roi_dilation_voxels (int): Dilation radius of the excluded lesion band (default 2, 0 disables).

    Outputs:
        torch.Tensor: Scalar unlabeled push loss.
    """
    if not instance_anchors:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    bg_mask = _confirmed_background_mask(target_mask, roi_dilation_voxels)
    bg_indices = torch.nonzero(bg_mask, as_tuple=False)  # (N_bg, 3)
    num_bg = bg_indices.shape[0]
    if num_bg == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Algorithmic & Volumetric Rationale:
    # A standard 3D volumetric patch (e.g. 192x192x192) contains over 7 million voxels.
    # Performing full-grid dense tensor broadcasting across all background voxels for every
    # anchor accumulates O(|BG| * K) uncompressed 3D activation maps in the autograd backward tape,
    # causing rapid memory scaling on multi-instance scans.
    # We apply standard Monte Carlo subsampling of background voxels (default 10,000):
    #   E_{i ~ BG}[hinge(delta_dist - ||e_i - a_k||)^2] ~= (1 / N) * sum_{j=1}^N hinge(delta_dist - ||e_j - a_k||)^2
    # This yields an unbiased estimator with identical expected gradient dynamics while reducing
    # intermediate autograd graph memory by several orders of magnitude.
    if num_bg > max_bg_samples:
        perm = torch.randperm(num_bg, device=embeddings.device)[:max_bg_samples]
        sampled_indices = bg_indices[perm]  # (S, 3)
    else:
        sampled_indices = bg_indices

    # Gather sampled background embedding vectors: (D, S)
    sz, sy, sx = sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]
    bg_embeds = embeddings[:, sz, sy, sx]  # (D, S)

    push_losses = []
    for az, ay, ax in instance_anchors:
        # min=1e-8 (not 0) keeps sqrt differentiable at coincident embeddings.
        dist_sq = _unit_sphere_dist_sq(embeddings[:, az, ay, ax], bg_embeds, "d, ds -> s")
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-8))  # (S,)
        hinged_push = torch.clamp(delta_dist - dist, min=0.0) ** 2  # (S,)
        push_losses.append(hinged_push.mean())

    return torch.stack(push_losses).mean()


def compute_spoco_total_loss(
    student_embeds: torch.Tensor,
    teacher_embeds: torch.Tensor,
    targets: torch.Tensor,
    is_absent: Optional[torch.Tensor] = None,
    delta_var: float = 0.5,
    delta_dist: float = 1.5,
    pmaps_threshold: float = 0.5,
    sigma: Optional[float] = None,
    w_con: float = 0.1,
    w_unl_push: float = 0.1,
    w_neg: float = 1.0,
    num_unlabeled_anchors: int = 8,
    volume_threshold: float = 0.05,
    negative_supervision: bool = True,
    max_annotated_components: int = 8,
    roi_dilation_voxels: int = 2,
    union_target: bool = True,
    return_details: bool = False,
) -> Any:
    """
    Signature:
        compute_spoco_total_loss(student_embeds: torch.Tensor, teacher_embeds: torch.Tensor, targets: torch.Tensor, is_absent: torch.Tensor | None = None, delta_var: float = 0.5, delta_dist: float = 1.5, pmaps_threshold: float = 0.5, sigma: float | None = None, w_con: float = 0.1, w_unl_push: float = 0.1, w_neg: float = 1.0, num_unlabeled_anchors: int = 8, volume_threshold: float = 0.05, negative_supervision: bool = True, max_annotated_components: int = 8, roi_dilation_voxels: int = 2, union_target: bool = True, return_details: bool = False) -> tuple

    Objective:
        Compute full 3D SPOCO loss: Instance-level soft Dice on annotated objects (L_obj),
        Unannotated Anchor Consistency (L_con) against EMA Teacher with iterative coverage suppression,
        Unlabeled Background Push repulsion (L_unl_push), and null-target supervision on confirmed
        absent findings (L_neg).

        L_obj and L_neg are reduced SEPARATELY and combined with an explicit w_neg. They used to
        share one list and one mean, but they are not the same quantity -- L_obj terms are soft
        Dice in [0, 1] while L_neg terms are raw Gaussian soft-mask mass (order 1e-3). Averaging
        them together made the balance between positive supervision and negative suppression a
        function of how many terms each happened to contribute in a given batch, which swings with
        how many positive prompts survived the random crop and how many components each held. It
        also made the training L_obj incomparable to the validation L_obj, since validation samples
        no negative prompts at all.

        Each (b, n) prompt is routed by whether the finding is genuinely absent from the
        scan (`is_absent`), NOT by whether its target happens to be empty in the crop:
          - absent finding                -> negative supervision (penalize soft-mask mass);
          - present finding, empty crop   -> skipped entirely (no signal, no penalty);
          - present finding, non-empty     -> L_obj + L_unl_push + L_con.
        When `is_absent` is None, the legacy behavior is used (empty target => absent).

    Inputs:
        student_embeds (torch.Tensor): Student embeddings of shape (B, N, D, Z, Y, X).
        teacher_embeds (torch.Tensor): EMA Teacher embeddings of shape (B, N, D, Z, Y, X).
        targets (torch.Tensor): Ground truth target tensor of shape (B, N, Z, Y, X).
        is_absent (torch.Tensor | None): Bool tensor of shape (B, N), True where the prompt
            is a confirmed-absent (sampled negative) finding. None => infer from empty target.
        delta_var (float): Intra-cluster pull margin (default 0.5).
        delta_dist (float): Inter-cluster push margin (default 1.5).
        pmaps_threshold (float): Soft mask probability cutoff at distance delta_var (default 0.5).
        sigma (float | None): Optional legacy sigma override.
        w_con (float): Consistency loss weight for unannotated anchors (default 0.1).
        w_unl_push (float): Unlabeled background push weight (default 0.1).
        w_neg (float): Weight on the confirmed-absent null-target term L_neg (default 1.0). The
            default is a starting point rather than a calibrated value: L_neg is roughly three
            orders of magnitude smaller than a Dice term, so w_neg = 1.0 makes negative
            supervision far weaker than the old implicit weighting. Log L_neg and tune from there.
        num_unlabeled_anchors (int): Maximum unlabeled anchors sampled per finding (default 8).
        volume_threshold (float): Stopping fraction for unlabeled coverage (default 0.05).
        negative_supervision (bool): Penalize false-positive anchor masks on absent findings (default True).
        max_annotated_components (int): Cap on connected components used for anchor *placement*
            per finding, largest first (default 8); bounds autograd-tape memory on multi-focal
            crops. Under `union_target=True` this no longer caps how many voxels receive
            supervision (every anchor is scored against the full finding mask), only how many
            redundant interior anchors are placed.
        roi_dilation_voxels (int): Lesion-band dilation (voxels) excluded from the push and
            unannotated-anchor "confirmed background" pools (default 2, 0 disables).
        union_target (bool): ReXGroundingCT's official evaluation is per-finding volumetric
            Dice/Hit-Rate against the union of all instances of that finding -- it carries no
            instance-matching term, so nothing in the challenge rewards telling instance 3 apart
            from instance 7 of the same pathology. When True (default), every sampled annotated
            anchor's Dice term is evaluated against the finding's full target mask (all
            components), not just its own isolated component: this (a) removes the false-positive
            penalty L_obj previously placed on an anchor's soft mask bleeding into a *different*
            true instance of the same finding -- a bleed that L_con is separately trying to
            encourage -- and (b) fixes a real coverage gap, since `max_annotated_components`
            previously left every component beyond the cap with zero gradient of either kind
            (too small for direct supervision, yet excluded from the unlabeled pool because it is
            genuinely annotated); categories such as micronodules (mean 12.14 components/finding)
            and septal thickening (7.82) already exceed the cap of 8 routinely. Set False to
            recover the original per-instance behavior (each anchor's Dice scored only against
            its own connected component) for an explicit instance-vs-union ablation.
        return_details (bool): If True, returns (total_loss, l_obj, l_con, l_push, l_neg). Otherwise (total_loss, l_obj, l_con).

    Outputs:
        tuple[torch.Tensor, ...]: (total_loss, l_obj, l_con) or, with return_details,
        (total_loss, l_obj, l_con, l_push, l_neg).
    """
    B, N_prompts, D, Z, Y, X = student_embeds.shape
    device = student_embeds.device

    loss_obj_list = []
    loss_con_list = []
    loss_push_list = []
    loss_neg_list = []

    for b in range(B):
        for n in range(N_prompts):
            s_embed = student_embeds[b, n]  # (D, Z, Y, X)
            t_embed = teacher_embeds[b, n]  # (D, Z, Y, X)
            tgt = targets[b, n]             # (Z, Y, X)

            tgt_empty = (tgt.sum() == 0)
            if is_absent is not None:
                finding_absent = bool(is_absent[b, n])
            else:
                finding_absent = bool(tgt_empty)

            # Present finding whose lesion fell outside this random crop: no usable
            # signal in the patch, and penalizing it as background is the label-noise
            # bug this flag fixes. Skip the prompt entirely.
            if not finding_absent and tgt_empty:
                continue

            # 1. Supervised Instance Soft Dice on Annotated Anchors (Connected-Component Multi-Instance)
            if not finding_absent:
                annotated_items = sample_annotated_anchors(tgt, max_components=max_annotated_components)
                instance_anchor_coords = []

                if annotated_items:
                    anchor_coords = [item[0] for item in annotated_items]
                    comp_masks = [item[1] for item in annotated_items]
                    instance_anchor_coords = anchor_coords

                    anchor_tensor = torch.tensor(anchor_coords, device=device, dtype=torch.long)
                    soft_masks = compute_gaussian_soft_mask(
                        embeddings=s_embed,
                        anchor_coords=anchor_tensor,
                        delta_var=delta_var,
                        pmaps_threshold=pmaps_threshold,
                        sigma=sigma,
                    )  # (K, Z, Y, X)

                    for k in range(len(annotated_items)):
                        # Union targeting (default): every anchor's soft mask is scored against
                        # the WHOLE finding mask (tgt), not just the component it was seeded
                        # from, so a bleed into a different true instance of the same finding
                        # is rewarded rather than penalized, and every annotated voxel gets
                        # gradient regardless of the max_annotated_components cap above.
                        dice_target = tgt if union_target else comp_masks[k]
                        l_dice = compute_instance_dice_loss(soft_masks[k], dice_target)
                        loss_obj_list.append(l_dice)

                    # 2. Unlabeled Push Repulsion Force: push annotated anchors away from background
                    if w_unl_push > 0:
                        l_push = compute_unlabeled_push_loss(
                            embeddings=s_embed,
                            instance_anchors=instance_anchor_coords,
                            target_mask=tgt,
                            delta_dist=delta_dist,
                            roi_dilation_voxels=roi_dilation_voxels,
                        )
                        loss_push_list.append(l_push)

                # 3. Unannotated Anchor Consistency with Iterative Coverage Suppression
                unlabeled_anchors = sample_unannotated_anchors(
                    embeddings=s_embed.detach(),
                    target_mask=tgt,
                    delta_var=delta_var,
                    num_anchors=num_unlabeled_anchors,
                    volume_threshold=volume_threshold,
                    roi_dilation_voxels=roi_dilation_voxels,
                )
                if unlabeled_anchors:
                    u_anchor_tensor = torch.tensor(unlabeled_anchors, device=device, dtype=torch.long)
                    s_u_soft = compute_gaussian_soft_mask(
                        embeddings=s_embed,
                        anchor_coords=u_anchor_tensor,
                        delta_var=delta_var,
                        pmaps_threshold=pmaps_threshold,
                        sigma=sigma,
                    )
                    t_u_soft = compute_gaussian_soft_mask(
                        embeddings=t_embed,
                        anchor_coords=u_anchor_tensor,
                        delta_var=delta_var,
                        pmaps_threshold=pmaps_threshold,
                        sigma=sigma,
                    )

                    for u in range(len(unlabeled_anchors)):
                        l_con = compute_instance_dice_loss(s_u_soft[u], t_u_soft[u].detach())
                        loss_con_list.append(l_con)
            else:
                # Confirmed absent negative finding: penalize mass for absent structures
                if negative_supervision:
                    neg_anchors = sample_unannotated_anchors(
                        embeddings=s_embed.detach(),
                        target_mask=tgt,
                        delta_var=delta_var,
                        num_anchors=min(4, num_unlabeled_anchors),
                        volume_threshold=volume_threshold,
                        roi_dilation_voxels=roi_dilation_voxels,
                    )
                    if neg_anchors:
                        neg_anchor_tensor = torch.tensor(neg_anchors, device=device, dtype=torch.long)
                        s_neg_soft = compute_gaussian_soft_mask(
                            embeddings=s_embed,
                            anchor_coords=neg_anchor_tensor,
                            delta_var=delta_var,
                            pmaps_threshold=pmaps_threshold,
                            sigma=sigma,
                        )
                        loss_neg_list.append(s_neg_soft.mean())

    # A zero that still carries a grad_fn back to `student_embeds`, so that a step in
    # which every prompt was skipped (e.g. a crop where all positives fell out of frame)
    # produces a valid no-op backward instead of a bare requires_grad leaf.
    zero = student_embeds.sum() * 0.0

    loss_obj = torch.stack(loss_obj_list).mean() if loss_obj_list else zero
    loss_con = torch.stack(loss_con_list).mean() if loss_con_list else zero
    loss_push = torch.stack(loss_push_list).mean() if loss_push_list else zero
    loss_neg = torch.stack(loss_neg_list).mean() if loss_neg_list else zero

    total_loss = loss_obj + (w_con * loss_con) + (w_unl_push * loss_push) + (w_neg * loss_neg)

    if return_details:
        return total_loss, loss_obj, loss_con, loss_push, loss_neg
    return total_loss, loss_obj, loss_con
