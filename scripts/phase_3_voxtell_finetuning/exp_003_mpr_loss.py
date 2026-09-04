"""
===============================================================================
SCRIPT:         VoxTell Multi-Planar Projection Regularization (MPR) Fine-Tuning
PHASE:          Phase 3 — Model Fine-Tuning & Loss Hypotheses Benchmarking
LOCATION:       scripts/phase_3_voxtell_finetuning/exp_003_mpr_loss.py
OBJECTIVE:      Fine-tune VoxTell using 3D Multi-Planar Projection Regularization (MPR)
                consistency loss with exponential ramp-up. Builds on the Student-Teacher
                Mean Teacher framework of Exp 002, but replaces the 3D voxel-wise MSE
                consistency term with the MPR loss of Gao et al., 2022 (SOUSA): for each
                of Nrot random 3D rotations, the unannotated-background predictions of the
                Student and Teacher are max-projected onto the axial, coronal and sagittal
                planes and compared with a soft Dice loss, then averaged over rotations.
                Max-projection turns a small dispersed false positive into an isolated 2D
                peak that Dice penalises strongly, whereas the same voxel is nearly
                invisible to a 3D MSE — the failure mode expected under partial annotation.
USAGE:          Single-GPU: python scripts/phase_3_voxtell_finetuning/exp_003_mpr_loss.py
                Multi-GPU:  torchrun --nproc_per_node=N scripts/phase_3_voxtell_finetuning/exp_003_mpr_loss.py
===============================================================================
"""

import os
import sys
import math
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure proper GPU isolation before loading torch
load_dotenv(override=False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# MONAI's Randomizable.R is a CLASS variable shared by every transform and seeded once at import;
# plain torch DataLoader never reseeds it per worker, so all workers otherwise replay one
# identical crop/flip stream. This reseeds it from worker_info.seed. Preferred over
# monai.data.DataLoader, which would also swap collate_fn to list_data_collate.
from monai.data.utils import worker_init_fn as monai_worker_init_fn
from tqdm import tqdm

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR,
    TEXT_CACHE_DIR, LOGS_DIR, MODEL_DIR
)

# Import Phase 3 Shared Common Infrastructure
from scripts.phase_3_voxtell_finetuning.common import (
    init_distributed,
    cleanup_distributed,
    setup_distributed_logger,
    get_unwrapped_state_dict,
    ddp_step,
    ReXDataset,
    resolve_num_workers,
    load_voxtell_model
)


# Setup experiment logging directory
EXP_LOG_DIR = LOGS_DIR / "phase_3_voxtell_finetuning" / "exp_003_mpr_loss"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("exp_003_mpr_loss")


def compute_roi_mask(seg_target: torch.Tensor, kernel_size: int = 11, padding: int = 5) -> torch.Tensor:
    """
    Signature:
        compute_roi_mask(seg_target: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor

    Objective:
        Generate a 3D Region of Interest (ROI) binary mask by applying 3D max-pooling dilation to targets.

    Inputs:
        seg_target (torch.Tensor): Binary target segmentation tensor of shape (B, F, Z, Y, X).
        kernel_size (int): 3D max pooling kernel size. Default 11.
        padding (int): Padding for max pooling. Default 5.

    Outputs:
        torch.Tensor: Dilated boolean mask tensor of shape (B, F, Z, Y, X).
    """
    dilated = F.max_pool3d(seg_target.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return dilated > 0


def compute_roi_masked_loss(logits: torch.Tensor, targets: torch.Tensor, roi_mask: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
    """
    Signature:
        compute_roi_masked_loss(logits: torch.Tensor, targets: torch.Tensor, roi_mask: torch.Tensor, pos_weight: float) -> torch.Tensor

    Objective:
        Compute ROI-Masked Supervised Loss (BCE + Dice) strictly confined within the dilated ROI mask.

    Inputs:
        logits (torch.Tensor): Pre-sigmoid logit tensor of shape (B, F, Z, Y, X).
        targets (torch.Tensor): Ground truth target tensor of shape (B, F, Z, Y, X).
        roi_mask (torch.Tensor): Dilated boolean ROI mask tensor.
        pos_weight (float): Positive class weight for BCE loss. Default 10.0.

    Outputs:
        torch.Tensor: Scalar loss tensor (BCE_masked + Dice_masked).
    """
    dtype = logits.dtype
    logits_clamped = torch.clamp(logits.float(), min=-30.0, max=30.0).to(dtype=dtype)
    # 1. BCE Loss confined to dilated ROI mask with class-weighted positives
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=dtype)
    bce = F.binary_cross_entropy_with_logits(logits_clamped, targets.to(dtype=dtype), pos_weight=pos_weight_tensor, reduction='none')
    bce_masked = (bce * roi_mask.to(dtype=dtype)).sum().float() / (roi_mask.to(dtype=dtype).sum().float() + 1e-6)
    
    # 2. Dice Loss confined to dilated ROI mask
    probs = torch.sigmoid(logits_clamped)
    probs_masked = probs * roi_mask.to(dtype=dtype)
    targets_masked = targets.to(dtype=dtype) * roi_mask.to(dtype=dtype)
    
    intersection = (probs_masked * targets_masked).sum(dim=(2, 3, 4)).float()
    union = probs_masked.sum(dim=(2, 3, 4)).float() + targets_masked.sum(dim=(2, 3, 4)).float()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

    # Average Dice only over channels that actually carry an ROI. A channel whose ROI mask is
    # empty scores 1 - (0 + 1e-6)/(0 + 1e-6) = exactly 0 and, counted in the denominator, silently
    # scales the real supervised Dice by the fraction of prompts that caught foreground in this
    # crop. That fraction swings item to item, so it is a moving weight on the loss rather than a
    # constant. Without this, excluding a prompt by emptying its ROI would not actually exclude it.
    roi_per_channel = roi_mask.sum(dim=(2, 3, 4))  # (B, F)
    active_channels = roi_per_channel > 0
    if bool(active_channels.any()):
        dice_term = dice[active_channels].mean()
    else:
        dice_term = logits.sum() * 0.0  # grad-carrying zero: keeps backward valid on an all-empty batch

    return bce_masked + dice_term



def _random_rotation_matrix(device: torch.device, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Signature:
        _random_rotation_matrix(device: torch.device, generator: torch.Generator) -> torch.Tensor

    Objective:
        Draw a uniformly random 3D rotation as a 3x3 matrix, built from a random unit axis and an
        angle sampled uniformly in [-pi, pi] (Rodrigues' formula). Used to rotate the background
        prediction volumes before max-projection in the MPR consistency loss (Gao et al., 2022).

    Inputs:
        device (torch.device): Device on which to allocate the matrix.
        generator (torch.Generator): Optional RNG for reproducibility. Default None (global RNG).

    Outputs:
        torch.Tensor: Rotation matrix of shape (3, 3), float32.
    """
    axis = torch.randn(3, device=device, dtype=torch.float32, generator=generator)
    axis = axis / (axis.norm() + 1e-8)
    angle = (torch.rand(1, device=device, dtype=torch.float32, generator=generator) * 2.0 - 1.0) * math.pi
    x, y, z = axis[0], axis[1], axis[2]
    c = torch.cos(angle).squeeze()
    s = torch.sin(angle).squeeze()
    C = 1.0 - c
    rot = torch.stack([
        torch.stack([c + x * x * C,     x * y * C - z * s, x * z * C + y * s]),
        torch.stack([y * x * C + z * s, c + y * y * C,     y * z * C - x * s]),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C]),
    ])
    return rot


def _rotate_volume(vol: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """
    Signature:
        _rotate_volume(vol: torch.Tensor, rot: torch.Tensor) -> torch.Tensor

    Objective:
        Apply an arbitrary 3D rotation to a (B, F, Z, Y, X) volume via trilinear grid sampling,
        keeping the output grid identical to the input grid (out-of-volume samples padded with zeros).
        Computed in float32 for numerical stability under bf16 autocast.

    Inputs:
        vol (torch.Tensor): Volume tensor of shape (B, F, Z, Y, X).
        rot (torch.Tensor): Rotation matrix of shape (3, 3).

    Outputs:
        torch.Tensor: Rotated volume, same shape and dtype as the float32-cast input.
    """
    b = vol.shape[0]
    theta = torch.zeros(b, 3, 4, device=vol.device, dtype=torch.float32)
    theta[:, :3, :3] = rot.to(torch.float32).unsqueeze(0).expand(b, -1, -1)
    grid = F.affine_grid(theta, list(vol.shape), align_corners=False)
    return F.grid_sample(vol.float(), grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def _triplanar_projection_loss(bg_student: torch.Tensor, bg_teacher: torch.Tensor, projection_loss: str) -> torch.Tensor:
    """
    Signature:
        _triplanar_projection_loss(bg_student: torch.Tensor, bg_teacher: torch.Tensor, projection_loss: str) -> torch.Tensor

    Objective:
        Max-project the (masked) Student and Teacher background predictions onto the axial (Z),
        coronal (Y) and sagittal (X) planes and compare the projections, averaging the three planes.
        The Teacher projections are detached (no gradient flows into the EMA network).

    Inputs:
        bg_student (torch.Tensor): Student background prediction (B, F, Z, Y, X), gradient-carrying.
        bg_teacher (torch.Tensor): Teacher background prediction (B, F, Z, Y, X).
        projection_loss (str): 'dice' for soft Dice between projections (Gao et al., 2022), or 'mse'.

    Outputs:
        torch.Tensor: Scalar loss averaged over the 3 orthogonal projection planes.
    """
    total = bg_student.new_zeros(())
    for dim in (2, 3, 4):
        p_s = torch.max(bg_student, dim=dim)[0].float()
        p_t = torch.max(bg_teacher, dim=dim)[0].float().detach()
        if projection_loss == "mse":
            total = total + F.mse_loss(p_s, p_t)
        else:
            intersection = (p_s * p_t).sum()
            denom = p_s.sum() + p_t.sum()
            total = total + (1.0 - (2.0 * intersection + 1e-6) / (denom + 1e-6))
    return total / 3.0


def compute_mpr_consistency_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    roi_mask: torch.Tensor,
    num_rotations: int = 4,
    rotation_mode: str = "affine",
    projection_loss: str = "dice",
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Signature:
        compute_mpr_consistency_loss(student_probs: torch.Tensor, teacher_probs: torch.Tensor, roi_mask: torch.Tensor, num_rotations: int, rotation_mode: str, projection_loss: str, valid_mask: torch.Tensor | None = None) -> torch.Tensor

    Objective:
        Compute the Multi-Planar Projection (MPR) consistency loss of Gao et al., 2022 across the
        unannotated background region. For rotation 0 (identity) and each of the remaining
        num_rotations - 1 random 3D rotations, the masked Student and Teacher predictions are
        max-projected onto the axial, coronal and sagittal planes and compared (soft Dice by
        default); the per-rotation losses are averaged. Rotations turn dispersed false positives
        into isolated 2D peaks that projection Dice penalises strongly, unlike a 3D voxel-wise MSE.

    Inputs:
        student_probs (torch.Tensor): Student sigmoid probabilities (B, F, Z, Y, X), gradient-carrying.
        teacher_probs (torch.Tensor): Teacher sigmoid probabilities (B, F, Z, Y, X).
        roi_mask (torch.Tensor): Dilated boolean ROI mask (B, F, Z, Y, X); its complement is the
            unannotated background over which the loss is computed.
        num_rotations (int): Total projections including the identity. 1 reproduces the plain
            axis-aligned tri-planar loss. Default 4. Gao et al. report an optimum around 9.
        rotation_mode (str): 'affine' for arbitrary-angle trilinear rotation, 'none' for identity only.
        projection_loss (str): 'dice' (Gao et al., 2022) or 'mse'.
        valid_mask (torch.Tensor | None): Optional broadcastable bool mask of prompts that carry a
            usable signal. Required whenever a prompt is dropped by emptying its ROI: the
            background is the COMPLEMENT of the ROI, so a dropped prompt would otherwise receive
            full-volume projection supervision -- the exact opposite of being skipped.

    Outputs:
        torch.Tensor: Scalar MPR consistency loss.
    """
    dtype = student_probs.dtype
    bg_bool = ~roi_mask
    if valid_mask is not None:
        bg_bool = bg_bool & valid_mask
    bg_mask = bg_bool.to(dtype=dtype)
    bg_student = student_probs * bg_mask
    bg_teacher = (teacher_probs * bg_mask).detach()

    n_rot = max(1, int(num_rotations))
    total = bg_student.new_zeros(())
    # Rotation 0 is always the identity so the loss degrades gracefully to the axis-aligned
    # tri-planar projection when num_rotations == 1 or rotation_mode == 'none'.
    total = total + _triplanar_projection_loss(bg_student, bg_teacher, projection_loss)

    if rotation_mode == "affine":
        for _ in range(n_rot - 1):
            rot = _random_rotation_matrix(bg_student.device)
            rot_student = _rotate_volume(bg_student, rot).to(dtype=dtype)
            rot_teacher = _rotate_volume(bg_teacher, rot).to(dtype=dtype)
            total = total + _triplanar_projection_loss(rot_student, rot_teacher, projection_loss)
        return total / float(n_rot)

    return total


def get_mpr_rampup_weight(epoch: int, max_epochs: int, max_weight: float = 0.5) -> float:
    """
    Signature:
        get_mpr_rampup_weight(epoch: int, max_epochs: int, max_weight: float) -> float

    Objective:
        Compute Gaussian exponential ramp-up consistency weight schedule gamma(t) = exp(-5 * (1 - t/T)^2) (Gao et al., 2022).

    Inputs:
        epoch (int): Current epoch index (1-indexed).
        max_epochs (int): Total number of training epochs (T).
        max_weight (float): Maximum consistency weight ceiling. Default 0.5.

    Outputs:
        float: Computed ramp-up consistency loss weight.
    """
    t = float(epoch)
    T = float(max_epochs)
    if T <= 1.0:
        return max_weight
    ratio = max(0.0, 1.0 - (t / T))
    gamma = math.exp(-5.0 * (ratio ** 2))
    return max_weight * gamma


@torch.no_grad()
def update_ema_variables(student_model: nn.Module, teacher_model: nn.Module, alpha: float) -> None:
    """
    Signature:
        update_ema_variables(student_model: nn.Module, teacher_model: nn.Module, alpha: float) -> None

    Objective:
        Update Teacher network parameters via Exponential Moving Average (EMA) from Student network.

    Inputs:
        student_model (nn.Module): Active Student network model (handles DDP wrapped or unwrapped).
        teacher_model (nn.Module): Target Teacher network model.
        alpha (float): EMA decay weighting factor (e.g. 0.999).

    Outputs:
        None (In-place parameter update).
    """
    src_model = student_model.module if hasattr(student_model, "module") else student_model
    for teacher_param, student_param in zip(teacher_model.parameters(), src_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
        
    for teacher_buffer, student_buffer in zip(teacher_model.buffers(), src_model.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command line arguments for Exp 003 MPR loss fine-tuning.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VoxTell MPR Consistency Loss Fine-Tuning Pipeline (Exp 003)")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json metadata")
    parser.add_argument("--img_dir", type=str, default=str(RAW_IMAGES_DIR), help="Path to raw CT images directory")
    parser.add_argument("--seg_dir", type=str, default=str(RAW_MASKS_DIR), help="Path to raw CT segmentations directory")
    parser.add_argument("--cache_dir", type=str, default=str(TEXT_CACHE_DIR), help="Path to Qwen text embeddings cache directory")
    parser.add_argument("--model_dir", type=str, default=str(MODEL_DIR), help="Path to pre-trained voxtell_v1.1 checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=str(EXP_LOG_DIR), help="Directory to save fine-tuned model checkpoints")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--alpha", type=float, default=0.999, help="Teacher EMA momentum decay parameter (default: 0.999)")
    parser.add_argument("--pos_weight", type=float, default=10.0, help="Positive class weight for BCE loss inside ROIs (default: 10.0)")
    parser.add_argument("--max_mpr_weight", type=float, default=0.5, help="Maximum MPR consistency loss weight (default: 0.5)")
    parser.add_argument("--mpr_num_rotations", type=int, default=4, help="Number of MPR projections incl. identity; 1 = plain axis-aligned tri-planar (default: 4, Gao et al. optimum ~9)")
    parser.add_argument("--mpr_rotation_mode", type=str, default="affine", choices=["affine", "none"], help="MPR rotation sampling: 'affine' arbitrary-angle trilinear, 'none' identity only (default: affine)")
    parser.add_argument("--mpr_projection_loss", type=str, default="dice", choices=["dice", "mse"], help="Loss between Student/Teacher 2D projections (default: dice, per Gao et al. 2022)")
    parser.add_argument("--patch_size", type=int, default=192, help="Patch size for MONAI spatial crop (default: 192)")
    parser.add_argument("--num_pos_prompts", type=int, default=2, help="Number of positive prompts per volume (default: 2)")
    parser.add_argument("--num_neg_prompts", type=int, default=1, help="Number of negative prompts per volume (default: 1)")
    parser.add_argument("--pos_ratio", type=float, default=0.85, help="Foreground patch sampling probability (default: 0.85)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device for standalone run (e.g. cuda:0)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers per GPU (default: auto-resolved based on SLURM / CPU count)")
    parser.add_argument("--use_volume_cache", action="store_true", default=False, help="Enable on-disk full-volume caching (default: False, streaming mode)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_model.pt if available")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging (default: True)")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rexgroundingct", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="exp_003_mpr_loss", help="Weights & Biases run name")
    return parser.parse_args()


def train_mpr_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    w_mpr: float,
    pos_weight: float,
    alpha: float,
    mpr_num_rotations: int = 4,
    mpr_rotation_mode: str = "affine",
    mpr_projection_loss: str = "dice",
    global_step: int = 0,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False
) -> tuple[float, float, float, int]:
    """
    Signature:
        train_mpr_epoch(student_model: nn.Module, teacher_model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: str, w_mpr: float, pos_weight: float, alpha: float, mpr_num_rotations: int, mpr_rotation_mode: str, mpr_projection_loss: str, global_step: int, rank: int, world_size: int, is_distributed: bool) -> tuple[float, float, float, int]

    Objective:
        Execute one training epoch using PU dilated ROI masked supervision + 3D Multi-Planar Projection (MPR) consistency with DDP synchronization.

    Inputs:
        student_model (nn.Module): Active Student network model.
        teacher_model (nn.Module): Target Teacher network model.
        dataloader (DataLoader): PyTorch training DataLoader.
        optimizer (Optimizer): PyTorch AdamW optimizer.
        scaler (GradScaler): AMP Gradient Scaler.
        device (str): Computation device string.
        w_mpr (float): MPR consistency loss ramp-up weight for current epoch.
        pos_weight (float): Positive class weight for BCE loss inside ROIs.
        alpha (float): EMA decay weighting factor for Teacher model update.
        mpr_num_rotations (int): Number of MPR projections including the identity. Default 4.
        mpr_rotation_mode (str): 'affine' or 'none'. Default 'affine'.
        mpr_projection_loss (str): 'dice' or 'mse'. Default 'dice'.
        global_step (int): Running global iteration counter across epochs. Default 0.
        rank (int): Process global rank. Default 0.
        world_size (int): Total number of distributed processes. Default 1.
        is_distributed (bool): Whether running in multi-GPU distributed mode. Default False.

    Outputs:
        tuple[float, float, float, int]: (avg_total_loss, avg_sup_loss, avg_mpr_loss, updated_global_step).
    """
    student_model.train()
    teacher_model.eval()
    
    total_loss_acc = 0.0
    sup_loss_acc = 0.0
    mpr_loss_acc = 0.0
    valid_batches = 0
    
    for batch in tqdm(dataloader, desc="Training Epoch (MPR Consistency)", leave=False, disable=(rank != 0)):
        images = batch['image'].to(device)
        targets = batch['seg'].to(device)
        text_embeds = batch['text_embeddings'].to(device)
        scan_id = batch.get('scan_id', ['unknown'])[0]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            # Student forward pass
            student_logits = student_model(images, text_embeds)
            student_probs = torch.sigmoid(student_logits)
            
            # Teacher forward pass (freeze gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images, text_embeds)
                teacher_probs = torch.sigmoid(teacher_logits)
                
            # Route each prompt by whether the finding is genuinely absent from the SCAN, not by
            # whether its target happens to be empty in this crop. ReXDataset already emits
            # `is_absent_finding`; the previous `targets.sum() > 0` test conflated "confirmed
            # absent" with "present, but the lesion fell outside this 192^3 window", then widened
            # that prompt's ROI to the entire volume and supervised it as background under
            # pos_weight=10 -- actively penalising the model for a lesion that is really there.
            # With pos_ratio=0.85 at least 15% of items are pure-background crops in which EVERY
            # positive prompt looks empty, and the true rate is higher because the crop is centred
            # on the union foreground, so one finding of several is typically in frame.
            has_fg = (targets.sum(dim=(2, 3, 4), keepdim=True) > 0)
            if "is_absent_finding" in batch:
                is_absent = batch["is_absent_finding"].to(device).view(*targets.shape[:2], 1, 1, 1)
            else:
                is_absent = ~has_fg  # backward-compatible fallback for batches predating the flag
            supervise = is_absent | has_fg  # present-but-out-of-crop carries no signal either way

            roi_mask = compute_roi_mask(targets, kernel_size=11, padding=5)
            # confirmed absent      -> full-volume suppression (penalise hallucinated mass)
            # present, in crop      -> dilated ROI
            # present, out of crop  -> empty ROI, i.e. dropped from the supervised loss entirely
            roi_mask = torch.where(is_absent, torch.ones_like(roi_mask), roi_mask) & supervise
            
            # Supervised loss strictly within ROI (or full volume for negative prompts)
            loss_sup = compute_roi_masked_loss(student_logits.float(), targets.float(), roi_mask, pos_weight=pos_weight)
            
            # 3D Multi-Planar Projection (MPR) consistency loss across unannotated background voxels
            loss_mpr = compute_mpr_consistency_loss(
                student_probs.float(), teacher_probs.float(), roi_mask,
                num_rotations=mpr_num_rotations,
                rotation_mode=mpr_rotation_mode,
                projection_loss=mpr_projection_loss,
                valid_mask=supervise,
            )
            
            # Combined total loss
            total_loss = loss_sup + w_mpr * loss_mpr
            
        step_ok = ddp_step(
            total_loss=total_loss,
            model=student_model,
            optimizer=optimizer,
            scaler=scaler,
            is_distributed=is_distributed,
            max_norm=1.0,
            logger=logger,
            scan_id=scan_id,
            rank=rank
        )
        
        if step_ok:
            # Synchronize Teacher parameters via EMA
            update_ema_variables(student_model, teacher_model, alpha)
            
            total_loss_acc += total_loss.item()
            sup_loss_acc += loss_sup.item()
            mpr_loss_acc += loss_mpr.item()
            valid_batches += 1
            global_step += 1


        if rank == 0:
            try:
                import wandb
                if wandb.run is not None and global_step % 5 == 0:
                    wandb.log({
                        "train/step_total_loss": total_loss.item(),
                        "train/step_sup_loss": loss_sup.item(),
                        "train/step_mpr_loss": loss_mpr.item(),
                        "train/w_mpr": w_mpr,
                        "step": global_step
                    })
            except Exception:
                pass
        
    if is_distributed:
        stats_tensor = torch.tensor([total_loss_acc, sup_loss_acc, mpr_loss_acc, float(valid_batches)], device=device, dtype=torch.float32)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        t_tot, t_sup, t_mpr, t_batches = stats_tensor.tolist()
        num_b = max(t_batches, 1.0)
        return t_tot / num_b, t_sup / num_b, t_mpr / num_b, global_step
    else:
        num_b = max(valid_batches, 1)
        return total_loss_acc / num_b, sup_loss_acc / num_b, mpr_loss_acc / num_b, global_step


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for VoxTell Multi-Planar Projection Regularization (MPR) fine-tuning execution supporting
        both standalone single-GPU and torchrun multi-GPU modes.

    Inputs:
        None

    Outputs:
        None
    """
    args = parse_args()
    is_distributed, rank, local_rank, world_size, default_device = init_distributed()
    target_device = default_device if is_distributed else args.device

    setup_distributed_logger(logger, EXP_LOG_DIR, rank)
    logger.info("Starting VoxTell MPR Fine-Tuning Pipeline (Exp 003)...")
    logger.info(f"Execution Mode: {'Distributed (DDP)' if is_distributed else 'Single-Device'} | Rank: {rank}/{world_size} | Device: {target_device}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}, Alpha: {args.alpha}, PosWeight: {args.pos_weight}, MaxMPRWeight: {args.max_mpr_weight}")
    logger.info(f"Prompts: {args.num_pos_prompts} Pos + {args.num_neg_prompts} Neg | Foreground Ratio: {args.pos_ratio}")
    logger.info(f"MPR: rotations={args.mpr_num_rotations}, mode={args.mpr_rotation_mode}, projection_loss={args.mpr_projection_loss}")
    
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Resolve server-agnostic DataLoader workers
        resolved_workers = resolve_num_workers(args.num_workers)
        logger.info(
            f"Resolved DataLoader workers: {resolved_workers} "
            f"(SLURM: {'SLURM_CPUS_PER_TASK' in os.environ}, Host CPUs: {os.cpu_count()}, Volume Cache: {args.use_volume_cache})"
        )

        # Initialize Dataset
        train_dataset = ReXDataset(
            dataset_json=args.dataset_json,
            split="train",
            img_dir=args.img_dir,
            seg_dir=args.seg_dir,
            cache_dir=args.cache_dir,
            is_train=True,
            patch_size=args.patch_size,
            num_positive_prompts=args.num_pos_prompts,
            num_negative_prompts=args.num_neg_prompts,
            pos_ratio=args.pos_ratio,
            use_volume_cache=args.use_volume_cache
        )
        
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=resolved_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(resolved_workers > 0),
                prefetch_factor=2 if resolved_workers > 0 else None,
                worker_init_fn=monai_worker_init_fn
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=resolved_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(resolved_workers > 0),
                prefetch_factor=2 if resolved_workers > 0 else None,
                worker_init_fn=monai_worker_init_fn
            )
        
        logger.info(f"Loaded training split: {len(train_dataset)} total scans.")
        
        # Instantiate Student and Teacher Models
        logger.info("Instantiating Student VoxTell network...")
        student_model = load_voxtell_model(args.model_dir, target_device, deep_supervision=False)
        
        logger.info("Instantiating Teacher VoxTell network...")
        teacher_model = load_voxtell_model(args.model_dir, target_device, deep_supervision=False)
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        # Optimizer, Scheduler, Scaler
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        
        start_epoch = 1
        best_loss = float("inf")
        best_model_path = output_dir / "best_model.pt"
        latest_model_path = output_dir / "latest_model.pt"

        if args.resume and latest_model_path.exists():
            logger.info(f"Resuming training from checkpoint: {latest_model_path}")
            checkpoint = torch.load(latest_model_path, map_location=target_device, weights_only=False)
            student_model.load_state_dict(checkpoint["student_state_dict"])
            teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            # Recover the true best loss from best_model.pt; latest_model.pt only carries the last
            # epoch's loss, which would let a worse checkpoint overwrite the best one after a resume.
            if best_model_path.exists():
                try:
                    best_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
                    if "loss" in best_ckpt:
                        best_loss = best_ckpt["loss"]
                except Exception:
                    pass
            # Fast-forward the cosine schedule so the resumed LR continues instead of restarting at epoch 1
            scheduler.last_epoch = start_epoch - 1
            logger.info(f"Successfully resumed from epoch {start_epoch}, previous best loss: {best_loss:.4f}")
        
        # Wrap student in DistributedDataParallel
        if is_distributed:
            student_model = DDP(
                student_model,
                device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True
            )

        # Initialize Weights & Biases on Rank 0
        if rank == 0 and args.wandb:
            import wandb
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    config=vars(args)
                )
                logger.info(f"Initialized Weights & Biases logging (Project: {args.wandb_project}, Run: {args.wandb_run_name})")
            except Exception as e:
                logger.warning(f"Could not initialize Weights & Biases ({e}). Proceeding without wandb logging.")
                args.wandb = False

        global_step = 0
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
                
            w_mpr = get_mpr_rampup_weight(epoch, max_epochs=args.epochs, max_weight=args.max_mpr_weight)
            
            epoch_loss, sup_loss, mpr_loss, global_step = train_mpr_epoch(
                student_model=student_model,
                teacher_model=teacher_model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=target_device,
                w_mpr=w_mpr,
                pos_weight=args.pos_weight,
                alpha=args.alpha,
                mpr_num_rotations=args.mpr_num_rotations,
                mpr_rotation_mode=args.mpr_rotation_mode,
                mpr_projection_loss=args.mpr_projection_loss,
                global_step=global_step,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            logger.info(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] — Global Avg Loss: {epoch_loss:.4f} "
                f"(Sup: {sup_loss:.4f}, MPR: {mpr_loss:.4f}, w_mpr: {w_mpr:.3f}) | LR: {current_lr:.6f}"
            )
            
            # Save checkpoints strictly on Rank 0
            if rank == 0:
                # Re-create the output directory in case it was renamed or moved mid-run
                output_dir.mkdir(parents=True, exist_ok=True)
                unwrapped_student = get_unwrapped_state_dict(student_model)
                unwrapped_teacher = get_unwrapped_state_dict(teacher_model)
                torch.save({
                    "epoch": epoch,
                    "student_state_dict": unwrapped_student,
                    "teacher_state_dict": unwrapped_teacher,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss
                }, latest_model_path)

                if math.isfinite(epoch_loss) and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_path = output_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "student_state_dict": unwrapped_student,
                        "teacher_state_dict": unwrapped_teacher,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss
                    }, best_model_path)
                    logger.info(f"Saved new best model checkpoint to: {best_model_path}")
                    
                if args.wandb:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/total_loss": epoch_loss,
                        "train/sup_loss": sup_loss,
                        "train/mpr_loss": mpr_loss,
                        "train/w_mpr": w_mpr,
                        "train/lr": current_lr,
                        "train/best_loss": best_loss
                    })

        if rank == 0 and args.wandb:
            import wandb
            wandb.finish()

        logger.info("MPR fine-tuning training complete.")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
