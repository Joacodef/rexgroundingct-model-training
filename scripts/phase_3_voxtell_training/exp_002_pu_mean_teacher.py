"""
===============================================================================
SCRIPT:         VoxTell Positive-Unlabeled (PU) Mean Teacher Fine-Tuning Pipeline
PHASE:          Phase 3 — Model Fine-Tuning & Loss Hypotheses Benchmarking
LOCATION:       scripts/phase_3_voxtell_training/exp_002_pu_mean_teacher.py
OBJECTIVE:      Fine-tune VoxTell using Positive-Unlabeled (PU) learning with 
                dilated ROI-masked supervision and Mean Teacher EMA consistency 
                regularization to resolve instance suppression bias (Wolny et al., 2022).
                Supports server-agnostic multi-GPU (DDP) and single-GPU execution.
USAGE:          Single-GPU: python scripts/phase_3_voxtell_training/exp_002_pu_mean_teacher.py
                Multi-GPU:  torchrun --nproc_per_node=N scripts/phase_3_voxtell_training/exp_002_pu_mean_teacher.py
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
from tqdm import tqdm

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    TEXT_CACHE_DIR, TMP_PREP_DIR, LOGS_DIR, MODEL_DIR
)

# Import Phase 3 Shared Common Infrastructure
from scripts.phase_3_voxtell_training.common import (
    init_distributed,
    cleanup_distributed,
    setup_distributed_logger,
    get_unwrapped_state_dict,
    ddp_step,
    ReXDataset,
    load_voxtell_model
)


# Setup experiment logging directory
EXP_LOG_DIR = LOGS_DIR / "phase_3_voxtell_training" / "exp_002_pu_mean_teacher"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("exp_002_pu_mean_teacher")


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
    
    return bce_masked + dice.mean()



def compute_unannotated_consistency_loss(student_probs: torch.Tensor, teacher_probs: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
    """
    Signature:
        compute_unannotated_consistency_loss(student_probs: torch.Tensor, teacher_probs: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor

    Objective:
        Compute Mean Squared Error (MSE) consistency loss across unannotated background voxels outside the dilated ROI.

    Inputs:
        student_probs (torch.Tensor): Student network probability predictions (B, F, Z, Y, X).
        teacher_probs (torch.Tensor): Teacher network probability predictions (B, F, Z, Y, X).
        roi_mask (torch.Tensor): Dilated boolean ROI mask tensor isolating annotated regions.

    Outputs:
        torch.Tensor: Scalar MSE consistency loss over unannotated voxels.
    """
    bg_mask = (~roi_mask).float()
    diff_sq = (student_probs - teacher_probs) ** 2
    loss_con = (diff_sq * bg_mask).sum() / (bg_mask.sum() + 1e-6)
    return loss_con


def get_consistency_weight(epoch: int, max_weight: float = 0.5, warm_up_epochs: int = 15) -> float:
    """
    Signature:
        get_consistency_weight(epoch: int, max_weight: float, warm_up_epochs: int) -> float

    Objective:
        Compute sigmoid warm-up scaling factor for consistency loss at current training epoch.

    Inputs:
        epoch (int): Current epoch index (1-indexed).
        max_weight (float): Target maximum consistency loss weight. Default 0.5.
        warm_up_epochs (int): Number of warmup epochs. Default 15.

    Outputs:
        float: Computed consistency weight scaling factor.
    """
    if warm_up_epochs <= 0 or epoch >= warm_up_epochs:
        return max_weight
    x = epoch - (warm_up_epochs / 2.0)
    sigmoid = 1.0 / (1.0 + math.exp(-2.0 * x))
    return max_weight * sigmoid


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
        Parse command line arguments for Exp 002 PU Mean Teacher fine-tuning.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VoxTell PU Mean Teacher Fine-Tuning Pipeline (Exp 002)")
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
    parser.add_argument("--max_consistency_weight", type=float, default=0.5, help="Maximum consistency loss weight (default: 0.5)")
    parser.add_argument("--consistency_warmup", type=int, default=15, help="Number of warmup epochs for consistency loss (default: 15)")
    parser.add_argument("--patch_size", type=int, default=192, help="Patch size for MONAI spatial crop (default: 192)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device for standalone run (e.g. cuda:0)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers per GPU")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_model.pt if available")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging (default: True)")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rexgroundingct", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="exp_002_pu_mean_teacher", help="Weights & Biases run name")
    return parser.parse_args()


def train_pu_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    w_con: float,
    pos_weight: float,
    alpha: float,
    global_step: int = 0,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False
) -> tuple[float, float, float, int]:
    """
    Signature:
        train_pu_epoch(student_model: nn.Module, teacher_model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: str, w_con: float, pos_weight: float, alpha: float, global_step: int, rank: int, world_size: int, is_distributed: bool) -> tuple[float, float, float, int]

    Objective:
        Execute one training epoch using PU dilated ROI masked supervision + Mean Teacher EMA consistency with DDP synchronization.

    Inputs:
        student_model (nn.Module): Active Student network model.
        teacher_model (nn.Module): Target Teacher network model.
        dataloader (DataLoader): PyTorch training DataLoader.
        optimizer (Optimizer): PyTorch AdamW optimizer.
        scaler (GradScaler): AMP Gradient Scaler.
        device (str): Computation device string.
        w_con (float): Consistency loss weight for the current epoch.
        pos_weight (float): Positive class weight for BCE loss inside ROIs.
        alpha (float): EMA decay weighting factor for Teacher model update.
        global_step (int): Running global iteration counter across epochs. Default 0.
        rank (int): Process global rank. Default 0.
        world_size (int): Total number of distributed processes. Default 1.
        is_distributed (bool): Whether running in multi-GPU distributed mode. Default False.

    Outputs:
        tuple[float, float, float, int]: (avg_total_loss, avg_sup_loss, avg_con_loss, updated_global_step).
    """
    student_model.train()
    teacher_model.eval()
    
    total_loss_acc = 0.0
    sup_loss_acc = 0.0
    con_loss_acc = 0.0
    valid_batches = 0
    
    for batch in tqdm(dataloader, desc="Training Epoch (PU Mean Teacher)", leave=False, disable=(rank != 0)):
        images = batch['image'].to(device)
        targets = batch['seg'].to(device)
        text_embeds = batch['text_embeddings'].to(device)
        scan_id = batch.get('scan_id', ['unknown'])[0]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Student forward pass
            student_logits = student_model(images, text_embeds)
            student_probs = torch.sigmoid(student_logits)
            
            # Teacher forward pass (freeze gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images, text_embeds)
                teacher_probs = torch.sigmoid(teacher_logits)
                
            # Compute dilated ROI mask
            roi_mask = compute_roi_mask(targets, kernel_size=11, padding=5)
            
            # Supervised loss strictly within ROI
            loss_sup = compute_roi_masked_loss(student_logits.float(), targets.float(), roi_mask, pos_weight=pos_weight)
            
            # Consistency regularization on unannotated voxels
            loss_con = compute_unannotated_consistency_loss(student_probs.float(), teacher_probs.float(), roi_mask)
            
            # Combined loss
            total_loss = loss_sup + w_con * loss_con
            
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
            con_loss_acc += loss_con.item()
            valid_batches += 1
            global_step += 1


        if rank == 0:
            try:
                import wandb
                if wandb.run is not None and global_step % 5 == 0:
                    wandb.log({
                        "train/step_total_loss": total_loss.item(),
                        "train/step_sup_loss": loss_sup.item(),
                        "train/step_con_loss": loss_con.item(),
                        "train/w_con": w_con,
                        "step": global_step
                    })
            except Exception:
                pass
        
    if is_distributed:
        stats_tensor = torch.tensor([total_loss_acc, sup_loss_acc, con_loss_acc, float(valid_batches)], device=device, dtype=torch.float32)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        t_tot, t_sup, t_con, t_batches = stats_tensor.tolist()
        num_b = max(t_batches, 1.0)
        return t_tot / num_b, t_sup / num_b, t_con / num_b, global_step
    else:
        num_b = max(valid_batches, 1)
        return total_loss_acc / num_b, sup_loss_acc / num_b, con_loss_acc / num_b, global_step


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for VoxTell Positive-Unlabeled (PU) Mean Teacher fine-tuning execution supporting
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
    logger.info("Starting VoxTell PU Mean Teacher Fine-Tuning Pipeline (Exp 002)...")
    logger.info(f"Execution Mode: {'Distributed (DDP)' if is_distributed else 'Single-Device'} | Rank: {rank}/{world_size} | Device: {target_device}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}, Alpha: {args.alpha}, PosWeight: {args.pos_weight}, PatchSize: {args.patch_size}")
    
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize Dataset
        train_dataset = ReXDataset(
            dataset_json=args.dataset_json,
            split="train",
            img_dir=args.img_dir,
            seg_dir=args.seg_dir,
            cache_dir=args.cache_dir,
            is_train=True,
            patch_size=args.patch_size
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
                num_workers=args.num_workers,
                pin_memory=True
            )
        else:
            train_sampler = None
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
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
        scaler = torch.amp.GradScaler('cuda')
        
        start_epoch = 1
        best_loss = float("inf")
        latest_model_path = output_dir / "latest_model.pt"

        if args.resume and latest_model_path.exists():
            logger.info(f"Resuming training from checkpoint: {latest_model_path}")
            checkpoint = torch.load(latest_model_path, map_location=target_device, weights_only=False)
            student_model.load_state_dict(checkpoint["student_state_dict"])
            teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "loss" in checkpoint:
                best_loss = checkpoint["loss"]
            start_epoch = checkpoint.get("epoch", 0) + 1
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
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            logger.info(f"Initialized Weights & Biases logging (Project: {args.wandb_project}, Run: {args.wandb_run_name})")

        global_step = 0
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
                
            w_con = get_consistency_weight(epoch, max_weight=args.max_consistency_weight, warm_up_epochs=args.consistency_warmup)
            
            epoch_loss, sup_loss, con_loss, global_step = train_pu_epoch(
                student_model=student_model,
                teacher_model=teacher_model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=target_device,
                w_con=w_con,
                pos_weight=args.pos_weight,
                alpha=args.alpha,
                global_step=global_step,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            logger.info(
                f"Epoch [{epoch:02d}/{args.epochs:02d}] — Global Avg Loss: {epoch_loss:.4f} "
                f"(Sup: {sup_loss:.4f}, Con: {con_loss:.4f}, w_con: {w_con:.3f}) | LR: {current_lr:.6f}"
            )
            
            # Save checkpoints strictly on Rank 0
            if rank == 0:
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
                        "train/con_loss": con_loss,
                        "train/w_con": w_con,
                        "train/lr": current_lr,
                        "train/best_loss": best_loss
                    })

        if rank == 0 and args.wandb:
            import wandb
            wandb.finish()

        logger.info("PU Mean Teacher fine-tuning training complete.")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
