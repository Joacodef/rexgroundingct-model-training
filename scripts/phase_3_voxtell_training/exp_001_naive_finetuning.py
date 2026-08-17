"""
===============================================================================
SCRIPT:         VoxTell Naïve Supervised Fine-Tuning Baseline
PHASE:          Phase 3 — Model Fine-Tuning & Adaptation
LOCATION:       scripts/phase_3_voxtell_training/exp_001_naive_finetuning.py
OBJECTIVE:      Naïve supervised fine-tuning of VoxTell baseline (voxtell_v1.1) 
                using standard BCE + Dice loss on 3D CT volume patches from the 
                2,992-scan training split. Supports server-agnostic multi-GPU (DDP)
                and single-GPU execution. Establishes the supervised lower bound.
USAGE:          Single-GPU: python scripts/phase_3_voxtell_training/exp_001_naive_finetuning.py
                Multi-GPU:  torchrun --nproc_per_node=N scripts/phase_3_voxtell_training/exp_001_naive_finetuning.py
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from monai.losses import DiceLoss

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
    ReXDataset,
    load_voxtell_model
)

# Setup experiment logging directory
EXP_LOG_DIR = LOGS_DIR / "phase_3_voxtell_training" / "exp_001_naive_finetuning"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("exp_001_naive_finetuning")


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command line arguments for naïve fine-tuning execution.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VoxTell Naïve Supervised Fine-Tuning Baseline")
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
    parser.add_argument("--patch_size", type=int, default=192, help="Patch size for MONAI spatial crop")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device for standalone run (e.g. cuda:0)")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers per GPU")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_model.pt if available")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging (default: True)")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rexgroundingct", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="exp_001_naive_finetuning", help="Weights & Biases run name")
    return parser.parse_args()


def train_naive_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scaler: torch.amp.GradScaler, 
    device: str, 
    bce_criterion: nn.Module, 
    dice_criterion: nn.Module, 
    global_step: int = 0,
    rank: int = 0,
    world_size: int = 1,
    is_distributed: bool = False
) -> tuple[float, int]:
    """
    Signature:
        train_naive_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: str, bce_criterion: nn.Module, dice_criterion: nn.Module, global_step: int, rank: int, world_size: int, is_distributed: bool) -> tuple[float, int]

    Objective:
        Execute one training epoch using naïve supervised BCE + Dice loss with DDP synchronization.

    Inputs:
        model (nn.Module): VoxTell model instance (or DDP wrapped model).
        dataloader (DataLoader): PyTorch training DataLoader.
        optimizer (Optimizer): PyTorch AdamW optimizer.
        scaler (GradScaler): AMP Gradient Scaler.
        device (str): Computation device string.
        bce_criterion (nn.Module): BCEWithLogitsLoss instance.
        dice_criterion (nn.Module): DiceLoss instance.
        global_step (int): Running global iteration counter across epochs. Default 0.
        rank (int): Process global rank. Default 0.
        world_size (int): Total number of distributed processes. Default 1.
        is_distributed (bool): Whether running in multi-GPU distributed mode. Default False.

    Outputs:
        tuple[float, int]: Average training loss across all processes and updated global_step.
    """
    model.train()
    running_loss = 0.0
    valid_batches = 0
    
    for batch in tqdm(dataloader, desc="Training Epoch", leave=False, disable=(rank != 0)):
        images = batch['image'].to(device) # (B, 1, Z, Y, X)
        targets = batch['seg'].to(device)   # (B, F, Z, Y, X)
        text_embeds = batch['text_embeddings'].to(device) # (B, F, 2560)
        scan_id = batch.get('scan_id', ['unknown'])[0]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # VoxTell forward pass
            logits = model(images, text_embeds) # (B, F, Z, Y, X)
            
            # Upcast logits and targets to float32 for loss stability
            logits_f32 = logits.float()
            targets_f32 = targets.float()
            
            loss_bce = bce_criterion(logits_f32, targets_f32)
            loss_dice = dice_criterion(logits_f32, targets_f32)
            total_loss = loss_bce + loss_dice
        
        if not torch.isfinite(total_loss):
            logger.warning(f"Scan {scan_id} on Rank {rank} produced non-finite loss (BCE={loss_bce.item()}, Dice={loss_dice.item()}). Skipping step.")
            continue
            
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Check gradient finiteness before clipping to prevent NaN corruption
        grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if grads_finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            logger.warning(f"Scan {scan_id} on Rank {rank} produced non-finite gradients. Skipping grad clipping.")
            
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        valid_batches += 1
        global_step += 1

        if rank == 0:
            try:
                import wandb
                if wandb.run is not None and global_step % 5 == 0:
                    wandb.log({
                        "train/step_loss": total_loss.item(),
                        "train/step_bce": loss_bce.item(),
                        "train/step_dice": loss_dice.item(),
                        "step": global_step
                    })
            except Exception:
                pass
        
    # Synchronize epoch loss across all DDP ranks
    if is_distributed:
        stats_tensor = torch.tensor([running_loss, float(valid_batches)], device=device, dtype=torch.float32)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        total_reduced_loss = stats_tensor[0].item()
        total_reduced_batches = stats_tensor[1].item()
        epoch_avg_loss = total_reduced_loss / max(total_reduced_batches, 1.0)
    else:
        epoch_avg_loss = running_loss / max(valid_batches, 1)

    return epoch_avg_loss, global_step


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for VoxTell naïve supervised fine-tuning execution supporting
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
    logger.info("Starting VoxTell Naïve Supervised Fine-Tuning Pipeline (Exp 001)...")
    logger.info(f"Execution Mode: {'Distributed (DDP)' if is_distributed else 'Single-Device'} | Rank: {rank}/{world_size} | Device: {target_device}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size / GPU: {args.batch_size}, Patch Size: {args.patch_size}")
    
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
        
        # Configure Sampler and DataLoader
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
        
        # Instantiate VoxTell Model and load pre-trained weights
        model = load_voxtell_model(args.model_dir, target_device)
        
        # Optimizer, Loss Criteria, and Scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler('cuda')
        
        bce_criterion = nn.BCEWithLogitsLoss()
        dice_criterion = DiceLoss(sigmoid=True)
        
        start_epoch = 1
        best_loss = float("inf")
        latest_model_path = output_dir / "latest_model.pt"

        if args.resume and latest_model_path.exists():
            logger.info(f"Resuming training from checkpoint: {latest_model_path}")
            checkpoint = torch.load(latest_model_path, map_location=target_device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "loss" in checkpoint:
                best_loss = checkpoint["loss"]
            start_epoch = checkpoint.get("epoch", 0) + 1
            logger.info(f"Successfully resumed from epoch {start_epoch}, previous best loss: {best_loss:.4f}")
        
        # Wrap model in DistributedDataParallel
        if is_distributed:
            model = DDP(
                model,
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
                
            epoch_loss, global_step = train_naive_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=target_device,
                bce_criterion=bce_criterion,
                dice_criterion=dice_criterion,
                global_step=global_step,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] — Global Avg Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")
            
            # Checkpoints serialized strictly from Rank 0
            if rank == 0:
                unwrapped_state = get_unwrapped_state_dict(model)
                latest_model_path = output_dir / "latest_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": unwrapped_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss
                }, latest_model_path)

                if math.isfinite(epoch_loss) and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_path = output_dir / "best_model.pt"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": unwrapped_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": best_loss
                    }, best_model_path)
                    logger.info(f"Saved new best model checkpoint to: {best_model_path}")
                    
                if args.wandb:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": epoch_loss,
                        "train/lr": current_lr,
                        "train/best_loss": best_loss
                    })

        if rank == 0 and args.wandb:
            import wandb
            wandb.finish()

        logger.info("Naïve supervised fine-tuning training complete.")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
