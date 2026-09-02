"""
===============================================================================
SCRIPT:         VoxTell-SPOCO Model Fine-Tuning (SPOCO Foundation)
PHASE:          Phase 4 — VoxTell-SPOCO Metric Learning
LOCATION:       scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py
OBJECTIVE:      Fine-tune VoxTell adapted for Sparse Object-level Consistency
                (SPOCO, Wolny et al., CVPR 2022). Maps 3D CT voxels into a continuous
                32D metric embedding space on a unit hypersphere, regularized via
                Student-Teacher unannotated consistency with iterative coverage suppression,
                multi-instance connected-component anchoring, dual-view intensity perturbations,
                and background repulsion to resolve instance suppression.
                Supports server-agnostic multi-GPU (DDP) and single-GPU execution.
USAGE:          Single-GPU: python scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py
                Multi-GPU:  torchrun --nproc_per_node=N scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py
===============================================================================
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure proper GPU isolation before loading torch
load_dotenv(override=False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Relative root directory path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATASET_JSON,
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    TEXT_CACHE_DIR,
    LOGS_DIR,
    MODEL_DIR,
)

# Phase 3 Shared Common Infrastructure (DDP, Dataset, Worker Resolution)
from scripts.phase_3_voxtell_finetuning.common import (
    init_distributed,
    cleanup_distributed,
    setup_distributed_logger,
    get_unwrapped_state_dict,
    ddp_step,
    ReXDataset,
    resolve_num_workers,
)

# Phase 4 Common Infrastructure (VoxTell-SPOCO Model & Loss Engine)
from scripts.phase_4_voxtell_spoco.common import (
    load_voxtell_spoco_model,
    compute_spoco_total_loss,
)

# Experiment log directory pairing
EXP_LOG_DIR = LOGS_DIR / "phase_4_voxtell_spoco" / "exp_001_voxtell_spoco"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("exp_001_voxtell_spoco")


def apply_student_view_perturbation(images: torch.Tensor) -> torch.Tensor:
    """
    Signature:
        apply_student_view_perturbation(images: torch.Tensor) -> torch.Tensor

    Objective:
        Apply lightweight, GPU-accelerated intensity perturbations (additive Gaussian
        noise and stochastic intensity scaling/shifting) to generate a distinct Student view
        while preserving 100% identical 3D spatial voxel coordinates with the Teacher view.

    Inputs:
        images (torch.Tensor): 3D CT volume tensor of shape (B, 1, Z, Y, X).

    Outputs:
        torch.Tensor: Perturbed 3D CT volume tensor of shape (B, 1, Z, Y, X).
    """
    B = images.shape[0]
    device = images.device
    dtype = images.dtype

    # 1. Random intensity scaling in [0.90, 1.10]
    scale = torch.empty((B, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(0.90, 1.10)
    # 2. Random intensity shift in [-0.10, 0.10]
    shift = torch.empty((B, 1, 1, 1, 1), device=device, dtype=dtype).uniform_(-0.10, 0.10)
    # 3. Additive Gaussian noise with std=0.03
    noise = torch.randn_like(images) * 0.03

    perturbed = (images * scale) + shift + noise
    return perturbed


def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float) -> None:
    """
    Signature:
        update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float) -> None

    Objective:
        Update Teacher parameters via Exponential Moving Average (EMA).
        theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student.

    Inputs:
        model (nn.Module): Student active model (or unwrapped model).
        ema_model (nn.Module): Teacher EMA model.
        alpha (float): Momentum decay rate (e.g. 0.999).

    Outputs:
        None
    """
    student_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    for key, ema_param in ema_model.state_dict().items():
        if key in student_state:
            student_param = student_state[key]
            if ema_param.dtype.is_floating_point:
                ema_param.data.mul_(alpha).add_(student_param.data.to(ema_param.device), alpha=1.0 - alpha)


def evaluate_val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    delta_var: float = 0.5,
    delta_dist: float = 1.5,
    pmaps_threshold: float = 0.5,
    sigma: float | None = None,
    w_con: float = 0.1,
    w_unl_push: float = 0.1,
    num_unlabeled_anchors: int = 8,
    volume_threshold: float = 0.05,
    max_batches: int = 20,
) -> float:
    """
    Signature:
        evaluate_val_loss(model: nn.Module, val_loader: DataLoader, device: str, delta_var: float = 0.5, delta_dist: float = 1.5, pmaps_threshold: float = 0.5, sigma: float | None = None, w_con: float = 0.1, w_unl_push: float = 0.1, num_unlabeled_anchors: int = 8, volume_threshold: float = 0.05, max_batches: int = 20) -> float

    Objective:
        Compute validation SPOCO loss across a fixed subset of validation batches.

    Inputs:
        model (nn.Module): Student model to evaluate.
        val_loader (DataLoader): Validation DataLoader.
        device (str): Computation device.
        delta_var (float): Intra-cluster pull distance margin (default 0.5).
        delta_dist (float): Inter-cluster push distance margin (default 1.5).
        pmaps_threshold (float): Kernel probability cutoff (default 0.5).
        sigma (float | None): Optional legacy sigma override.
        w_con (float): Consistency loss weight. Default 0.1.
        w_unl_push (float): Unlabeled background push weight. Default 0.1.
        num_unlabeled_anchors (int): Max unannotated anchors per volume. Default 8.
        volume_threshold (float): Stopping fraction for unlabeled coverage. Default 0.05.
        max_batches (int): Maximum number of validation batches to evaluate. Default 20.

    Outputs:
        float: Average validation loss.
    """
    model.eval()
    val_losses = []
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["seg"].to(device, non_blocking=True)
            text_embeds = (batch.get("text_embeddings") if "text_embeddings" in batch else batch["text_embedding"]).to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = model(images, text_embeds, return_embeddings=True)
                loss, _, _ = compute_spoco_total_loss(
                    student_embeds=s_embeds,
                    teacher_embeds=s_embeds,
                    targets=targets,
                    delta_var=delta_var,
                    delta_dist=delta_dist,
                    pmaps_threshold=pmaps_threshold,
                    sigma=sigma,
                    w_con=w_con,
                    w_unl_push=w_unl_push,
                    num_unlabeled_anchors=num_unlabeled_anchors,
                    volume_threshold=volume_threshold,
                    negative_supervision=True,
                    return_details=False,
                )
            if torch.isfinite(loss):
                val_losses.append(loss.item())

    model.train()
    return float(sum(val_losses) / max(1, len(val_losses)))


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse CLI arguments for Phase 4 Exp 001 VoxTell-SPOCO fine-tuning.
    """
    parser = argparse.ArgumentParser(description="Phase 4 Exp 001: VoxTell-SPOCO Model Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4)")
    parser.add_argument("--alpha", type=float, default=0.999, help="EMA decay rate for Teacher model (default: 0.999)")
    parser.add_argument("--delta_var", type=float, default=0.5, help="Intra-cluster pull margin (default: 0.5)")
    parser.add_argument("--delta_dist", type=float, default=1.5, help="Inter-cluster push margin (default: 1.5)")
    parser.add_argument("--kernel_threshold", type=float, default=0.5, help="Gaussian soft mask cutoff (default: 0.5)")
    parser.add_argument("--sigma", type=float, default=None, help="Legacy Gaussian bandwidth sigma override")
    parser.add_argument("--w_con", type=float, default=0.1, help="Consistency loss weight for unlabeled anchors (default: 0.1)")
    parser.add_argument("--w_unl_push", type=float, default=0.1, help="Unlabeled background push loss weight (default: 0.1)")
    parser.add_argument("--max_unlabeled_anchors", type=int, default=8, help="Max unannotated anchors per volume (default: 8)")
    parser.add_argument("--volume_threshold", type=float, default=0.05, help="Stopping fraction for uncovered background (default: 0.05)")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Metric embedding dimension D (default: 32)")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json")
    parser.add_argument("--output_dir", type=str, default=str(EXP_LOG_DIR), help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_model.pt in output_dir")
    parser.add_argument(
        "--use_volume_cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fast volume caching (default: False, streams directly from NVMe)",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker count (None for server-agnostic auto)")
    parser.add_argument("--dry_run", action="store_true", help="Execute single-batch verification step")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases telemetry logging")
    parser.add_argument("--wandb_project", type=str, default="rexgroundingct", help="WandB project name")
    return parser.parse_args()


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for Phase 4 Exp 001 VoxTell-SPOCO training pipeline.
        Initializes DDP, loads pre-trained foundation backbone weights, sets up
        Student and Teacher models, and executes the SPOCO optimization loop.
    """
    args = parse_args()
    is_distributed, rank, local_rank, world_size, device_str = init_distributed()
    device = torch.device(device_str)

    output_dir = Path(args.output_dir)
    setup_distributed_logger(logger, output_dir, rank)

    if rank == 0:
        logger.info("=" * 80)
        logger.info("PHASE 4 — EXP 001: VOXTELL-SPOCO MODEL FINE-TUNING")
        logger.info(f"Host Device: {device_str} | World Size: {world_size} | DDP: {is_distributed}")
        logger.info(f"Epochs: {args.epochs} | LR: {args.lr} | Alpha (EMA): {args.alpha}")
        logger.info(f"Delta Var: {args.delta_var} | Delta Dist: {args.delta_dist} | Kernel Threshold: {args.kernel_threshold}")
        logger.info(f"W_con: {args.w_con} | W_unl_push: {args.w_unl_push} | Max Anchors: {args.max_unlabeled_anchors}")
        logger.info(f"Metric Embedding Dim: {args.embedding_dim} | Model Dir: {MODEL_DIR}")
        logger.info("=" * 80)

    # Initialize WandB on Rank 0 if requested
    if rank == 0 and args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name="exp_001_voxtell_spoco",
                config=vars(args),
            )
            logger.info("Initialized Weights & Biases telemetry.")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

    # 1. Instantiate Student & Teacher VoxTell-SPOCO Models
    student_model = load_voxtell_spoco_model(
        model_dir=str(MODEL_DIR),
        device=device_str,
        embedding_dim=args.embedding_dim,
        deep_supervision=False,
    )

    teacher_model = load_voxtell_spoco_model(
        model_dir=str(MODEL_DIR),
        device=device_str,
        embedding_dim=args.embedding_dim,
        deep_supervision=False,
    )
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # Wrap Student in DDP if multi-GPU
    if is_distributed:
        student_model = DDP(
            student_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # 2. Setup Optimizer & Native bfloat16 Scaler
    raw_student = student_model.module if hasattr(student_model, "module") else student_model
    encoder_params = set(raw_student.encoder.parameters())
    transformer_params = set(raw_student.transformer_decoder.parameters())
    decoder_and_head_params = [
        p for p in raw_student.parameters()
        if p not in encoder_params and p not in transformer_params
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": list(encoder_params), "lr": args.lr * 0.1},
            {"params": list(transformer_params), "lr": args.lr * 0.5},
            {"params": decoder_and_head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # Disabled scaler for native bfloat16

    # 3. Setup ReXDataset & DataLoaders
    workers = resolve_num_workers(args.num_workers)
    if rank == 0:
        logger.info(f"DataLoader Worker Resolution: {workers} workers per GPU.")

    train_dataset = ReXDataset(
        dataset_json=args.dataset_json,
        split="train",
        img_dir=str(RAW_IMAGES_DIR),
        seg_dir=str(RAW_MASKS_DIR),
        cache_dir=str(TEXT_CACHE_DIR),
        is_train=True,
        patch_size=192,
        num_positive_prompts=2,
        num_negative_prompts=1,
        pos_ratio=0.85,
        use_volume_cache=args.use_volume_cache,
    )

    val_dataset = ReXDataset(
        dataset_json=args.dataset_json,
        split="val",
        img_dir=str(RAW_IMAGES_DIR),
        seg_dir=str(RAW_MASKS_DIR),
        cache_dir=str(TEXT_CACHE_DIR),
        is_train=False,
        patch_size=192,
        num_positive_prompts=2,
        num_negative_prompts=1,
        pos_ratio=0.85,
        use_volume_cache=args.use_volume_cache,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    # 4. Dry Run Mode
    if args.dry_run:
        if rank == 0:
            logger.info("Executing single-batch dry run verification on VoxTell-SPOCO with dual-view perturbation...")
        batch = next(iter(train_loader))
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["seg"].to(device, non_blocking=True)
        text_embeds = (batch.get("text_embeddings") if "text_embeddings" in batch else batch["text_embedding"]).to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Dual view perturbation: student receives perturbed view, teacher receives unperturbed view
            student_images = apply_student_view_perturbation(images)
            s_embeds = student_model(student_images, text_embeds, return_embeddings=True)
            with torch.no_grad():
                t_embeds = teacher_model(images, text_embeds, return_embeddings=True)

            loss, l_obj, l_con, l_push = compute_spoco_total_loss(
                student_embeds=s_embeds,
                teacher_embeds=t_embeds,
                targets=targets,
                delta_var=args.delta_var,
                delta_dist=args.delta_dist,
                pmaps_threshold=args.kernel_threshold,
                sigma=args.sigma,
                w_con=args.w_con,
                w_unl_push=args.w_unl_push,
                num_unlabeled_anchors=args.max_unlabeled_anchors,
                volume_threshold=args.volume_threshold,
                negative_supervision=True,
                return_details=True,
            )

        success = ddp_step(
            total_loss=loss,
            model=student_model,
            optimizer=optimizer,
            scaler=scaler,
            is_distributed=is_distributed,
            logger=logger,
            scan_id=batch["scan_id"][0] if "scan_id" in batch else "dry_run",
            rank=rank,
        )
        update_ema_variables(student_model, teacher_model, alpha=args.alpha)

        if rank == 0:
            logger.info(
                f"Dry Run Result: Success={success} | Total Loss: {loss.item():.4f} "
                f"(L_obj: {l_obj.item():.4f}, L_con: {l_con.item():.4f}, L_push: {l_push.item():.4f})"
            )
            logger.info(f"Student Metric Embeddings Shape: {tuple(s_embeds.shape)}")
        cleanup_distributed()
        return

    # 5. Multi-Epoch Training Loop
    best_val_loss = float("inf")
    total_steps = 0
    start_epoch = 1

    latest_model_path = output_dir / "latest_model.pt"
    if args.resume and latest_model_path.exists():
        if rank == 0:
            logger.info(f"Resuming training from checkpoint: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device, weights_only=False)
        # Checkpoints are serialized unwrapped (get_unwrapped_state_dict), so load into the inner
        # module: a DDP wrapper expects 'module.'-prefixed keys and would reject them outright.
        unwrapped_student = student_model.module if hasattr(student_model, "module") else student_model
        unwrapped_student.load_state_dict(checkpoint["student_state_dict"])
        teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        if rank == 0:
            logger.info(f"Successfully resumed at Epoch {start_epoch}, previous best val loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        student_model.train()
        epoch_loss_sum = 0.0
        epoch_obj_sum = 0.0
        epoch_con_sum = 0.0
        epoch_push_sum = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=(rank != 0))
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["seg"].to(device, non_blocking=True)
            text_embeds = (batch.get("text_embeddings") if "text_embeddings" in batch else batch["text_embedding"]).to(device, non_blocking=True)
            scan_id = batch["scan_id"][0] if "scan_id" in batch else f"step_{total_steps}"

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                student_images = apply_student_view_perturbation(images)
                s_embeds = student_model(student_images, text_embeds, return_embeddings=True)
                with torch.no_grad():
                    t_embeds = teacher_model(images, text_embeds, return_embeddings=True)

                loss, l_obj, l_con, l_push = compute_spoco_total_loss(
                    student_embeds=s_embeds,
                    teacher_embeds=t_embeds,
                    targets=targets,
                    delta_var=args.delta_var,
                    delta_dist=args.delta_dist,
                    pmaps_threshold=args.kernel_threshold,
                    sigma=args.sigma,
                    w_con=args.w_con,
                    w_unl_push=args.w_unl_push,
                    num_unlabeled_anchors=args.max_unlabeled_anchors,
                    volume_threshold=args.volume_threshold,
                    negative_supervision=True,
                    return_details=True,
                )

            step_ok = ddp_step(
                total_loss=loss,
                model=student_model,
                optimizer=optimizer,
                scaler=scaler,
                is_distributed=is_distributed,
                logger=logger,
                scan_id=scan_id,
                rank=rank,
            )

            if step_ok:
                update_ema_variables(student_model, teacher_model, alpha=args.alpha)
                epoch_loss_sum += loss.item()
                epoch_obj_sum += l_obj.item()
                epoch_con_sum += l_con.item()
                epoch_push_sum += l_push.item()
                step_count += 1
                total_steps += 1

                if rank == 0:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "l_obj": f"{l_obj.item():.4f}",
                        "l_con": f"{l_con.item():.4f}",
                        "l_push": f"{l_push.item():.4f}",
                    })

        avg_epoch_loss = epoch_loss_sum / max(1, step_count)
        avg_obj_loss = epoch_obj_sum / max(1, step_count)
        avg_con_loss = epoch_con_sum / max(1, step_count)
        avg_push_loss = epoch_push_sum / max(1, step_count)

        # Validation Evaluation
        val_loss = evaluate_val_loss(
            model=student_model,
            val_loader=val_loader,
            device=device_str,
            delta_var=args.delta_var,
            delta_dist=args.delta_dist,
            pmaps_threshold=args.kernel_threshold,
            sigma=args.sigma,
            w_con=args.w_con,
            w_unl_push=args.w_unl_push,
            num_unlabeled_anchors=args.max_unlabeled_anchors,
            volume_threshold=args.volume_threshold,
        )

        if rank == 0:
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Train Loss: {avg_epoch_loss:.4f} (Obj: {avg_obj_loss:.4f}, Con: {avg_con_loss:.4f}, Push: {avg_push_loss:.4f}) | "
                f"Val Loss: {val_loss:.4f}"
            )

            if args.wandb:
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/total_loss": avg_epoch_loss,
                        "train/obj_loss": avg_obj_loss,
                        "train/con_loss": avg_con_loss,
                        "train/push_loss": avg_push_loss,
                        "val/loss": val_loss,
                    })
                except Exception:
                    pass

            # Save Checkpoint if best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = output_dir / "best_model.pt"
                unwrapped_state = get_unwrapped_state_dict(student_model)
                teacher_state = teacher_model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "student_state_dict": unwrapped_state,
                        "teacher_state_dict": teacher_state,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
                logger.info(f"Saved new best checkpoint to {best_ckpt_path} (Val Loss: {val_loss:.4f})")

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                epoch_ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "student_state_dict": get_unwrapped_state_dict(student_model),
                        "teacher_state_dict": teacher_model.state_dict(),
                        "val_loss": val_loss,
                    },
                    epoch_ckpt_path,
                )

            # Always save latest_model.pt for robust checkpointing / auto-resume
            latest_ckpt_path = output_dir / "latest_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "student_state_dict": get_unwrapped_state_dict(student_model),
                    "teacher_state_dict": teacher_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                latest_ckpt_path,
            )
            logger.info(f"Updated latest checkpoint: {latest_ckpt_path}")

    if rank == 0:
        logger.info(f"Phase 4 Exp 001 Training Completed across {args.epochs} epochs. Best Val Loss: {best_val_loss:.4f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

