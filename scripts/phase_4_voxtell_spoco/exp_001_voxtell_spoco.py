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
import copy
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from monai.data.utils import worker_init_fn as monai_worker_init_fn
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

# Stamped into every checkpoint. Optimizer state is only restored from a checkpoint carrying this
# exact marker, because AdamW state is mapped positionally within each parameter group and older
# checkpoints were written with set-ordered (i.e. process-dependent) groups.
PARAM_GROUP_ORDER = "ordered_v1"


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


def consistency_rampup_weight(epoch: int, max_epochs: int, max_weight: float) -> float:
    """
    Signature:
        consistency_rampup_weight(epoch: int, max_epochs: int, max_weight: float) -> float

    Objective:
        Gaussian exponential ramp-up for the SPOCO consistency weight,
        gamma(t) = max_weight * exp(-5 * (1 - t/T)^2) (Gao et al., 2022; the same
        schedule as Phase 3 Exp 003 get_mpr_rampup_weight). Near 0 in the first epochs,
        when the student and the EMA teacher are both uninformative, and approaching
        max_weight by the final epoch.

    Inputs:
        epoch (int): Current 1-indexed epoch.
        max_epochs (int): Total number of epochs T.
        max_weight (float): Asymptotic consistency weight.

    Outputs:
        float: Consistency weight to use for this epoch.
    """
    T = float(max_epochs)
    if T <= 1.0:
        return max_weight
    ratio = max(0.0, 1.0 - (epoch / T))
    return max_weight * math.exp(-5.0 * (ratio ** 2))


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
    teacher_model: nn.Module,
    val_loader: DataLoader,
    device: str,
    delta_var: float = 0.5,
    delta_dist: float = 1.5,
    pmaps_threshold: float = 0.5,
    sigma: float | None = None,
    w_con: float = 0.1,
    w_unl_push: float = 0.1,
    w_neg: float = 1.0,
    num_unlabeled_anchors: int = 8,
    volume_threshold: float = 0.05,
    max_batches: int = 20,
    union_target: bool = True,
) -> float:
    """
    Signature:
        evaluate_val_loss(model: nn.Module, teacher_model: nn.Module, val_loader: DataLoader, device: str, delta_var: float = 0.5, delta_dist: float = 1.5, pmaps_threshold: float = 0.5, sigma: float | None = None, w_con: float = 0.1, w_unl_push: float = 0.1, num_unlabeled_anchors: int = 8, volume_threshold: float = 0.05, max_batches: int = 20, union_target: bool = True) -> float

    Objective:
        Compute validation SPOCO loss across a fixed subset of validation batches.

    Inputs:
        model (nn.Module): Student model to evaluate.
        teacher_model (nn.Module): EMA Teacher, used for the consistency term. Passing the student
            here instead (the previous behavior) degenerates L_con into 1 - sum(s^2)/sum(s), a
            soft-mask sharpness penalty rather than a teacher-student agreement measurement.
        val_loader (DataLoader): Validation DataLoader.
        device (str): Computation device.
        delta_var (float): Intra-cluster pull distance margin (default 0.5).
        delta_dist (float): Inter-cluster push distance margin (default 1.5).
        pmaps_threshold (float): Kernel probability cutoff (default 0.5).
        sigma (float | None): Optional legacy sigma override.
        w_con (float): Consistency loss weight. Pass the CURRENT epoch's ramped value, not the
            asymptotic maximum, or the validation objective differs from the trained one. Default 0.1.
        w_unl_push (float): Unlabeled background push weight. Default 0.1.
        w_neg (float): Confirmed-absent null-target weight. Default 1.0. Inert on the validation
            split, which samples no negative prompts, but kept aligned with training.
        num_unlabeled_anchors (int): Max unannotated anchors per volume. Default 8.
        volume_threshold (float): Stopping fraction for unlabeled coverage. Default 0.05.
        max_batches (int): Maximum number of validation batches to evaluate. Default 20.
        union_target (bool): Passed through to `compute_spoco_total_loss` -- must match the
            mode training was run under. Default True (see `--target_mode` on this script).

    Outputs:
        float: Average validation loss.
    """
    model.eval()
    teacher_model.eval()
    val_losses = []
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if b_idx >= max_batches:
                break
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["seg"].to(device, non_blocking=True)
            text_embeds = (batch.get("text_embeddings") if "text_embeddings" in batch else batch["text_embedding"]).to(device, non_blocking=True)
            is_absent = batch["is_absent_finding"].to(device, non_blocking=True) if "is_absent_finding" in batch else None

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_embeds = model(images, text_embeds, return_embeddings=True)
                t_embeds = teacher_model(images, text_embeds, return_embeddings=True)
                loss, _, _ = compute_spoco_total_loss(
                    student_embeds=s_embeds,
                    teacher_embeds=t_embeds,
                    targets=targets,
                    is_absent=is_absent,
                    delta_var=delta_var,
                    delta_dist=delta_dist,
                    pmaps_threshold=pmaps_threshold,
                    sigma=sigma,
                    w_con=w_con,
                    w_unl_push=w_unl_push,
                    w_neg=w_neg,
                    num_unlabeled_anchors=num_unlabeled_anchors,
                    volume_threshold=volume_threshold,
                    negative_supervision=True,
                    union_target=union_target,
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
    parser.add_argument("--w_neg", type=float, default=1.0, help="Confirmed-absent null-target loss weight L_neg (default: 1.0). L_neg is ~1e-3 in magnitude against Dice terms of order 1; this default is a starting point, not a calibrated value")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay, applied only to parameters with ndim >= 2 outside the final decoder stage (default: 1e-4)")
    parser.add_argument("--max_unlabeled_anchors", type=int, default=8, help="Max unannotated anchors per volume (default: 8)")
    parser.add_argument(
        "--target_mode", type=str, default="union", choices=["union", "instance"],
        help="L_obj Dice target for each sampled annotated anchor (default: union). "
             "'union' scores every anchor against the FULL finding mask (all instances): the "
             "official Dice/Hit-Rate metric is finding-level with no instance-matching term, "
             "so this both removes the false-positive penalty a per-instance target places on "
             "an anchor bleeding into a different true instance of the same finding (a bleed "
             "L_con is separately encouraging), and gives every annotated voxel gradient "
             "regardless of the max_annotated_components cap. 'instance' recovers the original "
             "per-component Dice target for an explicit ablation against 'union'."
    )
    parser.add_argument("--volume_threshold", type=float, default=0.05, help="Stopping fraction for uncovered background (default: 0.05)")
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
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
        help="Enable Weights & Biases telemetry logging (default: True)",
    )
    parser.add_argument(
        "--no_wandb",
        dest="wandb",
        action="store_false",
        help="Disable Weights & Biases telemetry logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "rexgroundingct-challenge"),
        help="WandB project name",
    )
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
        logger.info(f"W_con: {args.w_con} | W_unl_push: {args.w_unl_push} | W_neg: {args.w_neg} | Max Anchors: {args.max_unlabeled_anchors}")
        logger.info(f"Weight Decay: {args.weight_decay} (ndim >= 2 only, final decoder stage exempt) | LR Schedule: CosineAnnealingLR(T_max={args.epochs}, eta_min=1e-6)")
        logger.info(f"L_obj Target Mode: {args.target_mode} (union = scored against full finding mask, no instance-matching term in the official metric)")
        logger.info(f"Metric Embedding Dim: 32 (native decoder width) | Model Dir: {MODEL_DIR}")
        logger.info("=" * 80)

    # Initialize WandB on Rank 0 if requested
    if rank == 0 and args.wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name="exp_001_voxtell_spoco",
                config=vars(args),
                resume="allow",
            )
            logger.info("Initialized Weights & Biases telemetry.")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

    # 1. Instantiate Student & Teacher VoxTell-SPOCO Models. The Teacher is a deep copy
    #    of the Student (identical weights), avoiding a second multi-GB checkpoint load
    #    from disk that the previous load_state_dict form immediately overwrote.
    student_model = load_voxtell_spoco_model(
        model_dir=str(MODEL_DIR),
        device=device_str,
        deep_supervision=False,
    )

    teacher_model = copy.deepcopy(student_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # Wrap Student in DDP if multi-GPU. find_unused_parameters stays True: training only
    # exercises return_embeddings=True, so the decoder's seg_layers and the stage-0
    # text-query projection (used by the inference logit head) receive no gradient. An
    # auxiliary logit loss on the annotated ROI would let this go False and also
    # fine-tune the inference seed map -- tracked as a candidate Exp 001b, not part of
    # this baseline.
    if is_distributed:
        student_model = DDP(
            student_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # 2. Setup Optimizer, LR Schedule & Native bfloat16 Scaler
    raw_student = student_model.module if hasattr(student_model, "module") else student_model

    # Parameter group ORDER must be deterministic across processes. Building the groups from
    # `set(...)` made their order depend on id()-derived hashes, which vary between runs; because
    # Optimizer.load_state_dict maps saved state POSITIONALLY within each group, every --resume
    # re-attached Adam's exp_avg / exp_avg_sq to a different parameter than the one they were
    # accumulated for (a shape error where widths differ, silently wrong momentum where they
    # match). Membership still uses id() sets, but iteration order now always comes from
    # named_parameters(), which is stable.
    encoder_ids = {id(p) for p in raw_student.encoder.parameters()}
    transformer_ids = {id(p) for p in raw_student.transformer_decoder.parameters()}

    # Weight decay applies only to parameters with ndim >= 2. plans.json builds the backbone with
    # InstanceNorm3d(affine=True) and conv_bias=True, so every norm scale/shift and every conv
    # bias is 1-D, and decaying those toward zero discards pretrained calibration.
    # The final full-resolution decoder stage is additionally exempt: its output is L2-normalized
    # onto S^31, so the loss is invariant to that stage's weight magnitude. No gradient opposes
    # decay along the radial direction -- it only shrinks the weights while inflating the
    # effective learning rate on their direction.
    final_stage_prefix = f"decoder.stages.{len(raw_student.decoder.stages) - 1}."

    groups = {
        "encoder":      {"lr": args.lr * 0.1, "decay": [], "no_decay": []},
        "transformer":  {"lr": args.lr * 0.5, "decay": [], "no_decay": []},
        "decoder_head": {"lr": args.lr,       "decay": [], "no_decay": []},
    }
    for param_name, param in raw_student.named_parameters():
        if id(param) in encoder_ids:
            group_key = "encoder"
        elif id(param) in transformer_ids:
            group_key = "transformer"
        else:
            group_key = "decoder_head"
        decays = (param.ndim >= 2) and not param_name.startswith(final_stage_prefix)
        groups[group_key]["decay" if decays else "no_decay"].append(param)

    param_groups = []
    for group_key, spec in groups.items():
        if spec["decay"]:
            param_groups.append({"params": spec["decay"], "lr": spec["lr"], "weight_decay": args.weight_decay})
        if spec["no_decay"]:
            param_groups.append({"params": spec["no_decay"], "lr": spec["lr"], "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # Disabled scaler for native bfloat16

    if rank == 0:
        for group_key, spec in groups.items():
            logger.info(
                f"Param group '{group_key}': lr={spec['lr']:.2e} | "
                f"decay={len(spec['decay'])} tensors | no_decay={len(spec['no_decay'])} tensors"
            )

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
    # worker_init_fn: MONAI's Randomizable.R is a CLASS variable shared by every transform and
    # seeded once at import, and plain torch DataLoader never reseeds it per worker -- so all
    # workers otherwise replay one identical crop/flip stream. monai.data.utils.worker_init_fn
    # reseeds it from worker_info.seed. Used in preference to monai.data.DataLoader, which would
    # also swap collate_fn to list_data_collate.
    # persistent_workers must be guarded: resolve_num_workers returns --num_workers verbatim when
    # it is >= 0, so 0 is reachable and an unguarded True raises ValueError.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=monai_worker_init_fn,
        persistent_workers=(workers > 0),
    )

    # batch_size is pinned to 1 here, not args.batch_size. Negative-prompt sampling is gated on
    # is_train, so a val item emits N = min(F, num_positive_prompts) prompts -- 1 for the 26% of
    # val scans with a single finding, 2 otherwise -- and default_collate cannot stack a mixed
    # batch. Padding validation with synthetic negatives would change what the val loss measures,
    # so the loader stays scan-by-scan; at max_batches=20 this costs nothing.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=monai_worker_init_fn,
        persistent_workers=(workers > 0),
    )

    # 4. Dry Run Mode
    if args.dry_run:
        if rank == 0:
            logger.info("Executing single-batch dry run verification on VoxTell-SPOCO with dual-view perturbation...")
        batch = next(iter(train_loader))
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["seg"].to(device, non_blocking=True)
        text_embeds = (batch.get("text_embeddings") if "text_embeddings" in batch else batch["text_embedding"]).to(device, non_blocking=True)
        is_absent = batch["is_absent_finding"].to(device, non_blocking=True) if "is_absent_finding" in batch else None

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # Dual view perturbation: student receives perturbed view, teacher receives unperturbed view
            student_images = apply_student_view_perturbation(images)
            s_embeds = student_model(student_images, text_embeds, return_embeddings=True)
            with torch.no_grad():
                t_embeds = teacher_model(images, text_embeds, return_embeddings=True)

            loss, l_obj, l_con, l_push, l_neg = compute_spoco_total_loss(
                student_embeds=s_embeds,
                teacher_embeds=t_embeds,
                targets=targets,
                is_absent=is_absent,
                delta_var=args.delta_var,
                delta_dist=args.delta_dist,
                pmaps_threshold=args.kernel_threshold,
                sigma=args.sigma,
                w_con=args.w_con,
                w_unl_push=args.w_unl_push,
                w_neg=args.w_neg,
                num_unlabeled_anchors=args.max_unlabeled_anchors,
                volume_threshold=args.volume_threshold,
                negative_supervision=True,
                union_target=(args.target_mode == "union"),
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
                f"(L_obj: {l_obj.item():.4f}, L_con: {l_con.item():.4f}, "
                f"L_push: {l_push.item():.4f}, L_neg: {l_neg.item():.4f})"
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
        start_epoch = checkpoint["epoch"] + 1

        # Only load optimizer state from a checkpoint written under deterministic parameter
        # ordering. Pre-"ordered_v1" checkpoints were saved with set-ordered groups, so their
        # moments cannot be mapped back onto the current parameters.
        if checkpoint.get("param_group_order") == PARAM_GROUP_ORDER:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif rank == 0:
            logger.warning(
                f"Checkpoint predates deterministic parameter ordering "
                f"(param_group_order={checkpoint.get('param_group_order')!r}); "
                f"model weights restored but AdamW moments restart at epoch {start_epoch}."
            )

        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler.last_epoch = start_epoch - 1

        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        if rank == 0:
            logger.info(f"Successfully resumed at Epoch {start_epoch}, previous best val loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Gaussian ramp-up for the consistency weight: near 0 early (student and EMA
        # teacher are both uninformative, so enforcing agreement injects noise),
        # approaching args.w_con by the final epoch.
        w_con_epoch = consistency_rampup_weight(epoch, args.epochs, args.w_con)

        student_model.train()
        epoch_loss_sum = 0.0
        epoch_obj_sum = 0.0
        epoch_con_sum = 0.0
        epoch_push_sum = 0.0
        epoch_neg_sum = 0.0
        step_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=(rank != 0))
        for batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["seg"].to(device, non_blocking=True)
            text_embeds = (batch.get("text_embeddings") if "text_embeddings" in batch else batch["text_embedding"]).to(device, non_blocking=True)
            is_absent = batch["is_absent_finding"].to(device, non_blocking=True) if "is_absent_finding" in batch else None
            scan_id = batch["scan_id"][0] if "scan_id" in batch else f"step_{total_steps}"

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                student_images = apply_student_view_perturbation(images)
                s_embeds = student_model(student_images, text_embeds, return_embeddings=True)
                with torch.no_grad():
                    t_embeds = teacher_model(images, text_embeds, return_embeddings=True)

                loss, l_obj, l_con, l_push, l_neg = compute_spoco_total_loss(
                    student_embeds=s_embeds,
                    teacher_embeds=t_embeds,
                    targets=targets,
                    is_absent=is_absent,
                    delta_var=args.delta_var,
                    delta_dist=args.delta_dist,
                    pmaps_threshold=args.kernel_threshold,
                    sigma=args.sigma,
                    w_con=w_con_epoch,
                    w_unl_push=args.w_unl_push,
                    w_neg=args.w_neg,
                    num_unlabeled_anchors=args.max_unlabeled_anchors,
                    volume_threshold=args.volume_threshold,
                    negative_supervision=True,
                    union_target=(args.target_mode == "union"),
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
                epoch_neg_sum += l_neg.item()
                step_count += 1
                total_steps += 1

                if rank == 0:
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "l_obj": f"{l_obj.item():.4f}",
                        "l_con": f"{l_con.item():.4f}",
                        "l_push": f"{l_push.item():.4f}",
                        "l_neg": f"{l_neg.item():.4f}",
                    })
                    if args.wandb and total_steps % 10 == 0:
                        try:
                            import wandb
                            wandb.log({
                                "step/loss": loss.item(),
                                "step/obj_loss": l_obj.item(),
                                "step/con_loss": l_con.item(),
                                "step/push_loss": l_push.item(),
                                "step/neg_loss": l_neg.item(),
                                "step/global_step": total_steps,
                            })
                        except Exception:
                            pass

            # Release the embedding volumes before the next iteration's forward allocates. Each is
            # (B, N, 32, Z, Y, X) -- ~1.36 GB at B=1, N=3, 192^3 -- and without this both stay
            # bound until their names are rebound, carrying ~2.7 GB of dead tensor across the loop
            # boundary on top of peak activation memory.
            del s_embeds, t_embeds, loss, l_obj, l_con, l_push, l_neg

        avg_epoch_loss = epoch_loss_sum / max(1, step_count)
        avg_obj_loss = epoch_obj_sum / max(1, step_count)
        avg_con_loss = epoch_con_sum / max(1, step_count)
        avg_push_loss = epoch_push_sum / max(1, step_count)
        avg_neg_loss = epoch_neg_sum / max(1, step_count)
        # Captured before scheduler.step() below, so this is the LR this epoch actually ran at.
        current_lr = optimizer.param_groups[-1]["lr"]

        # Validation Evaluation. Wrapped defensively: an unexpected evaluation-time failure (a
        # malformed val scan, a transient OOM, etc.) must never discard a completed training epoch,
        # since this is typically run as an unattended multi-epoch SLURM batch job and this call
        # happens before any checkpoint for the epoch is written. On failure, val_loss is recorded as
        # +inf so it can never be mistaken for a new best checkpoint, training continues, and
        # latest_model.pt is still saved below so no progress or queue slot is lost.
        try:
            val_loss = evaluate_val_loss(
                model=student_model,
                teacher_model=teacher_model,
                val_loader=val_loader,
                device=device_str,
                delta_var=args.delta_var,
                delta_dist=args.delta_dist,
                pmaps_threshold=args.kernel_threshold,
                sigma=args.sigma,
                w_con=w_con_epoch,
                w_unl_push=args.w_unl_push,
                w_neg=args.w_neg,
                num_unlabeled_anchors=args.max_unlabeled_anchors,
                volume_threshold=args.volume_threshold,
                union_target=(args.target_mode == "union"),
            )
        except Exception as e:
            if rank == 0:
                logger.warning(f"Epoch {epoch:03d}: validation evaluation failed ({e}). Recording val_loss=inf and continuing.")
            val_loss = float("inf")
            unwrapped_for_mode = student_model.module if hasattr(student_model, "module") else student_model
            unwrapped_for_mode.train()

        if rank == 0:
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"Train Loss: {avg_epoch_loss:.4f} (Obj: {avg_obj_loss:.4f}, Con: {avg_con_loss:.4f}, "
                f"Push: {avg_push_loss:.4f}, Neg: {avg_neg_loss:.4f}) | "
                f"w_con: {w_con_epoch:.4f} | LR: {current_lr:.2e} | Val Loss: {val_loss:.4f}"
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
                        "train/neg_loss": avg_neg_loss,
                        "train/w_con": w_con_epoch,
                        "train/lr": current_lr,
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
                        "scheduler_state_dict": scheduler.state_dict(),
                        "param_group_order": PARAM_GROUP_ORDER,
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
                        "param_group_order": PARAM_GROUP_ORDER,
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
                    "scheduler_state_dict": scheduler.state_dict(),
                    "param_group_order": PARAM_GROUP_ORDER,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                latest_ckpt_path,
            )
            logger.info(f"Updated latest checkpoint: {latest_ckpt_path}")

        # Outside the rank-0 block: every rank must advance the schedule identically.
        scheduler.step()

    if rank == 0:
        logger.info(f"Phase 4 Exp 001 Training Completed across {args.epochs} epochs. Best Val Loss: {best_val_loss:.4f}")
        if args.wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    cleanup_distributed()


if __name__ == "__main__":
    main()

