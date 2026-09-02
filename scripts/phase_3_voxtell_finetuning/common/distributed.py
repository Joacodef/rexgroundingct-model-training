"""
===============================================================================
MODULE:         Distributed Utilities for Phase 3 Model Fine-Tuning
LOCATION:       scripts/phase_3_voxtell_finetuning/common/distributed.py
OBJECTIVE:      Provide server-agnostic PyTorch DistributedDataParallel (DDP) 
                initialization, clean teardown, rank-0 selective logging, and 
                unwrapped model serialization across Phase 3 training pipelines.
===============================================================================
"""

import os
import sys
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist


def init_distributed() -> tuple[bool, int, int, int, str]:
    """
    Signature:
        init_distributed() -> tuple[bool, int, int, int, str]

    Objective:
        Initialize PyTorch DistributedDataParallel (DDP) environment if launched via torchrun,
        or configure single-GPU/CPU fallback if running standalone.

    Inputs:
        None

    Outputs:
        tuple[bool, int, int, int, str]: (is_distributed, rank, local_rank, world_size, device_str)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device_str = f"cuda:{local_rank}"
        else:
            device_str = "cpu"
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
        is_distributed = True
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_distributed = False
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    return is_distributed, rank, local_rank, world_size, device_str


def cleanup_distributed() -> None:
    """
    Signature:
        cleanup_distributed() -> None

    Objective:
        Destroy PyTorch distributed process group if initialized.

    Inputs:
        None

    Outputs:
        None
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def setup_distributed_logger(target_logger: logging.Logger, exp_log_dir: Path, rank: int) -> None:
    """
    Signature:
        setup_distributed_logger(target_logger: logging.Logger, exp_log_dir: Path, rank: int) -> None

    Objective:
        Configure logging handlers so that Rank 0 logs INFO messages to both console
        and run.log file, while non-zero ranks log only WARNINGs to prevent log corruption.

    Inputs:
        target_logger (logging.Logger): Logger instance to configure.
        exp_log_dir (Path): Directory where run.log should be stored.
        rank (int): Process global rank index.

    Outputs:
        None
    """
    for handler in list(target_logger.handlers):
        target_logger.removeHandler(handler)
        
    target_logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter(f"%(asctime)s [Rank {rank}] [%(levelname)s] %(message)s")

    if rank == 0:
        exp_log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(exp_log_dir / "run.log"), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        target_logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        target_logger.addHandler(stream_handler)
    else:
        null_handler = logging.NullHandler()
        target_logger.addHandler(null_handler)


def get_unwrapped_state_dict(model: nn.Module) -> dict:
    """
    Signature:
        get_unwrapped_state_dict(model: nn.Module) -> dict

    Objective:
        Extract model state dictionary unwrapping DDP container if present.

    Inputs:
        model (nn.Module): Model instance.

    Outputs:
        dict: Clean state dictionary without 'module.' prefix.
    """
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def ddp_step(
    total_loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    is_distributed: bool,
    max_norm: float = 1.0,
    logger: logging.Logger | None = None,
    scan_id: str = "",
    rank: int = 0
) -> bool:
    """
    Signature:
        ddp_step(total_loss: torch.Tensor, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, is_distributed: bool, max_norm: float = 1.0, logger: logging.Logger | None = None, scan_id: str = "", rank: int = 0) -> bool

    Objective:
        Execute a robust, synchronized gradient backward, unscale, gradient clipping,
        and optimizer step across distributed ranks. Guarantees that all DDP ranks execute
        identical collective operations even if individual ranks encounter non-finite loss or gradients.

    Inputs:
        total_loss (torch.Tensor): Forward loss tensor.
        model (nn.Module): DDP or single-GPU model.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        scaler (torch.amp.GradScaler): AMP gradient scaler.
        is_distributed (bool): Whether running under PyTorch DDP.
        max_norm (float): Gradient clipping max norm. Default 1.0.
        logger (logging.Logger | None): Optional logger for diagnostic warnings.
        scan_id (str): Current scan identifier for logging.
        rank (int): Process global rank index.

    Outputs:
        bool: True if the optimizer step was successfully executed with finite loss and finite gradients, False otherwise.
    """
    loss_is_finite = torch.isfinite(total_loss)
    
    if is_distributed and dist.is_available() and dist.is_initialized():
        finite_flag = torch.tensor(1.0 if loss_is_finite else 0.0, device=total_loss.device if total_loss.is_cuda else torch.device("cuda"))
        dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
        all_finite = (finite_flag.item() > 0.5)
    else:
        all_finite = loss_is_finite.item() if isinstance(loss_is_finite, torch.Tensor) else bool(loss_is_finite)

    if not all_finite:
        if logger is not None and not loss_is_finite:
            logger.warning(f"Scan {scan_id} on Rank {rank} produced non-finite loss ({total_loss.item()}). Synchronously skipping step across all ranks.")
        # Perform safe zero-gradient backward to maintain DDP hook synchronization across all ranks
        safe_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        scaler.scale(safe_loss).backward()
        scaler.unscale_(optimizer)
        scaler.update()
        optimizer.zero_grad()
        return False

    # Standard backward pass
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)

    # Check gradient finiteness before clipping to prevent NaN corruption
    local_grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if is_distributed and dist.is_available() and dist.is_initialized():
        grad_flag = torch.tensor(1.0 if local_grads_finite else 0.0, device=total_loss.device if total_loss.is_cuda else torch.device("cuda"))
        dist.all_reduce(grad_flag, op=dist.ReduceOp.MIN)
        all_grads_finite = (grad_flag.item() > 0.5)
    else:
        all_grads_finite = local_grads_finite

    if all_grads_finite:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        scaler.step(optimizer)
        scaler.update()
        return True
    else:
        if logger is not None and not local_grads_finite:
            logger.warning(f"Scan {scan_id} on Rank {rank} produced non-finite gradients. Synchronously skipping optimizer step across all ranks.")
        scaler.update()
        optimizer.zero_grad()
        return False

