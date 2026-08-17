"""
===============================================================================
MODULE:         Distributed Utilities for Phase 3 Model Fine-Tuning
LOCATION:       scripts/phase_3_voxtell_training/common/distributed.py
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
