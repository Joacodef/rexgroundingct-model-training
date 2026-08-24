"""
===============================================================================
PACKAGE:        Phase 3 Common Infrastructure
LOCATION:       scripts/phase_3_voxtell_training/common/__init__.py
OBJECTIVE:      Expose standardized distributed training, dataset loading, and 
                model initialization utilities for Phase 3 experiments.
===============================================================================
"""

from scripts.phase_3_voxtell_training.common.distributed import (
    init_distributed,
    cleanup_distributed,
    setup_distributed_logger,
    get_unwrapped_state_dict,
    ddp_step
)
from scripts.phase_3_voxtell_training.common.dataset import ReXDataset, resolve_num_workers
from scripts.phase_3_voxtell_training.common.model_loader import load_voxtell_model

__all__ = [
    "init_distributed",
    "cleanup_distributed",
    "setup_distributed_logger",
    "get_unwrapped_state_dict",
    "ddp_step",
    "ReXDataset",
    "resolve_num_workers",
    "load_voxtell_model"
]

