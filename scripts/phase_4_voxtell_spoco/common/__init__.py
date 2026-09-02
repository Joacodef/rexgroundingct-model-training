"""
===============================================================================
PACKAGE:        Phase 4 VoxTell-SPOCO Common Infrastructure
LOCATION:       scripts/phase_4_voxtell_spoco/common/__init__.py
OBJECTIVE:      Export reusable components for Phase 4 VoxTell-SPOCO 3D vision-language
                models and metric architectures.
===============================================================================
"""

from scripts.phase_4_voxtell_spoco.common.voxtell_spoco import (
    VoxTellSpocoModel,
    VoxTellSpocoDecoder,
    load_voxtell_spoco_model,
)
from scripts.phase_4_voxtell_spoco.common.losses import (
    compute_gaussian_soft_mask,
    sample_annotated_anchors,
    sample_unannotated_anchors,
    compute_instance_dice_loss,
    compute_unlabeled_push_loss,
    compute_spoco_total_loss,
)
from scripts.phase_4_voxtell_spoco.common.clustering import (
    extract_instances_from_embeddings,
)

__all__ = [
    "VoxTellSpocoModel",
    "VoxTellSpocoDecoder",
    "load_voxtell_spoco_model",
    "compute_gaussian_soft_mask",
    "sample_annotated_anchors",
    "sample_unannotated_anchors",
    "compute_instance_dice_loss",
    "compute_unlabeled_push_loss",
    "compute_spoco_total_loss",
    "extract_instances_from_embeddings",
]

