"""
===============================================================================
MODULE:         Phase 2A Rule-Based Common Utilities
OBJECTIVE:      Exports shared spatial prior baseline engines and inference runners.
===============================================================================
"""

from .prior_engine import EmpiricalSpatialPDFBaseline, EMPIRICAL_VOLUME_QUANTILES
from .runner import run_prior_inference_and_eval

__all__ = [
    "EmpiricalSpatialPDFBaseline", 
    "run_prior_inference_and_eval", 
    "EMPIRICAL_VOLUME_QUANTILES"
]
