"""
===============================================================================
SCRIPT:         VoxTell Batch Zero-Shot Baseline Inference Pipeline (Wrapper)
PHASE:          Phase 2B — VoxTell Zero-Shot Baseline & Preprocessing Audit
LOCATION:       scripts/phase_2b_voxtell_baseline/exp_001_voxtell_inference.py
OBJECTIVE:      Backward-compatible execution wrapper for Phase 2B experiments.
                Delegates to the Centralized Universal VoxTell Inference Engine
                (scripts/common/voxtell_inference.py).
USAGE:          python scripts/phase_2b_voxtell_baseline/exp_001_voxtell_inference.py --split val
===============================================================================
"""

import sys
from pathlib import Path

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common.voxtell_inference import main

if __name__ == "__main__":
    main()