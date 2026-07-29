# Scripts Index — ReXGroundingCT Model Training

This directory contains executable modules organized by research phase. Every script adheres to standard phase-prefixed naming conventions and self-documenting module headers.

---

## 🗺️ Module & Objective Mapping Matrix

| Subdirectory / Script | Research Phase | Objective & Description | Primary Inputs | Primary Outputs |
|---|---|---|---|---|
| [`config.py`](config.py) | **Global** | Centralized path, environment variable, and category registry. | `.env`, System environment | Dynamic Path Constants |
| [`evaluate.py`](evaluate.py) | **Global** | Official Challenge evaluator CLI (Dice score & Hit Rate $\ge 0.1$). | GT NIfTI masks, Prediction NIfTI masks | CLI Metrics / Benchmark JSON |
| `baselines/` | **Phase 2A** | Non-neural statistical / rule-based prior segmentation baseline. | `phase_1_priors_bundle.json`, `dataset.json` | `data/predictions/phase_2a_rule_based/` |
| `voxtell/` | **Phase 2B / 3** | Zero-shot VoxTell inference, 4D Back-Reorientation, and PyTorch Mean Teacher fine-tuning. | Pretrained `voxtell_v1.1`, Text Cache, SSD Volumes | `data/predictions/`, `models/checkpoints/` |
| `data_prep/` | **Phase 3** | MONAI 3D patch extraction, text embedding caching, and SSD volume caching. | `raw/images/`, `raw/segmentations/` | `/tmp/rexgroundingct_preprocessed/` |

---

## 📜 Script Naming & Documentation Protocol

Every script in this repository must comply with the following standards:

1. **Phase-Prefixed Naming:**
   - Phase 2A scripts: `phase_2a_<name>.py` (e.g., `scripts/baselines/phase_2a_rule_based_prior_baseline.py`)
   - Phase 2B scripts: `phase_2b_<name>.py` (e.g., `scripts/voxtell/phase_2b_voxtell_zero_shot.py`)
   - Phase 3 scripts: `phase_3_<name>.py` (e.g., `scripts/voxtell/training/phase_3_train_mean_teacher.py`)

2. **Standardized Header Docstrings:**
   Every script MUST start with a block docstring detailing:
   - **PHASE:** Target research phase.
   - **OBJECTIVE:** 1-2 sentence description of what the script accomplishes.
   - **INPUTS:** Explicit path dependencies.
   - **OUTPUTS:** Generated artifacts or prediction directories.
   - **USAGE:** Example CLI invocation command.
