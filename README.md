# ReXGroundingCT Challenge 2026 — Model Training & Evaluation Workspace

Dedicated research workspace for **Phase 2 Baseline Audits & Phase 3 Model Fine-Tuning** for the **ReXGrounding Challenge @ MICCAI 2026** (3D radiological finding grounding in thoracic CT scans from free-text descriptions).

> [!IMPORTANT]
> **Repository Scope & Governance**:
> This repository is dedicated exclusively to **Phase 2 Non-Neural & VoxTell Baselines, Continuous Logit Probability Thresholding, MONAI Patch Dataloading, PyTorch Mean Teacher Fine-Tuning, Positive-Unlabeled (PU) Mean Teacher Fine-Tuning, and Test Submission Generation**.
> Phase 1 exploratory data profiling is handled in sibling workspace `rexgroundingct-data-profiling`.

---

## 📂 Project Structure

```text
rexgroundingct-model-training/
├── .agents/                    # Agentic rules, host setup docs, and governance
│   └── shared/                 # Server-agnostic master plan and paper digests
├── bash_scripts/               # SLURM batch/interactive job submission scripts (untracked)
├── logs/                       # Baseline audit & training execution logs
│   ├── phase_2a_rule_based/    # Phase 2A non-neural baseline evaluation logs
│   ├── phase_2b_voxtell_baseline/ # Phase 2B VoxTell off-the-shelf baseline audit logs
│   ├── phase_3_voxtell_finetuning/ # Phase 3 VoxTell fine-tuning logs
│   └── phase_4_voxtell_spoco/  # Phase 4 VoxTell-SPOCO metric learning logs
├── report/                     # LaTeX technical report, figures, and figure scripts (untracked)
├── requirements/               # Dependency manifests (base.txt, voxtell.txt, visualization-3d.txt)
├── scratch/                    # Fine-tuning scratch scripts & temporary evaluation tools
├── scripts/                    # Core inference, training, & dataloading pipeline
│   ├── analysis/               # Post-hoc diagnostic profilers & CT visualizers
│   ├── common/                 # Spatial orientation engine, volume preprocessor, & metric evaluator
│   ├── phase_2a_rule_based/    # Phase 2A non-neural statistical baseline pipeline
│   ├── phase_2b_voxtell_baseline/ # Phase 2B VoxTell off-the-shelf baseline audit pipeline
│   ├── phase_3_voxtell_finetuning/ # Phase 3 VoxTell fine-tuning pipeline
│   └── phase_4_voxtell_spoco/  # Phase 4 VoxTell-SPOCO metric learning pipeline
├── tests/                      # Automated test suite for spatial engine and utilities
├── .env.example                # Environment variable configuration template
└── README.md                   # Primary repository documentation
```

---

## 🚀 Setup & Execution

### 1. Environment Setup
Activate the project's standard Python virtual environment (`.venv`):
```bash
source .venv/bin/activate
```

If initializing a fresh virtual environment, install from the pinned manifests in `requirements/`
(`voxtell.txt` includes `base.txt` and pulls the CUDA-matched Torch build plus VoxTell and its
nnU-Net/transformers dependency chain):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements/voxtell.txt
```

> [!NOTE]
> The `voxtell` package constrains `torch<2.9`, so the pinned build is `torch==2.8.0+cu128`.
> PyVista (used only by `scripts/analysis/plot_3d_spatial_density_heatmaps.py`) is optional:
> `pip install -r requirements/visualization-3d.txt`.

### 2. Phase 2 Baselines & Diagnostics

#### Phase 2A: Empirical 3D Spatial PDF Prior Baselines
Run Exp 001 (Percentile Factor Spatial Prior Baseline):
```bash
python scripts/phase_2a_rule_based/exp_001_spatial_priors_percentile.py --split val --eval
```

Run Exp 002 (Empirical Volume Quantile Matching Baseline):
```bash
python scripts/phase_2a_rule_based/exp_002_spatial_priors_quantile.py --split val --eval
```

#### Automated Test Suite Execution
Run automated unit and integration test suites:
```bash
# Full suite (44 tests). Submit through SLURM on ih-condor rather than running on the login node.
.venv/bin/python -m pytest tests -q

# Individual suites
.venv/bin/python -m pytest tests/test_orientation.py -q      # spatial engine (14 tests)
.venv/bin/python -m pytest tests/test_prior_engine.py -q     # Phase 2A prior engine (8 tests)
.venv/bin/python -m pytest tests/test_mpr_loss.py -q         # Phase 3 Exp 003 MPR loss (7 tests)
.venv/bin/python -m pytest tests/test_voxtell_spoco.py -q    # Phase 4 SPOCO (7 tests)
```

> [!NOTE]
> `tests/test_exp001_diagnostics.py` is a GPU training-diagnostic harness, not a unit test: pytest
> collects zero tests from it. Run it only inside an allocated SLURM job.

#### Phase 2B: Off-the-Shelf VoxTell Baseline Inference
Run sliding window inference on validation scans with canonical RAS spatial alignment (`scripts/common/orientation.py`):
```bash
python scripts/phase_2b_voxtell_baseline/exp_001_voxtell_inference.py
```

#### Diagnostic Quantitative Statistical Profiler
Run quantitative 3D/4D segmentation mask statistical profiler:
```bash
python scripts/analysis/prediction_stats.py \
    --pred_dir ../data/predictions/phase_2a_rule_based \
    --gt_dir ../data/raw/segmentations \
    --split val \
    --output_json logs/phase_2a_rule_based/exp_001_spatial_priors_percentile/diagnostic_stats.json
```

#### 6-Slice 2D CT Cross-Sectional Visualizer & Downloadable NIfTI ZIP Exporter
Generate per-pathology 6-slice 2D CT cross-sectional overlays (Max GT and Max Pred slices across Axial, Coronal, Sagittal planes) and 3D-dimension-matched NIfTI bundles saved to `scan_visualizations/<scan_id>/`:
```bash
python scripts/analysis/plot_single_case.py --scan_id train_19891_a_2
```

### 3. Metric Evaluation
Standalone official metric evaluation for predicted 4D segmentation masks against ground-truth masks:
```bash
python scripts/common/evaluate.py --gt_dir ../data/raw/segmentations --img_dir ../data/raw/images --pred_dir ../data/predictions/phase_2a_rule_based --split val
```

### 4. VoxTell Fine-Tuning & Hypotheses (Phase 3)

> [!IMPORTANT]
> `ih-condor` is SLURM-governed. Per `.agents/AGENTS.md`, training MUST be submitted with `sbatch` —
> never run directly on the login shell, and never under a bare `tmux`/`nohup` session. The detached
> `tmux` pattern applies only to standalone non-SLURM development nodes.

Submit fine-tuning as a batch job:
```bash
sbatch bash_scripts/train_exp_003_mpr.slurm
```

To monitor progress:
```bash
# Queue state:
squeue -u $USER

# Live training log:
tail -f logs/phase_3_voxtell_finetuning/exp_003_mpr_loss/run.log

# SLURM stdout/stderr for a given job id:
tail -f logs/phase_3_voxtell_finetuning/exp_003_mpr_loss/slurm_<job_id>.out
```

---

## ⚙️ Shared Data & Hardware Configuration

* **Shared Data**: Dynamic path resolution in `scripts/config.py` automatically links to shared datasets in `../data/` and pretrained weights in `../models/`.
* **Fast Storage Caching**: Preprocessed volumetric tensors are cached in `/tmp/rexgroundingct_preprocessed/` (RAID SSD cache) to bypass slow CPU decompression bounds.
* **Hardware Isolation**: On SLURM hosts such as `ih-condor`, the scheduler sets `CUDA_VISIBLE_DEVICES` per job — do NOT set it yourself in `.env` or job scripts, as it overrides the allocation. Pin GPUs manually only on standalone non-SLURM nodes.

