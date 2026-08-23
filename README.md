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
├── logs/                       # Baseline audit & training execution logs
│   ├── phase_2a_rule_based/    # Phase 2A non-neural baseline evaluation logs
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell zero-shot baseline audit logs
│   └── phase_3_training/       # Phase 3 Mean Teacher fine-tuning logs
├── scratch/                    # Fine-tuning scratch scripts & temporary evaluation tools
├── scripts/                    # Core inference, training, & dataloading pipeline
│   ├── analysis/               # Post-hoc diagnostic profilers & CT visualizers
│   ├── common/                 # Spatial orientation engine, volume preprocessor, & metric evaluator
│   ├── phase_2a_rule_based/    # Phase 2A non-neural statistical baseline pipeline
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell zero-shot baseline audit pipeline
│   └── phase_3_training/       # Phase 3 PyTorch semi-supervised fine-tuning pipeline
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

If initializing a fresh virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install monai nibabel SimpleITK scikit-image scipy pandas numpy tqdm wandb python-dotenv
```

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

Run Exp 003 (Radiodensity HU Intensity Windowing + Quantile Baseline):
```bash
python scripts/phase_2a_rule_based/exp_003_hu_windowed_priors.py --split val --eval
```

#### Automated Test Suite Execution
Run automated unit and integration test suites:
```bash
# Prior Engine Test Suite (9/9 passed)
python tests/test_prior_engine.py

# Spatial Orientation Engine Test Suite (13/13 passed)
python tests/test_orientation.py

# Phase 2A Runner Test Suite
python tests/test_phase_2a_runner.py

# Run full unittest discovery across all tests
python -m unittest discover -s tests -p "test_*.py"
```

#### Phase 2B: Zero-Shot VoxTell Baseline Inference
Run sliding window inference on validation scans with canonical RAS spatial alignment (`scripts/common/orientation.py`):
```bash
python scripts/phase_2b_voxtell/exp_001_voxtell_inference.py
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
Run fine-tuning inside a persistent detached `tmux` session:
```bash
tmux new-session -d -s rex_phase3 "CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/phase_3_voxtell_training/exp_002_pu_mean_teacher.py --epochs 50 --batch_size 1 --lr 1e-4 --device cuda:0 --wandb 2>&1 | tee logs/phase_3_voxtell_training/exp_002_pu_mean_teacher/run.log"
```

To monitor progress:
```bash
# Attach to live tmux session:
tmux attach -t rex_phase3

# Or inspect the log file directly:
tail -f logs/phase_3_voxtell_training/exp_002_pu_mean_teacher/run.log
```

---

## ⚙️ Shared Data & Hardware Configuration

* **Shared Data**: Dynamic path resolution in `scripts/config.py` automatically links to shared datasets in `../data/` and pretrained weights in `../models/`.
* **Fast Storage Caching**: Preprocessed volumetric tensors are cached in `/tmp/rexgroundingct_preprocessed/` (RAID SSD cache) to bypass slow CPU decompression bounds.
* **Hardware Isolation**: Pin execution to host GPU via `CUDA_VISIBLE_DEVICES=1` as documented in `.agents/STATUS.md`.

