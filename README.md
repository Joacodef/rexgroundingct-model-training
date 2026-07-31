# ReXGroundingCT Challenge 2026 — Model Training & Evaluation Workspace

Dedicated research workspace for **Phase 2 Baseline Audits & Phase 3 Model Fine-Tuning** for the **ReXGrounding Challenge @ MICCAI 2026** (3D radiological finding grounding in thoracic CT scans from free-text descriptions).

> [!IMPORTANT]
> **Repository Scope & Governance**:
> This repository is dedicated exclusively to **VoxTell Zero-Shot Baseline Audit, Continuous Logit Probability Thresholding, MONAI Patch Dataloading, PyTorch Mean Teacher Fine-Tuning, Positive-Unlabeled (PU) SPOCO Loss Adaptations, and Test Submission Generation**.
> Data profiling and spatial prior generation are handled separately in `rexgroundingct-data-profiling`.

---

## 📂 Project Structure

```text
rexgroundingct-model-training/
├── .agents/                    # Agentic rules, host setup docs, and governance
│   ├── shared/                 # Server-agnostic master plan and paper digests
│   ├── AGENTS.md               # Repository operating rules & governance
│   ├── STATUS.md               # Local active macro progress matrix
│   ├── HANDSHAKE.md            # Tactical session bridge & transition handoff
│   └── server_documentation.txt# Host server hardware setup & guides
├── logs/                       # Baseline audit & training execution logs (per-experiment subfolders)
│   ├── common/                 # System logs and shared utility diagnostics
│   ├── phase_2a_rule_based/    # Phase 2A non-neural baseline evaluation logs
│   │   └── exp_001_seg_masks_priors/ # Dedicated experiment subfolder (eval.md, eval_results_val.json, run.log)
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell zero-shot baseline audit logs
│   │   └── exp_001_voxtell_inference/
│   └── phase_3_training/       # Phase 3 Mean Teacher fine-tuning logs
│       └── exp_001_train_mean_teacher/
├── scratch/                    # Fine-tuning scratch scripts & evaluation tools
├── scripts/                    # Core inference, training, & dataloading pipeline
│   ├── config.py               # Dynamic path resolver (shared ../data/ and ../models/)
│   ├── common/                 # Shared pipelines & utilities
│   │   ├── analyze_predictions.py # Diagnostic profiler & 2D qualitative failure snapshot generator
│   │   ├── evaluate.py         # Official challenge metric evaluator (Dice & Hit Rate)
│   │   ├── preprocess.py       # MONAI patch cropping, text cache, & volume processing
│   │   └── prompt_normalizer.py# Radiology report prompt normalization
│   ├── phase_2a_rule_based/    # Phase 2A non-neural statistical baseline
│   │   └── exp_001_seg_masks_priors/ # Dedicated experiment module
│   │       ├── prior_engine.py # Core 3D spatial PDF baseline class & threshold config
│   │       ├── 01_build_spatial_pdf_cache.py # Task 1: 3D spatial heatmap accumulator
│   │       └── 02_run_inference_and_eval.py  # Tasks 2 & 3: Resample, threshold, & evaluate
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell baseline & zero-shot audit
│   │   └── exp_001_voxtell_inference.py # VoxTell v1.1 sliding window inference
│   └── phase_3_training/       # Phase 3 PyTorch semi-supervised fine-tuning
│       └── exp_001_train_mean_teacher.py # Mean Teacher + PU SPOCO fine-tuning trainer
├── .env.example                # Environment variable configuration template
├── .env                        # Local environment settings (untracked by git)
└── README.md                   # Primary repository documentation
```

---

## 🚀 Setup & Execution

### 1. Environment Setup
Activate the primary host Conda virtual environment (`voxtell_env`):
```bash
conda activate voxtell_env
# Or run explicitly with full python binary path:
# /home/jdeferrari/miniconda3/envs/voxtell_env/bin/python
```

If setting up a local virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/voxtell.txt
```

### 2. Phase 2 Baselines & Diagnostics

#### Phase 2A: Empirical 3D Spatial PDF Prior Baseline
Build 3D spatial probability density heatmaps cache (Task 1, run once):
```bash
python scripts/phase_2a_rule_based/exp_001_seg_masks_priors/01_build_spatial_pdf_cache.py
```

Run non-neural spatial probability density baseline generator and evaluator (Tasks 2 & 3):
```bash
python scripts/phase_2a_rule_based/exp_001_seg_masks_priors/02_run_inference_and_eval.py --split val --eval
```

#### Phase 2B: Zero-Shot VoxTell Baseline Inference
Run sliding window inference with 4D Back-Reorientation on validation scans:
```bash
python scripts/phase_2b_voxtell/exp_001_voxtell_inference.py
```

#### Diagnostic Failure Snapshot Generator
Harvest 2D qualitative slice snapshots for top-K worst prediction failures:
```bash
python scripts/common/analyze_predictions.py \
    --pred_dir ../data/predictions/phase_2a_rule_based \
    --gt_dir ../data/raw/segmentations \
    --img_dir ../data/raw/images \
    --split val \
    --top_k_failures 10 \
    --save_snapshots \
    --output_dir logs/phase_2a_rule_based/exp_001_seg_masks_priors/failure_snapshots
```

### 3. Metric Evaluation
Standalone official metric evaluation for predicted 4D segmentation masks against ground-truth masks:
```bash
python scripts/common/evaluate.py --gt_dir ../data/raw/segmentations --pred_dir ../data/predictions/phase_2a_rule_based --split val
```

### 4. PyTorch Mean Teacher Fine-Tuning (Phase 3)
Run persistent Mean Teacher fine-tuning with gradient clipping and float32 upcasting:
```bash
WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup python -u scripts/phase_3_training/exp_001_train_mean_teacher.py --epochs 50 --wandb > logs/phase_3_training/exp_001_train_mean_teacher/run.log 2>&1 &
```

---

## ⚙️ Shared Data & Hardware Configuration

* **Shared Data**: Dynamic path resolution in `scripts/config.py` automatically links to shared datasets in `../data/` and pretrained weights in `../models/`.
* **Fast Storage Caching**: Preprocessed volumetric tensors are cached in `/tmp/rexgroundingct_preprocessed/` (RAID SSD cache) to bypass slow CPU decompression bounds.
* **Hardware Isolation**: Pin execution to host GPU via `CUDA_VISIBLE_DEVICES=1` as documented in `.agents/STATUS.md`.

