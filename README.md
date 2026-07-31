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
│   └── shared/                 # Server-agnostic master plan and paper digests
├── logs/                       # Baseline audit & training execution logs
│   ├── phase_2a_rule_based/    # Phase 2A non-neural baseline evaluation logs
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell zero-shot baseline audit logs
│   └── phase_3_training/       # Phase 3 Mean Teacher fine-tuning logs
├── scratch/                    # Fine-tuning scratch scripts & temporary evaluation tools
├── scripts/                    # Core inference, training, & dataloading pipeline
│   ├── common/                 # Shared pipelines, evaluators, & multi-angle visualizer
│   ├── phase_2a_rule_based/    # Phase 2A non-neural statistical baseline pipeline
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell zero-shot baseline audit pipeline
│   └── phase_3_training/       # Phase 3 PyTorch semi-supervised fine-tuning pipeline
├── .env.example                # Environment variable configuration template
└── README.md                   # Primary repository documentation
```

---

## 🚀 Setup & Execution

### 1. Environment Setup
Activate the primary host Conda virtual environment (`voxtell_env`):
```bash
conda activate voxtell_env
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

#### 3D/2D Rotational Visualizer & Downloadable NIfTI ZIP Exporter
Generate per-pathology multi-angle 3D rotational viewports, 2D CT slice overlays, and 3D-dimension-matched NIfTI bundles saved to `scan_visualizations/<scan_id>/`:
```bash
python scripts/common/plot_single_case.py --scan_id train_19891_a_2
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

