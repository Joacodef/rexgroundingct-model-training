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
├── logs/                       # Baseline audit & training execution logs
│   ├── common/                 # System logs and utility output
│   ├── phase_2a_rule_based/    # Phase 2A non-neural baseline evaluation logs
│   ├── phase_2b_voxtell/       # Phase 2B VoxTell zero-shot baseline audit logs
│   └── phase_3_training/       # Phase 3 Mean Teacher fine-tuning logs
├── scratch/                    # Fine-tuning scratch scripts & evaluation tools
├── scripts/                    # Core inference, training, & dataloading pipeline
│   ├── config.py               # Dynamic path resolver (shared ../data/ and ../models/)
│   ├── common/                 # Shared pipelines & utilities
│   │   ├── evaluate.py         # Official challenge metric evaluator (Dice & Hit Rate)
│   │   ├── preprocess.py       # MONAI patch cropping, text cache, & volume processing
│   │   └── prompt_normalizer.py# Radiology report prompt normalization
│   ├── phase_2a_rule_based/    # Phase 2A non-neural statistical baseline
│   │   └── exp_001_seg_masks_priors.py # Empirical 3D spatial PDF baseline
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
Activate the standard local virtual environment:
```bash
source .venv/bin/activate
```

If setting up on a new machine or GPU cluster node:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/voxtell.txt
```

### 2. Zero-Shot Baseline Inference (Phase 2)
Run sliding window inference with 4D Back-Reorientation on validation scans:
```bash
python scripts/phase_2b_voxtell/exp_001_voxtell_inference.py
```

### 3. Metric Evaluation
Evaluate predicted 4D segmentation masks against ground-truth masks:
```bash
python scripts/common/evaluate.py --gt_dir ../data/raw/segmentations --pred_dir ../data/predictions
```

### 4. PyTorch Mean Teacher Fine-Tuning (Phase 3)
Run persistent Mean Teacher fine-tuning with gradient clipping and float32 upcasting:
```bash
WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup python -u scripts/phase_3_training/exp_001_train_mean_teacher.py --epochs 50 --wandb > logs/phase_3_training/train_mean_teacher_50ep.log 2>&1 &
```

---

## ⚙️ Shared Data & Hardware Configuration

* **Shared Data**: Dynamic path resolution in `scripts/config.py` automatically links to shared datasets in `../data/` and pretrained weights in `../models/`.
* **Fast Storage Caching**: Preprocessed volumetric tensors are cached in `/tmp/rexgroundingct_preprocessed/` (RAID SSD cache) to bypass slow CPU decompression bounds.
* **Hardware Isolation**: Pin execution to host GPU via `CUDA_VISIBLE_DEVICES=1` as documented in `.agents/STATUS.md`.
