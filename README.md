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
│   ├── phase_2_inference_audit/ # Baseline audit logs & failure breakdown matrices
│   ├── phase_3_fine_tuning/     # Mean Teacher fine-tuning proof-of-concept logs
│   └── execution_raw/          # Detached execution logs and process IDs
├── scratch/                    # Fine-tuning scratch scripts & evaluation tools
│   ├── phase_2_inference_audit/ # Chunked inference & official pipeline verifiers
│   └── phase_3_fine_tuning/     # Proof-of-concept training & diagnostic utils
├── scripts/                    # Core inference, training, & dataloading pipeline
│   ├── config.py               # Dynamic path resolver (shared ../data/ and ../models/)
│   ├── evaluate.py             # Official challenge metric evaluator (Dice & Hit Rate)
│   ├── data_prep/
│   │   └── preprocess.py       # MONAI patch cropping, text cache, & volume processing
│   └── voxtell/
│       ├── prompt_normalizer.py# Radiology report prompt normalization
│       ├── voxtell_inference.py# Zero-shot sliding window inference & 4D Back-Reorientation
│       └── training/
│           └── train_mean_teacher.py # PyTorch Mean Teacher + PU SPOCO fine-tuning trainer
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
python scripts/voxtell/voxtell_inference.py
```

### 3. Metric Evaluation
Evaluate predicted 4D segmentation masks against ground-truth masks:
```bash
python scripts/evaluate.py --gt_dir ../data/raw/segmentations --pred_dir ../data/predictions
```

### 4. PyTorch Mean Teacher Fine-Tuning (Phase 3)
Run persistent Mean Teacher fine-tuning with gradient clipping and float32 upcasting:
```bash
WANDB_MODE=offline PYTHONUNBUFFERED=1 nohup python -u scripts/voxtell/training/train_mean_teacher.py --epochs 50 --wandb > logs/execution_raw/train_mean_teacher_50ep.log 2>&1 &
```

---

## ⚙️ Shared Data & Hardware Configuration

* **Shared Data**: Dynamic path resolution in `scripts/config.py` automatically links to shared datasets in `../data/` and pretrained weights in `../models/`.
* **Fast Storage Caching**: Preprocessed volumetric tensors are cached in `/tmp/rexgroundingct_preprocessed/` (RAID SSD cache) to bypass slow CPU decompression bounds.
* **Hardware Isolation**: Pin execution to host GPU via `CUDA_VISIBLE_DEVICES=1` as documented in `.agents/STATUS.md`.
