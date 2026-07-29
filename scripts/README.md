# Scripts Directory Index

This directory contains all operational scripts for the **ReXGroundingCT Model Training & Fine-Tuning Workspace**. Subdirectories are structured according to the research phase scheme to provide clear mapping across research objectives.

---

## 📁 Subdirectory Architecture

```text
scripts/
├── config.py                               # Dynamic path resolver & 14-category map
├── common/                                 # Shared pipelines & utilities
│   ├── evaluate.py                         # Official MICCAI evaluation metric calculator
│   ├── preprocess.py                       # MONAI volume reorientation & 1.5mm isotropic resampling
│   └── prompt_normalizer.py                # Hybrid medical report prompt cleaner & entity normalizer
│
├── phase_2a_rule_based/                    # Phase 2A: Non-Neural Statistical Baselines
│   └── exp_001_seg_masks_priors.py         # Empirical 3D spatial PDF baseline predictor
│
├── phase_2b_voxtell/                       # Phase 2B: Pre-Trained VoxTell Baseline & Audit
│   └── exp_001_voxtell_inference.py        # VoxTell v1.1 sliding window zero-shot inference pipeline
│
└── phase_3_training/                       # Phase 3: Semi-Supervised Fine-Tuning
    └── exp_001_train_mean_teacher.py       # Mean Teacher fine-tuning loop with EMA & float32 upcasting
```

---

## 📌 Phase-to-Script Mapping Matrix

| Category / Phase | Directory Path | Key Script | Objective & Description | Key Outputs / Artifacts |
|---|---|---|---|---|
| **Common Utilities** | **[`common/`](file:///home/jdeferrari/rex_project/rexgroundingct-model-training/scripts/common)** | `evaluate.py`<br>`preprocess.py`<br>`prompt_normalizer.py` | Official evaluation metrics, MONAI 1.5mm isotropic volume caching, and radiology report prompt normalization. | Evaluation JSONs, cached SSD float32 tensors, normalized prompts |
| **Phase 2A Baseline** | **[`phase_2a_rule_based/`](file:///home/jdeferrari/rex_project/rexgroundingct-model-training/scripts/phase_2a_rule_based)** | `exp_001_seg_masks_priors.py` | Data-driven non-neural baseline accumulating 3D empirical spatial PDF heatmaps ($P_c(z,y,x)$) from training segmentations. | `../data/phase_2a/empirical_spatial_pdf_14cat.npz`, `../data/predictions/phase_2a_rule_based/` |
| **Phase 2B Baseline** | **[`phase_2b_voxtell/`](file:///home/jdeferrari/rex_project/rexgroundingct-model-training/scripts/phase_2b_voxtell)** | `exp_001_voxtell_inference.py` | Pre-trained VoxTell zero-shot baseline inference & 4D Back-Reorientation coordinate mapping. | `../data/predictions/voxtell_baseline/`, `logs/phase_2b_voxtell/` |
| **Phase 3 Training** | **[`phase_3_training/`](file:///home/jdeferrari/rex_project/rexgroundingct-model-training/scripts/phase_3_training)** | `exp_001_train_mean_teacher.py` | Semi-supervised Mean Teacher fine-tuning loop with float32 upcasting, gradient clipping, & EMA. | Checkpoints in `checkpoints/`, `logs/phase_3_training/` |

---

## 📜 Directives & Conventions

1. **Phase-Prefixed & Common Subdirectories:** All scripts MUST reside in their respective functional directory (`common`, `phase_2a_rule_based`, `phase_2b_voxtell`, `phase_3_training`).
2. **Experiment Naming:** Experiment scripts follow `exp_XXX_<description>.py` and output to matching `logs/<phase>/exp_XXX_<description>.md` files.
3. **Relative Path Directives:** ALL imports, file paths, and documentation links MUST strictly use **relative paths**.
