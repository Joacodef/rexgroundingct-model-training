# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) supported by a comprehensive internal group technical report built on rigorous data understanding and zero-shot baseline inference mastery.

> [!IMPORTANT]
> **Phased Research Roadmap**:
> 1. **Phase 1 — ReXGroundingCT Data Profiling**: 3D CT metadata, sparse vs exhaustive mask profiling, 14 finding categories, component topology, prompt syntax, and internal group technical report.
> 2. **Phase 2 — Baselines & Preprocessing Audit**:
>    - **Phase 2A — Statistical / Rule-Based Prior Baseline**: Non-neural prior baseline leveraging Phase 1 empirical data distributions (`phase_1_priors_bundle.json`) to establish a non-learning lower-bound benchmark.
>    - **Phase 2B — VoxTell Zero-Shot Baseline & Audit**: Official `NibabelIOWithReorient` pipeline, sliding window tile overlap, per-category continuous logit threshold optimization, and failure mode profiling.
> 3. **Phase 3 — Model Fine-Tuning & Consistency Adaptations**: Systematic benchmarking of candidate loss hypotheses (PU-SPOCO, Asymmetric Focal, Soft-Label Consistency) and dynamic post-processing.

---

## 🗓️ Phased Research Roadmap & Deliverables

### Phase 1: Deep Data Profiling of ReXGroundingCT
* **Core Research Scope**:
  * **Dataset Architecture & Split Hygiene**: Patient hierarchy (Patient $\rightarrow$ Scan $\rightarrow$ Finding), cross-split leakage audit (`patient_id` grouping), and Train (sparse) vs. Val (exhaustive) annotation disparity.
  * **Free-Text Radiology Report & Prompt Dynamics**: Quantitative NLP analysis of finding descriptions, syntactic complexity, vocabulary TTR, spatial locators, and tokenizer expansion rates.
  * **Canonical 3D Spatial & Radiodensity Priors**: Canonical RAS coordinate centroids, 3D spatial probability density maps for the 14 finding categories, and Hounsfield Unit (HU) intensity window bounds (`[min_HU, max_HU]`).
  * **Component Topology & Morphology**: 3D connected-component analysis, volume distributions, sphericity indices, and category-level morphological profiles.
  * **Multi-Finding Co-Occurrence Structure**: Pairwise pathology co-occurrence matrix $P(c_i \text{ present} \mid c_j \text{ present})$ across multi-label CT scans.

* **Key Deliverables**:
  1. *Empirical Priors Bundle*: Exported metadata & prior distribution artifact (`../data/phase_1/phase_1_priors_bundle.json`) to inform downstream preprocessing and model training.
  2. *Internal Group Technical Report*: Comprehensive LaTeX report detailing dataset architecture, prompt syntax, spatial priors, topology, and actionable modeling recommendations (`logs/phase_1_report_overleaf/main.tex`).

---

### Phase 2: Baselines & Preprocessing Audit

#### Phase 2A: Statistical / Rule-Based Prior Baseline Model (First Step)
* **Core Research Scope**:
  * **Non-Neural Statistical Prior Generator**: Construct a rule-based baseline module (`scripts/baselines/rule_based_prior_baseline.py`) integrating Phase 1 prior distributions (`../data/phase_1/phase_1_priors_bundle.json`):
    * **3D Spatial Density Masking**: 3D RAS spatial probability masks and centroid bounding ellipsoids per category.
    * **HU Radiodensity Windowing**: Intensity thresholding within category-specific HU bounds ($[\text{min\_HU}, \text{max\_HU}]$).
    * **Text Prompt Spatial Parsing**: NLP directive extraction mapping anatomical locators (e.g., *right lower lobe*, *apical*) to spatial sub-volumes.
    * **Morphological Component Filtering**: 3D connected component shape and volume constraints.
  * **4D Back-Reorientation & Evaluation**: Apply standard 4D Back-Reorientation to map non-neural predictions to raw NIfTI space and evaluate on 200 validation scans via `scripts/evaluate.py`.

* **Key Deliverables**:
  1. *Statistical Prior Baseline Module*: Executable script `scripts/baselines/rule_based_prior_baseline.py`.
  2. *Empirical Baseline Metrics*: Quantitative benchmark (Dice & Hit Rate) establishing the non-learning lower-bound performance.

#### Phase 2B: VoxTell Zero-Shot Baseline & Preprocessing Audit
* **Core Research Scope**:
  * **Official Preprocessing & Reorientation Audit**: Validate 100% fidelity with official `NibabelIOWithReorient` and 4D Back-Reorientation pipeline.
  * **Inference Dynamics & Sensitivity**: Evaluate sliding window parameters (tile step size, patch padding, Gaussian weighting) and continuous logit distributions.
  * **Per-Category Logit Calibration**: Profile pre-sigmoid probability distributions per finding category to optimize binarization thresholds ($p_c$).
  * **Category-Level Failure Analysis**: Systematic error audit across all 14 finding categories isolating spatial misalignment, text shift, or suppression bias relative to Phase 2A statistical baseline.

* **Key Deliverables**:
  1. *Zero-Shot Benchmark & Error Matrix*: 200-scan validation evaluation of VoxTell v1.1 with 14-category error breakdown.
  2. *Phase 2 Baseline Audit Report*: Detailed failure mode analysis detailing root causes and actionable fine-tuning hypotheses (`logs/phase_2_inference_audit/`).

---

### Phase 3: Fine-Tuning & Model Adaptations
* **Core Research Scope**:
  * **Supervised & Loss Hypotheses Benchmarking**: Evaluate competing loss strategies to address sparse-to-exhaustive annotation disparity:
    * **H1**: Positive-Unlabeled (PU) SPOCO loss.
    * **H2**: Asymmetric Focal Loss / Focal Dice.
    * **H3**: Soft Pseudo-Label & Mean Teacher Logit Consistency.
  * **Multi-GPU Scaled Training**: Scale training across full 2,992-scan training split using RAID SSD volume caching (`/tmp/rexgroundingct_preprocessed/`).
  * **Dynamic Post-Processing & Ensembling**: Multi-checkpoint ensembling, category-validated dynamic component filtering, and 4D test submission generator.

* **Key Deliverables**:
  1. *Fine-Tuned Model Checkpoints & Pipeline*: Validated trainer and high-performing model weights resolving instance suppression bias.
  2. *Final Submission Package & Paper Manuscript*: Leaderboard submission package and MICCAI research manuscript.

---

## 🔬 Directory Conventions & Artifact Mapping
Exploratory fine-tuning scripts and logs are organized in phase-specific subfolders:
* `logs/phase_1_report_overleaf/`: LaTeX manuscript source for Phase 1 technical report.
* `logs/`: Individual experiment reports and logs (`exp_001_...md` to `exp_005_...md`).
* `scripts/`: Execution scripts (`exp_001_...py` to `exp_005_...py`).
