# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) supported by a comprehensive internal group technical report built on rigorous data understanding and zero-shot baseline inference mastery.

> [!IMPORTANT]
> **Phased Research Roadmap**:
> 1. **Phase 1 — ReXGroundingCT Data Profiling**: 3D CT metadata, sparse vs exhaustive mask profiling, 14 finding categories, component topology, prompt syntax, and internal group technical report.
> 2. **Phase 2 — VoxTell Zero-Shot Baseline & Audit**: Official `NibabelIOWithReorient` pipeline, sliding window tile overlap, continuous logit distributions, and failure modes.
> 3. **Phase 3 — Model Fine-Tuning & Consistency Adaptations**: Supervised fine-tuning, Positive-Unlabeled (PU) SPOCO, and MPR consistency learning.

---

## 🗓️ Phased Research Roadmap & Deliverables

### Phase 1: Deep Data Profiling of ReXGroundingCT
* **Core Research Scope**:
  * **Dataset Architecture & Split Hygiene**: Patient hierarchy (Patient $\rightarrow$ Scan $\rightarrow$ Finding), cross-split leakage audit, and Train (sparse) vs. Val (exhaustive) annotation disparity.
  * **Free-Text Radiology Report & Prompt Dynamics**: Quantitative NLP analysis of finding descriptions, syntactic complexity, vocabulary TTR, spatial locators, and tokenizer expansion rates.
  * **Canonical 3D Spatial & Radiodensity Priors**: Canonical RAS coordinate centroids, 3D spatial probability density maps for the 14 finding categories, and Hounsfield Unit (HU) intensity window bounds (`[min_HU, max_HU]`).
  * **Component Topology & Morphology**: 3D connected-component analysis, volume distributions, sphericity indices, and empirical noise pruning size thresholds.
  * **Multi-Finding Co-Occurrence Structure**: Pairwise pathology co-occurrence matrix $P(c_i \text{ present} \mid c_j \text{ present})$ across multi-label CT scans.

* **Key Deliverables**:
  1. *Empirical Priors Bundle*: Exported metadata & prior distribution artifact (`../data/phase_1/phase_1_priors_bundle.json`) to inform downstream preprocessing and model training.
  2. *Internal Group Technical Report*: Comprehensive LaTeX report detailing dataset architecture, prompt syntax, spatial priors, topology, and actionable modeling recommendations (`logs/phase_1_report_overleaf/main.tex`).

---

### Phase 2: VoxTell Zero-Shot Baseline & Preprocessing Audit
* **Core Research Scope**:
  * **Official Preprocessing & Reorientation Audit**: Validate 100% fidelity with the official `NibabelIOWithReorient` and 4D Back-Reorientation pipeline.
  * **Inference Dynamics & Sensitivity**: Evaluate sliding window parameters (tile step size, patch padding, Gaussian weighting) and continuous logit distributions.
  * **Category-Level Failure Analysis**: Systematic error audit across all 14 finding categories isolating spatial misalignment, text shift, or suppression bias.

* **Key Deliverables**:
  1. *Zero-Shot Benchmark & Error Matrix*: 200-scan validation evaluation of VoxTell v1.1 with 14-category error breakdown.
  2. *Phase 2 Baseline Audit Report*: Detailed failure mode analysis detailing root causes and actionable fine-tuning hypotheses.

---

### Phase 3: Fine-Tuning & Model Adaptations
* **Core Research Scope**:
  * **Supervised & Consistency Adaptations**: Fine-tune VoxTell using Positive-Unlabeled (PU) SPOCO loss and Multi-Planar Reconstruction (MPR) consistency learning.
  * **Multi-GPU Scaled Training**: Scale training across full 2,992-scan training split using RAID SSD volume caching.
  * **Post-Processing & Ensembling**: Multi-checkpoint ensembling, noise pruning, and 4D test submission generator.

* **Key Deliverables**:
  1. *Fine-Tuned Model Checkpoints & Pipeline*: Validated trainer and high-performing model weights resolving instance suppression bias.
  2. *Final Submission Package & Paper Manuscript*: Leaderboard submission package and MICCAI research manuscript.

---

## 🔬 Directory Conventions & Artifact Mapping
Exploratory fine-tuning scripts and logs are organized in phase-specific subfolders:
* `logs/phase_1_report_overleaf/`: LaTeX manuscript source for Phase 1 technical report.
* `logs/`: Individual experiment reports and logs (`exp_001_...md` to `exp_005_...md`).
* `scripts/`: Execution scripts (`exp_001_...py` to `exp_005_...py`).
