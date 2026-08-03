# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) supported by a comprehensive internal group technical report built on rigorous data understanding and zero-shot baseline inference mastery.

> [!IMPORTANT]
> **Phased Research Roadmap**:
> 1. **Phase 1 — ReXGroundingCT Data Profiling**: 3D CT metadata, sparse vs. exhaustive mask profiling, 14 finding categories, component topology, prompt syntax, and internal group technical report.
> 2. **Phase 2 — Baselines & Preprocessing Audit**:
>    - **Phase 2A — Statistical / Rule-Based Prior Baseline**: Non-neural prior baseline leveraging Phase 1 empirical data distributions to establish a non-learning lower-bound benchmark.
>    - **Phase 2B — VoxTell Zero-Shot Baseline & Audit**: Official reorientation pipeline, sliding window tile overlap, per-category continuous logit threshold optimization, and failure mode profiling.
> 3. **Phase 3 — Model Fine-Tuning & Consistency Adaptations**: Systematic benchmarking of candidate loss hypotheses (PU-SPOCO, Asymmetric Focal, Soft-Label Consistency) and dynamic post-processing.

---

## 🗓️ Phased Research Roadmap & Deliverables

### Phase 1: Deep Data Profiling of ReXGroundingCT
* **Core Research Scope**:
  * **Dataset Architecture & Split Hygiene**: Patient hierarchy (Patient $\rightarrow$ Scan $\rightarrow$ Finding), cross-split leakage audit (`patient_id` grouping), and Train (sparse) vs. Val (exhaustive) annotation disparity.
  * **Free-Text Radiology Report & Prompt Dynamics**: Quantitative NLP analysis of finding descriptions, syntactic complexity, vocabulary TTR, spatial locators, and tokenizer expansion rates.
  * **Canonical 3D Spatial & Radiodensity Priors**: Canonical RAS coordinate centroids, 3D spatial probability density maps for the 14 finding categories, and Hounsfield Unit (HU) intensity window bounds ($[\text{min\_HU}, \text{max\_HU}]$).
  * **Component Topology & Morphology**: 3D connected-component analysis, volume distributions, sphericity indices, and category-level morphological profiles.
  * **Multi-Finding Co-Occurrence Structure**: Pairwise pathology co-occurrence matrix $P(c_i \text{ present} \mid c_j \text{ present})$ across multi-label CT scans.

* **Key Deliverables**:
  1. *Empirical Priors Bundle*: Exported metadata & prior distribution artifacts to inform downstream preprocessing and model training.
  2. *Internal Group Technical Report*: Comprehensive LaTeX report detailing dataset architecture, prompt syntax, spatial priors, topology, and actionable modeling recommendations.

---

### Phase 2: Baselines & Preprocessing Audit

#### Phase 2A: Statistical / Rule-Based Prior Baseline Model (First Step)
* **Core Research Scope**:
  * **Non-Neural Statistical Prior Generator**: Construct a rule-based baseline module integrating Phase 1 prior distributions:
    * **3D Spatial Density Masking**: 3D RAS spatial probability masks and centroid bounding ellipsoids per category.
    * **HU Radiodensity Windowing**: Intensity thresholding within category-specific HU bounds ($[\text{min\_HU}, \text{max\_HU}]$).
    * **Text Prompt Spatial Parsing**: NLP directive extraction mapping anatomical locators (e.g., *right lower lobe*, *apical*) to spatial sub-volumes.
    * **Morphological Component Filtering**: 3D connected component shape and volume constraints.
  * **Validation Benchmark Evaluation**: Evaluate non-neural statistical prior performance on validation scans to establish non-learning lower-bound benchmarks (Dice & Hit Rate).

* **Key Deliverables**:
  1. *Statistical Prior Baseline Module*: Executable non-neural prior generator module.
  2. *Empirical Baseline Metrics*: Quantitative benchmark (Dice & Hit Rate) establishing the non-learning lower-bound performance.

#### Phase 2B: VoxTell Zero-Shot Baseline & Preprocessing Audit
* **Core Research Scope**:
  * **Zero-Shot Baseline Evaluation**: Establish official zero-shot validation benchmarks and inference fidelity across validation scans.
  * **Inference Dynamics & Sensitivity**: Evaluate sliding window parameters (tile step size, patch padding, Gaussian weighting) and continuous logit distributions.
  * **Per-Category Logit Calibration**: Profile pre-sigmoid probability distributions per finding category to optimize binarization thresholds ($p_c$).
  * **Category-Level Failure Analysis**: Systematic error audit across all 14 finding categories isolating spatial misalignment, text shift, or suppression bias relative to Phase 2A statistical baseline.

* **Key Deliverables**:
  1. *Zero-Shot Benchmark & Error Matrix*: Validation evaluation of VoxTell baseline with 14-category error breakdown.
  2. *Phase 2 Baseline Audit Report*: Detailed failure mode analysis detailing root causes and actionable fine-tuning hypotheses.

---

### Phase 3: Fine-Tuning & Model Adaptations
* **Core Research Scope**:
  * **Supervised & Loss Hypotheses Benchmarking**: Evaluate competing loss strategies to address sparse-to-exhaustive annotation disparity:
    * **H1**: Positive-Unlabeled (PU) SPOCO loss.
    * **H2**: Asymmetric Focal Loss / Focal Dice.
    * **H3**: Soft Pseudo-Label & Mean Teacher Logit Consistency.
  * **Multi-GPU Scaled Training**: Scale training across full training split using fast SSD volume caching.
  * **Dynamic Post-Processing & Ensembling**: Multi-checkpoint ensembling, category-validated dynamic component filtering, and 4D test submission generator.

* **Key Deliverables**:
  1. *Fine-Tuned Model Checkpoints & Pipeline*: Validated trainer and high-performing model weights resolving instance suppression bias.
  2. *Final Submission Package & Paper Manuscript*: Leaderboard submission package and MICCAI research manuscript.


