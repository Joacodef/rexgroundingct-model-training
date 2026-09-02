# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) supported by a comprehensive internal group technical report built on rigorous data understanding and zero-shot baseline inference mastery.

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

#### Phase 2B: Off-the-Shelf VoxTell Pre-Trained Baseline & Diagnostics
* **Core Research Scope**:
  * **Off-the-Shelf Baseline Evaluation**: Establish pre-trained foundation model validation benchmarks and inference fidelity across validation scans.
  * **Inference Dynamics & Sensitivity**: Evaluate sliding window parameters (tile step size, patch padding, Gaussian weighting) and continuous logit distributions.
  * **Per-Category Logit Calibration**: Profile pre-sigmoid probability distributions per finding category to optimize binarization thresholds ($p_c$).
  * **Category-Level Failure Analysis**: Systematic error audit across all 14 finding categories isolating spatial misalignment, text shift, or suppression bias relative to Phase 2A statistical baseline.

* **Key Deliverables**:
  1. *Off-the-Shelf Benchmark & Error Matrix*: Validation evaluation of VoxTell baseline with 14-category error breakdown.
  2. *Phase 2 Baseline Audit Report*: Detailed failure mode analysis detailing root causes and actionable fine-tuning hypotheses.

---

### Phase 3: VoxTell Model Fine-Tuning
* **Core Research Scope**:
  * **Targeted Loss Benchmarking**: Evaluate 3 specific loss strategies to address sparse-to-exhaustive annotation disparity on VoxTell:
    * **Exp 001**: Naïve Supervised Baseline (BCE + Dice).
    * **Exp 002**: Positive-Unlabeled (PU) Mean Teacher Loss.
    * **Exp 003**: Multi-Planar Projection Regularization (MPR) Loss *(Delegated to `peteroa` production cluster for multi-GPU batch throughput)*.
  * **Fast Volume Acceleration**: Execute training using fast local SSD volume caching.

* **Key Deliverables**:
  1. *VoxTell Fine-Tuned Model Checkpoints*: Validated VoxTell trainer and high-performing model weights.
  2. *Leaderboard Test Submission*: Official challenge submission package.

---

### Phase 4: Metric Learning and Voxel Embeddings via VoxTell-SPOCO
* **Core Research Scope**:
  * **Metric-Learning & Continuous Hypersphere Exploration**: Adapt the VoxTell vision-language foundation model for continuous metric representation learning on a 32D unit hypersphere ($\mathbb{S}^{31}$) via Sparse Object-Level Consistency (SPOCO, Wolny et al., CVPR 2022) to resolve the sparse-to-exhaustive annotation gap and false-negative penalties across four structured experiments:
    * **Exp 001: Canonical SPOCO Foundation Baseline** *(Delegated to `ih-condor` compute node)*: Full fine-tuning baseline directly leveraging VoxTell's native 32-channel decoder feature map with single-anchor connected-component supervision ($L_{\text{obj}}$), unannotated iterative coverage suppression consistency ($L_{\text{con}}$), and subsampled background push repulsion ($L_{\text{unl\_push}}$).
    * **Exp 002: Parameter-Efficient Adapters vs.\ Full Fine-Tuning for Representation Preservation (Hypothesis H1)**: Testing whether freezing the pre-trained vision-language backbone and training lightweight residual 3D convolutional adapter layers prevents catastrophic forgetting of anatomical priors while learning metric hyperspherical embeddings.
    * **Exp 003: Multi-Anchor Volumetric Scaling for Conglomerate \& Diffuse Pathologies (Hypothesis H2)**: Testing dynamic multi-anchor sampling scaled by component volume ($K_c \propto V_c^{1/3}$) along the 3D medial axis to eliminate variance strain on large non-focal entities (e.g., lobar consolidation, massive effusions, diffuse emphysema).
    * **Exp 004: Morphology-Adaptive Margin \& Kernel Calibration (Hypothesis H3)**: Testing category-adaptive variance margins $\delta_{\text{var}}(c) = \delta_0 (1 - \beta S_c)$ calibrated to anatomical sphericity ($S$) established in Phase~1 (tight clusters for compact nodules vs.\ flexible envelopes for sheet-like infiltrates).

* **Key Deliverables**:
  1. *VoxTell-SPOCO Model Suite \& Checkpoints*: Validated metric learning trainers, checkpoints, and benchmark reports across hypotheses.
  2. *Final Research Manuscript \& Technical Report*: Comprehensive synthesis detailing multi-phase grounding results.
