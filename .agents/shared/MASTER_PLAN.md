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

### Phase 3: VoxTell Model Fine-Tuning
* **Core Research Scope**:
  * **Targeted Loss Benchmarking**: Evaluate 3 specific loss strategies to address sparse-to-exhaustive annotation disparity on VoxTell:
    * **Exp 001**: Naïve Supervised Baseline (BCE + Dice).
    * **Exp 002**: Positive-Unlabeled (PU) Mean Teacher Loss.
    * **Exp 003**: Multi-Planar Projection Regularization (MPR) Loss.
  * **Fast Volume Acceleration**: Execute training using fast local SSD volume caching.

* **Key Deliverables**:
  1. *VoxTell Fine-Tuned Model Checkpoints*: Validated VoxTell trainer and high-performing model weights.
  2. *Leaderboard Test Submission*: Official challenge submission package.

---

### Phase 4: Alternative Architectures & Unbiased Models
* **Core Research Scope**:
  * **Clean-Slate Architecture Exploration**: Evaluate alternative 3D vision-language grounding models not pre-trained on ReXGroundingCT data.
  * **True SPOCO Evaluation**: Test metric-learning pixel embedding segmentation with anchor soft masks and clustering (Wolny et al., CVPR 2022) on compatible clean-slate architectures.
  * **Comparative Cross-Architecture Benchmark**: Systematic benchmark comparing VoxTell against clean-slate 3D grounding backbones to isolate pre-training bias vs. architectural strength.

* **Key Deliverables**:
  1. *Alternative Model Suite & Benchmarks*: Validated trainers and benchmark reports across alternative 3D grounding backbones.
  2. *Final Research Manuscript*: Comprehensive research manuscript detailing multi-model comparison.
