# Master Plan — ReXGroundingCT Challenge 2026

**Primary Goal:** Top-3 on the leaderboard (September 2026) AND an original paper accepted at MICCAI 2026, built on rigorous data understanding and zero-shot baseline inference mastery.

> [!IMPORTANT]
> **Phased Research Roadmap**:
> 1. **Phase 1 — ReXGroundingCT Data Profiling**: 3D CT metadata, sparse vs exhaustive mask profiling, 14 finding categories, component topology, and prompt syntax.
> 2. **Phase 2 — VoxTell Zero-Shot Baseline & Audit**: Official `NibabelIOWithReorient` pipeline, sliding window tile overlap, continuous logit distributions, and failure modes.
> 3. **Phase 3 — Model Fine-Tuning & Consistency Adaptations**: Supervised fine-tuning, Positive-Unlabeled (PU) SPOCO, and MPR consistency learning.

---

## 🗓️ Phased Research Roadmap & Deliverables

### Phase 1: Deep Data Profiling of ReXGroundingCT
* **Core Research Scope**:
  * **3D CT Image & Metadata Profiling**: Voxel spacings, orientation affines, physical dimensions, intensity distributions, and longitudinal field-of-view (FOV) bounds.
  * **Exhaustive vs Sparse Mask Analysis**: Quantitative comparison of ground-truth mask distributions between the training set (partially annotated) and validation set (exhaustively annotated).
  * **Positive-Unlabeled (PU) Noise & Inter-Class Overlap Profiling**: Empirical unannotated true-positive rate per category in background voxels and inter-class voxel-level IoU.
  * **The 14 Official Finding Categories**: Detailed error profiling across the 14 challenge categories:
    * *Non-focal (6)*: Bronchial wall thickening, Bronchiectasis, Emphysema, Septal thickening, Micronodules, Other non-focal.
    * *Focal (8)*: Linear opacities, Atelectasis / consolidation, Ground-glass opacity, Pulmonary nodules / masses, Pleural effusion / thickening, Honeycombing, Pneumothorax, Other focal.
  * **Finding Volume, Component & Topology Statistics**: Component counts, voxel volumes, spatial centroids, sphericity indices, and surface area-to-volume ratios.
  * **Free-Text Radiology Report & Spatial Directive Analysis**: Quantitative NLP analysis of finding descriptions in `dataset.json` (syntax, modifier adjectives, spatial locators, length, and anatomical jargon).

* **Expected Deliverables & Outputs**:
  1. *Statistical & Annotation Disparity Analysis*: Quantitative profiling of sparse vs. exhaustive mask distributions in Train (~1 mask/scan) vs Val (~3 masks/scan).
  2. *3D Average Mask & Spatial Density Maps*: Canonical RAS ($128 \times 128 \times 128$) probability density maps $P(\mathbf{x}' \in \text{mask} \mid c)$, 2D AIP projections, and spatial centroids for all 14 categories.
  3. *Free-Text Prompt & NLP Shift Analysis*: Quantitative analysis of word counts, punctuation shifts, syntactic complexity, TTR (-45.2%), and prompt normalization trade-offs.
  4. *Hounsfield Unit (HU) Radiodensity Profiling*: Category-level HU intensity distributions inside mask regions vs. healthy lung parenchyma to define optimal preprocessing intensity windowing (`[min_HU, max_HU]`).
  5. *Physical Resolution & Voxel Spacing Profiling*: Distribution of slice thickness ($\Delta z$) and in-plane voxel dimensions ($\Delta x, \Delta y$) across Train and Val splits.
  6. *3D Bounding Box Scale & Aspect Ratio Profiling*: Spatial extents ($\Delta X, \Delta Y, \Delta Z$) and volume aspect ratios per pathology.
  7. *Multi-Finding Co-Occurrence Matrix*: Pairwise co-occurrence matrix $P(c_i \text{ present} \mid c_j \text{ present})$ across CT scans.
  8. *Multi-Instance Component Profiling*: Connected-component analysis of mask fragmentation and instance count distributions.
  9. *PU Background Contamination & Inter-Class Overlap Profiling*: Empirical unannotated true-positive rate and voxel-level inter-class IoU overlap matrix.
  10. *Text-Spatial Directive & Syntax Parsing*: Parsing of spatial prepositions, compound prompts, and hedging.
  11. *Patient & Series Reconstruction Hierarchy*: 3-tier ID decomposition and cross-split patient leakage audit.
  12. *Comprehensive Phase 1 Technical Report*: Consolidated technical report detailing dataset architecture, spatial priors, prompt syntax, topology, and actionable recommendations (`logs/phase_1_data_profiling/phase_1_technical_report.md`).

---

### Phase 2: VoxTell Zero-Shot Baseline & Preprocessing Audit
* **Core Research Scope**:
  * **Official Preprocessing & Reorientation**: Audit official `nnunetv2.imageio.nibabel_reader_writer.NibabelIOWithReorient` and `VoxTellPredictor` to ensure 100% fidelity with the authors' intended input pipeline.
  * **Sliding Window Hyperparameter Sensitivity**: Evaluate tile step size (`tile_step_size` 0.5 vs 0.25), Gaussian tile weighting, and patch padding on prediction quality.
  * **Continuous Logit & Threshold Profiling**: Analyze raw sigmoid output probabilities prior to binarization (`> 0.5`) to determine whether false negatives are caused by low probability magnitude or spatial misalignment.
  * **Category-Level Failure Mode Profiling**: Systematically identify which of the 14 categories succeed zero-shot and which fail, analyzing spatial and text characteristics of failure cases.

* **Expected Deliverables & Outputs**:
  1. *Zero-Shot Baseline Benchmark*: Full 200-scan validation evaluation of VoxTell v1.1 using official `NibabelIOWithReorient` and 4D Back-Reorientation pipeline.
  2. *14-Category Error Breakdown Matrix*: Category-level Dice and Hit Rate ($\ge 0.1$) performance matrix isolating high-performing vs failing findings.
  3. *Sliding Window Sensitivity Study*: Quantitative evaluation of tile step size (`tile_step_size` 0.5 vs 0.25), patch padding, and Gaussian tile weighting.
  4. *Continuous Logit & Threshold Profiling*: Pre-sigmoid logit probability distribution analysis and category-specific binarization threshold optimization.
  5. *Phase 2 Failure Mode Audit Report*: Comprehensive failure analysis mapping root causes (text shift, spatial misalignment, low logit magnitude, or suppression bias) per category.

---

### Phase 3: Fine-Tuning & Model Adaptations
* **Core Research Scope**:
  * **Model Weight Adaptation**: Supervised fine-tuning using partial-annotation and consistency loss formulations.
  * **Positive-Unlabeled (PU) SPOCO + MPR Consistency**: Resolve instance suppression bias via PU-SPOCO loss and multi-planar reconstruction (MPR) consistency.
  * **Multi-GPU Scaled Training**: Scale training across full 2,992-scan training split using RAID SSD volume caching.

* **Expected Deliverables & Outputs**:
  1. *Stabilized Fine-Tuning Pipeline*: Robust trainer supporting float32 loss upcasting, L2 gradient clipping, and fast RAID SSD volume caching.
  2. *Positive-Unlabeled (PU) SPOCO + MPR Consistency Training*: Fine-tuned model resolving instance suppression bias via PU-SPOCO loss and multi-planar reconstruction (MPR) consistency.
  3. *Multi-GPU Scaled Fine-Tuning*: Scaled training runs on full 2,992-scan training split.
  4. *Ensemble & Post-Processing Pipeline*: Multi-checkpoint ensemble and 4D Back-Reorientation test submission generator.
  5. *Final Submission Package & MICCAI Paper Manuscript*: Competition submission artifact and research manuscript.

---

## 🔬 Directory Conventions & Artifact Mapping
Exploratory fine-tuning scripts and logs are organized in phase-specific subfolders:
* `logs/phase_1_data_profiling/`: Data profiling logs and Overleaf manuscript source.
* `logs/phase_2_inference_audit/`: VoxTell baseline evaluation logs and error audits.
* `logs/phase_3_fine_tuning/proof_of_concept/`: Phase 3 proof-of-concept experiment logs.
* `scratch/phase_3_fine_tuning/proof_of_concept/`: Phase 3 proof-of-concept training and evaluation scripts.
