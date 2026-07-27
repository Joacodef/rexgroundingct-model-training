# Experiment Log 001: [Phase 2] VoxTell v1.0 - Resampled Preprocessing Failure and Catastrophic Spatial Disalignment

* **Date:** May 17, 2026  

* **Project Milestone:** Milestone 1 (June 1, 2026) - Baseline and Methodological Verification  
* **Status:** **FAILED & ABANDONED** (This pipeline was officially retired due to catastrophic metrics; all core lessons were used to design the successful native-resolution pipeline in Exp 001)

---

> [!WARNING]
> **Executive Post-Mortem Conclusion:**
> This experiment was a **complete failure**, yielding a near-zero average Dice score of **`0.0083`**. The failure was caused by two main factors:
> 1. **Catastrophic Spatial Mismatch:** A metadata bug in the official ground-truth labels combined with axis-swapping inside the nnU-Net reader completely scrambled the spatial coordinate alignment between predictions and ground truths.
> 2. **Severe Normalization OOD Shift:** Downsampling to 1.5mm isotropic spacing and applying custom lung-window clipping (`[-1000, 400]`) with localized Z-scores created an out-of-distribution (OOD) intensity shift that blinded the zero-shot baseline model.
>
> **Action taken:** This resampled preprocessing pipeline has been permanently abandoned. We transitioned to a native-resolution, 4D Back-Reorientation pipeline in `exp_002_voxtell_v1.1_baseline_verification.md`, which successfully restored alignment and achieved a baseline Average Dice of **`0.2139`**.

---

## 1. Objective & Methodological Hypothesis
The initial phase of Milestone 1 sought to establish the zero-shot validation and preprocessing pipeline for the **ReXGroundingCT 2026 Challenge** using the baseline **VoxTell v1.0** model.

To standardize the inputs across the multi-modality CT-RATE corpus and reduce computational complexity, we proposed a traditional medical image processing pipeline with the following hypotheses:
* **Spatial Standardization:** Resampling both high-resolution CT volumes and their corresponding Ground Truth segmentations to a standardized **1.5mm isotropic spacing** (using MONAI spatial transforms) would provide a clean, spatially homogeneous domain for convolutional processing.
* **Targeted Clinical Normalization:** Restricting Hounsfield Units (HU) to a lung window of `[-1000, 400]` and calculating localized Z-score normalization (excluding background air/padding) would enhance clinical contrast and boost zero-shot localization performance.

---

## 2. Chronology of Diagnostic Investigations (Spatial & Semantic Pitfalls)
During pipeline integration, we observed that the initial zero-shot outputs yielded completely empty segmentations or near-zero overlap. Deep diagnostics revealed three severe technical and geometric barriers that broke the baseline evaluation:

### A. Voxel Spacing & Metadata Affine Discrepancy
* **Physical Diagnostic:** The raw CT scans (`images/`) contained correct, non-identity affine matrices (representing true voxel spacings of `~0.574 mm x 0.574 mm x 1.5 mm` and orientation flips). However, the Ground Truth masks (`segmentations/`) were saved with a generic **Identity Affine** (`np.eye(4)`), signifying a generic `1.0 mm` spacing.
* **Mechanism of Failure:** When MONAI's `Spacingd(..., pixdim=(1.5, 1.5, 1.5))` was applied:
  * The CT image was resampled relative to its physical spacing ratio (`0.574 / 1.5`), yielding `197` slices.
  * The Ground Truth mask was resampled relative to the identity spacing ratio (`1.0 / 1.5`), yielding `342` slices.
  * This discrepancy caused a catastrophic shape mismatch: `GT (4, 342, 342, 137) vs Pred (4, 197, 197, 205)`.
* **Resolution:** Implemented an explicit header-correction pass (`CopyAffined`) to overwrite the corrupt GT identity affine matrix with the correct, physical CT affine matrix prior to resampling.

### B. Anatomical Axis Permutation via Automated nnU-Net Reorientation
* **Physical Diagnostic:** The baseline pipeline loaded CT volumes using nnU-Net's internal `NibabelIOWithReorient()`, which automatically permuted and transposed spatial axes from standard `RAS` space to a decoder-optimized spatial format `(Z, X, Y)`.
* **Mechanism of Failure:** VoxTell's zero-shot inference API expected inputs in native `RAS` orientation. Feeding the permuted nnU-Net format resulted in the model segmenting a spatially scrambled 3D volume, rotating the clinical structures and yielding a residual Dice of only `0.0043`.
* **Resolution:** Replaced the automated nnU-Net reader with native Nibabel file loading to preserve standardized `RAS` spatial layout during network propagation.

### C. Clinical Text Prompt Tokenization Errors
* **Physical Diagnostic:** The competition dataset (`dataset.json`) represents findings in a dictionary format keyed by string indices (e.g., `"0"`, `"1"`). The initial pipeline looped directly over raw keys (`for f in findings`), which passed stringified characters like `'0'` or `'1'` to the zero-shot text encoder.
* **Mechanism of Failure:** The multi-modal text encoder could not map generic numbers to clinical features, causing the projection layers to collapse and output entirely empty predicted masks (filled with `0.0`).
* **Resolution:** Refactored the prompt extractor to sort keys numerically and map them to their actual string medical descriptions (e.g., "pleural effusion") before text tokenization.

---

## 3. Quantitative Results & Evaluation Metrics
After applying the coordinate corrections to allow the zero-shot pipeline to execute, we evaluated the model on the first 5 validation cases (representing 12 distinct clinical findings):

* **Evaluation Subset:** 5 validation scans, 12 findings
* **Average Dice Score:** **`0.0083`**
* **Hit Rate (Dice >= 0.1):** **`22.61%`**

These extremely poor metrics quantitatively confirmed that, despite resolving the coordinate/axis mismatches, the resampled and windowed preprocessing approach was biologically and computationally non-viable.

---

## 4. Empirical Post-Mortem & Scientific Conclusions
The failure of this initial iteration yielded two critical clinical and deep learning insights:

### 1. Loss of Sub-Voxel Anatomical Features (Resolution Degradation)
Resampling CT scans from their high-resolution native spacing (e.g., `0.6mm` isotropic) to `1.5mm` isotropic deforms and deletes fine-grained spatial features of micro-findings (e.g., small calcified nodules, sub-pleural ground-glass opacities). The spatial interpolation smeared out high-frequency anatomical boundaries, rendering them completely invisible to the convolutional encoder.

### 2. Normalization-Induced Out-of-Distribution (OOD) Shift
VoxTell was pre-trained using global, unclipped `ZScoreNormalization` calculated over the entire volumetric array (including air/background). By clipping Hounsfield Units to the lung window and calculating localized tissue Z-scores, we artificially altered the grayscale mean and standard deviation of the inputs. This created a severe OOD shift. The network's internal features were blinded by the shift in contrast distribution, causing the final sigmoid activation thresholds to skew completely and suppress true segmentations.

### Final Paradigm Shift:
The initial 1.5mm resampled preprocessing paradigm was **officially abandoned**. 

The failures and mathematical lessons learned during this attempt directly motivated the design of **[Experiment 001](exp_001_voxtell_v1.1_baseline_verification.md)** (Native Resolution & 4D Back-Reorientation Pipeline). By preserving native high-resolution spacing, retaining global unclipped normalization, and mathematically reversing the RAS coordinate flips at inference time, Experiment 001 successfully raised the Average Dice to a reproducible baseline of **`0.2139`**.
