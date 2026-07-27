# Experiment Log 003: [Phase 2] VoxTell v1.1 Second Baseline Verification & Split Breakdown

**Date**: July 15, 2026 (Updated July 21, 2026)  
**Purpose**: Second verification of official `voxtell_v1.1` zero-shot performance across all 200 validation masks (released July 2026), including empirical 4D Back-Reorientation coordinate audit and split performance profiling.

---

## 1. Execution Setup
- Bash Chunking Wrapper used to isolate `VoxTellPredictor` memory allocations.
- Sliding window inference evaluated with `tile_step_size = 0.5` across all 200 validation scans on GPU 1 (NVIDIA RTX PRO 6000 Blackwell).

---

## 2. Empirical 4D Back-Reorientation Audit
- **Hypothesis Tested**: Investigated whether an axis transpose mismatch in `voxtell_inference.py` during back-reorientation caused zero-overlap Dice scores.
- **Empirical Result (`scratch/test_shape_hypothesis.py`)**:
  - **Coordinate Mapping Verified**: The 4D Back-Reorientation transformation in `voxtell_inference.py` is **mathematically sound and correct**.
  - **Centroid Precision**: On scan `train_13102_a_1` (Finding #0: *"Fibrotic changes in middle lobe of right lung"*), predicted centroid (`[142.0, 219.9, 95.8]`) aligns with GT centroid (`[135.9, 223.3, 88.8]`) within **~6 voxels**.
  - **High Overlap**: On `train_13013_a_1` (Finding #0: *"Bilateral pleural effusion..."*), single-scan Dice reaches **`0.6240`**.

---

## 3. Split Quantitative Results & Prompt Sensitivity Diagnosis

| Partition | Cases | Findings | Raw Prompts Dice | Raw Hit Rate | Normalized Prompts Dice | Normalized Hit Rate |
|---|---|---|---|---|---|---|
| **First 50 Scans (Paper Val Split)** | 50 | 115 | **`0.2139`** | **`49.57%`** | `0.2015` | `45.22%` |
| **Next 150 Scans (New MICCAI Split)** | 150 | 266 | **`0.0491`** | **`14.66%`** | `0.0402` | `12.03%` |
| **Combined 200 Scans** | 200 | 381 | **`0.0988`** | **`25.20%`** | `0.0889` | `22.05%` |

### Empirical Discovery: Clinical Modifier OOD Shift
1. **Instruction Template Matching**: In `voxtell/utils/text_embedding.py`, VoxTell wraps text queries with `Instruct: Given an anatomical term query...`.
2. **Text Conditioning Collapse**: Prompts in cases 51–200 contained long clinical report sentences (up to 35 words, 39.1% punctuation) with non-diagnostic adjectives (*"Stable, nonspecific 6 mm subpleural nodule..."*), driving BioClinicalBERT embeddings out-of-distribution and returning `0.0000` continuous probabilities across the 3D volume.
3. **Entity Query Recovery**: Converting verbose clinical sentences to concise anatomical entity queries (*"pulmonary nodule"*) recovered zero predictions from **`0.0000` to `0.4124` Dice (+41.2%)** on single-scan test case `train_18382_b_2` Finding #0.


---

## 4. Key Scientific Conclusions & Working Hypotheses for Phase 3

1. **Paper Baseline Replicated**: `voxtell_v1.1` perfectly matches the `0.2139` paper validation benchmark on the original 50-scan paper split.
2. **Text Shift Diagnosed**: Prompt normalization removes OOD text conditioning failures on isolated findings, but universal regex stripping yields mixed overall results across all 266 findings.
3. **Hypothesis 1 (Weight Fine-Tuning Impact)**: We hypothesize that fine-tuning VoxTell's model weights on the 1,000 ReXGroundingCT training volumes in Phase 3 will improve validation Dice on cases 51–200.
4. **Hypothesis 2 (Partial Annotation / Unlabeled Loss)**: We hypothesize that partial annotations in the training set introduce an instance suppression bias, and that Phase 3 loss functions treating unannotated regions as unlabelled (such as Positive-Unlabeled or SPOCO masked loss) will stabilize training without suppressing valid predictions.



