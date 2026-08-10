# Official Rules: ReXGroundingCT Challenge @ MICCAI 2026

> Structured transcript of the official rules published at `https://rexrank.ai/ReXGroundingCT/challenge.html`, the dataset overview at `https://rexrank.ai/ReXGroundingCT/index.html`, submission guidelines at `https://rexrank.ai/explore/submission_guideline_ct.html`, arXiv preprint `arXiv:2507.22030`, and official organizer communications (M. Baharoon). Last updated: August 9, 2026. In case of any discrepancy with official pages, the official page takes precedence.

---

## 1. About the Challenge & Official Tracks

The ReXGrounding Challenge is an official MICCAI 2026 challenge designed to evaluate models on the localization of radiological findings described in unrestricted natural language, producing precise 3D segmentation masks in volumetric thoracic CTs.

Unlike previous challenges focused on lesion or organ segmentation by predefined category labels, this benchmark requires models to interpret diverse clinical language — including anatomical descriptors, spatial relationships, and morphological attributes — and accurately anchor them in 3D volumetric space. The dataset includes focal and diffuse abnormalities, covers a wide range of radiological patterns, and reflects real-world variability in how radiologists report.

The challenge is built on CT-RATE (a large-scale dataset of non-contrast thoracic CTs paired with free-text radiology reports), extended with expert-verified pixel-level 3D segmentations corresponding to individual findings from the reports.

Host: public leaderboard at `https://rexrank.ai/ReXGroundingCT/challenge.html`.

### The Two Official Competition Tracks

The challenge is structured into **two distinct tracks**, each with a separate set of 3 winners:

1. **Track 1 — Main Track: Free-Text Grounding**
   - **Requirement**: Methods must consume the exact free-text finding description as provided, conditioning segmentation directly on that text (e.g. as a text embedding).
   - **Constraint**: The raw free-text prompt MUST NOT be rewritten or parsed into a structured representation before reaching the model.
   - **Goal**: Evaluates flexible, end-to-end models that natively ground free-text radiology findings.

2. **Track 2 — Overall Track: Free-Text & Structured**
   - **Scope**: General track admitting both native free-text methods and methods that transform free-text findings into structured representations (e.g. parsing location, morphology, or size via an LLM or NLP text parser) prior to conditioning.
   - **Constraint**: Prompts cannot simply be reduced to a fixed class label (e.g., just "nodule"); models must take in spatial/descriptive context to differentiate separate findings in distinct locations.
   - **Automatic Dual Eligibility**: Every eligible Main Track submission is automatically evaluated for the Overall Track. A strong free-text method can win in both tracks simultaneously.

---

## 2. Task & Category Ontology

**Single task: free-text finding grounding.**

Model input:
- A 3D thoracic CT volume.
- A finding description in natural language extracted from a radiology report.

Expected output:
- A 3D segmentation mask corresponding to the description.

### The 14 Official Finding Categories

**Typically non-focal (6):**
1. Bronchial wall thickening
2. Bronchiectasis
3. Emphysema
4. Septal thickening
5. Micronodules
6. Other non-focal

**Typically focal (8):**
1. Linear opacities
2. Atelectasis / consolidation
3. Ground-glass opacity
4. Pulmonary nodules / masses
5. Pleural effusion / thickening
6. Honeycombing
7. Pneumothorax
8. Other focal

---

## 3. Dataset Pipeline, Statistics & Chain-of-Thought Resource

### Dataset Provenance & Construction
1. **Source Reports**: Reports originated from CT-RATE (originally written in Turkish, machine-translated to English).
2. **Finding Extraction & Standardization**: GPT-4 was utilized to extract and standardize findings, descriptors, and metadata from the translated reports.
3. **Hierarchical Ontology Categorization**: GPT-4o-mini categorized each finding into a hierarchical ontology of lung and pleural abnormalities.
4. **3D Segmentation & Verification**:
   - Training Set: Quality-assured by board-certified radiologists.
   - Validation & Test Sets: Fully and exhaustively annotated by board-certified radiologists.
5. **Overall Statistics**: Contains 16,301 annotated entities across 8,028 text-to-3D-segmentation pairs from 3,142 non-contrast CT scans (~79% focal abnormalities, ~21% non-focal abnormalities).

### Complementary Chain-of-Thought (CoT) Dataset
The authors provide a complementary **Chain-of-Thought (CoT) reasoning dataset** created using GPT-4o paired with 3D localization coordinates derived from organ segmentation models. This dataset provides step-by-step hierarchical anatomical reasoning for localizing findings within CT volumes, providing a structured textual resource for prompt engineering or reasoning-guided fine-tuning.

### Dataset Splits

| Split | Scans | Annotation Type | Description |
|---|---|---|---|
| Training | 2,992 CT scans | Partial | Up to 3 instances per finding annotated |
| Validation | 200 CT scans | Exhaustive | All visible instances annotated by board-certified radiologists |
| Test | 300 CT scans | Exhaustive | Held-out evaluation split (50% public / 50% private) |

### Note on Split Differences
* **Paper Split (`arXiv:2507.22030`)**: 2,992 train / 50 val / 100 test (used in initial publication benchmark).
* **MICCAI Challenge Split**: 2,992 train / 200 val / 300 test (expanded split for official leaderboard).

---

## 4. Timeline & Deadlines

| Date | Milestone |
|---|---|
| Pre-registration | Pre-registration open. Training data publicly available. |
| June 2026 | Challenge launch: registration opens, validation set (200) released. |
| June — September 2026 | Development phase: evaluate on val set, submit multiple test runs. |
| **September 14, 2026 (11:59 PM ET)** | **Official Submission Deadline** for final test set rankings. |
| Late September 2026 | Results announced and challenge session at MICCAI 2026. |

---

## 5. Evaluation Metrics & Leaderboard Table

### Primary Ranking Metric
**Average Dice Coefficient (DSC)**: Computed per finding and per case across all target instances.

### Evaluation Metrics Suite

| Metric | Displayed on Public Leaderboard | Threshold / Matching Criterion |
|---|---|---|
| Dice | Yes (Primary Rank Metric) | Average DSC per finding per case |
| Hit Rate | Yes | Proportion of findings where global $Dice \ge 0.1$ |
| Instance F1 | Yes | Harmonic mean of Instance Precision and Instance Recall ($Dice \ge 0.2$) |
| Instance Precision | No (Detailed Breakdown) | TP / (TP + FP), where TP requires predicted instance component $Dice \ge 0.2$ |
| Instance Recall | No (Detailed Breakdown) | TP / (TP + FN), with same $Dice \ge 0.2$ criterion |
| Distance Precision | No (Detailed Breakdown) | TP / (TP + FP), where TP requires ASSD (non-focal) or centroid distance (focal) $\le 2 \times \max(\text{voxel spacing})$ |
| Distance Recall | No (Detailed Breakdown) | TP / (TP + FN), with same distance criterion |
| Distance F1 | No (Detailed Breakdown) | Harmonic mean of Distance Precision and Distance Recall |

### Public Leaderboard & Fast Evaluator
* **Public Standings**: Live standings are computed on a fixed **public 50% subset** of the 300 test scans. The remaining 50% private test set determines final rankings.
* **Evaluator Processing Speed**: The backend evaluator is optimized (~20x faster), publishing scores to the public leaderboard within **~2 hours** of zip submission on `rexrank.ai`.

---

## 6. Permitted vs. Disqualified Techniques

### Permitted in Both Tracks
1. **Auxiliary Category Information**: Using finding categories (either passed directly or inferred via prompt classifier) as auxiliary input signals is permitted in both Track 1 and Track 2. (In Track 1, the raw free-text prompt must still be directly consumed).
2. **Anatomical Segmentations**: Incorporating organ/anatomical segmentations (e.g. lung lobes, pleura, airways) as extra input channels or spatial bounding constraints is fully permitted.
3. **Anatomical Post-Processing**: Restricting predicted masks to anatomical sub-regions specified in the text prompt (e.g., spatial locator masking) is fully permitted.

### Strictly Out of Scope & Disqualified
* **Fixed-Category Class Segmentation**: Models that output fixed class masks and select target categories solely by label (ignoring text descriptors and failing to separate two distinct instances in different locations) are **strictly disqualified**.

---

## 7. Technical Submission Process

### Prediction Format & Packaging
1. Run inference on the ReXGroundingCT test set (300 scans).
2. Save predictions as individual NIfTI files (`.nii.gz`) with **names matching the original CT files**.
3. Each prediction file must have 4D shape `(F, H, W, D)`, where:
   - `F` = number of findings for that scan (matching the exact order in `dataset.json`).
   - `H, W, D` = spatial dimensions matching ground truth.
4. Compress all 300 prediction NIfTI files into a **single `.zip` file** (do NOT compress a wrapper folder; files must sit at root).

### Web Submission Portal Workflow
Submissions are managed exclusively via the web portal at `https://rexrank.ai/ReXGroundingCT/challenge.html`:
1. **Account Setup & Registration**: Log in on `rexrank.ai`, register team name, and list team members (Full Name + Affiliation, max 8 members).
2. **Google Drive Upload**: Upload prediction `.zip` file to Google Drive and set permission to *"Anyone with the link can view"*.
3. **Submit**: Enter Model Name, target split (`test`), and shareable Google Drive URL.

---

## 8. Awards and Co-authorship

- Top 3 teams in each track receive official MICCAI awards and invited presentations.
- Up to 8 members per top-3 team qualify for co-authorship on the post-challenge MICCAI overview paper.
- No embargo period: teams retain full rights to publish independently.

---

## 9. Critical Operational Implications

1. **Submission Deadline**: **September 14, 2026 at 11:59 PM ET**.
2. **Track Alignment**: Main Track requires direct text embedding input of the raw prompt. Categorical and anatomical locators can be added as auxiliary inputs.
3. **Flat Zip Structure**: `.nii.gz` files must be at the root of the `.zip` archive.
4. **50% Public / 50% Private Leaderboard**: Standings evaluate 50% public test scans live; private 50% determines final MICCAI ranking.