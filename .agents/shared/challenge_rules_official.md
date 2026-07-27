# Official Rules: ReXGroundingCT Challenge @ MICCAI 2026

> Structured transcript of the official rules published at `https://rexrank.ai/ReXGroundingCT/challenge.html`, the dataset overview at `https://rexrank.ai/ReXGroundingCT/index.html`, submission guidelines at `https://rexrank.ai/explore/submission_guideline_ct.html`, and arXiv preprint `arXiv:2507.22030`. Last verified: July 23, 2026. In case of any discrepancy with official pages, the official page takes precedence.

---

## 1. About the Challenge

The ReXGrounding Challenge is an official MICCAI 2026 challenge designed to evaluate models on the localization of radiological findings described in unrestricted natural language, producing precise 3D segmentation masks in volumetric thoracic CTs.

Unlike previous challenges focused on lesion or organ segmentation by predefined category labels, this benchmark requires models to interpret diverse clinical language — including anatomical descriptors, spatial relationships, and morphological attributes — and accurately anchor them in 3D volumetric space. The dataset includes focal and diffuse abnormalities, covers a wide range of radiological patterns, and reflects real-world variability in how radiologists report.

The challenge is built on CT-RATE (a large-scale dataset of non-contrast thoracic CTs paired with free-text radiology reports), extended with expert-verified pixel-level 3D segmentations corresponding to individual findings from the reports.

Host: public leaderboard at `https://rexrank.ai/ReXGroundingCT/challenge.html`.

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

## 4. Timeline

| Date | Milestone |
|---|---|
| Pre-registration | Pre-registration open. Training data publicly available. |
| June 2026 | Challenge launch: registration opens, validation set (200) released. |
| June — September 2026 | Development phase: evaluate on val set, submit multiple runs. |
| September 2026 | Submission deadline. Final evaluation on held-out test set. |
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

### Public Leaderboard Display & Category Breakdown
* The main leaderboard table displays **Dice** (bolded primary metric), **Hit Rate**, and **Instance F1**.
* Clicking any submission row on `rexrank.ai` reveals an interactive per-category metric breakdown across all 14 finding classes.

### Important Note on Hit Rate Threshold
The official challenge Hit Rate uses **Dice $\ge 0.1$** as the matching threshold. The VoxTell paper reports $HIT_{5\%}$ (threshold 0.05), which is different. Local evaluation loops in `scripts/evaluate.py` enforce the official 0.1 threshold.

---

## 6. Participation Rules

1. **Pre-registration** via the official registration panel on `rexrank.ai`.
2. **Training data availability**: Method development and local benchmarking can proceed using available training and validation splits.
3. **External data permitted**: Any public or private training data, including pre-trained foundation models and external datasets, may be used. All external data sources must be declared in the final submission report.
4. **Fully automatic predictions**: All predictions on the test set must be generated automatically. Prohibited actions include:
   - Manual intervention or human mask editing.
   - Post-hoc manual threshold tuning per case.
   - Non-reproducible case-specific heuristics.
5. **Multiple submissions during development phase**: Teams may submit multiple runs to evaluate on the test split. Only the single best submission per team is officially ranked on the leaderboard.

---

## 7. Awards and Publication

- The top 3 teams receive:
  - Official MICCAI award certificates.
  - Invited oral / spotlight presentation at the MICCAI 2026 challenge session.
  - Recognition on the public leaderboard.
  - Recognition and co-authorship in the post-challenge overview paper.
- **Co-authorship in the challenge publication**: Members of top-3 teams qualify, up to 8 authors per team.
- **No embargo period**: Participating teams retain full rights to publish their methods and results independently without time restrictions.

---

## 8. Technical Submission Process

### Prediction Format & Packaging
1. Run inference on the ReXGroundingCT test set (300 scans).
2. Save predictions as individual NIfTI files (`.nii.gz`) with **names matching the original CT files**.
3. Each prediction file must have 4D shape `(F, H, W, D)`, where:
   - `F` = number of findings for that scan (matching the exact order in `dataset.json`).
   - `H, W, D` = spatial dimensions matching ground truth.
4. Compress all 300 prediction NIfTI files into a **single `.zip` file** (do not compress a wrapper folder).

### Web Submission Portal Workflow
Submissions are managed exclusively via the web portal at `https://rexrank.ai/ReXGroundingCT/challenge.html`:

1. **Account Setup**: Log in or create an account on `rexrank.ai` (requires email verification; password min 6 characters). Note: Check spam/junk folders for verification emails.
2. **Team Registration**: Register team name, contact email, and member list (each member must provide **Full Name** and **Affiliation**, max 8 members).
3. **Google Drive File Upload**: Upload the single prediction `.zip` file to Google Drive and set sharing permissions to *"Anyone with the link can view"*.
4. **Form Submission**: Select target split (`test`), enter **Model Name** (required), and provide the shareable Google Drive URL.

---

## 9. Public vs. Private Leaderboard Rules

* **Public Standings (50% Test Set)**: Live standings on the public leaderboard are calculated on a randomly selected **public 50% subset** of the 300 held-out test scans.
* **Private Final Ranking (100% Test Set)**: The remaining **50% subset is withheld**. Final challenge placement is evaluated on the complete 100% test set and will be revealed at MICCAI 2026.
* **Best Run Ranking**: Each team is ranked by their single best submission score.

---

## 10. Organizers

- Mohammed Baharoon — Harvard Medical School, USA (primary contact)
- Pranav Rajpurkar — Harvard Medical School, USA
- Luyang Luo — Harvard Medical School, USA
- Xiaoman Zhang — Harvard Medical School, USA
- Mahmoud Hussain Alabbad — King Fahad Hospital, Saudi Arabia
- Sungeun Kim — Harvard Medical School, USA

General challenge contact: `MohammedSalimAB@outlook.com`

---

## 11. Official Resources

- Leaderboard & Submission Portal: `https://rexrank.ai/ReXGroundingCT/challenge.html`
- Dataset Overview: `https://rexrank.ai/ReXGroundingCT/index.html`
- Submission Guidelines: `https://rexrank.ai/explore/submission_guideline_ct.html`
- Dataset on HuggingFace: `https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT`
- Official Evaluation Script: `https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT/blob/main/rexrank_eval.py`
- ReXGroundingCT Paper: `https://arxiv.org/abs/2507.22030`

---

## 12. Official Citations

**ReXGroundingCT:**
```bibtex
@article{baharoon2025rexgroundingct,
  title={ReXGroundingCT: A 3D Chest CT Dataset for Segmentation of Findings from Free-Text Reports},
  author={Baharoon, Mohammed and Luo, Luyang and Moritz, Michael and Kumar, Abhinav and Kim, Sung Eun and Zhang, Xiaoman and Zhu, Miao and Alabbad, Mahmoud Hussain and Alhazmi, Maha Sbayel and Mistry, Neel P and others},
  journal={arXiv preprint arXiv:2507.22030},
  year={2025}
}
```

**CT-RATE:**
```bibtex
@article{hamamci2026generalist,
  title={Generalist foundation models from a multimodal dataset for 3D computed tomography},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Wang, Chenyu and Almas, Furkan and Simsek, Ayse Gulnihan and Esirgun, Sevval Nil and Dogan, Irem and Durugol, Omer Faruk and Hou, Benjamin and Shit, Suprosanna and others},
  journal={Nature Biomedical Engineering},
  pages={1--19},
  year={2026},
  publisher={Nature Publishing Group UK London}
}
```

---

## 13. Critical Operational Implications

1. **50% Public / 50% Private Leaderboard Split**: Leaderboard scores reflect only half the test set. Overfitting to the public 50% risks performance degradation on the final private evaluation. Solid local cross-validation is essential.
2. **Chain-of-Thought (CoT) Prompting Leverage**: The availability of GPT-4o-generated CoT reasoning traces anchored to anatomical coordinates provides an additional textual grounding mechanism to experiment with during baseline fine-tuning.
3. **Output Format (F, H, W, D)**: Predictions must stack individual 3D finding segmentations into a single 4D volume per CT scan, maintaining exact correspondence with the GT finding sequence. Verify local outputs using `scripts/evaluate.py`.
4. **Google Drive Submission Portal**: Predictions are submitted via a shared Google Drive link in the web portal. Form fields require a designated Model Name and verified team members (Name + Affiliation).
5. **Co-authorship Limit**: Top-3 placement qualifies up to 8 team members for co-authorship on the official MICCAI challenge paper.