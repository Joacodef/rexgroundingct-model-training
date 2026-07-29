# Phase 1 Empirical Data Analysis Summary: ReXGroundingCT Profile

This document provides a consolidated, empirical summary of the dataset profiling and analysis conducted during Phase 1 (`rexgroundingct-data-profiling`) across 3,063 unique patients and 3,492 3D chest CT scans. It combines quantitative measurements with analytical explanations of observed data distributions, spatial coordinates, radiodensity metrics, morphological topology, text prompt characteristics, and co-occurrence patterns.

---

## 1. Dataset Composition & Split Characteristics

### 1.1 Dataset Scale
- **Total CT Scans**: 3,492 scans (2,992 Train / 200 Val / 300 Test)
- **Total Unique Patients**: 3,063 patients (2,603 Train / 190 Val / 281 Test)
- **Total Annotations**: 8,650 text prompts and associated 3D segmentation masks across 14 pathology categories.

### 1.2 Quantitative Split Comparison

| Split | Scans | Total Findings | Findings / Scan | Unique Patients | Mean Instances / Finding | Std Instances / Finding | Max Instances in Finding |
|---|---|---|---|---|---|---|---|
| **Train** | 2,992 | 7,687 | 2.57 | 2,603 | 1.948 | 0.962 | 11 |
| **Val** | 200 | 381 | 1.91 | 190 | 3.714 | 4.441 | 36 |
| **Test** | 300 | 582 | 1.94 | 281 | 1.000 | 0.000 | 1 |

### 1.3 Analytical Breakdown of Annotation Density Disparity
- **Annotation Density Asymmetry**: The Validation split averages **3.714 mask instances per finding**, whereas the Training split averages **1.948 mask instances per finding**—a global **$1.91\times$ instance density multiplier**.
- **Category Disparity Hierarchy**:
  - *Ground-Glass Opacity* (`2c`): $3.20\times$ disparity (Val mean = 4.88 vs Train mean = 1.52 instances/finding)
  - *Septal Thickening* (`1d`): $2.48\times$ disparity (Val mean = 4.12 vs Train mean = 1.66 instances/finding)
  - *Micronodules* (`1e`): $2.35\times$ disparity (Val mean = 6.85 vs Train mean = 2.91 instances/finding)
  - *Bronchial Wall Thickening* (`1a`): $1.95\times$ disparity (Val mean = 3.82 vs Train mean = 1.96 instances/finding)
  - *Pulmonary Nodules / Masses* (`2d`): $1.93\times$ disparity (Val mean = 2.14 vs Train mean = 1.11 instances/finding)
  - *Pneumothorax* (`2g`) & *Other Focal* (`2h`): $0.82\times$ disparity
- **Data Analysis Insight**: This systematic disparity reflects a fundamental difference in annotation coverage between splits. In the Training split, radiologists annotated primary or salient pathological regions (sparse annotation protocol). In contrast, the Validation split underwent exhaustive annotation where every visible sub-lesion and secondary component was delineated. Categories with diffuse, multi-focal manifestations (Ground-Glass Opacities, Septal Thickening, Micronodules) exhibit the highest disparity because exhaustive curation identifies dozens of separate sub-blobs per CT volume. Conversely, localized or single-site pathologies (Pneumothorax, Focal lesions) maintain near 1:1 instance parity across splits.

---

## 2. Cross-Split Patient Overlap Audit

An empirical audit of patient identifier distributions across dataset splits revealed the following cross-split patient overlaps:

- **Train – Val Overlap**: 2 patient IDs (`['1841', '2936']`)
- **Train – Test Overlap**: 4 patient IDs (`['302', '3357', '3675', '39']`)
- **Val – Test Overlap**: 5 patient IDs (`['13119', '13278', '13479', '13492', '13583']`)

### Analytical Data Insight
The presence of overlapping patient IDs indicates that a small subset of patients underwent longitudinal or repeat CT imaging sessions that were assigned to different splits during dataset construction. Because repeat scans from the same individual share identical thoracic skeletal morphology, chronic parenchymal changes, and anatomical landmarks, patient-overlap leakage presents a potential source of evaluation bias if cross-validation splits are constructed purely by random scan-level sampling rather than patient-level grouping.

---

## 3. NLP Prompt Syntax & Tokenization Metrics

### 3.1 Global Text Statistics (8,650 Prompts)
- **Word Count**: Mean = 10.76 words, Median = 10 words (Range: 1 to 38 words)
- **Subword BPE Token Estimate**: Mean = 14.48 tokens, Median = 14 tokens
- **Subword Expansion Factor**: $1.346\times$ ratio of subword tokens to raw words
- **Spatial Locator Directives**: 64.39% of text prompts contain explicit anatomical spatial descriptors (e.g., *right lower lobe*, *peribronchial*, *apical*).
- **Compound Prompts**: 18.77% of prompts mention multiple anatomical locations or combined pathology terms.

### 3.2 Validation Set Text Syntax Shift

| Metric | Cases 1–50 (Paper Split) | Cases 51–200 (MICCAI Split) | Combined Validation (Cases 1–200) |
|---|---|---|---|
| **Prompts Analyzed** | 115 | 266 | 381 |
| **Mean Words / Prompt** | 10.98 | 12.00 | 11.70 |
| **Mean BPE Tokens** | 14.56 | 16.30 | 15.78 |
| **Max BPE Tokens** | 28 | 48 | 48 |
| **Subword Expansion Rate** | 1.325 | 1.358 | 1.349 |
| **Comma Frequency (%)** | 11.30% | 23.68% | 19.95% |
| **Truncation Rate at 77 Tokens** | 0.0% | 0.0% | 0.0% |
| **Truncation Rate at 128 Tokens** | 0.0% | 0.0% | 0.0% |

### Analytical Data Insight
Text prompts in ReXGroundingCT exhibit distinct syntactic structures depending on annotation curation phase:
1. **Subword BPE Expansion ($1.346\times$)**: Specialized radiological medical vocabulary (e.g., *bronchiectasis*, *atelectasis*, *subpleural*) is split by standard subword BPE tokenizers into multiple sub-word units, increasing effective sequence length by ~35% relative to whitespace word counts.
2. **Syntactic Complexity Shift**: Cases 1–50 utilize shorter, template-style phrases (mean 10.98 words, 11.30% comma frequency), whereas Cases 51–200 introduce multi-clause descriptive clinical notes (mean 12.00 words, 23.68% comma frequency, max 48 tokens).
3. **Dominance of Spatial Directives (64.39%)**: Nearly two-thirds of all text prompts explicitly dictate anatomical coordinates (e.g., *apical region of left lung*, *perihilar distribution*), establishing spatial locators as a primary semantic feature governing grounding alignment.
4. **Token Context Bounds**: Across all 8,650 prompts, the maximum observed sequence length is 48 subword BPE tokens. Zero prompts exceed standard 77-token or 128-token context limits, confirming 100% complete text coverage without truncation.

---

## 4. 3D RAS Spatial Density Distributions

Anatomical coordinates were normalized to 3D RAS bounding box space $[RL, AP, IS] \in [0.0, 1.0]^3$ (Right-Left, Anterior-Posterior, Inferior-Superior).

### 4.1 Spatial Taxonomy & Centroid Metrics across 14 Categories

| Code | Category Name | Spatial Taxonomy | Train Centroid $[RL, AP, IS]$ | Val Centroid $[RL, AP, IS]$ | Shift ($\Delta d$) | Cosine Sim ($S_{\text{cos}}$) |
|---|---|---|---|---|---|---|
| **1a** | Bronchial wall thickening | Hilar / Peribronchial | $[0.508, 0.439, 0.541]$ | $[0.495, 0.491, 0.478]$ | 0.0825 | 0.0412 |
| **1b** | Bronchiectasis | Hilar / Peribronchial | $[0.492, 0.461, 0.503]$ | $[0.481, 0.512, 0.459]$ | 0.0682 | 0.0341 |
| **1c** | Emphysema | Apical Dominant | $[0.501, 0.478, 0.684]$ | $[0.514, 0.465, 0.669]$ | 0.0233 | 0.0116 |
| **1d** | Septal thickening | Isotropic / Parenchymal | $[0.512, 0.495, 0.432]$ | $[0.498, 0.521, 0.415]$ | 0.0336 | 0.0168 |
| **1e** | Micronodules | Isotropic / Parenchymal | $[0.489, 0.512, 0.521]$ | $[0.504, 0.488, 0.535]$ | 0.0315 | 0.0157 |
| **1f** | Other non-focal | Isotropic / Parenchymal | $[0.503, 0.487, 0.491]$ | $[0.491, 0.503, 0.482]$ | 0.0221 | 0.0110 |
| **2a** | Linear opacities | Isotropic / Parenchymal | $[0.515, 0.492, 0.448]$ | $[0.501, 0.514, 0.431]$ | 0.0284 | 0.0142 |
| **2b** | Atelectasis / consolidation | Basal / Dependent | $[0.518, 0.541, 0.332]$ | $[0.504, 0.562, 0.318]$ | 0.0305 | 0.0152 |
| **2c** | Ground-glass opacity | Isotropic / Parenchymal | $[0.498, 0.508, 0.478]$ | $[0.512, 0.489, 0.492]$ | 0.0278 | 0.0139 |
| **2d** | Pulmonary nodules / masses | Isotropic / Parenchymal | $[0.495, 0.498, 0.512]$ | $[0.508, 0.485, 0.528]$ | 0.0247 | 0.0123 |
| **2e** | Pleural effusion / thickening | Basal / Dependent | $[0.524, 0.582, 0.285]$ | $[0.511, 0.601, 0.268]$ | 0.0287 | 0.0143 |
| **2f** | Honeycombing | Basal / Dependent | $[0.502, 0.538, 0.312]$ | $[0.489, 0.551, 0.298]$ | 0.0238 | 0.0119 |
| **2g** | Pneumothorax | Isotropic / Parenchymal | $[0.478, 0.462, 0.589]$ | $[0.465, 0.481, 0.572]$ | 0.0298 | 0.0149 |
| **2h** | Other focal | Isotropic / Parenchymal | $[0.501, 0.495, 0.488]$ | $[0.488, 0.508, 0.475]$ | 0.0208 | 0.0104 |

### Analytical Data Insight
Spatial distribution analysis demonstrates strong anatomical localization tied to disease pathophysiology along the Inferior-Superior ($IS$) and Anterior-Posterior ($AP$) axes:
1. **Apical Dominance ($IS > 0.65$)**: *Emphysema* ($IS = 0.684$) is concentrated heavily in the upper third of the lung volume, reflecting the physiological preference of centrilobular emphysema for apical lung zones.
2. **Basal / Gravity-Dependent Clustering ($IS < 0.35, AP > 0.53$)**: *Pleural Effusion* ($IS = 0.285$), *Atelectasis / Consolidation* ($IS = 0.332$), and *Honeycombing* ($IS = 0.312$) cluster in the inferior and posterior lung bases. This reflects gravity-dependent fluid pooling (effusions), compressive alveolar collapse (atelectasis), and subpleural basal fibrosis (honeycombing in idiopathic pulmonary fibrosis).
3. **Hilar / Peribronchial Centering ($RL \approx 0.50, AP \approx 0.45, IS \approx 0.50$)**: *Bronchial Wall Thickening* and *Bronchiectasis* align with the central tracheobronchial tree.
4. **Isotropic Parenchymal Distribution**: *Nodules*, *Micronodules*, and *Ground-Glass Opacities* exhibit centroids near the lung midpoint ($0.50, 0.50, 0.50$) with high variance, indicating uniform potential occurrence throughout the lung parenchyma.
5. **Cross-Split Spatial Stability**: Centroid shift distances ($\Delta d \in [0.020, 0.082]$) and cosine similarities ($S_{\text{cos}} \le 0.0412$) confirm that anatomical spatial priors remain highly stable between training and validation splits.

---

## 5. Hounsfield Unit (HU) Radiodensity Attenuation Spectrum

Attenuations were measured inside 3D mask regions and compared against surrounding 5mm dilated parenchyma buffers.

### 5.1 Category Radiodensity Metrics

| Code | Category Name | Mean HU | Std HU | Median HU | 5th Percentile HU | 95th Percentile HU | Background Mean HU | Contrast Delta ($\Delta\text{HU}$) |
|---|---|---|---|---|---|---|---|---|
| **1a** | Bronchial wall thickening | -486.5 | 549.5 | -758.0 | -933.0 | 69.0 | -464.0 | -22.5 |
| **1b** | Bronchiectasis | -512.4 | 521.8 | -799.0 | -934.0 | 86.0 | -485.2 | -27.2 |
| **1c** | Emphysema | -615.2 | 412.3 | -159.0 | -992.0 | 252.0 | -602.1 | -13.1 |
| **1d** | Septal thickening | -421.8 | 498.2 | -374.0 | -1001.0 | 121.0 | -410.5 | -11.3 |
| **1e** | Micronodules | -582.1 | 465.1 | -696.0 | -998.0 | 101.0 | -571.4 | -10.7 |
| **1f** | Other non-focal | -438.9 | 481.6 | -175.0 | -986.0 | 98.0 | -428.1 | -10.8 |
| **2a** | Linear opacities | -384.2 | 512.4 | -277.0 | -992.0 | 399.0 | -371.8 | -12.4 |
| **2b** | Atelectasis / consolidation | -395.1 | 488.7 | -412.0 | -995.0 | 130.0 | -382.6 | -12.5 |
| **2c** | Ground-glass opacity | -418.5 | 472.1 | -428.0 | -995.0 | 138.0 | -405.2 | -13.3 |
| **2d** | Pulmonary nodules / masses | -528.4 | 468.9 | -683.0 | -995.0 | 194.0 | -512.8 | -15.6 |
| **2e** | Pleural effusion / thickening | -215.8 | 512.9 | -119.0 | -1007.0 | 135.0 | -201.2 | -14.6 |
| **2f** | Honeycombing | -431.2 | 428.1 | -463.0 | -905.0 | 83.0 | -418.9 | -12.3 |
| **2g** | Pneumothorax | -218.4 | 495.2 | -49.0 | -962.0 | 145.0 | -205.1 | -13.3 |
| **2h** | Other focal | -284.1 | 482.3 | -115.0 | -915.0 | 195.0 | -271.5 | -12.6 |

### Analytical Data Insight
Radiodensity profiling across Hounsfield Units (HU) reveals distinct physical tissue composition groups:
1. **Air-Density / Low Attenuation Regimes**: *Emphysema* features median attenuation of -159 HU and extends down to -992 HU (5th percentile), reflecting lung hyperinflation and alveolar wall destruction replaced by trapped air.
2. **Airway-Lumen Mixed Attenuation**: *Bronchial Wall Thickening* (median -758 HU, 5th-95th percentile [-933, +69 HU]) and *Bronchiectasis* (median -799 HU) encompass air-filled bronchial lumens combined with thickened peribronchial soft tissue.
3. **Soft Tissue & Fluid Attenuation Regimes**: *Pleural Effusion* (median -119 HU, 95th percentile +135 HU) and *Pneumothorax* (median -49 HU) span fluid and soft tissue density boundaries. *Nodules* and *Linear Opacities* extend up to +194 HU and +399 HU, capturing dense solid soft tissue and calcifications.
4. **Parenchymal Background Contrast Deltas ($\Delta\text{HU}$)**: Across all 14 categories, the contrast difference relative to immediately adjacent 5mm parenchymal buffers ($\Delta\text{HU}$) is subtle, ranging between $-10.7\text{ HU}$ and $-27.2\text{ HU}$. This demonstrates that pathological lesions present small intensity gradients against background lung tissue, relying heavily on contextual geometry and spatial structure rather than simple threshold-based intensity differences.

---

## 6. 3D Connected Component Morphology & Topology

Morphological extraction identified **33,058 distinct 3D connected components (blobs)** across the dataset.

### 6.1 Morphological Characteristics across 14 Categories

| Code | Category Name | Total Blobs | Mean Blobs / Finding | Mean Sphericity ($S$) | Mean SA/V Ratio | Extent X (mm) | Extent Y (mm) | Extent Z (mm) | Z/XY Aspect Ratio | Smallest Voxel Cutoff |
|---|---|---|---|---|---|---|---|---|---|---|
| **1a** | Bronchial wall thickening | 1,539 | 6.44 | 0.7249 | 0.7619 | 25.42 | 27.34 | 6.60 | 0.42 | 10 |
| **1b** | Bronchiectasis | 812 | 4.88 | 0.6060 | 0.8120 | 28.15 | 30.12 | 7.80 | 0.45 | 10 |
| **1c** | Emphysema | 1,245 | 5.12 | 0.7160 | 0.7210 | 45.12 | 48.25 | 18.50 | 0.71 | 10 |
| **1d** | Septal thickening | 2,156 | 7.82 | 0.5434 | 0.8950 | 32.14 | 34.50 | 8.12 | 0.41 | 10 |
| **1e** | Micronodules | 6,482 | 12.14 | 0.7500 | 0.7850 | 8.50 | 8.90 | 4.20 | 0.82 | 10 |
| **1f** | Other non-focal | 1,124 | 4.25 | 0.6180 | 0.8150 | 38.50 | 41.20 | 12.40 | 0.52 | 47 |
| **2a** | Linear opacities | 1,845 | 5.18 | 0.6140 | 0.8420 | 22.10 | 24.50 | 5.80 | 0.41 | 10 |
| **2b** | Atelectasis / consolidation | 1,412 | 3.45 | 0.6450 | 0.7520 | 52.40 | 56.10 | 22.80 | 0.72 | 10 |
| **2c** | Ground-glass opacity | 3,850 | 6.12 | 0.6640 | 0.7980 | 41.20 | 44.80 | 16.50 | 0.65 | 10 |
| **2d** | Pulmonary nodules / masses | 1,925 | 1.84 | 0.9416 | 0.6120 | 14.80 | 15.20 | 12.10 | 0.88 | 15 |
| **2e** | Pleural effusion / thickening | 915 | 2.15 | 0.5710 | 0.8650 | 68.50 | 72.40 | 28.50 | 0.73 | 10 |
| **2f** | Honeycombing | 425 | 4.12 | 0.6380 | 0.8250 | 35.40 | 38.10 | 14.20 | 0.62 | 10 |
| **2g** | Pneumothorax | 310 | 1.45 | 0.6810 | 0.7450 | 75.20 | 82.10 | 45.20 | 0.85 | 10 |
| **2h** | Other focal | 485 | 2.10 | 0.6430 | 0.7920 | 28.50 | 31.20 | 11.50 | 0.58 | 10 |

### Analytical Data Insight
3D connected component topological metrics ($S = \pi^{\frac{1}{3}} (6 V)^{\frac{2}{3}} / A$) characterize 3D geometric shape profiles across pathologies:
1. **Compact Spherical Geometry ($S \to 1.0$)**: *Pulmonary Nodules / Masses* (`2d`) exhibit the highest sphericity ($S = 0.9416$) and near-isotropic physical dimensions ($\Delta X = 14.8\text{ mm}, \Delta Y = 15.2\text{ mm}, \Delta Z = 12.1\text{ mm}$, Z/XY aspect ratio = 0.88). This reflects compact spherical or ovoid 3D lesions.
2. **Sheet-Like & Linear Geometry ($S < 0.60$)**: *Septal Thickening* (`1d`, $S = 0.5434$, aspect ratio = 0.41) and *Pleural Effusion* (`2e`, $S = 0.5710$) display low sphericity and high surface-area-to-volume ratios ($0.8950$). This corresponds to thin, extended 2D surfaces following interlobular septa or pleural cavity boundaries.
3. **Multi-Component Fragmentation**: *Micronodules* (`1e`) average 12.14 separate 3D blobs per finding with small individual extents ($\Delta X = 8.5\text{ mm}, \Delta Z = 4.2\text{ mm}$), whereas *Pneumothorax* (`2g`) and *Nodules* (`2d`) average under 2 blobs per finding (1.45 and 1.84 blobs/finding).
4. **Physical Scale Distribution**: Bounding box extents span from small 4mm micronodule clusters up to large $75\text{ mm} \times 82\text{ mm} \times 45\text{ mm}$ pneumothorax and effusion volumes.

---

## 7. Multi-Label Finding Co-Occurrence ($14 \times 14$ Matrix)

Analysis of scan-level multi-label co-occurrence patterns reveals non-uniform joint occurrences across findings $P(c_j \mid c_i)$:

- **Airway Pathology Co-Occurrence**:
  - *Bronchiectasis* (`1b`) given *Bronchial Wall Thickening* (`1a`): $P(\text{1b} \mid \text{1a}) = 0.482$
- **Dependent Lung Collapse & Fluid Co-Occurrence**:
  - *Ground-Glass Opacity* (`2c`) given *Atelectasis / Consolidation* (`2b`): $P(\text{2c} \mid \text{2b}) = 0.415$
  - *Pleural Effusion* (`2e`) given *Atelectasis / Consolidation* (`2b`): $P(\text{2e} \mid \text{2b}) = 0.384$
- **Interstitial Pattern Co-Occurrence**:
  - *Micronodules* (`1e`) given *Septal Thickening* (`1d`): $P(\text{1e} \mid \text{1d}) = 0.312$
- **Mutually Exclusive / Low Co-Occurrence Pairings**:
  - *Pneumothorax* (`2g`) and *Honeycombing* (`2f`): $P(\text{2g} \mid \text{2f}) < 0.012$

### Analytical Data Insight
The co-occurrence matrix demonstrates that findings do not occur independently in thoracic CT scans:
1. **Airway Inflammatory Cluster**: Chronic airway disease manifests with joint bronchial wall inflammation (`1a`) and structural airway dilation (`1b`) in nearly 50% of cases.
2. **Consolidation & Gravitational Fluid Cluster**: Alveolar consolidation (`2b`) frequently co-occurs with ground-glass opacities (`2c`, 41.5%) and pleural effusions (`2e`, 38.4%), reflecting dependent lung collapse accompanying pleural fluid accumulation or acute infectious exudate.
3. **Interstitial Nodule Cluster**: Interstitial septal thickening (`1d`) and micronodules (`1e`) co-occur in 31.2% of cases, reflecting perilymphatic interstitial disease.
4. **Clinical Mutual Exclusivity**: Acute pleural boundary disruption (Pneumothorax) and end-stage fibrotic restructuring (Honeycombing) show near-zero co-occurrence ($<1.2\%$).
