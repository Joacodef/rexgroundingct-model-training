# VoxTell Baseline Architecture, Training Protocol & Technical Reference

> [!IMPORTANT]
> This document details the complete ground-truth specifications, pre-training corpus, fine-tuning protocols, loss configurations, and operational guidelines for the **VoxTell** foundation model (`voxtell_v1.1`). It consolidates all official disclosures from the VoxTell publication (*"VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation"*, Rokuss et al., DKFZ / MIC-DKFZ, arXiv:2511.11450, Nov 2025) and deep codebase audits of the implementation.

---

## 🏁 1. Pre-Training Corpus & ReXGroundingCT Fine-Tuning Dataset

### A. Universal Pre-Training Corpus
* **Scale**: Pre-trained across **62,000+ volumetric 3D scans** spanning CT, MRI, and PET modalities across **1,000+ anatomical and pathological concepts**.
* **Vocabulary Harmonization**: Built a unified clinical vocabulary comprising **1,087 concepts and 9,682 rewritten labels** validated via LLM-assisted conflict detection and human expert verification.

### B. Instance-Focused Composite Dataset (Appendix B.3 & Table 6)
The authors explicitly identified that semantic pre-training alone cannot solve fine-grained, localized spatial queries. To fine-tune VoxTell for the **ReXGroundingCT** benchmark, they curated an **instance-focused composite dataset** comprising:
1. **Official ReXGroundingCT Training Split**: **2,992 chest CT scans** from the CT-RATE corpus with free-text report findings grounded to 3D voxel annotations.
2. **TotalSegmentator-Derived Pseudo-Instance Augmentation**: Converted public semantic lesion datasets (e.g. LIDC, Decathlon, StructSeg, TriALS) into localized instance annotations by extracting anatomical anchors (lung lobes, Couinaud liver segments, left/right kidneys) using TotalSegmentator and synthesizing spatial prompts (e.g., *"spiculated tumor in the left lower lobe"*, *"cluster of HCC lesions in Couinaud segment 5"*).
3. **TCIA Collections with Structured DICOM Metadata**: Brain and head–neck CT/MR datasets (RADCURE, BrainGammaKnife) converted into location prompts via slice positioning and series metadata.

---

## 🔬 2. Architecture & Code Specifications

* **Vision Backbone**: 3D `ResidualEncoder` (ResEncL from `dynamic_network_architectures` / nnU-Net v2) with 6 hierarchical resolution stages producing feature channels `[32, 64, 128, 256, 320, 320]`.
* **Bottleneck & Positional Encoding**: Bottleneck features at layer 4 ($12 \times 12 \times 12$) projected to `query_dim = 2048` and fused with 3D sinusoidal positional embeddings (`PositionalEncoding3D`).
* **Prompt Transformer Decoder**: 6 DETR-style transformer decoder layers (8 attention heads, LayerNorm pre-normalization, hidden dim 2048) performing cross-attention between text query embeddings ($Q$) and 3D vision bottleneck features ($K, V$).
* **MaskFormer Multi-Stage Decoder & Einsum Fusion**: `VoxTellDecoder` upsamples spatial features across 5 resolution stages, fusing text query embeddings at every stage via tensor contraction (`torch.einsum('b c h w d, b n c -> b n h w d')`).
* **Text Encoder**:
  - *Architecture & Pretrained Model*: Frozen `Qwen/Qwen3-Embedding-4B` ($D_{\text{text}} = 2560$). Both the publication paper and the official open-source `v1.1` release (`mrokuss/VoxTell`) use this 2560-dimensional embedding model with `project_text_embed: (2048, 2560)`.
  - *Instruction Wrapping Template*:  
    `"Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents\nQuery: {text}"`
  - *Token Pooling*: Hidden state of the last non-padded token extracted via `last_token_pool`.

---

## ⚙️ 3. Official Training Protocols & Hyperparameters (Appendix A.1 & A.3)

* **Framework**: Built on the **nnU-Net v2** framework (`nnUNetTrainer`-derived 3D pipeline).
* **Patch Size & Input Format**: **$192 \times 192 \times 192$ voxel patches**, depth-first C-contiguous ordering `(Z, Y, X)`.
* **Optimizer & Schedule**: **Stochastic Gradient Descent (SGD)** with initial learning rate $\text{lr} = 1 \times 10^{-4}$ decayed via polynomial schedule across 2,000 epochs (250 iterations/epoch).
* **Loss Function**: Combined **Binary Cross-Entropy (BCE) + Soft Dice Loss**.
* **Deep Supervision**: Applied across 5 decoder scales with nnU-Net default scale weights:  
  $$\lambda_s = [1, 1/2, 1/4, 1/8, 1/16]$$
* **Prompt Sampling per Volume Step**:
  - Each volume is queried with **3 prompts per step**:
    - **2 Positive Prompts**: Structures actually present in the sampled volume patch.
    - **1 Negative Prompt**: A structure absent from the volume (critical for training the model to output empty masks when targets are not present).
* **Foreground Oversampling**: **85% probability** of sampling patches centered on foreground annotated lesions (15% random background).
* **Dynamic Text Augmentation**: 75% probability of sampling an LLM-generated synonym/paraphrase and 25% probability of using the default label string.
* **Data Augmentations**: Standard nnU-Net 3D spatial and intensity augmentations.
  > [!CAUTION]
  > **Crucial Data Augmentation Constraint**: **Left-Right Mirroring is strictly DISABLED** during training to prevent destroying anatomical laterality (e.g. differentiating left vs. right lung pathology).

---

## 📊 4. Official ReXGroundingCT Benchmark Scores (Section 6, Figure 4)

* **Fine-Tuned VoxTell on ReXGroundingCT Validation Split**:
  - **Average Dice**: **`28.2`**
  - **HIT5%** ($\text{Dice} \ge 0.05$): **`67.8%`**
  - *Comparison*: State-of-the-art baseline SAT achieved `13.1` Dice and `49.8%` HIT5% on the identical data mix.
* *Metric Clarification*: The VoxTell publication reported **HIT5%** ($\text{Dice} \ge 0.05$), whereas the official MICCAI ReXGroundingCT Challenge evaluates **HIT10%** ($\text{Dice} \ge 0.10$).

---

## 🛠️ 5. Official Inference Protocols & Specifications

* **Sliding-Window Inference**: 3D sliding-window patch-based inference using $192 \times 192 \times 192$ patches.
* **Gaussian Weighting**: Spatial Gaussian weighting mask with $\sigma\_scale = 1/8$ to blend overlapping patch boundaries smoothly without stitching seams.
* **Tile Overlap**: Default evaluation uses `tile_step_size = 0.5` (50% spatial overlap along $Z, Y, X$).
* **Continuous Probabilities & Thresholding**: Model outputs continuous sigmoid probabilities; standard binarization threshold is $p > 0.5$.
* **Text Embedding Conditioning**: Text queries are wrapped with the official instruction template before Qwen embedding pooling:
  `"Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents\nQuery: {text}"`
* **Coordinate Orientation**: The official VoxTell predictor assumes input volumes formatted with depth-first ordering $(Z, Y, X)$. Output predictions must be mapped back to native scan affine orientation.

