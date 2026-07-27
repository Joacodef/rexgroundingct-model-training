# VoxTell Baseline Architecture & Technical Reference

> [!IMPORTANT]
> This document details key specifications, codebase architecture, pre-training history, and operational guidelines for the pre-trained **VoxTell** baseline model (`voxtell_v1.1`). It combines findings from the VoxTell paper (*cs.CV, Nov 2025*) with a deep source code audit of the installed package (`.venv-voxtell/lib/python3.13/site-packages/voxtell/`).

---

## 🏁 1. VoxTell Pre-Training Distribution & Instance Bias

### Pre-Training Provenance
* **Scale:** Pre-trained on a multi-modality 3D medical corpus of **62,000+ 3D scans** (CT, MRI, PET) spanning **1,000+ anatomical and pathological concepts**.
* **Text-Image Grounding Datasets:** Incorporates CT-RATE and multi-organ 3D segmentation benchmarks for spatially grounded anatomical queries.
* **Supervision Loss:** Trained with fully-supervised standard Dice + Binary Cross-Entropy (BCE) losses computed over volume masks.

### Instance Suppression Bias
Because pre-training relied on partially-annotated datasets with standard supervised losses, the model exhibits an **instance suppression bias**: unannotated positive instances in partially-labeled training scans are treated as background (0), suppressing candidate logit activations. This motivates Positive-Unlabeled (PU), SPOCO, or Mean Teacher consistency adaptations for Phase 3 fine-tuning.

---

## 🔬 2. Architecture & Code Specifications

Source code location: `.venv-voxtell/lib/python3.13/site-packages/voxtell/`

```
voxtell/
├── model/
│   ├── voxtell_model.py     # Main VoxTellModel & VoxTellDecoder classes
│   └── transformer.py       # DETR-style TransformerDecoder & TransformerDecoderLayer
├── inference/
│   ├── predictor.py         # VoxTellPredictor (sliding window, text embedding, binarization)
│   └── predict_from_raw_data.py # CLI entrypoint and NIfTI I/O writer
└── utils/
    └── text_embedding.py    # Instruction wrapping template and last_token_pool
```

### Key Subsystems & Hyperparameters

1. **Text Embedding & Instruction Pipeline**:
   * **Text Encoder Backbone**: `Qwen/Qwen3-Embedding-4B` (`.venv-voxtell/lib/python3.13/site-packages/voxtell/inference/predictor.py#L68-L70`) (embedding dimension $D_{\text{text}} = 2560$, max sequence length 8192).
   * **Instruction Wrapper**: Free-text prompts are formatted using `wrap_with_instruction` (`.venv-voxtell/lib/python3.13/site-packages/voxtell/utils/text_embedding.py#L14-L20`):  
     `"Instruct: Given an anatomical term query, retrieve the precise anatomical entity and location it represents\nQuery: {text}"`
   * **Token Pooling**: `last_token_pool` (`.venv-voxtell/lib/python3.13/site-packages/voxtell/utils/text_embedding.py#L3-L11`) extracts the hidden state of the last non-padded token.
   * **Linear Projection**: `project_text_embed`: `Linear(2560 → 2048) → GELU → Linear(2048 → 2048)`.

2. **3D Vision Encoder**:
   * **Backbone**: `ResidualEncoder` (from `dynamic_network_architectures`) extracting multi-scale 3D feature skip connections ($S_0 \dots S_5$).
   * **Input Channels**: 1 (single-channel CT volume).

3. **Bottleneck Projection & 3D Positional Encoding**:
   * Encoder features at layer 4 (channels: 320, spatial: $12 \times 12 \times 12$) are projected via `project_bottleneck_embed` to `query_dim = 2048`.
   * **Positional Encoding**: Fused with 3D sinusoidal positional encodings (`PositionalEncoding3D(query_dim)`).

4. **Prompt Transformer Decoder**:
   * `TransformerDecoder` (`.venv-voxtell/lib/python3.13/site-packages/voxtell/model/transformer.py#L17-L104`): 6 DETR-style transformer layers, 8 attention heads, LayerNorm pre-normalization.
   * Queries ($Q$) = Text prompt embeddings. Memory ($K, V$) = Bottleneck 3D image features.
   * Outputs refined text-conditioned mask query embeddings ($N_{\text{prompts}}, B, 2048$).

5. **MaskFormer Multi-Stage Decoder & Einsum Fusion**:
   * `VoxTellDecoder` (`.venv-voxtell/lib/python3.13/site-packages/voxtell/model/voxtell_model.py#L280-L472`) upsamples features across 5 resolution stages.
   * Fuses text mask embeddings at each stage via multi-head tensor contraction (`torch.einsum('b c h w d, b n c -> b n h w d')`).
   * Final segmentation logits are produced by voxel-wise dot-product einsum between decoder spatial features and prompt query embeddings.

---

## 🔍 3. Inference & Spatial Pipeline Protocols

1. **Spatial Reader & Alignment**:
   * Uses `nnunetv2.imageio.nibabel_reader_writer.NibabelIOWithReorient` to enforce standard RAS space.
   * Predictions are made in RAS space; final masks MUST undergo 4D Back-Reorientation to match original CT NIfTI metadata.

2. **Intensity Preprocessing**:
   * Background cropped to non-zero regions via `crop_to_nonzero`.
   * Intensity normalized using masked `ZScoreNormalization`.

3. **Sliding-Window Inference**:
   * **Patch Size**: `(192, 192, 192)`.
   * **Overlap**: Default `tile_step_size = 0.5` (50% tile overlap).
   * **Weighting**: 3D Gaussian kernel (`compute_gaussian`, $\sigma\_scale = 1/8$, `value_scaling_factor = 10`).
   * **Async Prefetching**: Multi-threaded producer-consumer queue (`Queue(maxsize=2)`) overlaps CPU patch loading with GPU forward passes.

4. **Binarization & Thresholding**:
   * Predicts continuous logits, converted to probabilities via `torch.sigmoid()`.
   * Standard zero-shot binarization threshold: `p > 0.5`.

---

## 🛠️ 4. Fine-Tuning & Training Infrastructure

Implementation: `scripts/voxtell/training/train_mean_teacher.py`

1. **Text Embedding Pre-Caching**:
   * Qwen text embeddings are pre-computed offline via `precompute_text_cache` and saved to `data/text_cache/*.pt` (shape: `(F, 2560)`).
   * Eliminates the heavy 4B parameter text model from GPU memory during network fine-tuning.

2. **Fast SSD Volume Caching**:
   * Reorientated, cropped, and Z-score normalized CT volumes are cached as PyTorch tensors in fast SSD temporary storage (`TMP_PREP_DIR`) to bypass slow CPU Gzip decompression bounds.

3. **Patch Augmentation & Sampling**:
   * MONAI `RandCropByPosNegLabeld` with `pos = 1.0` ensures positive-centered $192 \times 192 \times 192$ patch extraction.
   * 3D spatial random flips (`RandFlipd`) applied along all 3 axes during training.
