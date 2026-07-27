# Experiment Log 001: [Phase 3] VoxTell v1.1 Mean Teacher Semi-Supervised Fine-Tuning

* **Date:** May 27, 2026  

* **Project Milestone:** Milestone 1 (June 1, 2026) - Baseline and Methodological Validation  
* **Status:** Interrupted (Epoch 5) & Evaluated (Trivial Collapse Diagnosed)

---

## 1. Objective & Hypothesis
The pre-trained **VoxTell v1.1** model suffers from **Instance Suppression Bias** because it was trained with fully-supervised losses (Dice + BCE) on the partially-annotated ReXGroundingCT dataset. The model treats unannotated real instances of pathologies/anatomies as background, suppressing exhaustive segmentations.

**Hypothesis:** Applying a Positive-Unlabeled (PU) semi-supervised learning framework using a **Mean Teacher** structure will resolve this bias.
* **Strong Supervision:** DiceCE loss is restricted strictly to dilated Regions of Interest (ROIs) around annotated positives (**SPOCO** masked loss).
* **Weak Supervision:** 3D spatial consistency is maintained in unannotated regions via a Multi-Planar Reconstruction (**MPR**) consistency loss (MSE of Axial, Coronal, and Sagittal 2D max projections) between Student and EMA Teacher predictions.

---

## 2. Experimental Setup
* **Model Baseline:** VoxTell v1.1 (initialized from fold 0 weights).
* **Student-Teacher Architecture:** Student model optimized via backpropagation; Teacher model updated via Exponential Moving Average (EMA) with decay rate $\alpha = 0.999$.
* **Optimization:** AdamW optimizer ($LR = 10^{-4}$, weight decay $10^{-5}$), Cosine Annealing learning rate scheduler over 50 epochs.
* **Consistency Loss Weight ($w_{con}$):** Scaled via a sigmoid warmup from $0.0$ up to a maximum weight of $10.0$ over the first 5 epochs.
* **Hardware Allocation:** Isolated to GPU 2 (`NVIDIA RTX 6000 Ada`, 48 GB VRAM; indexed as `CUDA_VISIBLE_DEVICES=1` in PyTorch).
* **Low-Precision Optimization:** FP16/BF16 mixed precision using `torch.amp.autocast`. Memory footprint: **~7.69 GB VRAM** (stable, no leaks).

---

## 3. Training Progress, History & Duration Logs (Sunday, May 24, 2026)
The training was initiated for a 50-epoch run, logging offline to Weights & Biases (run ID: `run-20260524_100447-z7qyum89`). Below are the exact training loss convergences and precise computational durations logged per epoch on the isolated `NVIDIA RTX 6000 Ada` GPU:

### Epoch-Level Quantitative Metrics & Computational Durations:

| Epoch | Avg Train Loss | Supervised Loss (`Sup`) | Consistency Loss (`Con`) | Avg Val Loss | Training Duration (2,992 Steps) | Validation Duration (50 Cases) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 1.263249 | 1.259806 | 0.051442 | `nan` | **1h 57m 10s** (2.35s/it) | **11m 15s** (13.51s/it) |
| **2** | 1.205845 | 1.201259 | 0.009669 | `nan` | **1h 56m 26s** (2.33s/it) | **19m 39s** (23.60s/it) |
| **3** | 1.174802 | 1.168275 | 0.002427 | `nan` | **1h 59m 55s** (2.41s/it) | **19m 02s** (22.84s/it) |
| **4** | 1.173363 | 1.166457 | 0.000945 | `nan` | **2h 02m 50s** (2.46s/it) | **10m 51s** (13.03s/it) |
| **5** | *Interrupted* | *Interrupted* | *Interrupted* | *N/A* | *Stopped at step 2022* | *N/A* |

* **Total Active Training Time (4 Epochs):** **~9.5 hours** (8 hours of pure backpropagation, 1 hour of sliding-window validation evaluation).

### Key Observations during Training:
1. **Rapid Consistency Decay:** The consistency loss `Con` dropped extremely fast, from `0.0514` in Epoch 1 to `0.0009` in Epoch 4.
2. **Supervised Loss Convergence:** The supervised ROI loss converged steadily from `1.259` to `1.166`.
3. **Numerical Instability in Validation:** The validation loss consistently yielded `nan`. This indicates a potential mathematical anomaly during full-resolution sliding window validation inference under the SPOCO masked loss framework (e.g., division-by-zero when the dilated mask sum is zero, or projection edge cases).

---

## 4. Quantitative Evaluation (Epoch 4 Checkpoint)
To evaluate the intermediate performance of the model for thesis review, we extracted a fast validation subset containing the **first 10 CT scans** from the validation split. 

* **Checkpoint Evaluated:** `models/checkpoint_mean_teacher_latest.pth` (Epoch 4 / index 3)
* **Target Model:** Student Model (default)
* **Inference Parameters:** Native resolution, sliding window tile step size = `0.5`
* **GPU Used:** GPU 2 (RTX 6000 Ada)

### Quantitative Metrics:
* **Subset Size:** 10 cases (representing 28 pathology findings)
* **Average Dice (Primary Metric):** **`0.0000`**
* **Hit Rate (Dice >= 0.1):** **`0.0000`**

### Diagnostics & Forensic Analysis (Exploding Gradients & NaN Collapse):
1. **Model Output Inspection:** A dedicated diagnostic script was executed to extract raw, unbinarized sigmoid probability maps. This revealed that the model was producing **`nan`** (Not a Number) values during the forward pass.
2. **Checkpoint Weight Analysis:** To pinpoint the origin of the `nan` values, we ran checks on the network weight tensors inside three separate checkpoints to measure the maximum absolute weight magnitude:

| Checkpoint File | Description / Run Type | Max Absolute Weight | NaNs in Weights | Target Outputs |
| :--- | :--- | :---: | :---: | :---: |
| `voxtell_v1.1/fold_0/checkpoint_final.pth` | Baseline Pre-trained Model | **`1.1804`** | False | Valid Segmentation Maps |
| `checkpoint_mean_teacher_final.pth` | 5-Epoch Smoke Test (2 cases) | **`1.1807`** | False | Valid Segmentation Maps |
| `checkpoint_mean_teacher_latest.pth` | 4-Epoch Full Scale (2,992 cases) | **`2617.9135`** | False | **NaN (Numerical Overflow)** |

3. **Textbook Gradient Explosion:**
   * In the small-scale smoke test (2 cases), the model weights remained completely stable (`1.1807`).
   * However, under full-scale training on 2,992 chest CT scans, the complex backprojection geometry of the MPR consistency loss over unannotated space caused massive gradient spikes.
   * Because the training script has **no Gradient Clipping** implemented, these spikes multiplied the weights uncontrollably up to magnitudes of `2617.9135`.
   * When these massive weights are used in a standard float32 forward pass, they trigger numerical overflow (values tending to infinity), which immediately yields `nan` during layer normalization or sigmoid activation.

---

## 5. Diagnostic Action Plan & Proposed Remedies
1. **Implement Gradient Clipping [CRITICAL]:**
   Add `torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)` right before `optimizer.step()` in the training loop. This is the primary standard defense to stabilize training and completely prevent gradient explosion.
2. **Mitigate Background Consistency Penalty:**
   Reduce the maximum consistency loss weight ($w_{con}$) from `10.0` to `0.5` or `0.1` and extend the sigmoid warmup period to 15 epochs.
3. **Ponderate ROI Positives / Focal Loss:**
   Implement class-weighted BCE or Focal Loss inside the dilated ROIs to strongly penalize false negatives and encourage positive activations, helping the model pull away from zero-prediction tendencies.
