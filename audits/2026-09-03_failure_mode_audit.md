# ReXGroundingCT Failure-Mode Audit

**Audit date:** September 3, 2026
**Commit audited:** `14990ee` (*Phase 4 SPOCO: gate inference seeds on confidence and the candidate mask*)
**Host:** `ih-condor`
**Scope:** 11,353 lines across `scripts/` and `tests/`
**Method:** Reading and reasoning, cross-checked against the installed `voxtell`, `nnunetv2`, `acvl_utils` and `monai` sources, plus small standalone probes. No test suite was executed at any point, per the repository constraint. No job was submitted *during the audit*; the diagnostic and verification jobs listed under Disposition below were submitted afterwards, during remediation, all via `sbatch`.
**Runtime evidence:** Job **96800**, Phase 4 Exp 001, epoch 8/50 at time of audit.
**Published artifact:** <https://claude.ai/code/artifact/f73d2bdf-643d-4e40-b258-7a515a23e731>
**Revisions:**
- *September 3, 2026* — corrections to `LK-1` (patient-count parsing, plus reconciliation with the Phase 1 overlap audit), `LK-2` (8 test / 6 val scans affected, not 5), `LOSS-3` (bf16 exponent label) and `OPT-1` (epoch rate) after external review. No finding was withdrawn or downgraded.
- *September 4, 2026* — added `BC-3` (critical), found during remediation and the most consequential defect in the set; retracted two frequency figures this audit had asserted about the `Num foregrounds 0` anomaly (see Disposition); refreshed the status markers below, which described job 96800 as still running.

---

## Disposition (last updated September 4, 2026)

**All 14 code fixes are landed. Phase 4 Exp 001 restarted as job 96971 on 2026-09-04.** Job 96800
was cancelled at epoch 8/50 and discarded rather than resumed, because `LOSS-1`, `LOSS-3` and
`LOSS-4` change the objective and a mid-run switch would have made the run incoherent; its
checkpoints and logs are preserved under
`logs/phase_4_voxtell_spoco/exp_001_voxtell_spoco/prefix_run_96800/`. Tracked in
`.agents/STATUS.md` → "2026-09-03 Failure-Mode Audit — Remediation Status".

Supporting jobs, all via `sbatch`: **96834** (CPU diagnostic — 400/400 scans intact through the
load path, which localised `BC-3` to inside MONAI), **96833** and **96970** (single-batch
`--dry_run` verification, pre- and post-`BC-3`), **96971** (the restart).

**Still live under the old training signal:** Phase 3 Exp 002 on the sibling server. `LOSS-2`,
`LOSS-5` and `BC-3` all change what it trains on — `BC-3` reaches all four trainers, since
`ReXDataset` is shared. Whoever restarts it should know the signal changed.

| Finding | Disposition |
|---|---|
| `OPT-1`, `OPT-2`, `OPT-3`, `OPT-4` | Fixed — `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py` |
| `LOSS-1`, `LOSS-3`, `LOSS-6`, `MEM-1` | Fixed — `scripts/phase_4_voxtell_spoco/common/losses.py` |
| `LOSS-4`, `MEM-2` | Fixed — `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py` |
| `LOSS-2`, `LOSS-5` | Fixed — both Phase 3 mean-teacher trainers, backward-compatible fallback |
| `BC-1`, `AUG-1` | Fixed — `dataset.py` fixed-N padding; `worker_init_fn` on all four trainers |
| `BC-3` | Fixed — `dataset.py` single-channel union `label_key` (found 2026-09-04, not in the original audit) |
| `LK-1` | No action — verified clear |
| `LK-2` | Documented in `.agents/STATUS.md` split-mismatch note |
| `LK-3` | Open — protocol note for whoever runs the threshold sweep |
| `MEM-3` | Deferred — only bites when `--mpr_num_rotations` is raised on `peteroa` |

Two things are landed but **not yet measured**, and should be checked on epoch 1 of the restart:
`MEM-1`'s effect on the 96.6 min/epoch baseline, and whether train `Obj` and val `Obj` now sit in
the same range after the `LOSS-1` split.

**A 16th finding, `BC-3`, was found on 2026-09-04** while diagnosing an anomaly this audit noticed
but misread. It is the most consequential defect in the set: a third of the training split was
receiving no positive supervision at all. The audit's own error is worth recording — I read the
6,875 `Num foregrounds 0` stderr lines as an item rate when Python deduplicates warnings per
process (354 distinct messages × ~48 repeats), and separately floated a meaningless "91.2% of
scans" bound from matching volume voxel-counts. Neither number was real; the finding underneath
them was.

---

## Verdict board

| # | Failure mode | Verdict | Findings |
|---|---|---|---|
| 1 | Unintended tensor broadcasting | No silent broadcast in the loss *math* — but a channel-axis semantic mismatch with MONAI cost 35% of the training signal (`BC-3`, found later). | 1 critical, 1 high |
| 2 | Data leakage across splits | Training splits clean at scan **and** patient level. Two evaluation-side risks. | 2 medium |
| 3 | Computational graph memory | No unbounded leak. Three bounded costs, one dominating step time. | 1 medium, 2 low |
| 4 | Custom loss discrepancies | Two findings change what is actually being optimised versus what the docstrings claim. | 2 critical, 2 high, 1 medium, 1 low |
| 5 | Optimisation & weight decay | One resume-corruption bug that job 96800 would have hit at its epoch-44 resume. | 1 critical, 2 medium, 1 low |
| — | Outside the five modes | Augmentation RNG shared across DataLoader workers. | 1 medium |

**Status.** Every finding below is fixed except `LK-3` (open, protocol) and `MEM-3` (deferred) —
see Disposition above. Severity markers read *"was live"* where the finding was actively affecting
a run at audit time; `LOSS-2` is the one still affecting a running job (Phase 3 Exp 002 on the
sibling server, which predates the fix).

---

## 1. Unintended tensor broadcasting

Every deliberate broadcast in the loss path is correct, and every `einsum` contraction is shape-checked rather than broadcast. The one real shape defect crashes loudly rather than silently corrupting gradients — but it makes a documented CLI flag unusable.

### BC-1 — `--batch_size > 1` fails collation on roughly half of all random pairs
**Severity:** High · latent

`ReXDataset` emits `N = min(F, 2) + 1` prompts per scan, so a scan with a single finding yields `N = 2` while everything else yields `N = 3`. `seg`, `text_embeddings` and `is_absent_finding` all carry that variable leading dimension, so `default_collate` raises `stack expects each tensor to be equal size` the moment a batch mixes the two.

1,056 of 2,992 training scans (35.3%) have exactly one finding, so a random pair collides with probability ≈ 46%. Every trainer exposes `--batch_size` and defaults it to 1, which is why this has never fired.

- **Where:** `scripts/phase_3_voxtell_finetuning/common/dataset.py:286–335`; exposed by all four trainers via `--batch_size`.
- **Fix:** Pad prompts to a fixed `N` (repeating a sampled negative when `F < num_positive_prompts`), or attach a `collate_fn` that pads and emits a validity mask. Padding to a fixed `N` composes with the existing `is_absent_finding` routing at no extra cost.

### BC-3 — MONAI silently discards channel 0 of the label, so 35% of scans train on no foreground
**Severity:** Critical · found 2026-09-04 while diagnosing `Num foregrounds 0` · **not in the original audit**

`RandCropByPosNegLabeld` selects crop centres via `map_binary_to_indices`, whose third line is:

```python
if label.shape[0] > 1:
    label = label[1:]  # for One-Hot format data, remove the background channel
```

MONAI assumes a multi-channel label is one-hot with background in channel 0. `ReXDataset` passes
`seg` of shape `(F, Z, Y, X)` where **every channel is a different finding**, so channel 0 — the
first sampled positive — was being dropped from the foreground pool on every item. Measured
directly against the installed MONAI:

| Label as `ReXDataset` built it | True foreground voxels | Voxels MONAI sampled from |
|---|---:|---:|
| `[pos_0, neg]` — a 1-finding scan | 1 | **0** |
| `[pos_0, neg, neg]` — 1-finding, post-`BC-1` | 1 | **0** |
| `[pos_0, pos_1, neg]` — 2-finding scan | 2 | 1 |
| `[union]` — single channel | 1 | 1 |

Consequences: for the **1,056 of 2,992 train scans (35.3%) with a single finding**, foreground was
always empty, `pos_ratio = 0.85` was silently ignored, and *every* crop was a pure-background crop
carrying no positive supervision. For multi-finding scans, finding 0 could never anchor a crop — a
systematic sampling bias. This is what produced the `Num foregrounds 0` lines throughout job
96800's stderr; the masks on disk, the RAS load, the transpose and the bbox crop were all verified
intact (400/400 scans, job 96834), so the loss happened entirely inside MONAI.

- **Where:** `scripts/phase_3_voxtell_finetuning/common/dataset.py` — both the train and val
  `Compose` pipelines, shared by all four trainers.
- **Fix:** build a single-channel `(1, Z, Y, X)` foreground union and pass it as `label_key`,
  leaving `seg` in `keys` to be cropped. MONAI only drops channel 0 when `shape[0] > 1`, so a
  1-channel label sidesteps the heuristic rather than working around it. `DeleteItemsd` drops the
  union after the crop.
- **Confirmed:** post-fix dry run on real data reports `L_obj: 0.7728` where the pre-fix run on the
  same code path reported `L_obj: 0.0000`, and the `Num foregrounds 0` warnings are gone.

### BC-2 — The deliberate broadcasts and contractions check out
**Severity:** Verified clear

- `is_positive` of shape `(B,F,1,1,1)` broadcasting over an `(B,F,Z,Y,X)` ROI mask in `torch.where` — intended and correct.
- `pos_weight=torch.tensor([w])` in `binary_cross_entropy_with_logits` — a shape-`(1,)` tensor broadcasts to a scalar weight, numerically identical to passing a float.
- The SPOCO decoder's logit head, `einsum("b c h w d, b n c -> b n h w d", x, mask_embeddings_rev[-1])`, contracts 32 against `project_to_decoder_channels[0]`, whose output width is `decoder_config["channels"] = 32` (stage 0 alone omits the `× num_heads` factor). This reproduces stock `VoxTellDecoder` exactly.
- `VoxTellModel` returns `(B, N, Z, Y, X)` for `deep_supervision=False` and a list of that shape when enabled, matching `targets` — so the BCE/Dice pair in Exp 001 is not silently broadcasting a channel axis.
- Mask channel counts on disk match `len(findings)` in `dataset.json` for all 110 scans sampled (50 val + 60 random train), so the `text_embeddings` ↔ `seg` index pairing in `__getitem__` is sound.

---

## 2. Data leakage across splits

Nothing from val or test reaches training. Both remaining risks sit on the evaluation side, where a number gets reported against data that helped produce it.

### LK-1 — Train-time hygiene holds at scan and patient level
**Severity:** Verified clear

Parsing `<partition>_<patient>_<study>_<series>` out of every entry in the deployed `dataset.json`:

| Split | Scans | Patients | ∩ train (scan) | ∩ train (patient) |
|---|---:|---:|---:|---:|
| train | 2,992 | 2,717 | — | — |
| val | 50 | 49 | 0 | 0 |
| test | 100 | 92 | 0 | 0 |

**On the patient count.** The `train` split mixes two CT-RATE source partitions — 2,578 `train_*` and 414 `valid_*` filenames — and 114 numeric ids occur under both prefixes. Counting patients as `<partition>_<number>` gives **2,717**; collapsing to the bare number gives **2,603**, the figure recorded in `.agents/shared/PHASE_1_DATA_ANALYSIS_SUMMARY.md`. CT-RATE numbers patients independently within each partition, so `train_1294` and `valid_1294` are different people and 2,717 is the conservative reading; the bare-number count merges 114 unrelated pairs. The zero-overlap result below is identical under both parsings, so nothing downstream turns on which is preferred.

**Reconciling with the Phase 1 overlap audit.** `PHASE_1_DATA_ANALYSIS_SUMMARY.md §2` reports a train–val overlap of 2 patient ids (`1841`, `2936`) and a train–test overlap of 4 (`302`, `3357`, `3675`, `39`). None of those reproduce here: all six ids appear **only in the train split** of the deployed file, and `train ∩ val` and `train ∩ test` are empty under bare-number parsing as well as under `<partition>_<number>`. Those overlaps belong to the expanded MICCAI release Phase 1 profiled (3,492 scans, 190 val / 281 test) — the split mismatch `.agents/STATUS.md` already warns about. The val–test overlap is the one Phase 1 finding that *does* carry over, because both releases share the same val/test patient pool for it (see `LK-2`).

Supporting checks: Phase 2A spatial PDFs accumulate strictly from `metadata["train"]`; `ZScoreNormalization(intensityproperties={})` computes mean and σ per image, so no global intensity statistic crosses splits; the Qwen text cache is keyed by scan id, and scan ids are disjoint across splits, so no cache entry is shared or overwritten between them.

### LK-2 — Five patients appear in both the val and test splits
**Severity:** Medium · upstream

`train_13119`, `train_13278`, `train_13479`, `train_13492` and `train_13583` each contribute scans to both partitions in the shipped arXiv-split `dataset.json`. Nothing in this repository created that overlap, and it cannot contaminate training. This is the one cross-split overlap from `PHASE_1_DATA_ANALYSIS_SUMMARY.md §2` that does reproduce on the deployed file.

Five patients, but more than five scans — they account for **8 of the 100 test scans** and **6 of the 50 val scans**:

| Patient | Val scans | Test scans |
|---|---|---|
| `train_13119` | `_c_1` | `_b_1` |
| `train_13278` | `_b_2` | `_a_2`, `_c_1` |
| `train_13479` | `_e_1`, `_j_1` | `_f_1`, `_g_1`, `_i_1` |
| `train_13492` | `_b_2` | `_a_2` |
| `train_13583` | `_d_2` | `_a_2` |

So anything tuned on val — checkpoint selection, binarization thresholds, post-processing volume floors — is fitted to anatomy that reappears in 8% of the test set, not 5%. The effect is still small but it is one-directional: it flatters the leaderboard estimate rather than depressing it.

- **Fix:** Record it alongside the split-count caveat already in `.agents/STATUS.md`, and quote val-tuned numbers with the overlap stated rather than trying to re-split a challenge-provided partition.

### LK-3 — Binarization thresholds are swept on val and then applied to val
**Severity:** Medium · latent

`scripts/phase_2b_voxtell_baseline/exp_002_voxtell_logit_diagnostics.py` defaults to `--split val` and sweeps *p<sub>c</sub>* over `[0.01 … 0.50]`, reporting per-category Dice at each cut. `scripts/analysis/postprocess_predictions.py --thresholds_json` then consumes exactly that per-category mapping. If the resulting predictions are scored on val, the reported Dice is a maximum over eight thresholds selected on the same 50 scans and 115 findings — a per-category argmax over a sample that small carries real optimism.

No `thresholds_json` artifact exists on disk yet, so this is a methodology trap rather than a number currently in the logs.

- **Fix:** Sweep on a held-out slice of `train` (annotation sparsity differs from val, so state that caveat), or report the val number at a single pre-registered threshold and keep the swept value strictly for the test submission.

---

## 3. Computational graph memory

No unbounded retention: loss accumulators go through `.item()`, the teacher runs under `no_grad`, anchors sample from a detached copy, and the Workstream B component cap bounds the soft-mask tape. What remains is bounded but expensive, and one item dominates step time.

### MEM-1 — The unannotated-anchor loop materialises every background coordinate, eight times over
**Severity:** Medium · throughput

`sample_unannotated_anchors` calls `torch.nonzero(unlabeled_mask)` *inside* the anchor loop. At 192³ that allocates an `(N,3)` int64 tensor of up to 170 MB per iteration, and the loop runs up to 8 times per prompt, per batch item. Each iteration also forces two device syncs (`unlabeled_mask.sum().item()` and the `randint(...).item()`) plus a full 7.1M-voxel `einsum` for the suppression step.

Observed epoch time is ≈97 minutes for 2,992 single-scan steps — about 1.9 s/step against a model whose forward+backward at this patch size should be well under that.

- **Where:** `scripts/phase_4_voxtell_spoco/common/losses.py:206–228`
- **Fix:** Compute the candidate coordinates once before the loop, then draw indices against a shrinking boolean over that fixed array instead of re-running `nonzero`; keep the suppression `einsum` but drop the two per-iteration `.item()` syncs by tracking the count on-device.

### MEM-2 — Last iteration's embedding volumes are still alive when the next forward allocates
**Severity:** Low

`s_embeds` and `t_embeds` are each `(1, 3, 32, 192, 192, 192)` bf16 ≈ 1.36 GB. Their names are not rebound until the next iteration's forward has already begun allocating, so roughly 2.7 GB of stale output carries across the loop boundary on top of peak activation memory. Bounded, not a leak — but this pipeline has an OOM history, and the fix is one line.

- **Where:** `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py:528–530, 550`
- **Fix:** `del s_embeds, t_embeds, loss` after `ddp_step` returns and the scalars have been read.

### MEM-3 — Each MPR rotation pins an fp32 sampling grid on the backward tape
**Severity:** Low · scales with a flag

`_rotate_volume` runs `F.grid_sample` on the gradient-carrying student background. The grid itself — `(B, Z, Y, X, 3)` fp32, ≈85 MB at 192³ — plus the fp32 input cast are saved for backward, so `--mpr_num_rotations 4` retains roughly 0.9 GB. Gao et al. report an optimum near 9 rotations, which would take this past 2.5 GB on top of the model.

The teacher branch is correctly detached before rotation, so only the student side pays. Worth knowing before the rotation count is raised on `peteroa`.

- **Where:** `scripts/phase_3_voxtell_finetuning/exp_003_mpr_loss.py:160–181, 257–266`
- **Fix:** Build the grid once per rotation at reduced resolution and upsample, or checkpoint the rotate-and-project block so the grid is recomputed in backward rather than stored.

---

## 4. Custom loss discrepancies

The densest category. Two findings change what the optimiser is actually pursuing relative to what the docstrings describe, and the run log for job 96800 shows the first of them plainly.

### LOSS-1 — `L_obj` averages soft Dice together with raw soft-mask mass
**Severity:** Critical · was live (job 96800) · **fixed**

Present findings append a soft-Dice value in `[0,1]` to `loss_obj_list`; absent findings append `s_neg_soft.mean()` — the mean of a Gaussian soft mask over the whole patch, which is near zero for anything but a degenerate embedding. Both land in one `torch.stack(loss_obj_list).mean()`. The relative weight between positive supervision and negative suppression is therefore set by the *count* of terms in each batch, which swings with how many positive prompts survived the random crop and how many connected components each contributed.

The run log makes the consequence concrete. Validation carries no negative prompts at all (`is_train=False` skips the negative branch entirely), so its `L_obj` is pure positive Dice:

```text
Epoch 008/050 | Train Loss: 0.4132 (Obj: 0.4091, Con: 0.5916, Push: 0.0241)
                w_con: 0.0029 | Val Loss: 0.9819

train Obj 0.409  ->  Dice-like terms diluted by near-zero negative terms
val   Obj ~0.98  ->  soft Dice ~0.02 on positives, i.e. essentially no grounding yet
```

The train `Obj` curve descending from 0.451 to 0.409 over eight epochs reads as progress, but it is not measuring the same quantity as the val figure sitting at ~0.98. Any read of "is SPOCO learning" from the train column is unsafe until these are separated.

- **Where:** `scripts/phase_4_voxtell_spoco/common/losses.py:454, 516, 523`
- **Fix:** Accumulate negatives into their own list, reduce it separately, and combine as `L_obj + w_neg · L_neg` with an explicit weight. Log the two independently so the train and val `Obj` columns become comparable.

### LOSS-2 — Phase 3 Exp 002 and Exp 003 still infer absence from an empty crop
**Severity:** Critical · **fixed in code**, but Phase 3 Exp 002 is still running on the sibling server under the old signal

`ReXDataset` already emits `is_absent_finding`, the additive flag added in Workstream B1 precisely so a present finding whose lesion fell outside the random crop is not mistaken for a confirmed-absent one. Neither Phase 3 mean-teacher trainer consumes it. Both re-derive absence geometrically:

```python
is_positive = (targets.sum(dim=(2, 3, 4), keepdim=True) > 0)
roi_mask    = torch.where(is_positive, roi_mask, torch.ones_like(roi_mask, dtype=torch.bool))
```

A positive prompt with no foreground in the crop therefore gets its ROI widened to the *entire volume* and is supervised as confirmed-absent everywhere, under `pos_weight = 10.0` BCE driving every voxel to background. That is the exact training-signal bug Phase 4 had already fixed. It is now fixed in Phase 3 too, but Exp 002 was launched before the fix and is still running under the old signal.

The floor on how often this fires is exact: `pos_ratio = 0.85` means 15% of items are drawn from the pure-background pool, where *every* positive prompt is empty by construction. The true rate is higher — `RandCropByPosNegLabeld` centres the crop on the union foreground, so a 192³ window over a 512×512×N scan routinely contains one finding out of several.

- **Where:** `scripts/phase_3_voxtell_finetuning/exp_002_pu_mean_teacher.py:305–307`; `scripts/phase_3_voxtell_finetuning/exp_003_mpr_loss.py:440–442`
- **Fix:** Read `batch['is_absent_finding']` and drive `is_positive` from it, falling back to the geometric test only when the key is absent. Skip prompts that are neither absent nor present-in-crop rather than widening their ROI. This mirrors the routing already in `compute_spoco_total_loss`.

### LOSS-3 — bf16 quantises the SPOCO distance exactly where the pull term lives
**Severity:** High · was live (job 96800) · **fixed**

Squared distance on the hypersphere is computed as `2 − 2·⟨e_i, e_a⟩`. Under `autocast(bfloat16)` the `einsum` producing that dot product returns bf16, whose spacing at 1.0 is 2⁻⁷ = 0.0078125 (bf16 carries 8 mantissa bits, 7 of them stored). The subtraction is a textbook cancellation: the whole inner core of the Gaussian collapses into one or two representable bins.

| True d² | bf16 d² | Soft mask (bf16) | Soft mask (exact) |
|---|---:|---:|---:|
| 0.0001 | 0.000000 | 1.0000 | 0.9997 |
| 0.0010 | 0.000000 | 1.0000 | 0.9972 |
| 0.0040 | 0.007813 | 0.9786 | 0.9890 |
| 0.0100 | 0.007813 | 0.9786 | 0.9727 |
| 0.0500 | 0.046875 | 0.8781 | 0.8706 |
| 0.2500 | 0.250000 | 0.5000 | 0.5000 |

With `delta_var = 0.5` and `pmaps_threshold = 0.5`, `two_sigma ≈ 0.361`. Every true d² below ≈0.004 rounds to exactly zero, so the soft mask is pinned at 1.0000 and its gradient with respect to the embedding is identically zero across the entire tight-cluster regime the `L_obj` pull is supposed to sharpen. The same cancellation affects the coverage-suppression test in `sample_unannotated_anchors`, where `dist_sq < delta_var²` is decided on that quantised grid.

- **Where:** `scripts/phase_4_voxtell_spoco/common/losses.py:95–99` (soft mask), `:222–227` (suppression), `:316–318` (push)
- **Fix:** Wrap the distance computation in `torch.autocast("cuda", enabled=False)` and cast the embeddings to fp32 first. The embeddings are already L2-normalised, so fp32 here costs one 32-channel cast, not a full-precision forward.

### LOSS-4 — Validation passes the student to itself as its own teacher
**Severity:** High · was live (job 96800) · **fixed**

`evaluate_val_loss` calls `compute_spoco_total_loss(student_embeds=s_embeds, teacher_embeds=s_embeds, …)`. The consistency term then compares a soft mask against a detached copy of itself, and

```text
L_con = 1 − Σ s²  /  Σ s
```

which is not zero and is not a consistency measurement — it is a penalty on how non-binary the soft masks are. It is also weighted at the full asymptotic `args.w_con = 0.1`, while training at epoch 8 is using the ramped value 0.0029. So the quantity deciding which checkpoint is "best" differs from the training objective in both its definition and its weighting.

Checkpoint selection is currently ≈ positive soft Dice + a sharpness penalty + a push term. That is defensible as a proxy, but it is not what the docstring says, and it is not what A5 will measure once a real Dice harness runs.

- **Where:** `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py:212–215, 596–598`
- **Fix:** Pass the actual EMA teacher into `evaluate_val_loss`, and pass `w_con_epoch` rather than `args.w_con`, so the val figure tracks the objective being optimised. Or set `w_con=0.0` for validation and select on `L_obj` alone — either is honest; the current form is neither.

### LOSS-5 — ROI-masked Dice is diluted by every prompt whose ROI is empty
**Severity:** Medium

`compute_roi_masked_loss` reduces with `dice.mean()` over all `B×F` channels. A channel whose ROI mask is empty contributes exactly `1 − (0 + 1e-6)/(0 + 1e-6) = 0` to the numerator while still counting in the denominator. The effective weight on the real supervised Dice therefore scales with the fraction of prompts that happened to catch foreground in this crop — the same batch-composition coupling as `LOSS-1`, in a different loss.

- **Where:** `scripts/phase_3_voxtell_finetuning/exp_002_pu_mean_teacher.py:120–124`; `scripts/phase_3_voxtell_finetuning/exp_003_mpr_loss.py:120–124`
- **Fix:** Average only over channels with a non-empty ROI, guarding the all-empty case with a grad-carrying zero the way `compute_spoco_total_loss` already does.

### LOSS-6 — The Dice denominator is rounded to bf16 before it is cast to fp32
**Severity:** Low

`compute_instance_dice_loss` writes `soft_mask.sum().float()`. Under autocast `soft_mask` is bf16, so the reduction returns a bf16 scalar and *then* widens — the cast is one operation too late. A background-dominated sum of order 10⁶ keeps only 8 mantissa bits, snapping to multiples of ≈8,192. The intersection term is safe by accident: `soft_mask * target_mask` promotes to fp32 first because the target is fp32.

- **Where:** `scripts/phase_4_voxtell_spoco/common/losses.py:251–253`
- **Fix:** `soft_mask.float().sum()`, not `soft_mask.sum().float()`.

---

## 5. Optimisation & weight decay

Phase 3 builds its optimiser from an ordered `model.parameters()` and is fine. Phase 4 builds discriminative learning-rate groups out of Python sets, and that decision has a specific, dated consequence.

### OPT-1 — Set-ordered parameter groups corrupt Adam state on every `--resume`
**Severity:** Critical · was live (job 96800) · **fixed**

The Phase 4 optimiser is constructed from `list(encoder_params)` and `list(transformer_params)`, where both are Python `set`s of tensors. Set iteration order over objects follows `id()`-derived hashes, which vary between processes. Five runs of a trivial 16-parameter model produced three distinct orderings:

```text
run 1  [13, 14, 15, 11, 0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 10, 12]
run 2  [12, 13, 14, 15, 0, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11]
run 3  [12, 13, 14, 15, 0, 1, 4, 3, 2, 5, 6, 7, 8, 9, 10, 11]
run 4  [13, 14, 15, 11, 0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 10, 12]
run 5  [13, 14, 15, 10, 0, 1, 3, 4, 2, 5, 6, 7, 8, 9, 11, 12]
```

`Optimizer.load_state_dict` maps saved state onto current parameters *positionally* within each group. On resume, every `exp_avg` and `exp_avg_sq` in the encoder and transformer groups is therefore re-attached to a different parameter than the one it was accumulated for. Where the shapes differ, `AdamW.step` raises. Where they match — and a U-Net repeats `(32,)`, `(64,)`, `(320,320,3,3,3)` shapes constantly — it silently applies the wrong momentum and second moment.

This was not hypothetical for job 96800. Timestamps in its `run.log` give 96.6 min/epoch (11 h 17 min for epochs 2–8), so 50 epochs needed ≈80.5 h against a 72-hour batch limit; it would have been cut around epoch 44, and `bash_scripts/train_exp_001_spoco.slurm` passes `--resume` unconditionally. Its checkpoints were 6.2 GB, consistent with optimiser state in both `latest_model.pt` and `best_model.pt`. The same arithmetic applies to the replacement run (job 96971) unless `MEM-1` shortens the epoch — but its checkpoints now carry `param_group_order: "ordered_v1"`, so the resume restores optimiser state correctly instead of silently mismatching it.

- **Where:** `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py:355–368`; `bash_scripts/train_exp_001_spoco.slurm` (`--resume`)
- **Fix:** Build the group lists from the ordered generators and use `id()` sets for membership only:

  ```python
  enc_ids = {id(p) for p in raw_student.encoder.parameters()}
  tr_ids  = {id(p) for p in raw_student.transformer_decoder.parameters()}
  enc     = [p for p in raw_student.parameters() if id(p) in enc_ids]
  tr      = [p for p in raw_student.parameters() if id(p) in tr_ids]
  head    = [p for p in raw_student.parameters()
             if id(p) not in enc_ids and id(p) not in tr_ids]
  ```

  Ordering then derives from `parameters()` and is stable across processes. Note the existing checkpoints were written under an unknown ordering, so the first resume after the fix still cannot trust its optimiser state — restore the weights, drop the moments, and log that the run restarted Adam at epoch *k*.

### OPT-2 — Weight decay is applied to every normalisation scale and every conv bias
**Severity:** Medium

`plans.json` configures `InstanceNorm3d` with `affine: True` and `conv_bias: True`, so the backbone carries per-channel affine weights and biases throughout. All four trainers pass a single `weight_decay` to every parameter, decaying those affine scales toward zero alongside the convolution kernels — the standard misconfiguration, and one that bites harder on a fine-tune than on a from-scratch run because the pretrained scales are meaningful.

Phase 4 additionally hardcodes `weight_decay=1e-4` at the call site, with no CLI argument, unlike Phase 3's `--weight_decay`.

- **Where:** `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py:361–368`
- **Fix:** Split each group into decay / no-decay by `param.ndim >= 2`, giving the no-decay half `weight_decay=0.0`. Expose `--weight_decay` so Phase 4 matches the Phase 3 interface.

### OPT-3 — Decay acts on a scale-invariant output, so it only inflates the effective learning rate
**Severity:** Medium · SPOCO-specific

The SPOCO embedding is `F.normalize(x, p=2, dim=1)`. The loss is therefore invariant to the magnitude of the final decoder stage's weights: scaling them by *c* leaves every soft mask and every distance unchanged. Nothing in the gradient opposes decay along the radial direction, so those weights shrink monotonically toward zero while the effective step size on their *direction* grows as 1/‖w‖² — the well-documented effective-learning-rate drift in normalised networks. Combined with `OPT-2`, the InstanceNorm scales immediately upstream of the normalisation are decaying too.

- **Fix:** Exempt the final decoder stage (and, per `OPT-2`, all norm/bias parameters) from decay, or accept the drift deliberately and document it — but not while it is also being applied to the norm affines by default.

### OPT-4 — Phase 4 runs a constant learning rate for 50 epochs
**Severity:** Low

Exp 001 (Phase 3) uses `PolynomialLR` or `CosineAnnealingLR`, Exp 002 and Exp 003 use `CosineAnnealingLR`, each with a `scheduler.last_epoch` fast-forward on resume. Phase 4 has no scheduler at all: `1e-4` (×0.1 encoder, ×0.5 transformer) held flat throughout. Not wrong, but it is an unstated deviation from the pattern the other three experiments established, and it removes the annealing that usually sharpens a metric-learning embedding late in training.

- **Fix:** Add `CosineAnnealingLR(T_max=args.epochs, eta_min=1e-6)` with the same `last_epoch` fast-forward, or record in the experiment log that a constant LR is the intended Exp 001 baseline.

---

## Outside the five requested modes

### AUG-1 — Every DataLoader worker draws the same MONAI augmentation stream
**Severity:** Medium

MONAI's `Randomizable.R` is a *class* variable — one `np.random.RandomState` shared by every transform instance, seeded once at import. The repo uses `torch.utils.data.DataLoader`, whose default worker seeding covers `torch`, `random` and NumPy's global RNG but not MONAI's private state. `monai.data.DataLoader` exists precisely to install the `worker_init_fn` that fixes this.

So all workers within a rank replay an identical sequence of crop offsets and flip decisions for the *k*-th item each handles. Because the crop index lands in a different foreground list per scan, positions still differ; the `RandFlipd` coin flips do not. Phase 4 also omits `persistent_workers`, so workers re-fork from the same parent state and the stream restarts every epoch. Effective augmentation entropy drops by roughly the worker count (6 here).

- **Where:** `scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco.py:405–421`; `scripts/phase_3_voxtell_finetuning/exp_00{1,2,3}_*.py` (DataLoader construction)
- **Fix:** Pass `worker_init_fn=monai.data.utils.worker_init_fn`, or swap in `monai.data.DataLoader`. Note that `monai.data.DataLoader` also changes the default `collate_fn` to `list_data_collate`, so pair that swap with the `BC-1` fix rather than doing it blind.
