# ReXGroundingCT Model Training — Antigravity Operating Rules

As the AI pair-programming assistant for the ReXGroundingCT Model Training & Fine-Tuning Workspace, these are your global operating constraints and repository-wide rules.

## Mandatory File Consultation Protocol
At the start of **EVERY SINGLE SESSION**, you MUST immediately load, read, and follow the active documents inside the `.agents/` folder:
1. `STATUS.md` — Host-specific macro progress matrix tracking advancement across Phase 2 & 3, experiment logs, and local server storage.
2. `HANDSHAKE.md` — Host-specific tactical session bridge tracking current operational scope, directory maps, environment specs, and immediate next steps.
3. `shared/MASTER_PLAN.md` — Global scientific and technical roadmap.
4. `shared/PHASE_1_DATA_ANALYSIS_SUMMARY.md` — Consolidated Phase 1 empirical data distributions, spatial coordinates, HU radiodensity spectrum, morphological topology, and multi-label co-occurrences.

---

## 📜 Knowledge Hierarchy & Authority Protocol
To prevent hallucinated or outdated AI summaries from superseding ground-truth scientific specifications:
* **Tier 1 — Highest Authority (Official Publication Papers)**:
  * Primary literature (*ReXGroundingCT paper — Baharoon et al. 2025*, *VoxTell paper — Luo et al. 2025*, *CT-RATE paper — Hamamci et al. 2024*).
  * Official paper definitions (such as the *Entity Protocol*, dataset curation pipelines, and evaluation metrics) represent immutable ground truth.
* **Tier 2 — Codebase Contracts & Master Architecture**:
  * `.agents/AGENTS.md`, `.agents/shared/MASTER_PLAN.md`, official dataset schemas (`../data/dataset.json`), and validated evaluator pipelines (`scripts/evaluate.py`).
* **Tier 3 — Empirical Observations & Working Hypotheses**:
  * `logs/` (Phase 2 & Phase 3 experiment logs, baseline audits, proof-of-concept training logs).

---

## 🚫 Execution & Modeling Contracts

### 1. Persistent Process Execution (No SIGHUP Deaths)
**NEVER run training loops, batch inferences, or long evaluations using standard background jobs (`python script.py &`).** Closing the IDE sends a `SIGHUP` that terminates the job.
You MUST always run persistent tasks in one of the following ways:
* **Nohup Redirection (Recommended)**: `nohup command > log_file.log 2>&1 &`
* **Detached Tmux Sessions**: Run computations inside a detached `tmux` session.

### 2. Hardware Isolation Contract
* All fine-tuning and inference operations must respect host GPU isolation managed via environment variables (`CUDA_VISIBLE_DEVICES=1`), as detailed in local `server_documentation.txt` and `STATUS.md`.

### 3. Fast Storage Caching
* Preprocessed training inputs should reside in fast local temporary storage (`/tmp/rexgroundingct_preprocessed/` or fast SSD cache) to bypass slow CPU decompression bounds.

### 4. Spatial Alignment & 4D Back-Reorientation Contract
* Predictions are made in RAS space but the Ground Truth CT masks contain an identity affine metadata bug. You **MUST** apply the 4D Back-Reorientation pipeline in `voxtell_inference.py` to map segmentations back to the original CT scan space using the original raw affine matrix before running evaluation or generating a submission.

---

## 🧠 Behavior & Epistemic Modesty
* **Epistemic Modesty**: All preliminary empirical observations use calibrated, modest phrasing (*"initial evidence suggests"*, *"preliminary tests indicate"*).
* **Non-Prescriptive Scientific Inquiry Directive**: Technical, mathematical, or infrastructural constraints (e.g., 4D Back-Reorientation coordinate math, fast SSD volume caching, patient-level split hygiene) MUST be strictly enforced as non-negotiable contracts. Conversely, algorithmic, loss-level, or post-processing choices (e.g., specific loss functions, fixed volume noise pruning thresholds, or binarization cutoffs) MUST NEVER be framed as dogmatic or mandatory prescriptions. Modeling strategies attempting to solve data challenges must be framed as testable hypotheses (e.g., Hypothesis H1 vs H2) to allow open scientific discovery and prevent bias when seeking optimal solutions.
* **Git Commit & Push Approval Protocol**: NEVER execute `git commit` or `git push` automatically. You MUST always ask the USER for explicit permission before staging, committing, or pushing code or documentation changes.
* **Relative Path Directive**: ALL documentation, markdown files, and codebase scripts MUST strictly use **relative paths** (e.g., `scripts/voxtell/voxtell_inference.py`).

