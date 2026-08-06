# ReXGroundingCT Model Training — Antigravity Operating Rules

As the AI pair-programming assistant for the ReXGroundingCT Model Training & Fine-Tuning Workspace, these are your global operating constraints and repository-wide rules.

## 🚨 PRIME DIRECTIVE: Post-Edit Compliance Auditing
* **Mandatory Post-Edit Compliance Verification**: After creating or modifying any codebase script (`scripts/`) or experiment log file (`logs/`), you MUST immediately perform a self-audit to verify full compliance with all contracts, naming conventions, relative path directives, function signature docstrings, and 1:1 experiment subfolder pairing rules defined in `AGENTS.md`.

---

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
  * `.agents/AGENTS.md`, `.agents/shared/MASTER_PLAN.md`, official dataset schemas (`../data/dataset.json`), and validated evaluator pipelines (`scripts/common/evaluate.py`).
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

### 4. Centralized Spatial Alignment & Canonical RAS Orientation
* **Centralized Spatial Engine (`scripts/common/orientation.py`)**: All NIfTI volumes (ground truth, images, and model predictions) MUST be loaded, inspected, and saved using the centralized spatial functions `load_nifti_ras()` and `save_nifti()`.
* **Prohibited: Manual Transposes & Naive Shape Assertions**: NEVER rely on manual array transposes or naive shape checks (e.g. `shape == (512, 512, 456)`) to assume anatomical orientation. Always inspect and canonicalize spatial orientation using `scripts/common/orientation.py`.
* **Domain-Driven Spatial Anchoring**: Segmentation masks conceptually lack independent physical orientation. Their spatial affine MUST strictly be inherited from the parent CT scan. `scripts/common/orientation.py` enforces this natively by always anchoring 4D mask loads to their parent CT scan, ignoring the mask's internal NIfTI header.

### 5. Directory & Structure Conventions
* `scripts/`: Categorized into shared utilities (`common/`) and phase-specific execution pipelines (`phase_2a_rule_based/`, `phase_2b_voxtell/`, `phase_3_training/`).
* **Experiment Log Pairing & Subfolder Convention**: Experiment scripts follow `exp_XXX_<description>.py` and pair 1:1 with corresponding dedicated experiment subfolders `logs/<phase>/<exp_name>/` (containing `eval.md`, `eval_results_val.json`, `run.log`, and `failure_snapshots/`).
* **Function Signature & Docstring Directive**: Standard library/utility functions and class methods MUST include a docstring as their very first statement outlining the function signature, a concise explanation of what the function does, its input arguments, and expected return outputs (boilerplate CLI helpers like `main()` and `parse_args()` only require a concise high-level docstring without a signature block).
* **MONAI Framework Preference Contract**: Wherever reasonable and applicable (such as GPU-accelerated spatial resampling, intensity normalization, transform pipelines, and deep learning preprocessing), MONAI (`monai.transforms`) should be preferred for 3D medical image and mask operations, provided it respects the centralized spatial engine contracts in `scripts/common/orientation.py`.

---

## 🧠 Behavior & Epistemic Modesty
* **Epistemic Modesty**: All preliminary empirical observations use calibrated, modest phrasing (*"initial evidence suggests"*, *"preliminary tests indicate"*).
* **Non-Prescriptive Scientific Inquiry Directive**: Technical, mathematical, or infrastructural constraints (e.g., 4D Back-Reorientation coordinate math, fast SSD volume caching, patient-level split hygiene) MUST be strictly enforced as non-negotiable contracts. Conversely, algorithmic, loss-level, or post-processing choices (e.g., specific loss functions, fixed volume noise pruning thresholds, or binarization cutoffs) MUST NEVER be framed as dogmatic or mandatory prescriptions. Modeling strategies attempting to solve data challenges must be framed as testable hypotheses (e.g., Hypothesis H1 vs H2) to allow open scientific discovery and prevent bias when seeking optimal solutions.
* **Git Commit & Push Approval Protocol**: NEVER execute `git commit` or `git push` automatically. You MUST always ask the USER for explicit permission before staging, committing, or pushing code or documentation changes.
* **Relative Path Directive**: ALL documentation, markdown files, and codebase scripts MUST strictly use **relative paths** (e.g., `scripts/phase_2b_voxtell/exp_001_voxtell_inference.py`).
* **Master Plan Abstraction Directive**: The master plan (`.agents/shared/MASTER_PLAN.md`) MUST be maintained strictly as a big-picture, high-level scientific and technical roadmap. It MUST NOT contain references to specific file paths, script names, or implementation conventions (which belong exclusively in `AGENTS.md` or `README.md`).
