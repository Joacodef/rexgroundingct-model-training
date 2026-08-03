"""
===============================================================================
MODULE:         Centralized Path & Environmental Configuration
LOCATION:       scripts/config.py
OBJECTIVE:      Single source of truth for dataset, model, and log directories.
                Paths are dynamically resolved relative to repository root with
                environment variable overrides.
===============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load local .env environment variables if present
load_dotenv(override=True)

# 2. Determine Repository Root Directory dynamically
# scripts/config.py -> parent is scripts/ -> parent is sub-repo root
ROOT_DIR = Path(__file__).resolve().parent.parent
PARENT_DIR = ROOT_DIR.parent

# 3. Base Directory Definitions
# Check parent container directory (/home/.../rex_project) for shared data and models folders
DATA_DIR = Path(os.getenv("DATA_DIR") or (PARENT_DIR / "data" if (PARENT_DIR / "data").exists() else ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR") or (PARENT_DIR / "models" if (PARENT_DIR / "models").exists() else ROOT_DIR / "models"))
LOGS_DIR = ROOT_DIR / "logs"
SCRATCH_DIR = ROOT_DIR / "scratch"
VISUALIZATIONS_DIR = ROOT_DIR / "scan_visualizations"

# 4. Core Dataset & Asset Paths (Env Override -> Fallback to project relative default)
DATASET_JSON = Path(os.getenv("DATASET_JSON") or (DATA_DIR / "dataset.json"))
RAW_IMAGES_DIR = Path(os.getenv("IMG_RAW_DIR") or (DATA_DIR / "raw" / "images"))
RAW_MASKS_DIR = Path(os.getenv("SEG_RAW_DIR") or (DATA_DIR / "raw" / "segmentations"))
PREPROCESSED_DIR = Path(os.getenv("DATA_PREP_DIR") or (DATA_DIR / "preprocessed"))
PREDICTIONS_DIR = Path(os.getenv("DATA_PRED_DIR") or (DATA_DIR / "predictions"))
TEXT_CACHE_DIR = Path(os.getenv("TEXT_CACHE_DIR") or (DATA_DIR / "text_cache"))

# 5. Model & Checkpoint Paths
MODEL_DIR = Path(os.getenv("MODEL_DIR") or (MODELS_DIR / "voxtell_v1.1"))
CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR") or (MODELS_DIR / "checkpoints"))

# 6. Temporary / Fast SSD Storage (Fallback to system /tmp)
TMP_PREP_DIR = Path(os.getenv("TMP_PREP_DIR") or "/tmp/rexgroundingct_preprocessed")

# 7. Hardware & Hardware Isolation Settings
DEFAULT_DEVICE = os.getenv("DEFAULT_DEVICE", "cuda:0")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# 8. Challenge 14-Category Definitions & Taxonomy
CATEGORY_MAP = {
    '1a': 'Bronchial wall thickening',
    '1b': 'Bronchiectasis',
    '1c': 'Emphysema',
    '1d': 'Septal thickening',
    '1e': 'Micronodules',
    '1f': 'Other non-focal',
    '2a': 'Linear opacities',
    '2b': 'Atelectasis / consolidation',
    '2c': 'Ground-glass opacity',
    '2d': 'Pulmonary nodules / masses',
    '2e': 'Pleural effusion / thickening',
    '2f': 'Honeycombing',
    '2g': 'Pneumothorax',
    '2h': 'Other focal'
}

NON_FOCAL_CATEGORIES = {'1a', '1b', '1c', '1d', '1e', '1f'}
FOCAL_CATEGORIES = {'2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h'}
REVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}

# 4-Tier Spatial Prior Taxonomy Mapping
SPATIAL_TAXONOMY = {
    '1a': 'Hilar / Peribronchial',
    '1b': 'Hilar / Peribronchial',
    '1c': 'Apical Dominant',
    '1d': 'Isotropic / Parenchymal',
    '1e': 'Isotropic / Parenchymal',
    '1f': 'Isotropic / Parenchymal',
    '2a': 'Isotropic / Parenchymal',
    '2b': 'Basal / Dependent',
    '2c': 'Isotropic / Parenchymal',
    '2d': 'Isotropic / Parenchymal',
    '2e': 'Basal / Dependent',
    '2f': 'Basal / Dependent',
    '2g': 'Isotropic / Parenchymal',
    '2h': 'Isotropic / Parenchymal',
}


