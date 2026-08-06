"""
===============================================================================
SCRIPT:         Batch 3D Volume Preprocessing & Fast Caching Engine
LOCATION:       scripts/common/preprocess.py
OBJECTIVE:      Preprocess raw 3D CT scans into fast SSD temporary storage 
                (/tmp/rexgroundingct_preprocessed/) or persistent storage, 
                performing spatial canonicalization and Z-score normalization.
USAGE:          python scripts/common/preprocess.py
===============================================================================
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from monai.transforms import Compose, LoadImaged, Orientationd, Spacingd, SaveImage, EnsureChannelFirstd, MapTransform
from monai.data import Dataset, DataLoader, decollate_batch

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    TMP_PREP_DIR, PREPROCESSED_DIR, LOGS_DIR
)

# Configure logging
(LOGS_DIR / "common").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOGS_DIR / "common" / "preprocess.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocess")

# Load environment variables from .env file
load_dotenv(override=True)

# Dynamic paths from scripts.config
DATASET_JSON = str(DATASET_JSON)
IMG_DIR = str(RAW_IMAGES_DIR)
SEG_DIR = str(RAW_MASKS_DIR)
TMP_PREP_DIR = str(TMP_PREP_DIR) if TMP_PREP_DIR else None
DATA_PREP_DIR = str(PREPROCESSED_DIR) if PREPROCESSED_DIR else None

# Determine output directory based on environment (fast SSD cache vs persistent storage)
if TMP_PREP_DIR:
    OUT_DIR = TMP_PREP_DIR
    logger.info(f"Fast SSD cache mode detected. Writing tensors to: {OUT_DIR}")
elif DATA_PREP_DIR:
    OUT_DIR = DATA_PREP_DIR
    logger.info(f"Persistent storage mode detected. Writing tensors to: {OUT_DIR}")
else:
    logger.error("Configuration error: Neither TMP_PREP_DIR nor PREPROCESSED_DIR resolved.")
    raise ValueError("Configuration error: Neither TMP_PREP_DIR nor PREPROCESSED_DIR resolved.")

os.makedirs(OUT_DIR, exist_ok=True)

class CopyAffined(MapTransform):
    """
    Dictionary-based MONAI transform to copy affine and spatial metadata from a reference key to target keys.
    Fixes identity affine issues in raw segmentation masks.
    """
    def __init__(self, keys, ref_key="image", allow_missing_keys=False):
        """
        Signature:
            __init__(keys, ref_key: str, allow_missing_keys: bool) -> None

        Objective:
            Initialize CopyAffined transform mapping target keys to a reference metadata key.

        Inputs:
            keys (Sequence[str] | str): Keys of the target dictionary items to modify.
            ref_key (str): Key of the reference dictionary item containing ground-truth affine. Default 'image'.
            allow_missing_keys (bool): Whether to ignore missing keys. Default False.

        Outputs:
            None
        """
        super().__init__(keys, allow_missing_keys)
        self.ref_key = ref_key

    def __call__(self, data):
        """
        Signature:
            __call__(data: dict) -> dict

        Objective:
            Execute affine metadata copy from reference tensor ('image') to target tensors ('label').

        Inputs:
            data (dict): MONAI data dictionary containing image and label tensors.

        Outputs:
            dict: Updated MONAI data dictionary with synchronized affine matrices.
        """
        d = dict(data)
        ref_tensor = d[self.ref_key]
        for key in self.key_iterator(d):
            if key == self.ref_key:
                continue
            
            target_tensor = d[key]
            # Copy PyTorch tensor affine
            if hasattr(target_tensor, "affine") and hasattr(ref_tensor, "affine"):
                target_tensor.affine = ref_tensor.affine.clone()
            
            # Copy important metadata keys
            if hasattr(target_tensor, "meta") and hasattr(ref_tensor, "meta"):
                for meta_k in ["affine", "original_affine", "pixdim", "space", "spatial_shape"]:
                    if meta_k in ref_tensor.meta:
                        val = ref_tensor.meta[meta_k]
                        if hasattr(val, "clone"):
                            target_tensor.meta[meta_k] = val.clone()
                        elif hasattr(val, "copy"):
                            target_tensor.meta[meta_k] = val.copy()
                        else:
                            target_tensor.meta[meta_k] = val
        return d

def main():
    """Main CLI entry point for batch volume preprocessing, MONAI 1.5mm isotropic resampling, and tensor caching."""
    # 1. Define parser for CLI arguments
    parser = argparse.ArgumentParser(description="ReXGroundingCT Preprocessing Script")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"], 
                        help="Dataset split to preprocess (train, val, test)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"STARTING PREPROCESSING FOR SPLIT: {args.split}")
    logger.info("=" * 60)

    # 2. Read dataset.json
    logger.info(f"Reading metadata from: {DATASET_JSON}")
    with open(DATASET_JSON, 'r') as f:
        metadata = json.load(f)
    
    entries = metadata.get(args.split, [])
    if not entries:
        logger.warning(f"No cases found for split '{args.split}'. Check dataset.json structure.")
        return
    
    # 3. Path mapping and existence validation
    data_dicts = []
    missing_cases = 0

    for entry in entries:
        img_path = os.path.join(IMG_DIR, entry["name"])
        seg_path = os.path.join(SEG_DIR, entry["name"])
        
        # File existence check: Ensure both raw files exist before adding to MONAI pipeline
        if not os.path.exists(img_path) or not os.path.exists(seg_path):
            logger.warning(f"Missing raw files for {entry['name']}. Skipping.")
            missing_cases += 1
            continue

        num_f = len(entry.get("findings", {})) 
        data_dicts.append({
            "image": img_path, 
            "label": seg_path,
            "num_findings": num_f
        })

    logger.info(f"Found {len(data_dicts)} valid cases. Skipped {missing_cases} missing cases.")
    
    if len(data_dicts) == 0:
        logger.error("No valid data to process. Aborting.")
        return

    # 4. Spatial transformations pipeline (MONAI)
    preprocessing_pipeline = Compose([
        # Load NIfTI without automatic reordering
        LoadImaged(keys=["image", "label"], reader="NibabelReader"),
        
        # Ensure channel first [1, H, W, D] for the CT image.
        EnsureChannelFirstd(keys=["image"]), 
        
        # Register the findings channel at index 0 (start of tensor) for the multi-channel segmentation mask
        EnsureChannelFirstd(keys=["label"], channel_dim=0),
        
        # Copy affine and spatial metadata from image to label to align them (fixing identity affine in labels)
        CopyAffined(keys=["label"], ref_key="image"),
        
        # Standardize orientation to RAS
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # Resample to 1.5mm isotropic spacing
        Spacingd(
            keys=["image", "label"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=["bilinear", "nearest"]
        )
    ])

    dataset = Dataset(data=data_dicts, transform=preprocessing_pipeline)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # 5. Savers (resample=False prevents reverting Spacingd)
    save_img = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="ct", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False
    )
    
    # Save segmentation mask maintaining the channel dim (Finding axes)
    save_seg = SaveImage(
        output_dir=OUT_DIR, 
        output_postfix="seg", 
        output_ext=".nii.gz", 
        resample=False,
        separate_folder=False,
        channel_dim=0 # FORCE MONAI to know channel is at 0, don't move it to the end
    )

    logger.info(f"Starting batch preprocessing of {len(data_dicts)} volumes to {OUT_DIR}...")
    
    # 6. Execution and validation
    for batch in tqdm(dataloader, desc=f"Preprocessing {args.split}", unit="scan"):
        for data in decollate_batch(batch):
            f_expected = int(data["num_findings"])
            f_real = data["label"].shape[0]
            filename = data["image"].meta.get("filename_or_obj", "Unknown")
            
            # Log successful properties of loaded/aligned scan
            logger.info(
                f"Preprocessed case: {os.path.basename(filename)} | "
                f"CT final shape: {data['image'].shape} | "
                f"Seg final shape: {data['label'].shape} | "
                f"Findings (expected={f_expected}, real={f_real})"
            )
            
            # Hard validation of dimensionality (F, H, W, D) for VoxTell
            if f_real != f_expected:
                error_msg = (
                    f"Dimensionality error in {filename}:\n"
                    f"  dataset.json declares {f_expected} findings, but loaded mask has {f_real} channels.\n"
                    f"  Loaded CT Shape  : {data['image'].shape}\n"
                    f"  Loaded Seg Shape : {data['label'].shape}\n"
                    f"  Please run the diagnostic utility:\n"
                    f"  python scratch/phase_3_fine_tuning/proof_of_concept/data_utils/diagnose_dataset.py --split {args.split}"
                )
                logger.error(error_msg)
                raise AssertionError(error_msg)
            
            save_img(data["image"])
            save_seg(data["label"])

    logger.info("=" * 60)
    logger.info(f"PREPROCESSING SPLIT '{args.split}' COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()