"""
===============================================================================
SCRIPT:         scripts/common/download_missing_test_images.py
PHASE:          Shared Infrastructure / Dataset Alignment
LOCATION:       scripts/common/download_missing_test_images.py
OBJECTIVE:      Downloads all missing 200 test set CT volumes directly from
                HuggingFace (IBRAHIMHAMAMCI/CT-RATE) and extracts them into
                data/raw/images/.
USAGE:          python scripts/common/download_missing_test_images.py
===============================================================================
"""

import os
import sys
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import DATASET_JSON, RAW_IMAGES_DIR


def get_hf_path_for_volume(filename: str) -> str:
    """
    Signature:
        get_hf_path_for_volume(filename: str) -> str

    Objective:
        Derive the HuggingFace repository relative path in IBRAHIMHAMAMCI/CT-RATE
        from a raw CT volume filename.
        Example: 'train_302_a_2.nii.gz' -> 'dataset/train/train_302/train_302_a/train_302_a_2.nii.gz'

    Inputs:
        filename (str): Base filename of CT scan (e.g. 'train_302_a_2.nii.gz').

    Outputs:
        str: Relative path inside HuggingFace repo.
    """
    clean_name = filename.replace(".nii.gz", "")
    parts = clean_name.split("_")
    # parts: ['train', '302', 'a', '2']
    patient_id = f"train_{parts[1]}"
    study_id = f"{patient_id}_{parts[2]}"
    return f"dataset/train/{patient_id}/{study_id}/{filename}"


def download_missing_test_images():
    """
    Signature:
        download_missing_test_images() -> None

    Objective:
        Identify missing test scan volumes from dataset.json in data/raw/images/
        and download them directly from IBRAHIMHAMAMCI/CT-RATE on HuggingFace.
    """
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    test_entries = metadata.get("test", [])
    if not test_entries:
        print("[ERROR] No test entries found in dataset.json")
        return

    raw_images_dir = Path(RAW_IMAGES_DIR)
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    missing_names = sorted([
        item["name"] for item in test_entries
        if not (raw_images_dir / item["name"]).exists()
    ])

    print("=" * 70)
    print(f"Total Test Set Entries in dataset.json : {len(test_entries)}")
    print(f"Currently Present in data/raw/images/ : {len(test_entries) - len(missing_names)}")
    print(f"Missing Test Scans to Download       : {len(missing_names)}")
    print("=" * 70)

    if not missing_names:
        print("[SUCCESS] All 300 test CT images are already present on disk!")
        return

    successful_downloads = 0
    failed_downloads = []

    for filename in tqdm(missing_names, desc="Downloading missing test CT scans"):
        repo_file_path = get_hf_path_for_volume(filename)
        dest_path = raw_images_dir / filename

        try:
            downloaded_path = hf_hub_download(
                repo_id="IBRAHIMHAMAMCI/CT-RATE",
                filename=repo_file_path,
                repo_type="dataset"
            )
            shutil.copy(downloaded_path, dest_path)
            successful_downloads += 1
        except Exception as e:
            tqdm.write(f"[WARNING] Failed to download {filename} ({repo_file_path}): {e}")
            failed_downloads.append(filename)

    final_present = sum(1 for item in test_entries if (raw_images_dir / item["name"]).exists())

    print("\n" + "=" * 70)
    print("        DOWNLOAD & EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successfully Downloaded : {successful_downloads} / {len(missing_names)}")
    print(f"Failed Downloads        : {len(failed_downloads)}")
    print(f"Total Test Images On Disk: {final_present} / {len(test_entries)}")
    print("=" * 70)


if __name__ == "__main__":
    download_missing_test_images()
