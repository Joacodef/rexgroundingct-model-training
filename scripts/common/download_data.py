"""
===============================================================================
MODULE:         download_data.py
LOCATION:       scripts/common/download_data.py
OBJECTIVE:      Selective, high-throughput dataset downloader for ReXGroundingCT
                and CT-RATE volumetric scans via Hugging Face Hub API.
===============================================================================
"""

import os
import json
import shutil
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATA_DIR,
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    DATASET_JSON,
    HF_HOME,
)

# 1. Load active .env environment variables
load_dotenv(override=False)
if HF_HOME:
    os.environ["HF_HOME"] = str(HF_HOME)
    HF_HOME.mkdir(parents=True, exist_ok=True)


def download_rexgroundingct_metadata(
    repo_id: str = "rajpurkarlab/ReXGroundingCT",
    token: Optional[str] = None,
    dest_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Download ReXGroundingCT metadata and challenge specification JSON files.

    Signature:
        download_rexgroundingct_metadata(repo_id: str, token: Optional[str], dest_dir: Optional[Path]) -> dict[str, Path]

    Args:
        repo_id (str): Hugging Face repository identifier for ReXGroundingCT.
        token (Optional[str]): Hugging Face API access token.
        dest_dir (Optional[Path]): Destination directory (defaults to DATA_DIR).

    Returns:
        dict[str, Path]: Dictionary mapping filename keys to local Path destinations.
    """
    target_dir = dest_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.getenv("HF_TOKEN")

    metadata_files = [
        "dataset.json",
        "MICCAI_challenge_dataset.json",
        "anatomical_cot.json",
        "reports_dataset.json",
    ]

    downloaded_paths = {}
    print(f"=== Downloading ReXGroundingCT Metadata from {repo_id} ===")
    for filename in metadata_files:
        try:
            local_hf_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                token=hf_token,
                cache_dir=str(HF_HOME) if HF_HOME else None,
            )
            dest_file = target_dir / filename
            shutil.copy2(local_hf_path, dest_file)
            downloaded_paths[filename] = dest_file
            print(f"  [OK] Saved {filename} -> {dest_file}")
        except Exception as e:
            print(f"  [WARNING] Could not download {filename}: {e}")

    return downloaded_paths


def download_rexgroundingct_segmentations(
    repo_id: str = "rajpurkarlab/ReXGroundingCT",
    token: Optional[str] = None,
    dest_dir: Optional[Path] = None,
    max_workers: int = 8,
) -> int:
    """
    Download all ground-truth 3D segmentation masks (.nii.gz) for ReXGroundingCT using snapshot_download.

    Signature:
        download_rexgroundingct_segmentations(repo_id: str, token: Optional[str], dest_dir: Optional[Path], max_workers: int) -> int

    Args:
        repo_id (str): Hugging Face repository identifier.
        token (Optional[str]): Hugging Face API access token.
        dest_dir (Optional[Path]): Destination directory (defaults to RAW_MASKS_DIR).
        max_workers (int): Unused concurrency argument kept for backwards compatibility.

    Returns:
        int: Number of successfully downloaded / verified segmentation masks.
    """
    from huggingface_hub import snapshot_download

    target_dir = dest_dir or RAW_MASKS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.getenv("HF_TOKEN")

    print(f"=== Downloading all segmentation masks via snapshot_download from {repo_id} ===")
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["segmentations/*.nii.gz"],
        token=hf_token,
        cache_dir=str(HF_HOME) if HF_HOME else None,
    )

    seg_src = Path(snapshot_path) / "segmentations"
    seg_files = list(seg_src.glob("*.nii.gz"))
    print(f"Syncing {len(seg_files)} segmentation masks into {target_dir}...")

    success_count = 0
    for src_file in tqdm(seg_files, desc="Copying masks to target"):
        dest_file = target_dir / src_file.name
        if not dest_file.exists() or dest_file.stat().st_size == 0:
            shutil.copy2(src_file, dest_file)
        success_count += 1

    print(f"=== Ground Truth Masks Complete: {success_count} available in {target_dir} ===")
    return success_count


def get_required_scan_list(
    dataset_json_path: Path,
    splits: Optional[List[str]] = None,
) -> List[str]:
    """
    Extract the unique list of scan image filenames declared across specified splits.

    Signature:
        get_required_scan_list(dataset_json_path: Path, splits: Optional[List[str]]) -> List[str]

    Args:
        dataset_json_path (Path): Path to dataset.json file.
        splits (Optional[List[str]]): List of splits to extract ('train', 'val', 'test'). Defaults to all present.

    Returns:
        List[str]: Sorted list of unique scan filenames.
    """
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_splits = splits or [k for k in data.keys() if isinstance(data[k], list)]
    scan_names = set()

    for split in target_splits:
        if split in data and isinstance(data[split], list):
            for entry in data[split]:
                name = entry.get("name") or entry.get("image") or entry.get("scan_id")
                if name:
                    scan_names.add(name if name.endswith(".nii.gz") else f"{name}.nii.gz")

    return sorted(list(scan_names))


def resolve_ct_rate_remote_path(filename: str) -> str:
    """
    Construct the nested hierarchical repository path in CT-RATE for a given scan filename.

    Signature:
        resolve_ct_rate_remote_path(filename: str) -> str

    Args:
        filename (str): Base filename (e.g. 'train_2550_a_2.nii.gz' or 'valid_12_a_1.nii.gz').

    Returns:
        str: Relative path inside the Hugging Face CT-RATE dataset repository.
    """
    name_no_ext = filename.split(".")[0]
    parts = name_no_ext.split("_")
    prefix = parts[0]

    if prefix in ("train", "valid", "test"):
        folder1 = f"{parts[0]}_{parts[1]}"
        folder2 = f"{parts[0]}_{parts[1]}_{parts[2]}"
        return f"dataset/{prefix}/{folder1}/{folder2}/{filename}"

    raise ValueError(f"Unrecognized scan prefix in filename '{filename}' (expected train, valid, or test)")


def download_ct_rate_scans(
    scan_names: List[str],
    repo_id: str = "ibrahimhamamci/CT-RATE",
    token: Optional[str] = None,
    dest_dir: Optional[Path] = None,
    max_workers: int = 16,
) -> Tuple[int, int]:
    """
    Selectively download only the whitelisted CT-RATE volumetric scans into the image directory.

    Signature:
        download_ct_rate_scans(scan_names: List[str], repo_id: str, token: Optional[str], dest_dir: Optional[Path], max_workers: int) -> Tuple[int, int]

    Args:
        scan_names (List[str]): List of scan filenames to download.
        repo_id (str): Hugging Face repository identifier for CT-RATE.
        token (Optional[str]): Hugging Face API access token.
        dest_dir (Optional[Path]): Destination directory (defaults to RAW_IMAGES_DIR).
        max_workers (int): Number of concurrent worker threads.

    Returns:
        Tuple[int, int]: (Number of successful downloads, total requested scans).
    """
    target_dir = dest_dir or RAW_IMAGES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    hf_token = token or os.getenv("HF_TOKEN")

    print(f"=== Initiating Selective CT-RATE Download ({len(scan_names)} scans) ===")

    def _download_single_scan(filename: str) -> bool:
        dest_file = target_dir / filename
        if dest_file.exists() and dest_file.stat().st_size > 0:
            return True

        try:
            remote_path = resolve_ct_rate_remote_path(filename)
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                token=hf_token,
                cache_dir=str(HF_HOME) if HF_HOME else None,
            )
            real_blob = Path(cached_path).resolve()
            shutil.copy2(cached_path, dest_file)
            if real_blob.exists() and "blobs" in str(real_blob):
                try:
                    real_blob.unlink()
                except Exception:
                    pass
            return True
        except Exception as exc:
            print(f"Failed to download {filename}: {exc}")
            return False

    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_single_scan, name): name for name in scan_names}
        with tqdm(total=len(scan_names), desc="Downloading CT-RATE Volumes") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)

    print(f"=== CT-RATE Download Complete: {success_count}/{len(scan_names)} available in {target_dir} ===")
    return success_count, len(scan_names)


def main():
    """
    CLI entrypoint for downloading ReXGroundingCT metadata, masks, and selective CT-RATE scans.
    """
    parser = argparse.ArgumentParser(description="Selective ReXGroundingCT & CT-RATE Dataset Downloader")
    parser.add_argument("--metadata_only", action="store_true", help="Download only metadata JSON files")
    parser.add_argument("--masks_only", action="store_true", help="Download only ground-truth segmentation masks")
    parser.add_argument("--scans_only", action="store_true", help="Download only CT-RATE 3D scan volumes")
    parser.add_argument("--splits", nargs="+", default=None, help="Splits to download scans for (e.g. val train test)")
    parser.add_argument("--workers", type=int, default=16, help="Number of download threads (default: 16)")
    args = parser.parse_args()

    # Step 1: Download Metadata
    if not args.masks_only and not args.scans_only:
        download_rexgroundingct_metadata()

    # Step 2: Download Ground Truth Segmentation Masks
    if not args.metadata_only and not args.scans_only:
        download_rexgroundingct_segmentations(max_workers=args.workers)

    # Step 3: Download Selective CT-RATE Scans
    if not args.metadata_only and not args.masks_only:
        dataset_json_path = DATASET_JSON if DATASET_JSON.exists() else DATA_DIR / "dataset.json"
        if not dataset_json_path.exists():
            print(f"[ERROR] Cannot download scans: {dataset_json_path} does not exist.")
            return

        required_scans = get_required_scan_list(dataset_json_path, splits=args.splits)
        print(f"Total whitelisted scans for splits {args.splits or 'ALL'}: {len(required_scans)}")
        download_ct_rate_scans(required_scans, max_workers=args.workers)


if __name__ == "__main__":
    main()
