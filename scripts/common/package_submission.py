"""
===============================================================================
MODULE:         scripts/common/package_submission.py
PHASE:          Shared Infrastructure & Challenge Packaging
LOCATION:       scripts/common/package_submission.py
OBJECTIVE:      Validates 4D NIfTI prediction shape, spatial affines, non-zero
                contents, and compresses them into a flat .zip archive compliant
                with MICCAI 2026 / rexrank.ai challenge rules.
USAGE:          python scripts/common/package_submission.py \
                    --pred_dir ../data/predictions/phase_2a_exp_002_quantile_test \
                    --split test \
                    --output_zip ../data/submissions/exp_002_quantile_test_submission.zip
===============================================================================
"""

import os
import sys
import json
import hashlib
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import DATASET_JSON, RAW_IMAGES_DIR, PREDICTIONS_DIR, DATA_DIR
from scripts.common.orientation import load_nifti_ras


def compute_file_sha256(filepath: Path) -> str:
    """
    Signature:
        compute_file_sha256(filepath: Path) -> str

    Objective:
        Compute the SHA-256 hash checksum of a target file.

    Inputs:
        filepath (Path): Path to the target file.

    Outputs:
        str: Hexadecimal SHA-256 checksum string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def package_submission(
    pred_dir: Path,
    split: str,
    output_zip: Path,
    dataset_json_path: Path = DATASET_JSON,
    img_dir: Path = RAW_IMAGES_DIR,
    verify_contents: bool = True
) -> dict:
    """
    Signature:
        package_submission(
            pred_dir: Path,
            split: str,
            output_zip: Path,
            dataset_json_path: Path = DATASET_JSON,
            img_dir: Path = RAW_IMAGES_DIR,
            verify_contents: bool = True
        ) -> dict

    Objective:
        Inspects 4D NIfTI predictions for a specified split, validates array shape (F, H, W, D)
        against ground-truth finding count F, checks non-zero predictions, and bundles all files
        flat at the root of a compressed .zip archive.

    Inputs:
        pred_dir (Path): Directory containing predicted .nii.gz files.
        split (str): Split name ('val' or 'test').
        output_zip (Path): Target path for the output submission .zip archive.
        dataset_json_path (Path): Path to dataset.json metadata.
        img_dir (Path): Path to raw CT images for reference spatial affine alignment.
        verify_contents (bool): Whether to perform deep 4D shape and non-zero array inspection.

    Outputs:
        dict: Operational summary containing total_files_packaged, missing_cases,
              archive_size_mb, and sha256_hash.
    """
    pred_dir = Path(pred_dir).resolve()
    output_zip = Path(output_zip).resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Challenge Submission Packaging ===")
    print(f"Prediction Directory: {pred_dir}")
    print(f"Split               : {split}")
    print(f"Target Zip Output   : {output_zip}\n")

    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    split_entries = metadata.get(split, [])
    if not split_entries:
        raise ValueError(f"No entries found for split '{split}' in {dataset_json_path}")

    packaged_files = []
    missing_cases = []
    invalid_cases = []

    for entry in tqdm(split_entries, desc=f"Validating {split} predictions"):
        scan_filename = entry["name"]
        scan_id = scan_filename.replace(".nii.gz", "")
        pred_path = pred_dir / scan_filename

        if not pred_path.exists():
            missing_cases.append(scan_filename)
            continue

        if verify_contents:
            num_gt_findings = entry.get("num_findings", len(entry.get("categories", {})))
            ref_ct_path = img_dir / scan_filename if img_dir and (img_dir / scan_filename).exists() else None
            
            try:
                pred_arr, _, _ = load_nifti_ras(pred_path)

                if pred_arr.ndim == 3:
                    pred_arr = pred_arr[None, ...]

                if pred_arr.ndim != 4:
                    tqdm.write(f"[ERROR] {scan_filename} has invalid ndim {pred_arr.ndim} (expected 4D).")
                    invalid_cases.append(scan_filename)
                    continue

                if pred_arr.shape[0] != num_gt_findings:
                    tqdm.write(
                        f"[ERROR] {scan_filename} finding count mismatch: "
                        f"pred shape {pred_arr.shape[0]}, GT expected {num_gt_findings}."
                    )
                    invalid_cases.append(scan_filename)
                    continue

            except Exception as e:
                tqdm.write(f"[ERROR] Failed to load/verify {scan_filename}: {e}")
                invalid_cases.append(scan_filename)
                continue

        packaged_files.append(pred_path)

    if missing_cases:
        print(f"\n[WARNING] {len(missing_cases)} entries in split '{split}' were not found in {pred_dir}:")
        for m in missing_cases[:5]:
            print(f"  - Missing: {m}")
        if len(missing_cases) > 5:
            print(f"  ... and {len(missing_cases) - 5} more.")

    if invalid_cases:
        raise RuntimeError(f"Packaging aborted due to {len(invalid_cases)} invalid prediction files: {invalid_cases}")

    if not packaged_files:
        raise RuntimeError(f"No valid prediction files found to package for split '{split}'.")

    print(f"\nCreating flat ZIP archive with {len(packaged_files)} NIfTI files...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filepath in tqdm(packaged_files, desc="Writing files to ZIP"):
            # Enforce zero-directory flat layout (arcname is purely file basename)
            zipf.write(filepath, arcname=filepath.name)

    archive_bytes = output_zip.stat().st_size
    archive_mb = archive_bytes / (1024 * 1024)
    sha256_hash = compute_file_sha256(output_zip)

    summary = {
        "split": split,
        "total_split_entries": len(split_entries),
        "total_files_packaged": len(packaged_files),
        "missing_cases_count": len(missing_cases),
        "archive_path": str(output_zip),
        "archive_size_mb": round(archive_mb, 2),
        "sha256_hash": sha256_hash
    }

    print("\n" + "=" * 60)
    print("        SUBMISSION PACKAGING COMPLETED")
    print("=" * 60)
    print(f"Target Split            : {split}")
    print(f"Files Packaged          : {len(packaged_files)} / {len(split_entries)}")
    print(f"Archive Path            : {output_zip}")
    print(f"Archive Size            : {archive_mb:.2f} MB ({archive_bytes} bytes)")
    print(f"SHA-256 Checksum        : {sha256_hash}")
    print("=" * 60)

    return summary


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse CLI arguments for the submission packaging tool.
    """
    parser = argparse.ArgumentParser(description="ReXGroundingCT Challenge Submission Packager")
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory containing predicted NIfTI files"
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Dataset split to package (default: test)"
    )
    parser.add_argument(
        "--output_zip", type=str, required=True,
        help="Path for output submission .zip archive"
    )
    parser.add_argument(
        "--dataset_json", type=str, default=str(DATASET_JSON),
        help="Path to dataset.json"
    )
    parser.add_argument(
        "--img_dir", type=str, default=str(RAW_IMAGES_DIR),
        help="Path to raw CT images directory for affine verification"
    )
    parser.add_argument(
        "--no_verify", action="store_false", dest="verify_contents",
        help="Disable deep 4D array shape verification"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for package_submission."""
    args = parse_args()
    package_submission(
        pred_dir=Path(args.pred_dir),
        split=args.split,
        output_zip=Path(args.output_zip),
        dataset_json_path=Path(args.dataset_json),
        img_dir=Path(args.img_dir),
        verify_contents=args.verify_contents
    )


if __name__ == "__main__":
    main()
