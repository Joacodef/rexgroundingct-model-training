"""
===============================================================================
MODULE:         Standalone Offline Post-Processing Engine
LOCATION:       scripts/analysis/postprocess_predictions.py
OBJECTIVE:      Provide decoupled, multi-threaded CPU post-processing on raw 
                probability maps or binary segmentation masks (per-finding 
                probability thresholding and 3D connected-component noise pruning).
===============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.ndimage
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import RAW_IMAGES_DIR, DATASET_JSON, CATEGORY_MAP
from scripts.common.orientation import load_nifti_ras, save_nifti


def prune_small_components_3d(mask_3d: np.ndarray, min_voxels: int) -> np.ndarray:
    """
    Signature:
        prune_small_components_3d(mask_3d: np.ndarray, min_voxels: int) -> np.ndarray

    Objective:
        Remove isolated 3D connected components whose total volume in voxels is strictly less than min_voxels.

    Inputs:
        mask_3d (np.ndarray): 3D boolean or binary uint8 mask array (X, Y, Z).
        min_voxels (int): Minimum component voxel size threshold.

    Outputs:
        np.ndarray: Cleaned 3D binary uint8 mask array.
    """
    if min_voxels <= 0 or not np.any(mask_3d):
        return mask_3d.astype(np.uint8)

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_features = scipy.ndimage.label(mask_3d > 0, structure=structure)
    if num_features == 0:
        return np.zeros_like(mask_3d, dtype=np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    # Identify labels that meet or exceed the size threshold (excluding background label 0)
    valid_labels = np.where(component_sizes >= min_voxels)[0]
    valid_labels = valid_labels[valid_labels > 0]

    cleaned_mask = np.isin(labeled, valid_labels).astype(np.uint8)
    return cleaned_mask


def process_single_case(args_tuple: tuple) -> str:
    """
    Signature:
        process_single_case(args_tuple: tuple) -> str

    Objective:
        Worker function to load, threshold, noise-filter, and save a single scan prediction.

    Inputs:
        args_tuple (tuple): (pred_path, out_path, parent_ct_path, cat_codes, thresholds_by_cat,
            min_voxels_by_cat, default_threshold, default_min_voxels) where cat_codes maps each
            finding channel index of this scan to its category code ('1a'..'2h').

    Outputs:
        str: Success scan ID string or error message.
    """
    (pred_path, out_path, parent_ct_path, cat_codes, thresholds_by_cat,
     min_voxels_by_cat, default_threshold, default_min_voxels) = args_tuple
    try:
        data_ras, _, _ = load_nifti_ras(pred_path)

        # Determine if input is 3D or 4D
        if data_ras.ndim == 3:
            data_ras = data_ras[None] # (1, X, Y, Z)

        num_channels = data_ras.shape[0]
        cleaned_channels = []

        for c in range(num_channels):
            chan_data = data_ras[c]
            # Finding channel index c is NOT a category index: resolve this scan's category code first
            # so per-category thresholds land on the finding they were calibrated for.
            cat_code = cat_codes[c] if c < len(cat_codes) else None
            t_val = thresholds_by_cat.get(cat_code, default_threshold)
            v_min = min_voxels_by_cat.get(cat_code, default_min_voxels)

            # Binarize if continuous floating point probabilities
            if np.issubdtype(chan_data.dtype, np.floating):
                binary_chan = (chan_data >= t_val).astype(np.uint8)
            else:
                binary_chan = (chan_data > 0).astype(np.uint8)

            # Apply 3D connected-component noise filtering
            if v_min > 0:
                binary_chan = prune_small_components_3d(binary_chan, v_min)

            cleaned_channels.append(binary_chan)

        out_array = np.stack(cleaned_channels, axis=0) # (F, X, Y, Z)
        save_nifti(out_array, out_path, parent_ct_path, dtype=np.uint8)
        return pred_path.name
    except Exception as e:
        return f"ERROR: {pred_path.name}: {str(e)}"


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command line arguments for standalone prediction post-processing.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Decoupled CPU Prediction Post-Processor")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing input probability maps or binary prediction masks")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save post-processed binary prediction masks")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), 
                        help="Path to dataset.json metadata")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split being processed")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Uniform probability threshold for continuous maps (default: 0.5)")
    parser.add_argument("--thresholds_json", type=str, default=None,
                        help="Path to JSON file containing per-category threshold mapping or list")
    parser.add_argument("--min_volume", type=int, default=0, 
                        help="Uniform minimum 3D connected-component volume in voxels (default: 0 = disabled)")
    parser.add_argument("--min_volumes_json", type=str, default=None,
                        help="Path to JSON file containing per-category min volume mapping or list")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of parallel CPU worker processes (default: 4, max: 8)")
    return parser.parse_args()


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main execution entrypoint for multi-threaded prediction post-processing.

    Inputs:
        None

    Outputs:
        None
    """
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    # Load dataset entries to preserve scan order
    with open(args.dataset_json, 'r') as f:
        dataset_meta = json.load(f)
    
    entries = dataset_meta.get(args.split, [])
    if not entries:
        scan_files = sorted(list(input_dir.glob("*.nii.gz")))
    else:
        scan_files = [input_dir / entry['name'] for entry in entries if (input_dir / entry['name']).exists()]

    print(f"[INFO] Found {len(scan_files)} target prediction scans to post-process.")

    # Load per-category thresholds if specified. Mappings are keyed by category code ('1a'..'2h');
    # a bare list is accepted as an ordered override matching CATEGORY_MAP's declaration order.
    category_codes = list(CATEGORY_MAP.keys())

    def _load_per_category(json_path: str, default_value):
        """
        Signature:
            _load_per_category(json_path: str, default_value: float | int) -> dict

        Objective:
            Load a per-category override mapping keyed by category code, accepting either a
            {'1a': v, ...} dict or an ordered list aligned to CATEGORY_MAP.

        Inputs:
            json_path (str): Path to the JSON override file, or None to use the default for all categories.
            default_value (float | int): Value applied to categories absent from the override file.

        Outputs:
            dict: Mapping from category code ('1a'..'2h') to the resolved value.
        """
        resolved = {code: default_value for code in category_codes}
        if not json_path or not Path(json_path).exists():
            return resolved
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            for code, value in zip(category_codes, data):
                resolved[code] = value
        elif isinstance(data, dict):
            unknown = [k for k in data if k not in resolved]
            if unknown:
                print(f"[WARNING] Ignoring unknown category keys in {json_path}: {unknown}")
            for code in category_codes:
                if code in data:
                    resolved[code] = data[code]
        return resolved

    thresholds_by_cat = _load_per_category(args.thresholds_json, args.threshold)
    min_voxels_by_cat = _load_per_category(args.min_volumes_json, args.min_volume)

    # Map each scan's finding channel index to its category code, so per-category values are
    # applied to the matching finding rather than to whichever channel shares its ordinal position.
    categories_by_scan = {}
    for entry in entries:
        cats = entry.get("categories", {})
        categories_by_scan[entry["name"]] = [
            str(cats.get(str(i), "")) for i in range(len(cats))
        ]

    # Prepare worker tasks
    tasks = []
    for pred_path in scan_files:
        out_path = output_dir / pred_path.name
        parent_ct_path = RAW_IMAGES_DIR / pred_path.name
        cat_codes = categories_by_scan.get(pred_path.name, [])
        tasks.append((
            pred_path, out_path, parent_ct_path, cat_codes,
            thresholds_by_cat, min_voxels_by_cat, args.threshold, args.min_volume
        ))

    num_workers = min(args.num_workers, 8, os.cpu_count() or 4)
    print(f"[INFO] Executing post-processing across {num_workers} CPU workers...")
    distinct_t = sorted(set(thresholds_by_cat.values()))
    distinct_v = sorted(set(min_voxels_by_cat.values()))
    print(f"[INFO] Thresholds per category: {distinct_t} | Min volumes per category: {distinct_v}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_case, tasks), total=len(tasks), desc="Post-Processing"))

    errors = [r for r in results if r.startswith("ERROR")]
    if errors:
        print(f"[WARNING] {len(errors)} errors encountered during post-processing:")
        for err in errors[:5]:
            print(f"  {err}")
    else:
        print(f"[SUCCESS] Post-processed {len(results)} scans cleanly to {output_dir}")


if __name__ == "__main__":
    main()
