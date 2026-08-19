"""
===============================================================================
SCRIPT:         Create Mini Dataset for Fast Diagnostics
PHASE:          Shared Analysis & Diagnostics
LOCATION:       scripts/analysis/create_mini_dataset.py
OBJECTIVE:      Extract a balanced, representative subset of 20 training scans 
                and 5 validation scans from dataset.json covering all 14 pathology
                categories for rapid numerical stability and diagnostic testing.
USAGE:          python scripts/analysis/create_mini_dataset.py
===============================================================================
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import DATASET_JSON, DATASET_MINI_JSON, CATEGORY_MAP


def sample_representative_scans(entries: list[dict], target_count: int = 20) -> list[dict]:
    """
    Signature:
        sample_representative_scans(entries: list[dict], target_count: int) -> list[dict]

    Objective:
        Sample a diverse subset of scans prioritizing coverage across all 14 pathology categories.

    Inputs:
        entries (list[dict]): List of scan metadata dictionaries from dataset.json.
        target_count (int): Desired number of scans to sample. Default 20.

    Outputs:
        list[dict]: List of sampled scan metadata dictionaries.
    """
    category_to_entries = defaultdict(list)
    for entry in entries:
        categories = entry.get('categories', {})
        for cat in categories.values():
            category_to_entries[cat].append(entry)
            
    selected_scans = []
    selected_names = set()
    
    # 1. First pass: ensure at least one scan per known category
    for cat in sorted(CATEGORY_MAP.keys()):
        matching = category_to_entries.get(cat, [])
        for item in matching:
            if item['name'] not in selected_names:
                selected_scans.append(item)
                selected_names.add(item['name'])
                break
        if len(selected_scans) >= target_count:
            break
            
    # 2. Second pass: fill remaining quota with diverse multi-finding scans
    if len(selected_scans) < target_count:
        sorted_by_findings = sorted(
            entries,
            key=lambda x: len(x.get('findings', {})),
            reverse=True
        )
        for item in sorted_by_findings:
            if item['name'] not in selected_names:
                selected_scans.append(item)
                selected_names.add(item['name'])
            if len(selected_scans) >= target_count:
                break
                
    return selected_scans[:target_count]


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Parse arguments and generate dataset_mini.json in the global data directory.

    Inputs:
        None

    Outputs:
        None
    """
    parser = argparse.ArgumentParser(description="Create mini dataset for rapid diagnostics")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to full dataset.json")
    parser.add_argument("--num_train", type=int, default=20, help="Number of training scans (default: 20)")
    parser.add_argument("--num_val", type=int, default=5, help="Number of validation scans (default: 5)")
    parser.add_argument("--output", type=str, default=str(DATASET_MINI_JSON), help="Output path")
    args = parser.parse_args()

    input_path = Path(args.dataset_json)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing dataset.json at {input_path}")

    with open(input_path, 'r') as f:
        full_data = json.load(f)

    train_entries = full_data.get('train', [])
    val_entries = full_data.get('val', [])
    test_entries = full_data.get('test', [])

    sampled_train = sample_representative_scans(train_entries, target_count=args.num_train)
    sampled_val = sample_representative_scans(val_entries, target_count=args.num_val)
    sampled_test = test_entries[:args.num_val]

    mini_data = {
        "train": sampled_train,
        "val": sampled_val,
        "test": sampled_test
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(mini_data, f, indent=2)

    print(f"Successfully generated mini dataset at: {output_path}")
    print(f"  Train scans: {len(sampled_train)} / {len(train_entries)}")
    print(f"  Val scans:   {len(sampled_val)} / {len(val_entries)}")
    print(f"  Test scans:  {len(sampled_test)} / {len(test_entries)}")


if __name__ == "__main__":
    main()
