import os
import json
import argparse
import torch
import nibabel as nib
from dotenv import load_dotenv
from monai.transforms import LoadImaged, EnsureChannelFirstd

# Load environment variables from .env
load_dotenv(override=True)

DATASET_JSON = os.environ["DATASET_JSON"]
IMG_DIR = os.environ["IMG_RAW_DIR"]
SEG_DIR = os.environ["SEG_RAW_DIR"]

def main():
    parser = argparse.ArgumentParser(description="Dataset Dimensionality & Metadata Diagnostic Utility")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                        help="Dataset split to analyze (default: val)")
    parser.add_argument("--num_cases", type=int, default=5,
                        help="Number of cases to inspect (default: 5)")
    parser.add_argument("--case_name", type=str, default=None,
                        help="Specific case filename to inspect (e.g. train_13082_a_1.nii.gz)")
    args = parser.parse_args()

    print("=" * 70)
    print("          DATASET DIMENSIONALITY & METADATA DIAGNOSTIC UTILITY")
    print("=" * 70)
    print(f"DATASET_JSON: {DATASET_JSON}")
    print(f"IMG_RAW_DIR : {IMG_DIR}")
    print(f"SEG_RAW_DIR : {SEG_DIR}")
    print("=" * 70)

    if not DATASET_JSON or not IMG_DIR or not SEG_DIR:
        print("[ERROR] Environment variables not correctly set. Please check your .env file.")
        return

    if not os.path.exists(DATASET_JSON):
        print(f"[ERROR] Metadata file not found at: {DATASET_JSON}")
        return

    with open(DATASET_JSON, 'r') as f:
        metadata = json.load(f)

    cases = metadata.get(args.split, [])
    if not cases:
        print(f"[WARNING] No cases found for split: '{args.split}'")
        return

    # Filter by specific case if requested
    if args.case_name:
        cases = [c for c in cases if c.get("name") == args.case_name]
        if not cases:
            print(f"[ERROR] Case '{args.case_name}' not found in split '{args.split}'")
            return
    else:
        cases = cases[:args.num_cases]

    print(f"Inspecting {len(cases)} case(s) from split '{args.split}':\n")

    loader = LoadImaged(keys=["image", "label"], reader="NibabelReader")
    ensure_channel_first_img = EnsureChannelFirstd(keys=["image"])
    # According to our diagnostics, the segmentation findings channel is loaded first (dim 0)
    ensure_channel_first_seg = EnsureChannelFirstd(keys=["label"], channel_dim=0)

    for case in cases:
        name = case.get("name")
        print(f"--> [CASE] {name}")
        
        img_path = os.path.join(IMG_DIR, name)
        seg_path = os.path.join(SEG_DIR, name)
        
        if not os.path.exists(img_path) or not os.path.exists(seg_path):
            print(f"    [WARNING] Missing raw files. Image exists: {os.path.exists(img_path)}, Seg exists: {os.path.exists(seg_path)}")
            print("-" * 70)
            continue
            
        # 1. Nibabel inspection (raw headers)
        raw_img = nib.load(img_path)
        raw_seg = nib.load(seg_path)
        
        print(f"    [Raw Nibabel] Image Shape      : {raw_img.shape}")
        print(f"    [Raw Nibabel] Image Spacing    : {raw_img.header.get_zooms()[:3]}")
        print(f"    [Raw Nibabel] Image Affine Diag: {raw_img.affine.diagonal().tolist()}")
        print(f"    [Raw Nibabel] Seg Shape        : {raw_seg.shape}")
        print(f"    [Raw Nibabel] Seg Spacing      : {raw_seg.header.get_zooms()[:3]}")
        print(f"    [Raw Nibabel] Seg Affine Diag  : {raw_seg.affine.diagonal().tolist()}")
        
        # 2. MONAI Loading inspection
        try:
            data = loader({"image": img_path, "label": seg_path})
            print(f"    [MONAI Loaded] Image Shape     : {data['image'].shape} ({type(data['image']).__name__})")
            print(f"    [MONAI Loaded] Seg Shape       : {data['label'].shape} ({type(data['label']).__name__})")
            
            # Ensure channel first
            data = ensure_channel_first_img(data)
            data = ensure_channel_first_seg(data)
            print(f"    [MONAI ChannelFirst] Image     : {data['image'].shape}")
            print(f"    [MONAI ChannelFirst] Seg       : {data['label'].shape}")
            
            # Warn if segmentations lack spacing metadata
            affine_diag = raw_seg.affine.diagonal().tolist()[:3]
            if all(abs(v - 1.0) < 1e-5 for v in affine_diag):
                print("    [ALERT] Segmentation has default IDENTITY affine. It lacks correct voxel spacing!")
                print("            This will cause MONAI Spacingd to resample incorrectly if not aligned using CopyAffined!")
                
        except Exception as e:
            print(f"    [ERROR] Failed loading or processing: {str(e)}")
            
        print("-" * 70)

if __name__ == "__main__":
    main()
