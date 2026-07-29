import os
import json
import tempfile
import torch
import numpy as np
import nibabel as nib

from dotenv import load_dotenv
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

def compute_dice(a, b):
    a_b = a > 0
    b_b = b > 0
    i = (a_b & b_b).sum()
    u = a_b.sum() + b_b.sum()
    return 2.0 * i / u if u > 0 else (1.0 if i == 0 else 0.0)

def main():
    load_dotenv(override=True)
    gt_dir = os.environ["SEG_RAW_DIR"]
    img_dir = os.environ["IMG_RAW_DIR"]
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = os.environ["MODEL_DIR"]

    with open(dataset_json) as f:
        meta = json.load(f)

    # Let's take case 0 (train_13082_a_1) from paper val split (where Dice was 0.21)
    case_0 = meta["val"][0]
    scan_id = case_0["name"].replace(".nii.gz", "")
    gt_path = os.path.join(gt_dir, f"{scan_id}.nii.gz")
    img_path = os.path.join(img_dir, f"{scan_id}.nii.gz")

    print(f"=== TESTING OFFICIAL VOXTELL PIPELINE ON {scan_id} ===")
    
    # 1. Read CT using official reader
    reader = NibabelIOWithReorient()
    img_arr, props = reader.read_images([img_path])
    print("Official read_images img_arr shape:", img_arr.shape)

    # 2. Extract prompts
    findings = case_0["findings"]
    if isinstance(findings, dict):
        sorted_keys = sorted(findings.keys(), key=int)
        raw_prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
    else:
        raw_prompts = [f.get("text", "") if isinstance(f, dict) else str(f) for f in findings]
    print("Raw Prompts:", raw_prompts)


    # 3. Load official Predictor
    predictor = VoxTellPredictor(model_dir=model_dir, device=torch.device("cuda:0"))


    # 4. Predict single image using official method
    raw_seg = predictor.predict_single_image(img_arr, raw_prompts) # shape: (F, Z, Y, X)
    print("Official predict_single_image raw_seg shape:", raw_seg.shape)

    # Load GT
    gt_nii = nib.load(gt_path)
    gt_data = np.asanyarray(gt_nii.dataobj) # shape: (4, 512, 512, 205) or (512, 512, 205, 4)
    if gt_data.ndim == 4 and gt_data.shape[-1] < np.min(gt_data.shape[:-1]):
        gt_data = np.moveaxis(gt_data, -1, 0)
    print("GT array shape:", gt_data.shape)

    # Let's test different axis transposes of raw_seg against gt_data!
    print("\n--- DICE METRICS UNDER DIFFERENT AXIS PERMUTATIONS ---")
    
    # Permutation 1: (F, Z, Y, X) directly against gt_data (F, X, Y, Z)
    if raw_seg.shape == gt_data.shape:
        print("Direct (F, Z, Y, X) vs GT (F, X, Y, Z):", [compute_dice(raw_seg[c], gt_data[c]) for c in range(len(raw_prompts))])

    # Permutation 2: Transpose (F, Z, Y, X) -> (F, X, Y, Z) via (0, 3, 2, 1)
    seg_fxyz_1 = np.transpose(raw_seg, (0, 3, 2, 1))
    if seg_fxyz_1.shape == gt_data.shape:
        print("Transpose (0, 3, 2, 1) [F, X, Y, Z] vs GT:", [compute_dice(seg_fxyz_1[c], gt_data[c]) for c in range(len(raw_prompts))])

    # Permutation 3: Transpose (F, Z, Y, X) -> (F, X, Y, Z) via (0, 2, 1, 3)
    seg_fxyz_2 = np.transpose(raw_seg, (0, 2, 1, 3))
    if seg_fxyz_2.shape == gt_data.shape:
        print("Transpose (0, 2, 1, 3) vs GT:", [compute_dice(seg_fxyz_2[c], gt_data[c]) for c in range(len(raw_prompts))])

    # Permutation 4: Apply official write_seg back-reorientation
    out_tmp = os.path.join(tempfile.gettempdir(), f"test_{scan_id}_official.nii.gz")
    # write_seg expects 3D or 4D?
    # Let's test writing channel 0
    reader.write_seg(raw_seg[0], out_tmp, props)
    written_nii = nib.load(out_tmp)
    written_data = np.asanyarray(written_nii.dataobj)
    print("Official write_seg written_data shape:", written_data.shape)
    print("Official write_seg channel 0 vs GT channel 0:", compute_dice(written_data, gt_data[0]))

if __name__ == "__main__":
    main()
