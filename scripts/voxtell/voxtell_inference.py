"""
VoxTell Batch Zero-Shot Inference Pipeline for ReXGroundingCT.

This script performs batch inference using the VoxTell model on preprocessed 
3D CT scans, generating 4D segmentation masks (F, H, W, D) guided by free-text 
prompts. It ensures strict positional alignment with the ground truth JSON 
to maintain compatibility with the official `rexrank_eval.py` script.

Input/Output Contract:
- Inputs: Configuration relies exclusively on environment variables loaded via 
  `.env` (MODEL_DIR, DATA_PREP_DIR, DATA_PRED_DIR, DATASET_JSON). Optional 
  `--split` CLI argument is available to target specific dataset partitions.
- Outputs: 4D NIfTI files saved in `DATA_PRED_DIR` preserving the original affine matrix.
"""


import os
import gc
import ctypes
import json
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

# Load environment variables from .env file
load_dotenv(override=True)

# 1. Strictly isolate GPU (Node policy) MUST happen before VoxTell/nnU-Net imports
# Note: CUDA_VISIBLE_DEVICES must be explicitly set in the environment or .env

import sys
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from scripts.voxtell.prompt_normalizer import clean_finding_prompt

def main():
    # Parse CLI arguments for split selection
    parser = argparse.ArgumentParser(description="VoxTell Batch Zero-Shot Inference")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate (train, val, test)")
    parser.add_argument("--dataset_json", type=str, default=None,
                        help="Path to dataset.json (overrides .env)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for predictions (overrides .env)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to custom checkpoint file (.pth) to load weights from")
    parser.add_argument("--use_teacher", action="store_true",
                        help="Load teacher_state_dict instead of student_state_dict if custom checkpoint is specified")
    parser.add_argument("--tile_step_size", type=float, default=0.5,
                        help="Step size for sliding window inference (default: 0.5 = 50% overlap, increase to speed up)")
    parser.add_argument("--normalize_prompts", action="store_true", default=True,
                        help="Normalize free-text prompts to strip non-diagnostic clinical modifiers (default: True)")
    parser.add_argument("--no_normalize_prompts", action="store_false", dest="normalize_prompts",
                        help="Disable prompt normalization")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing dataset entries")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing dataset entries (exclusive)")
    args = parser.parse_args()


    # Inject paths from .env file
    download_dir = os.environ["MODEL_DIR"]
    img_raw_dir = os.environ["IMG_RAW_DIR"]
    output_dir = args.output_dir or os.environ.get("TMP_PRED_DIR") or os.environ["DATA_PRED_DIR"]
    dataset_json = args.dataset_json or os.environ["DATASET_JSON"]

    # Security validation for critical environment variables
    if not all([download_dir, img_raw_dir, output_dir, dataset_json]):
        raise ValueError("Error: Missing environment variables in .env (MODEL_DIR, IMG_RAW_DIR, DATA_PRED_DIR/TMP_PRED_DIR, DATASET_JSON).")

    # Prepare directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    # Device Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Resolve the base models folder to keep folders clean
    models_root = os.path.dirname(download_dir) if download_dir.endswith("voxtell_v1.0") else download_dir
    
    # Target recommended model version v1.1
    model_name = "voxtell_v1.1" 
    
    # Get Weights from Hugging Face
    snapshot_download(
        repo_id="mrokuss/VoxTell", 
        allow_patterns=[f"{model_name}/*", "*.json"], 
        local_dir=models_root
    )
    voxtell_weights_dir = os.path.join(models_root, model_name)

    # Initialize Predictor
    predictor = VoxTellPredictor(model_dir=voxtell_weights_dir, device=device)
    predictor.tile_step_size = args.tile_step_size
    
    # Load custom checkpoint weights if specified
    if args.checkpoint:
        print(f"Loading custom checkpoint weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        
        # Determine which state dict to use
        if 'student_state_dict' in checkpoint or 'teacher_state_dict' in checkpoint:
            key = 'teacher_state_dict' if args.use_teacher else 'student_state_dict'
            state_dict = checkpoint[key]
            print(f"Loaded '{key}' from checkpoint.")
        elif 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
            print("Loaded 'network_weights' from checkpoint.")
        else:
            state_dict = checkpoint
            print("Loaded raw state_dict from checkpoint.")
            
        predictor.network.load_state_dict(state_dict)
    
    # Data Reading and Reorientation (Critical for VoxTell)
    reader = NibabelIOWithReorient()

    # Load Ground Truth metadata
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get(args.split, [])
    if not entries:
        print(f"[WARNING] No cases found for split '{args.split}'. Check dataset.json structure.")
        return

    # Apply chunking bounds
    end_idx = args.end_idx if args.end_idx is not None else len(entries)
    entries = entries[args.start_idx:end_idx]
    
    missing_files_count = 0

    # Batch Inference Loop
    for entry in tqdm(entries, desc=f"Evaluating {args.split} Scans [{args.start_idx}:{end_idx}]"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue
            
        out_path = os.path.join(output_dir, f"{scan_id}.nii.gz")
        if os.path.exists(out_path):
            tqdm.write(f"[INFO] Prediction already exists for {scan_id}. Skipping.")
            continue
            
        nifti_path = os.path.join(img_raw_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(nifti_path):
            tqdm.write(f"[WARNING] Raw file not found: {nifti_path}. Skipping.")
            missing_files_count += 1
            continue
            
        # Load and reorient volume to RAS
        img, img_properties = reader.read_images([nifti_path])
        
        findings = entry.get('findings', {})
        if not findings:
            continue
            
        # Extract prompts, handling both dictionary and list formats (for testing compatibility)
        if isinstance(findings, dict):
            # Sort keys numerically to ensure strict alignment with the channel order
            sorted_keys = sorted(findings.keys(), key=int)
            text_prompts = []
            for k in sorted_keys:
                val = findings[k]
                if isinstance(val, dict):
                    text_prompts.append(val.get('text', ''))
                else:
                    text_prompts.append(str(val))
        else:
            text_prompts = [f['text'] if isinstance(f, dict) else f for f in findings]
        
        if args.normalize_prompts:
            text_prompts = [clean_finding_prompt(p) for p in text_prompts]
        
        # Inference
        with torch.no_grad():
            voxtell_seg = predictor.predict_single_image(img, text_prompts) # shape: (F, Z, Y, X)

            
        # Reorient 4D prediction back to original image space
        # 1. Transpose from (F, Z, Y, X) to (X, Y, Z, F) in RAS space
        pred_xyzf = np.transpose(voxtell_seg, (3, 2, 1, 0))
        
        # 2. Create NIfTI in RAS space using reoriented_affine
        reoriented_affine = img_properties['nibabel_stuff']['reoriented_affine']
        pred_nib = nib.Nifti1Image(pred_xyzf, reoriented_affine)
        
        # 3. Get the transformation to go from RAS back to original orientation
        original_affine = img_properties['nibabel_stuff']['original_affine']
        img_ornt = io_orientation(original_affine)
        ras_ornt = axcodes2ornt("RAS")
        from_canonical = ornt_transform(ras_ornt, img_ornt)
        
        # 4. Apply back-reorientation
        pred_nib_back = pred_nib.as_reoriented(from_canonical)
        
        # 5. Extract data and transpose back to (F, X, Y, Z) to preserve original shape contract
        # Use dataobj instead of get_fdata() to avoid a massive float64 memory spike
        pred_back_data = np.asanyarray(pred_nib_back.dataobj).astype(np.uint8) # shape: (X, Y, Z, F)
        pred_back_fxyz = np.transpose(pred_back_data, (3, 0, 1, 2)) # shape: (F, X, Y, Z)
        
        # Save prediction with the original raw image affine
        out_nii = nib.Nifti1Image(pred_back_fxyz, original_affine)
        out_path = os.path.join(output_dir, f"{scan_id}.nii.gz")
        nib.save(out_nii, out_path)
        
        # Explicit memory cleanup to prevent OS OOM killer
        del img, voxtell_seg, pred_xyzf, pred_nib, pred_nib_back, pred_back_data, pred_back_fxyz, out_nii
        gc.collect()
        # Force glibc to return freed memory to the OS (solves Python memory fragmentation)
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    if missing_files_count > 0:
        print(f"\n[INFO] Inference completed. Skipped {missing_files_count} missing files.")

if __name__ == "__main__":
    main()