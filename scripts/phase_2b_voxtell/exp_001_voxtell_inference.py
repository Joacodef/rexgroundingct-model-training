"""
===============================================================================
SCRIPT:         VoxTell Batch Zero-Shot Baseline Inference Pipeline
PHASE:          Phase 2B — VoxTell Zero-Shot Baseline & Preprocessing Audit
LOCATION:       scripts/phase_2b_voxtell/exp_001_voxtell_inference.py
OBJECTIVE:      Performs batch inference using pre-trained VoxTell v1.1 model on 
                3D CT scans, generating 4D segmentation masks (F, X, Y, Z) 
                guided by free-text radiology prompts.
USAGE:          python scripts/phase_2b_voxtell/exp_001_voxtell_inference.py --split val
===============================================================================
"""


import os
import gc
import ctypes
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables FIRST before PyTorch CUDA initialization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv(override=False)

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Strictly isolate GPU before VoxTell imports
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from voxtell.inference.predictor import VoxTellPredictor
from scripts.common.orientation import load_nifti_ras, save_nifti





def main():
    """
    Main CLI entry point for executing VoxTell v1.1 batch zero-shot inference 
    canonicalized to RAS space via Centralized Spatial Engine (orientation.py).
    """
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
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing dataset entries")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing dataset entries (exclusive)")
    args = parser.parse_args()

    # Inject paths from .env file or fallback config
    download_dir = os.environ.get("MODEL_DIR", "models/voxtell")
    img_raw_dir = os.environ.get("IMG_RAW_DIR", "../data/raw/images")
    output_dir = args.output_dir or os.environ.get("TMP_PRED_DIR") or os.path.join(os.environ.get("DATA_PRED_DIR", "../data/predictions"), "phase_2b_voxtell")
    dataset_json = args.dataset_json or os.environ.get("DATASET_JSON", "../data/dataset.json")

    # Security validation for critical environment variables
    if not all([download_dir, img_raw_dir, output_dir, dataset_json]):
        raise ValueError("Error: Missing required environment variables (MODEL_DIR, IMG_RAW_DIR, DATA_PRED_DIR/TMP_PRED_DIR, DATASET_JSON).")

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
    
    # Offload sliding window result accumulators to CPU memory to prevent CUDA OOM
    predictor.perform_everything_on_device = False
    
    # Memory optimization: Keep text backbone on CPU to prevent CUDA OOM
    predictor.text_backbone = predictor.text_backbone.to("cpu")

    def embed_text_prompts_cpu_safe(text_prompts):
        """
        Signature:
            embed_text_prompts_cpu_safe(text_prompts: List[str] | str) -> torch.Tensor

        Objective:
            Computes text embeddings using CPU-offloaded Qwen2-0.5B text backbone to prevent
            CUDA VRAM OOM while returning final prompt tensor embeddings on GPU device.

        Args:
            text_prompts (List[str] | str): Input free-text prompt strings.

        Returns:
            torch.Tensor: Normalized prompt embeddings tensor of shape (1, N_prompts, D_embed) on predictor device.
        """
        from voxtell.utils.text_embedding import wrap_with_instruction, last_token_pool
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)
        wrapped = wrap_with_instruction(text_prompts)
        tokens = predictor.tokenizer(wrapped, padding=True, truncation=True, max_length=predictor.max_text_length, return_tensors="pt")
        with torch.no_grad():
            text_embed = predictor.text_backbone(**tokens)
            embeddings = last_token_pool(text_embed.last_hidden_state, tokens['attention_mask'])
            embeddings = embeddings.view(1, n_prompts, -1)
        return embeddings.to(predictor.device)

    predictor.embed_text_prompts = embed_text_prompts_cpu_safe
    
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
        torch.cuda.empty_cache()
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
            
        # Step 1: Load CT image in canonical NIfTI RAS physical coordinate space
        # load_nifti_ras() standardizes DICOM/NIfTI headers so axis 0=Right, 1=Anterior, 2=Superior.
        # Image shape: (X, Y, Z)
        img_ras, ras_nii, raw_axcodes = load_nifti_ras(nifti_path)
        
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

        # Step 2: Transpose image array from NIfTI RAS indexing (X, Y, Z) to nnUNet/VoxTell memory layout (Z, Y, X)
        # CRITICAL TECHNICAL DIRECTIVE: VoxTell's 3D Swin UNet model was pre-trained on nnUNet v2 pipelines.
        # nnUNet's NibabelIOWithReorient applies .transpose((2, 1, 0)) to place the axial depth/slice axis (Z)
        # at index 0 (depth-first C-contiguous ordering). Without this transposition, 3D convolutions receive
        # rotated sagittal cross-sections, destroying spatial feature matching and collapsing logits to near-zero.
        img_nnunet = img_ras.transpose((2, 1, 0))  # Shape: (Z, Y, X)

        # Step 3: Preprocess and run sliding window inference on nnUNet-ordered image array (Z, Y, X)
        data_tensor, bbox, orig_shape = predictor.preprocess(img_nnunet)
        embeddings = predictor.embed_text_prompts(text_prompts)

        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data_tensor, embeddings).cpu()
            probs_nnunet_crop = torch.sigmoid(logits.float()).numpy()

        # Step 4: Revert cropping by inserting cropped probabilities back into original 3D volume shape (F, Z_orig, Y_orig, X_orig)
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        probs_nnunet_full = np.zeros([probs_nnunet_crop.shape[0], *orig_shape], dtype=np.float32)
        probs_nnunet_full = insert_crop_into_image(probs_nnunet_full, probs_nnunet_crop, bbox)

        # Step 5: Untranspose predicted binary mask from (F, Z, Y, X) back to canonical NIfTI RAS space (F, X, Y, Z)
        # This restores exact 3D spatial alignment with ground-truth RAS segmentation masks.
        voxtell_seg = ((probs_nnunet_full.transpose((0, 3, 2, 1))) > 0.5).astype(np.uint8)  # shape: (F, X, Y, Z)

        # Step 6: Save prediction anchored to parent CT scan header via Centralized Spatial Engine
        save_nifti(voxtell_seg, out_path, nifti_path)
        
        # Explicit memory cleanup to prevent OS OOM killer
        del img_ras, ras_nii, voxtell_seg
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    if missing_files_count > 0:
        print(f"\n[INFO] Inference completed. Skipped {missing_files_count} missing files.")


if __name__ == "__main__":
    main()