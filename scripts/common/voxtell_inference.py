"""
===============================================================================
MODULE:         Universal VoxTell Multi-GPU Batch Inference Engine
LOCATION:       scripts/common/voxtell_inference.py
OBJECTIVE:      Server-agnostic, multi-GPU batch inference engine for pre-trained
                and fine-tuned VoxTell models on 3D CT scans. Generates 4D 
                segmentation masks (F, X, Y, Z) canonicalized to RAS coordinate space.
USAGE:          Single-GPU: python scripts/common/voxtell_inference.py --split val
                Multi-GPU:  torchrun --nproc_per_node=N scripts/common/voxtell_inference.py --split val
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
import torch.distributed as dist
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
from scripts.phase_3_voxtell_training.common.distributed import init_distributed, cleanup_distributed
from scripts.config import TEXT_CACHE_DIR
from acvl_utils.cropping_and_padding.padding import pad_nd_image


def main():
    """
    Main CLI entry point for executing VoxTell batch inference canonicalized 
    to RAS space via Centralized Spatial Engine (orientation.py).
    Supports server-agnostic multi-GPU execution (torchrun) and single-GPU fallback.
    """
    # Initialize Server-Agnostic Distributed Environment
    is_distributed, rank, local_rank, world_size, device_str = init_distributed()
    device = torch.device(device_str)

    # Parse CLI arguments for split selection
    parser = argparse.ArgumentParser(description="Universal VoxTell Multi-GPU Batch Inference Engine")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], 
                        help="Dataset split to evaluate (train, val, test)")
    parser.add_argument("--dataset_json", type=str, default=None,
                        help="Path to dataset.json (overrides .env)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for predictions (overrides .env)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to custom checkpoint file (.pt/.pth) to load weights from")
    parser.add_argument("--use_teacher", action="store_true",
                        help="Load teacher_state_dict instead of student_state_dict if custom checkpoint is specified")
    parser.add_argument("--tile_step_size", type=float, default=0.5,
                        help="Step size for sliding window inference (default: 0.5 = 50%% overlap, increase to speed up)")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for processing dataset entries (single-GPU mode)")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for processing dataset entries (single-GPU mode, exclusive)")
    parser.add_argument("--save_raw_probs", action="store_true",
                        help="Save continuous float32 probability maps instead of binarized uint8 masks for offline threshold tuning")
    args = parser.parse_args()

    # Inject paths from .env file or fallback config
    download_dir = os.environ.get("MODEL_DIR", "models/voxtell")
    img_raw_dir = os.environ.get("IMG_RAW_DIR", "../data/raw/images")
    output_dir = args.output_dir or os.environ.get("TMP_PRED_DIR") or os.path.join(os.environ.get("DATA_PRED_DIR", "../data/predictions"), "phase_2b_voxtell")
    dataset_json = args.dataset_json or os.environ.get("DATASET_JSON", "../data/dataset.json")

    # Security validation for critical environment variables
    if not all([download_dir, img_raw_dir, output_dir, dataset_json]):
        raise ValueError("Error: Missing required environment variables (MODEL_DIR, IMG_RAW_DIR, DATA_PRED_DIR/TMP_PRED_DIR, DATASET_JSON).")

    # Prepare directories (Rank 0 creates directories)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(download_dir, exist_ok=True)

    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
    
    # Resolve the base models folder to keep folders clean
    models_root = os.path.dirname(download_dir) if download_dir.endswith("voxtell_v1.0") else download_dir
    
    # Target recommended model version v1.1
    model_name = "voxtell_v1.1" 
    
    # Get Weights from Hugging Face (Rank 0 checks/downloads first)
    if rank == 0:
        snapshot_download(
            repo_id="mrokuss/VoxTell", 
            allow_patterns=[f"{model_name}/*", "*.json"], 
            local_dir=models_root
        )
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        
    voxtell_weights_dir = os.path.join(models_root, model_name)

    # Initialize Predictor on isolated device
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
            Computes text embeddings using CPU-offloaded Qwen3-Embedding-4B text backbone to prevent
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
        return embeddings.to(device=predictor.device, dtype=torch.float32)

    predictor.embed_text_prompts = embed_text_prompts_cpu_safe

    # Ensure robust, numerical-stable sliding window without fp16 underflow/overflow NaNs
    @torch.inference_mode()
    def safe_predict_sliding_window_return_logits(input_image: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Signature:
            safe_predict_sliding_window_return_logits(input_image: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor

        Objective:
            Executes sliding-window inference with autocast disabled on CUDA to guarantee numerical stability
            and prevent float16 underflow/overflow NaNs during Gaussian weighting.

        Inputs:
            input_image (torch.Tensor): 4D input volume tensor (C, Z, Y, X).
            text_embeddings (torch.Tensor): Prompt text embedding tensor (1, N_prompts, D_embed).

        Outputs:
            torch.Tensor: Predicted logits tensor across full volume.
        """
        predictor.network = predictor.network.to(predictor.device)
        data, slicer_revert_padding = pad_nd_image(input_image, predictor.patch_size, 'constant', {'value': 0}, True, None)
        slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
        with torch.autocast('cuda', enabled=False) if predictor.device.type == 'cuda' else torch.no_grad():
            predicted_logits = predictor._internal_predict_sliding_window_return_logits(
                data, text_embeddings, slicers, predictor.perform_everything_on_device
            )
        return predicted_logits[(slice(None), *slicer_revert_padding[1:])]

    predictor.predict_sliding_window_return_logits = safe_predict_sliding_window_return_logits
    
    # Load custom checkpoint weights if specified
    if args.checkpoint:
        if rank == 0:
            print(f"Loading custom checkpoint weights from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        
        # Determine which state dict to use
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            if rank == 0:
                print("Loaded 'model_state_dict' from checkpoint.")
        elif 'student_state_dict' in checkpoint or 'teacher_state_dict' in checkpoint:
            key = 'teacher_state_dict' if args.use_teacher else 'student_state_dict'
            state_dict = checkpoint[key]
            if rank == 0:
                print(f"Loaded '{key}' from checkpoint.")
        elif 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
            if rank == 0:
                print("Loaded 'network_weights' from checkpoint.")
        else:
            state_dict = checkpoint
            if rank == 0:
                print("Loaded raw state_dict from checkpoint.")
            
        predictor.network.load_state_dict(state_dict)

    predictor.network = predictor.network.to(device)
    predictor.network.eval()

    # Load Ground Truth metadata
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    entries = metadata.get(args.split, [])
    if not entries:
        if rank == 0:
            print(f"[WARNING] No cases found for split '{args.split}'. Check dataset.json structure.")
        return

    # Apply server-agnostic distributed sharding or CLI slicing bounds
    if is_distributed and world_size > 1:
        # Shard entries across ranks in strided order
        entries = entries[rank::world_size]
        desc_str = f"[Rank {rank}/{world_size}] Evaluating {args.split} ({len(entries)} scans)"
    else:
        end_idx = args.end_idx if args.end_idx is not None else len(entries)
        entries = entries[args.start_idx:end_idx]
        desc_str = f"Evaluating {args.split} Scans [{args.start_idx}:{end_idx}]"
    
    missing_files_count = 0

    # Batch Inference Loop
    for entry in tqdm(entries, desc=desc_str, position=rank if is_distributed and world_size > 1 else 0):
        torch.cuda.empty_cache()
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue
            
        out_path = os.path.join(output_dir, f"{scan_id}.nii.gz")
        if os.path.exists(out_path):
            tqdm.write(f"[Rank {rank}] [INFO] Prediction already exists for {scan_id}. Skipping.")
            continue
            
        nifti_path = os.path.join(img_raw_dir, f"{scan_id}.nii.gz")
        
        if not os.path.exists(nifti_path):
            tqdm.write(f"[Rank {rank}] [WARNING] Raw file not found: {nifti_path}. Skipping.")
            missing_files_count += 1
            continue
            
        # Step 1: Load CT image in canonical NIfTI RAS physical coordinate space
        # Image shape: (X, Y, Z)
        img_ras, ras_nii, raw_axcodes = load_nifti_ras(nifti_path)
        
        findings = entry.get('findings', {})
        if not findings:
            continue
            
        # Extract prompts, handling both dictionary and list formats
        if isinstance(findings, dict):
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
        img_nnunet = img_ras.transpose((2, 1, 0))  # Shape: (Z, Y, X)

        # Step 3: Preprocess and run sliding window inference on nnUNet-ordered image array (Z, Y, X)
        data_tensor, bbox, orig_shape = predictor.preprocess(img_nnunet)
        
        cached_emb_path = TEXT_CACHE_DIR / f"{scan_id}.pt"
        if cached_emb_path.exists():
            embeddings = torch.load(cached_emb_path, map_location=device).unsqueeze(0).to(torch.float32)
        else:
            embeddings = predictor.embed_text_prompts(text_prompts).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            logits = predictor.predict_sliding_window_return_logits(data_tensor, embeddings).cpu()
            probs_nnunet_crop = torch.sigmoid(logits.float()).numpy()

        # Step 4: Revert cropping by inserting cropped probabilities back into original 3D volume shape (F, Z_orig, Y_orig, X_orig)
        from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
        probs_nnunet_full = np.zeros([probs_nnunet_crop.shape[0], *orig_shape], dtype=np.float32)
        probs_nnunet_full = insert_crop_into_image(probs_nnunet_full, probs_nnunet_crop, bbox)

        # Step 5: Untranspose predicted probabilities from (F, Z, Y, X) back to canonical NIfTI RAS space (F, X, Y, Z)
        probs_ras = probs_nnunet_full.transpose((0, 3, 2, 1))  # shape: (F, X, Y, Z)

        # Step 6: Save continuous probabilities or binarized mask anchored to parent CT scan header
        if args.save_raw_probs:
            save_nifti(probs_ras, out_path, nifti_path, dtype=np.float32)
        else:
            voxtell_seg = (probs_ras > 0.5).astype(np.uint8)
            save_nifti(voxtell_seg, out_path, nifti_path, dtype=np.uint8)
        
        # Explicit memory cleanup
        del img_ras, ras_nii, probs_nnunet_crop, probs_nnunet_full, probs_ras
        gc.collect()

    # Synchronize all ranks at completion
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        cleanup_distributed()

    if rank == 0:
        if missing_files_count > 0:
            print(f"\n[INFO] Inference completed. Skipped {missing_files_count} missing files.")
        else:
            print("\n[INFO] Batch inference completed successfully across all scans.")


if __name__ == "__main__":
    main()
