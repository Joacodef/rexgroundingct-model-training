"""
===============================================================================
TEST SCRIPT:    Exp 001 Diagnostic Harness for Numerical Stability
LOCATION:       tests/test_exp001_diagnostics.py
OBJECTIVE:      Execute a fast 2-epoch diagnostic run of Exp 001 on dataset_mini.json,
                logging exact per-batch loss components, gradient norms, and weight finiteness
                to isolate the root cause of NaNs.
USAGE:          CUDA_VISIBLE_DEVICES=1 python tests/test_exp001_diagnostics.py
===============================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceLoss

from scripts.config import (
    DATASET_JSON, DATASET_MINI_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    TEXT_CACHE_DIR, TMP_PREP_DIR, LOGS_DIR, MODEL_DIR
)
from scripts.phase_3_voxtell_training.exp_001_naive_finetuning import (
    ReXDataset, load_voxtell_model
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("test_exp001_diagnostics")


def run_diagnostics(dataset_json: str, epochs: int = 2, device: str = "cuda:0") -> dict:
    """
    Signature:
        run_diagnostics(dataset_json: str, epochs: int, device: str) -> dict

    Objective:
        Execute diagnostic training loop with detailed telemetry on losses, gradients, and model weights.

    Inputs:
        dataset_json (str): Path to mini dataset.json.
        epochs (int): Number of epochs to run. Default 2.
        device (str): Computation device. Default 'cuda:0'.

    Outputs:
        dict: Summary statistics of diagnostic run including NaN counts and gradient behaviors.
    """
    logger.info(f"Initializing diagnostic run on {device} using {dataset_json}...")
    
    dataset = ReXDataset(
        dataset_json=dataset_json,
        split="train",
        img_dir=str(RAW_IMAGES_DIR),
        seg_dir=str(RAW_MASKS_DIR),
        cache_dir=str(TEXT_CACHE_DIR),
        is_train=True,
        patch_size=192
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info(f"Loaded dataset: {len(dataset)} scans.")
    
    model = load_voxtell_model(str(MODEL_DIR), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss(sigmoid=True)
    
    diagnostic_history = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_running_loss = 0.0
        nan_batch_count = 0
        
        logger.info(f"--- Starting Diagnostic Epoch {epoch}/{epochs} ---")
        
        for batch_idx, batch in enumerate(dataloader):
            scan_id = batch['scan_id'][0]
            images = batch['image'].to(device)
            targets = batch['seg'].to(device)
            text_embeds = batch['text_embeddings'].to(device)
            
            fg_voxels = int(targets.sum().item())
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images, text_embeds)
                if isinstance(outputs, (list, tuple)):
                    raw_scale_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
                    weights = raw_scale_weights[:len(outputs)]
                    w_sum = sum(weights)
                    norm_weights = [w / w_sum for w in weights]
                    
                    total_loss = 0.0
                    loss_bce_val = 0.0
                    loss_dice_val = 0.0
                    for s_logits, w in zip(outputs, norm_weights):
                        s_logits_f32 = torch.clamp(s_logits.float(), min=-30.0, max=30.0)
                        if s_logits_f32.shape[2:] != targets.shape[2:]:
                            s_target = torch.nn.functional.interpolate(
                                targets.float(),
                                size=s_logits_f32.shape[2:],
                                mode='nearest'
                            )
                        else:
                            s_target = targets.float()
                        s_bce = bce_criterion(s_logits_f32, s_target)
                        s_dice = dice_criterion(s_logits_f32, s_target)
                        total_loss = total_loss + w * (s_bce + s_dice)
                        loss_bce_val = loss_bce_val + (w * s_bce).item()
                        loss_dice_val = loss_dice_val + (w * s_dice).item()
                else:
                    logits_f32 = torch.clamp(outputs.float(), min=-30.0, max=30.0)
                    targets_f32 = targets.float()
                    loss_bce = bce_criterion(logits_f32, targets_f32)
                    loss_dice = dice_criterion(logits_f32, targets_f32)
                    total_loss = loss_bce + loss_dice
                    loss_bce_val = loss_bce.item()
                    loss_dice_val = loss_dice.item()
                
            loss_val = total_loss.item()
            is_finite_loss = torch.isfinite(total_loss).item()
            
            # Record step stats
            step_record = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "scan_id": scan_id,
                "fg_voxels": fg_voxels,
                "loss_bce": loss_bce_val,
                "loss_dice": loss_dice_val,
                "total_loss": loss_val,
                "is_finite_loss": is_finite_loss
            }
            
            if not is_finite_loss:
                nan_batch_count += 1
                logger.warning(
                    f"Epoch {epoch} Batch {batch_idx:02d} ({scan_id}): Non-finite loss! "
                    f"BCE={loss_bce_val}, Dice={loss_dice_val}, Total={loss_val}, FG={fg_voxels}"
                )
            else:
                logger.info(
                    f"Epoch {epoch} Batch {batch_idx:02d} ({scan_id}): "
                    f"Loss={loss_val:.4f} (BCE={loss_bce_val:.4f}, Dice={loss_dice_val:.4f}), FG={fg_voxels}"
                )
                
            # Backward pass & gradient telemetry
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            
            # Check gradients
            grads_finite = True
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        grads_finite = False
                        logger.warning(f"Epoch {epoch} Batch {batch_idx:02d}: Non-finite gradient in {name}!")
                        break
            step_record["grads_finite"] = grads_finite
            
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            step_record["grad_norm"] = grad_norm.item()
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_running_loss += loss_val
            diagnostic_history.append(step_record)
            
        # Check weights at end of epoch
        weights_finite = True
        for name, param in model.named_parameters():
            if not torch.isfinite(param.data).all():
                weights_finite = False
                logger.error(f"Epoch {epoch}: Non-finite parameter weight detected in {name}!")
                break
                
        avg_loss = epoch_running_loss / len(dataloader)
        logger.info(
            f"=== Epoch {epoch} Complete === "
            f"Avg Loss: {avg_loss} | NaN Batches: {nan_batch_count}/{len(dataloader)} | Weights Finite: {weights_finite}"
        )
        
    return {
        "history": diagnostic_history,
        "weights_finite": weights_finite
    }


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        CLI entry point for Exp 001 diagnostic harness.

    Inputs:
        None

    Outputs:
        None
    """
    parser = argparse.ArgumentParser(description="Exp 001 Diagnostic Harness")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_MINI_JSON), help="Path to mini dataset.json")
    parser.add_argument("--epochs", type=int, default=2, help="Number of diagnostic epochs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target CUDA device")
    args = parser.parse_args()

    results = run_diagnostics(dataset_json=args.dataset_json, epochs=args.epochs, device=args.device)
    print("\n--- Diagnostic Run Summary ---")
    print(f"Total steps recorded: {len(results['history'])}")
    nan_steps = [s for s in results['history'] if not s['is_finite_loss']]
    print(f"NaN / Inf loss steps: {len(nan_steps)}")
    non_finite_grads = [s for s in results['history'] if not s['grads_finite']]
    print(f"Non-finite gradient steps: {len(non_finite_grads)}")
    print(f"Final model weights finite: {results['weights_finite']}")


if __name__ == "__main__":
    main()
