"""
===============================================================================
SCRIPT:         VoxTell Naïve Supervised Fine-Tuning Baseline
PHASE:          Phase 3 — Model Fine-Tuning & Adaptation
LOCATION:       scripts/phase_3_voxtell_training/exp_001_naive_finetuning.py
OBJECTIVE:      Naïve supervised fine-tuning of VoxTell baseline (voxtell_v1.1) 
                using standard BCE + Dice loss on 3D CT volume patches from the 
                2,992-scan training split. Establishes the supervised lower bound.
USAGE:          python scripts/phase_3_voxtell_training/exp_001_naive_finetuning.py
===============================================================================
"""

import os
import sys
import json
import math
import hashlib
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure proper GPU isolation before loading torch
load_dotenv(override=False)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from tqdm import tqdm

import monai
monai.data.set_track_meta(False)
import monai.transforms as mt
from monai.losses import DiceLoss
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    TEXT_CACHE_DIR, TMP_PREP_DIR, LOGS_DIR, MODEL_DIR
)

# Import Centralized Spatial Engine and VoxTell dependencies
from scripts.common.orientation import load_nifti_ras
from voxtell.model.voxtell_model import VoxTellModel

# Setup experiment logging directory
EXP_LOG_DIR = LOGS_DIR / "phase_3_voxtell_training" / "exp_001_naive_finetuning"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXP_LOG_DIR / "run.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("exp_001_naive_finetuning")


class ReXDataset(Dataset):
    """
    Native Resolution 3D CT Dataset for ReXGroundingCT fine-tuning.
    Loads images, 4D segmentations, and Qwen text embeddings, applying
    MONAI patch-based cropping, intensity Z-score normalization, and SSD caching.
    """

    def __init__(self, dataset_json: str, split: str, img_dir: str, seg_dir: str, cache_dir: str, is_train: bool = True, patch_size: int = 192):
        """
        Signature:
            __init__(dataset_json: str, split: str, img_dir: str, seg_dir: str, cache_dir: str, is_train: bool, patch_size: int) -> None

        Objective:
            Initialize ReXDataset instance, setup MONAI augmentation pipeline, Z-score intensity normalization, and SSD cache hash.

        Inputs:
            dataset_json (str): Path to dataset.json metadata.
            split (str): Dataset partition ('train', 'val', 'test').
            img_dir (str): Directory path containing raw CT images.
            seg_dir (str): Directory path containing raw GT segmentations.
            cache_dir (str): Directory path containing precomputed Qwen text embeddings.
            is_train (bool): Whether dataset is configured for training (applies random augmentations). Default True.
            patch_size (int): Spatial crop patch size (e.g. 192). Default 192.

        Outputs:
            None
        """
        self.split = split
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        
        with open(dataset_json, 'r') as f:
            data = json.load(f)
        self.entries = data.get(split, [])
        
        # Intensity Z-score normalization
        self.normalization = ZScoreNormalization(intensityproperties={})
        
        # MD5 hash based on preprocessing configuration
        norm_name = self.normalization.__class__.__name__
        prep_config = {
            "orientation": "RAS",
            "transpose_img": [2, 1, 0],
            "transpose_seg": [0, 3, 2, 1],
            "cropping": "crop_to_nonzero",
            "normalization": norm_name
        }
        config_str = json.dumps(prep_config, sort_keys=True)
        self.preprocessing_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:12]
        
        # MONAI Transform Pipeline
        if self.is_train:
            self.transforms = mt.Compose([
                mt.SpatialPadd(keys=['image', 'seg'], spatial_size=[patch_size, patch_size, patch_size], mode='constant'),
                mt.RandCropByPosNegLabeld(
                    keys=['image', 'seg'],
                    label_key='seg',
                    spatial_size=[patch_size, patch_size, patch_size],
                    pos=1.0,
                    neg=0.0,
                    num_samples=1
                ),
                mt.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=['image', 'seg'], prob=0.5, spatial_axis=2),
                mt.EnsureTyped(keys=['image', 'seg'], dtype=torch.float32)
            ])
        else:
            self.transforms = mt.Compose([
                mt.EnsureTyped(keys=['image', 'seg'], dtype=torch.float32)
            ])

    def __len__(self) -> int:
        """
        Signature:
            __len__() -> int

        Objective:
            Return total number of dataset entries.

        Inputs:
            None

        Outputs:
            int: Number of entries in dataset split.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        """
        Signature:
            __getitem__(idx: int) -> dict

        Objective:
            Load, normalize, crop, and augment a single CT volume patch and its text embeddings.

        Inputs:
            idx (int): Dataset entry index.

        Outputs:
            dict: Data dictionary containing 'image', 'seg', 'text_embeddings', and 'scan_id'.
        """
        entry = self.entries[idx]
        scan_id = entry['name'].replace('.nii.gz', '')
        
        img_path = os.path.join(self.img_dir, f"{scan_id}.nii.gz")
        seg_path = os.path.join(self.seg_dir, f"{scan_id}.nii.gz")
        
        # Fast local SSD-based volume caching
        tmp_prep_dir = os.getenv("TMP_PREP_DIR", "/tmp/rexgroundingct_preprocessed")
        ssd_cache_dir = os.path.join(
            tmp_prep_dir,
            f"volume_cache_{self.preprocessing_hash}"
        )
        os.makedirs(ssd_cache_dir, exist_ok=True)
        
        cache_img_path = os.path.join(ssd_cache_dir, f"{scan_id}_img.pt")
        cache_seg_path = os.path.join(ssd_cache_dir, f"{scan_id}_seg.pt")
        
        loaded_from_cache = False
        if os.path.exists(cache_img_path) and os.path.exists(cache_seg_path):
            try:
                img_normalized = torch.load(cache_img_path, map_location='cpu')
                seg_cropped = torch.load(cache_seg_path, map_location='cpu')
                if isinstance(img_normalized, torch.Tensor) and isinstance(seg_cropped, torch.Tensor):
                    loaded_from_cache = True
            except Exception as e:
                # Invalidate broken/corrupted cache files
                if os.path.exists(cache_img_path):
                    try:
                        os.remove(cache_img_path)
                    except OSError:
                        pass
                if os.path.exists(cache_seg_path):
                    try:
                        os.remove(cache_seg_path)
                    except OSError:
                        pass
                loaded_from_cache = False

        if not loaded_from_cache:
            # Load canonical RAS physical coordinate space via Centralized Spatial Engine
            img_ras, _, _ = load_nifti_ras(Path(img_path))
            img_data = img_ras.transpose((2, 1, 0))[None] # (1, Z, Y, X)
            
            seg_ras, _, _ = load_nifti_ras(Path(seg_path))
            seg_data = seg_ras.transpose((0, 3, 2, 1)) # (F, Z, Y, X)
            
            img_data = img_data.astype(np.float32)
            seg_data = seg_data.astype(np.float32)
            
            img_cropped, _, bbox = crop_to_nonzero(img_data, None)
            seg_cropped = seg_data[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
            
            img_normalized = self.normalization.run(img_cropped, None)
            
            img_normalized = torch.as_tensor(img_normalized, dtype=torch.float32)
            seg_cropped = torch.as_tensor(seg_cropped, dtype=torch.float32)
            
            # Atomic save to prevent corruption from concurrent workers or premature termination
            tmp_img = f"{cache_img_path}.tmp_{os.getpid()}_{idx}"
            tmp_seg = f"{cache_seg_path}.tmp_{os.getpid()}_{idx}"
            try:
                torch.save(img_normalized, tmp_img)
                torch.save(seg_cropped, tmp_seg)
                os.replace(tmp_img, cache_img_path)
                os.replace(tmp_seg, cache_seg_path)
            except Exception:
                for tmp_f in [tmp_img, tmp_seg]:
                    if os.path.exists(tmp_f):
                        try:
                            os.remove(tmp_f)
                        except OSError:
                            pass
        
        # Load pre-computed Qwen text embeddings
        cache_path = os.path.join(self.cache_dir, f"{scan_id}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing pre-computed text embeddings for case {scan_id} at {cache_path}")
        try:
            text_embeddings = torch.load(cache_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading text embeddings from {cache_path}: {e}")
        
        # Sample 1 finding per volume to manage GPU memory footprint
        num_findings = text_embeddings.shape[0]
        max_f = 1
        if num_findings > max_f:
            if self.is_train:
                selected_indices = np.random.choice(num_findings, max_f, replace=False)
            else:
                selected_indices = np.arange(max_f)
            
            text_embeddings = text_embeddings[selected_indices]
            seg_cropped = seg_cropped[selected_indices]
        
        data_dict = {
            'image': img_normalized,
            'seg': seg_cropped
        }
        
        if self.is_train:
            transformed = self.transforms(data_dict)
            transformed = transformed[0]
            image_tensor = torch.as_tensor(transformed['image'])
            seg_tensor = torch.as_tensor(transformed['seg'])
        else:
            transformed = self.transforms(data_dict)
            image_tensor = torch.as_tensor(transformed['image'])
            seg_tensor = torch.as_tensor(transformed['seg'])
            
        return {
            'image': image_tensor,
            'seg': seg_tensor,
            'text_embeddings': text_embeddings,
            'scan_id': scan_id
        }


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command line arguments for naïve fine-tuning execution.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VoxTell Naïve Supervised Fine-Tuning Baseline")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json metadata")
    parser.add_argument("--img_dir", type=str, default=str(RAW_IMAGES_DIR), help="Path to raw CT images directory")
    parser.add_argument("--seg_dir", type=str, default=str(RAW_MASKS_DIR), help="Path to raw CT segmentations directory")
    parser.add_argument("--cache_dir", type=str, default=str(TEXT_CACHE_DIR), help="Path to Qwen text embeddings cache directory")
    parser.add_argument("--model_dir", type=str, default=str(MODEL_DIR), help="Path to pre-trained voxtell_v1.1 checkpoint directory")
    parser.add_argument("--output_dir", type=str, default=str(EXP_LOG_DIR), help="Directory to save fine-tuned model checkpoints")
    
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--patch_size", type=int, default=192, help="Patch size for MONAI spatial crop")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (e.g. cuda:0)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_model.pt if available")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable Weights & Biases logging (default: True)")
    parser.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rexgroundingct", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="exp_001_naive_finetuning", help="Weights & Biases run name")
    return parser.parse_args()


def train_naive_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: str, bce_criterion: nn.Module, dice_criterion: nn.Module, global_step: int = 0) -> tuple[float, int]:
    """
    Signature:
        train_naive_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, device: str, bce_criterion: nn.Module, dice_criterion: nn.Module, global_step: int = 0) -> tuple[float, int]

    Objective:
        Execute one training epoch using naïve supervised BCE + Dice loss.

    Inputs:
        model (nn.Module): VoxTell model instance.
        dataloader (DataLoader): PyTorch training DataLoader.
        optimizer (Optimizer): PyTorch AdamW optimizer.
        scaler (GradScaler): AMP Gradient Scaler.
        device (str): Computation device string.
        bce_criterion (nn.Module): BCEWithLogitsLoss instance.
        dice_criterion (nn.Module): DiceLoss instance.
        global_step (int): Running global iteration counter across epochs.

    Outputs:
        tuple[float, int]: Average training loss over the epoch and updated global_step.
    """
    model.train()
    running_loss = 0.0
    valid_batches = 0
    
    for batch in tqdm(dataloader, desc="Training Epoch", leave=False):
        images = batch['image'].to(device) # (B, 1, Z, Y, X)
        targets = batch['seg'].to(device)   # (B, F, Z, Y, X)
        text_embeds = batch['text_embeddings'].to(device) # (B, F, 2560)
        scan_id = batch.get('scan_id', ['unknown'])[0]
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # VoxTell forward pass
            logits = model(images, text_embeds) # (B, F, Z, Y, X)
            
            # Upcast logits and targets to float32 for loss stability
            logits_f32 = logits.float()
            targets_f32 = targets.float()
            
            loss_bce = bce_criterion(logits_f32, targets_f32)
            loss_dice = dice_criterion(logits_f32, targets_f32)
            total_loss = loss_bce + loss_dice
        
        if not torch.isfinite(total_loss):
            logger.warning(f"Scan {scan_id} produced non-finite loss (BCE={loss_bce.item()}, Dice={loss_dice.item()}). Skipping step.")
            continue
            
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        
        # Check gradient finiteness before clipping to prevent NaN corruption
        grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if grads_finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            logger.warning(f"Scan {scan_id} produced non-finite gradients. Skipping grad clipping and letting scaler adapt.")
            
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += total_loss.item()
        valid_batches += 1
        global_step += 1

        try:
            import wandb
            if wandb.run is not None and global_step % 5 == 0:
                wandb.log({
                    "train/step_loss": total_loss.item(),
                    "train/step_bce": loss_bce.item(),
                    "train/step_dice": loss_dice.item(),
                    "step": global_step
                })
        except Exception:
            pass
        
    return running_loss / max(valid_batches, 1), global_step


def load_voxtell_model(model_dir: str, device: str) -> nn.Module:
    """
    Signature:
        load_voxtell_model(model_dir: str, device: str) -> nn.Module

    Objective:
        Load plans.json architectural hyperparameters and instantiate VoxTellModel,
        then load pre-trained checkpoint weights.

    Inputs:
        model_dir (str): Directory containing plans.json and checkpoint_final.pth.
        device (str): Computation device string (e.g. 'cuda:0').

    Outputs:
        nn.Module: Loaded VoxTellModel instance placed on device.
    """
    import pydoc
    model_dir_path = Path(model_dir)
    plans_file = model_dir_path / "plans.json"
    
    if not plans_file.exists():
        raise FileNotFoundError(f"Missing plans.json at {plans_file}")
        
    with open(plans_file, 'r') as f:
        plans = json.load(f)
        
    arch_kwargs = plans['configurations']['3d_fullres']['architecture']['arch_kwargs']
    arch_kwargs = dict(**arch_kwargs)
    for required_import_key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
        if arch_kwargs[required_import_key] is not None:
            arch_kwargs[required_import_key] = pydoc.locate(arch_kwargs[required_import_key])
            
    model = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False
    )
    
    ckpt_path = model_dir_path / "fold_0" / "checkpoint_final.pth"
    if not ckpt_path.exists():
        ckpt_path = model_dir_path / "checkpoint_final.pth"
        
    if ckpt_path.exists():
        logger.info(f"Loading pre-trained VoxTell weights from {ckpt_path}")
        checkpoint_data = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint_data.get("network_weights", checkpoint_data.get("model", checkpoint_data))
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.warning(f"Pre-trained checkpoint not found at {ckpt_path}. Initializing from scratch.")
        
    return model.to(device)


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for VoxTell naïve supervised fine-tuning execution.

    Inputs:
        None

    Outputs:
        None
    """
    args = parse_args()
    logger.info("Starting VoxTell Naïve Supervised Fine-Tuning Pipeline (Exp 001)...")
    logger.info(f"Target Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}, Patch Size: {args.patch_size}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Datasets and DataLoaders
    train_dataset = ReXDataset(
        dataset_json=args.dataset_json,
        split="train",
        img_dir=args.img_dir,
        seg_dir=args.seg_dir,
        cache_dir=args.cache_dir,
        is_train=True,
        patch_size=args.patch_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Loaded training split: {len(train_dataset)} scans.")
    
    # Instantiate VoxTell Model and load pre-trained weights via plans.json
    model = load_voxtell_model(args.model_dir, args.device)
    
    # Optimizer, Loss Criteria, and Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss(sigmoid=True)
    
    start_epoch = 1
    best_loss = float("inf")
    latest_model_path = output_dir / "latest_model.pt"

    if args.resume and latest_model_path.exists():
        logger.info(f"Resuming training from checkpoint: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "loss" in checkpoint:
            best_loss = checkpoint["loss"]
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Successfully resumed from epoch {start_epoch}, previous best loss: {best_loss:.4f}")
    
    # Initialize Weights & Biases if requested
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        logger.info(f"Initialized Weights & Biases logging (Project: {args.wandb_project}, Run: {args.wandb_run_name})")

    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss, global_step = train_naive_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=args.device,
            bce_criterion=bce_criterion,
            dice_criterion=dice_criterion,
            global_step=global_step
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"Epoch [{epoch:02d}/{args.epochs:02d}] — Loss: {epoch_loss:.4f} | LR: {current_lr:.6f}")
        
        # Save latest model checkpoint
        latest_model_path = output_dir / "latest_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss
        }, latest_model_path)

        # Save best model checkpoint based on training loss
        if math.isfinite(epoch_loss) and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, best_model_path)
            logger.info(f"Saved new best model checkpoint to: {best_model_path}")
            
        if args.wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/lr": current_lr,
                "train/best_loss": best_loss
            })

    if args.wandb:
        import wandb
        wandb.finish()

    logger.info("Naïve supervised fine-tuning training complete.")


if __name__ == "__main__":
    main()
