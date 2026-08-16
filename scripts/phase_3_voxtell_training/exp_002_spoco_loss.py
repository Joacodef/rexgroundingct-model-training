"""
===============================================================================
SCRIPT:         VoxTell Positive-Unlabeled (PU) SPOCO Fine-Tuning Pipeline
PHASE:          Phase 3 — Model Fine-Tuning & Loss Hypotheses Benchmarking
LOCATION:       scripts/phase_3_voxtell_training/exp_002_spoco_loss.py
OBJECTIVE:      Fine-tune VoxTell using Positive-Unlabeled (PU) SPOCO loss with 
                dilated ROI-masked supervision and Mean Teacher EMA consistency 
                regularization to resolve instance suppression bias (Wolny et al., 2022).
USAGE:          CUDA_VISIBLE_DEVICES=1 python scripts/phase_3_voxtell_training/exp_002_spoco_loss.py
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from tqdm import tqdm

import monai
monai.data.set_track_meta(False)
import monai.transforms as mt
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
EXP_LOG_DIR = LOGS_DIR / "phase_3_voxtell_training" / "exp_002_spoco_loss"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(EXP_LOG_DIR / "run.log"), mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("exp_002_spoco_loss")


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
            Load, normalize, crop, and augment a single CT volume patch and its text embeddings with self-healing caching.

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
            except Exception:
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
            
            # Atomic save to prevent corruption
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
        
        # Sample 1 finding per volume during training to manage memory footprint
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


def compute_roi_mask(seg_target: torch.Tensor, kernel_size: int = 11, padding: int = 5) -> torch.Tensor:
    """
    Signature:
        compute_roi_mask(seg_target: torch.Tensor, kernel_size: int, padding: int) -> torch.Tensor

    Objective:
        Generate a 3D Region of Interest (ROI) binary mask by applying 3D max-pooling dilation to targets.

    Inputs:
        seg_target (torch.Tensor): Binary target segmentation tensor of shape (B, F, Z, Y, X).
        kernel_size (int): 3D max pooling kernel size. Default 11.
        padding (int): Padding for max pooling. Default 5.

    Outputs:
        torch.Tensor: Dilated boolean mask tensor of shape (B, F, Z, Y, X).
    """
    dilated = F.max_pool3d(seg_target.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return dilated > 0


def compute_spoco_loss(logits: torch.Tensor, targets: torch.Tensor, roi_mask: torch.Tensor, pos_weight: float = 10.0) -> torch.Tensor:
    """
    Signature:
        compute_spoco_loss(logits: torch.Tensor, targets: torch.Tensor, roi_mask: torch.Tensor, pos_weight: float) -> torch.Tensor

    Objective:
        Compute SPOCO Masked Supervised Loss (BCE + Dice) strictly confined within the dilated ROI mask.

    Inputs:
        logits (torch.Tensor): Pre-sigmoid logit tensor of shape (B, F, Z, Y, X).
        targets (torch.Tensor): Ground truth target tensor of shape (B, F, Z, Y, X).
        roi_mask (torch.Tensor): Dilated boolean ROI mask tensor.
        pos_weight (float): Positive class weight for BCE loss. Default 10.0.

    Outputs:
        torch.Tensor: Scalar loss tensor (BCE_masked + Dice_masked).
    """
    dtype = logits.dtype
    # 1. BCE Loss confined to dilated ROI mask with class-weighted positives
    pos_weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=dtype), pos_weight=pos_weight_tensor, reduction='none')
    bce_masked = (bce * roi_mask.to(dtype=dtype)).sum().float() / (roi_mask.to(dtype=dtype).sum().float() + 1e-6)
    
    # 2. Dice Loss confined to dilated ROI mask
    probs = torch.sigmoid(logits)
    probs_masked = probs * roi_mask.to(dtype=dtype)
    targets_masked = targets.to(dtype=dtype) * roi_mask.to(dtype=dtype)
    
    intersection = (probs_masked * targets_masked).sum(dim=(2, 3, 4)).float()
    union = probs_masked.sum(dim=(2, 3, 4)).float() + targets_masked.sum(dim=(2, 3, 4)).float()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    return bce_masked + dice.mean()


def compute_unannotated_consistency_loss(student_probs: torch.Tensor, teacher_probs: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
    """
    Signature:
        compute_unannotated_consistency_loss(student_probs: torch.Tensor, teacher_probs: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor

    Objective:
        Compute Mean Squared Error (MSE) consistency loss across unannotated background voxels outside the dilated ROI.

    Inputs:
        student_probs (torch.Tensor): Student network probability predictions (B, F, Z, Y, X).
        teacher_probs (torch.Tensor): Teacher network probability predictions (B, F, Z, Y, X).
        roi_mask (torch.Tensor): Dilated boolean ROI mask tensor isolating annotated regions.

    Outputs:
        torch.Tensor: Scalar MSE consistency loss over unannotated voxels.
    """
    bg_mask = (~roi_mask).float()
    diff_sq = (student_probs - teacher_probs) ** 2
    loss_con = (diff_sq * bg_mask).sum() / (bg_mask.sum() + 1e-6)
    return loss_con


def get_consistency_weight(epoch: int, max_weight: float = 0.5, warm_up_epochs: int = 15) -> float:
    """
    Signature:
        get_consistency_weight(epoch: int, max_weight: float, warm_up_epochs: int) -> float

    Objective:
        Compute sigmoid warm-up scaling factor for consistency loss at current training epoch.

    Inputs:
        epoch (int): Current epoch index (1-indexed).
        max_weight (float): Target maximum consistency loss weight. Default 0.5.
        warm_up_epochs (int): Number of warmup epochs. Default 15.

    Outputs:
        float: Computed consistency weight scaling factor.
    """
    if warm_up_epochs <= 0 or epoch >= warm_up_epochs:
        return max_weight
    x = epoch - (warm_up_epochs / 2.0)
    sigmoid = 1.0 / (1.0 + math.exp(-2.0 * x))
    return max_weight * sigmoid


@torch.no_grad()
def update_ema_variables(student_model: nn.Module, teacher_model: nn.Module, alpha: float) -> None:
    """
    Signature:
        update_ema_variables(student_model: nn.Module, teacher_model: nn.Module, alpha: float) -> None

    Objective:
        Update Teacher network parameters via Exponential Moving Average (EMA) from Student network.

    Inputs:
        student_model (nn.Module): Active Student network model.
        teacher_model (nn.Module): Target Teacher network model.
        alpha (float): EMA decay weighting factor (e.g. 0.999).

    Outputs:
        None (In-place parameter update).
    """
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
        
    for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)


def load_voxtell_model(model_dir: str, device: str) -> nn.Module:
    """
    Signature:
        load_voxtell_model(model_dir: str, device: str) -> nn.Module

    Objective:
        Load plans.json architectural hyperparameters, instantiate VoxTellModel, and load checkpoint weights.

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


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command line arguments for Exp 002 PU-SPOCO fine-tuning.

    Inputs:
        None

    Outputs:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VoxTell PU-SPOCO Fine-Tuning Pipeline (Exp 002)")
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
    parser.add_argument("--alpha", type=float, default=0.999, help="Teacher EMA momentum decay parameter (default: 0.999)")
    parser.add_argument("--pos_weight", type=float, default=10.0, help="Positive class weight for BCE loss inside ROIs (default: 10.0)")
    parser.add_argument("--max_consistency_weight", type=float, default=0.5, help="Maximum consistency loss weight (default: 0.5)")
    parser.add_argument("--consistency_warmup", type=int, default=15, help="Number of warmup epochs for consistency loss (default: 15)")
    parser.add_argument("--patch_size", type=int, default=192, help="Patch size for MONAI spatial crop (default: 192)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device (e.g. cuda:0)")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest_model.pt if available")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="rexgroundingct", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default="exp_002_spoco_loss", help="Weights & Biases run name")
    return parser.parse_args()


def train_spoco_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    w_con: float,
    pos_weight: float,
    alpha: float
) -> tuple[float, float, float]:
    """
    Signature:
        train_spoco_epoch(student_model: nn.Module, teacher_model: nn.Module, dataloader: DataLoader, optimizer: Optimizer, scaler: GradScaler, device: str, w_con: float, pos_weight: float, alpha: float) -> tuple[float, float, float]

    Objective:
        Execute one training epoch using PU-SPOCO dilated ROI masked supervision + Mean Teacher EMA consistency.

    Inputs:
        student_model (nn.Module): Active Student network model.
        teacher_model (nn.Module): Target Teacher network model.
        dataloader (DataLoader): PyTorch training DataLoader.
        optimizer (Optimizer): PyTorch AdamW optimizer.
        scaler (GradScaler): AMP Gradient Scaler.
        device (str): Computation device string.
        w_con (float): Consistency loss weight for the current epoch.
        pos_weight (float): Positive class weight for BCE loss inside ROIs.
        alpha (float): EMA decay weighting factor for Teacher model update.

    Outputs:
        tuple[float, float, float]: (avg_total_loss, avg_sup_loss, avg_con_loss).
    """
    student_model.train()
    teacher_model.eval()
    
    total_loss_acc = 0.0
    sup_loss_acc = 0.0
    con_loss_acc = 0.0
    
    for batch in tqdm(dataloader, desc="Training Epoch (PU-SPOCO)", leave=False):
        images = batch['image'].to(device)
        targets = batch['seg'].to(device)
        text_embeds = batch['text_embeddings'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            # Student forward pass
            student_logits = student_model(images, text_embeds)
            student_probs = torch.sigmoid(student_logits)
            
            # Teacher forward pass (freeze gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images, text_embeds)
                teacher_probs = torch.sigmoid(teacher_logits)
                
            # Compute dilated ROI mask
            roi_mask = compute_roi_mask(targets, kernel_size=11, padding=5)
            
            # Supervised SPOCO loss strictly within ROI
            loss_sup = compute_spoco_loss(student_logits.float(), targets.float(), roi_mask, pos_weight=pos_weight)
            
            # Consistency regularization on unannotated voxels
            loss_con = compute_unannotated_consistency_loss(student_probs.float(), teacher_probs.float(), roi_mask)
            
            # Combined loss
            total_loss = loss_sup + w_con * loss_con
            
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Synchronize Teacher parameters via EMA
        update_ema_variables(student_model, teacher_model, alpha)
        
        total_loss_acc += total_loss.item()
        sup_loss_acc += loss_sup.item()
        con_loss_acc += loss_con.item()
        
    num_batches = max(len(dataloader), 1)
    return total_loss_acc / num_batches, sup_loss_acc / num_batches, con_loss_acc / num_batches


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for VoxTell Positive-Unlabeled (PU) SPOCO fine-tuning execution.

    Inputs:
        None

    Outputs:
        None
    """
    args = parse_args()
    logger.info("Starting VoxTell PU-SPOCO Fine-Tuning Pipeline (Exp 002)...")
    logger.info(f"Target Device: {args.device}")
    logger.info(f"Epochs: {args.epochs}, LR: {args.lr}, Alpha: {args.alpha}, PosWeight: {args.pos_weight}, PatchSize: {args.patch_size}")
    
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
    
    # Instantiate Student and Teacher Models
    logger.info("Instantiating Student VoxTell network...")
    student_model = load_voxtell_model(args.model_dir, args.device)
    
    logger.info("Instantiating Teacher VoxTell network...")
    teacher_model = load_voxtell_model(args.model_dir, args.device)
    for param in teacher_model.parameters():
        param.requires_grad = False
        
    # Optimizer, Scheduler, Scaler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    
    start_epoch = 1
    best_loss = float("inf")
    latest_model_path = output_dir / "latest_model.pt"

    if args.resume and latest_model_path.exists():
        logger.info(f"Resuming training from checkpoint: {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=args.device, weights_only=False)
        student_model.load_state_dict(checkpoint["student_state_dict"])
        teacher_model.load_state_dict(checkpoint["teacher_state_dict"])
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

    for epoch in range(start_epoch, args.epochs + 1):
        w_con = get_consistency_weight(epoch, max_weight=args.max_consistency_weight, warm_up_epochs=args.consistency_warmup)
        
        epoch_loss, sup_loss, con_loss = train_spoco_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=args.device,
            w_con=w_con,
            pos_weight=args.pos_weight,
            alpha=args.alpha
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] — Total Loss: {epoch_loss:.4f} "
            f"(Sup: {sup_loss:.4f}, Con: {con_loss:.4f}, w_con: {w_con:.3f}) | LR: {current_lr:.6f}"
        )
        
        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "student_state_dict": student_model.state_dict(),
            "teacher_state_dict": teacher_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss
        }, latest_model_path)

        # Save best model based on total loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "student_state_dict": student_model.state_dict(),
                "teacher_state_dict": teacher_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, best_model_path)
            logger.info(f"Saved new best model checkpoint to: {best_model_path}")
            
        if args.wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/total_loss": epoch_loss,
                "train/sup_loss": sup_loss,
                "train/con_loss": con_loss,
                "train/w_con": w_con,
                "train/lr": current_lr,
                "train/best_loss": best_loss
            })

    if args.wandb:
        import wandb
        wandb.finish()

    logger.info("PU-SPOCO fine-tuning training complete.")


if __name__ == "__main__":
    main()
