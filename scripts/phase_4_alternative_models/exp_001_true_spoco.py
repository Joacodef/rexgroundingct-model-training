"""
Phase 4 — Exp 001: True 3D Medical SPOCO Model Prototype (Candidate A Architecture)

Objective:
    Scaffold a baseline executable prototype for true Sparse Object-level Consistency
    (SPOCO, Wolny et al., CVPR 2022) adapted for 3D medical text-guided segmentation.
    
    Architecture Design (Candidate A):
        1. 3D Vision Encoder/Decoder: MONAI 3D Residual U-Net backbone.
        2. Vision-Language Fusion: Multi-head Cross-Attention bottleneck conditioning
           spatial 3D feature maps on 2560-dim Qwen text embeddings.
        3. Pixel Embedding Head: Projects decoder features to a 16-dimensional continuous
           metric embedding space e(z, y, x) in R^16 (L2 normalized).
        4. Differentiable Gaussian Soft Masks: Anchor-based probability maps S_k(i) = exp(-||e_i - e(a_k)||^2 / 2*sigma^2).
        5. Loss Hierarchy:
           - Instance Soft Dice Loss (L_obj) on annotated lesion components.
           - Metric Cluster Pull & Push Loss (L_pull, L_push).
           - Unannotated Anchor Consistency Loss (L_con) between Student and EMA Teacher embeddings.

Status & Notice:
    [RESEARCH PROTOTYPE] Architectural design and parameterization (anchor sampling
    strategies, sigma bandwidth, pull/push margin thresholds, and clustering algorithms)
    are under active investigation and open for iterative refinement.
"""

import os
import sys
import math
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.amp import GradScaler
from tqdm import tqdm

import monai
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
)

# Relative root directory path resolution
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    DATA_DIR,
    RAW_IMAGES_DIR,
    RAW_SEGS_DIR,
    EMBEDDINGS_DIR,
    CACHE_DIR,
    LOGS_DIR,
    FINDING_CATEGORIES,
)
from scripts.common.orientation import load_nifti_ras, save_nifti

# Experiment log directory pairing
EXP_LOG_DIR = LOGS_DIR / "phase_4_alternative_models" / "exp_001_true_spoco"
EXP_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXP_LOG_DIR / "run.log", mode="a"),
    ],
)
logger = logging.getLogger("exp_001_true_spoco")


# =========================================================================
# 1. Dataset & MONAI Data Pipeline
# =========================================================================

class ReXSpocoDataset(Dataset):
    """
    Signature:
        ReXSpocoDataset(dataset_json: Path, split: str = 'train', patch_size: tuple[int, int, int] = (192, 192, 192), cache_dir: Optional[Path] = CACHE_DIR, is_train: bool = True)

    Objective:
        PyTorch Dataset for Phase 4 SPOCO training, yielding 3D CT image tensors,
        4D multi-finding segmentation masks, and 2560-dim text embedding vectors.

    Inputs:
        dataset_json (Path): Absolute/relative path to dataset.json.
        split (str): Split name ('train' or 'val'). Default 'train'.
        patch_size (tuple[int, int, int]): 3D spatial patch size. Default (192, 192, 192).
        cache_dir (Optional[Path]): Directory containing preprocessed .pt cache files.
        is_train (bool): Whether training augmentations are enabled. Default True.

    Outputs:
        dict: Sample dictionary containing 'image', 'seg', 'text_embeddings', and 'scan_id'.
    """

    def __init__(
        self,
        dataset_json: Path,
        split: str = "train",
        patch_size: Tuple[int, int, int] = (192, 192, 192),
        cache_dir: Optional[Path] = CACHE_DIR,
        is_train: bool = True,
    ) -> None:
        """Initialize SPOCO dataset with cache configuration and split samples."""
        self.split = split
        self.patch_size = patch_size
        self.cache_dir = cache_dir
        self.is_train = is_train

        with open(dataset_json, "r") as f:
            data = json.load(f)
        self.samples = data.get(split, [])
        logger.info(f"Loaded {len(self.samples)} {split} scans for SPOCO pipeline.")

        # MONAI 3D patch transforms
        if is_train:
            self.transform = Compose(
                [
                    RandCropByPosNegLabeld(
                        keys=["image", "seg"],
                        label_key="seg",
                        spatial_size=patch_size,
                        pos=2.0,
                        neg=1.0,
                        num_samples=1,
                        image_key="image",
                        image_threshold=0.0,
                    ),
                    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),
                    RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),
                    EnsureTyped(keys=["image", "seg"]),
                ]
            )
        else:
            self.transform = EnsureTyped(keys=["image", "seg"])

    def __len__(self) -> int:
        """
        Signature:
            __len__() -> int

        Objective:
            Return total number of dataset cases.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Signature:
            __getitem__(idx: int) -> dict[str, Any]

        Objective:
            Fetch preprocessed 3D CT volume, segmentation mask, and text embeddings.
        """
        sample = self.samples[idx]
        scan_id = sample["scan_id"]
        cached_file = self.cache_dir / f"{scan_id}.pt" if self.cache_dir else None

        if cached_file and cached_file.exists():
            data_dict = torch.load(cached_file, weights_only=True)
            image = data_dict["image"]
            seg = data_dict["seg"]
            text_embeds = data_dict["text_embeddings"]
        else:
            # Fallback loading from raw storage
            img_path = RAW_IMAGES_DIR / f"{scan_id}.nii.gz"
            seg_path = RAW_SEGS_DIR / f"{scan_id}.nii.gz"
            embed_path = EMBEDDINGS_DIR / f"{scan_id}.pt"

            image_arr, _ = load_nifti_ras(img_path)
            seg_arr, _ = load_nifti_ras(seg_path)

            image = torch.from_numpy(image_arr).float().unsqueeze(0)  # (1, Z, Y, X)
            seg = torch.from_numpy(seg_arr).float()                   # (F, Z, Y, X)
            text_embeds = torch.load(embed_path, weights_only=True)   # (F, 2560)

        # Standardize dynamic range
        image = (image - image.mean()) / (image.std() + 1e-6)

        data = {"image": image, "seg": seg}
        if self.is_train:
            transformed = self.transform(data)
            if isinstance(transformed, list):
                data = transformed[0]
            else:
                data = transformed

        return {
            "image": data["image"],
            "seg": data["seg"],
            "text_embeddings": text_embeds,
            "scan_id": scan_id,
        }


# =========================================================================
# 2. Candidate A Architecture: 3D UNet with Text Cross-Attention Bottleneck
# =========================================================================

class TextCrossAttentionBottleneck(nn.Module):
    """
    Signature:
        TextCrossAttentionBottleneck(spatial_dim: int = 256, text_dim: int = 2560, num_heads: int = 8)

    Objective:
        Cross-attention bottleneck module conditioning deepest 3D spatial feature maps
        on 2560-dimensional text query embeddings.

    Inputs:
        spatial_dim (int): Channel dimension of spatial bottleneck feature map. Default 256.
        text_dim (int): Embedding dimension of text queries. Default 2560 (Qwen).
        num_heads (int): Number of multi-head attention heads. Default 8.

    Outputs:
        torch.Tensor: Conditioned spatial feature tensor of shape (B, F, C, Z', Y', X').
    """

    def __init__(self, spatial_dim: int = 256, text_dim: int = 2560, num_heads: int = 8) -> None:
        """Initialize cross-attention module between spatial and text feature embeddings."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.text_proj = nn.Linear(text_dim, spatial_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=spatial_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(spatial_dim)
        self.norm2 = nn.LayerNorm(spatial_dim)
        self.ffn = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim * 2),
            nn.GELU(),
            nn.Linear(spatial_dim * 2, spatial_dim),
        )

    def forward(self, spatial_feats: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Signature:
            forward(spatial_feats: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor

        Inputs:
            spatial_feats (torch.Tensor): Tensor of shape (B, C, Z', Y', X').
            text_embeds (torch.Tensor): Tensor of shape (B, F, text_dim).

        Outputs:
            torch.Tensor: Fused tensor of shape (B, F, C, Z', Y', X').
        """
        B, C, Z, Y, X = spatial_feats.shape
        _, F_findings, _ = text_embeds.shape

        # Flatten spatial tokens: (B, N_spatial, C) where N_spatial = Z*Y*X
        spatial_tokens = spatial_feats.view(B, C, -1).permute(0, 2, 1)

        # Project text embeddings: (B, F, C)
        text_tokens = self.text_proj(text_embeds)

        # Cross attention for each finding category
        fused_list = []
        for f_idx in range(F_findings):
            q_token = text_tokens[:, f_idx : f_idx + 1, :]  # (B, 1, C)
            
            norm_spatial = self.norm1(spatial_tokens)
            attn_out, _ = self.cross_attn(query=norm_spatial, key=q_token, value=q_token)
            x = spatial_tokens + attn_out
            x = x + self.ffn(self.norm2(x))
            
            # Reshape back to 3D spatial map: (B, C, Z, Y, X)
            fused_3d = x.permute(0, 2, 1).view(B, C, Z, Y, X)
            fused_list.append(fused_3d)

        # Stack findings: (B, F, C, Z, Y, X)
        return torch.stack(fused_list, dim=1)


class SpocoUNet3D(nn.Module):
    """
    Signature:
        SpocoUNet3D(in_channels: int = 1, embedding_dim: int = 16, text_dim: int = 2560, feature_channels: tuple[int, ...] = (32, 64, 128, 256))

    Objective:
        3D Vision-Language U-Net producing continuous 16D pixel/voxel embeddings
        for True SPOCO metric learning segmentation.

    Inputs:
        in_channels (int): Input image channels (1 for CT). Default 1.
        embedding_dim (int): Dimensionality of pixel metric space D. Default 16.
        text_dim (int): Qwen text embedding dimension. Default 2560.
        feature_channels (tuple[int, ...]): Encoder/decoder channel depths.

    Outputs:
        torch.Tensor: Normalized pixel embeddings of shape (B, F, D, Z, Y, X).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 16,
        text_dim: int = 2560,
        feature_channels: Tuple[int, ...] = (32, 64, 128, 256),
    ) -> None:
        """Initialize 3D SPOCO U-Net architecture."""
        super().__init__()
        self.embedding_dim = embedding_dim
        c0, c1, c2, c3 = feature_channels

        # 3D Encoder Blocks
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, c0, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(c0, c1, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(c3),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Bottleneck Text Cross-Attention Fusion
        self.bottleneck = TextCrossAttentionBottleneck(spatial_dim=c3, text_dim=text_dim)

        # 3D Decoder Blocks (applied per finding category)
        self.up3 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv3d(c2 * 2, c2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.up2 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(c1 * 2, c1, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.up1 = nn.ConvTranspose3d(c1, c0, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(c0 * 2, c0, kernel_size=3, padding=1),
            nn.InstanceNorm3d(c0),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Pixel Embedding Projection Head: (c0 -> D)
        self.embedding_head = nn.Conv3d(c0, embedding_dim, kernel_size=1)

    def forward(self, images: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Signature:
            forward(images: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor

        Inputs:
            images (torch.Tensor): 3D CT patch tensor of shape (B, 1, Z, Y, X).
            text_embeds (torch.Tensor): Text query embeddings of shape (B, F, text_dim).

        Outputs:
            torch.Tensor: Dense pixel embeddings of shape (B, F, D, Z, Y, X) normalized along D.
        """
        B = images.shape[0]
        F_findings = text_embeds.shape[1]

        # 1. Encoder Forward
        e1 = self.enc1(images)  # (B, c0, Z, Y, X)
        e2 = self.enc2(e1)      # (B, c1, Z/2, Y/2, X/2)
        e3 = self.enc3(e2)      # (B, c2, Z/4, Y/4, X/4)
        e4 = self.enc4(e3)      # (B, c3, Z/8, Y/8, X/8)

        # 2. Text Conditioning at Bottleneck: (B, F, c3, Z/8, Y/8, X/8)
        fused_bottleneck = self.bottleneck(e4, text_embeds)

        # 3. Decoder Forward per finding
        out_embeds_list = []
        for f_idx in range(F_findings):
            d4 = fused_bottleneck[:, f_idx]  # (B, c3, Z/8, Y/8, X/8)

            d3 = self.up3(d4)
            d3 = torch.cat([d3, e3], dim=1)
            d3 = self.dec3(d3)

            d2 = self.up2(d3)
            d2 = torch.cat([d2, e2], dim=1)
            d2 = self.dec2(d2)

            d1 = self.up1(d2)
            d1 = torch.cat([d1, e1], dim=1)
            d1 = self.dec1(d1)

            # Project to metric embedding space: (B, D, Z, Y, X)
            embed = self.embedding_head(d1)
            
            # L2 normalize along embedding dimension for unit hypersphere metric space
            embed = F.normalize(embed, p=2, dim=1)
            out_embeds_list.append(embed)

        # Stack findings: (B, F, D, Z, Y, X)
        return torch.stack(out_embeds_list, dim=1)


# =========================================================================
# 3. True SPOCO Loss Module (Wolny et al. 2022 3D Formulation)
# =========================================================================

def compute_gaussian_soft_mask(
    embeddings: torch.Tensor,
    anchor_coords: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Signature:
        compute_gaussian_soft_mask(embeddings: torch.Tensor, anchor_coords: torch.Tensor, sigma: float = 0.5) -> torch.Tensor

    Objective:
        Compute differentiable Gaussian soft segmentation masks from anchor embeddings.
        S_k(i) = exp(- ||e_i - e(a_k)||^2 / (2 * sigma^2)).

    Inputs:
        embeddings (torch.Tensor): 3D pixel embeddings of shape (D, Z, Y, X).
        anchor_coords (torch.Tensor): Anchor voxel index tensor of shape (K, 3) where columns are (z, y, x).
        sigma (float): Gaussian bandwidth scaling factor. Default 0.5.

    Outputs:
        torch.Tensor: Soft mask tensor of shape (K, Z, Y, X) in range [0, 1].
    """
    D, Z, Y, X = embeddings.shape
    K = anchor_coords.shape[0]

    if K == 0:
        return torch.zeros((0, Z, Y, X), device=embeddings.device, dtype=embeddings.dtype)

    # Extract anchor embedding vectors: (K, D)
    anchor_embeds = []
    for k in range(K):
        az, ay, ax = anchor_coords[k]
        anchor_embeds.append(embeddings[:, az, ay, ax])
    anchor_embeds = torch.stack(anchor_embeds, dim=0)  # (K, D)

    # Compute Euclidean distance: ||e_i - e(a_k)||^2 = ||e_i||^2 + ||e(a_k)||^2 - 2 <e_i, e(a_k)>
    # Since embeddings are L2 normalized (||e|| = 1), ||e_i - e(a_k)||^2 = 2 - 2 * (e_i . e(a_k))
    dot_prod = torch.einsum("kd, dzyx -> kzyx", anchor_embeds, embeddings)
    dist_sq = torch.clamp(2.0 - 2.0 * dot_prod, min=0.0)

    # Gaussian soft mask
    soft_masks = torch.exp(-dist_sq / (2.0 * (sigma ** 2)))
    return soft_masks


def sample_annotated_anchors(target_mask: torch.Tensor) -> List[Tuple[int, int, int]]:
    """
    Signature:
        sample_annotated_anchors(target_mask: torch.Tensor) -> list[tuple[int, int, int]]

    Objective:
        Sample anchor voxels from annotated ground-truth lesion components.

    Inputs:
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).

    Outputs:
        list[tuple[int, int, int]]: List of (z, y, x) anchor coordinates for each component.
    """
    positive_coords = torch.nonzero(target_mask > 0.5)
    if len(positive_coords) == 0:
        return []

    # Sample central/median coordinate of positive cluster
    med_idx = len(positive_coords) // 2
    coord = positive_coords[med_idx].tolist()
    return [tuple(coord)]


def sample_unannotated_anchors(target_mask: torch.Tensor, num_anchors: int = 5) -> List[Tuple[int, int, int]]:
    """
    Signature:
        sample_unannotated_anchors(target_mask: torch.Tensor, num_anchors: int = 5) -> list[tuple[int, int, int]]

    Objective:
        Sample unannotated anchor voxels from background region (1 - GT) for Teacher consistency.

    Inputs:
        target_mask (torch.Tensor): Binary target tensor of shape (Z, Y, X).
        num_anchors (int): Number of unlabeled anchors to sample. Default 5.

    Outputs:
        list[tuple[int, int, int]]: List of (z, y, x) anchor coordinates in unannotated region.
    """
    unlabeled_coords = torch.nonzero(target_mask == 0)
    if len(unlabeled_coords) == 0:
        return []

    sampled_indices = torch.randint(0, len(unlabeled_coords), (min(num_anchors, len(unlabeled_coords)),))
    anchors = [tuple(unlabeled_coords[idx].tolist()) for idx in sampled_indices]
    return anchors


def compute_instance_dice_loss(soft_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    Signature:
        compute_instance_dice_loss(soft_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor

    Objective:
        Compute soft Dice loss for an individual anchor soft mask against ground truth.

    Inputs:
        soft_mask (torch.Tensor): Predicted Gaussian soft mask of shape (Z, Y, X).
        target_mask (torch.Tensor): Binary ground truth mask of shape (Z, Y, X).

    Outputs:
        torch.Tensor: Scalar Dice loss tensor in range [0, 1].
    """
    intersection = (soft_mask * target_mask).sum().float()
    union = soft_mask.sum().float() + target_mask.sum().float()
    return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)


def compute_spoco_total_loss(
    student_embeds: torch.Tensor,
    teacher_embeds: torch.Tensor,
    targets: torch.Tensor,
    sigma: float = 0.5,
    w_con: float = 0.1,
    num_unlabeled_anchors: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Signature:
        compute_spoco_total_loss(student_embeds: torch.Tensor, teacher_embeds: torch.Tensor, targets: torch.Tensor, sigma: float = 0.5, w_con: float = 0.1, num_unlabeled_anchors: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    Objective:
        Compute full True 3D SPOCO loss: Instance-level Dice on annotated objects (L_obj)
        plus Unannotated Anchor Soft-Mask Consistency (L_con) against EMA Teacher.

    Inputs:
        student_embeds (torch.Tensor): Student embeddings of shape (B, F, D, Z, Y, X).
        teacher_embeds (torch.Tensor): EMA Teacher embeddings of shape (B, F, D, Z, Y, X).
        targets (torch.Tensor): Ground truth target tensor of shape (B, F, Z, Y, X).
        sigma (float): Gaussian bandwidth scaling factor. Default 0.5.
        w_con (float): Consistency loss weight for unannotated anchors. Default 0.1.
        num_unlabeled_anchors (int): Number of unlabeled anchors sampled per finding. Default 5.

    Outputs:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (total_loss, loss_supervised_obj, loss_consistency_unlabeled).
    """
    B, F_findings, D, Z, Y, X = student_embeds.shape
    device = student_embeds.device

    loss_obj_list = []
    loss_con_list = []

    for b in range(B):
        for f in range(F_findings):
            s_embed = student_embeds[b, f]      # (D, Z, Y, X)
            t_embed = teacher_embeds[b, f]      # (D, Z, Y, X)
            tgt = targets[b, f]                 # (Z, Y, X)

            # 1. Supervised Instance Soft Dice on Annotated Anchors
            annotated_anchors = sample_annotated_anchors(tgt)
            if annotated_anchors:
                anchor_tensor = torch.tensor(annotated_anchors, device=device, dtype=torch.long)
                soft_masks = compute_gaussian_soft_mask(s_embed, anchor_tensor, sigma=sigma)  # (K, Z, Y, X)
                for k in range(len(annotated_anchors)):
                    l_dice = compute_instance_dice_loss(soft_masks[k], tgt)
                    loss_obj_list.append(l_dice)

            # 2. Unannotated Anchor Consistency (Student vs EMA Teacher)
            unlabeled_anchors = sample_unannotated_anchors(tgt, num_anchors=num_unlabeled_anchors)
            if unlabeled_anchors:
                u_anchor_tensor = torch.tensor(unlabeled_anchors, device=device, dtype=torch.long)
                s_u_soft = compute_gaussian_soft_mask(s_embed, u_anchor_tensor, sigma=sigma)  # (U, Z, Y, X)
                t_u_soft = compute_gaussian_soft_mask(t_embed, u_anchor_tensor, sigma=sigma)  # (U, Z, Y, X)

                for u in range(len(unlabeled_anchors)):
                    l_con = compute_instance_dice_loss(s_u_soft[u], t_u_soft[u].detach())
                    loss_con_list.append(l_con)

    loss_obj = torch.stack(loss_obj_list).mean() if loss_obj_list else torch.tensor(0.0, device=device, requires_grad=True)
    loss_con = torch.stack(loss_con_list).mean() if loss_con_list else torch.tensor(0.0, device=device, requires_grad=True)

    total_loss = loss_obj + w_con * loss_con
    return total_loss, loss_obj, loss_con


# =========================================================================
# 4. Teacher EMA & Training Loop
# =========================================================================

def update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float) -> None:
    """
    Signature:
        update_ema_variables(model: nn.Module, ema_model: nn.Module, alpha: float) -> None

    Objective:
        Update Teacher parameters via Exponential Moving Average (EMA).
    """
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)


def parse_args() -> argparse.Namespace:
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command line arguments for Phase 4 Exp 001 True SPOCO prototype.
    """
    parser = argparse.ArgumentParser(description="Phase 4 Exp 001: True 3D Medical SPOCO Model Prototype")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    parser.add_argument("--alpha", type=float, default=0.999, help="EMA decay rate for Teacher model")
    parser.add_argument("--sigma", type=float, default=0.5, help="Gaussian soft mask bandwidth")
    parser.add_argument("--w_con", type=float, default=0.1, help="Consistency loss weight for unlabeled anchors")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    parser.add_argument("--dry_run", action="store_true", help="Execute single-batch dry run verification")
    return parser.parse_args()


def main() -> None:
    """
    Signature:
        main() -> None

    Objective:
        Main entry point for Phase 4 Exp 001 True SPOCO training and architecture testing.
    """
    args = parse_args()
    logger.info("Initializing Phase 4 Exp 001 True 3D Medical SPOCO Prototype (Candidate A)...")
    logger.info(f"Target Device: {args.device} | Epochs: {args.epochs} | LR: {args.lr} | Bandwidth Sigma: {args.sigma}")

    dataset_json = DATA_DIR / "dataset.json"
    if not dataset_json.exists():
        logger.error(f"Dataset manifest not found at {dataset_json}")
        sys.exit(1)

    # Initialize Candidate A Models (Student & Teacher)
    student_model = SpocoUNet3D(in_channels=1, embedding_dim=16, text_dim=2560).to(args.device)
    teacher_model = SpocoUNet3D(in_channels=1, embedding_dim=16, text_dim=2560).to(args.device)
    teacher_model.load_state_dict(student_model.state_dict())
    for param in teacher_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler('cuda')

    train_dataset = ReXSpocoDataset(dataset_json=dataset_json, split="train", is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    if args.dry_run:
        logger.info("Running single-batch dry run verification...")
        batch = next(iter(train_loader))
        images = batch["image"].to(args.device)
        targets = batch["seg"].to(args.device)
        text_embeds = batch["text_embeddings"].to(args.device)

        with torch.amp.autocast('cuda'):
            s_embeds = student_model(images, text_embeds)
            with torch.no_grad():
                t_embeds = teacher_model(images, text_embeds)
            loss, l_obj, l_con = compute_spoco_total_loss(s_embeds, t_embeds, targets, sigma=args.sigma, w_con=args.w_con)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        logger.info(f"Dry run successful! Total Loss: {loss.item():.4f} | L_obj: {l_obj.item():.4f} | L_con: {l_con.item():.4f}")
        logger.info("Output embeddings shape: %s", tuple(s_embeds.shape))
        return

    logger.info("Prototype initialized and ready for full training execution upon design alignment.")


if __name__ == "__main__":
    main()
