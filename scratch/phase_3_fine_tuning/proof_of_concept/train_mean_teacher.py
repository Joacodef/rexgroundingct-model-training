"""
VoxTell Mean Teacher Training Pipeline for ReXGroundingCT.

This script implements Phase 2 Task 1 of ReXGroundingCT:
1. Pre-computes and caches Qwen text embeddings for all prompts.
2. Initializes Student and Teacher VoxTell networks from v1.1 weights.
3. Implements native resolution patch-based dataloaders with positive-centered cropping.
4. Updates the Teacher network via Exponential Moving Average (EMA).
5. Runs training epochs and saves checkpoints.
"""

import os
from dotenv import load_dotenv
# Load environment variables first to ensure proper GPU isolation
load_dotenv(override=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import hashlib
import argparse
import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

import monai
monai.data.set_track_meta(False)
import monai.transforms as mt
from torch.utils.data import Dataset, DataLoader
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization


# Import VoxTell dependencies
from voxtell.model.voxtell_model import VoxTellModel
from voxtell.inference.predictor import VoxTellPredictor


class ReXDataset(Dataset):
    """
    Native Resolution 3D CT Dataset for ReXGroundingCT fine-tuning.
    Loads images and 4D segmentations in native RAS space, crops non-zero regions,
    and applies MONAI patch-based cropping and augmentations.
    """
    def __init__(self, dataset_json, split, img_dir, seg_dir, cache_dir, is_train=True, patch_size=192):
        self.split = split
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        
        with open(dataset_json, 'r') as f:
            data = json.load(f)
        self.entries = data.get(split, [])
        
        # Replicate nnUNet intensity normalization
        self.normalization = ZScoreNormalization(intensityproperties={})
        
        # Dynamically compute a fixed-length MD5 hash based on preprocessing transformations
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
        
        # Setup MONAI Augmentation Pipeline
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

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        scan_id = entry['name'].replace('.nii.gz', '')
        
        # Resolve paths
        img_path = os.path.join(self.img_dir, f"{scan_id}.nii.gz")
        seg_path = os.path.join(self.seg_dir, f"{scan_id}.nii.gz")
        
        # Fast local SSD-based volume caching to bypass CPU-bound Gzip decompression
        ssd_cache_dir = os.path.join(
            os.environ["TMP_PREP_DIR"],
            f"volume_cache_{self.preprocessing_hash}"
        )
        os.makedirs(ssd_cache_dir, exist_ok=True)
        
        cache_img_path = os.path.join(ssd_cache_dir, f"{scan_id}_img.pt")
        cache_seg_path = os.path.join(ssd_cache_dir, f"{scan_id}_seg.pt")
        
        if os.path.exists(cache_img_path) and os.path.exists(cache_seg_path):
            img_normalized = torch.load(cache_img_path, map_location='cpu')
            seg_cropped = torch.load(cache_seg_path, map_location='cpu')
        else:
            # 1. Load image and reorient to RAS using Nibabel
            nib_img = nib.load(img_path)
            from nibabel.orientations import io_orientation
            img_ornt = io_orientation(nib_img.affine)
            img_r = nib_img.as_reoriented(img_ornt)
            img_data = img_r.get_fdata().transpose((2, 1, 0))[None] # shape: (1, Z, Y, X)
            
            # 2. Load 4D segmentation mask
            nib_seg = nib.load(seg_path)
            seg_r = nib_seg.as_reoriented(img_ornt)
            seg_data = seg_r.get_fdata().transpose((0, 3, 2, 1)) # shape: (F, Z, Y, X)
            
            # 3. Perform cropping to non-zero region of the image
            img_data = img_data.astype(np.float32)
            seg_data = seg_data.astype(np.float32)
            
            img_cropped, _, bbox = crop_to_nonzero(img_data, None)
            # Apply identical spatial cropping to segmentation
            seg_cropped = seg_data[:, bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
            
            # 4. Perform intensity Z-score normalization
            img_normalized = self.normalization.run(img_cropped, None)
            
            # Convert to PyTorch tensors and cache on fast local SSD
            img_normalized = torch.as_tensor(img_normalized, dtype=torch.float32)
            seg_cropped = torch.as_tensor(seg_cropped, dtype=torch.float32)
            
            torch.save(img_normalized, cache_img_path)
            torch.save(seg_cropped, cache_seg_path)
        
        # 5. Load pre-computed Qwen text embeddings
        cache_path = os.path.join(self.cache_dir, f"{scan_id}.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing pre-computed text embeddings for case {scan_id}. Run caching first.")
        text_embeddings = torch.load(cache_path, map_location='cpu') # shape: (F, 2560)
        
        # Cap number of findings during training/validation to manage memory footprint and prevent OOM
        num_findings = text_embeddings.shape[0]
        max_f = 1
        if num_findings > max_f:
            if self.is_train:
                # Randomly sample 1 finding
                selected_indices = np.random.choice(num_findings, max_f, replace=False)
            else:
                # Deterministically select first 1 finding for consistent validation metrics
                selected_indices = np.arange(max_f)
            
            text_embeddings = text_embeddings[selected_indices]
            seg_cropped = seg_cropped[selected_indices]
        
        # 6. Apply MONAI patch-based crop / augmentations
        data_dict = {
            'image': img_normalized,
            'seg': seg_cropped
        }
        
        if self.is_train:
            transformed = self.transforms(data_dict)
            # RandCropByPosNegLabeld returns list due to num_samples=1
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


class ValidationPredictor(VoxTellPredictor):
    """
    Lightweight subclass of VoxTellPredictor that bypasses loading the heavy 
    Qwen text embedding model during validation to conserve GPU memory.
    """
    def __init__(self, model_dir: str, device: torch.device, network: torch.nn.Module):
        self.device = device
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
        self.normalization = ZScoreNormalization(intensityproperties={})
        
        self.tile_step_size = 0.5
        self.perform_everything_on_device = False
        self.max_text_length = 8192
        self.verbose = False
        
        # Load network settings
        from batchgenerators.utilities.file_and_folder_operations import join, load_json
        plans = load_json(join(model_dir, 'plans.json'))
        self.patch_size = plans['configurations']['3d_fullres']['patch_size']
        
        self.network = network

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        logits = super().predict_sliding_window_return_logits(input_image, text_embeddings)
        return logits


def precompute_text_cache(dataset_json, cache_dir, device):
    """
    Offline pre-computation of Qwen text embeddings for all dataset prompts.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    from transformers import AutoModel, AutoTokenizer
    from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
    
    print("Initializing Qwen text embedding model...")
    text_encoding_model = 'Qwen/Qwen3-Embedding-4B'
    tokenizer = AutoTokenizer.from_pretrained(text_encoding_model, padding_side='left')
    text_backbone = AutoModel.from_pretrained(text_encoding_model).eval().to(device)
    
    with open(dataset_json, 'r') as f:
        metadata = json.load(f)
        
    all_entries = metadata.get('train', []) + metadata.get('val', [])
    
    print(f"Pre-computing Qwen text embeddings for {len(all_entries)} cases...")
    for entry in tqdm(all_entries, desc="Caching text embeddings"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue
            
        cache_path = os.path.join(cache_dir, f"{scan_id}.pt")
        if os.path.exists(cache_path):
            continue
            
        findings = entry.get('findings', {})
        if not findings:
            torch.save(torch.zeros((0, 2560), dtype=torch.float32), cache_path)
            continue
            
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
            
        with torch.no_grad():
            instruct_text_prompts = wrap_with_instruction(text_prompts)
            text_tokens = tokenizer(
                instruct_text_prompts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}
            text_embed = text_backbone(**text_tokens)
            embeddings = last_token_pool(text_embed.last_hidden_state, text_tokens['attention_mask'])
            
        torch.save(embeddings.cpu(), cache_path)
        torch.cuda.empty_cache()
        
    print("\nText cache successfully generated!")


def load_voxtell_model(model_dir, device, deep_supervision=False):
    """
    Instantiates and loads weights for the VoxTellModel baseline checkpoint.
    """
    import pydoc
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    
    plans = load_json(join(model_dir, 'plans.json'))
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
        deep_supervision=deep_supervision
    )
    
    checkpoint = torch.load(
        join(model_dir, 'fold_0', 'checkpoint_final.pth'),
        map_location=torch.device('cpu'),
        weights_only=False
    )
    
    model.load_state_dict(checkpoint['network_weights'])
    model = model.to(device)
    return model


@torch.no_grad()
def update_ema_variables(student_model, teacher_model, alpha):
    """
    Updates the Teacher's parameters using Exponential Moving Average (EMA).
    """
    # Parameter update
    for teacher_param, student_param in zip(teacher_model.parameters(), student_model.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
        
    # Buffer synchronization
    for teacher_buffer, student_buffer in zip(teacher_model.buffers(), student_model.buffers()):
        teacher_buffer.data.copy_(student_buffer.data)


import torch.nn.functional as F

def compute_roi_mask(seg_target, kernel_size=11, padding=5):
    """
    Generates a Region of Interest (ROI) mask by applying 3D binary dilation to the targets.
    Dilation is performed efficiently on GPU using F.max_pool3d.
    """
    # seg_target shape: (batch, F, Z, Y, X)
    dilated = F.max_pool3d(seg_target.float(), kernel_size=kernel_size, stride=1, padding=padding)
    return dilated > 0


def compute_spoco_loss(logits, targets, roi_mask, pos_weight=1.0):
    """
    SPOCO Masked Supervised Loss. Confines BCE and Dice losses strictly within the dilated ROI.
    """
    dtype = logits.dtype
    # 1. BCE Loss confined to the dilated mask with class-weighted positives
    if pos_weight != 1.0:
        pos_weight_tensor = torch.tensor([pos_weight], device=logits.device, dtype=dtype)
        bce = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=dtype), pos_weight=pos_weight_tensor, reduction='none')
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=dtype), reduction='none')
    bce_masked = (bce * roi_mask.to(dtype=dtype)).sum().float() / (roi_mask.to(dtype=dtype).sum().float() + 1e-6)
    
    # 2. Dice Loss confined to the dilated mask
    probs = torch.sigmoid(logits)
    probs_masked = probs * roi_mask.to(dtype=dtype)
    targets_masked = targets.to(dtype=dtype) * roi_mask.to(dtype=dtype)
    
    intersection = (probs_masked * targets_masked).sum(dim=(2, 3, 4)).float()
    union = probs_masked.sum(dim=(2, 3, 4)).float() + targets_masked.sum(dim=(2, 3, 4)).float()
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    
    return bce_masked + dice.mean()


def compute_dice_score(logits, targets, threshold=0.0):
    """
    Computes global Dice Coefficient directly on full-volume logits vs targets.
    """
    probs = (logits > threshold).float()
    targets_f = targets.float()
    intersection = (probs * targets_f).sum(dim=(2, 3, 4))
    union = probs.sum(dim=(2, 3, 4)) + targets_f.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean()


def compute_mpr_consistency_loss(student_probs, teacher_probs, roi_mask):
    """
    MPR Consistency Loss. Computes mean consistency (MSE) over Axial, Coronal,
    and Sagittal max-projections in unannotated regions.
    """
    dtype = student_probs.dtype
    # Isolate unannotated background region
    bg_mask = ~(roi_mask)
    bg_student = student_probs * bg_mask.to(dtype=dtype)
    bg_teacher = teacher_probs * bg_mask.to(dtype=dtype)
    
    # 2D Max projections along Axial (Z), Coronal (Y), and Sagittal (X) axes
    p_axial_s = torch.max(bg_student, dim=2)[0]
    p_coronal_s = torch.max(bg_student, dim=3)[0]
    p_sagittal_s = torch.max(bg_student, dim=4)[0]
    
    p_axial_t = torch.max(bg_teacher, dim=2)[0]
    p_coronal_t = torch.max(bg_teacher, dim=3)[0]
    p_sagittal_t = torch.max(bg_teacher, dim=4)[0]
    
    # Compute MSE across projections in float32 to prevent exponent overflow
    loss_axial = F.mse_loss(p_axial_s.float(), p_axial_t.float())
    loss_coronal = F.mse_loss(p_coronal_s.float(), p_coronal_t.float())
    loss_sagittal = F.mse_loss(p_sagittal_s.float(), p_sagittal_t.float())
    
    return (loss_axial + loss_coronal + loss_sagittal) / 3.0


def get_consistency_weight(epoch, max_weight=10.0, warm_up_epochs=5):
    """
    Computes a sigmoid warm-up weight scaling factor for consistency loss.
    """
    if warm_up_epochs == 0:
        return max_weight
    if epoch >= warm_up_epochs:
        return max_weight
    
    import math
    x = epoch - (warm_up_epochs / 2.0)
    sigmoid = 1.0 / (1.0 + math.exp(-2.0 * x))
    return max_weight * sigmoid


def main():
    parser = argparse.ArgumentParser(description="VoxTell Mean Teacher Training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.999, help="Teacher EMA decay parameter")
    parser.add_argument("--cache-only", action="store_true", help="Only run offline text cache pre-computation")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 2-case 5-epoch test loop")
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint if available")
    parser.add_argument("--max-consistency-weight", type=float, default=0.5, help="Maximum consistency scaling weight (default: 0.5)")
    parser.add_argument("--consistency-warmup", type=int, default=15, help="Consistency loss warmup epochs (default: 15)")
    parser.add_argument("--pos-weight", type=float, default=10.0, help="Positive class weight for BCE loss inside ROIs (default: 10.0)")
    parser.add_argument("--patch-size", type=int, default=192, help="Patch size for training crops (default: 192)")
    parser.add_argument("--val-only", action="store_true", help="Run only the validation loop on the loaded checkpoint")
    args = parser.parse_args()

    # Paths isolation
    dataset_json = os.environ["DATASET_JSON"]
    model_dir = os.environ["MODEL_DIR"]
    img_dir = os.environ["IMG_RAW_DIR"]
    seg_dir = os.environ["SEG_RAW_DIR"]
    cache_dir = os.path.join(os.path.dirname(dataset_json), "text_cache")

    # If model_dir points to voxtell_v1.0 but voxtell_v1.1 exists, redirect to it
    if model_dir and model_dir.endswith("voxtell_v1.0"):
        v11_dir = os.path.join(os.path.dirname(model_dir), "voxtell_v1.1")
        if os.path.exists(v11_dir):
            print(f"[INFO] Automatically redirecting MODEL_DIR to {v11_dir} for training.")
            model_dir = v11_dir

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    # 1. Modify configs if in smoke test mode
    if args.smoke_test:
        print("[INFO] Running in SMOKE-TEST mode.")
        args.epochs = 5
        # Create a tiny custom debugging dataset file to run fast
        smoke_dataset_json = os.path.join(os.path.dirname(dataset_json), "dataset_test_smoke.json")
        if not os.path.exists(smoke_dataset_json):
            # Write a simple smoke test dataset mapping
            with open(dataset_json, 'r') as f:
                orig_dataset = json.load(f)
            smoke_dataset = {
                "train": orig_dataset["train"][:2],
                "val": orig_dataset["val"][:1]
            }
            with open(smoke_dataset_json, 'w') as f:
                json.dump(smoke_dataset, f, indent=4)
        dataset_json = smoke_dataset_json

    # 2. Check/Run pre-computed text cache
    if args.cache_only:
        precompute_text_cache(dataset_json, cache_dir, device)
        return
        
    # Standard cache check
    if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        print("[INFO] Text cache is empty. Initializing offline text caching...")
        precompute_text_cache(dataset_json, cache_dir, device)

    # 3. Load Models
    print("Loading baseline model weights into Student model...")
    student_model = load_voxtell_model(model_dir, device, deep_supervision=False)
    print("Loading baseline model weights into Teacher model...")
    teacher_model = load_voxtell_model(model_dir, device, deep_supervision=False)

    # Freeze Teacher gradients
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 4. Setup Dataloaders
    train_dataset = ReXDataset(dataset_json, "train", img_dir, seg_dir, cache_dir, is_train=True, patch_size=args.patch_size)
    val_dataset = ReXDataset(dataset_json, "val", img_dir, seg_dir, cache_dir, is_train=False, patch_size=args.patch_size)

    # Batch size is strictly 1 volume per GPU to fit high-res native training memory boundaries
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 5.1 Resume from latest checkpoint if specified
    start_epoch = 0
    latest_checkpoint_path = os.path.join(os.path.dirname(model_dir), "checkpoint_mean_teacher_latest.pth")
    if args.resume and os.path.exists(latest_checkpoint_path):
        print(f"[INFO] Resuming training from checkpoint: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        student_model.load_state_dict(checkpoint['student_state_dict'])
        teacher_model.load_state_dict(checkpoint['teacher_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"[INFO] Successfully resumed training from epoch {start_epoch}")

    # Initialize ValidationPredictor once outside the training/validation loops
    predictor = ValidationPredictor(model_dir=model_dir, device=device, network=student_model)
    # Ensure validation sliding window uses the configured patch size to prevent OOM
    predictor.patch_size = [args.patch_size, args.patch_size, args.patch_size]

    # 6. Initialize Weights & Biases
    if args.wandb:
        import wandb
        wandb.init(
            project="ReXGroundingCT-Phase2",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "ema_alpha": args.alpha,
                "smoke_test": args.smoke_test
            }
        )

    def run_validation(epoch_idx):
        student_model.eval()
        val_loss = 0.0
        val_dice = 0.0
        print(f"[DEBUG] GPU memory before validation: {torch.cuda.memory_allocated() / 1e9:.2f} GB | reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch_idx + 1}/{args.epochs} [Val]"):
                image = batch['image'][0].to(device)
                seg_target = batch['seg'][0].to(device)
                text_embeddings = batch['text_embeddings'].to(device)
                
                val_logits = predictor.predict_sliding_window_return_logits(image, text_embeddings).to(device)
                
                val_roi_mask = compute_roi_mask(seg_target[None], kernel_size=11, padding=5)
                loss = compute_spoco_loss(val_logits[None], seg_target[None], val_roi_mask, pos_weight=args.pos_weight)
                val_loss += loss.item()
                
                dice = compute_dice_score(val_logits[None], seg_target[None])
                val_dice += dice.item()
                
                del image, seg_target, text_embeddings, val_logits, val_roi_mask, loss, dice
                torch.cuda.empty_cache()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        print(f"Epoch {epoch_idx + 1} Validation. Average Val Loss: {avg_val_loss:.6f} | Average Dice: {avg_val_dice:.4f}")
        
        if args.wandb:
            wandb.log({"val/loss": avg_val_loss, "val/dice": avg_val_dice, "epoch": epoch_idx + 1})
            
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[DEBUG] GPU memory after validation cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB | reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    if args.val_only:
        print("\n[INFO] Running in validation-only mode.")
        run_validation(max(0, start_epoch - 1))
        if args.wandb:
            wandb.finish()
        return

    # 7. Training Loop
    print("\nStarting Mean Teacher fine-tuning pipeline...")
    for epoch in range(start_epoch, args.epochs):
        student_model.train()
        epoch_loss = 0.0
        epoch_loss_sup = 0.0
        epoch_loss_con = 0.0
        
        # Warmup epochs: scale down to 2 for smoke-test, standard 15 (or custom) otherwise
        warmup = 2 if args.smoke_test else args.consistency_warmup
        # In smoke-test, we also scale the max consistency weight for safety
        max_w = 0.1 if args.smoke_test else args.max_consistency_weight
        w_con = get_consistency_weight(epoch, max_weight=max_w, warm_up_epochs=warmup)
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]"):
            image = batch['image'].to(device)                           # shape: (1, 1, 192, 192, 192)
            seg_target = batch['seg'].to(device)                       # shape: (1, F, 192, 192, 192)
            text_embeddings = batch['text_embeddings'].to(device)       # shape: (1, F, 2560)
            
            optimizer.zero_grad()
            
            # Use Automatic Mixed Precision (AMP) with bfloat16 to optimize memory usage
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Student Forward
                student_logits = student_model(image, text_embeddings)     # shape: (1, F, 192, 192, 192)
                student_probs = torch.sigmoid(student_logits)
                
                # Teacher Forward (freeze gradients)
                with torch.no_grad():
                    teacher_logits = teacher_model(image, text_embeddings)  # shape: (1, F, 192, 192, 192)
                    teacher_probs = torch.sigmoid(teacher_logits)
                    
                # Compute ROI Mask and Semi-Supervised Losses
                roi_mask = compute_roi_mask(seg_target, kernel_size=11, padding=5)
                loss_sup = compute_spoco_loss(student_logits, seg_target, roi_mask, pos_weight=args.pos_weight)
                loss_con = compute_mpr_consistency_loss(student_probs, teacher_probs, roi_mask)
                
                # Total loss
                loss = loss_sup + w_con * loss_con
            
            # Backward pass & Optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Synchronize Teacher weights using EMA
            update_ema_variables(student_model, teacher_model, args.alpha)
            
            epoch_loss += loss.item()
            epoch_loss_sup += loss_sup.item()
            epoch_loss_con += loss_con.item()
            
            if args.wandb:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/loss_sup_step": loss_sup.item(),
                    "train/loss_con_step": loss_con.item(),
                    "train/w_con": w_con
                })
            
            # Explicit memory cleanup to prevent GPU OOM
            del image, seg_target, text_embeddings, student_logits, student_probs, teacher_logits, teacher_probs, roi_mask, loss_sup, loss_con, loss
            torch.cuda.empty_cache()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        avg_loss_sup = epoch_loss_sup / len(train_loader)
        avg_loss_con = epoch_loss_con / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.6f} (Sup: {avg_loss_sup:.6f}, Con: {avg_loss_con:.6f})")
        
        if args.wandb:
            wandb.log({
                "train/loss": avg_loss,
                "train/loss_sup": avg_loss_sup,
                "train/loss_con": avg_loss_con,
                "epoch": epoch + 1
            })

        # 8. Validation Pass
        run_validation(epoch)

        # Save latest checkpoint at the end of each epoch for training recovery
        latest_checkpoint_path = os.path.join(os.path.dirname(model_dir), "checkpoint_mean_teacher_latest.pth")
        torch.save({
            'epoch': epoch,
            'student_state_dict': student_model.state_dict(),
            'teacher_state_dict': teacher_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, latest_checkpoint_path)
        print(f"[INFO] Saved latest checkpoint for epoch {epoch + 1} to {latest_checkpoint_path}")

    # Save final model checkpoint
    checkpoint_path = os.path.join(os.path.dirname(model_dir), "checkpoint_mean_teacher_final.pth")
    torch.save({
        'student_state_dict': student_model.state_dict(),
        'teacher_state_dict': teacher_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"\nTraining pipeline completed. Saved final checkpoint to {checkpoint_path}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
