"""
===============================================================================
MODULE:         Dataset and Caching Infrastructure for Phase 3
LOCATION:       scripts/phase_3_voxtell_training/common/dataset.py
OBJECTIVE:      Provide native-resolution 3D CT dataset loader with MONAI 
                spatial cropping, intensity Z-score normalization, atomic fast 
                RAID SSD volume caching, and pre-computed text embeddings.
===============================================================================
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

import monai
monai.data.set_track_meta(False)
import monai.transforms as mt
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization

# Centralized Spatial Engine
from scripts.common.orientation import load_nifti_ras

logger = logging.getLogger("phase_3_dataset")


class ReXDataset(Dataset):
    """
    Native Resolution 3D CT Dataset for ReXGroundingCT fine-tuning.
    Loads images, 4D segmentations, and Qwen text embeddings, applying
    MONAI patch-based cropping, intensity Z-score normalization, and fast SSD caching.
    """

    def __init__(
        self, 
        dataset_json: str, 
        split: str, 
        img_dir: str, 
        seg_dir: str, 
        cache_dir: str, 
        is_train: bool = True, 
        patch_size: int = 192,
        max_findings_per_scan: int = 1
    ):
        """
        Signature:
            __init__(dataset_json: str, split: str, img_dir: str, seg_dir: str, cache_dir: str, is_train: bool, patch_size: int, max_findings_per_scan: int) -> None

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
            max_findings_per_scan (int): Maximum number of findings to sample per scan. Default 1.

        Outputs:
            None
        """
        self.split = split
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.cache_dir = cache_dir
        self.is_train = is_train
        self.max_findings_per_scan = max_findings_per_scan
        
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
            except Exception:
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
        
        # Sample findings per volume to manage GPU memory footprint
        num_findings = text_embeddings.shape[0]
        if num_findings > self.max_findings_per_scan:
            if self.is_train:
                selected_indices = np.random.choice(num_findings, self.max_findings_per_scan, replace=False)
            else:
                selected_indices = np.arange(self.max_findings_per_scan)
            
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
