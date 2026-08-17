"""
===============================================================================
MODULE:         VoxTell Model Loader for Phase 3
LOCATION:       scripts/phase_3_voxtell_training/common/model_loader.py
OBJECTIVE:      Instantiate VoxTellModel from plans.json configuration and load 
                pre-trained baseline checkpoint weights for Phase 3 fine-tuning.
===============================================================================
"""

import json
import logging
import pydoc
from pathlib import Path
import torch
import torch.nn as nn

from voxtell.model.voxtell_model import VoxTellModel

logger = logging.getLogger("phase_3_model_loader")


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
