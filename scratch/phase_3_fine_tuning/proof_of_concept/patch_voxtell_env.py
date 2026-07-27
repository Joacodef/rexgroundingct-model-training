#!/usr/bin/env python3
"""
VoxTell fp16 Underflow Bug Patcher
==================================

Context:
--------
During the fine-tuning and inference of the VoxTell v1.1 model (Experiment 003), 
we encountered a catastrophic issue where the model achieved exactly 0.0000 Dice score.
This was originally masked by a temporary hotfix (`torch.nan_to_num(nan=0.0)`) in the 
ValidationPredictor, which forced the script to predict foreground everywhere 
(because sigmoid(0.0) = 0.5, the threshold for foreground).

The Root Cause:
---------------
The bug is native to the `voxtell` pip package (specifically `voxtell.inference.predictor`).
When merging overlapping patches during 3D sliding window inference, the network multiplies 
predictions by a Gaussian mask. The Gaussian mask applies extremely tiny weights near 
the edges of a patch.
Because the accumulation buffers (`predicted_logits` and `n_predictions`) were initialized 
as `torch.half` (fp16), these tiny edge weights underflowed to exactly `0.0`.
When the network divided the logits by the weights to average them, it resulted in `0.0 / 0.0`, 
creating catastrophic `NaN`s that corrupted the entire prediction.

The Fix:
--------
This script automatically locates the installed `voxtell` pip package in the active 
Python environment and upgrades the accumulation buffers in `predictor.py` from 
`torch.half` to `torch.float32`. `float32` provides enough precision to prevent 
the Gaussian underflow.

Usage:
------
Run this script immediately after running `pip install voxtell` or whenever setting up 
a new virtual environment for this repository:
$ python scripts/patch_voxtell_env.py
"""

import os
import re
import importlib.util

def patch_voxtell_predictor():
    print("Locating voxtell package...")
    spec = importlib.util.find_spec("voxtell")
    if spec is None or spec.origin is None:
        print("Error: Could not find 'voxtell' package. Is it installed in this environment?")
        return

    # spec.origin is typically .../site-packages/voxtell/__init__.py
    voxtell_dir = os.path.dirname(spec.origin)
    predictor_path = os.path.join(voxtell_dir, "inference", "predictor.py")
    
    if not os.path.exists(predictor_path):
        print(f"Error: Could not find predictor.py at {predictor_path}")
        return

    print(f"Found predictor.py at: {predictor_path}")
    
    with open(predictor_path, 'r') as f:
        content = f.read()

    # Define the precise lines to target
    target_logits = "dtype=torch.half,\n                                        device=results_device)"
    target_n_preds = "n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)"
    
    replacement_logits = "dtype=torch.float32,\n                                        device=results_device)"
    replacement_n_preds = "n_predictions = torch.zeros(data.shape[1:], dtype=torch.float32, device=results_device)"

    if replacement_logits in content and replacement_n_preds in content:
        print("Success: The package is already patched with torch.float32.")
        return

    if target_logits not in content or target_n_preds not in content:
        print("Error: Could not find the expected torch.half lines in predictor.py. The package may have been updated or already modified.")
        return

    # Apply the patches
    content = content.replace(target_logits, replacement_logits)
    content = content.replace(target_n_preds, replacement_n_preds)

    with open(predictor_path, 'w') as f:
        f.write(content)

    print("Success: Patched predictor.py to use torch.float32 for Gaussian accumulation buffers!")

if __name__ == "__main__":
    patch_voxtell_predictor()
