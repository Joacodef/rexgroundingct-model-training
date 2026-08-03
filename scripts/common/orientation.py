"""
===============================================================================
MODULE:         Centralized Spatial Orientation Engine
LOCATION:       scripts/common/orientation.py
OBJECTIVE:      Minimal, clean single source of truth for loading, inspecting, 
                canonicalizing, and saving 3D/4D NIfTI volumes.
===============================================================================
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path


def load_nifti_ras(nifti_path: Path) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, str, str]]:
    """
    Signature:
        load_nifti_ras(nifti_path: Path) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, str, str]]

    Objective:
        Load a 3D or 4D NIfTI file, extract raw orientation axis codes via nib.orientations.aff2axcodes,
        reorient to canonical RAS space via nib.as_closest_canonical, and return array data in RAS space.

    Inputs:
        nifti_path (Path): Path to target NIfTI file (.nii or .nii.gz).

    Outputs:
        tuple containing:
            - data_ras (np.ndarray): Volume float32 array reoriented to canonical RAS coordinate space.
            - ras_nii (nib.Nifti1Image): Canonical RAS NIfTI image object (with RAS affine).
            - raw_axcodes (tuple[str, str, str]): Raw anatomical axis codes before canonicalization (e.g. ('L', 'P', 'S')).
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    raw_nii = nib.load(str(nifti_path))

    # 1. Affine Matrix Integrity Validation
    if not np.isfinite(raw_nii.affine).all():
        raise ValueError(f"Corrupt NIfTI affine matrix containing non-finite values in {nifti_path}")

    # 2. Extract Raw Anatomical Axis Codes
    raw_axcodes = nib.orientations.aff2axcodes(raw_nii.affine)

    # 3. Canonicalize to RAS Coordinate Space
    ras_nii = nib.as_closest_canonical(raw_nii)

    # 4. Extract Array Data with Header Slope/Intercept Scaling
    data_ras = ras_nii.get_fdata(dtype=np.float32)

    # 5. Standardize 4D Channel Axis to Front: (X, Y, Z, F) -> (F, X, Y, Z)
    if data_ras.ndim == 4 and data_ras.shape[-1] < np.min(data_ras.shape[:3]):
        data_ras = np.moveaxis(data_ras, -1, 0)

    return data_ras, ras_nii, raw_axcodes


def save_nifti(pred_array: np.ndarray, out_path: Path, affine: np.ndarray) -> None:
    """
    Signature:
        save_nifti(pred_array: np.ndarray, out_path: Path, affine: np.ndarray) -> None

    Objective:
        Save a 3D or 4D binary prediction array to disk as a NIfTI file with designated affine matrix,
        ensuring uint8 data typing and automatic parent directory creation.

    Inputs:
        pred_array (np.ndarray): Binary prediction mask array (3D or 4D).
        out_path (Path): File destination path for NIfTI file.
        affine (np.ndarray): 4x4 affine transformation matrix.

    Outputs:
        None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_array = np.asanyarray(pred_array, dtype=np.uint8)

    if affine is None or not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        affine = np.eye(4, dtype=np.float32)

    out_nii = nib.Nifti1Image(pred_array, affine)
    nib.save(out_nii, str(out_path))
