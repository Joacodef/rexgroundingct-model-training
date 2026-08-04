"""
===============================================================================
MODULE:         Centralized Spatial Orientation Engine
LOCATION:       scripts/common/orientation.py
OBJECTIVE:      Minimal, clean single source of truth for loading, inspecting, 
                canonicalizing, and saving 3D CT volumes and 4D segmentation masks.
===============================================================================
"""

import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path


def load_nifti_ras(nifti_path: Path, ref_affine: np.ndarray = None) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, str, str]]:
    """
    Signature:
        load_nifti_ras(nifti_path: Path, ref_affine: np.ndarray = None) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, str, str]]

    Objective:
        Load a 3D CT volume or 4D multi-finding segmentation mask NIfTI file, canonicalize to RAS coordinate space,
        and return array data in RAS space.
        First detects whether the input tensor is a 3D Volume or a 4D Segmentation Mask. For 4D segmentations,
        un-stacks the 4D tensor into a list of 3D finding matrices before spatial reorientation, reorients each 3D
        matrix independently in 3D spatial space using nib.as_closest_canonical, and re-stacks them into (F, X, Y, Z).

    Inputs:
        nifti_path (Path): Path to target NIfTI file (.nii or .nii.gz).
        ref_affine (np.ndarray, optional): 4x4 reference image affine matrix to override missing/identity affine headers.

    Outputs:
        tuple containing:
            - data_ras (np.ndarray): Volume float32 array reoriented to canonical RAS coordinate space.
              For 4D volumes, channel axis is placed first: (F, X, Y, Z).
            - ras_nii (nib.Nifti1Image): Canonical RAS NIfTI image object (with RAS affine).
            - raw_axcodes (tuple[str, str, str]): Raw anatomical axis codes before canonicalization (e.g. ('L', 'P', 'S')).
    """
    nifti_path = Path(nifti_path)
    if not nifti_path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")

    raw_nii = nib.load(str(nifti_path))
    data = raw_nii.get_fdata(dtype=np.float32)

    # 1. Affine Matrix Integrity & Reference Affine Repair Validation
    # Upstream Dataset Artifact: Raw segmentation masks in RAW_MASKS_DIR were stored with
    # uninformative identity matrix headers (np.eye(4)), while their 3D voxel arrays were
    # drawn directly on the raw CT image's DICOM LPS grid. If ref_affine is not passed,
    # auto-fetch the matching parent CT image affine from RAW_IMAGES_DIR if present.
    if not np.isfinite(raw_nii.affine).all():
        raise ValueError(f"Corrupt NIfTI affine matrix containing non-finite values in {nifti_path}")

    if ref_affine is not None and isinstance(ref_affine, np.ndarray) and ref_affine.shape == (4, 4):
        affine = ref_affine
    elif np.allclose(raw_nii.affine, np.eye(4)):
        # Auto-repair fallback: look for corresponding raw CT image in RAW_IMAGES_DIR
        try:
            from scripts.config import RAW_IMAGES_DIR
            matching_img = RAW_IMAGES_DIR / nifti_path.name
            if matching_img.exists() and matching_img != nifti_path:
                img_nii_tmp = nib.load(str(matching_img))
                if np.isfinite(img_nii_tmp.affine).all() and not np.allclose(img_nii_tmp.affine, np.eye(4)):
                    affine = img_nii_tmp.affine
                else:
                    affine = raw_nii.affine
            else:
                affine = raw_nii.affine
        except Exception:
            affine = raw_nii.affine
    else:
        affine = raw_nii.affine

    # 2. Detector of Segmentation Mask (4D) vs. Image Volume (3D)
    is_segmentation = (data.ndim == 4)

    if is_segmentation:
        d0, d1, d2, d3 = data.shape
        if d0 == d1 and d1 == d2 and d3 <= 64:
            finding_axis = 3
        elif d0 <= 64 and d0 < d3 and d0 < d1 and d0 < d2:
            finding_axis = 0
        elif d3 <= 64:
            finding_axis = 3
        else:
            finding_axis = 0

        if finding_axis == 0:
            masks_3d = [data[i] for i in range(d0)]
        else:
            masks_3d = [data[..., i] for i in range(d3)]

        # Separate and reorient each 3D finding matrix independently in 3D spatial space
        reoriented_3d_list = []
        first_ras_nii = None
        first_axcodes = None

        for m_3d in masks_3d:
            nii_3d = nib.Nifti1Image(m_3d, affine)
            if first_axcodes is None:
                first_axcodes = nib.orientations.aff2axcodes(nii_3d.affine)
            ras_nii_3d = nib.as_closest_canonical(nii_3d)
            if first_ras_nii is None:
                first_ras_nii = ras_nii_3d
            reoriented_3d_list.append(ras_nii_3d.get_fdata(dtype=np.float32))

        # Re-stack 3D finding matrices back into 4D array with finding channel first: (F, X, Y, Z)
        data_ras = np.stack(reoriented_3d_list, axis=0)
        return data_ras, first_ras_nii, first_axcodes

    else:
        # Standard 3D Spatial CT Volume Reorientation
        nii_3d = nib.Nifti1Image(data, affine)
        raw_axcodes = nib.orientations.aff2axcodes(nii_3d.affine)
        ras_nii = nib.as_closest_canonical(nii_3d)
        data_ras = ras_nii.get_fdata(dtype=np.float32)
        return data_ras, ras_nii, raw_axcodes


def save_nifti(pred_array: np.ndarray, out_path: Path, affine: np.ndarray) -> None:
    """
    Signature:
        save_nifti(pred_array: np.ndarray, out_path: Path, affine: np.ndarray) -> None

    Objective:
        Save a 3D or 4D binary prediction array to disk as a NIfTI file with designated affine matrix.
        For 4D segmentation masks, standardizes the tensor layout to Channel-First (F, X, Y, Z) matching the
        official ReXGroundingCT challenge dataset specification and leaderboard evaluation server requirement.

    Inputs:
        pred_array (np.ndarray): Binary prediction mask array (3D or 4D). If 4D with shape (X, Y, Z, F),
                                 it is automatically transposed to (F, X, Y, Z) for challenge compliance.
        out_path (Path): File destination path for NIfTI file.
        affine (np.ndarray): 4x4 affine transformation matrix.

    Outputs:
        None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_array = np.asanyarray(pred_array, dtype=np.uint8)

    # Standardize 4D array layout to Channel-First (F, X, Y, Z) matching official challenge dataset spec
    if pred_array.ndim == 4:
        if pred_array.shape[-1] <= 64:
            sorted_spatial = sorted(pred_array.shape[:3])
            if pred_array.shape[-1] < sorted_spatial[0] or (pred_array.shape[-1] == sorted_spatial[0] and pred_array.shape[-1] < sorted_spatial[1]) or (sorted_spatial[0] == sorted_spatial[1] and sorted_spatial[1] == sorted_spatial[2]):
                pred_array = np.moveaxis(pred_array, -1, 0)

    if affine is None or not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        affine = np.eye(4, dtype=np.float32)

    out_nii = nib.Nifti1Image(pred_array, affine)
    nib.save(out_nii, str(out_path))
