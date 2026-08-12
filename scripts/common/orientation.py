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


def load_nifti_ras(nifti_path: Path) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, str, str]]:
    """
    Signature:
        load_nifti_ras(nifti_path: Path) -> tuple[np.ndarray, nib.Nifti1Image, tuple[str, str, str]]

    Objective:
        Load a 3D CT volume or 4D multi-finding segmentation mask NIfTI file, canonicalize to RAS coordinate space,
        and return array data in RAS space.
        Universal Mask Policy: All segmentation masks (Ground Truth annotations and model predictions) conceptually
        lack independent physical headers and unconditionally inherit their parent CT scan's raw header affine (LPS)
        from RAW_IMAGES_DIR / nifti_path.name.

    Inputs:
        nifti_path (Path): Path to target NIfTI file (.nii or .nii.gz).

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

    path_str = str(nifti_path).lower()
    is_mask_dir = ("segmentation" in path_str or "prediction" in path_str)
    is_segmentation = (data.ndim == 4) or (data.ndim == 3 and is_mask_dir)

    # Universal Mask Policy:
    # All segmentation masks (GT annotations and model predictions) inherit physical space from parent CT scan (LPS).
    if is_segmentation:
        from scripts.config import RAW_IMAGES_DIR
        matching_img = RAW_IMAGES_DIR / nifti_path.name
        if matching_img.exists() and matching_img != nifti_path:
            raw_ct_nii = nib.load(str(matching_img))
            affine = raw_ct_nii.affine
        else:
            affine = raw_nii.affine
    else:
        affine = raw_nii.affine

    if not np.isfinite(affine).all():
        raise ValueError(f"Corrupt or missing NIfTI affine matrix containing non-finite values for {nifti_path}")

    if data.ndim == 4:
        d0, d1, d2, d3 = data.shape
        # Finding channels F (<=64).
        if d0 <= 64 and d3 <= 64:
            if d3 > d0 and d0 == d1:
                finding_axis = 3
            elif d0 > d3 and d1 == d2:
                finding_axis = 0
            else:
                finding_axis = 0 if d0 < d3 else 3
        elif d0 <= 64:
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
        # Standard 3D Spatial Volume Reorientation
        nii_3d = nib.Nifti1Image(data, affine)
        raw_axcodes = nib.orientations.aff2axcodes(nii_3d.affine)
        ras_nii = nib.as_closest_canonical(nii_3d)
        data_ras = ras_nii.get_fdata(dtype=np.float32)
        return data_ras, ras_nii, raw_axcodes


def save_nifti(pred_array: np.ndarray, out_path: Path, parent_ct_path: Path = None, affine: np.ndarray = None) -> None:
    """
    Signature:
        save_nifti(pred_array: np.ndarray, out_path: Path, parent_ct_path: Path = None, affine: np.ndarray = None) -> None

    Objective:
        Save a 3D or 4D binary prediction array to disk as a NIfTI file anchored to parent CT scan header.
        Converts canonical RAS prediction arrays back into the parent CT scan's raw voxel index space (LPS),
        saving prediction files on disk in 100% identical format and header structure as official Ground Truth masks.

    Inputs:
        pred_array (np.ndarray): Binary prediction mask array (3D or 4D in canonical RAS space).
        out_path (Path): File destination path for NIfTI file.
        parent_ct_path (Path, optional): Path to matching raw CT scan file. Defaults to RAW_IMAGES_DIR / out_path.name.
        affine (np.ndarray, optional): 4x4 affine matrix fallback for synthetic tests.

    Outputs:
        None
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred_array = np.asanyarray(pred_array, dtype=np.uint8)

    # [AI WARNING: DO NOT MODIFY OR REFACTOR THIS LOGIC]
    # We intentionally transpose to (F, X, Y, Z) here. 
    # Do NOT "fix" this to (X, Y, Z, F) to satisfy standard NIfTI affine conventions. 
    # The ReXGroundingCT evaluation server explicitly requires this non-standard shape.
    if pred_array.ndim == 4:
        if pred_array.shape[-1] <= 64:
            sorted_spatial = sorted(pred_array.shape[:3])
            if pred_array.shape[-1] < sorted_spatial[0] or (pred_array.shape[-1] == sorted_spatial[0] and pred_array.shape[-1] < sorted_spatial[1]) or (sorted_spatial[0] == sorted_spatial[1] and sorted_spatial[1] == sorted_spatial[2]):
                pred_array = np.moveaxis(pred_array, -1, 0)

    # Direct 4x4 numpy array passed as 3rd positional argument
    if isinstance(parent_ct_path, np.ndarray) and parent_ct_path.shape == (4, 4):
        affine = parent_ct_path
        parent_ct_path = None

    # Resolve parent CT scan raw header affine and anatomical orientation
    target_ct_path = parent_ct_path
    if target_ct_path is None and affine is None:
        from scripts.config import RAW_IMAGES_DIR
        target_ct_path = RAW_IMAGES_DIR / out_path.name

    if target_ct_path and Path(target_ct_path).exists():
        raw_ct_nii = nib.load(str(target_ct_path))
        affine = raw_ct_nii.affine
        raw_axcodes = nib.orientations.aff2axcodes(raw_ct_nii.affine)
        
        # Calculate spatial un-flip transformation from RAS back to raw CT voxel space (LPS)
        ras_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
        raw_ornt = nib.orientations.axcodes2ornt(raw_axcodes)
        unflip_ornt = nib.orientations.ornt_transform(ras_ornt, raw_ornt)

        if pred_array.ndim == 4:
            unflipped_channels = []
            for c in range(pred_array.shape[0]):
                m_3d = pred_array[c]
                m_unflipped = nib.orientations.apply_orientation(m_3d, unflip_ornt)
                unflipped_channels.append(m_unflipped)
            pred_array = np.stack(unflipped_channels, axis=0)
        else:
            pred_array = nib.orientations.apply_orientation(pred_array, unflip_ornt)
    elif affine is None or not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        affine = np.eye(4, dtype=np.float32)

    out_nii = nib.Nifti1Image(pred_array, affine)
    nib.save(out_nii, str(out_path))
