"""
===============================================================================
SCRIPT:         Shared 6-Slice 2D Multi-Plane Cross-Sectional Visualizer Engine
LOCATION:       scripts/analysis/plot_single_case.py
OBJECTIVE:      Generates 6 high-contrast 2D CT slice overlays across 3 orthogonal 
                planes (2 Axial, 2 Coronal, 2 Sagittal) per pathology finding category.
                Selects the slice with the largest Ground Truth area and the slice 
                with the largest Prediction area for each anatomical axis.
                Exports high-contrast PNG figures and 3D NIfTI bundles to scan_visualizations/.
USAGE:          python scripts/analysis/plot_single_case.py --scan_id train_19891_a_2
===============================================================================
"""
import os
import sys
import json
import random
import zipfile
import tempfile
import textwrap
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import RAW_IMAGES_DIR, RAW_MASKS_DIR, PREDICTIONS_DIR, DATASET_JSON, CATEGORY_MAP, SCRATCH_DIR, VISUALIZATIONS_DIR
from scripts.common.orientation import load_nifti_ras, save_nifti

WINDOW_PRESETS = {
    'lung': (-1000.0, 200.0),
    'soft_tissue': (-160.0, 240.0),
    'bone': (-200.0, 1000.0),
    'chest_default': (-1000.0, 400.0)
}

CATEGORY_WINDOW_MAP = {
    '1a': 'lung', '1b': 'lung', '1c': 'lung', '1d': 'lung', '1e': 'lung', '1f': 'lung',
    '2a': 'lung', '2b': 'lung', '2c': 'lung', '2f': 'lung', '2g': 'lung',
    '2d': 'soft_tissue', '2e': 'soft_tissue', '2h': 'soft_tissue'
}

def resolve_contrast_window(cat_code: str, img_data: np.ndarray, user_preset: str = 'auto') -> tuple[float, float]:
    """
    resolve_contrast_window(cat_code: str, img_data: np.ndarray, user_preset: str = 'auto') -> tuple[float, float]
    Resolves (vmin, vmax) intensity contrast window bounds for a CT volume based on category preset or user override.

    Args:
        cat_code (str): Pathology category code (e.g. '1a', '2e').
        img_data (np.ndarray): 3D CT volume array.
        user_preset (str): Specified window preset ('auto', 'lung', 'soft_tissue', 'bone', 'chest_default').

    Returns:
        tuple[float, float]: Resolved (vmin, vmax) bounds.
    """
    if user_preset != 'auto' and user_preset in WINDOW_PRESETS:
        target_preset = user_preset
    else:
        target_preset = CATEGORY_WINDOW_MAP.get(cat_code, 'chest_default')

    base_vmin, base_vmax = WINDOW_PRESETS[target_preset]

    img_min, img_max = float(img_data.min()), float(img_data.max())
    if img_min >= -1050.0 and img_max <= 3500.0 and img_min < -500.0:
        return base_vmin, base_vmax
    else:
        p1 = float(np.percentile(img_data, 1.0))
        p99 = float(np.percentile(img_data, 99.0))
        if p99 > p1:
            hu_range = 1400.0
            vmin_ratio = (base_vmin + 1000.0) / hu_range
            vmax_ratio = (base_vmax + 1000.0) / hu_range
            vmin = p1 + vmin_ratio * (p99 - p1)
            vmax = p1 + vmax_ratio * (p99 - p1)
            return float(vmin), float(vmax)
        else:
            return img_min, img_max

def load_canonical_ras(nifti_path: Path, override_affine: np.ndarray = None) -> np.ndarray:
    """
    load_canonical_ras(nifti_path: Path, override_affine: np.ndarray = None) -> np.ndarray
    Loads a NIfTI file in canonical RAS space using centralized spatial engine scripts.common.orientation.

    Args:
        nifti_path (Path): Path to target .nii.gz file.
        override_affine (np.ndarray, optional): 4x4 affine matrix to override NIfTI header affine before reorienting.

    Returns:
        np.ndarray: Reoriented RAS data array with 4D shape (channels, X, Y, Z).
    """
    data_ras, _, _ = load_nifti_ras(nifti_path, ref_affine=override_affine)
    if data_ras.ndim == 3:
        data_ras = np.expand_dims(data_ras, axis=0)
    return data_ras

def load_dataset_metadata() -> dict:
    """
    Signature:
        load_dataset_metadata() -> dict

    Objective:
        Reads dataset.json file and constructs a lookup map from scan_id to finding metadata.

    Inputs:
        None

    Outputs:
        dict: Mapping from scan_id (e.g. 'train_19891_a_2') to item dict containing findings & categories.
    """
    meta_map = {}
    if DATASET_JSON.exists():
        with open(DATASET_JSON, 'r') as f:
            data = json.load(f)
        for split in ['train', 'val', 'test']:
            if split in data:
                for item in data[split]:
                    name = item['name'].replace('.nii.gz', '')
                    meta_map[name] = item
    return meta_map

def compute_dice_and_stats(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """
    compute_dice_and_stats(gt_mask: np.ndarray, pred_mask: np.ndarray) -> dict
    Calculates Dice score, IoU, voxel counts, and centroid distance between 3D binary masks.

    Args:
        gt_mask (np.ndarray): 3D binary ground truth mask.
        pred_mask (np.ndarray): 3D binary prediction mask.

    Returns:
        dict: Summary metrics dictionary containing dice, iou, hit_status, gt_voxels, pred_voxels, intersection.
    """
    gt_bool = (gt_mask > 0)
    pred_bool = (pred_mask > 0)

    gt_voxels = int(gt_bool.sum())
    pred_voxels = int(pred_bool.sum())
    intersection = int(np.logical_and(gt_bool, pred_bool).sum())
    union = int(np.logical_or(gt_bool, pred_bool).sum())

    denom = gt_voxels + pred_voxels
    dice = (2.0 * intersection / denom) if denom > 0 else (1.0 if intersection == 0 else 0.0)
    iou = (intersection / union) if union > 0 else (1.0 if union == 0 else 0.0)
    hit_status = "HIT (DSC ≥ 0.1)" if dice >= 0.1 else "MISS (DSC < 0.1)"

    if gt_voxels > 0 and pred_voxels > 0:
        gt_c = np.array(np.where(gt_bool)).mean(axis=1)
        pred_c = np.array(np.where(pred_bool)).mean(axis=1)
        centroid_dist = np.linalg.norm(gt_c - pred_c)
        dist_str = f"{centroid_dist:.1f} vox"
    else:
        dist_str = "N/A"

    return {
        'dice': dice,
        'iou': iou,
        'hit_status': hit_status,
        'gt_voxels': gt_voxels,
        'pred_voxels': pred_voxels,
        'intersection': intersection,
        'centroid_dist_str': dist_str
    }

def create_scan_zip_bundle(scan_id: str, img_path: Path, gt_path: Path, pred_path: Path, zip_out_path: Path) -> Path:
    """
    create_scan_zip_bundle(scan_id: str, img_path: Path, gt_path: Path, pred_path: Path, zip_out_path: Path) -> Path
    Creates a ZIP archive containing the 3D CT volume and 3D-dimension-matched segmentation mask NIfTI files
    (converting multi-channel 4D masks to 3D shape-matched volumes for immediate visualizer compatibility).

    Args:
        scan_id (str): Target CT scan identifier.
        img_path (Path): Path to raw CT image volume .nii.gz file.
        gt_path (Path): Path to ground truth segmentation mask .nii.gz file.
        pred_path (Path): Path to predicted segmentation mask .nii.gz file.
        zip_out_path (Path): Target output path for the created ZIP archive.

    Returns:
        Path: Path to saved ZIP archive file.
    """
    img_data, img_nii_ras, _ = load_nifti_ras(img_path) if img_path.exists() else (None, None, None)
    img_affine = img_nii_ras.affine if img_nii_ras is not None else None

    def _convert_4d_to_3d_nii(nii_path: Path, base_affine: np.ndarray):
        """
        Signature:
            _convert_4d_to_3d_nii(nii_path: Path, base_affine: np.ndarray) -> tuple

        Objective:
            Convert a 4D finding mask volume into a 3D labeled composite array and per-finding dict.

        Inputs:
            nii_path (Path): Path to 4D NIfTI mask file.
            base_affine (np.ndarray): Target reference affine matrix.

        Outputs:
            tuple: (combined_3d, per_finding_3d, affine)
        """
        if not nii_path.exists():
            return None, {}, None
        data, nii_ras, _ = load_nifti_ras(nii_path)
        affine = base_affine if base_affine is not None else nii_ras.affine
        
        if data.ndim == 3:
            combined_3d = (data > 0).astype(np.uint8)
            per_finding_3d = {1: combined_3d}
        elif data.ndim == 4:
            num_f = data.shape[0]
            shape_3d = data.shape[1:]
            combined_3d = np.zeros(shape_3d, dtype=np.uint8)
            per_finding_3d = {}
            for i in range(num_f):
                f_mask = (data[i] > 0).astype(np.uint8)
                combined_3d[f_mask > 0] = i + 1
                per_finding_3d[i + 1] = f_mask
        else:
            return None, {}, None

        return combined_3d, per_finding_3d, affine

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_out_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
            if img_path.exists():
                zipf.write(img_path, arcname=f"{scan_id}.nii.gz")

            gt_combined, gt_findings, affine_gt = _convert_4d_to_3d_nii(gt_path, img_affine)
            if gt_combined is not None:
                gt_3d_file = tmp_path / f"{scan_id}_gt.nii.gz"
                save_nifti(gt_combined, gt_3d_file, affine=affine_gt)
                zipf.write(gt_3d_file, arcname=f"{scan_id}_gt.nii.gz")

                if len(gt_findings) > 1:
                    for f_idx, f_mask in gt_findings.items():
                        f_file = tmp_path / f"{scan_id}_gt_finding_{f_idx}.nii.gz"
                        save_nifti(f_mask, f_file, affine=affine_gt)
                        zipf.write(f_file, arcname=f"{scan_id}_gt_finding_{f_idx}.nii.gz")

            pred_combined, pred_findings, affine_pred = _convert_4d_to_3d_nii(pred_path, img_affine)
            if pred_combined is not None:
                pred_3d_file = tmp_path / f"{scan_id}_pred.nii.gz"
                save_nifti(pred_combined, pred_3d_file, affine=affine_pred)
                zipf.write(pred_3d_file, arcname=f"{scan_id}_pred.nii.gz")

                if len(pred_findings) > 1:
                    for f_idx, f_mask in pred_findings.items():
                        f_file = tmp_path / f"{scan_id}_pred_finding_{f_idx}.nii.gz"
                        save_nifti(f_mask, f_file, affine=affine_pred)
                        zipf.write(f_file, arcname=f"{scan_id}_pred_finding_{f_idx}.nii.gz")

            if gt_path.exists():
                zipf.write(gt_path, arcname=f"{scan_id}_gt_4d.nii.gz")
            if pred_path.exists():
                zipf.write(pred_path, arcname=f"{scan_id}_pred_4d.nii.gz")

    return zip_out_path

def select_gt_and_pred_max_slices(gt_mask: np.ndarray, pred_mask: np.ndarray, axis: int) -> tuple[int, int, str, str]:
    """
    select_gt_and_pred_max_slices(gt_mask: np.ndarray, pred_mask: np.ndarray, axis: int) -> tuple[int, int, str, str]
    Selects the slice index with the largest Ground Truth area and the slice index with the
    largest Prediction area along a specified volume axis (0=X/Sagittal, 1=Y/Coronal, 2=Z/Axial).

    Args:
        gt_mask (np.ndarray): 3D boolean Ground Truth mask array.
        pred_mask (np.ndarray): 3D boolean Prediction mask array.
        axis (int): Target volume axis (0=X, 1=Y, 2=Z).

    Returns:
        tuple[int, int, str, str]: (slice_gt_idx, slice_pred_idx, label_gt, label_pred)
    """
    max_len = gt_mask.shape[axis]
    sum_axes = tuple(i for i in range(3) if i != axis)

    gt_counts = gt_mask.sum(axis=sum_axes)     # 1D GT voxel counts per slice
    pred_counts = pred_mask.sum(axis=sum_axes) # 1D Pred voxel counts per slice

    # Determine Max GT Slice
    gt_indices = np.where(gt_counts > 0)[0]
    if len(gt_indices) > 0:
        s_gt = int(gt_indices[np.argmax(gt_counts[gt_indices])])
        label_gt = "Max GT"
    else:
        s_gt = max_len // 2
        label_gt = "Center"

    # Determine Max Pred Slice
    pred_indices = np.where(pred_counts > 0)[0]
    if len(pred_indices) > 0:
        s_pred = int(pred_indices[np.argmax(pred_counts[pred_indices])])
        label_pred = "Max Pred"
    else:
        s_pred = max_len // 2
        label_pred = "Center"

    # Handle overlapping slice indices to ensure 2 distinct informative slice views
    if s_gt == s_pred:
        if len(pred_indices) > 1:
            sorted_pred = pred_indices[np.argsort(pred_counts[pred_indices])]
            s_pred = int(sorted_pred[-2])
            label_pred = "2nd Pred"
        elif len(gt_indices) > 1:
            sorted_gt = gt_indices[np.argsort(gt_counts[gt_indices])]
            s_pred = int(sorted_gt[-2])
            label_pred = "2nd GT"
        else:
            s_pred = min(max_len - 1, s_gt + 1) if s_gt < max_len - 1 else max(0, s_gt - 1)
            label_pred = "Adjacent"

    return s_gt, s_pred, label_gt, label_pred

def plot_single_scan_case(scan_id: str, meta_map: dict, pred_dir: Path, out_dir: Path, fix_gt_affine: bool = True, window_preset: str = 'auto') -> Path:
    """
    plot_single_scan_case(scan_id: str, meta_map: dict, pred_dir: Path, out_dir: Path, fix_gt_affine: bool = True, window_preset: str = 'auto') -> Path
    Generates a per-pathology multi-row 2D cross-sectional visualization figure for a CT scan volume.
    Each active pathology gets its own row featuring 6 2D CT slice overlays across 3 orthogonal planes
    (Max GT and Max Pred slice per plane for Axial, Coronal, Sagittal) + 1 Pathology Statistics & Prompt Card.

    Args:
        scan_id (str): Target CT scan identifier.
        meta_map (dict): Dataset metadata mapping.
        pred_dir (Path): Predictions directory.
        out_dir (Path): Output directory for saved PNG images.
        fix_gt_affine (bool): If True, overrides uninformative GT mask header affine with raw image affine before canonical reorientation.
        window_preset (str): Specified contrast window preset ('auto', 'lung', 'soft_tissue', 'bone', 'chest_default').

    Returns:
        Path: Path to saved output image file.
    """
    img_path = RAW_IMAGES_DIR / f"{scan_id}.nii.gz"
    gt_path = RAW_MASKS_DIR / f"{scan_id}.nii.gz"
    pred_path = pred_dir / f"{scan_id}.nii.gz"

    img_4d, img_nii_ras, _ = load_nifti_ras(img_path)
    raw_img_affine = img_nii_ras.affine if img_nii_ras is not None else None
    if img_4d.ndim == 3:
        img_4d = np.expand_dims(img_4d, axis=0)

    # Extract physical voxel spacing (zooms) for true anatomical millimeter proportions
    raw_zooms = img_nii_ras.header.get_zooms()[:3]
    dx = float(raw_zooms[0]) if len(raw_zooms) > 0 and float(raw_zooms[0]) > 0 else 1.0
    dy = float(raw_zooms[1]) if len(raw_zooms) > 1 and float(raw_zooms[1]) > 0 else 1.0
    dz = float(raw_zooms[2]) if len(raw_zooms) > 2 and float(raw_zooms[2]) > 0 else 1.0

    aspect_axial = dy / dx
    aspect_coronal = dz / dx
    aspect_sagittal = dz / dy

    target_ref_affine = raw_img_affine if fix_gt_affine else None
    gt_4d = load_canonical_ras(gt_path, override_affine=target_ref_affine)
    pred_4d = load_canonical_ras(pred_path, override_affine=target_ref_affine)

    img_data = img_4d[0] # (X, Y, Z)
    num_findings = gt_4d.shape[0]

    scan_meta = meta_map.get(scan_id, {})
    categories_meta = scan_meta.get('categories', {})
    findings_text_meta = scan_meta.get('findings', {})

    # Gather finding statistics
    findings_stats = []
    active_indices = []

    for f_idx in range(num_findings):
        gt_f = gt_4d[f_idx]
        pred_f = pred_4d[f_idx] if f_idx < pred_4d.shape[0] else np.zeros_like(gt_f)

        stats = compute_dice_and_stats(gt_f, pred_f)
        cat_code = categories_meta.get(str(f_idx), "Unknown")
        cat_name = CATEGORY_MAP.get(cat_code, f"Category {cat_code}")
        prompt_text = findings_text_meta.get(str(f_idx), "")

        stats['f_idx'] = f_idx
        stats['cat_code'] = cat_code
        stats['cat_name'] = cat_name
        stats['prompt'] = prompt_text
        stats['gt_mask'] = (gt_f > 0)
        stats['pred_mask'] = (pred_f > 0)

        findings_stats.append(stats)
        if stats['gt_voxels'] > 0 or stats['pred_voxels'] > 0:
            active_indices.append(f_idx)

    if not active_indices:
        active_indices = list(range(num_findings))

    nx, ny, nz = img_data.shape
    num_rows = len(active_indices)
    fig = plt.figure(figsize=(27, 4.5 * num_rows), dpi=200)

    for row_idx, f_idx in enumerate(active_indices):
        f_item = findings_stats[f_idx]
        gt_mask = f_item['gt_mask']
        pred_mask = f_item['pred_mask']

        # Resolve category-specific or user-overridden contrast window
        row_vmin, row_vmax = resolve_contrast_window(f_item['cat_code'], img_data, user_preset=window_preset)

        # Select Max GT and Max Pred slice for each axis
        z_gt, z_pred, lbl_z_gt, lbl_z_pred = select_gt_and_pred_max_slices(gt_mask, pred_mask, axis=2) # Axial (Z)
        y_gt, y_pred, lbl_y_gt, lbl_y_pred = select_gt_and_pred_max_slices(gt_mask, pred_mask, axis=1) # Coronal (Y)
        x_gt, x_pred, lbl_x_gt, lbl_x_pred = select_gt_and_pred_max_slices(gt_mask, pred_mask, axis=0) # Sagittal (X)

        slices_config = [
            # (panel_idx, ct_slice, gt_s, pred_s, title, aspect)
            (1, img_data[:, :, z_gt].T, gt_mask[:, :, z_gt].T, pred_mask[:, :, z_gt].T, f"Axial [{lbl_z_gt}] (Z={z_gt})", aspect_axial),
            (2, img_data[:, :, z_pred].T, gt_mask[:, :, z_pred].T, pred_mask[:, :, z_pred].T, f"Axial [{lbl_z_pred}] (Z={z_pred})", aspect_axial),
            (3, img_data[:, y_gt, :].T, gt_mask[:, y_gt, :].T, pred_mask[:, y_gt, :].T, f"Coronal [{lbl_y_gt}] (Y={y_gt})", aspect_coronal),
            (4, img_data[:, y_pred, :].T, gt_mask[:, y_pred, :].T, pred_mask[:, y_pred, :].T, f"Coronal [{lbl_y_pred}] (Y={y_pred})", aspect_coronal),
            (5, img_data[x_gt, :, :].T, gt_mask[x_gt, :, :].T, pred_mask[x_gt, :, :].T, f"Sagittal [{lbl_x_gt}] (X={x_gt})", aspect_sagittal),
            (6, img_data[x_pred, :, :].T, gt_mask[x_pred, :, :].T, pred_mask[x_pred, :, :].T, f"Sagittal [{lbl_x_pred}] (X={x_pred})", aspect_sagittal),
        ]

        base_sub = row_idx * 7

        # Helper to plot 2D slice overlay
        def plot_2d_pathology_slice(ax, ct_slice, gt_s, pred_s, title_str, aspect='auto'):
            """
            Signature:
                plot_2d_pathology_slice(ax, ct_slice, gt_s, pred_s, title_str, aspect) -> None

            Objective:
                Render a 2D CT slice with colored GT (green) and predicted (red) segmentation overlays.

            Inputs:
                ax (matplotlib.axes.Axes): Target subplot axis.
                ct_slice (np.ndarray): 2D CT image slice array.
                gt_s (np.ndarray): 2D ground truth binary mask slice.
                pred_s (np.ndarray): 2D predicted binary mask slice.
                title_str (str): Subplot title string.
                aspect (str|float): Aspect ratio for imshow.

            Outputs:
                None
            """
            ct_norm = np.clip(ct_slice, row_vmin, row_vmax)
            ct_norm = (ct_norm - row_vmin) / (row_vmax - row_vmin)

            gt_overlay = np.zeros((*ct_norm.shape, 4), dtype=np.float32)
            gt_overlay[..., 1] = 1.0 # Lime Green
            gt_overlay[..., 3] = gt_s.astype(np.float32) * 0.65

            pred_overlay = np.zeros((*ct_norm.shape, 4), dtype=np.float32)
            pred_overlay[..., 0] = 1.0 # Crimson Red
            pred_overlay[..., 3] = pred_s.astype(np.float32) * 0.55

            gt_overlay = np.clip(gt_overlay, 0.0, 1.0)
            pred_overlay = np.clip(pred_overlay, 0.0, 1.0)

            ax.imshow(ct_norm, cmap='gray', origin='lower', aspect=aspect)
            ax.imshow(gt_overlay, origin='lower', aspect=aspect)
            ax.imshow(pred_overlay, origin='lower', aspect=aspect)
            ax.set_title(title_str, fontsize=9.5, fontweight='bold')
            ax.axis('off')

        # Plot 6 Slices
        for col_idx, ct_s, gt_s, pred_s, title_str, aspect_val in slices_config:
            ax = fig.add_subplot(num_rows, 7, base_sub + col_idx)
            plot_2d_pathology_slice(ax, ct_s, gt_s, pred_s, title_str, aspect=aspect_val)

        # Panel 7: Finding Statistics & Prompt Card
        ax_stats = fig.add_subplot(num_rows, 7, base_sub + 7)
        ax_stats.axis('off')
        if row_idx == 0:
            ax_stats.set_title("Pathology & Text Prompt", fontsize=10, fontweight='bold')

        box_color = 'lightgreen' if f_item['dice'] >= 0.1 else ('salmon' if f_item['gt_voxels'] > 0 else 'lightgray')
        
        prompt_str = f_item['prompt'] if f_item['prompt'] else "No prompt text provided."
        prompt_wrapped = textwrap.fill(prompt_str, width=28)

        card_text = (
            f"Finding {f_idx} [{f_item['cat_code']}]\n"
            f"Category: {f_item['cat_name']}\n"
            f"---------------------------\n"
            f"Text Prompt:\n\"{prompt_wrapped}\"\n"
            f"---------------------------\n"
            f"Dice DSC: {f_item['dice']:.4f}\n"
            f"IoU Score: {f_item['iou']:.4f}\n"
            f"Status: {f_item['hit_status']}\n"
            f"GT Voxels: {f_item['gt_voxels']:,}\n"
            f"Pred Voxels: {f_item['pred_voxels']:,}\n"
            f"Centroid Shift: {f_item['centroid_dist_str']}"
        )

        ax_stats.text(0.02, 0.5, card_text, transform=ax_stats.transAxes, fontsize=8.0, fontweight='bold', va='center',
                      bbox=dict(boxstyle='round,pad=0.6', facecolor=box_color, alpha=0.45))

    # Add Legend at bottom
    green_patch = mpatches.Patch(color='lime', alpha=0.7, label='Ground Truth Mask')
    red_patch = mpatches.Patch(color='crimson', alpha=0.7, label='Prediction Mask')
    fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, frameon=True, fontsize=10)

    plt.suptitle(f"Per-Pathology 6-Slice 2D CT Cross-Sectional Visualization | Scan: {scan_id}", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96], w_pad=2.2, h_pad=3.0)

    scan_subfolder = out_dir / scan_id
    scan_subfolder.mkdir(parents=True, exist_ok=True)
    out_path = scan_subfolder / f"pathology_row_scan_{scan_id}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

    zip_out_path = scan_subfolder / f"{scan_id}_bundle.zip"
    create_scan_zip_bundle(scan_id, img_path, gt_path, pred_path, zip_out_path)

    print(f"[SUCCESS] Saved figure to: {out_path.resolve()}", flush=True)
    print(f"[SUCCESS] Saved downloadable data bundle ZIP to: {zip_out_path.resolve()}", flush=True)
    return out_path

def main():
    """
    main() -> None
    CLI entry point to generate 6-slice 2D cross-sectional visualizer figures
    for one or multiple CT scan volumes.
    """
    parser = argparse.ArgumentParser(description="Per-Pathology 6-Slice 2D Cross-Sectional CT Visualizer Generator")
    parser.add_argument("--scan_id", type=str, default=None, help="Specific scan ID or comma-separated IDs (e.g. train_19891_a_2,train_13098_a_2).")
    parser.add_argument("--pred_subdir", type=str, default="phase_2a_rule_based", help="Subdirectory name inside predictions directory (default: phase_2a_rule_based).")
    parser.add_argument("--num_scans", type=int, default=1, help="Number of random scans to plot if --scan_id is omitted.")
    parser.add_argument("--no_fix_gt_affine", action="store_true", help="If set, disables automatic GT mask header affine repair with raw CT image DICOM affine.")
    parser.add_argument("--window_preset", type=str, default="auto", choices=["auto", "lung", "soft_tissue", "bone", "chest_default"], help="Contrast window preset for CT rendering (default: auto).")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sampling.")
    parser.add_argument("--out_dir", type=str, default=str(VISUALIZATIONS_DIR), help="Directory to save generated PNG images (default: scan_visualizations/).")
    args = parser.parse_args()

    # Default fix_gt_affine to True: repairs raw GT mask headers (np.eye(4)) using parent CT image DICOM affine
    fix_gt_affine = not args.no_fix_gt_affine

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    meta_map = load_dataset_metadata()

    mask_files = list(RAW_MASKS_DIR.glob("*.nii.gz"))
    if not mask_files:
        raise FileNotFoundError(f"No mask NIfTI files found in {RAW_MASKS_DIR}")

    pred_dir = PREDICTIONS_DIR / args.pred_subdir
    
    candidate_ids = []
    for f in mask_files:
        scan_name = f.name.replace('.nii.gz', '')
        if (pred_dir / f.name).exists():
            candidate_ids.append(scan_name)

    if not candidate_ids:
        raise FileNotFoundError(f"No prediction files matching mask files found in {pred_dir}")

    # Determine target scan IDs
    if args.scan_id:
        target_ids = [s.strip() for s in args.scan_id.split(',') if s.strip()]
    else:
        num_to_pick = min(args.num_scans, len(candidate_ids))
        target_ids = random.sample(candidate_ids, num_to_pick)

    print(f"[INFO] Selected {len(target_ids)} scan(s) to process: {target_ids}", flush=True)

    saved_paths = []
    out_dir = Path(args.out_dir)

    for i, scan_id in enumerate(target_ids, 1):
        print(f"\n--- Processing Scan [{i}/{len(target_ids)}]: '{scan_id}' ---", flush=True)
        out_p = plot_single_scan_case(
            scan_id, meta_map, pred_dir, out_dir,
            fix_gt_affine=fix_gt_affine,
            window_preset=args.window_preset
        )
        saved_paths.append(out_p)

    print(f"\n[COMPLETE] Successfully generated {len(saved_paths)} visualizer figures in '{out_dir.resolve()}'.", flush=True)

if __name__ == "__main__":
    main()
