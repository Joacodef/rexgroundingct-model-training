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
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import RAW_IMAGES_DIR, RAW_MASKS_DIR, PREDICTIONS_DIR, DATASET_JSON, CATEGORY_MAP, SCRATCH_DIR, VISUALIZATIONS_DIR

def load_canonical_ras(nifti_path: Path, override_affine: np.ndarray = None) -> np.ndarray:
    """
    load_canonical_ras(nifti_path: Path, override_affine: np.ndarray = None) -> np.ndarray
    Loads a NIfTI file, optionally overrides its header affine matrix, reorients it to canonical RAS space,
    and returns the 4D/3D float32 array.

    Args:
        nifti_path (Path): Path to target .nii.gz file.
        override_affine (np.ndarray, optional): 4x4 affine matrix to override NIfTI header affine before reorienting.

    Returns:
        np.ndarray: Reoriented RAS data array with 4D shape (channels, X, Y, Z).
    """
    nii = nib.load(str(nifti_path))
    if override_affine is not None:
        nii = nib.Nifti1Image(nii.get_fdata(), override_affine)
        
    nii_ras = nib.as_closest_canonical(nii)
    data = np.asanyarray(nii_ras.dataobj).astype(np.float32)

    if data.ndim == 3:
        data_ras = np.expand_dims(data, axis=0)
    elif data.ndim == 4:
        if data.shape[-1] < np.min(data.shape[:3]):
            data_ras = np.moveaxis(data, -1, 0)
        else:
            data_ras = data
            
    return data_ras

def load_dataset_metadata() -> dict:
    """
    load_dataset_metadata() -> dict
    Reads dataset.json file and constructs a lookup map from scan_id to finding metadata.

    Returns:
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
    img_nii = nib.load(str(img_path)) if img_path.exists() else None
    img_affine = img_nii.affine if img_nii is not None else None

    def _convert_4d_to_3d_nii(nii_path: Path, base_affine: np.ndarray):
        """
        _convert_4d_to_3d_nii(nii_path: Path, base_affine: np.ndarray) -> tuple[nib.Nifti1Image | None, dict, np.ndarray | None]
        Loads a 4D/3D NIfTI file, collapses 4D channels into a single 3D multi-label mask,
        and extracts per-finding 3D binary masks.

        Args:
            nii_path (Path): Target NIfTI file path.
            base_affine (np.ndarray): Base image 4x4 affine matrix for coordinate alignment.

        Returns:
            tuple: (combined_3d_nii, per_finding_3d_dict, affine_matrix)
        """
        if not nii_path.exists():
            return None, {}, None
        nii = nib.load(str(nii_path))
        data = np.asanyarray(nii.dataobj)
        affine = base_affine if base_affine is not None else nii.affine
        
        if data.ndim == 3:
            combined_3d = (data > 0).astype(np.uint8)
            per_finding_3d = {1: combined_3d}
        elif data.ndim == 4:
            if data.shape[0] < np.min(data.shape[1:]):
                num_f = data.shape[0]
                shape_3d = data.shape[1:]
                combined_3d = np.zeros(shape_3d, dtype=np.uint8)
                per_finding_3d = {}
                for i in range(num_f):
                    f_mask = (data[i] > 0).astype(np.uint8)
                    combined_3d[f_mask > 0] = i + 1
                    per_finding_3d[i + 1] = f_mask
            else:
                num_f = data.shape[-1]
                shape_3d = data.shape[:3]
                combined_3d = np.zeros(shape_3d, dtype=np.uint8)
                per_finding_3d = {}
                for i in range(num_f):
                    f_mask = (data[..., i] > 0).astype(np.uint8)
                    combined_3d[f_mask > 0] = i + 1
                    per_finding_3d[i + 1] = f_mask
        else:
            return None, {}, None

        comb_nii = nib.Nifti1Image(combined_3d, affine)
        return comb_nii, per_finding_3d, affine

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_out_path, 'w', compression=zipfile.ZIP_STORED) as zipf:
            # 1. Add main 3D CT volume
            if img_path.exists():
                zipf.write(img_path, arcname=f"{scan_id}.nii.gz")

            # 2. Add 3D-dimension-matched GT masks
            gt_3d_nii, gt_findings, affine_gt = _convert_4d_to_3d_nii(gt_path, img_affine)
            if gt_3d_nii is not None:
                gt_3d_file = tmp_path / f"{scan_id}_gt.nii.gz"
                nib.save(gt_3d_nii, str(gt_3d_file))
                zipf.write(gt_3d_file, arcname=f"{scan_id}_gt.nii.gz")

                if len(gt_findings) > 1:
                    for f_idx, f_mask in gt_findings.items():
                        f_nii = nib.Nifti1Image(f_mask, affine_gt)
                        f_file = tmp_path / f"{scan_id}_gt_finding_{f_idx}.nii.gz"
                        nib.save(f_nii, str(f_file))
                        zipf.write(f_file, arcname=f"{scan_id}_gt_finding_{f_idx}.nii.gz")

            # 3. Add 3D-dimension-matched Pred masks
            pred_3d_nii, pred_findings, affine_pred = _convert_4d_to_3d_nii(pred_path, img_affine)
            if pred_3d_nii is not None:
                pred_3d_file = tmp_path / f"{scan_id}_pred.nii.gz"
                nib.save(pred_3d_nii, str(pred_3d_file))
                zipf.write(pred_3d_file, arcname=f"{scan_id}_pred.nii.gz")

                if len(pred_findings) > 1:
                    for f_idx, f_mask in pred_findings.items():
                        f_nii = nib.Nifti1Image(f_mask, affine_pred)
                        f_file = tmp_path / f"{scan_id}_pred_finding_{f_idx}.nii.gz"
                        nib.save(f_nii, str(f_file))
                        zipf.write(f_file, arcname=f"{scan_id}_pred_finding_{f_idx}.nii.gz")

            # 4. Include raw 4D masks for reference
            if gt_path.exists():
                zipf.write(gt_path, arcname=f"{scan_id}_gt_4d.nii.gz")
            if pred_path.exists():
                zipf.write(pred_path, arcname=f"{scan_id}_pred_4d.nii.gz")

    return zip_out_path

def plot_single_scan_case(scan_id: str, meta_map: dict, pred_dir: Path, out_dir: Path, fix_gt_affine: bool = False, render_body: bool = True, body_alpha: float = 0.08, body_step_size: int = 4) -> Path:
    """
    plot_single_scan_case(scan_id: str, meta_map: dict, pred_dir: Path, out_dir: Path, fix_gt_affine: bool = False, render_body: bool = True, body_alpha: float = 0.08, body_step_size: int = 4) -> Path
    Generates a per-pathology multi-row visualization figure for a CT scan volume.
    Each active pathology gets its own dedicated row featuring 4 high-contrast 3D rotational viewports (with semi-transparent CT torso contour),
    3 2D slice overlays (Axial, Coronal, Sagittal through pathology centroid), and 1 pathology statistics card including clinical text.

    Args:
        scan_id (str): Target CT scan identifier.
        meta_map (dict): Dataset metadata mapping.
        pred_dir (Path): Predictions directory.
        out_dir (Path): Output directory for saved PNG images.
        fix_gt_affine (bool): If True, overrides GT mask header affine with raw image affine before canonical reorientation.
        render_body (bool): If True, precomputes and renders a semi-transparent thoracic body contour in 3D.
        body_alpha (float): Opacity value for the 3D thoracic body mesh (default: 0.08).
        body_step_size (int): Marching cubes step size for background body mesh decimation (default: 4).

    Returns:
        Path: Path to saved output image file.
    """
    img_path = RAW_IMAGES_DIR / f"{scan_id}.nii.gz"
    gt_path = RAW_MASKS_DIR / f"{scan_id}.nii.gz"
    pred_path = pred_dir / f"{scan_id}.nii.gz"

    img_nii = nib.load(str(img_path))
    img_affine = img_nii.affine

    img_4d = load_canonical_ras(img_path)
    gt_4d = load_canonical_ras(gt_path, override_affine=img_affine if fix_gt_affine else None)
    pred_4d = load_canonical_ras(pred_path)

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

    # Pre-compute semi-transparent 3D thoracic body contour isosurface mesh from CT volume
    body_mesh_verts = None
    body_mesh_faces = None
    if render_body:
        try:
            # Downsample CT volume by 4x (64x voxel reduction) for instant Matplotlib 3D rendering
            img_sub = img_data[::4, ::4, ::4]
            non_bg = img_sub[img_sub > 100]
            body_level = float(np.percentile(non_bg, 10)) if len(non_bg) > 0 else 500.0
            
            body_verts, body_faces, _, _ = marching_cubes(img_sub, level=body_level, step_size=body_step_size)
            body_mesh_verts = body_verts * 4.0  # Scale back to full volume coordinates
            body_mesh_faces = body_faces
        except Exception as e:
            print(f"[WARNING] Body isosurface generation failed for {scan_id}: {e}", flush=True)

    num_rows = len(active_indices)
    fig = plt.figure(figsize=(25, 5.2 * num_rows), dpi=200)

    # 4 distinct rotational 3D perspective angles around the volume
    views_3d = [
        (25, 45, "3D ISO (Ant-Right)"),
        (25, 135, "3D ISO (Ant-Left)"),
        (25, 225, "3D ISO (Post-Left)"),
        (25, 315, "3D ISO (Post-Right)")
    ]

    vmin, vmax = -1000.0, 400.0

    for row_idx, f_idx in enumerate(active_indices):
        f_item = findings_stats[f_idx]
        gt_mask = f_item['gt_mask']
        pred_mask = f_item['pred_mask']

        # Determine pathology centroid for 2D slice positioning
        if gt_mask.sum() > 0:
            x_idx, y_idx, z_idx = np.where(gt_mask)
            x_mid, y_mid, z_mid = int(np.median(x_idx)), int(np.median(y_idx)), int(np.median(z_idx))
        elif pred_mask.sum() > 0:
            x_idx, y_idx, z_idx = np.where(pred_mask)
            x_mid, y_mid, z_mid = int(np.median(x_idx)), int(np.median(y_idx)), int(np.median(z_idx))
        else:
            x_mid, y_mid, z_mid = nx // 2, ny // 2, nz // 2

        # Pre-compute 3D GT and Pred meshes ONCE per pathology (4x rendering speedup)
        gt_verts, gt_faces = None, None
        if gt_mask.sum() > 30:
            try:
                v, f, _, _ = marching_cubes(gt_mask.astype(np.float32), level=0.5, step_size=2)
                gt_verts, gt_faces = v, f
            except Exception:
                pass

        pred_verts, pred_faces = None, None
        if pred_mask.sum() > 30:
            try:
                vp, fp, _, _ = marching_cubes(pred_mask.astype(np.float32), level=0.5, step_size=2)
                pred_verts, pred_faces = vp, fp
            except Exception:
                pass

        base_sub = row_idx * 8

        # Panels 1-4: 4 Rotational 3D Viewports for this specific pathology
        for v_idx, (elev, azim, v_title) in enumerate(views_3d):
            ax_3d = fig.add_subplot(num_rows, 8, base_sub + v_idx + 1, projection='3d')
            
            # Pure white 3D background panes with contrasting slategray CT torso volume
            ax_3d.set_facecolor('white')
            ax_3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax_3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax_3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            # 1. Render high-contrast thoracic body contour mesh (slategray against white panes)
            if body_mesh_verts is not None and body_mesh_faces is not None:
                mesh_body = Poly3DCollection(body_mesh_verts[body_mesh_faces], facecolors='slategray', edgecolors='none', alpha=body_alpha)
                ax_3d.add_collection3d(mesh_body)

            # 2. Render GT mask (lime green)
            if gt_verts is not None and gt_faces is not None:
                mesh_gt = Poly3DCollection(gt_verts[gt_faces], facecolors='lime', edgecolors='none', alpha=0.65)
                ax_3d.add_collection3d(mesh_gt)

            # 3. Render Pred mask (crimson red)
            if pred_verts is not None and pred_faces is not None:
                mesh_p = Poly3DCollection(pred_verts[pred_faces], facecolors='crimson', edgecolors='none', alpha=0.55)
                ax_3d.add_collection3d(mesh_p)

            ax_3d.set_xlim(0, nx); ax_3d.set_ylim(0, ny); ax_3d.set_zlim(0, nz)
            ax_3d.set_xlabel('X (RL)', fontsize=7); ax_3d.set_ylabel('Y (AP)', fontsize=7); ax_3d.set_zlabel('Z (IS)', fontsize=7)
            ax_3d.view_init(elev=elev, azim=azim)
            if row_idx == 0:
                ax_3d.set_title(v_title, fontsize=10, fontweight='bold')

        # Helper to plot 2D slice overlay for single pathology
        def plot_2d_pathology_slice(ax, ct_slice, gt_s, pred_s, title_str):
            ct_norm = np.clip(ct_slice, vmin, vmax)
            ct_norm = (ct_norm - vmin) / (vmax - vmin)

            gt_overlay = np.zeros((*ct_norm.shape, 4), dtype=np.float32)
            gt_overlay[..., 1] = 1.0 # Lime Green
            gt_overlay[..., 3] = gt_s.astype(np.float32) * 0.65

            pred_overlay = np.zeros((*ct_norm.shape, 4), dtype=np.float32)
            pred_overlay[..., 0] = 1.0 # Crimson Red
            pred_overlay[..., 3] = pred_s.astype(np.float32) * 0.55

            gt_overlay = np.clip(gt_overlay, 0.0, 1.0)
            pred_overlay = np.clip(pred_overlay, 0.0, 1.0)

            ax.imshow(ct_norm, cmap='gray', origin='lower')
            ax.imshow(gt_overlay, origin='lower')
            ax.imshow(pred_overlay, origin='lower')
            ax.set_title(title_str, fontsize=9, fontweight='bold')
            ax.axis('off')

        # Panel 5: Axial Slice
        ax_ax = fig.add_subplot(num_rows, 8, base_sub + 5)
        plot_2d_pathology_slice(ax_ax, img_data[:, :, z_mid].T, gt_mask[:, :, z_mid].T, pred_mask[:, :, z_mid].T, f"Axial (Z={z_mid})")

        # Panel 6: Coronal Slice
        ax_cor = fig.add_subplot(num_rows, 8, base_sub + 6)
        plot_2d_pathology_slice(ax_cor, img_data[:, y_mid, :].T, gt_mask[:, y_mid, :].T, pred_mask[:, y_mid, :].T, f"Coronal (Y={y_mid})")

        # Panel 7: Sagittal Slice
        ax_sag = fig.add_subplot(num_rows, 8, base_sub + 7)
        plot_2d_pathology_slice(ax_sag, img_data[x_mid, :, :].T, gt_mask[x_mid, :, :].T, pred_mask[x_mid, :, :].T, f"Sagittal (X={x_mid})")

        # Panel 8: Finding Statistics & Prompt Card
        ax_stats = fig.add_subplot(num_rows, 8, base_sub + 8)
        ax_stats.axis('off')
        if row_idx == 0:
            ax_stats.set_title("Pathology & Text Prompt", fontsize=10, fontweight='bold')

        box_color = 'lightgreen' if f_item['dice'] >= 0.1 else ('salmon' if f_item['gt_voxels'] > 0 else 'lightgray')
        
        prompt_str = f_item['prompt'] if f_item['prompt'] else "No prompt text provided."
        prompt_wrapped = textwrap.fill(prompt_str, width=32)

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
    gray_patch = mpatches.Patch(color='gray', alpha=0.5, label=f'CT Torso Volume Contour (Alpha={body_alpha})')
    fig.legend(handles=[green_patch, red_patch, gray_patch], loc='lower center', ncol=3, frameon=True, fontsize=10)

    mode_str = "Fixed GT Affine" if fix_gt_affine else "Raw GT Affine"
    plt.suptitle(f"Per-Pathology 3D Rotational & 2D CT Visualization | Scan: {scan_id} ({mode_str})", fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    scan_subfolder = out_dir / scan_id
    scan_subfolder.mkdir(parents=True, exist_ok=True)
    suffix = "_fixed_affine" if fix_gt_affine else ""
    out_path = scan_subfolder / f"pathology_row_scan_{scan_id}{suffix}.png"
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
    CLI entry point to generate per-pathology multi-row 3D isosurface mesh and 2D slice visualizers
    for one or multiple CT scan volumes.
    """
    parser = argparse.ArgumentParser(description="Per-Pathology Multi-Row CT Visualization Generator")
    parser.add_argument("--scan_id", type=str, default=None, help="Specific scan ID or comma-separated IDs (e.g. train_19891_a_2,train_13098_a_2).")
    parser.add_argument("--pred_subdir", type=str, default="phase_2a_rule_based", help="Subdirectory name inside predictions directory (default: phase_2a_rule_based).")
    parser.add_argument("--num_scans", type=int, default=1, help="Number of random scans to plot if --scan_id is omitted.")
    parser.add_argument("--fix_gt_affine", action="store_true", help="If set, overrides GT mask header affine with image affine before canonical reorientation.")
    parser.add_argument("--no_render_body", action="store_true", help="If set, disables rendering of the semi-transparent 3D thoracic body contour.")
    parser.add_argument("--body_alpha", type=float, default=0.08, help="Opacity value for the 3D thoracic body contour mesh (default: 0.08).")
    parser.add_argument("--body_step_size", type=int, default=4, help="Step size for marching cubes body isosurface decimation (default: 4).")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sampling.")
    parser.add_argument("--out_dir", type=str, default=str(VISUALIZATIONS_DIR), help="Directory to save generated PNG images (default: scan_visualizations/).")
    args = parser.parse_args()

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
            fix_gt_affine=args.fix_gt_affine,
            render_body=not args.no_render_body,
            body_alpha=args.body_alpha,
            body_step_size=args.body_step_size
        )
        saved_paths.append(out_p)

    print(f"\n[COMPLETE] Successfully generated {len(saved_paths)} visualizer figures in '{out_dir.resolve()}'.", flush=True)

if __name__ == "__main__":
    main()
