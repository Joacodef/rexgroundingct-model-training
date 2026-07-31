"""
===============================================================================
SCRATCH SCRIPT:   Phase 2A 3D Hybrid Visualization & Figure Generator
OBJECTIVE:        Analyze exp_001 validation predictions, harvest 3D isosurface 
                  mesh + multi-planar (Axial, Coronal, Sagittal) snapshots for 
                  best-performing findings and qualitative failure cases, saving
                  high-resolution PNG figures for report documentation.
USAGE:            python scratch/phase_2a_rule_based/generate_3d_figures.py
===============================================================================
"""

import os
import sys
import json
import gc
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from scripts.config import (
    DATA_DIR, DATASET_JSON, RAW_IMAGES_DIR, RAW_MASKS_DIR, 
    PREDICTIONS_DIR, LOGS_DIR, CATEGORY_MAP
)


def load_canonical_ras(nifti_path: Path) -> np.ndarray:
    """
    Signature:
        load_canonical_ras(nifti_path: Path) -> np.ndarray

    Objective:
        Load NIfTI image/mask file using nib.as_closest_canonical and return 
        array in canonical RAS space (X, Y, Z) where 0=X (RL), 1=Y (AP), 2=Z (IS).

    Inputs:
        nifti_path (Path): Path to .nii.gz file.

    Outputs:
        np.ndarray: Float32 numpy array with shape (F, X, Y, Z) in canonical RAS space.
    """
    nii = nib.load(str(nifti_path))
    nii_ras = nib.as_closest_canonical(nii)
    data = np.asanyarray(nii_ras.dataobj).astype(np.float32)

    # Handle 3D vs 4D finding channels
    if data.ndim == 3:
        data_ras = np.expand_dims(data, axis=0) # (1, X, Y, Z)
    elif data.ndim == 4:
        if data.shape[-1] < np.min(data.shape[:3]):
            data_ras = np.moveaxis(data, -1, 0) # (F, X, Y, Z)
        else:
            data_ras = data # (F, X, Y, Z)
            
    return data_ras


def main():
    exp_dir = LOGS_DIR / "phase_2a_rule_based" / "exp_001_seg_masks_priors"
    figs_dir = exp_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = PREDICTIONS_DIR / "phase_2a_rule_based"
    gt_dir = RAW_MASKS_DIR
    img_dir = RAW_IMAGES_DIR
    dataset_json_path = DATASET_JSON

    print(f"[INFO] Reading metadata from {dataset_json_path}...")
    with open(dataset_json_path, 'r') as f:
        metadata = json.load(f)

    val_entries = metadata.get("val", [])
    print(f"[INFO] Analyzing {len(val_entries)} validation scans for best/worst finding instances...")

    finding_records = []

    for entry in tqdm(val_entries, desc="Auditing Val Findings"):
        scan_id = entry.get("name", "").replace(".nii.gz", "")
        if not scan_id:
            continue

        pred_path = pred_dir / f"{scan_id}.nii.gz"
        gt_path = gt_dir / f"{scan_id}.nii.gz"
        img_path = img_dir / f"{scan_id}.nii.gz"

        if not (pred_path.exists() and gt_path.exists() and img_path.exists()):
            continue

        findings = entry.get("findings", {})
        categories_dict = entry.get("categories", {})

        if isinstance(findings, dict):
            sorted_keys = sorted(findings.keys(), key=int)
            prompts = [findings[k].get("text", "") if isinstance(findings[k], dict) else str(findings[k]) for k in sorted_keys]
        else:
            prompts = [f.get("text", "") if isinstance(f, dict) else str(f) for f in findings]

        if not prompts:
            continue

        try:
            gt_data_ras = load_canonical_ras(gt_path)     # (F, X, Y, Z)
            pred_data_ras = load_canonical_ras(pred_path) # (F, X, Y, Z)

            num_findings = min(gt_data_ras.shape[0], pred_data_ras.shape[0], len(prompts))

            for f_idx in range(num_findings):
                cat_code = str(categories_dict.get(str(f_idx), "2h"))
                cat_name = CATEGORY_MAP.get(cat_code, "Other focal")
                prompt_text = prompts[f_idx]

                gt_3d = (gt_data_ras[f_idx] > 0)
                pred_3d = (pred_data_ras[f_idx] > 0)

                gt_voxels = int(gt_3d.sum())
                pred_voxels = int(pred_3d.sum())

                tp = int(np.logical_and(gt_3d, pred_3d).sum())
                union = gt_voxels + pred_voxels

                if union == 0:
                    dice = 1.0
                else:
                    dice = float(2.0 * tp / union)

                record = {
                    "scan_id": scan_id,
                    "f_idx": f_idx,
                    "cat_code": cat_code,
                    "cat_name": cat_name,
                    "prompt": prompt_text,
                    "dice": dice,
                    "gt_voxels": gt_voxels,
                    "pred_voxels": pred_voxels,
                    "tp": tp,
                    "gt_path": str(gt_path),
                    "pred_path": str(pred_path),
                    "img_path": str(img_path)
                }
                finding_records.append(record)

            del gt_data_ras, pred_data_ras
            gc.collect()

        except Exception as e:
            tqdm.write(f"[WARNING] Error reading {scan_id}: {e}")
            continue

    print(f"[INFO] Audited {len(finding_records)} total finding instances across validation split.")

    sorted_by_dice = sorted(finding_records, key=lambda x: x["dice"], reverse=True)

    # Select Best Cases (highest non-1.0 Dice)
    best_cases = [r for r in sorted_by_dice if r["dice"] > 0.01 and r["gt_voxels"] > 50][:6]

    # Select Failure Cases (zero Dice with substantial GT & Pred volume)
    failures_zero_dice = [r for r in sorted_by_dice if r["dice"] == 0.0 and r["gt_voxels"] > 200 and r["pred_voxels"] > 200]
    failures_zero_dice.sort(key=lambda x: x["gt_voxels"], reverse=True)
    failure_cases = failures_zero_dice[:6]

    print(f"[INFO] Selected {len(best_cases)} best cases and {len(failure_cases)} failure cases for 3D figure generation.")

    def render_3d_hybrid_snapshot(case_info: dict, out_name: str, title_prefix: str) -> None:
        """
        Signature:
            render_3d_hybrid_snapshot(case_info: dict, out_name: str, title_prefix: str) -> None

        Objective:
            Render 4-panel hybrid visualization figure (3D Isosurface Mesh, Axial, Coronal, 
            Sagittal slice overlays) for a specified finding case and save PNG asset.

        Inputs:
            case_info (dict): Finding record metadata dictionary containing paths and scores.
            out_name (str): Target PNG output filename.
            title_prefix (str): Figure title prefix (e.g. 'Best Result #1').

        Outputs:
            None
        """
        img_path = case_info["img_path"]
        gt_path = case_info["gt_path"]
        pred_path = case_info["pred_path"]
        f_idx = case_info["f_idx"]

        img_data_ras = load_canonical_ras(img_path)   # (1, X, Y, Z)
        gt_data_ras = load_canonical_ras(gt_path)     # (F, X, Y, Z)
        pred_data_ras = load_canonical_ras(pred_path) # (F, X, Y, Z)

        img_3d = img_data_ras[0] # (X, Y, Z)
        gt_3d = (gt_data_ras[f_idx] > 0)
        pred_3d = (pred_data_ras[f_idx] > 0)

        # Spatial dimensions in RAS: X=dim0 (RL), Y=dim1 (AP), Z=dim2 (IS)
        nx_len, ny_len, nz_len = img_3d.shape

        # Compute centroid of combined mask for multi-planar slicing
        combined_mask = np.logical_or(gt_3d, pred_3d)
        if combined_mask.any():
            x_indices, y_indices, z_indices = np.where(combined_mask)
            x_mid = int(np.median(x_indices))
            y_mid = int(np.median(y_indices))
            z_mid = int(np.median(z_indices))
        else:
            x_mid, y_mid, z_mid = nx_len // 2, ny_len // 2, nz_len // 2

        # Create 4-panel hybrid figure (Panel 1: 3D Mesh, Panels 2-4: Axial, Coronal, Sagittal)
        fig = plt.figure(figsize=(18, 4.8), dpi=300)

        # Panel 1: 3D Isosurface Mesh
        ax_3d = fig.add_subplot(1, 4, 1, projection='3d')
        
        if gt_3d.sum() > 20:
            try:
                verts_gt, faces_gt, _, _ = marching_cubes(gt_3d.astype(np.float32), level=0.5, step_size=2)
                mesh_gt = Poly3DCollection(verts_gt[faces_gt], facecolors='green', edgecolors='none', alpha=0.45)
                ax_3d.add_collection3d(mesh_gt)
            except Exception:
                pass

        if pred_3d.sum() > 20:
            try:
                verts_pred, faces_pred, _, _ = marching_cubes(pred_3d.astype(np.float32), level=0.5, step_size=2)
                mesh_pred = Poly3DCollection(verts_pred[faces_pred], facecolors='red', edgecolors='none', alpha=0.35)
                ax_3d.add_collection3d(mesh_pred)
            except Exception:
                pass

        ax_3d.set_xlim(0, nx_len) # X (RL)
        ax_3d.set_ylim(0, ny_len) # Y (AP)
        ax_3d.set_zlim(0, nz_len) # Z (IS)
        ax_3d.set_xlabel('X (RL)', fontsize=8)
        ax_3d.set_ylabel('Y (AP)', fontsize=8)
        ax_3d.set_zlabel('Z (IS)', fontsize=8)
        ax_3d.view_init(elev=20, azim=45)
        ax_3d.set_title("3D Isosurface Mesh\n(GT=Green, Pred=Red)", fontsize=10, fontweight='bold')

        # Window CT HU [-1000, 400]
        vmin, vmax = -1000.0, 400.0

        # Helper to plot 2D slice overlay
        def plot_2d_slice(ax, ct_slice, gt_s, pred_s, title_str):
            ct_norm = np.clip(ct_slice, vmin, vmax)
            ct_norm = (ct_norm - vmin) / (vmax - vmin)

            gt_overlay = np.zeros((*ct_norm.shape, 4), dtype=np.float32)
            gt_overlay[..., 1] = 1.0 # Green
            gt_overlay[..., 3] = gt_s.astype(np.float32) * 0.55

            pred_overlay = np.zeros((*ct_norm.shape, 4), dtype=np.float32)
            pred_overlay[..., 0] = 1.0 # Red
            pred_overlay[..., 3] = pred_s.astype(np.float32) * 0.50

            ax.imshow(ct_norm, cmap='gray', origin='lower')
            ax.imshow(gt_overlay, origin='lower')
            ax.imshow(pred_overlay, origin='lower')
            ax.set_title(title_str, fontsize=10, fontweight='bold')
            ax.axis('off')

        # Panel 2: Axial Slice (Z fixed) -> slice shape (Y, X)
        ax_ax = fig.add_subplot(1, 4, 2)
        plot_2d_slice(ax_ax, img_3d[:, :, z_mid].T, gt_3d[:, :, z_mid].T, pred_3d[:, :, z_mid].T, f"Axial Slice (Z={z_mid})")

        # Panel 3: Coronal Slice (Y fixed) -> slice shape (Z, X)
        ax_cor = fig.add_subplot(1, 4, 3)
        plot_2d_slice(ax_cor, img_3d[:, y_mid, :].T, gt_3d[:, y_mid, :].T, pred_3d[:, y_mid, :].T, f"Coronal Slice (Y={y_mid})")

        # Panel 4: Sagittal Slice (X fixed) -> slice shape (Z, Y)
        ax_sag = fig.add_subplot(1, 4, 4)
        plot_2d_slice(ax_sag, img_3d[x_mid, :, :].T, gt_3d[x_mid, :, :].T, pred_3d[x_mid, :, :].T, f"Sagittal Slice (X={x_mid})")

        # Legend
        green_patch = mpatches.Patch(color='green', alpha=0.6, label='Ground Truth 3D Mask')
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='Empirical PDF Prior')
        fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, frameon=True, fontsize=10)

        prompt_clean = case_info['prompt'][:60] + "..." if len(case_info['prompt']) > 60 else case_info['prompt']
        plt.suptitle(f"{title_prefix}: Scan {case_info['scan_id']} | Category '{case_info['cat_code']}' ({case_info['cat_name']}) | Dice: {case_info['dice']:.4f}\n\"{prompt_clean}\"", fontsize=11, y=1.03)

        plt.tight_layout()
        save_path = figs_dir / out_name
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"[INFO] Saved 3D hybrid figure: {save_path}")

        del img_data_ras, gt_data_ras, pred_data_ras
        gc.collect()

    # Render 3D figures for best cases
    for i, c in enumerate(best_cases, 1):
        fname = f"best_case_{i}_scan_{c['scan_id']}_{c['cat_code']}_3d.png"
        render_3d_hybrid_snapshot(c, fname, f"Best Result #{i}")

    # Render 3D figures for failure cases
    for i, c in enumerate(failure_cases, 1):
        fname = f"failure_case_{i}_scan_{c['scan_id']}_{c['cat_code']}_3d.png"
        render_3d_hybrid_snapshot(c, fname, f"Failure Case #{i}")

    print(f"[SUCCESS] Completed rendering all 3D hybrid figures to {figs_dir}")


if __name__ == "__main__":
    main()
