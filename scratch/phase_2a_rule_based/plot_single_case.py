import json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.config import RAW_IMAGES_DIR, RAW_MASKS_DIR, PREDICTIONS_DIR

def load_canonical_ras(nifti_path: Path) -> np.ndarray:
    nii = nib.load(str(nifti_path))
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

def main():
    scan_id = "train_19891_a_2"
    f_idx = 1 # Pleural effusion / thickening (2e)

    img_path = RAW_IMAGES_DIR / f"{scan_id}.nii.gz"
    gt_path = RAW_MASKS_DIR / f"{scan_id}.nii.gz"
    pred_path = PREDICTIONS_DIR / "phase_2a_rule_based" / f"{scan_id}.nii.gz"

    print(f"[INFO] Loading canonical RAS volumes for {scan_id}...")
    img_data = load_canonical_ras(img_path)[0]   # (X, Y, Z)
    gt_data = load_canonical_ras(gt_path)[f_idx] # (X, Y, Z)
    pred_data = load_canonical_ras(pred_path)[f_idx] # (X, Y, Z)

    gt_3d = (gt_data > 0)
    pred_3d = (pred_data > 0)

    nx, ny, nz = img_data.shape

    # Find centroid of active GT mask
    x_indices, y_indices, z_indices = np.where(gt_3d)
    x_mid, y_mid, z_mid = int(np.median(x_indices)), int(np.median(y_indices)), int(np.median(z_indices))

    fig = plt.figure(figsize=(18, 4.8), dpi=200)

    # Panel 1: 3D Isosurface Mesh
    ax_3d = fig.add_subplot(1, 4, 1, projection='3d')
    
    verts_gt, faces_gt, _, _ = marching_cubes(gt_3d.astype(np.float32), level=0.5, step_size=2)
    verts_pred, faces_pred, _, _ = marching_cubes(pred_3d.astype(np.float32), level=0.5, step_size=2)

    mesh_gt = Poly3DCollection(verts_gt[faces_gt], facecolors='green', edgecolors='none', alpha=0.45)
    mesh_pred = Poly3DCollection(verts_pred[faces_pred], facecolors='red', edgecolors='none', alpha=0.35)

    ax_3d.add_collection3d(mesh_gt)
    ax_3d.add_collection3d(mesh_pred)

    ax_3d.set_xlim(0, nx); ax_3d.set_ylim(0, ny); ax_3d.set_zlim(0, nz)
    ax_3d.set_xlabel('X (RL)'); ax_3d.set_ylabel('Y (AP)'); ax_3d.set_zlabel('Z (IS)')
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.set_title("3D Isosurface Mesh\n(GT=Green, Pred=Red)", fontsize=10, fontweight='bold')

    # Window CT HU [-1000, 400]
    vmin, vmax = -1000.0, 400.0

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

    # Panel 2: Axial Slice (Z fixed)
    ax_ax = fig.add_subplot(1, 4, 2)
    plot_2d_slice(ax_ax, img_data[:, :, z_mid].T, gt_3d[:, :, z_mid].T, pred_3d[:, :, z_mid].T, f"Axial Slice (Z={z_mid})")

    # Panel 3: Coronal Slice (Y fixed)
    ax_cor = fig.add_subplot(1, 4, 3)
    plot_2d_slice(ax_cor, img_data[:, y_mid, :].T, gt_3d[:, y_mid, :].T, pred_3d[:, y_mid, :].T, f"Coronal Slice (Y={y_mid})")

    # Panel 4: Sagittal Slice (X fixed)
    ax_sag = fig.add_subplot(1, 4, 4)
    plot_2d_slice(ax_sag, img_data[x_mid, :, :].T, gt_3d[x_mid, :, :].T, pred_3d[x_mid, :, :].T, f"Sagittal Slice (X={x_mid})")

    green_patch = mpatches.Patch(color='green', alpha=0.6, label='Ground Truth Mask')
    red_patch = mpatches.Patch(color='red', alpha=0.6, label='Prediction Mask')
    fig.legend(handles=[green_patch, red_patch], loc='lower center', ncol=2, frameon=True, fontsize=10)

    plt.suptitle(f"Scan {scan_id} | Finding {f_idx} (Pleural Effusion 2e)", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = Path("single_case_train_19891_a_2.png")
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Saved single case figure to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
