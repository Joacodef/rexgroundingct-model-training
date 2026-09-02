"""
===============================================================================
SCRIPT:         plot_3d_spatial_density_heatmaps.py
LOCATION:       scripts/analysis/plot_3d_spatial_density_heatmaps.py
OBJECTIVE:      Generate vibrant 3D GPU-accelerated volumetric density renders
                for all 14 pathologies using PyVista / VTK, rendering 3 distinct 
                isometric camera viewports per row with bright Magma lighting, 
                anatomical 3D orientation triads (R-L, A-P, Superior-Inferior) in 
                the bottom-right corner of each view, clear row pathology headers, 
                per-row colorbars, and tight border-free layout.
REQUIRES:       PyVista, which is NOT part of the default .venv. Install it first:
                    .venv/bin/pip install -r requirements/visualization-3d.txt
USAGE:          python scripts/analysis/plot_3d_spatial_density_heatmaps.py
===============================================================================
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import DATA_DIR, VISUALIZATIONS_DIR, CATEGORY_MAP, SPATIAL_TAXONOMY, PHASE_2A_PDFS_DIR


def render_3d_pathology_isometric_views(
    density_3d: np.ndarray,
    cat_code: str,
    cat_name: str,
    target_render_size: tuple = (128, 128, 128)
) -> list[np.ndarray]:
    """
    Signature:
        render_3d_pathology_isometric_views(
            density_3d: np.ndarray, cat_code: str, cat_name: str, target_render_size: tuple
        ) -> list[np.ndarray]

    Objective:
        Render 3 GPU-accelerated 3D isometric camera views (Front-Right, Front-Left, Top) 
        for a single 3D pathology density heatmap with enhanced ambient/diffuse lighting, 
        vivid opacity mapping, and an anatomical 3D orientation triad (R-L, A-P, Sup-Inf)
        placed in the bottom-right corner of each viewport.

    Inputs:
        density_3d (np.ndarray): 3D float32 spatial probability density volume of shape (X, Y, Z).
        cat_code (str): 2-character category code string (e.g. '1c').
        cat_name (str): Human-readable pathology category name.
        target_render_size (tuple): Interpolated volume shape for fast rendering.

    Outputs:
        list[np.ndarray]: List of 3 RGB uint8 numpy image arrays corresponding to the 3 camera views.
    """
    # Interpolate to target_render_size for smooth GPU mesh extraction
    if density_3d.shape != target_render_size:
        tensor_in = torch.from_numpy(density_3d).unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)
        tensor_res = F.interpolate(tensor_in, size=target_render_size, mode='trilinear', align_corners=False)
        vol_data = tensor_res.squeeze(0).squeeze(0).numpy()
    else:
        vol_data = density_3d.copy()

    p_max = float(vol_data.max())
    if p_max <= 0:
        vol_data = np.full(target_render_size, 0.01, dtype=np.float32)
        p_max = 0.01

    grid = pv.ImageData(dimensions=vol_data.shape)
    grid.point_data["density"] = vol_data.ravel(order="F")

    # Multi-level 3D isosurfaces starting from low density to capture full structure
    p_min = max(0.02 * p_max, 1e-5)
    iso_levels = np.linspace(p_min, p_max * 0.95, 10)
    contours = grid.contour(isosurfaces=iso_levels, scalars="density")

    outline = grid.outline()

    pv.set_plot_theme("dark")
    images = []

    # 3 Isometric Camera Positions
    camera_positions = [
        # View 1: Front-Right Isometric (Anterior-Right-Superior)
        [(-1.5 * target_render_size[0], -2.0 * target_render_size[1], 1.8 * target_render_size[2]),
         (target_render_size[0] / 2, target_render_size[1] / 2, target_render_size[2] / 2),
         (0, 0, 1)],
        # View 2: Front-Left Isometric (Anterior-Left-Superior)
        [(2.5 * target_render_size[0], -2.0 * target_render_size[1], 1.8 * target_render_size[2]),
         (target_render_size[0] / 2, target_render_size[1] / 2, target_render_size[2] / 2),
         (0, 0, 1)],
        # View 3: Top Overhead View (Superior-Down)
        [(target_render_size[0] / 2, target_render_size[1] / 2, 3.0 * target_render_size[2]),
         (target_render_size[0] / 2, target_render_size[1] / 2, target_render_size[2] / 2),
         (0, 1, 0)]
    ]

    for cam_pos in camera_positions:
        plotter = pv.Plotter(off_screen=True, window_size=(700, 700))
        plotter.set_background("#000000")  # Pitch-black background
        plotter.enable_lightkit()

        if contours.n_cells > 0:
            plotter.add_mesh(
                contours,
                scalars="density",
                cmap="magma",
                clim=[p_min, p_max],
                opacity=[0.15, 0.4, 0.7, 0.9, 1.0],  # Progressive opacity transfer
                ambient=0.6,                         # Bright ambient lighting
                diffuse=0.9,                         # Vibrant diffuse reflection
                specular=0.4,
                smooth_shading=True,
                show_scalar_bar=False,
            )

        plotter.add_mesh(outline, color="#5a5a68", line_width=1.5)
        plotter.camera_position = cam_pos

        # Add Anatomical Orientation Triad Widget in the bottom-right corner
        # X = Right-Left (RL), Y = Anterior-Posterior (AP), Z = Superior-Inferior (Sup)
        plotter.add_axes(
            line_width=3,
            xlabel="R-L",
            ylabel="A-P",
            zlabel="Sup",
            color="white",
            viewport=(0.68, 0.0, 1.0, 0.30)
        )

        img_rgb = plotter.screenshot(return_img=True)
        plotter.close()
        images.append(img_rgb)

    return images


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for 3D spatial probability density heatmap rendering.
    """
    parser = argparse.ArgumentParser(
        description="Generate vibrant 3D GPU-accelerated isometric spatial probability density heatmaps"
    )
    default_cache = str(PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat.npz") if (PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat.npz").exists() else str(DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat.npz")
    parser.add_argument(
        "--density_cache", type=str, default=default_cache,
        help="Path to precomputed 3D spatial probability density cache file (.npz)"
    )
    parser.add_argument(
        "--output_png", type=str, default=str(VISUALIZATIONS_DIR / "spatial_3d_density_heatmaps_grid.png"),
        help="Path for output 3D isometric overview PNG grid"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for 3D spatial density heatmap isometric rendering with anatomical orientation triad."""
    args = parse_args()
    cache_path = Path(args.density_cache)
    output_png_path = Path(args.output_png)

    if not cache_path.exists():
        print(f"[ERROR] Probability density cache file not found at {cache_path}. Run Task 1 first to build cache.")
        sys.exit(1)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Vibrant 3D GPU Spatial Density Heatmaps PNG Grid (Anatomical Triads)")
    print(f"Loading Cache:   {cache_path}")
    print(f"Output PNG Grid: {output_png_path}")
    print("=" * 80)

    with np.load(cache_path) as cache_data:
        categories = [code for code in CATEGORY_MAP.keys() if code in cache_data]

    if not categories:
        print(f"[ERROR] No category arrays found inside {cache_path}")
        sys.exit(1)

    num_cats = len(categories)
    fig = plt.figure(figsize=(16, 2.8 * num_cats), facecolor="#0c0c0e")

    # Main Atlas Title
    fig.suptitle(
        "3D GPU Volumetric Density Atlas — 14 Pathology Categories",
        fontsize=22, color="#ffffff", fontweight="bold", y=0.992
    )

    # Tight GridSpec Layout removing excess vertical padding
    gs = fig.add_gridspec(
        num_cats, 4,
        width_ratios=[1, 1, 1, 0.08],
        wspace=0.04, hspace=0.28,
        top=0.975, bottom=0.005, left=0.03, right=0.97
    )

    cmap_magma = plt.cm.magma

    with np.load(cache_path) as cache_data:
        for cat_idx, cat_code in enumerate(categories):
            cat_name = CATEGORY_MAP.get(cat_code, "Unknown")
            taxonomy = SPATIAL_TAXONOMY.get(cat_code, "Parenchymal")
            density_3d = cache_data[cat_code]
            p_max = float(density_3d.max())

            print(f"[INFO] Rendering 3D Isometric Views for [{cat_code}] {cat_name} (P_max = {p_max:.4f})...")
            view_imgs = render_3d_pathology_isometric_views(density_3d, cat_code, cat_name)

            # Render 3 Camera Viewports for this row
            view_axes = []
            for v_idx in range(3):
                ax = fig.add_subplot(gs[cat_idx, v_idx])
                ax.imshow(view_imgs[v_idx])
                ax.set_facecolor("#000000")
                ax.axis("off")
                view_axes.append(ax)

            # Row Pathology Header Banner cleanly anchored above center viewport
            row_title_str = f"[{cat_code}] {cat_name}  |  Taxonomy: {taxonomy}  |  Peak Density P_max = {p_max:.4f}"
            view_axes[1].text(
                0.5, 1.04, row_title_str, color="#ffdb4d", fontsize=13, fontweight="bold",
                ha="center", va="bottom", transform=view_axes[1].transAxes
            )

            # Dedicated Per-Row Colorbar in Column 4
            cbar_ax = fig.add_subplot(gs[cat_idx, 3])
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max(p_max, 1e-4))
            cb = fig.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_magma),
                cax=cbar_ax, orientation="vertical"
            )
            cb.set_label("Density P_c(x,y,z)", color="#ffffff", fontsize=10, labelpad=6)
            cb.ax.yaxis.set_tick_params(color="#ffffff", labelcolor="#ffffff", labelsize=9)
            cb.outline.set_edgecolor("#444450")

    plt.savefig(output_png_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f"\n[SUCCESS] High-resolution vibrant 3D overview grid saved to: {output_png_path}")


if __name__ == "__main__":
    main()
