"""
===============================================================================
SCRIPT:         plot_spatial_density_heatmaps.py
LOCATION:       scripts/analysis/plot_spatial_density_heatmaps.py
OBJECTIVE:      Generate publication-quality dark-themed spatial probability density 
                heatmaps for all 14 pathologies on the Coronal plane at peak probability 
                slices using the Magma colormap (pitch-black background), exporting a 
                high-resolution PNG overview grid with correct upright anatomical orientation.
USAGE:          python scripts/analysis/plot_spatial_density_heatmaps.py
===============================================================================
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.config import DATA_DIR, VISUALIZATIONS_DIR, CATEGORY_MAP, PHASE_2A_PDFS_DIR


def extract_coronal_peak_slice(density_3d: np.ndarray) -> tuple[int, int, int, float, np.ndarray]:
    """
    Signature:
        extract_coronal_peak_slice(density_3d: np.ndarray) -> tuple[int, int, int, float, np.ndarray]

    Objective:
        Identify global maximum probability coordinate (x_max, y_max, z_max) in 3D RAS space
        (X: Right-Left, Y: Anterior-Posterior, Z: Inferior-Superior) and extract the 2D Coronal 
        plane slice transposed to shape (Z, X) for upright vertical display in Matplotlib.

    Inputs:
        density_3d (np.ndarray): 3D float32 spatial probability density volume of shape (X, Y, Z).

    Outputs:
        tuple containing:
            - x_max (int): Right-Left sagittal index of global peak probability.
            - y_max (int): Anterior-Posterior coronal slice index of global peak probability.
            - z_max (int): Inferior-Superior axial index of global peak probability.
            - p_max (float): Global peak probability density value.
            - coronal_slice (np.ndarray): 2D float32 slice array transposed to shape (Z, X).
    """
    p_max = float(density_3d.max())
    if p_max <= 0:
        x_max, y_max, z_max = density_3d.shape[0] // 2, density_3d.shape[1] // 2, density_3d.shape[2] // 2
    else:
        max_idx = np.unravel_index(np.argmax(density_3d), density_3d.shape)
        x_max, y_max, z_max = int(max_idx[0]), int(max_idx[1]), int(max_idx[2])
    
    # Raw Coronal slice along Y axis has shape (X, Z) where Axis 0=X (Right-Left) and Axis 1=Z (Inferior-Superior)
    raw_coronal = density_3d[:, y_max, :]  # Shape: (X, Z)

    # =========================================================================
    # [CRITICAL ANATOMICAL ORIENTATION CONTRACT — DO NOT MODIFY OR REMOVE .T]
    # Matplotlib's ax.imshow(matrix, origin="lower") maps Matrix Row Axis 0 to 
    # the Vertical Y-screen axis and Matrix Column Axis 1 to the Horizontal X-screen axis.
    # Without transposing raw_coronal (X, Z) -> (Z, X):
    #   - Row Axis 0 (X) would be rendered vertically (causing lungs to lie sideways).
    #   - Column Axis 1 (Z) would be rendered horizontally (causing apex/neck to point right).
    # By transposing raw_coronal.T to shape (Z, X):
    #   - Row Axis 0 becomes Z (Inferior-Superior), placing Apex/Superior at top & Base at bottom.
    #   - Column Axis 1 becomes X (Right-Left), rendering the coronal view upright.
    # =========================================================================
    coronal_slice = raw_coronal.T  # Transposed Shape: (Z, X)

    return x_max, y_max, z_max, p_max, coronal_slice


def parse_args():
    """
    Signature:
        parse_args() -> argparse.Namespace

    Objective:
        Parse command-line arguments for Coronal spatial density heatmap PNG grid generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate dark-themed Coronal spatial probability density PNG heatmaps for all 14 pathologies"
    )
    default_cache = str(PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat.npz") if (PHASE_2A_PDFS_DIR / "empirical_spatial_pdf_14cat.npz").exists() else str(DATA_DIR / "phase_2a" / "empirical_spatial_pdf_14cat.npz")
    parser.add_argument(
        "--density_cache", type=str, default=default_cache,
        help="Path to precomputed 3D spatial probability density cache file (.npz)"
    )
    parser.add_argument(
        "--output_png", type=str, default=str(VISUALIZATIONS_DIR / "spatial_density_heatmaps_grid.png"),
        help="Path for output consolidated high-res PNG overview grid"
    )
    return parser.parse_args()


def main():
    """Main CLI entry point for generating upright Coronal spatial density PNG heatmaps."""
    args = parse_args()
    cache_path = Path(args.density_cache)
    output_png_path = Path(args.output_png)

    if not cache_path.exists():
        print(f"[ERROR] Probability density cache file not found at {cache_path}. Run Task 1 first to build cache.")
        sys.exit(1)

    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Generating Upright Coronal Spatial Density Heatmaps PNG Grid (Magma Palette)")
    print(f"Loading Cache:   {cache_path}")
    print(f"Output PNG Grid: {output_png_path}")
    print("=" * 80)

    # Load 14-category spatial density heatmaps
    with np.load(cache_path) as cache_data:
        categories = [code for code in CATEGORY_MAP.keys() if code in cache_data]

    if not categories:
        print(f"[ERROR] No category arrays found inside {cache_path}")
        sys.exit(1)

    # Magma colormap with pitch-black background
    cmap = plt.cm.magma.copy()
    cmap.set_under("#000000")

    # -------------------------------------------------------------------------
    # CONSOLIDATED HIGH-RES PNG GRID (4x4 Grid of Upright Coronal Slices)
    # -------------------------------------------------------------------------
    fig_grid, axes_grid = plt.subplots(4, 4, figsize=(20, 20), facecolor="#0c0c0e")
    axes_flat = axes_grid.flatten()

    fig_grid.suptitle(
        "3D Empirical Spatial Density Prior Heatmaps\n(Upright Coronal Peak-Probability Slices - Magma Palette)",
        fontsize=18, color="#ffffff", fontweight="bold", y=0.98
    )

    with np.load(cache_path) as cache_data:
        for idx, cat_code in enumerate(categories):
            ax = axes_flat[idx]
            cat_name = CATEGORY_MAP.get(cat_code, "Unknown")
            density_3d = cache_data[cat_code]

            x_max, y_max, z_max, p_max, coronal_slice = extract_coronal_peak_slice(density_3d)

            im = ax.imshow(
                coronal_slice, cmap=cmap, vmin=1e-5, vmax=max(p_max, 1e-4),
                origin="lower", aspect="auto"
            )
            ax.set_title(f"[{cat_code}] {cat_name}\nP_max = {p_max:.4f} (Coronal Y={y_max})", color="#ffffff", fontsize=10, pad=6, fontweight="bold")
            ax.set_xlabel("Right ← X → Left", color="#808080", fontsize=8)
            ax.set_ylabel("Inferior ← Z → Superior", color="#808080", fontsize=8)
            ax.set_facecolor("#000000")

            ax.tick_params(colors="#606060", labelsize=7)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#2a2a30")

        # Turn off remaining unused subplots if any
        for idx in range(len(categories), len(axes_flat)):
            axes_flat[idx].axis("off")

    fig_grid.tight_layout(rect=[0, 0.04, 1, 0.95])

    # Shared Horizontal Colorbar for Grid
    cbar_ax_grid = fig_grid.add_axes([0.20, 0.015, 0.60, 0.012])
    cbar_g = fig_grid.colorbar(im, cax=cbar_ax_grid, orientation="horizontal")
    cbar_g.set_label("Spatial Density Probability P_c(z, y, x) [Magma Palette: Black → Purple → Coral → Light Yellow]", color="#ffdb4d", fontsize=11, labelpad=4)
    cbar_g.ax.xaxis.set_tick_params(color="#ffffff", labelcolor="#ffffff", labelsize=9)
    cbar_g.outline.set_edgecolor("#333338")

    plt.savefig(output_png_path, dpi=300, facecolor=fig_grid.get_facecolor(), edgecolor="none")
    plt.close(fig_grid)

    print(f"[SUCCESS] High-resolution upright PNG overview grid saved to: {output_png_path}")


if __name__ == "__main__":
    main()
