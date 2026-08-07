"""
===============================================================================
MODULE:         nlp_locators.py
PHASE:          Common Shared Utilities
LOCATION:       scripts/common/nlp_locators.py
OBJECTIVE:      Extract anatomical directional locators from free-text radiology 
                prompts and generate 3D spatial ROI bounding box masks.
                Shared across Phase 2 and Phase 3 pipelines.
===============================================================================
"""

import numpy as np

# Anatomical spatial locator keywords mapping to normalized 3D RAS bounding box coordinates [RL, AP, IS].
# DERIVATION: Derived from Phase 1 text prompt directive profiling (64.39% spatial descriptors).
ANATOMICAL_LOCATOR_KEYWORDS = {
    "right": {"rl": (0.45, 1.0)},
    "left": {"rl": (0.0, 0.55)},
    "bilateral": {"rl": (0.0, 1.0)},
    "upper lobe": {"is": (0.50, 1.0)},
    "apical": {"is": (0.60, 1.0)},
    "apex": {"is": (0.60, 1.0)},
    "lower lobe": {"is": (0.0, 0.55)},
    "basal": {"is": (0.0, 0.50)},
    "base": {"is": (0.0, 0.50)},
    "middle lobe": {"rl": (0.45, 1.0), "is": (0.35, 0.70)},
    "anterior": {"ap": (0.50, 1.0)},
    "posterior": {"ap": (0.0, 0.55)},
    "subpleural": {"margin": 0.20},  # Outer 20% margin
    "peripheral": {"margin": 0.20},  # Outer 20% margin
    "perihilar": {"core": 0.60},     # Central 60% core
    "hilar": {"core": 0.60},         # Central 60% core
}


def parse_prompt_spatial_locators(prompt_text: str) -> dict:
    """
    Signature:
        parse_prompt_spatial_locators(prompt_text: str) -> dict

    Objective:
        Extract normalized RAS 3D bounding box coordinate constraints [RL, AP, IS] from free-text prompt.

    Inputs:
        prompt_text (str): Radiology finding text description string.

    Outputs:
        dict: Normalized 3D bounding box dict with keys 'rl', 'ap', 'is', 'margin', 'core'.
    """
    if not prompt_text or not isinstance(prompt_text, str):
        return {}

    text_lower = prompt_text.lower()
    bounds = {}

    for kw, rule in ANATOMICAL_LOCATOR_KEYWORDS.items():
        if kw in text_lower:
            for k, val in rule.items():
                if k in ("rl", "ap", "is"):
                    if k not in bounds:
                        bounds[k] = list(val)
                    else:
                        bounds[k] = [max(bounds[k][0], val[0]), min(bounds[k][1], val[1])]
                elif k in ("margin", "core"):
                    bounds[k] = val

    return bounds


def generate_text_spatial_mask(prompt_text: str, target_shape_ras: tuple) -> np.ndarray:
    """
    Signature:
        generate_text_spatial_mask(prompt_text: str, target_shape_ras: tuple) -> np.ndarray

    Objective:
        Generate a 3D float32 ROI spatial mask for a target volume shape based on parsed NLP text locators.

    Inputs:
        prompt_text (str): Radiology finding text prompt string.
        target_shape_ras (tuple): Target 3D volume shape (X, Y, Z).

    Outputs:
        np.ndarray: 3D float32 mask array with 1.0 inside anatomical ROI and 0.0 outside.
    """
    mask = np.ones(target_shape_ras, dtype=np.float32)
    bounds = parse_prompt_spatial_locators(prompt_text)
    if not bounds:
        return mask

    nx, ny, nz = target_shape_ras

    # Coordinate axis bounds (RL=X, AP=Y, IS=Z)
    if "rl" in bounds:
        x_min = int(bounds["rl"][0] * nx)
        x_max = int(bounds["rl"][1] * nx)
        mask[:x_min, :, :] = 0.0
        mask[x_max:, :, :] = 0.0

    if "ap" in bounds:
        y_min = int(bounds["ap"][0] * ny)
        y_max = int(bounds["ap"][1] * ny)
        mask[:, :y_min, :] = 0.0
        mask[:, y_max:, :] = 0.0

    if "is" in bounds:
        z_min = int(bounds["is"][0] * nz)
        z_max = int(bounds["is"][1] * nz)
        mask[:, :, :z_min] = 0.0
        mask[:, :, z_max:] = 0.0

    # Peripheral / Subpleural margin filter
    if "margin" in bounds:
        m_frac = bounds["margin"]
        mx, my, mz = int(m_frac * nx), int(m_frac * ny), int(m_frac * nz)
        core_mask = np.zeros(target_shape_ras, dtype=bool)
        core_mask[mx:nx-mx, my:ny-my, mz:nz-mz] = True
        mask[core_mask] = 0.0

    # Central / Perihilar core filter
    if "core" in bounds:
        c_frac = (1.0 - bounds["core"]) / 2.0
        cx, cy, cz = int(c_frac * nx), int(c_frac * ny), int(c_frac * nz)
        outer_mask = np.ones(target_shape_ras, dtype=bool)
        outer_mask[cx:nx-cx, cy:ny-cy, cz:nz-cz] = False
        mask[outer_mask] = 0.0

    return mask
