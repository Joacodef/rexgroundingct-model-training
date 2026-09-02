"""
===============================================================================
SCRIPT:         VoxTell-SPOCO Inference & Instance Extraction
PHASE:          Phase 4 — VoxTell-SPOCO Metric Learning
LOCATION:       scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco_inference.py
OBJECTIVE:      Run sliding-window inference with a fine-tuned VoxTell-SPOCO checkpoint,
                producing per-prompt dense 32D metric embeddings AND VoxTell's native
                text-query logit map in one pass. The logit map argmax is used as the
                text-conditioned seed; the calibrated Gaussian soft mask on the unit
                hypersphere is expanded from that seed and binarized into a 3D instance
                mask. Predictions are saved as 4D (F, X, Y, Z) NIfTI anchored to the
                parent CT scan header, ready for scripts/common/evaluate.py.
USAGE:          python scripts/phase_4_voxtell_spoco/exp_001_voxtell_spoco_inference.py \
                    --split val --checkpoint logs/phase_4_voxtell_spoco/exp_001_voxtell_spoco/latest_model.pt
===============================================================================
"""

import os
import sys
import gc
import json
import math
import argparse
import logging
import itertools
from pathlib import Path

from dotenv import load_dotenv

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
load_dotenv(override=False)

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from scripts.config import DATASET_JSON, RAW_IMAGES_DIR, TEXT_CACHE_DIR, LOGS_DIR, MODEL_DIR
from scripts.common.orientation import load_nifti_ras, save_nifti
from scripts.phase_4_voxtell_spoco.common import (
    load_voxtell_spoco_model,
    extract_instances_from_embeddings,
)

logger = logging.getLogger("exp_001_voxtell_spoco_inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PATCH = 192


def build_gaussian_weight(patch: int = PATCH, sigma_scale: float = 1.0 / 8.0) -> np.ndarray:
    """
    Signature:
        build_gaussian_weight(patch: int = 192, sigma_scale: float = 0.125) -> np.ndarray

    Objective:
        Build a separable 3D Gaussian importance weight for blending overlapping
        sliding-window patches (matches VoxTell's sigma_scale = 1/8), floored to a small
        positive value so no voxel receives zero weight.

    Inputs:
        patch (int): Cubic patch edge length.
        sigma_scale (float): Gaussian sigma as a fraction of the patch edge.

    Outputs:
        np.ndarray: Float32 weight volume of shape (patch, patch, patch), max 1.0.
    """
    coords = np.arange(patch, dtype=np.float64) - (patch - 1) / 2.0
    sigma = patch * sigma_scale
    g1 = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    g = g / g.max()
    g = np.clip(g, a_min=g.max() * 1e-3, a_max=None)
    return g.astype(np.float32)


def sliding_window_positions(shape: tuple, patch: int, step_size: float) -> list:
    """
    Signature:
        sliding_window_positions(shape: tuple[int, int, int], patch: int, step_size: float) -> list[tuple[int, int, int]]

    Objective:
        Compute evenly-spaced top-left corner positions for cubic sliding-window
        patches over a 3D volume (nnU-Net style: ceil coverage, uniform stride).

    Inputs:
        shape (tuple[int, int, int]): Volume spatial shape (Z, Y, X), each >= patch.
        patch (int): Cubic patch edge length.
        step_size (float): Fraction of patch to advance between windows (e.g. 0.5).

    Outputs:
        list[tuple[int, int, int]]: (z0, y0, x0) start corners.
    """
    stride = max(1, int(round(patch * step_size)))
    axis_positions = []
    for dim in range(3):
        last = shape[dim] - patch
        if last <= 0:
            axis_positions.append([0])
            continue
        n_steps = int(math.ceil(last / stride)) + 1
        pos = [int(round(i * last / (n_steps - 1))) for i in range(n_steps)]
        axis_positions.append(sorted(set(pos)))
    return list(itertools.product(*axis_positions))


def preprocess_scan(nifti_path: Path) -> tuple:
    """
    Signature:
        preprocess_scan(nifti_path: Path) -> tuple[np.ndarray, list, tuple]

    Objective:
        Reproduce the ReXDataset preprocessing for a full scan: load canonical RAS,
        transpose to (Z, Y, X), crop to the non-zero bounding box, and Z-score
        normalize intensities.

    Inputs:
        nifti_path (Path): Path to the raw CT NIfTI.

    Outputs:
        tuple[np.ndarray, list, tuple]: (normalized (1, Z, Y, X) float32 volume,
        crop bbox, original (Z, Y, X) shape before cropping).
    """
    img_ras, _, _ = load_nifti_ras(nifti_path)
    img_zyx = img_ras.transpose((2, 1, 0))[None].astype(np.float32)  # (1, Z, Y, X)
    orig_shape = img_zyx.shape[1:]
    img_cropped, _, bbox = crop_to_nonzero(img_zyx, None)
    img_norm = ZScoreNormalization(intensityproperties={}).run(img_cropped, None)
    return img_norm.astype(np.float32), bbox, orig_shape


@torch.inference_mode()
def infer_scan(
    model: torch.nn.Module,
    img_norm: np.ndarray,
    text_embed: torch.Tensor,
    device: torch.device,
    step_size: float,
    finding_chunk: int,
) -> tuple:
    """
    Signature:
        infer_scan(model, img_norm: np.ndarray, text_embed: torch.Tensor, device, step_size: float, finding_chunk: int) -> tuple[np.ndarray, np.ndarray]

    Objective:
        Run Gaussian-weighted sliding-window inference over one preprocessed scan,
        accumulating per-finding 32D metric embeddings and native text-query logits on
        the CPU, then normalizing (embeddings re-projected onto the unit hypersphere).

    Inputs:
        model (torch.nn.Module): VoxTellSpocoModel in eval mode.
        img_norm (np.ndarray): Preprocessed (1, Z, Y, X) float32 volume.
        text_embed (torch.Tensor): (1, F, D_text) float32 text query embeddings.
        device (torch.device): Compute device.
        step_size (float): Sliding-window step as a fraction of the patch.
        finding_chunk (int): Number of findings pushed through the model per forward.

    Outputs:
        tuple[np.ndarray, np.ndarray]: (embeddings (F, 32, Z, Y, X) float32 on S^31,
        logits (F, Z, Y, X) float32), both at the preprocessed (cropped) resolution.
    """
    num_findings = text_embed.shape[1]
    vol = torch.from_numpy(img_norm)  # (1, Z, Y, X)
    vol, revert = pad_nd_image(vol, [PATCH, PATCH, PATCH], "constant", {"value": 0}, True, None)
    _, Zp, Yp, Xp = vol.shape

    gauss = build_gaussian_weight(PATCH)
    positions = sliding_window_positions((Zp, Yp, Xp), PATCH, step_size)

    # Geometry weight is identical for every finding: accumulate once.
    weight_acc = np.zeros((Zp, Yp, Xp), dtype=np.float32)
    for (z0, y0, x0) in positions:
        weight_acc[z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH] += gauss
    weight_acc = np.maximum(weight_acc, 1e-6)

    emb_chunks, logit_chunks = [], []
    for f_start in range(0, num_findings, finding_chunk):
        f_end = min(f_start + finding_chunk, num_findings)
        fc = f_end - f_start
        t_chunk = text_embed[:, f_start:f_end].to(device, dtype=torch.float32)

        emb_acc = np.zeros((fc, 32, Zp, Yp, Xp), dtype=np.float32)
        logit_acc = np.zeros((fc, Zp, Yp, Xp), dtype=np.float32)
        for (z0, y0, x0) in positions:
            patch = vol[:, z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH].unsqueeze(0).to(device, dtype=torch.float32)
            emb, logit = model(patch, t_chunk, return_embeddings=True, return_logits=True)
            # emb: (1, fc, 32, P, P, P) ; logit: (1, fc, P, P, P)
            emb_np = emb[0].float().cpu().numpy() * gauss[None, None]
            logit_np = logit[0].float().cpu().numpy() * gauss[None]
            emb_acc[:, :, z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH] += emb_np
            logit_acc[:, z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH] += logit_np
            del patch, emb, logit, emb_np, logit_np

        emb_acc /= weight_acc[None, None]
        logit_acc /= weight_acc[None]
        # Re-project the blended embedding volume back onto the unit hypersphere.
        emb_acc /= np.maximum(np.linalg.norm(emb_acc, axis=1, keepdims=True), 1e-8)
        # Revert padding (revert is a length-4 slicer: channel + 3 spatial).
        emb_chunks.append(emb_acc[:, :, revert[1], revert[2], revert[3]].copy())
        logit_chunks.append(logit_acc[:, revert[1], revert[2], revert[3]].copy())
        del emb_acc, logit_acc

    return np.concatenate(emb_chunks, axis=0), np.concatenate(logit_chunks, axis=0)


def top_k_seeds(prob: np.ndarray, k: int, min_separation: int) -> list:
    """
    Signature:
        top_k_seeds(prob: np.ndarray, k: int, min_separation: int) -> list[tuple[int, int, int]]

    Objective:
        Pick up to k local-maximum seed voxels from a 3D probability map with greedy
        non-maximum suppression so seeds are not all drawn from one blob.

    Inputs:
        prob (np.ndarray): 3D probability map (Z, Y, X).
        k (int): Maximum number of seeds.
        min_separation (int): Minimum Chebyshev voxel distance between seeds.

    Outputs:
        list[tuple[int, int, int]]: Seed (z, y, x) coordinates, highest probability first.
    """
    work = prob.copy()
    seeds = []
    for _ in range(k):
        flat = int(np.argmax(work))
        coord = np.unravel_index(flat, work.shape)
        if work[coord] <= 0:
            break
        seeds.append(tuple(int(c) for c in coord))
        z0 = max(0, coord[0] - min_separation); z1 = coord[0] + min_separation + 1
        y0 = max(0, coord[1] - min_separation); y1 = coord[1] + min_separation + 1
        x0 = max(0, coord[2] - min_separation); x1 = coord[2] + min_separation + 1
        work[z0:z1, y0:y1, x0:x1] = 0.0
    return seeds


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Phase 4 Exp 001 VoxTell-SPOCO inference."""
    p = argparse.ArgumentParser(description="Phase 4 Exp 001: VoxTell-SPOCO inference & instance extraction")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--checkpoint", type=str, required=True, help="Path to a VoxTell-SPOCO .pt checkpoint")
    p.add_argument("--use_teacher", action="store_true", help="Load teacher_state_dict instead of student_state_dict")
    p.add_argument("--dataset_json", type=str, default=str(DATASET_JSON))
    p.add_argument("--img_dir", type=str, default=str(RAW_IMAGES_DIR))
    p.add_argument("--text_cache_dir", type=str, default=str(TEXT_CACHE_DIR))
    p.add_argument("--output_dir", type=str, required=True, help="Directory for predicted 4D NIfTI masks")
    p.add_argument("--candidate_mask_dir", type=str, default=None,
                   help="Optional dir of per-scan lung/foreground masks (<scan_id>.nii.gz) restricting seed/expansion")
    p.add_argument("--tile_step_size", type=float, default=0.5)
    p.add_argument("--p_cand", type=float, default=0.30, help="Min sigmoid(logit) peak to emit a non-empty mask")
    p.add_argument("--delta_var", type=float, default=0.5)
    p.add_argument("--pmaps_threshold", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--soft_mask_threshold", type=float, default=0.5)
    p.add_argument("--min_volume_voxels", type=int, default=10)
    p.add_argument("--seeds_per_finding", type=int, default=1)
    p.add_argument("--seed_min_separation", type=int, default=8)
    p.add_argument("--finding_chunk", type=int, default=1,
                   help="Findings sharing one sliding pass. 1 bounds RAM (~7 GB/finding); raise to trade RAM for speed")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Re-run scans whose prediction file already exists")
    return p.parse_args()


def load_checkpoint_into(model: torch.nn.Module, ckpt_path: str, use_teacher: bool) -> None:
    """
    Signature:
        load_checkpoint_into(model: torch.nn.Module, ckpt_path: str, use_teacher: bool) -> None

    Objective:
        Load a VoxTell-SPOCO training checkpoint into `model`, selecting the student or
        teacher weights and tolerating the raw-state-dict and nnU-Net key layouts.

    Inputs:
        model (torch.nn.Module): Target model.
        ckpt_path (str): Path to the checkpoint file.
        use_teacher (bool): Load teacher_state_dict when available.

    Outputs:
        None
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "student_state_dict" in ckpt or "teacher_state_dict" in ckpt:
        key = "teacher_state_dict" if use_teacher else "student_state_dict"
        state_dict = ckpt[key]
    elif "network_weights" in ckpt:
        state_dict = ckpt["network_weights"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"Loaded {'teacher' if use_teacher else 'student'} weights "
                f"(missing: {len(missing)}, unexpected: {len(unexpected)})")


def main() -> None:
    """Main entry point: sliding-window VoxTell-SPOCO inference over a dataset split."""
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_voxtell_spoco_model(model_dir=str(MODEL_DIR), device=str(device), deep_supervision=False)
    load_checkpoint_into(model, args.checkpoint, args.use_teacher)
    model.eval()

    with open(args.dataset_json, "r") as fh:
        entries = json.load(fh).get(args.split, [])
    end_idx = args.end_idx if args.end_idx is not None else len(entries)
    entries = entries[args.start_idx:end_idx]
    logger.info(f"Split '{args.split}': {len(entries)} scans [{args.start_idx}:{end_idx}]")

    img_dir = Path(args.img_dir)
    text_cache_dir = Path(args.text_cache_dir)
    cand_dir = Path(args.candidate_mask_dir) if args.candidate_mask_dir else None

    for entry in entries:
        scan_id = entry["name"].replace(".nii.gz", "")
        out_path = out_dir / f"{scan_id}.nii.gz"
        if out_path.exists() and not args.overwrite:
            logger.info(f"{scan_id}: prediction exists, skipping.")
            continue

        nifti_path = img_dir / f"{scan_id}.nii.gz"
        text_path = text_cache_dir / f"{scan_id}.pt"
        if not nifti_path.exists() or not text_path.exists():
            logger.warning(f"{scan_id}: missing CT or text cache, skipping.")
            continue

        text_embed = torch.load(text_path, map_location="cpu").float()
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(0)          # (1, F, D_text)
        elif text_embed.dim() == 3 and text_embed.shape[0] != 1:
            text_embed = text_embed.unsqueeze(0).squeeze(2)
        num_findings = text_embed.shape[1]

        img_norm, bbox, orig_shape = preprocess_scan(nifti_path)
        emb, logit = infer_scan(model, img_norm, text_embed, device,
                                args.tile_step_size, args.finding_chunk)

        candidate_mask = None
        if cand_dir is not None:
            cand_path = cand_dir / f"{scan_id}.nii.gz"
            if cand_path.exists():
                cand_ras, _, _ = load_nifti_ras(cand_path)
                cand_zyx = (cand_ras.transpose((2, 1, 0)) > 0).astype(np.uint8)
                candidate_mask = cand_zyx[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

        crop_shape = emb.shape[2:]
        pred_cropped = np.zeros((num_findings, *crop_shape), dtype=np.uint8)
        for n in range(num_findings):
            prob = 1.0 / (1.0 + np.exp(-logit[n]))
            if float(prob.max()) < args.p_cand:
                continue
            seeds = top_k_seeds(prob, args.seeds_per_finding, args.seed_min_separation)
            pred_cropped[n] = extract_instances_from_embeddings(
                embeddings=emb[n],
                delta_var=args.delta_var,
                pmaps_threshold=args.pmaps_threshold,
                sigma=args.sigma,
                threshold=args.soft_mask_threshold,
                min_volume_voxels=args.min_volume_voxels,
                candidate_mask=candidate_mask,
                seed_coords=seeds,
            )

        # Reinsert into the original (pre-crop) volume, then back to RAS (F, X, Y, Z).
        pred_full = np.zeros((num_findings, *orig_shape), dtype=np.uint8)
        pred_full = insert_crop_into_image(pred_full, pred_cropped, bbox)
        pred_ras = pred_full.transpose((0, 3, 2, 1))
        save_nifti(pred_ras, str(out_path), str(nifti_path), dtype=np.uint8)
        logger.info(f"{scan_id}: wrote {out_path.name}  (findings={num_findings}, "
                    f"non-empty={int((pred_cropped.reshape(num_findings, -1).sum(1) > 0).sum())})")

        del emb, logit, pred_cropped, pred_full, pred_ras
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("VoxTell-SPOCO inference complete.")


if __name__ == "__main__":
    main()
