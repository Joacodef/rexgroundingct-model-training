"""
===============================================================================
MODULE:         precompute_text_embeddings.py
LOCATION:       scripts/common/precompute_text_embeddings.py
OBJECTIVE:      Precompute offline Qwen text embeddings for all ReXGroundingCT 
                finding text prompts across train and val splits to eliminate 
                NLP forward pass overhead and VRAM usage during 3D training.
===============================================================================
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.config import DATASET_JSON, TEXT_CACHE_DIR, HF_HOME
from voxtell.utils.text_embedding import wrap_with_instruction, last_token_pool

load_dotenv(override=False)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("precompute_text_embeddings")


def extract_all_findings_by_scan(dataset_json_path: Path, splits: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Extract ordered finding prompt strings per scan across target splits from dataset.json.

    Signature:
        extract_all_findings_by_scan(dataset_json_path: Path, splits: Optional[List[str]]) -> Dict[str, List[str]]

    Args:
        dataset_json_path (Path): Path to dataset.json file.
        splits (Optional[List[str]]): Splits to process (default: ['train', 'val']).

    Returns:
        Dict[str, List[str]]: Dictionary mapping scan_id to ordered list of finding text prompt strings.
    """
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_json_path}")

    with open(dataset_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_splits = splits or ["train", "val"]
    scans_dict: Dict[str, List[str]] = {}

    for split in target_splits:
        if split not in data:
            continue
        for entry in data[split]:
            scan_name = entry.get("name") or entry.get("image") or entry.get("scan_id")
            scan_id = scan_name.replace(".nii.gz", "")
            findings = entry.get("findings", {})
            if isinstance(findings, dict):
                # Sort finding keys integer-wise: '0', '1', '2', ...
                sorted_keys = sorted(findings.keys(), key=lambda k: int(k) if k.isdigit() else k)
                ordered_prompts = [findings[k] for k in sorted_keys]
            elif isinstance(findings, list):
                ordered_prompts = findings
            else:
                ordered_prompts = [str(findings)]
            scans_dict[scan_id] = ordered_prompts

    return scans_dict


def precompute_embeddings(
    scans_dict: Dict[str, List[str]],
    cache_dir: Path,
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    device: str = "cuda:0",
    batch_size: int = 64,
    overwrite: bool = False,
) -> int:
    """
    Precompute and save Qwen text embedding tensors to the cache directory.

    Signature:
        precompute_embeddings(scans_dict: Dict[str, List[str]], cache_dir: Path, model_name: str, device: str, batch_size: int, overwrite: bool) -> int

    Args:
        scans_dict (Dict[str, List[str]]): Mapping of scan_id to prompt lists.
        cache_dir (Path): Destination cache directory.
        model_name (str): Hugging Face text model name (default: 'Qwen/Qwen3-Embedding-4B').
        device (str): Computation device.
        batch_size (int): Batch size for text embedding forward pass.
        overwrite (bool): Whether to overwrite existing cache files.

    Returns:
        int: Number of successfully cached scan embedding files.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    comp_device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading text encoder '{model_name}' onto {comp_device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(HF_HOME) if HF_HOME else None)
    text_backbone = AutoModel.from_pretrained(model_name, cache_dir=str(HF_HOME) if HF_HOME else None).to(comp_device)
    text_backbone.eval()

    # Determine unique prompts across all scans to avoid redundant computation
    unique_prompts: Dict[str, Optional[torch.Tensor]] = {}
    for scan_id, prompts in scans_dict.items():
        for p in prompts:
            if p not in unique_prompts:
                unique_prompts[p] = None

    logger.info(f"Identified {len(unique_prompts)} unique text prompts across {len(scans_dict)} scans.")
    prompt_list = list(unique_prompts.keys())

    # Batch embedding of unique prompts
    with torch.no_grad():
        for i in tqdm(range(0, len(prompt_list), batch_size), desc="Embedding unique prompts"):
            batch_prompts = prompt_list[i : i + batch_size]
            wrapped = wrap_with_instruction(batch_prompts)
            tokens = tokenizer(
                wrapped,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(comp_device)

            outputs = text_backbone(**tokens)
            batch_embeds = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
            batch_embeds = batch_embeds.cpu().to(torch.float32)

            for p, emb in zip(batch_prompts, batch_embeds):
                unique_prompts[p] = emb

    # Save per-scan tensors (F, D_embed)
    saved_count = 0
    for scan_id, prompts in tqdm(scans_dict.items(), desc="Saving scan embedding files"):
        out_file = cache_dir / f"{scan_id}.pt"
        if out_file.exists() and not overwrite:
            saved_count += 1
            continue

        scan_embeds = torch.stack([unique_prompts[p] for p in prompts], dim=0) # (F, D_embed)
        torch.save(scan_embeds, out_file)
        saved_count += 1

    logger.info(f"Successfully cached {saved_count} text embedding files into {cache_dir}.")
    return saved_count


def main():
    """Boilerplate CLI entrypoint for text embedding precomputation."""
    parser = argparse.ArgumentParser(description="Precompute Qwen text embeddings for ReXGroundingCT")
    parser.add_argument("--dataset_json", type=str, default=str(DATASET_JSON), help="Path to dataset.json")
    parser.add_argument("--cache_dir", type=str, default=str(TEXT_CACHE_DIR), help="Destination cache directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to process")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-4B", help="Text backbone model name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computation device")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args()

    scans_dict = extract_all_findings_by_scan(Path(args.dataset_json), splits=args.splits)
    precompute_embeddings(
        scans_dict=scans_dict,
        cache_dir=Path(args.cache_dir),
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
