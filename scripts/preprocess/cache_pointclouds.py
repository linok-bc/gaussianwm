"""
Preprocess dataset by running Splatt3r and caching full-resolution point clouds + images.

Caches pred1 (14D), pred2 (14D), and the source image for each frame,
preserving the H×W spatial grid layout needed for rendering.

Usage:
    python scripts/preprocess/cache_pointclouds.py \
        dataset=droid \
        dataset.data_path=$GWM_PATH/data/ \
        cache.dataset_name=droid_100 \
        cache.root_dir=$GWM_PATH/data/cached_pointclouds \
        cache.shard_size=256
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import numpy as np
import einops
from tqdm import tqdm
from termcolor import cprint

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gaussianwm.processor.regressor import Splatt3rRegressor, get_gaussain_tensor
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset


def process_single(image: torch.Tensor, splatt3r) -> dict:
    """Run Splatt3r on a single image and return raw features.

    Args:
        image: [1, C, H, W] float32 tensor, range [0, 1]
        splatt3r: Splatt3rRegressor instance

    Returns:
        dict with pred1 [H*W, 14], pred2 [H*W, 14], H, W
    """
    with torch.no_grad():
        pred1, pred2 = splatt3r(image)

    B, H, W = pred1['means'].shape[:3]

    pred1_tensor = get_gaussain_tensor(pred1)  # [1, H*W, 14]
    pred2_tensor = get_gaussain_tensor(pred2)  # [1, H*W, 14]

    return {
        'pred1': pred1_tensor[0].cpu(),
        'pred2': pred2_tensor[0].cpu(),
        'H': H,
        'W': W,
    }


def cache_split(
    split: str,
    cfg: DictConfig,
    splatt3r,
    device: torch.device,
    cache_dir: str,
    shard_size: int,
):
    """Cache point clouds and images for one split (train or val)."""
    split_cache_dir = os.path.join(cache_dir, split)
    os.makedirs(split_cache_dir, exist_ok=True)

    # Check if already completed
    metadata_path = os.path.join(split_cache_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        cprint(f"[{split}] Already cached: {meta['total_frames']} frames in {meta['num_shards']} shards", "green")
        return

    # Build dataset
    cprint(f"[{split}] Building dataset...", "blue")
    dataset = build_gaussian_splatting_reconstruction_dataset(split, cfg.dataset)
    cprint(f"[{split}] Dataset size: {len(dataset)}", "blue")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
    )

    shard_pred1_buffer = []
    shard_pred2_buffer = []
    shard_image_buffer = []
    shard_idx = 0
    total_frames = 0
    grid_H, grid_W = None, None

    cprint(f"[{split}] Processing frames...", "blue")
    for batch in tqdm(data_loader, desc=f"Caching {split}"):
        obs, action, reward = batch

        image = obs.float()
        if image.max() > 1.0:
            image = image / 255.0

        # Handle temporal dimension
        if image.ndim == 5:
            image = image[0, 0]  # [H, W, C]
        elif image.ndim == 4:
            image = image[0]  # [H, W, C]

        # Store original image as uint8
        image_uint8 = (image * 255).to(torch.uint8).cpu()

        # Prepare for Splatt3r: [1, C, H, W]
        image_input = image.permute(2, 0, 1).unsqueeze(0).to(device)

        result = process_single(image_input, splatt3r)

        if grid_H is None:
            grid_H, grid_W = result['H'], result['W']
        else:
            assert grid_H == result['H'] and grid_W == result['W'], \
                f"Grid size mismatch: expected ({grid_H}, {grid_W}), got ({result['H']}, {result['W']})"

        shard_pred1_buffer.append(result['pred1'])
        shard_pred2_buffer.append(result['pred2'])
        shard_image_buffer.append(image_uint8)

        if len(shard_pred1_buffer) >= shard_size:
            shard_data = {
                'pred1': torch.stack(shard_pred1_buffer[:shard_size]),
                'pred2': torch.stack(shard_pred2_buffer[:shard_size]),
                'images': torch.stack(shard_image_buffer[:shard_size]),
            }
            shard_path = os.path.join(split_cache_dir, f"shard_{shard_idx:04d}.pt")
            torch.save(shard_data, shard_path)

            total_frames += shard_size
            shard_idx += 1

            shard_pred1_buffer = shard_pred1_buffer[shard_size:]
            shard_pred2_buffer = shard_pred2_buffer[shard_size:]
            shard_image_buffer = shard_image_buffer[shard_size:]

            cprint(f"  Saved shard {shard_idx - 1} ({total_frames} frames so far)", "cyan")

    # Save final partial shard
    if shard_pred1_buffer:
        shard_data = {
            'pred1': torch.stack(shard_pred1_buffer),
            'pred2': torch.stack(shard_pred2_buffer),
            'images': torch.stack(shard_image_buffer),
        }
        shard_path = os.path.join(split_cache_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(shard_data, shard_path)
        total_frames += len(shard_pred1_buffer)
        shard_idx += 1

    metadata = {
        "total_frames": total_frames,
        "shard_size": shard_size,
        "num_shards": shard_idx,
        "grid_H": grid_H,
        "grid_W": grid_W,
        "feature_dim": 14,
        "dataset_name": cfg.cache.dataset_name,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    cprint(f"[{split}] Done: {total_frames} frames in {shard_idx} shards (grid: {grid_H}x{grid_W})", "green")


@hydra.main(version_base=None, config_path="../../configs", config_name="train_vae")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cprint(f"Using device: {device}", "blue")

    cprint("Loading Splatt3r...", "blue")
    splatt3r = Splatt3rRegressor().to(device).eval()
    cprint("Splatt3r loaded", "green")

    cache_dir = os.path.join(cfg.cache.root_dir, cfg.cache.dataset_name)
    shard_size = cfg.cache.get("shard_size", 256)

    cprint(f"Cache dir: {cache_dir}", "blue")
    cprint(f"Shard size: {shard_size}", "blue")

    for split in ["train", "val"]:
        cache_split(
            split=split,
            cfg=cfg,
            splatt3r=splatt3r,
            device=device,
            cache_dir=cache_dir,
            shard_size=shard_size,
        )

    cprint("All done!", "green")


if __name__ == "__main__":
    main()
