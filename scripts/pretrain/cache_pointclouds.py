"""
Preprocess dataset by running Splatt3r and caching point clouds to disk.

Usage:
    python scripts/preprocess/cache_pointclouds.py \
        dataset=droid \
        dataset.data_path=$GWM_PATH/data/ \
        cache.dataset_name=droid_100 \
        cache.root_dir=$GWM_PATH/data/cached_pointclouds \
        cache.shard_size=1024 \
        cache.splatt3r_batch_size=8
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
from pytorch3d.ops import sample_farthest_points as fps

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gaussianwm.processor.regressor import Splatt3rRegressor
from gaussianwm.processor.datasets import build_gaussian_splatting_reconstruction_dataset


SH_C0 = 0.28209479177387814


def process_batch(images: torch.Tensor, splatt3r, point_cloud_size: int) -> torch.Tensor:
    """Run Splatt3r + FPS + color normalization on a batch of images.

    Args:
        images: [B, C, H, W] float32 tensor, range [0, 1]
        splatt3r: Splatt3rRegressor instance
        point_cloud_size: Number of points to keep after FPS

    Returns:
        [B, point_cloud_size, 14] float32 tensor of point cloud features
    """
    with torch.no_grad():
        points, _ = splatt3r.forward_tensor(images)

    # SH color normalization
    colors = 0.5 + SH_C0 * points[..., -4:-1]
    points[..., -4:-1] = colors / 255.0

    # Farthest point sampling
    points, _ = fps(points, K=point_cloud_size)

    return points.cpu()


def get_num_completed_frames(cache_dir: str, shard_size: int) -> int:
    """Count how many frames have already been cached (for resuming)."""
    metadata_path = os.path.join(cache_dir, "metadata.json")
    if os.path.exists(metadata_path):
        # Fully completed
        with open(metadata_path, "r") as f:
            return json.load(f)["total_frames"]

    # Count existing shards
    num_shards = 0
    while os.path.exists(os.path.join(cache_dir, f"shard_{num_shards:04d}.pt")):
        num_shards += 1

    if num_shards == 0:
        return 0

    # Last shard may be partial — load to check size
    last_shard = torch.load(os.path.join(cache_dir, f"shard_{num_shards - 1:04d}.pt"))
    full_shards = num_shards - 1
    return full_shards * shard_size + last_shard.shape[0]


def cache_split(
    split: str,
    cfg: DictConfig,
    splatt3r,
    device: torch.device,
    cache_dir: str,
    shard_size: int,
    splatt3r_batch_size: int,
    point_cloud_size: int,
):
    """Cache point clouds for one split (train or val)."""
    split_cache_dir = os.path.join(cache_dir, split)
    os.makedirs(split_cache_dir, exist_ok=True)

    # Check if already completed
    metadata_path = os.path.join(split_cache_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        cprint(f"[{split}] Already cached: {meta['total_frames']} frames in {meta['num_shards']} shards", "green")
        return

    # Check for partial progress
    completed_frames = get_num_completed_frames(split_cache_dir, shard_size)
    if completed_frames > 0:
        cprint(f"[{split}] Resuming from {completed_frames} already-cached frames", "yellow")

    # Build dataset
    cprint(f"[{split}] Building dataset...", "blue")
    dataset = build_gaussian_splatting_reconstruction_dataset(split, cfg.dataset)
    cprint(f"[{split}] Dataset size: {len(dataset)}", "blue")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # DroidDataset yields individual frames
        num_workers=0,
    )

    # Accumulate frames
    shard_buffer = []
    shard_idx = completed_frames // shard_size
    total_frames = 0
    image_batch = []
    frames_skipped = 0

    cprint(f"[{split}] Processing frames...", "blue")
    for batch in tqdm(data_loader, desc=f"Caching {split}"):
        obs, action, reward = batch

        # Skip already-cached frames
        if frames_skipped < completed_frames:
            frames_skipped += 1
            continue

        # obs shape: [1, T, H, W, C] — extract and convert
        image = obs.float()
        if image.max() > 1.0:
            image = image / 255.0
        image = einops.rearrange(image, 'b t h w c -> (b t) c h w')
        image_batch.append(image)

        # Process in batches through Splatt3r
        if len(image_batch) >= splatt3r_batch_size:
            batch_tensor = torch.cat(image_batch, dim=0).to(device)
            points = process_batch(batch_tensor, splatt3r, point_cloud_size)

            for i in range(points.shape[0]):
                shard_buffer.append(points[i])

                if len(shard_buffer) >= shard_size:
                    shard_tensor = torch.stack(shard_buffer[:shard_size])
                    shard_path = os.path.join(split_cache_dir, f"shard_{shard_idx:04d}.pt")
                    torch.save(shard_tensor, shard_path)
                    shard_idx += 1
                    total_frames += shard_size
                    shard_buffer = shard_buffer[shard_size:]

            image_batch = []

    # Process remaining images in the batch buffer
    if image_batch:
        batch_tensor = torch.cat(image_batch, dim=0).to(device)
        points = process_batch(batch_tensor, splatt3r, point_cloud_size)
        for i in range(points.shape[0]):
            shard_buffer.append(points[i])

    # Save final partial shard
    if shard_buffer:
        shard_tensor = torch.stack(shard_buffer)
        shard_path = os.path.join(split_cache_dir, f"shard_{shard_idx:04d}.pt")
        torch.save(shard_tensor, shard_path)
        total_frames += len(shard_buffer)
        shard_idx += 1

    total_frames += completed_frames

    # Write metadata
    metadata = {
        "total_frames": total_frames,
        "shard_size": shard_size,
        "num_shards": shard_idx,
        "point_cloud_size": point_cloud_size,
        "feature_dim": 14,
        "dataset_name": cfg.cache.dataset_name,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    cprint(f"[{split}] Done: {total_frames} frames in {shard_idx} shards", "green")


@hydra.main(version_base=None, config_path="../../configs", config_name="train_vae")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    cprint(f"Using device: {device}", "blue")

    # Load Splatt3r
    cprint("Loading Splatt3r...", "blue")
    splatt3r = Splatt3rRegressor().to(device).eval()
    cprint("Splatt3r loaded", "green")

    cache_dir = os.path.join(cfg.cache.root_dir, cfg.cache.dataset_name)
    shard_size = cfg.cache.get("shard_size", 1024)
    splatt3r_batch_size = cfg.cache.get("splatt3r_batch_size", 8)
    point_cloud_size = cfg.vae.point_cloud_size

    cprint(f"Cache dir: {cache_dir}", "blue")
    cprint(f"Shard size: {shard_size}", "blue")
    cprint(f"Splatt3r batch size: {splatt3r_batch_size}", "blue")
    cprint(f"Point cloud size: {point_cloud_size}", "blue")

    for split in ["train", "val"]:
        cache_split(
            split=split,
            cfg=cfg,
            splatt3r=splatt3r,
            device=device,
            cache_dir=cache_dir,
            shard_size=shard_size,
            splatt3r_batch_size=splatt3r_batch_size,
            point_cloud_size=point_cloud_size,
        )

    cprint("All done!", "green")


if __name__ == "__main__":
    main()
