"""
Cached point cloud dataset for VAE training.

Loads precomputed Splatt3r point clouds from sharded .pt files,
avoiding redundant Splatt3r inference during training.
"""

import os
import json

import torch
from torch.utils.data import Dataset


class CachedPointCloudDataset(Dataset):
    """Map-style dataset that loads precomputed point clouds from sharded .pt files.

    All shards are loaded into memory at init for fast random access.
    For droid_100 (~10K frames) this is ~1.1GB — trivially fits in RAM.
    """

    def __init__(self, cache_dir: str, point_cloud_size: int):
        """
        Args:
            cache_dir: Path to the cache directory containing metadata.json and shard files.
            point_cloud_size: Expected point cloud size from training config (validated against cache).
        """
        self.cache_dir = cache_dir

        # Load metadata
        metadata_path = os.path.join(cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No metadata.json found in {cache_dir}. "
                f"Run the preprocessing script first."
            )

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.total_frames = self.metadata["total_frames"]
        self.num_shards = self.metadata["num_shards"]
        self.feature_dim = self.metadata["feature_dim"]
        cached_pc_size = self.metadata["point_cloud_size"]

        # Validate point cloud size matches config
        if cached_pc_size != point_cloud_size:
            raise ValueError(
                f"Cache was built with point_cloud_size={cached_pc_size}, "
                f"but training config specifies point_cloud_size={point_cloud_size}. "
                f"Regenerate the cache with the correct point_cloud_size."
            )

        # Load all shards into memory
        print(f"Loading {self.num_shards} shards from {cache_dir}...")
        shards = []
        for i in range(self.num_shards):
            shard_path = os.path.join(cache_dir, f"shard_{i:04d}.pt")
            shards.append(torch.load(shard_path, map_location="cpu"))
        self.data = torch.cat(shards, dim=0)
        print(f"Loaded {self.data.shape[0]} frames, shape {self.data.shape}")

        assert self.data.shape[0] == self.total_frames, (
            f"Expected {self.total_frames} frames from metadata, "
            f"but loaded {self.data.shape[0]}"
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def __len__(self) -> int:
        return self.total_frames


def build_cached_dataset(split: str, cfg) -> CachedPointCloudDataset:
    """Build a CachedPointCloudDataset from config.

    Args:
        split: 'train' or 'val'
        cfg: Config object, expected to have cfg.cache.root_dir, cfg.cache.dataset_name,
             and cfg.vae.point_cloud_size (or cfg.model.point_cloud_size).
    """
    cache_dir = os.path.join(cfg.cache.root_dir, cfg.cache.dataset_name, split)

    # Support both cfg.vae.point_cloud_size and cfg.model.point_cloud_size
    if hasattr(cfg, "vae") and hasattr(cfg.vae, "point_cloud_size"):
        point_cloud_size = cfg.vae.point_cloud_size
    elif hasattr(cfg, "model") and hasattr(cfg.model, "point_cloud_size"):
        point_cloud_size = cfg.model.point_cloud_size
    else:
        raise ValueError("Config must specify point_cloud_size in either vae or model section")

    return CachedPointCloudDataset(
        cache_dir=cache_dir,
        point_cloud_size=point_cloud_size,
    )
