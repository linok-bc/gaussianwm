"""
Visualize VAE reconstruction quality on cached point clouds.

Usage:
    python scripts/eval/visualize_vae.py \
        resume=logs/ae/checkpoint-50.pth \
        cache.dataset_name=droid_100
"""

import os
import sys
import json
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from termcolor import cprint

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gaussianwm.encoder.models_ae import create_autoencoder
from gaussianwm.processor.cached_dataset import build_cached_dataset


SH_C0 = 0.28209479177387814

# Feature dimension layout for pred1: [means(3), scales(3), rotations(4), sh(3), opacity(1)] = 14D
FEATURE_NAMES = [
    "mean_x", "mean_y", "mean_z",
    "scale_x", "scale_y", "scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "sh_r", "sh_g", "sh_b",
    "opacity",
]

FEATURE_GROUPS = {
    "means": slice(0, 3),
    "scales": slice(3, 6),
    "rotations": slice(6, 10),
    "sh_color": slice(10, 13),
    "opacity": slice(13, 14),
}


def sh_to_rgb(sh: np.ndarray) -> np.ndarray:
    """Convert SH-normalized color values to RGB [0, 1] for visualization.

    The dataset's __getitem__ applies SH normalization: (0.5 + SH_C0 * raw_sh) / 255.
    These values are very small, so we scale them up for scatter plot coloring.
    """
    # Scale to visible range — these are small normalized values
    rgb = sh * 255.0
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb


def reconstruct(model, points, device):
    """Run VAE encode -> decode and return reconstruction."""
    model.eval()
    with torch.no_grad():
        points_dev = points.unsqueeze(0).to(device)
        outputs = model(points_dev, points_dev)
        recon = outputs["logits"].squeeze(0).cpu()
    return recon


def plot_point_cloud_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    sample_idx: int,
    mse: float,
    save_path: str,
):
    """Plot original vs reconstructed point cloud side by side, colored by SH."""
    fig = plt.figure(figsize=(16, 7))

    # Extract xyz and colors
    orig_xyz = original[:, :3]
    recon_xyz = reconstructed[:, :3]

    # SH color is always at dims 10:13 in 14D layout
    feat_dim = original.shape[1]
    sh_slice = slice(10, 13)

    orig_colors = sh_to_rgb(original[:, sh_slice])
    recon_colors = sh_to_rgb(reconstructed[:, sh_slice])

    # Compute shared axis limits
    all_xyz = np.concatenate([orig_xyz, recon_xyz], axis=0)
    xyz_min = all_xyz.min(axis=0)
    xyz_max = all_xyz.max(axis=0)
    margin = (xyz_max - xyz_min) * 0.05

    # Original
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(orig_xyz[:, 0], orig_xyz[:, 1], orig_xyz[:, 2],
                c=orig_colors, s=1, alpha=0.6)
    ax1.set_xlim(xyz_min[0] - margin[0], xyz_max[0] + margin[0])
    ax1.set_ylim(xyz_min[1] - margin[1], xyz_max[1] + margin[1])
    ax1.set_zlim(xyz_min[2] - margin[2], xyz_max[2] + margin[2])
    ax1.set_title("Original")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Reconstructed
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(recon_xyz[:, 0], recon_xyz[:, 1], recon_xyz[:, 2],
                c=recon_colors, s=1, alpha=0.6)
    ax2.set_xlim(xyz_min[0] - margin[0], xyz_max[0] + margin[0])
    ax2.set_ylim(xyz_min[1] - margin[1], xyz_max[1] + margin[1])
    ax2.set_zlim(xyz_min[2] - margin[2], xyz_max[2] + margin[2])
    ax2.set_title("Reconstructed")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    fig.suptitle(f"Sample {sample_idx} — MSE: {mse:.6f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_dimension_error(all_errors: np.ndarray, feat_dim: int, save_path: str):
    """Bar chart of mean absolute error per feature dimension."""
    mean_errors = np.mean(np.abs(all_errors), axis=(0, 1))  # average over samples and points

    fig, ax = plt.subplots(figsize=(14, 5))

    labels = FEATURE_NAMES[:feat_dim] if feat_dim <= len(FEATURE_NAMES) else [f"dim_{i}" for i in range(feat_dim)]
    colors_list = []
    group_colors = {
        "means": "#1f77b4",
        "means_other_view": "#ff7f0e",
        "scales": "#2ca02c",
        "rotations": "#d62728",
        "sh_color": "#9467bd",
        "opacity": "#8c564b",
    }
    for i in range(feat_dim):
        assigned = "#7f7f7f"
        for group_name, group_slice in FEATURE_GROUPS.items():
            if group_slice.start <= i < group_slice.stop:
                assigned = group_colors[group_name]
                break
        colors_list.append(assigned)

    bars = ax.bar(range(feat_dim), mean_errors, color=colors_list)
    ax.set_xticks(range(feat_dim))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Per-Dimension Reconstruction Error")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=name) for name, c in group_colors.items()
                       if any(group_colors[name] == colors_list[i] for i in range(feat_dim))]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_point_error_histogram(all_errors: np.ndarray, save_path: str):
    """Histogram of per-point MSE across all samples."""
    per_point_mse = np.mean(all_errors ** 2, axis=-1).flatten()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(per_point_mse, bins=100, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Per-Point MSE")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Point MSE Distribution (mean={per_point_mse.mean():.6f}, median={np.median(per_point_mse):.6f})")
    ax.axvline(per_point_mse.mean(), color="red", linestyle="--", label=f"Mean: {per_point_mse.mean():.6f}")
    ax.axvline(np.median(per_point_mse), color="orange", linestyle="--", label=f"Median: {np.median(per_point_mse):.6f}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@hydra.main(version_base=None, config_path="../../configs", config_name="train_vae")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_samples = 10

    # Output directory
    output_dir = Path(cfg.output_dir) / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    cprint(f"Saving visualizations to {output_dir}", "blue")

    # Load model
    cprint("Loading model...", "blue")
    model = create_autoencoder(
        depth=cfg.vae.vae_depth,
        dim=cfg.vae.latent_dim,
        M=cfg.vae.num_latents,
        latent_dim=cfg.vae.latent_dim,
        output_dim=cfg.vae.output_dim,
        N=cfg.vae.point_cloud_size,
        deterministic=not cfg.vae.use_kl,
    ).to(device)

    # Load checkpoint
    if cfg.resume is None:
        raise ValueError("Must specify resume=<path_to_checkpoint> to load trained weights")

    cprint(f"Loading checkpoint from {cfg.resume}", "blue")
    checkpoint = torch.load(cfg.resume, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    cprint("Model loaded", "green")

    # Load val dataset
    cprint("Loading val dataset...", "blue")
    dataset = build_cached_dataset("val", cfg)
    cprint(f"Val dataset size: {len(dataset)}", "blue")

    # Select evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    cprint(f"Visualizing {num_samples} samples at indices: {indices.tolist()}", "blue")

    # Run reconstructions and collect errors
    all_errors = []
    per_sample_metrics = []

    for i, idx in enumerate(indices):
        points = dataset[idx]
        recon = reconstruct(model, points, device)

        error = (recon - points).numpy()
        mse = float(np.mean(error ** 2))
        all_errors.append(error)

        per_sample_metrics.append({
            "sample_idx": int(idx),
            "mse": mse,
        })

        # Plot 3D comparison
        save_path = output_dir / f"sample_{i:03d}_idx{idx}.png"
        plot_point_cloud_comparison(
            original=points.numpy(),
            reconstructed=recon.numpy(),
            sample_idx=int(idx),
            mse=mse,
            save_path=str(save_path),
        )
        cprint(f"Sample {i} (idx={idx}): MSE={mse:.6f} — saved to {save_path.name}", "cyan")

    all_errors = np.stack(all_errors)  # [num_samples, num_points, feat_dim]
    feat_dim = all_errors.shape[-1]

    # Per-dimension error plot
    plot_per_dimension_error(all_errors, feat_dim, str(output_dir / "per_dimension_error.png"))
    cprint("Saved per-dimension error plot", "green")

    # Per-point error histogram
    plot_per_point_error_histogram(all_errors, str(output_dir / "per_point_error_hist.png"))
    cprint("Saved per-point error histogram", "green")

    # Save summary
    overall_mse = float(np.mean(all_errors ** 2))
    per_dim_mse = np.mean(all_errors ** 2, axis=(0, 1)).tolist()

    summary = {
        "overall_mse": overall_mse,
        "per_dimension_mse": per_dim_mse,
        "num_samples": num_samples,
        "samples": per_sample_metrics,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    cprint(f"Saved summary to {summary_path}", "green")

    cprint(f"\nOverall MSE: {overall_mse:.6f}", "blue")
    cprint(f"Done! All outputs in {output_dir}", "green")


if __name__ == "__main__":
    main()
