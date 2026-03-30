"""
Rendering-based VAE visualization using cached full-resolution data.

Loads full-resolution pred1, pred2, and images from the cache.
Runs VAE encode->decode, writes reconstructed features back into pred dicts,
renders both original and reconstructed Gaussians using Splatt3r's decoder.

Usage:
    python scripts/eval/visualize_vae_render.py \
        resume=logs/ae/checkpoint-50.pth \
        cache.dataset_name=droid_100
"""

import os
import sys
import copy
import json
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from termcolor import cprint

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch3d.ops import sample_farthest_points as fps

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gaussianwm.encoder.models_ae import create_autoencoder
from gaussianwm.processor.cached_dataset import CachedPointCloudDataset
from gaussianwm.processor.regressor import gaussian_feature_to_dim

# Add splatt3r paths for the decoder and geometry
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r'))
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r/src/pixelsplat_src'))
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r/src/mast3r_src'))
sys.path.append(str(ROOT_DIR / 'third_party/splatt3r/src/mast3r_src/dust3r'))

import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.geometry as geometry

SH_C0 = 0.28209479177387814


def tensor_14d_to_pred_dict(features_14d, H, W, is_pred2=False):
    """Convert a [1, H*W, 14] tensor back into a Splatt3r pred dict.

    For pred1: dims are means(3), scales(3), rotations(4), sh(3), opacity(1)
    For pred2: dims are means_in_other_view(3), scales(3), rotations(4), sh(3), opacity(1)
    """
    B = features_14d.shape[0]
    f = features_14d

    if is_pred2:
        pred = {
            'pts3d_in_other_view': f[..., 0:3].reshape(B, H, W, 3),
            'means_in_other_view': f[..., 0:3].reshape(B, H, W, 3),
        }
    else:
        pred = {
            'pts3d': f[..., 0:3].reshape(B, H, W, 3),
            'means': f[..., 0:3].reshape(B, H, W, 3),
        }

    pred['scales'] = f[..., 3:6].reshape(B, H, W, 3)
    pred['rotations'] = f[..., 6:10].reshape(B, H, W, 4)
    pred['sh'] = f[..., 10:13].reshape(B, H, W, 3, 1)  # [B, H, W, C, D_sh]
    pred['opacities'] = f[..., 13:14].reshape(B, H, W, 1)
    pred['covariances'] = geometry.build_covariance(pred['scales'], pred['rotations'])

    return pred


def render_gaussians(decoder, pred1, pred2, image_size, device):
    """Render Gaussians using Splatt3r's decoder."""
    H, W = image_size

    # The decoder uses context[0]['camera_pose'] as the base coordinate frame,
    # then transforms target poses relative to it. Using identity for both
    # means we render from the same viewpoint as the Gaussians' coordinate frame.
    batch = {
        'target': [{
            'camera_pose': torch.eye(4).unsqueeze(0).to(device),
            'camera_intrinsics': torch.eye(3).unsqueeze(0).to(device),
        }],
        'context': [
            {'camera_pose': torch.eye(4).unsqueeze(0).to(device)},
            {},
        ],
    }

    try:
        color, _ = decoder(batch, pred1, pred2, (H, W))
        rendered = color[0, 0].detach().permute(1, 2, 0).cpu().numpy()
        rendered = np.clip(rendered, 0, 1)
        return rendered
    except Exception as e:
        cprint(f"Rendering failed: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


def vae_reconstruct_full_grid(model, features_14d_normalized, num_inputs, device):
    """Run VAE encode->decode on normalized 14D features at full grid resolution.

    Args:
        model: VAE model
        features_14d_normalized: [1, H*W, 14] SH-normalized features
        num_inputs: VAE encoder's expected input size (e.g. 2048)
        device: torch device

    Returns:
        [1, H*W, 14] reconstructed normalized features
    """
    points = features_14d_normalized.to(device)

    # FPS downsample to match VAE encoder's expected input size
    points_fps, fps_indices = fps(points[..., :3], K=num_inputs)
    # Gather the full 14D features at FPS indices
    points_fps_full = points[0, fps_indices[0]].unsqueeze(0)  # [1, num_inputs, 14]

    # Encode
    latent = model.encode(points_fps_full)

    # Decode with ALL original points as queries
    recon = model.decode(latent, queries=points)

    return recon  # [1, H*W, 14]


def save_comparison(original_img, original_render, recon_render, recon_geo_render,
                    sample_idx, mse, save_path):
    """Save a 4-panel comparison image."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    panels = [
        (original_img, "Source Image"),
        (original_render, "Original Gaussians Render"),
        (recon_render, f"VAE Full Reconstruction\nMSE: {mse:.6f}"),
        (recon_geo_render, "VAE Geometry Only\n(Original SH Colors)"),
    ]

    for ax, (img, title) in zip(axes, panels):
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Render failed", ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(f"Sample {sample_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


@hydra.main(version_base=None, config_path="../../configs", config_name="train_vae")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_samples = 5

    output_dir = Path(cfg.output_dir) / "render_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    cprint(f"Saving visualizations to {output_dir}", "blue")

    # Load VAE
    cprint("Loading VAE...", "blue")
    model = create_autoencoder(
        depth=cfg.vae.vae_depth,
        dim=cfg.vae.latent_dim,
        M=cfg.vae.num_latents,
        latent_dim=cfg.vae.latent_dim,
        output_dim=cfg.vae.output_dim,
        N=cfg.vae.point_cloud_size,
        deterministic=not cfg.vae.use_kl,
    ).to(device)

    if cfg.resume is None:
        raise ValueError("Must specify resume=<path_to_checkpoint>")

    checkpoint = torch.load(cfg.resume, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    cprint("VAE loaded", "green")

    # Load Splatt3r's renderer
    cprint("Loading renderer...", "blue")
    decoder = pixelsplat_decoder.DecoderSplattingCUDA(
        background_color=[0.0, 0.0, 0.0]
    ).to(device)
    cprint("Renderer loaded", "green")

    # Load cached dataset (val split) with raw access
    cache_dir = os.path.join(cfg.cache.root_dir, cfg.cache.dataset_name, "val")
    cprint(f"Loading cache from {cache_dir}...", "blue")
    dataset = CachedPointCloudDataset(
        cache_dir=cache_dir,
        point_cloud_size=cfg.vae.point_cloud_size,
    )
    cprint(f"Cache loaded: {len(dataset)} frames", "green")

    # Select evenly spaced samples
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    metrics = []

    for i, idx in enumerate(indices):
        cprint(f"\nSample {i} (cache idx {idx})...", "yellow")

        raw = dataset.get_raw(idx)
        pred1_14d = raw['pred1'].unsqueeze(0)  # [1, H*W, 14]
        pred2_14d = raw['pred2'].unsqueeze(0)  # [1, H*W, 14]
        image_uint8 = raw['image']               # [H_img, W_img, 3] uint8
        H, W = raw['H'], raw['W']

        original_img = image_uint8.numpy().astype(np.float32) / 255.0

        # Build pred dicts from raw cached data (no SH normalization applied)
        pred1 = tensor_14d_to_pred_dict(pred1_14d.to(device), H, W, is_pred2=False)
        pred2 = tensor_14d_to_pred_dict(pred2_14d.to(device), H, W, is_pred2=True)

        # Render original Gaussians
        cprint("  Rendering original...", "cyan")
        original_render = render_gaussians(decoder, pred1, pred2, (H, W), device)

        with torch.no_grad():
            recon = vae_reconstruct_full_grid(
                model, pred1_14d, cfg.vae.point_cloud_size, device
            )

        # Compute MSE in normalized space (matches training loss)
        mse = float(((recon - pred1_14d.to(device)) ** 2).mean().cpu())
        cprint(f"  MSE: {mse:.6f}", "cyan")

        # Denormalize and build reconstructed pred dict
        recon = recon.cpu()
        pred1_recon = tensor_14d_to_pred_dict(recon.to(device), H, W, is_pred2=False)

        # Render full VAE reconstruction
        cprint("  Rendering full reconstruction...", "cyan")
        recon_render = render_gaussians(decoder, pred1_recon, pred2, (H, W), device)

        # Render geometry-only reconstruction (VAE geometry + original SH/opacity)
        cprint("  Rendering geometry-only reconstruction...", "cyan")
        recon_geo = recon.clone()
        # Overwrite SH and opacity with originals
        recon_geo[..., 10:14] = pred1_14d[..., 10:14]
        pred1_geo = tensor_14d_to_pred_dict(recon_geo.to(device), H, W, is_pred2=False)
        recon_geo_render = render_gaussians(decoder, pred1_geo, pred2, (H, W), device)

        # Save comparison
        save_path = output_dir / f"render_sample_{i:03d}_idx{idx}.png"
        save_comparison(original_img, original_render, recon_render, recon_geo_render,
                        idx, mse, str(save_path))
        cprint(f"  Saved to {save_path.name}", "green")

        metrics.append({"sample_idx": int(idx), "mse": mse})

    # Save summary
    summary = {
        "num_samples": len(metrics),
        "overall_mse": float(np.mean([m["mse"] for m in metrics])),
        "samples": metrics,
    }
    summary_path = output_dir / "render_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    cprint(f"\nOverall MSE: {summary['overall_mse']:.6f}", "blue")
    cprint(f"Done! All outputs in {output_dir}", "green")


if __name__ == "__main__":
    main()
