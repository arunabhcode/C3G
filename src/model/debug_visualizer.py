import numpy as np
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.loggers.wandb import WandbLogger
from matplotlib import cm

from src.misc.utils import inverse_normalize
from src.model.utils import run_pca


def log_debug_visualizations(
    logger,
    global_step,
    checkpoint_interval,
    target_rgb,
    target_sam_feature,
    rendered_feature,
    img_size,
):
    """Log debug visualization table and feature norm stats to wandb."""
    if global_step % checkpoint_interval != 0:
        return
    if not isinstance(logger, WandbLogger):
        return
    if rendered_feature is None:
        return

    h, w = img_size
    target_rgb_norm = inverse_normalize(target_rgb)

    target_pca = run_pca(target_sam_feature.unsqueeze(0), (h, w))
    rendered_pca = run_pca(rendered_feature.unsqueeze(0), (h, w))

    mse_map = compute_mse_heatmap(rendered_feature, target_sam_feature)
    cosine_map = compute_cosine_error_map(rendered_feature, target_sam_feature)

    mse_resized = (
        F.interpolate(
            mse_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    cosine_resized = (
        F.interpolate(
            cosine_map.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )

    mse_colored = colorize_heatmap(mse_resized)
    cosine_colored = colorize_heatmap(cosine_resized)

    mse_overlay = alpha_blend(mse_colored, target_rgb_norm)
    cosine_overlay = alpha_blend(cosine_colored, target_rgb_norm)

    table = wandb.Table(
        columns=[
            "target_rgb",
            "target_sam_pca",
            "rendered_feature_pca",
            "mse_heatmap_overlay",
            "cosine_error_overlay",
        ]
    )
    table.add_data(
        wandb.Image(to_hwc_uint8(target_rgb_norm)),
        wandb.Image(to_hwc_uint8(target_pca.squeeze(0))),
        wandb.Image(to_hwc_uint8(rendered_pca.squeeze(0))),
        wandb.Image(to_hwc_uint8(mse_overlay)),
        wandb.Image(to_hwc_uint8(cosine_overlay)),
    )
    logger.experiment.log({"val/debug_visualizations": table}, step=global_step)

    target_norms = compute_feature_norms(target_sam_feature)
    rendered_norms = compute_feature_norms(rendered_feature)

    logger.experiment.log(
        {
            "val/feature_norm_histogram": wandb.Histogram(
                np.concatenate(
                    [
                        target_norms.detach().cpu().numpy(),
                        rendered_norms.detach().cpu().numpy(),
                    ]
                )
            ),
            "val/target_feature_norm_min": target_norms.min().item(),
            "val/target_feature_norm_max": target_norms.max().item(),
            "val/target_feature_norm_std": target_norms.std().item(),
            "val/rendered_feature_norm_min": rendered_norms.min().item(),
            "val/rendered_feature_norm_max": rendered_norms.max().item(),
            "val/rendered_feature_norm_std": rendered_norms.std().item(),
        },
        step=global_step,
    )


def to_hwc_uint8(tensor):
    """Convert (3, H, W) float tensor in [0, 1] to HWC uint8 numpy array."""
    return (
        (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0) * 255)
        .to(torch.uint8)
        .numpy()
    )


def compute_mse_heatmap(rendered_feature, target_feature):
    """Compute per-pixel MSE across channels, returns (H, W) tensor."""
    return ((rendered_feature - target_feature) ** 2).mean(dim=0)


def compute_cosine_error_map(rendered_feature, target_feature):
    """Compute 1 - cosine_similarity per pixel, returns (H, W) tensor."""
    eps = 1e-8
    rendered_norm = rendered_feature.norm(dim=0, keepdim=True).clamp(min=eps)
    target_norm = target_feature.norm(dim=0, keepdim=True).clamp(min=eps)
    rendered_normalized = rendered_feature / rendered_norm
    target_normalized = target_feature / target_norm
    cosine_similarity = (rendered_normalized * target_normalized).sum(dim=0)
    return 1 - cosine_similarity


def colorize_heatmap(heatmap, vmin=0.0, vmax=None):
    """Apply viridis colormap to a heatmap, returns (3, H, W) in [0, 1]."""
    if vmax is None:
        vmax = heatmap.max()
    normalized = ((heatmap - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)
    cmap = cm.get_cmap("viridis")
    rgb_np = cmap(normalized.detach().cpu().numpy())[..., :3]
    rgb = torch.tensor(rgb_np, device=heatmap.device, dtype=torch.float32)
    return rgb.permute(2, 0, 1)


def alpha_blend(overlay, background, alpha=0.5):
    """Blend overlay onto background, both (3, H, W) in [0, 1], returns (3, H, W)."""
    return alpha * overlay + (1 - alpha) * background


def compute_feature_norms(feature):
    """Compute per-spatial-position L2 norms, returns flat vector of length H*W."""
    return feature.norm(dim=0).flatten()
