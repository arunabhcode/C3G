import torch


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
