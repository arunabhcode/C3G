"""Generic binary mask metrics: IoU, boundary IoU, and warp IoU (pred vs pred)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass
class MaskMetricScores:
    iou: float
    boundary_iou: float
    best_index: int = 0
    warp_iou: Optional[float] = None


def _to_bool_numpy(mask: np.ndarray | Tensor) -> np.ndarray:
    if isinstance(mask, Tensor):
        mask = mask.detach().cpu().numpy()
    if mask.dtype == np.bool_:
        return mask
    return mask > 128 if mask.dtype != np.bool_ else mask


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Boundary band of a binary mask (Gaussian-Grouping / boundary-IoU style)."""
    import cv2

    mask_u8 = (_to_bool_numpy(mask)).astype(np.uint8)
    h, w = mask_u8.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)[1 : h + 1, 1 : w + 1]
    return mask_u8 - eroded


def mask_iou(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
) -> float:
    """Standard binary IoU between prediction and ground-truth (or reference) mask."""
    m1 = _to_bool_numpy(pred)
    m2 = _to_bool_numpy(gt)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def boundary_iou(
    pred: np.ndarray | Tensor,
    gt: np.ndarray | Tensor,
    dilation_ratio: float = 0.02,
) -> float:
    """IoU computed on boundary pixels only (pred vs label mask)."""
    pred_b = mask_to_boundary(_to_bool_numpy(pred).astype(np.uint8), dilation_ratio)
    gt_b = mask_to_boundary(_to_bool_numpy(gt).astype(np.uint8), dilation_ratio)
    intersection = ((pred_b > 0) & (gt_b > 0)).sum()
    union = ((pred_b > 0) | (gt_b > 0)).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def best_multimask_scores(
    pred_masks: np.ndarray | Tensor,
    gt_mask: np.ndarray | Tensor,
    *,
    dilation_ratio: float = 0.02,
) -> MaskMetricScores:
    """
    When the model returns K masks (e.g. SAM multimask), pick the one with highest IoU
    vs ground truth and report its IoU and boundary IoU.
    """
    if isinstance(pred_masks, Tensor):
        pred_masks = pred_masks.detach().cpu().numpy()
    if pred_masks.ndim == 2:
        pred_masks = pred_masks[None]

    gt_mask = _to_bool_numpy(gt_mask)
    best_iou, best_biou, best_k = 0.0, 0.0, 0
    for k in range(pred_masks.shape[0]):
        iou = mask_iou(pred_masks[k], gt_mask)
        biou = boundary_iou(pred_masks[k], gt_mask, dilation_ratio=dilation_ratio)
        if iou > best_iou:
            best_iou, best_biou, best_k = iou, biou, k
    return MaskMetricScores(iou=best_iou, boundary_iou=best_biou, best_index=best_k)


def warp_mask_to_pose(
    mask: np.ndarray | Tensor,
    src_extrinsics: np.ndarray | Tensor,
    dst_extrinsics: np.ndarray | Tensor,
    src_intrinsics: np.ndarray | Tensor,
    dst_intrinsics: np.ndarray | Tensor,
    image_size: tuple[int, int],
) -> np.ndarray:
    """
    Warp a binary mask from the source camera into the destination camera frame.

    TODO: Implement via depth-based reprojection, plane homography, or mesh rasterization.
    """
    raise NotImplementedError(
        "warp_mask_to_pose is not implemented yet. "
        "Project the mask from src_extrinsics/intrinsics into dst_extrinsics/intrinsics "
        "using scene depth or a reference surface, then threshold the warped mask."
    )


def warp_mask_iou(
    pred_mask: np.ndarray | Tensor,
    reference_pred_mask: np.ndarray | Tensor,
    src_extrinsics: np.ndarray | Tensor,
    dst_extrinsics: np.ndarray | Tensor,
    src_intrinsics: np.ndarray | Tensor,
    dst_intrinsics: np.ndarray | Tensor,
    image_size: tuple[int, int],
) -> float:
    """
    IoU between a predicted mask warped into another camera and a reference prediction.

    Unlike boundary_iou, both operands are predictions (not GT).
    """
    warped = warp_mask_to_pose(
        pred_mask,
        src_extrinsics,
        dst_extrinsics,
        src_intrinsics,
        dst_intrinsics,
        image_size,
    )
    return mask_iou(warped, reference_pred_mask)


def scores_from_logits(
    pred_logits: Tensor,
    gt_masks: Tensor,
    *,
    threshold: float = 0.5,
    dilation_ratio: float = 0.02,
) -> list[MaskMetricScores]:
    """
    Decode (N, K, H, W) or (N, 1, H, W) logits vs (N, 1, H, W) GT; return per-sample scores.
    """
    if gt_masks.dim() == 5:
        gt_masks = gt_masks.squeeze(2)
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)

    pred = pred_logits.sigmoid() > threshold
    if pred.shape[-2:] != gt_masks.shape[-2:]:
        pred = torch.nn.functional.interpolate(
            pred.float(),
            size=gt_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ) > threshold

    n = pred.shape[0]
    results: list[MaskMetricScores] = []
    for i in range(n):
        pred_i = pred[i].detach().cpu().numpy()
        gt_i = gt_masks[i, 0].detach().cpu().numpy()
        results.append(
            best_multimask_scores(pred_i, gt_i, dilation_ratio=dilation_ratio)
        )
    return results


def mean_scores(scores: list[MaskMetricScores]) -> dict[str, float]:
    if not scores:
        return {"iou": 0.0, "boundary_iou": 0.0}
    warp_vals = [s.warp_iou for s in scores if s.warp_iou is not None]
    out = {
        "iou": float(np.mean([s.iou for s in scores])),
        "boundary_iou": float(np.mean([s.boundary_iou for s in scores])),
    }
    if warp_vals:
        out["warp_iou"] = float(np.mean(warp_vals))
    return out
