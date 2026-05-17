"""IoU and Boundary-IoU metrics for LERF-Mask (Gaussian-Grouping benchmark)."""

from __future__ import annotations

import numpy as np


def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Convert a binary mask to its boundary band (Gaussian-Grouping reference)."""
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    h, w = mask_u8.shape
    img_diag = np.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)[1 : h + 1, 1 : w + 1]
    return mask_u8 - eroded


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    m1 = mask1 > 128 if mask1.dtype != np.bool_ else mask1
    m2 = mask2 > 128 if mask2.dtype != np.bool_ else mask2
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def boundary_iou(gt: np.ndarray, pred: np.ndarray, dilation_ratio: float = 0.02) -> float:
    gt_u8 = ((gt > 128) if gt.dtype != np.bool_ else gt).astype(np.uint8)
    pred_u8 = ((pred > 128) if pred.dtype != np.bool_ else pred).astype(np.uint8)
    gt_boundary = mask_to_boundary(gt_u8, dilation_ratio)
    pred_boundary = mask_to_boundary(pred_u8, dilation_ratio)
    intersection = ((gt_boundary * pred_boundary) > 0).sum()
    union = ((gt_boundary + pred_boundary) > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def best_mask_iou(
    pred_masks: np.ndarray,
    gt_mask: np.ndarray,
) -> tuple[float, float, int]:
    """
    Score multi-mask SAM output by best IoU (and matching Boundary-IoU).

    pred_masks: (K, H, W) bool or uint8
    gt_mask: (H, W) bool or uint8
  """
    if pred_masks.ndim == 2:
        pred_masks = pred_masks[None]
    best_iou, best_biou, best_k = 0.0, 0.0, 0
    for k in range(pred_masks.shape[0]):
        iou = calculate_iou(gt_mask, pred_masks[k])
        biou = boundary_iou(gt_mask, pred_masks[k])
        if iou > best_iou:
            best_iou, best_biou, best_k = iou, biou, k
    return best_iou, best_biou, best_k
