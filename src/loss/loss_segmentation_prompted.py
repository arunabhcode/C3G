"""Prompted BCE + Dice loss with best-of-3 mask selection."""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.prompt_sampler import PromptSampler, decompose_label_map
from ..model.sam_decoder import SAMMaskDecoderWrapper
from ..model.types import Gaussians
from .loss import Loss

logger = logging.getLogger(__name__)


@dataclass
class LossSegmentationPromptedCfg:
    weight: float
    sam_checkpoint: str = ""
    sam_model_variant: str = "sam_vit_h"
    use_lora: bool = False
    lora_rank: int = 4
    prompt_strategy: str = "centroid"
    min_object_pixels: int = 16


@dataclass
class LossSegmentationPromptedCfgWrapper:
    segmentation_prompted: LossSegmentationPromptedCfg


class LossSegmentationPrompted(
    Loss[LossSegmentationPromptedCfg, LossSegmentationPromptedCfgWrapper]
):
    """Prompted BCE + Dice loss with best-of-3 mask selection."""

    def __init__(self, cfg: LossSegmentationPromptedCfgWrapper):
        super().__init__(cfg)

        self.mask_decoder = SAMMaskDecoderWrapper(
            sam_checkpoint=self.cfg.sam_checkpoint,
            model_variant=self.cfg.sam_model_variant,
            use_lora=self.cfg.use_lora,
            lora_rank=self.cfg.lora_rank,
        )

        self.prompt_sampler = PromptSampler(
            strategy=self.cfg.prompt_strategy,
            min_object_pixels=self.cfg.min_object_pixels,
        )

    def dice_loss(self, pred, target):
        """Compute dice loss between sigmoid predictions and binary targets (fp32)."""
        pred_sigmoid = torch.sigmoid(pred.float())
        target = target.float()
        pred_flat = pred_sigmoid.flatten(1)
        target_flat = target.flatten(1)
        intersection = (pred_flat * target_flat).sum(1)
        union = pred_flat.sum(1) + target_flat.sum(1)
        loss = 1 - (2 * intersection + 1) / (union + 1)
        return loss.mean()

    def sigmoid_bce_loss(self, pred, target):
        """Compute binary cross-entropy loss with logits (fp32 for stability)."""
        return F.binary_cross_entropy_with_logits(
            pred.float(), target.float(), reduction="mean"
        )

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        target_image: Float[Tensor, "batch view 3 height width"] | None = None,
    ) -> Float[Tensor, ""]:
        label_maps = batch["target"]["label"]
        target_view_count = label_maps.shape[1]
        rendered_features = prediction.feature[:, :target_view_count]
        B, V, C, H, W = rendered_features.shape
        device = rendered_features.device

        total_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        for b in range(B):
            for v in range(V):
                label_map = label_maps[b, v]
                binary_masks = decompose_label_map(label_map).to(device)

                if binary_masks.shape[0] == 0:
                    logger.warning(
                        f"Skipping prompted loss for batch {b}, view {v}: label map is all background"
                    )
                    continue

                valid_masks = [
                    i
                    for i in range(binary_masks.shape[0])
                    if binary_masks[i].sum().item() >= self.cfg.min_object_pixels
                ]
                if len(valid_masks) == 0:
                    logger.warning(
                        f"Skipping prompted loss for batch {b}, view {v}: no mask with enough pixels"
                    )
                    continue

                point_coords, point_labels, gt_mask = self.prompt_sampler.sample(
                    binary_masks
                )
                point_coords = point_coords.unsqueeze(0).to(device)
                point_labels = point_labels.unsqueeze(0).to(device)

                features = rendered_features[b, v].unsqueeze(0)
                features_64 = F.interpolate(
                    features, size=(64, 64), mode="bilinear", align_corners=False
                )

                pred_masks = self.mask_decoder(
                    features_64, point_coords=point_coords, point_labels=point_labels
                )

                _, num_masks, MH, MW = pred_masks.shape
                gt_mask_2d = gt_mask.to(device).unsqueeze(0).unsqueeze(0)
                gt_resized = F.interpolate(gt_mask_2d, size=(MH, MW), mode="nearest")

                best_loss = None
                for m in range(num_masks):
                    candidate = pred_masks[:, m : m + 1, :, :]
                    bce = self.sigmoid_bce_loss(candidate, gt_resized)
                    dice = self.dice_loss(candidate, gt_resized)
                    candidate_loss = bce + dice
                    if best_loss is None or candidate_loss < best_loss:
                        best_loss = candidate_loss

                total_loss = total_loss + best_loss
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return self.cfg.weight * (total_loss / valid_count)
