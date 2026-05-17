"""C3G SAM mask decoder wrapper (precomputed 64x64 embeddings, not full vanilla SAM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.sam.constants import GRID_SIZE, SAM_MODELS
from src.model.sam.loader import load_sam
from src.model.sam.preprocess import generate_grid_points


class SAMMaskDecoderWrapper(nn.Module):
    """Wraps SAM's mask decoder to accept rendered Gaussian features (Bx256x64x64)."""

    def __init__(
        self, sam_checkpoint, model_variant="sam_vit_h", use_lora=False, lora_rank=4
    ):
        super().__init__()

        if model_variant not in SAM_MODELS:
            raise ValueError(
                f"Unsupported SAM model variant '{model_variant}'. "
                f"Supported: {list(SAM_MODELS.keys())}"
            )

        sam = load_sam(model_variant, sam_checkpoint, freeze=True)
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder

        self.use_lora = use_lora
        if use_lora:
            self.inject_lora(lora_rank)

    def inject_lora(self, rank):
        """Inject LoRA on v_proj layers in token-to-image cross-attention."""
        from src.model.lora import inject_lora

        for layer in self.mask_decoder.transformer.layers:
            inject_lora(layer, "cross_attn_token_to_image.v_proj", rank=rank)

        inject_lora(
            self.mask_decoder.transformer, "final_attn_token_to_image.v_proj", rank=rank
        )

    def forward(
        self, rendered_features, point_coords=None, point_labels=None, box=None
    ):
        B, C, H, W = rendered_features.shape

        if H != 64 or W != 64:
            image_embeddings = F.interpolate(
                rendered_features, size=(64, 64), mode="bilinear", align_corners=False
            )
        else:
            image_embeddings = rendered_features

        image_pe = self.prompt_encoder.get_dense_pe()
        has_prompts = point_coords is not None or box is not None

        all_masks = []
        for i in range(B):
            if has_prompts:
                pts = None
                if point_coords is not None:
                    pts = (point_coords[i : i + 1], point_labels[i : i + 1])
                bx = box[i : i + 1] if box is not None else None
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=pts, boxes=bx, masks=None
                )
            else:
                grid_pts, grid_labels = generate_grid_points(
                    1, rendered_features.device, grid_size=GRID_SIZE
                )
                sparse_emb, dense_emb = self.prompt_encoder(
                    points=(grid_pts, grid_labels), boxes=None, masks=None
                )

            masks_i, _ = self.mask_decoder(
                image_embeddings=image_embeddings[i : i + 1],
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            all_masks.append(masks_i)

        return torch.cat(all_masks, dim=0)
