from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything import sam_model_registry

SAM_MODELS = {
    "sam_vit_h": "vit_h",
    "sam_vit_l": "vit_l",
    "sam_vit_b": "vit_b",
}

GRID_SIZE = 8


class SAMMaskDecoderWrapper(nn.Module):
    """Wraps SAM's mask decoder to accept rendered Gaussian features."""

    def __init__(
        self, sam_checkpoint, model_variant="sam_vit_h", use_lora=False, lora_rank=4
    ):
        super().__init__()

        if model_variant not in SAM_MODELS:
            raise ValueError(
                f"Unsupported SAM model variant '{model_variant}'. "
                f"Supported: {list(SAM_MODELS.keys())}"
            )
        if not os.path.isfile(sam_checkpoint):
            raise FileNotFoundError(
                f"SAM checkpoint not found at '{sam_checkpoint}'. "
                f"Please download the SAM weights to this path."
            )

        sam_type = SAM_MODELS[model_variant]
        sam = sam_model_registry[sam_type](checkpoint=sam_checkpoint)

        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder

        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        self.use_lora = use_lora
        if use_lora:
            self.inject_lora(lora_rank)

    def inject_lora(self, rank):
        """Inject LoRA on v_proj layers in token-to-image cross-attention."""
        from src.model.lora import inject_lora

        for i, layer in enumerate(self.mask_decoder.transformer.layers):
            inject_lora(layer, "cross_attn_token_to_image.v_proj", rank=rank)

        inject_lora(
            self.mask_decoder.transformer, "final_attn_token_to_image.v_proj", rank=rank
        )

    def generate_grid_points(self, batch_size, device):
        """Generate evenly-spaced grid points for segment-everything mode."""
        step = 1024 // GRID_SIZE
        offset = step // 2
        coords_h = torch.arange(offset, 1024, step, device=device, dtype=torch.float32)
        coords_w = torch.arange(offset, 1024, step, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords_h, coords_w, indexing="ij")
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        points = points.unsqueeze(0).expand(batch_size, -1, -1)
        labels = torch.ones(batch_size, points.shape[1], device=device, dtype=torch.int)
        return points, labels

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
                grid_pts, grid_labels = self.generate_grid_points(
                    1, rendered_features.device
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
