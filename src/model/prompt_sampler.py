"""Generates point prompts from GT binary masks for prompted SAM training."""

from __future__ import annotations

import torch

from src.model.sam.constants import SAM_IMAGE_SIZE


class PromptSampler:
    """Generates point prompts from GT binary masks for prompted SAM training."""

    def __init__(
        self, strategy="centroid", min_object_pixels=16, image_size=SAM_IMAGE_SIZE
    ):
        self.strategy = strategy
        self.min_object_pixels = min_object_pixels
        self.image_size = image_size

    def sample(self, binary_masks):
        """Pick a random valid mask, generate point prompt."""
        K, H, W = binary_masks.shape
        valid_indices = []
        for i in range(K):
            if binary_masks[i].sum().item() >= self.min_object_pixels:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            raise ValueError("No mask has enough foreground pixels for prompt sampling")

        idx = valid_indices[torch.randint(len(valid_indices), size=()).item()]
        selected_mask = binary_masks[idx]

        if self.strategy == "centroid":
            row, col = self.compute_centroid(selected_mask)
        else:
            row, col = self.sample_random_point(selected_mask)

        col_norm = col * (self.image_size / W)
        row_norm = row * (self.image_size / H)

        point_coords = torch.tensor([[col_norm, row_norm]], dtype=torch.float32)
        point_labels = torch.tensor([1], dtype=torch.int64)

        return point_coords, point_labels, selected_mask

    def compute_centroid(self, mask):
        """Mean row/col of foreground pixels, rounded to nearest int."""
        fg = torch.nonzero(mask, as_tuple=False)
        mean_row = fg[:, 0].float().mean().round().long().item()
        mean_col = fg[:, 1].float().mean().round().long().item()
        return mean_row, mean_col

    def sample_random_point(self, mask):
        """Uniform random foreground pixel."""
        fg = torch.nonzero(mask, as_tuple=False)
        idx = torch.randint(fg.shape[0], size=()).item()
        row = fg[idx, 0].item()
        col = fg[idx, 1].item()
        return row, col
