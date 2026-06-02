#!/usr/bin/env python3
"""Export C3G-rendered SAM masks on Replica + ScanNet test (vanilla-eval layout).

One C3G forward per labeled frame, then batched SAM mask-decoder calls per class
(same coverage pattern as ``modal_eval_sam``).

Output layout::

    <output_root>/replica/<scene_id>/<frame_id>/<class_id>.png
    <output_root>/scannet/<scene_id>/<frame_id>/<class_id>.png

Run on Modal via ``modal_eval_c3gsam.py`` or locally::

    python -m src.eval_c3g_sam_scannet_masks \\
        +evaluation=c3g_sam_distill \\
        checkpointing.load=/path/to/distillation-base.ckpt
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

import hydra
import torch
import torch.nn.functional as F
from einops import repeat
from omegaconf import DictConfig
from PIL import Image

from src.config import load_typed_root_config
from src.dataset import get_dataset
from src.dataset.data_module import get_data_shim
from src.dataset.types import BatchedExample
from src.evaluation.mask_metrics import best_multimask_scores
from src.global_cfg import set_cfg
from src.misc.step_tracker import StepTracker
from src.modal_infra.modal_common import (
    C3G_EVAL_DATASETS,
    expected_mask_class_ids,
    filter_scenes_for_mask_export,
    frame_mask_export_complete,
    iter_dataset_frames,
)
from src.model.decoder import get_decoder
from src.model.distillation_wrapper import (
    DebugDecoderCfg,
    DistillationModelWrapper,
    DistillTrainCfg,
    OptimizerCfg as DistillOptimizerCfg,
)
from src.model.encoder import get_encoder
from src.model.prompt_sampler import PromptSampler, decompose_label_map
from src.model.sam_decoder import SAMMaskDecoderWrapper
from src.model.types import Gaussians
from src.loss import get_losses

logger = logging.getLogger(__name__)


class DistillMaskExportDataset(Protocol):
    cfg: object

    def _build_visualization_batch(self, scene: str, frame_id: str) -> dict | None: ...


def _batch_to_device(batch: BatchedExample, device: torch.device) -> BatchedExample:
    moved: dict = {"scene": batch["scene"]}
    for stage in ("context", "target"):
        moved[stage] = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch[stage].items()
        }
    return moved  # type: ignore[return-value]


def _add_batch_dim(example: dict) -> BatchedExample:
    batched: dict = {"scene": [example["scene"]]}
    for stage in ("context", "target"):
        stage_dict = {}
        for key, value in example[stage].items():
            if isinstance(value, torch.Tensor):
                stage_dict[key] = value.unsqueeze(0)
            else:
                stage_dict[key] = value
        batched[stage] = stage_dict
    return batched  # type: ignore[return-value]


def class_prompts_from_label(
    label_path: Path,
    image_shape: tuple[int, int],
    *,
    prompt_strategy: str,
    min_object_pixels: int,
) -> list[tuple[int, list[list[float]], list[int], torch.Tensor]]:
    """Return ``(class_id, point_coords, point_labels, gt_mask)`` per valid object."""
    import numpy as np

    label_np = np.array(Image.open(label_path))
    binary_masks = decompose_label_map(label_np)
    if binary_masks.shape[0] == 0:
        raise ValueError(f"No foreground objects in label map: {label_path}")

    h, w = image_shape
    sampler = PromptSampler(
        strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
        image_size=max(h, w),
    )

    unique_ids = np.unique(label_np)
    class_ids = [int(obj_id) for obj_id in unique_ids if obj_id != 0]

    prompts: list[tuple[int, list[list[float]], list[int], torch.Tensor]] = []
    for class_id, mask_idx in zip(class_ids, range(binary_masks.shape[0])):
        mask = binary_masks[mask_idx]
        if mask.sum().item() < min_object_pixels:
            continue

        if prompt_strategy == "centroid":
            row, col = sampler.compute_centroid(mask)
        else:
            row, col = sampler.sample_random_point(mask)

        x = col * (sampler.image_size / w)
        y = row * (sampler.image_size / h)
        prompts.append((class_id, [[float(x), float(y)]], [1], mask.float()))
    return prompts


def _save_mask_png(mask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype("uint8") * 255)).save(path)


@torch.no_grad()
def render_target_features(
    wrapper: DistillationModelWrapper,
    batch: BatchedExample,
) -> torch.Tensor:
    """Run distillation forward; return 64x64 features for target view 0."""
    _, _, _, h, w = batch["target"]["image"].shape
    wrapper._validate_distill_batch_shapes(batch, h, w)

    context_sam = batch["context"]["sam_features"]
    context_sam_enc = wrapper._downsample_for_encoder(context_sam, h, w)

    gaussians = wrapper.encoder(
        batch["context"],
        0,
        context_feature=context_sam_enc,
    )
    gaussians_detached = Gaussians(
        means=gaussians.means.detach(),
        covariances=gaussians.covariances.detach(),
        harmonics=gaussians.harmonics.detach(),
        opacities=gaussians.opacities.detach(),
        feature=gaussians.feature,
    )

    output = wrapper.decoder.forward(
        gaussians_detached,
        batch["target"]["extrinsics"],
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (h, w),
    )

    target_feat = output.feature[0, 0]
    return F.interpolate(
        target_feat.unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _flush_mask_batch(
    mask_decoder: SAMMaskDecoderWrapper,
    rendered_feat_64: torch.Tensor,
    batch_items: list[tuple[str, str, int, torch.Tensor]],
    batch_prompts: list[tuple[list[list[float]], list[int]]],
    pred_root: Path,
) -> int:
    if not batch_items:
        return 0

    device = rendered_feat_64.device
    num_items = len(batch_items)
    features = repeat(rendered_feat_64, "c h w -> n c h w", n=num_items)
    point_coords = torch.tensor(
        [coords for coords, _ in batch_prompts],
        dtype=torch.float32,
        device=device,
    )
    point_labels = torch.tensor(
        [labels for _, labels in batch_prompts],
        dtype=torch.int,
        device=device,
    )

    pred_masks = mask_decoder(
        features,
        point_coords=point_coords,
        point_labels=point_labels,
    )

    saved = 0
    for index, (scene_id, frame_id, class_id, gt_mask) in enumerate(batch_items):
        pred = pred_masks[index].detach().cpu()
        gt = gt_mask.detach().cpu().numpy()
        best = best_multimask_scores(pred, gt)
        mask = (torch.sigmoid(pred[best.best_index]) > 0.0).numpy()
        out_path = pred_root / scene_id / frame_id / f"{class_id}.png"
        _save_mask_png(mask, out_path)
        saved += 1
    return saved


@torch.no_grad()
def export_distill_masks(
    dataset_name: str,
    wrapper: DistillationModelWrapper,
    mask_decoder: SAMMaskDecoderWrapper,
    distill_dataset: DistillMaskExportDataset,
    frames: list,
    pred_root: Path,
    image_shape: tuple[int, int],
    *,
    prompt_strategy: str,
    min_object_pixels: int,
    mask_batch_size: int,
) -> dict[str, int]:
    """Export masks for every labeled frame and class in one dataset split."""
    data_shim = get_data_shim(wrapper.encoder)
    device = next(wrapper.parameters()).device

    saved_masks = 0
    skipped_frames = 0
    skipped_existing_frames = 0
    total = len(frames)

    for index, (scene_id, paths) in enumerate(frames):
        if index % 50 == 0:
            print(
                f"[{dataset_name}] Progress {index}/{total} — "
                f"{scene_id}/{paths.frame_id}",
                flush=True,
            )

        class_ids = expected_mask_class_ids(
            paths.label, min_object_pixels=min_object_pixels
        )
        if frame_mask_export_complete(
            pred_root, scene_id, paths.frame_id, class_ids
        ):
            skipped_existing_frames += 1
            continue

        try:
            class_prompts = class_prompts_from_label(
                paths.label,
                image_shape,
                prompt_strategy=prompt_strategy,
                min_object_pixels=min_object_pixels,
            )
        except ValueError as exc:
            print(f"[{dataset_name}] Skip {scene_id}/{paths.frame_id}: {exc}")
            skipped_frames += 1
            continue

        if not class_prompts:
            print(
                f"[{dataset_name}] Skip {scene_id}/{paths.frame_id}: "
                "no objects with enough pixels"
            )
            skipped_frames += 1
            continue

        example = distill_dataset._build_visualization_batch(scene_id, paths.frame_id)
        if example is None:
            print(
                f"[{dataset_name}] Skip {scene_id}/{paths.frame_id}: "
                "could not build multi-view example"
            )
            skipped_frames += 1
            continue

        batch = _batch_to_device(data_shim(_add_batch_dim(example)), device)

        rendered_feat_64 = render_target_features(wrapper, batch)

        batch_items: list[tuple[str, str, int, torch.Tensor]] = []
        batch_prompts: list[tuple[list[list[float]], list[int]]] = []

        for class_id, coords, labels, gt_mask in class_prompts:
            batch_items.append((scene_id, paths.frame_id, class_id, gt_mask))
            batch_prompts.append((coords, labels))
            if len(batch_items) >= mask_batch_size:
                saved_masks += _flush_mask_batch(
                    mask_decoder,
                    rendered_feat_64,
                    batch_items,
                    batch_prompts,
                    pred_root,
                )
                batch_items.clear()
                batch_prompts.clear()

        saved_masks += _flush_mask_batch(
            mask_decoder,
            rendered_feat_64,
            batch_items,
            batch_prompts,
            pred_root,
        )

    return {
        "saved_masks": saved_masks,
        "skipped_frames": skipped_frames,
        "skipped_existing_frames": skipped_existing_frames,
    }


def build_distillation_wrapper(cfg_dict: DictConfig) -> DistillationModelWrapper:
    cfg = load_typed_root_config(cfg_dict)
    cfg.model.encoder.feature_dim = cfg.model.encoder.gaussian_feature_dim

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)

    distill_train_cfg = DistillTrainCfg(
        feature_cosine_loss_weight=cfg_dict.train.get("feature_cosine_loss_weight", 1.0),
        feature_mag_loss_weight=cfg_dict.train.get("feature_mag_loss_weight", 0.1),
        depth_mode=cfg.train.depth_mode,
        context_view_loss=cfg.train.context_view_loss,
    )
    distill_optimizer_cfg = DistillOptimizerCfg(
        lr=cfg.optimizer.lr,
        warm_up_steps=cfg.optimizer.warm_up_steps,
        weight_decay=cfg_dict.optimizer.get("weight_decay", 0.05),
        feature_head_weight_decay=cfg_dict.optimizer.get(
            "feature_head_weight_decay", 0.01
        ),
    )

    debug_decoder_cfg = None
    if cfg_dict.get("debug_decoder", {}).get("enabled", False):
        debug_decoder_cfg = DebugDecoderCfg(
            enabled=True,
            sam_checkpoint=cfg_dict.debug_decoder.get("sam_checkpoint"),
            sam_model_variant=cfg_dict.debug_decoder.get(
                "sam_model_variant", "sam_vit_h"
            ),
        )

    return DistillationModelWrapper(
        distill_optimizer_cfg,
        distill_train_cfg,
        encoder,
        decoder,
        get_losses(cfg.loss),
        StepTracker(),
        debug_decoder_cfg=debug_decoder_cfg,
    )


def load_checkpoint_into_wrapper(
    wrapper: DistillationModelWrapper,
    checkpoint_path: Path,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = wrapper.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint {checkpoint_path} — "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig) -> None:
    set_cfg(cfg_dict)
    eval_cfg = cfg_dict.get("eval", {})

    checkpoint_path = cfg_dict.checkpointing.load
    if checkpoint_path is None:
        raise ValueError("checkpointing.load is required for mask export.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wrapper = build_distillation_wrapper(cfg_dict)
    load_checkpoint_into_wrapper(wrapper, checkpoint_path)
    wrapper.eval()
    wrapper.to(device)

    sam_ckpt = cfg_dict.debug_decoder.get(
        "sam_checkpoint", cfg_dict.train.get("sam_checkpoint", "/weights/sam_vit_h.pth")
    )
    sam_variant = cfg_dict.debug_decoder.get(
        "sam_model_variant", cfg_dict.train.get("sam_model_variant", "sam_vit_h")
    )
    mask_decoder = SAMMaskDecoderWrapper(
        sam_ckpt,
        model_variant=sam_variant,
    ).to(device)
    mask_decoder.eval()

    typed_cfg = load_typed_root_config(cfg_dict)
    datasets = get_dataset(typed_cfg.dataset, "test", StepTracker())
    datasets_by_name = {ds.cfg.name: ds for ds in datasets}

    from src.dataset.scannet_2dseg_splits import scenes_for_stage

    output_root = Path(eval_cfg.get("mask_output_dir", "outputs/c3g_sam_eval"))
    output_root.mkdir(parents=True, exist_ok=True)

    limit_frames = eval_cfg.get("limit_frames")
    prompt_strategy = eval_cfg.get("prompt_strategy", "centroid")
    min_object_pixels = int(eval_cfg.get("min_object_pixels", 16))
    mask_batch_size = int(eval_cfg.get("mask_batch_size", 32))

    dataset_stats: dict[str, dict] = {}
    total_saved = 0
    total_skipped = 0
    total_skipped_existing = 0

    for dataset_name, cfg_name, default_scenes in C3G_EVAL_DATASETS:
        if cfg_name not in cfg_dict.dataset:
            raise ValueError(
                f"Eval requires dataset.{cfg_name} in config; "
                f"add it to the evaluation Hydra defaults."
            )

        distill_cfg = cfg_dict.dataset[cfg_name]
        distill_dataset = datasets_by_name.get(cfg_name)
        if distill_dataset is None:
            raise ValueError(f"No test dataset instance for {cfg_name}")

        root = Path(distill_cfg.roots[0])
        if cfg_name == "scannet_distill":
            scene_ids = scenes_for_stage(
                "test",
                root=root,
                num_val=distill_cfg.val_scene_count,
                num_test=distill_cfg.test_scene_count,
            )
        else:
            scene_ids = list(distill_cfg.get("scenes", default_scenes))

        pred_root = output_root / dataset_name
        pred_root.mkdir(parents=True, exist_ok=True)
        scenes_to_run, skipped_scenes = filter_scenes_for_mask_export(
            root,
            scene_ids,
            pred_root,
            min_object_pixels=min_object_pixels,
        )
        if skipped_scenes:
            print(
                f"[{dataset_name}] Skipping {len(skipped_scenes)} scenes "
                f"already on volume under {pred_root}",
                flush=True,
            )
        if not scenes_to_run:
            print(
                f"[{dataset_name}] All scenes already exported — nothing to run.",
                flush=True,
            )
            dataset_stats[dataset_name] = {
                "dataset_root": str(root),
                "output_root": str(pred_root),
                "scenes": scene_ids,
                "scenes_to_run": [],
                "skipped_scenes": skipped_scenes,
                "saved_masks": 0,
                "skipped_frames": 0,
                "skipped_existing_frames": 0,
            }
            if cfg_name == "scannet_distill":
                dataset_stats[dataset_name]["split"] = "test"
            continue

        frames = iter_dataset_frames(root, scenes_to_run)
        if limit_frames is not None:
            frames = frames[: int(limit_frames)]

        image_shape = tuple(distill_cfg.input_image_shape)

        print(
            f"Evaluating {dataset_name}: {len(scenes_to_run)} scenes "
            f"({len(skipped_scenes)} already on volume), "
            f"{len(frames)} labeled frames under {root}",
            flush=True,
        )

        stats = export_distill_masks(
            dataset_name,
            wrapper,
            mask_decoder,
            distill_dataset,
            frames,
            pred_root,
            image_shape,
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
            mask_batch_size=mask_batch_size,
        )

        entry: dict = {
            "dataset_root": str(root),
            "output_root": str(pred_root),
            "scenes": scene_ids,
            "scenes_to_run": scenes_to_run,
            "skipped_scenes": skipped_scenes,
            **stats,
        }
        if cfg_name == "scannet_distill":
            entry["split"] = "test"

        dataset_stats[dataset_name] = entry
        total_saved += stats["saved_masks"]
        total_skipped += stats["skipped_frames"]
        total_skipped_existing += stats["skipped_existing_frames"]
        print(
            f"[{dataset_name}] Done — saved {stats['saved_masks']} masks "
            f"({stats['skipped_frames']} frames skipped, "
            f"{stats['skipped_existing_frames']} already on volume).",
            flush=True,
        )

    manifest = {
        "checkpoint": str(checkpoint_path),
        "datasets": dataset_stats,
        "prompt_strategy": prompt_strategy,
        "min_object_pixels": min_object_pixels,
        "mask_batch_size": mask_batch_size,
        "saved_masks": total_saved,
        "skipped_frames": total_skipped,
        "skipped_existing_frames": total_skipped_existing,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Mask export complete — saved {total_saved} masks "
        f"({total_skipped} frames skipped, "
        f"{total_skipped_existing} already on volume) under {output_root}"
    )


if __name__ == "__main__":
    main()
