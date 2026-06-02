"""Shared Replica + ScanNet test mask export (vanilla SAM and C3G-SAM).

Output layout (both backends)::

    <output_root>/replica/<scene_id>/<frame_id>/<class_id>.png
    <output_root>/scannet/<scene_id>/<frame_id>/<class_id>.png
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

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
    VANILLA_EVAL_DATASETS,
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

CoordFrame = Literal["original", "sam"]
DEFAULT_PROMPT_STRATEGY = "centroid"
DEFAULT_MIN_OBJECT_PIXELS = 16


@dataclass
class DatasetExportStats:
    dataset_root: str
    output_root: str
    scenes: list[str]
    scenes_to_run: list[str]
    skipped_scenes: list[str]
    saved_masks: int
    skipped_frames: int
    skipped_existing_frames: int
    split: str | None = None


def save_mask_png(mask, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype("uint8") * 255)).save(path)


def class_prompts_from_label(
    label_path: Path,
    *,
    coord_frame: CoordFrame = "original",
    image_shape: tuple[int, int] | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    include_gt_masks: bool = False,
) -> list[tuple[int, list[list[float]], list[int]]] | list[
    tuple[int, list[list[float]], list[int], torch.Tensor]
]:
    """Return per-class point prompts (and optionally full-res GT masks)."""
    import numpy as np

    label_np = np.array(Image.open(label_path))
    binary_masks = decompose_label_map(label_np)
    if binary_masks.shape[0] == 0:
        raise ValueError(f"No foreground objects in label map: {label_path}")

    h_label, w_label = label_np.shape[:2]
    sampler = PromptSampler(
        strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
        image_size=max(image_shape) if image_shape else max(h_label, w_label),
    )

    unique_ids = np.unique(label_np)
    class_ids = [int(obj_id) for obj_id in unique_ids if obj_id != 0]

    prompts: list = []
    for class_id, mask_idx in zip(class_ids, range(binary_masks.shape[0])):
        mask = binary_masks[mask_idx]
        if mask.sum().item() < min_object_pixels:
            continue

        if prompt_strategy == "centroid":
            row, col = sampler.compute_centroid(mask)
        else:
            row, col = sampler.sample_random_point(mask)

        if coord_frame == "original":
            coords = [[float(col), float(row)]]
        else:
            if image_shape is None:
                raise ValueError("image_shape is required when coord_frame='sam'")
            h, w = image_shape
            x = col * (sampler.image_size / w)
            y = row * (sampler.image_size / h)
            coords = [[float(x), float(y)]]

        entry: tuple = (class_id, coords, [1])
        if include_gt_masks:
            entry = (*entry, mask.float())
        prompts.append(entry)
    return prompts


def write_export_manifest(output_root: Path, manifest: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Vanilla SAM (end-to-end src.model.sam.forward)
# ---------------------------------------------------------------------------


@dataclass
class VanillaPredictPayload:
    images_b64: list[str]
    point_coords: list[list[list[float]]] | None = None
    point_labels: list[list[int]] | None = None
    boxes: list[list[float]] | None = None
    multimask_output: bool = True
    return_logits: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VanillaPredictPayload:
        return cls(
            images_b64=list(payload["images_b64"]),
            point_coords=payload.get("point_coords"),
            point_labels=payload.get("point_labels"),
            boxes=payload.get("boxes"),
            multimask_output=bool(payload.get("multimask_output", True)),
            return_logits=bool(payload.get("return_logits", False)),
        )


def decode_image_bytes(data: bytes) -> torch.Tensor:
    import numpy as np

    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.array(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def encode_mask_array(masks) -> dict[str, Any]:
    import numpy as np

    packed = masks.astype(np.uint8, copy=False)
    return {
        "masks_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "shape": list(packed.shape),
        "dtype": "uint8",
    }


def run_vanilla_sam_predict(
    sam,
    device: torch.device,
    payload: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    from src.model.sam import forward as sam_forward

    request = VanillaPredictPayload.from_dict(payload)
    if not request.images_b64:
        raise ValueError("images_b64 must contain at least one image.")

    images = [decode_image_bytes(base64.b64decode(item)) for item in request.images_b64]
    batch = torch.cat(images, dim=0).to(device)

    point_coords = point_labels = boxes = None
    if request.point_coords is not None:
        point_coords = torch.tensor(
            request.point_coords, dtype=torch.float32, device=device
        )
    if request.point_labels is not None:
        point_labels = torch.tensor(request.point_labels, dtype=torch.int, device=device)
    if request.boxes is not None:
        boxes = torch.tensor(request.boxes, dtype=torch.float32, device=device)

    result = sam_forward(
        sam,
        batch,
        point_coords=point_coords,
        point_labels=point_labels,
        boxes=boxes,
        multimask_output=request.multimask_output,
        return_logits=request.return_logits,
    )

    masks = result["masks"].detach().cpu().numpy()
    if request.return_logits:
        masks = (masks > 0.0).astype(np.uint8)
    else:
        masks = masks.astype(np.uint8)

    response: dict[str, Any] = {
        "masks": encode_mask_array(masks),
        "iou_predictions": result["iou_predictions"].detach().cpu().tolist(),
    }
    if "low_res_logits" in result:
        low_res = result["low_res_logits"].detach().cpu().numpy()
        response["low_res_logits"] = encode_mask_array(low_res.astype(np.float32))
    return response


def masks_from_predict_response(masks_payload: dict[str, Any]):
    import numpy as np

    shape = tuple(masks_payload["shape"])
    raw = base64.b64decode(masks_payload["masks_b64"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def best_masks_from_predict_response(result: dict[str, Any]) -> list:
    import numpy as np

    masks = masks_from_predict_response(result["masks"])
    ious = np.asarray(result["iou_predictions"])
    batch_size = masks.shape[0]
    return [masks[b, int(np.argmax(ious[b]))] for b in range(batch_size)]


def sample_vanilla_prompt_fields(
    label_path: Path,
    *,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> tuple[list[list[float]], list[int]]:
    """One random valid-object prompt in original image pixels (smoke / API)."""
    import numpy as np

    label_np = np.array(Image.open(label_path))
    binary_masks = decompose_label_map(label_np)
    if binary_masks.shape[0] == 0:
        raise ValueError(f"No foreground objects in label map: {label_path}")

    sampler = PromptSampler(
        strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
    )
    point_coords, point_labels, _ = sampler.sample(
        binary_masks, coord_frame="original"
    )
    return point_coords.tolist(), point_labels.tolist()


def build_vanilla_batch_predict_payload(
    items: list[tuple[bytes, Path]] | list[tuple[bytes, list[list[float]], list[int]]],
    *,
    from_paths: bool = True,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> dict[str, Any]:
    if not items:
        raise ValueError("items must not be empty.")

    images_b64: list[str] = []
    point_coords: list[list[list[float]]] = []
    point_labels: list[list[int]] = []

    if from_paths:
        for image_bytes, label_path in items:
            if not label_path.is_file():
                raise FileNotFoundError(f"Label map not found: {label_path}")
            coords, labels = sample_vanilla_prompt_fields(
                label_path,
                prompt_strategy=prompt_strategy,
                min_object_pixels=min_object_pixels,
            )
            images_b64.append(base64.b64encode(image_bytes).decode("ascii"))
            point_coords.append(coords[0])
            point_labels.append(labels[0])
    else:
        for image_bytes, coords, labels in items:
            images_b64.append(base64.b64encode(image_bytes).decode("ascii"))
            point_coords.append(coords)
            point_labels.append(labels)

    return {
        "images_b64": images_b64,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "multimask_output": True,
    }


def export_vanilla_sam_masks(
    sam,
    device: torch.device,
    *,
    output_root: Path,
    dataset_roots: dict[str, str | Path],
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Export vanilla SAM masks for Replica + ScanNet test."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    dataset_stats: dict[str, dict[str, Any]] = {}
    total_saved = 0
    total_skipped_frames = 0
    total_skipped_existing = 0

    for dataset_name, scenes in VANILLA_EVAL_DATASETS:
        root = Path(dataset_roots[dataset_name])
        pred_root = output_root / dataset_name
        scenes_to_run, skipped_scenes = filter_scenes_for_mask_export(
            root,
            list(scenes),
            pred_root,
            min_object_pixels=min_object_pixels,
        )
        if skipped_scenes:
            print(
                f"[vanilla/{dataset_name}] Skipping {len(skipped_scenes)} scenes "
                f"already on volume under {pred_root}"
            )
        if not scenes_to_run:
            print(f"[vanilla/{dataset_name}] All scenes already exported.")
            dataset_stats[dataset_name] = DatasetExportStats(
                str(root),
                str(pred_root),
                list(scenes),
                [],
                skipped_scenes,
                0,
                0,
                0,
                "test" if dataset_name == "scannet" else None,
            ).__dict__
            continue

        print(
            f"[vanilla/{dataset_name}] {len(scenes_to_run)} scenes "
            f"({len(skipped_scenes)} skipped) under {root}"
        )
        frames = iter_dataset_frames(root, scenes_to_run)
        saved, skipped_frames, skipped_existing = _export_vanilla_sam_frames(
            sam,
            device,
            dataset_name=dataset_name,
            frames=frames,
            pred_root=pred_root,
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
            batch_size=batch_size,
        )
        stats = DatasetExportStats(
            str(root),
            str(pred_root),
            list(scenes),
            scenes_to_run,
            skipped_scenes,
            saved,
            skipped_frames,
            skipped_existing,
            "test" if dataset_name == "scannet" else None,
        )
        dataset_stats[dataset_name] = stats.__dict__
        total_saved += saved
        total_skipped_frames += skipped_frames
        total_skipped_existing += skipped_existing
        print(
            f"[vanilla/{dataset_name}] Done — saved {saved} masks "
            f"({skipped_frames} skipped, {skipped_existing} existing)"
        )

    manifest = {
        "backend": "vanilla_sam",
        "datasets": dataset_stats,
        "prompt_strategy": prompt_strategy,
        "min_object_pixels": min_object_pixels,
        "batch_size": batch_size,
        "saved_masks": total_saved,
        "skipped_frames": total_skipped_frames,
        "skipped_existing_frames": total_skipped_existing,
    }
    write_export_manifest(output_root, manifest)
    print(
        f"Vanilla mask export complete — {total_saved} masks "
        f"under {output_root}"
    )
    return manifest


def _export_vanilla_sam_frames(
    sam,
    device: torch.device,
    *,
    dataset_name: str,
    frames: list,
    pred_root: Path,
    prompt_strategy: str,
    min_object_pixels: int,
    batch_size: int,
) -> tuple[int, int, int]:
    saved_masks = 0
    skipped_frames = 0
    skipped_existing = 0
    batch_items: list[tuple[str, str, int, bytes]] = []
    batch_prompts: list[tuple[list[list[float]], list[int]]] = []

    def flush() -> None:
        nonlocal saved_masks
        if not batch_items:
            return
        payload = build_vanilla_batch_predict_payload(
            [
                (image_bytes, coords, labels)
                for (_, _, _, image_bytes), (coords, labels) in zip(
                    batch_items, batch_prompts
                )
            ],
            from_paths=False,
        )
        result = run_vanilla_sam_predict(sam, device, payload)
        masks = best_masks_from_predict_response(result)
        for (scene_id, frame_id, class_id, _), mask in zip(batch_items, masks):
            save_mask_png(mask, pred_root / scene_id / frame_id / f"{class_id}.png")
            saved_masks += 1
        batch_items.clear()
        batch_prompts.clear()

    total = len(frames)
    for index, (scene_id, paths) in enumerate(frames):
        if index % 50 == 0:
            print(
                f"[vanilla/{dataset_name}] {index}/{total} — "
                f"{scene_id}/{paths.frame_id}"
            )

        class_ids = expected_mask_class_ids(
            paths.label, min_object_pixels=min_object_pixels
        )
        if frame_mask_export_complete(pred_root, scene_id, paths.frame_id, class_ids):
            skipped_existing += 1
            continue

        try:
            class_prompts = class_prompts_from_label(
                paths.label,
                coord_frame="original",
                prompt_strategy=prompt_strategy,
                min_object_pixels=min_object_pixels,
            )
        except ValueError as exc:
            print(f"[vanilla/{dataset_name}] Skip {scene_id}/{paths.frame_id}: {exc}")
            skipped_frames += 1
            continue

        if not class_prompts:
            skipped_frames += 1
            continue

        image_bytes = paths.image.read_bytes()
        for class_id, coords, labels in class_prompts:
            batch_items.append((scene_id, paths.frame_id, class_id, image_bytes))
            batch_prompts.append((coords, labels))
            if len(batch_items) >= batch_size:
                flush()
    flush()
    return saved_masks, skipped_frames, skipped_existing


# ---------------------------------------------------------------------------
# C3G-SAM (rendered features + SAM mask decoder)
# ---------------------------------------------------------------------------


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


@torch.no_grad()
def render_target_features(
    wrapper: DistillationModelWrapper,
    batch: BatchedExample,
) -> torch.Tensor:
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


def _flush_c3g_mask_batch(
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
        gt = gt_mask.detach().cpu()
        if gt.dim() == 3:
            gt = gt.squeeze(0)
        gt_size = tuple(gt.shape[-2:])
        pred_full = F.interpolate(
            pred.unsqueeze(0).float(),
            size=gt_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        best = best_multimask_scores(pred_full, gt.numpy())
        mask = (torch.sigmoid(pred_full[best.best_index]) > 0.0).numpy()
        save_mask_png(mask, pred_root / scene_id / frame_id / f"{class_id}.png")
        saved += 1
    return saved


@torch.no_grad()
def export_c3g_sam_masks(
    wrapper: DistillationModelWrapper,
    mask_decoder: SAMMaskDecoderWrapper,
    distill_datasets: dict[str, DistillMaskExportDataset],
    *,
    output_root: Path,
    cfg_dict: DictConfig,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    mask_batch_size: int = 32,
    limit_frames: int | None = None,
) -> dict[str, Any]:
    """Export C3G-rendered SAM masks for Replica + ScanNet test."""
    from src.dataset.scannet_2dseg_splits import scenes_for_stage

    data_shim = get_data_shim(wrapper.encoder)
    device = next(wrapper.parameters()).device

    dataset_stats: dict[str, dict[str, Any]] = {}
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
        distill_dataset = distill_datasets.get(cfg_name)
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
                f"[c3g/{dataset_name}] Skipping {len(skipped_scenes)} scenes "
                f"already on volume under {pred_root}",
                flush=True,
            )
        if not scenes_to_run:
            print(f"[c3g/{dataset_name}] All scenes already exported.", flush=True)
            dataset_stats[dataset_name] = DatasetExportStats(
                str(root),
                str(pred_root),
                scene_ids,
                [],
                skipped_scenes,
                0,
                0,
                0,
                "test" if cfg_name == "scannet_distill" else None,
            ).__dict__
            continue

        frames = iter_dataset_frames(root, scenes_to_run)
        if limit_frames is not None:
            frames = frames[: int(limit_frames)]

        image_shape = tuple(distill_cfg.input_image_shape)
        print(
            f"[c3g/{dataset_name}] {len(scenes_to_run)} scenes, {len(frames)} frames",
            flush=True,
        )

        saved_masks = 0
        skipped_frames = 0
        skipped_existing_frames = 0
        total = len(frames)

        for index, (scene_id, paths) in enumerate(frames):
            if index % 50 == 0:
                print(
                    f"[c3g/{dataset_name}] {index}/{total} — "
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
                    coord_frame="sam",
                    image_shape=image_shape,
                    prompt_strategy=prompt_strategy,
                    min_object_pixels=min_object_pixels,
                    include_gt_masks=True,
                )
            except ValueError as exc:
                print(
                    f"[c3g/{dataset_name}] Skip {scene_id}/{paths.frame_id}: {exc}",
                    flush=True,
                )
                skipped_frames += 1
                continue

            if not class_prompts:
                skipped_frames += 1
                continue

            example = distill_dataset._build_visualization_batch(
                scene_id, paths.frame_id
            )
            if example is None:
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
                    saved_masks += _flush_c3g_mask_batch(
                        mask_decoder,
                        rendered_feat_64,
                        batch_items,
                        batch_prompts,
                        pred_root,
                    )
                    batch_items.clear()
                    batch_prompts.clear()

            saved_masks += _flush_c3g_mask_batch(
                mask_decoder,
                rendered_feat_64,
                batch_items,
                batch_prompts,
                pred_root,
            )

        stats = DatasetExportStats(
            str(root),
            str(pred_root),
            scene_ids,
            scenes_to_run,
            skipped_scenes,
            saved_masks,
            skipped_frames,
            skipped_existing_frames,
            "test" if cfg_name == "scannet_distill" else None,
        )
        dataset_stats[dataset_name] = stats.__dict__
        total_saved += saved_masks
        total_skipped += skipped_frames
        total_skipped_existing += skipped_existing_frames
        print(
            f"[c3g/{dataset_name}] Done — saved {saved_masks} masks",
            flush=True,
        )

    manifest = {
        "backend": "c3g_sam",
        "checkpoint": str(cfg_dict.checkpointing.load),
        "datasets": dataset_stats,
        "prompt_strategy": prompt_strategy,
        "min_object_pixels": min_object_pixels,
        "mask_batch_size": mask_batch_size,
        "saved_masks": total_saved,
        "skipped_frames": total_skipped,
        "skipped_existing_frames": total_skipped_existing,
    }
    write_export_manifest(output_root, manifest)
    print(f"C3G mask export complete — {total_saved} masks under {output_root}")
    return manifest


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


def run_c3g_mask_export_from_hydra(cfg_dict: DictConfig) -> dict[str, Any]:
    """Hydra entry: load distillation checkpoint and export masks."""
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
    mask_decoder = SAMMaskDecoderWrapper(sam_ckpt, model_variant=sam_variant).to(device)
    mask_decoder.eval()

    typed_cfg = load_typed_root_config(cfg_dict)
    datasets = get_dataset(typed_cfg.dataset, "test", StepTracker())
    datasets_by_name = {ds.cfg.name: ds for ds in datasets}

    output_root = Path(eval_cfg.get("mask_output_dir", "outputs/c3g_sam_eval"))
    return export_c3g_sam_masks(
        wrapper,
        mask_decoder,
        datasets_by_name,
        output_root=output_root,
        cfg_dict=cfg_dict,
        prompt_strategy=eval_cfg.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY),
        min_object_pixels=int(
            eval_cfg.get("min_object_pixels", DEFAULT_MIN_OBJECT_PIXELS)
        ),
        mask_batch_size=int(eval_cfg.get("mask_batch_size", 32)),
        limit_frames=eval_cfg.get("limit_frames"),
    )


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    """Local C3G mask export (Hydra)."""
    run_c3g_mask_export_from_hydra(cfg_dict)


if __name__ == "__main__":
    main()
