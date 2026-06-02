#!/usr/bin/env python3
"""Modal runner for vanilla SAM eval (:mod:`src.model.sam`).

GT point prompts from label maps. SAM weights on the ``c3g-weights`` volume.

Deploy HTTP endpoint::

    modal deploy src/modal_infra/modal_eval_sam.py

Smoke (one frame from a dataset volume)::

    modal run src/modal_infra/modal_eval_sam.py::smoke --dataset replica --wait

Full eval (Replica + ScanNet test split)::

    modal run src/modal_infra/modal_eval_sam.py --wait
    modal run --detach src/modal_infra/modal_eval_sam.py

Scenes (and individual frames) that already have mask PNGs on the output volume
are skipped automatically so reruns resume where they left off.

Local image + label::

    modal run src/modal_infra/modal_eval_sam.py \\
        --image-path frame_x.jpg --label-path frame_y.png --wait

Upload SAM weights (once)::

    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth

Masks on ``vanilla-sam-outputs``::

    vanilla-sam-outputs/<dataset>/<scene>/<frame_id>/<class_id>.png
"""

from __future__ import annotations

import base64
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

from src.modal_infra.modal_common import (
    DATASET_SPECS,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    VANILLA_EVAL_DATASETS,
    VANILLA_SAM_OUTPUT_MOUNT,
    VANILLA_SAM_OUTPUT_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    DatasetName,
    build_eval_sam_modal_image,
    expected_mask_class_ids,
    filter_scenes_for_mask_export,
    find_smoke_frame,
    frame_mask_export_complete,
    iter_dataset_frames,
    resolve_dataset_root,
    resolve_detach,
    resolve_sam_checkpoint,
)

APP_NAME = "c3g-vanilla-sam"
DEFAULT_VARIANT = "sam_vit_h"
DEFAULT_BATCH_SIZE = 32
DEFAULT_PROMPT_STRATEGY = "centroid"
DEFAULT_MIN_OBJECT_PIXELS = 16

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
vanilla_output_volume = modal.Volume.from_name(
    VANILLA_SAM_OUTPUT_VOLUME, create_if_missing=True
)
eval_image = build_eval_sam_modal_image()


@dataclass
class PredictPayload:
    images_b64: list[str]
    point_coords: list[list[list[float]]] | None = None
    point_labels: list[list[int]] | None = None
    boxes: list[list[float]] | None = None
    multimask_output: bool = True
    return_logits: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictPayload:
        return cls(
            images_b64=list(payload["images_b64"]),
            point_coords=payload.get("point_coords"),
            point_labels=payload.get("point_labels"),
            boxes=payload.get("boxes"),
            multimask_output=bool(payload.get("multimask_output", True)),
            return_logits=bool(payload.get("return_logits", False)),
        )


def _decode_image_bytes(data: bytes):
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.array(image, dtype=np.float32)
    import torch

    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _encode_mask_array(masks) -> dict[str, Any]:
    import numpy as np

    packed = masks.astype(np.uint8, copy=False)
    return {
        "masks_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "shape": list(packed.shape),
        "dtype": "uint8",
    }


def _run_vanilla_sam_predict(
    sam,
    device,
    payload: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    import torch

    from src.model.sam import forward as sam_forward

    request = PredictPayload.from_dict(payload)
    if not request.images_b64:
        raise ValueError("images_b64 must contain at least one image.")

    images = [_decode_image_bytes(base64.b64decode(item)) for item in request.images_b64]
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
        "masks": _encode_mask_array(masks),
        "iou_predictions": result["iou_predictions"].detach().cpu().tolist(),
    }
    if "low_res_logits" in result:
        low_res = result["low_res_logits"].detach().cpu().numpy()
        response["low_res_logits"] = _encode_mask_array(low_res.astype(np.float32))
    return response


def _masks_from_response(masks_payload: dict[str, Any]):
    import numpy as np

    shape = tuple(masks_payload["shape"])
    raw = base64.b64decode(masks_payload["masks_b64"])
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def _best_masks_from_response(result: dict[str, Any]) -> list:
    import numpy as np

    masks = _masks_from_response(result["masks"])
    ious = np.asarray(result["iou_predictions"])
    batch_size = masks.shape[0]
    return [masks[b, int(np.argmax(ious[b]))] for b in range(batch_size)]


def _save_mask_png(mask, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype("uint8") * 255)).save(path)


def _class_prompts_from_label(
    label_path: Path,
    *,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> list[tuple[int, list[list[float]], list[int]]]:
    """Return ``(class_id, point_coords, point_labels)`` for every valid GT object."""
    import numpy as np
    from PIL import Image

    from src.model.prompt_sampler import PromptSampler, decompose_label_map

    label_np = np.array(Image.open(label_path))
    binary_masks = decompose_label_map(label_np)
    if binary_masks.shape[0] == 0:
        raise ValueError(f"No foreground objects in label map: {label_path}")

    unique_ids = np.unique(label_np)
    class_ids = [int(obj_id) for obj_id in unique_ids if obj_id != 0]

    sampler = PromptSampler(
        strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
    )
    prompts: list[tuple[int, list[list[float]], list[int]]] = []
    for class_id, mask_idx in zip(class_ids, range(binary_masks.shape[0])):
        mask = binary_masks[mask_idx]
        if mask.sum().item() < min_object_pixels:
            continue

        if prompt_strategy == "centroid":
            row, col = sampler.compute_centroid(mask)
        else:
            row, col = sampler.sample_random_point(mask)

        prompts.append((class_id, [[float(col), float(row)]], [1]))
    return prompts


def _sample_gt_prompt_payload_fields(
    label_path: Path,
    *,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> tuple[list[list[list[float]]], list[list[int]]]:
    import numpy as np
    from PIL import Image

    from src.model.prompt_sampler import PromptSampler, decompose_label_map

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


def _build_batch_predict_payload(
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
            coords, labels = _sample_gt_prompt_payload_fields(
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


def _eval_labeled_frames(
    sam,
    device,
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
    skipped_existing_frames = 0
    batch_items: list[tuple[str, str, int, bytes]] = []
    batch_prompts: list[tuple[list[list[float]], list[int]]] = []

    def _flush_batch() -> None:
        nonlocal saved_masks
        if not batch_items:
            return

        payload = _build_batch_predict_payload(
            [
                (image_bytes, coords, labels)
                for (_, _, _, image_bytes), (coords, labels) in zip(
                    batch_items, batch_prompts
                )
            ],
            from_paths=False,
        )
        result = _run_vanilla_sam_predict(sam, device, payload)
        masks = _best_masks_from_response(result)
        for (scene_id, frame_id, class_id, _), mask in zip(batch_items, masks):
            out_path = pred_root / scene_id / frame_id / f"{class_id}.png"
            _save_mask_png(mask, out_path)
            saved_masks += 1
        batch_items.clear()
        batch_prompts.clear()

    total = len(frames)
    for index, (scene_id, paths) in enumerate(frames):
        if index % 50 == 0:
            print(
                f"[{dataset_name}] Progress {index}/{total} — {scene_id}/{paths.frame_id}"
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
            class_prompts = _class_prompts_from_label(
                paths.label,
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

        image_bytes = paths.image.read_bytes()
        for class_id, coords, labels in class_prompts:
            batch_items.append((scene_id, paths.frame_id, class_id, image_bytes))
            batch_prompts.append((coords, labels))
            if len(batch_items) >= batch_size:
                _flush_batch()

    _flush_batch()
    return saved_masks, skipped_frames, skipped_existing_frames


@app.cls(
    image=eval_image,
    gpu="H200",
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(SCANNET_MOUNT): scannet_volume,
        str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
    },
    timeout=60 * 60 * 24,
    scaledown_window=300,
)
class VanillaSAMInference:
    """Stateful vanilla SAM worker (loads checkpoint once per container)."""

    @modal.enter()
    def load_model(self) -> None:
        import torch

        from src.model.sam import load_sam

        sam_path = resolve_sam_checkpoint()
        print(f"SAM checkpoint: {sam_path}")

        self.device = torch.device("cuda")
        self.sam = load_sam(
            DEFAULT_VARIANT,
            str(sam_path),
            freeze=True,
        ).to(self.device)
        self.sam.eval()

    @modal.method()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _run_vanilla_sam_predict(self.sam, self.device, payload)

    @modal.method()
    def predict_smoke_frame(
        self,
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
        min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    ) -> dict[str, Any]:
        spec = DATASET_SPECS[dataset]
        root = resolve_dataset_root(dataset, dataset_root)
        scene_id, paths = find_smoke_frame(root, scenes=list(spec["scenes"]))  # type: ignore[arg-type]
        summary = f"{dataset} scene={scene_id} frame={paths.frame_id} ({paths.image.name})"
        print(f"Smoke test image: {summary}")

        payload = _build_batch_predict_payload(
            [(paths.image.read_bytes(), paths.label)],
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
        )
        result = _run_vanilla_sam_predict(self.sam, self.device, payload)
        result["smoke_frame"] = summary
        result["prompt_mode"] = "prompted"
        result["prompt_strategy"] = prompt_strategy
        if payload.get("point_coords"):
            result["point_coords"] = payload["point_coords"]
        return result

    @modal.method()
    def eval_all(
        self,
        replica_root: str | None = None,
        scannet_root: str | None = None,
        prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
        min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> str:
        """Run vanilla SAM on Replica + ScanNet test; write to ``vanilla-sam-outputs``."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        print(f"batch_size: {batch_size}")

        dataset_roots = {
            "replica": resolve_dataset_root("replica", replica_root),
            "scannet": resolve_dataset_root("scannet", scannet_root),
        }
        for dataset_name, data_root in dataset_roots.items():
            spec = DATASET_SPECS[dataset_name]  # type: ignore[index]
            if not Path(data_root).is_dir():
                raise FileNotFoundError(
                    f"{spec['label']} dataset not found at {data_root}. "
                    f"Populate the `{spec['volume']}` volume via the download script."
                )

        output_root = VANILLA_SAM_OUTPUT_MOUNT
        dataset_stats: dict[str, dict[str, Any]] = {}
        total_saved = 0
        total_skipped_frames = 0
        total_skipped_existing_frames = 0

        for dataset_name, scenes in VANILLA_EVAL_DATASETS:
            root = Path(dataset_roots[dataset_name])
            pred_root = output_root / dataset_name
            scenes_to_run, skipped_scenes = filter_scenes_for_mask_export(
                root,
                scenes,
                pred_root,
                min_object_pixels=min_object_pixels,
            )
            if skipped_scenes:
                print(
                    f"[{dataset_name}] Skipping {len(skipped_scenes)} scenes "
                    f"already on volume under {pred_root}"
                )
            if not scenes_to_run:
                print(f"[{dataset_name}] All scenes already exported — nothing to run.")
                dataset_stats[dataset_name] = {
                    "dataset_root": str(root),
                    "output_root": str(pred_root),
                    "scenes": scenes,
                    "scenes_to_run": [],
                    "skipped_scenes": skipped_scenes,
                    "saved_masks": 0,
                    "skipped_frames": 0,
                    "skipped_existing_frames": 0,
                }
                if dataset_name == "scannet":
                    dataset_stats[dataset_name]["split"] = "test"
                continue

            print(
                f"Evaluating {dataset_name}: {len(scenes_to_run)} scenes "
                f"({len(skipped_scenes)} already on volume) under {root}"
            )
            frames = iter_dataset_frames(root, scenes_to_run)
            saved_masks, skipped_frames, skipped_existing_frames = (
                _eval_labeled_frames(
                    self.sam,
                    self.device,
                    dataset_name=dataset_name,
                    frames=frames,
                    pred_root=pred_root,
                    prompt_strategy=prompt_strategy,
                    min_object_pixels=min_object_pixels,
                    batch_size=batch_size,
                )
            )
            dataset_stats[dataset_name] = {
                "dataset_root": str(root),
                "output_root": str(pred_root),
                "scenes": scenes,
                "scenes_to_run": scenes_to_run,
                "skipped_scenes": skipped_scenes,
                "saved_masks": saved_masks,
                "skipped_frames": skipped_frames,
                "skipped_existing_frames": skipped_existing_frames,
            }
            if dataset_name == "scannet":
                dataset_stats[dataset_name]["split"] = "test"
            total_saved += saved_masks
            total_skipped_frames += skipped_frames
            total_skipped_existing_frames += skipped_existing_frames
            print(
                f"[{dataset_name}] Done — saved {saved_masks} masks "
                f"({skipped_frames} frames skipped, "
                f"{skipped_existing_frames} already on volume)."
            )

        manifest = {
            "datasets": dataset_stats,
            "prompt_strategy": prompt_strategy,
            "min_object_pixels": min_object_pixels,
            "batch_size": batch_size,
            "saved_masks": total_saved,
            "skipped_frames": total_skipped_frames,
            "skipped_existing_frames": total_skipped_existing_frames,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        vanilla_output_volume.commit()

        print(
            f"Eval complete — saved {total_saved} masks "
            f"({total_skipped_frames} frames skipped, "
            f"{total_skipped_existing_frames} already on volume)."
        )
        print(f"  Output volume: `{VANILLA_SAM_OUTPUT_VOLUME}`")
        return str(output_root)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(self):
        from fastapi import FastAPI
        from pydantic import BaseModel, Field

        class PredictBody(BaseModel):
            images_b64: list[str] = Field(
                ..., description="Batch of base64-encoded RGB images."
            )
            point_coords: list[list[list[float]]] | None = Field(
                None,
                description="Point prompts in original-image pixels (required unless boxes set).",
            )
            point_labels: list[list[int]] | None = None
            boxes: list[list[float]] | None = None
            multimask_output: bool = True
            return_logits: bool = False

        service = self
        api = FastAPI(
            title="C3G Vanilla SAM",
            description="End-to-end SAM via src.model.sam.forward",
        )

        @api.post("/predict")
        def predict_endpoint(body: PredictBody) -> dict[str, Any]:
            return service.predict.local(body.model_dump())

        return api


@app.local_entrypoint()
def main(
    image_path: str | None = None,
    label_path: str | None = None,
    replica_root: str | None = None,
    scannet_root: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    detach: bool = True,
    wait: bool = False,
) -> None:
    """Full vanilla SAM eval, or a single local image + label predict."""
    from src.misc.modal_run import dispatch_remote

    use_detach = resolve_detach(detach=detach, remote_job=not wait)

    if image_path:
        image_file = Path(image_path)
        if not image_file.is_file():
            print(f"Image not found: {image_file}", file=sys.stderr)
            raise SystemExit(1)
        if not label_path:
            print("--label-path is required for local predict.", file=sys.stderr)
            raise SystemExit(2)
        label_file = Path(label_path)
        if not label_file.is_file():
            print(f"Label not found: {label_file}", file=sys.stderr)
            raise SystemExit(1)

        payload = _build_batch_predict_payload(
            [(image_file.read_bytes(), label_file)],
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
        )
        result = dispatch_remote(
            VanillaSAMInference().predict,
            payload,
            detach=use_detach,
            job_name="vanilla SAM predict",
            app_name=APP_NAME,
        )
        if use_detach:
            return
        print(f"OK — masks shape {result['masks']['shape']}")
        print(f"IoU heads: {result['iou_predictions']}")
        return

    result = dispatch_remote(
        VanillaSAMInference().eval_all,
        replica_root=replica_root,
        scannet_root=scannet_root,
        prompt_strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
        batch_size=batch_size,
        detach=use_detach,
        job_name="vanilla SAM eval",
        app_name=APP_NAME,
    )
    if use_detach:
        return
    print(f"Remote run finished: {result}")


@app.local_entrypoint()
def smoke(
    dataset: DatasetName = "replica",
    dataset_root: str | None = None,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    detach: bool = True,
    wait: bool = False,
) -> None:
    """One-frame SAM smoke test with GT point prompts from a dataset volume."""
    from src.misc.modal_run import dispatch_remote

    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    use_detach = resolve_detach(detach=detach, remote_job=not wait)
    result = dispatch_remote(
        VanillaSAMInference().predict_smoke_frame,
        dataset=dataset,
        dataset_root=dataset_root,
        prompt_strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
        detach=use_detach,
        job_name=f"vanilla SAM smoke ({dataset})",
        app_name=APP_NAME,
    )
    if use_detach:
        return
    print(f"OK — masks shape {result['masks']['shape']}")
    print(f"IoU heads: {result['iou_predictions']}")
    if "smoke_frame" in result:
        print(f"Frame: {result['smoke_frame']}")
    if result.get("point_coords"):
        print(f"GT prompt: {result['point_coords']}")
