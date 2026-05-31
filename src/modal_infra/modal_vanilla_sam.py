#!/usr/bin/env python3
"""Modal inference interface for vanilla SAM (:mod:`src.model.sam`).

Uses GT point prompts from label maps (same ``PromptSampler`` / centroid strategy as
:mod:`src.modal_infra.modal_train_c3g_sam` prompted training). SAM weights are read
from the ``c3g-weights`` volume (``sam_vit_h.pth`` at ``/weights``).

Deploy the HTTP endpoint::

    modal deploy src/modal_infra/modal_vanilla_sam.py

Smoke test on one dataset frame (Modal GPU)::

    modal run src/modal_infra/modal_vanilla_sam.py::smoke --dataset replica --wait

Full eval on all Replica + ScanNet test split (spawns on Modal by default; ``--wait`` to block)::

    modal run --detach src/modal_infra/modal_vanilla_sam.py::main
    modal run src/modal_infra/modal_vanilla_sam.py::main --batch-size 32 --wait

ScanNet test = last 24 scenes on the volume (``scene0783_00`` … ``scene0806_00``).
Per view, one binary mask per GT class id is written under::

    vanilla-sam-outputs/replica/<scene>/<frame_id>/<class_id>.png
    vanilla-sam-outputs/scannet/<scene>/<frame_id>/<class_id>.png

Local image + matching ``*_y.png`` label::

    modal run src/modal_infra/modal_vanilla_sam.py \\
        --image-path frame_x.jpg --label-path frame_y.png --wait

Upload SAM weights to the Modal volume (once)::

    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
"""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.modal_infra.modal_sam_common import DatasetName

APP_NAME = "c3g-vanilla-sam"
DEFAULT_VARIANT = "sam_vit_h"
DEFAULT_BATCH_SIZE = 32


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


def _best_mask_from_response(result: dict[str, Any]):
    return _best_masks_from_response(result)[0]


def _save_mask_png(mask, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype("uint8") * 255)).save(path)


def _load_smoke_frame(
    dataset: str,
    dataset_root: str | None,
) -> tuple[bytes, Path, str]:
    from src.modal_infra.modal_sam_common import (
        DATASET_SPECS,
        find_smoke_frame,
        resolve_dataset_root,
    )

    spec = DATASET_SPECS[dataset]  # type: ignore[index]
    root = resolve_dataset_root(dataset, dataset_root)
    scene_id, paths = find_smoke_frame(root, scenes=list(spec["scenes"]))  # type: ignore[arg-type]
    summary = f"{dataset} scene={scene_id} frame={paths.frame_id} ({paths.image.name})"
    return paths.image.read_bytes(), paths.label, summary


def _class_prompts_from_label(
    label_path: Path,
    *,
    prompt_strategy: str = "centroid",
    min_object_pixels: int = 16,
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
    prompt_strategy: str = "centroid",
    min_object_pixels: int = 16,
) -> tuple[list[list[list[float]]], list[list[int]], str]:
    """Build API point prompt fields from a GT ``*_y.png`` label map."""
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
    coords = point_coords.tolist()
    labels = point_labels.tolist()
    return [coords], [labels], f"prompt=({coords[0][0]:.1f},{coords[0][1]:.1f}) strategy={prompt_strategy}"


def _build_predict_payload(
    image_bytes: bytes,
    *,
    label_path: Path,
    prompt_strategy: str = "centroid",
    min_object_pixels: int = 16,
) -> dict[str, Any]:
    """Build a predict payload with one GT-sampled point prompt."""
    return _build_batch_predict_payload(
        [(image_bytes, label_path)],
        prompt_strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
    )


def _build_batch_predict_payload_from_prompts(
    items: list[tuple[bytes, list[list[float]], list[int]]],
) -> dict[str, Any]:
    """Build a predict payload from precomputed per-object point prompts."""
    if not items:
        raise ValueError("items must contain at least one (image_bytes, coords, labels) tuple.")

    images_b64: list[str] = []
    point_coords: list[list[list[float]]] = []
    point_labels: list[list[int]] = []

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


def _build_batch_predict_payload(
    items: list[tuple[bytes, Path]],
    *,
    prompt_strategy: str = "centroid",
    min_object_pixels: int = 16,
) -> dict[str, Any]:
    """Build a predict payload for one or more images with GT point prompts."""
    if not items:
        raise ValueError("items must contain at least one (image_bytes, label_path) pair.")

    images_b64: list[str] = []
    point_coords: list[list[list[float]]] = []
    point_labels: list[list[int]] = []

    for image_bytes, label_path in items:
        if not label_path.is_file():
            raise FileNotFoundError(f"Label map not found: {label_path}")

        coords, labels, _ = _sample_gt_prompt_payload_fields(
            label_path,
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
        )
        images_b64.append(base64.b64encode(image_bytes).decode("ascii"))
        point_coords.append(coords[0])
        point_labels.append(labels[0])

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
    frames: list[tuple[str, Any]],
    pred_root: Path,
    prompt_strategy: str,
    min_object_pixels: int,
    batch_size: int,
) -> tuple[int, int]:
    """Run vanilla SAM for every GT class in each view; write masks under ``pred_root``."""
    saved_masks = 0
    skipped_frames = 0
    batch_items: list[tuple[str, str, int, bytes]] = []
    batch_prompts: list[tuple[list[list[float]], list[int]]] = []

    def _flush_batch() -> None:
        nonlocal saved_masks
        if not batch_items:
            return

        payload = _build_batch_predict_payload_from_prompts(
            [
                (image_bytes, coords, labels)
                for (_, _, _, image_bytes), (coords, labels) in zip(
                    batch_items, batch_prompts
                )
            ]
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
    return saved_masks, skipped_frames


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    from src.modal_infra.modal_sam_common import (
        DATASET_SPECS,
        REPLICA_MOUNT,
        REPLICA_VOLUME,
        SCANNET_MOUNT,
        SCANNET_VOLUME,
        VANILLA_SAM_OUTPUT_MOUNT,
        VANILLA_SAM_OUTPUT_VOLUME,
        VANILLA_EVAL_DATASETS,
        WEIGHTS_MOUNT,
        WEIGHTS_VOLUME,
        DatasetName,
        build_vanilla_sam_modal_image,
        iter_dataset_frames,
        resolve_dataset_root,
        resolve_detach,
        resolve_sam_checkpoint,
    )

    def _validate_dataset_root(dataset: DatasetName, dataset_root: str) -> None:
        spec = DATASET_SPECS[dataset]
        if not Path(dataset_root).is_dir():
            raise FileNotFoundError(
                f"{spec['label']} dataset not found at {dataset_root}. "
                f"Populate the `{spec['volume']}` volume via the download script."
            )

    app = modal.App(APP_NAME)
    weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
    scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
    vanilla_output_volume = modal.Volume.from_name(
        VANILLA_SAM_OUTPUT_VOLUME, create_if_missing=True
    )
    inference_image = build_vanilla_sam_modal_image()

    @app.cls(
        image=inference_image,
        gpu="H200",
        volumes={
            str(WEIGHTS_MOUNT): weights_volume,
            str(REPLICA_MOUNT): replica_volume,
            str(SCANNET_MOUNT): scannet_volume,
            str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
        },
        timeout=60 * 60 * 2,
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
            prompt_strategy: str = "centroid",
            min_object_pixels: int = 16,
        ) -> dict[str, Any]:
            image_bytes, label_path, summary = _load_smoke_frame(dataset, dataset_root)
            print(f"Smoke test image: {summary}")
            payload = _build_predict_payload(
                image_bytes,
                label_path=label_path,
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
            prompt_strategy: str = "centroid",
            min_object_pixels: int = 16,
            batch_size: int = DEFAULT_BATCH_SIZE,
        ) -> str:
            """Run vanilla SAM on all Replica + ScanNet test; write to ``vanilla-sam-outputs``."""
            import json

            if batch_size < 1:
                raise ValueError(f"batch_size must be >= 1, got {batch_size}")

            print(f"batch_size: {batch_size}")

            dataset_roots = {
                "replica": resolve_dataset_root("replica", replica_root),
                "scannet": resolve_dataset_root("scannet", scannet_root),
            }
            for dataset_name, data_root in dataset_roots.items():
                _validate_dataset_root(dataset_name, data_root)  # type: ignore[arg-type]

            output_root = VANILLA_SAM_OUTPUT_MOUNT
            dataset_stats: dict[str, dict[str, Any]] = {}
            total_saved = 0
            total_skipped_frames = 0

            for dataset_name, scenes in VANILLA_EVAL_DATASETS:
                root = Path(dataset_roots[dataset_name])
                pred_root = output_root / dataset_name
                print(
                    f"Evaluating {dataset_name}: {len(scenes)} scenes under {root}"
                )
                frames = iter_dataset_frames(root, scenes)
                saved_masks, skipped_frames = _eval_labeled_frames(
                    self.sam,
                    self.device,
                    dataset_name=dataset_name,
                    frames=frames,
                    pred_root=pred_root,
                    prompt_strategy=prompt_strategy,
                    min_object_pixels=min_object_pixels,
                    batch_size=batch_size,
                )
                dataset_stats[dataset_name] = {
                    "dataset_root": str(root),
                    "output_root": str(pred_root),
                    "scenes": scenes,
                    "saved_masks": saved_masks,
                    "skipped_frames": skipped_frames,
                }
                if dataset_name == "scannet":
                    dataset_stats[dataset_name]["split"] = "test"
                total_saved += saved_masks
                total_skipped_frames += skipped_frames
                print(
                    f"[{dataset_name}] Done — saved {saved_masks} masks "
                    f"({skipped_frames} frames skipped)."
                )

            manifest = {
                "datasets": dataset_stats,
                "prompt_strategy": prompt_strategy,
                "min_object_pixels": min_object_pixels,
                "batch_size": batch_size,
                "saved_masks": total_saved,
                "skipped_frames": total_skipped_frames,
            }
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            vanilla_output_volume.commit()

            print(
                f"Eval complete — saved {total_saved} masks "
                f"({total_skipped_frames} frames skipped)."
            )
            print(f"  Output volume: `{VANILLA_SAM_OUTPUT_VOLUME}`")
            print(
                f"  Replica masks: {output_root}/replica/<scene>/<frame_id>/<class_id>.png"
            )
            print(
                f"  ScanNet masks: {output_root}/scannet/<scene>/<frame_id>/<class_id>.png"
            )
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

    def _dispatch_smoke_frame(
        *,
        dataset: DatasetName,
        dataset_root: str | None,
        prompt_strategy: str,
        min_object_pixels: int,
        detach: bool | None,
        wait: bool,
    ) -> None:
        from src.misc.modal_run import dispatch_remote

        use_detach = resolve_detach(detach=detach, remote_job=not wait)
        result = dispatch_remote(
            VanillaSAMInference().predict_smoke_frame,
            dataset=dataset,
            dataset_root=dataset_root,
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
            detach=use_detach,
            job_name=f"vanilla SAM smoke (prompted, {dataset})",
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

    def _dispatch_eval_all(
        *,
        replica_root: str | None,
        scannet_root: str | None,
        prompt_strategy: str,
        min_object_pixels: int,
        batch_size: int,
        detach: bool | None,
        wait: bool,
    ) -> None:
        from src.misc.modal_run import dispatch_remote

        use_detach = resolve_detach(detach=detach, remote_job=not wait)
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
    def main(
        image_path: str | None = None,
        label_path: str | None = None,
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        replica_root: str | None = None,
        scannet_root: str | None = None,
        prompt_strategy: str = "centroid",
        min_object_pixels: int = 16,
        batch_size: int = DEFAULT_BATCH_SIZE,
        smoke_test: bool = False,
        detach: bool = True,
        wait: bool = False,
    ) -> None:
        """Vanilla SAM eval on all Replica + ScanNet test; writes to ``vanilla-sam-outputs``."""
        from src.misc.modal_run import dispatch_remote

        use_detach = detach and not wait

        if smoke_test:
            if dataset not in DATASET_SPECS:
                print(
                    f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            _dispatch_smoke_frame(
                dataset=dataset,
                dataset_root=dataset_root,
                prompt_strategy=prompt_strategy,
                min_object_pixels=min_object_pixels,
                detach=use_detach,
                wait=wait,
            )
            return

        if not image_path:
            _dispatch_eval_all(
                replica_root=replica_root,
                scannet_root=scannet_root,
                prompt_strategy=prompt_strategy,
                min_object_pixels=min_object_pixels,
                batch_size=batch_size,
                detach=use_detach,
                wait=wait,
            )
            return

        image_file = Path(image_path)
        if not image_file.is_file():
            print(f"Image not found: {image_file}", file=sys.stderr)
            raise SystemExit(1)

        if not label_path:
            print("--label-path is required (GT label PNG for point prompts).", file=sys.stderr)
            raise SystemExit(2)
        label_file = Path(label_path)
        if not label_file.is_file():
            print(f"Label not found: {label_file}", file=sys.stderr)
            raise SystemExit(1)

        payload = _build_predict_payload(
            image_file.read_bytes(),
            label_path=label_file,
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

    @app.local_entrypoint()
    def smoke(
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        prompt_strategy: str = "centroid",
        min_object_pixels: int = 16,
        detach: bool = True,
        wait: bool = False,
    ) -> None:
        """One-image SAM smoke test with GT point prompts from the dataset volume."""
        if dataset not in DATASET_SPECS:
            print(
                f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        _dispatch_smoke_frame(
            dataset=dataset,
            dataset_root=dataset_root,
            prompt_strategy=prompt_strategy,
            min_object_pixels=min_object_pixels,
            detach=detach and not wait,
            wait=wait,
        )

except ImportError:
    app = None  # type: ignore[assignment]
    main = None  # type: ignore[assignment,misc]
    smoke = None  # type: ignore[assignment,misc]
