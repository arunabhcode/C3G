#!/usr/bin/env python3
"""Modal inference interface for vanilla SAM (:mod:`src.model.sam`).

Uses GT point prompts from label maps (same ``PromptSampler`` as C3G prompted training).

Deploy the HTTP endpoint::

    modal deploy src/inference/modal_vanilla_sam.py

Smoke test on one dataset frame (Modal GPU)::

    modal run src/inference/modal_vanilla_sam.py::smoke --dataset replica --wait

Local image + matching ``*_y.png`` label::

    modal run src/inference/modal_vanilla_sam.py \\
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
    from src.inference.modal_sam_common import DatasetName

APP_NAME = "c3g-vanilla-sam"
DEFAULT_VARIANT = "sam_vit_h"


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


def _load_smoke_frame(
    dataset: str,
    dataset_root: str | None,
) -> tuple[bytes, Path, str]:
    from src.inference.modal_sam_common import (
        DATASET_SPECS,
        find_smoke_frame,
        resolve_dataset_root,
    )

    spec = DATASET_SPECS[dataset]  # type: ignore[index]
    root = resolve_dataset_root(dataset, dataset_root)
    scene_id, paths = find_smoke_frame(root, scenes=list(spec["scenes"]))  # type: ignore[arg-type]
    summary = f"{dataset} scene={scene_id} frame={paths.frame_id} ({paths.image.name})"
    return paths.image.read_bytes(), paths.label, summary


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
    if not label_path.is_file():
        raise FileNotFoundError(f"Label map not found: {label_path}")

    point_coords, point_labels, _ = _sample_gt_prompt_payload_fields(
        label_path,
        prompt_strategy=prompt_strategy,
        min_object_pixels=min_object_pixels,
    )
    return {
        "images_b64": [base64.b64encode(image_bytes).decode("ascii")],
        "point_coords": point_coords,
        "point_labels": point_labels,
        "multimask_output": True,
    }


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    from src.inference.modal_sam_common import (
        DATASET_SPECS,
        REPLICA_MOUNT,
        REPLICA_VOLUME,
        SCANNET_MOUNT,
        SCANNET_VOLUME,
        WEIGHTS_MOUNT,
        WEIGHTS_VOLUME,
        DatasetName,
        build_vanilla_sam_modal_image,
        resolve_detach,
        resolve_sam_checkpoint,
    )

    app = modal.App(APP_NAME)
    weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
    scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
    inference_image = build_vanilla_sam_modal_image()

    @app.cls(
        image=inference_image,
        gpu="A100-40GB",
        volumes={
            str(WEIGHTS_MOUNT): weights_volume,
            str(REPLICA_MOUNT): replica_volume,
            str(SCANNET_MOUNT): scannet_volume,
        },
        timeout=60 * 30,
        scaledown_window=300,
    )
    class VanillaSAMInference:
        """Stateful vanilla SAM worker (loads checkpoint once per container)."""

        @modal.enter()
        def load_model(self) -> None:
            import torch

            from src.model.sam import load_sam

            checkpoint_path = str(resolve_sam_checkpoint())

            self.device = torch.device("cuda")
            self.sam = load_sam(
                DEFAULT_VARIANT,
                checkpoint_path,
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

    @app.local_entrypoint()
    def smoke(
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        prompt_strategy: str = "centroid",
        min_object_pixels: int = 16,
        detach: bool | None = None,
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
            detach=detach,
            wait=wait,
        )

    @app.local_entrypoint()
    def modal_main(
        image_path: str | None = None,
        label_path: str | None = None,
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        prompt_strategy: str = "centroid",
        min_object_pixels: int = 16,
        smoke_test: bool = False,
        detach: bool | None = None,
        wait: bool = False,
    ) -> None:
        from src.misc.modal_run import dispatch_remote

        if dataset not in DATASET_SPECS:
            print(
                f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
                file=sys.stderr,
            )
            raise SystemExit(2)

        if smoke_test:
            _dispatch_smoke_frame(
                dataset=dataset,
                dataset_root=dataset_root,
                prompt_strategy=prompt_strategy,
                min_object_pixels=min_object_pixels,
                detach=detach,
                wait=wait,
            )
            return

        if not image_path:
            print(
                "Use entrypoint ::smoke for a detached Modal GPU run on one dataset frame, "
                "or pass --image-path and --label-path for a local pair.",
                file=sys.stderr,
            )
            raise SystemExit(2)

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
        use_detach = resolve_detach(detach=detach, remote_job=False)
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

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]
    smoke = None  # type: ignore[assignment,misc]
