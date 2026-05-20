#!/usr/bin/env python3
"""Modal inference interface for vanilla SAM (:mod:`src.model.sam`).

Deploy the HTTP endpoint::

    modal deploy src/inference/modal_vanilla_sam.py

Detached smoke test on one dataset frame (Modal GPU; no local image/GPU)::

    modal run src/inference/modal_vanilla_sam.py::smoke --dataset replica
    modal run src/inference/modal_vanilla_sam.py::smoke --dataset scannet

Optional local image (reads file on your machine before upload)::

    modal run src/inference/modal_vanilla_sam.py --image-path path/to.jpg

Upload SAM weights to the Modal volume (once)::

    modal volume put c3g-weights sam_vit_h.pth /path/to/sam_vit_h.pth
"""

from __future__ import annotations

import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

from src.inference.modal_sam_common import (
    DATASET_SPECS,
    DEFAULT_SAM_CHECKPOINT,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    DatasetName,
    resolve_dataset_root,
    resolve_detach,
)

APP_NAME = "c3g-vanilla-sam"
DEFAULT_VARIANT = "sam_vit_h"


def _build_image():
    import modal

    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "numpy==1.26.4",
            "pillow==11.0.0",
            "fastapi==0.118.0",
            "pydantic==2.11.4",
            "torch==2.5.1",
            "torchvision==0.20.1",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        )
        .add_local_dir(
            str(SRC_ROOT),
            remote_path="/root/src",
            ignore=["**/__pycache__/**", "**/.DS_Store"],
        )
        .env({"PYTHONPATH": "/root"})
    )


@dataclass
class PredictPayload:
    images_b64: list[str]
    segment_everything: bool = False
    point_coords: list[list[list[float]]] | None = None
    point_labels: list[list[int]] | None = None
    boxes: list[list[float]] | None = None
    multimask_output: bool = True
    return_logits: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictPayload:
        return cls(
            images_b64=list(payload["images_b64"]),
            segment_everything=bool(payload.get("segment_everything", False)),
            point_coords=payload.get("point_coords"),
            point_labels=payload.get("point_labels"),
            boxes=payload.get("boxes"),
            multimask_output=bool(payload.get("multimask_output", True)),
            return_logits=bool(payload.get("return_logits", False)),
        )


def _decode_image_bytes(data: bytes):
    from PIL import Image

    image = Image.open(io.BytesIO(data)).convert("RGB")
    array = np.array(image, dtype=np.float32)
    import torch

    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _encode_mask_array(masks: np.ndarray) -> dict[str, Any]:
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
        segment_everything=request.segment_everything,
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


def _load_smoke_image_bytes(dataset: DatasetName, dataset_root: str | None) -> tuple[bytes, str]:
    spec = DATASET_SPECS[dataset]
    root = resolve_dataset_root(dataset, dataset_root)
    scene_id, paths = find_smoke_frame(root, scenes=list(spec["scenes"]))  # type: ignore[arg-type]
    summary = f"{dataset} scene={scene_id} frame={paths.frame_id} ({paths.image.name})"
    return paths.image.read_bytes(), summary


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App(APP_NAME)
    weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
    scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
    inference_image = _build_image()

    @app.cls(
        image=inference_image,
        gpu="A10G",
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

            checkpoint_path = str(DEFAULT_SAM_CHECKPOINT)
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(
                    f"SAM checkpoint not found at {checkpoint_path}. "
                    f"Upload with: modal volume put {WEIGHTS_VOLUME} sam_vit_h.pth <local.pth>"
                )

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
            segment_everything: bool = False,
        ) -> dict[str, Any]:
            image_bytes, summary = _load_smoke_image_bytes(dataset, dataset_root)
            print(f"Smoke test image: {summary}")
            payload = {
                "images_b64": [base64.b64encode(image_bytes).decode("ascii")],
                "segment_everything": segment_everything,
                "multimask_output": True,
            }
            result = _run_vanilla_sam_predict(self.sam, self.device, payload)
            result["smoke_frame"] = summary
            return result

        @modal.fastapi_endpoint(method="POST", docs=True)
        def web(self):
            from fastapi import FastAPI
            from pydantic import BaseModel, Field

            class PredictBody(BaseModel):
                images_b64: list[str] = Field(
                    ..., description="Batch of base64-encoded RGB images."
                )
                segment_everything: bool = False
                point_coords: list[list[list[float]]] | None = None
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
        segment_everything: bool,
        detach: bool | None,
        wait: bool,
    ) -> None:
        from src.misc.modal_run import dispatch_remote

        use_detach = resolve_detach(detach=detach, remote_job=not wait)
        result = dispatch_remote(
            VanillaSAMInference().predict_smoke_frame,
            dataset=dataset,
            dataset_root=dataset_root,
            segment_everything=segment_everything,
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

    @app.local_entrypoint()
    def smoke(
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        segment_everything: bool = False,
        detach: bool | None = None,
        wait: bool = False,
    ) -> None:
        """Detached one-image SAM smoke test using the dataset volume on Modal GPU."""
        if dataset not in DATASET_SPECS:
            print(
                f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
                file=sys.stderr,
            )
            raise SystemExit(2)
        _dispatch_smoke_frame(
            dataset=dataset,
            dataset_root=dataset_root,
            segment_everything=segment_everything,
            detach=detach,
            wait=wait,
        )

    @app.local_entrypoint()
    def modal_main(
        image_path: str | None = None,
        dataset: DatasetName = "replica",
        dataset_root: str | None = None,
        segment_everything: bool = False,
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
                segment_everything=segment_everything,
                detach=detach,
                wait=wait,
            )
            return

        if not image_path:
            print(
                "Use entrypoint ::smoke for a detached Modal GPU run on one dataset frame, "
                "or pass --image-path for a local file.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        image_file = Path(image_path)
        if not image_file.is_file():
            print(f"Image not found: {image_file}", file=sys.stderr)
            raise SystemExit(1)

        payload = {
            "images_b64": [base64.b64encode(image_file.read_bytes()).decode("ascii")],
            "segment_everything": segment_everything,
            "multimask_output": True,
        }
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
