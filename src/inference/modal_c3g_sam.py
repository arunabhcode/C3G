#!/usr/bin/env python3
"""Modal inference interface for the C3G SAM mask-decoder pipeline.

Decodes masks from rendered Gaussian features (Bx256x64x64) using
:class:`src.model.sam_decoder.SAMMaskDecoderWrapper` — the SAM head used during
C3G training/eval, not the full vanilla SAM image encoder.

Deploy the HTTP endpoint::

    modal deploy src/inference/modal_c3g_sam.py

Run a detached smoke test on Modal GPU (random feature tensor; no local GPU)::

    modal run src/inference/modal_c3g_sam.py::smoke

    modal run src/inference/modal_c3g_sam.py --smoke-test

Upload SAM weights to the Modal volume (once)::

    modal volume put c3g-weights sam_vit_h.pth /path/to/sam_vit_h.pth
"""

from __future__ import annotations

import base64
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

APP_NAME = "c3g-sam-pipeline"
DEFAULT_VARIANT = "sam_vit_h"


def _build_image():
    import modal

    return (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "numpy==1.26.4",
            "einops==0.6.1",
            "fastapi==0.118.0",
            "pydantic==2.11.4",
            "torch==2.5.1",
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
class DecodePayload:
    features_b64: str
    shape: list[int]
    dtype: str = "float32"
    point_coords: list[list[list[float]]] | None = None
    point_labels: list[list[int]] | None = None
    boxes: list[list[float]] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DecodePayload:
        return cls(
            features_b64=str(payload["features_b64"]),
            shape=[int(x) for x in payload["shape"]],
            dtype=str(payload.get("dtype", "float32")),
            point_coords=payload.get("point_coords"),
            point_labels=payload.get("point_labels"),
            boxes=payload.get("boxes"),
        )


def _decode_features_b64(features_b64: str, shape: list[int], dtype: str):
    import numpy as np

    if len(shape) != 4:
        raise ValueError(f"shape must be [B, C, H, W], got {shape}")
    np_dtype = np.dtype(dtype)
    raw = base64.b64decode(features_b64)
    array = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
    if array.size == 0:
        raise ValueError("features payload is empty")
    return array


def _encode_float_array(array) -> dict[str, Any]:
    import numpy as np

    packed = np.ascontiguousarray(array)
    return {
        "data_b64": base64.b64encode(packed.tobytes()).decode("ascii"),
        "shape": list(packed.shape),
        "dtype": str(packed.dtype),
    }


def _run_c3g_sam_decode(mask_decoder, device, payload: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    request = DecodePayload.from_dict(payload)
    features_np = _decode_features_b64(
        request.features_b64,
        request.shape,
        request.dtype,
    )
    features = torch.from_numpy(features_np).to(device)

    point_coords = point_labels = boxes = None
    if request.point_coords is not None:
        point_coords = torch.tensor(
            request.point_coords, dtype=torch.float32, device=device
        )
    if request.point_labels is not None:
        point_labels = torch.tensor(request.point_labels, dtype=torch.int, device=device)
    if request.boxes is not None:
        boxes = torch.tensor(request.boxes, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = mask_decoder(
            features,
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
        )

    logits_np = logits.detach().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_np))
    return {
        "logits": _encode_float_array(logits_np),
        "probabilities": _encode_float_array(probs.astype(np.float32)),
    }


def _smoke_test_payload() -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(0)
    features = rng.standard_normal((1, 256, 64, 64), dtype=np.float32)
    return {
        "features_b64": base64.b64encode(features.tobytes()).decode("ascii"),
        "shape": [1, 256, 64, 64],
        "dtype": "float32",
    }


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    from src.inference.modal_sam_common import (
        DEFAULT_SAM_CHECKPOINT,
        WEIGHTS_MOUNT,
        WEIGHTS_VOLUME,
        resolve_detach,
    )

    app = modal.App(APP_NAME)
    weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    inference_image = _build_image()

    @app.cls(
        image=inference_image,
        gpu="A10G",
        volumes={str(WEIGHTS_MOUNT): weights_volume},
        timeout=60 * 30,
        scaledown_window=300,
    )
    class C3GSAMPipeline:
        """C3G SAM mask decoder on precomputed rendered feature maps."""

        @modal.enter()
        def load_model(self) -> None:
            import torch

            from src.model.sam_decoder import SAMMaskDecoderWrapper

            checkpoint_path = str(DEFAULT_SAM_CHECKPOINT)
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(
                    f"SAM checkpoint not found at {checkpoint_path}. "
                    f"Upload with: modal volume put {WEIGHTS_VOLUME} sam_vit_h.pth <local.pth>"
                )

            self.device = torch.device("cuda")
            self.mask_decoder = SAMMaskDecoderWrapper(
                sam_checkpoint=checkpoint_path,
                model_variant=DEFAULT_VARIANT,
                use_lora=False,
                lora_rank=4,
            ).to(self.device)
            self.mask_decoder.eval()

        @modal.method()
        def decode(self, payload: dict[str, Any]) -> dict[str, Any]:
            return _run_c3g_sam_decode(self.mask_decoder, self.device, payload)

        @modal.method()
        def smoke_decode(self) -> dict[str, Any]:
            """Decode one random feature map on Modal GPU (no local inputs)."""
            return _run_c3g_sam_decode(
                self.mask_decoder, self.device, _smoke_test_payload()
            )

        @modal.fastapi_endpoint(method="POST", docs=True)
        def web(self):
            from fastapi import FastAPI
            from pydantic import BaseModel, Field

            class DecodeBody(BaseModel):
                features_b64: str = Field(
                    ..., description="float32 tensor bytes (base64), shape [B,C,H,W]."
                )
                shape: list[int]
                dtype: str = "float32"
                point_coords: list[list[list[float]]] | None = None
                point_labels: list[list[int]] | None = None
                boxes: list[list[float]] | None = None

            service = self
            api = FastAPI(
                title="C3G SAM Pipeline",
                description="SAMMaskDecoderWrapper on rendered Gaussian features",
            )

            @api.post("/decode")
            def decode_endpoint(body: DecodeBody) -> dict[str, Any]:
                return service.decode.local(body.model_dump())

            return api

    def _dispatch_smoke(*, detach: bool | None, wait: bool) -> None:
        from src.misc.modal_run import dispatch_remote

        use_detach = resolve_detach(detach=detach, remote_job=not wait)
        result = dispatch_remote(
            C3GSAMPipeline().smoke_decode,
            detach=use_detach,
            job_name="C3G SAM decode smoke",
            app_name=APP_NAME,
        )
        if use_detach:
            return
        print("OK — logits shape", result["logits"]["shape"])

    @app.local_entrypoint()
    def modal_main(
        smoke_test: bool = False,
        detach: bool | None = None,
        wait: bool = False,
    ) -> None:
        if not smoke_test:
            print("Pass --smoke-test or use entrypoint ::smoke for a detached Modal GPU run.")
            return
        _dispatch_smoke(detach=detach, wait=wait)

    @app.local_entrypoint()
    def smoke(detach: bool | None = None, wait: bool = False) -> None:
        """Detached SAM decoder smoke test on Modal GPU."""
        _dispatch_smoke(detach=detach, wait=wait)

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]
    smoke = None  # type: ignore[assignment,misc]
