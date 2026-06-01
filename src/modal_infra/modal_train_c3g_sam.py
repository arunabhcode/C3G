#!/usr/bin/env python3
"""Modal runner for C3G SAM distillation with precomputed features.

Prerequisites (once)::

    modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt
    modal run src/modal_infra/modal_precompute_sam_features.py::main --dataset scannet

Training uses Hydra config only (no CLI overrides)::

    modal run src/modal_infra/modal_train_c3g_sam.py
    modal run src/modal_infra/modal_train_c3g_sam.py --wait

Smoke (one optimizer step)::

    modal run src/modal_infra/modal_train_c3g_sam.py::smoke --wait

Checkpoints: ``c3g-train-outputs`` volume at ``/outputs/runs/<wandb.name>/``.
"""

from __future__ import annotations

import os
import subprocess

import modal

from src.modal_infra.modal_sam_common import (
    C3G_MODAL_WORKSPACE,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    PRECOMPUTE_SAM_FEATURES_MOUNT,
    PRECOMPUTE_SAM_FEATURES_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    build_c3g_modal_image,
    resolve_detach,
)

APP_NAME = "c3g-sam-precomputed-train"
TRAINING_CONFIG = "feature_head_sam_precomputed"
SMOKE_TRAINING_CONFIG = "feature_head_sam_precomputed_smoke"
WANDB_SECRET = modal.Secret.from_name("wandb")

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

VOLUMES = {
    str(WEIGHTS_MOUNT): weights_volume,
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
    str(OUTPUT_MOUNT): output_volume,
}


def _train_cmd(training_config: str) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "src.main",
        f"+training={training_config}",
    ]


@app.function(
    image=build_c3g_modal_image(),
    gpu="T4",
    timeout=60 * 60 * 24,
    secrets=[WANDB_SECRET],
    volumes=VOLUMES,
)
def train_sam_precomputed(training_config: str = TRAINING_CONFIG) -> str:
    """Run ``src.main`` with the given ``+training=`` config (YAML-only)."""
    if training_config == TRAINING_CONFIG and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "feature_head_sam_precomputed sets wandb.mode=online. "
            "Create a Modal secret: modal secret create wandb WANDB_API_KEY=<key> "
            "Or use the smoke entrypoint (wandb disabled)."
        )

    cmd = _train_cmd(training_config)
    print("Running:", " ".join(cmd))
    print(f"Outputs on volume `{OUTPUT_VOLUME}` under {OUTPUT_MOUNT / 'runs'}")
    subprocess.run(cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))
    output_volume.commit()
    return str(OUTPUT_MOUNT / "runs")


@app.local_entrypoint()
def main(detach: bool | None = None, wait: bool = False) -> None:
    """Full SAM distillation training (see ``config/training/feature_head_sam_precomputed.yaml``)."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        train_sam_precomputed,
        training_config=TRAINING_CONFIG,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G SAM distill train",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def smoke(detach: bool | None = None, wait: bool = False) -> None:
    """One optimizer-step smoke test (``feature_head_sam_precomputed_smoke.yaml``)."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        train_sam_precomputed,
        training_config=SMOKE_TRAINING_CONFIG,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G SAM distill smoke",
        app_name=APP_NAME,
    )
