#!/usr/bin/env python3
"""Modal runner for C3G SAM distillation eval on ScanNet test.

Loads ``distillation-base.ckpt`` from the ``c3g-weights`` volume, runs
``src.main`` in test mode on the ScanNet test split, and logs five training-style
debug visualizations (PCA / MSE / cosine / SAM decoder masks) for random frames
chosen with a fixed seed.

Prerequisites (once)::

    modal volume put c3g-weights /path/to/distillation-base.ckpt distillation-base.ckpt
    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
    # ScanNet frames on ``scannet`` volume.
    # Precomputed ``*_sam.pt`` on ``precompute_sam_features`` at ``scannet/``.

Full eval (ScanNet test split)::

    modal run src/modal_infra/modal_eval_c3gsam.py --wait
    modal run --detach src/modal_infra/modal_eval_c3gsam.py

Smoke (one test batch, wandb disabled)::

    modal run src/modal_infra/modal_eval_c3gsam.py::smoke --wait

Outputs: ``c3g-train-outputs`` volume under ``/outputs/eval/<wandb.name>/``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

from src.modal_infra.modal_common import (
    C3G_MODAL_WORKSPACE,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    PRECOMPUTE_SAM_FEATURES_MOUNT,
    PRECOMPUTE_SAM_FEATURES_VOLUME,
    SCANNET_2DSEG_TEST_SCENES,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    build_c3g_modal_image,
    resolve_detach,
    resolve_distillation_checkpoint,
    sample_eval_visualization_keys,
)

APP_NAME = "c3g-sam-distill-eval"
EVALUATION_CONFIG = "c3g_sam_scannet_distill"
WANDB_SECRET = modal.Secret.from_name("wandb")

DEFAULT_CHECKPOINT = "distillation-base.ckpt"
DEFAULT_VISUALIZATION_COUNT = 5
DEFAULT_VISUALIZATION_SEED = 42

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

VOLUMES = {
    str(WEIGHTS_MOUNT): weights_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
    str(OUTPUT_MOUNT): output_volume,
}


def _build_eval_command(
    *,
    evaluation_config: str,
    checkpoint_path: Path,
    visualization_keys: list[str],
    wandb_mode: str,
    limit_test_batches: int | None = None,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "src.main",
        f"+evaluation={evaluation_config}",
        f"checkpointing.load={checkpoint_path}",
        f"eval.visualization_keys={json.dumps(visualization_keys)}",
        f"wandb.mode={wandb_mode}",
    ]
    if limit_test_batches is not None:
        cmd.append(f"trainer.limit_test_batches={limit_test_batches}")
    return cmd


@app.function(
    image=build_c3g_modal_image(),
    gpu="H200",
    cpu=8,
    timeout=60 * 60 * 24,
    memory=131072,
    secrets=[WANDB_SECRET],
    volumes=VOLUMES,
)
def eval_c3gsam(
    evaluation_config: str = EVALUATION_CONFIG,
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    visualization_count: int = DEFAULT_VISUALIZATION_COUNT,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    wandb_mode: str = "online",
    limit_test_batches: int | None = None,
) -> str:
    """Run distillation eval on ScanNet test with seeded debug visualizations."""
    checkpoint_path = resolve_distillation_checkpoint(WEIGHTS_MOUNT / checkpoint_name)
    if not SCANNET_MOUNT.is_dir():
        raise FileNotFoundError(
            f"ScanNet volume not mounted at {SCANNET_MOUNT}. "
            f"Populate the `{SCANNET_VOLUME}` volume via the download script."
        )

    visualization_keys = sample_eval_visualization_keys(
        SCANNET_MOUNT,
        list(SCANNET_2DSEG_TEST_SCENES),
        count=visualization_count,
        seed=visualization_seed,
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test scenes: {len(SCANNET_2DSEG_TEST_SCENES)}")
    print(
        f"Debug visualizations ({len(visualization_keys)} frames, seed={visualization_seed}):"
    )
    for key in visualization_keys:
        print(f"  - {key}")

    if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb.mode=online requires WANDB_API_KEY in the Modal `wandb` secret. "
            "Use wandb_mode=disabled for smoke runs."
        )

    cmd = _build_eval_command(
        evaluation_config=evaluation_config,
        checkpoint_path=checkpoint_path,
        visualization_keys=visualization_keys,
        wandb_mode=wandb_mode,
        limit_test_batches=limit_test_batches,
    )
    print("Running:", " ".join(cmd))
    print(f"Outputs on volume `{OUTPUT_VOLUME}` under {OUTPUT_MOUNT / 'eval'}")
    subprocess.run(cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))
    output_volume.commit()
    return str(OUTPUT_MOUNT / "eval")


@app.local_entrypoint()
def main(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    visualization_count: int = DEFAULT_VISUALIZATION_COUNT,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    wandb_mode: str = "online",
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full C3G distillation eval on ScanNet test."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        eval_c3gsam,
        checkpoint_name=checkpoint_name,
        visualization_count=visualization_count,
        visualization_seed=visualization_seed,
        wandb_mode=wandb_mode,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G SAM ScanNet distill eval",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def smoke(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """One-batch eval smoke test (wandb disabled)."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        eval_c3gsam,
        checkpoint_name=checkpoint_name,
        visualization_count=1,
        visualization_seed=visualization_seed,
        wandb_mode="disabled",
        limit_test_batches=1,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G SAM ScanNet distill eval smoke",
        app_name=APP_NAME,
    )
