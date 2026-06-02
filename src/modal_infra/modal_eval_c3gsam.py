#!/usr/bin/env python3
"""Modal runner for C3G SAM distillation eval on ScanNet test.

Exports per-class masks (same layout as ``modal_eval_sam``) using a distillation
checkpoint on ``c3g-weights``: one C3G forward per frame, batched SAM mask-decoder
calls per class.

Prerequisites (once)::

    modal volume put c3g-weights /path/to/distillation-base.ckpt distillation-base.ckpt
    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
    # ScanNet frames on ``scannet`` volume.
    # Precomputed ``*_sam.pt`` on ``precompute_sam_features`` at ``scannet/``.

Full eval (ScanNet test split, mask export)::

    modal run src/modal_infra/modal_eval_c3gsam.py --wait
    modal run --detach src/modal_infra/modal_eval_c3gsam.py

Optional Lightning test metrics + W&B debug tables::

    modal run src/modal_infra/modal_eval_c3gsam.py --with-lightning-test --wait

Smoke (one frame, masks only)::

    modal run src/modal_infra/modal_eval_c3gsam.py::smoke --wait

Masks on ``c3g-sam-eval-outputs``::

    c3g-sam-eval-outputs/scannet/<scene>/<frame_id>/<class_id>.png
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal

from src.modal_infra.modal_common import (
    C3G_MODAL_WORKSPACE,
    C3G_SAM_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EVAL_OUTPUT_VOLUME,
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
DEFAULT_MASK_BATCH_SIZE = 32
DEFAULT_VISUALIZATION_COUNT = 5
DEFAULT_VISUALIZATION_SEED = 42

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)
c3g_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EVAL_OUTPUT_VOLUME, create_if_missing=True
)

VOLUMES = {
    str(WEIGHTS_MOUNT): weights_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
}


def _build_mask_export_command(
    *,
    evaluation_config: str,
    checkpoint_path: Path,
    mask_batch_size: int,
    limit_frames: int | None = None,
) -> list[str]:
    cmd = [
        "python",
        "-m",
        "src.eval_c3g_sam_scannet_masks",
        f"+evaluation={evaluation_config}",
        f"checkpointing.load={checkpoint_path}",
        f"eval.mask_batch_size={mask_batch_size}",
        f"eval.mask_output_dir={C3G_SAM_EVAL_OUTPUT_MOUNT}",
    ]
    if limit_frames is not None:
        cmd.append(f"eval.limit_frames={limit_frames}")
    return cmd


def _build_lightning_test_command(
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
        f"hydra.run.dir={C3G_SAM_EVAL_OUTPUT_MOUNT}/lightning",
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
    mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    visualization_count: int = DEFAULT_VISUALIZATION_COUNT,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    wandb_mode: str = "online",
    with_lightning_test: bool = False,
    limit_frames: int | None = None,
    limit_test_batches: int | None = None,
) -> str:
    """Export C3G masks on ScanNet test; optionally run Lightning test metrics."""
    checkpoint_path = resolve_distillation_checkpoint(WEIGHTS_MOUNT / checkpoint_name)
    if not SCANNET_MOUNT.is_dir():
        raise FileNotFoundError(
            f"ScanNet volume not mounted at {SCANNET_MOUNT}. "
            f"Populate the `{SCANNET_VOLUME}` volume via the download script."
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test scenes: {len(SCANNET_2DSEG_TEST_SCENES)}")
    print(f"Mask batch size (classes per SAM forward): {mask_batch_size}")

    mask_cmd = _build_mask_export_command(
        evaluation_config=evaluation_config,
        checkpoint_path=checkpoint_path,
        mask_batch_size=mask_batch_size,
        limit_frames=limit_frames,
    )
    print("Running mask export:", " ".join(mask_cmd))
    subprocess.run(mask_cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))

    if with_lightning_test:
        if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "wandb.mode=online requires WANDB_API_KEY in the Modal `wandb` secret."
            )
        visualization_keys = sample_eval_visualization_keys(
            SCANNET_MOUNT,
            list(SCANNET_2DSEG_TEST_SCENES),
            count=visualization_count,
            seed=visualization_seed,
        )
        print("Debug visualization frames:")
        for key in visualization_keys:
            print(f"  - {key}")
        test_cmd = _build_lightning_test_command(
            evaluation_config=evaluation_config,
            checkpoint_path=checkpoint_path,
            visualization_keys=visualization_keys,
            wandb_mode=wandb_mode,
            limit_test_batches=limit_test_batches,
        )
        print("Running Lightning test:", " ".join(test_cmd))
        subprocess.run(test_cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))

    c3g_eval_output_volume.commit()
    print(f"Outputs on volume `{C3G_SAM_EVAL_OUTPUT_VOLUME}` at {C3G_SAM_EVAL_OUTPUT_MOUNT}")
    return str(C3G_SAM_EVAL_OUTPUT_MOUNT)


@app.local_entrypoint()
def main(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    mask_batch_size: int = DEFAULT_MASK_BATCH_SIZE,
    visualization_count: int = DEFAULT_VISUALIZATION_COUNT,
    visualization_seed: int = DEFAULT_VISUALIZATION_SEED,
    wandb_mode: str = "online",
    with_lightning_test: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full C3G distillation mask export on ScanNet test."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        eval_c3gsam,
        checkpoint_name=checkpoint_name,
        mask_batch_size=mask_batch_size,
        visualization_count=visualization_count,
        visualization_seed=visualization_seed,
        wandb_mode=wandb_mode,
        with_lightning_test=with_lightning_test,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G SAM ScanNet distill eval",
        app_name=APP_NAME,
    )


@app.local_entrypoint()
def smoke(
    checkpoint_name: str = DEFAULT_CHECKPOINT,
    mask_batch_size: int = 8,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Export masks for a single test frame (fast smoke)."""
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        eval_c3gsam,
        checkpoint_name=checkpoint_name,
        mask_batch_size=mask_batch_size,
        limit_frames=1,
        detach=resolve_detach(detach=detach, remote_job=not wait),
        job_name="C3G SAM ScanNet distill eval smoke",
        app_name=APP_NAME,
    )
