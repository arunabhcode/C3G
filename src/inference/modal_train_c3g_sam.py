#!/usr/bin/env python3
"""Modal runner for prompted SAM training on the Replica volume.

All training weights are read from the ``c3g-weights`` volume (mounted at
``/weights``): ``sam_vit_h.pth`` and ``gaussian_decoder.ckpt`` (encoder init).

Upload weights::

    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
    modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt

Smoke test::

    modal run src/inference/modal_train_c3g_sam.py::smoke

Full training::

    modal run src/inference/modal_train_c3g_sam.py --run-name sam_prompted_replica
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import modal

from src.inference.modal_sam_common import (
    C3G_MODAL_WORKSPACE,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    build_c3g_modal_image,
    find_smoke_scene,
    resolve_training_weights,
)

APP_NAME = "c3g-sam-prompted-simple"
WORKSPACE = C3G_MODAL_WORKSPACE


def _build_replica_prompted_overrides(
    *,
    run_name: str,
    max_steps: int,
    encoder_weights: Path,
    sam_checkpoint: Path,
    smoke_scene: str | None = None,
    smoke: bool = False,
) -> list[str]:
    run_dir = OUTPUT_MOUNT / "runs" / run_name
    overrides = [
        "+training=feature_head_sam_prompted",
        "dataset@dataset.replica_semseg=replica_2dseg",
        "wandb.mode=disabled",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={10_000 if smoke else 1000}",
        f"data_loader.train.batch_size={1 if smoke else 2}",
        f"model.encoder.pretrained_weights={encoder_weights}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"dataset.replica_semseg.roots=[{REPLICA_MOUNT}]",
        "train.prompt_strategy=centroid",
        "train.prompted_seg_loss_weight=1.0",
        "train.min_object_pixels=16",
    ]
    if smoke_scene is not None:
        overrides.append(f"dataset.replica_semseg.overfit_to_scene={smoke_scene}")
    if smoke:
        overrides.extend(
            [
                "data_loader.train.num_workers=0",
                "checkpointing.every_n_train_steps=1000000",
            ]
        )
    return overrides


app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


@app.function(
    image=build_c3g_modal_image(),
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def train_prompted_replica(
    run_name: str = "sam_prompted_replica",
    max_steps: int = 5001,
    smoke: bool = False,
) -> str:
    """Run prompted SAM training against the prepared Replica Modal volume."""
    encoder_weights, sam_checkpoint = resolve_training_weights()
    print(f"Encoder init: {encoder_weights}")
    print(f"SAM checkpoint: {sam_checkpoint}")
    if not REPLICA_MOUNT.is_dir():
        raise FileNotFoundError(f"Missing Replica volume mount: {REPLICA_MOUNT}")

    smoke_scene = find_smoke_scene(REPLICA_MOUNT) if smoke else None
    if smoke_scene is not None:
        print(f"Smoke scene: {smoke_scene}")

    overrides = _build_replica_prompted_overrides(
        run_name=run_name,
        max_steps=1 if smoke else max_steps,
        encoder_weights=encoder_weights,
        sam_checkpoint=sam_checkpoint,
        smoke_scene=smoke_scene,
        smoke=smoke,
    )

    cmd = ["uv", "run", "--no-sync", "python", "-m", "src.main", *overrides]
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE))

    run_dir = OUTPUT_MOUNT / "runs" / run_name
    output_volume.commit()
    return str(run_dir)


@app.local_entrypoint()
def main(run_name: str = "sam_prompted_replica", max_steps: int = 5001) -> None:
    print(train_prompted_replica.remote(run_name=run_name, max_steps=max_steps, smoke=False))


@app.local_entrypoint()
def smoke(run_name: str = "sam_smoke_replica") -> None:
    """One-example prompted training test on the Replica Modal volume."""
    print(train_prompted_replica.remote(run_name=run_name, max_steps=1, smoke=True))
