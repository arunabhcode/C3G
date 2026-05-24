#!/usr/bin/env python3
"""Modal runner for prompted SAM training on Replica or ScanNet volumes.

All training weights are read from the ``c3g-weights`` volume (mounted at
``/weights``): ``sam_vit_h.pth`` and ``gaussian_decoder.ckpt`` (encoder init).

Upload weights::

    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
    modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt

Smoke test (one step, first scene; detached by default)::

    modal run src/inference/modal_train_c3g_sam.py::smoke --dataset replica
    modal run src/inference/modal_train_c3g_sam.py::smoke --dataset scannet --wait

Full training (spawn on Modal by default; use ``--wait`` to block locally)::

    modal run src/inference/modal_train_c3g_sam.py --run-name sam_prompted_replica --dataset replica
    modal run --detach src/inference/modal_train_c3g_sam.py --run-name sam_prompted_scannet --dataset scannet
    modal run src/inference/modal_train_c3g_sam.py --run-name my_run --dataset scannet --batch-size 4

Checkpoints are written to the ``c3g-train-outputs`` volume at
``runs/{run_name}/checkpoints/`` and committed after each new ``.ckpt`` file.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import modal

from src.inference.modal_sam_common import (
    C3G_MODAL_WORKSPACE,
    DATASET_SPECS,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    DatasetName,
    build_c3g_modal_image,
    build_prompted_train_overrides,
    find_smoke_scene,
    resolve_dataset_root,
    resolve_detach,
    resolve_training_weights,
    run_subprocess_with_output_commit,
)

APP_NAME = "c3g-sam-prompted-simple"
WORKSPACE = C3G_MODAL_WORKSPACE
# Create before online runs: modal secret create wandb WANDB_API_KEY=<your-key>
WANDB_SECRET = modal.Secret.from_name("wandb")

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


def _validate_dataset_root(dataset: DatasetName, dataset_root: str) -> None:
    spec = DATASET_SPECS[dataset]
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(
            f"{spec['label']} dataset not found at {dataset_root}. "
            f"Populate the `{spec['volume']}` volume via the download script."
        )


@app.function(
    image=build_c3g_modal_image(),
    gpu="A100-80GB",
    timeout=60 * 60 * 4,
    secrets=[WANDB_SECRET],
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(SCANNET_MOUNT): scannet_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def train_prompted_sam(
    run_name: str,
    dataset: DatasetName = "replica",
    max_steps: int = 5001,
    smoke: bool = False,
    dataset_root: str | None = None,
    wandb_mode: str = "disabled",
    gaussian_weights: str | None = None,
    sam_checkpoint: str | None = None,
    batch_size: int | None = None,
    val_check_interval: int | None = None,
) -> str:
    """Run prompted SAM training on a prepared Replica or ScanNet Modal volume."""
    if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb.mode=online requires a Modal secret named 'wandb' with WANDB_API_KEY. "
            "Create it with: modal secret create wandb WANDB_API_KEY=<your-key> "
            "(get a key at https://wandb.ai/authorize). "
            "Or rerun with --wandb-mode disabled."
        )

    encoder_weights, sam_path = resolve_training_weights(
        gaussian_weights=gaussian_weights,
        sam_checkpoint=sam_checkpoint,
    )
    print(f"Encoder init: {encoder_weights}")
    print(f"SAM checkpoint: {sam_path}")

    data_root = resolve_dataset_root(dataset, dataset_root)
    _validate_dataset_root(dataset, data_root)

    smoke_scene = find_smoke_scene(data_root) if smoke else None
    if smoke_scene is not None:
        print(f"Smoke scene: {smoke_scene}")

    overrides = build_prompted_train_overrides(
        run_name=run_name,
        dataset=dataset,
        dataset_root=data_root,
        max_steps=1 if smoke else max_steps,
        wandb_mode=wandb_mode,
        gaussian_weights=encoder_weights,
        sam_checkpoint=sam_path,
        val_interval=10_000 if smoke else val_check_interval,
        batch_size=1 if smoke else batch_size,
        resume=None,
        smoke_scene=smoke_scene,
    )
    if smoke:
        overrides.extend(
            [
                "data_loader.train.num_workers=0",
                "checkpointing.every_n_train_steps=1000000",
            ]
        )

    run_dir = OUTPUT_MOUNT / "runs" / run_name
    cmd = ["uv", "run", "--no-sync", "python", "-m", "src.main", *overrides]
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    print(f"Checkpoints on volume `{OUTPUT_VOLUME}`: {run_dir / 'checkpoints'}")
    run_subprocess_with_output_commit(
        cmd=cmd,
        cwd=WORKSPACE,
        run_dir=run_dir,
        commit=output_volume.commit,
    )
    return str(run_dir)


def _dispatch_train(
    *,
    run_name: str,
    dataset: DatasetName,
    max_steps: int,
    smoke: bool,
    dataset_root: str | None,
    wandb_mode: str,
    gaussian_weights: str | None,
    sam_checkpoint: str | None,
    batch_size: int | None,
    val_check_interval: int | None,
    detach: bool,
) -> None:
    from src.misc.modal_run import dispatch_remote

    mode_label = "smoke train" if smoke else "train"
    result = dispatch_remote(
        train_prompted_sam,
        run_name=run_name,
        dataset=dataset,
        max_steps=max_steps,
        smoke=smoke,
        dataset_root=dataset_root,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        sam_checkpoint=sam_checkpoint,
        batch_size=batch_size,
        val_check_interval=val_check_interval,
        detach=detach,
        job_name=f"C3G SAM {mode_label} ({run_name}, {dataset})",
        app_name=APP_NAME,
    )
    if detach:
        return
    print(f"Remote run finished: {result}")


def _default_run_name(dataset: DatasetName, *, smoke: bool) -> str:
    if smoke:
        return f"sam_smoke_{dataset}"
    return f"sam_prompted_{dataset}"


@app.local_entrypoint()
def main(
    run_name: str | None = None,
    dataset: DatasetName = "replica",
    max_steps: int = 5001,
    batch_size: int | None = None,
    val_check_interval: int | None = None,
    dataset_root: str | None = None,
    wandb_mode: str = "disabled",
    gaussian_weights: str | None = None,
    sam_checkpoint: str | None = None,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full prompted SAM training on Replica or ScanNet."""
    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _dispatch_train(
        run_name=run_name or _default_run_name(dataset, smoke=False),
        dataset=dataset,
        max_steps=max_steps,
        smoke=False,
        dataset_root=dataset_root,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        sam_checkpoint=sam_checkpoint,
        batch_size=batch_size,
        val_check_interval=val_check_interval,
        detach=resolve_detach(detach=detach, remote_job=not wait),
    )


@app.local_entrypoint()
def smoke(
    run_name: str | None = None,
    dataset: DatasetName = "replica",
    batch_size: int | None = None,
    dataset_root: str | None = None,
    wandb_mode: str = "disabled",
    gaussian_weights: str | None = None,
    sam_checkpoint: str | None = None,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """One-step prompted training on the first scene (Replica or ScanNet)."""
    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _dispatch_train(
        run_name=run_name or _default_run_name(dataset, smoke=True),
        dataset=dataset,
        max_steps=1,
        smoke=True,
        dataset_root=dataset_root,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        sam_checkpoint=sam_checkpoint,
        batch_size=batch_size,
        val_check_interval=None,
        detach=resolve_detach(detach=detach, remote_job=not wait),
    )
