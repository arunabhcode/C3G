#!/usr/bin/env python3
"""Modal runner for C3G SAM distillation training (doc Step 2 only).

Follows ``docs/distillation_training.md``: **Step 1** (SAM feature precompute) must
already be on the ``precompute_sam_features`` volume; this app only runs **Step 2**
(``uv run python -m src.main +training=feature_head_sam_precomputed ...``).

Step 1 — precompute (once per dataset)::

    modal run src/modal_infra/modal_precompute_sam_features.py::main --dataset scannet
    modal run src/modal_infra/modal_precompute_sam_features.py::main --dataset replica

Upload encoder init weights (once)::

    modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt

Step 2 — smoke (one optimizer step, first scene; detached by default)::

    modal run src/modal_infra/modal_train_c3g_sam.py::smoke --dataset replica
    modal run src/modal_infra/modal_train_c3g_sam.py::smoke --dataset scannet --wait

Step 2 — full training (spawn on Modal by default; use ``--wait`` to block locally)::

    modal run src/modal_infra/modal_train_c3g_sam.py \\
        --run-name sam_distill_scannet --dataset scannet --wandb-mode online
    modal run --detach src/modal_infra/modal_train_c3g_sam.py \\
        --run-name sam_distill_replica --dataset replica

Checkpoints are written to ``c3g-train-outputs`` at
``runs/{run_name}/checkpoints/``.
"""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import modal

from src.modal_infra.modal_sam_common import (
    C3G_MODAL_WORKSPACE,
    DISTILL_DATASET_SPECS,
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
    DatasetName,
    build_c3g_modal_image,
    build_distill_train_cmd,
    build_precomputed_train_overrides,
    find_smoke_scene,
    resolve_dataset_root,
    resolve_detach,
    resolve_encoder_pretrained_weights,
    resolve_sam_features_root,
    run_subprocess_with_output_commit,
    validate_sam_features_root,
)

APP_NAME = "c3g-sam-precomputed-train"
# Create before online runs: modal secret create wandb WANDB_API_KEY=<your-key>
WANDB_SECRET = modal.Secret.from_name("wandb")


app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
precompute_volume = modal.Volume.from_name(
    PRECOMPUTE_SAM_FEATURES_VOLUME, create_if_missing=True
)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


def _validate_dataset_root(dataset: DatasetName, dataset_root: str) -> None:
    spec = DISTILL_DATASET_SPECS[dataset]
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(
            f"{spec['label']} dataset not found at {dataset_root}. "
            f"Populate the `{spec['volume']}` volume via the download script."
        )


@app.function(
    image=build_c3g_modal_image(),
    gpu="T4",
    timeout=60 * 60 * 24,
    secrets=[WANDB_SECRET],
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(SCANNET_MOUNT): scannet_volume,
        str(PRECOMPUTE_SAM_FEATURES_MOUNT): precompute_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def train_sam_precomputed(
    run_name: str,
    dataset: DatasetName = "replica",
    max_steps: int = 5001,
    smoke: bool = False,
    dataset_root: str | None = None,
    sam_features_root: str | None = None,
    wandb_mode: str = "disabled",
    gaussian_weights: str | None = None,
    batch_size: int | None = None,
    val_check_interval: int | None = None,
    debug_decoder: bool = False,
) -> str:
    """Run SAM feature distillation via ``src.main`` (``train.pipeline=distillation``)."""
    if wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "wandb.mode=online requires a Modal secret named 'wandb' with WANDB_API_KEY. "
            "Create it with: modal secret create wandb WANDB_API_KEY=<your-key> "
            "(get a key at https://wandb.ai/authorize). "
            "Or rerun with --wandb-mode disabled."
        )

    encoder_weights = resolve_encoder_pretrained_weights(gaussian_weights)
    print(f"Encoder init: {encoder_weights}")

    data_root = resolve_dataset_root(dataset, dataset_root)
    _validate_dataset_root(dataset, data_root)

    features_root = resolve_sam_features_root(dataset, sam_features_root)
    validate_sam_features_root(dataset, features_root)
    print(f"Dataset root: {data_root}")
    print(f"SAM features root (precomputed, doc Step 1): {features_root}")

    smoke_scene = find_smoke_scene(
        data_root, scenes=list(DISTILL_DATASET_SPECS[dataset]["scenes"])  # type: ignore[arg-type]
    ) if smoke else None
    if smoke_scene is not None:
        print(f"Smoke scene: {smoke_scene}")

    overrides = build_precomputed_train_overrides(
        run_name=run_name,
        dataset=dataset,
        dataset_root=data_root,
        sam_features_root=features_root,
        max_steps=1 if smoke else max_steps,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        val_interval=10_000 if smoke else val_check_interval,
        batch_size=batch_size,
        resume=None,
        debug_decoder=debug_decoder,
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
    cmd = build_distill_train_cmd(overrides)
    print("Running (distillation_training.md Step 2):", " ".join(shlex.quote(part) for part in cmd))
    print(f"Checkpoints on volume `{OUTPUT_VOLUME}`: {run_dir / 'checkpoints'}")
    run_subprocess_with_output_commit(
        cmd=cmd,
        cwd=C3G_MODAL_WORKSPACE,
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
    sam_features_root: str | None,
    wandb_mode: str,
    gaussian_weights: str | None,
    batch_size: int | None,
    val_check_interval: int | None,
    debug_decoder: bool,
    detach: bool,
) -> None:
    from src.misc.modal_run import dispatch_remote

    mode_label = "smoke distill" if smoke else "distill train"
    result = dispatch_remote(
        train_sam_precomputed,
        run_name=run_name,
        dataset=dataset,
        max_steps=max_steps,
        smoke=smoke,
        dataset_root=dataset_root,
        sam_features_root=sam_features_root,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        batch_size=batch_size,
        val_check_interval=val_check_interval,
        debug_decoder=debug_decoder,
        detach=detach,
        job_name=f"C3G SAM {mode_label} ({run_name}, {dataset})",
        app_name=APP_NAME,
    )
    if detach:
        return
    print(f"Remote run finished: {result}")


def _default_run_name(dataset: DatasetName, *, smoke: bool) -> str:
    if smoke:
        return f"sam_distill_smoke_{dataset}"
    return f"sam_distill_{dataset}"


@app.local_entrypoint()
def main(
    run_name: str | None = None,
    dataset: DatasetName = "replica",
    max_steps: int = 5001,
    batch_size: int | None = None,
    val_check_interval: int | None = None,
    dataset_root: str | None = None,
    sam_features_root: str | None = None,
    wandb_mode: str = "disabled",
    gaussian_weights: str | None = None,
    debug_decoder: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Full SAM distillation training with precomputed features on Replica or ScanNet."""
    if dataset not in DISTILL_DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: "
            f"{', '.join(DISTILL_DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _dispatch_train(
        run_name=run_name or _default_run_name(dataset, smoke=False),
        dataset=dataset,
        max_steps=max_steps,
        smoke=False,
        dataset_root=dataset_root,
        sam_features_root=sam_features_root,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        batch_size=batch_size,
        val_check_interval=val_check_interval,
        debug_decoder=debug_decoder,
        detach=resolve_detach(detach=detach, remote_job=not wait),
    )


@app.local_entrypoint()
def smoke(
    run_name: str | None = None,
    dataset: DatasetName = "replica",
    batch_size: int | None = None,
    dataset_root: str | None = None,
    sam_features_root: str | None = None,
    wandb_mode: str = "disabled",
    gaussian_weights: str | None = None,
    debug_decoder: bool = False,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """One optimizer-step distillation smoke test on the first scene."""
    if dataset not in DISTILL_DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: "
            f"{', '.join(DISTILL_DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _dispatch_train(
        run_name=run_name or _default_run_name(dataset, smoke=True),
        dataset=dataset,
        max_steps=1,
        smoke=True,
        dataset_root=dataset_root,
        sam_features_root=sam_features_root,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        batch_size=batch_size,
        val_check_interval=None,
        debug_decoder=debug_decoder,
        detach=resolve_detach(detach=detach, remote_job=not wait),
    )
