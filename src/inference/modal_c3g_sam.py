#!/usr/bin/env python3
"""Modal runner for C3G-F + SAM evaluation (Hydra ``mode=test``).

Runs one test batch on ``replica_2dseg`` or ``scannet_2dseg`` data from Modal
volumes. Requires a trained checkpoint on the ``c3g-train-outputs`` volume.

Prerequisites::

    modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
    modal run src/dataset/download_replica.py

Eval smoke (one batch, first scene; checkpoint on output volume)::

    modal run src/inference/modal_c3g_sam.py::smoke --dataset replica \\
        --resume /outputs/runs/sam_prompted_replica/checkpoints/last.ckpt

Full eval (same Hydra overrides, not limited to smoke scene)::

    modal run src/inference/modal_c3g_sam.py \\
        --run-name sam_prompted_replica_eval --dataset replica \\
        --resume /outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
"""

from __future__ import annotations

import shlex
import subprocess
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
    build_prompted_test_overrides,
    find_smoke_scene,
    resolve_dataset_root,
    resolve_detach,
    resolve_sam_checkpoint,
)

APP_NAME = "c3g-sam-inference"


def _validate_eval_paths(
    *,
    dataset: DatasetName,
    dataset_root: str,
    resume: str,
) -> None:
    spec = DATASET_SPECS[dataset]
    checkpoint = Path(resume)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint}. "
            f"Train first or upload to the `{OUTPUT_VOLUME}` volume."
        )
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(
            f"{spec['label']} dataset not found at {dataset_root}. "
            f"Populate the `{spec['volume']}` volume via the download script."
        )


app = modal.App(APP_NAME)
inference_image = build_c3g_modal_image()
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


@app.function(
    image=inference_image,
    gpu="A100-80GB",
    timeout=60 * 60 * 2,
    volumes={
        str(WEIGHTS_MOUNT): weights_volume,
        str(REPLICA_MOUNT): replica_volume,
        str(SCANNET_MOUNT): scannet_volume,
        str(OUTPUT_MOUNT): output_volume,
    },
)
def eval_c3g_sam(
    run_name: str,
    dataset: DatasetName = "replica",
    resume: str = "",
    dataset_root: str | None = None,
    sam_checkpoint: str | None = None,
    smoke: bool = False,
) -> str:
    """Run C3G-F + SAM test/eval via Hydra on a Modal dataset volume."""
    if not resume:
        raise ValueError("resume= is required (checkpoint path on the output volume).")

    sam_path = resolve_sam_checkpoint(sam_checkpoint)
    print(f"SAM checkpoint: {sam_path}")

    data_root = resolve_dataset_root(dataset, dataset_root)
    _validate_eval_paths(dataset=dataset, dataset_root=data_root, resume=resume)

    smoke_scene = find_smoke_scene(data_root) if smoke else None
    if smoke_scene is not None:
        print(f"Smoke scene: {smoke_scene}")

    overrides = build_prompted_test_overrides(
        run_name=run_name,
        dataset=dataset,
        dataset_root=data_root,
        checkpoint_path=resume,
        sam_checkpoint=sam_path,
        smoke_scene=smoke_scene,
    )

    cmd = ["uv", "run", "--no-sync", "python", "-m", "src.main", *overrides]
    print("Running:", " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(C3G_MODAL_WORKSPACE))

    run_dir = OUTPUT_MOUNT / "runs" / run_name
    output_volume.commit()
    label = "Smoke eval" if smoke else "Eval"
    print(f"{label} complete. Artifacts under {run_dir} (volume `{OUTPUT_VOLUME}`).")
    return str(run_dir)


def _dispatch_eval(
    *,
    run_name: str,
    dataset: DatasetName,
    resume: str,
    dataset_root: str | None,
    sam_checkpoint: str | None,
    smoke: bool,
    detach: bool,
) -> None:
    from src.misc.modal_run import dispatch_remote

    mode_label = "smoke eval" if smoke else "eval"
    result = dispatch_remote(
        eval_c3g_sam,
        run_name=run_name,
        dataset=dataset,
        resume=resume,
        dataset_root=dataset_root,
        sam_checkpoint=sam_checkpoint,
        smoke=smoke,
        detach=detach,
        job_name=f"C3G SAM {mode_label} ({run_name}, {dataset})",
        app_name=APP_NAME,
    )
    if detach:
        return
    print(f"Remote run finished: {result}")


@app.local_entrypoint()
def smoke(
    run_name: str = "sam_prompted_eval_smoke",
    dataset: DatasetName = "replica",
    resume: str = "",
    dataset_root: str | None = None,
    sam_checkpoint: str | None = None,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """One-batch eval on the first scene (requires ``--resume``)."""
    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not resume:
        print("--resume is required (checkpoint on the c3g-train-outputs volume).", file=sys.stderr)
        raise SystemExit(2)
    _dispatch_eval(
        run_name=run_name,
        dataset=dataset,
        resume=resume,
        dataset_root=dataset_root,
        sam_checkpoint=sam_checkpoint,
        smoke=True,
        detach=resolve_detach(detach=detach, remote_job=not wait),
    )


@app.local_entrypoint()
def main(
    run_name: str = "sam_prompted_eval",
    dataset: DatasetName = "replica",
    resume: str = "",
    dataset_root: str | None = None,
    sam_checkpoint: str | None = None,
    detach: bool | None = None,
    wait: bool = True,
) -> None:
    """Full test pass (``trainer.limit_test_batches=1`` in overrides)."""
    if dataset not in DATASET_SPECS:
        print(
            f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if not resume:
        print("--resume is required (checkpoint on the c3g-train-outputs volume).", file=sys.stderr)
        raise SystemExit(2)
    _dispatch_eval(
        run_name=run_name,
        dataset=dataset,
        resume=resume,
        dataset_root=dataset_root,
        sam_checkpoint=sam_checkpoint,
        smoke=False,
        detach=resolve_detach(detach=detach, remote_job=not wait),
    )
