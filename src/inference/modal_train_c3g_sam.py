#!/usr/bin/env python3
"""Modal runner for prompted SAM training on the Replica volume.

All training weights are read from the ``c3g-weights`` volume (mounted at
``/weights``): ``sam_vit_h.pth``, ``gaussian_decoder.ckpt`` (preferred encoder
init), or ``model.pt`` (VGGT fallback).

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
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SAM_NUM_CHANNELS,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    find_smoke_scene,
    resolve_training_weights,
)

APP_NAME = "c3g-sam-prompted-simple"
WORKSPACE = Path("/workspace")
CONFIG_H = "submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate

    # Modal imports this file as /root/simp.py inside the container, while the
    # copied repository lives at /workspace.
    workspace = Path("/workspace")
    if (workspace / "pyproject.toml").is_file() and (workspace / "src").is_dir():
        return workspace

    raise RuntimeError(f"Could not find repo root from {here}")


def _image() -> modal.Image:
    return (
        modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
        .apt_install(
            "git",
            "curl",
            "ca-certificates",
            "build-essential",
            "clang",
            "libgl1",
            "libglib2.0-0",
        )
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "echo 'export PATH=\"/root/.local/bin:$PATH\"' >> /root/.bashrc",
        )
        .env(
            {
                "PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
                "FORCE_CUDA": "1",
            }
        )
        .add_local_dir(
            str(_repo_root()),
            remote_path=str(WORKSPACE),
            copy=True,
            ignore=[
                "**/.git/**",
                "**/__pycache__/**",
                "**/.venv/**",
                "**/datasets/**",
                "**/outputs/**",
                "**/.DS_Store",
                "src/dataset/replica_data/replica_semseg/**",
            ],
        )
        .workdir(str(WORKSPACE))
        .run_commands(
            f"sed -i 's/#define NUM_SEMANTIC_CHANNELS 512/#define NUM_SEMANTIC_CHANNELS {SAM_NUM_CHANNELS}/' {CONFIG_H}",
            "uv sync --frozen",
            "uv run --no-sync python -c \"from submodules.diff_gaussian_rasterization_w_feature_detach.setup import _C\"",
            "uv run --no-sync python -c \"from submodules.diff_gaussian_rasterization_w_pose.setup import _C\"",
        )
        .env({"PYTHONPATH": str(WORKSPACE)})
    )


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
    image=_image(),
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
