
#!/usr/bin/env python3
"""Small Modal runner for prompted SAM training on the Replica Modal volume.

This mirrors the prompted-training flow while keeping all dataset access on Modal:

    modal run src/inference/simp.py::smoke

For a real run:

    modal run src/inference/simp.py::main --run-name sam_prompted_replica --max-steps 5001
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import modal

from src.inference.modal_sam_common import (
    DEFAULT_GAUSSIAN_WEIGHTS,
    DEFAULT_SAM_CHECKPOINT,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SAM_NUM_CHANNELS,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    build_prompted_train_overrides,
    find_smoke_scene,
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


app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)


@app.function(
    image=_image(),
    gpu="A10G",
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
    if not DEFAULT_SAM_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing SAM checkpoint: {DEFAULT_SAM_CHECKPOINT}")
    if not DEFAULT_GAUSSIAN_WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing Gaussian weights: {DEFAULT_GAUSSIAN_WEIGHTS}")
    if not REPLICA_MOUNT.is_dir():
        raise FileNotFoundError(f"Missing Replica volume mount: {REPLICA_MOUNT}")

    smoke_scene = find_smoke_scene(REPLICA_MOUNT) if smoke else None
    if smoke_scene is not None:
        print(f"Smoke scene: {smoke_scene}")

    overrides = build_prompted_train_overrides(
        run_name=run_name,
        dataset="replica",
        dataset_root=str(REPLICA_MOUNT),
        max_steps=1 if smoke else max_steps,
        wandb_mode="disabled",
        gaussian_weights=str(DEFAULT_GAUSSIAN_WEIGHTS),
        sam_checkpoint=str(DEFAULT_SAM_CHECKPOINT),
        val_interval=10_000 if smoke else 1000,
        batch_size=1 if smoke else 2,
        prompt_strategy="centroid",
        prompted_seg_loss_weight=1.0,
        min_object_pixels=16,
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
