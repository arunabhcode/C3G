#!/usr/bin/env python3
"""Modal training for the C3G-F (feature) decoder with SAM mask-decoder head.

Trains the Gaussian feature renderer + :class:`SAMMaskDecoderWrapper` pipeline using
Hydra config ``+training=feature_head_sam`` (``reproj_model: sam``,
``gaussian_feature_dim: 256``).

Prerequisites on Modal volumes::

    # SAM + VGGT backbone (and optional Gaussian init checkpoint)
    modal volume put c3g-weights sam_vit_h.pth /path/to/sam_vit_h.pth
    modal volume put c3g-weights model.pt /path/to/model.pt
    modal volume put c3g-weights gaussian_decoder.ckpt /path/to/gaussian_decoder.ckpt

    # Prepared Replica 2D-seg scenes (see download_replica.py)
    modal volume put replica office0_x.jpg /replica/office0/...  # or populate via:
    modal run src/dataset/replica_data/download_replica.py

Run training::

    modal run --detach src/inference/modal_train_c3g_sam.py \\
        --run-name replica_sam_run1

Resume from a checkpoint on the output volume::

    modal run src/inference/modal_train_c3g_sam.py \\
        --run-name replica_sam_run1 \\
        --resume /outputs/checkpoints/last.ckpt
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

APP_NAME = "c3g-train-sam-feature"
WEIGHTS_VOLUME = "c3g-weights"
OUTPUT_VOLUME = "c3g-train-outputs"
REPLICA_VOLUME = "replica"

WEIGHTS_MOUNT = Path("/weights")
DATA_MOUNT = Path("/data/replica")
OUTPUT_MOUNT = Path("/outputs")
WORKSPACE = Path("/workspace")

SAM_NUM_CHANNELS = 256
CONFIG_H = (
    "submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h"
)


def _build_training_image():
    import modal

    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "git",
            "curl",
            "ca-certificates",
            "build-essential",
            "libgl1",
            "libglib2.0-0",
        )
        .run_commands(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "echo 'export PATH=\"/root/.local/bin:$PATH\"' >> /root/.bashrc",
        )
        .env({"PATH": "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        .add_local_dir(
            str(REPO_ROOT),
            remote_path=str(WORKSPACE),
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
            # SAM feature dim for diff_gaussian_rasterization_w_feature_detach
            f"sed -i 's/#define NUM_SEMANTIC_CHANNELS 512/#define NUM_SEMANTIC_CHANNELS {SAM_NUM_CHANNELS}/' {CONFIG_H}",
            "uv sync --frozen",
            # JIT-compile CUDA rasterizers once at image build time
            "python -c \"from submodules.diff_gaussian_rasterization_w_feature_detach.diff_gaussian_rasterization import GaussianRasterizationSettings\"",
            "python -c \"from submodules.diff_gaussian_rasterization_w_pose.diff_gaussian_rasterization import GaussianRasterizationSettings\"",
        )
        .env({"PYTHONPATH": str(WORKSPACE)})
    )


def _build_train_command(
    *,
    run_name: str,
    max_steps: int,
    wandb_mode: str,
    resume: str | None,
    gaussian_weights: str,
    sam_checkpoint: str,
    vggt_weights: str,
    dataset_root: str,
    val_interval: int,
    batch_size: int,
) -> list[str]:
    run_dir = OUTPUT_MOUNT / "runs" / run_name
    overrides = [
        "+training=feature_head_sam",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_interval}",
        f"data_loader.train.batch_size={batch_size}",
        f"model.encoder.pretrained_weights={gaussian_weights}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"dataset.replica.roots=[{dataset_root}]",
        # Required so main.py sets encoder.feature_dim from the SAM encoder (256).
        "train.feature_rendering_loss=0.01",
    ]
    if resume:
        overrides.append(f"checkpointing.load={resume}")

    return [
        "python",
        "-m",
        "src.main",
        *overrides,
    ]


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App(APP_NAME)
    training_image = _build_training_image()
    weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
    output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

    @app.function(
        image=training_image,
        gpu="A100",
        timeout=60 * 60 * 24,
        volumes={
            str(WEIGHTS_MOUNT): weights_volume,
            str(DATA_MOUNT): replica_volume,
            str(OUTPUT_MOUNT): output_volume,
        },
    )
    def train_c3g_sam_feature(
        run_name: str,
        max_steps: int = 5001,
        wandb_mode: str = "disabled",
        resume: str | None = None,
        gaussian_weights: str | None = None,
        sam_checkpoint: str | None = None,
        vggt_weights: str | None = None,
        dataset_root: str | None = None,
        val_interval: int = 1000,
        batch_size: int = 2,
    ) -> str:
        """Train C3G-F + SAM decoder; returns the Hydra run directory path."""
        gaussian_path = gaussian_weights or str(WEIGHTS_MOUNT / "gaussian_decoder.ckpt")
        if not Path(gaussian_path).is_file():
            # Fall back to VGGT backbone weights only (slower convergence).
            gaussian_path = vggt_weights or str(WEIGHTS_MOUNT / "model.pt")

        sam_path = sam_checkpoint or str(WEIGHTS_MOUNT / "sam_vit_h.pth")
        data_root = dataset_root or str(DATA_MOUNT)

        for required, label in (
            (gaussian_path, "Gaussian init / VGGT weights"),
            (sam_path, "SAM checkpoint"),
        ):
            if not Path(required).is_file():
                raise FileNotFoundError(
                    f"{label} not found at {required}. "
                    f"Upload to the `{WEIGHTS_VOLUME}` volume."
                )

        if not Path(data_root).is_dir():
            raise FileNotFoundError(
                f"Replica dataset not found at {data_root}. "
                f"Populate the `{REPLICA_VOLUME}` volume via download_replica.py."
            )

        cmd = _build_train_command(
            run_name=run_name,
            max_steps=max_steps,
            wandb_mode=wandb_mode,
            resume=resume,
            gaussian_weights=gaussian_path,
            sam_checkpoint=sam_path,
            vggt_weights=vggt_weights or str(WEIGHTS_MOUNT / "model.pt"),
            dataset_root=data_root,
            val_interval=val_interval,
            batch_size=batch_size,
        )

        print("Running:", " ".join(shlex.quote(part) for part in cmd))
        subprocess.run(cmd, check=True, cwd=str(WORKSPACE))

        run_dir = OUTPUT_MOUNT / "runs" / run_name
        output_volume.commit()
        print(f"Training complete. Artifacts under {run_dir} (volume `{OUTPUT_VOLUME}`).")
        return str(run_dir)

    @app.local_entrypoint()
    def modal_main(
        run_name: str,
        max_steps: int = 5001,
        wandb_mode: str = "disabled",
        resume: str | None = None,
        gaussian_weights: str | None = None,
        sam_checkpoint: str | None = None,
        dataset_root: str | None = None,
        val_interval: int = 1000,
        batch_size: int = 2,
    ) -> None:
        run_dir = train_c3g_sam_feature.remote(
            run_name=run_name,
            max_steps=max_steps,
            wandb_mode=wandb_mode,
            resume=resume,
            gaussian_weights=gaussian_weights,
            sam_checkpoint=sam_checkpoint,
            dataset_root=dataset_root,
            val_interval=val_interval,
            batch_size=batch_size,
        )
        print(f"Remote run finished: {run_dir}")

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]
