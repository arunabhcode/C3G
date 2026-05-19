#!/usr/bin/env python3
"""Modal training for the C3G-F (feature) decoder with SAM mask-decoder head.

Trains the Gaussian feature renderer + :class:`SAMMaskDecoderWrapper` pipeline using
Hydra config ``+training=feature_head_sam`` with point-prompted segmentation loss on
the ``replica_2dseg`` or ``scannet_2dseg`` loaders (flat frame layout from
``download_replica.py`` / ``download_scannet.py``).

Prerequisites on Modal volumes::

    modal volume put c3g-weights sam_vit_h.pth /path/to/sam_vit_h.pth
    modal volume put c3g-weights model.pt /path/to/model.pt
    modal volume put c3g-weights gaussian_decoder.ckpt /path/to/gaussian_decoder.ckpt

    modal run src/dataset/download_replica.py
    modal run src/dataset/download_scannet.py --accept-tos

Full training::

    modal run src/inference/modal_train_c3g_sam.py \\
        --run-name sam_prompted_replica --dataset replica

Smoke test (one training step on the first scene/frame)::

    modal run src/inference/modal_train_c3g_sam.py \\
        --smoke-test --dataset replica

Eval smoke test (one test batch; requires a checkpoint on the output volume)::

    modal run src/inference/modal_train_c3g_sam.py \\
        --test --dataset replica \\
        --run-name sam_prompted_replica_eval \\
        --resume /outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from src.inference.modal_sam_common import (
    DATASET_SPECS,
    DEFAULT_GAUSSIAN_WEIGHTS,
    DEFAULT_SAM_CHECKPOINT,
    DEFAULT_VGGT_WEIGHTS,
    OUTPUT_MOUNT,
    OUTPUT_VOLUME,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    SAM_NUM_CHANNELS,
    WEIGHTS_MOUNT,
    WEIGHTS_VOLUME,
    DatasetName,
    build_prompted_test_overrides,
    build_prompted_train_overrides,
    find_smoke_scene,
    resolve_dataset_root,
)

APP_NAME = "c3g-train-sam-feature"
WORKSPACE = Path("/workspace")

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
            f"sed -i 's/#define NUM_SEMANTIC_CHANNELS 512/#define NUM_SEMANTIC_CHANNELS {SAM_NUM_CHANNELS}/' {CONFIG_H}",
            "uv sync --frozen",
            "python -c \"from submodules.diff_gaussian_rasterization_w_feature_detach.diff_gaussian_rasterization import GaussianRasterizationSettings\"",
            "python -c \"from submodules.diff_gaussian_rasterization_w_pose.diff_gaussian_rasterization import GaussianRasterizationSettings\"",
        )
        .env({"PYTHONPATH": str(WORKSPACE)})
    )


def _build_main_command(overrides: list[str]) -> list[str]:
    return ["python", "-m", "src.main", *overrides]


def _resolve_weights(gaussian_weights: str | None) -> str:
    gaussian_path = gaussian_weights or str(DEFAULT_GAUSSIAN_WEIGHTS)
    if not Path(gaussian_path).is_file():
        gaussian_path = str(DEFAULT_VGGT_WEIGHTS)
    return gaussian_path


def _validate_paths(
    *,
    spec: dict[str, str],
    gaussian_path: str,
    sam_path: str,
    data_root: str,
) -> None:
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
            f"{spec['label']} 2D-seg dataset not found at {data_root}. "
            f"Populate the `{spec['volume']}` volume via download script."
        )


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App(APP_NAME)
    training_image = _build_training_image()
    weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
    replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
    scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
    output_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

    @app.function(
        image=training_image,
        gpu="A100-40GB",
        timeout=60 * 60 * 24,
        volumes={
            str(WEIGHTS_MOUNT): weights_volume,
            str(REPLICA_MOUNT): replica_volume,
            str(SCANNET_MOUNT): scannet_volume,
            str(OUTPUT_MOUNT): output_volume,
        },
    )
    def train_c3g_sam_feature(
        run_name: str,
        dataset: DatasetName = "replica",
        max_steps: int = 5001,
        wandb_mode: str = "disabled",
        resume: str | None = None,
        gaussian_weights: str | None = None,
        sam_checkpoint: str | None = None,
        dataset_root: str | None = None,
        val_interval: int = 1000,
        batch_size: int = 2,
        prompt_strategy: str = "centroid",
        prompted_seg_loss_weight: float = 1.0,
        min_object_pixels: int = 16,
        smoke_test: bool = False,
        test: bool = False,
    ) -> str:
        """Train, smoke-test (1 step), or eval (1 batch) C3G-F + SAM on 2dseg data."""
        if smoke_test and test:
            raise ValueError("Pass only one of smoke_test=True or test=True.")

        spec = DATASET_SPECS[dataset]
        gaussian_path = _resolve_weights(gaussian_weights)
        sam_path = sam_checkpoint or str(DEFAULT_SAM_CHECKPOINT)
        data_root = resolve_dataset_root(dataset, dataset_root)
        _validate_paths(
            spec=spec,
            gaussian_path=gaussian_path,
            sam_path=sam_path,
            data_root=data_root,
        )

        smoke_scene = find_smoke_scene(data_root)
        print(f"Smoke scene (first in index): {smoke_scene}")

        if test:
            if not resume:
                raise ValueError(
                    "test=True requires resume= with a checkpoint on the output volume."
                )
            if not Path(resume).is_file():
                raise FileNotFoundError(f"Checkpoint not found: {resume}")
            overrides = build_prompted_test_overrides(
                run_name=run_name,
                dataset=dataset,
                dataset_root=data_root,
                sam_checkpoint=sam_path,
                checkpoint_path=resume,
                wandb_mode=wandb_mode,
                smoke_scene=smoke_scene,
            )
        else:
            overrides = build_prompted_train_overrides(
                run_name=run_name,
                dataset=dataset,
                dataset_root=data_root,
                max_steps=1 if smoke_test else max_steps,
                wandb_mode="disabled" if smoke_test else wandb_mode,
                gaussian_weights=gaussian_path,
                sam_checkpoint=sam_path,
                val_interval=10_000 if smoke_test else val_interval,
                batch_size=1 if smoke_test else batch_size,
                prompt_strategy=prompt_strategy,
                prompted_seg_loss_weight=prompted_seg_loss_weight,
                min_object_pixels=min_object_pixels,
                resume=None if smoke_test else resume,
                smoke_scene=smoke_scene if smoke_test else None,
            )
            if smoke_test:
                overrides.extend(
                    [
                        "data_loader.train.num_workers=0",
                        "checkpointing.every_n_train_steps=1000000",
                    ]
                )

        cmd = _build_main_command(overrides)
        print("Running:", " ".join(shlex.quote(part) for part in cmd))
        subprocess.run(cmd, check=True, cwd=str(WORKSPACE))

        run_dir = OUTPUT_MOUNT / "runs" / run_name
        output_volume.commit()
        label = "Smoke test" if smoke_test else "Test" if test else "Training"
        print(f"{label} complete. Artifacts under {run_dir} (volume `{OUTPUT_VOLUME}`).")
        return str(run_dir)

    @app.local_entrypoint()
    def modal_main(
        run_name: str = "sam_smoke",
        dataset: DatasetName = "replica",
        max_steps: int = 5001,
        wandb_mode: str = "disabled",
        resume: str | None = None,
        gaussian_weights: str | None = None,
        sam_checkpoint: str | None = None,
        dataset_root: str | None = None,
        val_interval: int = 1000,
        batch_size: int = 2,
        prompt_strategy: str = "centroid",
        prompted_seg_loss_weight: float = 1.0,
        min_object_pixels: int = 16,
        smoke_test: bool = False,
        test: bool = False,
        detach: bool = False,
    ) -> None:
        from src.misc.modal_run import dispatch_remote

        if dataset not in DATASET_SPECS:
            print(
                f"Unknown dataset {dataset!r}; choose one of: {', '.join(DATASET_SPECS)}",
                file=sys.stderr,
            )
            raise SystemExit(2)

        if test and not resume:
            print("--test requires --resume with a checkpoint path.", file=sys.stderr)
            raise SystemExit(2)

        mode_label = "smoke test" if smoke_test else "test" if test else "training"
        result = dispatch_remote(
            train_c3g_sam_feature,
            run_name=run_name,
            dataset=dataset,
            max_steps=max_steps,
            wandb_mode=wandb_mode,
            resume=resume,
            gaussian_weights=gaussian_weights,
            sam_checkpoint=sam_checkpoint,
            dataset_root=dataset_root,
            val_interval=val_interval,
            batch_size=batch_size,
            prompt_strategy=prompt_strategy,
            prompted_seg_loss_weight=prompted_seg_loss_weight,
            min_object_pixels=min_object_pixels,
            smoke_test=smoke_test,
            test=test,
            detach=detach,
            job_name=f"C3G-F SAM {mode_label} ({run_name}, {dataset})",
            app_name=APP_NAME,
        )
        if detach:
            return
        print(f"Remote run finished: {result}")

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]
