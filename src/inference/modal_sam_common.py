"""Shared constants and helpers for C3G SAM Modal inference/training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from src.dataset.frame_layout import FramePaths, list_frame_ids

DatasetName = Literal["replica", "scannet"]

WEIGHTS_VOLUME = "c3g-weights"
OUTPUT_VOLUME = "c3g-train-outputs"
REPLICA_VOLUME = "replica"
SCANNET_VOLUME = "scannet"

WEIGHTS_MOUNT = Path("/weights")
REPLICA_MOUNT = Path("/replica")
SCANNET_MOUNT = Path("/scannet")
OUTPUT_MOUNT = Path("/outputs")

SAM_NUM_CHANNELS = 256
DEFAULT_SAM_CHECKPOINT = WEIGHTS_MOUNT / "sam_vit_h.pth"
DEFAULT_VGGT_WEIGHTS = WEIGHTS_MOUNT / "model.pt"
DEFAULT_GAUSSIAN_WEIGHTS = WEIGHTS_MOUNT / "gaussian_decoder.ckpt"

DATASET_SPECS: dict[DatasetName, dict[str, str]] = {
    "replica": {
        "hydra_add": "replica_2dseg=replica_2dseg",
        "roots_key": "dataset.replica_2dseg.roots",
        "overfit_key": "dataset.replica_2dseg.overfit_to_scene",
        "default_root": str(REPLICA_MOUNT),
        "volume": REPLICA_VOLUME,
        "label": "Replica",
    },
    "scannet": {
        "hydra_add": "scannet_2dseg=scannet_2dseg",
        "roots_key": "dataset.scannet_2dseg.roots",
        "overfit_key": "dataset.scannet_2dseg.overfit_to_scene",
        "default_root": str(SCANNET_MOUNT),
        "volume": SCANNET_VOLUME,
        "label": "ScanNet",
    },
}


def resolve_dataset_root(dataset: DatasetName, dataset_root: str | None) -> str:
    return dataset_root or DATASET_SPECS[dataset]["default_root"]


def resolve_detach(*, detach: bool | None, remote_job: bool) -> bool:
    """Smoke/remote jobs detach by default; pass ``detach=False`` or ``--wait`` to block."""
    if detach is not None:
        return detach
    return remote_job


def find_smoke_scene(dataset_root: str | Path) -> str:
    """Return the first scene id listed in ``selected_seqs_test.json``."""
    root = Path(dataset_root)
    index_path = root / "selected_seqs_test.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing scene index: {index_path}")
    with index_path.open("r") as file_handle:
        scenes = json.load(file_handle)
    if not scenes:
        raise ValueError(f"No scenes in {index_path}")
    return next(iter(scenes))


def find_smoke_frame(dataset_root: str | Path) -> tuple[str, FramePaths]:
    """Return the first scene and frame triplet available on disk."""
    root = Path(dataset_root)
    scene_id = find_smoke_scene(root)
    scene_dir = root / scene_id
    frame_ids = list_frame_ids(scene_dir)
    if not frame_ids:
        raise FileNotFoundError(f"No frames under {scene_dir}")
    frame_id = frame_ids[0]
    paths = FramePaths.from_frame_id(scene_dir, frame_id)
    for path in (paths.image, paths.camera, paths.label):
        if not path.is_file():
            raise FileNotFoundError(f"Missing smoke-test frame file: {path}")
    return scene_id, paths


def build_prompted_train_overrides(
    *,
    run_name: str,
    dataset: DatasetName,
    dataset_root: str,
    max_steps: int,
    wandb_mode: str,
    gaussian_weights: str,
    sam_checkpoint: str,
    val_interval: int,
    batch_size: int,
    prompt_strategy: str,
    prompted_seg_loss_weight: float,
    min_object_pixels: int,
    resume: str | None,
    output_mount: Path = OUTPUT_MOUNT,
    smoke_scene: str | None = None,
) -> list[str]:
    spec = DATASET_SPECS[dataset]
    run_dir = output_mount / "runs" / run_name
    overrides = [
        "+training=feature_head_sam",
        "~dataset@_group_.replica",
        f"+dataset@_group_.{spec['hydra_add']}",
        "train.prompt_mode=prompted",
        f"train.prompted_seg_loss_weight={prompted_seg_loss_weight}",
        f"train.prompt_strategy={prompt_strategy}",
        f"train.min_object_pixels={min_object_pixels}",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_interval}",
        f"data_loader.train.batch_size={batch_size}",
        f"model.encoder.pretrained_weights={gaussian_weights}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"{spec['roots_key']}=[{dataset_root}]",
        "train.feature_rendering_loss=0.01",
    ]
    if smoke_scene is not None:
        overrides.append(f"{spec['overfit_key']}={smoke_scene}")
    if resume:
        overrides.append(f"checkpointing.load={resume}")
    return overrides


def build_prompted_test_overrides(
    *,
    run_name: str,
    dataset: DatasetName,
    dataset_root: str,
    sam_checkpoint: str,
    checkpoint_path: str,
    wandb_mode: str = "disabled",
    smoke_scene: str | None = None,
    output_mount: Path = OUTPUT_MOUNT,
) -> list[str]:
    spec = DATASET_SPECS[dataset]
    run_dir = output_mount / "runs" / run_name
    overrides = [
        "+training=feature_head_sam",
        "~dataset@_group_.replica",
        f"+dataset@_group_.{spec['hydra_add']}",
        "mode=test",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"{spec['roots_key']}=[{dataset_root}]",
        "train.feature_rendering_loss=0.01",
        "test.save_compare=true",
        "test.save_image=false",
        "test.save_video=false",
        f"checkpointing.load={checkpoint_path}",
        "trainer.limit_test_batches=1",
    ]
    if smoke_scene is not None:
        overrides.append(f"{spec['overfit_key']}={smoke_scene}")
    return overrides
