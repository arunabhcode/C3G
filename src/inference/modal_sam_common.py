"""Shared constants and helpers for C3G SAM Modal inference/training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from src.dataset.dataset_replica_2dseg import SCENES as REPLICA_2DSEG_SCENES
from src.dataset.dataset_scannet_2dseg import SCENES as SCANNET_2DSEG_SCENES
from src.misc.frame_layout import FramePaths, list_frame_ids

DatasetName = Literal["replica", "scannet"]
TrainingConfigName = Literal["feature_head_sam_prompted", "feature_head_sam"]
TRAINING_CONFIG_PROMPTED: TrainingConfigName = "feature_head_sam_prompted"
TRAINING_CONFIG_SAM: TrainingConfigName = "feature_head_sam"

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

DATASET_SPECS: dict[DatasetName, dict[str, str | list[str]]] = {
    "replica": {
        "hydra_add": "replica_2dseg=replica_2dseg",
        "roots_key": "dataset.replica_2dseg.roots",
        "overfit_key": "dataset.replica_2dseg.overfit_to_scene",
        "default_root": str(REPLICA_MOUNT),
        "volume": REPLICA_VOLUME,
        "label": "Replica",
        "scenes": REPLICA_2DSEG_SCENES,
    },
    "scannet": {
        "hydra_add": "scannet_2dseg=scannet_2dseg",
        "roots_key": "dataset.scannet_2dseg.roots",
        "overfit_key": "dataset.scannet_2dseg.overfit_to_scene",
        "default_root": str(SCANNET_MOUNT),
        "volume": SCANNET_VOLUME,
        "label": "ScanNet",
        "scenes": SCANNET_2DSEG_SCENES,
    },
}


def resolve_dataset_root(dataset: DatasetName, dataset_root: str | None) -> str:
    return dataset_root or DATASET_SPECS[dataset]["default_root"]


def resolve_detach(*, detach: bool | None, remote_job: bool) -> bool:
    """Smoke/remote jobs detach by default; pass ``detach=False`` or ``--wait`` to block."""
    if detach is not None:
        return detach
    return remote_job


def find_smoke_scene(
    dataset_root: str | Path,
    *,
    scenes: list[str] | None = None,
) -> str:
    """Return the first scene that has prepared frames on disk."""
    root = Path(dataset_root)
    index_path = root / "selected_seqs_test.json"
    if index_path.is_file():
        with index_path.open("r") as file_handle:
            indexed = json.load(file_handle)
        for scene_id in indexed:
            if list_frame_ids(root / scene_id):
                return scene_id

    for scene_id in scenes or []:
        if list_frame_ids(root / scene_id):
            return scene_id

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith(("_", ".")):
            continue
        if list_frame_ids(scene_dir):
            return scene_dir.name

    raise FileNotFoundError(
        f"No scene with frames found under {root}. "
        "Run the dataset download script to populate the volume."
    )


def find_smoke_frame(
    dataset_root: str | Path,
    *,
    scenes: list[str] | None = None,
) -> tuple[str, FramePaths]:
    """Return the first scene and frame triplet available on disk."""
    root = Path(dataset_root)
    scene_id = find_smoke_scene(root, scenes=scenes)
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


def build_sam_train_overrides(
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
    training_config: TrainingConfigName = TRAINING_CONFIG_PROMPTED,
    prompt_strategy: str | None = None,
    prompted_seg_loss_weight: float | None = None,
    min_object_pixels: int | None = None,
    resume: str | None,
    output_mount: Path = OUTPUT_MOUNT,
    smoke_scene: str | None = None,
) -> list[str]:
    spec = DATASET_SPECS[dataset]
    run_dir = output_mount / "runs" / run_name
    overrides = [
        f"+training={training_config}",
        "~dataset@_group_.replica",
        "~dataset@_group_.replica_semseg",
        f"+dataset@_group_.{spec['hydra_add']}",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_interval}",
        f"data_loader.train.batch_size={batch_size}",
        f"model.encoder.pretrained_weights={gaussian_weights}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"{spec['roots_key']}=[{dataset_root}]",
    ]
    if training_config == TRAINING_CONFIG_PROMPTED:
        if prompt_strategy is not None:
            overrides.append(f"train.prompt_strategy={prompt_strategy}")
        if prompted_seg_loss_weight is not None:
            overrides.append(
                f"train.prompted_seg_loss_weight={prompted_seg_loss_weight}"
            )
        if min_object_pixels is not None:
            overrides.append(f"train.min_object_pixels={min_object_pixels}")
    if smoke_scene is not None:
        overrides.append(f"{spec['overfit_key']}={smoke_scene}")
    if resume:
        overrides.append(f"checkpointing.load={resume}")
    return overrides


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
    """Hydra overrides for prompted SAM training on ``replica_2dseg`` / ``scannet_2dseg``."""
    return build_sam_train_overrides(
        run_name=run_name,
        dataset=dataset,
        dataset_root=dataset_root,
        max_steps=max_steps,
        wandb_mode=wandb_mode,
        gaussian_weights=gaussian_weights,
        sam_checkpoint=sam_checkpoint,
        val_interval=val_interval,
        batch_size=batch_size,
        training_config=TRAINING_CONFIG_PROMPTED,
        prompt_strategy=prompt_strategy,
        prompted_seg_loss_weight=prompted_seg_loss_weight,
        min_object_pixels=min_object_pixels,
        resume=resume,
        output_mount=output_mount,
        smoke_scene=smoke_scene,
    )


def build_prompted_test_overrides(
    *,
    run_name: str,
    dataset: DatasetName,
    dataset_root: str,
    sam_checkpoint: str,
    checkpoint_path: str,
    wandb_mode: str = "disabled",
    training_config: TrainingConfigName = TRAINING_CONFIG_PROMPTED,
    smoke_scene: str | None = None,
    output_mount: Path = OUTPUT_MOUNT,
) -> list[str]:
    spec = DATASET_SPECS[dataset]
    run_dir = output_mount / "runs" / run_name
    overrides = [
        f"+training={training_config}",
        "~dataset@_group_.replica",
        "~dataset@_group_.replica_semseg",
        f"+dataset@_group_.{spec['hydra_add']}",
        "mode=test",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"train.sam_checkpoint={sam_checkpoint}",
        f"{spec['roots_key']}=[{dataset_root}]",
        "test.save_compare=true",
        "test.save_image=false",
        "test.save_video=false",
        f"checkpointing.load={checkpoint_path}",
        "trainer.limit_test_batches=1",
    ]
    if smoke_scene is not None:
        overrides.append(f"{spec['overfit_key']}={smoke_scene}")
    return overrides
