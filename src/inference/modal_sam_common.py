"""Shared constants and helpers for C3G SAM Modal inference/training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from src.misc.frame_layout import FramePaths

# Keep in sync with dataset_replica_2dseg / dataset_scannet_2dseg (avoid importing
# src.dataset here — that package pulls in torch at import time).
REPLICA_2DSEG_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]
SCANNET_2DSEG_SCENES = [f"scene{i:04d}_00" for i in range(697, 712)]

DatasetName = Literal["replica", "scannet"]
TrainingConfigName = Literal["feature_head_sam_prompted", "feature_head_sam"]
TRAINING_CONFIG_PROMPTED: TrainingConfigName = "feature_head_sam_prompted"
TRAINING_CONFIG_SAM: TrainingConfigName = "feature_head_sam"

WEIGHTS_VOLUME = "c3g-weights"
OUTPUT_VOLUME = "c3g-train-outputs"
SAM_EVAL_OUTPUT_VOLUME = "sam-eval-outputs"
REPLICA_VOLUME = "replica"
SCANNET_VOLUME = "scannet"

WEIGHTS_MOUNT = Path("/weights")
REPLICA_MOUNT = Path("/replica")
SCANNET_MOUNT = Path("/scannet")
OUTPUT_MOUNT = Path("/outputs")
SAM_EVAL_OUTPUT_MOUNT = Path("/sam-eval-outputs")

SAM_NUM_CHANNELS = 256
DEFAULT_SAM_CHECKPOINT = WEIGHTS_MOUNT / "sam_vit_h.pth"
DEFAULT_VGGT_WEIGHTS = WEIGHTS_MOUNT / "model.pt"
DEFAULT_GAUSSIAN_WEIGHTS = WEIGHTS_MOUNT / "gaussian_decoder.ckpt"


def resolve_encoder_pretrained_weights(override: str | Path | None = None) -> Path:
    """Return encoder init weights from the ``c3g-weights`` volume.

    Prefers ``gaussian_decoder.ckpt``; falls back to ``model.pt`` when the
    Gaussian decoder checkpoint is not on the volume.
    """
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(
                f"Encoder weights not found at {path}. "
                f"Upload to the `{WEIGHTS_VOLUME}` volume."
            )
        return path

    if DEFAULT_GAUSSIAN_WEIGHTS.is_file():
        return DEFAULT_GAUSSIAN_WEIGHTS
    if DEFAULT_VGGT_WEIGHTS.is_file():
        return DEFAULT_VGGT_WEIGHTS

    raise FileNotFoundError(
        f"Missing encoder weights on `{WEIGHTS_VOLUME}` volume. "
        f"Upload one of:\n"
        f"  modal volume put {WEIGHTS_VOLUME} /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt\n"
        f"  modal volume put {WEIGHTS_VOLUME} /path/to/model.pt model.pt"
    )


def resolve_sam_checkpoint(override: str | Path | None = None) -> Path:
    """Return the SAM ViT-H checkpoint from the ``c3g-weights`` volume."""
    path = Path(override) if override is not None else DEFAULT_SAM_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM checkpoint not found at {path}. "
            f"Upload with:\n"
            f"  modal volume put {WEIGHTS_VOLUME} /path/to/sam_vit_h.pth sam_vit_h.pth"
        )
    return path


def resolve_training_weights(
    *,
    gaussian_weights: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve SAM and encoder init paths on the ``c3g-weights`` volume."""
    return (
        resolve_encoder_pretrained_weights(gaussian_weights),
        resolve_sam_checkpoint(sam_checkpoint),
    )

DATASET_SPECS: dict[DatasetName, dict[str, str | list[str]]] = {
    "replica": {
        "hydra_add": "replica_2dseg",
        "hydra_override_group": "dataset@dataset.replica_semseg",
        "roots_key": "dataset.replica_2dseg.roots",
        "overfit_key": "dataset.replica_2dseg.overfit_to_scene",
        # Prepared flat-frame Replica dataset lives on the Modal volume mounted here.
        "default_root": str(REPLICA_MOUNT),
        "volume": REPLICA_VOLUME,
        "label": "Replica",
        "scenes": REPLICA_2DSEG_SCENES,
    },
    "scannet": {
        "hydra_add": "scannet_2dseg",
        "hydra_override_group": "dataset@dataset.scannet_semseg",
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
    from src.misc.frame_layout import list_frame_ids

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
    from src.misc.frame_layout import FramePaths, list_frame_ids

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
    gaussian_weights: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
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
    encoder_weights, sam_path = resolve_training_weights(
        gaussian_weights=gaussian_weights,
        sam_checkpoint=sam_checkpoint,
    )
    spec = DATASET_SPECS[dataset]
    run_dir = output_mount / "runs" / run_name
    overrides = [
        f"+training={training_config}",
        f"{spec['hydra_override_group']}={spec['hydra_add']}",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"trainer.max_steps={max_steps}",
        f"trainer.val_check_interval={val_interval}",
        f"data_loader.train.batch_size={batch_size}",
        f"model.encoder.pretrained_weights={encoder_weights}",
        f"train.sam_checkpoint={sam_path}",
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
    gaussian_weights: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
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
    checkpoint_path: str,
    sam_checkpoint: str | Path | None = None,
    wandb_mode: str = "disabled",
    training_config: TrainingConfigName = TRAINING_CONFIG_PROMPTED,
    smoke_scene: str | None = None,
    output_mount: Path = SAM_EVAL_OUTPUT_MOUNT,
) -> list[str]:
    sam_path = resolve_sam_checkpoint(sam_checkpoint)
    spec = DATASET_SPECS[dataset]
    run_dir = output_mount / "runs" / run_name
    pred_root = output_mount / "runs"
    overrides = [
        f"+training={training_config}",
        f"{spec['hydra_override_group']}={spec['hydra_add']}",
        "mode=test",
        f"wandb.mode={wandb_mode}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"test.output_path={pred_root}",
        f"train.sam_checkpoint={sam_path}",
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


CONFIG_H = (
    "submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h"
)
C3G_MODAL_WORKSPACE = Path("/workspace")
VANILLA_SAM_MODAL_ROOT = Path("/root")


def repo_root_for_modal(start: Path | None = None) -> Path:
    """Locate the repo root when running locally or inside a Modal container."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate

    workspace = C3G_MODAL_WORKSPACE
    if (workspace / "pyproject.toml").is_file() and (workspace / "src").is_dir():
        return workspace

    raise RuntimeError(f"Could not find repo root from {here}")


def build_vanilla_sam_modal_image(
    *,
    src_root: Path | None = None,
    remote_root: Path = VANILLA_SAM_MODAL_ROOT,
):
    """Lightweight CUDA image for vanilla SAM inference on Modal.

    Installs ``git`` before any pip git dependencies (same ordering as
    :func:`build_c3g_modal_image`, which uses ``uv sync`` for segment-anything).
    """
    import modal

    # Resolve from this file so local dev and vanilla Modal workers (/root/src) both work
    # without a full repo checkout (workers mount only src/, not pyproject.toml).
    src = src_root or Path(__file__).resolve().parent.parent
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "ca-certificates")
        .pip_install(
            "numpy==1.26.4",
            "pillow==11.0.0",
            "fastapi==0.118.0",
            "pydantic==2.11.4",
        )
        .pip_install(
            "torch==2.5.1",
            "torchvision==0.20.1",
            index_url="https://download.pytorch.org/whl/cu124",
        )
        .pip_install(
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        )
        .env({"PYTHONPATH": str(remote_root)})
        .add_local_dir(
            str(src),
            remote_path=str(remote_root / "src"),
            ignore=["**/__pycache__/**", "**/.DS_Store"],
        )
    )


def build_c3g_modal_image(
    *,
    repo_root: Path | None = None,
    workspace: Path = C3G_MODAL_WORKSPACE,
):
    """CUDA + uv image for full C3G Hydra runs on Modal (training or test)."""
    import modal

    root = repo_root or repo_root_for_modal()
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
            str(root),
            remote_path=str(workspace),
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
        .workdir(str(workspace))
        .run_commands(
            f"sed -i 's/#define NUM_SEMANTIC_CHANNELS 512/#define NUM_SEMANTIC_CHANNELS {SAM_NUM_CHANNELS}/' {CONFIG_H}",
            "uv sync --frozen",
            "uv run --no-sync python -c \"from submodules.diff_gaussian_rasterization_w_feature_detach.setup import _C\"",
            "uv run --no-sync python -c \"from submodules.diff_gaussian_rasterization_w_pose.setup import _C\"",
        )
        .env({"PYTHONPATH": str(workspace)})
    )
