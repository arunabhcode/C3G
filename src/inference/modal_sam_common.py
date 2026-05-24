"""Shared constants and helpers for C3G SAM Modal inference/training scripts."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
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
# Train split: scene0000_00 … scene0774_00 (775 scenes). Val then test = last 32 on volume.
# See src/dataset/scannet_2dseg_splits.py (avoid importing src.dataset here — pulls torch).
SCANNET_2DSEG_SCENES = [f"scene{i:04d}_00" for i in range(807 - 8 - 24)]

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
DEFAULT_ENCODER_WEIGHTS = WEIGHTS_MOUNT / "gaussian_decoder.ckpt"


def resolve_encoder_pretrained_weights(override: str | Path | None = None) -> Path:
    """Return encoder init weights from the ``c3g-weights`` volume.

    Requires ``gaussian_decoder.ckpt`` (C3G Gaussian-decoder-trained encoder).
    """
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(
                f"Encoder weights not found at {path}. "
                f"Upload to the `{WEIGHTS_VOLUME}` volume."
            )
        return path

    if DEFAULT_ENCODER_WEIGHTS.is_file():
        return DEFAULT_ENCODER_WEIGHTS

    raise FileNotFoundError(
        f"Missing encoder weights on `{WEIGHTS_VOLUME}` volume. "
        f"Upload:\n"
        f"  modal volume put {WEIGHTS_VOLUME} /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt"
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

def dataset_group_hydra_overrides(
    hydra_add: str,
    training_config: TrainingConfigName,
) -> list[str]:
    """Add the 2dseg dataset config group (training preset has no dataset default)."""
    _ = training_config
    return [f"+dataset@_group_.{hydra_add}={hydra_add}"]


DATASET_SPECS: dict[DatasetName, dict[str, str | list[str]]] = {
    "replica": {
        "hydra_add": "replica_2dseg",
        "dataset_cfg_key": "dataset.replica_2dseg",
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
        "dataset_cfg_key": "dataset.scannet_2dseg",
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


def run_subprocess_with_output_commit(
    *,
    cmd: list[str],
    cwd: Path | str,
    run_dir: Path,
    commit: Callable[[], None],
    poll_interval_s: float = 15.0,
) -> None:
    """Run a training subprocess and persist ``run_dir`` to a Modal output volume.

    Commits when the Hydra run directory first appears and whenever a new
    ``*.ckpt`` file shows up under ``run_dir/checkpoints/``. A final commit
    always runs after the subprocess exits (success or failure).
    """
    stop_event = threading.Event()
    checkpoints_dir = run_dir / "checkpoints"
    seen_checkpoints: set[str] = set()
    run_dir_committed = False

    def _commit_loop() -> None:
        nonlocal run_dir_committed
        while not stop_event.is_set():
            if run_dir.is_dir() and not run_dir_committed:
                commit()
                run_dir_committed = True
                print(f"Committed output volume (run dir): {run_dir}")

            if checkpoints_dir.is_dir():
                for ckpt in checkpoints_dir.glob("*.ckpt"):
                    if ckpt.name not in seen_checkpoints:
                        seen_checkpoints.add(ckpt.name)
                        commit()
                        print(f"Committed output volume (checkpoint): {ckpt.name}")

            stop_event.wait(poll_interval_s)

    thread = threading.Thread(
        target=_commit_loop,
        name="modal-output-commit",
        daemon=True,
    )
    thread.start()
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd))
    finally:
        stop_event.set()
        thread.join(timeout=poll_interval_s + 5)
        commit()
        print(f"Committed output volume (final): {run_dir}")


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


def iter_dataset_frames(
    dataset_root: str | Path,
    scenes: list[str],
) -> list[tuple[str, FramePaths]]:
    """List every (scene_id, frame paths) with image + label on disk."""
    from src.misc.frame_layout import FramePaths, list_frame_ids

    root = Path(dataset_root)
    frames: list[tuple[str, FramePaths]] = []
    for scene_id in scenes:
        scene_dir = root / scene_id
        if not scene_dir.is_dir():
            continue
        for frame_id in list_frame_ids(scene_dir):
            paths = FramePaths.from_frame_id(scene_dir, frame_id)
            if paths.image.is_file() and paths.label.is_file():
                frames.append((scene_id, paths))
    if not frames:
        raise FileNotFoundError(
            f"No labeled frames found under {root} for scenes {scenes}. "
            "Run the dataset download script to populate the volume."
        )
    return frames


def build_sam_train_overrides(
    *,
    run_name: str,
    dataset: DatasetName,
    dataset_root: str,
    max_steps: int | None,
    wandb_mode: str,
    gaussian_weights: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
    val_interval: int | None = None,
    batch_size: int | None = None,
    training_config: TrainingConfigName = TRAINING_CONFIG_PROMPTED,
    prompt_strategy: str | None = None,
    prompted_seg_loss_weight: float | None = None,
    min_object_pixels: int | None = None,
    max_distance_between_context_views: int | None = None,
    min_distance_between_context_views: int | None = None,
    view_sampler_warm_up_steps: int | None = None,
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
        *dataset_group_hydra_overrides(spec["hydra_add"], training_config),
        f"wandb.mode={wandb_mode}",
        f"wandb.project={run_name}",
        f"wandb.name={run_name}",
        f"hydra.run.dir={run_dir}",
        f"model.encoder.pretrained_weights={encoder_weights}",
        f"train.sam_checkpoint={sam_path}",
        f"{spec['roots_key']}=[{dataset_root}]",
    ]
    if max_steps is not None:
        overrides.append(f"trainer.max_steps={max_steps}")
    if val_interval is not None:
        overrides.append(f"trainer.val_check_interval={val_interval}")
    if batch_size is not None:
        overrides.append(f"data_loader.train.batch_size={batch_size}")
    if max_distance_between_context_views is not None:
        overrides.append(
            f"{spec['dataset_cfg_key']}.view_sampler.max_distance_between_context_views={max_distance_between_context_views}"
        )
    if min_distance_between_context_views is not None:
        overrides.append(
            f"{spec['dataset_cfg_key']}.view_sampler.min_distance_between_context_views={min_distance_between_context_views}"
        )
    if view_sampler_warm_up_steps is not None:
        overrides.append(
            f"{spec['dataset_cfg_key']}.view_sampler.warm_up_steps={view_sampler_warm_up_steps}"
        )
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
    if training_config == TRAINING_CONFIG_PROMPTED:
        overrides.extend(
            [
                "model.encoder.freeze_backbone=true",
                "model.encoder.freeze_instill_qk=true",
            ]
        )
    return overrides


def build_prompted_train_overrides(
    *,
    run_name: str,
    dataset: DatasetName,
    dataset_root: str,
    max_steps: int | None,
    wandb_mode: str,
    gaussian_weights: str | Path | None = None,
    sam_checkpoint: str | Path | None = None,
    val_interval: int | None = None,
    batch_size: int | None = None,
    prompt_strategy: str | None = None,
    prompted_seg_loss_weight: float | None = None,
    min_object_pixels: int | None = None,
    max_distance_between_context_views: int | None = None,
    min_distance_between_context_views: int | None = None,
    view_sampler_warm_up_steps: int | None = None,
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
        max_distance_between_context_views=max_distance_between_context_views,
        min_distance_between_context_views=min_distance_between_context_views,
        view_sampler_warm_up_steps=view_sampler_warm_up_steps,
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
        *dataset_group_hydra_overrides(spec["hydra_add"], training_config),
        "mode=test",
        f"wandb.mode={wandb_mode}",
        f"wandb.project={run_name}",
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
