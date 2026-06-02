"""Shared constants and helpers for C3G Modal scripts in ``src/modal_infra/``."""

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
# Test split: scene0783_00 … scene0806_00 (24 scenes).
SCANNET_2DSEG_TEST_SCENES = [f"scene{i:04d}_00" for i in range(807 - 24, 807)]

DatasetName = Literal["replica", "scannet"]

# Vanilla SAM eval runs all Replica scenes plus ScanNet test only.
VANILLA_EVAL_DATASETS: list[tuple[DatasetName, list[str]]] = [
    ("replica", REPLICA_2DSEG_SCENES),
    ("scannet", SCANNET_2DSEG_TEST_SCENES),
]
WEIGHTS_VOLUME = "c3g-weights"
OUTPUT_VOLUME = "c3g-train-outputs"
VANILLA_SAM_OUTPUT_VOLUME = "vanilla-sam-outputs"
C3G_SAM_EVAL_OUTPUT_VOLUME = "c3g-sam-eval-outputs"
PRECOMPUTE_SAM_FEATURES_VOLUME = "precompute_sam_features"
REPLICA_VOLUME = "replica"
SCANNET_VOLUME = "scannet"

WEIGHTS_MOUNT = Path("/weights")
REPLICA_MOUNT = Path("/replica")
SCANNET_MOUNT = Path("/scannet")
OUTPUT_MOUNT = Path("/outputs")
VANILLA_SAM_OUTPUT_MOUNT = Path("/vanilla-sam-outputs")
C3G_SAM_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-eval-outputs")
PRECOMPUTE_SAM_FEATURES_MOUNT = Path("/precompute_sam_features")

DEFAULT_SAM_CHECKPOINT = WEIGHTS_MOUNT / "sam_vit_h.pth"
DEFAULT_DISTILLATION_CHECKPOINT = WEIGHTS_MOUNT / "distillation-base.ckpt"


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


def resolve_distillation_checkpoint(override: str | Path | None = None) -> Path:
    """Return the distillation Lightning checkpoint from the ``c3g-weights`` volume."""
    path = (
        Path(override) if override is not None else DEFAULT_DISTILLATION_CHECKPOINT
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Distillation checkpoint not found at {path}. "
            f"Upload with:\n"
            f"  modal volume put {WEIGHTS_VOLUME} /path/to/distillation-base.ckpt "
            f"distillation-base.ckpt"
        )
    return path


def sample_eval_visualization_keys(
    dataset_root: str | Path,
    scenes: list[str],
    *,
    count: int = 5,
    seed: int = 42,
) -> list[str]:
    """Pick ``count`` random ``scene/frame_id`` keys for test debug visualizations."""
    import random

    frames = iter_dataset_frames(dataset_root, scenes)
    if not frames:
        raise FileNotFoundError(
            f"No labeled frames under {dataset_root} for scenes {scenes}"
        )
    rng = random.Random(seed)
    picked = rng.sample(frames, min(count, len(frames)))
    return [f"{scene}/{paths.frame_id}" for scene, paths in picked]


DATASET_SPECS: dict[DatasetName, dict[str, str | list[str]]] = {
    "replica": {
        "default_root": str(REPLICA_MOUNT),
        "volume": REPLICA_VOLUME,
        "label": "Replica",
        "scenes": REPLICA_2DSEG_SCENES,
    },
    "scannet": {
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


def run_subprocess_with_precompute_commit(
    *,
    cmd: list[str],
    cwd: Path | str,
    output_root: Path,
    commit: Callable[[], None],
    poll_interval_s: float = 30.0,
) -> None:
    """Run precompute subprocess and persist ``*_sam.pt`` files to a Modal volume."""
    stop_event = threading.Event()
    seen_features: set[str] = set()

    def _commit_loop() -> None:
        while not stop_event.is_set():
            if output_root.is_dir():
                for feature_path in output_root.rglob("*_sam.pt"):
                    rel = str(feature_path.relative_to(output_root))
                    if rel not in seen_features:
                        seen_features.add(rel)
                        commit()
                        print(f"Committed precompute volume: {rel}")

            stop_event.wait(poll_interval_s)

    thread = threading.Thread(
        target=_commit_loop,
        name="modal-precompute-commit",
        daemon=True,
    )
    thread.start()
    try:
        subprocess.run(cmd, check=True, cwd=str(cwd))
    finally:
        stop_event.set()
        thread.join(timeout=poll_interval_s + 5)
        commit()
        print(f"Committed precompute volume (final): {output_root}")


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


C3G_MODAL_WORKSPACE = Path("/workspace")
C3G_MODAL_PYTHON = "3.12"
VANILLA_SAM_MODAL_ROOT = Path("/root")
VANILLA_SAM_PYTHON = "3.11"
PYTORCH_CU124_INDEX = "https://download.pytorch.org/whl/cu124"
PYTORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
MODAL_UV_PATH = "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Match distillation_sanity_check.sh / pyproject.toml (torch cu128, Blackwell sm_120).
C3G_CUDA_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu24.04"
C3G_TORCH_CUDA_ARCH_LIST = "9.0"

def build_eval_sam_modal_image(
    *,
    src_root: Path | None = None,
    remote_root: Path = VANILLA_SAM_MODAL_ROOT,
):
    """Lightweight image for ``modal_eval_sam`` (``uv pip install``)."""
    import modal

    # Resolve from this file so local dev and vanilla Modal workers (/root/src) both work
    # without a full repo checkout (workers mount only src/, not pyproject.toml).
    src = src_root or Path(__file__).resolve().parent.parent
    return (
        modal.Image.debian_slim(python_version=VANILLA_SAM_PYTHON)
        .apt_install("git", "ca-certificates", "curl")
        .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
        .env({"PATH": MODAL_UV_PATH})
        .run_commands(
            f"uv pip install --system --python {VANILLA_SAM_PYTHON} "
            "numpy==1.26.4 pillow==11.0.0 fastapi==0.118.0 pydantic==2.11.4",
            f"uv pip install --system --python {VANILLA_SAM_PYTHON} "
            f"torch==2.5.1 torchvision==0.20.1 --index-url {PYTORCH_CU124_INDEX}",
            f"uv pip install --system --python {VANILLA_SAM_PYTHON} "
            '"segment-anything @ git+https://github.com/facebookresearch/segment-anything.git"',
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
    cuda_image: str = C3G_CUDA_IMAGE,
    torch_cuda_arch_list: str = C3G_TORCH_CUDA_ARCH_LIST,
):
    """CUDA image for C3G on Modal: copy repo to ``/workspace`` and ``uv pip install -e``."""
    import modal

    if repo_root is None:
        here = Path(__file__).resolve()
        for candidate in (here.parent, *here.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
                repo_root = candidate
                break
        else:
            if (workspace / "pyproject.toml").is_file() and (workspace / "src").is_dir():
                repo_root = workspace
            else:
                raise RuntimeError(f"Could not find repo root from {here}")

    python_version = C3G_MODAL_PYTHON
    return (
        modal.Image.from_registry(cuda_image, add_python=python_version)
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
                "PATH": MODAL_UV_PATH,
                "TORCH_CUDA_ARCH_LIST": torch_cuda_arch_list,
                "FORCE_CUDA": "1",
                "UV_INDEX_PYTORCH_CU128_URL": PYTORCH_CU128_INDEX,
            }
        )
        .add_local_dir(
            str(repo_root),
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
            f"cd {workspace} && uv pip install --system --python {python_version} -e ."
        )
        .env({"PYTHONPATH": str(workspace)})
    )
