#!/usr/bin/env python3
"""Modal runner to score exported SAM / C3G-SAM masks against GT test labels.

Reads predictions from ``vanilla-sam-outputs`` (``sam``) or ``c3g-sam-eval-outputs``
(``c3gsam``) and compares them to Replica + ScanNet **test** labels:

- **IoU** (dense pred vs GT label map; mean over classes present in both)
- **boundary mIoU** (same shared-class set, boundary IoU per class)
- **warp mIoU** (dense pred maps warped across adjacent frames; classes present in both frames only; uses `{frame_id}_depth.png`)

Examples::

    modal run src/modal_infra/modal_get_scores.py --experiment sam --wait
    modal run src/modal_infra/modal_get_scores.py --experiment c3gsam --wait
    modal run src/modal_infra/modal_get_scores.py::smoke --experiment sam --wait
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import modal

from src.modal_infra.modal_common import (
    C3G_SAM_EVAL_OUTPUT_MOUNT,
    C3G_SAM_EVAL_OUTPUT_VOLUME,
    REPLICA_2DSEG_SCENES,
    REPLICA_MOUNT,
    REPLICA_VOLUME,
    SCANNET_2DSEG_TEST_SCENES,
    SCANNET_MOUNT,
    SCANNET_VOLUME,
    VANILLA_SAM_OUTPUT_MOUNT,
    VANILLA_SAM_OUTPUT_VOLUME,
    build_eval_sam_modal_image,
    expected_mask_class_ids,
    find_smoke_scene,
    resolve_dataset_root,
    resolve_detach,
)

APP_NAME = "c3g-get-scores"
DEFAULT_MIN_OBJECT_PIXELS = 16
DEFAULT_DILATION_RATIO = 0.02
SCORES_FILENAME = "scores.json"

ExperimentName = Literal["sam", "c3gsam"]
DatasetName = Literal["replica", "scannet"]

TEST_SCENES: dict[DatasetName, list[str]] = {
    "replica": list(REPLICA_2DSEG_SCENES),
    "scannet": list(SCANNET_2DSEG_TEST_SCENES),
}

EXPERIMENT_PRED_ROOTS: dict[ExperimentName, Path] = {
    "sam": VANILLA_SAM_OUTPUT_MOUNT,
    "c3gsam": C3G_SAM_EVAL_OUTPUT_MOUNT,
}

app = modal.App(APP_NAME)

replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
vanilla_output_volume = modal.Volume.from_name(
    VANILLA_SAM_OUTPUT_VOLUME, create_if_missing=True
)
c3g_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EVAL_OUTPUT_VOLUME, create_if_missing=True
)

scores_image = build_eval_sam_modal_image()

SCORE_VOLUMES = {
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
}


@dataclass
class DatasetScores:
    iou: float
    boundary_iou: float
    warp_iou: float | None
    num_scored_classes: int
    num_warp_pairs: int
    missing_predictions: int
    skipped_warp_pairs: int


def _resolve_experiment(experiment: str) -> ExperimentName:
    normalized = experiment.strip().lower()
    if normalized not in EXPERIMENT_PRED_ROOTS:
        raise ValueError(
            f"Unknown experiment {experiment!r}; expected 'sam' or 'c3gsam'."
        )
    return normalized  # type: ignore[return-value]


def _load_camera_npz(camera_path: Path) -> tuple:
    import numpy as np

    metadata = np.load(camera_path)
    pose = metadata["camera_pose"].astype(np.float32)
    intrinsics = metadata["camera_intrinsics"].astype(np.float32)
    return pose, intrinsics


def _load_pred_mask(pred_path: Path):
    import numpy as np
    from PIL import Image

    if not pred_path.is_file():
        return None
    return np.array(Image.open(pred_path))


def _pred_mask_to_bool(mask):
    import numpy as np

    if mask.dtype == np.bool_:
        return mask
    return mask > 128


def _classes_in_dense_map(dense) -> set[int]:
    import numpy as np

    return {int(class_id) for class_id in np.unique(dense) if class_id != 0}


def _build_dense_pred_mask(
    frame_pred_dir: Path,
    label_shape: tuple[int, int],
    class_ids: list[int],
) -> tuple:
    """Merge per-class binary PNGs into one dense label map."""
    import numpy as np

    dense = np.zeros(label_shape, dtype=np.int32)
    missing_predictions = 0
    for class_id in class_ids:
        binary = _load_pred_mask(frame_pred_dir / f"{class_id}.png")
        if binary is None:
            missing_predictions += 1
            continue
        dense[_pred_mask_to_bool(binary)] = class_id
    return dense, missing_predictions


def _shared_class_iou_scores(
    pred_dense,
    ref_dense,
    class_ids: set[int],
    *,
    dilation_ratio: float,
) -> tuple[list[float], list[float]]:
    from src.evaluation.mask_metrics import boundary_iou, mask_iou

    ious: list[float] = []
    boundary_ious: list[float] = []
    for class_id in sorted(class_ids):
        pred_bin = pred_dense == class_id
        ref_bin = ref_dense == class_id
        if not pred_bin.any() or not ref_bin.any():
            continue
        ious.append(mask_iou(pred_bin, ref_bin))
        boundary_ious.append(
            boundary_iou(pred_bin, ref_bin, dilation_ratio=dilation_ratio)
        )
    return ious, boundary_ious


def _frame_depth_path(scene_dir: Path, frame_id: str) -> Path:
    return scene_dir / f"{frame_id}_depth.png"


def _depth_to_meters(raw, dataset: DatasetName):
    import numpy as np

    depth = raw.astype(np.float32)
    if dataset == "scannet":
        # Prepared ScanNet volume stores sens depth as uint16 millimeters.
        return depth / 1000.0
    # Prepared Replica volume stores renderer depth as uint16 millimeters.
    return depth / 1000.0


def _load_frame_depth_meters(
    scene_dir: Path,
    frame_id: str,
    *,
    dataset: DatasetName,
    image_size: tuple[int, int],
):
    import cv2

    depth_path = _frame_depth_path(scene_dir, frame_id)
    if not depth_path.is_file():
        return None

    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[..., 0]

    depth_m = _depth_to_meters(raw, dataset)
    if tuple(depth_m.shape[:2]) != image_size:
        depth_m = cv2.resize(
            depth_m,
            (image_size[1], image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    depth_m = depth_m.copy()
    depth_m[depth_m <= 0] = 0.0
    return depth_m


def _score_gt_masks(
    *,
    dataset_root: Path,
    pred_root: Path,
    scenes: list[str],
    min_object_pixels: int,
    dilation_ratio: float,
) -> tuple[list[float], list[float], int]:
    import numpy as np
    from PIL import Image

    from src.misc.frame_layout import FramePaths, list_frame_ids

    ious: list[float] = []
    boundary_ious: list[float] = []
    missing_predictions = 0

    for scene_id in scenes:
        scene_dir = dataset_root / scene_id
        if not scene_dir.is_dir():
            continue
        for frame_id in list_frame_ids(scene_dir):
            paths = FramePaths.from_frame_id(scene_dir, frame_id)
            if not paths.label.is_file():
                continue

            gt_dense = np.array(Image.open(paths.label))
            gt_classes = set(
                expected_mask_class_ids(
                    paths.label, min_object_pixels=min_object_pixels
                )
            )
            pred_dense, missing = _build_dense_pred_mask(
                pred_root / scene_id / frame_id,
                gt_dense.shape[:2],
                sorted(gt_classes),
            )
            missing_predictions += missing

            shared_classes = gt_classes & _classes_in_dense_map(pred_dense)
            frame_ious, frame_boundary_ious = _shared_class_iou_scores(
                pred_dense,
                gt_dense,
                shared_classes,
                dilation_ratio=dilation_ratio,
            )
            ious.extend(frame_ious)
            boundary_ious.extend(frame_boundary_ious)

    return ious, boundary_ious, missing_predictions


def _score_adjacent_warp_masks(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    pred_root: Path,
    scenes: list[str],
    min_object_pixels: int,
) -> tuple[list[float], int]:
    import numpy as np
    from PIL import Image

    from src.evaluation.mask_metrics import warp_mask_iou
    from src.misc.frame_layout import FramePaths, list_frame_ids

    warp_ious: list[float] = []
    skipped_warp_pairs = 0

    for scene_id in scenes:
        scene_dir = dataset_root / scene_id
        if not scene_dir.is_dir():
            continue

        frame_ids = list_frame_ids(scene_dir)
        for frame_index in range(len(frame_ids) - 1):
            frame_a = frame_ids[frame_index]
            frame_b = frame_ids[frame_index + 1]
            paths_a = FramePaths.from_frame_id(scene_dir, frame_a)
            paths_b = FramePaths.from_frame_id(scene_dir, frame_b)

            if not (
                paths_a.camera.is_file()
                and paths_b.camera.is_file()
                and paths_a.label.is_file()
                and paths_b.label.is_file()
            ):
                skipped_warp_pairs += 1
                continue

            label_a = np.array(Image.open(paths_a.label))
            image_size = tuple(label_a.shape[:2])
            depth_a = _load_frame_depth_meters(
                scene_dir, frame_a, dataset=dataset, image_size=image_size
            )
            depth_b = _load_frame_depth_meters(
                scene_dir, frame_b, dataset=dataset, image_size=image_size
            )
            if depth_a is None or depth_b is None:
                skipped_warp_pairs += 1
                continue

            ext_a, int_a = _load_camera_npz(paths_a.camera)
            ext_b, int_b = _load_camera_npz(paths_b.camera)

            gt_classes_a = set(
                expected_mask_class_ids(
                    paths_a.label, min_object_pixels=min_object_pixels
                )
            )
            gt_classes_b = set(
                expected_mask_class_ids(
                    paths_b.label, min_object_pixels=min_object_pixels
                )
            )
            pred_dense_a, _ = _build_dense_pred_mask(
                pred_root / scene_id / frame_a,
                image_size,
                sorted(gt_classes_a),
            )
            pred_dense_b, _ = _build_dense_pred_mask(
                pred_root / scene_id / frame_b,
                image_size,
                sorted(gt_classes_b),
            )
            shared_classes = _classes_in_dense_map(
                pred_dense_a
            ) & _classes_in_dense_map(pred_dense_b)
            if not shared_classes:
                skipped_warp_pairs += 1
                continue

            pair_scores: list[float] = []
            for class_id in sorted(shared_classes):
                pred_a = (pred_dense_a == class_id).astype(np.uint8)
                pred_b = (pred_dense_b == class_id).astype(np.uint8)
                pair_scores.append(
                    warp_mask_iou(
                        pred_a,
                        pred_b,
                        ext_a,
                        ext_b,
                        int_a,
                        int_b,
                        image_size,
                        depth=depth_a,
                    )
                )
                pair_scores.append(
                    warp_mask_iou(
                        pred_b,
                        pred_a,
                        ext_b,
                        ext_a,
                        int_b,
                        int_a,
                        image_size,
                        depth=depth_b,
                    )
                )

            if not pair_scores:
                skipped_warp_pairs += 1
                continue
            warp_ious.append(float(np.mean(pair_scores)))

    return warp_ious, skipped_warp_pairs


def _score_dataset(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    pred_root: Path,
    min_object_pixels: int,
    dilation_ratio: float,
    scenes: list[str] | None = None,
) -> DatasetScores:
    import numpy as np

    scene_ids = scenes if scenes is not None else TEST_SCENES[dataset]
    predictions = pred_root / dataset

    ious, boundary_ious, missing_predictions = _score_gt_masks(
        dataset_root=dataset_root,
        pred_root=predictions,
        scenes=scene_ids,
        min_object_pixels=min_object_pixels,
        dilation_ratio=dilation_ratio,
    )
    warp_ious, skipped_warp_pairs = _score_adjacent_warp_masks(
        dataset=dataset,
        dataset_root=dataset_root,
        pred_root=predictions,
        scenes=scene_ids,
        min_object_pixels=min_object_pixels,
    )

    return DatasetScores(
        iou=float(np.mean(ious)) if ious else 0.0,
        boundary_iou=float(np.mean(boundary_ious)) if boundary_ious else 0.0,
        warp_iou=float(np.mean(warp_ious)) if warp_ious else None,
        num_scored_classes=len(ious),
        num_warp_pairs=len(warp_ious),
        missing_predictions=missing_predictions,
        skipped_warp_pairs=skipped_warp_pairs,
    )


def _run_scoring(
    *,
    experiment: ExperimentName,
    replica_root: str,
    scannet_root: str,
    pred_root: Path,
    min_object_pixels: int,
    dilation_ratio: float,
    scenes: dict[DatasetName, list[str]] | None = None,
) -> dict:
    results: dict[str, dict] = {}

    for dataset, dataset_root in (
        ("replica", Path(replica_root)),
        ("scannet", Path(scannet_root)),
    ):
        dataset_scenes = None
        if scenes is not None:
            dataset_scenes = scenes[dataset]  # type: ignore[index]

        print(f"[scores/{experiment}/{dataset}/test] scoring...")
        scores = _score_dataset(
            dataset=dataset,  # type: ignore[arg-type]
            dataset_root=dataset_root,
            pred_root=pred_root,
            min_object_pixels=min_object_pixels,
            dilation_ratio=dilation_ratio,
            scenes=dataset_scenes,
        )
        print(
            f"[scores/{experiment}/{dataset}/test] "
            f"iou={scores.iou:.4f} "
            f"boundary_iou={scores.boundary_iou:.4f} "
            f"warp_iou={scores.warp_iou} "
            f"classes={scores.num_scored_classes} "
            f"missing={scores.missing_predictions}"
        )
        results[dataset] = asdict(scores)

    return {
        "experiment": experiment,
        "split": "test",
        "pred_root": str(pred_root),
        "min_object_pixels": min_object_pixels,
        "dilation_ratio": dilation_ratio,
        "results": results,
    }


def _commit_pred_volume(experiment: ExperimentName) -> None:
    if experiment == "sam":
        vanilla_output_volume.commit()
    else:
        c3g_eval_output_volume.commit()


@app.function(
    image=scores_image,
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 6,
    volumes=SCORE_VOLUMES,
)
def compute_scores(
    experiment: str = "sam",
    replica_root: str | None = None,
    scannet_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    output_path: str | None = None,
) -> dict:
    """Score exported masks on Modal worker volumes."""
    experiment_name = _resolve_experiment(experiment)
    replica_data_root = resolve_dataset_root("replica", replica_root)
    scannet_data_root = resolve_dataset_root("scannet", scannet_root)
    predictions_root = (
        Path(pred_root) if pred_root else EXPERIMENT_PRED_ROOTS[experiment_name]
    )

    for dataset_name, root in (
        ("replica", replica_data_root),
        ("scannet", scannet_data_root),
    ):
        if not Path(root).is_dir():
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {root}. "
                f"Populate the `{dataset_name}` volume."
            )
    if not predictions_root.is_dir():
        raise FileNotFoundError(
            f"Predictions not found at {predictions_root}. "
            f"Run mask export for experiment={experiment_name!r} first."
        )

    report = _run_scoring(
        experiment=experiment_name,
        replica_root=replica_data_root,
        scannet_root=scannet_data_root,
        pred_root=predictions_root,
        min_object_pixels=min_object_pixels,
        dilation_ratio=dilation_ratio,
    )

    out_path = Path(output_path) if output_path else predictions_root / SCORES_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _commit_pred_volume(experiment_name)
    print(f"Wrote scores to {out_path}")
    return report


@app.function(
    image=scores_image,
    cpu=2,
    memory=8192,
    timeout=60 * 30,
    volumes=SCORE_VOLUMES,
)
def smoke_scores(
    experiment: str = "sam",
    dataset: str = "replica",
    dataset_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> dict:
    """Score one test scene as a quick sanity check."""
    experiment_name = _resolve_experiment(experiment)
    if dataset not in TEST_SCENES:
        raise ValueError(f"Unknown dataset {dataset!r}")

    data_root = resolve_dataset_root(dataset, dataset_root)  # type: ignore[arg-type]
    predictions_root = (
        Path(pred_root) if pred_root else EXPERIMENT_PRED_ROOTS[experiment_name]
    )
    scene_id = find_smoke_scene(data_root, scenes=TEST_SCENES[dataset])  # type: ignore[index]

    scores = _score_dataset(
        dataset=dataset,  # type: ignore[arg-type]
        dataset_root=Path(data_root),
        pred_root=predictions_root,
        min_object_pixels=min_object_pixels,
        dilation_ratio=DEFAULT_DILATION_RATIO,
        scenes=[scene_id],
    )

    payload = {
        "experiment": experiment_name,
        "dataset": dataset,
        "split": "test",
        "scene_id": scene_id,
        "scores": asdict(scores),
    }
    print(json.dumps(payload, indent=2))
    return payload


def _dispatch(
    fn,
    *,
    job_name: str,
    detach: bool,
    **kwargs,
) -> None:
    from src.misc.modal_run import dispatch_remote

    dispatch_remote(
        fn,
        detach=detach,
        job_name=job_name,
        app_name=APP_NAME,
        **kwargs,
    )


@app.local_entrypoint()
def main(
    experiment: str = "sam",
    replica_root: str | None = None,
    scannet_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    dilation_ratio: float = DEFAULT_DILATION_RATIO,
    output_path: str | None = None,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Score exported masks for Replica + ScanNet test splits."""
    _resolve_experiment(experiment)
    _dispatch(
        compute_scores,
        job_name=f"{experiment} mask scoring",
        detach=resolve_detach(detach=detach, remote_job=not wait),
        experiment=experiment,
        replica_root=replica_root,
        scannet_root=scannet_root,
        pred_root=pred_root,
        min_object_pixels=min_object_pixels,
        dilation_ratio=dilation_ratio,
        output_path=output_path,
    )


@app.local_entrypoint()
def smoke(
    experiment: str = "sam",
    dataset: str = "replica",
    dataset_root: str | None = None,
    pred_root: str | None = None,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Smoke test: score a single test scene."""
    _resolve_experiment(experiment)
    _dispatch(
        smoke_scores,
        job_name=f"{experiment} mask scoring smoke ({dataset})",
        detach=resolve_detach(detach=detach, remote_job=not wait),
        experiment=experiment,
        dataset=dataset,
        dataset_root=dataset_root,
        pred_root=pred_root,
        min_object_pixels=min_object_pixels,
    )
