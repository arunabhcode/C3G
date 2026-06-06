#!/usr/bin/env python3
"""Visualize dense segmentation masks from Modal eval volumes.

For each eval output volume (vanilla SAM + C3G-SAM variants), picks two consecutive
frames from one random test scene per dataset (Replica + ScanNet), loads
``dense_mask.png`` predictions (or merges per-class exports), and writes a
side-by-side colored figure. Ground-truth label maps get the same treatment with
identical class colors across all figures (background = black).

Examples::

    modal run src/modal_infra/seg_viz.py --wait
    modal run src/modal_infra/seg_viz.py --output-dir seg_results --wait
    python -m src.modal_infra.seg_viz \\
        --replica-root /path/to/replica \\
        --scannet-root /path/to/scannet \\
        --sam-root /path/to/vanilla-sam-outputs \\
        --c3gsam-root /path/to/c3g-sam-eval-outputs \\
        --output-dir seg_results
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
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
    resolve_dataset_root,
    resolve_detach,
)

APP_NAME = "c3g-seg-viz"
DEFAULT_SEED = 42
DEFAULT_MIN_OBJECT_PIXELS = 16
DENSE_MASK_FILENAME = "dense_mask.png"
REMOTE_SCRATCH_DIR = Path("/tmp/seg_viz")


def default_local_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "seg_results"

C3G_SAM_DFT_EVAL_OUTPUT_VOLUME = "c3g-sam-dft-eval-outputs"
C3G_SAM_DFT_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-dft-eval-outputs")
C3G_SAM_NOMAGHEAD_EVAL_OUTPUT_VOLUME = "c3g-sam-nomaghead-eval-outputs"
C3G_SAM_NOMAGHEAD_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-nomaghead-eval-outputs")
C3G_SAM_EMA_NOMAG_EVAL_OUTPUT_VOLUME = "c3g-sam-ema-nomag-eval-outputs"
C3G_SAM_EMA_NOMAG_EVAL_OUTPUT_MOUNT = Path("/c3g-sam-ema-nomag-eval-outputs")

ExperimentName = Literal[
    "sam", "c3gsam", "c3gsam-dft", "c3gsam-nomaghead", "c3gsam-ema-nomag"
]
DatasetName = Literal["replica", "scannet"]

TEST_SCENES: dict[DatasetName, list[str]] = {
    "replica": list(REPLICA_2DSEG_SCENES),
    "scannet": list(SCANNET_2DSEG_TEST_SCENES),
}

EXPERIMENT_PRED_ROOTS: dict[ExperimentName, Path] = {
    "sam": VANILLA_SAM_OUTPUT_MOUNT,
    "c3gsam": C3G_SAM_EVAL_OUTPUT_MOUNT,
    "c3gsam-dft": C3G_SAM_DFT_EVAL_OUTPUT_MOUNT,
    "c3gsam-nomaghead": C3G_SAM_NOMAGHEAD_EVAL_OUTPUT_MOUNT,
    "c3gsam-ema-nomag": C3G_SAM_EMA_NOMAG_EVAL_OUTPUT_MOUNT,
}

# Same palette as src/visualization/colors.py, excluding black/white for foreground.
FOREGROUND_HEX_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
]

app = modal.App(APP_NAME)

replica_volume = modal.Volume.from_name(REPLICA_VOLUME, create_if_missing=True)
scannet_volume = modal.Volume.from_name(SCANNET_VOLUME, create_if_missing=True)
vanilla_output_volume = modal.Volume.from_name(
    VANILLA_SAM_OUTPUT_VOLUME, create_if_missing=True
)
c3g_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EVAL_OUTPUT_VOLUME, create_if_missing=True
)
c3g_dft_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_DFT_EVAL_OUTPUT_VOLUME, create_if_missing=True
)
c3g_nomaghead_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_NOMAGHEAD_EVAL_OUTPUT_VOLUME, create_if_missing=True
)
c3g_ema_nomag_eval_output_volume = modal.Volume.from_name(
    C3G_SAM_EMA_NOMAG_EVAL_OUTPUT_VOLUME, create_if_missing=True
)

viz_image = build_eval_sam_modal_image()

SEG_VIZ_VOLUMES = {
    str(REPLICA_MOUNT): replica_volume,
    str(SCANNET_MOUNT): scannet_volume,
    str(VANILLA_SAM_OUTPUT_MOUNT): vanilla_output_volume,
    str(C3G_SAM_EVAL_OUTPUT_MOUNT): c3g_eval_output_volume,
    str(C3G_SAM_DFT_EVAL_OUTPUT_MOUNT): c3g_dft_eval_output_volume,
    str(C3G_SAM_NOMAGHEAD_EVAL_OUTPUT_MOUNT): c3g_nomaghead_eval_output_volume,
    str(C3G_SAM_EMA_NOMAG_EVAL_OUTPUT_MOUNT): c3g_ema_nomag_eval_output_volume,
}


@dataclass(frozen=True, slots=True)
class FramePairSelection:
    dataset: DatasetName
    scene_id: str
    frame_a: str
    frame_b: str


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


def _resize_logits_to_label_shape(logits, label_shape: tuple[int, int]):
    import cv2

    if tuple(logits.shape[:2]) == label_shape:
        return logits
    height, width = label_shape
    return cv2.resize(
        logits.astype("float32", copy=False),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def _build_dense_pred_mask(
    frame_pred_dir: Path,
    label_shape: tuple[int, int],
    class_ids: list[int],
) -> tuple:
    import numpy as np

    dense = np.zeros(label_shape, dtype=np.int32)
    best_logit = np.full(label_shape, -np.inf, dtype=np.float32)
    missing_predictions = 0
    for class_id in class_ids:
        binary = _load_pred_mask(frame_pred_dir / f"{class_id}.png")
        logits_path = frame_pred_dir / f"{class_id}_logits.npy"
        if binary is None or not logits_path.is_file():
            missing_predictions += 1
            continue
        mask = _pred_mask_to_bool(binary)
        logits = np.load(logits_path).astype(np.float32)
        if tuple(logits.shape[:2]) != label_shape:
            logits = _resize_logits_to_label_shape(logits, label_shape)
        update = mask & (logits > best_logit)
        dense[update] = class_id
        best_logit[update] = logits[update]
    return dense, missing_predictions


def _load_dense_mask(
    frame_pred_dir: Path,
    label_path: Path,
    *,
    min_object_pixels: int,
):
    import numpy as np
    from PIL import Image

    gt_dense = np.array(Image.open(label_path))
    dense_path = frame_pred_dir / DENSE_MASK_FILENAME
    if dense_path.is_file():
        dense = np.array(Image.open(dense_path))
        if dense.shape[:2] != gt_dense.shape[:2]:
            raise ValueError(
                f"Dense mask shape {dense.shape[:2]} != label shape "
                f"{gt_dense.shape[:2]} at {dense_path}"
            )
        return dense.astype(np.int32)

    class_ids = expected_mask_class_ids(
        label_path, min_object_pixels=min_object_pixels
    )
    dense, _ = _build_dense_pred_mask(
        frame_pred_dir, gt_dense.shape[:2], class_ids
    )
    return dense


def _load_gt_dense(label_path: Path):
    import numpy as np
    from PIL import Image

    return np.array(Image.open(label_path)).astype(np.int32)


def _frame_has_predictions(
    frame_pred_dir: Path,
    label_path: Path,
    *,
    min_object_pixels: int,
) -> bool:
    if (frame_pred_dir / DENSE_MASK_FILENAME).is_file():
        return True
    if not frame_pred_dir.is_dir():
        return False
    class_ids = expected_mask_class_ids(
        label_path, min_object_pixels=min_object_pixels
    )
    if not class_ids:
        return True
    return all(
        (frame_pred_dir / f"{class_id}.png").is_file()
        and (frame_pred_dir / f"{class_id}_logits.npy").is_file()
        for class_id in class_ids
    )


def _frame_pair_candidates(
    *,
    dataset: DatasetName,
    dataset_root: Path,
    pred_roots: dict[ExperimentName, Path],
    scenes: list[str],
    min_object_pixels: int,
) -> list[FramePairSelection]:
    from src.misc.frame_layout import FramePaths, list_frame_ids

    candidates: list[FramePairSelection] = []
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
            if not (paths_a.label.is_file() and paths_b.label.is_file()):
                continue

            complete = True
            for pred_root in pred_roots.values():
                pred_scene_dir = pred_root / dataset / scene_id
                for frame_id, label_path in (
                    (frame_a, paths_a.label),
                    (frame_b, paths_b.label),
                ):
                    if not _frame_has_predictions(
                        pred_scene_dir / frame_id,
                        label_path,
                        min_object_pixels=min_object_pixels,
                    ):
                        complete = False
                        break
                if not complete:
                    break

            if complete:
                candidates.append(
                    FramePairSelection(
                        dataset=dataset,
                        scene_id=scene_id,
                        frame_a=frame_a,
                        frame_b=frame_b,
                    )
                )
    return candidates


def _pick_frame_pairs(
    *,
    replica_root: Path,
    scannet_root: Path,
    pred_roots: dict[ExperimentName, Path],
    seed: int,
    min_object_pixels: int,
) -> dict[DatasetName, FramePairSelection]:
    rng = random.Random(seed)
    selections: dict[DatasetName, FramePairSelection] = {}

    for dataset, dataset_root in (
        ("replica", replica_root),
        ("scannet", scannet_root),
    ):
        candidates = _frame_pair_candidates(
            dataset=dataset,  # type: ignore[arg-type]
            dataset_root=dataset_root,
            pred_roots=pred_roots,
            scenes=TEST_SCENES[dataset],  # type: ignore[index]
            min_object_pixels=min_object_pixels,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No scene with consecutive labeled frames and complete predictions "
                f"for all experiments under {dataset_root}."
            )
        selections[dataset] = rng.choice(candidates)  # type: ignore[index]

    return selections


def _collect_class_ids(
    *,
    selections: dict[DatasetName, FramePairSelection],
    replica_root: Path,
    scannet_root: Path,
    pred_roots: dict[ExperimentName, Path],
    min_object_pixels: int,
) -> set[int]:
    import numpy as np

    from src.misc.frame_layout import FramePaths

    class_ids: set[int] = set()
    dataset_roots = {"replica": replica_root, "scannet": scannet_root}

    for dataset, selection in selections.items():
        scene_dir = dataset_roots[dataset] / selection.scene_id
        for frame_id in (selection.frame_a, selection.frame_b):
            label_path = FramePaths.from_frame_id(scene_dir, frame_id).label
            gt_dense = _load_gt_dense(label_path)
            class_ids.update(
                int(class_id) for class_id in np.unique(gt_dense) if class_id
            )
            for pred_root in pred_roots.values():
                pred_dense = _load_dense_mask(
                    pred_root / dataset / selection.scene_id / frame_id,
                    label_path,
                    min_object_pixels=min_object_pixels,
                )
                class_ids.update(
                    int(class_id)
                    for class_id in pred_dense.flat
                    if class_id != 0
                )

    class_ids.discard(0)
    return class_ids


def _build_class_colormap(class_ids: set[int]) -> dict[int, tuple[int, int, int]]:
    from PIL import ImageColor

    colormap: dict[int, tuple[int, int, int]] = {0: (0, 0, 0)}
    for index, class_id in enumerate(sorted(class_ids)):
        hex_color = FOREGROUND_HEX_COLORS[index % len(FOREGROUND_HEX_COLORS)]
        colormap[class_id] = ImageColor.getcolor(hex_color, "RGB")
    return colormap


def _colorize_dense_mask(dense, colormap: dict[int, tuple[int, int, int]]):
    import numpy as np

    rgb = np.zeros((*dense.shape[:2], 3), dtype=np.uint8)
    for class_id, color in colormap.items():
        rgb[dense == class_id] = color
    return rgb


def _save_two_mask_figure(
    *,
    mask_a,
    mask_b,
    colormap: dict[int, tuple[int, int, int]],
    title: str,
    subtitles: tuple[str, str],
    output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    rgb_a = _colorize_dense_mask(mask_a, colormap)
    rgb_b = _colorize_dense_mask(mask_b, colormap)
    gap = 8
    header_height = 48
    panel_width = rgb_a.shape[1] + rgb_b.shape[1] + gap
    panel_height = max(rgb_a.shape[0], rgb_b.shape[0])
    canvas = Image.new(
        "RGB",
        (panel_width, header_height + panel_height),
        color=(0, 0, 0),
    )
    canvas.paste(Image.fromarray(rgb_a), (0, header_height))
    canvas.paste(Image.fromarray(rgb_b), (rgb_a.shape[1] + gap, header_height))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)
    draw.text((4, 22), f"{subtitles[0]}  |  {subtitles[1]}", fill=(200, 200, 200), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _dataset_roots(replica_root: str, scannet_root: str) -> dict[DatasetName, Path]:
    return {
        "replica": Path(resolve_dataset_root("replica", replica_root)),
        "scannet": Path(resolve_dataset_root("scannet", scannet_root)),
    }


def _resolve_pred_roots(
    overrides: dict[ExperimentName, str | None] | None = None,
) -> dict[ExperimentName, Path]:
    overrides = overrides or {}
    roots: dict[ExperimentName, Path] = {}
    for experiment, default_root in EXPERIMENT_PRED_ROOTS.items():
        override = overrides.get(experiment)
        roots[experiment] = Path(override) if override else default_root
    return roots


def generate_seg_viz_figures(
    *,
    replica_root: str | Path,
    scannet_root: str | Path,
    pred_roots: dict[ExperimentName, Path],
    output_dir: str | Path,
    seed: int = DEFAULT_SEED,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> dict:
    from src.misc.frame_layout import FramePaths

    replica_path = Path(replica_root)
    scannet_path = Path(scannet_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selections = _pick_frame_pairs(
        replica_root=replica_path,
        scannet_root=scannet_path,
        pred_roots=pred_roots,
        seed=seed,
        min_object_pixels=min_object_pixels,
    )
    colormap = _build_class_colormap(
        _collect_class_ids(
            selections=selections,
            replica_root=replica_path,
            scannet_root=scannet_path,
            pred_roots=pred_roots,
            min_object_pixels=min_object_pixels,
        )
    )

    written: list[str] = []
    dataset_roots = {"replica": replica_path, "scannet": scannet_path}

    for dataset, selection in selections.items():
        scene_dir = dataset_roots[dataset] / selection.scene_id
        paths_a = FramePaths.from_frame_id(scene_dir, selection.frame_a)
        paths_b = FramePaths.from_frame_id(scene_dir, selection.frame_b)
        gt_a = _load_gt_dense(paths_a.label)
        gt_b = _load_gt_dense(paths_b.label)
        subtitles = (
            f"{selection.frame_a}",
            f"{selection.frame_b}",
        )
        gt_path = out_dir / f"gt_{dataset}.png"
        _save_two_mask_figure(
            mask_a=gt_a,
            mask_b=gt_b,
            colormap=colormap,
            title=(
                f"GT — {dataset} / {selection.scene_id} "
                f"({selection.frame_a}, {selection.frame_b})"
            ),
            subtitles=subtitles,
            output_path=gt_path,
        )
        written.append(str(gt_path))

        for experiment, pred_root in pred_roots.items():
            pred_scene_dir = pred_root / dataset / selection.scene_id
            pred_a = _load_dense_mask(
                pred_scene_dir / selection.frame_a,
                paths_a.label,
                min_object_pixels=min_object_pixels,
            )
            pred_b = _load_dense_mask(
                pred_scene_dir / selection.frame_b,
                paths_b.label,
                min_object_pixels=min_object_pixels,
            )
            pred_path = out_dir / f"{experiment}_{dataset}.png"
            _save_two_mask_figure(
                mask_a=pred_a,
                mask_b=pred_b,
                colormap=colormap,
                title=(
                    f"{experiment} — {dataset} / {selection.scene_id} "
                    f"({selection.frame_a}, {selection.frame_b})"
                ),
                subtitles=subtitles,
                output_path=pred_path,
            )
            written.append(str(pred_path))

    payload = {
        "output_dir": str(out_dir),
        "seed": seed,
        "colormap_class_ids": sorted(k for k in colormap if k != 0),
        "selections": {
            dataset: {
                "scene_id": selection.scene_id,
                "frame_a": selection.frame_a,
                "frame_b": selection.frame_b,
            }
            for dataset, selection in selections.items()
        },
        "figures": written,
    }
    print(json_dumps(payload))
    return payload


def json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2) + "\n"


def _attach_figure_bytes(report: dict) -> dict:
    report["figure_bytes"] = {
        Path(figure_path).name: Path(figure_path).read_bytes()
        for figure_path in report["figures"]
    }
    return report


def save_figures_locally(report: dict, output_dir: Path | None = None) -> Path:
    """Write PNGs returned from a Modal run into a local directory."""
    out_dir = (output_dir or default_local_output_dir()).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    local_paths: list[str] = []
    for name, data in report.get("figure_bytes", {}).items():
        path = out_dir / name
        path.write_bytes(data)
        local_paths.append(str(path))

    manifest = {key: value for key, value in report.items() if key != "figure_bytes"}
    manifest["figures"] = local_paths
    manifest["output_dir"] = str(out_dir)
    (out_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
    print(f"Wrote {len(local_paths)} figures to {out_dir}")
    return out_dir


@app.function(
    image=viz_image,
    cpu=4,
    memory=8192,
    timeout=60 * 30,
    volumes=SEG_VIZ_VOLUMES,
    nonpreemptible=True,
)
def render_seg_viz(
    replica_root: str | None = None,
    scannet_root: str | None = None,
    output_dir: str | None = None,
    seed: int = DEFAULT_SEED,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
) -> dict:
    """Render segmentation comparison figures on Modal worker volumes."""
    del output_dir  # Figures are always pulled to the local machine after the run.
    pred_roots = _resolve_pred_roots()
    out_dir = REMOTE_SCRATCH_DIR

    for dataset_name, root in _dataset_roots(
        resolve_dataset_root("replica", replica_root),
        resolve_dataset_root("scannet", scannet_root),
    ).items():
        if not root.is_dir():
            raise FileNotFoundError(
                f"{dataset_name} dataset not found at {root}. "
                f"Populate the `{dataset_name}` volume."
            )

    for experiment, pred_root in pred_roots.items():
        if not pred_root.is_dir():
            raise FileNotFoundError(
                f"Predictions for {experiment!r} not found at {pred_root}."
            )

    report = generate_seg_viz_figures(
        replica_root=resolve_dataset_root("replica", replica_root),
        scannet_root=resolve_dataset_root("scannet", scannet_root),
        pred_roots=pred_roots,
        output_dir=out_dir,
        seed=seed,
        min_object_pixels=min_object_pixels,
    )
    return _attach_figure_bytes(report)


def _dispatch(fn, *, job_name: str, detach: bool, **kwargs):
    from src.misc.modal_run import dispatch_remote

    return dispatch_remote(
        fn,
        detach=detach,
        job_name=job_name,
        app_name=APP_NAME,
        **kwargs,
    )


@app.local_entrypoint()
def main(
    replica_root: str | None = None,
    scannet_root: str | None = None,
    output_dir: str | None = None,
    seed: int = DEFAULT_SEED,
    min_object_pixels: int = DEFAULT_MIN_OBJECT_PIXELS,
    detach: bool | None = None,
    wait: bool = False,
) -> None:
    """Render seg viz figures from Modal eval volumes."""
    detached = resolve_detach(detach=detach, remote_job=not wait)
    report = _dispatch(
        render_seg_viz,
        job_name="segmentation viz",
        detach=detached,
        replica_root=replica_root,
        scannet_root=scannet_root,
        output_dir=output_dir,
        seed=seed,
        min_object_pixels=min_object_pixels,
    )
    if detached:
        print(
            "Detached run: figures are not saved locally. "
            "Re-run with --wait to write into seg_results/."
        )
        return

    local_out = Path(output_dir) if output_dir else default_local_output_dir()
    save_figures_locally(report, local_out)


def _parse_local_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render colored dense-mask figures from eval outputs."
    )
    parser.add_argument("--replica-root", type=Path, default=None)
    parser.add_argument("--scannet-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_local_output_dir(),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-object-pixels",
        type=int,
        default=DEFAULT_MIN_OBJECT_PIXELS,
    )
    parser.add_argument("--sam-root", type=Path, default=None)
    parser.add_argument("--c3gsam-root", type=Path, default=None)
    parser.add_argument("--c3gsam-dft-root", type=Path, default=None)
    parser.add_argument("--c3gsam-nomaghead-root", type=Path, default=None)
    parser.add_argument("--c3gsam-ema-nomag-root", type=Path, default=None)
    return parser.parse_args(argv)


def main_local(argv: list[str] | None = None) -> None:
    args = _parse_local_args(argv)
    pred_roots = _resolve_pred_roots(
        {
            "sam": args.sam_root,
            "c3gsam": args.c3gsam_root,
            "c3gsam-dft": args.c3gsam_dft_root,
            "c3gsam-nomaghead": args.c3gsam_nomaghead_root,
            "c3gsam-ema-nomag": args.c3gsam_ema_nomag_root,
        }
    )
    generate_seg_viz_figures(
        replica_root=args.replica_root or REPLICA_MOUNT,
        scannet_root=args.scannet_root or SCANNET_MOUNT,
        pred_roots=pred_roots,
        output_dir=args.output_dir,
        seed=args.seed,
        min_object_pixels=args.min_object_pixels,
    )


if __name__ == "__main__":
    if "modal" in sys.modules and hasattr(modal, "is_local") and not modal.is_local():
        raise SystemExit(
            "Run with `modal run src/modal_infra/seg_viz.py`, not as a worker entrypoint."
        )
    main_local()
