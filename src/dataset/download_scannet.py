#!/usr/bin/env python3
"""Download and prepare ScanNet scenes for :mod:`dataset_scannet_2dseg`.

Writes one directory per scene under the output root::

    <out>/<scene_id>/{frame_id}_x.jpg
    <out>/<scene_id>/{frame_id}_cam.npz
    <out>/<scene_id>/{frame_id}_y.png
    <out>/<scene_id>/{frame_id}_depth.png

Also writes ``scannetv2-labels.combined.tsv`` and ``selected_seqs_test.json``.

Run on Modal::

    modal run src/dataset/download_scannet.py

Run detached::

    modal run --detach src/dataset/download_scannet.py --accept-tos

Run locally::

    python -m src.dataset.download_scannet --out-dir ./datasets/scannet --accept-tos
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import ssl
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

BASE_URL = "http://kaldir.vc.cit.tum.de/scannet/"
TOS_URL = BASE_URL + "ScanNet_TOS.pdf"
RELEASE = "v2/scans"
RELEASE_TASKS = "v2/tasks"
LABEL_MAP_FILE = "scannetv2-labels.combined.tsv"

# 15 scenes with 2D semantic labels (scene0697_00 … scene0711_00).
SCENES: tuple[str, ...] = tuple(f"scene{i:04d}_00" for i in range(697, 712))

FRAME_SKIP = 20
IMAGE_SIZE = (480, 640)  # (height, width)
FRAME_ID_WIDTH = 5
VOLUME_NAME = "scannet"
VOLUME_MOUNT = Path("/scannet")
RAW_SUBDIR = "_raw"
MODAL_RAW_SCRATCH = Path("/tmp/scannet_raw")

# Train scenes reuse v1 .sens streams; later scenes use v2.
V1_SENS_SCENES = frozenset(f"scene{i:04d}_00" for i in range(697, 707))

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {-1: "unknown", 0: "raw_ushort", 1: "zlib_ushort", 2: "occi_ushort"}


def format_frame_id(frame_index: int) -> str:
    return f"{frame_index:0{FRAME_ID_WIDTH}d}"


# ---------------------------------------------------------------------------
# ScanNet .sens reader (Python 3 port of ScanNet/SensReader/python/SensorData.py)
# ---------------------------------------------------------------------------


class RGBDFrame:
    def load(self, file_handle) -> None:
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type: str) -> bytes:
        if compression_type == "zlib_ushort":
            return zlib.decompress(self.depth_data)
        raise ValueError(f"Unsupported depth compression: {compression_type}")

    def decompress_color(self, compression_type: str) -> np.ndarray:
        if compression_type == "jpeg":
            return imageio.imread(self.color_data)
        raise ValueError(f"Unsupported color compression: {compression_type}")


class SensorData:
    def __init__(self, filename: str | os.PathLike[str]) -> None:
        self.version = 4
        self.load(filename)

    def load(self, filename: str | os.PathLike[str]) -> None:
        with open(filename, "rb") as file_handle:
            version = struct.unpack("I", file_handle.read(4))[0]
            if version != self.version:
                raise ValueError(f"Unsupported .sens version {version}")
            strlen = struct.unpack("Q", file_handle.read(8))[0]
            file_handle.read(strlen)
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            file_handle.read(16 * 4)  # extrinsic_color
            file_handle.read(16 * 4)  # intrinsic_depth
            file_handle.read(16 * 4)  # extrinsic_depth
            color_compression = struct.unpack("i", file_handle.read(4))[0]
            depth_compression = struct.unpack("i", file_handle.read(4))[0]
            self.color_compression_type = COMPRESSION_TYPE_COLOR[color_compression]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[depth_compression]
            self.color_width = struct.unpack("I", file_handle.read(4))[0]
            self.color_height = struct.unpack("I", file_handle.read(4))[0]
            self.depth_width = struct.unpack("I", file_handle.read(4))[0]
            self.depth_height = struct.unpack("I", file_handle.read(4))[0]
            file_handle.read(4)  # depth_shift (unused for export)
            num_frames = struct.unpack("Q", file_handle.read(8))[0]
            self.frames: list[RGBDFrame] = []
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(file_handle)
                self.frames.append(frame)

    def export_frames_strided(
        self,
        output_dir: str | os.PathLike[str],
        *,
        frame_skip: int = 1,
        image_size: tuple[int, int] | None = None,
    ) -> list[int]:
        output_dir = Path(output_dir)
        color_dir = output_dir / "color"
        pose_dir = output_dir / "pose"
        color_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        frame_indices = list(range(0, len(self.frames), frame_skip))
        for frame_idx in frame_indices:
            frame = self.frames[frame_idx]
            color = frame.decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_AREA,
                )
            imageio.imwrite(color_dir / f"{frame_idx}.jpg", color)
            np.savetxt(pose_dir / f"{frame_idx}.txt", frame.camera_to_world)
        return frame_indices


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_file(url: str, out_file: str | os.PathLike[str]) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.is_file():
        print(f"Skipping existing file {out_file}")
        return
    print(f"Downloading {url} -> {out_file}")
    with tempfile.NamedTemporaryFile(dir=out_file.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(out_file)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url} -> {out_file}: {exc}") from None


def download_scan_file(
    scan_id: str,
    suffix: str,
    out_dir: str | os.PathLike[str],
    *,
    use_v1_sens: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_release = "v1/scans" if use_v1_sens and suffix == ".sens" else RELEASE
    url = f"{BASE_URL}{scan_release}/{scan_id}/{scan_id}{suffix}"
    out_file = out_dir / f"{scan_id}{suffix}"
    download_file(url, out_file)
    return out_file


def download_label_map(out_dir: str | os.PathLike[str]) -> Path:
    url = f"{BASE_URL}{RELEASE_TASKS}/{LABEL_MAP_FILE}"
    out_path = Path(out_dir) / LABEL_MAP_FILE
    download_file(url, out_path)
    return out_path


def read_label_mapping(label_map_file: str | os.PathLike[str]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    with open(label_map_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t")
        for row in reader:
            mapping[int(row["id"])] = int(row["nyu40id"])
    return mapping


def map_label_image(image: np.ndarray, label_mapping: dict[int, int]) -> np.ndarray:
    mapped = np.copy(image)
    for raw_id, nyu40_id in label_mapping.items():
        mapped[image == raw_id] = nyu40_id
    return mapped.astype(np.uint16)


def adjust_intrinsic(
    intrinsic: np.ndarray,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    orig_h, orig_w = original_size
    out_h, out_w = output_size
    scaled = intrinsic.copy()
    scaled[0, 0] *= out_w / orig_w
    scaled[1, 1] *= out_h / orig_h
    scaled[0, 2] *= out_w / orig_w
    scaled[1, 2] *= out_h / orig_h
    return scaled


def extract_label_zip(zip_path: Path, scene_dir: Path) -> Path:
    label_dir = scene_dir / "label-filt"
    if label_dir.is_dir() and any(label_dir.glob("*.png")):
        return label_dir
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(scene_dir)
    if not label_dir.is_dir():
        for candidate in scene_dir.iterdir():
            if candidate.is_dir() and "label" in candidate.name.lower():
                label_dir = candidate
                break
    if not label_dir.is_dir():
        raise FileNotFoundError(f"No label-filt directory found under {scene_dir}")
    return label_dir


def resize_depth(depth: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        depth,
        (image_size[1], image_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def decompress_frame_depth(
    sensor: SensorData,
    frame_idx: int,
    *,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    frame = sensor.frames[frame_idx]
    depth_bytes = frame.decompress_depth(sensor.depth_compression_type)
    depth = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(
        sensor.depth_height, sensor.depth_width
    )
    if image_size is not None:
        depth = resize_depth(depth, image_size)
    return depth


def resize_label(
    label_path: Path,
    label_mapping: dict[int, int],
    image_size: tuple[int, int],
) -> np.ndarray:
    image = np.array(imageio.imread(label_path))
    image = cv2.resize(
        image,
        (image_size[1], image_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return map_label_image(image, label_mapping)


def build_scene(
    scan_id: str,
    raw_dir: Path,
    out_root: Path,
    label_map_file: Path,
    *,
    frame_skip: int = FRAME_SKIP,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> list[str]:
    scene_raw = raw_dir / scan_id
    sens_path = scene_raw / f"{scan_id}.sens"
    label_zip = scene_raw / f"{scan_id}_2d-label-filt.zip"
    use_v1_sens = scan_id in V1_SENS_SCENES

    if not sens_path.is_file():
        download_scan_file(scan_id, ".sens", scene_raw, use_v1_sens=use_v1_sens)
    if not label_zip.is_file():
        download_scan_file(scan_id, "_2d-label-filt.zip", scene_raw)

    work_dir = scene_raw / "extracted"
    sensor = SensorData(sens_path)
    frame_indices = sensor.export_frames_strided(
        work_dir,
        frame_skip=frame_skip,
        image_size=image_size,
    )
    label_filt_dir = extract_label_zip(label_zip, scene_raw)
    label_mapping = read_label_mapping(label_map_file)
    original_size = (sensor.color_height, sensor.color_width)
    intrinsic = adjust_intrinsic(
        sensor.intrinsic_color[:3, :3].astype(np.float32),
        original_size,
        image_size,
    )

    scene_out = out_root / scan_id
    scene_out.mkdir(parents=True, exist_ok=True)

    frame_names: list[str] = []
    for frame_idx in frame_indices:
        label_src = label_filt_dir / f"{frame_idx}.png"
        if not label_src.is_file():
            print(f"Warning: missing label for {scan_id} frame {frame_idx}, skipping")
            continue

        frame_name = format_frame_id(frame_idx)
        frame_names.append(frame_name)
        shutil.copy2(
            work_dir / "color" / f"{frame_idx}.jpg",
            scene_out / f"{frame_name}_x.jpg",
        )
        label = resize_label(label_src, label_mapping, image_size)
        imageio.imwrite(scene_out / f"{frame_name}_y.png", label)

        depth = decompress_frame_depth(sensor, frame_idx, image_size=image_size)
        imageio.imwrite(scene_out / f"{frame_name}_depth.png", depth)

        pose = np.loadtxt(work_dir / "pose" / f"{frame_idx}.txt", dtype=np.float32)
        np.savez(
            scene_out / f"{frame_name}_cam.npz",
            camera_pose=pose,
            camera_intrinsics=intrinsic,
        )

    return frame_names


def prepare_scannet(
    out_root: str | os.PathLike[str],
    *,
    scenes: tuple[str, ...] = SCENES,
    accept_tos: bool = False,
    raw_dir: str | os.PathLike[str] | None = None,
) -> None:
    if not accept_tos:
        print(
            "By continuing you confirm agreement to the ScanNet terms of use:\n"
            f"  {TOS_URL}\n"
            "Re-run with --accept-tos to proceed."
        )
        sys.exit(1)

    out_root = Path(out_root)
    raw_dir = Path(raw_dir) if raw_dir is not None else out_root / RAW_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    label_map_path = download_label_map(out_root)

    selected_seqs: dict[str, list[str]] = {}
    for scan_id in scenes:
        print(f"Preparing {scan_id} ...")
        frame_names = build_scene(scan_id, raw_dir, out_root, label_map_path)
        if frame_names:
            selected_seqs[scan_id] = frame_names
        print(f"  {len(frame_names)} frames")

    with open(out_root / "selected_seqs_test.json", "w") as file_handle:
        json.dump(selected_seqs, file_handle, indent=2)

    print(f"Done. {len(selected_seqs)} scenes under {out_root}")
    print(f"Point dataset configs at: dataset.scannet_2dseg.roots=[{out_root}]")


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------

try:
    import modal

    app = modal.App("c3g-scannet-download")
    scannet_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install(
            "numpy==1.26.4",
            "opencv-python-headless==4.10.0.84",
            "imageio==2.37.0",
        )
    )

    @app.function(
        image=image,
        volumes={str(VOLUME_MOUNT): scannet_volume},
        timeout=60 * 60 * 12,
        cpu=4,
        memory=32768,
    )
    def populate_scannet_volume() -> None:
        # Keep bulky .sens / label zips off the Modal volume (only prepared frames are committed).
        if MODAL_RAW_SCRATCH.exists():
            shutil.rmtree(MODAL_RAW_SCRATCH)
        MODAL_RAW_SCRATCH.mkdir(parents=True, exist_ok=True)
        try:
            prepare_scannet(
                VOLUME_MOUNT,
                accept_tos=True,
                raw_dir=MODAL_RAW_SCRATCH,
            )
        finally:
            shutil.rmtree(MODAL_RAW_SCRATCH, ignore_errors=True)

        raw_on_volume = VOLUME_MOUNT / RAW_SUBDIR
        if raw_on_volume.exists():
            shutil.rmtree(raw_on_volume)

        scannet_volume.commit()

    @app.local_entrypoint()
    def modal_main(accept_tos: bool = False, detach: bool = False) -> None:
        from src.misc.modal_run import dispatch_remote

        if not accept_tos:
            print(
                "By continuing you confirm agreement to the ScanNet terms of use:\n"
                f"  {TOS_URL}\n"
                "Re-run with --accept-tos to proceed."
            )
            sys.exit(1)
        dispatch_remote(
            populate_scannet_volume,
            detach=detach,
            job_name="ScanNet volume populate",
            app_name=app.name,
        )

except ImportError:
    app = None  # type: ignore[assignment]
    modal_main = None  # type: ignore[assignment,misc]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare ScanNet scenes for 2D segmentation."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Local output directory.",
    )
    parser.add_argument(
        "--accept-tos",
        action="store_true",
        help="Confirm ScanNet terms of use and start downloading.",
    )
    parser.add_argument(
        "--modal",
        action="store_true",
        help="Populate the Modal volume via modal run (requires modal package).",
    )
    args = parser.parse_args()

    if args.modal:
        if app is None:
            print("Install modal (`pip install modal`) to use --modal.", file=sys.stderr)
            sys.exit(1)
        modal_main(accept_tos=args.accept_tos)  # type: ignore[misc]
        return

    out_dir = args.out_dir or Path("datasets") / VOLUME_NAME
    prepare_scannet(out_dir, accept_tos=args.accept_tos)


if __name__ == "__main__":
    main()
