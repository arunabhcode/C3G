"""ScanNet 2D semantic segmentation loader (flat per-scene layout).

Expects data prepared by :mod:`download_scannet`::

    <root>/<scene_id>/{frame_id}_x.jpg
    <root>/<scene_id>/{frame_id}_cam.npz
    <root>/<scene_id>/{frame_id}_y.png

See :mod:`src.misc.frame_layout` for naming conventions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..misc.cam_utils import camera_normalization
from .cropping import (
    bbox_from_intrinsics_in_out,
    camera_matrix_of_crop,
    crop_image_depthmap,
    rescale_image_depthmap,
)
from .dataset import DatasetCfgCommon
from ..misc.frame_layout import FramePaths
from .types import Stage
from .view_sampler import ViewSampler

SCANNET_TARGET_CLASSES = [
    "wall",
    "floor",
    "ceiling",
    "chair",
    "table",
    "sofa",
    "bed",
    "other",
]


def build_label_mapper(label_map_path: str | os.PathLike[str]):
    labels = [label.lower() for label in SCANNET_TARGET_CLASSES]
    df = pd.read_csv(label_map_path, sep="\t")
    id_to_nyu40class = pd.Series(
        df["nyu40class"].str.lower().values, index=df["id"]
    ).to_dict()
    nyu40class_to_newid = {
        cls: labels.index(cls) + 1 if cls in labels else labels.index("other") + 1
        for cls in set(id_to_nyu40class.values())
    }
    id_to_newid = {
        id_: nyu40class_to_newid[cls] for id_, cls in id_to_nyu40class.items()
    }
    return np.vectorize(
        lambda value: id_to_newid.get(value, labels.index("other") + 1)
        if value != 0
        else 0
    )


def imread_cv2(path: str | os.PathLike[str], options=cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(str(path), options)
    if img is None:
        raise OSError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


@dataclass
class Scannet2dSegCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool
    llff_hold: int = 8
    test_ids: list[int] = field(default_factory=lambda: [1, 4])
    num_of_inputs: int = 2
    context_eval: bool = False


@dataclass
class DatasetScannet2dSegCfgWrapper:
    scannet_2dseg: Scannet2dSegCfg


class DatasetScannet2dSeg(IterableDataset):
    cfg: Scannet2dSegCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.01
    far: float = 100.0

    def __init__(
        self,
        cfg: Scannet2dSegCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        label_map_path = Path(cfg.roots[0]) / "scannetv2-labels.combined.tsv"
        self.map_labels = build_label_mapper(label_map_path)

        with (Path(cfg.roots[0]) / "selected_seqs_test.json").open("r") as file_handle:
            self.scenes = {
                scene: sorted(frame_ids)
                for scene, frame_ids in json.load(file_handle).items()
                if frame_ids
            }
        self.scene_list = list(self.scenes.keys())

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, info=None):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        width, height = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, width - cx)
        min_margin_y = min(cy, height - cy)
        left, top = cx - min_margin_x, cy - min_margin_y
        right, bottom = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (left, top, right, bottom)
        image, depthmap, intrinsics = crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        width, height = image.size
        assert resolution[0] >= resolution[1]
        if height > 1.1 * width:
            resolution = resolution[::-1]

        target_resolution = np.array(resolution)
        image, depthmap, intrinsics = rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        intrinsics2 = camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2 = crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )
        return image, depthmap, intrinsics2

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        scene_list = self.scene_list
        if self.stage == "test" and worker_info is not None:
            scene_list = [
                scene
                for scene_index, scene in enumerate(scene_list)
                if scene_index % worker_info.num_workers == worker_info.id
            ]

        for scene_id in scene_list:
            frame_ids = self.scenes[scene_id]
            selected_views = [
                view_index
                for view_index in range(len(frame_ids))
                if view_index % self.cfg.llff_hold in self.cfg.test_ids
            ]

            for target_view in selected_views:
                left_idxs = [
                    max(target_view - offset, 0)
                    for offset in range(1, (self.cfg.num_of_inputs + 2) // 2)
                ]
                right_idxs = [
                    min(target_view + offset, len(frame_ids) - 1)
                    for offset in range(1, (self.cfg.num_of_inputs + 2) // 2)
                ]
                idxs: list[int] = []
                for left, right in zip(left_idxs, right_idxs):
                    idxs.extend([left, right])
                idxs.append(target_view)

                scene_dir = Path(self.cfg.roots[0]) / scene_id
                extrinsics_list: list[np.ndarray] = []
                intrinsics_list: list[np.ndarray] = []
                images_list: list[Tensor] = []
                label_list: list[Tensor] = []

                for view_index in idxs:
                    paths = FramePaths.from_frame_id(scene_dir, frame_ids[view_index])
                    metadata = np.load(paths.camera)
                    camera_pose = metadata["camera_pose"].astype(np.float32)
                    if np.any(np.isinf(camera_pose)):
                        continue

                    intrinsics = metadata["camera_intrinsics"].astype(np.float32)
                    rgb_image = imread_cv2(paths.image)
                    labelmap = imread_cv2(paths.label, options=cv2.IMREAD_UNCHANGED)
                    labelmap = self.map_labels(labelmap)

                    depthmap = np.ones(labelmap.shape[:2], dtype=np.uint16)
                    maskmap = np.ones_like(depthmap) * 255
                    depth_mask_label = np.stack([depthmap, maskmap, labelmap], axis=-1)

                    rgb_image, depth_mask_label, intrinsics = self._crop_resize_if_necessary(
                        rgb_image,
                        depth_mask_label,
                        intrinsics,
                        self.cfg.input_image_shape,
                        info=str(paths.image),
                    )

                    intrinsics[0, :] /= self.cfg.input_image_shape[0]
                    intrinsics[1, :] /= self.cfg.input_image_shape[1]
                    labelmap = depth_mask_label[:, :, 2]

                    extrinsics_list.append(camera_pose)
                    intrinsics_list.append(intrinsics)
                    images_list.append(self.to_tensor(rgb_image))
                    label_list.append(
                        torch.from_numpy(labelmap.astype(np.int64)).unsqueeze(0)
                    )

                if len(extrinsics_list) < len(idxs):
                    continue

                extrinsics = torch.from_numpy(
                    np.stack(extrinsics_list, axis=0).astype(np.float32)
                )
                intrinsics = torch.from_numpy(
                    np.stack(intrinsics_list, axis=0).astype(np.float32)
                )
                images = torch.stack(images_list, dim=0)
                labels = torch.cat(label_list, dim=0)

                context_extrinsics = extrinsics[: self.cfg.num_of_inputs]
                if self.cfg.make_baseline_1:
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1.0

                if self.cfg.relative_pose:
                    extrinsics = camera_normalization(extrinsics[0:1], extrinsics)

                class_names = SCANNET_TARGET_CLASSES
                if self.cfg.context_eval:
                    for view_offset in range(self.cfg.num_of_inputs):
                        yield {
                            "context": {
                                "extrinsics": extrinsics[: self.cfg.num_of_inputs],
                                "intrinsics": intrinsics[: self.cfg.num_of_inputs],
                                "image": images[: self.cfg.num_of_inputs],
                                "label": labels[: self.cfg.num_of_inputs],
                                "near": self.get_bound("near", self.cfg.num_of_inputs)
                                / scale,
                                "far": self.get_bound("far", self.cfg.num_of_inputs)
                                / scale,
                                "index": torch.tensor(
                                    idxs[: self.cfg.num_of_inputs], dtype=torch.int64
                                ),
                                "overlap": 0,
                                "text": class_names,
                            },
                            "target": {
                                "extrinsics": extrinsics[view_offset : view_offset + 1],
                                "intrinsics": intrinsics[view_offset : view_offset + 1],
                                "image": images[view_offset : view_offset + 1],
                                "label": labels[view_offset : view_offset + 1],
                                "near": self.get_bound("near", 1) / scale,
                                "far": self.get_bound("far", 1) / scale,
                                "index": torch.tensor(
                                    idxs[self.cfg.num_of_inputs :], dtype=torch.int64
                                ),
                                "text": class_names,
                            },
                            "scene": scene_id,
                        }
                else:
                    yield {
                        "context": {
                            "extrinsics": extrinsics[: self.cfg.num_of_inputs],
                            "intrinsics": intrinsics[: self.cfg.num_of_inputs],
                            "image": images[: self.cfg.num_of_inputs],
                            "label": labels[: self.cfg.num_of_inputs],
                            "near": self.get_bound("near", self.cfg.num_of_inputs)
                            / scale,
                            "far": self.get_bound("far", self.cfg.num_of_inputs) / scale,
                            "index": torch.tensor(
                                idxs[: self.cfg.num_of_inputs], dtype=torch.int64
                            ),
                            "overlap": 0,
                            "text": class_names,
                        },
                        "target": {
                            "extrinsics": extrinsics[self.cfg.num_of_inputs :],
                            "intrinsics": intrinsics[self.cfg.num_of_inputs :],
                            "image": images[self.cfg.num_of_inputs :],
                            "label": labels[self.cfg.num_of_inputs :],
                            "near": self.get_bound(
                                "near", len(idxs) - self.cfg.num_of_inputs
                            )
                            / scale,
                            "far": self.get_bound(
                                "far", len(idxs) - self.cfg.num_of_inputs
                            )
                            / scale,
                            "index": torch.tensor(
                                idxs[self.cfg.num_of_inputs :], dtype=torch.int64
                            ),
                            "text": class_names,
                        },
                        "scene": scene_id,
                    }

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
