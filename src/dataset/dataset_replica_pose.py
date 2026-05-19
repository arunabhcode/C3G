"""Replica relative-pose evaluation loader (``pose/`` layout)."""

from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
import torchvision.transforms as tf
from einops import repeat
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from .dataset import DatasetCfgCommon
from .dataset_replica import imread_cv2
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

REPLICA_LABELS: dict[str, list[str]] = {
    "office3": ["wall", "ceiling", "floor", "chair", "table"],
    "office4": ["wall", "ceiling", "floor", "chair", "tv-screen", "table"],
    "room1": ["wall", "ceiling", "floor", "bed", "blinds"],
}

REPLICA_LABEL_MAPPING: dict[str, dict[int, int]] = {
    "office3": {
        0: 0,
        8: 3,
        10: 0,
        12: 1,
        14: 0,
        15: 0,
        17: 0,
        20: 4,
        22: 0,
        29: 4,
        31: 2,
        35: 5,
        37: 1,
        40: 3,
        47: 2,
        56: 0,
        62: 0,
        76: 4,
        79: 0,
        80: 5,
        82: 0,
        83: 0,
        88: 0,
        92: 0,
        93: 1,
        95: 0,
        97: 1,
    },
    "office4": {
        0: 0,
        8: 3,
        10: 0,
        17: 0,
        20: 4,
        22: 0,
        31: 2,
        37: 0,
        40: 3,
        47: 2,
        56: 5,
        80: 6,
        87: 5,
        92: 0,
        93: 1,
        95: 0,
        97: 1,
    },
    "room1": {
        0: 0,
        3: 0,
        7: 4,
        11: 4,
        12: 5,
        13: 0,
        18: 0,
        26: 0,
        31: 2,
        37: 1,
        40: 3,
        44: 0,
        47: 0,
        54: 0,
        56: 0,
        59: 1,
        61: 4,
        64: 0,
        79: 0,
        91: 0,
        92: 0,
        93: 1,
        95: 0,
        97: 1,
    },
}

SCENE_FROM_ID = {0: "office3", 1: "office4", 2: "room1"}


@dataclass
class DatasetReplicaPoseCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool


@dataclass
class DatasetReplicaPoseCfgWrapper:
    replica_pose: DatasetReplicaPoseCfg


class DatasetReplicaPose(IterableDataset):
    cfg: DatasetReplicaPoseCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetReplicaPoseCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.data_root = cfg.roots[0]
        pair_file = os.path.join(cfg.roots[0], "pose", "test.npz")
        data_pairs = np.load(pair_file)
        self.pairs = data_pairs["name"]
        self.rel_pose = data_pairs["rel_pose"]

    def map_labels(self, scene_id: str, label_map: np.ndarray) -> np.ndarray:
        mapping = REPLICA_LABEL_MAPPING[scene_id]
        remapped = label_map.copy()
        for raw_id, mapped_id in mapping.items():
            remapped[label_map == raw_id] = mapped_id
        for unique_label in np.unique(label_map):
            if unique_label not in mapping:
                remapped[label_map == unique_label] = 0
        return remapped

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        pairs = self.pairs
        rel_poses = self.rel_pose
        if self.stage == "test" and worker_info is not None:
            pairs = [
                pair
                for pair_index, pair in enumerate(pairs)
                if pair_index % worker_info.num_workers == worker_info.id
            ]
            rel_poses = [
                pose
                for pose_index, pose in enumerate(rel_poses)
                if pose_index % worker_info.num_workers == worker_info.id
            ]

        for scene, rel_pose in zip(pairs, rel_poses):
            scene_name = SCENE_FROM_ID[int(scene[0])]
            frame_a = int(scene[2])
            frame_b = int(scene[3])
            scene_root = os.path.join(self.data_root, "pose", scene_name)

            im_a_path = os.path.join(scene_root, "color", f"{frame_a}.jpg")
            im_b_path = os.path.join(scene_root, "color", f"{frame_b}.jpg")
            label_a_path = os.path.join(scene_root, "labels", f"{frame_a}.png")
            label_b_path = os.path.join(scene_root, "labels", f"{frame_b}.png")

            context_images = self.convert_images([im_a_path, im_b_path])
            context_labels = self.convert_labels(
                scene_name, [label_a_path, label_b_path]
            )

            h, w = context_images.shape[-2:]

            K = np.stack(
                [
                    np.array([float(i) for i in r.split()])
                    for r in open(
                        osp.join(scene_root, "intrinsic", "intrinsic_color.txt"),
                        "r",
                    )
                    .read()
                    .split("\n")
                    if r
                ]
            )

            def center_principal_point(image, labels, cx, cy, h, w):
                cx = round(cx)
                cy = round(cy)
                center_x, center_y = w // 2, h // 2
                shift_x = center_x - cx
                shift_y = center_y - cy
                new_w = round(max(w, w - 2 * shift_x))
                new_h = round(max(h, h - 2 * shift_y))

                new_image = torch.zeros((2, 3, new_h, new_w), dtype=torch.float32)
                new_labels = torch.zeros((2, 1, new_h, new_w), dtype=torch.long)

                pad_left = max(0, -shift_x)
                pad_top = max(0, -shift_y)
                src_left = max(0, shift_x)
                src_top = max(0, shift_y)
                src_right = min(w, w + shift_x)
                src_bottom = min(h, h + shift_y)

                new_image[
                    :,
                    :,
                    pad_top : pad_top + src_bottom - src_top,
                    pad_left : pad_left + src_right - src_left,
                ] = image[:, :, src_top:src_bottom, src_left:src_right]
                new_labels[
                    :,
                    :,
                    pad_top : pad_top + src_bottom - src_top,
                    pad_left : pad_left + src_right - src_left,
                ] = labels[
                    :, :, src_top:src_bottom, src_left:src_right
                ]

                return new_image, new_labels, new_w // 2, new_h // 2

            context_images, context_labels, tgt_cx, tgt_cy = center_principal_point(
                context_images, context_labels, K[0, 2], K[1, 2], h, w
            )
            K[0, 2] = tgt_cx
            K[1, 2] = tgt_cy

            h, w = context_images.shape[-2:]
            target_images = context_images.clone()
            target_labels = context_labels.clone()

            pose1 = torch.eye(4)
            pose2 = torch.eye(4)
            pose2[:3, :4] = torch.tensor(rel_pose.reshape(3, 4)).to(torch.float32)
            pose2 = torch.inverse(pose2)
            extrinsics = torch.stack((pose1, pose2), dim=0)

            K = K[:3, :3]
            K[0, :3] /= w
            K[1, :3] /= h

            intrinsics = (
                torch.tensor(K, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
            )

            overlap = torch.tensor([0.5], dtype=torch.float32)
            scale = torch.tensor([1.0], dtype=torch.float32)
            context_indices = torch.tensor([frame_a, frame_b], dtype=torch.int64)

            example = {
                "context": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "label": context_labels,
                    "near": self.get_bound("near", 2),
                    "far": self.get_bound("far", 2),
                    "index": context_indices,
                    "overlap": overlap,
                    "scale": scale,
                    "text": REPLICA_LABELS[scene_name],
                },
                "target": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": target_images,
                    "label": target_labels,
                    "near": self.get_bound("near", 2),
                    "far": self.get_bound("far", 2),
                    "index": context_indices,
                    "text": REPLICA_LABELS[scene_name],
                },
                "scene": scene_name,
            }
            yield apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    def convert_images(
        self,
        images: list[str],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            torch_images.append(self.to_tensor(Image.open(image)))
        return torch.stack(torch_images)

    def convert_labels(
        self,
        scene_id: str,
        label_paths: list[str],
    ) -> Float[Tensor, "batch 1 height width"]:
        label_tensors = []
        for label_path in label_paths:
            label_map = imread_cv2(label_path, options=cv2.IMREAD_UNCHANGED)
            label_map = self.map_labels(scene_id, label_map)
            label_tensors.append(
                torch.from_numpy(label_map).long().unsqueeze(0)
            )
        return torch.stack(label_tensors, dim=0)

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
