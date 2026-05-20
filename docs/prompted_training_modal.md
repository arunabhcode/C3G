# Prompted Training with SAM on Replica and ScanNet (Modal)

Train C3G-F with point-prompted SAM segmentation loss on [Modal](https://modal.com/) using the prepared **`replica`** and **`scannet`** volumes. Data layout matches `src/misc/frame_layout.py` and the download scripts `src/dataset/download_replica.py` and `src/dataset/download_scannet.py`.

## Prerequisites

### 1. Modal CLI

```bash
pip install modal
modal setup
```

### 2. Pretrained weights (`c3g-weights` volume)

Upload checkpoints to the `c3g-weights` volume (mounted at `/weights` in training jobs). Same layout as `src/inference/modal_train_c3g_sam.py`.

```bash
modal volume put c3g-weights /path/to/sam_vit_h.pth sam_vit_h.pth
modal volume put c3g-weights /path/to/model.pt model.pt
# Optional: faster convergence than VGGT backbone alone
modal volume put c3g-weights /path/to/gaussian_decoder.ckpt gaussian_decoder.ckpt
```

| File | Purpose |
|------|---------|
| `sam_vit_h.pth` | SAM ViT-H mask decoder (`train.sam_checkpoint`) |
| `model.pt` | VGGT-1B encoder init (`model.encoder.pretrained_weights`) |
| `gaussian_decoder.ckpt` | Optional Gaussian decoder init |

## Populate dataset volumes

Both datasets share the same **flat** on-disk layout: one directory per scene, frames named `{frame_id}_x.jpg`, `{frame_id}_y.png`, `{frame_id}_cam.npz`. Download scripts may also write `selected_seqs_test.json` at the volume root (optional; loaders discover scenes via `scenes` in config). ScanNet also includes `scannetv2-labels.combined.tsv`.

### Replica (`replica` volume → `/replica`)

Eight scenes: `office0`–`office4`, `room0`–`room2`. Frames are strided by 20 from the source trajectories.

```bash
modal run src/dataset/download_replica.py
modal run --detach src/dataset/download_replica.py
```

Local preparation (writes `./datasets/replica` by default):

```bash
python -m src.dataset.download_replica --source /path/to/raw/replica --out-dir ./datasets/replica
```

Point training configs at `dataset.replica_2dseg.roots=[/replica]` (or the mount path you use).

### ScanNet (`scannet` volume → `/scannet`)

Fifteen labeled test scenes: `scene0697_00` … `scene0711_00`. Requires accepting the ScanNet terms of use.

```bash
modal run src/dataset/download_scannet.py --accept-tos
modal run --detach src/dataset/download_scannet.py --accept-tos
```

Local preparation:

```bash
python -m src.dataset.download_scannet --out-dir ./datasets/scannet --accept-tos
```

Point training configs at `dataset.scannet_2dseg.roots=[/scannet]`.

Use `src/misc/modal_run.py` (`--detach`) for long download jobs; raw `.sens` archives are downloaded to ephemeral scratch on Modal, not stored on the volume.

## Volume layout

```
/<volume_root>/
├── selected_seqs_test.json
├── scannetv2-labels.combined.tsv    # ScanNet only
├── <scene_id>/
│   ├── 00000_x.jpg                  # RGB
│   ├── 00000_y.png                  # semantic labels (uint16)
│   ├── 00000_cam.npz                # camera_pose (4×4), camera_intrinsics (3×3)
│   ├── 00020_x.jpg
│   └── ...
└── <scene_id>/
    └── ...
```

Loaders: `replica_2dseg` (`src/dataset/dataset_replica_2dseg.py`), `scannet_2dseg` (`src/dataset/dataset_scannet_2dseg.py`). They use the same sampling and batch layout as `dataset_replica_semseg` (random context/target pairs for train; full sweep for test).

## Training

Training runs in a CUDA image built from this repo (see `src/inference/modal_train_c3g_sam.py` for volume mounts, image build, and `subprocess.run` pattern). Mount volumes:

| Volume | Mount | Dataset config |
|--------|-------|----------------|
| `c3g-weights` | `/weights` | SAM + encoder checkpoints |
| `c3g-train-outputs` | `/outputs` | Hydra runs and checkpoints |
| `replica` | `/replica` | `dataset.replica_2dseg.roots=[/replica]` |
| `scannet` | `/scannet` | `dataset.scannet_2dseg.roots=[/scannet]` |

Use `+training=feature_head_sam_prompted` with the `replica_2dseg` or `scannet_2dseg` dataset group (Modal CLI in `src/inference/modal_sam_common.py` sets this automatically). For non-prompted SAM feature distillation only, use `+training=feature_head_sam` instead.

### Replica

```bash
python -m src.main \
    +training=feature_head_sam_prompted \
    ~dataset@_group_.replica \
    ~dataset@_group_.replica_semseg \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    wandb.mode=online \
    wandb.name=sam_prompted_replica \
    hydra.run.dir=/outputs/runs/sam_prompted_replica \
    dataset.replica_2dseg.roots=[/replica] \
    train.sam_checkpoint=/weights/sam_vit_h.pth \
    model.encoder.pretrained_weights=/weights/gaussian_decoder.ckpt
```

Use `model.encoder.pretrained_weights=/weights/model.pt` if `gaussian_decoder.ckpt` is not on the volume.

### ScanNet

```bash
python -m src.main \
    +training=feature_head_sam_prompted \
    ~dataset@_group_.replica \
    ~dataset@_group_.replica_semseg \
    +dataset@_group_.scannet_2dseg=scannet_2dseg \
    wandb.mode=online \
    wandb.name=sam_prompted_scannet \
    hydra.run.dir=/outputs/runs/sam_prompted_scannet \
    dataset.scannet_2dseg.roots=[/scannet] \
    train.sam_checkpoint=/weights/sam_vit_h.pth \
    model.encoder.pretrained_weights=/weights/gaussian_decoder.ckpt
```

### Common overrides

**Random-point prompts:**

```bash
train.prompt_strategy=random_point
```

**Disable Weights & Biases:**

```bash
wandb.mode=disabled
```

**Resume** from a checkpoint on the output volume:

```bash
checkpointing.load=/outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
```

### Modal CLI (`src/inference/modal_train_c3g_sam.py`)

Full training:

```bash
modal run src/inference/modal_train_c3g_sam.py \
    --run-name sam_prompted_replica --dataset replica

modal run src/inference/modal_train_c3g_sam.py \
    --run-name sam_prompted_scannet --dataset scannet
```

Smoke tests run **detached on Modal GPU** by default (no local GPU, dataset, or blocking wait). Use `--wait` to block until completion.

Training smoke (one step on the first indexed scene; no checkpoint):

```bash
modal run src/inference/modal_train_c3g_sam.py::smoke --dataset replica
modal run src/inference/modal_train_c3g_sam.py::smoke --dataset scannet
```

Follow logs: `modal app logs c3g-train-sam-feature`

Eval smoke (one test batch; detached by default; requires `--resume`):

```bash
modal run src/inference/modal_c3g_sam.py::smoke --dataset replica \
    --resume /outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
```

Vanilla SAM on one dataset frame:

```bash
modal run src/inference/modal_vanilla_sam.py::smoke --dataset replica
```

Use `modal run --detach …` for full training jobs. Shared volume paths and Hydra overrides live in `src/inference/modal_sam_common.py`.

### Key training parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train.prompt_mode` | `prompted` | Point prompts from GT label maps |
| `train.prompted_seg_loss_weight` | `1.0` | Weight for prompted BCE + Dice loss |
| `train.prompt_strategy` | `centroid` | `centroid` or `random_point` |
| `train.min_object_pixels` | `16` | Minimum foreground pixels for a valid prompt |
| `train.sam_model_variant` | `sam_vit_h` | SAM variant |
| `train.use_lora` | `false` | LoRA on SAM decoder |
| `train.lora_rank` | `4` | LoRA rank (if enabled) |

## Evaluation (segment-everything mode)

Evaluation uses grid prompts regardless of training `prompt_mode`. Example for Replica:

```bash
python -m src.main \
    +training=feature_head_sam_prompted \
    ~dataset@_group_.replica \
    ~dataset@_group_.replica_semseg \
    +dataset@_group_.replica_2dseg=replica_2dseg \
    mode=test \
    wandb.mode=online \
    wandb.name=sam_prompted_replica_eval \
    dataset.replica_2dseg.roots=[/replica] \
    train.sam_checkpoint=/weights/sam_vit_h.pth \
    checkpointing.load=/outputs/runs/sam_prompted_replica/checkpoints/last.ckpt
```

Use `scannet_2dseg` and `/scannet` for ScanNet eval runs.

## Expected outputs

### Checkpoints

```
/outputs/runs/<wandb.name>/checkpoints/
```

Defaults: top 5 checkpoints every 10,000 steps (`checkpointing.every_n_train_steps`, `checkpointing.save_top_k`).

### Evaluation visualizations

With `test.save_compare=true`, per-scene mask overlays are written to the
`sam-eval-outputs` volume (not `c3g-train-outputs`):

```
/sam-eval-outputs/runs/<wandb.name>/<scene>/seg/
```

Download with `modal volume get sam-eval-outputs ...`.

### Wandb monitoring

When `wandb.mode=online`:

- `loss/prompted_segmentation` — prompted BCE + Dice loss
- `loss/total` — combined training loss
- Reconstruction losses (MSE, LPIPS) when enabled
- Multi-view IoU during validation
