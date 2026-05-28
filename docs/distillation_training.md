# Distillation Training with Pre-computed SAM Features

This document describes how to train C3G-F using MSE feature distillation against pre-computed SAM ViT-H encoder features on the Replica dataset.

Unlike prompted training (which runs SAM live during training with point prompts), distillation training pre-computes SAM features offline and trains the model to reproduce them via Gaussian splatting. This is faster at training time and avoids loading the full SAM model into GPU memory during training.

## Prerequisites

### 1. Environment Setup

```bash
uv sync --frozen
source .venv/bin/activate
```

### 2. SAM Checkpoint (for pre-computation only)

Download the SAM ViT-H checkpoint into `pretrained_weights/`:

```bash
mkdir -p pretrained_weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./pretrained_weights/sam_vit_h.pth
```

This checkpoint is only needed during the pre-computation step. It is not loaded during training.

### 3. Encoder Pretrained Weights

Download the VGGT pretrained weights:

```bash
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt?download=true -O ./pretrained_weights/model.pt
```

### 4. Dataset

The Replica dataset should be available at the configured root path. The default config expects it at `datasets/replica`. Override via the command line if your path differs.

Expected directory structure (before pre-computation):

```
datasets/replica/
├── office0/
│   ├── frame000000_x.jpg
│   ├── frame000000_cam.npz
│   └── ...
├── office1/
│   └── ...
└── room2/
    └── ...
```

Scenes used: `office0`, `office1`, `office2`, `office3`, `office4`, `room0`, `room1`, `room2`.

## Step 1: Pre-compute SAM Features

Before training, encode every frame with the frozen SAM image encoder. This produces a `{frame_id}_sam.pt` file (256×64×64 float32 tensor) alongside each frame.

```bash
uv run python scripts/precompute_sam_features.py \
    --dataset-root datasets/replica \
    --dataset replica \
    --sam-checkpoint pretrained_weights/sam_vit_h.pth \
    --batch-size 8
```

To process only specific scenes:

```bash
uv run python scripts/precompute_sam_features.py \
    --dataset-root datasets/replica \
    --dataset replica \
    --scenes office0 office1 \
    --sam-checkpoint pretrained_weights/sam_vit_h.pth \
    --batch-size 8
```

To overwrite existing feature files:

```bash
uv run python scripts/precompute_sam_features.py \
    --dataset-root datasets/replica \
    --dataset replica \
    --sam-checkpoint pretrained_weights/sam_vit_h.pth \
    --overwrite
```

After pre-computation, the dataset directory will contain:

```
datasets/replica/
├── office0/
│   ├── frame000000_x.jpg
│   ├── frame000000_cam.npz
│   ├── frame000000_sam.pt    ← new
│   └── ...
└── ...
```

### Pre-computation Arguments

| Argument             | Default      | Description                                      |
| -------------------- | ------------ | ------------------------------------------------ |
| `--dataset-root`     | (required)   | Root directory of the dataset                    |
| `--dataset`          | (required)   | Dataset type: `replica` or `scannet`             |
| `--scenes`           | all scenes   | Specific scenes to process                       |
| `--sam-checkpoint`   | (required)   | Path to SAM checkpoint file                      |
| `--sam-model-variant`| `sam_vit_h`  | SAM model variant                                |
| `--batch-size`       | `8`          | Frames per encoding batch                        |
| `--overwrite`        | `false`      | Overwrite existing `.pt` files                   |

## Step 2: Run Distillation Training

```bash
uv run python -m src.main +training=feature_head_sam_precomputed_replica
```

To override the dataset root:

```bash
uv run python -m src.main +training=feature_head_sam_precomputed_replica \
    dataset.replica_distill.roots="[/path/to/replica]"
```

To disable wandb logging:

```bash
uv run python -m src.main +training=feature_head_sam_precomputed_replica wandb.mode=disabled
```

### Key Training Parameters

| Parameter                          | Default  | Description                                         |
| ---------------------------------- | -------- | --------------------------------------------------- |
| `train.pipeline`                   | `distillation` | Selects the distillation training loop        |
| `train.feature_mse_loss_weight`    | `1.0`    | Weight for the MSE feature loss                     |
| `train.context_view_loss`          | `true`   | Include context views in the feature loss           |
| `optimizer.lr`                     | `1e-5`   | Learning rate                                       |
| `optimizer.warm_up_steps`          | `1000`   | Linear warmup steps                                 |
| `trainer.max_steps`                | `5001`   | Total training steps                                |
| `trainer.accumulate_grad_batches`  | `2`      | Gradient accumulation steps                         |
| `data_loader.train.batch_size`     | `4`      | Batch size per GPU                                  |

### Frozen Components

The distillation config freezes most of the encoder to train only the feature rendering head:

- `freeze_backbone: true` — VGGT backbone weights are frozen
- `freeze_instill_qk: true` — Cross-attention QK projections are frozen
- `freeze_geometry_head: true` — Gaussian geometry head is frozen
- `feature_detach: true` — Gaussian geometry is detached from the feature loss gradient

Only the feature head and decoder feature rendering path receive gradients.

## Evaluation

Run evaluation on a trained checkpoint:

```bash
uv run python -m src.main +training=feature_head_sam_precomputed_replica \
    mode=test \
    wandb.name="sam_distill_eval" \
    checkpointing.load="path/to/checkpoint.ckpt"
```

## Expected Outputs

### Checkpoints

Checkpoints are saved to the Hydra output directory:

```
outputs/<date>/<time>/checkpoints/
```

The config saves the top 2 checkpoints every 50 steps (configurable via `checkpointing.every_n_train_steps` and `checkpointing.save_top_k`).

### Wandb Monitoring

When `wandb.mode=online`, the following metrics are logged:

- `loss/feature_mse` — MSE between rendered and target SAM features
- `loss/total` — combined training loss (feature MSE + any auxiliary losses)
- `val/feature_mse` — validation feature MSE
- Learning rate schedule via `LearningRateMonitor`

## How It Works

1. The dataset loader (`DatasetReplicaDistill`) loads pre-computed SAM features from `{frame_id}_sam.pt` files alongside RGB images and camera parameters.
2. The encoder produces Gaussians with a learned 256-dim feature channel from context views.
3. The decoder renders these Gaussian features into target viewpoints.
4. The rendered features are bilinearly interpolated to match the SAM feature resolution (64×64).
5. An MSE loss is computed between the rendered features and the ground-truth SAM features.
6. Optionally, context views are also included in the loss (controlled by `train.context_view_loss`).

## Comparison with Prompted Training

| Aspect                  | Prompted Training                    | Distillation Training                |
| ----------------------- | ------------------------------------ | ------------------------------------ |
| SAM at train time       | Yes (full model in GPU memory)       | No (pre-computed offline)            |
| Loss function           | BCE + Dice on segmentation masks     | MSE on encoder features              |
| Training speed          | Slower (live SAM forward pass)       | Faster (features loaded from disk)   |
| Disk usage              | Lower                                | Higher (`.pt` files per frame)       |
| Supervision signal      | Prompted segmentation masks          | Dense encoder features               |
| Config                  | `feature_head_sam_prompted`          | `feature_head_sam_precomputed_replica` |
