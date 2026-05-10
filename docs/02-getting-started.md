# Getting Started

## Installation

The codebase requires Python 3.11, PyTorch 2.5.1, and CUDA 12.4.

```bash
conda create -n c3g python=3.11
conda activate c3g

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Pretrained Weight Downloads

### VGGT Backbone

Download the VGGT-1B pretrained weights from [VGGT](https://github.com/facebookresearch/vggt):

```bash
mkdir -p pretrained_weights
wget https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt?download=true -O ./pretrained_weights/model.pt
```

### LSeg (for feature distillation)

Download LSeg pretrained weights for semantic feature lifting:

```bash
gdown 1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7 -O ./pretrained_weights/demo_e200.ckpt
```

### HuggingFace Checkpoints

All pretrained C3G checkpoints are available at [huggingface.co/honggyuAn/C3G](https://huggingface.co/honggyuAn/C3G/tree/main):

| Checkpoint | Description |
|-----------|-------------|
| `gaussian_decoder.ckpt` | Gaussian Decoder trained for 2-view input |
| `gaussian_decoder_multiview.ckpt` | Gaussian Decoder trained for multi-view input |
| `feature_decoder_lseg.ckpt` | Feature Decoder trained with LSeg |
| `feature_decoder_dinov3L.ckpt` | Feature Decoder trained with DINOv3-L |
| `feature_decoder_dinov2.ckpt` | Feature Decoder trained with DINOv2-L |

## Dataset Preparation

### RealEstate10K (Training + NVS Evaluation)

Used for training and novel view synthesis evaluation. Follow the preprocessing from [pixelSplat](https://github.com/dcharatan/pixelsplat) or [MVSplat](https://github.com/donydchen/mvsplat).

Place the processed data at `datasets/re10k/` (or update `config/dataset/re10k.yaml` roots).

Expected structure:

```
datasets/re10k/
├── train/
│   ├── index.json
│   └── *.torch          # chunked scene files
└── test/
    ├── index.json
    └── *.torch
```

### ScanNet (Scene Understanding Evaluation)

Used for 3D semantic segmentation evaluation. Follow [LSM](https://github.com/NVlabs/LSM/blob/main/data_process/data.md) for preprocessing.

Required files:

- `scannetv2-labels.combined.tsv` (label mapping)
- `selected_seqs_test.json` (test scene selection)
- Per-scene: images, depths, labels directories

### Replica (Scene Understanding Evaluation)

Used for 3D semantic segmentation evaluation. Follow the preprocessing and evaluation protocol from [Feature 3DGS](https://github.com/ShijieZhou-UCLA/feature-3dgs).

Requires COLMAP sparse reconstructions per scene (`sparse/0/images.bin`, `sparse/0/cameras.bin`).

## Training Commands

### Gaussian Decoder Training

Train the Gaussian Decoder (2-view):

```bash
python -m src.main +training=gaussian_head wandb.mode=online wandb.name="wandb_name"
```

Train the Gaussian Decoder (multi-view):

```bash
python -m src.main +training=gaussian_head_multiview wandb.mode=online wandb.name="wandb_name"
```

Continue from 2-view checkpoint for faster multi-view training:

```bash
python -m src.main +training=gaussian_head wandb.mode=online wandb.name="wandb_name" \
  checkpointing.load="2view_checkpoint" model.decoder.low_pass_filter=0.3
```

### Feature Decoder Training

> **Important: Update `NUM_SEMANTIC_CHANNELS`**
>
> Before training a feature decoder, update the value in:
> `./submodules/diff_gaussian_rasterization_w_feature_detach/cuda_rasterizer/config.h`
>
> | VFM | NUM_SEMANTIC_CHANNELS |
> |-----|----------------------|
> | LSeg | 512 |
> | DINOv2-base | 768 |
> | DINOv2-large / DINOv3-large | 1024 |
> | VGGT-tracking | 128 |

Train with various VFMs:

```bash
# LSeg
python -m src.main +training=feature_head_lseg wandb.mode=online wandb.name="wandb_name" \
  model.encoder.pretrained_weights="2view_checkpoint"

# DINOv2-base
python -m src.main +training=feature_head_dinov2_B wandb.mode=online wandb.name="wandb_name" \
  model.encoder.pretrained_weights="2view_checkpoint"

# DINOv2-large
python -m src.main +training=feature_head_dinov2_L wandb.mode=online wandb.name="wandb_name" \
  model.encoder.pretrained_weights="2view_checkpoint"

# DINOv3-large
python -m src.main +training=feature_head_dinov3_L wandb.mode=online wandb.name="wandb_name" \
  model.encoder.pretrained_weights="2view_checkpoint"

# VGGT-tracking
python -m src.main +training=feature_head_vggt wandb.mode=online wandb.name="wandb_name" \
  model.encoder.pretrained_weights="2view_checkpoint"
```

Multi-view feature decoder (LSeg example):

```bash
python -m src.main +training=feature_head_lseg_multiview wandb.mode=online wandb.name="wandb_name" \
  model.encoder.pretrained_weights="multiview_checkpoint"
```

To disable W&B logging, set `wandb.mode=disabled`.

## Evaluation Commands

### Novel View Synthesis on RE10K (2-view)

```bash
python -m src.main +evaluation=re10k mode=test \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
  test.save_compare=true wandb.mode=online \
  checkpointing.load="checkpoint_path" wandb.name="wandb_name"
```

### Novel View Synthesis on RE10K (multi-view)

```bash
python -m src.main +evaluation=re10k_multiview mode=test \
  dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
  dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
  test.save_compare=true wandb.mode=online \
  checkpointing.load="checkpoint_path" wandb.name="wandb_name"
```

### 3D Scene Understanding on ScanNet

```bash
python -m src.main +evaluation=scannet wandb.mode=online mode=test \
  test.save_compare=true test.pose_align_steps=1000 \
  checkpointing.load="checkpoint_path" wandb.name="wandb_name"
```
