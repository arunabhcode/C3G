# Implementation Plan: Pre-Computed SAM Distillation Pipeline

## Overview

This plan implements a standalone training pipeline that distills SAM ViT-H encoder features into the C3G model's InstillTransformer using pre-computed features and MSE loss. The implementation proceeds bottom-up: pre-computation script → dataset loaders → model wrapper → configs → integration wiring.

## Tasks

- [ ] 1. Implement the pre-computation script
  - [x] 1.1 Create `scripts/precompute_sam_features.py` with CLI interface
    - Implement `argparse` CLI accepting `--dataset-root`, `--dataset` (replica|scannet), `--scenes`, `--sam-checkpoint`, `--sam-model-variant`, `--batch-size`, `--overwrite`
    - Use `src.model.sam.loader.load_sam` to load the frozen SAM model
    - Use `src.model.sam.preprocess.resize_images_longest_side` and `preprocess_images` for normalization
    - Iterate all frames using `src.misc.frame_layout.list_frame_ids`, encode in batches, save each as `{frame_id}_sam.pt` (float32, shape 256×64×64)
    - Skip existing `.pt` files unless `--overwrite` is set
    - Log warning and continue on unreadable frames
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 1.2 Write property test for SAM feature save/load round-trip
    - **Property 1: SAM Feature Save/Load Round-Trip**
    - **Validates: Requirements 1.2, 2.2, 2.3, 2.5, 3.2, 3.3, 3.5**

  - [x] 1.3 Write property test for SAM preprocessing output shape
    - **Property 2: SAM Preprocessing Produces Correct Output**
    - **Validates: Requirements 1.4**

  - [x] 1.4 Write property test for batch processing completeness
    - **Property 3: Batch Processing Completeness**
    - **Validates: Requirements 1.7**

- [ ] 2. Implement the Replica distillation dataset loader
  - [x] 2.1 Create `src/dataset/dataset_replica_distill.py`
    - Implement `DatasetReplicaDistill(IterableDataset)` following the pattern of `dataset_replica_2dseg.py`
    - Reuse view sampling, camera loading, image preprocessing, and baseline normalization logic
    - Load `{frame_id}_sam.pt` for each view instead of `{frame_id}_y.png`
    - Produce `sam_features` key (V, 256, 64, 64) in both context and target dicts
    - Do NOT produce `label` or `sam_image` keys
    - Skip samples where any `_sam.pt` file is missing (log warning)
    - Export `DATASET_CLASS`, `DATASET_NAMES = ("replica_distill",)`, and `CFG_WRAPPERS`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 2.2 Create `config/dataset/replica_distill.yaml`
    - Follow the structure of `config/dataset/replica_2dseg.yaml`
    - Set `name: replica_distill`
    - Remove `prompt_strategy` and `min_object_pixels` (not needed for distillation)
    - _Requirements: 6.6_

- [ ] 3. Implement the ScanNet distillation dataset loader
  - [x] 3.1 Create `src/dataset/dataset_scannet_distill.py`
    - Implement `DatasetScannetDistill(IterableDataset)` following the pattern of `dataset_scannet_2dseg.py`
    - Reuse view sampling, scene splitting (`scannet_2dseg_splits`), camera loading, image preprocessing
    - Load `{frame_id}_sam.pt` for each view instead of `{frame_id}_y.png`
    - Produce `sam_features` key (V, 256, 64, 64) in both context and target dicts
    - Do NOT produce `label` or `sam_image` keys
    - Skip samples where any `_sam.pt` file is missing (log warning)
    - Export `DATASET_CLASS`, `DATASET_NAMES = ("scannet_distill",)`, and `CFG_WRAPPERS`
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [x] 3.2 Create `config/dataset/scannet_distill.yaml`
    - Follow the structure of `config/dataset/scannet_2dseg.yaml`
    - Set `name: scannet_distill`
    - Remove `prompt_strategy` and `min_object_pixels`
    - _Requirements: 6.6_

- [x] 4. Checkpoint — Ensure dataset loaders are correct
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement the distillation model wrapper
  - [x] 5.1 Create `src/model/distillation_wrapper.py`
    - Implement `DistillationModelWrapper(LightningModule)` as a new standalone file
    - Accept `optimizer_cfg`, `train_cfg` (DistillTrainCfg dataclass), `encoder` (EncoderVGGT), `decoder` (Decoder), `losses` (list[Loss]), `step_tracker`
    - In `training_step`: extract context SAM features from batch, reshape to (B, V*64*64, 256), pass as `context_feature` to encoder
    - Detach geometry (means, covariances, harmonics, opacities) before rendering features
    - Interpolate rendered features to 64×64 with bilinear mode
    - Compute MSE loss between interpolated rendered features and target SAM features from batch
    - Weight MSE loss by `feature_mse_loss_weight` (default 1.0)
    - Optionally compute RGB losses (MSE, LPIPS) for joint training
    - Do NOT instantiate or reference SAM encoder/decoder
    - Do NOT compute cosine similarity or prompted segmentation loss
    - Reuse existing `EncoderVGGT` and `Decoder` without modification
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_

  - [ ]* 5.2 Write property test for MSE loss computation correctness
    - **Property 4: MSE Loss Computation Correctness**
    - **Validates: Requirements 4.4, 4.7**

  - [x] 5.3 Write property test for InstillTransformer parameter freeze correctness
    - **Property 5: InstillTransformer Parameter Freeze Correctness**
    - **Validates: Requirements 5.4**

  - [x] 5.4 Write property test for context feature shape invariant
    - **Property 6: Context Feature Shape Invariant**
    - **Validates: Requirements 4.2**

- [x] 6. Checkpoint — Ensure model wrapper is correct
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Create training configuration and wire everything together
  - [x] 7.1 Create `config/training/feature_head_sam_precomputed.yaml`
    - Follow the structure of `config/training/feature_head_sam_prompted.yaml`
    - Reference the new `replica_distill` dataset (default)
    - Set `freeze_backbone: true`, `freeze_instill_qk: true`, `freeze_geometry_head: true`
    - Set `gaussian_feature_dim: 256`
    - Set decoder `feature_detach: true`
    - Set `train.feature_mse_loss_weight: 1.0`
    - Do NOT include `sam_checkpoint`, `sam_model_variant`, or `feature_rendering_loss`
    - Do NOT include `prompt_mode`, `prompted_seg_loss_weight`, or segmentation-related fields
    - Include standard optimizer, data_loader, trainer, and checkpointing sections
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 7.2 Register new dataset loaders in `src/dataset/__init__.py`
    - Add imports for `dataset_replica_distill` and `dataset_scannet_distill`
    - Ensure the data module can discover and instantiate the new loaders by name
    - _Requirements: 2.1, 3.1_

  - [x] 7.3 Wire the distillation model wrapper into the training entry point
    - Ensure `main.py` or the model instantiation logic can select `DistillationModelWrapper` based on config
    - The wrapper should be instantiated when the training config specifies the distillation pipeline
    - _Requirements: 4.1, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 8. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- The implementation language is Python (matching the existing codebase)
- Property tests validate universal correctness properties from the design document
- The existing live-SAM pipeline remains completely untouched — all new code is in separate files
- Checkpoints ensure incremental validation between major components
