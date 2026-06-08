# Implementation Plan: SAM Prompted Training Pipeline

## Overview

This plan implements a prompted-mode training pipeline for finetuning C3G-F via SAM on the Replica SemSeg dataset. The implementation proceeds incrementally: dataset loader → prompt sampler → prompted loss → config integration → training loop wiring → property-based tests.

## Tasks

- [ ] 1. Implement the Replica SemSeg dataset loader
  - [x] 1.1 Create `src/dataset/dataset_replica_semseg.py` with `DatasetReplicaSemSeg` class
    - Implement `ReplicaSemSegCfg` dataclass with fields: name, roots, scenes, baseline_min/max, max_fov, make_baseline_1, augment, relative_pose, skip_bad_shape, num_of_inputs, prompt_strategy, min_object_pixels
    - Implement `load_intrinsics()` to read `{root}/replica/cam_params.json` (fx=600, fy=600, cx=599.5, cy=339.5)
    - Implement `load_trajectory(scene)` to parse `{root}/replica/{scene}/traj.txt` (16 floats per line → 4x4 matrix)
    - Implement `decompose_labels(label_map)` to convert integer label map to (K, H, W) binary masks for non-background objects
    - Implement `__iter__()` yielding batches with context/target views including `label` key (integer semantic label map)
    - Load RGB from `{root}/replica/{scene}/results/frame{id:06d}.jpg`, labels from `{root}/replica_label_maps/{scene}/semantic_{id:06d}.png`
    - Resize RGB with bilinear interpolation, labels with nearest-neighbor to `input_image_shape`
    - Raise `FileNotFoundError` if dataset root does not exist at init time
    - Follow the `DatasetReplica2dSeg` pattern for view sampling and batch structure
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 1.2 Write property test for trajectory parsing round-trip
    - **Property 1: Trajectory Parsing Round-Trip**
    - **Validates: Requirements 1.2**

  - [x] 1.3 Write property test for nearest-neighbor resize preserving label set
    - **Property 2: Nearest-Neighbor Resize Preserves Label Set**
    - **Validates: Requirements 1.4, 2.5**

  - [x] 1.4 Write property test for label map decomposition correctness
    - **Property 3: Label Map Decomposition Correctness**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

- [ ] 2. Implement the Prompt Sampler
  - [x] 2.1 Create `src/model/prompt_sampler.py` with `PromptSampler` class
    - Implement `__init__(strategy, min_object_pixels, image_size)` storing config
    - Implement `sample(binary_masks)` that picks a random valid mask (≥ min_object_pixels foreground pixels), generates point prompt, returns (point_coords, point_labels, selected_gt_mask)
    - Implement `compute_centroid(mask)` returning mean row/col of foreground pixels rounded to nearest int
    - Implement `sample_random_point(mask)` returning a uniform random foreground pixel
    - Normalize point coordinates to SAM input space [0, image_size] and set foreground label to 1
    - Re-sample if selected mask has fewer than `min_object_pixels` foreground pixels
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [x] 2.2 Write property test for prompt point lying within selected mask
    - **Property 4: Prompt Point Lies Within Selected Mask**
    - **Validates: Requirements 3.2, 3.3**

  - [x] 2.3 Write property test for prompt sampler output validity
    - **Property 5: Prompt Sampler Output Validity**
    - **Validates: Requirements 3.5, 3.6**

- [ ] 3. Implement the Prompted Segmentation Loss
  - [x] 3.1 Create `src/loss/loss_segmentation_prompted.py` with `LossSegmentationPrompted` class
    - Implement `LossSegmentationPromptedCfg` dataclass with weight, sam_checkpoint, sam_model_variant, use_lora, lora_rank, prompt_strategy, min_object_pixels
    - Instantiate `SAMMaskDecoderWrapper` and `PromptSampler` in `__init__`
    - Implement `forward(prediction, batch, gaussians, global_step, target_image)`:
      - For each target view: decompose label map into binary masks
      - Sample a point prompt from a random object via `PromptSampler`
      - Pass rendered features (interpolated to 64x64) + point prompt to SAM decoder with `multimask_output=True` → 3 candidates
      - Compute BCE + Dice loss for each candidate against GT binary mask
      - Select candidate with minimum loss, backpropagate through that one
      - Return `weight * (bce + dice)` for the best candidate
    - Skip loss for views where label map is all background, log warning
    - Interpolate predicted mask to GT resolution before loss computation
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.5_

  - [x] 3.2 Write property test for best-of-3 mask selection minimizing loss
    - **Property 6: Best-of-3 Mask Selection Minimizes Loss**
    - **Validates: Requirements 4.2**

  - [x] 3.3 Write property test for prompted loss equaling weighted BCE + Dice
    - **Property 7: Prompted Loss Equals Weighted BCE + Dice**
    - **Validates: Requirements 4.3, 4.5**

- [ ] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Integrate training config and training loop
  - [x] 5.1 Add new fields to `TrainCfg` in `src/model/model_wrapper.py`
    - Add `prompt_mode: str = "grid"` (values: "grid" or "prompted")
    - Add `prompted_seg_loss_weight: float = 1.0`
    - Add `prompt_strategy: str = "centroid"` (values: "centroid" or "random_point")
    - Add `min_object_pixels: int = 16`
    - Add validation for `prompt_mode` and `prompt_strategy` in `validate_sam_config()`
    - _Requirements: 5.1, 8.2, 8.3, 8.4_

  - [x] 5.2 Modify `training_step()` in `src/model/model_wrapper.py` to use prompted loss
    - When `prompt_mode == "prompted"` and `output.feature is not None`:
      - Instantiate or use cached `LossSegmentationPrompted` (similar to existing `self.segmentation_loss` pattern)
      - Compute prompted loss using target view label maps from `batch["target"]["label"]`
      - Log as `loss/prompted_segmentation`
      - Add `prompted_seg_loss_weight * loss` to total_loss
    - When `prompt_mode == "grid"`, keep existing segmentation loss behavior unchanged
    - Skip prompted loss when target label map is all background, log warning
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

  - [x] 5.3 Write property test for gradient flow through prompted loss
    - **Property 8: Gradient Flow Through Prompted Loss**
    - **Validates: Requirements 5.2**

- [ ] 6. Create configuration files
  - [x] 6.1 Create `config/dataset/replica_semseg.yaml`
    - Set `name: replica_semseg`, `roots: [/home/arunabh/replica_semseg]`
    - Set `scenes: [office0, office1, office2, office3, office4, room0, room1, room2]`
    - Set `input_image_shape: [256, 256]`, `original_image_shape: [680, 1200]`
    - Set `prompt_strategy: centroid`, `min_object_pixels: 16`
    - Include view_sampler and baseline settings matching existing replica config pattern
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 6.2 Create `config/training/feature_head_sam_prompted.yaml`
    - Set `train.prompt_mode: prompted`
    - Set `train.prompted_seg_loss_weight: 1.0`
    - Set `train.prompt_strategy: centroid`
    - Set `train.min_object_pixels: 16`
    - Retain SAM integration params (sam_model_variant, sam_checkpoint, use_lora, lora_rank)
    - Reference the `replica_semseg` dataset config
    - Follow the pattern of `feature_head_sam.yaml` for model/encoder/loss defaults
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ] 7. Wire dataset registration and ensure evaluation uses grid prompts
  - [x] 7.1 Register `DatasetReplicaSemSeg` in the dataset module
    - Add `DATASET_CLASS`, `DATASET_NAMES`, and `CFG_WRAPPERS` exports to the dataset module
    - Ensure the dataset is discoverable by the data module when `name: replica_semseg` is configured
    - _Requirements: 1.1, 7.1_

  - [x] 7.2 Ensure evaluation mode uses grid prompts regardless of `prompt_mode`
    - In validation/test step, always use grid prompts (segment-everything mode) for mask generation
    - Compute and log multi-view IoU metrics using existing `compute_sam_mask_metrics` infrastructure
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 8. Add run instructions documentation
  - [x] 8.1 Add prompted training section to README or create `docs/prompted_training.md`
    - Document prerequisites: dataset path, SAM checkpoint download, environment setup
    - Include exact training command with config overrides
    - Include evaluation command for segment-everything mode
    - Document expected outputs: checkpoint location, eval visualizations, wandb monitoring
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- The implementation reuses existing `SAMMaskDecoderWrapper`, `LossSegmentation` patterns, and `DatasetReplica2dSeg` structure
- All property tests use Hypothesis with minimum 100 iterations per property
