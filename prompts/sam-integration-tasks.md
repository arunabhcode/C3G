# Implementation Plan: SAM C3G-F Integration

## Overview

This plan implements the SAM (Segment Anything Model) integration with the C3G-F feature distillation pipeline. The implementation follows an incremental approach: first loading the SAM encoder and extracting features, then wiring features through the existing InstillTransformer, adding the mask decoder with optional LoRA, implementing the segmentation loss, creating the training configuration, and finally integrating everything into the training loop.

## Tasks

- [ ] 1. SAM encoder loading and feature extraction
  - [x] 1.1 Add SAM encoder loading to `src/model/load_foundation_model.py`
    - Add `elif 'sam' in cfg.train.reproj_model:` branch
    - Load SAM image encoder from checkpoint (sam_vit_h, sam_vit_l, or sam_vit_b)
    - Freeze all encoder parameters (`requires_grad = False`)
    - Set `feature_dim = 256`
    - Return the SAM encoder alongside existing model slots
    - _Requirements: 1.4, 1.5_

  - [x] 1.2 Add SAM feature extraction branch in `src/model/model_wrapper.py`
    - Add `elif 'sam' in self.train_cfg.reproj_model:` in `forward_foundation_model()`
    - Resize input images to 1024×1024 before passing to SAM encoder
    - Run frozen SAM encoder to produce `(B*V, 256, 64, 64)` features
    - Reshape to `(B, V, 256, 64, 64)`
    - Interpolate to `(H//14, W//14)` when `interpolate=True` to match pipeline grid
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.3 Wire SAM encoder into `ModelWrapper.__init__` in `src/model/model_wrapper.py`
    - Accept SAM encoder as constructor parameter (similar to `dino`, `lseg_feature_extractor`)
    - Store as `self.sam_encoder`
    - Update `src/main.py` to pass SAM encoder from `load_foundation_model()` return value
    - _Requirements: 1.4, 1.5_

  - [x] 1.4 Write property test for feature extraction shape invariant
    - **Property 1: Feature Extraction Shape Invariant**
    - **Validates: Requirements 1.1, 1.3**
    - Use Hypothesis to generate arbitrary (B, V, H, W) dimensions
    - Verify output shape is always `(B, V, 256, H//14, W//14)` after the full pipeline

- [x] 2. Checkpoint - Ensure SAM encoder loads and extracts features correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. LoRA adapter module
  - [x] 3.1 Create `src/model/lora.py` with `LoRALinear` class
    - Implement `LoRALinear(nn.Module)` wrapping an existing `nn.Linear`
    - Accept `rank` parameter (default 4)
    - Initialize `lora_A` as `nn.Linear(in_features, rank, bias=False)` with default init
    - Initialize `lora_B` as `nn.Linear(rank, out_features, bias=False)` with zeros
    - Freeze original linear layer weights
    - Forward: `original(x) + lora_B(lora_A(x))`
    - Add helper function `inject_lora(module, target_layer_name, rank)` to replace a linear layer with LoRA-wrapped version
    - _Requirements: 5.1, 5.3, 5.4_

  - [x] 3.2 Write property test for LoRA parameter count
    - **Property 6: LoRA Parameter Count**
    - **Validates: Requirements 5.2**
    - Use Hypothesis to generate arbitrary (d_in, d_out, rank) dimensions
    - Verify trainable parameter count equals `rank * (d_in + d_out)` per LoRA layer

- [ ] 4. SAM mask decoder wrapper
  - [x] 4.1 Create `src/model/sam_decoder.py` with `SAMMaskDecoderWrapper` class
    - Load SAM's `MaskDecoder` and `PromptEncoder` from checkpoint
    - Freeze all mask decoder and prompt encoder parameters by default
    - Implement `forward(rendered_features, point_coords, point_labels, box)`:
      - Interpolate `rendered_features` to 64×64 if spatial dims differ
      - Generate positional encoding via SAM's `PromptEncoder.get_dense_pe()`
      - Encode sparse/dense prompts (or use grid points for segment-everything mode)
      - Run mask decoder with rendered features as `image_embeddings`
      - Return predicted masks
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [x] 4.2 Add optional LoRA injection to `SAMMaskDecoderWrapper`
    - When `use_lora=True`, inject `LoRALinear` on `v_proj` layers in token-to-image cross-attention
    - Freeze all original mask decoder parameters
    - Only LoRA parameters are trainable
    - When `use_lora=False`, mask decoder operates with original frozen weights
    - _Requirements: 5.1, 5.4, 5.5, 5.6_

  - [x] 4.3 Write property test for mask decoder accepting rendered features
    - **Property 5: Mask Decoder Accepts Rendered Features**
    - **Validates: Requirements 3.1, 3.3, 3.4**
    - Use Hypothesis to generate arbitrary (B, 256, H, W) feature tensors and valid prompts
    - Verify output masks have valid shape and no errors

  - [x] 4.4 Write property test for InstillAttention dual-stream output
    - **Property 2: InstillAttention Dual-Stream Output**
    - **Validates: Requirements 2.1, 2.3**
    - Use Hypothesis to generate arbitrary token sequences with matching batch dims
    - Verify two output tensors with correct shapes are produced

- [x] 5. Checkpoint - Ensure LoRA and mask decoder work in isolation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Segmentation loss
  - [x] 6.1 Create `src/loss/loss_segmentation.py` with `SegmentationLoss` class
    - Inherit from the project's `Loss` base class
    - Implement combined BCE + dice loss matching SAM's original training objective
    - Accept `weight` parameter for loss scaling (from config `segmentation_loss_weight`)
    - In `forward()`: receive rendered features from `prediction.feature`, run SAM mask decoder, compute loss against ground-truth masks from batch
    - Handle missing ground-truth masks gracefully (skip loss, log info)
    - Interpolate predicted masks to match ground-truth spatial dimensions if needed
    - _Requirements: 4.1, 4.2, 4.5, 4.6_

  - [x] 6.2 Write property test for segmentation loss correctness
    - **Property 3: Segmentation Loss Correctness**
    - **Validates: Requirements 4.1, 4.2, 4.5**
    - Use Hypothesis to generate arbitrary equal dimensioned pred/gt mask tensors
    - Verify loss equals `weight * (BCE(pred, gt) + dice_loss(pred, gt))`
    - Verify loss scales linearly with weight parameter

- [ ] 7. Gradient routing and training integration
  - [x] 7.1 Integrate segmentation loss into training loop in `src/model/model_wrapper.py`
    - Add `segmentation_loss_weight` field to `TrainCfg` dataclass
    - Add `sam_model_variant`, `sam_checkpoint`, `use_lora`, `lora_rank` fields to `TrainCfg`
    - In `training_step()`, after feature rendering, compute segmentation loss when `segmentation_loss_weight > 0`
    - Combine with existing feature distillation loss and reconstruction losses
    - Log segmentation loss to wandb
    - _Requirements: 4.3, 4.5, 4.6, 6.5_

  - [x] 7.2 Ensure correct gradient routing
    - When LoRA is disabled: SAM mask decoder parameters have no gradients, only C3G-F decoder (`to_anotherv`, `to_yout`, `ff2`) receives gradient updates
    - When LoRA is enabled: LoRA parameters and C3G-F decoder parameters receive gradients, original SAM decoder weights remain frozen
    - Verify gradients flow through: mask decoder → rendered features → Gaussian rasterizer → per-Gaussian features → feature head → InstillTransformer
    - _Requirements: 4.3, 4.4, 5.4_

  - [x] 7.3 Write property test for gradient routing correctness
    - **Property 4: Gradient Routing Correctness**
    - **Validates: Requirements 4.3, 4.4**
    - Verify C3G-F decoder trainable params have non-zero gradients after backward
    - Verify SAM mask decoder params have zero/None gradients when LoRA disabled

  - [x] 7.4 Write property test for feature rendering differentiability
    - **Property 8: Feature Rendering Differentiability**
    - **Validates: Requirements 8.1, 8.2**
    - Verify backward pass produces non-zero gradients on per-Gaussian feature vectors
    - Verify gradients propagate through feature head to C3G-F decoder parameters

- [x] 8. Checkpoint - Ensure loss computation and gradient flow work correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Training configuration
  - [x] 9.1 Create `config/training/feature_head_sam.yaml`
    - Follow structure of existing `feature_head_lseg.yaml`
    - Set `reproj_model: sam` and `gaussian_feature_dim: 256`
    - Set `feature_rendering_loss: 0.01`
    - Add `segmentation_loss_weight: 1.0`
    - Add `sam_model_variant: sam_vit_h`
    - Add `sam_checkpoint: './pretrained_weights/sam_vit_h.pth'`
    - Add `use_lora: false` and `lora_rank: 4`
    - Set `feature_detach: false` to allow gradients through geometry
    - _Requirements: 6.1, 6.2, 6.3, 6.5_

  - [x] 9.2 Add config validation in model initialization
    - Validate SAM checkpoint file exists and is accessible at load time
    - Validate `sam_model_variant` is one of supported variants
    - Raise clear error messages for invalid configuration
    - _Requirements: 6.4_

- [ ] 10. Multi-view consistency evaluation
  - [x] 10.1 Add multi-view consistency metric logging in `model_wrapper.py`
    - During validation, render features from two different viewpoints
    - Run SAM mask decoder on both rendered feature maps
    - Compute IoU between masks projected from one view to another using known camera geometry
    - Log multi-view consistency metrics to wandb
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 10.2 Write property test for multi-view feature rendering consistency
    - **Property 7: Multi-View Feature Rendering Consistency**
    - **Validates: Requirements 7.1, 7.2**
    - Verify that rendering the same Gaussians from two viewpoints produces consistent features
    - Verify IoU between projected masks exceeds minimum threshold for overlapping regions

- [ ] 11. Feature rendering robustness
  - [x] 11.1 Add NaN/Inf clamping to rendered feature maps
    - In the decoder's feature rendering path, clamp feature values to `[-1e4, 1e4]`
    - Log a warning with step number and view index when clamping occurs
    - Ensure training continues without interruption
    - _Requirements: 8.3, 8.4_

  - [x] 11.2 Ensure `feature_detach` configuration is respected
    - When `feature_detach: false`, gradients flow through both Gaussian geometry and feature vectors
    - When `feature_detach: true`, gradients only flow through feature vectors (geometry detached)
    - _Requirements: 8.2, 8.3_

- [x] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document (8 properties total)
- The implementation uses Python with PyTorch, following existing codebase patterns
- SAM model weights should be downloaded separately to `./pretrained_weights/`
- The existing `InstillTransformer` and `FeatureDetachGaussianRasterizer` require no modifications
