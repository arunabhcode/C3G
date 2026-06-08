# Implementation Plan: Wandb Debug Visualizations

## Overview

Implement a debug visualization module that logs structured diagnostic artifacts to Weights & Biases during validation at checkpoint frequency. The module is a collection of pure functions in `src/model/debug_visualizer.py`, called from the existing `validation_step` in `src/model/model_wrapper.py`.

## Tasks

- [ ] 1. Create debug visualizer module with core computation functions
  - [x] 1.1 Create `src/model/debug_visualizer.py` with `compute_mse_heatmap` and `compute_cosine_error_map`
    - Implement `compute_mse_heatmap(rendered_feature, target_feature)` returning (H, W) tensor of per-pixel MSE
    - Implement `compute_cosine_error_map(rendered_feature, target_feature)` returning (H, W) tensor of 1 - cosine_similarity with epsilon for zero-norm safety
    - _Requirements: 5.1, 6.1, 6.2, 6.3_

  - [ ]* 1.2 Write property test for MSE heatmap non-negativity and shape
    - **Property 2: MSE heatmap non-negativity and shape**
    - **Validates: Requirements 5.1**

  - [ ]* 1.3 Write property test for cosine error map range and shape
    - **Property 3: Cosine error map range and shape**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [x] 1.4 Implement `colorize_heatmap` and `alpha_blend`
    - Implement `colorize_heatmap(heatmap, vmin=0.0, vmax=None)` using viridis colormap, returns (3, H, W) in [0, 1]
    - Implement `alpha_blend(overlay, background, alpha=0.5)` returning (3, H, W) in [0, 1]
    - _Requirements: 5.3, 5.4, 6.5, 6.6_

  - [ ]* 1.5 Write property test for colormap output validity
    - **Property 4: Colormap produces valid 3-channel RGB**
    - **Validates: Requirements 5.3, 6.5**

  - [ ]* 1.6 Write property test for alpha blend value range preservation
    - **Property 5: Alpha blend preserves value range**
    - **Validates: Requirements 5.4, 6.6**

  - [x] 1.7 Implement `compute_feature_norms`
    - Implement `compute_feature_norms(feature)` computing per-spatial-position L2 norms, returns flat vector of length H*W
    - _Requirements: 7.1_

  - [ ]* 1.8 Write property test for feature norm statistics consistency
    - **Property 6: Feature norm statistics consistency**
    - **Validates: Requirements 7.1, 7.3**

- [ ] 2. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 3. Implement orchestration function and wandb table logging
  - [x] 3.1 Implement `log_debug_visualizations` orchestration function
    - Gate on `global_step % checkpoint_interval == 0`, return early otherwise
    - Check `isinstance(logger, WandbLogger)` and return early if not
    - Check rendered feature is not None, return early if so
    - Compute PCA for target SAM feature and rendered feature using `run_pca` from `src/model/utils.py`
    - Compute MSE heatmap and cosine error map, colorize, resize to target RGB dimensions, alpha-blend onto inverse-normalized target RGB
    - Build `wandb.Table` with columns: "target_rgb", "target_sam_pca", "rendered_feature_pca", "mse_heatmap_overlay", "cosine_error_overlay"
    - Convert images to numpy HWC uint8 and wrap in `wandb.Image`
    - Log feature norm histogram and scalar stats (min, max, std) for both target and rendered features
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.2, 5.5, 6.4, 6.7, 7.2, 7.3, 8.1, 8.2, 8.3_

  - [ ]* 3.2 Write property test for PCA output validity
    - **Property 1: PCA output validity**
    - **Validates: Requirements 3.1, 3.2, 4.1, 4.2**

  - [ ]* 3.3 Write property test for checkpoint frequency gating
    - **Property 7: Checkpoint frequency gating**
    - **Validates: Requirements 8.2, 8.3**

  - [ ]* 3.4 Write unit tests for table schema and skip behavior
    - Test that wandb.Table column names match spec: "target_rgb", "target_sam_pca", "rendered_feature_pca", "mse_heatmap_overlay", "cosine_error_overlay"
    - Test that passing None rendered feature results in no logging
    - Test that non-WandbLogger results in no logging
    - _Requirements: 1.2, 1.3_

- [ ] 4. Integrate debug visualizer into validation_step
  - [x] 4.1 Add call to `log_debug_visualizations` at end of `validation_step` in `src/model/model_wrapper.py`
    - Import `log_debug_visualizations` from `src.model.debug_visualizer`
    - Call at the end of the feature rendering visualization block (after PCA comparison logging)
    - Pass `self.logger`, `self.global_step`, `get_cfg()["checkpointing"]["every_n_train_steps"]`, target RGB tensor, target SAM feature, rendered feature, and `(h, w)` image size
    - _Requirements: 1.1, 8.1, 8.2_

- [ ] 5. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- All visualization logic lives in `src/model/debug_visualizer.py` as pure functions (no class state)
- Coding style: no type annotations, one-line docstrings only, no inline comments, no private member functions
- Use `uv run pytest tests/test_debug_visualizer.py` to run tests
- Property tests use `hypothesis` with `hypothesis[numpy]` for tensor generation
- The existing `run_pca` from `src/model/utils.py` and `inverse_normalize` from `src/misc/utils.py` are reused
