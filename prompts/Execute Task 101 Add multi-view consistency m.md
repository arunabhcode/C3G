# Execute Task: 10.1 Add multi-view consistency m...

Exported: 6/8/2026, 12:24:15 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 10.1 Add multi-view consistency metric logging in `model_wrapper.py`

---

**Assistant:**

Let me read the relevant source files to understand the current structure.

Now let me look at the `warp_mask_iou` function to understand how it works:

Now let me look at the decoder's forward method to understand how features are rendered from different viewpoints:

Now let me look at how the batch data is structured, particularly the extrinsics and intrinsics:

Now let me look at how the batch provides context and target views with their camera parameters:

Now let me look at the imports at the top of model_wrapper.py:

Now I have a good understanding of the codebase. Let me look at how the `compute_warp_mask_miou` is currently used and understand the `warp_mask_to_pose` limitation:

The `warp_mask_to_pose` function is not yet implemented (raises `NotImplementedError`). For multi-view consistency, I need to:
1. Render features from two different viewpoints (context views)
2. Decode masks from both
3. Compute IoU between the masks using a simpler approach since `warp_mask_to_pose` isn't implemented

Since `warp_mask_to_pose` is not implemented, I'll use the depth from the decoder output to do the reprojection. Let me check what depth information is available:

Since `warp_mask_to_pose` is not implemented, I'll implement the multi-view consistency metric using a different approach: render features from two viewpoints, decode masks from both, and compute a direct IoU between the masks (since both are predictions from the same 3D Gaussians, they should produce consistent masks for overlapping regions). For a proper cross-view comparison, I'll implement a simple depth-based reprojection within the metric function itself.

Let me look at how depth is rendered and what camera utilities are available:

Now I have enough context. I'll implement the multi-view consistency metric. The approach:

1. During validation, when we have features rendered from both context and target viewpoints, decode masks from two different views
2. Since `warp_mask_to_pose` is not implemented, I'll implement a depth-based reprojection within the metric to warp masks between views
3. Compute IoU between the warped mask and the mask from the other view
4. Log the metric to wandb

Let me first implement the `warp_mask_to_pose` function in `mask_metrics.py` since it's needed:

Now let me look at how the decoder output provides depth, which I'll need for reprojection:

Good. Now I have all the information I need. Let me implement the multi-view consistency metric. The approach:

1. Implement `warp_mask_to_pose` using depth-based reprojection in `mask_metrics.py`
2. Add a `compute_multiview_consistency` method to `ModelWrapper`
3. Call it during `validation_step` and log to wandb

Let me implement the depth-based reprojection first:

Now update the `warp_mask_iou` function to pass depth through:

Now let me add the multi-view consistency computation method to `ModelWrapper` and integrate it into `validation_step`. First, let me add the method after `compute_warp_mask_miou`:

Now let me add the call to `compute_multiview_consistency` in the `validation_step`. I'll add it after the existing SAM metrics block:

Now I'll add the multi-view consistency metric logging right after the SAM metrics block:

Now let me verify the changes compile correctly:

Let me also verify the syntax by running a quick import check:

Let me also run the existing tests to make sure nothing is broken:

This is a pre-existing import error (missing `dataset_replica_pose` module), not related to my changes. Let me run just the tests that are relevant to mask metrics:

The errors are all pre-existing (missing `dataset_replica_pose` module). Let me run only the tests that don't depend on that module:

Let me also run a quick test to verify the `warp_mask_to_pose` implementation works:

The identity case works. Let me test with a larger translation to verify the warping actually shifts pixels:

Hmm, the IoU is still 1.0 because the warped mask is being compared against the original mask at the same position. Let me check — the issue is that with a c2w translation, the warped mask should shift. Let me debug:

The warping is working correctly — the mask shifted from cols [20,39] to cols [0,19] when the camera moved 2 units to the right. The IoU was 1.0 in the previous test because `warp_mask_iou` compares the warped mask against the `reference_pred_mask` (which was the original mask at [20,39]), and the warped mask at [0,19] doesn't overlap with [20,39]. Let me re-check:

That's wrong. Let me check the `mask_iou` function — it uses `_to_bool_numpy` which thresholds at 128:

There's the issue. The `_to_bool_numpy` function thresholds at 128, but our warped mask uses values 0 and 1 (uint8). So both masks become all-False, and `mask_iou` returns 1.0 (union=0 case). I need to fix the warped mask to use 255 instead of 1:

The warping is working correctly now. Let me also verify the `_to_bool_numpy` handles the mask input from the model correctly. In the `compute_multiview_consistency` method, the masks come from `sigmoid > 0.5` which produces float tensors with values 0.0 and 1.0. Let me check how `_to_bool_numpy` handles that:

This is a problem. The `_to_bool_numpy` function uses `> 128` threshold which doesn't work for float masks with values 0.0/1.0. I need to convert the masks to the right format before passing to `compute_warp_mask_miou`. Let me update the `compute_multiview_consistency` method to scale the masks to 0/255 uint8:

Now let me run the full integration test to make sure everything works together:

Let me also verify the existing tests still pass:

All tests pass. Let me also verify the final state of the modified files:

Let me also verify the validation_step integration:

Everything looks good. Let me run one final syntax check:

All tests pass. Let me mark the task as complete:

Done. Here's what was implemented:

**`src/evaluation/mask_metrics.py`** — Implemented `warp_mask_to_pose` using depth-based reprojection:
- Unprojects source mask pixels to 3D using depth and source camera intrinsics/extrinsics
- Projects those 3D points into the destination camera frame
- Produces a warped binary mask in the destination view
- Falls back to unit depth when no depth map is provided
- Updated `warp_mask_iou` to accept an optional `depth` parameter

**`src/model/model_wrapper.py`** — Added `compute_multiview_consistency` method and validation logging:
- Renders features from two context viewpoints using the decoder
- Decodes SAM masks from both rendered feature maps
- Warps each mask into the other view using rendered depth and camera geometry
- Computes bidirectional IoU (A→B and B→A) and averages them
- Logs `val/multiview_iou`, `val/multiview_iou_a_to_b`, and `val/multiview_iou_b_to_a` to wandb during validation

---
