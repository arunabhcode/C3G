# Execute task: 11.1 Add NaN/Inf clamping to rendered feature maps

Exported: 6/8/2026, 12:24:38 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 11.1 Add NaN/Inf clamping to rendered feature maps

---

**Assistant:**

Let me find the feature rendering path in the model wrapper to understand where clamping should be added.

Now I can see the feature rendering path. The `output.feature` is the rendered feature map from Gaussian splatting. I need to add NaN/Inf clamping right after `output.feature` is accessed. Let me look at where `output.feature` is first produced/used and understand the flow better.

Now let me look at the training_step more closely to understand where the best place to add clamping is. The rendered features come from `output.feature` which is produced by the decoder. I should add clamping right after `output.feature` is accessed in the training step, before it's used for loss computation.

Let me also check if there's a `logging` import or how warnings are logged:

I see the project uses `print` for logging warnings (line 621 area) and `self.log` for metrics. Let me look at how warnings are typically handled in this codebase:

The codebase uses `print` for logging. Now let me look at the training step more carefully to understand where `output.feature` is produced and where the best place to add clamping is:

Now I have a clear picture. The `output.feature` is the rendered feature map from the decoder. I need to add NaN/Inf clamping after `output.feature` is accessed. The best approach is to add a helper method that clamps the features and logs warnings, then apply it wherever `output.feature` is used in the training step.

Looking at the training step, `output.feature` is used in two places:
1. Line 547: `gaussian_feature = output.feature` (for feature rendering loss)
2. Line 583: `if gt_masks is not None and output.feature is not None:` (for segmentation loss)

The cleanest approach is to clamp `output.feature` in-place right after the decoder produces it, before any downstream usage. Let me also check the validation step usage:

Now I have a complete understanding. I'll add the clamping right after the decoder forward call produces `output`, so it applies to all downstream uses (training, validation, test). The best place is a dedicated method that clamps `output.feature` in-place and logs a warning.

Let me look at the exact line where `output` is produced in training_step:

Now let me find a good place to add the clamping method. I'll add it as a method on `ModelWrapper` and call it right after the decoder forward in the training step. Let me find the class methods area:

Now I'll add the clamping method and integrate it into the training step. The approach:
1. Add a `_clamp_rendered_features` method that checks for NaN/Inf, clamps, and logs warnings
2. Call it right after `output` is produced in `training_step`

Now I need to call `_clamp_rendered_features` right after the decoder forward in the training step:

Now let me verify the changes look correct:

Now let me check if there are any diagnostics issues:

Clean. Let me run the existing tests to make sure nothing is broken:

There's a module import error unrelated to my changes (missing `dataset_replica_pose` module). Let me try a simpler test to verify the code at least parses correctly:

The file parses correctly. Let me also write a quick unit test to verify the clamping logic works:

The clamping logic works correctly. Let me verify the final state of the method one more time:

Now let me mark the task as completed:

Done. Here's what I added to `src/model/model_wrapper.py`:

1. A `_clamp_rendered_features` method that:
   - Short-circuits if `output.feature` is `None`
   - Detects NaN/Inf values using `torch.isfinite`
   - Logs a warning per affected view with the step number and view index
   - Replaces NaN/Inf with zeros (so they don't poison the clamp)
   - Clamps all values to `[-1e4, 1e4]`

2. A call to `self._clamp_rendered_features(output)` immediately after `self.decoder.forward(...)` in the training step, ensuring all downstream consumers (feature rendering loss, segmentation loss) receive clean feature maps.

Training continues uninterrupted since the method replaces bad values rather than raising exceptions.

---
