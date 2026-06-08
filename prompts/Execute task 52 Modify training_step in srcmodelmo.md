# Execute task: 5.2 Modify `training_step()` in `src/model/model_wrapper.py` to use prompted loss

Exported: 6/8/2026, 12:28:33 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 5.2 Modify `training_step()` in `src/model/model_wrapper.py` to use prompted loss

---

**Assistant:**

Now let me read the training_step method to understand the existing segmentation loss integration:

Now let me look at the existing `LossSegmentationPrompted` class to understand its interface:

Now let me look at the imports at the top of model_wrapper.py:

Now I have a clear picture. I need to:

1. Import `LossSegmentationPrompted`, `LossSegmentationPromptedCfg`, and `LossSegmentationPromptedCfgWrapper`
2. Add prompted loss initialization in `__init__` (similar to the existing `segmentation_loss` pattern)
3. Add prompted loss computation in `training_step()` when `prompt_mode == "prompted"`

Let me also check the `_clamp_rendered_features` method reference and the output structure:

The code uses `self._clamp_rendered_features(output)` but the method is defined as `clamp_rendered_features`. This is likely an alias or the code already works. Let me not worry about that — it's existing code.

Now I have everything I need. Let me implement the changes:

Now let me add the prompted loss initialization in `__init__`, right after the existing `segmentation_loss` initialization block:

Let me re-read the exact text around the segmentation_loss initialization:

Now let me add the prompted loss computation in `training_step()`. I need to add it after the existing segmentation loss block and before the total_loss logging. Let me re-read the relevant section with the new line numbers:

Now I'll add the prompted loss block after the existing segmentation loss block and before the total_loss logging:

Now let me verify the changes look correct by reading the relevant section:

Let me also verify the `__init__` section looks correct:

Good. Now let me also check the imports section to make sure it looks right:

Now let me run diagnostics to check for any issues:

Let me also verify the `LossSegmentationPrompted` class properly handles the `weight` application (since the task says to add `prompted_seg_loss_weight * loss` to total_loss). Looking at the `forward` method of `LossSegmentationPrompted`, it already applies `self.cfg.weight * (total_loss / valid_count)` at the end, and we pass `self.train_cfg.prompted_seg_loss_weight` as the `weight` in the config. So the weight is already applied inside the loss function — we just add the result directly to `total_loss`.

Let me also verify the `validate_sam_config` method still works correctly with the new prompted mode validation:

I notice the `validate_sam_config` only runs when `"sam" in self.train_cfg.reproj_model or self.train_cfg.segmentation_loss_weight > 0`. But in prompted mode, the user might have `segmentation_loss_weight = 0` (using only prompted loss, not grid-based). The validation should also trigger when `prompt_mode == "prompted"`. Let me fix this:

Now let me run a quick syntax check to make sure the file is valid Python:

Let me also verify the import resolves correctly:

The import fails due to CUDA dependencies (not available in this environment), but that's unrelated to our changes. The syntax is valid and the import path is correct. Let me verify the module structure is correct by checking the loss `__init__.py`:

The `LossSegmentationPrompted` doesn't need to be in the `LOSSES` dict because it's not instantiated through the generic `get_losses` mechanism — it's directly instantiated in `ModelWrapper.__init__` (similar to how `self.segmentation_loss` is also directly instantiated there rather than going through `get_losses`). The import in `model_wrapper.py` uses the relative path `..loss.loss_segmentation_prompted` which will work fine.

Let me run the existing tests to make sure nothing is broken:

All existing tests pass. Let me also run a quick check on the other test files:

All 49 tests pass. Let me run diagnostics one more time to confirm no issues:

Everything is clean. Let me mark the task as complete:

Done. Here's what I implemented in `src/model/model_wrapper.py`:

1. **Added import** for `LossSegmentationPrompted`, `LossSegmentationPromptedCfg`, and `LossSegmentationPromptedCfgWrapper`.

2. **Added prompted loss initialization in `__init__`** — when `prompt_mode == "prompted"` and a SAM checkpoint is configured, it instantiates `LossSegmentationPrompted` with the relevant config (weight, checkpoint, model variant, LoRA settings, prompt strategy, min object pixels). Otherwise sets it to `None`.

3. **Added prompted loss computation in `training_step()`** — after the existing grid-based segmentation loss block, a new block checks if `prompt_mode == "prompted"`, the prompted loss module exists, and features are available. It then checks for target label maps in the batch, computes the prompted loss, logs it as `loss/prompted_segmentation`, and adds it to `total_loss`.

4. **Updated `validate_sam_config()`** — added `self.train_cfg.prompt_mode == "prompted"` to the early-return guard so validation also triggers when using prompted mode (even if `segmentation_loss_weight` is 0).

---

**User:**

wait are you logging loss from single segmentation or multiple segmentations?

---

**Assistant:**

The prompted loss logs a **single scalar** — it's the average loss across all valid (batch, view) pairs. Inside `LossSegmentationPrompted.forward()`, it iterates over all B×V target views, accumulates the best-of-3 loss for each, then returns `weight * (total_loss / valid_count)`. So what gets logged as `loss/prompted_segmentation` is one averaged value representing the prompted segmentation loss across all target views in the batch.

The existing grid-based `loss/segmentation` works the same way — single scalar for the batch.

Is that what you were asking, or are you wondering whether it should log per-view losses separately?

---
