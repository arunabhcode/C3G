# Execute task: 7.1 Integrate segmentation loss into training loop in `src/model/model_wrapper.py`

Exported: 6/8/2026, 12:22:25 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 7.1 Integrate segmentation loss into training loop in `src/model/model_wrapper.py`

---

**Assistant:**

The `TrainCfg` dataclass already has the fields I need to add (`segmentation_loss_weight`, `sam_model_variant`, `sam_checkpoint`, `use_lora`, `lora_rank`). They were added in a previous task. Now let me look at the `training_step` method to understand where to integrate the segmentation loss.

Now let me look at the existing segmentation loss integration. I can see there's already a block at line ~380 that attempts to compute segmentation loss, but it's using `self.losses[0]` which is incorrect — it should use the SAM mask decoder. Let me also check what `self.losses` contains and how the `SegmentationLoss` class works.

Now I have a clear picture. Looking at the current `training_step`, there's already a segmentation loss block (around line 380) but it's incorrectly using `self.losses[0]` — which is just the first loss in the loss list (likely the reconstruction loss), not the segmentation loss. 

The correct approach is to:
1. Store a `SAMMaskDecoderWrapper` instance on the model (or use a dedicated `LossSegmentation` instance)
2. Compute the segmentation loss properly using the rendered features and the SAM mask decoder

Let me check how the losses are configured and passed in from `main.py`:

Now let me look at how `get_losses` works to understand the loss configuration:

Now I understand the full picture. The `LossSegmentation` is already registered as a configurable loss. The current implementation in `training_step` has a segmentation loss block, but it's incorrectly using `self.losses[0]` instead of finding the actual `LossSegmentation` instance from the losses list.

The proper approach is:
1. Find the `LossSegmentation` instance from `self.losses` (if configured)
2. Compute segmentation loss using it with the rendered features
3. The segmentation loss should NOT be computed in the main loss loop (since it needs special handling with `output.feature`)

Let me also check if there's a `masks` field expected in the batch:

Now I understand the full picture. The `LossSegmentation` class handles everything internally — it takes the rendered features from `prediction.feature`, runs the SAM mask decoder, and computes the loss. So it can be used through the standard loss loop.

However, the current code has a problem: the segmentation loss block at line ~380 uses `self.losses[0]` which is wrong. The correct approach is to either:
1. Let the segmentation loss be computed in the main loss loop (since `LossSegmentation.forward` already handles everything), OR
2. Compute it separately with proper identification

Looking more carefully at the `LossSegmentation.forward()`, it accesses `batch["target"]["masks"]` which might not always be present. The loss should handle missing masks gracefully. Also, the segmentation loss should be separate from the main loss loop because:
- It needs `output.feature` which is the rendered features
- The main losses use `output.color` for reconstruction
- The weight is applied differently (via `segmentation_loss_weight` in TrainCfg, not the loss's own weight)

Wait, looking again at `LossSegmentation`, it already applies `self.cfg.weight` internally. So there's a design choice: either the weight comes from the loss config or from `TrainCfg.segmentation_loss_weight`. The current code applies both, which would double-weight it.

Let me re-examine the current segmentation block more carefully. The issue is that `self.losses[0]` is just the first loss (likely MSE or LPIPS), not the segmentation loss. I need to properly find and use the segmentation loss instance.

The cleanest approach based on the design doc: store a reference to the `LossSegmentation` instance (if present) and call it separately, OR instantiate a dedicated `SAMMaskDecoderWrapper` on the model. Since `LossSegmentation` already wraps the mask decoder, let me use it properly.

Let me now fix the `training_step` to properly integrate the segmentation loss:

Now I need to understand the proper fix. The segmentation loss should find the `LossSegmentation` instance from `self.losses`. Let me also check if `LossSegmentation` is meant to be included in the main loss loop or handled separately:

Looking at the design, the segmentation loss should be computed **separately** from the main loss loop because:
1. It needs `output.feature` (rendered features) specifically
2. The `segmentation_loss_weight` in `TrainCfg` controls its contribution
3. It should only run when `segmentation_loss_weight > 0` AND ground-truth masks are available

The `LossSegmentation` already has its own internal weight (`self.cfg.weight`). To avoid double-weighting, the approach should be:
- The `LossSegmentation` instance should NOT be in the main loss loop (it would be called separately)
- OR it should be in the main loss loop and the `segmentation_loss_weight` block should be removed

Looking at the design doc more carefully, the segmentation loss is meant to be a **separate** computation from the standard losses. The `TrainCfg.segmentation_loss_weight` controls it. So the proper approach is:

1. Store a dedicated `LossSegmentation` (or `SAMMaskDecoderWrapper`) on the model, separate from `self.losses`
2. Call it in the segmentation loss block

But wait — looking at `get_losses` and how it's configured, if `LossSegmentation` is in the loss config, it would be in `self.losses` and computed in the main loop. That's a conflict.

The cleanest solution matching the design: the segmentation loss block should find the `LossSegmentation` from `self.losses` and skip it in the main loop, OR we instantiate a separate `SAMMaskDecoderWrapper` on the model.

Let me look at how the loss config works to see if segmentation loss would be in the main loop:

Now I understand the loss configuration system. Losses are configured via Hydra config groups (e.g., `override /loss: [mse, lpips]`). The segmentation loss would be added as another loss config option. 

Given the architecture, the cleanest approach for the segmentation loss integration is:
1. The `LossSegmentation` can be included in the loss list via config (like `[mse, lpips, segmentation]`)
2. BUT the current `training_step` code has a separate block that uses `self.train_cfg.segmentation_loss_weight > 0` as a gate

The issue is that if `LossSegmentation` is in `self.losses`, it would be called in the main loop AND in the separate block. The design intent is clear: the segmentation loss should be computed **separately** from the main reconstruction losses, controlled by `segmentation_loss_weight`.

The best approach: Don't put `LossSegmentation` in the main loss list. Instead, instantiate it directly on the `ModelWrapper` when `segmentation_loss_weight > 0`. This matches the pattern of how `feature_rendering_loss` is handled (it's computed separately, not through the loss list).

Let me now implement the fix. I need to:
1. Import `LossSegmentation` in `model_wrapper.py`
2. In `__init__`, create a `LossSegmentation` instance when `segmentation_loss_weight > 0`
3. Fix the segmentation loss block in `training_step` to use this dedicated instance
4. Handle missing masks gracefully

Now let me look at the `LossSegmentation` import and the `SAMMaskDecoderWrapper`:

Now I have a complete understanding. The current segmentation loss block in `training_step` uses `self.losses[0]` which is wrong. I need to:

1. Import `LossSegmentation` and `LossSegmentationCfg`
2. In `__init__`, create a dedicated `self.segmentation_loss` instance when `segmentation_loss_weight > 0`
3. Fix the segmentation loss block to use `self.segmentation_loss` and handle missing masks

Let me also check if there's a `"masks"` key expected in the batch data:

Good — `masks` isn't in the dataset types yet, so the segmentation loss needs to handle missing masks gracefully. Now let me implement the changes:

Now let me add the segmentation loss initialization in `__init__`. I need to find the right place after the existing model setup:

Now let me fix the segmentation loss block in `training_step`. The current code incorrectly uses `self.losses[0]`. I need to replace it with proper logic that:
1. Checks if `self.segmentation_loss` exists
2. Gets rendered features from `output.feature`
3. Gets ground-truth masks from batch (with graceful handling if missing)
4. Runs the mask decoder and computes BCE + dice loss

Now let me verify the file compiles correctly by checking for diagnostics:

All the diagnostics are pre-existing warnings (unused imports, f-strings without placeholders, etc.) — none are errors related to my changes. Let me verify the specific area I modified looks correct:

Now let me verify the `__init__` section looks correct:

Let me also verify the import was added correctly:

Now let me run the existing tests to make sure nothing is broken:

The import fails due to CUDA not being available in this environment (it's trying to compile CUDA extensions). This is expected for a GPU-dependent project. Let me verify the syntax is correct by checking just the model_wrapper file in isolation:

Let me also run the linter on the file:

You're right. The `LossSegmentation` class already does everything — it wraps the mask decoder, computes BCE + dice, handles the feature reshaping. I should just instantiate it and call its `forward()` directly. Let me fix this properly.

Now fix the `__init__` to instantiate `LossSegmentation` instead of `SAMMaskDecoderWrapper`:

Now fix the `training_step` block to just call `self.segmentation_loss.forward()`:

Now note that `LossSegmentation.forward()` already applies `self.cfg.weight` internally (which we set to `segmentation_loss_weight`), so we don't multiply again. The loss is already weighted. Let me verify the final state:

Wait — there's a problem with the ordering. The `total_loss` is logged *before* the segmentation loss is added to it. That means the logged `loss/total` won't include the segmentation loss. Let me fix the ordering so the total loss is logged after all losses are accumulated:

I need to move the `loss/total` log to after the segmentation loss block. Let me do that:

Now let me verify the final result looks correct:

Let me also verify the `__init__` section:

Now let me verify syntax is still valid:

The only error is a pre-existing line-too-long on line 1370, unrelated to my changes. Everything looks good.

Let me also check that the `LossSegmentation` won't double-apply the weight. Looking at its `forward()`, it returns `self.cfg.weight * (bce + dice)`. Since I set `weight=self.train_cfg.segmentation_loss_weight` in the config, and then in `training_step` I do `total_loss = total_loss + segmentation_loss` (no additional multiplication), the weight is applied exactly once. That's correct.

All pre-existing warnings, no new issues.

Done. Here's what I did:

1. **Import**: Added `LossSegmentation`, `LossSegmentationCfg`, and `LossSegmentationCfgWrapper` from `src/loss/loss_segmentation.py`.

2. **`__init__`**: When `segmentation_loss_weight > 0` and a SAM checkpoint is configured, instantiates a `LossSegmentation` with the config from `TrainCfg` fields. Stored as `self.segmentation_loss`.

3. **`training_step`**: After the feature rendering loss block, checks if segmentation loss is enabled and ground-truth masks are present in the batch. If so, calls `self.segmentation_loss.forward(output, batch, gaussians, global_step)` which handles everything internally (feature reshaping, mask decoder inference, BCE + dice computation, weight application). The result is added to `total_loss` and logged to wandb as `loss/segmentation`.

4. **Ordering fix**: Moved the `loss/total` log to *after* the segmentation loss is accumulated, so the logged total reflects all loss components.

The `TrainCfg` fields (`segmentation_loss_weight`, `sam_model_variant`, `sam_checkpoint`, `use_lora`, `lora_rank`) were already present from a previous task.

---
