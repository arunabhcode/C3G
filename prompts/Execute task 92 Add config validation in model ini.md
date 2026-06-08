# Execute task: 9.2 Add config validation in model initialization

Exported: 6/8/2026, 12:23:32 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 9.2 Add config validation in model initialization

---

**Assistant:**

I can see that `load_sam_encoder` already has validation for both the model variant and checkpoint path. Now let me check where this is called from to see if there's a model initialization path that should also validate the config earlier (e.g., in `ModelWrapper.__init__` or `main.py`).

Now let me look at `src/main.py` to see how `load_foundation_model` is called and where the model is initialized:

Now I have a clear picture. The `load_sam_encoder` function already validates both the model variant and checkpoint path. However, looking at the task requirements, the validation should happen at model initialization time — which it does since `load_foundation_model` is called during initialization. But the current validation only triggers when `'sam' in cfg.train.reproj_model`. 

Let me also check if there's a case where the `sam_checkpoint` and `sam_model_variant` are used for the segmentation loss path without going through `load_foundation_model`:

Now I have the full picture. Both `load_sam_encoder` and `SAMMaskDecoderWrapper.__init__` already validate the checkpoint path and model variant. However, the task asks for validation at model initialization time — specifically in the `ModelWrapper.__init__` where the config is first processed. This provides early, clear error messages before any heavy model loading begins.

The current flow is:
1. `load_foundation_model()` validates when `reproj_model` contains "sam" (encoder path)
2. `SAMMaskDecoderWrapper.__init__` validates when segmentation loss is created (decoder path)

But there's a gap: if someone sets `segmentation_loss_weight > 0` with an invalid `sam_checkpoint` or `sam_model_variant`, the error only surfaces deep in the `SAMMaskDecoderWrapper` constructor. And if `reproj_model` doesn't contain "sam" but segmentation loss is configured, there's no early validation.

I'll add config validation in `ModelWrapper.__init__` that runs before any heavy loading, providing clear upfront error messages:

Now I need to add the `_validate_sam_config` method. Let me find a good place to add it — right after `__init__`:

Now let me verify the file has no diagnostics issues:

All warnings are pre-existing (unused imports, f-strings without placeholders, etc.) — none related to my changes. Let me verify the validation method looks correct by reading it back:

The validation method is in place. Let me also verify the call site:

The validation runs early in `__init__`, before the heavier `SAMMaskDecoderWrapper` and segmentation loss setup. Now let me run the existing tests to make sure nothing is broken:

All 33 tests pass. Let me mark the task as complete:

Done. I added a `_validate_sam_config` method to `ModelWrapper.__init__` that runs early during initialization — before the heavier SAM decoder/loss objects are constructed. It:

1. Checks if SAM is needed (either `reproj_model` contains "sam" or `segmentation_loss_weight > 0`)
2. Validates `sam_model_variant` is one of the three supported variants (`sam_vit_h`, `sam_vit_l`, `sam_vit_b`)
3. Validates `sam_checkpoint` is specified and the file actually exists on disk

Each failure raises a clear error (`ValueError` or `FileNotFoundError`) with an actionable message. All 33 existing tests continue to pass.

---
