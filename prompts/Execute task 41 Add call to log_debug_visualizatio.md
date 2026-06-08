# Execute task: 4.1 Add call to `log_debug_visualizations` at end of `validation_step` in `src/model/model_wrapper.py`

Exported: 6/8/2026, 12:37:31 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 4.1 Add call to `log_debug_visualizations` at end of `validation_step` in `src/model/model_wrapper.py`

---

**Assistant:**

Now I can see the structure. The feature rendering visualization block is inside the `if self.train_cfg.feature_rendering_loss > 0:` condition, ending with the PCA comparison logging. I need to add the `log_debug_visualizations` call after that block. Let me also check the imports at the top of the file.

Now I have a clear picture. I need to:
1. Add the import for `log_debug_visualizations`
2. Add the call inside the `if self.train_cfg.feature_rendering_loss > 0:` block, after the PCA comparison logging

Let me look at the exact location where the PCA logging ends and what variables are available:

Now I understand the structure. I need to look at what variables are available. From the design doc:
- `target_sam_feature` comes from `foundation_feature[:, CV:]` for target views
- `rendered_feature` comes from `gaussian_feature` for target views

Let me check how `foundation_feature` is structured - specifically the context/target split:

Now I understand the data flow:
- `foundation_feature` has shape `(B, CN+N, C, H, W)` — context views concatenated with target views
- `gaussian_feature` has shape `(B, CN+N, C, H, W)` — context + target rendered features
- The number of context views is `CN`, target views is `N`

The target SAM feature would be `foundation_feature[0, CN:]` (the target portion), and the rendered feature for targets is `output.feature[0]` (before the concatenation with context).

Let me also check the `debug_visualizer.py` to see the exact signature:

The function expects:
- `target_rgb`: a single image tensor (3, H, W)
- `target_sam_feature`: a single feature tensor (C, H, W) 
- `rendered_feature`: a single feature tensor (C, H, W) or None
- `img_size`: tuple (h, w)

Now let me look at the variables available in the validation_step. The target RGB is `batch["target"]["image"][0]` which has shape `(N, 3, H, W)` — I need the first target view. The `foundation_feature` contains context+target, and `output.feature` is the rendered feature for targets.

Let me check how many context views there are:

From the code:
- `context_gaussian_feature` has shape `(B, CN, C, H, W)` 
- `gaussian_feature = output.feature` has shape `(B, N, C, H, W)` where N is target views

So `foundation_feature` is computed from `torch.cat((batch["context"]["image"], batch["target"]["image"] * 2 - 1), dim=1)` — it has `CN + N` views. The target SAM features start at index `CN`.

For the first target view:
- `target_rgb` = `batch["target"]["image"][0, 0]` → (3, H, W)
- `target_sam_feature` = `foundation_feature[0, CN]` → (C, 64, 64) where CN is the number of context views
- `rendered_feature` = `output.feature[0, 0]` → (C, H, W) — the first target view's rendered feature

Now I'll make the changes:

Now add the call to `log_debug_visualizations` at the end of the feature rendering visualization block (after the PCA comparison logging, still inside the `if self.train_cfg.feature_rendering_loss > 0:` block):

Now let me verify the changes look correct:

Let me also verify the import was added correctly:

Now let me check for any diagnostics:

Clean. Let me also quickly verify the import path resolves correctly by checking the module structure:

Everything is in order. Let me mark the task as complete:

Done. I added two changes to `src/model/model_wrapper.py`:

1. **Import** — added `from .debug_visualizer import log_debug_visualizations` alongside the existing utils import.

2. **Call** — at the end of the `if self.train_cfg.feature_rendering_loss > 0:` block (after PCA comparison logging), added the call passing:
   - `self.logger`
   - `self.global_step`
   - `get_cfg()["checkpointing"]["every_n_train_steps"]`
   - `batch["target"]["image"][0, 0]` — first target view RGB
   - `foundation_feature[0, CN]` — target SAM feature (skipping context views)
   - `output.feature[0, 0]` — rendered feature for first target view
   - `(h, w)` — image size

The function handles frequency gating, logger type checking, and None-feature skipping internally.

---
