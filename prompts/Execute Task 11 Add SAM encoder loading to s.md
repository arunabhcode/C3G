# Execute Task: 1.1 Add SAM encoder loading to `s...

Exported: 6/8/2026, 1:01:37 AM

---

**User:**

Execute Task: 1.1 Add SAM encoder loading to `src/model/load_foundation_model.py`

---

**Assistant:**

Now I understand the pattern. The function loads various foundation models and returns them alongside a `feature_dim`. I need to add a SAM encoder branch and include it in the return value. Let me also check how this function's return value is consumed.

Now let me check the ModelWrapper constructor to understand how it accepts foundation models:

Now I have a clear picture. Let me also check if there's a `segment_anything` package available or how SAM is typically loaded in this project:

I notice `segment-anything` is not in the dependencies. Let me check if there's a SAM-related package or if the project uses a local SAM implementation:

Now I have all the context I need. The `segment-anything` package needs to be added as a dependency, and I need to add the SAM encoder loading branch. Let me implement this:

Now add the SAM branch before the `maskclip` branch and update the return statement:

Now I can see the exact formatting. Let me apply the changes:

Now update the return statement to include `sam_encoder`:

Now I need to update `src/main.py` to unpack the new return value:

Now let me also add `segment-anything` to the project dependencies:

Now let me verify the final state of the modified file:

Now let me check the diagnostics:

Clean. Let me also verify the main.py change looks correct in context:

Everything looks good. Let me now mark the task as completed:

Done. Here's what was implemented:

**`src/model/load_foundation_model.py`:**
- Added `SAM_MODEL_REGISTRY` mapping variant names (`sam_vit_h`, `sam_vit_l`, `sam_vit_b`) to SAM's internal type strings
- Created `load_sam_encoder()` function that validates the variant and checkpoint path, loads the SAM model via `segment_anything.sam_model_registry`, extracts the image encoder, freezes all parameters, and returns it with `feature_dim=256`
- Added `elif "sam" in cfg.train.reproj_model:` branch in `load_foundation_model()` that reads `sam_model_variant` and `sam_checkpoint` from config (with sensible defaults)
- Extended the return tuple to include `sam_encoder` as a new slot

**`src/main.py`:**
- Updated the unpacking of `load_foundation_model()` to include the new `sam_encoder` return value

**`pyproject.toml`:**
- Added `segment-anything` as a git dependency from Facebook's official repository

---
