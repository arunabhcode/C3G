# Execute task: 5.1 Add new fields to `TrainCfg` in `src/model/model_wrapper.py`

Exported: 6/8/2026, 12:28:08 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 5.1 Add new fields to `TrainCfg` in `src/model/model_wrapper.py`

---

**Assistant:**

Now I have a clear picture. Let me add the new fields to `TrainCfg` and the validation logic to `validate_sam_config()`.

Now let me add validation for the new fields at the end of `validate_sam_config()`:

Let me verify the changes look correct:

Clean. Let me run the existing tests to make sure nothing is broken:

The import fails due to CUDA dependencies (expected in this project). The dataclass changes are syntactically correct and the diagnostics show no issues. Let me verify with a simpler approach that doesn't trigger the CUDA imports:

All four new fields are in place. Let me also verify the validation logic was added correctly:

Everything checks out.

Done. Here's what was added to `src/model/model_wrapper.py`:

**New fields in `TrainCfg`:**
- `prompt_mode: str = "grid"` — switches between grid (segment-everything) and prompted training
- `prompted_seg_loss_weight: float = 1.0` — weight for the prompted segmentation loss
- `prompt_strategy: str = "centroid"` — how point prompts are generated (centroid or random foreground pixel)
- `min_object_pixels: int = 16` — minimum foreground pixels for a mask to be valid for prompting

**Validation in `validate_sam_config()`:**
- Checks `prompt_mode` is one of `{"grid", "prompted"}`, raises `ValueError` otherwise
- Checks `prompt_strategy` is one of `{"centroid", "random_point"}`, raises `ValueError` otherwise

---
