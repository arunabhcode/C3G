# Execute task: 6.2 Create `config/training/feature_head_sam_prompted.yaml`

Exported: 6/8/2026, 12:29:17 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 6.2 Create `config/training/feature_head_sam_prompted.yaml`

---

**Assistant:**

Now I have the pattern. Let me also check the dataset config for `replica_semseg` to make sure the reference is correct:

Good. Now I'll create the prompted training config following the pattern of `feature_head_sam.yaml`:

Done. Created `config/training/feature_head_sam_prompted.yaml` following the pattern of `feature_head_sam.yaml` with:

- `prompt_mode: prompted` to activate prompted training
- `prompted_seg_loss_weight: 1.0`, `prompt_strategy: centroid`, `min_object_pixels: 16` for prompted loss config
- SAM params retained (`sam_model_variant`, `sam_checkpoint`, `use_lora`, `lora_rank`)
- References the `replica_semseg` dataset config
- Same model/encoder/optimizer/trainer structure as the existing SAM config

---
