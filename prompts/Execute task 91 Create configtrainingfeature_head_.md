# Execute task: 9.1 Create `config/training/feature_head_sam.yaml`

Exported: 6/8/2026, 12:23:17 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 9.1 Create `config/training/feature_head_sam.yaml`

---

**Assistant:**

Now I'll also check if there's already a `config/training/feature_head_sam.yaml` file:

Done. Created `config/training/feature_head_sam.yaml` following the structure of the existing LSeg config with these SAM-specific changes:

- `reproj_model: sam` and `gaussian_feature_dim: 256`
- `feature_rendering_loss: 0.01`
- `segmentation_loss_weight: 1.0`
- `sam_model_variant: sam_vit_h` and `sam_checkpoint: './pretrained_weights/sam_vit_h.pth'`
- `use_lora: false` and `lora_rank: 4`
- `feature_detach: false` to allow gradients through geometry

---

**User:**

make the config so that we can run it with scannet not re10k

---

**Assistant:**

Done. Updated `config/training/feature_head_sam.yaml` to use the ScanNet dataset instead of RE10K — switched the dataset default to `scannet`, updated wandb name/tags, and pointed the dataset override section at `scannet`.

---
