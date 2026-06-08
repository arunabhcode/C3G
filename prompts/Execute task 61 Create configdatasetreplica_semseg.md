# Execute task: 6.1 Create `config/dataset/replica_semseg.yaml`

Exported: 6/8/2026, 12:28:57 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 6.1 Create `config/dataset/replica_semseg.yaml`

---

**Assistant:**

Now I have the pattern. Let me create the new config file:

Done. Created `config/dataset/replica_semseg.yaml` following the existing replica config pattern with:

- Dataset name and root path pointing to `/home/arunabh/replica_semseg`
- All 8 scenes listed
- Image shapes matching the design (256×256 input, 680×1200 original)
- View sampler and baseline settings matching the existing replica configs
- Prompted training fields: `prompt_strategy: centroid` and `min_object_pixels: 16`

---
