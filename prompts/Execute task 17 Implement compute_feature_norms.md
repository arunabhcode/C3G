# Execute task: 1.7 Implement `compute_feature_norms`

Exported: 6/8/2026, 12:34:06 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 1.7 Implement `compute_feature_norms`

---

**Assistant:**

Let me verify the implementation works:

Done. Added `compute_feature_norms` to `src/model/debug_visualizer.py` — it computes the L2 norm along the channel dimension (dim=0) and flattens the spatial dimensions into a vector of length H×W. Verified it produces the correct shape (4096 for a 64×64 feature map) with all non-negative values.

---
