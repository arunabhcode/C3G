# Execute task: 1.1 Create `src/model/debug_visualizer.py` with `compute_mse_heatmap` and `compute_cosine_error_map`

Exported: 6/8/2026, 12:33:36 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 1.1 Create `src/model/debug_visualizer.py` with `compute_mse_heatmap` and `compute_cosine_error_map`

---

**Assistant:**

Let me look at the existing code structure to understand the patterns used in this project.

Now let me look at a few more files to understand the coding style better:

Now I have a clear picture of the project structure and coding style. Let me create the `debug_visualizer.py` module:

Let me verify there are no issues with the file:

Let me run a quick sanity check to make sure the functions work correctly:

All checks pass. Let me mark the task as complete:

Done. Created `src/model/debug_visualizer.py` with both functions:

- `compute_mse_heatmap(rendered_feature, target_feature)` — computes `((a - b) ** 2).mean(dim=0)`, returning an (H, W) tensor of per-pixel MSE values (always non-negative).
- `compute_cosine_error_map(rendered_feature, target_feature)` — L2-normalizes both inputs along the channel dimension (with epsilon=1e-8 clamping for zero-norm safety), computes cosine similarity, and returns `1 - cosine_similarity` as an (H, W) tensor.

Both functions handle edge cases correctly — identical inputs produce zeros, and zero-norm inputs don't produce NaN.

---
