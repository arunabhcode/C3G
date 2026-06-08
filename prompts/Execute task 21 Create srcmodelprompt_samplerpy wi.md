# Execute task: 2.1 Create `src/model/prompt_sampler.py` with `PromptSampler` class

Exported: 6/8/2026, 12:27:30 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 2.1 Create `src/model/prompt_sampler.py` with `PromptSampler` class

---

**Assistant:**

Let me look at the existing dataset and model code to understand the patterns used in this project.

Now let me look at the SAM decoder to understand the coordinate space and how prompts are passed:

Now I have a clear picture. The SAM decoder expects `point_coords` of shape `(B, N, 2)` and `point_labels` of shape `(B, N)`. Let me also check the SAM constants to understand the image size:

Now I have everything I need. The SAM input coordinate space is [0, 1024]. Let me create the `PromptSampler` class:

Now let me verify there are no issues with the file:

Let me also run a quick import check to make sure it loads correctly:

Let me also run a quick functional test to verify the core logic works:

Everything works correctly. Let me also verify the error case when no valid masks exist:

All working. Let me mark the task as complete:

Done. Created `src/model/prompt_sampler.py` with the `PromptSampler` class implementing:

- `__init__(strategy, min_object_pixels, image_size)` — stores config, defaults to centroid strategy with SAM's 1024 image size
- `sample(binary_masks)` — picks a random valid mask (≥ min_object_pixels), generates a point prompt, returns `(point_coords, point_labels, selected_gt_mask)`
- `compute_centroid(mask)` — mean row/col of foreground pixels rounded to nearest int
- `sample_random_point(mask)` — uniform random foreground pixel
- Coordinates are normalized to SAM input space `[0, image_size]` with `(x, y)` = `(col_scaled, row_scaled)` ordering matching SAM's expected format
- Masks with fewer than `min_object_pixels` foreground pixels are filtered out before sampling

---
