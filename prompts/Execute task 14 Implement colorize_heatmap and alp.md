# Execute task: 1.4 Implement `colorize_heatmap` and `alpha_blend`

Exported: 6/8/2026, 12:33:53 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 1.4 Implement `colorize_heatmap` and `alpha_blend`

---

**Assistant:**

Now let me check if there's an existing color map utility mentioned in the design doc:

I have everything I need. The design says to use viridis colormap. I can leverage `apply_color_map_to_image` from the existing utility, but the design specifies the function should live in `debug_visualizer.py` as a self-contained pure function. I'll implement it directly using matplotlib's viridis colormap to keep it simple and avoid an extra dependency on the visualization module.

Now let me verify the file looks correct:

Let me run a quick smoke test to make sure the functions work:

All tests pass. Let me also check for any diagnostics:

Clean. Let me mark the task as complete:

Done. Both functions are implemented in `src/model/debug_visualizer.py`:

- **`colorize_heatmap(heatmap, vmin=0.0, vmax=None)`** — normalizes the input to [0, 1] using vmin/vmax (defaults vmax to the heatmap's max), applies the viridis colormap via matplotlib, and returns a `(3, H, W)` tensor in [0, 1].

- **`alpha_blend(overlay, background, alpha=0.5)`** — simple linear blend `alpha * overlay + (1 - alpha) * background`, returning `(3, H, W)` in [0, 1] when both inputs are in that range.

---
