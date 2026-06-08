# Execute task: 3.1 Implement `log_debug_visualizations` orchestration function

Exported: 6/8/2026, 12:37:17 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 3.1 Implement `log_debug_visualizations` orchestration function

---

**Assistant:**

Let me read the existing debug visualizer module and the relevant utilities to understand the current state.

Now let me check how `WandbLogger` is imported in the codebase:

Now let me check how `inverse_normalize` is imported in `model_wrapper.py`:

Now I have all the context I need. Let me also check how `wandb` is used for logging tables and histograms in the codebase:

Now I have a clear picture. The `WandbLogger` wraps the wandb experiment. To log tables and histograms, I'll need to access `logger.experiment` which gives the underlying `wandb.run`. Let me now implement the `log_debug_visualizations` function:

Now let me verify the full file looks correct:

The file looks good. Let me check if there are any actual import errors or issues by running diagnostics:

These are just style warnings (import sorting and line length on the colorize_heatmap docstring). Let me fix them:

Clean. Now let me verify the module can be imported without errors:

Let me also run a quick smoke test to verify the orchestration function works end-to-end with mocked wandb:

Everything works correctly. Let me also run the existing tests to make sure nothing is broken:

No test file exists yet (that's for optional tasks 1.2, 1.3, etc.). The implementation is complete and working.

Done. The `log_debug_visualizations` orchestration function is implemented in `src/model/debug_visualizer.py`. Here's what it does:

1. **Gates on checkpoint frequency** — returns early if `global_step % checkpoint_interval != 0`
2. **Checks logger type** — returns early if not a `WandbLogger`
3. **Checks rendered feature** — returns early if `None`
4. **Computes PCA** for both target SAM and rendered features using `run_pca`, resized to target RGB dimensions
5. **Computes MSE heatmap and cosine error map**, resizes to target RGB dimensions via bilinear interpolation, colorizes with viridis, and alpha-blends onto the inverse-normalized target RGB
6. **Builds a `wandb.Table`** with the five required columns and logs it as `"val/debug_visualizations"`
7. **Logs feature norm histogram** and scalar stats (min, max, std) for both target and rendered features

---
