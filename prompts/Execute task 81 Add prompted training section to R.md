# Execute task: 8.1 Add prompted training section to README or create `docs/prompted_training.md`

Exported: 6/8/2026, 12:31:52 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 8.1 Add prompted training section to README or create `docs/prompted_training.md`

---

**Assistant:**

Let me look at the existing project structure to understand the config patterns and any existing README.

Now let me look at the training config files to understand the exact config patterns used:

Let me also check where checkpoints are saved and how evaluation works:

Now I have enough context. Let me create the documentation file:

Created `docs/prompted_training.md` covering all the required documentation:

- **Prerequisites** — environment setup with `uv`, SAM checkpoint download, VGGT weights, and dataset path/structure
- **Training command** — exact `uv run` command with config overrides for dataset root, prompt strategy, and wandb
- **Evaluation command** — segment-everything mode eval with checkpoint loading
- **Expected outputs** — checkpoint save location, eval visualization paths, and wandb metrics logged during training

---
