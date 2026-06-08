# add the same debug visualizations to the training...

Exported: 6/8/2026, 1:14:26 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

add the same debug visualizations to the training data, like images, hist etc. right now they are only in validation

---

**Assistant:**

Now let me look at the training_step to see where to add it:

Now let me look at how the validation step calls it to match the pattern:

The `log_debug_visualizations` function uses `"val/"` prefix hardcoded. I need to either make it configurable or create a training-specific version. Let me check the function again — it logs to `"val/debug_visualizations"`, `"val/target_feature_norms"`, etc. I'll add a `prefix` parameter:

Now add the call in `training_step`:

Done. Training will now log the same debug visualizations (PCA images, MSE/cosine heatmap overlays, feature norm histograms and stats) under the `train/` prefix at every checkpoint interval. The validation ones continue logging under `val/` as before.

---

**User:**

ensure train also logs at checkpoint freq

---

**Assistant:**

It already does. The `log_debug_visualizations` function has this check at the top:

```python
if global_step % checkpoint_interval != 0:
    return
```

And the training call passes `get_cfg()["checkpointing"]["every_n_train_steps"]` as the `checkpoint_interval` — same as validation. So it only logs at checkpoint frequency (every 50 steps per your config).

---

**User:**

rendered_feature_pca for train is always zero even though it is not for validation, wtf

---

**Assistant:**

The issue is that in training, `rendered_interp` has shape `(b, v_total, c, fh, fw)` where `v_total` includes both target AND context views (because `context_view_loss=True`). You're passing `rendered_interp[0, 0]` which is the first target view — that should be fine.

But wait — the real problem is likely that `log_debug_visualizations` is being called inside the `if self.global_rank == 0 and (batch_idx + 1) % accum == 0:` block. With `accumulate_grad_batches=2`, this only fires on odd `batch_idx` values. But `global_step` is 0 at the first checkpoint interval check, which passes the `global_step % checkpoint_interval == 0` gate.

Actually, the real issue is simpler. Look at when it fires: `(batch_idx + 1) % accum == 0`. With `accum=2`, this fires at `batch_idx=1, 3, 5...` But `global_step=0` at `batch_idx=1`. The checkpoint interval is 50, so `0 % 50 == 0` — it logs at step 0.

At step 0, the model hasn't done a backward pass yet. The rendered features might be in a weird state. But you said it's **always** zero for train...

Let me look more carefully — is `rendered_interp` computed under `torch.no_grad()` in training? No, it's not. But there's another possibility: the `rendered_interp` tensor still has gradients attached, and `.detach()` inside `log_debug_visualizations` → `compute_feature_norms` → `to_hwc_uint8` should handle that.

Wait — I bet the issue is that `rendered_interp[0, 0]` is being passed while the tensor is on GPU with autocast (bf16). The PCA function might be getting near-zero values due to half precision. Let me check `run_pca`:

There it is. Look at the `except` block:

```python
except:
    B, C1, FH, FW = feature.shape
    H, W = img_size
    pca_rgb = torch.zeros((B, 3, H, W), device=feature.device)
return pca_rgb
```

The PCA is **silently failing** during training and returning zeros. This is likely because during training the tensor has gradients attached or is in bf16, and `pca_torch` does SVD which can fail on bf16 tensors or tensors that require grad.

The validation step runs under `inference_mode=True` (or `torch.no_grad()`), so the tensor is clean. In training, the tensor still has autograd graph attached.

Fix: detach and cast to float32 before passing to `log_debug_visualizations`:

The issue is that `pca_torch` does SVD which fails on bf16 or grad-attached tensors, hits the bare `except`, and returns zeros. Adding `.detach().float()` ensures the PCA gets clean fp32 tensors without autograd.

---
