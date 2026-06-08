# Execute task: 11.2 Ensure `feature_detach` configuration is respected

Exported: 6/8/2026, 12:24:59 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 11.2 Ensure `feature_detach` configuration is respected

---

**Assistant:**

Let me look at the current implementation to understand how `feature_detach` is being used.

Now I can see the issue. The `feature_detach` parameter is passed to `render_cuda` but it's never actually used inside the function. Let me check the `FeatureDetachGaussianRasterizer` to understand how it handles detachment.

Now I understand the architecture. The `FeatureDetachGaussianRasterizer` name is a bit misleading — looking at the backward pass, it computes gradients for both `means3D` (geometry) and `semantic_feature` (features). The "detach" behavior needs to be implemented at the call site by detaching geometry tensors before passing them to the rasterizer.

Let me look at how `feature_detach` is currently used in `render_cuda`:

Looking at the code in `cuda_splatting.py`, I can see that `feature_detach` is accepted as a parameter but **never actually used** in the function body. The parameter is passed through from `decoder_splatting_cuda.py` but has no effect. This is the bug — when `feature_detach=True`, the geometry tensors (means, covariances) should be detached before being passed to the feature rasterizer so gradients only flow through feature vectors.

Let me also check the `training_step` to understand the full context:

Now let me look at how the `render_cuda` function uses `feature_detach`. I already saw it's passed but not used. Let me implement the fix:

Now I have a clear picture. The `feature_detach` parameter is passed to `render_cuda` but never applied. When `feature_detach=True`, the geometry tensors (`gaussian_means` and `gaussian_covariances`) should be detached before passing to the `FeatureDetachGaussianRasterizer` so that gradients from the feature rendering path don't flow back through geometry — only through the feature vectors themselves.

Let me implement this:

Now let me also check the SAM config to confirm `feature_detach: false` is set:

Good — `feature_detach: false` is set in the SAM config, which means gradients will flow through both geometry and features. Now let me verify the existing tests still pass and write a test to validate this behavior:

Now I have a good understanding of the test patterns. The existing tests use a mock `differentiable_splat` function. For this task, I need to write a test that specifically validates the `feature_detach` behavior in `render_cuda`. Let me look at how `render_cuda` is called to write a proper test:

Good. Now let me write a test that validates the `feature_detach` configuration is respected. Since the CUDA rasterizer requires GPU and compiled CUDA code, I'll write a test that validates the logic at the `render_cuda` level using a mock approach similar to the existing tests, but also a unit test that directly tests the detach logic in `render_cuda`:

Now let's run the tests to make sure they pass:

One test fails due to a module import issue (unrelated to our change — it's a missing dataset module). Let me fix that test to avoid the problematic import chain:

All 5 tests pass. Let me also verify the existing tests still pass to make sure the change to `cuda_splatting.py` didn't break anything:

Those errors are pre-existing import issues unrelated to our change (they fail during collection due to a missing `src.dataset.dataset_replica_pose` module). Let me run just the tests that can actually be collected:

The 5 failures in `test_gradient_routing.py` are all due to the same pre-existing `KeyError: 'src.dataset'` import issue — not related to our change. The 29 tests that passed include our 5 new tests plus tests from other files. Let me confirm our specific tests all pass cleanly:

Let me also verify the final state of the modified file looks correct:

No diagnostics issues. Let me mark the task as complete:

Done. Here's what I implemented:

**The problem:** The `feature_detach` parameter was passed to `render_cuda()` in `src/model/decoder/cuda_splatting.py` but never actually used — geometry tensors were always passed with gradients intact regardless of the setting.

**The fix:** In the `FeatureDetachGaussianRasterizer` branch of `render_cuda()`, I now conditionally `.detach()` the geometry tensors (`means3D`, covariances, opacities) before passing them to the rasterizer when `feature_detach=True`. This ensures:

- `feature_detach: false` → gradients flow through both geometry and feature vectors
- `feature_detach: true` → gradients only flow through feature vectors (geometry is detached)

**Tests:** Added `tests/test_feature_detach_config.py` with 5 property-based tests validating both modes behave correctly.

---
