# Execute Task: 1.1 Create `src/dataset/dataset_r...

Exported: 6/8/2026, 12:26:52 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 1.1 Create `src/dataset/dataset_replica_semseg.py` with `DatasetReplicaSemSeg` class

---

**Assistant:**

Let me look at the existing `DatasetReplica2dSeg` to follow its pattern.

Now let me look at the `DatasetCfgCommon` and the `__init__.py` to understand registration:

Now let me look at the existing `DatasetReplica` to understand the trajectory/intrinsics loading pattern for the standard Replica dataset:

Now I have a good understanding of the patterns. Let me also check the `view_sampler` module briefly:

Now I have a complete picture. Let me also check the `cam_utils` module for the `camera_normalization` function:

Good. Now I have everything I need. Let me create the new dataset file:

Now let me verify the file has no syntax errors:

Let me clean up the unused imports and fix the line length:

Let me check the actual line 190 more carefully:

Clean. Now let me verify the module can be imported by checking for any import issues:

Let me also run a quick sanity check on the key methods by testing them with synthetic data:

All methods work correctly. Let me also test the `load_intrinsics` method:

Let me also verify the `FileNotFoundError` is raised when the root doesn't exist:

Everything works as expected. Let me read the final file to confirm it's complete:

I notice the `_get_num_frames` was renamed to `get_num_frames` (following the "no private member functions" rule). But I see there's still `_sample_views` which is private. Let me fix that per the coding style rules:

Let me run the import test one more time to make sure everything still works:

Now let me mark the task as complete:

Done. Here's what I implemented in `src/dataset/dataset_replica_semseg.py`:

- **`ReplicaSemSegCfg`** dataclass with all required fields (name, roots, scenes, baseline_min/max, max_fov, make_baseline_1, augment, relative_pose, skip_bad_shape, num_of_inputs, prompt_strategy, min_object_pixels)
- **`load_intrinsics()`** reads `{root}/replica/cam_params.json` and builds a 3×3 intrinsic matrix
- **`load_trajectory(scene)`** parses `{root}/replica/{scene}/traj.txt` (16 floats per line → 4×4 matrix), with validation
- **`decompose_labels(label_map)`** converts an integer label map to (K, H, W) binary masks for non-background objects, returning an empty (0, H, W) tensor when all-background
- **`__iter__()`** yields batches with context/target views including the `label` key, loading RGB with bilinear interpolation and labels with nearest-neighbor resize
- **`FileNotFoundError`** raised at init if the dataset root doesn't exist
- Module-level exports (`DATASET_CLASS`, `DATASET_NAMES`, `CFG_WRAPPERS`) for registration in the dataset module

---
