# Execute Task: 1.3 Wire SAM encoder into `ModelW...

Exported: 6/8/2026, 12:16:59 AM

---

**User:**

Execute Task: 1.3 Wire SAM encoder into `ModelWrapper.__init__` in `src/model/model_wrapper.py`

---

**Assistant:**

Now let me look at `src/main.py` to see how the model is instantiated and how `load_foundation_model()` is called:

Now let me also check `load_foundation_model.py` to confirm the SAM encoder is already being returned:

Now I have a clear picture. The `load_foundation_model()` already returns `sam_encoder` as the 5th element, and `main.py` already unpacks it. I need to:

1. Add `sam_encoder=None` parameter to `ModelWrapper.__init__`
2. Store it as `self.sam_encoder`
3. Pass `sam_encoder=sam_encoder` in the `ModelWrapper(...)` call in `main.py`

Let me also check how `self.sam_encoder` is referenced in `forward_foundation_model` to make sure it's consistent:

Good — the `forward_foundation_model` already references `self.sam_encoder`. Now I'll make the changes:

Now I need to store `self.sam_encoder` in the constructor body. I'll add it after the `self.lseg_feature_extractor` assignment:

Now update `src/main.py` to pass `sam_encoder` to `ModelWrapper`:

Now let me verify the changes look correct:

All diagnostics are pre-existing warnings (unused imports, f-strings without placeholders, etc.) — none related to my changes. Let me quickly verify the final state of the relevant sections:

Everything looks good. Let me mark the task as completed:

Done. Here's what I changed:

**`src/model/model_wrapper.py`:**
- Added `sam_encoder=None` parameter to `ModelWrapper.__init__` (alongside the existing `dino`, `lseg_feature_extractor`, `clip` params)
- Stored it as `self.sam_encoder` in the constructor body

**`src/main.py`:**
- Added `sam_encoder=sam_encoder` to the `ModelWrapper(...)` constructor call, passing through the value already unpacked from `load_foundation_model()`

The `forward_foundation_model()` method already references `self.sam_encoder` in its SAM branch (from task 1.2), so this wiring completes the connection.

---
