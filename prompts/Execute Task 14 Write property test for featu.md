# Execute Task: 1.4 Write property test for featu...

Exported: 6/8/2026, 12:17:30 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 1.4 Write property test for feature extraction shape invariant

---

**Assistant:**

Let me look at the existing SAM feature extraction code to understand what we're testing.

Now I have a clear picture of the `forward_foundation_model` method. The SAM branch:
1. Resizes input to 1024×1024
2. Runs SAM encoder → `(B*V, 256, 64, 64)`
3. Reshapes to `(B, V, 256, 64, 64)`
4. When `interpolate=True`, interpolates to `(H//14, W//14)`

Let me check if there's an existing test directory structure:

The file seems truncated. Let me check if it's complete:

The file is incomplete — it cuts off mid-function. Let me check the pyproject.toml for test configuration:

I see `hypothesis` isn't in the dependencies. Let me check if it's available:

---

**User:**

continue

---

**Assistant:**

On it.

---

**User:**

continue

---

**Assistant:**

Let me check if hypothesis and pytest are available, then write the property test.

Good, both are available. Now let me write the complete property test file:

Now let's run the tests to verify they pass:

All 4 property tests pass. Let me mark the task as complete:

Done. The property test file at `tests/test_sam_feature_extraction.py` covers Property 1 (Feature Extraction Shape Invariant) with four test cases:

1. **Shape with interpolation** — verifies output is `(B, V, 256, H//14, W//14)` for arbitrary dimensions
2. **Shape without interpolation** — verifies output retains SAM's native `(B, V, 256, 64, 64)`
3. **Channel count invariant** — confirms the feature dimension is always 256
4. **No gradients** — confirms the frozen encoder produces outputs without gradient tracking

Each test uses Hypothesis to generate 100 random combinations of `(B, V, H, W)` dimensions, validating the shape invariant holds universally. A `MockSAMEncoder` isolates the pipeline logic from the actual SAM weights.

---
