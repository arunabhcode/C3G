# Execute task: 1.2 Add SAM feature extraction branch in `src/model/model_wrapper.py`

Exported: 6/8/2026, 12:16:32 AM

---

**User:**

Execute Task: 1.2 Add SAM feature extraction branch in `src/model/model_wrapper.py`

---

**Assistant:**

### Sub-Agent: Spec Task Execution

**Explanation:** Delegating implementation of task 1.2 (adding SAM feature extraction branch in model_wrapper.py) to the spec-task-execution subagent.

**Prompt:**

Execute the following task from the spec at .kiro/specs/sam-c3g-f-integration/tasks.md.

Task ID: 1.2
Task Title: Add SAM feature extraction branch in `src/model/model_wrapper.py`
Spec Path: .kiro/specs/sam-c3g-f-integration

Task Details:
- Add `elif 'sam' in self.train_cfg.reproj_model:` in `forward_foundation_model()`
- Resize input images to 1024×1024 before passing to SAM encoder
- Run frozen SAM encoder to produce `(B*V, 256, 64, 64)` features
- Reshape to `(B, V, 256, 64, 64)`
- Interpolate to `(H//14, W//14)` when `interpolate=True` to match pipeline grid
- _Requirements: 1.1, 1.2, 1.3_

Relevant context from the design document:

The intended implementation per the design (section 2 — SAM Feature Extraction in `src/model/model_wrapper.py`):

```python
def forward_foundation_model(self, input_image, interpolate=True):
    # ... existing branches ...
    elif 'sam' in self.train_cfg.reproj_model:
        # Resize to 1024×1024 (SAM's expected input)
        sam_input = F.interpolate(input_image.reshape(B*V, C, H, W),
                                   size=(1024, 1024), mode='bilinear')
        # Run frozen SAM encoder
        with torch.no_grad():
            context_feature = self.sam_encoder(sam_input)  # (B*V, 256, 64, 64)
        context_feature = context_feature.reshape(B, V, 256, 64, 64)
```

The output is then interpolated to `(H//14, W//14)` to match the pipeline's standard feature grid (consistent with how DINOv2 features are handled).

Requirements being addressed:
- Requirement 1.1: SAM_Encoder produces feature maps of shape (B, V, 256, H/16, W/16)
- Requirement 1.2: SAM_Encoder remains fully frozen with no gradient computation during training
- Requirement 1.3: When input images have resolution different from 1024×1024, SAM_Encoder receives images resized to 1024×1024 and output features are interpolated to match pipeline's expected spatial resolution (H/14, W/14)

Important context from prior task (1.1):
- Task 1.1 added SAM encoder loading to `src/model/load_foundation_model.py` (status: completed)
- The SAM encoder is referenced as `self.sam_encoder` on the ModelWrapper in line with task 1.3 (which has not yet been completed but will wire it in)

Important notes:
- For this task (1.2), focus only on adding the feature extraction branch in `forward_foundation_model()` in `src/model/model_wrapper.py`. Match the patterns used by existing foundation models (DINOv2, LSeg, VGGT, MaskCLIP) in the same method.
- Look at how existing branches handle:
  * The shape unpacking of `input_image`
  * The `interpolate=True` flag and the target H//14, W//14 grid
  * Whether torch.no_grad() / freezing is handled at the call site or via parameter freezing
- Reference `self.sam_encoder` even though `__init__` wiring is task 1.3 — the field will exist when task 1.3 is completed.
- Do not implement task 1.3 (constructor wiring); that is a separate task.
- Do NOT execute the optional sub-task 1.4 (the property test marked with `*`).

After implementation:
- Run any project linters or type checks if available (check pyproject.toml for tools).
- Do not run training or full model inference; just verify the change is syntactically valid and imports resolve.
- Return a clear summary of what was changed.

**Response:**

Completed task 1.2: added SAM feature extraction branch in `src/model/model_wrapper.py`.

**Change made** (in `forward_foundation_model()`, between the VGGT and MaskCLIP branches):

```python
elif 'sam' in self.train_cfg.reproj_model:
    # SAM expects 1024x1024 input; resize before encoder.
    sam_input = F.interpolate(
        input_image.reshape(B * V, C, H, W),
        size=(1024, 1024),
        mode="bilinear",
        align_corners=False,
    )
    # Frozen SAM encoder produces (B*V, 256, 64, 64) features.
    context_feature = self.sam_encoder(sam_input)
    context_feature = context_feature.reshape(B, V, 256, 64, 64)
```

**How it satisfies the requirements:**
- **Req 1.1** — SAM encoder output is reshaped to `(B, V, 256, 64, 64)`, and (when `interpolate=True`) is then resized by the existing trailing `F.interpolate(... size=(H//14, W//14) ...)` block, matching the pipeline's standard feature grid (B, V, 256, H/14, W/14).
- **Req 1.2** — Freezing is honoured two ways: the encoder's parameters were set to `requires_grad=False` in task 1.1, and the entire branch executes inside the existing `with torch.no_grad():` block at the top of `forward_foundation_model()` (matching how every other foundation-model branch is handled).
- **Req 1.3** — Inputs are resized to 1024×1024 prior to the SAM encoder regardless of original resolution, and the existing post-encoder `if interpolate:` block aligns to (H//14, W//14).

**Notes / scope discipline:**
- Did not modify `__init__` — `self.sam_encoder` wiring is task 1.3 (separate task).
- Did not implement task 1.4 (optional property test, marked with `*`).
- Branch placement matches existing patterns (uses `'sam' in self.train_cfg.reproj_model`, mirroring `'vggt' in ...` / `'dinov3' in ...`).
- The autoformatter (matching the file's existing style) reflowed the code into multi-line form; semantically identical to the design snippet.

**Verification performed:**
- `ast.parse` of `src/model/model_wrapper.py` — syntax OK.
- `ruff check --select E9,F63,F7,F82` — only pre-existing F722 jaxtyping warnings; no new issues introduced by the edit.
- `getDiagnostics` returned no diagnostics for the file.

Task 1.2 is complete. Added the SAM branch in `forward_foundation_model()` in `src/model/model_wrapper.py`:

- Resizes inputs to 1024×1024 with bilinear interpolation
- Runs `self.sam_encoder` (executes inside the existing `torch.no_grad()` block, matching the other foundation-model branches)
- Reshapes to `(B, V, 256, 64, 64)`
- The existing trailing `if interpolate:` block then resamples to `(H//14, W//14)` to match the pipeline grid

Branch placement mirrors the `vggt` / `dinov3` patterns. Constructor wiring of `self.sam_encoder` is deferred to task 1.3, and the optional property test (1.4) was not implemented.

---
