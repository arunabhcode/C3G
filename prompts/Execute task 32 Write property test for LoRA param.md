# Execute task: 3.2 Write property test for LoRA parameter count

Exported: 6/8/2026, 12:19:58 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 3.2 Write property test for LoRA parameter count

---

**Assistant:**

Now let me check if there are existing property tests to follow the same pattern:

Now I have a clear picture of the testing patterns. Let me write the property test for LoRA parameter count:

Now let me run the tests to verify they pass:

All 4 property tests pass. Let me mark the task as complete:

Done. Created `tests/test_lora_parameter_count.py` with 4 property-based tests using Hypothesis (100 examples each):

1. **test_lora_trainable_parameter_count** — verifies trainable params equal `rank * (d_in + d_out)`
2. **test_lora_original_weights_frozen** — confirms original layer weights have `requires_grad=False`
3. **test_lora_a_and_b_shapes** — validates lora_A is `(rank, d_in)` and lora_B is `(d_out, rank)`
4. **test_inject_lora_parameter_count** — verifies the `inject_lora` helper produces the same correct trainable count

All tests pass across arbitrary `(d_in, d_out, rank)` combinations.

---
