# Execute task: 4.2 Add optional LoRA injection to `SAMMaskDecoderWrapper`

Exported: 6/8/2026, 12:20:44 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 4.2 Add optional LoRA injection to `SAMMaskDecoderWrapper`

---

**Assistant:**

The LoRA injection is already implemented in `SAMMaskDecoderWrapper.inject_lora()`. Let me verify it works correctly by checking that:
1. When `use_lora=True`, LoRA is injected on `v_proj` layers
2. All original mask decoder parameters are frozen
3. Only LoRA parameters are trainable
4. When `use_lora=False`, the mask decoder operates with frozen weights

Let me look at the SAM transformer structure to confirm the layer paths are correct:

The SAM transformer structure confirms:
- `self.layers` contains `TwoWayAttentionBlock` instances, each with `cross_attn_token_to_image.v_proj`
- `self.final_attn_token_to_image.v_proj` is the final attention layer

The existing `inject_lora` method in `SAMMaskDecoderWrapper` already targets the correct layers. Let me now verify the implementation is complete by running the existing tests:

Now let me run the existing LoRA tests to make sure they pass:

All LoRA tests pass. Now let me write a quick integration test to verify the LoRA injection works correctly on the SAM mask decoder structure specifically. Let me check if there's already a test for the SAM decoder with LoRA:

The implementation is already in place. Looking at the current `sam_decoder.py`, the `inject_lora` method:

1. ✅ Injects `LoRALinear` on `v_proj` layers in token-to-image cross-attention (`cross_attn_token_to_image.v_proj` in each layer)
2. ✅ Also injects on the final attention layer (`final_attn_token_to_image.v_proj`)
3. ✅ All original mask decoder parameters are frozen in `__init__` (before LoRA injection)
4. ✅ LoRA parameters are trainable by default (from `LoRALinear` constructor)
5. ✅ When `use_lora=False`, the mask decoder operates with original frozen weights

The implementation is complete and correct. Let me write a quick unit test to verify the LoRA injection behavior on the SAM decoder specifically, to make sure the integration works end-to-end:

The LoRA injection works correctly on SAM's transformer structure. The `v_proj` layers have shape `(128, 256)` (downsampled by rate 2), and after injection:
- Trainable params: `4 * (256 + 128) = 1536` ✅
- Frozen params: original `v_proj` weight + bias = `128*256 + 128 = 32896` ✅

Let me also verify the full `inject_lora` method on the SAM decoder works with all layers:

Everything works correctly. The implementation in `src/model/sam_decoder.py` already fully satisfies task 4.2:

1. **When `use_lora=True`**: `inject_lora` is called on `v_proj` layers in all token-to-image cross-attention blocks (both per-layer and final) — ✅
2. **All original mask decoder parameters are frozen**: Done in `__init__` before LoRA injection — ✅
3. **Only LoRA parameters are trainable**: Verified that only `lora_A` and `lora_B` params have `requires_grad=True` — ✅ (4608 trainable params for rank=4 with 3 attention layers)
4. **When `use_lora=False`**: The mask decoder operates with original frozen weights, no LoRA modules are created — ✅

Task 4.2 is complete. The `SAMMaskDecoderWrapper` already has the LoRA injection fully implemented:

- `inject_lora()` targets `cross_attn_token_to_image.v_proj` on each transformer layer plus `final_attn_token_to_image.v_proj`
- With rank=4 and SAM's 256→128 downsampled attention, this adds 4,608 trainable parameters (3 layers × 4 × (256+128))
- All original decoder weights stay frozen; only LoRA A/B matrices are trainable
- When `use_lora=False`, no LoRA modules are created and the decoder runs with its original frozen weights

---
