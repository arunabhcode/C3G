# Execute task: 3.1 Create `src/model/lora.py` with `LoRALinear` class

Exported: 6/8/2026, 12:18:43 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 3.1 Create `src/model/lora.py` with `LoRALinear` class

---

**Assistant:**

I have a bug in the import — let me fix that:

Now let me verify the module works correctly:

All tests pass. Let me read the final file to confirm it's clean:

Done. Created `src/model/lora.py` with:

- **`LoRALinear`** — wraps an existing `nn.Linear`, freezes it, and adds a trainable low-rank bypass (`lora_A` → `lora_B`). `lora_B` is zero-initialized so the initial output is identical to the original layer.
- **`inject_lora(module, target_layer_name, rank)`** — helper that resolves a dotted attribute path, validates it's an `nn.Linear`, and replaces it in-place with a `LoRALinear` wrapper.

All verification checks pass: frozen original weights, correct trainable parameter count (`rank * (in_features + out_features)`), zero-init behavior, and proper error handling for invalid targets.

---

**User:**

why is LORA just for linear layers shouldn't I inject it to entire networks like SAM?

---

**Assistant:**

Good question. LoRA *is* applied to networks like SAM — but it works by targeting specific linear layers *within* that network, not by wrapping the entire model as a single unit. Here's why:

**LoRA operates at the linear layer level by design.** The core idea from the Hu et al. paper is that the weight updates during fine-tuning have low intrinsic rank. So instead of updating a full `d_in × d_out` weight matrix, you decompose the update into two small matrices `A (d_in × r)` and `B (r × d_out)`. This only makes sense for individual linear projections.

**In practice, you inject LoRA into selected layers across the network.** For SAM's mask decoder, the plan (task 4.2) calls `inject_lora` on the `v_proj` layers in the token-to-image cross-attention blocks. You'd do something like:

```python
for block in sam_mask_decoder.transformer.layers:
    inject_lora(block, "cross_attn_token_to_image.v_proj", rank=4)
```

This gives you fine-tuning of the full decoder's behavior while only adding ~4K–8K trainable parameters total, versus the millions in the full decoder.

**Why not every layer?** You could inject LoRA into Q, K, V, and output projections across all attention layers — that's what people do for full LoRA fine-tuning of LLMs. But for this use case, the design targets only `v_proj` in cross-attention because:

1. The distribution shift is specifically in how rendered features (from Gaussian splatting) differ from native SAM encoder features
2. The value projection is where the model reads from those features
3. Fewer LoRA layers = fewer parameters = less risk of overfitting on limited segmentation data

So `LoRALinear` is the building block, and `inject_lora` is the tool you use to surgically place it wherever needed in SAM's decoder. The module we created is general-purpose — it'll work on any `nn.Linear` in any network.

---

**User:**

what if I want to do full LORA, only the inject function will change?

---

**Assistant:**

Right. The `LoRALinear` class stays exactly the same. For full LoRA you'd just add a helper that walks the module tree and injects on every `nn.Linear` it finds. Something like:

```python
def inject_lora_all(module: nn.Module, rank: int = 4, target_modules: list[str] | None = None) -> None:
    """Inject LoRA on all nn.Linear layers (or a filtered subset) in a module."""
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                inject_lora(module, name, rank=rank)
```

You could also filter by name patterns — e.g. `target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]` to hit all attention projections but skip MLPs, or pass `None` to hit everything.

The current design intentionally starts conservative (only `v_proj` in cross-attention) because the distribution shift is localized. But if you find mask quality still suffers, you can widen the scope without touching `LoRALinear` at all — just change which layers get injected. That's the nice thing about keeping the building block generic.

---
