# Execute task: 4.1 Create `src/model/sam_decoder.py` with `SAMMaskDecoderWrapper` class

Exported: 6/8/2026, 12:20:28 AM

## Steering Documents

**Workspace:**
- python.md

---

**User:**

Execute Task: 4.1 Create `src/model/sam_decoder.py` with `SAMMaskDecoderWrapper` class

---

**Assistant:**

Let me look at how SAM is loaded in the existing codebase and understand the SAM model structure.

Now let me look at how the `segment_anything` package is structured to understand the mask decoder and prompt encoder interfaces:

Now let me look at the existing `lora.py` to understand the LoRA implementation already in place:

Now let me look at the SAM mask decoder's internal transformer to understand where the `v_proj` layers are (for future LoRA injection in task 4.2):

Now I have a complete understanding of the SAM architecture. Let me also check the `build_sam.py` to understand how the full SAM model is constructed (to know how to load just the mask decoder and prompt encoder):

Now I have all the information needed. The SAM model is loaded as a whole, and we can access `sam.mask_decoder` and `sam.prompt_encoder` from it. Let me create the `SAMMaskDecoderWrapper`:

Now let me verify there are no diagnostics issues:

Let me run a quick import check to make sure the module loads correctly:

Let me also run a quick unit test to verify the class can be instantiated and the forward pass works (using a mock since we don't have the actual checkpoint):

The issue is that SAM's mask decoder uses `torch.repeat_interleave` to expand image embeddings per-mask based on the number of sparse prompt embeddings. The batch dimension of `sparse_embeddings` and `image_embeddings` need to match. Let me look at how SAM's original code handles this — the issue is that with grid points, we get a batch of 2 images but the sparse embeddings have a different batch size. Let me trace through the logic more carefully:

The problem is that `sparse_embeddings` from the prompt encoder has shape `(B, N_prompts, embed_dim)` and the mask decoder does `torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)`. Looking at the mask decoder code more carefully:

```python
output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
```

The issue is `tokens.shape[0]` is the batch size of sparse_prompt_embeddings, and it's being used to repeat image_embeddings. When `image_embeddings` has batch=2 and `sparse_prompt_embeddings` has batch=2, it repeats each image embedding 2 times, giving batch=4. But `dense_prompt_embeddings` still has batch=2.

Looking at SAM's original usage in `sam.py`, it processes one image at a time (`curr_embedding.unsqueeze(0)`) with prompts for that single image. The mask decoder expects `image_embeddings` to have batch=1 (single image) and `sparse_prompt_embeddings` to have batch=N (number of prompt sets for that image).

So the correct approach is to process each image in the batch individually through the mask decoder. Let me fix this:

Now let me run the test again:

All forward pass modes work correctly. Let me also verify the LoRA injection works:

Everything works correctly:
- LoRA adds 4608 trainable parameters (3 v_proj layers × 2 × rank × dim = 3 × 2 × 4 × 128 = 3072... actually let me check: SAM vit_b has embedding_dim=256, downsample_rate=2, so internal_dim=128. LoRA on each v_proj: rank×128 + 4×128 = 4×128 + 4×128 = 1024 per layer. With 3 layers (2 in the loop + 1 final): 3 × 1024 + some... 4608 = 3 × 1536. Actually: in_features=256, rank=4, out_features=128 → lora_A: 256×4=1024, lora_B: 4×128=512, total per layer=1536, × 3 layers = 4608. That checks out.)
- Gradients flow through the rendered features when LoRA is enabled
- All prompt modes work (points, boxes, segment-everything grid)

Let me also run the existing tests to make sure nothing is broken:

All existing tests pass. Let me check the final state of the file:

Task 4.1 is complete. Here's what was implemented in `src/model/sam_decoder.py`:

**`SAMMaskDecoderWrapper`** — loads SAM's `MaskDecoder` and `PromptEncoder` from a checkpoint, freezes all their parameters, and provides a clean `forward()` interface that:

- Accepts rendered features of any spatial size and interpolates to 64×64 when needed
- Generates positional encodings via `PromptEncoder.get_dense_pe()`
- Supports three prompt modes: point prompts, box prompts, or segment-everything (8×8 grid of foreground points)
- Processes each image individually through the mask decoder (matching SAM's expected per-image interface)
- Returns mask logits of shape `(B, 3, 256, 256)` (multimask output)
- Includes `_inject_lora()` for optional LoRA adaptation on `v_proj` layers (used when `use_lora=True`)

All existing tests pass and the new module was verified with point prompts, box prompts, segment-everything mode, non-64×64 inputs, and LoRA gradient flow.

---
