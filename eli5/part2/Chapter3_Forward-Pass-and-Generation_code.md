# Part II Chapter 3 Forward Pass and Generation (Code Walkthrough): ELI5 companion

Original: [Chapter3_Forward-Pass-and-Generation_code.md](../../course-material/eng/part2/Chapter3_Forward-Pass-and-Generation_code.md). Same headings, same numbers. Concept half: [Chapter3_Forward-Pass-and-Generation.md](./Chapter3_Forward-Pass-and-Generation.md). Cast and picture legend: [../README.md](../README.md).

Every file named here carries ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang`; paths are relative to `python/minisgl/`.

## The one-minute version

**Kid version.** We climb the building floor by floor in the real source. The model code is surprisingly short (a few dozen lines per model), because the heavy lifting hides in small reusable blocks (`layers/`) and in GPU kernels.

```text
   models/qwen3.py            Qwen3ForCausalLM.forward
     ├─ Qwen3Model.forward                      embed ─▶ [ Qwen3DecoderLayer ] x N ─▶ final norm
     │    └─ Qwen3DecoderLayer.forward          norm ▶ attention ▶ norm ▶ MLP   (with the fused shortcut)
     │         ├─ self_attn = RopeAttn          models/utils.py:  qkv_proj ▶ AttentionLayer ▶ o_proj
     │         └─ mlp       = GatedMLP          models/utils.py:  gate_up_proj ▶ silu_and_mul ▶ down_proj
     └─ lm_head = ParallelLMHead                layers/embedding.py: pick last positions ▶ logits
```

## 1 Learning Objectives

| Goal | Where it is covered |
|---|---|
| Locate embedding, decoder layers, attention, LM head in Qwen3 | Section 2's table and the tree above |
| Explain Q/K/V, the causal mask, residuals through shapes | 3.2 to 3.4 and the shape table in 5 |
| Why generation reads the last position of each request, and when it stops | 4.1, 4.2, 4.3 |

## 2 Locate the Source

| File and entry point | What to look for | Kid gloss |
|---|---|---|
| `models/qwen3.py`: `Qwen3Model`, `Qwen3DecoderLayer` | Order of operations in a floor | The building plan |
| `models/utils.py`: `RopeAttn`, `GatedMLP` | QKV projection, attention, gated MLP | The meeting room and the private office |
| `layers/attention.py`: `AttentionLayer.forward` | Q/K norm, RoPE, attention backend | Where words are rotated and compared |
| `layers/norm.py`: `RMSNormFused.forward` | Fused norm + residual add | The thermostat with the shortcut built in |
| `layers/embedding.py`: `ParallelLMHead.forward` | Choosing last positions, vocabulary scores | The scoreboard |
| `engine/engine.py`, `engine/sample.py` | Running the pass; choosing the next token | The chef and the word-picker |

Other models (`llama.py`, `mistral.py`, `qwen2.py`, `qwen3_moe.py`) are copies of the same layout with tiny differences:

| Model file | Difference from the Qwen3 layout |
|---|---|
| `qwen3.py` | `has_qk_norm=True`: Q and K are normalised per head before RoPE |
| `qwen2.py` | `has_qk_norm=False`, `has_attn_bias=True`: attention projections have a bias |
| `llama.py`, `mistral.py` | Plain attention (no bias, no QK norm) |
| `qwen3_moe.py` | The MLP is a mixture of experts (`MoEMLP`) instead of `GatedMLP` |

## 3 Follow One Forward Pass

### 3.1 Enter the model with token IDs

```python
# models/qwen3.py: Qwen3Model.forward
x = self.embed_tokens.forward(input_ids)
residual: torch.Tensor | None = None
for layer in self.layers.op_list:
    x, residual = layer.forward(x, residual)
return self.norm.forward(x, residual)[0]
```

*Line by line:*

| Line | Kid words |
|---|---|
| `x = self.embed_tokens.forward(input_ids)` | Look up the starting vector for every card |
| `residual = None` | The shortcut road does not exist yet |
| `for layer in self.layers.op_list:` | Go up the floors one at a time (`op_list` is the plain Python list inside an `OPList`) |
| `x, residual = layer.forward(x, residual)` | Each floor takes the paper `x` and the road `residual`, returns updated versions |
| `return self.norm.forward(x, residual)[0]` | Last thermostat: add what is left of the road, cool it, return only the cooled vectors (index `[0]`; index `[1]` would be the road) |

**The packed layout `[N, C]` vs the book's `[B, T, C]`.** Conceptually the input is a rectangle: `B` sentences by `T` positions. In the engine, sentences have different lengths, so padding them to a rectangle would waste work. Instead all tokens of all requests are **concatenated into one long list** of length `N = total tokens`. Hidden states are `[N, C]`; a small array of boundaries (`cu_seqlens`) says where each request starts. Example (the prefill tray from chapter 2): request A has 5 tokens to compute, request B has 4:

```text
   input_ids  : [ a0 a1 a2 a3 a4 | b2 b3 b4 b5 ]        N = 9 tokens (no padding)
   positions  : [  0  1  2  3  4 |  2  3  4  5 ]        (B's first two tokens were cached)
   cu_seqlens_q = [0, 5, 9]                              boundaries between requests
   hidden     : [9, C]
   LM head keeps rows [4, 8]  (= cu_seqlens_q[1:] - 1)   the last token of each request
```

For a decode tray with 3 requests, `N = 3` and `cu_seqlens_q = [0, 1, 2, 3]`.

### 3.2 Computation inside a decoder layer

```python
# models/qwen3.py: Qwen3DecoderLayer.forward
x, residual = self.input_layernorm.forward(x, residual)
x = self.self_attn.forward(x)
x, residual = self.post_attention_layernorm.forward(x, residual)
x = self.mlp.forward(x)
return x, residual
```

The book asks: where is `x + attention(x)`? **Inside the next norm call.** `RMSNormFused.forward(x, residual)` does: if `residual is None` (very first call): return `(rmsnorm(x), x)`, i.e. save the raw input as the road; else do `residual += x`, then `x = rmsnorm(residual)`, returning both. So the shortcut addition is *deferred* and merged into the next thermostat (one pass over memory instead of two). Tracing the first two floors:

| Call | Road (`residual`) afterwards | Paper (`x`) afterwards |
|---|---|---|
| embedding output | n/a | `x0` |
| floor 0, `input_layernorm(x0, None)` | `x0` | `norm(x0)` |
| `self_attn` | `x0` | `a0` |
| floor 0, `post_attention_layernorm(a0, x0)` | `x0 + a0` | `norm(x0 + a0)` |
| `mlp` | `x0 + a0` | `m0` |
| floor 1, `input_layernorm(m0, x0 + a0)` | `x0 + a0 + m0` | `norm(x0 + a0 + m0)` |

Mathematically that is still Pre-Norm: `h = x + Attention(RMSNorm(x))`, `out = h + MLP(RMSNorm(h))`. The final `self.norm` after the last floor adds the last MLP output to the road.

### 3.3 QKV, RoPE, and causal attention

```python
# models/utils.py: RopeAttn.forward (memory-release statements omitted)
qkv = self.qkv_proj.forward(x)
o = self.attn.forward(qkv)
return self.o_proj.forward(o)
```

*Line by line:* (1) one merged matrix multiply produces Questions, Labels and Gifts side by side; (2) `AttentionLayer.forward` splits them, normalises and rotates Q/K, runs the attention kernel; (3) the output projection mixes the heads and returns to hidden size `C`. The real source also has `del x` lines to release memory early (the "memory-release statements" the book omits).

**What `AttentionLayer.forward` does, in order** (read it with the code open):

| Step | Code | Kid words |
|---|---|---|
| Split | `q, k, v = qkv.split([qo_dim, kv_dim, kv_dim], dim=-1)` | Cut the wide row into Question, Label, Gift parts |
| QK norm (Qwen3 only) | `q_norm.forward_inplace(q.view(-1, heads, head_dim))` | Thermostat on each head's Question and Label |
| RoPE | `rotary.forward(ctx.batch.positions, q, k)` | Rotate Q and K by their seat numbers (V stays) |
| Attend | `ctx.attn_backend.forward(q, k, v, layer_id, batch)` | Store the new K/V in the pantry, then run the attention kernel against the pantry |
| Reshape | `o.view(-1, qo_dim)` | Glue the heads back into one row per token |

**Important:** the backend call also **writes this pass's K and V into the KV cache** (`kvcache.store_kv(k, v, batch.out_loc, layer_id)`) *before* running attention, so the new tokens attend to themselves and to everything cached. Attention reads K/V from the cache through the page table, not from the `k`, `v` tensors passed in.

**The book's teaching simplification, line by line.** (It is *not* the GPU backend's source; it is a readable stand-in.)

```python
import torch

groups = q.shape[1] // k.shape[1]
k = k.repeat_interleave(groups, dim=1)
v = v.repeat_interleave(groups, dim=1)
scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
T = q.shape[-2]
allowed = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
scores = scores.masked_fill(~allowed, float("-inf"))
weights = scores.float().softmax(dim=-1).to(q.dtype)
attention = weights @ v
```

| Line | Kid words | Shape (B=1, Hq=4, Hkv=2, T=3, D=8) |
|---|---|---|
| `groups = q.shape[1] // k.shape[1]` | How many Question heads share one Label/Gift head | `4 // 2 = 2` |
| `k = k.repeat_interleave(groups, dim=1)` | Photocopy each Label head for its group so shapes line up | K: `[1,2,3,8]` becomes `[1,4,3,8]` |
| `v = ...` | Same for Gifts | V: `[1,4,3,8]` |
| `scores = q @ k.transpose(-1,-2) / sqrt(D)` | Every Question against every Label, scaled | `[1,4,3,3]` |
| `allowed = ones(T,T).tril()` | A triangle of "allowed" cells (lower triangle incl. diagonal) | `[3,3]` |
| `masked_fill(~allowed, -inf)` | Cross out the future | `[1,4,3,3]` |
| `softmax(dim=-1)` in float32, cast back | Scores to percentages per row (float32 for accuracy) | `[1,4,3,3]` |
| `weights @ v` | Take the weighted mix of Gifts | `[1,4,3,8]` |

`repeat_interleave(2, dim=1)` on heads `[kv0, kv1]` gives `[kv0, kv0, kv1, kv1]`: query head `h` uses kv head `h // 2`. The real GPU kernels do **not** make these copies; they just index the same K/V for the whole group (that's the whole memory saving). The lower-triangular mask means position `i` sees positions `0..i`. And again: `head_dim` is its own setting, so `C = Hq · D` is not guaranteed (Qwen3-0.6B: `C = 1024` but `Hq · D = 2048`).

**Real matrix shapes** (rows = outputs, columns = inputs; Qwen3-0.6B: `C = 1024`, `Hq = 16`, `Hkv = 8`, `D = 128`):

| Matrix | Shape | Parameters |
|---|---|---|
| `qkv_proj` | `[(16 + 8 + 8)·128, 1024] = [4096, 1024]` | 4,194,304 |
| `o_proj` | `[1024, 16·128] = [1024, 2048]` | 2,097,152 |
| attention total per layer | | **6,291,456** |

(The same total as in the Deploy lesson's parameter count.)

### 3.4 The gated MLP

```python
# models/utils.py: GatedMLP.forward (memory-release statements omitted)
gate_up = self.gate_up_proj.forward(x)
y = self.act_fn(gate_up)
return self.down_proj.forward(y)
```

*Line by line:* (1) one matrix multiply produces *both* the gate part and the up part, side by side, `[N, 2·I]`; (2) `act_fn` is `silu_and_mul`: apply SiLU to the first half and multiply by the second half, giving `[N, I]`; (3) `down_proj` returns to hidden size `[N, C]`.

| Matrix (Qwen3-0.6B, `I = 3072`) | Shape | Parameters |
|---|---|---|
| `gate_up_proj` | `[2·3072, 1024] = [6144, 1024]` | 6,291,456 |
| `down_proj` | `[1024, 3072]` | 3,145,728 |
| MLP total per layer | | **9,437,184** |

The MLP is 60% of the layer's weights here (9.44 M of 15.73 M). The activation is chosen by name from `config.hidden_act` (`"silu"` or `"gelu"`).

**Tensor parallelism preview.** With two GPUs, `qkv_proj` and `gate_up_proj` are split by **columns** (each GPU computes half the outputs: `[2048, 1024]` and `[3072, 1024]`), and `o_proj` and `down_proj` are split by **rows** (each GPU holds half of the inputs: `[1024, 1024]` and `[1024, 1536]`), so each produces a partial sum that one `all_reduce` adds up. That is why those two layers call `self._comm.all_reduce(y)` (chapter 9).

## 4 From Logits to a Generation Loop

### 4.1 Keep the last position of each request

```python
# models/qwen3.py: Qwen3ForCausalLM.forward
output = self.model.forward(get_global_ctx().batch.input_ids)
logits = self.lm_head.forward(output)
return logits
```

*Line by line:* the model takes **no argument**; it reads the tray from the whiteboard (`get_global_ctx().batch`). That is how `forward()` is called as `self.model.forward()` in the engine.

In `ParallelLMHead.forward` during prefill: `indices = batch.attn_metadata.get_last_indices(bs)`, then `x = x[indices]`, *then* the vocabulary multiply. `get_last_indices` is `cu_seqlens_q[1 : 1 + bs] - 1`. With the tray above (`cu_seqlens_q = [0, 5, 9]`) it gives `[4, 8]`: the last token of A and of B. The multiply therefore produces **2 rows**, not 9. For decode the input already has one token per request, so no selection is needed.

For input `[1, 2, 3]` the logits at position 2 (the token `3`) predict the token *after* 3; earlier rows predict tokens you already know.

**Two more details in this layer:** if the model "ties" the output layer to the embedding table (`tie_word_embeddings`), the LM head reuses the embedding weights instead of storing a second copy; with tensor parallelism the vocabulary is split across GPUs, each computes its slice of the scores, and an `all_gather` stitches them together.

### 4.2 Put one step into a loop

```python
import torch

@torch.inference_mode()
def generate(forward_all, tokens, max_new_tokens, eos_id=None):
    output = []
    for _ in range(max_new_tokens):
        logits = forward_all(tokens)[0, -1]
        token = int(logits.argmax())
        output.append(token)
        tokens = torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)
        if token == eos_id:
            break
    return output
```

| Line | Kid words |
|---|---|
| `@torch.inference_mode()` | No gradient bookkeeping (we are not training) |
| `forward_all(tokens)[0, -1]` | Run the whole model on the whole list; take batch item 0, last position: that row of `[1, T, V]` logits |
| `token = int(logits.argmax())` | Greedy: pick the top-scoring word |
| `output.append(token)` | Remember it |
| `torch.cat((tokens, tokens.new_tensor([[token]])), dim=1)` | Put the new card on the end of the pile (`[[token]]` is a 1x1 tensor with the same dtype and device) |
| `if token == eos_id: break` | Stop after writing the end-of-sequence card (it is included in the output) |

It recomputes the **entire** prefix every iteration: for a 10-token prompt and 8 new tokens that is 108 token-passes instead of 17 (chapter 4 removes the waste). A `max_new_tokens` of 0 makes no model calls.

### 4.3 Find sampling in the engine

```python
# engine/sample.py: Sampler.sample (profiling context omitted)
if args.temperatures is None:
    return torch.argmax(logits, dim=-1)
return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
```

*Line by line:* `temperatures is None` is how `Sampler.prepare` says "every request on this tray is greedy" → plain `argmax` per row. Otherwise: float32 softmax with per-row temperature, optionally trimmed by top-k/top-p, then a weighted draw. **Who stops the loop?** Not the sampler: the *scheduler* checks length limits and EOS after each pass. The book's last line contrasts two PyTorch switches: `eval()` changes layer behaviour (such as dropout); `inference_mode()` turns off gradient recording. Different jobs.

## 5 Check Shapes and Causality

The book's regular-layout example: `B=1, T=3, C=32, Hq=4, Hkv=2, D=8, V=64`.

| Stage | Tensor shape | Why |
|---|---|---|
| Input ids / after embedding | `[1,3]` / `[1,3,32]` | 3 cards, each a 32-number vector |
| Q / original K and V | `[1,4,3,8]` / `[1,2,3,8]` | 4 query heads, 2 label/gift heads, each of size 8 |
| Attention scores | `[1,4,3,3]` | per query head, every position vs every position |
| All-position logits / next-token scores | `[1,3,64]` / `[1,64]` | every position scored vs only the last one kept |

Also useful (computed): `qkv_proj` weight `[(4 + 2·2)·8, 32] = [64, 32]`.

**Three checks to run on paper:**
1. *Changing a future token must not change earlier outputs.* True because of the causal mask: position `i` never reads positions `> i`.
2. *Each generation step appends exactly one token.*
3. *Generation stops at the length limit or at a confirmed EOS.*

Then ask what the LM head avoids by keeping only the final positions: the vocabulary multiply for every other position (for a 3,000-token prompt with a 151,936-word vocabulary: 3,000 times less FLOPs and about 1.8 GB less float32 logits).

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** Embedding → floors (norm, attention, norm, MLP, with the fused shortcut) → final norm → LM head on the *last* position of each request → sampler. mini-sglang packs all requests into one token list, fuses the residual add into the norm, and scores only the last positions.

### 6.2 Exercises

**1. Why does the prefill LM head not need a vocabulary projection at every input position?**

<details><summary>Answer</summary>

Only the last position's logits are needed to choose the first new token; earlier positions predict tokens that are already in the prompt. `get_last_indices` selects those rows first, so the big vocabulary multiply runs on `batch size` rows instead of `N`.

</details>

**2. Where does the residual addition happen if it is absent from `Qwen3DecoderLayer.forward`?**

<details><summary>Answer</summary>

Inside `RMSNormFused.forward`: it adds the previous sublayer's output to the `residual` (in place) and then normalises. The MLP's residual add is deferred to the next layer's `input_layernorm`, or to the final `self.norm` after the last layer.

</details>

**3. Which shapes change if `Hkv` goes from 2 to 1?**

<details><summary>Answer</summary>

K and V become `[1,1,3,8]` (and `k_proj`/`v_proj`/merged `qkv_proj` shrink: `qkv_proj` becomes `[(4+2)·8, 32] = [48, 32]`); the group size becomes 4. Q, scores `[1,4,3,3]` and the logits do not change. The KV cache is half as large.

</details>

**4. Without the causal mask, which output positions could change when future tokens change?**

<details><summary>Answer</summary>

Every position before the changed token (and the changed one itself) could change, because each position would be allowed to attend to later positions. With the mask, only positions at or after the change are affected.

</details>

**5. What does cache-less generation recompute, and what could be saved?**

<details><summary>Answer</summary>

Every step recomputes K and V (and all hidden states) for all earlier tokens. The K and V of earlier tokens never change (causal mask), so they can be saved and reused: the KV cache.

</details>

**Extra drills (not in the book).**

6. For a decode tray of 5 requests, what is `cu_seqlens_q`? (`[0, 1, 2, 3, 4, 5]`.) And `get_last_indices(5)`? (`[0, 1, 2, 3, 4]`: every row is already a last row.)
7. Why does `RopeAttn` use one merged `qkv_proj` instead of three matrices? (One bigger matrix multiply uses the GPU better and reads the input once.)
8. Why is the softmax in the teaching snippet done in float32 and cast back? (Softmax sums exponentials; low-precision formats lose accuracy there.)
