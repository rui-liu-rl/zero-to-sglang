# Part II Chapter 3 Forward Pass and Generation: ELI5 companion

Original: [Chapter3_Forward-Pass-and-Generation.md](../../course-material/eng/part2/Chapter3_Forward-Pass-and-Generation.md). Same headings, same numbers. The code half is [Chapter3_Forward-Pass-and-Generation_code.md](./Chapter3_Forward-Pass-and-Generation_code.md). Cast and picture legend: [../README.md](../README.md).

This chapter reuses ideas from Part I chapters 1, 2 and 4 (attention, RoPE, norm, the KV cache). Where an idea was already explained there, this page gives the short version and points back.

## The one-minute version

**Kid version.** The chef cooks one word in five steps:

```text
   cards ──▶ ① VECTORS ──▶ ② N FLOORS of (meet, then think) ──▶ ③ SCORES for every word ──▶ ④ PICK ONE ──▶ new card
   (ids)    (embedding)     (attention + MLP, N times)           (LM head → logits)         (sampling)
                                    ▲                                                           │
                                    └──────────── append the new card, repeat ◀─────────────────┘
```

① look up a vector for each card; ② run the vectors up a tall building where on each floor words first *look at each other* (attention) and then *think alone* (MLP); ③ turn the top floor's vector into one score per vocabulary word; ④ pick a word; then append it and go again. **Prefill** and **decode** are the same loop, with a different amount of text already known.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Core concepts of this chapter | The five steps above |
| Trace a token's path from input to the next sampled token | Cards → vectors → floors → scores → pick |
| Roles of forward pass, generation, prefill, decode, KV cache | Forward pass = one trip up the building; generation = many trips in a loop; prefill/decode = the first trip vs the later trips; KV cache = notes that save re-doing old floors' work |

## 2 The Core of a Forward Pass: Attention, Blocks, and Logits

### 2.1 From Text to Embeddings: Tokenization

**Kid version.** The model can only read numbers. The **tokenizer** chops text into reusable *pieces* and gives each piece a number. An **embedding** is then a giant *lookup table*: row number = card number, row contents = a list of numbers that is the model's first guess about that card.

```text
   "I love datawhale"  ─split─▶  ["I", "love", "data", "whale"]  ─look up─▶  [40, 3021, 695, 12084]
                                         (pieces)                              (ids; made up)
   embedding table (vocab_size rows x d_model columns):
        row 0     [ . . . . . . . . ]
        ...
        row 3021  [ 0.12 -0.40 ... ]   ← this card's starting vector
```

**Where do the pieces come from? A tiny BPE run.** Byte-pair encoding (BPE) starts with single letters and repeatedly *glues together the most frequent neighbouring pair*. Here is a real run on a four-word toy corpus (`low` x5, `lower` x2, `newest` x6, `widest` x3; `</w>` marks "end of word"; ties broken arbitrarily):

| Step | Merge (count) | Pieces now |
|---|---|---|
| 1 | `t` + `</w>` (9) | `newest` = n e w e s **t</w>** |
| 2 | `s` + `t</w>` (9) | n e w e **st</w>** |
| 3 | `e` + `st</w>` (9) | n e w **est</w>** |
| 4 | `o` + `w` (7) | **ow** (so `low` = l **ow** </w>) |
| 5 | `l` + `ow` (7) | **low** |
| 6 | `w` + `est</w>` (6) | n e **west</w>** |
| 7 | `n` + `e` (6) | **ne** west</w> |
| 8 | `ne` + `west</w>` (6) | **newest</w>** (a whole word) |

Frequent words become one piece; rare words stay as a few pieces; any word can always fall back to single letters (or bytes), so nothing is "unknown". Real models do ~150,000 such merges (Qwen3's vocabulary has 151,936 pieces).

| Method | Kid version |
|---|---|
| BPE | Glue the most common neighbours, again and again |
| WordPiece | Like BPE, but chooses merges by how much they help a probability model (BERT) |
| SentencePiece | A toolkit that works on raw text including spaces; good for many languages |

**The embedding in numbers.** Table shape `[vocab_size, d_model]`. For Qwen3-0.6B: `151,936 x 1,024 = 155.6 million` numbers (about 0.31 GB in bf16, and the same table is reused as the output layer because the model "ties" them). After lookup, a sentence of `T` cards becomes a `[T, d_model]` matrix; with a batch of `B` sentences, `[B, T, d_model]`. In the real engine tokens from all requests are packed into one long list `[N, d_model]` (see the code half).

**Where the story lies.** The tokenizer does *no* thinking and takes no part in the model's math; it is a dictionary. The embedding vector knows only *which card* it is, not what is around it. Context arrives later (attention).

### 2.2 Positional Encoding: Telling the Model About Order

**Kid version.** The lookup table gives "cat" the same vector wherever it sits, so "cat chases dog" and "dog chases cat" would start identical. RoPE fixes this by **rotating** each word's Question and Label arrows by an angle that grows with the word's seat number. After rotation, how well two words match depends only on *how far apart* they sit (Part I chapter 1, section 4.1, has the clock-hand picture and the numbers).

In the code, the seat numbers come from the **scheduler**: `Batch.positions` holds, for every token being computed, its position inside its own request (for a cached prefix of 6 tokens and a 4-token extension, positions 6, 7, 8, 9). The rotation happens inside the attention layer (`AttentionLayer.forward` → `rotary.forward`) on Q and K only; V is not rotated.

### 2.3 Attention: Tokens Begin to Communicate

**Kid version.** Meetings again (Part I chapter 1, 4.2). After the first floor's meeting, each word is a blend of itself and the words it was allowed to see. This is the costliest part of a forward pass, because every word is compared with every earlier word.

**Q, K, V.** Three small matrices turn the same input vector into three versions: Question ("what am I looking for?"), Label ("what do I contain?"), Gift ("what would I hand over?").

**Multi-head split.** The vector is cut into `H` slices; each slice has its own meeting at the same time; results are glued back together.

**Grouped-query attention (GQA), the modern twist.** Real models give the Questions more slices than the Labels and Gifts. Example (this chapter's own numbers): `Hq = 4` Question heads, `Hkv = 2` Label/Gift heads, each of size `D = 8`, hidden size `C = 32`:

```text
   Question heads:   q0   q1   q2   q3          (4 heads)
                      \   /     \   /
   shared Label/Gift:  kv0       kv1             (2 heads)  → group size = Hq / Hkv = 2
   head h uses kv head  h // 2  →  [0, 0, 1, 1]
```

The projection matrices have these shapes (rows = outputs, columns = inputs):

| Matrix | Shape | Why |
|---|---|---|
| `q_proj` | `[Hq·D, C] = [32, 32]` | 4 heads of size 8 |
| `k_proj`, `v_proj` | `[Hkv·D, C] = [16, 32]` each | 2 heads of size 8 |
| merged `qkv_proj` | `[(Hq + 2·Hkv)·D, C] = [64, 32]` | the three stacked: one matrix multiply instead of three |

Why merge? One big matrix multiply keeps the GPU busier than three small ones and reads the input once.

**A real GQA run (my toy, computed).** `T = 3` words, `D = 2`, `Hq = 2`, `Hkv = 1`, so both Question heads share ONE set of Labels/Gifts but still ask different questions:

| | position 0 | position 1 | position 2 |
|---|---|---|---|
| query head 0: weights | [1.00] | [0.33, 0.67] | [0.25, 0.25, 0.50] |
| query head 1: weights | [1.00] | [0.67, 0.33] | [0.58, 0.14, 0.28] |

Same K and V, different Q, so different weights, different outputs. That is why GQA keeps most of the quality while the notepad (KV cache) is `Hq / Hkv` times smaller.

**The causal mask.** Future positions' scores are set to minus infinity before softmax, so position `t` sees only positions `0..t`. (Part I chapter 4 shows why this makes old K and V cacheable.)

**Where the story lies.** "The hidden size `C` equals `Hq · D`" is **not** always true. Qwen3-0.6B has `C = 1024` but `Hq · D = 16 · 128 = 2048`; the attention is "wider" than the hidden size, and the output projection maps `2048 → 1024` back.

### 2.4 Transformer Blocks: Attention, MLP, Residuals, and Normalization

**Kid version.** One floor = meeting (attention) then private thinking (MLP), with the hallway shortcut (residual) and a thermostat (norm) *before* each of the two steps (**Pre-Norm**). Stack `N` floors, then one last thermostat (**final norm**).

```text
   x ─▶ norm ─▶ attention ─▶ (+ x) ─▶ norm ─▶ MLP ─▶ (+ previous) ─▶ next floor ...
```

Attention works **across** tokens; the MLP works on **one token at a time**. Residuals let signals and learning skip floors; Pre-Norm keeps numbers tame. The "fused add" detail (adding the shortcut inside the next norm) is explained step by step in Part I chapter 1, section 4.3.3, and in the code half here.

### 2.5 LM Head: From Vectors to Logits

**Kid version.** The top floor hands back one vector per word (`d_model` numbers). The **LM head** is a final matrix with one row per vocabulary word: multiply, and you get **one score for every word in the vocabulary**: the **logits**. They are raw scores, not percentages.

```text
   hidden vector  [d_model = 1024]  x  LM-head matrix  [vocab = 151,936  x  1024]  =  logits  [151,936]
```

**Only the LAST position matters when generating.** Position `t`'s logits say what should come *after* word `t`. During training all positions are scored (each has a known next word). During inference you only need the logits after the final word. **Tiny numbers** for a prompt of 3,000 tokens (Qwen3-0.6B):

| | Score all positions | Score only the last position |
|---|---|---|
| LM-head FLOPs (2 · tokens · 1024 · 151,936) | 9.3 x 10^11 | 3.1 x 10^8 |
| Logits tensor in float32 | 1.82 GB | 0.61 MB |

That is **3,000 times** less work and memory, which is why mini-sglang's LM head, during prefill, first keeps only each request's last hidden vector (`get_last_indices`) and *then* does the big multiply.

### 2.6 Common Structural Variants in Modern LLMs

The book's table is explained in Part I chapter 1, section 4.5, which adds a column "what this repo uses". The one-line takeaway: the forward pass is always *embed → N blocks → final norm → LM head → logits*; the variants (MoE, GQA, RMSNorm, Pre-Norm, RoPE, FlashAttention) change what happens *inside* a block, not the route.

## 3 From Logits to Text

**Kid version.** The scores tell the chef how much each word is *liked*. Someone must now actually *choose*. The choice rule (the **sampling policy**) is the only place randomness enters, which is why the same prompt can give different answers.

### 3.1 Why Sampling Is Needed

Language generation has many right answers; the model gives a *distribution* over next words. Always taking the top word is just one policy.

### 3.2 Common Sampling Strategies

#### 3.2.1 Greedy Decoding

Always take the highest score. Deterministic, good for factual answers, can be repetitive. (Because softmax preserves order, greedy never needs softmax.)

#### 3.2.2 Temperature

Divide the scores by `T` before turning them into percentages: `p_i = exp(logit_i / T) / sum_j exp(logit_j / T)`. Numbers for logits `[2.0, 1.0, 0.1]`:

| `T` | Probabilities | Feel |
|---|---|---|
| 0.1 | ~1.00, ~0.00, ~0.00 | nearly greedy |
| 0.5 | 0.864, 0.117, 0.019 | focused |
| 1.0 | 0.659, 0.242, 0.099 | natural |
| 2.0 | 0.502, 0.304, 0.194 | adventurous |
| `T → ∞` | → 1/3 each | pure noise |

#### 3.2.3 Top-k and Top-p

Both **cut off the unlikely tail**, renormalise what is left, then roll the die.

- **Top-k:** keep exactly the `k` likeliest words.
- **Top-p (nucleus):** keep the *smallest* set whose probabilities add up to at least `p`.

**Why top-p is "smarter":** its set size adapts to how confident the model is. Same settings on two different distributions:

| Situation | Probabilities | `top_k = 3` keeps | `top_p = 0.9` keeps |
|---|---|---|---|
| Model is sure (peaked) | 0.70, 0.20, 0.05, 0.03, 0.01, 0.01 | 3 words | **2 words** (0.70 + 0.20 = 0.90) |
| Model is unsure (flat) | 0.20, 0.20, 0.20, 0.20, 0.10, 0.10 | 3 words | **5 words** (0.20 x 4 + 0.10 = 0.90) |

Top-k is blind to the situation; top-p keeps few words when the model is sure and many when it is not.

#### 3.2.4 Strategy Summary

| Name | Where it acts | Kid version | Extreme case |
|---|---|---|---|
| Greedy | Final choice | Always the best | = top-k 1 = `T → 0` |
| Temperature | Reshapes the scores | Sharpen or flatten | `T = 1` changes nothing |
| Top-k | Trims candidates | Keep the best `k` | `k = vocab size` = no trimming |
| Top-p | Trims candidates | Keep the smallest group covering `p` | `p = 1` = no trimming |

### 3.3 The Autoregressive Generation Loop

Predict, append, repeat, forever until a stop. Each step depends only on the past (the causal mask guarantees it). In **training** the true next words are known, so all positions are scored at once; in **inference** each next word must be produced before the one after it can be.

**Stop conditions:** (1) the end-of-sequence token is sampled; (2) `max_new_tokens` reached; (3) the model's maximum context length reached.

## 4 Prefill, Decode, and KV Cache

Both phases are the same loop; the question is only "which tokens are already known?".

| | Prefill | Decode |
|---|---|---|
| Known | The whole prompt | Everything so far, but only the newest token is *new* |
| Tokens run through the model | Many at once (parallel) | One per order per step |
| Produces | The KV cache for the prompt and the first answer word | One more answer word + one more K/V column |
| Limited by | Compute (Tensor Cores) | Memory bandwidth |
| What the user feels | Time to first token (TTFT) | Time per output token (TPOT) |

(Numbers behind "compute-bound vs memory-bound": Part I chapter 2, sections 4 and 5. In the code: `Batch.phase` is `"prefill"` or `"decode"`, and `Req.extend_len` is "many" for prefill and exactly 1 for decode.)

## 5 Key Concepts

| Term | Kid version |
|---|---|
| **Hidden state** | The vector a word carries up the building; gets richer floor by floor |
| **Causal mask** | A rule: no peeking right |
| **MLP / FFN** | Each word thinking alone in its private office |
| **LM head** | The final matrix that turns a vector into one score per vocabulary word |
| **Logits** | Raw scores; can be negative; don't sum to 1 |
| **Softmax** | Scores → percentages that add to 100% |
| **Sampling** | Choosing a word from the percentages |
| **Stopping conditions** | EOS, `max_new_tokens`, maximum context length, a custom stop string, structured output finishing, or an external cancel |
| **EOS** | The model's "I'm done" token |
| **`max_new_tokens`** | A hard cap on how many words may be written |
| **Maximum context length** | The most cards (prompt + answer) the model can handle at once |
| **TTFT** | Time to first bite (driven by prefill) |
| **TPOT / ITL** | Time between bites (driven by decode) |

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** Text → cards → vectors → `N` floors (meet, think) → final thermostat → LM head → logits → sampling policy (greedy / temperature / top-k / top-p) → a new card, appended, repeat. Prefill reads the whole prompt in one go and fills the notepad; decode writes one card at a time reading the notepad.

### 6.2 Exercises

**1. The path of one token, from input to the next sampled token.**

<details><summary>Hint</summary>

Token id → embedding row → add/rotate position info → `N` blocks (attention across tokens, MLP per token, residual + norm) → final norm → LM head → logits at the last position → softmax with temperature → filter (top-k/top-p) → sample → append.

</details>

**2. Logits vs probabilities; why not always argmax; effect of temperature, top-k, top-p.**

<details><summary>Hint</summary>

Logits are unbounded raw scores; softmax makes them non-negative and sum to 1. Always taking the largest is deterministic and repetitive; sampling gives variety. Temperature reshapes the distribution; top-k and top-p remove the unlikely tail (top-p adapts to the shape: 2 words in the peaked example, 5 in the flat one).

</details>

**3. Differences between prefill and decode; why compute-bound vs memory-bound.**

<details><summary>Hint</summary>

Prefill processes many tokens per weight read (arithmetic intensity about the prompt length); decode processes one token per weight read plus a growing KV read (intensity about the batch size, near 1 for a single request).

</details>

**Extra drills (not in the book).**

4. A prompt has 10 tokens and the model has 28 layers. After prefill, how many (token, layer) K/V entries are in the cache? (10 x 28 = 280.)
5. With `Hq = 16` and `Hkv = 8`, how many Question heads share each Label/Gift head? (2.)
6. If you compute logits for all 3,000 prompt positions instead of the last, how much extra memory does the float32 logits tensor use for a 151,936-word vocabulary? (About 1.8 GB.)
