# Part II Chapter 4 KV Cache Optimization: ELI5 companion

Original: [Chapter4_KV-Cache-Optimization.md](../../course-material/eng/part2/Chapter4_KV-Cache-Optimization.md). Same headings, same numbers. The code half is [Chapter4_KV-Cache-Optimization_code.md](./Chapter4_KV-Cache-Optimization_code.md). Part I chapter 4 ([../part1/Chapter4_KV-Cache-The-Core-Data-Structure-of-Inference.md](../part1/Chapter4_KV-Cache-The-Core-Data-Structure-of-Inference.md)) explains *what* the cache is and how big it is; this chapter is about *how much work it saves*.

## The one-minute version

**Kid version.** Without a notepad the chef, for each new word, **rereads the entire order from the top and redoes every earlier word's notes**. With a notepad the chef writes each word's notes **once** and, for each new word, only (1) writes that word's own notes and (2) glances over the pad. Two different savings hide in that sentence, and the chapter's maths is about telling them apart:

| What is saved | Without the pad | With the pad |
|---|---|---|
| Re-making old words' **K and V** | redone every step | **never redone** (computed once) |
| Comparing the new word with old words | a full `T x T` table every step | a single row of length `T` |

The second one is *not* removed, only shrunk: the new word must still look at every old Label.

```text
   step t, WITHOUT pad                         step t, WITH pad
   ┌─────────────────────────────┐             ┌─────────────────────────────┐
   │ x1 x2 ... y(t-1)  (T words) │             │ only y(t-1)  (1 word)       │
   │ make K,V for ALL T words    │             │ make K,V for 1 word, append │
   │ compare ALL pairs: T x T    │             │ compare it with T Labels    │
   └─────────────────────────────┘             └─────────────────────────────┘
```

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Why computation repeats without a cache | The forward pass has no memory: each call starts from zero |
| What is cached | Labels and Gifts (K and V) of old words, on every floor. Not Questions, not scores, not logits |
| Derive the complexity | Add up `1 + 2 + ... + n`, or add up squares, and see which one you are counting |
| Prefill vs decode | Prefill fills the pad in one go; decode adds one column per step |

## 2 Repeated Computation Without KV Cache

### 2.1 Generation Loop Without KV Cache

**Kid version.** Each step hands the model the **whole text so far**; the model returns scores for *every* position, but we use only the last row (it predicts the next word).

```text
   step 1: [ x1 x2 ... xp ]                   → logits for each position; use the LAST  → y1
   step 2: [ x1 x2 ... xp y1 ]                → again all positions; use the LAST       → y2
   step 3: [ x1 x2 ... xp y1 y2 ]             → ...                                      → y3
```

The old words' K and V did not change between steps. They were thrown away only because the forward pass is **stateless**: it recomputes whatever it is given and remembers nothing.

### 2.2 Where Computation Is Wasted: Repeated Historical K/V Projections

In every attention layer, **all `T` positions** are projected into Q, K and V again (each a `B x H x T x D` tensor), then `Q K^T / sqrt(D)` forms a `B x H x T x T` table (with the causal mask hiding the upper triangle), softmax, multiply by V, output projection, residual, norm, MLP. `T` is the full history length. The first `T - 1` words' K/V were already made last step; making them again is **the repeated work**.

### 2.3 A Small Example: From [x1, x2] to [x1, x2, y1, y2]

Prompt `[x1, x2]`, three new words. Counting exactly:

| Step | Input | K/V made for | Score table | Output |
|---|---|---|---|---|
| 1 | `[x1, x2]` | x1, x2 (2 words) | 2 x 2 = 4 | y1 |
| 2 | `[x1, x2, y1]` | x1, x2, y1 (3 words) | 3 x 3 = 9 | y2 |
| 3 | `[x1, x2, y1, y2]` | x1, x2, y1, y2 (4 words) | 4 x 4 = 16 | y3 |
| **Total** | | **9 word-projections** | **29 score cells** | |

And with a notepad:

| Step | New work | K/V made for | Score cells | Output |
|---|---|---|---|---|
| 1 (prefill) | read the prompt | x1, x2 | 2 x 2 = 4 | y1 |
| 2 (decode) | feed y1 | y1 only | 1 x 3 = 3 | y2 |
| 3 (decode) | feed y2 | y2 only | 1 x 4 = 4 | y3 |
| **Total** | | **4 word-projections** | **11 score cells** | |

Already 9 vs 4 and 29 vs 11 on a toy; the gap grows with length (next section).

### 2.4 How Much Is Repeated: O(n²) or O(n³)?

It depends **what you count**.

- **Count word-projections** (K/V made): step 1 makes `p` of them, step 2 `p + 1`, ..., so the total is `p + (p+1) + ... = ` roughly `n²/2`: **quadratic, O(n²)**.
- **Count score-table cells**: step `t` has a `T x T` table, and `1² + 2² + ... + n² =` roughly `n³/3`: **cubic, O(n³)**.

Both are true statements about different quantities. The phrase **"from O(n²) to O(n)"** refers to the first one.

**Exact counts** for a prompt of `p` words and `n` generated words (decode steps only; "ratio" = no-pad divided by pad):

| `p` | `n` | K/V projections, no pad | with pad | ratio | Score cells, no pad | with pad | ratio |
|---|---|---|---|---|---|---|---|
| 100 | 200 | 40,100 | 200 | **200x** | 8,706,700 | 40,100 | **217x** |
| 100 | 1,000 | 600,500 | 1,000 | **600x** | 443,933,500 | 600,500 | **739x** |
| 1,000 | 1,000 | 1,500,500 | 1,000 | **1,500x** | 2,334,833,500 | 1,500,500 | **1,556x** |
| 100 | 10,000 | 51,005,000 | 10,000 | **5,100x** | 343,484,335,000 | 51,005,000 | **6,734x** |

(The table uses the book's convention `T_t = p + t`. Counting passes literally, the first pass is over `p` words rather than `p + 1`, which shifts every figure by a small amount; the growth is what matters.) Longer answers make the gap grow roughly with `n`: that is the signature of "quadratic vs linear".

**Where the story lies.** These count *operations*. Real decode time is dominated by reading weights and the notepad from memory (Part I chapter 2), so the actual speed-up is smaller than the operation ratio. But the cache is what makes long generation practical at all.

## 3 What Does KV Cache Store?

**Kid version.** The chef's notepad has **one row per floor**, and on each row, one **Label card and one Gift card per word**. That's all.

### 3.1 Common Misconceptions

| Wrong idea | Why it is wrong |
|---|---|
| "It caches the hidden states of the whole sequence" | Only K and V (projections *inside* attention). The word's running vector is recomputed or passed along |
| "One cache shared by all layers" | Each floor has its own recipe, so its own K and V |
| "It caches the attention table" | The table is temporary, recomputed per step and thrown away |
| "It caches logits" | Logits exist only after the last floor and are used once, for the current word |
| "Only the last layer needs a cache" | Every self-attention layer needs its own |
| "Keys are cached before RoPE" | With RoPE, the **rotated** key is cached; the new Question is rotated at use time. Values are not rotated |

One more: the forward pass stays stateless; the **inference system** keeps the pad outside the model and passes it in.

## 4 KV Cache: From O(n²) to O(n)

Setup: prompt length `p`, `n` new words, total length at decode step `t` is `T_t = p + t`. We ignore constants (layers, width, heads) and look only at growth in length.

### 4.1 Without KV Cache: Historical K/V Is Recomputed Every Step

Projection work at step `t` is `O(T_t)`, so the total is `sum (p + t) = n p + n(n+1)/2 = O(n²)`. The attention table at step `t` has `T_t²` cells: `sum (p + t)² = O(n³)`. A single step's attention is `O(T_t²)`.

### 4.2 With KV Cache: Historical K/V Is Computed Only Once

Prefill makes K/V for all `p` words once. Each decode step makes **one** word's q, k, v: `O(1)` projection, total `O(n)`. Attention for the new Question runs over `T_t` Labels: `O(T_t)` per step, `sum (p + t) = O(n²)` over all steps. Per-step attention falls from `O(T²)` to `O(T)`.

### 4.3 Complexity Comparison

| What | Without cache | With cache | Kid reading |
|---|---|---|---|
| Per-step K/V projection | O(T) | O(1) | Redo all old words vs just the new one |
| Total K/V projection over n steps | O(n²) | **O(n)** | |
| Per-step attention | O(T²) | **O(T)** | Whole table vs one row |
| Total attention over n steps | O(n³) | O(n²) | Still quadratic: each step reads all history |
| Cache memory | 0 | O(T) | The price |

(With width `d`: attention becomes `O(T² d)` vs `O(T d)`; projection `O(T d²)` vs `O(d²)`. Length remains the main variable.)

### 4.4 From O(n²) to O(n)

The famous sentence means *only* the K/V projection total: `sum T_t = O(n²)` → `sum 1 = O(n)`. It does **not** say all inference becomes linear: the new word must still compare itself with every earlier Label, so total attention across the answer stays `O(n²)`, unless the model uses sliding-window, sparse or linear attention.

### 4.5 Total Complexity Including Prefill

| | Prefill | Decode | Whole job, with pad | Whole job, no pad |
|---|---|---|---|---|
| Attention | O(p²) | O(n²) | O((p + n)²) | O((p + n)³) |
| K/V projection | O(p) | O(n) | O(p + n) | O((p + n)²) |

**Where the story lies.** "Linear vs quadratic" talks about *how work grows with length*; it is silent about the memory price: the pad grows linearly with length and is what ends up limiting how many conversations fit (Part I chapter 4).

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** No pad: every word redoes all the old words' notes, so total notes-making grows like `n²` (and score cells like `n³`). With a pad: notes are made once, so notes-making grows like `n`; each new word still reads the whole pad, so attention still grows like `n²` overall. In exchange, memory grows linearly with layers, conversations and length.

### 5.2 Exercises

**1. Why are old tokens' K and V recomputed every decode step without the cache?**

<details><summary>Answer</summary>

The model's forward pass is a stateless function: it recomputes everything for the input it is given and retains nothing. Since the input is the whole history, the K and V of old tokens (which have not changed) are recomputed.

</details>

**2. Why store only K and V, not Q, the attention table or logits?**

<details><summary>Answer</summary>

A word's Question is used once, for its own output, and no later word reads it. The attention table is a temporary per-step value. Logits are only needed for the current word. K and V of old words are read by every later word, and never change (causal mask), so they are the only reusable data.

</details>

**3. How do prefill and decode read and write the cache, and what characterises each?**

<details><summary>Answer</summary>

Prefill writes K and V for all prompt tokens in one pass (bulk write, then reads them for causal attention); compute-heavy and parallel. Decode writes one token's K/V per step and reads the whole cache; memory-heavy and serial.

</details>

**4. What does "from O(n²) to O(n)" refer to? Why does decode attention remain O(n²) in total?**

<details><summary>Answer</summary>

It refers to cumulative K/V projection over `n` generated tokens (`sum T_t` becomes `n`). The new token still attends to every earlier Label each step, `O(T_t)` per step, so the sum over `n` steps is `O(n²)`.

</details>

**Extra drills (not in the book).**

5. For the 2-word-prompt example, how many score cells does the 4th step add with a pad? (It would be `1 x 5 = 5`.)
6. With `p = 100`, how much faster (in K/V projections) is generating `n = 2,000` words with a pad versus without? (`100 x 2000 + 2000·2001/2 = 2,201,000` vs `2,000`: about 1,100x.)
7. Why does the ratio grow roughly like `n / 2`? (Without a pad the average step redoes about `p + n/2` words; with a pad it does 1.)
