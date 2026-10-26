# Chapter 4 KV Cache: The Core Data Structure of Inference: ELI5 companion

Original: [Chapter4_KV Cache The Core Data Structure of Inference.md](../../course-material/eng/part1/Chapter4_KV%20Cache%20The%20Core%20Data%20Structure%20of%20Inference.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

## The one-minute version

**Kid version.** While the chef writes an answer, each new word needs to "look back" at every earlier word. To look back, the chef needs, for each earlier word, its **Label (Key)** and its **Gift (Value)**. Those never change once computed. So after working them out the first time, the chef **writes them on a notepad** and just reads the pad next time. That notepad is the **KV cache**.

```text
   NOTEPAD for one customer (one request):     one row per floor (layer), one column per word so far
                       word 1   word 2   word 3   word 4  ...
   floor 1   K,V    [ k1,v1 ] [ k2,v2 ] [ k3,v3 ] [ k4,v4 ]
   floor 2   K,V    [ k1,v1 ] [ k2,v2 ] [ k3,v3 ] [ k4,v4 ]      every box = different numbers,
   ...                                                          because each floor has its own recipe
   floor L   K,V    [ k1,v1 ] [ k2,v2 ] [ k3,v3 ] [ k4,v4 ]
   new word arrives ─▶ compute ITS q,k,v on every floor ─▶ add a new column ─▶ read the whole pad
```

**What it saves, in numbers.** Prompt of 100 tokens, then 200 generated tokens (the book's example).

| Strategy | Token-forward-passes of work |
|---|---|
| No notepad: every step redoes all tokens so far (passes over 100, 101, ..., 299 tokens) | **39,900** |
| Notepad: prefill the 100 once, then 1 per new token (199 more passes; the 200th word is never fed back) | 100 + 199 = **299** |

That is a 133x difference in work. The price: **memory** that grows with every word (the rest of the chapter).

## 1 Learning Objectives

| Original question | Short kid answer (full answer in the section named) |
|---|---|
| Why a KV cache from the attention formula? Which computations are reused? | Old K and V never change (nobody looks at future words), so reuse them. New Q, K, V are still computed. See 2 |
| Cache lifecycle of one request | Reserve space, write in prefill, append per decode step, free (or keep for reuse). See 3 |
| How many bytes? | `tokens x layers x 2 x kv_heads x head_dim x bytes`. See 4 |
| Numbers for a real model | Qwen3-4B-style: 144 KiB per token. See 4 |
| Why does naive management waste memory? | Fragmentation, over-reservation, duplicated prefixes. See 5 |

## 2 The KV Cache in the Attention Mechanism

### 2.1 Where the KV Cache Comes From

**Kid version.** Every word, on every floor, writes three index cards from its current vector (a matrix multiply each):

| Card | What it says | Kept? | Why |
|---|---|---|---|
| **Q** (Question) | "What am I looking for?" | **No** | Used only to compute *this* word's own result, right now |
| **K** (Label) | "What do I contain?" | **Yes** | Every *later* word will compare its Question against this |
| **V** (Gift) | "What do I hand over?" | **Yes** | Every later word may take a share of this |

**Why Q is thrown away (the one-line proof).** The result for word `i` depends only on word `i`'s Question and words `1..i`'s Labels and Gifts (the causal mask forbids looking right). Once word `i`'s result is computed it never needs recomputing, and no *later* word ever uses word `i`'s Question. So Questions are dead after use; Labels and Gifts stay alive for the rest of the conversation.

**Who creates what:**

```text
   PREFILL :  prompt tokens 1..n all at once  → their K,V written to the pad (all floors)
                                              → only the LAST position's output is used to pick word n+1
   DECODE  :  one new token each step         → its q,k,v computed → k,v APPENDED → q reads the pad
```

**Why only the last position?** After prefill, positions 1 to n-1 already "know" their next words (they are in the prompt). Only position n predicts something new. A lean engine runs the final output layer (the vocabulary scoring) just for those last positions, not for every prompt token.

### 2.2 How the KV Cache Works

**Kid version.** The same story as formulas. At position `t` the model has the word's vector `h_t` and makes:

```text
   q_t = h_t W_q     k_t = h_t W_k     v_t = h_t W_v         (three small matrix multiplies)
```

The pad grows by one column, then the new Question reads it:

```text
   K_{1:t+1} = [ K_{1:t} ; k_{t+1} ]      V_{1:t+1} = [ V_{1:t} ; v_{t+1} ]        (append)
   o_{t+1}   = softmax( q_{t+1} K_{1:t+1}^T / sqrt(d_h) ) V_{1:t+1}                  (read all)
```

*In words:* compare my Question to every Label (including my own), turn that into percentages, take that mix of Gifts. No mask is needed in decode because my Question is a single row that is allowed to see everything already cached.

**One pad per floor.** `KV = [ (K1,V1), (K2,V2), ..., (KL,VL) ]`. Each floor has its own recipe weights, so each floor makes different K and V for the same word.

**The loop after attention:**

```text
   o_{t+1} → FFN, residual, norm → next floor ... → last floor → scores over the vocabulary (logits)
        → sampling picks a token ID → look up that ID's vector → this is the NEXT step's input
```

Note the token ID is what is fed back (not the last hidden vector). Only at the very end are IDs turned back into text.

**How the real engine stores the pad** (a pointer into the code you will read). In `mini-sglang`, [`kvcache/mha_pool.py`](../../../mini-sglang/python/minisgl/kvcache/mha_pool.py) class `MHAKVCache` allocates **one big tensor up front**:

```text
   _kv_buffer shape = ( 2 , num_layers , num_pages , page_size , local_kv_heads , head_dim )
                        K/V   floors       which page   slot in page   heads         numbers per head
```

It is a *shared pantry* for all customers, not a notepad per customer. `num_pages * page_size` is the number of **token slots** in the whole pool. When a layer computes new K and V, `store_kv(k, v, out_loc, layer_id)` writes them into the slots named by `out_loc` (a list of slot numbers handed out by the scheduler). Everything in section 5 is about *which customer gets which slots*.

**Pause and predict.** The pool has 1,000 pages of 16 slots each. How many tokens of notes can it hold in total, across all customers?

<details>
<summary>Answer (try first)</summary>

1,000 x 16 = 16,000 token slots (per layer, per K/V; the buffer holds all layers and both K and V for each slot). Memory is that many slots times the per-token cost from section 4.

</details>

## 3 The KV Cache Lifecycle

**Kid version.** We follow one tiny customer from arrival to departure, by hand. The mini model: 4 numbers per word, 1 floor, 1 head, and (to keep arithmetic clean) Q = K = V = the word's own vector. These simplifications change nothing about how the pad behaves.

The order is "I love AI." and the words become:

| Position | Word | Vector |
|---|---|---|
| 1 | I | [1, 0, 1, 0] |
| 2 | love | [0, 1, 1, 0] |
| 3 | AI | [1, 1, 0, 1] |

### 3.1 Prefill

**Kid version.** The chef reads all three words at once, writes three columns on the pad, and uses the *last* word to guess what comes next.

**Step by step (checked by computer).**

1. Scores `Q K^T / sqrt(4)` for every pair, before the mask:

```text
              k1    k2    k3
   q1  [      1.0   0.5   0.5 ]
   q2  [      0.5   1.0   0.5 ]
   q3  [      0.5   0.5   1.5 ]
```

2. Hide the future (set to minus infinity):

```text
   q1  [      1.0    -∞    -∞ ]
   q2  [      0.5   1.0    -∞ ]
   q3  [      0.5   0.5   1.5 ]
```

3. Softmax each row:

```text
   q1  [  100%      0%      0%  ]
   q2  [ 37.8%   62.2%      0%  ]
   q3  [ 21.2%   21.2%   57.6%  ]
```

4. Output rows (mix of Gifts):

```text
   o1 = [1.000, 0.000, 1.000, 0.000]
   o2 = [0.378, 0.622, 1.000, 0.000]
   o3 = [0.788, 0.788, 0.424, 0.576]
```

The pad now holds:

```text
   K = V = [ [1,0,1,0] , [0,1,1,0] , [1,1,0,1] ]     (3 columns)
```

5. Only `o3` continues (through the rest of the floor, then to the output layer) to produce scores over the 10-word vocabulary. Say softmax gives: `very 0.63, models 0.12, great 0.08, are 0.05 ...`. Greedy picks **very**.

**What Prefill accomplished:** (1) built the initial pad; (2) used the last position to get the first new word.

### 3.2 Decode

**Kid version.** The new word "very" arrives. It writes its own three cards, **appends** its Label and Gift to the pad, then reads the *entire* pad with its Question.

Using the book's assumed vector for "very": `x4 = [0.82, 0.726, 0.548, 0]`, so `q4 = k4 = v4 = x4`.

The pad grows to four columns:

```text
   K = V = [ [1,0,1,0] , [0,1,1,0] , [1,1,0,1] , [0.82,0.726,0.548,0] ]
```

**The step the book leaves out (computed here).**

```text
   scores  q4·K^T / 2 =  [ 0.684 , 0.637 , 0.773 , 0.750 ]       (match with I, love, AI, very)
   softmax            =  [ 24.3% , 23.2% , 26.6% , 26.0% ]        (fairly even: "very" looks everywhere)
   output  o4         =  [ 0.721 , 0.686 , 0.617 , 0.266 ]
```

Then `o4` goes through the rest of the floor and the output layer, and sampling picks the fifth word ("much" in the book's story).

**What was NOT recomputed:** the K and V of "I", "love", "AI". Without a pad, step 4 would redo the projections for all 4 words; with it, just for 1.

**Why the old columns stay valid (the key fact).** Compare the prefill table with what step 4 would show if we redid everything: rows 1-3 of the attention table do not change, because each row only looks left. Adding word 4 only adds a new row (and a new column that earlier rows cannot see). That is why appending is *correct*, not just fast.

**The per-step recipe, from the book:**

```text
   new token ID → embedding → q,k,v → append k,v → q reads all cached K,V → predict next → repeat
```

**Where the story lies.** "Append" sounds like one contiguous strip growing in memory. In real engines (paged caches, prefix reuse, sliding windows, rollback in speculative decoding) the new entries may land in scattered pages. The *logical* picture stays "append a column"; the *physical* layout is up to the allocator.

**RoPE footnote.** In real models the rotation by position (Chapter 1) is applied to K before it is written, so cached Labels already contain their seat number. A cached prefix is therefore only valid at the same positions.

### 3.3 Request Completion

**Kid version.** The customer leaves when (a) the chef writes the "I'm done" token (**EOS**), (b) the answer hits the length limit, or (c) the customer walks out (cancelled). The pad's pages are handed back.

How the hand-back works: every page has a **reference count**, "how many people are still using this page". When a request leaves, each of its pages' counts goes down by one. A page whose count reaches 0 returns to the free pool.

```text
   page 7: used by request A, request B   count=2
   A finishes  →  count=1   (B still needs it; page stays)
   B finishes  →  count=0   → back to the free pool
```

If **prefix caching** is on, a cache index (Part II chapter 8) keeps pointing at pages that hold reusable prefixes, so they are *not* returned right away. They sit in the pool until an **eviction** policy reclaims them when space is needed.

**The whole life in one line:** `allocate → bulk-write (prefill; skip what a prefix hit already has) → read+append (decode) → free, or keep for reuse`.

**Pause and predict.** A request is cancelled halfway through decode. Which of the three finish conditions is that, and what must the engine do besides stop computing?

<details>
<summary>Answer (try first)</summary>

Cancellation. It must release the request's pages (decrement reference counts) and remove it from the running batch; otherwise memory leaks and the GPU keeps computing for nobody (Part II chapter 2's point about disconnects).

</details>

## 4 GPU Memory Usage of a Naive KV Cache

**Kid version.** How big is the notepad? Count the boxes and multiply by the size of a box.

```text
   one box   = the Key (or Value) vector of ONE word on ONE floor, for ONE kv-head   = d_h numbers
   per word  = 2 (K and V) x n_kv heads x d_h numbers x L floors x s bytes
   per request = tokens x per word
```

**Symbols** (the book's): `B` requests, `T` prompt tokens, `L` layers, `n_kv` KV heads, `d_h` head size, `s` bytes per number, `l` generated tokens.

**1. After prefill:** `KV_bytes = B x T x L x 2 x n_kv x d_h x s`.
**2. Added by decode:** `B x l x L x 2 x n_kv x d_h x s`.
**Total:** `B x (T + l) x L x 2 x n_kv x d_h x s`. With unequal lengths, replace `B x (T + l)` with the sum of each request's length.

### Real models, one table

Bytes per token = `2 x L x n_kv x d_h x s`. All in bf16 (s = 2):

| Model | L | n_kv | d_h | Bytes per token | Why |
|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | 8 | 128 | **112 KiB** | verified from its config |
| Qwen3-4B-Thinking-2507 (book) | 36 | 8 | 128 | **144 KiB** | book's example |
| Qwen2.5-1.5B (book exercise 3) | 28 | 2 | 128 | **28 KiB** | only 2 KV heads: tiny pad |
| LLaMA-2-7B style (classic MHA) | 32 | 32 | 128 | **512 KiB** | every head has its own K,V |
| LLaMA-3-8B style (GQA) | 32 | 8 | 128 | **128 KiB** | 4x smaller than MHA at the same L |

*Why the first-principles formula catches design choices:* more layers, more KV heads, bigger head size, more bytes per number each multiply the pad. GQA divides it by `heads / n_kv`; KV quantization (fp8) halves `s`.

### The book's worked example, checked

Qwen3-4B-Thinking-2507: `2 x 36 x 8 x 128 x 2 = 147,456 B = 144 KiB` per token.

| Quantity | Calculation | Value |
|---|---|---|
| After prefill, `T = 100` | 100 x 147,456 | 14,745,600 B = **14.06 MiB** |
| One decode step adds | 147,456 | **0.141 MiB** |
| Total after 1 step | 101 x 147,456 | 14.20 MiB |
| After `l = 200` | 300 x 147,456 | 44,236,800 B = **42.19 MiB** |
| One request at 4K tokens | 4,096 x 147,456 | 0.56 GiB |
| One request at 32K tokens | 32,768 x 147,456 | 4.50 GiB |
| Eight 32K requests | 8 x 4.50 | **36 GiB** |
| The weights | 4e9 x 2 bytes | 7.45 GiB |

**Drift note on units.** The book writes "MB" and "GB", but its numbers are in binary units (1 MiB = 1,048,576 B). 14,745,600 B is 14.06 MiB and 14.75 MB. Not a mistake in the maths; just a label.

**Reading the table:** eight long conversations make the notepads **4.8 times the size of the entire recipe book** (36 vs 7.45). With short conversations the weights dominate. Long contexts and many concurrent customers flip it.

### Why the notepad matters more than the "temporary stuff"

During a forward pass each floor also makes temporary results (**activations**), e.g. one vector per word per floor. For Qwen3-0.6B (hidden size 1,024) and a 1,000-token prompt, one such tensor is `1,000 x 1,024 x 2 B = 2 MB`, and an engine keeps only a few at a time, then reuses the space. The pad for those same 1,000 tokens is `1,000 x 112 KiB = 109 MiB`, and it must **stay for the whole conversation**, on every floor. Temporary = small and short-lived; the pad = big and long-lived.

| | Activations | KV cache |
|---|---|---|
| Lives for | one floor's computation | the whole request |
| Grows with | tokens in this step | all tokens so far x layers |
| Reused? | the memory is reused immediately | no; kept until the request ends |

**Pause and predict.** You switch a model from full multi-head attention (32 KV heads) to GQA with 8 KV heads, everything else equal. By what factor does the pad shrink, and how many more conversations fit?

<details>
<summary>Answer (try first)</summary>

The pad shrinks 4x (32 to 8 KV heads). If the pad is the memory limit, about 4x as many conversations fit (the weights stay the same, so slightly less than 4x in total).

</details>

## 5 GPU Memory and Traditional Cache Management Problems with Multiple Requests

**Kid version.** Imagine the pantry is a long shelf of 100 slots, and you give each customer one **fixed-length block of slots**, big enough for their *longest possible* answer. That's the "one contiguous tensor per request" approach. It sounds simple. It goes wrong in four ways.

### Problem 1: wasted reservations and fragmentation

Numbers: the pool has 100 slots; every request reserves 30 (the maximum it might use). Three customers arrive:

```text
   slots  0         30        60        90   99
          ┌─────────┬─────────┬─────────┬────┐
          │ A: 10/30│ B: 12/30│ C:  8/30│free│        used 30 of 90 reserved = 33%
          └─────────┴─────────┴─────────┴────┘        and the free 10 can't fit anyone new
```

Two thirds of what's reserved sits **empty**: nobody else may use it. Now A and C leave:

```text
          ┌─────────┬─────────┬──────────────────┐
          │  free 30│ B: 12/30│     free 40      │      70 slots free in total
          └─────────┴─────────┴──────────────────┘
   New customer D reserves 50 contiguous slots → impossible (biggest run is 40), though 70 are free.
```

The first waste is **internal** (reserved but unused). The second is **external fragmentation** (free but in unusable pieces). Growing a block in place may also force a copy to a new place.

### Problem 2: duplicated prefixes

Many requests start with the same system prompt. If each has its own pad, the same Labels and Gifts are stored many times. For Qwen3-4B, a 2,000-token system prompt is 281 MiB; a thousand customers store **275 GiB** of identical notes, versus 281 MiB if shared.

### Problem 3: scheduling and reclamation

Requests arrive, finish, get paused, get resumed. With scattered, resizable blocks the engine must also track locations, sizes, priorities and deadlines, and reclaim or relocate blocks. More bookkeeping; more copying.

### Problem 4: awkward continuous batching

Continuous batching (Part II chapter 6) swaps requests in and out of the running batch every step. With contiguous storage:

- you must track each request's start and length, or
- you pad every request to the longest one so the data forms a rectangle. Lengths 100, 2,000 and 50 padded to 2,000 means 6,000 slots of work for 2,150 real tokens: **64% waste**.
- repacking to avoid padding costs gathers and data movement.

### The moral, and a preview of the fix

**Don't give each customer one big strip; give them small fixed pages from a shared pile.** Page size 4, as a picture:

```text
   pool of pages :  P0 P1 P2 P3 P4 P5 P6 P7 ...                       (all the same size)
   customer A    :  token 0-3 → P5 | tokens 4-7 → P2 | tokens 8-9 → P7 (half-used)
   page table A  :  [5, 2, 7]                                            (logical → physical)
   waste per customer ≤ page_size - 1 slots (only the last page can be partly empty)
```

| Tokens | Pages (size 4) | Wasted slots |
|---|---|---|
| 1 | 1 | 3 |
| 5 | 2 | 3 |
| 9 | 3 | 3 |
| 17 | 5 | 3 |

Waste per request never exceeds 3 slots, pages need not be next to each other, freed pages go straight back to the pile, and two customers can point at the **same** page for a shared prefix (reference count 2). That is **paged KV cache** (Part II chapter 7) plus a prefix index (**radix cache**, chapter 8).

**Pause and predict.** With page size 16 versus 4, which wastes less at the end of a request, and what is the cost of the smaller pages?

<details>
<summary>Answer (try first)</summary>

Smaller pages waste less (at most 3 vs 15 slots). The cost is more pages to track: bigger page tables, more indirection when kernels gather pages, and less favourable memory access. Page size is a tuning knob (mini-sglang exposes `--page-size`).

</details>

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** The KV cache is the chef's notepad of Labels and Gifts for every word on every floor. It is produced in prefill, appended in decode, and thrown away (or kept for reuse) when the customer leaves. Its size is `tokens x floors x 2 x kv-heads x head-size x bytes`, so it grows linearly with conversation length and quickly outgrows the weights. Giving each customer one big contiguous strip wastes space, cannot share common beginnings, and fights continuous batching. So engines move to small pages from a shared pool.

```text
   work saved by the cache  :  recompute ALL history every step   →   compute each token once
   price paid               :  memory proportional to (tokens x layers x kv-heads x head-size)
   what engines then build  :  paging (no fragmentation) + prefix sharing (no duplication)
```

### 6.2 Exercises

**1. Which computations produce the KV cache, and what is it for?**

<details><summary>Hint</summary>

Prefill projects every prompt token to K and V on every layer and stores them; decode projects only the new token and appends. They let each new Question read old Labels/Gifts without recomputing them.

</details>

**2. The full lifecycle of one request's cache.**

<details><summary>Hint</summary>

Allocate slots/pages; bulk write in prefill (skipping prefix hits); per decode step read everything and append one entry; on EOS/limit/cancel, decrement reference counts and free (or keep reusable prefix pages until evicted).

</details>

**3. Qwen2.5-1.5B memory with `T = 200` and `l = 100`.**

<details><summary>Worked answer (try first)</summary>

Config: 28 layers, 2 KV heads, head size 128 (hidden 1536 / 12 heads), bf16. Per token: `2 x 28 x 2 x 128 x 2 = 28,672 B = 28 KiB`.

- End of prefill (`T = 200`): `200 x 28,672 = 5,734,400 B` = **5.47 MiB**
- Per decode token: **28 KiB** (0.027 MiB)
- At the end (`300` tokens): `8,601,600 B` = **8.20 MiB**

Reading: the prompt already accounts for two thirds (5.47 of 8.20); each generated token adds a small constant; total grows linearly. Compared with Qwen3-4B's 144 KiB per token, this model's pad per token is 5x smaller, because it has only 2 KV heads.

</details>

**4. What goes wrong with a separate contiguous region per request?**

<details><summary>Hint</summary>

Over-reservation and fragmentation (section 5 problem 1), no prefix sharing, extra bookkeeping, padding or repacking for continuous batching.

</details>

**5. Could some or all of the KV cache live in CPU memory? Benefits and costs.**

<details><summary>Worked answer (try first)</summary>

Yes, and engines do (hierarchical caching, Part III chapter 3). Benefit: GPU memory stops being the cap, so more conversations and longer contexts. Cost: speed. For one 32K-token conversation of Qwen3-4B (4.5 GiB of notes):

| Where the notes are read from | Time to read them once |
|---|---|
| GPU HBM (about 2 TB/s) | about 2.4 ms |
| CPU memory over PCIe 4.0 x16 (about 32 GB/s) | about 151 ms |

A decode step reads the whole pad, so keeping *hot* notes on the GPU is essential. What works well is moving **cold** pages (idle conversations, old prefixes) to the CPU and loading them back only when needed, ideally overlapped with other compute.

</details>

**Extra drills (not in the book).**

6. LLaMA-3-8B style (32 layers, 8 KV heads, head size 128, bf16): how much notes memory for one 8K-token conversation? (128 KiB x 8,192 = 1 GiB.)
7. Why would a model with 64 KV heads of size 128 be a poor choice for long-context serving? (The pad scales linearly with `n_kv`: at 32 layers and bf16 that is 1 MiB per token, so one 32K-token conversation needs 32 GiB of notes, 64 times the pad of a single-KV-head variant.)
8. A request ends and its prefix pages are kept by the cache index. What stops those pages from filling the pool forever? (An eviction policy that returns the least valuable unreferenced pages to the free pool.)
