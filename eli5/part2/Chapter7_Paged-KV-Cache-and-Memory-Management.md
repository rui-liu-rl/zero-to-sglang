# Part II Chapter 7 Paged KV Cache and Memory Management: ELI5 companion

Original: [Chapter7_Paged-KV-Cache-and-Memory-Management.md](../../course-material/eng/part2/Chapter7_Paged-KV-Cache-and-Memory-Management.md). Same headings, same numbers. This chapter has no code half; the mini-sglang side is in section 6 below ("Where this lives in mini-sglang") and in the [chapter 4 code companion](./Chapter4_KV-Cache-Optimization_code.md). Cast and picture legend: [../README.md](../README.md).

Runnable toy with every number below: `python3 -I eli5/tools/ch7_paging_toy.py` (standard library only).

## The one-minute version

**Kid version.** The pantry (GPU memory for notes) is a wall of small, **same-size lockers**. Instead of reserving a huge cupboard for every order, the clerk gives each order **one locker at a time, whichever is free**, and writes the locker numbers in the order's **list** (the block table). The chef reads the notes by walking the list, locker by locker. Nothing needs to sit side by side.

```text
   RESERVE-A-CUPBOARD (contiguous)                    LOCKERS + A LIST (paged)
   ┌────────────────────────────┐                     order's list:  [ 12 ][ 7 ][ 42 ][ 3 ]
   │ order 1: 2,048 slots held, │                                        │    │     │     │
   │ only 200 ever used         │                     lockers: ... [3] ... [7] ... [12] ... [42] ...
   └────────────────────────────┘                     (any free locker, anywhere, one at a time)
   huge waste + holes nobody can fill                 waste is at most one partly-used locker
```

The one rule that makes it work: **the chef still reads the notes in order**; she just looks up each locker's address in the list first. The maths of attention does not change at all.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Memory problems of contiguous allocation; the central tension | Reserved cupboards waste space and leave holes; we want lockers (anywhere) but attention wants the notes in a row |
| Paging abstractions and PagedAttention | Locker, shelf-spot inside a locker, list of lockers; and how the chef reads locker by locker and still gets the exact answer |

## 2 Why Paging Is Needed: The GPU-Memory Dilemma of Contiguous KV Cache

### 2.1 How Contiguous KV Cache Allocation Works and Where It Fails

**Kid version.** Every order gets one **big cupboard** sized for the longest answer it *might* ever need, because nobody knows the real length in advance and the notes must sit in one unbroken stretch (efficient to read).

**Tiny numbers (my toy, Part 1).** For Qwen3-0.6B one word's notes cost `28 floors x 8 KV heads x 128 numbers x 2 (K and V) x 2 bytes = 114,688 B = 112 KiB`. A cupboard for 2,048 words is `2048 x 112 KiB = 224 MiB`. One request is a small thing; a server holds hundreds.

| | Cupboard of 2,048 words |
|---|---|
| Reserved per order | 224 MiB |
| Really used by an answer of 200 words | 21.875 MiB |
| Used / idle | **9.8% / 90.2%** |

(The book says "roughly 90% idle"; the number checks out.)

### 2.2 GPU-Memory Fragmentation and Waste

**Kid version.** Three different kinds of waste, each with its own picture.

**(a) Internal fragmentation: empty space *inside* a reserved cupboard.**

```text
   cupboard for 2,048:   [██████ 200 used ██████][░░░░░░░░░░░░ 1,848 reserved, empty, nobody else may use ░░░░░░░░░░░░]
```

**(b) External fragmentation: holes *between* cupboards that are each too small.**

```text
   memory:  [cup A][ hole 512 ][cup B][ hole 256 ][cup C][ hole 1024 ][cup D][ hole 768 ][cup E][ hole 1024 ]
   a new order needs 1,536 contiguous. Total free? plenty. Biggest single hole? 1,024. → "out of memory".
```

**Tiny numbers (my toy, Part 2).** The book's five holes add up to `512 + 256 + 1024 + 768 + 1024 = 3,584 MiB = 3.50 GiB` (the book writes 3.60 GiB: a small slip, logged as D6 in the notes; the argument is unchanged). A 1.5 GiB order cannot fit in the biggest 1 GiB hole. With 16-word lockers of 1.75 MiB each, the same order needs `ceil(1536 / 1.75) = 878` lockers, and the holes contain 2,048 lockers' worth: it fits.

**A simulation that shows the effect (my toy, Part 5).** 3,000 random arrivals, a pool of 2,000 word-slots, sizes 20 to 200, lifetimes 5 to 60 steps. The contiguous allocator is given the **exact** final size (the best possible case, no reservation waste) so that only the holes can hurt it:

| Allocator | Served | Refused although enough TOTAL memory was free | Refused because truly full |
|---|---|---|---|
| Contiguous (best case) | 1,800 | **1,174** | 26 |
| Paged (4-slot lockers) | 1,865 | **0** | 1,135 |

The paged allocator is never refused while memory exists; contiguous is refused over a thousand times with enough free memory in total. (This workload is deliberately overloaded, so many refusals are real for both; the counts of *false* refusals are the point. Refusals change what is free later, so the "served" totals are only roughly comparable.)

**(c) Hard to grow, and hard to share.** A sequence grows one word per step; either reserve it all up front (waste, section a) or reserve little and **copy everything** into a bigger cupboard when it overflows. And two answers that start the same (parallel sampling, beam search, a shared system prompt) would each need their own copy of the shared beginning.

### 2.3 The Central Tension That Paging Resolves

**Kid version.** Two things we want that look like they clash:

| The clerk wants | The chef wants |
|---|---|
| Hand out space **in small pieces, wherever it is free** (so scattered) | The notes **in order, one after another** (so she can read them without thinking) |

Old way: force the pieces to sit together (cupboards). New way: let them be scattered and keep an **address list** so the chef sees an in-order view anyway. Operating systems did this decades ago for program memory ("virtual memory"). The three consequences:

| Problem | After paging |
|---|---|
| Wasted space inside | At most the unfilled part of **one** locker |
| Holes between | None: every locker is the same size, any free locker will do |
| Growing | Add one more locker to the list; nothing is copied |

**Where the story lies.** The operating-system picture has *hardware* that translates addresses. On a GPU, the **kernel itself** does the lookup in software (it reads the list and computes addresses). Also OS paging can swap pages to disk; a GPU KV cache normally does not (chapter "Hierarchical Caching" in Part III is about that).

## 3 Paging Abstractions: Blocks, the Block Table, and Logical-to-Physical Mapping

### 3.1 Core Concepts and Their Hierarchy

**Kid version.**

| Real word | Kid word | Meaning |
|---|---|---|
| **Block** (also "page") | A locker | Fixed-size unit of GPU memory; holds the notes of `block_size` words (the book's default is 16; 32 and 64 are used too) |
| **Slot** | A shelf spot inside a locker | One word's notes inside a block |
| **Block table** | The order's list of locker numbers | Per order; entry `j` = which physical locker holds the order's `j`-th chunk of words |

```text
   word number  ──▶  (logical block j, offset t)  ──▶  block table[j] = physical block P  ──▶  slot  P x block_size + t
   position 37, block size 16:   37 // 16 = 2 ,  37 % 16 = 5
                                 table = [12, 7, 42, 3]  →  table[2] = 42
                                 address = 42 x 16 + 5 = 677
```

**Tiny numbers (my toy, Part 3).** A 50-word order, blocks of 16, table `[12, 7, 42, 3]`:

| Word | Logical block | Offset | Physical block | Slot address |
|---|---|---|---|---|
| 0 | 0 | 0 | 12 | 192 |
| 15 | 0 | 15 | 12 | 207 |
| 16 | 1 | 0 | 7 | 112 |
| 31 | 1 | 15 | 7 | 127 |
| 32 | 2 | 0 | 42 | 672 |
| 37 | 2 | 5 | 42 | 677 |
| 47 | 2 | 15 | 42 | 687 |
| 48 | 3 | 0 | 3 | 48 |
| 49 | 3 | 1 | 3 | 49 |

Four lockers used, and the last holds only 2 words (so 14 empty spots: that is the *entire* internal waste of this order, and it is bounded by one locker).

**The CPU copy and the GPU copy of the list.** The scheduler decides allocations, so it keeps a CPU-side copy; the attention kernel needs it while running, so a GPU-side copy exists too.

**Pause and predict**

1. A 2,048-word order with 16-word blocks: how many lockers? How many are completely full?

<details><summary>Answer</summary>

`2048 / 16 = 128` lockers, all full (2,048 is an exact multiple of 16). If it had 2,049 words, there would be 129 lockers and the last would hold one word.

</details>

2. In the table above, if table entry 2 is wrongly changed from 42 to 7, which words read the wrong notes?

<details><summary>Answer</summary>

Words 32 to 47 (logical block 2). They would read locker 7's notes (the notes of words 16 to 31) and attention would silently use wrong K and V: no crash, wrong text. (The book's exercise 2 asks this.)

</details>

### 3.2 How Logical Blocks Map to Physical Blocks

**Kid version.** Four steps.

| Step | What happens | Example (50 words, block 16) |
|---|---|---|
| 1 | Cut the order's words into blocks of 16 | blocks 0-3: words 0-15, 16-31, 32-47, 48-49 |
| 2 | Give each a free locker, record it in the list | list `[12, 7, 42, 3]`: scattered, in any order |
| 3 | To find word *w*: `w // 16` picks the list entry, `w % 16` the spot | word 37 → entry 2 → locker 42, spot 5 |
| 4 | To read the whole order: walk the list in order, read each locker's valid spots | locker 12, then 7, then 42, then 3 (2 spots) |

**Where the story lies.** "Interchangeable lockers" is true for *space*. For *speed*, scattered lockers cost extra address lookups and less tidy memory reads, which is what the kernel optimisations in section 4 are about.

## 4 PagedAttention: Performing Attention over Non-Contiguous KV

### 4.1 How PagedAttention Reads KV

**Kid version.** For one question (query), the chef walks the list: for each locker she **reads its Labels (K)**, scores the question against them, and notes down a **running tally**; then she moves to the next locker. At the end she has the same answer as if all the notes had been laid out in one row.

```text
   for each logical block j = 0 .. ceil(L / b) - 1:
       P = block_table[j]                      # look up the locker
       read K, V of the valid spots in P       # up to b spots (fewer in the last block)
       scores = q . K / sqrt(d)                # one score per spot
       mask: spots after this word are -infinity (a word can't see its future)
       update the running tally (m, l, O)      # next section
   answer = O / l
```

The kernel keeps these memory tricks: adjacent threads read adjacent addresses (**coalesced access**), the locker is first copied to fast on-chip **shared memory** (so scattered global reads are paid once), and the block size is tuned (**small blocks**: more list lookups; **large blocks**: more unfilled last-locker waste).

### 4.2 Proof of Mathematical Equivalence and Its Premises

**The problem in kid words.** Softmax turns scores into shares that add to 1: each score gets `exp(score)`, divided by the sum of all `exp`s. To compute it **safely** you first subtract the largest score (otherwise `exp` of a big number overflows). But the chef reads locker by locker and **does not know the largest score until the last locker**. And she does not want to re-read old lockers.

**The trick: a running tally measured against the best score so far.** Keep three numbers:

| Symbol | Kid word | Meaning |
|---|---|---|
| `m` | the best score so far | running maximum |
| `l` | total weight so far | `sum of exp(score - m)` over words seen |
| `O` | weighted notes so far | `sum of exp(score - m) x value` over words seen |

When a new locker brings a **bigger best score**, everything in the tally was measured against the old, smaller best. To compare fairly, turn the old tally **down** by one common factor `exp(m_old - m_new)` (which is at most 1), then add the new locker's contribution measured against the new best. Both `l` and `O` must be turned down by the **same** factor.

```text
   m_new = max(m_old, m_local)
   l_new = l_old x exp(m_old - m_new)  +  l_local x exp(m_local - m_new)
   O_new = O_old x exp(m_old - m_new)  +  O_local x exp(m_local - m_new)
   at the very end:  answer = O / l
```

**Tiny numbers: a full run (my toy, Part 4).** Six words with scores `[2.0, 1.0, 3.0, 0.5, 4.0, 1.5]` and scalar values `[1, 2, 3, 4, 5, 6]`, read in three lockers of two:

| Locker | Local best `m` | Local `l` | Local `O` | Shrink old by | Shrink local by | Running `m` | Running `l` | Running `O` |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.000 | 1.368 | 1.736 | (nothing yet) | 1.000 | 2.000 | 1.368 | 1.736 |
| 1 | 3.000 | 1.082 | 3.328 | **0.368** | 1.000 | 3.000 | 1.585 | 3.967 |
| 2 | 4.000 | 1.082 | 5.493 | **0.368** | 1.000 | 4.000 | 1.665 | 6.952 |

Final answer `O / l = 6.952 / 1.665 = 4.174571`. The direct formula (look at all six scores at once) gives **4.174571** as well; the difference is `9e-16`, which is floating-point rounding, not approximation.

(`0.368 = exp(-1)`: in locker 1 the best score rose from 2 to 3, so the old tally shrinks by `exp(2 - 3)`.)

**The wrong way (same toy).** Add each locker's local `l` and `O` **without** shrinking: you get `2.988806`, off by **1.186**. The tallies were measured against different "best scores"; adding them is like adding a length in metres to one in feet.

**Why bother subtracting the best?** With scores near 1000, `exp(1000)` overflows (my toy: `OverflowError`). The online recurrence only ever exponentiates differences, so it stays safe; at scores `[1000, 1001, 999, 1002]` it gives **3.232743**, identical to the max-subtracted direct formula.

**Why the final answer is exact.** At the end, every term in `O` and `l` carries the same factor `exp(-m_final)`. It appears in numerator and denominator and cancels, leaving exactly `sum(exp(score) x value) / sum(exp(score))`, the plain softmax-weighted sum. Blocking only changed the **order** of the additions, not the maths. (The idea is from the "online normalizer" softmax paper in the references; FlashAttention uses the same recurrence.)

**What PagedAttention does and does not change.**

| Standard attention | PagedAttention |
|---|---|
| Notes in one contiguous matrix | Look up the list, gather locker by locker |
| One softmax | Running tally across lockers |
| One big matrix multiply | A series of small ones accumulated |

It changes **where the numbers are fetched from**, not **what is computed**.

**Pause and predict**

3. After a locker, the running best `m` did not change (the new locker's best was smaller). What are the two "shrink" factors?

<details><summary>Answer</summary>

Old tally: `exp(m_old - m_new) = exp(0) = 1` (unchanged). Local tally: `exp(m_local - m_new) < 1` (the new locker is shrunk). Only whichever side has the smaller best is shrunk.

</details>

4. Why can't the chef just add each locker's local `l` and `O`?

<details><summary>Answer</summary>

Each local pair was computed against its own best score; `exp` is not linear, so the pairs live on different scales. They must first be moved to a common reference (the new running best), with the same factor applied to `l` and `O`.

</details>

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** Reserved cupboards waste most of memory (in the book's example, ninety percent idle) and leave unusable holes. Same-size lockers plus a per-order list remove the holes and cap the waste at one locker, and growth is just "add a locker". The chef reads locker by locker and keeps a running tally that is shrunk whenever a new best score appears, so the final answer equals ordinary attention. Paging changes how notes are *addressed*, never what attention *computes*.

### 5.2 Exercises

**1. Draw the data flow: token sequence → logical blocks → block table → slots in physical blocks.**

<details><summary>Answer</summary>

Use the table in section 3.1: tokens 0..49 → logical blocks 0..3 (cut every 16) → block table `[12, 7, 42, 3]` → in physical block *P* the token with offset *t* sits in slot `P x 16 + t`. Word 37 → (2, 5) → locker 42 → address 677.

</details>

**2. Roles of Block, Slot, Block Table; the hierarchy; what if one block-table entry is wrong?**

<details><summary>Answer</summary>

Block = unit of allocation; Slot = one token's spot inside a block; Block table = per-order list from logical to physical block. Hierarchy: token → (logical block, offset) → table → (physical block, slot). A wrong entry makes every token in that logical block read another block's K and V: silent wrong output.

</details>

**3. Why is blocked attention equal to standard attention? Use the online-softmax update.**

<details><summary>Answer</summary>

Because `l` and `O` are always shrunk by the same factor `exp(m_old - m_new)` whenever the running best changes, and the new locker is brought to the same reference before adding. At the end every term carries one common factor `exp(-m_final)`, which cancels in `O / l`, leaving exactly the standard softmax-weighted sum. Table in section 4.2: 4.174571 both ways.

</details>

## 6 Where this lives in mini-sglang

The book's vocabulary (Block, Slot, Block Table) is the general idea. mini-sglang's words differ a little:

| Book | mini-sglang |
|---|---|
| Block (physical) | **page** of `page_size` token slots. `page_size` defaults to **1**, i.e. every token is its own page (no scattered-locker overhead, no last-block waste, but one table entry per token) |
| Slot | token slot number (an integer index into the flattened pool: `page * page_size + offset`) |
| Block table | `page_table[row, position] = slot number`, **per token** (row = the order's peg; see chapter 4). It is "paged" in the sense that slots need not be consecutive |
| Free block pool | `CacheManager.free_slots`: a tensor of page-start slot numbers (for `page_size = 4`: `[0, 4, 8, ...]`) |
| Block table the kernel reads | Built per backend: the FlashAttention-style backend takes every `page_size`-th entry of the row and divides by `page_size` to get **page numbers**; the TRT-LLM-style backend does the same slicing; the FlashInfer-style backend does **not** use pages: it re-views the pool as one-token pages and hands the kernel the **per-token** slot numbers directly |

| Backend | How it treats pages |
|---|---|
| `fi` (FlashInfer) | Tells the library "page = 1 token" always (the metadata's `page_size` is the constant 1), so it works with a pool of **any** `page_size` |
| `fa` (FlashAttention, `sgl_kernel`) | Real pages: the pool's `page_size`, any value |
| `trtllm` | Real pages: 16, 32 or 64 only (the engine overrides other values to 64) |

(I first wrote that `fi` "supports only `page_size = 1`", from its assert. That was wrong: the assert checks a constant. `forward()` calls `_flatten_cache`, which views the pool as `[-1, 1, heads, head_dim]`, and the indices are per-token slots. The offline benchmark `bench.py` even uses `page_size=256`.)

So mini-sglang with default settings is the "block size 1" end of the book's tuning spectrum: no internal waste at all, at the cost of a bigger table and, for kernels, the least tidy reads. Larger pages are an option, and the free-list and prefix-cache bookkeeping then works in whole pages.

## 7 Extra drills (not in the book)

4. With `page_size = 1`, how many table entries does a 2,048-word order need? With 16? (2,048; 128.)
5. Qwen3-0.6B, 112 KiB per word, block of 16: how big is one physical block, and how many fit in 1 GiB? (`16 x 112 KiB = 1.75 MiB`; `1024 MiB / 1.75 = 585` blocks.)
6. In the simulation, why are the paged allocator's refusals all "truly full"? (Any free block can serve any request; if there are enough free blocks in total it always succeeds.)
7. A kernel reads a block with only 2 valid spots out of 16. What happens to the other 14 in the softmax? (They are masked out: their scores are treated as `-infinity`, so their weights are 0.)
