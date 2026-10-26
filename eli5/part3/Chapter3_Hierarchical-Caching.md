# Part III Chapter 3 Hierarchical Caching: ELI5 companion

Original: [Chapter3_Hierarchical-Caching.md](../../course-material/eng/part3/Chapter3_Hierarchical-Caching.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

This chapter is about the **full SGLang engine's** HiCache. mini-sglang has no host-memory tier or storage tier (its prefix cache lives only on the GPU: [chapter 8](../part2/Chapter8_RadixAttention-and-Prefix-Caching.md)). I have not read HiCache's source for these notes, so facts about it are the chapter's. What I can check is the arithmetic, which the toy does.

Runnable toy: `python3 -I eli5/tools/p3ch3_hicache_toy.py` (standard library only). **Every bandwidth and speed in it is an assumption I chose to illustrate the trade-off, not a measurement.** The constants are at the top of the file; change them and rerun.

## The one-minute version

**Kid version.** The kitchen has three places to keep finished notes:

| Place | Kitchen | Real |
|---|---|---|
| **L1** | The pantry beside the chef | GPU memory: fast, small |
| **L2** | A storeroom in the same building | This machine's CPU (host) memory: bigger, a walk away |
| **L3** | A warehouse across town (optionally shared with other kitchens) | External storage: biggest, slowest |

Notes that would otherwise be thrown out of the pantry are **kept in the storeroom or warehouse**; when a customer returns, you **fetch** them instead of **cooking them again**. That is only worth it if fetching is faster than cooking.

```text
   customer comes back ──▶ find the longest matching beginning ──▶ fetch from L1 / L2 / L3 ──▶ cook only the rest
```

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Why bother when prefix caching exists | The pantry is small; popular notes get thrown out (2) |
| What L1, L2, L3 hold; what a hit means | 3.1 |
| Trace a prefix from compute and backup to restore | 4 |
| Compare restore and recompute; prefetch and overlap | 5 |
| Which workloads benefit | 6 |

## 2 Why Prefix Caching Alone Is Not Enough

**Kid version.** A user uploads a long document and asks a question; later they ask a second question about the same document. Prefix caching lets the second question skip re-reading the document, *if* the document's notes are still in the pantry. But between the two questions the kitchen served hundreds of other orders and needed shelf space, so the unused notes were thrown out. Next time the chef must read the whole document again.

| Tool | What it solves |
|---|---|
| Paged KV cache | Using the pantry's space without waste |
| Prefix caching | Not cooking the same beginning twice **while its notes are still in the pantry** |
| HiCache | Keeping notes **somewhere else** when the pantry has no room, and bringing them back |

**Where the story lies.** More cache tiers don't make the *pantry* bigger; they make the *fetch* option available. If fetching is slower than cooking, they don't help (section 5).

## 3 What HiCache's Three Tiers Do

### 3.1 L1, L2, and L3

| Tier | Where | Main purpose | A hit means |
|---|---|---|---|
| L1 | GPU memory | Notes for current work and recent reuse | Already there; use directly |
| L2 | This instance's host (CPU) memory | More history that has left the GPU | Copy the notes **back to the GPU first** |
| L3 | Optional external storage; may be **shared across instances** | Even more history | Usually read into host memory first, then copied to the GPU |

```text
   restore path:   L3 (storage) ──▶ L2 (host memory) ──▶ L1 (GPU)
                     slow link          PCIe-class link
```

Capacity grows as you go down; speed shrinks. Both the copy and the storage access add latency. L3 is "not always shared storage": it is optional, and sharing depends on configuration.

### 3.2 How Prefix Matching Works Across Tiers

**Kid version.** Still the same rule as before: look for an exact beginning, from the first word. The twist: the beginning's pieces may be in different places. Two questions: **how far does the match go, and where do I get each piece?**

Three steps:

1. **Local tree first (L1 and L2).** Walk the local radix tree word by word; each branch remembers whether its notes are on the GPU, in host memory, or both. This only checks the index; nothing is copied yet.
2. **Ask L3 if enabled.** For the part not matched locally, compute a **lookup key** per cache page and ask the storage. The key is a **prefix fingerprint**: it combines this page's words **with the whole preceding prefix**, so identical text in a different context doesn't collide. There is no global map of what's in L3; pages are looked up as needed.
3. **Use only the continuous prefix that is actually ready.** You can't skip a missing page and resume later (the notes after the gap were written having read the missing part).

**Checked by running (toy Part 2).** A 1,024-word prefix in pages of 256:

| Pages from the start | Reusable words |
|---|---|
| L1, L1, L2, L3 | 1,024 |
| L1, L1, L2, **missing** | 768 |
| L1, L1, **missing**, L3 | **512** (the L3 page after the gap is useless) |
| **missing**, L1, L1, L1 | **0** |

**Prefix fingerprints, illustrated (toy Part 6; my own scheme with a standard hash, not SGLang's exact format).** Chain a hash through the pages: each page's key = hash(previous key + this page's words):

| Situation | Keys | Equal? |
|---|---|---|
| Same document, **different** system prompt | `34217f53 ...` vs `897aaaf5 ...` | **No**: different context, no reuse |
| Same system prompt + document computed on another instance | identical | **Yes**: shared storage can serve it |

Exact words are needed: a document with similar meaning doesn't match. And a different system prompt or template in front changes the whole prefix.

Hybrid model designs have extra states (sliding-window K/V, recurrent states) that must also be complete for a hit; the chapter points to the unified radix cache for that. This companion uses plain full-attention K/V.

**Pause and predict**

1. Page 1 of the document is in L3 only; pages 2 and 3 are in L1. How much can be reused?

<details><summary>Answer</summary>

All three pages, provided page 1 is first read from L3 into host memory and restored to the GPU. The pieces may be in different tiers; what matters is that the prefix is continuous and every piece becomes ready.

</details>

## 4 The Lifecycle of a Cached Document

### 4.1 The First Request: Compute and Keep a Copy

**Kid version.** First request: "document + question A". Nothing is cached, so the chef reads everything (prefill), the notes land in the pantry (L1), and the answer is written. Depending on the policy, the notes of prefixes worth keeping are **copied** ("backed up") to the storeroom (L2) and, if configured, to the warehouse (L3). A backup is an extra copy; the pantry copy stays until the pantry needs the space.

Backing up everything immediately gives the best chance of a later hit but costs link bandwidth and storeroom space. If a prefix is thrown out of the pantry before its backup finishes, or was never chosen for backup, it must be recooked.

### 4.2 The Second Request: Restore According to the Hit Location

**Kid version.** Second request: "same document + question B".

| Case | What happens |
|---|---|
| **L1 hit** | Notes still in the pantry; use directly |
| **L2 hit** | Pantry copy gone but the storeroom has them: reserve pantry slots, copy back, then cook only question B |
| **L3 hit** | Read warehouse → storeroom, then storeroom → pantry |
| **Nothing usable** | Recook the missing part and make new notes |

```text
   document: [ first part in L1 ][ middle in L2 ][ end in L3 ][ question B: new ]
              use directly        copy up          read, copy up   cook this
```

**Caching a document does not mean the question stops using the document.** The saving is the *forward computation over the document*. The new words still attend to the document's K/V (that is the whole point).

## 5 Performance Tradeoffs

**Kid version.** Enabling a tier does **not** automatically speed things up. It saves cooking time and adds fetching time. Which is bigger?

### 5.1 Compare Restoration and Recomputation Time

```text
   restore time  ≈  (storage → host)  +  (host → GPU)        (an L2 hit has no first term)
   worth it only if   restore time  <  time to recompute the prefix
```

**Tiny numbers (toy Part 1; assumed 25 GB/s host→GPU, 2 GB/s storage→host, 100 TFLOP/s effective).** A 1,024-word prefix:

| Model | K/V per word | Prefix K/V | Recompute | L2 restore | L3 read + restore | Break-even host link |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-like | 112 KiB | 112 MiB | 12.3 ms | **4.7 ms** (restore wins) | 63.4 ms (**recompute wins**) | 9.6 GB/s |
| 32B-class | 256 KiB | 256 MiB | 655.4 ms | **10.7 ms** (restore wins) | **145.0 ms** (restore wins) | 0.4 GB/s |

Reading it:

- **Recompute cost grows with the model's parameter count** (arithmetic per word), while **K/V size grows only with layers and KV heads**. So the bigger the model, the better restoring looks.
- For the small model, slow storage loses: 63 ms to fetch vs 12 ms to recompute.
- The break-even speed tells you how slow a link can be before recomputing is better.

(These are teaching numbers. The chapter says the same: "These numbers are not measurements; actual time depends on cache size, effective bandwidth, storage load, and model computation speed.")

**The book's own example, run (toy Part 5, exercise 3).** Recompute 150 ms; L2 restore 30 ms; L3: storage read 200 ms + restore 30 ms.

| Hit | Time | vs recompute 150 ms |
|---|---|---|
| L2 | 30 ms | **helps** |
| L3 | `200 + 30 = 230` ms | **hurts** |

**Backup also costs.** If most prefixes are never reused, you pay for copies that save nothing. The question for any workload: *does reuse actually happen, and does the saved cooking exceed the backup and restore overhead?*

### 5.2 Prefetching: Read While the Request Is Waiting

**Kid version.** Request B's document is only in the warehouse, and B is waiting behind A. Don't wait to be served: **start the warehouse run while A is still cooking**. If the delivery arrives in time, B loses no time.

**Tiny numbers (toy Part 4).** A computes for 120 ms; B's read takes 100 ms: **0 ms** exposed (fully hidden). If B's read takes 180 ms: **60 ms** still exposed.

When B's turn comes and the read isn't finished, three policies:

| Policy | What happens | Trade-off |
|---|---|---|
| `best_effort` | Don't wait; use what's ready and recompute the rest | Starts sooner, may waste reuse |
| `wait_complete` | Wait until the read finishes | Reuses more, may wait long |
| `timeout` | Wait a limited time, then use what's ready | A hedge |

**Tiny numbers (toy Part 4).** The whole prefix would cost 300 ms to recompute; the read still needs *r* ms; the timeout is 30 ms. Delay before B can proceed:

| *r* (remaining read) | `wait_complete` | `best_effort` | `timeout` (30 ms) |
|---|---|---|---|
| 10 | 10 | 300 | 10 |
| 30 | 30 | 300 | 30 |
| 60 | 60 | 300 | **330** |
| 400 | 400 | **300** | 330 |

- If *r* is small, **waiting wins**. If *r* exceeds the recompute cost (400 vs 300), **giving up wins**.
- `timeout` is bad exactly when the read needs *slightly more* than the timeout: it pays the wait **and** the recompute (330 ms, worse than both of the others at *r* = 60).

No policy is always best. What users feel is the **total** request time, not the hit rate.

### 5.3 Restore Layer by Layer: Overlap Transfers with Computation

**Kid version.** The model runs floor by floor. To cook floor 0 you need floor 0's old notes, not all floors' notes. So: copy floor 0's notes, then **start cooking floor 0 while floor 1's notes are travelling**, and so on. Each floor must still wait for its own notes.

**Tiny numbers (toy Part 3, 4 layers, hypothetical times).**

| Per layer: transfer + compute | Copy-all-then-compute | Overlapped | Saved |
|---|---|---|---|
| 3 + 5 | 32 | 23 | 9 |
| 6 + 2 | 32 | 26 | 6 |
| 4 + 4 | 32 | 20 | 12 |

The first layer's transfer can't be hidden, and whichever of transfer or compute is slower sets the pace. Overlap can't break dependencies: if the next layer's notes haven't arrived, compute waits.

**Pause and predict**

2. Transfer 6 and compute 2 per layer, 4 layers: why is the overlapped total 26 and not 6 + 8 = 14?

<details><summary>Answer</summary>

Transfers are the bottleneck: they run back-to-back (6, 12, 18, 24), and each layer's compute can only start after its transfer, so the last layer finishes at `24 + 2 = 26`. Compute for earlier layers is hidden inside the waiting; transfer is what is exposed.

</details>

## 6 How to Decide Whether a Workload Fits HiCache

### 6.1 Look for Repeated Prefixes, Then Capacity Pressure

**Kid version.** Likely winners: multi-turn chats that resend history, repeated questions about one long document, agents that reuse the same task prefix. The shared trait: **a long prefix that has already been computed will be used again.** Then check whether that prefix actually gets **thrown out of the pantry** between uses: if hot prefixes stay in L1, lower tiers add little. Little sharing, short inputs, or an already-saturated link: more cache won't help.

HiCache reduces repeated **prefill** only. It does not speed up decode and does not raise the model's maximum context length.

### 6.2 Verify Benefits with Service Metrics

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-0.6B \
  --enable-hierarchical-cache
```

This enables the GPU + host tiers; it does not configure shared L3 storage. How to test: repeat requests with the same long prefix and put other traffic between them, so the prefix is evicted from the GPU; check that it comes back from a lower tier; compare with HiCache off under identical settings. Watch TTFT, input throughput and **tail** latency, plus tokens reused per tier, host memory use, and transfer load.

## 7 Summary and Exercises

### 7.1 Summary

**Kid version.** Keep notes in more places than the pantry, fetch them when a customer returns, and cook only what's missing. L1 hits are free; L2 and L3 hits must be copied up. Prefetching and layer-by-layer copying hide some of the waiting. Whether it helps depends on repeated long prefixes, pressure on the pantry, and fetching being cheaper than cooking.

### 7.2 Exercises

**1. Why HiCache when paged KV and prefix caching exist? What does each solve?**

<details><summary>Answer</summary>

Paged KV: efficient allocation of the pantry. Prefix caching: not recomputing a shared beginning while its notes remain. Neither can keep unlimited history in finite GPU memory; HiCache keeps evicted notes elsewhere and brings them back.

</details>

**2. What must happen before execution for L1, L2, L3 hits? Is L3 always shared?**

<details><summary>Answer</summary>

L1: nothing (already on the GPU). L2: allocate GPU slots and copy host → GPU. L3: read storage → host, then host → GPU. L3 is optional, and it is shared only if configured as shared storage.

</details>

**3. Recompute 150 ms, host restore 30 ms, storage read adds 200 ms, ignoring overlap: do both lower-tier hits help?**

<details><summary>Answer</summary>

L2: 30 ms < 150 ms: yes. L3: `200 + 30 = 230 ms` > 150 ms: no; recomputing is faster (toy Part 5).

</details>

**4. How does prefetching help while queued? What do the three policies do when the read is late?**

<details><summary>Answer</summary>

It moves the storage read into the queueing time so it overlaps with other requests' compute. If late: `best_effort` uses the ready part and recomputes the rest; `wait_complete` waits for the full read; `timeout` waits up to a limit, then behaves like `best_effort`. (Toy table: `timeout` is worst when the remaining read is just above the timeout.)

</details>

**5. Why can computation start once layer 0's K/V is ready? What dependencies must hold?**

<details><summary>Answer</summary>

Layer 0 attends only to layer 0's old K/V, not the other layers'. Each later layer still needs its own K/V to have arrived before it runs, and layer *i* needs layer *i−1*'s output; so computation can only be as far ahead as the transfers (toy Part 3).

</details>

**6. Hit rate went up but TTFT tail latency got worse. What do you investigate?**

<details><summary>Answer</summary>

Restore time vs recompute time for the hits that now happen (slow link or slow storage, as in the 0.6B L3 row); queueing of transfers behind each other; host memory pressure; prefetch policy (`wait_complete` can add long waits); backup traffic competing for the same link; whether hits are concentrated on prefixes that were cheap to recompute.

</details>

**Extra drills (not in the book).**

7. Qwen3-0.6B-like, 4,096-word prefix, same assumptions as the toy: L2 restore time? (`4096 x 112 KiB = 448 MiB`; `/ 25 GB/s ≈ 18.8 ms`. Recompute is 4x the 12.3 ms of the 1,024-word case, 49.2 ms; restore still wins.)
8. A prefix is in L1 for pages 1 and 3 but missing for page 2 (pages of 256). Reusable words? (256: only page 1, since page 2 is missing.)
9. Why does the fingerprint of page 3 depend on pages 1 and 2? (The notes of page 3 were written after reading pages 1 and 2; a chain hash makes the key encode exactly that context.)
