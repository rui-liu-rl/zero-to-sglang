# Part III Chapter 1 Attention Backends: ELI5 companion

Original: [Chapter1_Attention-Backends.md](../../course-material/eng/part3/Chapter1_Attention-Backends.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

This chapter is about the **full SGLang engine's** attention options. I have not read the full engine's source for these notes, so statements about FlashMLA, Triton and the name list below are the chapter's, not mine. What I **can** check is the maths (toy below) and mini-sglang's own three backends (`fi`, `fa`, `trtllm`), which this companion points at in section 7.

Runnable toy: `python3 -I eli5/tools/p3ch1_attention_toy.py` (standard library only).

## The one-minute version

**Kid version.** "Attention" is one **recipe** (the formula). A **kernel** is one **cooking technique** for carrying out that recipe on a GPU. A **backend** is the **sous-chef** who gets the ingredients to the chef's station in the right form (which request owns which words, where each word's notes are in the pantry) and picks a technique that suits the dish (long prompt or one new word) and the kitchen (which GPU).

```text
   model gives Q, K, V for the new words
        │
        ▼
   BACKEND (sous-chef):  ① prepare the paperwork (metadata)   ② store new K/V, point at the old ones   ③ pick a kernel for this phase
        │
        ▼
   KERNEL (technique) runs on the GPU  ──▶  attention output goes back to the model
```

Switching backends changes **how** the same attention is computed. It does not change the model.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Tell apart mechanism, kernel, backend | Recipe / technique / sous-chef (2.1) |
| Why prefill, decode, architecture, GPU need different backends | Different dishes and kitchens need different techniques (3) |
| FlashInfer, Triton, FA3, FlashMLA: ideas, strengths, limits | Section 4 |
| Pick implementations to compare, then check speed and correctness | Section 5 |

## 2 What an Attention Backend Does

### 2.1 From the Formula to a GPU Implementation

**Kid version.** The formula says *what* to compute:

```text
   O = softmax( Q Kᵀ / sqrt(d) + M ) V          (M = the mask: "you can't see the future")
```

Three words people mix up:

| Word | Question it answers | Kid version | Examples |
|---|---|---|---|
| **Attention mechanism** | What attention structure does the model use? | The recipe's *design* | MHA (every head its own K/V), GQA (heads share K/V), MLA (compressed cards) |
| **Attention kernel** | How does one GPU program carry it out? | A cooking technique | tiled matrix multiply, fused softmax, split-KV |
| **Attention backend** | How does the engine prepare inputs and call the right kernel for this phase? | The sous-chef | SGLang's FlashInfer, Triton, FA3 backends |

**Different technique, same dish, slightly different crumbs.** Different operation orders or number formats give tiny floating-point differences. So switching backends needs a numerical check, not blind trust. (Toy Part 3: splitting 4,096 keys into up to 16 segments changed the result by about `7e-18`, which is rounding.)

### 2.2 The Backend's Responsibilities

**Kid version.** Three jobs.

| Job | Kid version | Example (the book's batch: histories of 128, 1,024 and 4,096 words) |
|---|---|---|
| ① Prepare the **metadata** | A packing list: which new words belong to which order, how long each history is, where its notes are, what it may look at | Three rows, with lengths `128, 1024, 4096` and each row's slot list |
| ② Handle K/V reads and writes | Shelve the new notes at the places the pantry clerk chose; let the kernel read old notes even if they are scattered over many pages | New K/V go to `out_loc`; old K/V are found through the page list |
| ③ Call a kernel suited to the phase | Prefill: many new words per order; decode: usually one | Two code paths |

The full engine's interface has three entry points: `init_forward_metadata` (do job ① once per forward pass), `forward_extend` (prefill-like: many new words) and `forward_decode` (one new word per order). mini-sglang's version is in section 7.

## 3 Why Different Backends Are Needed

### 3.1 Prefill and Decode Have Different Workload Shapes

| Phase | Questions per order this step | The usual limit | So the kernel wants to... |
|---|---|---|---|
| Prefill | Many | **Compute** (lots of work per byte read) | Reuse the same K/V tile for many questions; avoid writing big intermediate tables |
| Decode | Usually one | **Memory bandwidth** (little work per byte read) | Read the history efficiently; keep the GPU busy despite so little work |

(These are the common cases; measure to confirm. Part I chapter 2 gives the same picture with numbers.)

```text
   prefill (3 new words)                        decode (1 new word)
   keys:   k1 k2 k3 k4 k5 k6                    keys:   k1 k2 k3 k4 k5 k6
   q4:     ■  ■  ■  ■  ·  ·                     q7:     ■  ■  ■  ■  ■  ■   ← one row, but reads ALL the history
   q5:     ■  ■  ■  ■  ■  ·
   q6:     ■  ■  ■  ■  ■  ■
```

**Split-KV.** In decode there is little parallel work (one question), so the GPU might be mostly idle. The trick: cut the long history into **segments**, let different parts of the GPU compute each segment's **partial answer** independently, and then **merge** them using each segment's softmax totals.

**Checked by running (toy Part 3, exact merge).** 4,096 keys, split into 1, 2, 8 and 16 segments:

| Segments | Merged output | Direct output | Difference |
|---|---|---|---|
| 1 | -0.035251032129 | -0.035251032129 | 0 |
| 2 | -0.035251032129 | -0.035251032129 | 6.9e-18 |
| 8 | -0.035251032129 | -0.035251032129 | 6.9e-18 |
| 16 | -0.035251032129 | -0.035251032129 | 6.9e-18 |

Each segment reports **three numbers**: its local best score, its total weight, and its weighted output. The merge shifts them to a common reference (the same trick as chapter 7's online softmax), so the answer is exact.

**Why it matters (toy Part 4).** A decode batch where order A has 128 words of history and order B has 8,192, with 8 KV heads each:

| Plan | Tasks | Longest task | Total work |
|---|---|---|---|
| One task per (order, head) | 16 | 8,192 words | 66,560 |
| Split-KV, 512-word segments | 136 | **512 words** | 66,560 |

Same total work, but the slowest piece shrinks 16 times; so if there are enough compute units the whole step finishes about 16 times sooner (in this idealised picture). The cost: B now produces 16 partial results per head that must be merged. For short histories or already-busy GPUs splitting may not help.

### 3.2 Models Use Different Attention Structures

**Kid version.** MHA: every head has its own Labels and Gifts. GQA: groups of heads **share** them (smaller notepad). MLA: keep a **compressed summary card** per word and expand it when needed (much smaller notepad; a different computation path).

**Consequence:** a technique built for MHA/GQA doesn't just work for MLA by changing a name, and an MLA-specific sous-chef can't turn an ordinary model into an MLA model. Sliding windows, special masks, verification trees for speculative decoding, and unusual K/V number formats are further features a backend may or may not support. **Check feature support before comparing speed.**

### 3.3 GPUs Have Different Architectures

**Kid version.** Different kitchens have different equipment. The chapter's example: Hopper GPUs have a **TMA** (a hardware engine that copies tiles by itself) and **WGMMA** (a matrix instruction that runs asynchronously). A technique built around them can overlap copying and cooking, but it only works, and only pays off, on kitchens that have them. CUDA and compiler versions matter too.

**Rule:** the right backend depends on the **model** (structure), the **workload** (shape) and the **hardware** (capabilities).

**Pause and predict**

1. A decode step has 4 orders, each with 100 words of history. Does split-KV help much?

<details><summary>Answer</summary>

Probably not. Histories are short, and 4 orders times several heads may already give the GPU enough tasks. Splitting adds partial results and merging with little parallelism gain. (The chapter: "More splits may not help when the context is short or the batch already supplies enough work.")

</details>

## 4 How the Four Main Backends Work

**Kid version (where each sits in the stack).**

```text
   model's attention call
        │
        ▼
   SGLang BACKEND (adapter: prepares metadata, picks calls)
        │
   ┌────┼────────────────┬──────────────────┬─────────────────┐
   ▼    ▼                ▼                  ▼                 ▼
 FlashInfer (library)  Triton (language+     FA3 (kernels for   FlashMLA (kernels for
 of inference kernels  compiler for GPU      Hopper hardware)   MLA models)
                       programs)
        │
        ▼
   GPU kernels run
```

### 4.1 The Core Idea: Tiling and Fusion

**Kid version.** The naive recipe builds a giant **score table** (every question against every key), writes it to the far warehouse (HBM), reads it back to turn it into weights, writes the weights, reads them again to mix the Gifts. The table is `n x n`, and it is enormous.

**Tiny numbers (toy Part 1).** At 32,768 words, for *one* layer and *one* head, with 2 bytes per number:

| Item | Size |
|---|---|
| Score table `S` | 1,073,741,824 entries = **2.00 GiB** |
| Write `S` + read it back | **4.0 GiB** |
| Also write and read the weights `P` | **8.0 GiB** (the figure's number) |

FlashAttention's two ideas:

1. **Work in tiles.** Bring a small tile of Q, K, V into the fast on-chip memory, do everything you can with it there, move on. The full table is never built.
2. **Normalise as you go.** Softmax needs the weights of a row to add to 1. Keep two running totals (how much weight so far, how much output so far); when a new tile brings a bigger best score, shrink the old totals (online softmax, chapter 7).

**The book's 4x4 example, checked (toy Part 2).** Values `V = [10, 20, 30, 40]`. Query 4 sees all four keys with unnormalised weights `[1, 1, 2, 2]`:

| Step | Total weight | Output so far |
|---|---|---|
| Naive: weights `[1/6, 1/6, 1/3, 1/3]` | 6 | `5 + 23.33 = ` **28.3333** |
| Tile A (keys 1, 2) | 2 | **15** |
| Tile B alone (keys 3, 4) | 4 | 35 |
| Merge: A's share `2 / 6 = 1/3` | 6 | `1/3 x 15 + 2/3 x 35 = ` **28.3333** |
| **Wrong**: average the two tile outputs `(15 + 35) / 2` | | 25.0 |

**An honest footnote (my estimate, toy Part 1).** The book's story is about avoiding the `n x n` table. It does not make the traffic vanish: for each tile of query rows the kernel re-reads all of K and V. With head dim 128, bf16 and no causal saving, one pass over K and V is 16 MiB:

| Query-tile height | K and V passes | Total traffic (one head, one layer, 32k words) |
|---|---|---|
| 64 rows | 512 | 8.02 GiB |
| 128 rows | 256 | **4.02 GiB** |
| 256 rows | 128 | 2.02 GiB |

So in this example the tiled kernel moves roughly half the bytes of the naive 8 GiB (with a 128-row tile) **and** needs no 2 GiB table in memory. The real wins are less storage, fewer steps, and fused work, not zero traffic. Also: the computation itself is still quadratic in sequence length for full prefill.

### 4.2 FlashInfer: Organizing Computation for Inference Serving

**Kid version.** Serving brings a mess tiling doesn't solve: orders of **very different lengths**, notes **scattered over pages**. FlashInfer's three steps:

| Step | Kid version |
|---|---|
| ① **Locate** | Give the kernel an address list (page indices) per order; it reads the pages directly, no need to copy an order's notes into one contiguous buffer |
| ② **Plan** | Decide how to divide the work into tasks of similar size: maybe split long histories into segments (split-KV) so every compute unit is busy; note which segment each task takes |
| ③ **Run** | Each layer runs the plan with its own Q and K/V, merging partial results |

**Plan once, run many times.** All layers have the same shapes, lengths and page lists; only Q, K, V values differ. So build the plan **once per forward pass** and `run` it for each layer, instead of re-planning 64 times.

```text
   forward pass:   plan ───────────────────────────────────────────────────▶ (once)
                   run(layer 0) run(layer 1) run(layer 2) ... run(layer 63)   (each with its own Q, K, V)
```

The costs: planning takes CPU time; split-KV adds partial-result storage and merging. The next decode step has different lengths (orders grow, join, finish) so the plan must be rebuilt every step.

### 4.3 Triton: Using a Compiler to Build Adaptable GPU Kernels

**Kid version.** Triton is a **language and compiler** for GPU programs: you describe tiles, loads and matrix operations in Python-like syntax, and the compiler writes the GPU code. An "attention backend `triton`" means attention kernels written in it. Strength: **easy to read and change**. Costs: speed depends on the compiler and tuning; the first run compiles (so warm up before timing). The book adds a careful note: "easier to write" doesn't mean "always slower"; measure.

### 4.4 FlashAttention 3: Using Hopper's Asynchronous Capabilities

**Kid version.** A **pipeline**. While the chef cooks tile *i*, the helper is already fetching tile *i+1* and the oven is already warm for the next softmax step. Hopper's TMA (copy engine) and asynchronous matrix instructions make that overlap possible. Condition: the GPU, software versions, head size, and features must match.

**Tiny numbers (illustrative; the book's figure also uses teaching assumptions).** 8 tiles, 3 units to fetch and 5 units to compute per tile:

| Schedule | Total |
|---|---|
| Fetch then compute, one after another | `8 x (3 + 5) = ` **64** |
| Overlapped (fetch the next tile while computing this one) | `3 + 8 x 5 = ` **43** (about 1.5x) |

The first fetch can't be hidden and the longer of the two stages sets the pace.

### 4.5 FlashMLA: Adapting to MLA Caches and Decode

**Kid version.** MLA is a *model design* (compressed notes). FlashMLA is a *set of kernels* that run that design well, on supported model, hardware, page size and number formats. Two lessons from the chapter: **cache compression comes from the model, not from FlashMLA**; and **one backend can mix kernels** (the SGLang FlashMLA backend uses FlashMLA mainly for decode, and inherits an ordinary prefill implementation), so a backend's name doesn't mean every phase uses the same kernel.

### 4.6 Comparing the Backends

| Backend | Main idea | Strength | Trade-off |
|---|---|---|---|
| FlashInfer | Organise variable-length requests and paged notes, then run suitable kernels | Broad serving support, several execution paths | Metadata and planning cost; dependency and feature requirements |
| Triton | Express tiled, fused, split-KV attention through a GPU compiler | Easy to read, change, extend | Speed depends on compilation and tuning; first-run compile |
| FA3 | Tiled attention with Hopper's asynchronous pipelines | High hardware use where supported | Hardware and shape constraints |
| FlashMLA | Organise reads, compute, scheduling around MLA's latent cache | Fits MLA | Limited to certain models/cache formats; prefill and decode may differ |

There is no ranking without a workload.

### 4.7 SGLang Attention Backend Name Reference

The book lists 24 `--attention-backend` names (22 plus two deprecated aliases) in the version it pins. I grouped them by what the names suggest; I did **not** verify the groups against the code, so treat the grouping as a reading aid only.

| Group | Names |
|---|---|
| The four of this chapter | `flashinfer`, `triton`, `fa3`, `flashmla` |
| Other FlashAttention generations | `fa4` |
| PyTorch built-ins | `torch_native`, `flex_attention` |
| TRT-LLM-style kernels | `trtllm_mha`, `trtllm_mla` |
| MLA variants | `cutedsl_mla`, `tokenspeed_mla` |
| Sparse or model-specific attention | `dsa`, `qsa`, `dsv4`, `minicpm_flashattn`, `minicpm_flashinfer` |
| Other vendors / hardware | `aiter`, `wave`, `ascend`, `intel_amx`, `intel_xpu` |
| Other | `hpc_ops` |
| Deprecated aliases | `nsa` (= `dsa`), `compressed` (= `dsv4`) |

These options don't all support the same models or hardware. Linear and multimodal attention are configured separately.

## 5 Choosing and Validating a Backend

### 5.1 Start with Compatible Implementations

**Kid version.** First rule out what can't work: MHA/GQA or MLA? Which GPU? Which K/V number format? Sliding windows, speculative decoding, other special features? Only then compare. With no choice given, the engine picks a sensible default from hardware and model: use that as your **baseline**. Two flags let prefill and decode use different backends (`--prefill-attention-backend`, `--decode-attention-backend`), provided their K/V layouts and features are compatible.

### 5.2 Compare the Actual Request Experience

**Checklist.** Same model, hardware, K/V format, lengths, concurrency, same warm-up. Then compare:

| Measure | Question |
|---|---|
| TTFT | Is long-input prefill faster? Does queuing affect it? |
| TPOT / ITL | Is token-by-token decode faster? |
| Throughput and memory | Does a busy server finish more work? Do workspaces eat memory? |
| Correctness | Do fixed inputs still give the same (within tolerance) answers? |

Long-input, short-output services care about prefill and TTFT; short-input, long-output services care about decode. The same backend can win one and lose the other.

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** The formula says what; the kernel says how on a GPU; the backend wires the engine to the kernel. Prefill (many questions, compute-bound) and decode (one question, bandwidth-bound), model structure, and GPU type explain why there are many backends. FlashInfer plans and reuses; Triton is readable; FA3 pipelines on Hopper; FlashMLA fits MLA. Check compatibility, then measure speed and correctness on real requests.

### 6.2 Exercises

**1. What belongs to mechanism, kernel and backend? Why does switching backends leave the architecture unchanged?**

<details><summary>Answer</summary>

Mechanism: the structure (MHA/GQA/MLA). Kernel: one GPU program computing it. Backend: prepares metadata, handles K/V reads and writes, and chooses a kernel. The backend only changes *how* the same maths executes; the weights and structure are untouched (small rounding differences aside).

</details>

**2. How does the query count per order differ between prefill and ordinary decode? Effect on kernel design?**

<details><summary>Answer</summary>

Prefill: many; decode: one. With many queries, one K/V tile is reused many times (compute-bound), so kernels focus on tiling and data reuse. With one, parallelism is scarce and K/V reading dominates, so kernels focus on bandwidth and on splitting long histories (split-KV).

</details>

**3. Why can FlashAttention avoid storing the full attention matrix? Does it make prefill linear in length?**

<details><summary>Answer</summary>

It computes one tile of scores at a time on chip, uses it immediately, and keeps only running softmax totals and a running output; nothing `n x n` is ever written. The amount of arithmetic is unchanged, so full prefill is still quadratic in sequence length.

</details>

**4. What does each of the four emphasise? Why can't FlashMLA run any MHA model?**

<details><summary>Answer</summary>

FlashInfer: serving (ragged lengths, paged notes, planning). Triton: readable, adaptable kernels. FA3: Hopper's asynchronous pipeline. FlashMLA: MLA's compressed cache and decode. An ordinary MHA model has no latent cache and a different layout, so an MLA kernel has nothing it can read.

</details>

**5. A backend has better TTFT on long inputs but worse throughput with high concurrency and long outputs. What do you check?**

<details><summary>Answer</summary>

The real request mix (is it prefill-heavy or decode-heavy?), decode speed (TPOT/ITL), kernel workspace memory (does it shrink the KV pool, and so concurrency?), correct warm-up, and whether a prefill/decode hybrid suits both.

</details>

**Extra drills (not in the book).**

6. At 16,384 words, how big is the score table for one head and one layer in bf16? (`16384^2 x 2 B = 512 MiB`: a quarter of the 32k size, because it scales with the square.)
7. Three segments report (local best, weight total, output) = (2.0, 1.0, 4.0), (3.0, 2.0, 5.0), (1.0, 1.0, 1.0). What is the merged output? (Shift to best 3.0: weights `1.0 e^-1 = 0.368`, `2.0`, `1.0 e^-2 = 0.135` → total `2.503`; outputs `4.0 e^-1 = 1.472`, `5.0`, `1.0 e^-2 = 0.135` → `6.607`; result `6.607 / 2.503 = 2.64`.)
8. Why does the plan have to be rebuilt every decode step? (Lengths grow by one, orders join or finish, so the work split and page lists change.)

## 7 Where this lives in mini-sglang

mini-sglang has the same three-part idea in a much smaller form (`python/minisgl/attention/`).

| Chapter concept | mini-sglang |
|---|---|
| Backend interface (`init_forward_metadata`, `forward_extend`, `forward_decode`) | `BaseAttnBackend`: `prepare_metadata(batch)` (once per batch, on the CPU, before any layer) and `forward(q, k, v, layer_id, batch)` (stores this pass's K/V, then attends). The prefill/decode split is by `batch.is_prefill` |
| FlashInfer: plan once, run per layer | `attention/fi.py` (`fi`): `wrapper.plan(...)` happens lazily on the first layer's call (`_initialize_metadata_once`) and is reused by the other layers; `wrapper.run(q, kv)` per layer. It always tells FlashInfer "page = 1 token" (it flattens the pool and passes per-token slot numbers), so FlashInfer's own multi-token paging is never used, but the pool itself can have any `page_size`. Tensor cores are switched on for decode when 4 or more query heads share a KV head |
| FA3 | `attention/fa.py` (`fa`): `flash_attn_with_kvcache` from `sgl_kernel` (FA3; FA4 on Blackwell); real pages (`page_size > 1`) |
| TRT-LLM-style kernels | `attention/trtllm.py` (`trtllm`): page sizes 16, 32, 64 only; two entry points for prefill and decode |
| Separate prefill and decode backends | A hybrid written `prefill,decode`, e.g. `fa,fi`; `HybridBackend` dispatches on the phase. `auto` gives `fa,fi` on Hopper, `trtllm` on Blackwell, `fi` on older GPUs |
| Triton, FlashMLA, the other names | Not in mini-sglang |

Everything the book says about **metadata** shows up in `FAMetadata`: `cu_seqlens_q` (cumulative new-word counts), `cu_seqlens_k` (cumulative context lengths), `cache_seqlens`, `page_table` (`[bs, max_pages]`), and `get_last_indices` (the last word of each order, which the LM head uses in prefill).
