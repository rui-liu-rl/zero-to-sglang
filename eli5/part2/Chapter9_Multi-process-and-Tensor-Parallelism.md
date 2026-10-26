# Part II Chapter 9 Multi-process and Tensor Parallelism: ELI5 companion

Original: [Chapter9_Multi-process-and-Tensor-Parallelism.md](../../course-material/eng/part2/Chapter9_Multi-process-and-Tensor-Parallelism.md). Same headings, same numbers. The code half is [Chapter9_Multi-process-and-Tensor-Parallelism_code.md](./Chapter9_Multi-process-and-Tensor-Parallelism_code.md). Cast and picture legend: [../README.md](../README.md).

Runnable toy with every number below ("GPUs" are list slices, so the maths is checked for real): `python3 -I eli5/tools/ch9_tp_toy.py`.

## The one-minute version

**Kid version.** One chef can't hold a giant recipe book (too heavy) or can't keep enough notepads (too little shelf). So **several chefs share the job**. Two separate ideas:

| Idea | Kid version | Real name |
|---|---|---|
| **Who is in the kitchen and how do they talk?** | Each chef has her own head waiter (one process per GPU). They all get **the same order slips in the same sequence** | Multi-process execution |
| **How is one dish cooked by several chefs at once?** | The recipe book is cut into pieces, **every chef holds a piece**; for each step they all cook their share, then **pass numbers to each other** to put the dish together | Tensor parallelism (TP) |

```text
   chef 0 (GPU 0)  ┐                                   ┌ own slice of every big matrix
   chef 1 (GPU 1)  ├── same slips, same order ─────────┤ cook the SAME tray together
   chef 2 (GPU 2)  ┘      (control plane: ZMQ + Gloo)  └ combine partial results (data plane: NCCL all-reduce / all-gather)
```

## 1 Learning Objectives

| Original question | Kid answer (and where) |
|---|---|
| What do multi-process and TP each solve? | Organising people vs. splitting the dish (2) |
| How do ranks stay consistent and combine results? | Same slips in the same order + two combining moves (3.1) |
| How are linear layers, vocabulary and KV cache split? | Section 3.2, checked numerically |
| How do the processes cooperate at start-up and per request? | Section 4 |

## 2 Multi-process and Tensor Parallelism

### 2.1 Why Consider Multiple GPUs When the Weights Fit on One GPU?

**Kid version.** "The recipe book fits on the shelf" is not the same as "the kitchen works". The shelf also needs room for **notepads** (KV cache), scratch paper (activations), and the chef's tools (communication buffers, recorded CUDA graphs).

**Tiny numbers (my toy, Parts 8 and 9).** A 32-billion-parameter model in bf16 is `3.2e10 x 2 = 64 GB` of weights (59.6 GiB). On an 80 GB card that leaves about **16 GB** for everything else. With a model of this shape (64 layers, 8 KV heads, head dim 128 → **256 KiB of notes per word**, illustrative) 16 GB holds about **61,000 words** of notes, i.e. about 30 chats of 2,000 words each, before counting any scratch space.

Split over `tp` GPUs, **two** things shrink on each GPU: the weights, and (while there are enough KV heads) the notes per word.

| tp | Weights/GPU | Free/GPU | Notes per word per GPU | Words of notes that fit | vs tp = 1 |
|---|---|---|---|---|---|
| 1 | 64 GB | 16 GB | 256 KiB | 61,035 | 1.0x |
| 2 | 32 GB | 48 GB | 128 KiB | 366,211 | **6.0x** |
| 4 | 16 GB | 64 GB | 64 KiB | 976,562 | **16.0x** |
| 8 | 8 GB | 72 GB | 32 KiB | 2,197,266 | **36.0x** |
| 16 | 4 GB | 76 GB | 32 KiB (**stuck**) | 2,319,336 | 38.0x |

Doubling the GPUs from 1 to 2 gives **six** times the room for notes, not two, because the two savings multiply. From 8 to 16 it barely moves: there are only 8 KV heads, so each GPU already holds a single head and can't hold less (section 3.2). (Upper bounds: the table ignores activations, graph buffers and communication buffers.)

**The other side: more GPUs is not automatically faster.** Each layer now needs communication (section 3.1), and that grows with the number of GPUs. For a trillion-parameter model (`1.6e12 x 2 = 3.2 TB`) a single card is hopeless: at least 40 cards of 80 GB for the weights alone.

**Where the story lies.** Chefs share a kitchen but the dish is still one dish: you can't pay for four chefs and get a dish four times faster. Speed depends on how big the tray is, how fast the chefs can pass numbers (the link between GPUs), and whether there is enough work to hide the passing.

### 2.2 How do the two support multi-card parallelism?

**Kid version.**

| Idea | In the book's words | Kid version |
|---|---|---|
| **Multi-process** | Separate processes with separate memory spaces cooperating by messages. Needed partly to get around Python's GIL | One person per job, each in their own room, passing notes |
| **Tensor parallelism** | The weights of the **same layer** (and the vocabulary) are cut across ranks; they jointly compute the same batch | Each chef holds a slice of every page of the recipe book |

The processes mini-sglang starts (default settings):

| Process | Count | Kitchen role |
|---|---|---|
| API server (main) | 1 | Front desk |
| Scheduler (`minisgl-TP{i}-scheduler`) | one **per GPU** (`tp_size`), each bound to its own GPU and each containing an Engine | A head waiter + her chef, per GPU |
| Detokenizer (`minisgl-detokenizer-0`) | 1; with the default `--num-tokenizer 0` it **also** tokenizes | Card-reader (and card-maker) |
| Extra tokenizers | `--num-tokenizer N` | More card-makers |

A **rank** is a GPU's number in the group (0 to `tp_size - 1`). "Every token is computed jointly by the whole TP group", and the ranks must always be working on the **same batch**.

*Not covered:* mini-sglang supports only TP. The full engine also has data parallelism (several complete copies) and pipeline parallelism (different layers on different GPUs).

## 3 Communication and Partitioning

**Kid version.** Two conditions for several chefs to cook one dish:

1. Everyone cooks **the same step of the same order** at the same time. (Scheduling and control messages.)
2. The partial results can be **combined into exactly what one big chef would have produced**. (Partitioning rules and number-passing.)

### 3.1 Communication: Align Tasks First, Then Combine Results

**Control plane: same slips, same order.**

| Tool | Carries | Kid version |
|---|---|---|
| **ZeroMQ** | Requests, token ids, sampling parameters, replies | The post: between front desk, card-room, and rank 0; and rank 0's **loudspeaker** (publish) to the other ranks (subscribe) |
| **Gloo** (CPU process group) | Tiny control messages: barriers, the NCCL start-up id, and the **count** of slips this round | A bell and a headcount |

**Only rank 0 talks to the outside.** Letting every rank read the real mailbox would let them drift (one reads three slips, another two). So rank 0 reads, then:

```text
   rank 0:  read all slips now waiting  →  Gloo broadcast: "N slips this round"  →  announce each slip on the loudspeaker (ZMQ publish)
   rank 1+: Gloo receive N                →  listen for exactly N announcements (ZMQ subscribe)
```

Every rank then updates its own local queues and caches **from the same messages**, running the same deterministic scheduling code. No rank ever receives a ready-made batch; they *each build the same batch*. Only rank 0 sends replies back; the others' results are identical anyway.

**Data plane: passing numbers between GPUs (NCCL).**

NCCL is NVIDIA's library for moving tensors between GPUs; mini-sglang wraps it as **PyNCCL** (default for multi-GPU; `--disable-pynccl` uses PyTorch's NCCL instead). Two moves:

| Move | Picture | Used for |
|---|---|---|
| **all_reduce** | Everyone has a partial sum; afterwards **everyone has the total** | After a row-parallel layer |
| **all_gather** | Everyone has one slice; afterwards **everyone has all slices glued together** | The LM head (vocabulary slices) |

```text
   all_reduce:  GPU0 [a0 a1]   GPU1 [b0 b1]    →   both: [a0+b0  a1+b1]
   all_gather:  GPU0 [a0 a1]   GPU1 [b0 b1]    →   both: [a0 a1 b0 b1]
```

If a needed collective is missing, later layers get wrong numbers and the output is garbage, not just slow.

**Tiny numbers: how much traffic? (my toy, Part 7)**, for a hidden size 5120, 64-layer model in bf16. Each layer has 2 all_reduces (after `o_proj` and after `down_proj`), plus one for the embedding: `2 x 64 + 1 = 129` calls per forward pass. In a ring all_reduce each GPU sends about `2(tp-1)/tp` times the tensor's size.

| Situation | One tensor | tp = 2 | tp = 4 | tp = 8 |
|---|---|---|---|---|
| Decode, 64 requests | 0.62 MiB | 0.08 GiB | 0.12 GiB | 0.14 GiB |
| Prefill, 8,192 tokens | 80 MiB | 10.1 GiB | 15.1 GiB | 17.6 GiB |

Per-GPU traffic **grows** with `tp` (the tensor stays the same size, the factor rises from 1 to 1.75) while per-GPU compute **shrinks**. That is exactly why more GPUs do not give proportionally more speed. For tiny decode tensors (under a MiB), the cost is usually dominated by the *latency* of 129 handshakes rather than by bandwidth (a general rule about collectives; I have not measured it here).

### 3.2 Partitioning

**Kid version.** Rules of thumb:

| Part | Treatment | Why |
|---|---|---|
| Big matrices | **Cut** | That's where the weight is |
| Small things (RMSNorm weights) | **Copy to everyone** | Tiny; cutting them would need extra passing |
| Heads and KV cache | **Cut by head** | Attention for different heads is independent |

Three ways to cut a matrix. (Note on row/column words: PyTorch stores a layer's weight as `[out, in]`, so "output dimension" is **dimension 0** and "input dimension" is **dimension 1**. The loader's `_SPLIT_DIM_0` and `_SPLIT_DIM_1` lists use exactly that.)

**(a) Column parallel: cut the OUTPUT dimension.** Everyone gets the **whole** input and produces **their slice of the output**. No communication needed yet. Used for `qkv_proj` and `gate_up_proj`.

**(b) Row parallel: cut the INPUT dimension.** Everyone already holds **their slice of the input** (from the previous column-parallel layer) and produces a **partial sum of the whole output**. An all_reduce adds the partials. Used for `o_proj` and `down_proj`.

**(c) Vocab parallel: cut the vocabulary dimension** of the embedding table and LM head.

**Checked by running (my toy).**

| Check | Result |
|---|---|
| Column parallel + all_gather equals the full matmul | **True** |
| Row parallel + all_reduce equals the full matmul | **True** |
| One partial sum alone equals the full result | False (as it should be) |
| Column → local ReLU → row → one all_reduce equals the full MLP | **True** |
| Row-parallel first, ReLU on partial sums, then add, equals ReLU of the full sum | **False** |
| Bias added on **every** rank before the all_reduce equals the full result | **False** (counted `tp` times) |
| Bias added once after the sum | True |

Three lessons:

1. **Column-then-row costs one all_reduce per block.** Attention: `qkv_proj` (column) → attention on the local heads → `o_proj` (row) → all_reduce. MLP: `gate_up_proj` (column) → SiLU and multiply (locally!) → `down_proj` (row) → all_reduce.
2. **You can't put a nonlinearity on partial sums.** `relu(a) + relu(b)` is not `relu(a + b)`. After any row-parallel layer you must all_reduce **before** the next nonlinearity. Column parallel gives each GPU *whole* neurons, so a local activation is exact.
3. **No bias on row-parallel layers**: it would be added on every GPU, and the all_reduce would count it `tp` times. (mini-sglang creates those layers without a bias.)

**Vocabulary (my toy, Part 5):** vocabulary of 10 words on 4 GPUs: `ceil(10/4) = 3` rows each, so the owned row ranges are `[0,3) [3,6) [6,9) [9,10)`; the last GPU has only one real row.

- *Embedding:* each GPU looks up the ids in **its** range and writes zeros for the others; exactly one GPU is non-zero per token; `all_reduce` sums them, and the result equals an ordinary lookup (True).
- *LM head:* each GPU computes the scores of **its** words; `all_gather` glues the slices to `4 x 3 = 12` columns; the extra **padding columns are cut off** (`[:, :num_embeddings]`) to give the true 10. The padding values are never used (in the real code they are uninitialised memory, so the cut matters).

**Attention heads and the KV cache (my toy, Part 6)**, with 32 query heads and 8 KV heads:

| tp | Query heads/GPU | KV heads/GPU | KV bytes/word/GPU (64 layers, head dim 128, bf16) | All GPUs together vs one copy |
|---|---|---|---|---|
| 1 | 32 | 8 | 256 KiB | 1.0x |
| 2 | 16 | 4 | 128 KiB | 1.0x |
| 4 | 8 | 2 | 64 KiB | 1.0x |
| 8 | 4 | 1 | 32 KiB | 1.0x |
| 16 | 2 | **1 (replicated)** | 32 KiB | **2.0x** |
| 3 | not allowed | | | |

- **Query heads** must divide evenly by `tp` (so `tp = 3` fails here).
- **KV heads** that outnumber GPUs are divided; if there are **fewer KV heads than GPUs**, each GPU keeps one head and several GPUs hold **copies** of the same head (allowed only if `tp` is a multiple of the KV head count).
- The cost of replication: going from 8 to 16 GPUs does not shrink the per-GPU notes at all, and the group as a whole stores **two** copies.

**Choosing `tp`:** (1) enough total memory for weights and notes, (2) a value that divides the head counts, and (3) enough work per step to justify the communication.

**Pause and predict**

1. A model has 32 query heads and 4 KV heads. Which of `tp = 2, 4, 8, 16` are allowed and what happens to the KV heads at 8?

<details><summary>Answer</summary>

All four divide 32, so the query heads split. KV heads: 2 and 4 divide 4 evenly (2 and 1 per GPU); 8 and 16 exceed 4 and are multiples of 4, so each GPU keeps 1 KV head, with copies on 2 and 4 GPUs respectively. All four are allowed; at 8 and 16 each KV head is replicated.

</details>

2. Why can a row-parallel layer not simply be followed by the next layer's activation function on each GPU?

<details><summary>Answer</summary>

Each GPU holds only a partial sum, and `f(a) + f(b)` differs from `f(a + b)` for a nonlinear `f`. The all_reduce has to turn partial sums into the true total first. (Toy Part 3: `False`.)

</details>

## 4 From Startup to Execution

**Kid version.** You run one command and the system hires the staff:

```bash
python -m minisgl --model "Qwen/Qwen3-32B" --tp-size 2
```

```text
   main process (front desk)
     │  run_api_server(...) builds its in-tray, then calls start_subprocess:
     │     ┌─ spawn minisgl-TP0-scheduler  (rank 0, GPU 0)   ──▶ Engine loads its slice of weights, KV pantry, graphs ─▶ sync_all_ranks ─▶ run_forever
     │     ├─ spawn minisgl-TP1-scheduler  (rank 1, GPU 1)   ──▶ (same)
     │     ├─ spawn minisgl-detokenizer-0                    ──▶ loads tokenizer, raises its flag
     │     └─ (optionally N tokenizers)
     │  waits on ack_queue for:  rank 0 + detokenizer + each tokenizer  =  num_tokenizers + 2 flags
     ▼
   uvicorn.run(app): the front desk opens for HTTP
```

| Step | What happens |
|---|---|
| 1 | The main process makes one copy of the server settings per rank with that rank's number filled in, and starts one scheduler process per rank (start method **spawn**, so each process gets a clean Python) |
| 2 | Inside each scheduler: create the **Engine** (set TP info first; bind `cuda:{rank}`; start communication; build the model on the meta device; load **this rank's slices** of the weights; allocate the KV pantry; capture graphs), then the managers and the mailbox, then wait at a barrier for all ranks, then loop forever |
| 3 | Only after every flag is up does the front desk start listening for HTTP |

Only the rank-0 scheduler raises a flag, since it speaks for all the GPU processes.

**Where the story lies.** "Cooperating processes" suggests shared memory. They share **none**: all coordination is explicit messages (ZMQ, Gloo) and explicit tensor exchange (NCCL). That is why every message goes through rank 0 and why identical code on identical inputs is the safety net.

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** Several chefs, each in her own room, each with a slice of every recipe page. They get the same slips in the same order (rank 0 reads the mail and tells the others how many and what). Each cooks her share, and at fixed points (after attention's output projection, after the MLP's down projection, at the vocabulary ends) they pass numbers to rebuild the full result. More chefs means more room for notes (much more!) but more passing; the benefit depends on the size of the tray and the speed of the corridor between chefs.

### 5.2 Exercises

**1. What problems do multi-process execution and TP each solve?**

<details><summary>Answer</summary>

Multi-process: organising the system (front desk, card-room, one scheduler per GPU), working around Python's GIL, and message-based coordination. TP: making a model that does not fit (or does not leave enough room for KV) on one GPU run by splitting each layer's weights and computation across ranks.

</details>

**2. Why can't each rank independently receive any number of requests? What must stay consistent, and how is it ensured?**

<details><summary>Answer</summary>

Because every rank must run the same batch, in the same order, with the same lengths and cache state; otherwise collectives would combine mismatched tensors. Consistency comes from rank 0 being the only reader, broadcasting the count of messages (Gloo) and then each message (ZMQ pub/sub), after which all ranks run the same deterministic scheduling code on the same inputs (the decode batch is also sorted by `uid` so rank-local sets list requests in the same order).

</details>

**3. Why doesn't doubling TP necessarily halve GPU memory use or double speed?**

<details><summary>Answer</summary>

Memory: weights halve (apart from the small replicated parts), but KV per word per GPU stops shrinking once `tp` exceeds the number of KV heads (toy: tp 8 → 16 leaves 32 KiB/word, and the group stores 2 copies). Speed: per-GPU compute halves but per-GPU all_reduce traffic grows with `2(tp-1)/tp` and the number of handshakes stays at 129 per pass, so communication takes a larger share. Small batches (decode) are latency-bound.

</details>

**Extra drills (not in the book).**

4. `tp = 4`; the LM head vocabulary is 151,936. Rows per GPU? (`ceil(151936 / 4) = 37,984`; 151,936 is divisible by 4, so no padding.) With `tp = 3`? (`ceil(151936 / 3) = 50,646`, giving 151,938 columns after gather, so 2 padding columns are cut off.)
5. How many all_reduces does one forward pass of a 28-layer model do, ignoring the LM head? (`2 x 28 + 1 = 57`: two per layer plus the embedding.)
6. rank 0 reads three slips and rank 1 is told "2" by mistake. What happens? (Rank 1 would stop listening early; the third slip would be read at the start of the next round, so rank 1 would run that request one iteration later than rank 0: the ranks go out of step and collectives would combine tensors from different batches.)

## 6 Extra: what the toy leaves out

The toy cuts lists in the same pattern as the real code but ignores: the fused QKV layout (q, k, v slices of one rank concatenated), the speed of NCCL kernels, overlap of communication and computation, and the PyNCCL staging buffer (see the code half). It proves the **algebra**, not the performance.
