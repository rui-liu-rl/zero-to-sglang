# Chapter 2 Introduction to Inference: ELI5 companion

Original: [Chapter2_Introduction to Inference.md](../../course-material/eng/part1/Chapter2_Introduction%20to%20Inference.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

> **Spoiler notice.** Your "do this now" task is §4 (Prefill and Decode) plus three sentences, the last one about decode and batch size. Sections 4.3 and 5.2 below contain hidden answer blocks that give that bridge away. **Write your own three sentences first**, then open them.

## The one-minute version

**Kid version.** The chef (model) already knows the recipe book. *Inference* is simply the chef cooking an order. It happens in two phases:

1. **Prefill**: the chef reads the *whole* order slip at once and jots notes on every word. Big job, but the chef can think about all the words in parallel. This is fast per word and limited by how hard the chef can *think*.
2. **Decode**: the chef writes the answer **one word at a time**. For every single word the chef must flip through the *whole* recipe book again. Tiny thinking, but a lot of page-flipping. Limited by how fast the chef can *fetch pages*.

```text
   ORDER SLIP                                   ANSWER
   "What is KV cache?"                          "It is notes the chef keeps ..."
   ┌───────────────────────┐                    ┌─────┬─────┬─────┬─────┬─────┐
   │ PREFILL: read it all  │ ───────────────▶   │ It  │ is  │notes│ ... │ ... │
   │ at once, write notes  │   DECODE: one word │  ①  │  ②  │  ③  │     │     │
   └───────────────────────┘   at a time, each   └─────┴─────┴─────┴─────┴─────┘
        thinking-limited       needs the book        fetching-limited
```

The one idea that explains most of this chapter: **reading the recipe book costs the same whether the chef cooks 1 portion or 200 portions in that same pass.** So the trick of inference engineering is to make every page-flip serve as many portions as possible.

## 1 Learning Objectives

| Original question | Kid answer (the full answer is in the section named) |
|---|---|
| Training vs inference: objectives, computation, resources | Training *changes* the recipe book (read, taste, correct: costs 6N per word). Inference only *uses* it (2N per word). See 3.1 |
| Life cycle of a token | Scheduled, prefilled, decoded, returned. See 4.1 |
| Prefill vs decode | Think-limited batch of words vs fetch-limited one-word-at-a-time. See 4.2, 4.3 |
| What affects observed performance? | Compute speed, memory speed, memory size, batch size, prompt length, reuse of notes |
| Compute-bound or memory-bound? | Compare the job's *arithmetic intensity* with the GPU's *ridge point* (the roofline). See 5 |
| Bottlenecks in agents and multi-turn | The same prompt prefix gets re-read again and again. See 6 |

## 2 What Is Inference?

**Kid version.** Training is the chef going to school. Inference is the chef working a shift. An order goes through five steps:

```text
   words ─▶ cards (token IDs) ─▶ vectors ─▶ [ floors: meet + think ] x L ─▶ scores for every word
      ▲                                                                        │
      │                          pick one, add it to the end                   ▼
      └───────────────────────── repeat until "done" ◀─────────────────────────┘
                      this loop is called AUTOREGRESSIVE
```

**A tiny worked loop.** Prompt `"Hi there"` (suppose 2 tokens). Here is what the engine does, using made-up outputs:

| Step | What goes in | Phase | What comes out |
|---|---|---|---|
| 0 | `Hi`, `there` (both at once) | prefill | notes for 2 tokens + the first answer token: `How` |
| 1 | `How` (just this one) | decode | `are` |
| 2 | `are` | decode | `you` |
| 3 | `you` | decode | `?` |
| 4 | `?` | decode | `<EOS>` (end-of-sequence: "I'm done") |

One prefill, then one decode step per remaining word. The answer's length decides how many decode steps.

**Where the story lies.** The book's cake analogy ("ingredients, processing, cake") is fine for the idea, but the real wrinkle is that the "cake" is made *one crumb at a time, and each crumb needs the whole recipe book*. That is the core of the rest of this chapter.

## 3 Inference and Training

### 3.1 Differences Between Inference and Training

**Kid version.** Training is a student doing homework with the answer key: for every word the student (a) answers, (b) checks against the key, (c) figures out which pencil-erasing fixes the mistake. Inference is the same student taking the real test: only (a).

Because in training the true next words are *known* (they are in the book), the student can do all positions of a sentence at the same time (this is *parallel*). At test time the student's own previous words are the only source, so words must come out one after another (*sequential*).

```text
   TRAINING (answer key known)                   INFERENCE (no key)
   position:  1   2   3   4                      position:  1  ──▶ 2  ──▶ 3  ──▶ 4
   input   :  A   B   C   D   all at once        each step needs the word just produced
   mask    :  each position sees only the left   (but inside one step, lots of parallel math)
```

**The 2N / 4N question, answered from scratch.** Let `N` be the number of weights in the model.

1. A layer does a matrix multiply `y = W x`. For one token, every weight in `W` is used once: one multiply and one add. That is **2 FLOPs per weight**. Adding over the whole model: **forward costs about 2N per token**. (Attention's score math is extra, but for ordinary lengths it is small compared with the weight multiplies; the book ignores it.)
2. Backward has to compute two things for every one of those multiplies:
   - how the loss changes with respect to the **input** `x` (so the earlier layer can learn): `W^T * (error)`. Same size of multiply as forward: **2N**.
   - how the loss changes with respect to the **weights** `W` (so we can update them): `(error) * x^T`. Again the same size: **2N**.
3. So backward = **4N**, and one training step per token = 2N + 4N = **6N**. Inference per token = **2N**.

```text
   forward   :  y = W x                        2N   ◀── also all that inference does
   backward  :  dx = W^T dy                    2N
                dW = dy x^T                    2N
   training total per token                    6N
```

Check yourself: if you use *activation checkpointing* (throw away intermediate results during the forward pass and recompute them during backward to save memory), what does the per-token cost become?

<details>
<summary>Answer (try first)</summary>

Forward (2N) + recompute forward (2N) + backward (4N) = **8N** per token. You pay about 33% more compute to save memory.

</details>

**Tiny numbers: one training run vs one request** (the book's 8-billion-parameter model, so `N = 8 x 10^9`):

| Quantity | Formula | Value |
|---|---|---|
| Training tokens (rule of thumb: 20 per parameter) | 20 x N | 160 billion |
| Training FLOPs | 6 N D | 6 x 8e9 x 1.6e11 = **7.7e21** |
| One H100 at peak (the book uses 990 TFLOPS) | | 8.6e19 FLOPs per day |
| GPUs for one day at 100% usage | 7.7e21 / 8.6e19 | **about 90** (more in real life: real usage is 30-50%) |
| One request (512 in, 128 out) | 2 x N x 640 | **1.0e13 FLOPs** |
| Requests that equal one training run | 7.7e21 / 1.0e13 | **750 million** |

So a single request is *tiny* next to training. Why does the book say serving can still end up costing more than training? Because the **bill is in GPU-seconds, not FLOPs**, and decode wastes GPU-seconds when it runs unbatched. A rough calculation with the same 8B model on an H100 (3.35 TB/s memory):

| How decode is run | GPU-seconds per request | For 1M requests/day: GPUs busy | GPU-days per year |
|---|---|---|---|
| One request at a time (batch 1) | 0.63 s | 7.3 | about 2,700 |
| 16 requests per step | 0.059 s | 0.68 | about 250 |
| 64 requests per step | 0.030 s | 0.35 | about 130 |

(Assumptions: prefill at 40% of peak; each decode step costs the time to read 16 GB of weights; KV reads and queueing ignored.) Training at 40% utilisation is about 225 GPU-days. So with careless serving the annual bill is over 10 times a training run; with good batching it is about the same. **That gap is the entire business case for inference engineering.**

**Drift note.** The book says "the annual total can exceed the training cost" without showing the arithmetic. By raw FLOPs alone, 1M requests/day for a year is about 3.7e21, around half of the training run. It only exceeds training when utilisation is poor or traffic is higher. The table above shows why.

**Real words.**

| Kid word | Real word |
|---|---|
| Answer key known | Teacher forcing / ground-truth prefix |
| Pencil-erasing fix | Backward pass / gradient |
| Seeing only the left | Causal mask |
| `N` | Parameter count |
| `D` | Number of training tokens |
| FLOPs | Floating-point operations (one multiply or one add each) |

**Where the story lies.** "Inference is sequential, training is parallel" is about the *positions*. Inside one step, both do enormous parallel math.

**KV cache, first sighting.** At test time, every new word needs a meeting (attention) with all previous words. Rereading and recomputing all earlier words' Labels and Gifts every step would be crazy. So the chef saves them in a **notepad (KV cache)**. Saves compute; costs memory.

**Pause and predict.** Is the 2N figure the same for a prompt of 10 tokens and a prompt of 10,000 tokens, *per token*?

<details>
<summary>Answer (try first)</summary>

The weight-multiply part is the same per token (2N). Attention adds a term that grows with how many earlier tokens each token meets, so very long prompts cost somewhat more per token. For ordinary lengths the weight part dominates.

</details>

### 3.2 How Training and Inference Work Together

**Kid version.** A chef's school and the restaurant are not separate worlds. The school *uses* the restaurant (to taste-test its own cooking) and the restaurant is *shaped by* the school (what recipe-book layout was chosen).

```text
        ┌──── train ───▶ model ───▶ serve ───▶ users ────┐
        │                                                 │
        └──── evaluate, collect feedback, fix ◀───────────┘
```

**1. Inference is part of training.** Validation runs inference. Reinforcement-learning post-training makes the model *write answers*, then a reward score judges them, then the model is updated. So the engine you are learning (SGLang) is also the "answer-writing" half of training loops.

**2. Training sets the ceiling for inference.** Choices made before training are locked in:

| Training-time choice | Effect at serving time |
|---|---|
| Number of layers and width | Weights memory (70B x 2 bytes = 140 GB just for the book) |
| **GQA** (share Key/Value heads) | Smaller notepad. You cannot turn an MHA model into a GQA model afterwards without retraining |
| Quantization-aware training | Lets you serve in INT8 with little accuracy loss. Quantizing a model that wasn't prepared can hurt quality |

**Pause and predict.** Why can't you just "turn on" GQA for a model trained with full multi-head attention?

<details>
<summary>Answer (try first)</summary>

The Key/Value projection weights were trained as separate per-head matrices. The Question heads were trained to expect those separate K/V heads. Forcing groups to share changes the function the model computes. You would need to retrain, or at least fine-tune carefully.

</details>

## 4 Prefill and Decode

> **Spoiler notice reminder.** Write your three sentences before opening the 4.3 answer block.

### 4.1 A Token's Life Cycle

**Kid version.** An order's journey through the kitchen:

```text
 ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   ┌──────────────┐
 │ 1. SCHEDULE │──▶│ 2. PREFILL  │──▶│ 3. DECODE (repeat)  │──▶│ 4. RETURN    │
 │ head waiter │   │ read slip,  │   │ one new word per    │   │ cards turned │
 │ picks the   │   │ write notes │   │ step, add its notes │   │ back to words│
 │ order+batch │   │ for all     │   │ until EOS or limit  │   │ and sent out │
 └─────────────┘   └─────────────┘   └─────────────────────┘   └──────────────┘
```

| Stage | What happens | Cost shape |
|---|---|---|
| Request scheduling | The scheduler assigns the request to a GPU (or group) | Mostly CPU bookkeeping |
| Prefill | All prompt tokens in one pass; builds the full KV cache; picks the first output token | Lots of compute, highly parallel |
| Decode | One token per step, using the cache; appends the new token's K/V | Little compute per step, but each step reads weights and the whole cache |
| Return | Token IDs decoded back to text and sent to the user | Cheap |

**Where the story lies.** Prefix caching (reusing notes from earlier requests) changes the prefill stage, but it is an optimisation on top of this basic cycle; the book leaves it out of the figure on purpose.

### 4.2 Prefill

**Kid version.** The chef reads the whole order slip, and for *every* word jots (a) what the word is looking for, (b) its label, (c) its gift. All words are processed side by side, so the big, well-organised matrix multiplications use the stove at full blast.

**Where the numbers come from (derive it once).** Take a layer with width `d`, an order of `N` tokens.

| Part | Matrices | FLOPs (2 per weight per token) |
|---|---|---|
| Q, K, V and output projections | 4 matrices of `d x d` | `2 x 4 x N x d^2 = 8 N d^2` |
| Feed-forward (three matrices of `d x 8/3 d`) | 3 matrices | `2 x 3 x N x (8/3) d^2 = 16 N d^2` |
| Attention scores `Q K^T` | `N x N` scores, each a `d`-long dot product | `2 N^2 d` |
| Score-weighted sum `scores x V` | `N x N` times `N x d` | `2 N^2 d` |
| **One layer total** | | **`24 N d^2 + 4 N^2 d`** |

**Tiny numbers (the book's LLaMA-7B, `d = 4096`, `L = 32` layers, `N = 3000`):**

- linear part per layer: `24 x 3000 x 4096^2 = 1.21e12`
- attention part per layer: `4 x 3000^2 x 4096 = 1.47e11` (11% of the layer)
- all 32 layers: **about 43.4 TFLOPs**; on an A100 at its 312 TFLOPS peak: **0.14 s** at the very best.

**When does the quadratic part take over?** It equals the linear part when `4 N^2 d = 24 N d^2`, i.e. `N = 6d`, which is 24,576 tokens for this model.

| Prompt length `N` | Quadratic share of a layer's FLOPs |
|---|---|
| 500 | 2.0% |
| 3,000 | 10.9% |
| 10,000 | 28.9% |
| 24,576 | 50.0% |
| 100,000 | 80.3% |

So for "a few thousand" tokens, compute grows about *linearly* with prompt length (the book's Figure 4). Very long prompts bend the curve upward.

**Memory traffic.** The weights are read once: 14 GB. The new notes are written: each token needs `2 (K and V) x 32 layers x 4096 x 2 bytes = 0.5 MiB`, so 3000 tokens write about 1.5 GB. Total about **15-16 GB**.

**The key ratio.** `43.4 TFLOPs / 15.5 GB = about 2,800 FLOPs per byte`. Thousands of calculations for every byte fetched: the stove is the limit, not the pantry.

**Real words.**

| Kid word | Real word |
|---|---|
| Think-limited | Compute-bound |
| Reading the recipe book once | Loading weights from HBM (GPU memory) once per forward pass |
| FLOPs per byte | **Arithmetic intensity (AI)** |

**Where the story lies.** The book's quick estimate counts FLOPs for multi-head attention. Real models with GQA have smaller K/V projections and fewer cached bytes. Orders of magnitude stay the same.

**Pause and predict.** If you halve the prompt (1,500 tokens), roughly what happens to prefill FLOPs and to the KV bytes written?

<details>
<summary>Answer (try first)</summary>

FLOPs: a bit under half (linear part halves, quadratic part quarters). KV bytes written: exactly half (0.75 GB). Weights read: unchanged (14 GB). So **arithmetic intensity falls**, because the same weight read now serves fewer tokens.

</details>

### 4.3 Decode

**Kid version.** The chef writes **one** new word. For that one word the chef must (1) use every page of the recipe book, (2) reread the whole notepad. But only one word's worth of thinking happens. The stove is idle most of the time; the chef spends nearly the whole step **walking to the pantry**.

**Tiny numbers (same LLaMA-7B, 3,000-token context).**

| Step cost | Value | Time at A100 limits |
|---|---|---|
| FLOPs for one token: `32 x (24 d^2 + 4 S d)` | 14.4 GFLOPs | `14.4e9 / 312e12` = **0.046 ms** |
| Bytes moved: weights 14 GB + KV 1.5 GB (+ 0.5 MiB new) | about 15.5 GB | `15.5e9 / 1.6e12` = **9.7 ms** |

The pantry walk takes **210 times longer** than the thinking. The stove is busy about 0.5% of the time.

**The notepad keeps growing.**

| After generating | Size of the notepad |
|---|---|
| 1 token | about 1.5 GB |
| 100 tokens | about 1.55 GB |
| 2,000 tokens | about 2.5 GB |

Two different limits appear:

- **Speed limit = memory bandwidth** (how fast you can fetch bytes): weights plus the notepad, every step.
- **Capacity limit = memory size** (how much fits): the notepad grows with every word, for every conversation.

```text
   one decode step, B = 1                       time ──▶
   thinking  ▏                                  0.05 ms
   fetching  ▕███████████████████████████████▏  9.7 ms     ← this is what you wait for
```

**Now for the idea behind your sentence 3.**

<details>
<summary>Spoiler: what happens to decode at a batch of 32? (open only after writing your own three sentences)</summary>

**The recipe book is shared; the notepads are not.** In one decode step, the chef can handle `B` customers' next words together. The 14 GB of weights are fetched *once* for all of them. But each customer has their *own* notepad, and each notepad must be read in full.

```text
   bytes per step  =  14 GB (weights, once)  +  B x 1.5 GB (each customer's notes)
   FLOPs per step  =  B x 14.4 GFLOPs
```

Using the book's formulas, with every customer holding 3,000 tokens of context:

| Batch `B` | Arithmetic intensity (FLOP/byte) | Step time | Tokens/s (all customers) |
|---|---|---|---|
| 1 | 0.9 | 9.7 ms | 103 |
| 2 | 1.7 | 10.7 ms | 187 |
| 8 | 4.4 | 16.6 ms | 482 |
| **32** | **7.2** | **40 ms** | **796** |
| 64 | 8.1 | 72 ms | 893 |
| 256 | 8.9 | 260 ms | 983 |
| limit, `B` very large | 9.2 | | about 1,000 |

What this says:

1. **Batching helps a lot at first**: going from `B = 1` to `B = 32` multiplies total throughput by about 8 for a 4x longer step. Each individual customer waits 4x longer per word, but the *kitchen* serves 8x more words per second.
2. **It stops helping**: once the notepads dominate the bytes, intensity plateaus near 9 FLOP/byte, far below the A100's ridge of 195. With 3,000-token conversations, decode stays memory-bound at *any* batch size.
3. **What limits `B` in practice is notepad capacity**: after 14 GB of weights, an 80 GB card holds only about 42 such conversations (66 GB / 1.57 GB).
4. With shorter conversations (300 tokens) the plateau rises to 83 FLOP/byte, and throughput at `B = 256` reaches 7,500 tokens/s. Short contexts batch far better.

The **memory-bound arithmetic of the weight multiplies alone** is exactly `intensity = B`: one weight read serves `B` tokens. For prefill it is the prompt length `N`. The same principle, two names. Everything in the scheduler chapters is a way of keeping `B` (tokens per forward pass) high enough without blowing the notepad budget.

</details>

**Pause and predict.** Why does the 4.2 prefill (3000 tokens) look compute-bound while a decode step for the same model is memory-bound, even though both read the same 14 GB of weights?

<details>
<summary>Answer (try first)</summary>

Prefill spends those 14 GB of fetching on 3,000 tokens of thinking, so each fetched byte feeds many calculations. Decode spends it on a single token. The *same* page-flip, spread over very different numbers of portions.

</details>

## 5 Identifying Inference Bottlenecks

### 5.1 Roofline Model Principles

**Kid version.** Imagine a cyclist who is limited by **two** things: (1) how hard they can pedal (compute), and (2) how fast their water bottles can be handed to them (memory). If the ride needs lots of pedalling per bottle, pedalling is the limit; if it needs a lot of bottles per pedal-stroke, bottle hand-off is the limit. The roofline asks: *which limit hits first for my job?*

**The formula in time form (easier to feel):**

```text
   step time  ≈  max( FLOPs / peak_compute ,  bytes / memory_bandwidth )
                       "thinking time"          "fetching time"
```

Whichever is bigger *is* the bottleneck. Divide both by bytes and you get the two roofline lines.

**Picture.**

```text
   performance
   (FLOP/s)
        ▲                        ┌──────────────────────────  compute roof  (e.g. 312 TFLOPS)
 312 T  │                     ╱  │
        │                  ╱     │
        │               ╱        │   ◀── prefill N=3000 sits out here (AI ≈ 3000)
        │            ╱           │
        │         ╱ slope = memory bandwidth (1.6 TB/s)
        │      ╱
        │   ●  decode B=1 (AI ≈ 1) way down at the bottom-left
        └───┴──────────────────┴────────────────────▶  arithmetic intensity (FLOP/byte)
                          ridge point = 195
        memory-bound   ◀── ridge ──▶   compute-bound
```

**Two numbers you need:**

| Quantity | Formula | A100 (book) | H100 (approx.) |
|---|---|---|---|
| **Ridge point** | peak FLOP/s / bandwidth | 312e12 / 1.6e12 = **195** | 990e12 / 3.35e12 = **about 295** |
| **Arithmetic intensity (AI)** | FLOPs / bytes moved | depends on the job | depends on the job |

**Decision rule.** If AI < ridge: speed ≈ AI x bandwidth (**memory-bound**, stove waits). If AI >= ridge: speed ≈ peak compute (**compute-bound**, pantry keeps up).

**The three steps:** (1) read the hardware's peak and bandwidth, compute the ridge. (2) Compute the job's AI. (3) Compare, then choose the optimisation:

| If you are... | Typical fixes |
|---|---|
| Memory-bound | Move fewer bytes: quantize weights/KV, share K/V heads, reuse cached notes, fuse kernels, **batch more tokens per weight read** |
| Compute-bound | Do fewer or faster FLOPs: better kernels, tensor cores, lower precision, avoid recomputation |

**Where the story lies.** The roofline ignores latency overheads (kernel launches, communication, waiting for the CPU). A job can be far below its roofline for those reasons; Chapter 3 and Part II's CUDA-graph material deal with that.

### 5.2 Roofline Model Example

**Kid version.** Almost everything a model does is a **matrix multiply** (GEMM). So let's work out the AI of one multiply, then see what prefill and decode look like through that lens.

**GEMM: `[M x K] times [K x N]`, in bf16 (2 bytes per number):**

```text
   FLOPs = 2 M K N                       (each of M*N outputs = K multiplies + K adds)
   Bytes = 2 (M K + K N + M N)           (read A, read B, write C)
   AI    = M K N / (M K + K N + M N)     (the 2s cancel)
```

For a square weight matrix (`K = N = 4096`) and `M` rows (= tokens processed together):

| `M` (tokens in the batch) | AI | Memory-bound or compute-bound on an A100 (ridge 195)? |
|---|---|---|
| 1 | 1.0 | memory |
| 8 | 8.0 | memory |
| 32 | 31.5 | memory |
| 128 | 120.5 | memory |
| **216** | **195** | the ridge |
| 512 | 409.6 | compute |
| 4,096 | 1,365 | compute |

**The takeaway in one line: for a weight matrix, AI is about `M`, the number of tokens you push through it together.** One fetch of the weights, `M` portions cooked. (When `M` is large relative to `K` the formula flattens, but that is far from our range.)

**Prefill vs decode, with the book's notation** (`N` = prompt length, `S` = context length, `d`, `L`, bf16, batch 1):

| | AI (book's formula) | At A100 ridge 195 |
|---|---|---|
| Prefill | `(24 N d^2 + 4 N^2 d) / (24 d^2 + 4 N d)` which simplifies to exactly **`N`** | `N = 100`: AI = 100 (memory-bound); `N = 3000`: AI = 3000 (compute-bound) |
| Decode | `(24 d^2 + 4 S d) / (24 d^2 + 4 d + 4 S d)` which is about **1** | memory-bound |

**Clarification of an apparent contradiction.** The book's §4.2 uses `N = 3000` and concludes prefill is compute-friendly, while §5.2 uses `N = 100` and says "it is still memory-bound". Both are right: the same formula crosses the ridge near `N = 195`. A short prompt (under about 200 tokens) alone on an A100 is memory-bound too; a long one is compute-bound. That is also why servers *combine* several short prompts into one prefill pass.

**Drift notes.** (1) The text says "discussed in Section 2.3"; that is a leftover from an older numbering and means §4.2-4.3. (2) The decode analysis assumes batch size 1 and the list of decode fixes (quantization, KV reuse) does not mention **batching**, which is the biggest one. See the spoiler in 4.3.

**Pause and predict.** A decode batch of 64 sequences of 3,000 tokens has total AI of about 8. Which single change would raise that AI the most: doubling the weights' precision to fp32, halving the KV bytes per token (GQA, KV quantization), or doubling the batch to 128?

<details>
<summary>Answer (try first)</summary>

Halving the KV bytes per token. At this batch size, the notepads (64 x 1.57 GB = 100 GB) dwarf the 14 GB of weights, so the weights are no longer the main cost. Doubling the batch barely changes AI (8.1 to 8.6). Doubling precision to fp32 would make it worse. This is why GQA, MLA and KV quantization matter so much for serving.

</details>

## 6 Inference in Real-World Scenarios

### 6.1 Agent Collaboration

**Kid version.** An agent chef works in a loop: think, use a tool, read what the tool said, think again. Each time the chef re-reads the *whole story so far*. If there are no saved notes for that story, the chef rereads everything again from the top.

```text
   loop 1: prefill [ system prompt + tools + task ]                 → tool call
   loop 2: prefill [ system prompt + tools + task + result 1 ]      → tool call
   loop 3: prefill [ ......................... + result 2 ]         → tool call
           └──────── mostly the SAME prefix, re-read every time ────┘
```

**Tiny numbers.** A 5,000-token base prompt; each of 20 tool steps adds 500 tokens.

| Strategy | Tokens that must be prefilled over the whole job |
|---|---|
| No reuse of notes (re-read everything each loop) | **195,000** |
| Perfect prefix reuse (only the new part each loop) | **15,000** |

13x less work with reuse. This is the reason for the prefix-cache chapter (Part II chapter 8).

**Cross-agent communication.** If several agents run one after another, each agent's output becomes the next agent's input. So messages lengthen the next prefill, and waiting for another agent adds latency, which is time the GPU may sit idle or compete with other decode requests.

### 6.2 Multi-Turn Conversations

**Kid version.** Each new chat message resends the *entire* conversation. Without saved notes, the chef rereads everything each turn, so the wait for the first word (TTFT) grows with every turn.

| Turn | TTFT with no reuse (200 tok per turn, prefill at 1,000 tok/s) | TTFT with prefix reuse |
|---|---|---|
| 1 | 200 ms | 200 ms |
| 2 | 400 ms | 200 ms |
| 5 | 1,000 ms | 200 ms |
| 10 | 2,000 ms | 200 ms |

**The notepad cost.** A 100,000-token conversation on the 7B multi-head model needs about 49 GiB of notes by itself. Only one such customer fits next to the weights on an 80 GB card, so **long contexts shrink the batch**, which (per section 4.3) lowers throughput.

**Pause and predict.** The book's exercise 5 (a financial-documents RAG assistant with a long chat). Name three bottlenecks before reading the answer.

<details>
<summary>Answer (try first)</summary>

(1) Every question injects retrieved documents, so prompts are long and prefill is heavy. (2) The growing history is re-prefilled each turn unless prefix-cached, so TTFT grows. (3) Long contexts make big notepads, so fewer simultaneous users fit and decode throughput drops. (Also: if retrieved chunks differ per question, the cacheable shared prefix is short; put stable text first.)

</details>

## 7 Summary and Exercises

### 7.1 Summary

**Kid version.** Inference = the chef cooking. Prefill = read the whole slip (thinking-limited). Decode = write one word at a time (fetching-limited, and the notepad grows). The roofline tells you which limit you hit: compare *FLOPs per byte* to the GPU's ridge. For weight multiplies, FLOPs per byte is about the number of tokens you process together, which is why batching and prefix reuse dominate serving design.

```text
   prefill  : tokens together = N   → AI ≈ N      → usually compute-bound once N > ~200
   decode   : tokens together = B   → AI ≈ B      → memory-bound; notepads cap B
```

### 7.2 Exercises

Try them on paper; hints are collapsed.

**1. Training vs inference: differences and connections.**

<details><summary>Hint</summary>

Weights change vs fixed; 6N vs 2N FLOPs per token; training parallel over positions with known targets, inference sequential; training keeps activations and optimizer state, inference keeps the KV cache; inference runs *inside* training (evaluation, RL rollouts) and architecture chosen in training sets inference cost (GQA, size).

</details>

**2. Compare prefill and decode.**

<details><summary>Hint</summary>

Prefill: all prompt tokens at once, builds the cache, AI ≈ N, compute-bound for long prompts. Decode: one token per step per sequence, reads weights + cache, AI ≈ B (about 1 at batch 1), memory-bound, cache grows.

</details>

**3. What is the significance of the roofline model?**

<details><summary>Hint</summary>

It tells you in one picture whether a job is limited by compute or memory, and therefore which kind of optimisation can possibly help. Axes: x = arithmetic intensity (FLOP/byte), y = achieved FLOP/s; flat roof = peak compute; slanted line = bandwidth x AI; the corner = ridge point.

</details>

**4. Use the roofline on a concrete case.**

<details><summary>Hint</summary>

Pick hardware (A100: 312 TFLOPS, 1.6 TB/s, ridge 195). Pick a job (decode, LLaMA-7B, batch 1, S = 3000): FLOPs 14.4 G, bytes 15.5 G, AI = 0.93. Compare: 0.93 << 195, so memory-bound. Then say what helps: quantize the weights, share K/V heads, or batch more sequences.

</details>

**5. A RAG + long-chat assistant: main bottlenecks.** (see the answer block in 6.2)

**Extra drills (not in the book).**

6. Show that the book's prefill AI formula equals `N` exactly. (Factor `N` out of the numerator: `N (24 d^2 + 4 N d)`; the denominator is `24 d^2 + 4 N d`.)
7. At what prompt length is a prefill on an H100 (ridge about 295) compute-bound? (Around 295 tokens.) What does that mean for a server that wants to keep the GPU busy with short prompts?
8. Why does 80 GB of GPU memory limit decode throughput through the notepad, not through weights, for long contexts?
