# Part II Chapter 1 mini-sglang: What an Inference Engine Looks Like: ELI5 companion

Original: [Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like.md](../../course-material/eng/part2/Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

The code in this chapter lives in `/Users/rui/code/mini-sglang` (branch `eli5-comments`, where every file named here carries ELI5 comments). The book's reference commit is `9a91cfa`, which is exactly the commit that branch was cut from, so there is **no drift** between book and source.

## The one-minute version

**Kid version.** A **model** is the chef's brain: give it words, it gives back scores for the next word. An **inference engine** is the whole restaurant built around that chef: the front desk that takes orders, the card-maker, the head waiter who decides what to cook next, the pantry that holds the notepads, and the runners who carry words back to customers. The chef alone cannot run a restaurant.

```text
   a MODEL (just the chef's brain)           an ENGINE (the whole restaurant)
   ┌─────────────────────────────┐           ┌────────────────────────────────────────────────┐
   │ model(token ids) → scores   │           │ front desk · card-maker · head waiter · chef    │
   └─────────────────────────────┘           │ pantry · runners · rules for many customers     │
                                             └────────────────────────────────────────────────┘
```

Part II builds this restaurant **one problem at a time**, in the order the problems appear. This chapter is the floor plan.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Difference between model invocation and an inference engine | The chef cooking once vs the restaurant serving hundreds |
| Draw the main components and data flow | Front desk → card-maker → head waiter → chef → card-reader → front desk |
| States a request goes through | Waiting, being read (prefill), being written (decode), done, or cancelled |
| What `Req`, `Batch`, `Context` store | The order slip, the tray of slips, the kitchen whiteboard |
| Why Part II is built in this order | Each new part fixes the biggest pain of the one before |

## 2 From Model Invocation to an Inference Engine

### 2.1 Starting with an Autoregressive Generation Loop

**Kid version.** Here is the simplest possible way to get text out of a model:

```python
input_ids = tokenizer.encode(prompt)

for _ in range(max_new_tokens):
    logits = model(input_ids)
    next_token = sample(logits[:, -1, :])
    input_ids = append(input_ids, next_token)

output = tokenizer.decode(input_ids)
```

*Line by line:*

| Line | Kid words |
|---|---|
| `input_ids = tokenizer.encode(prompt)` | Turn the order slip's words into numbered cards |
| `for _ in range(max_new_tokens):` | Repeat once per word we want to write |
| `logits = model(input_ids)` | Chef: read *all* the cards so far and give a score for every possible next word |
| `sample(logits[:, -1, :])` | Look only at the scores for the *last* position, and pick a word |
| `append(input_ids, next_token)` | Put the new card on the end of the pile |
| `tokenizer.decode(input_ids)` | Turn all the cards back into words |

**What is wrong with it** (the book's point): it re-feeds the *entire* pile every time, so it recomputes old cards over and over. With a 10-token prompt and 8 new tokens:

| | Token-forward-passes |
|---|---|
| This loop (re-reads everything each time) | 10 + 11 + 12 + ... + 17 = **108** |
| With a notepad (KV cache) | 10 (prefill) + 7 (one per extra token) = **17** |

**What else it cannot do** (the six questions from the book, as a table):

| The question | Who answers it in a real engine |
|---|---|
| How are many simultaneous orders queued? | The scheduler (`scheduler/prefill.py`) |
| Which orders cook together in this pass? | The scheduler (builds a `Batch`) |
| How many words has each order produced? Is it done? | `Req` (`core.py`) and the scheduler loop |
| Where do old Labels and Gifts live? | The KV cache pool (`kvcache/`) |
| How is space freed when an order ends? | The cache manager (`scheduler/cache.py`) |
| How do tokens become streamed text? | The detokenizer (`tokenizer/detokenize.py`) and the API server |

> **The sentence to remember:** *the model performs neural-network computation; the engine organises that computation.*

### 2.2 Four Core Responsibilities of an Inference Engine

| Responsibility | Kid version | Where it lives in mini-sglang |
|---|---|---|
| **Input and output management** | Words in, words out, streamed politely | `server/` (front desk), `tokenizer/` (card-maker and card-reader), `message/` (the forms they pass) |
| **Request orchestration** | A file for every order: how long, how far along, when to stop, how to clean up | `core.py` (`Req`), `scheduler/prefill.py`, `scheduler/decode.py`, `scheduler/table.py` |
| **Computation scheduling** | Deciding who gets cooked together, and whether this pass is "read the slip" or "write a word" | `scheduler/scheduler.py` |
| **State and memory management** | Keeping the notepads, handing out pantry slots, taking them back | `kvcache/`, `scheduler/cache.py` |

> *The engine decides which requests to compute, when, which tokens to process, and how to preserve the results.*

## 3 Overall Architecture of mini-sglang

### 3.1 From Responsibility Layers to the Process Structure

**Kid version.** There are two different ways to draw the same restaurant:

- the **layers** picture asks *"what job does each part do?"*
- the **processes** picture asks *"which worker, in which room, does it?"*

**Layers** (jobs):

```text
   ┌────────────────────────────────────────────────────────────┐
   │ INTERFACE      how orders get in and dishes get out        │  server/, tokenizer/, message/
   ├────────────────────────────────────────────────────────────┤
   │ ORCHESTRATION  who is waiting, what to cook this pass      │  scheduler/
   ├────────────────────────────────────────────────────────────┤
   │ EXECUTION      the forward pass and picking the next word  │  engine/, models/, layers/, attention/
   ├────────────────────────────────────────────────────────────┤
   │ STATE & MEMORY the history and the notepads                │  kvcache/, core.py (Req), scheduler/cache.py
   └────────────────────────────────────────────────────────────┘
```

**Processes** (rooms), as the code actually starts them (`server/launch.py`) with the default settings and one GPU:

```text
   you (curl / browser / OpenAI client)
        │ HTTP
        ▼
   ┌──────────────────────────────┐                      PROCESS 1: the main process
   │ API server (FastAPI)         │                      front desk, assigns each order a uid
   └───────┬──────────▲───────────┘
           │ ZeroMQ   │ ZeroMQ
           ▼          │
   ┌──────────────────┴───────────┐                      PROCESS 2: "minisgl-detokenizer-0"
   │ tokenizer + detokenizer      │                      (by default ONE worker does both jobs)
   └───────┬──────────▲───────────┘
           │ UserMsg  │ DetokenizeMsg (token ids)
           ▼          │
   ┌──────────────────┴───────────┐                      PROCESS 3: "minisgl-TP0-scheduler"
   │ scheduler + engine (one GPU) │                      one such process PER GPU
   └──────────────────────────────┘
```

Facts read from `launch.py`:

- The API server runs in the **main process**; it asks `start_subprocess()` to start the others when it boots.
- One scheduler process **per GPU** (tensor-parallel rank), named `minisgl-TP{i}-scheduler`. The scheduler and the engine are in the **same** process: the scheduler calls the engine like an ordinary function.
- `--num-tokenizer N` (default 0). With 0, the one `minisgl-detokenizer-0` worker also tokenizes. With N > 0, N extra workers `minisgl-tokenizer-{i}` handle tokenizing and the detokenizer only detokenizes.
- Startup waits for "ready" acknowledgements: from the primary scheduler, the detokenizer, and each extra tokenizer (`N + 2` in total).
- Processes are started with `spawn` (fresh interpreters), not `fork`. This is the safe choice when GPUs are involved.

**The phone lines between rooms** (ZeroMQ sockets, as local files under `/tmp`):

| Line | From → to | Carries |
|---|---|---|
| `minisgl_0` | tokenizer → scheduler | `UserMsg` (token ids + params), `AbortBackendMsg` |
| `minisgl_1` | scheduler → detokenizer (and API server → tokenizer when shared) | `DetokenizeMsg`; `TokenizeMsg`; `AbortMsg` |
| `minisgl_2` | scheduler rank 0 → other ranks | The same requests, copied to every GPU worker |
| `minisgl_3` | detokenizer → API server | `UserReply` (new text pieces) |
| `minisgl_4` | API server → tokenizers (only if `--num-tokenizer > 0`) | `TokenizeMsg` |

Every file name also ends in `.pid=<number>`, so two servers on one machine do not collide. The "who creates, who dials in" detail (`create=True` means **bind**, `False` means **connect**) is why the config has the odd `backend_create_detokenizer_link` flag. In the default shared mode the one tokenizer worker owns its in-tray and everyone dials in. With `--num-tokenizer N` the ownership flips: the front desk owns the tokenizers' in-tray and the scheduler owns the detokenizer's in-tray. (`server/args.py` holds the three flags that decide this.)

**Where the story lies.** The book's layers are *jobs*, not processes. For example the "execution layer" is not a separate process: it is the engine, called as a function inside the scheduler's process.

### 3.2 How a Request Flows Through the Architecture

**Kid version.** Two kinds of things travel through the restaurant:

- **Paperwork** (the *control path*): "order #7 is waiting", "order #7 now has this next word", "cancel order #7".
- **Ingredients** (the *data path*): the actual numbers the chef cooks with: token ids, positions, scores, notepads.

```text
   CONTROL PATH (small messages, who is doing what)         DATA PATH (big tensors, on the GPU)
   ───────────────────────────────────────────              ───────────────────────────────────
   UserMsg / AbortBackendMsg / DetokenizeMsg                token_pool  (token ids per request)
   Req state: waiting → prefill → decode → done             page_table  (KV slot per token)
   which requests are in this batch                         input_ids, positions, out_loc
   handled by: scheduler + managers                         logits, KV cache pool
                                                            handled by: Batch, engine, attention
```

#### The state machine, and where each state really lives

The book says the five states are "an educational abstraction": the source has no enum called `WAITING`. A request's state is implied by **which container it sits in**:

| Book state | In the code |
|---|---|
| `WAITING` | A `PendingReq` in `PrefillManager.pending_list` |
| `PREFILL` | A `Req` (or `ChunkedReq` for a long prompt processed in pieces) inside a `Batch` whose `phase` is `"prefill"` |
| `DECODE` | A `Req` in `DecodeManager.running_reqs` |
| `FINISHED` | The scheduler saw EOS, or `can_decode` became False, and freed the request's row and pages |
| `ABORTED` | An `AbortBackendMsg` arrived; both managers are asked to drop the uid and its resources are freed |

#### The 8 steps, with real names

Order: `"Please explain what the KV Cache is."`, `max_new_tokens = 8`.

| # | Book step | In the code |
|---|---|---|
| 1 | Server receives text and parameters | `server/api_server.py`; the order gets a uid |
| 2 | Tokenizer encodes into `input_ids` | `TokenizeManager.tokenize` (applies the chat template if the input is a list of messages, then `encode`); result is a 1-D int32 CPU tensor |
| 3 | A request object is created and queued | `UserMsg` → `Scheduler._process_one_msg` → `PrefillManager.add_one_req` → a `PendingReq` in `pending_list` |
| 4 | Scheduler builds a prefill batch | `Scheduler._schedule_next_batch`: try a prefill batch first, otherwise a decode batch |
| 5 | Model runner does the forward pass, writes KV | `Engine.forward_batch` |
| 6 | Sampler picks the first token | `Sampler.sample` (inside `forward_batch`) |
| 7 | Decode: one more token per iteration | Each pass: `DecodeManager` supplies the batch; the token goes straight into the GPU token pool for the next pass |
| 8 | Detokenizer returns text | `DetokenizeMsg` → `DetokenizeManager.detokenize` → `UserReply` → API server |

**Tiny numbers: one request's bookkeeping.** Suppose the prompt is 10 tokens, 6 of which are already in the prefix cache (another request did them), and `max_new_tokens = 8`. At creation: `device_len = 10`, `cached_len = 6`, `max_device_len = 18`. Every forward pass the engine calls `complete_one()`: `cached_len = device_len`, then `device_len += 1`.

| Forward pass | Phase | `extend_len` during it | `cached_len` after | `device_len` after | `remain_len` after | Token produced |
|---|---|---|---|---|---|---|
| 1 | prefill | 10 - 6 = **4** | 10 | 11 | 7 | #1 |
| 2 | decode | 1 | 11 | 12 | 6 | #2 |
| 3 | decode | 1 | 12 | 13 | 5 | #3 |
| ... | decode | 1 | ... | ... | ... | ... |
| 8 | decode | 1 | 17 | 18 | **0** | #8 (finished) |

Notice three things:

1. The prefill only *computes* 4 tokens, because 6 were reused. That is the whole point of prefix caching (chapter 8).
2. Eight forward passes produce eight tokens: one prefill and seven decode steps.
3. After the last pass, `cached_len` is 17, not 18. **The 8th token is never run through the model, so its notes are never written**, because nobody will ever need them.

**Where the story lies.** The table describes a request in isolation. In reality the scheduler overlaps work: while the GPU runs pass N, the CPU is already building pass N+1, and `complete_one()` is called before the sampled token's value has reached the CPU. The bookkeeping is one step ahead of the data (chapter 6).

### 3.3 From Architecture Modules to Source Code

**The source tree, with kid glosses:**

```text
python/minisgl/
├── attention/    the "meeting room" kernels (FlashAttention / FlashInfer / TensorRT-LLM variants)
├── benchmark/    stopwatch tools (Part I chapter 5, in code)
├── distributed/  how several GPUs talk (all-reduce, all-gather)
├── engine/       the chef's hands on ONE GPU: model + notepad + CUDA graphs + sampling
├── kernel/       hand-written GPU code (CUDA / Triton)
├── kvcache/      the pantry: KV pools, the radix-tree prefix index
├── layers/       the building blocks of a transformer (linear, norm, RoPE, attention layer, ...)
├── llm/          a tiny Python-library front door (no HTTP): `LLM(...)`
├── message/      the forms passed between processes
├── models/       whole model recipes (Llama, Qwen2/3, Mistral, Qwen3-MoE) + weight loading
├── moe/          mixture-of-experts backends
├── scheduler/    the head waiter
├── server/       the front desk (FastAPI), command-line flags, the launcher
├── tokenizer/    card-maker and card-reader worker
├── utils/        small shared tools (logging, ZeroMQ wrappers, registry, ...)
├── core.py       the four shared forms: SamplingParams, Req, Batch, Context
├── env.py        environment-variable switches (not in the book's tree)
├── shell.py      an interactive chat mode (not in the book's tree)
└── __main__.py   entry point for `python -m minisgl`
```

**Reading by question** (the book's advice): "what does a request look like?" → `core.py`. "How is the next batch chosen?" → `scheduler/`. "Where do K and V live?" → `kvcache/`. "How does a request enter and leave?" → `server/`. Working backwards from a feature: models and weights → `models/`; multi-GPU layers → `layers/` and `distributed/`; attention code → `attention/`; GPU kernels → `kernel/`.

#### The four shared forms

**`SamplingParams`: the "how adventurous?" card.**

| Field | Default | Kid meaning |
|---|---|---|
| `temperature` | 0.0 | 0 = always the single most likely word |
| `top_k` | -1 | Consider only the k likeliest words (-1 = all) |
| `top_p` | 1.0 | Consider the smallest set of words covering p of the probability |
| `ignore_eos` | False | Keep writing past the "I'm done" token (benchmarks) |
| `max_tokens` | 1024 | The longest answer allowed |

`is_greedy` is true when `temperature <= 0` (or `top_k == 1`) and `top_p == 1.0`: then no dice are needed.

**`Req`: one order slip plus progress notes.**

| Field | Kid meaning |
|---|---|
| `input_ids` | The cards so far (prompt plus words appended as they are generated), kept on the **CPU** |
| `table_idx` | This order's numbered peg on the head waiter's wall (a row of `page_table` and `token_pool`) |
| `cached_len` | How many words at the front already have notes in the pantry |
| `output_len` | The most new words this order may still produce |
| `uid` | The order number the front desk gave |
| `sampling_params` | The adventurousness card |
| `cache_handle` | A claim ticket for the shared-prefix entry this order uses |
| (derived) `device_len` | Words that exist from the GPU's point of view |
| (derived) `max_device_len` | Prompt length plus `output_len`: the ceiling |
| (derived) `remain_len` | `max_device_len - device_len` |
| (derived) `extend_len` | `device_len - cached_len`: words the chef must really think about this pass |

The book names "input token IDs, current and maximum sequence length, cached count, request id, sampling params, cache handle". The code has two extras the book leaves out: `table_idx` and `output_len`.

**Current length vs cached length** (the book stresses this): *current* = how many tokens the request contains; *cached* = how many of them already have K and V ready. Their difference, `extend_len`, is the number of tokens the model must process in this pass.

**`Batch`: the tray.**

| Field | Kid meaning |
|---|---|
| `reqs` | The order slips on the tray |
| `phase` | `"prefill"` (fresh slips) or `"decode"` (orders underway); never mixed |
| `input_ids` | The cards to feed in, all slips concatenated (filled by the scheduler) |
| `positions` | Each card's seat number inside its own slip (for RoPE) |
| `out_loc` | For each card, the pantry slot where its new notes must be written |
| `padded_reqs` | The slips plus blank dummies, to match the size a CUDA graph was recorded for |
| `attn_metadata` | Extra paperwork for the attention kernels (filled by the attention backend) |

**`Context`: the kitchen whiteboard.** Holds `page_size`, `page_table`, `attn_backend`, `moe_backend`, `kv_cache` and the current `_batch`. Layers deep inside the model call `get_global_ctx()` to read it, instead of every function receiving all of those as arguments. `with ctx.forward_batch(batch):` hangs the tray on the board for the duration of one forward pass and always clears it afterwards.

```text
   Context  (one per worker process, set once at engine start)
     ├── page_table      row = a request's peg;  column = position;  value = KV slot
     ├── kv_cache        the big pool of K/V slots
     ├── attn_backend    which attention implementation
     ├── moe_backend     (MoE models only)
     └── _batch  ──────▶ Batch
                           ├── reqs ──▶ Req, Req, Req ...
                           │              └── sampling_params + cache_handle
                           └── input_ids, positions, out_loc, attn_metadata
```

**Pause and predict.** A request arrives with a 12-token prompt, 12 tokens of which could be matched in the prefix cache. What does the scheduler do, and why?

<details>
<summary>Answer (try first)</summary>

It matches only the first 11 (`input_ids[: input_len - 1]` in `scheduler/cache.py`). The `Req` has the invariant `cached_len < device_len`, because at least one token must be run through the model to produce the logits for the next word. A fully cached prompt would have nothing to compute.

</details>

## 4 Implementation Roadmap for Part II

### 4.1 Why We Use an Incremental Implementation

**Kid version.** Building the whole restaurant in one go makes it hard to see why each room exists. So each chapter: (1) shows the biggest remaining pain, (2) adds exactly one fix, (3) measures that the fix helped.

| Chapter | The pain in the current version | The fix | How you know it worked |
|---|---|---|---|
| 2 | We don't know how an order flows through the real source | Trace one request | Draw its path |
| 3 | No working word-by-word generation yet | Forward pass, loop, sampling | Output is right |
| 4 | Every step recomputes old words | KV cache | Less compute, lower TPOT |
| 5 | Only callable from Python | HTTP, concurrency, streaming | End-to-end latency |
| 6 | One order at a time, or fixed groups | Continuous batching and the scheduler | Throughput, TTFT |
| 7 | Cache space is wasted and fragmented | Paged KV cache | Memory use, concurrency |
| 8 | Common beginnings get recomputed | RadixAttention (prefix cache) | Hit rate, TTFT |
| 9 | One GPU is too small or too slow | Multi-process and tensor parallelism | Multi-GPU scaling |
| 10 | Still one word per full-model pass | Speculative decoding | Acceptance rate, speed |

### 4.2 Moving from One Problem to the Next

```text
   understand the flow → make generation correct → repeated work? → KV cache
        → can't serve users? → HTTP + streaming → orders block each other? → continuous batching
        → wasted/fragmented pantry? → paged KV → shared beginnings recomputed? → prefix cache
        → one GPU not enough? → tensor parallelism → still one word per pass? → speculative decoding
```

Every new mechanism should answer four questions: (1) What is the current bottleneck? (2) Which path (control or data) does it change? (3) What objects and interfaces must be added? (4) Which metric shows it helped?

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** The chef (model) does the cooking; the restaurant (engine) does everything else: taking orders, deciding order of cooking, keeping notepads, serving words. An order's life is waiting, reading, writing, done. Three forms tie it together: **Req** (one order), **Batch** (a tray), **Context** (the whiteboard). Part II adds capabilities in the order the problems appear.

### 5.2 Exercises

**1. Why is `model(input_ids)` not yet an inference engine?**

<details><summary>Hint</summary>

It computes scores for one sequence. It does not queue or batch requests, keep per-request state, reuse earlier work, manage memory, stream text, handle cancels, or expose a service.

</details>

**2. Responsibilities of the model vs the engine (three each).**

<details><summary>Hint</summary>

Model: embedding, attention, FFN, normalisation, output scores. Engine: tokenizing, queueing and scheduling, KV-cache allocation and release, sampling, stopping conditions, streaming, cancellation.

</details>

**3. What do `Req`, `Batch`, `Context` represent, and how are they related?**

<details><summary>Hint</summary>

One request's state; the group of requests run in one pass plus the inputs prepared for it; the per-process shared environment (cache pool, attention backend, page table, current batch). The context points at the current batch; the batch holds the requests.

</details>

**4. States between arrival and response; what leads to FINISHED or ABORTED?**

<details><summary>Hint</summary>

Waiting → prefill → decode → finished. Finished on EOS (unless `ignore_eos`) or reaching the length limit. Aborted on cancel, disconnect, timeout or error.

</details>

**5. Why separate the control path from the data path? Give an example of each.**

<details><summary>Hint</summary>

Scheduling decisions (which requests, which phase) are small and CPU-side; model inputs and outputs are big and GPU-side. Separating them lets you reason about and optimise each independently. Control: "request 7 aborted". Data: `positions`, `out_loc`, logits, KV.

</details>

**6. Why implement single-request generation before KV cache, HTTP and continuous batching?**

<details><summary>Hint</summary>

Correctness first: you need a trusted baseline output to compare every later optimisation against. If you add batching and caching at once, a wrong output could come from any of five places.

</details>

**7. What do `engine`, `scheduler`, `kvcache`, `server`, `tokenizer` do?**

<details><summary>Hint</summary>

See the source-tree table in 3.3.

</details>

**Extra drills (not in the book).**

8. In the tiny-numbers table, why is `extend_len` 4 for prefill but 1 for every decode pass?
9. With `--num-tokenizer 0`, which process receives `DetokenizeMsg` and also `TokenizeMsg`? (The single detokenizer worker; both arrive on `minisgl_1`.)
10. Which object would you look at to find out which GPU memory slot holds the 5th token of request #3? (`page_table[req.table_idx, 4]`.)

## Where to read the code next

All of these now carry ELI5 comments on the `eli5-comments` branch of mini-sglang: `core.py`, `scheduler/table.py`, `scheduler/utils.py`, `scheduler/config.py`, `scheduler/decode.py`. The next chapter walks the whole path through `server/`, `tokenizer/`, `message/`, `scheduler/` and `engine/`.
