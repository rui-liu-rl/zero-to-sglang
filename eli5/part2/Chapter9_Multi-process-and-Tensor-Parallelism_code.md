# Part II Chapter 9 Multi-process and Tensor Parallelism (Code Walkthrough): ELI5 companion

Original: [Chapter9_Multi-process-and-Tensor-Parallelism_code.md](../../course-material/eng/part2/Chapter9_Multi-process-and-Tensor-Parallelism_code.md). Same headings, same numbers. Concept half: [Chapter9_Multi-process-and-Tensor-Parallelism.md](./Chapter9_Multi-process-and-Tensor-Parallelism.md). Cast and picture legend: [../README.md](../README.md).

Files named here carry ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang` (paths relative to `python/minisgl/`): `server/launch.py`, `server/args.py`, `engine/engine.py`, `distributed/{info,impl}.py`, `scheduler/io.py`, `models/weight.py`, `layers/{linear,embedding}.py`, `scheduler/decode.py`.

Run the maths yourself: `python3 -I eli5/tools/ch9_tp_toy.py`.

## The one-minute version

**Kid version.** Four things happen in the life of a multi-GPU server, in this order:

```text
   ① launch.py       hire one scheduler+engine process per GPU, plus a detokenizer; wait for the "ready" flag
   ② engine.py       each process: claim cuda:{rank}, phone the others (Gloo + PyNCCL), build the model skeleton
   ③ weight.py       each process: read the recipe book, keep ONLY its slice of every matrix, glue q/k/v and gate/up
   ④ io.py           rank 0 reads the mail and announces every slip to the others (count over Gloo, slips over ZMQ)
   ...then forever:   same slips → same batches → each GPU cooks its slice → all_reduce / all_gather at the seams
```

## 1 Learning Objectives

| Goal | Where |
|---|---|
| Roles of ZMQ broadcast, Gloo control, NCCL tensors, and how they cooperate | 3.2, 3.4 |
| Weight shapes, input/output shapes, communication points of column- and row-parallel layers | 3.3 |
| Sharding rules for QKV, vocabulary, KV cache; constraints | 3.3, concept 3.2 |
| Trace a request while all ranks keep the same order and state | 3.4 |

## 2 Locating the Source Code

| File | Purpose | Kid gloss |
|---|---|---|
| `server/launch.py` | Parse TP etc.; start the API server, schedulers, tokenizer and detokenizer processes; wait for flags | The manager who opens the restaurant |
| `distributed/info.py` | Current process's rank and world size; `get_tp_info()` | A name badge: "I am chef 1 of 4" |
| `distributed/impl.py` | A common interface for all-reduce and all-gather on top of torch.distributed or PyNCCL | The kitchen's intercom |
| `engine/engine.py` | Bind `cuda:{rank}`, start communication, load/shard weights, allocate KV cache, forward and sample | The chef's station |
| `scheduler/io.py` | Rank 0 talks to tokenizer/detokenizer, broadcasts requests, synchronises counts | The mailbox and loudspeaker |
| `scheduler/scheduler.py` | The scheduling loop | Head waiter |
| `scheduler/decode.py` | Sorts running requests by `uid` so every rank lists them in the same order | The seating order |
| `kernel/pynccl.py` | Rank 0 makes the NCCL unique id and broadcasts it; initialises PyNCCL | Dialling the phone |
| `kernel/csrc/src/pynccl.cu` | Calls NCCL through TVM-FFI to enqueue all-reduce / all-gather on a CUDA stream | The wire |
| `models/weight.py` | Cut checkpoint tensors by name into column/row/vocab slices, merge q/k/v and gate/up | The recipe-book reader |

## 3 From Startup to Output

### 3.1 Creating Processes

```bash
python -m minisgl --model "Qwen/Qwen3-32B" --tp-size 2
```

(The book's sentence says "three RTX 5090 GPUs" but the command uses two; see `NOTES-open-questions.md`, D9. Both flag spellings, `--tp-size` and `--tensor-parallel-size`, exist; the default is one GPU.)

```python
kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
```

*Line by line:* `parse_args` turns the command line into a settings object. `tp_info = DistributedInfo(rank=0, size=tp_size)` is the *parent's* placeholder: rank 0 of `tp_size`. Each child gets its own.

```python
def start_subprocess() -> None:
    mp.set_start_method("spawn", force=True)
    world_size = server_args.tp_info.size
    ack_queue: mp.Queue[str] = mp.Queue()

    for i in range(world_size):
        new_args = replace(server_args, tp_info=DistributedInfo(i, world_size))
        mp.Process(
            target=_run_scheduler,
            args=(new_args, ack_queue),
            daemon=False,
            name=f"minisgl-TP{i}-scheduler",
        ).start()
    ...
    for _ in range(num_tokenizers + 2):
        logger.info(ack_queue.get())
```

| Line | Kid words |
|---|---|
| `mp.set_start_method("spawn", force=True)` | Each child starts as a **brand-new Python interpreter**. Copying (forking) a process that has touched CUDA is unsafe, so "spawn" is the standard choice with GPUs |
| `world_size = tp_info.size` | How many GPUs |
| `ack_queue = mp.Queue()` | The "ready flag" mailbox every hire drops a note into |
| `for i in range(world_size)` | One iteration per GPU |
| `replace(server_args, tp_info=DistributedInfo(i, world_size))` | A copy of the settings with *this* child's rank: the name badge |
| `mp.Process(target=_run_scheduler, ..., daemon=False, name=...)` | Start the child. `daemon=False`: not killed automatically if the parent exits normally, so shutdown has to be explicit |
| `for _ in range(num_tokenizers + 2): ack_queue.get()` | Wait for the flags: **one** from the rank-0 scheduler (only rank 0 raises it), **one** from the detokenizer, plus one per extra tokenizer |

The processes on the way (default `--num-tokenizer 0`): `minisgl-TP0-scheduler`, ..., `minisgl-TP{n-1}-scheduler`, and **one** `minisgl-detokenizer-0`. With zero tokenizers that single worker reads from the *detokenizer* address, which both the front desk (`TokenizeMsg`) and the schedulers (`DetokenizeMsg`) send to, so it does both jobs. (Chapter 2 companion draws the sockets.)

**What each scheduler child does** (`_run_scheduler`):

```python
scheduler = Scheduler(args)          # slow: loads weights, KV pantry, CUDA graphs
scheduler.sync_all_ranks()           # barrier: everybody finished loading
if args.tp_info.is_primary():
    ack_queue.put("Scheduler is ready")
scheduler.run_forever()
```

*Line by line:* construct (slow), wait for **all** ranks at a barrier, rank 0 raises the flag, loop forever. The barrier guarantees that "ready" really means "every GPU is ready".

```python
def run_api_server(config, start_backend, run_shell) -> None:
    ...
    _GLOBAL_STATE = FrontendManager(config=config, recv_tokenizer=..., send_tokenizer=...)
    start_backend()                  # <- start_subprocess above
    logger.info(f"API server is ready to serve on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
```

The front desk first builds its own phone lines (the in-tray it owns), **then** calls `start_backend()` (hires everyone and waits for flags), and only then calls `uvicorn.run`. Model and communication start-up happens in the window between "processes created" and "HTTP open".

### 3.2 Initializing the Engine and Communication

```python
self.engine = Engine(config)                          # in Scheduler.__init__
...
set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)   # in Engine.__init__, FIRST
self.device = torch.device(f"cuda:{config.tp_info.rank}")
torch.cuda.set_device(self.device)
self.stream = torch.cuda.Stream()
torch.cuda.set_stream(self.stream)

self.tp_cpu_group = self._init_communication(config)

with torch.device("meta"), torch_dtype(config.dtype):
    self.model = create_model(config.model_config)
self.model.load_state_dict(self._load_weight_state_dict(config))
```

| Line | Kid words |
|---|---|
| `set_tp_info(rank, size)` | Write the name badge into a global **before** any layer is built: each layer calls `get_tp_info()` to decide how big its local slice is |
| `device = cuda:{rank}` | Rank *i* uses GPU *i* (the index refers to the **visible** devices) |
| `torch.cuda.Stream()` and `set_stream` | The chef's private lane on the GPU |
| `_init_communication(config)` | Set up the phones (below) |
| `with torch.device("meta") ...: create_model(...)` | Build the **empty skeleton** of the model: shapes and types only, no memory |
| `load_state_dict(...)` | Put the real, already-sliced tensors into the skeleton (swap in, no copy) |

The order matters: badge first, skeleton second (it reads the badge), weights third.

**Phones.** `_init_communication` has two paths:

```python
if config.tp_info.size == 1 or config.use_pynccl:
    torch.distributed.init_process_group(backend="gloo", rank=..., world_size=..., ...)
    tp_cpu_group = torch.distributed.group.WORLD
    max_bytes = config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
    enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
else:
    torch.distributed.init_process_group(backend="nccl", ...)
    tp_cpu_group = torch.distributed.new_group(backend="gloo")
```

| Configuration | PyTorch's own group | GPU tensor traffic | CPU control traffic |
|---|---|---|---|
| `tp_size = 1` | Gloo | none | Gloo |
| `tp_size > 1`, default | Gloo | a **separate PyNCCL communicator** | Gloo (the world group) |
| `tp_size > 1`, `--disable-pynccl` | NCCL | torch.distributed NCCL | a second, separate Gloo group |

Notes: (1) on one GPU a Gloo group still exists, for jobs like `_sync_get_memory`; (2) `max_bytes` is the largest tensor one all_reduce will ever carry: the longest forward pass (in tokens) times the hidden size times bytes per number; PyNCCL sizes its staging buffer from it.

**The intercom.**

```python
class DistributedCommunicator:
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    def all_reduce(self, x): return self.plugins[-1].all_reduce(x)
    def all_gather(self, x): return self.plugins[-1].all_gather(x)
```

A list of back-ends; the **last** one is used. `enable_pynccl_distributed` appends `PyNCCLDistributedImpl`, so from then on every `all_reduce` goes through PyNCCL. Layers only call `DistributedCommunicator().all_reduce(x)` and never know which wire is underneath.

**Consistency starts here too.** Before building the KV pool, every rank must agree on `num_pages` (the scheduling code is the same on every rank). `_sync_get_memory` measures free memory, does an **all_reduce with MIN** over a two-element tensor `[free, -free]` (a trick to get both the minimum and the maximum in one call) and raises if they differ by more than 2 GiB. The page count is then computed from the shared number, so all ranks get the same `num_pages`. (Hypothesis **H4** in the notes: the code uses the *maximum* free memory, not the minimum.)

### 3.3 Loading Weights and Model State

Each rank reads the **same checkpoint files** and keeps its own slice. The cutting rules, from `models/weight.py` (the checkpoint stores weights as `[out, in]`):

| Name contains | Cut | Which layer type |
|---|---|---|
| `.q_proj` `.k_proj` `.v_proj` `.gate_proj` `.up_proj` | dim 0 (**output**) | column parallel |
| `.o_proj` `.down_proj` | dim 1 (**input**) | row parallel |
| `embed_tokens`, `lm_head` | rows (vocabulary) in `ceil` pieces | vocab parallel |
| anything else (norm weights, ...) | not cut | replicated |

```python
def _shard_tensor(key, value, r, n, num_kv_heads):
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        ...  # rows [r * ceil(V/n) : min((r+1) * ceil(V/n), V)]
    else:
        return value
```

| Line | Kid words |
|---|---|
| `any(key.count(sub) ...)` | Does the weight's name contain one of the listed fragments? |
| KV special case (`num_kv_heads < n`) | Fewer KV heads than GPUs: GPU `r` takes **one whole head**, number `r * num_kv_heads // n`; several GPUs hold the same head |
| `value.chunk(n, dim=0)[r]` | Cut dimension 0 into `n` equal pieces; keep piece `r` |
| `.clone()` | A slice is only a *view* of the big tensor; cloning makes an independent tensor so the big one can be freed |
| `chunk(n, dim=1)` | The same for the input dimension |
| vocabulary branch | `ceil`-sized pieces; the last rank may get fewer real rows |

**Tiny numbers (my toy, Part 10b).** 8 KV heads and 16 GPUs: GPU `r` gets head `r * 8 // 16 = r // 2`, i.e. ranks `0,1 → head 0`, `2,3 → head 1`, ..., `14,15 → head 7`. Every head sits on two GPUs.

**Shard first, then glue.** The loader keeps a glue-waiting table; once `q`, `k`, `v` slices (or `gate`, `up`) have all arrived, it concatenates them: `torch.cat([q_slice, k_slice, v_slice], dim=0)` and yields `...qkv_proj`. MoE experts are stacked into one `[E, ...]` tensor in expert order.

**Tiny numbers (my toy, Part 10).** Four query rows `q0..q3`, two key rows `k0 k1`, two value rows `v0 v1`, two GPUs:

| | shard then merge (the loader) | merge then split |
|---|---|---|
| rank 0 | `q0 q1 k0 v0` | `q0 q1 q2 q3` |
| rank 1 | `q2 q3 k1 v1` | `k0 k1 v0 v1` |

Merging first and splitting evenly would hand rank 0 only queries and rank 1 only keys and values: neither could run its heads. Shard-then-merge gives each rank **a self-contained group of heads** (its queries plus the matching keys and values).

**Streaming.** `safetensors.safe_open(file, framework="pt", device=str(device))` opens a checkpoint file and reads one tensor at a time directly onto this rank's GPU; the full tensor `raw` is dropped (`del raw`) the moment its slice has been cloned. So at any moment a rank holds one full unsharded tensor plus a small glue buffer. (The book and a docstring call this a *CPU*-memory saving; the code reads straight to the GPU, so it is GPU memory; logged as D10.)

**The vocabulary layers at run time** (`layers/embedding.py`):

```python
class VocabParallelEmbedding(BaseOP):
    def forward(self, x):
        y = indexing(weights=self.weight, indices=x,
                     vocab_range=self.vocab_range if self.tp_size > 1 else None)
        return self._comm.all_reduce(y) if self.tp_size > 1 else y
```

| Line | Kid words |
|---|---|
| `indexing(...)` | A custom gather: row `x[i] - start` if the id is in my range, **zeros** otherwise |
| `vocab_range` | `(first row I own, how many rows I own)` |
| `all_reduce(y)` | Exactly one rank was non-zero per token, so the sum is the right row |

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(bs)].contiguous()   # only the LAST position of each request
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        output = self._comm.all_gather(logits)
        ...                                         # reshape to [bs, tp * slice]
        return output[:, : self.num_embeddings]     # cut the padding columns
```

*Line by line:* keep only the last position (the big 3,000-times saving from chapter 3), compute this rank's slice of the scores, all_gather the slices, rearrange them to `[batch, full vocabulary]`, and cut padding columns off. **Tiny numbers (toy Part 5):** vocabulary 10, 4 GPUs: `ceil(10/4) = 3` rows each, so 12 columns after the gather and 2 padding columns removed.

**Tied embeddings.** If the model "ties" the LM head to the embedding, the head borrows the embedding's weights (`tied_embedding`); it then ignores any `lm_head.weight` in the checkpoint.

**The parallel linear layers** (`layers/linear.py`, covered in the chapter 3 companion): `LinearQKVMerged` and `LinearColParallelMerged` (column parallel; sizes divided with `div_even`), `LinearOProj` and `LinearRowParallel` (row parallel; end with `all_reduce` when `tp_size > 1`). `div_even(num_kv_heads, tp, allow_replicate=True)` returns `1` if there are more GPUs than KV heads (and the GPU count is a multiple of the head count); otherwise it insists on exact divisibility.

### 3.4 Receiving and Broadcasting Requests

```python
def _recv_msg_multi_rank0(self, blocking=False):
    pending_msgs = []
    if blocking:
        self.run_when_idle()
        raw = self._recv_from_tokenizer.get_raw()
        self._send_into_ranks.put_raw(raw)
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))

    pending_raw_msgs = []
    while not self._recv_from_tokenizer.empty():
        pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

    src_tensor = torch.tensor(len(pending_raw_msgs))
    self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

    for raw in pending_raw_msgs:
        self._send_into_ranks.put_raw(raw)
        pending_msgs.append(self._recv_from_tokenizer.decode(raw))
    return pending_msgs

def _recv_msg_multi_rank1(self, blocking=False):
    pending_msgs = []
    if blocking:
        self.run_when_idle()
        pending_msgs.append(self._recv_from_rank0.get())

    dst_tensor = torch.tensor(-1)
    self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
    for _ in range(int(dst_tensor.item())):
        pending_msgs.append(self._recv_from_rank0.get())
    return pending_msgs
```

*Line by line, rank 0:*

| Line | Kid words |
|---|---|
| `if blocking: ... get_raw()` | Nothing to do anywhere, so sleep until the first slip arrives; announce it on the loudspeaker (`put_raw`), then decode it for myself. `raw` = the exact bytes: forwarding them avoids decode-then-re-encode |
| `while not ...empty(): get_raw()` | Scoop up everything *already* waiting, without waiting for more |
| `src_tensor = torch.tensor(len(...))` + `broadcast(..., root=0)` | Tell everyone **how many** slips this round (Gloo, a reliable collective) |
| `for raw in pending_raw_msgs: put_raw(raw); decode` | Announce each slip (ZMQ publish) and process it locally |

*Rank 1 and up:* mirror image. If blocking, wait for the first announcement. Then receive the count (`-1` is just a placeholder that is overwritten) and read exactly that many announcements from the subscription socket.

**Why a count, and why the same blocking flag?** The announcements travel on a different path from the count, and a rank that read too few or too many slips would drift out of step. The count is the safety net. The `blocking` flag comes from `not (prefill.runnable or decode.runnable ...)`, which is computed from state that is **identical on all ranks** if they have stayed in step, so all ranks block or none do.

**The principle (from the book).** *Same request set + same order + same token/length state*: each rank then runs the same scheduling code and updates its **own** local queues and caches. The full batch object is never broadcast. Your picture should be: eight chefs reading the same slips and independently arriving at the same tray. `DecodeManager.schedule_next_batch` sorts by `uid` for exactly this reason (a Python set has no guaranteed order). **This is also the prerequisite for the radix prefix cache to behave identically on all ranks** (same inserts, same evictions, same `num_pages`).

**Sending replies.** `_reply_tokenizer_rank0` sends the new tokens to the detokenizer (a single message as is; several as a `BatchTokenizerMsg`). `_reply_tokenizer_rank1` is a no-op: every rank computed the same tokens; only rank 0 reports.

**Pause and predict**

1. Which collectives happen during one decode step of a 28-layer model with `tp = 2`, ignoring the barrier?

<details><summary>Answer</summary>

One all_reduce for the embedding, two per layer (after `o_proj` and `down_proj`) = 57, and one all_gather for the LM head. Control traffic (Gloo count, ZMQ announcements) happens per scheduler round, not per layer.

</details>

2. Where could rank 1 and rank 0 disagree if only the slips were broadcast and not the count?

<details><summary>Answer</summary>

Rank 1 would not know how many announcements to read before scheduling. It could read too few (and run a request one iteration late) or block waiting for a slip rank 0 has not sent yet.

</details>

## 4 Mapping to SGLang

This table repeats the book's claims about the full engine's file layout. I have not read those files for these notes, so treat it as unchecked:

| Function | mini-sglang | Full SGLang (per the book) |
|---|---|---|
| Multi-process start-up | `server/launch.py` | `entrypoints/engine.py` |
| Scheduling and model execution | `scheduler/scheduler.py`, `engine/engine.py` | `managers/scheduler.py`, `managers/tp_worker.py`, `model_executor/model_runner` |
| Request reception and coordination | `scheduler/io.py`, `scheduler/scheduler.py` | `managers/scheduler.py` |
| Communication groups and tensor communication | `distributed/impl.py`, `kernel/pynccl.py` | `distributed/parallel_state.py` |
| Column/Row/QKV parallel | `layers/linear.py`, `models/weight.py` | `layers/linear.py` |
| Vocabulary sharding and loading | `layers/embedding.py`, `models/weight.py` | `layers/vocab_parallel_embedding.py` |

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** One process per GPU, each with its badge; they phone each other twice over (Gloo for tiny control messages and barriers, NCCL for tensors). Each loads the same book but keeps its slice, gluing q/k/v after cutting. Rank 0 alone reads the mail and tells the rest how many slips and which; everybody schedules the same way, so everybody builds the same tray.

### 5.2 Exercises

**1. Each rank starts a Scheduler. What purpose does that serve?**

<details><summary>Answer</summary>

Each GPU needs its own process to hold its engine (slice of the weights, own KV pool slice, CUDA graphs) and to run the scheduling loop on identical inputs. The schedulers are the ranks.

</details>

**2. How are weights loaded, and how do the ranks coordinate to process the same batch?**

<details><summary>Answer</summary>

Single GPU: stream the checkpoint, merge q/k/v and gate/up, load the lot. Multi-GPU: every rank streams the same files and keeps only its slice (`_shard_tensor`), then merges; rank 0 reads requests and announces the count (Gloo) and the bytes (ZMQ) to the others, so all ranks run the same scheduling logic on the same inputs and agree on `num_pages` through the min/max all-reduce.

</details>

**3. Why stream weights, and why yield only after merging/stacking?**

<details><summary>Answer</summary>

Streaming keeps the transient memory to about one full tensor plus the glue buffer instead of the whole checkpoint (and, in this code, that memory is on the GPU). The runtime layers hold fused tensors (`qkv_proj`, `gate_up_proj`, stacked experts), so the loader has to hand over final shapes; a half-built fused tensor cannot be loaded.

</details>

**Extra drills (not in the book).**

4. `tp = 4`, vocabulary 151,936, which rows does rank 3 own? (`ceil = 37,984`; rows `113,952` to `151,935`, which is exactly the last 37,984 rows.)
5. A model has 40 query heads and 8 KV heads. Is `tp = 16` valid? (No: 40 query heads do not divide by 16.)
6. Why does `ParallelLMHead` cut the gathered output to `[:, :num_embeddings]`? (The last rank's slice may contain padding rows that were never filled; they must not appear as vocabulary words.)
