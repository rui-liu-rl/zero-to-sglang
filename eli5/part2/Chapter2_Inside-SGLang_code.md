# Part II Chapter 2 Inside SGLang: The Path of a Request (Code Walkthrough): ELI5 companion

Original: [Chapter2_Inside-SGLang_code.md](../../course-material/eng/part2/Chapter2_Inside-SGLang_code.md). Same headings, same numbers. The concept half is [Chapter2_Inside-SGLang.md](./Chapter2_Inside-SGLang.md). Cast and picture legend: [../README.md](../README.md).

Every source file named below has ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang`. Paths are relative to `python/minisgl/`. Open the file next to this page.

## The one-minute version

**Kid version.** Last time we drew the five desks. Now we walk to each desk, read its notebook, and see exactly **which function** handles the letter and **which message** it hands on. The book's code boxes are shortened; this page also tells you what each box **left out**.

```text
   curl ──▶ api_server.py:v1_completions ──TokenizeMsg──▶ tokenizer/server.py:tokenize_worker
                                                              │ UserMsg
                                                              ▼
   curl ◀── api_server.py:listen ◀──UserReply── tokenize_worker ◀──DetokenizeMsg── scheduler/scheduler.py
                                                                                    │ calls
                                                                                    ▼
                                                                          engine/engine.py:forward_batch
```

## 1 Learning Objectives

| Goal | Where this page covers it |
|---|---|
| Start a server, send a streaming request, inspect the processes | Sections 2 and 3 |
| Find the files for API server, tokenizer, scheduler, engine, detokenizer | Section 4 (one subsection each) |
| Explain where a request waits, where it is computed, how results return | 4.2 (waits in queue), 4.4-4.5 (computed), 4.6-4.7 (returns) |
| Find the cleanup code for normal finish and cancel | 4.2 (abort), 4.4 (free on finish) |

## 2 Prepare and Send a Request

**Start the server.**

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B"
```

*Line by line:* `python -m minisgl` runs the package's `__main__.py`, which just calls `launch_server()`. `--model` (alias of `--model-path`) is a Hugging Face repo id or a local folder. Defaults you will meet: **host `127.0.0.1`, port `1919`** (not 30000 as in the SGLang lesson), one GPU (`--tp-size 1`), radix prefix cache, CUDA graphs enabled. (This needs a Linux machine with an NVIDIA GPU; your Mac cannot run it, see Part 0.2.)

**Send a streaming request.**

```bash
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is KV Cache?"}],
    "max_tokens": 128,
    "stream": true
  }'
```

*Line by line:* `curl URL` is a POST because of `-d`. `-H` labels the body as JSON. In the JSON: `model` is required by the request schema but **not used to pick a model** (the server serves the one it loaded); `messages` is the chat; `max_tokens: 128` limits the answer; `"stream": true` asks for SSE chunks. You will see lines like `data: {...}` and finally `data: [DONE]`.

**Defaults to know** (from `OpenAICompletionRequest` in `api_server.py`): `max_tokens = 16` if you omit it, `temperature = 1.0` (the OpenAI convention, even though mini-sglang's own `SamplingParams` default is 0.0), `top_k = -1`, `top_p = 1.0`.

## 3 Inspect Processes and Find the Source

```bash
ps aux | grep '[m]inisgl'
```

*Line by line:* `ps aux` lists every process. `grep '[m]inisgl'` keeps lines containing "minisgl". The **brackets are a trick**: the pattern `[m]inisgl` matches the text `minisgl` but the grep command's own command line contains `[m]inisgl` (with brackets), which does *not* match, so grep doesn't list itself.

**What you should see, and a drift note.** The book's table lists these names:

| Name | Responsibility | Is it there **by default**? |
|---|---|---|
| `minisgl-TP0-scheduler` | Scheduler (calls the engine inside) | Yes, one per GPU (`TP0`, `TP1`, ...) |
| `minisgl-detokenizer-0` | Reads token ids, makes text | Yes. **By default it also tokenizes** |
| `minisgl-tokenizer-0` | Text to token ids | **No**, only if you pass `--num-tokenizer 1` or more |
| main process | API server | Yes |

The book's table lists `minisgl-tokenizer-0` as if it always exists. In the source, `--num-tokenizer` defaults to **0**, which means *"the tokenizer is shared with the detokenizer"*, so a default launch has **no** `minisgl-tokenizer-N` process. (The book's own section 4.1 later says both start with the same `tokenize_worker` function, which is consistent.) Try `--num-tokenizer 4` and you get four `minisgl-tokenizer-0..3` plus the detokenizer.

**Finding the message types:**

```bash
rg -n 'TokenizeMsg|UserMsg|DetokenizeMsg|UserReply|AbortMsg|AbortBackendMsg' python/minisgl
```

*Line by line:* `rg` is ripgrep, a fast recursive search; `-n` prints line numbers; the quoted pattern is an alternation (`A|B|C`) of the six message class names. It shows which file creates or consumes each one.

The six messages, who sends them, who reads them:

| Message | Fields | Sent by → read by |
|---|---|---|
| `TokenizeMsg` | `uid`, `text` (string **or** list of chat messages), `sampling_params` | API server → tokenizer |
| `AbortMsg` | `uid` | API server → tokenizer |
| `UserMsg` | `uid`, `input_ids` (CPU 1-D int32), `sampling_params` | tokenizer → scheduler |
| `AbortBackendMsg` | `uid` | tokenizer → scheduler |
| `DetokenizeMsg` | `uid`, `next_token` (an int), `finished` | scheduler → detokenizer |
| `UserReply` | `uid`, `incremental_output` (text), `finished` | detokenizer → API server |

Each family has a `Batch...Msg` wrapper (a list of messages sent as one packet), and `message/utils.py` turns any of them into a plain dictionary for `msgpack`: dataclasses become `{"__type__": "ClassName", field: value ...}`, and a tensor becomes `{"__type__": "Tensor", "buffer": raw bytes, "dtype": ...}`. **Only 1-D tensors can be sent** (there's an assert).

## 4 Follow the Request Through the Source

The book's snippets are "shortened and simplified". Below, for each: the snippet explained, then **what the real code does that it leaves out**.

### 4.1 Starting the processes: `server/launch.py`

```python
# server/launch.py (outline; specific args / kwargs omitted)
for i in range(world_size):
    mp.Process(target=_run_scheduler, args=..., name=f"minisgl-TP{i}-scheduler").start()
mp.Process(target=tokenize_worker, kwargs=..., name="minisgl-detokenizer-0").start()
for i in range(num_tokenizers):
    mp.Process(target=tokenize_worker, kwargs=..., name=f"minisgl-tokenizer-{i}").start()

run_api_server(server_args, start_subprocess, run_shell=run_shell)
```

*Line by line:*

| Line | Kid words |
|---|---|
| `for i in range(world_size): ... _run_scheduler ...` | Hire one head-waiter-plus-chef per GPU, each told which GPU is theirs (`TP{i}`) |
| `mp.Process(target=tokenize_worker, ... "minisgl-detokenizer-0")` | Hire the card-reader (who, by default, is also the card-maker) |
| `for i in range(num_tokenizers): ...` | Hire extra card-makers if you asked for them (default: none) |
| `run_api_server(...)` | The front desk opens in the **main** process, and is told how to start the others |

**Left out of the book's box:**

- The real flow starts with `parse_args(sys.argv[1:])` (`server/args.py`) to turn command-line flags into a `ServerArgs`. Then `launch_server` defines an inner `start_subprocess()` function and passes it to `run_api_server`, which calls it *after* creating its own ZeroMQ queues but *before* starting uvicorn.
- `mp.set_start_method("spawn", force=True)`: workers start as **fresh Python interpreters** instead of copies of the parent. That is the safe choice with CUDA.
- Each scheduler gets its own copy of the settings with a different rank: `replace(server_args, tp_info=DistributedInfo(i, world_size))`.
- **Readiness.** `.start()` returns immediately but loading a model takes a while, so each worker puts a "ready" string on a shared `ack_queue`; `start_subprocess` reads exactly `num_tokenizers + 2` of them (the primary scheduler, the detokenizer, each extra tokenizer) before the server begins accepting requests. The scheduler also calls `sync_all_ranks()` first so all GPUs are ready together.
- Inside `_run_scheduler`: `Scheduler(args)` builds the engine (loads weights, allocates the KV pool, captures CUDA graphs), then `scheduler.run_forever()`. Ctrl+C is caught as `KeyboardInterrupt` and leads to `scheduler.shutdown()`.

### 4.2 The frontend: `server/api_server.py`

```python
# server/api_server.py (excerpt)
@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    uid = state.new_user()                     # Assign a request ID
    await state.send_one(TokenizeMsg(uid=uid, text=prompt, sampling_params=...))
    if req.stream:
        return StreamingResponse(...)          # Return a response that streams text
```

*Line by line:*

| Line | Kid words |
|---|---|
| `@app.post("/v1/chat/completions")` | "Orders for the chat window come through this door" |
| `async def` | The waiter can serve others while waiting |
| `req: OpenAICompletionRequest` | FastAPI (via pydantic) checks the JSON against a schema before our code runs |
| `state = get_global_state()` | Fetch the one shared `FrontendManager` (the desk's ledger) |
| `uid = state.new_user()` | Write a new ticket number and prepare its pile (`ack_map`) and light (`event_map`) |
| `await state.send_one(TokenizeMsg(...))` | Hand the slip to the card-maker (async ZeroMQ push) |
| `return StreamingResponse(...)` | Keep the connection open and stream pieces as they appear |

**Left out:** (1) if the JSON has `messages`, the prompt is a *list of dicts* (`[msg.model_dump() for msg in req.messages]`), otherwise it is the raw `prompt` string; (2) `SamplingParams` is built from `ignore_eos`, `max_tokens`, `temperature`, `top_k`, `top_p`; the other fields in the schema (`n`, `stop`, `presence_penalty`, `frequency_penalty`) are accepted but **ignored** (there's a TODO); (3) when `stream` is false, the handler loops `async for ack in state.wait_for_ack(uid)` collecting text and returns one JSON object. In that JSON `usage` token counts are zeros and `finish_reason` is always `"stop"`.

```python
# server/api_server.py (excerpt)
async def listen(self):
    while True:
        msg = await self.recv_tokenizer.get()   # Receive results
        for msg in _unwrap_msg(msg):
            self.ack_map[msg.uid].append(msg)   # Save to this request's result list
            self.event_map[msg.uid].set()       # Wake the coroutine waiting for it
```

*Line by line:* `listen` is **one long-running background task** for the whole server: the **mail sorter**. `await ...get()` waits for the next packet from the card-reader. `_unwrap_msg` splits a batch packet into single `UserReply`s. Each reply is dropped in its ticket's pile and the ticket's light is switched on.

**Left out:** the real loop first checks `if msg.uid not in self.ack_map: continue`, so replies for a ticket that was already cancelled are silently dropped. `listen` itself is started lazily by `_create_listener_once()` on the first `send_one`, because it needs the running event loop.

**The waiting side** (`wait_for_ack`, not in the book's box):

```python
async def wait_for_ack(self, uid: int):
    event = self.event_map[uid]
    while True:
        await event.wait()          # sleep until the light turns on
        event.clear()               # turn it off again
        pending = self.ack_map[uid]; self.ack_map[uid] = []   # take the whole pile
        ...
            yield ack               # hand each reply to the streaming code
```

Because several `UserReply`s can land before this coroutine runs again, it takes **all** pending ones at once. The generator ends after a reply with `finished=True`, then deletes the ticket's entries.

**Cancellation (the paragraph about Ctrl+C):**

| Step | Code |
|---|---|
| Notice the customer left | `stream_with_cancellation` checks `await request.is_disconnected()` between chunks and raises `asyncio.CancelledError` |
| Start the abort | on `CancelledError`: `asyncio.create_task(self.abort_user(uid))` |
| Clean the desk's own books | `abort_user` waits 0.1 s, deletes `ack_map[uid]` and `event_map[uid]` |
| Tell the kitchen | `await self.send_one(AbortMsg(uid=uid))` → tokenizer converts it to `AbortBackendMsg` → scheduler |

### 4.3 Tokenization: `tokenizer/server.py`

```python
# tokenizer/server.py (excerpt)
while True:
    pending_msg = _unwrap_msg(recv_listener.get())        # Receive
    ...
    tensors = tokenize_manager.tokenize(tokenize_msg)     # Text -> token IDs
    send_backend.put(BatchBackendMsg(data=[
        UserMsg(uid=msg.uid, input_ids=t, sampling_params=msg.sampling_params)
        for msg, t in zip(tokenize_msg, tensors)
    ]))                                                   # Send
```

*Line by line:* the worker is an endless loop: take the next packet (waits if none), unpack, tokenize every `TokenizeMsg` in it, wrap each result as a `UserMsg` carrying the *same* `uid` and `sampling_params`, send them to the scheduler as one `BatchBackendMsg`.

**Left out:** the same loop also handles the *detokenizer's* two jobs. It sorts each packet into three lists (`DetokenizeMsg`, `TokenizeMsg`, `AbortMsg`) and handles each: detokenize → `UserReply`s sent to the front desk; tokenize → `UserMsg`s sent to the scheduler; abort → `AbortBackendMsg`s sent to the scheduler. That's why the default has one worker for both. Small packaging detail: when a list has exactly one message, it sends the single message instead of a one-item batch.

**The chat template step** (explained in the concept page 2.2): inside `TokenizeManager.tokenize`, if `msg.text` is a list it calls `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` first, then `encode(..., return_tensors="pt")`, then `.view(-1).to(torch.int32)`.

### 4.4 Scheduling: `scheduler/scheduler.py`

The book first checks the message: requests that are too long cannot proceed, and valid ones go into a waiting queue. In the real code (`_process_one_msg` for a `UserMsg`): `max_output_len = max_seq_len - input_len`; **if that is not positive, the request is dropped with a log warning** (no reply is ever sent, so the client would wait); if `max_tokens` is bigger than what fits, it is **clamped** with a warning. Then `prefill_manager.add_one_req(msg)` creates a `PendingReq`.

```python
# scheduler/scheduler.py (simplified)
def normal_loop(self) -> None:
    blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):  # ① Wait only when there is no work
        self._process_one_msg(msg)
    forward_input = self._schedule_next_batch()    # ② Select requests and form a batch
    ongoing = None
    if forward_input is not None:
        ongoing = (forward_input, self._forward(forward_input))   # ③ Run a GPU step
    self._process_last_data(ongoing)               # ④ Process the completed results
```

*Line by line (the head waiter's routine):*

| Step | Kid words |
|---|---|
| ① `blocking = not (...)` and `receive_msg` | If nothing is waiting or eating, sleep until a new order arrives; otherwise just grab whatever is already in the mailbox without waiting |
| ② `_schedule_next_batch()` | Decide the tray |
| ③ `_forward(forward_input)` | Hand the tray to the chef |
| ④ `_process_last_data(ongoing)` | Read the dishes that came back: append words, check "done?", notify the card-reader, free finished orders |

**Important: this is not the loop that runs by default.** `run_forever()` picks `normal_loop` only if the environment variable `MINISGL_DISABLE_OVERLAP_SCHEDULING` is true. By default it runs **`overlap_loop`**:

```text
   normal_loop  (what the book shows, easier to read)
   CPU: [schedule n][ launch n ]·····wait for GPU·····[process results n][schedule n+1][launch n+1]...
   GPU:                        [====== run n ======]                                  [== run n+1 ==]
                                                      ▲ GPU sits idle while the CPU does bookkeeping

   overlap_loop (the default)
   CPU: [schedule n][launch n][process results of n-1 ········]  [schedule n+1][launch n+1][process n ···]
   GPU:                       [============ run n ============]                [======= run n+1 ========]
                                       ▲ the CPU's bookkeeping hides inside the GPU's work
```

In `overlap_loop(last_data)` the order is: receive → **schedule and launch batch n** → *then* process the results of batch **n-1** while the GPU is busy → return batch n's data for the next iteration. It runs the GPU work on a separate CUDA stream (`engine.stream`) and does the CPU metadata work on another (`self.stream`), synchronised with `wait_stream`.

**Why it can work:** the next batch's input tokens are read from the GPU-side `token_pool` (the previous step's sampled tokens were written there directly by the GPU), so the CPU does not have to see the sampled values before scheduling the next step. The price: a request that finished by EOS in batch n-1 may already be in batch n, so one extra step is run for it. The code guards this: `finished_reqs` stops a request from being freed twice.

```python
# scheduler/scheduler.py (excerpt)
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

*Line by line:* try a prefill tray first (new orders, within the per-pass word budget `prefill_budget = max_extend_tokens`, default 8192). `or` means "if that returned nothing (`None`), try the decode tray". So **prefill has priority over decode**; the source has a TODO for other policies. A long queue does not guarantee a prefill batch: the prefill adder checks that the order's worst-case notes fit alongside everything already running.

**What `_prepare_batch` does after a batch is chosen** (not in the book): (1) `graph_runner.pad_batch(batch)` adds dummy requests so the size matches a recorded CUDA graph; (2) `cache_manager.allocate_paged(batch.reqs)` gives each request pages for the tokens it is about to write; (3) it builds `positions`; (4) two index lists (`input_tuple`, `write_tuple`, below); (5) `batch.out_loc = page_table[input_mapping]`; (6) `attn_backend.prepare_metadata(batch)`; (7) `sampler.prepare(batch)`.

**Tiny numbers: assembling a prefill tray.** Two orders: **A** is on peg 3, nothing cached, 5 tokens. **B** is on peg 7, 2 tokens already cached, 6 tokens total.

| Request | `table_idx` | `cached_len` | `device_len` | `extend_len` | Positions to compute |
|---|---|---|---|---|---|
| A | 3 | 0 | 5 | 5 | 0, 1, 2, 3, 4 |
| B | 7 | 2 | 6 | 4 | 2, 3, 4, 5 |

```text
   positions       = [0,1,2,3,4,  2,3,4,5]          (9 tokens, A's then B's)
   token rows      = [3,3,3,3,3,  7,7,7,7]          (which peg each token belongs to)
   input_tuple     = (token rows, positions)        → read  token_pool[row, position]  = the token ids
   batch.out_loc   = page_table[(rows, positions)]  → the KV slot where each token's new K and V go
   write_tuple     = ([3, 7], [5, 6])               → after sampling, write A's new token at token_pool[3,5]
                                                       and B's at token_pool[7,6]  (position = device_len)
```

Notes: for a request that **cannot decode further**, the write position is `-1`, a sentinel meaning "throw this one away" (a prompt chunk whose logits we ignore, for example). The tray's `input_ids` is gathered with `self.token_pool[input_mapping]`, and the sampled tokens are scattered back with `self.token_pool[output_mapping] = forward_output.next_tokens_gpu`.

For a **decode** tray the same helper functions give: one token per request, its position = `cached_len` (the newest token), its row = the request's peg.

```python
# scheduler/scheduler.py (excerpt)
for i, req in enumerate(batch.reqs):
    req.append_host(next_token)                        # Append the new token
    finished = not req.can_decode                      # Reached max_tokens?
    if not req.sampling_params.ignore_eos:
        finished |= next_token == self.eos_token_id    # Generated EOS?
    reply.append(DetokenizeMsg(uid=req.uid, next_token=..., finished=finished))
self.send_result(reply)
```

*Line by line:* for each request on the tray that just finished its pass: record the new word on the CPU copy; decide if it's done (ran out of allowed words, or wrote the end-of-sequence token unless `ignore_eos`); queue a message for the card-reader; send them all. **Left out:** (1) chunked prefill requests are skipped (`isinstance(req, ChunkedReq): continue`), since their pass produced no real next word; (2) for finished requests: `decode_manager.remove_req(req)` and `_free_req_resources(req)` (gives back the peg and saves/frees the cache pages) and the freeing happens inside `cache_manager.lazy_free_region()` so many small frees become one batched operation; (3) for a normal prefill request that did not finish, `cache_manager.cache_req(req, finished=False)` inserts the prompt into the prefix cache so later requests can reuse it; (4) before reading the sampled tokens it calls `copy_done.synchronize()`, waiting for the GPU-to-CPU copy to land.

### 4.5 Computation: `engine/engine.py`

```python
# scheduler/scheduler.py (excerpt; called by step ③ of the loop)
def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
    batch, sample_args, input_mapping, output_mapping = forward_input
    batch.input_ids = self.token_pool[input_mapping]        # Tokens for this iteration
    forward_output = self.engine.forward_batch(batch, sample_args)   # Call the engine
    self.token_pool[output_mapping] = forward_output.next_tokens_gpu # Save new tokens
    return forward_output
```

*Line by line:* unpack the prepared bundle; gather this pass's token ids out of the GPU token pool using the (row, position) pairs; run the engine (an ordinary function call in the same process); scatter the new tokens back into the pool at their positions. **Left out:** one more line, `self.decode_manager.filter_reqs(forward_input.batch.reqs)`, which adds this tray's requests to the "tables still eating" set and drops any that can no longer decode.

```python
# engine/engine.py (simplified)
def forward_batch(self, batch, args):
    logits = self.model.forward()                     # Score the candidate tokens
    next_tokens = self.sampler.sample(logits, args)   # Handles softmax and token selection internally
    next_tokens_cpu = next_tokens.to("cpu", non_blocking=True)   # Copy to the CPU
    ...
```

*Line by line, and the real version:*

| Simplified | Real code |
|---|---|
| `self.model.forward()` | Runs inside `with self.ctx.forward_batch(batch):` (hangs the tray on the whiteboard). If `graph_runner.can_use_cuda_graph(batch)` it **replays a pre-recorded CUDA graph**, otherwise calls the model normally |
| (nothing) | `for req in batch.reqs: req.complete_one()` bookkeeping runs now, right after launching |
| `sampler.sample(logits, args)` | `sample(logits[: batch.size], args).to(torch.int32)`: only the real rows (not padding) |
| `.to("cpu", non_blocking=True)` | Same, plus a `torch.cuda.Event` recorded after the copy so the scheduler can later wait for exactly that copy |
| (returns) | `ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)` |

**The sampler** (`engine/sample.py`), as the book simplifies it:

```python
# engine/sample.py (simplified, combining sample and sample_impl)
def sample(self, logits, args):
    if args.temperatures is None:                      # The entire batch uses greedy
        return torch.argmax(logits, dim=-1)
    probs = sampling.softmax(logits.float(), args.temperatures)  # Temperature scaling + softmax
    return sampling.sampling_from_probs(probs)         # Draw a token from the probabilities
```

*Line by line:* if every request in the tray is greedy, just take the biggest score per row. Otherwise convert logits to float32, divide by each request's own temperature and softmax (one temperature per row), then draw. **Left out:** the top-k / top-p / both variants (chosen according to which filters any request uses), and a clever detail in `Sampler.prepare`: in a **mixed** tray, a greedy request is given a tiny temperature (1e-6) instead of 0, so it still flows through the same softmax code and effectively picks its top word. (Real softmax kernels subtract the largest score first so such tiny temperatures do not overflow.)

### 4.6 Detokenization: `tokenizer/detokenize.py`

```python
# tokenizer/detokenize.py (excerpt)
new_text = read_str[len(surr_str):]
if len(new_text) > 0 and not new_text.endswith("�"):    # Complete text can be emitted
    output_str = s.decoded_str + new_text
    ...
else:
    new_text = find_printable_text(new_text)             # Emit only the safe portion
```

*Line by line:* `read_str` is the text of the recent cards, `surr_str` is the text of the older "context" cards in that window; subtracting leaves only the **brand-new text**. If it is non-empty and does not end in the broken-character symbol, it is complete: add it to the running string (and advance the offsets, in the `...`). Otherwise, only the safely printable part is sent. The two worked traces (`café!` and `你好.`) are in the concept page, section 2.4. The class is `DetokenizeManager`, keeping one `DecodeStatus` per uid; when a message has `finished=True`, the uid's state is deleted.

**Left out:** on the final message, if `msg.finished` and the token **is** the end-of-sequence token, the token is *not* appended to the text (so the answer doesn't end with a literal `<|im_end|>`).

### 4.7 Returning to the frontend

*Kid version:* the card-reader pushes a `UserReply` to the front desk's mailbox; the mail sorter (`listen`) puts it in the right ticket's pile and turns the light on; the ticket's generator wakes, wraps the new text in an SSE chunk, and yields it. After a reply with `finished=True` it yields the closing chunk and `data: [DONE]`.

**Two streaming formats exist in this file:** `/v1/chat/completions` yields OpenAI-style JSON chunks (`data: {...}\n\n`); the plain `/generate` endpoint yields bare text lines (`data: <text>\n`).

## 5 Finding the Corresponding Code in SGLang

| Component | mini-sglang | SGLang (`python/sglang/srt/`) | Kid gloss |
|---|---|---|---|
| Frontend | `server/api_server.py` | `entrypoints/http_server.py` + `managers/tokenizer_manager.py` | The front desk (SGLang merges it with the card-maker) |
| Tokenizer | `tokenizer/server.py` | `managers/tokenizer_manager.py` | The card-maker |
| Scheduler | `scheduler/scheduler.py` | `managers/scheduler.py` (`event_loop_normal` / `event_loop_overlap`) | The head waiter (same two loops!) |
| Engine | `engine/engine.py` | `managers/tp_worker.py` + `model_executor/model_runner.py` | The chef's hands |
| Detokenizer | `tokenizer/detokenize.py` | `managers/detokenizer_manager.py` | The card-reader |
| Between components | ZMQ | ZMQ | Same phone lines |

SGLang has far more code (LoRA, weight updates, multimodal, ...). On a first pass follow one plain text request and find where each message type is received.

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** The `uid` is the thread through the whole maze: front desk makes it, every message carries it, and the front desk uses it to find the right waiting customer. Normal completion and cancellation both end by **freeing the order's peg and pages**; cancellation just starts one hop earlier, at the front desk.

### 6.2 Exercises

**1. Where is the uid created, the `TokenizeMsg` sent, and the `UserReply` received? How do the messages stay linked?**

<details><summary>Answer</summary>

Created: `FrontendManager.new_user()` in `server/api_server.py`. Sent: `await state.send_one(TokenizeMsg(uid=...))` in the endpoint handler. Received: `FrontendManager.listen()`. The `uid` is copied into `UserMsg`, `DetokenizeMsg` and `UserReply`, and `ack_map`/`event_map` are keyed by it.

</details>

**2. Compare `_schedule_next_batch` and `_forward`. Which decides, which calls the engine?**

<details><summary>Answer</summary>

`_schedule_next_batch` decides (prefill batch if possible, else decode batch) and `_prepare_batch` builds the tensors. `_forward` gathers input ids from the token pool, calls `engine.forward_batch`, and scatters the sampled tokens back.

</details>

**3. Interrupt a streaming client. Follow `AbortMsg` to the cleanup.**

<details><summary>Answer</summary>

`stream_with_cancellation` → `abort_user(uid)` → `AbortMsg` → `tokenize_worker` converts to `AbortBackendMsg` → `Scheduler._process_one_msg` → `prefill_manager.abort_req(uid)` or `decode_manager.abort_req(uid)` → `_free_req_resources(req)` → `table_manager.free(req.table_idx)` and `cache_manager.cache_req(req, finished=True)`.

</details>

**4. Find the length/EOS checks, and where completion becomes `data: [DONE]`.**

<details><summary>Answer</summary>

`Scheduler._process_last_data`: `finished = not req.can_decode` (length), `finished |= next_token == self.eos_token_id` (EOS, unless `ignore_eos`). The flag rides in `DetokenizeMsg.finished` → `UserReply.finished` → `stream_chat_completions` breaks, yields the closing chunk and `data: [DONE]`.

</details>

**Extra drills (not in the book).**

5. In the overlap loop, why is it safe to schedule batch n before knowing batch n-1's sampled tokens on the CPU? (The tokens are read from the GPU token pool, where the GPU already wrote them.)
6. What happens to a request whose prompt is longer than the model's maximum length? (It is dropped with a warning and never replied to; the client will keep waiting.)
7. Why does `abort_user` wait 0.1 s before sending the abort? (Likely to let in-flight chunks drain; the source gives no reason, so treat that as a guess.)
