# Part II Chapter 6 Continuous Batching and the Scheduler (Code Walkthrough): ELI5 companion

Original: [Chapter6_Continuous-Batching-and-the-Scheduler_code.md](../../course-material/eng/part2/Chapter6_Continuous-Batching-and-the-Scheduler_code.md). Same headings, same numbers. Concept half (written by me, because the original is a placeholder): [Chapter6_Continuous-Batching-and-the-Scheduler.md](./Chapter6_Continuous-Batching-and-the-Scheduler.md). Cast and picture legend: [../README.md](../README.md).

Files named here carry ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang` (paths relative to `python/minisgl/`): `scheduler/scheduler.py`, `prefill.py`, `decode.py`, `table.py`, `cache.py`, and `core.py`.

Run the numbers yourself: `python3 -I eli5/tools/ch6_scheduler_toy.py`.

## The one-minute version

**Kid version.** The whole chapter is this loop and four helper clipboards:

```text
   forever:
      mail  ──▶  _process_one_msg  ──▶  new order goes on the WAITING clipboard   (prefill_manager.pending_list)
      tray  ──▶  _schedule_next_batch:  prefill tray if any waiting order passes admission
                                        else decode tray of everyone "still eating" (decode_manager.running_reqs)
      cook  ──▶  _forward:               chef runs the tray; those who can still talk stay on "still eating"
      tidy  ──▶  _process_last_data:     record words, finish orders, return pegs, tell the card-reader
```

## 1 Learning Objectives

| Goal | Where |
|---|---|
| Locate the waiting queue, running set, and the moves between them | Sections 2, 4.1, 4.3 |
| Explain prefill priority, request slots, token budgets from the source | Sections 3.2, 4.2 |
| Trace output updates, EOS, length limits, cleanup | Section 5 |

## 2 Locate the Source

| File and entry point | What to look for | Kid gloss |
|---|---|---|
| `scheduler/scheduler.py`: `Scheduler.normal_loop` | Receive, schedule, compute, process results in one turn | The waiter's routine |
| `scheduler/prefill.py`: `PrefillManager`, `PrefillAdder` | Admit waiting requests, check budgets | The waiting clipboard and the admission clerk |
| `scheduler/decode.py`: `DecodeManager` | Requests that continue generating | The "still eating" clipboard |
| `core.py`: `Req` | Cached length, remaining budget, state | One order slip with its counters |
| `scheduler/table.py`, `scheduler/cache.py` | Request slots, cache resources | Pegs; pantry clerk |

**A `Req`'s counters, in one picture** (a prompt of 10 words, 6 already cached, answer limit 4):

```text
   position:      0 1 2 3 4 5 | 6 7 8 9 | 10 11 12 13
                  └─ cached ─┘ └ to read ┘ └ may be written later ┘
   cached_len = 6   device_len = 10   max_device_len = 14   (= 10 + output_len 4)
   extend_len = device_len − cached_len = 4   (words the chef must actually process this pass)
   remain_len = max_device_len − device_len = 4   (words still allowed to be produced)
   can_decode = remain_len > 0
```

After the pass launches, `complete_one()` runs: `cached_len = device_len` (10), `device_len += 1` (11). Notes now cover the prompt, and one new word is *owed*.

**The mail step.** After HTTP handling and tokenization, the scheduler receives a `UserMsg`. `_process_one_msg` (excerpt, comments removed):

```python
input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
max_output_len = max_seq_len - input_len
if max_output_len <= 0:
    return logger.warning_rank0(...)          # dropped, and no reply is ever sent (Q3)
if msg.sampling_params.max_tokens > max_output_len:
    msg.sampling_params.max_tokens = max_output_len
    logger.warning_rank0(...)
self.prefill_manager.add_one_req(msg)
```

| Line | Kid words |
|---|---|
| `max_output_len = max_seq_len - input_len` | How many words of answer still fit in the model's total length after this prompt |
| `if max_output_len <= 0: return ...` | The prompt alone fills (or exceeds) the whole limit: the order is thrown away with only a log line. The customer is never told (quirk Q3) |
| clamp `max_tokens` | Asked for more than fits? Quietly reduce the request to what fits |
| `add_one_req(msg)` | Put it at the end of the waiting clipboard as a `PendingReq(uid, input_ids, sampling_params)` |

## 3 Read One Scheduling Step

### 3.1 Receive, schedule, execute, and process results

```python
blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
for msg in self.receive_msg(blocking=blocking):
    self._process_one_msg(msg)

forward_input = self._schedule_next_batch()
ongoing_data = None
if forward_input is not None:
    ongoing_data = (forward_input, self._forward(forward_input))

self._process_last_data(ongoing_data)
```

*Line by line:*

| Line | Kid words |
|---|---|
| `blocking = not (A or B)` | If someone is waiting **or** someone is eating, don't sleep waiting for mail: there is work. Only when both clipboards are empty does the waiter sit and wait for the next slip |
| `for msg in self.receive_msg(blocking=blocking)` | Collect slips (new orders, cancels, "close the restaurant") |
| `_process_one_msg(msg)` | File each one |
| `forward_input = self._schedule_next_batch()` | Build the tray and its index lists, or `None` if there is nothing to cook |
| `self._forward(forward_input)` | Hand it to the chef; the result holds the new words |
| `ongoing_data = (input, output)` | Keep the tray and its result together so each new word is matched to the right order |
| `self._process_last_data(ongoing_data)` | Read the dishes. (In the plain loop it reads **this** turn's tray right away) |

**This is the plain `normal_loop`.** The default is `overlap_loop`, which processes the **previous** turn's results after launching this turn's tray:

```python
# overlap_loop (shape only)
for msg in self.receive_msg(blocking=blocking): self._process_one_msg(msg)
forward_input = self._schedule_next_batch()
if forward_input is not None:
    with self.engine_stream_ctx:
        self.engine.stream.wait_stream(self.stream)
        ongoing_data = (forward_input, self._forward(forward_input))
self._process_last_data(last_data)      # <- results of the PREVIOUS turn, while the GPU runs this one
return ongoing_data                      # <- becomes `last_data` next turn
```

Only `normal_loop` runs when `MINISGL_DISABLE_OVERLAP_SCHEDULING` is set. (Doc drift D2 in `NOTES-open-questions.md`.) In `overlap_loop` the `blocking` test also includes `last_data is not None`: a pending previous result counts as work.

### 3.2 Try prefill first, otherwise decode

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
return self._prepare_batch(batch) if batch else None
```

| Line | Kid words |
|---|---|
| `A or B` | If `A` is a real batch use it; if `A` is `None` (nothing admissible) use `B`. This is the policy "new orders first" |
| `prefill_manager.schedule_next_batch(self.prefill_budget)` | Try to fill a prefill tray with at most `prefill_budget` prompt words (default 8192) |
| `decode_manager.schedule_next_batch()` | Else everyone eating, in `uid` order |
| `_prepare_batch(batch) if batch else None` | If there is a tray, set it (below); otherwise idle this turn |

**A non-empty waiting list is not a prefill tray.** Anything can refuse the first waiting order (no peg, pantry worst-case, no budget), and then the `or` falls to decode.

**`_prepare_batch`, the seven steps** (the tray-setting, in the order the code does them):

| # | Step | Kid words |
|---|---|---|
| 1 | `graph_runner.pad_batch(batch)` | Pad with dummy orders to a size a recorded CUDA graph exists for |
| 2 | `cache_manager.allocate_paged(batch.reqs)` | Give every order pantry pages for the words it is about to write |
| 3 | `batch.positions = _make_positions(...)` | Seat number of every word (for RoPE) |
| 4 | `_make_input_tuple`, `_make_write_tuple` | Index lists: where to *read* input ids, where to *write* the new ones |
| 5 | `batch.out_loc = page_table[input_mapping]` | The pantry slot each new word's K/V will be written to |
| 6 | `attn_backend.prepare_metadata(batch)` | Extra paperwork for the attention kernels |
| 7 | `sampler.prepare(batch)` | Per-order temperature / top-k / top-p tensors |

**Continuous batching does not mean prefill and decode share a pass.** A pass is all-prefill or all-decode. The *members* change.

**Pause and predict**

1. The waiting list holds one order and 256 orders are eating, `max_running_req = 256`. Which tray is built?

<details><summary>Answer</summary>

A decode tray. `PrefillAdder._try_allocate_one` sees `table_manager.available_size == 0` and returns `None`, so `schedule_next_batch` returns `None` and the `or` picks decode.

</details>

## 4 How Requests Enter and Stay in a Batch

### 4.1 Try waiting requests in order

```python
reqs = []
for pending_req in self.pending_list:
    if req := adder.try_add_one(pending_req):
        reqs.append(req)
    else:
        break
```

*Line by line:* walk the waiting clipboard from the front; ask the admission clerk about each; add it if she says yes; **stop at the first no**. The real method also (1) starts the clerk with `reserved_size = decode_manager.inflight_tokens` and (2) handles half-read orders:

```python
# prefill.py schedule_next_batch (shape only)
adder = PrefillAdder(token_budget=prefill_budget, reserved_size=self.decode_manager.inflight_tokens, ...)
...
if isinstance(req, ChunkedReq):               # only part of the prompt fit this pass
    pending_req.chunked_req = req             # remember progress on the clipboard entry
    chunked_list.append(pending_req)
...
self.pending_list = chunked_list + self.pending_list[len(reqs):]   # half-read orders first, then the unreached
```

| Piece | Kid words |
|---|---|
| `reserved_size = decode_manager.inflight_tokens` | Start the "memory already promised" counter with the worst-case needs of everyone currently eating |
| `ChunkedReq` branch | An order that was only partly read stays on the clipboard, **at the front**, remembering how far it got |
| `chunked_list + self.pending_list[len(reqs):]` | New clipboard = half-read orders first, then the orders we never reached. Fully admitted orders drop off |

### 4.2 Keep three resource limits distinct

| Constraint | Source check | Meaning |
|---|---|---|
| Request slots | `table_manager.available_size` | A free peg exists |
| Compute budget this step | `token_budget` | Prompt words this pass may still read |
| Estimated KV capacity | `estimated_len + reserved_size` | Token space promised to new and running orders |

```python
extend_len = req.input_len - cached_len
estimated_len = extend_len + req.output_len
if estimated_len + self.reserved_size > self.cache_manager.available_size:
    return None
```

| Line | Kid words |
|---|---|
| `extend_len = input_len - cached_len` | Words that still need new notes (the already-cached beginning is free) |
| `estimated_len = extend_len + output_len` | Add the **most** it may write afterwards. Worst case |
| `... + self.reserved_size` | Add what is already promised (to eaters and to earlier arrivals in this same pass) |
| `> cache_manager.available_size` | Compare to free pages plus evictable cache entries (nobody using them) |
| `return None` | Too much: refuse |

**The order of checks in `_try_allocate_one`:**

```text
   1. no free peg?                                    → None
   2. match the prefix cache → cached_len, a "handle"
   3. worst-case check #1                              → None if it does not fit
   4. LOCK the handle (pin the matched beginning)
   5. worst-case check #2  (locking made available_size SMALLER)   → undo the lock, None if it does not fit now
   6. take a peg; if cached_len > 0, copy the cached ids and slot numbers into the peg's row
```

Step 5 exists because locking moves a matched prefix from "could be thrown out if needed" to "untouchable", which shrinks `available_size`. A check that passed before locking can fail after.

**Tiny numbers (my toy, Part 3).** Pantry 100 tokens; one eating order has used 30 and may still produce 40, so free = 70, promised = 40:

| Case | Waiting list as (prompt + answer) | Result |
|---|---|---|
| A | N1 15+20 = 35 | 35 + 40 = 75 > 70 → **refused** (although 70 are free right now) |
| B | N1 15+10 = 25, then N2 5+3 = 8 | N1: 25 + 40 = 65 ≤ 70 → admitted, promised 65. N2: 8 + 65 = 73 > 70 → **refused** |
| C | BIG 50+20 = 70, then small 3+2 = 5 | BIG: 110 > 70 → refused; `small` (45 ≤ 70, would fit) is **never asked** |

**Chunked prefill.** `_add_one_req`:

```python
remain_len = pending_req.input_len - cached_len
chunk_size = min(self.token_budget, remain_len)
is_chunked = chunk_size < remain_len
CLS = ChunkedReq if is_chunked else Req
self.token_budget -= chunk_size
self.reserved_size += remain_len + pending_req.output_len
```

| Line | Kid words |
|---|---|
| `remain_len = input_len - cached_len` | Unread words of the prompt |
| `chunk_size = min(token_budget, remain_len)` | Read as much as the pass has budget for, but no more than is left |
| `is_chunked = chunk_size < remain_len` | If we could not finish the prompt, this is a *partial* read |
| `CLS = ChunkedReq if is_chunked else Req` | Partial reads use `ChunkedReq`: no word is sampled and it never joins `running_reqs` |
| `token_budget -= chunk_size` | Spend the pass's budget |
| `reserved_size += remain_len + output_len` | Promise pantry for the **whole rest** of the prompt plus the full answer, not only this chunk |

`ChunkedReq` is a `Req` with two overrides:

```python
class ChunkedReq(Req):
    def append_host(self, next_token): raise NotImplementedError("ChunkedReq should not be sampled")
    @property
    def can_decode(self): return False
```

`_process_last_data` skips chunked requests (`if isinstance(req, ChunkedReq): continue`), so the `append_host` raise is only a guard.

Chunk bookkeeping (my toy, Part 4; prompt 20, budget 8):

| Pass | `cached_len` going in | remain | chunk | Kind |
|---|---|---|---|---|
| 1 | 0 | 20 | 8 | `ChunkedReq` |
| 2 | 8 | 12 | 8 | `ChunkedReq` |
| 3 | 16 | 4 | 4 | `Req` (first word sampled) |

For a continuing chunk, `try_add_one` takes `cache_handle`, `table_idx` and `cached_len` from the stored `chunked_req` (its `cached_len` was advanced to the end of the previous chunk by `complete_one()`), so the order keeps its peg and its protected beginning.

### 4.3 Keep requests that still need to generate

```python
self.running_reqs = {
    req for req in self.running_reqs.union(reqs) if req.can_decode
}
```

| Line | Kid words |
|---|---|
| `self.running_reqs.union(reqs)` | Everyone already eating plus everyone who was just on the tray |
| `if req.can_decode` | Keep only those that may still produce words (`remain_len > 0`) |
| `{ ... }` | A set: no duplicates, so an order on two trays is still listed once |

It runs in `_forward`, **after** `engine.forward_batch` has called `complete_one()` on every order on the tray. So `can_decode` already reflects the pass just launched. A length-limited order drops off *at launch*. EOS cannot be known yet: the word is still on the GPU.

**Decode tray order.** `DecodeManager.schedule_next_batch` sorts by `uid`. Reason: with several GPUs, each worker builds its own tray from its own set; a Python set has no guaranteed order, and without the sort the workers could put the same requests in different orders, so their results would not line up.

**Also here: `inflight_tokens`.** `sum(remain_len over running) + (page_size − 1) × len(running)`. The second term reserves one page of slack per order (a page being written may be only partly used).

**Pause and predict**

2. An order is on tray *n*, reaches its length limit on that pass, and is launched. Is it on tray *n*+1?

<details><summary>Answer</summary>

No. `complete_one()` runs inside `forward_batch`, so `remain_len` is already 0 when `filter_reqs` runs, and `can_decode` is false: it is dropped from `running_reqs`. (An order that ends by EOS is different: that is only seen one turn later under overlap scheduling.)

</details>

## 5 Update Outputs and Change the Batch

### 5.1 Complete requests and release resources

```python
next_token = next_tokens_cpu[i]
req.append_host(next_token.unsqueeze(0))
finished = not req.can_decode
if not req.sampling_params.ignore_eos:
    finished |= int(next_token.item()) == self.eos_token_id
if finished:
    self.decode_manager.remove_req(req)
    self._free_req_resources(req)
```

| Line | Kid words |
|---|---|
| `next_token = next_tokens_cpu[i]` | The i-th order's new word, already copied to the CPU (`copy_done.synchronize()` ran first) |
| `req.append_host(...)` | Add it to the CPU copy of the order's words. The shared index will use these as its lookup keys |
| `finished = not req.can_decode` | Hit the length limit? |
| `finished \|= token == eos_token_id` | Or wrote the end-of-sequence word. (Unless the order set `ignore_eos`, used by benchmarks that need exact lengths) |
| `decode_manager.remove_req(req)` | Cross it off the eating list |
| `_free_req_resources(req)` | Return the peg and offer its words to the shared index (below) |

The pieces the book's excerpt leaves out:

| Omitted bit | Why it exists |
|---|---|
| `with self.cache_manager.lazy_free_region():` | Collect every page-freeing call in this loop and apply them once at the end (one `torch.cat` instead of many small ones) |
| `isinstance(req, ChunkedReq): continue` | A half-read prompt produced no word |
| `req not in self.finished_reqs` | Under overlap scheduling, an EOS-ended order can already be on the next tray; when that tray's result arrives it "finishes" a second time. The set from the previous turn lets the scheduler skip the second free |
| `elif batch.is_prefill: cache_manager.cache_req(req, finished=False)` | A freshly read, unfinished prompt is written into the shared index **now** so a later customer with the same beginning can reuse it |
| `DetokenizeMsg(uid, next_token, finished)` | The new word on its way to the card-reader and, from there, the customer |

```python
def _free_req_resources(self, req):
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

`table_manager.free` is a plain list `append` (the peg's numbers are *not* cleared; the next owner overwrites). `cache_req(finished=True)` splits the order's pages into three groups (borrowed beginning, newly inserted, duplicates + tail) as described in the chapter-4 companion.

**Cancellation** (`AbortBackendMsg` branch of `_process_one_msg`):

```python
req_to_free = self.prefill_manager.abort_req(msg.uid)
req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
if req_to_free is not None:
    self._free_req_resources(req_to_free)
```

| Line | Kid words |
|---|---|
| `prefill_manager.abort_req(uid)` | Find it on the waiting clipboard and remove it. Returns its half-read `chunked_req` if it had one (that holds a peg), else `None` |
| `... or decode_manager.abort_req(uid)` | Not waiting? Look among the eaters and remove it, returning that `Req` |
| `if req_to_free is not None:` | A queued order with no resources only needed removing; one that held a peg and pages needs `_free_req_resources` |

Note what an abort does **not** do: it does not add the order to `finished_reqs`. Hypothesis **H3** in the notes asks what happens if the order is also in the batch still being processed.

### 5.2 Work through A, B, and C

Assumptions (from the book): two request slots, enough pantry, no prefix hits, each prompt prefilled in one pass; A, B, C need 2, 5, 2 words; C arrives after step 0; finishing prefill produces the first word.

**The book's table, plus the clipboards** (toy output verified against the book's rows; the clipboard columns are mine, derived by hand from the same rules):

| Step | Tray | Words after the step | Waiting | Eating (after) | Free pegs (after) | Finished |
|---|---|---|---|---|---|---|
| 0 | Prefill A, B | A(1), B(1) | C (arrives after this step) | A, B | 0 | none |
| 1 | Decode A, B | A(2), B(2) | C (**no peg free, so no prefill**) | B | 1 | A |
| 2 | Prefill C | C(1) | none | B, C | 0 | none (B **pauses**) |
| 3 | Decode B, C | B(3), C(2) | none | B | 1 | C |
| 4 | Decode B | B(4) | none | B | 1 | none |
| 5 | Decode B | B(5) | none | none | 2 | B |

Step 1 is the key row: C is waiting but both pegs are taken, so `PrefillAdder` refuses it and the `or` picks decode. A's finish at the end of step 1 frees a peg, so step 2 can prefill C.

**Where this differs from the default loop.** Under `overlap_loop`, A's finish is *processed* one turn later than it is *launched*, so C would be admitted one turn later than the table says. The book uses `normal_loop` to keep the story simple; the toy does the same.

**Fairness.** B waited an extra pass at step 2 because new orders go first. With a steady stream of arrivals, existing orders' decode steps could be delayed repeatedly; the code has no counter-measure (see the `TODO: support other policies: e.g. DECODE first` comment in `_schedule_next_batch`).

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** One loop. New orders are admitted FIFO under three limits (peg, word budget, worst-case pantry), long prompts are read in chunks, and each pass is all-prefill or all-decode. Orders leave on length or EOS and give back their peg while their notes may stay. The default loop overlaps paperwork with cooking, so the book's tidy step-by-step table is accurate in shape but one turn off in timing.

### 6.2 Exercises

**1. Why might decode run even when the waiting queue is not empty?**

<details><summary>Answer</summary>

Step 1 of the A/B/C trace is the example: C is waiting but no peg is free. Equally, a failed worst-case pantry check or a spent word budget gives `None` from `PrefillManager`.

</details>

**2. Can a request enter with a free slot but too little KV? Which code checks it?**

<details><summary>Answer</summary>

Yes. `PrefillAdder._try_allocate_one` checks the peg (`table_manager.available_size`) first, then `estimated_len + reserved_size > cache_manager.available_size`, twice (before and after locking the matched prefix).

</details>

**3. Why must intermediate prefill chunks not commit generated tokens?**

<details><summary>Answer</summary>

The prompt is incomplete, so no real next-word exists yet; sampling at the end of a chunk would insert a word the user never wrote. `ChunkedReq.append_host` raises, `can_decode` is false, and `_process_last_data` skips chunked requests.

</details>

**4. Under the section 5.2 assumptions, when can C first enter if A generates only one token?**

<details><summary>Answer</summary>

Step 1. A finishes in the step-0 prefill pass and frees its peg at the end of step 0; C arrives after step 0, so the next turn prefills it. (Toy: change A's length from 2 to 1.)

</details>

**5. Where do completion and cancellation enter cleanup? Why doesn't freeing a slot delete all KV?**

<details><summary>Answer</summary>

Both reach `_free_req_resources`: completion through `_process_last_data`, cancellation through the `AbortBackendMsg` branch. Freeing the peg only returns a table row. `cache_req(finished=True)` keeps whole reusable pages in the prefix cache and frees only duplicates and the partial tail page.

</details>

**Extra drills (not in the book).**

6. `prefill_budget = 8192`, three waiting prompts of 5000, 4000, 100 words, plenty of pegs and pantry. Describe the first prefill tray. (Tray 1: order 1 whole, 5000 words, budget left 3192; then the first 3192 words of order 2 as a `ChunkedReq`, budget left 0; order 3 is not reached because `try_add_one` sees `token_budget <= 0`. Tray 2: order 2's last 808 words, now a normal `Req` whose first word is sampled, then all 100 words of order 3. Order 1, already eating, pauses for both prefill trays.)
7. Why does `DecodeManager.schedule_next_batch` sort the running set by uid? (So that, with several GPUs, every worker lists the same orders in the same order.)
8. An order is aborted while it is the half-read `ChunkedReq` of a prefill. What does `PrefillManager.abort_req` return, and why does it matter? (Its `chunked_req`; that object holds a peg and a pinned cache handle that must be freed.)
