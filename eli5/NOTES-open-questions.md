# Open questions and suspected upstream quirks (mini-sglang)

Things noticed while reading `mini-sglang` commit `9a91cfa` for the ELI5 comments. **None of these has been verified by running mini-sglang itself** (this Mac cannot run it: Linux + CUDA only). The one exception in spirit is Q8, which I reproduced with a standard-library replay of the relevant methods. Each has the reasoning so you can check it on a GPU box or ask upstream. Some may make good small first contributions; read the Part 0 ethics lesson first (reproduce, measure, write it in your own words).

Legend: **Hypothesis** = my reading of the code, could be wrong. **Quirk** = harmless but surprising. **Doc drift** = book vs source.

## Hypotheses worth testing

### H1. Duplicate-prefix path in `CacheManager.cache_req` may leave a running request pointing at freed pages
- **Where:** `scheduler/cache.py`, `cache_req`, the line `self._free(page_indices[old_handle.cached_len : cached_len])`.
- **Reading:** when two requests with the same prompt are prefilled before either has inserted its prefix (for example in the same batch), the second `insert_prefix` returns `cached_len > 0` ("already in the cache"). The code frees the second request's own pages for that region, which is right for avoiding a leak. But the request's row in `page_table` still holds those (now freed) slot numbers, and for an unfinished request (`finished=False`) it keeps decoding while reading K/V through that row. If the freed pages are handed to another request and overwritten, attention would read wrong K/V.
- **What would settle it:** a test that submits two identical long prompts in one prefill batch and compares outputs against running them one at a time; or a code path elsewhere that repoints the row at the cache's indices (I did not find one).
- **Confidence:** medium-low. There may be an invariant I missed.

### H2. Possible small leak in the detokenizer state under overlap scheduling
- **Where:** `tokenizer/detokenize.py` (`decode_map`), `scheduler/scheduler.py` (`_process_last_data`, `finished_reqs`).
- **Reading:** with overlap scheduling, a request that finished by EOS in batch n-1 may already be in batch n, so one extra token for it reaches the detokenizer with `finished=False`. The detokenizer then creates a fresh `DecodeStatus` for a uid whose old state was already deleted, and nothing ever deletes the new one. The front desk ignores the reply (the uid is gone), so users see nothing, but each EOS-finished request might leave one small dict entry behind.
- **Confidence:** medium-low. Depends on whether the extra token can have `finished=False` (it can unless it is also EOS).

### H3. An abort that lands while the request's last batch is in flight may free its table row and pages twice
- **Where:** `scheduler/scheduler.py` (`overlap_loop`, `_process_one_msg`, `_process_last_data`, `_free_req_resources`), `scheduler/table.py` (`free`), `scheduler/cache.py` (`cache_req`).
- **Reading:** in `overlap_loop` the order inside one turn is (1) handle new mail, including an `AbortBackendMsg`, then (2) launch batch n, then (3) process batch n-1's results. An abort for request R handled in step 1 pops R from `running_reqs` and calls `_free_req_resources(R)` (row back to the table, pages back via `cache_req(finished=True)`). But R is still inside `last_data` (batch n-1), so step 3 touches R again:
  - if R's token in batch n-1 ends it (EOS or length), `finished and req not in self.finished_reqs` is true (aborts never add R to `finished_reqs`), so `_free_req_resources(R)` runs a **second** time. `TableManager.free` is a bare `append`, so the row would sit in `_free_slots` twice and could be handed to two different requests; `cache_req(finished=True)` would free the same pages again.
  - if batch n-1 was R's non-chunked prefill and R does not end, the `elif batch.is_prefill` branch calls `cache_req(R, finished=False)` on a request that was already released.
- **Window:** a single scheduler turn, so it needs an abort and a natural finish (or a prefill) to coincide. Rare, which is why it may never have been seen.
- **What would settle it:** a GPU stress test that aborts many short requests at random times while others run, with `check_integrity()` (the page-count self-check in `CacheManager`) enabled afterwards; or a unit test that calls `_process_one_msg(AbortBackendMsg)` and then `_process_last_data` with a batch that contains the same request.
- **Confidence:** medium-low. I read the code but did not run it, and there may be a guard I missed.

### H4. The KV page count is sized from the MAX free memory across TP ranks, not the MIN
- **Where:** `engine/engine.py`, `Engine.__init__` (`init_free_memory = self._sync_get_memory()[1]`) and `_determine_num_pages` (`new_free_memory = self._sync_get_memory()[1]`).
- **Reading:** `_sync_get_memory` returns `(min_free, max_free)` after an all-reduce, and it raises if they differ by more than 2 GiB. Both call sites take index `[1]`, the **maximum**. Every rank gets the same number, which is what the shared scheduling logic needs (all ranks must agree on `num_pages`). But if rank 3 has up to 2 GiB less free memory than rank 0, the pool is sized from rank 0's figure, and rank 3 is asked to allocate the same pool. The 10% slack of the default `memory_ratio = 0.9` probably absorbs that.
- **What would settle it:** start with one GPU deliberately holding 1.5 to 2 GiB of unrelated memory and check whether the weakest rank runs out of memory at start-up or under load.
- **Confidence:** low. It may be deliberate (use the largest consistent figure and rely on the slack).

## Quirks (probably harmless)

- **Q1.** `server/api_server.py`, `shell_completion`: `background=BackgroundTask(lambda: _abort)` returns the function without calling it, so the abort never runs. Probably intended `lambda: _abort()` or passing the coroutine function directly.
- **Q2.** `server/api_server.py`, `/v1/chat/completions`: request fields `n`, `stop`, `presence_penalty`, `frequency_penalty` are accepted and ignored; `usage` token counts are always 0; `finish_reason` is always `"stop"` (even if cut by `max_tokens`).
- **Q3.** `scheduler/scheduler.py`, `_process_one_msg`: a request whose prompt is longer than the model's maximum length is dropped with a log warning and **no reply is ever sent**, so an HTTP client waits indefinitely.
- **Q4.** `kvcache/mha_pool.py`: the `MHAKVCache` docstring says "Base class for key-value caches" (copy-pasted from the base class).
- **Q5.** `scheduler/utils.py`: `ScheduleResult` is defined but not used anywhere in the repo.
- **Q6.** `kvcache/base.py`: the docstring of `match_prefix` says it "will not modify the cache", but `RadixPrefixCache._tree_walk` can *split* a node and update timestamps while matching. The cached content is unchanged, but the tree structure and LRU order are.
- **Q7.** `server/api_server.py`, `OpenAICompletionRequest.temperature` defaults to 1.0 while `SamplingParams.temperature` defaults to 0.0; a client that omits `temperature` gets random sampling over HTTP but greedy through `/generate` and `LLM.generate`.

- **Q8.** `server/api_server.py`, `FrontendManager.wait_for_ack`: its trailing `del self.ack_map[uid]` / `del self.event_map[uid]` never run, because every caller leaves the loop when it sees `finished` and a closed async generator does not execute ordinary statements after its loop. **Reproduced** with a stdlib-only replay of the three methods (`new_user`, `listen`, `wait_for_ack`): after break-on-finished the uid stays in both maps; after draining the generator it is removed. My measurement of one leftover entry (empty list, `asyncio.Event`, two dict slots) was about 1 KB, so roughly 1 GB per million requests. The book already describes the gap (code chapter 5, section 5.1); this just confirms and sizes it. A fix would be `try/finally` inside `wait_for_ack` (so closing the generator also cleans up); the same toy (Test 7) shows that version removes the ticket even when the caller still breaks on `finished`.

- **Q9.** `attention/fi.py`: the assert text "Currently only page_size=1 is supported." and the field comment "currently only support page_size=1" read like a limitation of the backend, but they describe FlashInfer's own paging, which this backend never uses. `FIMetadata.page_size` is always the constant 1, `forward()` re-views the pool as one-token pages (`_flatten_cache`), and `indices` are per-token slot numbers, so any pool `--page-size` works with `fi` (`benchmark/offline/bench.py` uses 256). I misread this myself at first and corrected my earlier comments.
- **Q10.** `moe/fused.py`, `moe_align_block_size` docstring: the example says `num_experts = 4` but numbers its experts 1 to 4. With zero-based ids the example would be 0 to 3. I replayed the example in plain Python and the printed `sorted_token_ids` is right for 1-based ids.
- **Q11.** `kernel/csrc/include/minisgl/tensor.h`, `SymbolicSize`: 0 doubles as "not set" (`has_value()` is `m_value != 0`), so a real size of 0 can never be bound, and `unwrap()` on it says "Size value is not set". Kernels here never receive empty tensors, so it has no effect today.
- **Q12.** `kernel/tensor.py`: `test_tensor(...) -> int`, but the C++ function it calls (`test` in `tensor.cpp`) returns `void`.

## Doc drift (book vs source)

- **D1.** Part II ch.2 code walkthrough process table lists `minisgl-tokenizer-0` as a default process. By default (`--num-tokenizer 0`) there is none; `minisgl-detokenizer-0` also tokenizes.
- **D2.** Part II ch.2 code walkthrough shows `normal_loop` as the scheduler loop. The default is `overlap_loop`; `normal_loop` runs only if `MINISGL_DISABLE_OVERLAP_SCHEDULING` is set.
- **D3.** Part I ch.2 §5.2 refers to "Section 2.3" (leftover numbering; should be §4.2-4.3) and analyses decode at batch size 1 only without mentioning that batching is the main lever.
- **D4.** Part I ch.4 labels binary units as "MB"/"GB" (they are MiB/GiB).
- **D5.** Part I ch.3 vs ch.2 use different A100 memory bandwidths (2 TB/s for the 80 GB model vs 1.6 TB/s for the 40 GB model) without saying so.
- **D6.** Part II ch.7 §2.2: the five free holes (512, 256, 1,024, 768, 1,024 MiB) sum to 3,584 MiB = 3.50 GiB, not "3.60 GiB". The argument (largest hole 1,024 MiB < 1.5 GiB needed) is unchanged.
- **D7.** Part II ch.5 code: the book's `listen` excerpt names the loop variable `reply`; the source uses `msg`. Part II ch.8 §3.2: the book's `RadixTreeNode.__init__` shows `time.monotonic()`; the source uses `time.monotonic_ns()` (and a class attribute `counter`). Cosmetic.
- **D9.** Part II ch.9 code walkthrough, §3.1: the example says "run Qwen3-32B on three RTX 5090 GPUs" but the command uses `--tp-size 2`. (Both flag spellings, `--tensor-parallel-size` and `--tp-size`, do exist in `server/args.py`. Also worth remembering: the number of query heads must divide evenly by the TP size, and the KV heads must either divide evenly or the TP size must be a multiple of the KV head count, so TP = 3 is rarely valid for real models.)
- **D10.** Part II ch.9 (exercise 3 and the note after the loader excerpt) and the docstring of `load_weight` talk about peak **CPU** memory. The code opens the safetensors file with `device=str(device)`, so each tensor is read straight onto the **GPU**; the temporary full (unsharded) tensor `raw` lives on the GPU until `del raw`. The streaming argument still holds, only the memory in question is GPU memory.
- **D8.** Part II ch.6: the book's step-by-step trace is for `normal_loop`; the default `overlap_loop` notices a finished request one turn later (so a freed slot is reused one pass later). Not an error (the book says which loop it uses) but easy to forget.
