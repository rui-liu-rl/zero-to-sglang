# Part II Chapter 4 KV Cache Optimization (Code Walkthrough): ELI5 companion

Original: [Chapter4_KV-Cache-Optimization_code.md](../../course-material/eng/part2/Chapter4_KV-Cache-Optimization_code.md). Same headings, same numbers. Concept half: [Chapter4_KV-Cache-Optimization.md](./Chapter4_KV-Cache-Optimization.md). Cast and picture legend: [../README.md](../README.md).

Files named here carry ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang` (paths relative to `python/minisgl/`).

## The one-minute version

**Kid version.** Four little jobs hide inside "use a notepad", each done by a different object:

| Question | Who answers | Kid version |
|---|---|---|
| WHERE are the numbers? | `MHAKVCache` (`kvcache/mha_pool.py`) | One giant pantry shelf with numbered slots; knows nothing about orders |
| WHICH SLOT is for this word of this order? | `page_table` + `Batch.out_loc` | The peg board: row = order, column = word position, cell = slot number |
| CAN another order reuse these notes? | `BasePrefixCache` (`kvcache/`) | The shared index of "who already wrote notes for this beginning" |
| WHO hands out and takes back slots? | `CacheManager` (`scheduler/cache.py`) | The pantry clerk with the list of free shelves |

```text
   token word #5 of order on peg 3 ──▶ page_table[3, 5] ──▶ slot 42 ──▶ _kv_buffer[K/V][layer][slot 42] = the actual numbers
   (the order)                          (peg board)          (address)      (the pantry)
```

## 1 Learning Objectives

| Goal | Where |
|---|---|
| Where new K/V is stored on the GPU | Sections 4.1, 5.3 |
| How a request finds the K/V of one of its tokens | Section 4.3 (the peg board) |
| When a finished segment can be reused by another request | Sections 3.2, 5.1, 5.4 |

## 2 Source Modules Involved in KV Cache Reads and Writes

| File | Responsibility | Kid gloss |
|---|---|---|
| `kvcache/base.py` | Abstract interface of the physical pool | The pantry's job description |
| `kvcache/__init__.py` | Creates the pool, registers cache policies | The hiring office |
| `kvcache/mha_pool.py` | Stores K/V tensors for all layers, per-layer views | The shelf itself |
| `engine/engine.py` | Creates the pool, `page_table` and other runtime objects at start | Builds the shelf and the peg board |
| `core.py` | `Req` records progress and the span to write now | The order slip |
| `scheduler/cache.py` | Allocates / frees pages, manages prefix cache | The clerk |
| `scheduler/prefill.py` | Matches and locks reusable prefixes | The admission desk |
| `scheduler/scheduler.py` | Builds batches, computes write locations, cleans up | The head waiter |
| `attention/fa.py` (and `fi.py`, `trtllm.py`) | Writes and reads the cache inside attention | The meeting room |

Three steps to remember: (1) the scheduler prepares physical locations for the tokens being computed; (2) attention writes the new K/V there and then reads what it needs; (3) when an order ends or leaves a batch, the system keeps reusable slots and returns the rest.

## 3 K/V Storage, Position Mapping, and Prefix Reuse

| Component | Problem | Core data |
|---|---|---|
| `MHAKVCache` | Where the floating-point K/V tensors live | `_kv_buffer`, per-layer views |
| `page_table`, `Batch.out_loc` | Which slot a token in this pass writes to | `int32` slot numbers |
| `BasePrefixCache` | Can a finished prefix be used by someone else | `BaseCacheHandle`, `cached_len`, matched indices |
| `CacheManager` | Which pages can be handed out / reclaimed | `free_slots` |

`MHAKVCache` stores **only numbers**. It does not know which order owns a slot or which token a slot belongs to. The prefix cache stores **only addresses** (which slots correspond to which tokens). So two orders can reach the same K/V through their own rows of the peg board.

### 3.1 Per-Layer Reads from Physical KV Cache

```python
# kvcache/base.py (excerpt)
class BaseKVCachePool(ABC):
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...
    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...
    @abstractmethod
    def store_kv(self, k, v, out_loc, layer_id) -> None: ...
    # (plus properties: device, dtype, num_layers)
```

*Line by line:* three actions: "give me this layer's Label shelf", "give me this layer's Gift shelf", "write these new K and V rows into these slots of this layer". The three properties say which GPU, which number format, how many layers. The interface has no idea about requests.

### 3.2 How Prefix Cache Policies Reuse Existing K/V Locations

`kvcache/__init__.py` has a factory `create_kvcache_pool(...)` (always an `MHAKVCache` today, with a `TODO: support other variants (e.g. MLA)`) and registers two reuse policies on a **Registry** (the menu board from `utils/registry.py`): `"naive"` and `"radix"`.

| Policy | What `match_prefix` returns | Effect |
|---|---|---|
| `naive` (`kvcache/naive_cache.py`) | Always "nothing matched" (`cached_len = 0`); insert and evict do nothing (evict of >0 raises) | No sharing: every order computes its whole prompt |
| `radix` (`kvcache/radix_cache.py`) | The longest stored beginning that matches | Sharing: chapter 8 |

**They do not change the K/V tensor layout.** Attention reads and writes the same `MHAKVCache` through the same three methods either way.

## 4 Physical KV Storage and Position Mapping

### 4.1 MHAKVCache Stores K/V for All Layers

```python
# kvcache/mha_pool.py (excerpt)
class MHAKVCache(BaseKVCachePool):
    def __init__(self, num_kv_heads, num_layers, head_dim, num_pages, page_size, dtype, device):
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device, dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
```

*Line by line:*

| Line | Kid words |
|---|---|
| `local_kv_heads = div_even(...)` | How many Label/Gift heads THIS GPU keeps notes for (all of them with one GPU) |
| `torch.empty((2, L, pages, page_size, heads, head_dim))` | Build the whole shelf in one go: K and V, every floor, every page, every slot in a page, every head, every number. `empty` = don't clear it, just reserve memory |
| `_k_buffer = _kv_buffer[0]`, `_v_buffer = _kv_buffer[1]` | Two views: the Label half and the Gift half (no copy) |
| `_storage_shape = (pages * page_size, heads, head_dim)` | A flattened view used when writing: "slot number" instead of "page and offset" |

**Reading the shape from the outside in:**

| Index | Meaning |
|---|---|
| `[0]` or `[1]` | Label (K) or Gift (V) |
| `[layer]` | which floor's notes |
| `[page]` | which physical page |
| `[offset]` | which slot inside the page (`0 .. page_size - 1`) |
| `[head]` | which KV head |
| `[number]` | one of the `head_dim` values |

**Example size (computed).** Qwen3-0.6B, `page_size = 1`, 21,000 pages (about what a 4 GB card allows, see the Deploy lesson): shape `(2, 28, 21000, 1, 8, 128)` = 1,204,224,000 numbers x 2 bytes = **2.24 GiB**.

```python
# kvcache/mha_pool.py (excerpt)
def k_cache(self, index: int) -> torch.Tensor:
    return self._k_buffer[index]
def v_cache(self, index: int) -> torch.Tensor:
    return self._v_buffer[index]
```

`_k_buffer[index]` is the layer-`index` slice: shape `[num_pages, page_size, local_kv_heads, head_dim]`. It is a **view** (no data copied). Even if two floors use the same slot number, their numbers live at different `layer` positions, so they never collide.

### 4.2 Mapping Free Pages to Writable Token Slots

`CacheManager.free_slots` is a GPU tensor holding the **first slot of each free page**:

```python
self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
```

With `page_size = 4` and three free pages, `free_slots = [0, 4, 8]`:

| Page start | Expanded token slots |
|---:|---|
| 0 | `[0, 1, 2, 3]` |
| 4 | `[4, 5, 6, 7]` |
| 8 | `[8, 9, 10, 11]` |

```python
def _page_to_token(self, pages):
    if self.page_size == 1:
        return pages
    offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
    return (pages.unsqueeze(1) + offsets).flatten()
```

*Line by line:* with `page_size == 1`, a page is a slot, nothing to do. Otherwise: `pages.unsqueeze(1)` turns `[0, 4, 8]` into a column `[[0], [4], [8]]`; adding `offsets = [0, 1, 2, 3]` broadcasts to the 3x4 grid `[[0,1,2,3],[4,5,6,7],[8,9,10,11]]`; `flatten()` reads it row by row. So the clerk **hands out pages** but writes **slot numbers** into the peg board: free-space accounting is per page, addresses are per token.

### 4.3 page_table Maps Logical Positions to Physical Positions

```python
# engine/engine.py (excerpt)
self.num_pages = self._determine_num_pages(init_free_memory, config)
num_tokens = self.num_pages * config.page_size
self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(... num_pages=self.num_pages + 1, ...)  # +1 for dummy page
self.max_seq_len = min(config.max_seq_len, num_tokens)
aligned_max_seq_len = _align_up_32(self.max_seq_len)
self.ctx.page_table = self.page_table = torch.zeros(
    (config.max_running_req + 1, aligned_max_seq_len), dtype=torch.int32, device=self.device)
```

*Line by line:* (1) work out how many pages fit; (2) total token slots `num_tokens`; (3) build the shelf with **one extra page**, the dummy page for CUDA-graph rehearsals; (4) longest allowed sequence: the model's limit or the whole shelf, whichever is smaller; (5) round that up to a multiple of 32 so rows start at tidy memory addresses; (6) the peg board: one row per possible running order plus one spare, `aligned_max_seq_len` columns, all zeros.

Two details:
1. The shelf has `num_pages + 1` pages; the scheduler hands out only the first `num_pages`.
2. A cell holds a **slot number** (a token address), not a page id. The attention backend converts to pages just before calling its kernel.

**Example.** `page_size = 4`, order on peg 3 with 6 words. Its row begins `[40, 41, 42, 43, 16, 17, ...]`: words 0 to 3 live in page `[40..43]`, words 4 to 5 in page `[16..19]` (slots 16 and 17 used, 18 and 19 reserved for the next words). Reading it: `page_table[3, 4] = 16` means "word 4 of order 3 is in slot 16".

## 5 Writing, Reading, and Reclaiming KV Cache for a Request

```python
# core.py (excerpt)
@dataclass(eq=False)
class Req:
    input_ids, table_idx, cached_len, output_len, cache_handle
    # device_len = len(input_ids); max_device_len = len(input_ids) + output_len
    extend_len = device_len - cached_len
    def complete_one(self): self.cached_len = self.device_len; self.device_len += 1
```

View an order as a strip of words split in two: the **first part already has notes** (`cached_len` words), the **rest must be computed this pass** (`extend_len = device_len - cached_len`). After a pass, `complete_one()` moves the processed part into the "has notes" region and makes room for the word just sampled. (Worked example table: Part II chapter 1.)

### 5.1 Prefill Matches and Uses Existing Prefixes

The last prompt token can never be skipped (its logits give the first answer word), so the lookup omits it:

```python
# scheduler/cache.py (excerpt)
def match_req(self, req: PendingReq) -> MatchResult:
    input_len = req.input_len
    assert input_len > 0
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

Then `PrefillAdder._try_allocate_one` applies the admission rule. **Worked numbers:** the pantry reports `available_size = 1000` slots; running decodes have promised `reserved_size = 400`; a new order has 300 prompt tokens, of which 100 are cached; it may produce 128.

```text
   extend_len     = 300 - 100            = 200
   estimated_len  = 200 + 128            = 328
   first check    : 328 + 400 = 728  <=  1000                      → looks OK
   lock the 100 cached tokens → they move from "evictable" to "protected":   available = 900
   second check   : 728 <= 900                                     → admitted
```

Same order when `available_size = 750`: first check passes (728 <= 750), but after locking, `available = 650 < 728` → **unlock and refuse**. That is why the code checks twice.

```python
# scheduler/prefill.py (excerpt)
if cached_len > 0:
    device_ids = self.table_manager.token_pool[table_idx][:cached_len]
    page_entry = self.table_manager.page_table[table_idx][:cached_len]
    device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
    page_entry.copy_(handle.get_matched_indices())
```

*Line by line:* two different things are copied for the borrowed beginning: the **token ids** go into `token_pool` (the model needs them as input), and the **slot numbers** where those words' notes already live go into the peg board row (attention needs them to find the notes). After this, the new order's row starts with the old order's slot numbers: it reads someone else's notes as if they were its own.

**Why lock before use?** If the cache could evict that beginning, its slots might be handed to another order and overwritten while this order still points at them: silent corruption. The lock pins them.

The unmatched part gets only its token ids written now:

```python
_slice = slice(cached_len, cached_len + chunk_size)
device_ids = self.table_manager.token_pool[table_idx, _slice]
device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
```

Its pages are allocated later, together with the rest of the tray (next subsection).

### 5.2 The Scheduler Allocates K/V Locations

```python
# scheduler/scheduler.py (excerpt)
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    self.engine.graph_runner.pad_batch(batch)
    self.cache_manager.allocate_paged(batch.reqs)
    batch.positions = _make_positions(batch, self.device)
    input_mapping = _make_input_tuple(batch, self.device)
    write_mapping = _make_write_tuple(batch, self.device)
    batch.out_loc = self.engine.page_table[input_mapping]
    self.engine.attn_backend.prepare_metadata(batch)
    ...
```

*Order matters:* (1) pad the tray, (2) **give pages and fill the peg board** (`allocate_paged`), (3) compute positions and index lists, (4) read the write addresses **from the filled peg board**.

```python
# scheduler/cache.py (excerpt)
def allocate_paged(self, reqs):
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

*Line by line (page_size = 4):* an order needs pages `ceil(cached_len/4)` up to `ceil(device_len/4)`. A decode step at `device_len = 5` needs page index 1 (positions 4-7) because it is the first word to land there. At `device_len = 6, 7, 8` nothing new is needed. At `device_len = 9` page index 2 is needed. `_allocate` takes pages off the shelf (evicting unused cached notes if short), `_page_to_token` expands them to slot numbers, `_write_page_table` pins them into the right cells in one GPU operation.

Then `out_loc`:

```text
   batch.positions            = logical positions of the tokens computed now          [N]
   input_mapping              = (peg row of each token, position)                     two lists of length N
   batch.out_loc              = page_table[input_mapping]                             [N] slot numbers
```

`out_loc` is *not* the whole peg board and not a whole row: it is the list of **write addresses for the new tokens of this tray**, in the order the model processes tokens. Its length is the sum of every order's `extend_len`.

### 5.3 How Attention Writes New K/V and Reads Historical K/V

```python
# attention/fa.py (excerpt)
def forward(self, q, k, v, layer_id, batch):
    metadata = batch.attn_metadata
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    return _fa_sgl_impl(q=q, k_cache=self.kvcache.k_cache(layer_id), v_cache=self.kvcache.v_cache(layer_id),
                        page_table=metadata.page_table, cache_seqlens=metadata.cache_seqlens, ...)
```

**Write first, then read.** The new tokens' K/V must be in the shelf before the kernel runs so a new token can attend to itself and to its neighbours in the same pass.

```python
# kvcache/mha_pool.py (excerpt)
def store_kv(self, k, v, out_loc, layer_id):
    from minisgl.kernel import store_cache
    store_cache(k_cache=self._k_buffer[layer_id].view(self._storage_shape),
                v_cache=self._v_buffer[layer_id].view(self._storage_shape),
                indices=out_loc, k=k, v=v)
```

*Line by line:* take this floor's K shelf and re-view `[pages, page_size, heads, dim]` as `[pages * page_size, heads, dim]` (a flat list of slots); same for V; then a custom GPU kernel copies row `i` of `k` to slot `out_loc[i]` (and likewise for V). In `kernel/csrc/jit/store.cu`, **one warp (32 threads) copies one token's K row and V row**. Example: `out_loc = [4, 5]` writes the two new tokens into slots 4 and 5 of **this floor only**; other floors write their own K/V for the same words into their own shelves.

The kernel reads by **page**, so `FlashAttentionBackend.prepare_metadata` converts: take every `page_size`-th entry of each peg row (the first slot of each page), then integer-divide by `page_size`:

```python
new_page_table = torch.stack([page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs])
if self.page_size > 1:
    new_page_table.div_(self.page_size, rounding_mode="floor")
```

Example (`page_size = 4`): row `[40,41,42,43, 16,17,18,19]` → every 4th = `[40, 16]` → divide by 4 = pages `[10, 4]`. Internally everything stays in slots; conversion to page ids happens only at the kernel's door.

### 5.4 Reusing and Releasing K/V After a Request Finishes

```python
# scheduler/cache.py (excerpt)
def cache_req(self, req: Req, *, finished: bool) -> None:
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    self.unlock(old_handle)
    self._free(page_indices[old_handle.cached_len : cached_len])
    if finished:
        self._free(page_indices[new_handle.cached_len :])
    else:
        req.cache_handle = new_handle
        self.lock(new_handle)
```

*Line by line:* hand the valid region `[0, cached_len)` (words with finished notes, plus their slots) to the prefix cache; it tells you how much was *already* there and gives a ticket for the stored path. Release the old pin. Free the slots in `[old.cached_len, cached_len)` (duplicates: someone else stored the same words meanwhile). If the order is over, also free the tail the index could not take (partial page). If it is still running, swap in the new ticket and pin it.

Three kinds of slots, three fates:

| Kind | Fate |
|---|---|
| Borrowed beginning (`old_handle`) | Unpinned when this order stops needing it |
| Newly computed whole pages, successfully indexed | Stay on the shelf for future orders with the same beginning |
| Duplicates, and the tail that doesn't fill a page (finished orders) | Returned to `free_slots` |

**Returning a slot does not erase the numbers.** It only says "no longer yours"; the next owner overwrites them. (See `NOTES-open-questions.md`, H1, for a case where I think this deserves a second look.)

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** Shelf: `_kv_buffer[K/V][layer][page][slot][head][number]`. Peg board: `page_table[order row, word position] = slot`. Each pass: allocate pages, fill the board, read the new words' write addresses (`out_loc`), attention writes new K/V there and reads the old ones through the board. At the end, the index keeps the reusable beginnings and the clerk takes back the rest.

### 6.2 Exercises

**1. What do dimensions 0, 1, 2, 3 of `_kv_buffer` mean? Why is `layer_id` still needed when reading?**

<details><summary>Answer</summary>

0: K or V; 1: layer; 2: physical page; 3: slot within the page. `_k_buffer` already fixes dim 0; `k_cache(layer_id)` fixes dim 1. Each floor has its own K/V for the same word, so the layer must be chosen even when slot numbers coincide.

</details>

**2. With `page_size = 4` and a `free_slots` entry of `8`: which page? What does `_page_to_token` expand it to?**

<details><summary>Answer</summary>

Page index `8 / 4 = 2`; it expands to token slots `[8, 9, 10, 11]`.

</details>

**3. Why call `allocate_paged` before `page_table[input_mapping]`? What if reversed?**

<details><summary>Answer</summary>

`allocate_paged` fills the peg board cells for the new positions. Reading `out_loc` first would read stale or zero cells (the table starts as zeros and rows are reused), so K/V would be written to the wrong slots, possibly another order's notes, silently corrupting them.

</details>

**Extra drills (not in the book).**

4. A decode tray has 3 orders. How long is `batch.out_loc`? (3: one write address per order.)
5. A prefill tray has orders with `extend_len` 5 and 4. How long is `out_loc`? (9.)
6. Why does `store_kv` run before the attention kernel inside `forward`? (So new tokens can attend to themselves and to each other.)
