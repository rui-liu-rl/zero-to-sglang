# Part II Chapter 8 RadixAttention and Prefix Caching (Code Walkthrough): ELI5 companion

Original: [Chapter8_RadixAttention-and-Prefix-Caching_code.md](../../course-material/eng/part2/Chapter8_RadixAttention-and-Prefix-Caching_code.md). Same headings, same numbers. Concept half: [Chapter8_RadixAttention-and-Prefix-Caching.md](./Chapter8_RadixAttention-and-Prefix-Caching.md). Cast and picture legend: [../README.md](../README.md).

Files named here carry ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang` (paths relative to `python/minisgl/`): `kvcache/radix_cache.py` (has a worked tree example in its header), `kvcache/__init__.py`, `kvcache/base.py`, `kvcache/mha_pool.py`, `scheduler/cache.py`, `scheduler/prefill.py`, `scheduler/scheduler.py`.

Run the traces yourself: `python3 -I eli5/tools/ch8_radix_toy.py`. It is a plain-Python copy of the tree and the cache bookkeeping, with a clock instead of `time.monotonic_ns()`.

## The one-minute version

**Kid version.** Four objects, four jobs:

| Object | File | Job in one line |
|---|---|---|
| `RadixPrefixCache` | `kvcache/radix_cache.py` | The **index**: words → pantry slots. `match_prefix`, `insert_prefix`, `lock_handle`, `evict` |
| `MHAKVCache` | `kvcache/mha_pool.py` | The **pantry shelf**: the actual K and V numbers. Knows nothing about orders |
| `CacheManager` | `scheduler/cache.py` | The **clerk**: free list, hands out pages, calls the index, takes pages back |
| `PrefillAdder` | `scheduler/prefill.py` | The **admission clerk**: asks the cache clerk if a new order fits |

```text
   new order ─▶ match_req ─▶ lock ─▶ allocate_paged (only the missing words) ─▶ forward writes K/V ─▶ cache_req (insert)
   ... decode ... ─▶ finished: cache_req(finished=True) ─▶ unlock + free duplicates/tail ─▶ later, evict when short
```

## 1 Learning Objectives

| Goal | Where |
|---|---|
| Locate prefix cache, page table, pool; how they cooperate | Sections 2, 3 |
| One request's lifecycle, line by line | Section 4 |
| What each core operation does | Sections 4.1 to 4.4 |

## 2 Locating the source code

| File | Purpose | Kid gloss |
|---|---|---|
| `kvcache/base.py` | Abstract interfaces for prefix cache and pool | Job descriptions |
| `kvcache/radix_cache.py` | The radix tree, handles, match, insert, lock, evict | The index |
| `scheduler/cache.py` | Free list, page table writes, glue to the tree | The clerk |
| `scheduler/prefill.py` | Match before batching, lock, write the matched part, chunk | Admission |
| `scheduler/table.py` | Pegs: rows of `token_pool` / `page_table` | Peg board |
| `scheduler/decode.py` | Decode set, `inflight_tokens` | The "still eating" list |
| `scheduler/scheduler.py` | Build batches, allocate pages, reclaim on finish or cancel | Head waiter |
| `kvcache/mha_pool.py` | K and V tensors, layer by layer | Shelf |
| `kernel/csrc/jit/store.cu` | CUDA kernel that writes K/V into slots | The hand that stocks the shelf |
| `kernel/csrc/src/radix.cpp`, `kernel/radix.py` | C++ speed-up for comparing token runs | Fast eyes |

When reading, ask of every function: **is it maintaining a mapping, or touching real K/V numbers?**

## 3 Distinguishing the mapping table from the actual KV storage

| Component | Stores | Does NOT store |
|---|---|---|
| **Radix tree** | Word ids → slot numbers; tree shape; `ref_count`; timestamps | Any K or V number |
| **`MHAKVCache`** | The K and V numbers, shape `(2, layers, pages, page_size, heads, head_dim)` | Anything about orders or words |

```text
   tree node:  key [900 10 11 12]   value [0 1 2 3]           ◀── an address list
                                           │
                                           ▼
   MHAKVCache._k_buffer[layer][slot 0..3] , _v_buffer[layer][slot 0..3]   ◀── the notes
```

The pool's three methods (`k_cache`, `v_cache`, `store_kv`) are walked in the [chapter 4 code companion](./Chapter4_KV-Cache-Optimization_code.md). `store_kv` receives `out_loc`, the slot for each new word, produced by the scheduler. A **cache hit means nothing new is written for the matched words**: attention reads their slots through the order's row of the peg board.

**Core relationship in one line:** Radix tree = index (words → slots). `MHAKVCache` = the memory that holds the numbers.

## 4 Walking through the source code for a cache hit

### 4.1 Path-compressed nodes: RadixTreeNode

```python
@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    return RadixPrefixCache(device=device)

def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](device)
```

*Line by line:* a **Registry** (the menu board from `utils/registry.py`) maps the name `"radix"` to a function that builds the index. `--cache-type radix` (the default) picks it; `"naive"` picks the do-nothing index. `CacheManager.__init__` calls `create_prefix_cache(device, type)`.

```python
class RadixTreeNode:
    def __init__(self, key_fn, tic=None):
        self.children = {}
        self._parent = None
        self.ref_count = 0
        self.timestamp = tic or time.monotonic_ns()

    def set_key_value(self, key, value):
        assert len(key) == len(value)
        self._key = key      # token-id segment
        self._value = value  # corresponding physical-slot indices
        self._length = len(key)
```

| Field | Kid words |
|---|---|
| `children` | Branches below, found by the first word (page) of their key |
| `_parent` | The branch above (`None` for the root) |
| `ref_count` | How many orders lean on this branch; 0 = may be evicted if a leaf |
| `timestamp` | Last-used time in nanoseconds; **older leaves go first** |
| `_key` | The run of word ids, a **CPU** tensor (comparisons happen on the CPU) |
| `_value` | The slot number for each word, a **GPU** tensor |
| `assert len(key) == len(value)` | One address per word, always |

The root has an empty key and `ref_count = 1` ("always protected").

```python
def split_at(self, pos):
    parent = self.parent
    new_node = RadixTreeNode(self.key_fn, self.timestamp)
    new_node.set_key_value(self._key[:pos], self._value[:pos])
    new_node.set_parent(parent)
    new_node.ref_count = self.ref_count   # Inherit the reference count

    self.set_key_value(self._key[pos:], self._value[pos:])
    self.set_parent(new_node)
    return new_node
```

| Line | Kid words |
|---|---|
| `new_node = RadixTreeNode(...)` | Make the **upper half**, copying this node's timestamp |
| `set_key_value(self._key[:pos], self._value[:pos])` | It takes the first `pos` words and their slots |
| `new_node.set_parent(parent)` | Hang it where this node was. **`set_parent` replaces** the parent's dictionary entry for that first word, so the parent now points to the upper half |
| `new_node.ref_count = self.ref_count` | Whoever was leaning on the whole branch also leans on its upper half |
| `self.set_key_value(self._key[pos:], ...)` | This node keeps the **lower half** |
| `self.set_parent(new_node)` | Hang the lower half under the upper half |

**Worked example.** Node key `[10, 11, 12, 13]`, slots `[a, b, c, d]`. `split_at(2)`:

```text
   before:  parent ── [10 11 12 13]  (a b c d)
   after:   parent ── [10 11]  (a b)  ── [12 13]  (c d)       ← the returned node is [10 11]
```

No K/V moves; the node splitting is index surgery. The order of the two `set_parent` calls matters: the first one replaces the parent's old entry, the second hangs the lower half beneath the new upper one.

### 4.2 Longest-prefix matching: deriving the reusable range from input_ids

**The dictionary key.**

```python
def _get_key_fn(page_size):
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())
```

For `page_size = 1` a branch is found by its first word id (a Python `int`); for bigger pages by the tuple of its first page of ids (tuples can be dictionary keys). *Consequence (toy Part 4):* with page size 4, `[1,2,3,4,...]` and `[1,2,3,9,...]` share nothing, because their first pages differ; with page size 1 they share 3 words.

**The walk.**

```python
def _tree_walk(self, input_ids):
    prefix_len = 0
    node = self.root_node
    tic = time.monotonic_ns()

    while prefix_len < len(input_ids):
        child = node.children.get(self.key_fn(input_ids[prefix_len:]))
        if child is None:
            return node, prefix_len
        node = child

        match_len = node.get_match_len(input_ids[prefix_len:])
        match_len = align_down(match_len, self.page_size)
        prefix_len += match_len

        if match_len != node.length:  # Stop in the middle of a node → split it
            node = node.split_at(match_len)
            node.timestamp = tic
            return node, prefix_len

        node.timestamp = tic  # Full-node hit: refresh the LRU timestamp

    return node, prefix_len
```

| Line | Kid words |
|---|---|
| `tic = time.monotonic_ns()` | One clock reading for the whole walk |
| `while prefix_len < len(input_ids)` | Keep going while there are unmatched words |
| `node.children.get(key_fn(input_ids[prefix_len:]))` | Is there a branch starting with the next word (page)? |
| `child is None → return node, prefix_len` | No: stop here; we matched `prefix_len` words |
| `match_len = node.get_match_len(...)` | How many words of that branch agree with the order (C++ comparison) |
| `align_down(match_len, page_size)` | Only whole pages count (no-op for page size 1) |
| `match_len != node.length` | Matched only part of the branch → **split it** and stop at the new upper half |
| `node.timestamp = tic` | Fully matched branch: mark as just used and go deeper |

The same `tic` stamps every node on the path, so a path ages together and gets evicted from the tail inwards, not torn in the middle.

**Tiny trace (toy Part 3).** Tree: `S = [900,10,11,12]`. Order prompt `[900,10,11,12]`: `match_prefix(prompt[:-1])` = ids `[900,10,11]` → root child `900` found → `get_match_len = 3 < 4` → **split** into `[900,10,11]` and `[12]`. The tree changed shape even though we only asked a question (quirk **Q6** in the notes).

```python
def match_prefix(self, input_ids):
    node, prefix_len = self._tree_walk(input_ids)
    return MatchResult(RadixCacheHandle(prefix_len, node))
```

The **handle** is just `(cached_len = prefix_len, node)`. Its slot list:

```python
def get_matched_indices(self):
    node = self.node
    value_list = []
    while not node.is_root():
        value_list.append(node.value)
        node = node.parent
    value_list.reverse()
    return torch.cat(value_list)
```

Walk up to the root collecting each branch's slots, reverse to root-first order, glue together. Each node only knows its own segment, so the answer exists only by walking.

**Even with no hit**, `match_prefix` runs and returns `cached_len = 0`. The order still gets inserted into the tree later, after its own prefill.

### 4.3 From a match result to an executable Batch: allocating pages and locking

The admission loop (`PrefillManager.schedule_next_batch`), the chunking (`_add_one_req`), and `_try_allocate_one` are the subject of the [chapter 6 code companion](./Chapter6_Continuous-Batching-and-the-Scheduler_code.md), section 4. Here is only the cache side.

**`CacheManager.match_req`**

```python
def match_req(self, req):
    input_len = req.input_len
    return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])
```

It leaves out the **last** prompt token. Why (toy Part 3): if the whole prompt matched, `extend_len = input_len - cached_len = 0`: nothing would go through the model, so there would be no logits, hence no first output word. Matching `[:-1]` guarantees at least one word to compute. `Req` also asserts `cached_len < device_len`.

**`available_size`, and the two checks around `lock`.**

```python
@property
def available_size(self) -> int:
    return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size
```

= unused tree notes (can be thrown out on demand) + free pages. `lock` moves the matched path from evictable to protected, so `available_size` **shrinks**.

**Tiny numbers (a case the second check exists for).** Free list empty; the tree holds `S` (4 words, unused → evictable 4). An order matches `S` and needs 3 new words (no output reserved, nothing else promised):

| Moment | Calculation | Verdict |
|---|---|---|
| Before lock | available = evictable 4 + free 0 = 4; need 3 → `3 + 0 <= 4` | pass |
| After `lock(handle)` | `S` is protected: evictable 0 → available = 0; `3 + 0 > 0` | **fail** → `unlock`, refuse |

Without the second look the order would be admitted by counting notes it is itself about to pin, and the later allocation would find nothing to evict.

**`allocate_paged`** (run just before the forward pass, once for the whole tray):

```python
def allocate_paged(self, reqs):
    needed_pages = 0
    allocation_info = []
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            needed_pages += last_page - first_page
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

| Line | Kid words |
|---|---|
| `first_page = div_ceil(cached_len, page_size)` | First page index this order does **not** already have (the cached part is already in its row) |
| `last_page = div_ceil(device_len, page_size)` | One past the last page it needs for the words it is about to write |
| `if last_page > first_page` | Often 0: a decode step usually stays inside the current page |
| `needed_pages += ...`, `allocation_info.append(...)` | Add up needs for the whole tray; remember (peg row, first page, last page) per order |
| `_allocate(needed_pages)` | One call to take that many pages from the free list |
| `_page_to_token(...)` | Expand each page start into its `page_size` consecutive slot numbers (`[8, 20] → [8..11, 20..23]` at page size 4; unchanged at page size 1) |
| `_write_page_table(...)` | Build the (row, position) lists in pinned memory, copy once to the GPU, do **one** assignment `page_table[rows, positions] = slots` |

```python
def _allocate(self, needed_pages):
    if needed_pages > (free_pages := len(self.free_slots)):
        evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
        self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
        ...
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated
```

| Line | Kid words |
|---|---|
| `if needed_pages > free_pages` | Not enough loose pages |
| `prefix_cache.evict(shortfall * page_size)` | Throw out old unused notes, oldest leaves first; ask only for the shortfall (in slots) |
| `evicted[::page_size]` | Keep each page's first slot as the page's id |
| `torch.cat([free_slots, ...])` | The evicted pages join the free list |
| `allocated = free_slots[:needed_pages]` | Take from the front |

**Tiny trace (toy Part 2, step 7b).** Free list `[8, 9]`, R2 needs 3 words. Shortfall 1 → `evict(1)` removes the whole leaf `[31, 32]` (slots `[6, 7]`) and returns both → free list `[8, 9, 6, 7]` → allocate `[8, 9, 6]` → free list `[7]`.

**Notes on who frees what.** `RadixPrefixCache.evict` only **detaches** nodes from the tree and returns their slots. The caller (`CacheManager._allocate`) puts them back on the free list. `free_slots` always holds page-aligned starting slots.

**`cache_req`: writing the result back.**

```python
def cache_req(self, req, *, finished):
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

| Line | Kid words |
|---|---|
| `insert_ids = input_ids[:cached_len]` | The words whose notes now exist |
| `page_indices = page_table[row, :cached_len]` | Their slot numbers, from this order's peg row |
| `insert_prefix(...)` | File them. Returns `cached_len` (how many leading words the tree **already had**) and a new ticket covering everything inserted |
| `self.unlock(old_handle)` | Release the borrowed beginning. Done **after** the insert so the tree can't evict it in between |
| `self._free(page_indices[old_handle.cached_len : cached_len])` | Words the order computed itself that someone else filed first meanwhile: **duplicates**, return them (else they'd leak) |
| `if finished: self._free(page_indices[new_handle.cached_len :])` | Order over: return the part that can't be filed (partial last page) |
| `else: ... self.lock(new_handle)` | Order continues: swap to the longer ticket and keep it pinned for decode |

**The three groups on an order's strip of words** (from the comment in `scheduler/cache.py`):

```text
   |<-- old_handle.cached_len -->|<------------- computed by THIS order ------------->|
   |   borrowed from the tree    |  duplicates  |  newly filed (new leaf)  |   tail    |
                                 ^ old.cached_len  ^ cached_len        ^ new.cached_len
   duplicates [old.cached_len : cached_len]  → freed now
   newly filed [cached_len : new.cached_len] → now owned by the tree
   tail [new.cached_len : ]                  → freed only if the order is finished
```

**Tiny trace (toy Part 2).** R1 prefill done (`finished=False`): the tree already had 4 words; a new leaf `[31, 32]→[6, 7]` is added; old ticket unlocked, new one locked → protected 6. R1 finishes (`finished=True`): everything is already in the tree, so duplicates `[]` and tail `[]`; unlock → evictable 6.

**Insert in the tree.**

```python
def insert_prefix(self, input_ids, indices):
    insert_len = align_down(len(input_ids), self.page_size)
    input_ids, indices = input_ids[:insert_len], indices[:insert_len]
    node, prefix_len = self._tree_walk(input_ids)
    if prefix_len != insert_len:
        new_node = RadixTreeNode(self.key_fn)
        new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
        new_node.set_parent(node)
        self.evictable_size += new_node.length
        node = new_node
    return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))
```

| Line | Kid words |
|---|---|
| `insert_len = align_down(len, page_size)` | Only whole pages can be filed |
| `_tree_walk(input_ids)` | Walk as far as the tree already agrees (this may split a node) |
| `if prefix_len != insert_len` | There is a remainder to add |
| `RadixTreeNode(...)`, `set_key_value(ids[prefix:], indices[prefix:].clone())` | One new leaf for the remainder, holding a **clone** of the slot list so the tree owns it even if the order's peg row changes |
| `evictable_size += new_node.length` | New notes start unlocked, so they count as evictable |
| `return InsertResult(prefix_len, handle)` | `prefix_len` is the "already there" count used for duplicates |

**Lock and unlock.**

```python
# lock                                         # unlock
while not node.is_root():                      while not node.is_root():
    if node.ref_count == 0:                        node.ref_count -= 1
        evictable -= node.length                   if node.ref_count == 0:
        protected += node.length                       evictable += node.length
    node.ref_count += 1                                protected -= node.length
    node = node.parent                             node = node.parent
```

Walk from the ticket's node to the root; a branch crossing 0→1 moves from evictable to protected, and 1→0 moves back. Each lock is paired with exactly one unlock, so counts return to zero.

**Evict.**

```python
leave_nodes = self._collect_leave_nodes_for_evict()
heapq.heapify(leave_nodes)
...
while evicted_size < size:
    node = heapq.heappop(leave_nodes)
    evicted_size += node.length
    evicted_indices.append(node.value)
    self.evictable_size -= node.length
    parent = node.parent
    del parent.children[self.key_fn(node._key)]
    if parent.is_leaf() and parent.ref_count == 0:
        heapq.heappush(leave_nodes, parent)
return torch.cat(evicted_indices)
```

| Line | Kid words |
|---|---|
| `_collect_leave_nodes_for_evict()` | Depth-first search for leaves with `ref_count == 0` |
| `heapq.heapify(...)` | A min-heap ordered by `timestamp` (the `__lt__` method): **oldest first** |
| `while evicted_size < size` | Keep going until enough slots are freed |
| `heappop` | The least recently used leaf |
| `del parent.children[...]` | Cut it off the tree |
| `if parent.is_leaf() and ref_count == 0: heappush(parent)` | The parent just became a free-to-evict leaf: it joins the queue (the **cascade**) |
| `torch.cat(evicted_indices)` | Return all freed slots |

**Tiny trace (toy Part 5).** A tree `[1 2]` with children `[3 4]` (touched most recently), `[5 6]`, `[7]`. `evict(3)` removes `[5 6]` then `[7]` (oldest two) and returns slots `[6, 7, 10]`. Later `evict(3)` on `[1 2]` → `[3 4]` returns four slots: it popped `[3 4]` (2), the parent `[1 2]` joined the queue, and was popped too (2 more): asked for 3, got 4.

**`check_integrity`** (run when the scheduler is idle): `len(free_slots) + cache_pages == num_pages`, and with `page_size > 1` every free slot must be a multiple of `page_size`. It catches leaked pages: pages that are in neither the free list nor the tree. It runs when the scheduler is idle, i.e. when no running order holds private pages, so the sum must come out exact. (The toy's "books balance" column is the same check, with the private slots of the unrelated running order X added in.)

### 4.4 Decode, normal completion, and cancellation

**During decode** the order's row of the peg board gains one slot per page crossing (`allocate_paged`); nothing is inserted into the tree per step. The engine calls `req.complete_one()` after each launched pass:

```python
def complete_one(self):
    self.cached_len = self.device_len
    self.device_len += 1
```

`cached_len` catches up with everything computed; `device_len` grows by one for the word now owed. The scheduler appends the word's value to `input_ids` with `append_host` (so the tree has the real ids to use as keys later). The order keeps the ticket from its prefill-time insert pinned.

**Completion and cancellation share one exit:**

```python
def _free_req_resources(self, req):
    self.table_manager.free(req.table_idx)
    self.cache_manager.cache_req(req, finished=True)
```

The peg returns; `cache_req(..., finished=True)` files `prompt + answer` (whole pages) and frees duplicates and the unaligned tail. The cancel branch (`AbortBackendMsg`) calls the same function, so there is one cleanup path.

**Why file the answer too.** In a chat the next turn's prompt is "previous prompt + previous answer + new question", so the answer's notes are exactly what the next turn can borrow. (The book's closing note: "the prompt plus output ... is written to the Radix Tree".)

**Batching the frees.** `_process_last_data` wraps its loop in `with self.cache_manager.lazy_free_region():`. Inside, `_free` is swapped for a function that only **collects** page numbers into a basket; on exit the original `_free` is restored and the basket is appended to `free_slots` once (one `torch.cat` instead of many). `finished_reqs` stops the same finished order from being freed twice under overlap scheduling.

## 5 Comparing with real SGLang

The book relates mini-sglang's files to the full engine's. I have not read the full engine's code for these notes, so this table repeats the book's claims and I have not checked them:

| Feature | mini-sglang | Full SGLang (per the book) |
|---|---|---|
| Radix / prefix cache | `kvcache/radix_cache.py` | `srt/mem_cache/radix_cache.py`, moving to a unified radix cache, with variants such as hierarchical cache |
| Request-side cache coordination | `scheduler/cache.py` | split across `srt/mem_cache/` and `srt/managers/scheduler.py` |
| Scheduling policy | First come first served: `pending_list` order; stop at the first that doesn't fit | `srt/managers/schedule_policy.py` (prefix-aware and unaware policies) |
| KV storage and allocation | `kvcache/mha_pool.py` and friends | `srt/mem_cache/pool/`, `srt/mem_cache/storage/` |

The same five questions guide you through either codebase: where is the match produced; when is it locked; where is the missing part allocated; who writes it into the index; who releases it at the end?

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** The index finds a beginning and gives a ticket. The clerk locks it, allocates only the missing words (evicting old unused leaves if the shelf is short), the chef writes only those, and after the pass the result is filed in the index. At the end the order unlocks, returns duplicates and the partial tail, and its filed words stay until needed elsewhere.

### 6.2 Exercises

**1. Why does `CacheManager.match_req` receive only `input_ids[:input_len - 1]`? What if the whole prompt could hit?**

<details><summary>Answer</summary>

Prefill must produce the logits for the first answer word, so at least one prompt word must go through the model. With a full hit, `extend_len = 0`: nothing computed, no logits (toy: `cached_len 4, extend_len 0`). Matching `[:-1]` guarantees `extend_len >= 1` and keeps `cached_len < device_len`.

</details>

**2. Why does `_try_allocate_one` check `available_size` before and after `lock(handle)`?**

<details><summary>Answer</summary>

`lock` turns evictable notes into protected ones, shrinking `available_size`. The numbers above: before lock available 4 (need 3, ok), after lock available 0 (not ok). The first check is a cheap filter; the second is the real one.

</details>

**3. Why does `get_matched_indices` walk up to the root and reverse?**

<details><summary>Answer</summary>

Each node stores only its own segment of slots. From the matched node you can only go up through `parent`, so the segments are collected leaf-to-root and reversed into root-to-leaf order before concatenating.

</details>

**4. In `cache_req`, what does `page_indices[old_handle.cached_len:cached_len]` represent? What if it isn't released?**

<details><summary>Answer</summary>

Slots this order computed itself for words that another order filed in the tree in the meantime (duplicates). Not freeing them leaks them: they'd be in neither the free list nor the tree, so `check_integrity` would find `free + cache != num_pages`.

</details>

**5. Why must eviction start from unreferenced leaves, and who returns the pages to the free list?**

<details><summary>Answer</summary>

An inner node is the beginning of everything below it; removing it would break all those paths. A node with `ref_count > 0` is in use. `RadixPrefixCache.evict` only detaches nodes and returns slot numbers; `CacheManager._allocate` appends them (`evicted[::page_size]`) to `free_slots`.

</details>

**Extra drills (not in the book).**

6. `split_at` is called on a node with `ref_count = 2`. What are the `ref_count`s of the two halves? (Both 2: the upper half copies it; the lower keeps it.)
7. After R1 finishes in the toy, `evictable = 6`. Which slots are those and why isn't any freed yet? (`S` = 0..3 and R1's leaf = 6, 7. They are filed in the tree, unlocked; they only return to the free list if an allocation needs them.)
8. With `lazy_free_region` active, a `_free` call happens. When does `free_slots` actually grow? (When the `with` block exits.)
