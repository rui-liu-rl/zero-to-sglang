# Part II Chapter 8 RadixAttention and Prefix Caching: ELI5 companion

Original: [Chapter8_RadixAttention-and-Prefix-Caching.md](../../course-material/eng/part2/Chapter8_RadixAttention-and-Prefix-Caching.md). Same headings, same numbers. The code half is [Chapter8_RadixAttention-and-Prefix-Caching_code.md](./Chapter8_RadixAttention-and-Prefix-Caching_code.md). Cast and picture legend: [../README.md](../README.md).

Runnable toy with every trace below (a plain-Python copy of the tree and the cache bookkeeping): `python3 -I eli5/tools/ch8_radix_toy.py`.

## The one-minute version

**Kid version.** After an order is served, **don't throw its notes away**. Keep them in the pantry, filed in a **tree by the words they were written for**. When a new order starts with the same words as a filed one, skip re-reading those words: borrow the notes, and only read the new part.

```text
   filed in the tree:                           new order: "the cat sat on the mat"
        root                                    walk down:  the ─ cat ─ sat   (matched! notes already exist)
         └─ [the cat]                                         └ still has to read: "on the mat"
              ├─ [sat] ─ [on a rug]
              └─ [ran]
```

Three promises make it safe:

1. **Only a *beginning* can be borrowed** (section 2.2): a word's notes depend on every word to its left.
2. **Borrowed notes are locked** while someone is using them (nobody may throw them out).
3. **Unused notes are thrown out oldest-first, leaves first**, only when the pantry needs space.

RadixAttention does **not** change attention maths. It is the filing system plus a reuse policy.

## 1 Learning Objectives

| Original question | Kid answer (and where) |
|---|---|
| What is left over after a request ends in traditional KV management? | Nothing: the notes are binned with the order (2.1) |
| Why can a prefix be reused but not an arbitrary segment? | Notes depend on everything to the left (2.2, with a computed toy) |
| Why a compressed radix tree, not a word-by-word trie? | One node per branch instead of one per word (3.1) |
| Complete process of match, insert, branch, split? | Section 3.3, with the book's A/B/C run step by step |
| What does a node store, and how do tokens map to pool slots? | A run of words + the pantry slot of each (3.2, 4.1) |
| How do match, schedule, lock, evict, free connect? | The seven-step lifecycle in 4.2 with real counters |

## 2 From KV Cache to Prefix Cache

### 2.1 Limitations of Traditional KV Cache

**Kid version.** In the old way the notes belong to the **order**: made when it arrives, binned when it leaves. If a hundred customers all begin with the same house rules ("You are a helpful assistant ..."), the chef re-reads those rules a hundred times.

**Tiny numbers.** A 1,000-word system prompt, a 50-word question, 100 customers:

| | Words the chef must read |
|---|---|
| Notes binned after each order | `100 x (1000 + 50) = 105,000` |
| Notes kept and shared | first customer `1,050`, then `99 x 50 = 4,950` → **6,000** |

That is **17.5 times** less prefill work. Chat is the same story: in chapter 5 we computed that a 10-turn conversation resends 2,000 words in total but only 380 are new, a factor of about 5.3 that a prefix cache removes.

The change in one sentence: **the notes now belong to the *beginning*, not to the order.** The order merely borrows them. That creates the question of section 2.2.

### 2.2 Core Principle of Prefix Cache

**Kid version.** Each word's notes were written by someone who had **read everything to its left**. Change a word on the left, and every note to its right changes.

```text
   A: [ weather ] [ is ] [ fine ]          B: [ mood ] [ is ] [ fine ]
                                same words "is fine", same seats (1, 2), but A's "is" was written after
                                reading "weather" and B's "is" after reading "mood".  Different notes.
```

**A computed check (my toy, Part 0).** A tiny 2-layer attention model with random fixed weights. For each pair, are the K and V numbers **identical**?

| Pair | Layer 1 | Layer 2 |
|---|---|---|
| `is` in [weather is fine] vs [mood is fine] (same word, same seat, different left context) | identical | **different** |
| `fine` in the same two sentences | identical | **different** |
| `weather` at seat 0 and `is` at seat 1, in [weather is fine] vs [weather is **bad**] (same prefix, different later word) | identical | **identical** |
| `fine` vs `bad` at seat 2 | different | different (different words) |

Read it like this:

- **Floor 1's notes depend only on the word and its seat** (that is why they match). The left context has not been mixed in yet.
- **From floor 2 upward** the input to each floor has already been through meetings with the left neighbours, so the notes carry that context. Real models have dozens of floors, so for reuse you need *all floors' notes* and therefore the *whole* left context to match.
- **A change to the right never touches notes to its left** (the causal mask: a word never looks right). That is why a shared *beginning* is safe to reuse even if the endings differ (the third row).
- **RoPE (seat numbers)** adds a second reason: the same word at a different seat has a different K, so even a matching *middle* segment can't be moved to another position.

**Where it happens.** The lookup happens when an order **enters prefill**. During decode no matching is done: existing notes are used and new ones appended.

**Pause and predict**

1. Two prompts: `[1,2,3,4,5]` and `[9,2,3,4,5]`. How much can the second reuse from the first?

<details><summary>Answer</summary>

Nothing. The first word differs, so every later word's notes differ (from floor 2 up). Only a shared beginning counts, and the beginning is not shared.

</details>

### 2.3 RadixAttention and Attention

**Kid version.** The name sounds like a new kind of attention. It isn't. Attention is computed exactly as before. RadixAttention is the **front desk of the pantry**: the tree that says where each beginning's notes live, the pool that holds them, and the rules about order of service.

| Part | Kid word | Job |
|---|---|---|
| KV Cache Pool | The pantry shelf | Holds the actual K and V numbers |
| Radix Tree | The index | Words → pantry slot numbers; supports match, insert, split, evict |
| Cache-aware scheduler | The waiter's habits | Prefer orders that keep shared beginnings warm (full SGLang does; mini-sglang is plain first-come first-served) |

## 3 Core Data Structure: Radix Tree

### 3.1 Data-Structure Requirements of Prefix Cache

**Kid version.** What we need from the index:

| Need | Why |
|---|---|
| Find the **longest matching beginning** quickly | That's what we can reuse |
| **Add** new sequences, **split** a branch when a newcomer shares only part of it | Orders partly overlap |
| Share the beginning, keep different endings **independent** | No interference |
| **Compact**: few nodes | A prompt can be 10,000 words |

| Structure | One edge holds | Node count for a 10,000-word unbranched prompt |
|---|---|---|
| Word-by-word trie | 1 word | 10,000 |
| Radix tree (compressed) | any run of words | **1** (split only when a real fork appears) |

### 3.2 Basic Components

**Kid version.** A node is **one branch** of the phone-menu tree:

```text
   ┌─ node ───────────────────────────────────────────────────────┐
   │ key      the run of word ids on this branch      (on the CPU) │
   │ value    the pantry slot of EACH of those words  (on the GPU) │   len(key) == len(value), always
   │ children branches below, found by their first word (or first page)│
   │ parent   the branch above                                      │
   │ ref_count how many running orders are leaning on this branch   │
   │ timestamp when it was last used (older = thrown out sooner)    │
   └──────────────────────────────────────────────────────────────┘
```

The words of any beginning = the keys glued together along the path from the root. The pantry slots = the values glued together along the same path. The root is empty and always counted as locked.

| Method | Kid version |
|---|---|
| `get_match_len(ids)` | How many words at the front of `ids` agree with this branch |
| `split_at(pos)` | Cut this branch in two at `pos`; the upper half is a new node and this node keeps the lower half |
| `is_leaf`, `is_root` | Is this a dead end / the top |

### 3.3 Adding a New Node, Longest-Prefix Matching, and Splitting

**Kid version.** Three rules, applied while walking down:

| Situation while walking | What the tree does |
|---|---|
| The next branch matches **completely** | Go deeper |
| The next branch matches only **part** (0 < match < branch length) | **Split** the branch at the first difference; the shared part stays, the rest hangs below |
| No branch starts with the next word | Stop; a new branch is hung here (on insert) |

**The book's example, run by the toy (Part 1).** Three requests, slot names in brackets:

| Request | Token ids |
|---|---|
| A | `[101, 11, 12, 21, 22]` → slots `p0..p4` |
| B | `[101, 11, 12, 31, 32]` |
| C | `[101, 11, 99]` |

```text
   after A:    root ── [101 11 12 21 22]  slots p0 p1 p2 p3 p4

   after B:    B matches 3 words (101 11 12) of A's branch  → SPLIT at 3, hang [31 32] under the upper part
               root ── [101 11 12]  p0 p1 p2
                          ├── [21 22]  p3 p4
                          └── [31 32]  q3 q4            ← B's own q0..q2 duplicate p0..p2 and are given back

   after C:    C matches 2 words (101 11) of [101 11 12] → SPLIT at 2, hang [99] under the upper part
               root ── [101 11]  p0 p1
                          ├── [12]  p2
                          │     ├── [21 22]  p3 p4
                          │     └── [31 32]  q3 q4
                          └── [99]  r2
```

The toy output matches: `insert` of B reports "3 already in the tree", of C "2 already in the tree". A's original branch ends up cut into `[101 11]`, `[12]`, `[21 22]`, exactly as the book says.

**Splitting moves no notes.** The shared beginning still points at `p0, p1, p2`. Only the *index* is cut; the pantry is untouched.

**Where the story lies.** A phone-menu tree is read word by word. Real prompts hold thousands of ids, so the comparison of a branch with the new order's words is done by a small C++ routine (`fast_compare_key`) instead of a Python loop. That is a speed detail, not part of the idea.

## 4 How the Radix Tree Supports Prefix Cache

### 4.1 Relationship Between the Radix Tree and KV Cache Pool

| Object | Content | Kid version |
|---|---|---|
| Tree **key** | word ids | "which words" |
| Tree **value** | slot numbers (or page numbers) | "where in the pantry" |
| KV Cache Pool | the K and V numbers for every floor | "the notes themselves" |

**The tree stores no K or V numbers.** It stores only addresses. Two orders sharing a beginning each have, in their own row of the peg board, the same slot numbers for those words. That is the whole trick of sharing.

### 4.2 Cache Lifecycle from Arrival to Request Completion

**Kid version.** The vocabulary first.

| Word | Kid version |
|---|---|
| page / token slot | The pantry's unit / one word's spot |
| page table | The peg board row: word position → slot |
| free list | Slots nobody owns |
| `ref_count` | How many orders lean on a branch (0 = not in use) |
| handle | A **ticket**: the matched length and the deepest matched branch. Use it to fetch the slot list, and to lock/unlock |
| `evictable_size` | Words in the tree that nobody is using (may be thrown out) |
| `protected_size` | Words in the tree that someone is using (may not) |

**The setup (toy Part 2).** A pool of 10 slots. A beginning `S = [900, 10, 11, 12]` sits in the tree on slots `0..3`. Slots `4, 5` belong to an unrelated running order X (not in the tree). Slots `6..9` are free. Two orders arrive:

| | Prompt | Matches | Still to read |
|---|---|---|---|
| R1 | `[900, 10, 11, 12, 31, 32]` | 4 words (`S`) | `31, 32` |
| R2 | `[900, 10, 11, 12, 41, 42, 43]` | 4 words (`S`) | `41, 42, 43` |

**The seven steps, with the real counters from the toy.**

| Step | What happens | `evictable` | `protected` | free slots |
|---|---|---|---|---|
| 0 | Start: `S` filed, nobody using it | 4 | 0 | 6,7,8,9 |
| 1 | **Match** R1: finds `S`, ticket for 4 words, slots `[0,1,2,3]`. *Nothing is locked yet.* | 4 | 0 | 6,7,8,9 |
| 2 | **Schedule**: the waiter admits R1 (room check passes: available = 0 + 4) | | | |
| 3 | **Lock** `S` (ref 0→1): moves from evictable to protected | **0** | **4** | 6,7,8,9 |
| 4 | **Allocate only the missing part**: 2 slots (6, 7) for words `31, 32`; R1's row is `[0,1,2,3,6,7]` | 0 | 4 | 8,9 |
| 5 | After prefill: **insert** `[31,32]→[6,7]` as a new leaf; unlock the old ticket, lock the new one (it covers both branches) | 0 | **6** | 8,9 |
| 6 | R1 finishes: **unlock** (ref→0) | **6** | 0 | 8,9 |
| 7a | R2 arrives, **matches** `S`, **locks** it. R1's leaf `[31,32]` is unlocked, hence evictable | 2 | 4 | 8,9 |
| 7b | R2 needs 3 slots but only 2 are free → **evict** one: the leaf `[31,32]` goes, **both** its slots (6, 7) return | 0 | 4 | 8,9,6,7 → after taking 3 (8, 9, 6): **7** |

(After R2's own insert: evictable 0, protected 7 (`S` + R2's leaf), free `[7]`, X still holds 2: `1 + 7 + 2 = 10`. The toy checks that the books balance after every step.)

Three things to notice:

1. **Matched is not held.** Between steps 1 and 3 the tree could still throw `S` away. Locking is what makes the borrow safe.
2. **Locking protects; it does not allocate.** Step 4 is a separate step that hands out only the missing slots. The borrowed slots `0..3` are never requested again.
3. **Eviction frees whole nodes.** R2 asked for 1 slot and got 2 back (the leaf has two). Free space beyond the ask is simply left in the free list.

**What happens to duplicates (the race).** If another order inserted the same beginning *while this one was being prefilled*, `insert` reports "this much was already there", and the order's own copy of those slots becomes a duplicate. The pantry clerk gives them straight back to the free list so they don't linger as two copies. (This is the `_free(page_indices[old_handle.cached_len : cached_len])` line in the code half. Notes `H1` ask whether the order's peg board row might still point at those freed slots.)

**Finished versus kept.** When an order finishes, the whole-page part of its **prompt + answer** goes into the tree (if not already there) and stays until evicted. Only the part that cannot be filed (the partial last page, or duplicates) is returned at once. A kept beginning costs nothing to keep until the pantry needs the space.

**Scheduling.** The book describes full SGLang's prefix-aware policies (prefer a long shared prefix, or the shortest uncached work, or weight by tree depth). mini-sglang does none of that: it admits waiting orders **first come, first served** until space or word budget runs out.

**Where the story lies.** The tree's `evictable` slots count as "available" when the waiter checks whether a new order fits, because they can be reclaimed on demand. But they are only really free *after* an eviction. R2's check in step 7a passes (`2 evictable + 2 free = 4 >= 3`) even though only 2 slots are free right now.

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** Keep the notes filed by beginning. Match walks the tree and splits a branch if the newcomer only shares part of it. Lock before use; allocate only what is missing; insert after the pass; unlock when done; the notes stay until space is needed and then the oldest unused leaves go first.

### 5.2 Exercises

**1. Why a radix tree (path compression) instead of an ordinary trie?**

<details><summary>Answer</summary>

Real prompts share long beginnings (system prompts, earlier turns). A word-by-word trie makes one node per word and one step per word; a radix tree stores an unbranched run as a single node and only creates nodes at real forks. In the chapter's A/B/C example the three requests (13 words in total) end up as 5 nodes; a word-by-word trie would need 8 (one per distinct word along the paths: 101, 11, 12, 21, 22, 31, 32, 99).

</details>

**2. Why can eviction start only from leaf nodes?**

<details><summary>Answer</summary>

An inner node is the beginning of everything below it. Removing it would invalidate the notes of all its descendants, whose meaning depends on that beginning. A leaf is the end of a path, so removing it hurts no other branch; once its parent has no children left and no users, the parent is a leaf and becomes a candidate too (toy Part 5: `evict(3)` removed `[3 4]` and then `[1 2]`).

</details>

**3. In scheduling, when are *evict* and *free* used, and how do they differ?**

<details><summary>Answer</summary>

**Evict** takes unlocked leaves **out of the tree** when a new order needs space the free list can't supply; it returns their slots, and the pantry clerk puts them on the free list. **Free** returns slots that the tree never owned or doesn't need: duplicates found at insert time, and the unfileable tail of a finished order. Evict is on demand and works on tree nodes; free is immediate and works on loose slots.

</details>

**Extra drills (not in the book).**

4. A tree has `[1,2,3]` (slots 5,6,7) locked by one order. A new order `[1,2,9]` arrives. What does `match` return and what happens to the tree? (Match length 2. The branch `[1,2,3]` is split into `[1,2]` and `[3]`; the new upper node inherits the old `ref_count`, so the locked count is unchanged. The slots returned are `5, 6`.)
5. Page size 4: `A = [1..8]`, `B = [1,2,3,9,10,11,12,13]`. How much does B share? (Zero: the first page `(1,2,3,9)` differs from `(1,2,3,4)`; the dictionary lookup misses. With page size 1 it shares 3: my toy, Part 4.)
6. Why does a prefix cache help *decode* only indirectly? (Matching happens at prefill. Decode only reads/writes the order's own row, but the borrowed beginning is what saved the prefill work.)
