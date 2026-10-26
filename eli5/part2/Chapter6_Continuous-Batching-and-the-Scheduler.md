# Part II Chapter 6 Continuous Batching and the Scheduler: ELI5 companion

**This companion is written from the code walkthrough and the source.** The original concept page is a three-line placeholder ("This chapter is being written"), so there are no original section numbers to mirror; the sections below are mine. The code half, which follows the original's numbering, is [Chapter6_Continuous-Batching-and-the-Scheduler_code.md](./Chapter6_Continuous-Batching-and-the-Scheduler_code.md). Original code walkthrough: [here](../../course-material/eng/part2/Chapter6_Continuous-Batching-and-the-Scheduler_code.md). Cast and picture legend: [../README.md](../README.md).

Runnable toy with every number below: `python3 -I eli5/tools/ch6_scheduler_toy.py` (standard library only).

## The one-minute version

**Kid version.** The **head waiter** (scheduler) stands at the pass between the dining room and the kitchen. Every few milliseconds she asks one question: **"Who gets the chef's next pass?"** Two kinds of pass exist:

| Pass | Who is on the tray | What the chef does |
|---|---|---|
| **Prefill** | New orders | Reads each whole prompt at once and writes its first word |
| **Decode** | Orders already underway | Writes exactly one more word for each |

Her rule is simple: **if a new order can be admitted, read it first; otherwise give every underway order one more word.** And the tray is rebuilt from scratch every single pass, so orders can *join* and *leave* between any two passes. That is **continuous batching**.

```text
   waiting list ──admit──▶ [ TRAY ] ──chef──▶ new words ──▶ unfinished? stay on the "still eating" list
   (FIFO)                  one pass                          finished?  leave; give back peg + pantry
```

## 1 Learning Objectives

| Goal | Kid version |
|---|---|
| Find the waiting queue, running set, and the moves between them | Two clipboards (`pending_list`, `running_reqs`) and the doors between them |
| Explain prefill priority, request slots, token budgets | "New orders first", "only so many pegs", "only so many words read per pass" |
| Trace output, EOS, length limits, cleanup | How an order ends and what it gives back |

## 2 Why Batch at All?

**Kid version.** To write one word the chef must flip through the **entire recipe book** (all the model weights). Flipping takes the same time whether she is writing a word for one order or for thirty. So serving thirty orders on one pass is nearly as cheap as serving one. (This is Part I chapter 2's "decode is memory-bound", in a sentence.)

**Tiny numbers (illustrative, not measured).** Qwen3-0.6B has about 1.2 GB of weights (bf16). Suppose the GPU can read 1 TB/s. One pass then costs at least `1.2 GB / 1 TB/s = 1.2 ms`, whoever is on the tray:

| Orders on the tray | Words produced per pass | Best-case words per second |
|---|---|---|
| 1 | 1 | 1 / 1.2 ms = **833** |
| 8 | 8 | 8 / 1.2 ms = **6,667** |
| 32 | 32 | 32 / 1.2 ms = **26,667** |

This is a *ceiling*: it ignores reading each order's notes (the KV cache) and the compute itself, both of which grow with the tray. But it shows why the head waiter's whole job is "keep the tray full".

**Where the story lies.** A real waiter loses nothing by carrying a bigger tray. A bigger batch has costs: more K/V to read per pass, more pantry space, and a longer wait for each member to get *their* next word. Chapter 7 and chapter 3 of Part I cover the pantry and the memory arithmetic.

## 3 Static vs Continuous Batching

**Kid version.** *Static*: the waiter loads a tray with four orders, and **nobody new boards until all four are done**, even if three are finished and one is a very long dish. *Continuous*: the instant a table finishes, its seat can go to a waiting order on the very next pass.

```text
   STATIC   slot 1  [ r0 ][ r0 ][idle][idle][idle][idle]...      slots sit idle until the longest order ends
            slot 2  [ r1 ][ r1 ][ r1 ][ r1 ][ r1 ][idle]...
            slot 3  [ r2 ][ r2 ][idle][idle]...
            slot 4  [ r3 ][ r3 ][ r3 ][ r3 ][ r3 ][ r3 ]...  (30 words long)

   CONTINUOUS
            slot 1  [ r0 ][ r0 ][ r4 ][ r4 ][ r4 ][ r6 ]...  a free slot is reused at once
            slot 2  [ r1 ][ r1 ][ r1 ][ r1 ][ r1 ][ r5 ]...
            slot 3  [ r2 ][ r2 ][ r7 ][ r7 ]...
            slot 4  [ r3 ][ r3 ][ r3 ][ r3 ][ r3 ][ r3 ]...
```

**Tiny numbers (my toy, `ch6_scheduler_toy.py` Part 2).** Eight orders arrive together; four slots; outputs of 2, 5, 2, 30, 3, 4, 2, 8 words; every pass costs "one unit" (a big simplification: real prefill and decode passes cost different times):

| Policy | Total passes | Mean passes until the first word | Mean passes until finished | Slots doing useful work |
|---|---|---|---|---|
| Static | 38 | **16.0** | 22.0 | 56 of 152 = 36.8% |
| Continuous | 33 | **3.0** | 9.75 | 56 of 132 = 42.4% |

The throughput gain looks modest (38 → 33 passes) because one 30-word order dominates both schedules. The big win is **waiting time**: orders see their first word after 3 passes on average instead of 16, and finish in 9.75 instead of 22. In a real server where requests arrive all day (not all at once), the static policy's idle slots are pure waste, and the gap grows.

**Where the story lies.** "Continuous" does not mean "new and old orders share every pass". In mini-sglang a pass is **either** all-prefill **or** all-decode. What is continuous is the *membership*: it can change between any two passes.

## 4 The Scheduler's World: Two Clipboards, Pegs, and the Pantry

**Kid version.** The head waiter owns four things.

```text
   ┌── PrefillManager ──────────────┐   ┌── DecodeManager ───────────┐   ┌─ TableManager ─┐   ┌─ CacheManager ──────────┐
   │ pending_list: the WAITING      │   │ running_reqs: the orders   │   │ numbered PEGS: │   │ the PANTRY CLERK:       │
   │ clipboard, oldest first        │   │ "still eating"             │   │ one row per    │   │ free pages + the shared │
   │ (also holds half-read orders)  │   │ (a Python set)             │   │ running order  │   │ index (prefix cache)    │
   └────────────────────────────────┘   └────────────────────────────┘   └────────────────┘   └─────────────────────────┘
```

| Object | Real file | What it answers |
|---|---|---|
| `PrefillManager` / `PrefillAdder` | `scheduler/prefill.py` | "Which waiting orders may start now, and how much of each?" |
| `DecodeManager` | `scheduler/decode.py` | "Who is mid-answer? How much more pantry might they still need?" |
| `TableManager` | `scheduler/table.py` | "Is there a free peg (table row)?" |
| `CacheManager` | `scheduler/cache.py` | "Is there room in the pantry? Which beginnings are already cached?" |

**The life of one order.**

```text
        new slip (UserMsg)
              │ _process_one_msg: too long? shrink max_tokens / drop
              ▼
        WAITING (pending_list)  ◀────────────────┐
              │ admitted by PrefillAdder          │ not finished reading? (ChunkedReq) go back to the front
              ▼                                   │
        PREFILL pass (reads prompt, writes 1st word)──┘
              │ prompt complete
              ▼
        RUNNING (running_reqs)  ◀──┐
              │ decode pass: +1 word│  (repeat)
              ├────────────────────┘
              │ finished? (length limit OR end-of-sequence word)
              ▼
        FREE its peg; keep the reusable beginning in the shared index; return the rest
              ▲
              └── an AbortBackendMsg can pull it out of WAITING or RUNNING at any time
```

## 5 One Turn of the Loop

**Kid version (the plain loop, `normal_loop`).**

```text
   1. check the mailbox        (block only if there is nothing at all to do)
   2. choose the next tray     (prefill if possible, else decode)
   3. hand the tray to the chef, wait for her result
   4. read the dishes: record each new word, decide who is finished, free their stuff
```

**The default loop is a smarter version (`overlap_loop`).** Step 4 of turn *n* is *not* done before step 2 of turn *n*+1 starts. Instead:

```text
   CPU:  schedule(n)  launch(n)  process results of (n-1) ....  schedule(n+1)  launch(n+1)  process (n) ....
   GPU:               [============= run batch n ============]                [========= run n+1 =========]
                                    ▲ the paperwork about the PREVIOUS batch hides under the GPU's work
```

The chef is never left waiting for the waiter's paperwork. The price: results are one turn *late* when the scheduler makes its next decision. That has two consequences you will meet in the code:

| Consequence | Why | Where |
|---|---|---|
| Length-limited orders stop **exactly** | `Req.complete_one()` moves the counters forward the moment a batch is *launched*, so `can_decode` flips to false immediately | `core.py`, `DecodeManager.filter_reqs` |
| EOS-ended orders are noticed **one turn late** | The word is only known when it reaches the CPU, one turn after launch. The order may already be on the next tray. The scheduler keeps `finished_reqs` so it does not free the same order twice | `_process_last_data` |

**Where the story lies.** The book's trace (section 5.2 of the code half) uses the plain `normal_loop` for clarity. In the default `overlap_loop`, a freed slot becomes visible one turn later, so a waiting order such as C would board one pass later than the table shows. The ideas are identical.

## 6 Prefill First, or Decode?

```python
batch = (
    self.prefill_manager.schedule_next_batch(self.prefill_budget)
    or self.decode_manager.schedule_next_batch()
)
```

**Kid version.** `A or B` in Python means "A if A exists, else B". So: *try to build a tray of new orders; if that fails, build a tray of underway orders.*

| Why prefill first | The cost |
|---|---|
| A new customer sees their **first word** sooner (low time-to-first-token) | Everyone already eating **pauses** for that pass (their next word is late) |
| Admitting early keeps the slots full (continuous batching's whole point) | A steady stream of new orders can keep delaying the others (nothing here stops that) |

A non-empty waiting list is not the same as "a prefill tray exists": the admission rules (next section) can refuse everyone, and then the decode tray goes instead.

## 7 Admission: Three Different Limits

**Kid version.** Before an order may board, three separate questions are asked. Keep them apart; they run out for different reasons.

| # | Limit | Question | Counted in |
|---|---|---|---|
| 1 | **Pegs** | Is there a free numbered peg? | request slots (default `max_running_req` = 256) |
| 2 | **Word budget** for this pass | How many prompt words may the chef read in one pass? | tokens (default `max_extend_tokens` = 8192) |
| 3 | **Pantry** | Will there be room for this order's notes, **in the worst case**, without hurting anyone already eating? | KV tokens |

### The worst-case promise (limit 3)

An order needs pantry for its still-unread prompt plus every word it *might* generate (`max_tokens`). The waiter counts that **maximum**, not what it will probably use, and also counts the maximum still owed to everyone already eating. Rule: `new order's maximum + already promised <= free pantry`, otherwise: no.

```text
   free pantry now: 70 tokens          already promised to the order eating: 40 more tokens
   ─────────────────────────────────────────────────────────────────────────────────────────────
   Case A  new order needs 15 (prompt) + 20 (answer) = 35:   35 + 40 = 75 > 70   → REFUSED
           (even though 70 tokens are free right now: 40 of them are spoken for)
   Case B  new order needs 15 + 10 = 25:                     25 + 40 = 65 ≤ 70   → admitted; promised is now 65
           the NEXT order needs 5 + 3 = 8:                    8 + 65 = 73 > 70    → REFUSED
           (promises accumulate inside one pass, though nothing has been written yet)
```

**Why so careful?** Because once admitted, an order must always be able to finish. If the pantry could run dry mid-answer, the engine would have to throw a running order out (or crash). Worst-case counting is a simple way to guarantee that, at the price of leaving pantry empty on average (most answers end before `max_tokens`).

### First-in-first-out, and it stops at the first refusal

```text
   waiting list: [ BIG (50+20) ] [ small (3+2) ]       free 70, promised 40
   BIG: 70 + 40 = 110 > 70  → refused → STOP.
   small would have fit (5 + 40 = 45 ≤ 70) but is never asked.
```

Fair (no queue-jumping) but it allows **head-of-line blocking**: a big order at the front holds back small ones behind it.

### Chunked prefill (limit 2)

A very long prompt may exceed the pass's word budget. Then the chef reads it in pieces.

| Pass | `cached_len` (already read) | words left | chunk read now | What the order is |
|---|---|---|---|---|
| 1 | 0 | 20 | 8 | `ChunkedReq`: no word written, **not** added to the "still eating" list |
| 2 | 8 | 12 | 8 | `ChunkedReq` again |
| 3 | 16 | 4 | 4 | a normal `Req`: its first word is sampled; now it can decode |

(Prompt of 20 words, budget 8; my toy, Part 4.) Between chunks the order keeps its peg and its protected pantry beginning. The waiter promised pantry for the **whole** prompt plus the answer the first time (see `_add_one_req`: `reserved_size += remain_len + output_len`).

**Where the story lies.** Tokens, not bytes. All three limits count *token slots*; none looks at actual GPU bytes. Converting between them is the pantry's job (chapter 7).

## 8 After the Pass: Finishing, Cancelling, Tidying

**Kid version.** After each pass the waiter reads the dishes:

| For each order on the tray | What she does |
|---|---|
| It was a half-read chunk | Nothing yet (no word was produced) |
| It produced a new word | Appends it to her copy; sends it to the card-reader |
| That word is end-of-sequence (and the order did not say "ignore it"), or the length limit is hit | **Finished**: crosses it off the "still eating" list, returns its peg, and offers its words to the shared index |
| It was a freshly read prompt (not finished) | Writes the prompt into the shared index **now**, so a later customer with the same beginning can reuse it |

A **cancel** goes to the same cleanup: found on the waiting list (a half-read order has a peg to return; one that never started holds nothing) or on the "still eating" list.

**Returning a peg does not delete the notes.** The peg and the pantry pages are returned; the useful beginnings stay in the shared index until the pantry needs the space (chapters 7 and 8).

## 9 Summary and Exercises

### Summary

**Kid version.** One loop: mail, choose the tray, cook, tidy. The tray is prefill if any new order passes three limits (peg, word budget, worst-case pantry), else decode. Orders leave when they hit their length or the end word, giving back their peg; their notes can stay for others. The tray's membership changes every pass, and that, not mixing prefill and decode, is what "continuous" means. The default loop overlaps the waiter's paperwork with the chef's cooking, so results are processed one turn late.

### Exercises (the book's five, plus mine)

**1. Why might decode run even when the waiting list is not empty?**

<details><summary>Answer</summary>

No peg is free, or the worst-case pantry check refuses the first waiting order (FIFO admission stops there), or the word budget is spent. Then `prefill_manager.schedule_next_batch` returns `None` and the `or` falls through to decode.

</details>

**2. Can an order get a free peg but still be refused? Which code checks it?**

<details><summary>Answer</summary>

Yes. `PrefillAdder._try_allocate_one` checks the peg first (`table_manager.available_size`), then `estimated_len + reserved_size > cache_manager.available_size`. A free peg does not guarantee pantry room.

</details>

**3. Why must intermediate prefill chunks not commit generated tokens?**

<details><summary>Answer</summary>

The prompt is not finished, so the model's output at that position is not "the answer's first word". Committing it would insert a word the user never asked for into the sequence. `ChunkedReq` forbids `append_host` and reports `can_decode = False`.

</details>

**4. With A, B, C needing 2, 5, 2 words, two slots, C arriving after step 0: when does C first board if A generates only one word?**

<details><summary>Answer</summary>

A finishes in the prefill pass of step 0 (its first word is its only word). Its slot is freed at the end of step 0, and C arrives after step 0, so C is prefilled at **step 1**, one step earlier than in the book's table (step 2). (Run it: change `R("A", 0, 2)` to `R("A", 0, 1)` in the toy.)

</details>

**5. Where do completion and cancellation enter cleanup? Why does freeing a peg not delete all KV?**

<details><summary>Answer</summary>

Both call `_free_req_resources`: completion from `_process_last_data`, cancellation from the `AbortBackendMsg` branch of `_process_one_msg`. It returns the peg and calls `cache_req(finished=True)`, which hands reusable full pages to the prefix cache instead of freeing them.

</details>

**Extra drills (not in the book).**

6. Free pantry 90, promised 60, free pegs 2, waiting list `[(10,15), (4,4), (2,2)]` as (prompt, answer). Who boards? (First: 25 + 60 = 85 ≤ 90 → yes, promised becomes 85. Second: 8 + 85 = 93 > 90 → refused, so the third is never asked. Only the first boards.)
7. Why does the waiter promise pantry for the *whole* prompt on the first chunk of a chunked order? (So the order can always finish; it cannot be thrown out between chunks for lack of space.)
8. In the overlap loop, an order ends on EOS in batch *n*, but batch *n*+1 was already launched with it. What stops it being freed twice? (`finished_reqs`: the previous turn's finished set; the second sighting is skipped.)
