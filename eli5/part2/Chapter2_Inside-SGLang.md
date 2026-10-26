# Part II Chapter 2 Inside SGLang: The Path of a Request: ELI5 companion

Original: [Chapter2_Inside-SGLang.md](../../course-material/eng/part2/Chapter2_Inside-SGLang.md). Same headings, same numbers. The code half is in [Chapter2_Inside-SGLang_code.md](./Chapter2_Inside-SGLang_code.md). Cast and picture legend: [../README.md](../README.md).

## The one-minute version

**Kid version.** A request is a **letter that passes through five desks** and comes back as a stream of words.

```text
                 desk 1             desk 2           desk 3              desk 4            desk 5
   customer ──▶ FRONT DESK ──▶ CARD-MAKER ──▶ HEAD WAITER + CHEF ──▶ CARD-READER ──▶ FRONT DESK ──▶ customer
   (text)       gives it a     words → numbered  decides what to cook   numbered cards    streams text
                ticket number  cards             and cooks it           → words           back as it comes
                (uid)          (tokenizer)       (scheduler + engine)   (detokenizer)     (SSE)
```

Three transformations happen on the way: **text → token ids**, **context → new token** (repeated), **token ids → text**. The ticket number (`uid`) stays on the letter the whole way, so every desk knows whose it is.

## 1 Learning Objectives

| Goal | Kid version |
|---|---|
| Draw the whole path with inputs and outputs | The five-desk picture above, from memory |
| Scheduler vs execution engine; where prefill and decode fit | The head waiter *decides*, the chef *cooks*; prefill and decode are two kinds of cooking pass |
| Why streaming needs decoding state; why resources must be reclaimed on finish or cancel | A word can arrive in broken pieces; a customer can walk out |
| How multiple processes help, and what they cost | Many rooms working at once, at the price of passing notes between rooms |

## 2 Following a Request

The running example: the customer asks `"What is KV Cache?"` and wants the answer **as it is written**, not all at once. The book notes the five stages are not five processes; the last stage is done by the same front desk that took the order.

### 2.1 The frontend receives the request

**Kid version.** The front desk (the **API server**) takes your order, writes a **ticket number** on it, hands the slip to the card-maker, and **immediately turns to the next customer**. For every open ticket it keeps a little **light** that switches on whenever new words for that ticket arrive.

```text
   ticket 0 ─ light off ─ waiting ...
   ticket 1 ─ light ON  ─ new words arrived! → send them to customer 1
   ticket 2 ─ light off ─ waiting ...
```

**Real words.**

| Kid word | Real thing in the code |
|---|---|
| Front desk | `server/api_server.py`, a FastAPI app run by uvicorn in the **main process** |
| Ticket number | `uid`, from `FrontendManager.new_user()`: a plain counter, 0, 1, 2, ... |
| The light | An `asyncio.Event` per uid (`event_map[uid]`) |
| The pile of words that arrived for a ticket | `ack_map[uid]`, a list of `UserReply` messages |
| Turning to the next customer | The handler is `async` and `await`s: Python switches to another request while this one waits |

**Why "async" and not threads?** The front desk spends almost all its time *waiting* (for the network, for the scheduler's reply). `async` is like **one waiter juggling many tables**: when table A is waiting for food, the waiter serves table B and comes back when A's food is ready. No extra waiters needed, so no locking headaches.

**Where the story lies.** The "light" is *one light per ticket*, but several replies can arrive before the desk looks at it. The desk empties the whole pile each time it wakes up (the `ack_map` list), which is why a slow reader still gets every piece, in order.

### 2.2 The tokenizer converts text into tokens

**Kid version.** The model only understands **numbered cards**, not letters. The card-maker has a big dictionary of word-pieces and a card number for each. For a chat, first it fills in an **order form** (the *chat template*) that stamps who said what, then cuts the text into cards.

**A tiny example** (shape only; the exact template comes from the model's tokenizer config, and the card numbers here are made up):

```text
   messages = [ {role: "user", content: "What is KV Cache?"} ]
        │  chat template (add the "assistant is about to speak" marker)
        ▼
   "<|im_start|>user\nWhat is KV Cache?<|im_end|>\n<|im_start|>assistant\n"
        │  encode: cut into word-pieces, look up each one's number
        ▼
   [ 151644, 872, 198, 3838, 374, ... ]        ← a 1-D list of integers
```

The result is a 1-D **int32 tensor on the CPU**. Notice what does *not* change: the `uid` and the sampling parameters travel along with the cards, unchanged.

| Kid word | Real thing |
|---|---|
| Card-maker | `TokenizeManager.tokenize` in `tokenizer/tokenize.py` |
| Order form | `tokenizer.apply_chat_template(..., add_generation_prompt=True)` |
| Cards | `input_ids`, int32, 1-D, CPU |
| Passing the cards on | A `UserMsg(uid, input_ids, sampling_params)` pushed to the scheduler over ZeroMQ |

The worker handles one message at a time (`local_bs = 1` in `launch.py`; the code has a `TODO: batch tokenization`).

**Pause and predict.** Why does the scheduler never see the original text?

<details>
<summary>Answer (try first)</summary>

Everything downstream (scheduling, KV cache, the model) works on token ids. Keeping the text out of the scheduler also keeps CPU-heavy tokenizing out of the scheduler's busy loop, which is one of the reasons for the separate process (section 3).

</details>

### 2.3 The scheduler selects work, and the engine executes it

**Kid version.** New orders sit on a **waiting clipboard** (the scheduler's queue). The **head waiter** (scheduler) looks at the clipboard and the pantry and says: "These orders go in the next pass, and this pass is a *reading* pass (prefill) or a *writing* pass (decode)." The **chef** (engine) then does the cooking pass on exactly that tray. The head waiter and chef are in the *same* room (same process): the head waiter just calls the chef, no phone call needed.

```text
   waiting clipboard ─▶ HEAD WAITER ─▶ tray (a Batch) ─▶ CHEF (engine.forward_batch) ─▶ scores (logits)
   (PendingReq list)    "who fits?"                      └─ sampler ─▶ next token for each order
```

**Choosing the tray.** The waiter does not just take orders in line; it also checks: Will their notepads **fit** in the pantry? Can part of the order **reuse notes** another order already made (prefix cache)? Will the tray stay **within the word budget** for one pass? An order being on the clipboard does not guarantee it runs in this pass.

**From scores to a word (the sampler).** The chef's output is **logits**: a score for every word in the vocabulary. They are not percentages. Two ways to pick:

- **Greedy:** take the highest score. Done.
- **Sampling:** (1) divide the scores by a **temperature**, (2) turn them into percentages with **softmax**, (3) roll a weighted die.

**Tiny numbers.** Logits for a 3-word vocabulary `[2.0, 1.0, 0.1]`:

| Temperature | Percentages after softmax | Behaviour |
|---|---|---|
| 1.0 | 65.9%, 24.2%, 9.9% | The natural spread |
| 0.5 | 86.4%, 11.7%, 1.9% | Sharper: more predictable |
| 2.0 | 50.2%, 30.4%, 19.4% | Flatter: more adventurous |
| 0.1 | ~100%, ~0%, ~0% | Practically greedy |

Two optional filters before the die roll:

| Filter | Meaning | On the example at T = 1 |
|---|---|---|
| `top_k = 2` | Keep only the 2 likeliest words, renormalise | 73.1%, 26.9% (third word dropped) |
| `top_p = 0.8` | Keep the smallest set whose probabilities add to at least 0.8 | First two words (cumulative 90.1%), renormalised to 73.1%, 26.9% |

**Why greedy can skip softmax:** softmax never changes the *order* of the scores, so the biggest logit is the biggest probability. `argmax(logits)` gives the same answer with less work.

**What happens to the result.** The new token id is **appended** to its request's sequence and becomes part of the context for the next pass (this is the autoregressive loop of Part I).

**Where the story lies.** The engine does the GPU work, but the *orders* it receives come from the scheduler. "A module boundary is not necessarily a process boundary": scheduler and engine are two classes in one process.

### 2.4 The detokenizer converts tokens back into text

**Kid version.** The card-reader turns number cards back into words. The trap: **one card is not always one whole letter**. Some letters are made of several bytes, and the tokenizer may cut the bytes between two cards. If you read each card alone, you see a broken character (shown as �).

**Real example** (bytes shown in hex; "é" is two bytes `C3 A9`, "你" is three bytes `E4 BD A0`):

```text
   card ' caf'   → " caf"
   card 'C3'     → "�"          (half of é: nonsense on its own)
   card 'A9'     → "�"          (the other half: also nonsense on its own)
   decoded one by one and glued:  " caf��"      ← WRONG
   decoded together:              " café"        ← RIGHT
```

**The fix:** remember, for each ticket, how much has been decoded and sent, and re-decode a **short window** of recent cards together. Only send text when the window ends in a complete character. The code keeps three offsets per ticket:

| Offset | Meaning |
|---|---|
| `surr_offset` | Start of the small "context window" re-decoded each time (a few cards of look-behind) |
| `read_offset` | How many cards have been fully turned into text |
| `sent_offset` | How many characters of text have been sent to the front desk |

**Trace of the real algorithm** (I ran the exact logic with a mock tokenizer; each row is one new card arriving):

*"Hello world café!"* with `é` split across two cards:

| New card (bytes) | Recent cards decode to | Brand-new text in that | Sent to the front desk | Why |
|---|---|---|---|---|
| `Hello` | `Hello` | `Hello` | `Hello` | complete |
| ` wor` | `Hello wor` | ` wor` | ` wor` | complete |
| `ld` | ` world` | `ld` | `ld` | complete |
| ` caf` | `ld caf` | ` caf` | ` caf` | complete |
| `C3` | ` caf�` | `�` | *(nothing)* | ends in � → **hold back** |
| `A9` | ` café` | `é` | `é` | now complete; send the new character |
| `!` | `é!` | `!` | `!` | complete |

*"Hello 你好."* with `你` split across two cards:

| New card (bytes) | Recent cards decode to | Brand-new text in that | Sent to the front desk | Why |
|---|---|---|---|---|
| `Hello` | `Hello` | `Hello` | `Hello` | complete |
| ` E4 BD` | `Hello �` | ` �` | ` ` (just the space) | ends in �: send only the safely printable part |
| `A0 E5 A5 BD` | `Hello 你好` | ` 你好` | `你好` (the space went out last time) | the held-back bytes complete both characters |
| `.` | ` 你好.` | `.` | `.` | complete |

**The second safety net:** `find_printable_text` also avoids cutting a word in half (it prints up to the last space), and prints CJK characters immediately, since each one is a word by itself.

**Where the story lies.** This is not extra polish: without it a streaming chat window shows garbage for every accented letter or emoji. Streaming makes the state necessary.

**Pause and predict.** Why must the decoder keep state *per request* (per uid), not one global state?

<details>
<summary>Answer (try first)</summary>

Because many requests are interleaved: card 5 of request A may arrive between card 9 and card 10 of request B. Each needs its own half-finished character, its own offsets and its own sent-so-far count.

</details>

### 2.5 The frontend streams the response

**Kid version.** Instead of waiting for the whole dish, the front desk sends the words **one spoonful at a time** down a connection that stays open. The customer sees the answer being "typed". This does not make the chef faster; it makes the *first* word appear sooner, which is what people feel.

**Real words.** The format is **SSE (Server-Sent Events)**: a normal HTTP response that never closes, with text lines starting `data:`. The final line is `data: [DONE]`. In `mini-sglang`'s chat endpoint, one chunk looks like (shown across lines for readability):

```text
data: {"id": "cmpl-0", "object": "text_completion.chunk",
       "choices": [{"delta": {"role": "assistant", "content": "Hel"}, "index": 0, "finish_reason": null}]}

data: {"id": "cmpl-0", ... "choices": [{"delta": {"content": "lo"}, ...}]}

data: {"id": "cmpl-0", ... "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}

data: [DONE]
```

(`cmpl-0` is the uid.) Without streaming, the user waits until the whole answer exists. Today almost every chat client streams.

**When the customer walks out.** If the browser tab closes or you hit Ctrl+C, the front desk must notice and tell the scheduler to **cancel** the order. Otherwise the GPU keeps producing words nobody will read, and the notes for that order keep occupying the pantry. In the code: the stream wrapper checks `request.is_disconnected()` between chunks, and on disconnect starts `abort_user(uid)`, which sends an `AbortMsg` (it travels as: front desk → card-maker, which converts it to an `AbortBackendMsg` → scheduler, which frees the order's resources).

**When the answer ends normally.** The generation stops at a limit or an end-of-sequence token; the last `UserReply` carries `finished=True`; the front desk sends the closing chunk and `[DONE]` and removes its `ack_map`/`event_map` entries.

### 2.6 Putting the path together

```text
   ┌──────────────┐ TokenizeMsg ┌───────────────┐ UserMsg  ┌─────────────────────┐
   │ API server   │────────────▶│ tokenizer     │─────────▶│ scheduler + engine  │
   │ (main proc)  │             │ + detokenizer │          │ (one proc per GPU)  │
   │              │◀────────────│ (one worker   │◀─────────│  loop: pick batch,  │
   └──────────────┘  UserReply  │  by default)  │ Detokenize│  forward, sample    │
        ▲ │ HTTP stream         └───────────────┘   Msg    └─────────────────────┘
        │ ▼
     customer
```

Three transformations: text → ids (card-maker); context → new token (chef, looped); ids → text (card-reader). The **`uid`** is the thread you follow through every desk.

## 3 Why Use Multiple Processes?

**Kid version.** One real restaurant has a host, a card-maker, a head waiter and a chef working **at the same time**. If one person did all four jobs in turn, every customer would wait for every other customer's job. Splitting jobs into rooms lets different customers' stages overlap. The price: the rooms must pass notes to each other.

| What each stage mostly needs | |
|---|---|
| HTTP (front desk) | network waiting |
| Tokenize / detokenize | CPU work |
| Forward pass (chef) | the GPU |

### 3.1 Let different components work at the same time

**Story.** Request A is mid-answer. Request B arrives with a very long prompt. If the head waiter had to tokenize B before starting A's next pass, A's words would **pause** and the GPU would sit idle waiting for the CPU. With a separate card-maker room, B is tokenized *in parallel* while the waiter keeps A moving. B still has to wait for its own tokenizing.

**Why processes and not just threads?** In standard Python only one thread runs Python code at a time (the **GIL**, the "one pen on the table" rule). Two threads doing CPU work in one Python process take turns, they do not run side by side. Separate processes each have their own pen, so CPU work (tokenizing) truly overlaps with the scheduler's loop.

SGLang's usual layout has the same idea: HTTP server and tokenizer manager in one process, scheduler and detokenizer in separate subprocesses.

**Where the story lies.** A single process *can* overlap work with threads, `async`, and asynchronous GPU launches (the scheduler itself does the last one, "overlap scheduling" in chapter 6). Processes just make the pieces easier to keep apart. They do not remove dependencies within one request.

### 3.2 Coordinate multiple GPUs

**Kid version.** A model too big for one chef is split among several chefs, each holding part of the recipe book. They must all work on the **same tray, in the same order**, and swap partial results (that's tensor parallelism, chapter 9). So each GPU gets its own head-waiter-plus-chef process, and **they must agree on every decision**: they cannot each pick different orders.

How mini-sglang keeps them in agreement (a preview of chapter 9):

- Only **rank 0** listens to the card-maker.
- Rank 0 **copies each raw message** to the other ranks over a broadcast channel (`minisgl_2`), and also tells them *how many* messages to expect this round (through a small CPU-side collective), so every rank reads exactly the same messages in the same iteration.
- Decode batches are **sorted by uid** before use so every rank orders the tray identically.

### 3.3 Adjust the number of workers where requests are backing up

**Kid version.** If the line is at the card-maker (long prompts, lots of text), hire **more card-makers** (`--num-tokenizer N`), don't buy more chefs. If the chef is the bottleneck, more card-makers just move the queue to the next desk. **Measure where the time goes first** (the "profile first" rule from Part 0).

By default `--num-tokenizer 0` means "the detokenizer worker also tokenizes", so there is no separate tokenizer process.

### 3.4 Make communication and debugging boundaries clear

**Kid version.** Passing notes between rooms is slower than walking across one room. In code: a function call becomes a **message**: encode, send through a socket, decode. mini-sglang uses **ZeroMQ** for control messages (small) and **GPU collectives (NCCL or its own wrapper)** for tensors between GPUs.

Benefits and costs in one table:

| Benefit | Cost |
|---|---|
| A bug can be traced by following the `uid` through each desk's log | If a process dies, its half-finished requests are lost and others may wait forever for its messages |
| Each desk can be scaled alone | Message encode/decode and copying overhead |
| Clear ownership of memory and state | Startup, shutdown and state sync must be coordinated |

(The book also notes separate processes can still share memory in some cases. Here the messages carry tensors as raw bytes, and only 1-D tensors are supported.)

Moving prefill and decode onto **different machines** (Part III) adds KV-cache transfer across machines. Process splitting alone does not solve that.

**Pause and predict.** A user reports "the answer just stopped halfway, no error". Name two desks you would check first and what you would look for.

<details>
<summary>Answer (try first)</summary>

(1) Detokenizer: did it receive the later `DetokenizeMsg`s for that uid, or was it stuck holding back an incomplete character? (2) Scheduler: was the request aborted, or finished early (EOS or length limit), and did it send `finished=True`? Following the same `uid` in each desk's log shows where the trail goes cold.

</details>

## 4 Summary and Exercises

### 4.1 Summary

**Kid version.** A request is a letter: front desk (ticket) → card-maker (cards) → head waiter + chef (cook in a loop) → card-reader (words) → front desk (stream). Splitting these into processes lets stages of different requests overlap, helps scale and debug, and costs message passing and coordination.

### 4.2 Exercises

**1. Draw the path from memory, with inputs and outputs and the three transformations.**

<details><summary>Hint</summary>

Customer (text + params) → front desk (uid, `TokenizeMsg`) → tokenizer (`UserMsg` with token ids) → scheduler → engine (logits → sampler → next token, repeated) → `DetokenizeMsg(uid, token, finished)` → detokenizer (`UserReply` with new text) → front desk → SSE chunks.

</details>

**2. Mark the generation loop. What do scheduler and engine each do? Must they be separate processes?**

<details><summary>Hint</summary>

The loop is: schedule a batch → engine forward + sampling → process results (append token, check finish, send to detokenizer) → repeat. Scheduler decides which requests/phase and manages memory; engine runs the model and sampler. No: in mini-sglang they are one process (a function call).

</details>

**3. Why does the detokenizer need per-request state? What breaks if each token is decoded alone?**

<details><summary>Hint</summary>

A character can be split across tokens; decoding alone yields �. State lets it hold back incomplete bytes and send only complete text (see the two traces in 2.4).

</details>

**4. If the user disconnects midway, what must the frontend and the scheduler each do?**

<details><summary>Hint</summary>

Frontend: notice the disconnect, drop the uid's bookkeeping, send an abort. Scheduler: stop scheduling the request, remove it from waiting/running sets, free its table row and its KV pages (decrement reference counts).

</details>

**5. What do you gain by separating tokenizer and scheduler? Why still count communication cost?**

<details><summary>Hint</summary>

Gain: CPU tokenizing overlaps with GPU scheduling; independent scaling; clean debugging boundaries. Cost: encode/decode and copies per message, and coordination of startup/shutdown.

</details>

**Extra drills (not in the book).**

6. Why can greedy decoding skip softmax, but temperature sampling cannot?
7. For logits `[2.0, 1.0, 0.1]` and `top_p = 0.6`, which words are kept? (Only the first: 65.9% already exceeds 0.6.)
8. Two clients both send their first request at once. What guarantees that the two replies are not mixed up? (Each gets a different uid from the front desk's counter; every message carries it.)
