# Part II Chapter 5 Serving It: HTTP and Concurrent Requests: ELI5 companion

Original: [Chapter5_HTTP-and-Concurrent-Requests.md](../../course-material/eng/part2/Chapter5_HTTP-and-Concurrent-Requests.md). Same headings, same numbers. The code half is [Chapter5_HTTP-and-Concurrent-Requests_code.md](./Chapter5_HTTP-and-Concurrent-Requests_code.md). Cast and picture legend: [../README.md](../README.md).

Until now the Kitchen had one customer: you, typing in Python. This chapter opens the **front desk** so that *many* customers, on other machines, can order at the same time.

## The one-minute version

**Kid version.** Three jobs, nothing more:

1. **Speak a standard language** so any customer can order (HTTP + JSON).
2. **Never stand still while the kitchen cooks.** The front desk takes the next order while the last one is still cooking (concurrency).
3. **Hand each customer only their own dish, spoonful by spoonful** (ticket numbers + streaming).

```text
  customer A ──┐                                              ┌── spoonfuls for A ──▶ customer A
  customer B ──┼─▶ FRONT DESK ─▶ card-maker ─▶ head waiter ─▶ chef ─▶ card-reader ─▶ FRONT DESK
  customer C ──┘   (ticket #)    (tokenizer)   (scheduler)  (GPU)   (detokenizer)     (sorts by #) └── for B, for C
```

The front desk does **not** cook. It only takes orders, numbers them, and delivers.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Explain clients, servers, HTTP; why serve a local function | A function only works inside your own kitchen. A restaurant needs a door and a menu |
| Read a request: address, method, headers, body (`messages`, `stream`) | Read an order slip: where it goes, what kind of slip it is, the printed header, the handwriting |
| Path from client through the system and back | The nine-stop journey table in section 4 |
| Concurrency, request isolation; vs parallelism vs batching | One waiter / many tables, versus two waiters, versus one pot for two orders |
| Streaming vs non-streaming; completion, cancellation, cleanup | Whole dish vs spoonfuls; what happens when the customer walks out |

## 2 From a Local Program to an HTTP Service

### 2.1 What Clients and Servers Do

**Kid version.** Cooking at home: you hold the spoon and the pot. A restaurant: customers sit far away and the kitchen is behind a wall. They need a **window** where slips go in and plates come out.

```text
   AT HOME (local function)                 RESTAURANT (service)
   ┌───────────────────────┐               ┌────────┐   slip    ┌─────────────────────┐
   │ your Python program   │               │ client │ ────────▶ │ server (model loaded │
   │  + the model, loaded  │               │ (app)  │ ◀──────── │  once, kept running) │
   └───────────────────────┘               └────────┘   plate   └─────────────────────┘
   one program, one copy of the weights       many apps share ONE copy of the weights
```

| Kid word | Real word |
|---|---|
| Customer | **Client**: any program that sends a request (web page, phone app, `curl`, another server) |
| Kitchen | **Server**: the program that accepts requests, does the work, returns results |
| Window | The network interface (an address + a protocol) |

**Tiny numbers.** Qwen3-0.6B has about 0.6 billion numbers; in bf16 (2 bytes each) that is roughly **1.2 GB** of weights. Ten apps each loading their own copy: 12 GB of GPU memory. One shared server: 1.2 GB. That, more than anything, is why we "serve" a model instead of importing it everywhere.

**Where the story lies.** "Client" and "server" are **roles in one conversation**, not kinds of computer. Your business backend can be a *server* to its own users and a *client* of the model server in the same second.

### 2.2 How HTTP Defines Requests and Responses

**Kid version.** HTTP is a **postal format**: which box to write the address in, which box for the kind of letter, where the letter body goes. Both sides agree on the format; nothing else is promised. It says nothing about *how* the kitchen cooks.

Take the book's address and cut it into pieces:

```text
   http://localhost:1919/v1/chat/completions
   └─┬─┘  └───┬───┘ └─┬┘ └────────┬────────┘
   protocol  which   which     which window
   (HTTP)   machine  door      (endpoint path)
            (this    number
             one)    (port)
```

| Piece | Kid version |
|---|---|
| `http` | "We will write our slips in the HTTP format" |
| `localhost` | "The machine I am sitting on". From another machine you need a real address |
| `1919` | The **port**: the numbered door of that machine where *this* server is listening. A machine has thousands of doors; each program picks one |
| `/v1/chat/completions` | The **endpoint**: which counter in the restaurant ("chat orders here") |

**Where the story lies.** HTTP has **no memory**. Every slip stands alone; the server does not remember your previous slip unless *you* include the history again. (You will see the cost of that in section 3.1.)

**Pause and predict**

1. Which of these is *not* HTTP's job: finding the door, writing the slip, deciding the model's next word?

<details><summary>Answer</summary>

Deciding the next word. HTTP is only the envelope format. The API (which path means "generate", which JSON fields exist) is defined by the service, and the model is far behind the wall.

</details>

## 3 Reading a Request and a Response

### 3.1 What a Request Contains

**Kid version.** An order slip has four zones:

```text
   POST /v1/chat/completions HTTP/1.1      ◀── ① KIND + COUNTER: "I am SUBMITTING something, at this counter"
   Host: localhost:1919                    ◀──┐
   Content-Type: application/json          ◀──┴─ ② HEADERS: printed boxes: which machine, what language the body is in
                                           ◀── (blank line = "header over, handwriting starts")
   {                                       ◀──┐
     "model": "Qwen/Qwen3-0.6B",              │
     "messages": [ {"role":"user",            ├─ ③ BODY: the actual order, in JSON
                    "content":"What is KV      │
                    Cache?"} ],                │
     "max_tokens": 128,                        │
     "stream": false                        ◀──┘
   }
```

| Body field | Kid version |
|---|---|
| `model` | "Which chef do you want?" Here: **ignored**; this server cooks with the one model it loaded at startup |
| `messages` | The conversation so far, oldest first. Each message is `{role, content}` |
| `role` | Who said it: `system` (house rules), `user` (customer), `assistant` (the model) |
| `max_tokens` | "Stop after this many *cards*." It is a **ceiling**, not a promise: the answer may end sooner |
| `stream` | `false`: bring the whole dish at the end. `true`: bring spoonfuls as they are made |

**Two different "messages".** The HTTP request is itself a *message*; and inside the body there is a list called `messages`. Different things. The list is **conversation turns**.

**Tiny numbers: why the whole history is resent.** HTTP forgets, so a chat app resends everything every turn. Suppose every message is 20 tokens. Turn *k* sends *2k − 1* messages (all the earlier ones plus the new question):

| Turn | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Tokens the server receives | 20 | 60 | 100 | 140 | 180 | 220 | 260 | 300 | 340 | 380 |

Over ten turns the server is *sent* **2,000** tokens, but only **380** are new (the first question plus 9 pairs of "answer + next question"). Ratio about **5.3 to 1**, and it keeps growing. Without help the engine would re-read the same beginning every turn. That "help" is the **prefix cache** (chapter 8): the beginning of every turn's prompt is identical to the last turn's, so its notes can be reused.

**Where the story lies.** Notice that `max_tokens` is counted in tokens, not characters, and not in "words". And note that `model` looks like a menu choice but behaves like a decoration (see the code half).

### 3.2 What a Response Contains

**Kid version.** The plate comes back with a receipt:

```text
   HTTP/1.1 200 OK                 ◀── status: 200 = "your order was understood and served"
   Content-Type: application/json  ◀── header: "the plate is JSON"

   { "choices": [ { "message": {"role":"assistant","content":"KV Cache stores ..."},
                    "finish_reason": "stop" } ] }
                                                    ▲ why the chef stopped
```

| Piece | Kid version |
|---|---|
| `200` | The kitchen *processed* the slip. It says nothing about whether the answer is *correct* |
| `choices[0].message` | The dish |
| `finish_reason` | Why the chef stopped: `"stop"` (reached the end marker) or `"length"` (ran into `max_tokens`) |

Error statuses exist too (for example `422` when your slip was filled in wrongly, `500` when the kitchen broke).

**Where the story lies.** mini-sglang's `finish_reason` is **always `"stop"`**, even when the answer was cut off by `max_tokens`. A client that trusts it cannot tell "finished" from "truncated". (The engine knows the difference; the front desk does not pass it on. See the code half.)

**Pause and predict**

2. You send the same question twice. Does the server remember the first one when answering the second?

<details><summary>Answer</summary>

No. HTTP is stateless and so is this endpoint. Only what is in `messages` of *this* request exists for the model. (The prefix cache may reuse *computed notes* for an identical beginning, but that is a speed-up, not memory of the conversation.)

</details>

## 4 Following a Request Through the Inference System

**Kid version.** The slip goes on a journey. Each stop has one job.

| # | Stop | Kitchen role | What happens | In mini-sglang |
|---|---|---|---|---|
| 1 | Client | Customer | Writes the slip | your `curl`, browser, app |
| 2 | API Server | Front desk | Checks the slip, writes a **ticket number**, hands it on | `server/api_server.py`, FastAPI + uvicorn |
| 3 | Tokenizer | Card-maker | Words → numbered cards (applies the chat template to `messages`) | `tokenizer/server.py` |
| 4 | Scheduler | Head waiter | Queues the order, decides which orders share a tray | `scheduler/scheduler.py` |
| 5 | Engine | Chef | Cooks one word per pass | `engine/engine.py` on the GPU |
| 6 | (back to the scheduler) | Head waiter | Sees the new card, decides "finished?" | `_process_last_data` |
| 7 | Detokenizer | Card-reader | New card → piece of text | `tokenizer/detokenize.py` |
| 8 | API Server | Front desk | Finds whose ticket it is; passes the text on | `listen()` |
| 9 | Client | Customer | Reads the spoonful | your screen |

Stops 5 to 8 **repeat once per generated word**. Streaming means stop 9 happens each time. Non-streaming means stop 8 *collects* the pieces and stop 9 happens once at the end.

```text
            ┌────────── repeats for every new word ──────────┐
   ①──▶②──▶③──▶④──▶⑤──▶⑥──▶⑦──▶⑧──▶⑨ (stream)   ... or ... ⑧ keeps collecting, ⑨ only at the end
```

**These are responsibilities, not processes.** By default mini-sglang runs the card-maker and the card-reader in the *same* process, and the front desk in another, plus one head-waiter-and-chef process per GPU. (Chapter 2 and chapter 9 count them.)

**Why the front desk is separate from the kitchen.** Cooking takes seconds; parsing a slip takes microseconds. If the front desk stood and watched each dish cook, nobody else could order. So it hands over the slip and goes back to the door. That is the next section.

**Pause and predict**

3. Which stop turns text into numbers? Which one decides whose turn it is to cook?

<details><summary>Answer</summary>

Stop 3 (tokenizer) turns text into token ids. Stop 4 (scheduler) decides who is cooked next and who shares a tray.

</details>

## 5 Serving Multiple Users at Once

### 5.1 Why Concurrency Is Needed

**Kid version.** One waiter, many tables. A waiter who stands at table 1 until the food arrives is a bad waiter. A good waiter takes table 1's order, walks away, takes table 2's, and comes back when a bell rings.

Two customers, with an answer taking 5 s each, B arriving at t = 1 s (illustrative numbers):

```text
   SERIAL front desk                       CONCURRENT front desk
   t=0  A ordered   ░░░░░░░░░░ A cooks     t=0  A ordered   ░░░░░░░░░░ A cooks
   t=1  B arrives, NOBODY IS LISTENING      t=1  B ordered   ░░░░░░░░░░ B cooks (queued in the kitchen)
   t=5  A done. B is taken ONLY NOW         t=5  A done      t=6  B done
   t=10 B done
```

B waits 4 extra seconds to even be *heard* in the serial case. In the concurrent case the front desk heard B at t = 1 and the kitchen decides when to cook B.

**Real word:** **concurrency** = *keeping several unfinished orders alive at once*. It is a statement about the *front desk*, not about the GPU.

**Where the story lies.** More chefs (bigger clusters) do not remove the need. Every front desk still has to track many open connections.

### 5.2 Keeping Requests Separate

**Kid version.** Replies arrive jumbled: A's first spoonful, then B's, then B's again, then A's. The front desk needs a **ticket number** on every spoonful, a **pile** per ticket, and a **light** per ticket.

```text
   spoonfuls arriving, in this order:    A1  B1  B2  A2
                                          │   │   │   │
                                  ┌───────▼───▼───▼───▼───────┐
                                  │ MAIL SORTER (listen)      │   ← ONE task for the whole server
                                  └───┬───────────────────┬───┘
                                      │ ticket 0          │ ticket 1
                              ┌───────▼──────┐     ┌──────▼───────┐
                              │ pile: A1, A2 │     │ pile: B1, B2 │   ← ack_map[uid]
                              │ light: ON    │     │ light: ON    │   ← event_map[uid]
                              └───────┬──────┘     └──────┬───────┘
                                      ▼                   ▼
                               customer A's coroutine   customer B's coroutine
```

| Kid word | Real word |
|---|---|
| Ticket number | `uid` (an integer counter in the API server; it restarts at 0 when the server restarts) |
| Pile of spoonfuls | `ack_map[uid]`: a list of `UserReply` |
| Light | `event_map[uid]`: an `asyncio.Event` |
| Mail sorter | `FrontendManager.listen()`: one background task for the whole server |
| Customer waiting for the light | `wait_for_ack(uid)` |

**Why both a pile and a light?** A light alone says "something is there" but cannot carry the something. A pile alone holds things but cannot ring a bell, so the customer would have to keep checking ("polling"), burning time. One structure holds, one wakes.

**Tiny numbers: a trace of the real logic (my toy replay, no GPU).** The customers are slow, so all four spoonfuls arrive before anyone wakes.

| Step | What happens | pile 0 (A) | pile 1 (B) | light 0 | light 1 |
|---|---|---|---|---|---|
| 0 | two tickets issued | [ ] | [ ] | off | off |
| 1 | A1 arrives | [A1] | [ ] | **ON** | off |
| 2 | B1 arrives | [A1] | [B1] | ON | **ON** |
| 3 | B2 arrives | [A1] | [B1, B2] | ON | ON |
| 4 | A2 arrives | [A1, A2] | [B1, B2] | ON | ON |
| 5 | customer A wakes: clears light, grabs pile, leaves a fresh empty one | [ ] | [B1, B2] | off | ON |
| 6 | customer B wakes, same | [ ] | [ ] | off | off |

Run for real (see the code half), customer A received exactly `A1 A2` and B exactly `B1 B2`, in order within each ticket.

**One wake-up, many replies.** In another run I put three replies (`x`, `y`, `z`) in before the customer woke: it woke **once** and took **all three** (`wake-ups: 1, replies per wake-up: [3]`). The light is a *flag*, not a counter. That is fine because the data lives in the pile.

**Where the story lies.** "Separate" means separate *state and results*. It does **not** mean separate model copies or GPUs. All tickets share one model.

### 5.3 Concurrency, Parallelism, and Batching

**Kid version.** Three different ideas that people mix up:

| Idea | Kitchen picture | Real meaning |
|---|---|---|
| **Concurrency** | One waiter juggling many tables | Many unfinished orders are *managed* at once. The waiter is only ever doing one thing at an instant |
| **Parallelism** | Two waiters walking at the same time | Work *actually* happens at the same instant (several CPU cores or several GPUs) |
| **Batching** | One pot cooking two orders in one go | One model forward pass serves several requests together |

You can have any one without the others:

```text
   10 orders accepted (concurrency), cooked one at a time (no batching)  ◀── possible
   a script feeding 100 prompts to the GPU in one pass, no HTTP at all   ◀── batching without a server
```

So "the server accepts ten requests" does **not** mean "the GPU computes ten together". Whether they share a pot is the head waiter's decision. That is **continuous batching** (chapter 6), and the reason it matters is Part I chapter 2: a decode step is bound by reading the recipe book (weights), and a bigger batch reads the book *once* for many dishes.

**Pause and predict**

4. A request B arrives while A is still being generated. Which of the three ideas lets the *front desk* accept B? Which one lets the *GPU* work on both in one pass?

<details><summary>Answer</summary>

Concurrency (the async event loop) lets the front desk accept B. Batching (the scheduler putting A and B in one forward pass) lets the GPU serve both together. Parallelism is not needed for either.

</details>

## 6 Returning Once or Streaming Incrementally

### 6.1 The User-Visible Difference

**Kid version.** Whole dish: you wait, then get everything. Spoonfuls: you taste the first one almost at once.

**Tiny numbers (illustrative, not measured).** 128 words at 20 ms per word = **2.56 s** of cooking. Whole-dish: the screen is blank for 2.56 s (plus any prompt-reading time), then everything appears. Spoonfuls: the first word appears after about 20 ms (plus the same prompt-reading time) and then the text *keeps growing* for 2.56 s.

```text
   non-streaming:  [ blank .............................. ][ ALL THE TEXT ]
   streaming:      [ blank ][w][o][r][d][s] [ keep ] [ arriving ] ......
                   ▲ first word visible early       ▲ total time is about the SAME
```

**Total work is identical.** The chef still cooks 128 words and does 128 passes. Streaming only changes **when you see them** (time to first token, "TTFT"), not **when the last one is done**.

**Where the story lies.** "Non-streaming blocks the server" is false. While one customer waits for the whole dish, the front desk keeps serving others. Only *that customer's* screen is blank.

### 6.2 Carrying New Content with SSE

**Kid version.** **Server-Sent Events** is a way to keep the phone line open and let the server talk whenever it has something, in a fixed format: each message is a line starting `data: ` and *ended by a blank line*.

```text
   data: {"choices":[{"delta":{"content":"KV Cache"}}]}↵↵                 ◀── one event (↵↵ = blank line)
   data: {"choices":[{"delta":{"content":" stores ..."}}]}↵↵              ◀── next event
   data: [DONE]↵↵                                                         ◀── "I am finished" (a convention, not the standard)
```

| Piece | Kid version |
|---|---|
| `data: ` | "Here comes a message" |
| blank line | "That message is complete". This is how the reader knows where one message ends |
| `delta.content` | **Only the new piece**, not the whole text so far. The client glues pieces together |
| `[DONE]` | A sentinel this kind of API adds so the client knows to stop reading. SSE itself has no end marker |

**Where the story lies.** One event is **not** one token and not one network packet. Text decoding can hold back half a character (chapter 2's detokenizer), and the network can glue two events into one read or split one in two. A client must read until the blank line, not "one read = one event". Also, mini-sglang's simple `/generate` endpoint writes `data: <text>↵` with **no** blank line, so it is "SSE-flavoured" only; its own terminal shell parses that simpler format.

**Pause and predict**

5. A client reads the stream and gets `data: {"choices":[{"delta":{"content":"KV"}}]}↵↵data: {"choic` in one read. How many complete events does that read hold?

<details><summary>Answer</summary>

One. The second event has not finished (no blank line yet). The client must keep the leftover text and wait for the rest.

</details>

## 7 Handling Completion and Users Who Leave

**Kid version.** A dish has a life: **ordered, cooking, served, cleared**. Every stop on the journey keeps a little something for each order (a pile, a light, a peg on the peg board, pantry pages). When the order is over, *everybody* must clear their something. If anyone forgets, clutter piles up.

| What keeps something | What it keeps | Cleared when |
|---|---|---|
| Front desk | pile + light (`ack_map`, `event_map`) | the customer has taken the last spoonful (or leaves) |
| Card-reader | partial-word state per ticket | the finished marker passes through |
| Head waiter | a peg (row of the peg board) + pantry pages | the order finishes or is cancelled |
| Prefix cache | reusable beginnings | by design it *keeps* them, evicting later |

**When the customer just walks out.** Closing the browser tab closes the *phone line* to the front desk. The kitchen does not know. Unless someone tells it, the chef keeps cooking a dish nobody will eat, holding pantry pages. So the front desk must **send a cancel slip** down the journey:

```text
   customer leaves
        │
        ▼
   front desk: notices (between spoonfuls) ──▶ tears up its pile + light ──▶ sends AbortMsg
                                                                                  │
                                                          card-maker: re-labels it AbortBackendMsg
                                                                                  │
                                                          head waiter: finds the order (waiting? cooking?)
                                                                       and gives back its peg and pages
```

This is **cooperative cancellation**: nobody is forcibly stopped. A cancel slip is passed along and each station stops *future* work when it reads the slip. In kitchen terms: you call to the chef "table 7 left" rather than yanking the chef's apron. The dish already on the stove keeps cooking until the chef next looks up. So the cancel is **not instant** and a few more words may still be cooked.

**Late spoonfuls are dropped.** After the front desk tears up ticket 7, a spoonful for ticket 7 may still be in the post. The mail sorter checks "is ticket 7 still in my books?" and, if not, silently throws it away. I confirmed this in the toy replay.

**Slow customers.** If the customer reads slower than the chef cooks, spoonfuls pile up in the front desk's pile (memory). The chef is **never told to slow down** (no **backpressure**). For a single order that is bounded by `max_tokens`; for many slow orders it is not.

**Tiny numbers: the leaky ledger (my measurement of the toy).** The book points out (section 5.1 of the code half) that when the customer's loop stops at the `finished` spoonful, the generator's tidy-up lines at the bottom never run. I ran exactly that code in a stripped-down replay using only Python's standard library:

| Experiment | Result |
|---|---|
| Caller `break`s when it sees `finished` (what both the streaming and non-streaming paths do) | the ticket **stays** in both `ack_map` and `event_map` |
| Caller reads the generator to its very end | the ticket is removed |

Measured size of one forgotten ticket (an empty list, an `asyncio.Event`, two dictionary slots): about **1 KB** on my Mac's Python. Scaled up: 1,000 requests, about 1 MB; **1,000,000 requests, about 1 GB** of slowly growing memory in the front-desk process. Harmless for a classroom demo, a real problem for a service that runs for weeks. (The book names this as a known gap; the numbers are my toy's, not the live server's.)

**Where the story lies.** "Finished" and "cleaned up" are different events. The reply saying `finished=True` arrives *before* some stations have cleaned their stuff, and some stations here never do.

**Pause and predict**

6. A customer closes the browser during a *non-streaming* request. Does the front desk send a cancel slip?

<details><summary>Answer</summary>

No. Only the streaming path wraps its output in the "is the customer still there?" watcher. In non-streaming mode the front desk is busy waiting for the whole dish and never checks, so the kitchen cooks the whole answer for nobody. (Verified in the source; the code half shows the exact lines.)

</details>

7. Why can a cancelled order still produce one more word?

<details><summary>Answer</summary>

The cancel slip has to travel front desk → card-maker → head waiter, and the head waiter only reads new slips between passes. The pass already launched keeps going, and its result may come back before the cancel is applied.

</details>

## 8 Summary and Exercises

### 8.1 Summary

**Kid version.** The front desk speaks HTTP, gives every order a ticket, and never stands still. One mail sorter files every returning spoonful into its ticket's pile and rings its light. Customers either wait for the whole dish or taste spoonfuls through SSE. If a customer leaves, a cancel slip travels back; late spoonfuls are thrown away. Not every station tidies up perfectly: the front desk leaks a ticket per finished request, the card-reader may keep half-words for cancelled orders, and an over-long order is silently dropped.

```text
   ticket ─▶ tokenize ─▶ schedule ─▶ cook ─▶ detokenize ─▶ sort by ticket ─▶ customer
                                                               ▲
                                  cancel slip travels the other way, one station at a time
```

### 8.2 Exercises

**1. Why is a local text-generation function not enough for other apps? What do client, server and HTTP each do?**

<details><summary>Answer</summary>

Other apps are in other processes or on other machines; they cannot call your Python function or hold the weights themselves (and ten copies would cost ten times the GPU memory). The server keeps the model loaded and listens; clients send requests; HTTP is the agreed envelope format between them.

</details>

**2. What do the example request's method, path, headers and body hold? What do `messages` and `stream` mean?**

<details><summary>Answer</summary>

Method `POST` ("submit something"), path `/v1/chat/completions` (which counter), headers `Host` and `Content-Type` (where, and "body is JSON"), body = the order. `messages` is the conversation so far; `stream` chooses whole-dish or spoonfuls.

</details>

**3. Which stops carry a request out and back? Why separate network handling from model computation?**

<details><summary>Answer</summary>

Client, API server, tokenizer, scheduler, engine, (scheduler), detokenizer, API server, client. Network handling is fast and must always be responsive; computation is slow and must not block it.

</details>

**4. Why does a service need concurrency? How are interleaved replies kept from reaching the wrong client?**

<details><summary>Answer</summary>

Without it, a long request delays everyone behind it. Every request gets a `uid`; the same `uid` rides with each reply; the mail sorter files replies into `ack_map[uid]` and rings `event_map[uid]`.

</details>

**5. How do HTTP concurrency, computation parallelism and backend batching differ?**

<details><summary>Answer</summary>

Concurrency: the front desk manages many unfinished orders. Parallelism: work really runs at the same instant on multiple cores/devices. Batching: several requests share one model forward pass. Any can exist without the others.

</details>

**6. Which part of the user experience does streaming improve, and why does it not reduce model computation?**

<details><summary>Answer</summary>

Time to the first visible text. The model still does the same number of forward passes; only the delivery moment changes.

</details>

**7. What must a cancel notice pass through? Why is cancellation not instant? What needs cleanup?**

<details><summary>Answer</summary>

Front desk → card-maker → head waiter. Not instant because it travels through queues and is applied between passes; work already launched finishes. Cleanup targets: the front desk's pile and light, the card-reader's partial-word state, the peg and pantry pages, plus ignoring late replies.

</details>

**Extra drills (not in the book).**

8. A client is on its 6th turn and resends the full history, 20 tokens per message. How many tokens does the request carry? (220: 11 messages.)
9. Two replies for ticket 3 arrive; ticket 3's light was already ON when the second one arrived. How many wake-ups does the customer get, and how many replies does it see? (One wake-up; both replies, because the light is a flag and the pile holds the data.)
10. The mail sorter receives a reply for ticket 12, which has already been torn up. What happens? (It is dropped silently; no error.)
11. Why does the front desk send `AbortMsg` through the *card-maker* instead of straight to the head waiter? (In this design the card-maker's in-tray is the front desk's only line towards the backend; it simply re-labels the slip as `AbortBackendMsg` and passes it on.)
