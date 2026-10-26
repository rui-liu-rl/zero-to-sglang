# Part II Chapter 5 HTTP and Concurrent Requests (Code Walkthrough): ELI5 companion

Original: [Chapter5_HTTP-and-Concurrent-Requests_code.md](../../course-material/eng/part2/Chapter5_HTTP-and-Concurrent-Requests_code.md). Same headings, same numbers. Concept half: [Chapter5_HTTP-and-Concurrent-Requests.md](./Chapter5_HTTP-and-Concurrent-Requests.md). Cast and picture legend: [../README.md](../README.md).

Files named here carry ELI5 comments on the `eli5-comments` branch of `/Users/rui/code/mini-sglang` (paths relative to `python/minisgl/`). The main one is `server/api_server.py`; its header comment draws the same front desk.

**Run it yourself, on your Mac.** The front-desk logic does not need a GPU. [`../tools/ch5_frontend_toy.py`](../tools/ch5_frontend_toy.py) re-implements `new_user`, `listen` and `wait_for_ack` with only Python's standard library and runs seven experiments (the traces below are its real output):

```bash
python3 -I eli5/tools/ch5_frontend_toy.py
```

## The one-minute version

**Kid version.** The front desk's whole file fits in one picture:

```text
   HTTP slip ─▶ [pydantic form check] ─▶ new_user() = TICKET # ─▶ send TokenizeMsg ─▶ (goes away; front desk is free)
                                                                                           │
   ... later, on its own ...                                                               ▼  backend works
   MAIL SORTER listen():  reply arrives ─▶ file in ack_map[ticket] ─▶ switch on event_map[ticket]
                                                                                │
   CUSTOMER wait_for_ack(ticket):  sleep until light ─▶ clear ─▶ grab pile ─▶ yield spoonfuls ─▶ stop at finished
```

| Part | Where | Job |
|---|---|---|
| The order forms | `OpenAICompletionRequest`, `Message`, `GenerateRequest` | Check what the customer wrote |
| The ledger | `FrontendManager` | Ticket counter, piles, lights, out-tray, in-tray |
| The mail sorter | `FrontendManager.listen` | One endless task that files replies |
| The customer's wait | `FrontendManager.wait_for_ack` | Turn "light on" into spoonfuls |
| The formatters | `stream_generate`, `stream_chat_completions` | Wrap spoonfuls in the right text format |
| The watcher | `stream_with_cancellation`, `abort_user` | Notice a walk-out and cancel |

## 1 Learning Objectives

| Goal | Where |
|---|---|
| Locate request models, routes, conversion to an internal message | Sections 2.1, 2.2 |
| How `new_user`, `listen`, `wait_for_ack` combine | Sections 3.2 to 3.4 |
| Non-streaming collection and streaming SSE encoding | Sections 4.1, 4.2 |
| Cancellation creation, conversion, handling, and cleanup gaps | Section 5 |

## 2 From HTTP Input to Backend Work

Run these in a GPU environment (the original shows the commands): the server with `python -m minisgl --model "Qwen/Qwen3-0.6B"` and, in another terminal, the `curl -N` request. `-N` tells curl not to buffer, so you see spoonfuls as they arrive. Reading the source does not need a GPU.

| Responsibility | Source | Kid gloss |
|---|---|---|
| Parse, state, respond | `OpenAICompletionRequest`, `v1_completions`, `FrontendManager` in `server/api_server.py` | Front desk |
| Internal message structures | `TokenizeMsg`, `UserReply`, `AbortMsg`, `AbortBackendMsg` in `message/` | The printed internal forms |
| Tokenize, detokenize, convert aborts | `tokenize_worker` in `tokenizer/server.py` | The card room |
| Schedule and reclaim | `_process_one_msg`, `_process_last_data` in `scheduler/scheduler.py` | The head waiter |

### 2.1 Defining Input with a Request Model

```python
class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int = 16
    stream: bool = False
```

*Line by line:*

| Line | Kid words |
|---|---|
| `class ... (BaseModel)` | A **printed order form**. pydantic checks every incoming slip against it *before* your code runs |
| `model: str` | Required box. Customers must write a model name; the server then **ignores it** (it only serves the model loaded at start) |
| `prompt: str \| None = None` | Optional box: a plain string |
| `messages: list[Message] \| None = None` | Optional box: a chat. Give this **or** `prompt` |
| `max_tokens: int = 16` | If left blank, 16 cards. A small default borrowed from the OpenAI convention |
| `stream: bool = False` | If left blank, whole-dish |

The real class also has `temperature` (default **1.0**), `top_k` (-1 = off), `top_p` (1.0), `ignore_eos`, and four fields that are **accepted but ignored**: `n`, `stop`, `presence_penalty`, `frequency_penalty`. `Message` has `role` (only `system`, `user`, `assistant` allowed) and `content`.

**What pydantic does for free.** Sending `{"max_tokens": "lots"}` gets an automatic `422` error explaining which box is wrong. Sending `{"max_tokens": "128"}` is quietly converted to the number 128 (pydantic's default "lax" conversion). Your code later sees a clean Python object, never raw JSON.

**Where the story lies.** "OpenAI-style" is a *shape*, not a promise. Gaps worth memorising so you never trust them by accident:

| Looks supported | Actually |
|---|---|
| `model` | Required, ignored |
| `n`, `stop`, penalties | Accepted, never read |
| `usage.prompt_tokens` etc. | Always 0 |
| `finish_reason` | Always `"stop"` (even when `max_tokens` cut the answer) |
| `temperature` omitted | HTTP default 1.0 means *random* sampling, but `/generate` and `LLM.generate` use `SamplingParams` default 0.0 (*greedy*). Same words, different behaviour |

### 2.2 Establishing a Boundary Between HTTP and the Inference Backend

```python
state = get_global_state()
if req.messages:
    prompt = [msg.model_dump() for msg in req.messages]
else:
    assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
    prompt = req.prompt

uid = state.new_user()
await state.send_one(
    TokenizeMsg(
        uid=uid,
        text=prompt,
        sampling_params=SamplingParams(
            ignore_eos=req.ignore_eos,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        ),
    )
)
```

*Line by line:*

| Line | Kid words |
|---|---|
| `get_global_state()` | Fetch the one shared ledger (it was made in `run_api_server`). Asserts if used too early |
| `if req.messages: prompt = [m.model_dump() ...]` | A chat: turn each pydantic `Message` into a plain dict `{role, content}`. The tokenizer will later apply the **chat template** to this list |
| `else: prompt = req.prompt` | A plain string: tokenized as is |
| `assert req.prompt is not None` | If neither was given, fail loudly (this becomes an error response) |
| `uid = state.new_user()` | Take a ticket; make an empty pile and an unlit light for it |
| `TokenizeMsg(uid, text, sampling_params)` | The internal form: ticket, the text (string or chat list), and the customer's settings |
| `SamplingParams(...)` | Only five of the customer's fields make it in. `n`, `stop`, ... are dropped here |
| `await state.send_one(...)` | Put the form in the out-tray towards the card-maker. `await` = "I may be put on hold while the tray accepts it; serve others meanwhile" |

**What happens next in this handler.**

```text
   streaming?  yes ─▶ return StreamingResponse(...)   (connection stays open; the generator keeps producing)
               no  ─▶ async for ack in wait_for_ack(uid): glue pieces ... then return one JSON
```

**The ZeroMQ lines.** `send_one` ends up at a **push queue** (`ZmqAsyncPushQueue`, the async out-tray). The reply comes back on a **pull queue** (`ZmqAsyncPullQueue`, the async in-tray). Messages are packed with msgpack (compact binary, like JSON in a suitcase). The async variants never block the event loop while waiting.

**Why this boundary matters.** Everything left of `send_one` is cheap (microseconds). Everything right of it is slow (seconds) and happens in *other processes*. So the front desk can hand off and immediately serve someone else.

**Pause and predict**

1. A request arrives with both `messages` and `prompt`. Which wins?

<details><summary>Answer</summary>

`messages`. The `if req.messages:` branch is checked first and `prompt` is ignored.

</details>

2. Which of `temperature`, `top_k`, `n` ends up inside the `TokenizeMsg`?

<details><summary>Answer</summary>

`temperature` and `top_k` (via `SamplingParams`). `n` is accepted by the form but never copied.

</details>

## 3 How the API Server Manages Concurrent Requests

### 3.1 `async` Does Not Automatically Eliminate Blocking

```python
@app.post("/generate")
async def generate(req: GenerateRequest):
    text = blocking_generate(req.prompt)   # <-- no `await`
    return {"text": text}
```

**Kid version.** `async def` is a *costume*, not a power. If the waiter walks into the kitchen and stands over the pot with no way to be interrupted, the whole restaurant freezes, costume or not. Only an `await` is a point where the waiter can be called away.

**Tiny numbers (my toy, Test 5).** Two handlers that each "work" for 0.5 s, and a second customer who arrives just after they start:

| Handler body | Second customer is first looked at after |
|---|---|
| `time.sleep(0.5)` (blocking: no `await`) | **0.51 s** |
| `await asyncio.sleep(0.5)` (yields) | **0.00 s** |

Same 0.5 s of "work", completely different front desk. mini-sglang's handlers never run the model; they `await` the backend.

### 3.2 Isolating Request State with uid

```python
ack_map: dict[int, list[UserReply]]
event_map: dict[int, asyncio.Event]
```

```python
def new_user(self) -> int:
    uid = self.uid_counter
    self.uid_counter += 1
    self.ack_map[uid] = []
    self.event_map[uid] = asyncio.Event()
    return uid
```

*Line by line:*

| Line | Kid words |
|---|---|
| `uid = self.uid_counter` | Read the next ticket number |
| `self.uid_counter += 1` | Next one is one higher. Never reused while the server runs; restarts at 0 on restart |
| `self.ack_map[uid] = []` | An **empty pile** for this ticket |
| `self.event_map[uid] = asyncio.Event()` | An **unlit light** for this ticket |
| `return uid` | Hand the ticket to the handler |

`asyncio.Event` has three moves: `set()` (light on), `clear()` (light off), `await wait()` (sleep until the light is on; if it already is, return at once).

**`UserReply`** (`message/frontend.py`) is the form the card-reader sends back:

| Field | Meaning |
|---|---|
| `uid` | Whose ticket |
| `incremental_output` | Only the **new** letters since last time |
| `finished` | `True` on the last one |

### 3.3 Dispatching Results from a Shared Channel

```python
async def listen(self):
    while True:
        msg = await self.recv_tokenizer.get()        # book excerpt names this variable `reply`
        for msg in _unwrap_msg(msg):
            if msg.uid not in self.ack_map:
                continue
            self.ack_map[msg.uid].append(msg)
            self.event_map[msg.uid].set()
```

*(Cosmetic drift: the book writes `reply`; the source writes `msg`. Behaviour is identical.)*

*Line by line:*

| Line | Kid words |
|---|---|
| `while True` | The sorter never retires |
| `await self.recv_tokenizer.get()` | Sleep until a packet is in the in-tray. **This `await` is why the whole server doesn't freeze**: the sorter is parked, other tasks run |
| `_unwrap_msg(msg)` | A packet is either one `UserReply` or a `BatchFrontendMsg` holding several; get a flat list either way |
| `if msg.uid not in self.ack_map: continue` | **Late spoonfuls for torn-up tickets are dropped.** (Test 4: `sorter: DROP late reply for uid 0`) |
| `self.ack_map[msg.uid].append(msg)` | File it in that ticket's pile |
| `self.event_map[msg.uid].set()` | Light that ticket's light |

**Starting the sorter lazily.** `_create_listener_once` runs on the first `send_one`:

```python
def _create_listener_once(self):
    if not self.initialized:
        asyncio.create_task(self.listen())
        self.initialized = True
```

`asyncio.create_task` needs an event loop that is *already running*, and one only exists after uvicorn starts serving. At construction time there is none, so the sorter is started on the first request.

Now the customer's side:

```python
async def wait_for_ack(self, uid: int):
    event = self.event_map[uid]

    while True:
        await event.wait()
        event.clear()

        pending = self.ack_map[uid]
        self.ack_map[uid] = []
        ack = None
        for ack in pending:
            yield ack
        if ack and ack.finished:
            break

    del self.ack_map[uid]
    del self.event_map[uid]
```

*Line by line:*

| Line | Kid words |
|---|---|
| `async def ... yield` | An **async generator**: it hands out spoonfuls one by one. Callers write `async for ack in state.wait_for_ack(uid)` |
| `event = self.event_map[uid]` | Remember *my* light |
| `await event.wait()` | Sleep until my light is on (other customers keep being served) |
| `event.clear()` | Switch it off *right after waking*, so the next `set()` can wake me again |
| `pending = self.ack_map[uid]` | Take the whole pile |
| `self.ack_map[uid] = []` | Leave a fresh empty pile in its place for future spoonfuls |
| `for ack in pending: yield ack` | Hand over each spoonful. At `yield`, control goes to the caller (it may write to the network) |
| `if ack and ack.finished: break` | After handing over the pile, if the **last** spoonful was the final one, stop looping |
| `del self.ack_map[uid]`, `del self.event_map[uid]` | **Tidy-up: this is the part that never runs** (section 5.1) |

**Why clear → take → replace without an `await` in between.** There is no `await` between `event.clear()` and `self.ack_map[uid] = []`, so the sorter (a different task on the *same* event loop) can't squeeze in between those lines. Single-threaded cooperative scheduling replaces a lock. (Once the loop reaches `yield`, the caller may suspend and the sorter can append to the *new* list and relight the light.)

**Tiny numbers (my toy, Test 1).** Four replies in the order A1, B1, B2, A2, each customer breaking on `finished`:

```text
   A got: A1 A2        B got: B1 B2        delivery order: A:'A1 ' B:'B1 ' B:'B2' A:'A2'
```

**Tiny numbers (my toy, Test 2).** Three replies put in the queue before the customer woke:

```text
   wake-ups: 1   replies per wake-up: [3]   got: ['x', 'y', 'z']
```

One wake-up, three replies. The light says "go look at your pile"; the pile has the data.

### 3.4 Checking Isolation Through Producers and Consumers

**Kid version.** One **producer** (the sorter) fills piles; many **consumers** (one per customer) empty their own pile. No customer ever reads from the in-tray directly, so they never fight over the same slip.

```text
   in-tray (ZeroMQ) ─▶ ONE listen() ─▶ ack_map[0] ─▶ consumer 0
                                     ├▶ ack_map[1] ─▶ consumer 1
                                     └▶ ack_map[2] ─▶ consumer 2
```

**Pause and predict**

3. While consumer 0 is writing its spoonfuls to the network (suspended at `yield`), a new reply for ticket 0 arrives. Where does it go?

<details><summary>Answer</summary>

Into the **new** empty list that `wait_for_ack` placed in `ack_map[0]`, and `event_map[0]` is set again. On its next loop turn the consumer wakes immediately because the light is already on.

</details>

4. Why not give each customer its own ZeroMQ socket instead of one shared sorter?

<details><summary>Answer</summary>

The backend reaches the front desk through one shared channel (it does not know which customers exist); one listener plus a ticket lookup is the simplest way to route. (This is my reading of the design, not a statement from the book.)

</details>

## 4 Streaming and Non-Streaming Responses

### 4.1 Non-Streaming: Collecting the Complete Result

```python
full_content = ""
async for ack in state.wait_for_ack(uid):
    full_content += ack.incremental_output
    if ack.finished:
        break
```

*Line by line:* start with an empty string; for every spoonful that arrives, glue its new letters on; stop at the final one. The handler then returns one JSON dictionary (`id`, `object: "chat.completion"`, `choices[0].message.content`, `usage` of zeros).

**Not blocking.** While this loop sits at `async for`, the event loop is serving other customers. "Non-streaming" is about *what this customer sees*, not about the server stalling.

**Not watched.** This path has **no** "is the customer still there?" check. If the client disconnects, generation continues to the end. (Section 5.3.)

### 4.2 Streaming: Returning Results Incrementally with SSE

```python
return StreamingResponse(
    state.stream_with_cancellation(
        state.stream_chat_completions(uid),
        request,
        uid,
    ),
    media_type="text/event-stream",
)
```

| Piece | Kid words |
|---|---|
| `StreamingResponse(...)` | "Don't send one dish; keep the line open and send whatever this generator yields, whenever it yields it" |
| `stream_chat_completions(uid)` | The **formatter**: turns each `UserReply` into an SSE chunk |
| `stream_with_cancellation(..., request, uid)` | The **watcher** wrapped around it |
| `media_type="text/event-stream"` | The header telling the client "this is SSE" |

The formatter:

```python
async def stream_chat_completions(self, uid: int):
    first_chunk = True
    async for ack in self.wait_for_ack(uid):
        delta = {}
        if first_chunk:
            delta["role"] = "assistant"
            first_chunk = False
        if ack.incremental_output:
            delta["content"] = ack.incremental_output

        chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()

        if ack.finished:
            break

    # send final finish_reason
    end_chunk = {
        "id": f"cmpl-{uid}",
        "object": "text_completion.chunk",
        "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"
    logger.debug("Finished streaming response for user %s", uid)
```

| Line | Kid words |
|---|---|
| `first_chunk = True` | "Have I announced the speaker yet?" |
| `async for ack in self.wait_for_ack(uid)` | Same customer-wait as before; each pass is one spoonful |
| `delta["role"] = "assistant"` on the first one | The very first chunk says *who* is talking, then never again |
| `if ack.incremental_output:` | An empty spoonful is possible (the detokenizer may hold back half a character). Then the chunk has a `delta` with no `content` |
| `"object": "text_completion.chunk"` | OpenAI-style label (not the exact one real OpenAI uses for chat, hence "style", not "compatible") |
| `yield f"data: {...}\n\n".encode()` | Wrap in SSE form, **blank line at the end**, and turn to bytes for the wire |
| `if ack.finished: break` | Leave the loop after the last spoonful |
| `end_chunk ... "finish_reason": "stop"` | A closing chunk that always says `"stop"` (**even when cut by `max_tokens`**) |
| `yield b"data: [DONE]\n\n"` | Tell the client to stop reading |

**What a client really receives for a 2-word answer** (shape only):

```text
   data: {"id":"cmpl-0","object":"text_completion.chunk","choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0,"finish_reason":null}]}

   data: {"id":"cmpl-0","object":"text_completion.chunk","choices":[{"delta":{"content":" world"},"index":0,"finish_reason":null}]}

   data: {"id":"cmpl-0","object":"text_completion.chunk","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}

   data: [DONE]
```

Notice that the role and the first text can arrive together in chunk 1, so a client must not assume they are separate events.

**The simpler `/generate` stream.** `stream_generate` writes `data: <text>\n` (no blank line, no JSON) then `data: [DONE]\n`. It is "SSE-flavoured": the interactive terminal shell parses exactly this simple format by stripping `data: ` and the trailing newline.

**Pause and predict**

5. A request ends with `finished=True` in a reply that also has `incremental_output = "!"`. How many `data:` lines does a streaming client get after that reply, counting the reply's own chunk?

<details><summary>Answer</summary>

Three: the chunk carrying `"!"`, the closing chunk with `finish_reason: "stop"`, and `data: [DONE]`.

</details>

## 5 Request Completion, Cancellation, and State Cleanup

### 5.1 Normal Completion

The path on the finishing side:

```text
   scheduler._process_last_data:  finished? ─▶ remove from running set ─▶ _free_req_resources (peg + pages)
        └─▶ DetokenizeMsg(finished=True) ─▶ card-reader: clears this ticket's half-word state
                                              └─▶ UserReply(finished=True) ─▶ front desk
```

And on the front desk: both consumers (`stream_chat_completions` and the non-streaming loop) **stop at `finished`**. That is the problem. A generator is a paused function. When the caller breaks, the generator is left paused at its last `yield`; it is later closed, and closing delivers a `GeneratorExit` at the `yield`, so the two `del` lines **below the loop never execute**.

**Tiny numbers (my toy, Test 3).**

| What the caller does | `uid` in `ack_map`? | `uid` in `event_map`? |
|---|---|---|
| `break` when `finished` (what the real code does) | **yes** | **yes** |
| reads the generator to its very end | no | no |

**Tiny numbers (my toy, Test 6).** One forgotten ticket is an empty list + an `asyncio.Event` + two dict slots, measured at **about 1 KB**. A million requests would be about **1 GB** held forever. Invisible in a demo, a slow leak in a service that stays up for weeks.

**What a fix looks like (idea only; this branch changes comments only).**

```python
# sketch, not applied: put the tidy-up in finally so closing the generator also runs it
async def wait_for_ack(self, uid):
    event = self.event_map[uid]
    try:
        while True:
            ...
    finally:
        self.ack_map.pop(uid, None)
        self.event_map.pop(uid, None)
```

(`finally` runs on `GeneratorExit`, unlike statements placed after the loop. Toy Test 7 tries exactly this: the caller still `break`s on `finished`, and after a 50 ms wait for Python's generator-finalizer the ticket is gone from both maps.) Logged as quirk **Q8** in [`../NOTES-open-questions.md`](../NOTES-open-questions.md).

**Where the story lies.** "The reply said finished" is not "everything is tidy". The scheduler tidies its things; the card-reader tidies its things; the front desk leaves a ticket behind.

### 5.2 Streaming Client Disconnects and Cooperative Cancellation

```python
async def stream_with_cancellation(self, generator, request: Request, uid: int):
    try:
        async for chunk in generator:
            # detect if the client has disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected for user %s", uid)
                raise asyncio.CancelledError
            yield chunk
    except asyncio.CancelledError:
        asyncio.create_task(self.abort_user(uid))
        raise

async def abort_user(self, uid: int):
    await asyncio.sleep(0.1)
    if uid in self.ack_map:
        del self.ack_map[uid]
    if uid in self.event_map:
        del self.event_map[uid]
    logger.warning("Aborting request for user %s", uid)
    await self.send_one(AbortMsg(uid=uid))
```

*Line by line:*

| Line | Kid words |
|---|---|
| `async for chunk in generator` | Pull each spoonful from the formatter |
| `await request.is_disconnected()` | "Between spoonfuls, is the customer still at the counter?" (Starlette checks the connection) |
| `raise asyncio.CancelledError` | Pretend a cancel arrived (the same exception the web server raises if it cancels the handler) |
| `yield chunk` | Otherwise pass the spoonful on |
| `except asyncio.CancelledError:` | Whichever way the cancel came, we land here |
| `asyncio.create_task(self.abort_user(uid))` | Do the cleanup as its **own task**, because this generator is itself being torn down and may not get to `await` |
| `raise` | Re-raise so the cancellation keeps propagating upward |
| `await asyncio.sleep(0.1)` | A 0.1 s pause. The source gives no reason; my guess is that it lets in-flight chunks drain before the books are torn up |
| `del self.ack_map[uid]`, `del self.event_map[uid]` | Tear up pile and light (guarded, because they may already be gone) |
| `await self.send_one(AbortMsg(uid=uid))` | Send the cancel slip to the card-maker |

The cancel then takes two more hops (excerpts):

```python
# tokenizer/server.py, tokenize_worker: job 3
batch_output = BatchBackendMsg(data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg])
```

```python
# scheduler/scheduler.py, _process_one_msg: AbortBackendMsg branch
req_to_free = self.prefill_manager.abort_req(msg.uid)
req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
if req_to_free is not None:
    self._free_req_resources(req_to_free)
```

| Piece | Kid words |
|---|---|
| `AbortBackendMsg(uid=...)` | The card-maker only re-labels `AbortMsg` as the form the head waiter understands |
| `prefill_manager.abort_req(uid)` | Look on the **waiting clipboard** (`pending_list`). If the order is there, remove it. It returns the order's half-read `chunked_req` if it had one (that one already holds a peg and a ticket), else `None` |
| `req_to_free or decode_manager.abort_req(uid)` | If not found there, look among the tables **still eating** (`running_reqs`) |
| `if req_to_free is not None` | Only an order that actually held resources needs them returned |
| `_free_req_resources` | Give back the peg (`table_manager.free`) and the pantry pages (`cache_req(finished=True)`, which keeps the reusable prefix in the shared index) |

The whole trip:

```text
   customer leaves
      └▶ front desk watcher: raise CancelledError
           └▶ abort_user (own task): 0.1 s, tear up pile + light, send AbortMsg
                └▶ card-maker: AbortMsg → AbortBackendMsg
                     └▶ head waiter, next turn of its loop: remove order, free peg + pages
                          └▶ any spoonful still in the post reaches listen(); uid unknown → dropped
```

**Cooperative, not instant.** Each hop is a queue. The head waiter reads new mail only at the top of its loop; with the default overlapping loop, the batch launched on the previous turn is already on the GPU and its results are processed *after* the abort. So a few more words can still be computed and discarded. Hypothesis **H3** in the notes asks what happens when that already-launched batch is the order's last one.

### 5.3 Error Paths Not Yet Fully Handled

Four gaps, in kitchen words:

| # | Gap | Kitchen picture | Where |
|---|---|---|---|
| 1 | Non-streaming has no walk-out check | A customer who ordered "to be served whole" leaves, and nobody looks up from the pot | `v1_completions`, non-stream branch |
| 2 | Aborts do not reach the card-reader | The cancel slip goes to the head waiter, but the card-reader's note "ticket 7: half a word so far" is never torn up | `tokenize_worker` job 3 (no call into the detokenizer manager) |
| 3 | Over-long prompt is dropped silently | The head waiter takes the slip, sees it cannot be cooked, throws it away and **tells nobody**; the customer waits forever, and their pile and light are never cleared | `_process_one_msg`, `max_output_len <= 0` branch (Q3) |
| 4 | Frontend cleanup on finish never runs | Section 5.1 | `wait_for_ack` (Q8) |

A complete lifecycle would need: an error reply type, a "finished or failed" marker that every station understands, and cleanup that runs even when something went wrong (`finally`).

**Pause and predict**

6. A prompt is 10,000 tokens and the model allows 8,192. What does the customer see?

<details><summary>Answer</summary>

Nothing, forever (the connection stays open). The scheduler logs a warning and returns; no `UserReply` is ever produced. The customer's pile and light stay in the front desk's books too.

</details>

### 5.4 Slow Clients and Result Backlog

```text
   chef ─▶ card-reader ─▶ listen() ─▶ ack_map[uid] = [ s1, s2, s3, ..., s400 ]  ──▶ slow customer's network
                                       ▲ keeps growing while the network is slow, and the chef is never told to slow down
```

`StreamingResponse` is paced by the client's network, but `listen()` is **not**: it keeps filing replies into `ack_map[uid]` as fast as they come. One request is capped by `max_tokens` (and by the model's maximum length), so one slow reader costs a bounded pile. Many slow readers with long outputs can add up. A production system would cap the pile, set time limits, or cancel the order of a customer who is far behind. That "feed the consumer's speed back to the producer" idea is **backpressure**; this design has none.

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** Form check → ticket → hand off. A mail sorter files replies by ticket and lights lights. Customers sleep on their light, grab their pile, hand out spoonfuls. Whole-dish and spoonful modes differ only in what the front desk does with the spoonfuls. Walk-outs send a cancel slip back through card-maker to head waiter. The front desk still leaves tickets behind, non-streaming customers who leave are not noticed, and some failures give no reply at all.

### 6.2 Exercises

**1. Why can a synchronous generation function inside an `async def` route block other connections? Where must execution yield?**

<details><summary>Answer</summary>

`async def` only allows suspension at `await` points. A blocking call has none, so the event loop cannot run other tasks while it runs (my toy: second customer waited 0.51 s instead of 0.00 s). The route must `await` the backend: `state.send_one(...)` and `wait_for_ack(...)`.

</details>

**2. Trace `messages`, `max_tokens`, `stream` through `v1_completions`.**

<details><summary>Answer</summary>

`messages` (if non-empty) → list of dicts → `TokenizeMsg.text`. `max_tokens` → `SamplingParams.max_tokens` → later clamped by the scheduler to what fits the model's length. `stream` → `if req.stream:` picks `StreamingResponse(...)`; otherwise the code collects and returns one JSON.

</details>

**3. What roles do `ack_map` and `event_map` play and why is neither enough alone?**

<details><summary>Answer</summary>

`ack_map` holds the replies (data); `event_map` wakes the waiting coroutine (signal). A flag with no list can't carry the data; a list with no flag forces the consumer to poll.

</details>

**4. Walk through wait → clear → take-and-replace → yield. Why are several replies preserved when `set()` is called several times before one wake-up?**

<details><summary>Answer</summary>

`set()` only turns a flag on; every reply was appended to the list. One wake-up detaches the whole list, so all replies are delivered (toy Test 2: 1 wake-up, 3 replies). `clear()` and the list swap share a stretch with no `await`, so `listen` cannot interleave there.

</details>

**5. Inject interleaved A and B replies. How would you check isolation and cleanup?**

<details><summary>Answer</summary>

Run [`ch5_frontend_toy.py`](../tools/ch5_frontend_toy.py) Test 1 (each consumer gets only its own, in order) and compare `ack_map`/`event_map` keys afterwards. The leftover keys are the Q8 gap.

</details>

**6. Locate role, incremental text, finish reason and `[DONE]` in `stream_chat_completions`.**

<details><summary>Answer</summary>

Role: the `first_chunk` block. Text: `delta["content"]`. Finish reason: `end_chunk`. `[DONE]`: the last `yield b"data: [DONE]\n\n"`.

</details>

**7. If the caller breaks on `finished`, does `wait_for_ack` run its trailing deletions? Design a GPU-free check.**

<details><summary>Answer</summary>

No. Check: build the three methods in a small script, consume with `break`, then assert the `uid` is still in both maps; consume without `break`, assert it is gone. That is Test 3 of the toy. Cancellation and errors additionally need `finally`-style cleanup and an error reply type.

</details>

**Extra drills (not in the book).**

8. The listener receives a `BatchFrontendMsg` holding replies for tickets 4, 4, 9. How many lights are switched on? (Two: ticket 4's light is set twice, ticket 9's once.)
9. `abort_user` finds `uid` already missing from both maps. Does it still send `AbortMsg`? (Yes. The `if` guards only protect the `del`s; the `send_one` runs regardless.)
10. Where would you add a check so a non-streaming request cancels when the client leaves? (Wrap the non-streaming `wait_for_ack` loop in the same disconnect test used in `stream_with_cancellation`, and call `abort_user` when it fires.)
11. Why is the 0.1 s `sleep` inside `abort_user` risky if the next request reuses the same `uid`? (It can't: `uid_counter` only grows. The sleep only delays the cleanup.)
