# Chapter 5 Introduction to Benchmark: ELI5 companion

Original: [Chapter5_Introduction_to_Benchmark.md](../../course-material/eng/part1/Chapter5_Introduction_to_Benchmark.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

## The one-minute version

**Kid version.** "It feels faster" is not evidence. A **benchmark** is a fair race: same track, same rules, a stopwatch, and the result written down so anyone can rerun it. For a restaurant (our inference server) you time three things: **how long until the first bite arrives**, **how long between bites after that**, and **how many customers you can feed per hour without anyone waiting too long**.

```text
   one customer's meal, on a timeline

   order sent        first bite             last bite
      │                  │                      │
      ├──── TTFT ────────┤                      │
      │    (queue +      ├─ gap ─ gap ─ gap ─ … ┤        gap = ITL (each one);  TPOT = their average
      │     prefill)     │                      │
      ├─────────────────── E2E ─────────────────┤
```

Two lists of numbers, never mixed up:

| Question | Metrics |
|---|---|
| How fast is **one** customer served? (latency) | TTFT, TPOT, ITL, E2E |
| How much can the **whole kitchen** serve? (throughput) | Throughput, Goodput |

The thing most beginners get wrong: **look at the slowest customers, not the average**, and **look at latency before throughput**.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| Latency vs throughput; TTFT, TPOT, ITL, throughput, Goodput | Per-customer speed vs whole-kitchen output |
| Percentiles and tail latency | Judge by the unlucky customers (P99), not the average |
| Choose a dataset; design a reproducible test | Same food orders, same rules, every time |
| Run SGLang's benchmark tools and read the results | Press start; know which number to look at first |

## 2 What is a benchmark and why does it matter?

### 2.1 Introduction

**Kid version.** When a new model is announced you see a table of **scores** (maths, coding). Websites like Artificial Analysis also show **speed** and **price**, and plot a **Pareto frontier**: the set of models where you can't improve one thing (say, smartness) without paying in another (price or speed). A smaller model on the frontier can be the better *buy*.

There are two completely different kinds of benchmark; mixing them up is a classic mistake:

| | Model capability benchmark | Inference workload benchmark (this chapter) |
|---|---|---|
| Question | Does it **answer well**? | Does it **run quickly and reliably**? |
| Examples | MMLU, GSM8K, HumanEval, SWE-bench | TTFT, TPOT, throughput, P99 |
| What you feed it | Test questions with answers | A stream of requests (a *workload*) |
| What you measure | Correctness | Time and capacity |
| Kitchen version | A taste test of the food | A stopwatch on the service |

Unless said otherwise, "benchmark" below means the stopwatch kind (also called an AI-infra benchmark).

### 2.2 Why use benchmarks?

1. **To know the limits.** You can't tell a change helped by feel. Run the same requests before and after.
2. **To quantify an optimisation.** Two versions are only comparable under identical conditions.
3. **To plan capacity.** How many customers can one GPU feed, and how slow do they get when it's busy? Measure before launch, not after the pager goes off.

(The Part 0 lesson's rule "profile first" is this idea applied to engineering decisions.)

## 3 Core metrics

### 3.1 Latency metrics

**Kid version.** Four stopwatches on one customer:

| Metric | Kid words | Real definition |
|---|---|---|
| **TTFT** | "How long until my first bite?" | Time from sending the request to the first generated token. Includes waiting in the queue, the prefill, and the first decode step |
| **TPOT** | "How long between bites, on average?" | Average time per output token **after** the first (mainly decode speed) |
| **ITL** | "How long was *this particular* gap?" | Time between two adjacent tokens. Same thing as TPOT but looked at one gap at a time, so it shows **hiccups** |
| **E2E** | "How long was the whole meal?" | Request sent to last token received |

**The book's example, with the arithmetic.** Input 1,000 tokens, output 200, measured TTFT = 250 ms, TPOT = 25 ms.

```text
   remaining tokens after the first = 200 - 1 = 199
   remaining time   = 25 ms x 199 = 4,975 ms
   E2E              = 250 + 4,975 = 5,225 ms  ≈ 5.2 s
   prefill share    = 250 / 5,225 = 4.8%     ← decode is 95% of the wait
```

So for **long answers**, TPOT is what you feel; for **long prompts with short answers** (question answering over a document), TTFT dominates. This is Chapter 2's prefill/decode split showing up in the stopwatch.

**TPOT to speed.** 25 ms per token is 40 tokens per second. A common comfort floor is 100 ms per token = 10 tokens/s, which at about 0.75 words per token is about 7.5 words/s = 450 words per minute. Slower than that and reading feels laggy.

**ITL vs TPOT, with a hiccup.** Suppose the gaps (ms) were `25, 25, 25, 25, 400, 25, 25, 25`. TPOT (the mean) is `(4 x 25 + 400 + 3 x 25) / 8 = 72 ms`, which sounds mediocre. ITL shows one 400 ms stall (a scheduling hiccup or a long prefill hogging the GPU) and everything else perfect. The cure differs: the mean tells you nothing about *where* to look.

**Where the story lies.** Definitions vary slightly between tools (some count the first token in TPOT). The book's choice, `(E2E - TTFT) / (tokens - 1)`, is a common one; always check the definition when comparing numbers from different places.

### 3.2 Throughput metrics

| Metric | Kid words | Real definition |
|---|---|---|
| **Throughput** | "Meals served per hour" | Tokens per second or requests per second, across everyone |
| **Goodput** | "Meals served per hour *that arrived hot enough*" | Throughput counting only requests that meet the latency target (the SLA) |

**Why Goodput exists.** You can inflate throughput by cramming in more customers. Everyone's food gets cold, but meals-per-hour looks great.

**Tiny numbers.** 1,000 requests in 100 s = 10 requests/s throughput. The SLA says TTFT must be at most 500 ms, and 30% of requests miss it. Goodput = 700 / 100 = **7 requests/s**. The 3 requests/s that missed were work done, but not work the business can count.

**A worked sweep: why "more concurrency" is a trade-off, not a free win.** Take a toy GPU: 14 GB of weights, 1.6 TB/s memory, 66 GB left for notes, and every chat is 300 tokens long (0.5 MiB of notes per token). Each decode step reads the weights once plus each active chat's notes (Chapter 2's model), so:

```text
   step time = ( 14 GB + C x 0.157 GB ) / 1.6 TB/s      C = number of simultaneous chats
```

| Concurrency `C` | TPOT (step time) | Speed per user | Total throughput | Meets TPOT <= 25 ms? |
|---|---|---|---|---|
| 1 | 8.8 ms | 113 tok/s | 113 tok/s | yes |
| 8 | 9.5 ms | 105 tok/s | 839 tok/s | yes |
| 32 | 11.9 ms | 84 tok/s | 2,690 tok/s | yes |
| 64 | 15.0 ms | 67 tok/s | 4,255 tok/s | yes |
| 128 | 21.3 ms | 47 tok/s | 6,000 tok/s | yes |
| **165** | **25.0 ms** | **40 tok/s** | **6,608 tok/s** | **yes (the limit)** |
| 256 | 33.9 ms | 30 tok/s | 7,548 tok/s | no |
| 419 (notes memory full) | 49.9 ms | 20 tok/s | 8,390 tok/s | no |

(Model, not measurement: ignores prefill and scheduling overhead.) Read it three ways:

1. Total throughput keeps rising (113 to 8,390 tok/s) but every user gets slower (113 to 20 tok/s each).
2. If the promise is TPOT <= 25 ms, the useful capacity is **165 users and 6,608 tok/s**, not the 8,390 tok/s at saturation. The extra 1,800 tok/s is throughput with no Goodput value.
3. Past 419 chats the notepads are full; extra customers wait in a queue, so **TTFT** (not TPOT) is what blows up.

That is the shape of the book's Figure 5.5: throughput climbs and flattens; latency keeps climbing; the best operating point is the knee, before latency runs away.

**Little's law (a free sanity check).** For any steady system: `concurrency = throughput x latency`. With 32 chats in flight and E2E = 5.2 s, throughput must be `32 / 5.2 = 6.1 requests/s`, which at 200 output tokens is about 1,225 tok/s. If your measured numbers don't satisfy this, something is wrong with the measurement.

**Which metric matters where:**

| Scenario | Look at | Because |
|---|---|---|
| Chat | TTFT | People notice the pause before text starts |
| Long writing | TPOT | Most of the time is spent streaming |
| Offline batch / evaluations | Throughput | Nobody is waiting live |
| Voice or multi-step agents | TTFT **and** tail latency | A chain of calls is as slow as its slowest link (see 5.1) |

## 4 How to design a benchmark

**Kid version.** First decide **what you're testing**. Then choose **matching food orders**. Then **fix every rule** so the race is fair.

### 4.1 Define the evaluation target and workload

The dataset must match what you want to measure:

> Measuring coding ability with conversation data produces meaningless results.

SGLang ships four tools, aimed at different depths of the engine:

```text
   client ──HTTP──▶ [ API server ] ──▶ [ scheduler ] ──▶ [ model/kernels ]
   bench_serving             ◀───────── covers everything ─────────▶
   bench_one_batch_server    ◀───────── one batch, whole path ──────▶
   bench_offline_throughput                  ◀──── no network ──────▶
   bench_one_batch                                     ◀─ kernels only ▶
```

| Tool | HTTP? | Scheduler? | Kid description |
|---|---|---|---|
| `bench_serving` | yes | yes | Real customers through the front door: the closest to production |
| `bench_one_batch_server` | yes | yes | One big party, timed through the front door |
| `bench_offline_throughput` | no | yes | Skip the front desk; how much can the kitchen cook? |
| `bench_one_batch` | no | no | Time the chef's hands only (kernel analysis) |

`bench_serving` is enough most of the time. Its built-in request sets:

| Dataset | What it is | Use it for |
|---|---|---|
| `sharegpt` | Real conversations | Chat-like load |
| `random` | Random fixed-length requests | Ceiling tests only; real traffic isn't uniform |
| `generated-shared-prefix` | Requests that share a beginning | Measuring the benefit of prefix reuse |
| `agentic-trace` | Multi-turn traces with tool calls | Agent workloads |

mini-sglang also ships small benchmark scripts of its own (under `mini-sglang/benchmark/` and `mini-sglang/python/minisgl/benchmark/`); they follow the same ideas on a smaller scale.

### 4.2 Fix runtime parameters

| Rule | Kid reason | Notes |
|---|---|---|
| Fix concurrency **or** request rate | Same crowd every race | See open vs closed loop below |
| Fix hardware, model, precision, quantization | Same kitchen, same chef | FP16 vs INT8 can differ by over 2x in throughput |
| Warm up first | The first customer wakes the chef | CUDA graphs and the allocator need to reach steady state |
| Fix the output length (or its distribution) | Same meal size | Else the token counts are not comparable |

**Open loop vs closed loop (the two ways to apply load).**

| | `--max-concurrency N` (closed loop) | `--request-rate R` (open loop) |
|---|---|---|
| Kid version | N tables, always full; a new customer sits down only when one leaves | Customers arrive at R per second whether or not there's a free table |
| What happens when the server is slow | Load drops automatically | The queue grows without bound |
| Good for | Finding steady-state capacity | Testing behaviour under real arrival patterns, including overload |

Tiny numbers: a server that can finish 10 requests/s, hit with an open-loop 12 requests/s, falls behind by 2 requests every second: after a minute the queue holds 120 requests and TTFT keeps growing. The same server under closed-loop concurrency just settles at 10 requests/s.

## 5 How to interpret benchmark results

### 5.1 Look at latency before throughput

**Kid version of the three-step reading rule:**

1. **Decide the promise first** (the SLA): "TTFT <= 500 ms, TPOT <= 50 ms".
2. **Look at the spread, not the average.**
3. **Check that every rule matches** between runs (model, precision, hardware, concurrency, dataset, output length).

**Percentiles, explained.** Sort all the measurements. **P50** (the median) is the middle value: half the requests were faster. **P99** is the value that 99% of requests beat; only the slowest 1% were slower. Made-up TTFT samples (ms), already sorted, 20 requests:

```text
   150 158 165 172 178 183 188 194 199 202 | 206 211 217 224 232 248 300 380 520 862
                                        P50                                  P90  P95  P99
```

| Statistic | Value |
|---|---|
| Mean | **259 ms** |
| P50 (median) | **202 ms** |
| P90 | 380 ms |
| P95 | 520 ms |
| P99 | **862 ms** (4.3x the median) |

The mean (259) is pulled above the median (202) by the long tail, and says nothing about the 862 ms customer. This mirrors the book's Figure 5.8 (P50 about 202 ms, mean 246 ms, P99 862 ms). "Percentile" here uses the nearest-rank rule: P99 of 20 samples is the 20th (last) one, which also shows that P99 needs *many* samples to be meaningful.

**Why tails matter more than they look: tail amplification.** If one call has a 1% chance of being slow, a task that makes several calls is much more likely to hit at least one slow call:

| Calls in the task | Chance at least one is in the slowest 1% |
|---|---|
| 1 | 1.0% |
| 5 | 4.9% |
| 20 (an agent working through steps) | **18.2%** |
| 100 | **63.4%** |

(`1 - 0.99^k`.) That is why agent and voice workloads care about P99, not the mean.

### 5.2 Common pitfalls

**Kid version for each:**

| # | Pitfall | The kid story | A number |
|---|---|---|---|
| 1 | Throughput without latency | "We serve more meals!" (all cold) | Concurrency 64 beats 16 on throughput, but P99 TTFT went 300 ms to 3 s |
| 2 | Mean without tail | "Average wait is fine" while one customer waited forever | 99 requests at 100-300 ms and one at 2 s: the mean is only 218 ms |
| 3 | Not warmed up | Timing the first customer who woke the chef | Made-up: 3,200, 410, 305, 298, 301 ms. Mean of the first five: 903 ms; steady: 329 ms (2.7x off) |
| 4 | Treating `bench_one_batch` as online speed | Timing only the chef's hands, then promising that to customers | A 20 ms kernel number doesn't include queueing, scheduling, network |
| 5 | Comparing throughput across different lengths | Comparing a sandwich shop to a banquet | 500-in/100-out vs 4,000-in/500-out cannot be compared |
| 6 | Ignoring tokenizer differences | Counting "words" in different languages | The same text is 1,000 tokens for model A and 1,500 for B. If both write 100 characters/s, B "scores" 150 tok/s vs A's 100, though neither is faster |
| 7 | Token/s without Goodput | Counting cold meals | 30% of requests over the 500 ms limit still count in raw throughput |
| 8 | Single-machine conclusions applied to multi-GPU | Chefs sharing one recipe need to talk to each other | Tensor parallelism adds NVLink/RDMA traffic; the best concurrency on one GPU (say 16) may be wrong on eight |
| 9 | Unfixed random seed | Different dice rolls each race | Sampling kernels use random numbers; unfixed seeds can swing results by more than 5%, hiding or inventing an improvement |

**A practical checklist** (steal this for your own runs): model + precision + hardware + driver; dataset; input/output length distribution; concurrency or rate; warm-up done; seed fixed; at least a few hundred requests; report mean **and** P50/P90/P99 for TTFT, TPOT and E2E; report throughput **and** Goodput against a stated SLA.

**Pause and predict.** A teammate says "my change improved average TTFT from 220 ms to 200 ms". What two further questions would you ask before believing it?

<details>
<summary>Answer (try first)</summary>

(1) What happened to P95/P99? A 10% mean improvement can hide a worse tail. (2) Were the run conditions identical (same dataset, concurrency, warm-up, seed, output lengths) and how many requests were sampled? A 20 ms difference could be noise.

</details>

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** A benchmark is a fair, repeatable race. Time the first bite (TTFT), the gaps between bites (TPOT/ITL), and the whole meal (E2E); count meals per hour (Throughput) and hot meals per hour (Goodput). Judge by the slow tail (P99), not the average. Use matching food orders, fix every rule, warm up first, and compare only like with like. The same methodology will be used all through Part II to see whether each change to mini-sglang really helps.

### 6.2 Exercises

**1. What do TTFT, TPOT, ITL measure? Why does optimising TPOT help long-output workloads more?**

<details><summary>Hint</summary>

TTFT = request to first token (queue + prefill + first step); TPOT = mean gap after the first token; ITL = each individual gap. For long outputs the time is almost all decode steps (the book's example: 95%), so shaving TPOT shaves the total; TTFT is a single fixed chunk.

</details>

**2. If mean and P99 differ by an order of magnitude, what does it mean? Why watch P99?**

<details><summary>Hint</summary>

The distribution has a heavy tail: most requests are fast, but a few are very slow. The mean hides them; stability promises (SLAs) are about the slow ones, and with several calls per task the tail hits many users (18% at 20 calls).

</details>

**3. Chat users say there's a long pause after sending, but once text starts it's fast. Which metric, what to fix?**

<details><summary>Hint</summary>

TTFT is abnormal, TPOT is fine. Look at queueing (too much concurrency or too small a batch budget), long prefills blocking others, missing prefix reuse for the growing chat history, or cold-cache effects. Do not tune decode kernels.

</details>

**4. Why is a direct token/s comparison across models unreliable?**

<details><summary>Hint</summary>

Different tokenizers turn the same text into different numbers of tokens (pitfall 6), so token/s measures different amounts of real content.

</details>

**5. Which benchmark for a maths-reasoning model, and which for a speech-synthesis model? Why not share a dataset?**

<details><summary>Hint</summary>

Maths: a capability benchmark like GSM8K (and an inference benchmark with matching length distributions if you also care about serving cost). Speech synthesis: audio-quality and intelligibility measures, plus a latency benchmark using text-to-audio request sets. Inputs, outputs and success criteria differ completely.

</details>

**6. List the considerations for a reproducible, comparable benchmark.**

<details><summary>Hint</summary>

Use the checklist in 5.2: model, precision, hardware and software versions, dataset, input/output lengths, concurrency or rate, warm-up, seed, sample size, and the metrics to report (percentiles and Goodput vs a stated SLA).

</details>

**Extra drills (not in the book).**

7. With 64 chats in flight and E2E = 8 s, what is the request throughput? (Little's law: 64 / 8 = 8 requests/s.)
8. Why can't you read P99 from a 50-request run? (Nearest-rank P99 of 50 samples is the 50th, i.e. the single worst sample; it is the maximum, very noisy.)
9. A change makes TPOT 30% lower but TTFT 50% higher. For which of the four scenarios in the table (chat, long writing, offline batch, agents) is this a good trade?
