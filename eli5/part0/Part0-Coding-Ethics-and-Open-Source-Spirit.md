# Part 0.1 Coding Ethics and Open-Source Spirit: ELI5 companion

Original: [Part0-Coding-Ethics-and-Open-Source-Spirit.md](../../course-material/eng/part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md). Same headings, same order.

> **Heads-up about these notes.** They are AI-written study aids, for *you*. The lesson below says: do not paste AI text into issues or PRs, write in your own words, and only submit what you can defend line by line. That applies to these notes too. If you ever want to share something from them, re-explain it yourself first.

## The one-minute version

**Kid version.** Imagine a huge shared kitchen that anyone may cook in for free. Because it is free and shared, there are manners: *clean up your own mess*, *taste your food before you hand it to someone*, *do not waste the cleaners' time*, and *leave a recipe card for the next cook*. This lesson is those manners, written for people who share a code kitchen.

```text
  What you give the project      What the project gives you
  ┌────────────────────┐         ┌──────────────────────────┐
  │ small, checked PRs │ ─────▶  │ PyTorch, SGLang, weights │
  │ written-up pitfalls│         │ for free, forever        │
  └────────────────────┘         └──────────────────────────┘
        ▲                                   │
        └───────── the cycle keeps turning ─┘
```

## Coding ethics

### 1. Take responsibility for your own code

**Kid version.** A robot can bake 500 cupcakes in 30 seconds. But a *person* still has to taste-test all 500 to see if any are bad. The robot's speed did not remove the work; it moved the work onto the taster. If you hand 500 untasted cupcakes to the taster, you have secretly given them an hour of homework.

```text
  Before AI:   you write 500 lines  ████████████████████ (1 hour)
               reviewer reads       ████████████████████ (1 hour)    fair trade

  With AI:     you "write" 500 lines █ (30 seconds)
               reviewer reads        ████████████████████ (1 hour)   the cost moved
                                     └── your job: read it first, so the
                                         reviewer reads *less* to trust it
```

| Kid word | Real word |
|---|---|
| Taster | Reviewer / maintainer |
| Handing over a plate | Opening a PR (pull request) |
| "I tasted it myself" | You read every line and can explain it |
| Secret homework | The invisible transfer of workload the lesson talks about |

**Tiny numbers.** Suppose a careful review runs about 200 lines per hour (an assumption; real speed varies a lot). Then 500 lines cost the reviewer 2.5 hours, while generating them cost you 30 seconds. A thousand-line PR you never read costs a stranger a whole workday. That is why the lesson says to keep PRs small and understood.

**The four questions, in plain words.**

| The lesson asks | What it really means |
|---|---|
| What problem does this PR solve? | Can you say, in one sentence, why it exists? |
| What does it deliver, and how can someone reproduce it? | Can a stranger run one command and see the result? |
| Which code does it touch, and how invasive is it? | Do you know the blast radius if you are wrong? |
| Pick any function at random: what if it were deleted? | The real test: do you understand it, or did you only receive it? |

**Where the story lies.** Using AI is *not* the sin; the lesson says so plainly. The sin is shipping what you have not read. A handwritten PR that you do not understand is just as bad.

**Pause and predict.**

1. You get a 40-line AI-written fix that makes a failing test pass. What is the quickest honest check that you own it?
2. Why is "it passes the tests" not enough to answer the fourth question?

<details>
<summary>Answers (try first)</summary>

1. Delete or change one function in your head and say what breaks, then actually do it locally and watch the test that should notice. If nothing notices, either the test is weak or you do not yet understand the code.
2. Tests only cover cases someone thought of. "What if this were deleted" tests whether you understand the *role* of the code, which is what a maintainer needs when it breaks at 2 a.m.

</details>

### 2. Communicate like a human

**Kid version.** If a friend asks "did you finish your homework?", you say "yes, the math part took 20 minutes." You do not hand them a five-paragraph essay that starts "Thank you for your valuable question." People trust short, specific answers, because those can only come from someone who did the thing.

**Picture: signal vs packaging.**

```text
  AI-flavored reply (200 words):
  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ packaging
  ██ ← ~15 words of real information

  Human reply (30 words):
  █████████████ ← nearly all information
```

The lesson's two samples, side by side:

| | Sample 1 (the bad one) | Sample 2 (the good one) |
|---|---|---|
| Opens with | "Thank you for your review! The point you raised is very valuable." | "Tried it." |
| Evidence | "I conducted an in-depth analysis" (no numbers) | "decode step at batch=n drops from xx ms to xx ms" |
| Attachments | none | before/after profiles |
| Cost stated | "memory needs to be considered" | "graph capture costs an extra yy of memory" |
| What the reader feels | "A bot wrote this; was the PR also unread?" | "This person measured it." |

**A reusable template** for any reply you write: *What I tried → what number changed → proof → what it costs.* Four short lines beat four paragraphs.

**Where the story lies.** AI can help you *organise* your thoughts. The rule is that the words you post are yours and you stand behind every sentence.

**Pause and predict.** A reviewer asks "does CUDA Graph help?" You have not measured. What is the honest reply?

<details>
<summary>Answer (try first)</summary>

"Not measured yet. I'll run decode at batch sizes 1, 8, 32 with and without it and post the numbers." Admitting you have not measured is more trusted than a confident essay.

</details>

### 3. Profile first, always

**Kid version.** Your room is messy and you have one hour. If you spend the hour polishing one tiny shelf that is 3% of the mess, the room still looks 97% messy. First *look* at where the mess is; then clean the biggest pile.

**Picture: the loop.**

```text
        ┌──────────────────────────────────────────────┐
        ▼                                              │
  1. profile  ──▶  2. find the biggest  ──▶  3. change ──▶ 4. profile again
  (measure)         time-eater                (one thing)    and compare
                                                                │
                                           ┌────────────────────┘
                                           ▼
                                  is it really faster, and *why*?
```

**Real name for the 3% story: Amdahl's law.** If a part takes a fraction `p` of the total time and you make it `s` times faster, the whole thing gets faster by:

```text
   speedup = 1 / ( (1 - p) + p / s )
```

**Tiny numbers** (computed from that formula):

| Part you optimized takes... | ...if you make it 2x faster | ...if you make it *infinitely* fast |
|---|---|---|
| 3% of total time | 1.015x (1.5% better) | 1.031x (3.1% better) |
| 10% | 1.053x | 1.111x |
| 50% | 1.333x | 2.000x |
| 90% | 1.818x | 10.000x |

Read the last column: a part that is 3% of the time can never give you more than ~3% back, *even if you delete it entirely*. That is the "two weeks lost" the lesson warns about, and the one hour of measuring that would have saved them.

**The three things a performance PR must carry**, with the kid reason for each:

| Must include | Kid reason |
|---|---|
| Before/after numbers | "Show me the scale, not just that you feel lighter." |
| A reproducible command | "Let me weigh it myself." |
| Profiling evidence (why it is faster) | "Show me *where* the weight came off, so I know it was your change and not luck." |

**Where the story lies.** Real programs have several parts changing at once, and speeding one part up can move the bottleneck to another (a *new* biggest pile). That is why step 4 says "profile again", not "done".

**Pause and predict.** You find a function that is 40% of runtime and make it 4x faster. What total speedup do you expect?

<details>
<summary>Answer (try first)</summary>

`1 / ((1 - 0.4) + 0.4/4) = 1 / (0.6 + 0.1) = 1.43x`. Even a 4x win on 40% of the time gives 1.43x overall. This is the arithmetic to run *before* starting.

</details>

### 4. Summary: what kind of first PR is unwelcome

**Kid version.** Five ways to annoy the cleaners, and what to do instead:

| Unwelcome PR | What the reviewer feels | Do this instead |
|---|---|---|
| Thousands of AI lines the author never read | "I am being asked to do your job." | Keep it small; read all of it |
| "Faster!" with no numbers | "Prove it." | Before/after + command + profile |
| Core module refactor with no design | "This could break everything." | Discuss first; make it small |
| New dependency / abstraction / default changed without talk | "Who decided this?" | Open an issue, get agreement |
| Ten spam PRs at once | "Noise." | One PR, one real motivation |

**What a good first PR looks like:** small, complete, fixes a real existing issue, comes with verification. The lesson's motto: *understand first, then ask, then contribute.* Even *finding a problem and filing a clear issue* counts as a real contribution.

**Pause and predict.** You want to add a new default value for a server flag. Which row of the table is that, and what is the first step?

<details>
<summary>Answer (try first)</summary>

Row 4. The first step is not code: open an issue or ask in the discussion, explain what you measured, and wait for agreement before changing a default.

</details>

## Open-source spirit

### One for all, all for one

**Kid version.** A neighbourhood potluck: everyone brings a dish, everyone eats. You did not cook the bread on the table, but you ate it; so next time you bring something. In this course nearly everything you use (PyTorch, FlashAttention, SGLang, the weights you download, even the course itself) came off someone else's potluck table.

```text
        ┌───────────┐   take    ┌───────────────┐
        │ the shared │ ───────▶ │ you, learning │
        │  table     │          └───────┬───────┘
        └───────────┘ ◀──────── give ───┘
        pitfall write-ups, docs, translations, repro of benchmarks, bug reports
```

**Tiny numbers.** You spend 10 minutes writing up a pitfall. If 100 later people would each have lost 2 hours to it, those 10 minutes saved about 200 hours.

**What counts as contributing** (not only code): documentation, translation, reproducing a benchmark, helping pin down an issue, writing a clean pitfall report.

**The one distinction to remember.** Wanting experience on your résumé is fine; almost everyone has some of that motive. The problem is *confusing the means with the end*: when the goal becomes "number of merged PRs" instead of "problem solved", the project fills with junk PRs and the good ones drown.

**The three closing promises:** respect other people's time; respect the craft; take responsibility for your own code.

**Where the story lies.** The lesson notes that a bad commit does not collapse a bridge, so there is no legal liability. The manners are *chosen*, not enforced by law. That is exactly why they work only if each person holds them.

**Pause and predict.** Which of these is the better first contribution: (a) a 600-line refactor of the scheduler, (b) a clear issue reporting a crash with the command, the error text and `nvidia-smi` output?

<details>
<summary>Answer (try first)</summary>

(b). It is small, verifiable, real, and costs the reviewer minutes. (a) lands in the "refactor core modules without design" bucket from section 4.

</details>

## Check yourself before moving on

- Can you explain, in your own words, why 30 seconds of generation does not mean 30 seconds of total cost?
- Given any "X% faster" claim, can you compute the best-case end-to-end win from Amdahl's formula?
- Can you list the three things every performance PR carries?
