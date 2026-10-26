# Chapter 1 Introduction to LLM: ELI5 companion

Original: [Chapter1_Introduction_to_LLM.md](../../course-material/eng/part1/Chapter1_Introduction_to_LLM.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

## The one-minute version

**Kid version.** A large language model (LLM) is a *very* good guesser of the next word. You give it some words; it guesses the next one; you add that word and ask again. Do this a few hundred times and you have a paragraph. All the clever things it does (answering, translating, writing code) come from getting that one guess right, over and over.

```text
   "The cat sat on the"  ──▶ [ LLM ] ──▶  mat 55%   floor 20%   sofa 12%  ...
                                              │
                          pick one (say "mat") ▼
   "The cat sat on the mat" ──▶ [ LLM ] ──▶  .  70%   and 8%  ...
                                   ⋮
                          (repeat until it says "I'm done")
```

The rest of this chapter has two jobs: (1) tell you what an LLM is made of (the **Transformer**), and (2) hand you the vocabulary the whole course keeps using (token, parameter, context window, autoregressive).

**The inference connection, stated early.** Every single guess requires running the *entire* model once over the words so far. A model has billions of numbers inside. Doing that fast, cheaply, and for thousands of people at once is the whole subject of this course.

## 1 Learning Objectives

**Kid version of the four goals.**

| Original objective | After this chapter you can tell a 5-year-old... |
|---|---|
| 1. Definition, history, boundaries | "It guesses the next word. It got good around 2017-2022. It can be confidently wrong and doesn't know today's news." |
| 2. Typical applications | "Chatbots, writing helpers, coding helpers. All of them are waiting for the model to answer." |
| 3. Transformer: positional encoding, multi-head attention, layer norm + residual, feed-forward | "Words take turns *looking at each other*, then *each thinks alone*, with a safety shortcut and a thermostat, repeated many floors high." |
| 4. Core concepts: token, parameter, context window, autoregressive | "Words are chopped into numbered cards; the model is a giant pile of numbers; it can only hold so many cards at once; it writes one card at a time." |

## 2 Definition, Brief History, and Boundaries of LLMs

### What is an LLM

**Kid version.** On your phone, the keyboard suggests the next word. An LLM is that idea, scaled up enormously: a giant calculator trained by reading a huge pile of text, practising "what comes next?" billions of times, until its guesses became good enough to look like understanding.

**Why does "just guessing the next word" lead to so many skills?** To guess the next word well in a math book, you have to be good at math. In a story, you have to track who is who. In code, you have to follow the syntax and the logic. The guessing game is a trick for forcing the model to learn all of that.

```text
   text:   "17 + 25 = "          the best guess needs arithmetic
   text:   "def add(a, b): re"   the best guess needs programming
   text:   "Paris is the capital of"   the best guess needs facts
```

**Real words.**

| Kid word | Real word |
|---|---|
| The guess | Next-token prediction |
| The table of percentages | A probability distribution over the vocabulary |
| Picking | Sampling (random by percentage) or greedy (always the top one) |
| "Large" | Many parameters *and* a lot of training data |
| The pile of numbers it learned | Parameters / weights |

**Tiny numbers: greedy vs sampling.** Suppose the table is `mat 0.55, floor 0.20, sofa 0.12, bed 0.08, moon 0.05`. Greedy always says "mat". Sampling says "mat" 55 times out of 100, "floor" 20 times, and "moon" 5 times. A *temperature* knob flattens or sharpens the table before picking. (Part II chapter 3 builds exactly this.)

### History in one table

| Year | What happened | Kid gloss |
|---|---|---|
| 2017 | Transformer paper "Attention Is All You Need" | A new, parallel-friendly way for words to look at each other |
| 2018-2019 | BERT (reads both directions) and GPT (writes left to right) | Two schools: "fill in the blank" vs "keep writing" |
| 2020 | GPT-3, 175 billion parameters | Big enough to learn tasks from a few examples |
| 2022 on | ChatGPT; open models (LLaMA, Qwen, DeepSeek) | Everyone gets one; architectures keep improving (RoPE, GQA/MLA, RMSNorm, SwiGLU, all met below) |

### Capability boundaries

**Kid version.** The model is a student who read a lot up to last year and *always* answers, even when unsure. That causes four typical problems:

| Limit | Why it happens (from the "guess" view) |
|---|---|
| **Hallucination** | The model always produces *some* probability table. A fluent guess and a true fact look the same inside it |
| **Knowledge cutoff** | It only learned from text up to its training date |
| **Limited memory of long text** | It can only look at a fixed number of cards at once (the context window) |
| **No live view of the world** | It sees only the cards you give it, unless a tool feeds it fresh ones |

**Where the story lies.** "It just autocompletes" undersells it: the guesser is shaped further after pretraining (instruction tuning, reinforcement learning), so it follows requests instead of merely continuing text.

**Pause and predict.** (1) If the model never says "I don't know" by default, how could you make it say so? (2) Why does a model forget the start of a very long document?

<details>
<summary>Answers (try first)</summary>

1. Train it (or prompt it) to give "I don't know" a high probability when evidence is missing, or give it retrieval so the answer is *in* its context (RAG, section 5.4).
2. The context window is a hard limit on how many tokens it sees in one go. Anything outside is simply not an input.

</details>

## 3 Typical Applications of LLMs

### 3.1 Dialogue and Q&A

**Kid version.** The chat helper. You ask, it answers, and today it can see pictures and search the web. Many people now ask it before they search.

### 3.2 Text Generation

**Kid version.** The writing helper: drafts, reports, minutes, and translations. The small, fast "flash" models do much of the translation work that older translation software used to do.

### 3.3 Code-related

**Kid version.** The coding helper. You say what you want; it writes a working draft. Coding agents run it in a loop: write, run, read the error, fix.

### 3.4 The Foundation of AI Applications: Inference Cost and Latency

**Kid version.** Every one of those helpers is only as good, fast, and cheap as the kitchen cooking the answers. Two numbers matter to users:

| Name | Kid words | Real meaning |
|---|---|---|
| **TTFT** (time to first token) | How long until the first bite arrives | Delay from sending the request to the first token |
| **TPOT** (time per output token) | How long between bites afterwards | Average gap between successive tokens |

```text
  send ──────────────▶ first token ──▶ ──▶ ──▶ ──▶ ──▶ last token
  |<------- TTFT ------->|<-TPOT->|<-TPOT->|<-TPOT->| ...
```

The book says a bar for "keeps up with a person reading" is at most 150 ms per token. That is 1000 / 150 = about 6.7 tokens per second, which at roughly 0.75 words per token is about 5 words per second, or 300 words per minute.

**Why memory is the villain: the notepad grows.** While generating, the model keeps "notes" about every token so far (the **KV cache**, previewed in section 4.2 and treated in Part I chapter 4). Notes grow with *both* the number of conversations and their lengths:

```text
   notes memory  =  (number of conversations)  x  (tokens in each)  x  (bytes per token)
```

**Tiny numbers** (a 7-billion-parameter model, a common size):

| Fact | Number |
|---|---|
| Weights in half precision (2 bytes each) | 7 B x 2 = 14 GB |
| One 80 GB GPU, minus weights | 66 GB left for notes (82% of the card) |
| Notes per token (a classic multi-head layout) | 0.5 MiB |
| One 3,000-token conversation | about 1.5 GB |
| Conversations that fit at once | 66 / 1.5 = about 44 |

That is where the book's "more than 70% of memory is the KV cache" comes from: once the weights are loaded, almost everything left becomes notes. If notes are thrown away to make room, they must be recomputed later, which is the extra cost the chapter quotes.

**Where the story lies.** The book's cost figures (2-3x extra, 78% of a monthly bill) are quoted from a news article about one kind of deployment. Treat them as an illustration of the *direction*, not as constants.

## 4 Core Architecture: Transformer

### The whole picture first

**Kid version.** Picture a tall building. Each floor is identical in design. On every floor, two things happen:

1. **Meeting room (attention).** Each word looks at the other words and pulls in what's useful.
2. **Private office (feed-forward).** Each word thinks alone about what it just collected.

Two helpers on every floor: a **hallway shortcut** (residual connection) so information can skip the floor if the floor has nothing to add, and a **thermostat** (normalization) that keeps the numbers from getting too hot or too cold.

```text
   token ids    "the" "cat" "sat"                     (3 integers)
       │
       ▼  look up a vector for each id
   ┌──────────────────────────────┐
   │ embedding   [3 words x d]    │   d = vector length ("hidden size"), e.g. 1024
   └──────────────┬───────────────┘
                  ▼         one floor (a "decoder layer") ... repeated L times
        ┌──────────────────────────────────────────────┐
        │  thermostat ▶ MEETING ROOM ▶ + shortcut        │
        │  thermostat ▶ PRIVATE OFFICE ▶ + shortcut      │
        └──────────────────┬───────────────────────────┘
                           ▼
                  final thermostat
                           ▼
                  output layer ▶ a table of scores over every word in the vocabulary
                           ▼
                  pick the next word
```

**Encoder vs decoder.** The original 2017 design had two stacks: an *encoder* (reads the input) and a *decoder* (writes the output). Every chat model in this course is **decoder-only**: one stack that reads and writes, left to right. When the book says "stack the decoder and encoder N times", that is the 2017 picture; the real engines you will read run the decoder-only version.

**Real words.**

| Kid word | Real word |
|---|---|
| Floor | Decoder layer / Transformer block |
| Number of floors | `num_layers` (28 for Qwen3-0.6B) |
| Width of each word's vector | `hidden_size` (1024 for Qwen3-0.6B) |
| Meeting room | Self-attention |
| Private office | Feed-forward network (FFN / MLP) |
| Hallway shortcut | Residual connection |
| Thermostat | Normalization (LayerNorm / RMSNorm) |

### 4.1 Positional Encoding: Sinusoidal Positional Encoding

**Kid version.** Imagine the meeting room has no name tags and no seating chart. Everyone just throws their words on the table, so "man bites dog" and "dog bites man" look identical. We need to give each word a **seat number**.

The original trick: give each position a unique "fingerprint" made of **clock hands that spin at different speeds**. One hand spins fast (tells apart neighbours), another spins slowly (tells apart far-away positions), like the seconds, minutes and hours of a clock. Add that fingerprint to the word's vector.

**Picture: two clocks, two speeds.**

```text
   position:      0      1      2      3
   fast clock   ↑ 0°   ↗ 57°  → 115°  ↘ 172°        one turn every ~6.3 positions
   slow clock   ↑ 0°   ↑ 0.6° ↑ 1.1°  ↑ 1.7°        one turn every ~628 positions
```

**Tiny numbers** (model width `d_model = 4`, so two clocks; each clock gives a sine and a cosine):

| position | fast sin | fast cos | slow sin | slow cos |
|---|---|---|---|---|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | -0.416 | 0.020 | 1.000 |
| 3 | 0.141 | -0.990 | 0.030 | 1.000 |

Every row is different, so every position has a unique fingerprint. The fast clock separates neighbours; the slow clock barely moves, but over hundreds of positions it distinguishes "early" from "late". Bigger models use many more clocks (`d_model / 2`).

**Reading the formula.**

```text
   PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
   PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
```

`10000^(2i/d)` is the *wavelength divisor* for clock number `i`: clock 0 divides by 1 (fast), the last clock divides by nearly 10000 (very slow).

**The relative-position property, shown with a rotation.** For one clock with speed `w`, moving from position `p` to `p + k` is just turning its hand by a fixed angle `k*w`:

```text
   [sin(p+k)]   [  cos(kw)  sin(kw) ] [sin(p)]        (from sin(a+b) = sin a cos b + cos a sin b,
   [cos(p+k)] = [ -sin(kw)  cos(kw) ] [cos(p)]         and the same for cos)
```

The matrix depends on `k` only, not on `p`. So "3 seats further on" is the same operation anywhere in the sentence. That is the book's "naturally supports relative positions".

**Real words.**

| Kid word | Real word |
|---|---|
| Seat number fingerprint | Positional encoding |
| Clock speed | Frequency (set by `10000^(2i/d)`) |
| Added to the word | `X = Token + PE(pos)` |

#### What real LLMs do instead: RoPE (the one in this repo's code)

**Kid version.** Instead of *adding* a seat number to each word, **rotate** the word's question-arrow (Query) and label-arrow (Key) by an angle proportional to the seat number. When two words later compare arrows (their dot product), the answer depends only on **how far apart their seats are**, not on where they sit.

```text
   word at seat m: arrow turned by m*θ          word at seat n: arrow turned by n*θ
   how well they match  =  cos( (m - n) * θ )   ← only the DIFFERENCE appears
```

**Tiny numbers** (θ = 0.5, both arrows start pointing the same way):

| seat of query m | seat of key n | offset m - n | match score |
|---|---|---|---|
| 3 | 1 | 2 | 0.540 |
| 5 | 3 | 2 | 0.540 |
| 100 | 98 | 2 | 0.540 |
| 3 | 2 | 1 | 0.878 |

Same offset, same score, wherever it sits. That is why RoPE handles "relative position" so cleanly.

**Where it lives in the code.** [`layers/rotary.py`](../../../mini-sglang/python/minisgl/layers/rotary.py) builds a table once (`_cos_sin_cache`, one cos and sin per position and per clock) and `AttentionLayer.forward` in `layers/attention.py` rotates `q` and `k` in place **before** the attention backend stores `k` in the KV cache. Two consequences you will use later:

- The cached keys already have their seat number baked in.
- A cached prefix is only reusable if it sits at the same seats (positions starting from 0). That is exactly the situation prefix caching (Part II chapter 8) is built around.

**Where the story lies.** Clocks and arrows are pictures. The real thing is a bank of 2-D rotations applied to pairs of numbers inside each head's vector.

**Pause and predict.** With sinusoidal encoding, can you tell position 1 from position 2 using only the *slow* clock's values (0.010 vs 0.020)? Is the difference useful to the model?

<details>
<summary>Answer (try first)</summary>

Technically yes (they differ), but the gap is tiny and easily lost among other numbers. The fast clock carries the neighbour-level signal; the slow clock carries the long-range signal. That is the reason for many clock speeds.

</details>

### 4.2 Multi-Head Attention

**Kid version.** The meeting room. Every word writes three things on index cards:

- a **Question card (Query)**: "what am I looking for?"
- a **Label card (Key)**: "this is what I contain."
- a **Gift card (Value)**: "this is what I will hand over if you pick me."

Each word compares its Question against everyone's Labels. A good match scores high. Scores are turned into percentages that add up to 100% (**softmax**). Then the word takes that percentage-weighted **mix of Gifts**. That mix is the word's new, enriched self.

**Multi-head** means several independent meetings happen at once, each with its own Questions and Labels, so one meeting might track grammar while another tracks "who is *she*?". At the end, the results are glued together.

**Picture: one head, one word.**

```text
   my Question ──▶ compare with every Label ──▶ scores ──▶ softmax ──▶ percentages
                    (dot product)                                         │
   Gift 1  Gift 2  Gift 3   ◀── weight each Gift by its percentage ◀─────┘
        └─────────┬─────────┘
                  ▼
          add them up = my new vector
```

**Real words.**

| Kid word | Real word |
|---|---|
| Question / Label / Gift | Query `Q` / Key `K` / Value `V` |
| How well they match | `Q K^T` (dot products) |
| Scale the scores down | divide by `sqrt(d_k)` |
| Make percentages | `softmax` |
| Mix of gifts | multiply the percentages by `V` |
| One meeting | one attention head |
| Glue the meetings together | concatenate heads, then multiply by `W^O` |

#### 4.2.1 The Approach and Limitations of Single-Head Attention

**Tiny numbers, all of it computed.** Three words: "the", "cat", "sat", each a vector of 4 numbers. To keep it checkable, we use identity projections (`Q = K = V = X`):

```text
   X = [ the: [1,0,1,0]
         cat: [0,1,0,1]
         sat: [1,1,0,0] ]
```

Scores `Q K^T / sqrt(4)` (the divide-by-2 is the book's `sqrt(d_k)`):

```text
              the   cat   sat
   the  [     1.0   0.0   0.5 ]       "the" matches itself best
   cat  [     0.0   1.0   0.5 ]
   sat  [     0.5   0.5   1.0 ]
```

**The causal mask.** When *generating*, word `i` may only look at words `0..i`: later words do not exist yet. So the upper-right triangle is blocked before softmax:

```text
              the   cat   sat
   the  [     1.0    -     -  ]
   cat  [     0.0   1.0    -  ]
   sat  [     0.5   0.5   1.0 ]
```

After softmax each row becomes percentages:

```text
              the     cat     sat
   the  [    100%      0%      0%  ]
   cat  [   26.9%   73.1%      0%  ]
   sat  [   27.4%   27.4%   45.2%  ]
```

And the output of each word is that mix of Gifts (`V = X`):

```text
   the: 1.0*[1,0,1,0]                                   = [1.000, 0.000, 1.000, 0.000]
   cat: 0.269*[1,0,1,0] + 0.731*[0,1,0,1]               = [0.269, 0.731, 0.269, 0.731]
   sat: 0.274*the + 0.274*cat + 0.452*[1,1,0,0]         = [0.726, 0.726, 0.274, 0.274]
```

"sat" ends up as a blend of itself and what came before. That blending is the whole job of attention.

**Why divide by `sqrt(d_k)`?** A dot product of two vectors that each have `d` numbers adds up `d` products. For random numbers with spread 1, the total's spread is `sqrt(d)`: for `d = 64` it is 8. Large scores make softmax collapse to "all weight on one word", which kills learning. Dividing by 8 brings the spread back to 1.

**Why one meeting is not enough.** One meeting gives one set of percentages per word. Language needs several simultaneous questions ("who did it?", "when?", "what does *it* refer to?").

#### 4.2.2 The Design Idea of Multi-Head Attention

**Kid version.** Split each word's vector into `h` equal slices. Each slice goes to its own meeting. A meeting works in a smaller space (`d_k = d_model / h`), so the total work stays the same as one big meeting, but you get `h` different views.

**Same toy, two heads** (`h = 2`, so each head sees 2 of the 4 numbers: head 0 gets the first two, head 1 the last two):

```text
   head 0 weights            head 1 weights
   the [100%   0%    0%]     the [100%   0%    0%]
   cat [33.0% 67.0%  0%]     cat [33.0% 67.0%  0%]
   sat [24.8% 24.8% 50.3%]   sat [33.3% 33.3% 33.3%]   ← head 1 can't tell them apart here

   outputs glued together:
   the: [1.000, 0.000 | 1.000, 0.000]
   cat: [0.330, 0.670 | 0.330, 0.670]
   sat: [0.752, 0.752 | 0.333, 0.333]
```

Notice "sat" gets different mixes in the two heads (50% on itself vs 33% each). Different heads, different opinions. Real heads learn their roles in training; nobody assigns them.

#### 4.2.3 The Detailed Computation Process of Multi-Head Attention

**Step 1: Split into heads (the `reshape` and `transpose`).**

```python
Q = Q.reshape(batch_size, seq_len, h, d_k)   # [bs, seq_len, h, d_k]
Q = Q.transpose(1, 2)                        # [bs, h, seq_len, d_k]
```

*Line by line:* `Q` starts as one long vector per word (`d_model` numbers). `reshape` cuts that vector into `h` slices of `d_k` numbers each. `transpose(1, 2)` swaps the "which word" axis and the "which slice" axis, so the head index moves forward. Why? Because GPU matrix code does best when the heads are treated like extra *batch* items.

```text
   toy: bs=1, seq_len=3, d_model=4, h=2, d_k=2
   start        [1, 3, 4]    one 4-vector per word
   reshape      [1, 3, 2, 2] each word's 4-vector cut into 2 slices of 2
   transpose    [1, 2, 3, 2] now: 2 heads, each holding 3 words of size 2
```

**Step 2:** each head runs the whole `softmax(Q K^T / sqrt(d_k)) V` recipe on its own. The score matrix `Q K^T` per head is `seq_len x seq_len`, so cost grows with the **square of the sequence length** (this is why long prompts hurt prefill, Part I chapter 2).

**Step 3:** glue the head outputs back into `d_model` numbers and multiply by one more matrix `W^O`, which mixes the heads' findings together.

#### 4.2.4 The Specific Parameters in the Original Paper

| Quantity | Paper | Kid meaning |
|---|---|---|
| `d_model` | 512 | Length of each word's vector |
| `h` | 8 | Number of meetings |
| `d_k = d_v` | 512 / 8 = 64 | Size of each meeting's vectors |
| `sqrt(d_k)` | 8 | The score-scaling divisor |

**Why the cost doesn't grow with more heads:** one big head does matrices of size `d x d`. Eight small heads each do `d x d/8`, eight times, which is the same total multiply-adds.

#### Looking ahead: why this section is the seed of the KV cache

During generation, when the newest word "sat" arrives, look at what its meeting needs:

- its **own Question** (new, must compute)
- every **Label and Gift of earlier words** ("the", "cat"): *these never change*, because earlier words cannot see later ones (causal mask)

```text
   step 1: process "the"            compute q0,k0,v0           save k0,v0
   step 2: process "cat"            compute q1,k1,v1           save k1,v1   (reuse k0,v0)
   step 3: process "sat"            compute q2,k2,v2           save k2,v2   (reuse k0,k1,v0,v1)
                                                   └── these saved Labels and Gifts are the KV cache
```

So only Questions are used once and thrown away. Keys and Values are saved. That saved pile is the **KV cache** (Part I chapter 4).

**GQA preview.** Many real models give the Questions more heads than the Keys/Values (Qwen3-0.6B: 16 Question heads, 8 Key/Value heads, so two Question heads share each Label/Gift set). That halves the size of the pile you must keep.

**Pause and predict.** In the causal-masked table, why is the first row exactly `[100%, 0%, 0%]`?

<details>
<summary>Answer (try first)</summary>

The first word may only look at itself (nothing earlier exists), so the only unmasked score gets all the softmax weight.

</details>

### 4.3 Layer Normalization (LayerNorm) and Residual Connections

#### 4.3.1 What Is Normalization

**Kid version.** A thermostat. After each floor's work, a word's numbers can drift to huge or tiny values, which makes training unstable. Normalization recentres them: subtract the average, then divide by the spread, so every word's vector has average 0 and spread 1. Two small learnable knobs (`gamma` to stretch, `beta` to shift) let the model undo it if the plain version isn't ideal.

**Tiny numbers.**

| Input | mean | spread (std) | After `(v - mean) / std` |
|---|---|---|---|
| [3, 3, 7, 7] | 5 | 2.000 | [-1, -1, +1, +1] |
| [2, 4, 6, 8] | 5 | 2.236 | [-1.342, -0.447, +0.447, +1.342] |

**Real words.** `epsilon` (1e-6) in the formula is a tiny number added under the square root so you never divide by zero when all entries are equal.

**What real LLMs use: RMSNorm.** Simpler: skip the "subtract the mean" step and drop `beta`. Just divide by the root-mean-square, then scale by a learned `weight`:

```text
   RMSNorm(x) = x / sqrt( mean(x^2) + eps )  *  weight
```

On `[3, 3, 7, 7]`: mean of squares = (9+9+49+49)/4 = 29, root = 5.385, result = `[0.557, 0.557, 1.300, 1.300]`. Cheaper (one fewer pass over the numbers), and works as well in practice. It is in `layers/norm.py`: class `RMSNorm`.

**Qwen3's extra thermostats.** Qwen3 also normalises the Question and Label vectors of each head before attention ("QK-norm", `has_qk_norm=True` in `models/qwen3.py`). It keeps attention scores in a safe range.

#### 4.3.2 What Is a Residual

**Kid version.** When you edit a document, you don't rewrite it from scratch each time; you mark **changes** on top of the existing draft. A residual connection does exactly that: `output = input + change`. If a floor has nothing useful to add, it can learn "change = 0" and the input passes straight through, so adding floors can never make things much worse. It also gives learning signals a straight road back down through all the floors.

```text
        input ───────────────────────┐   the hallway shortcut
          │                          │
          ▼                          ▼
      [ floor work ] ──▶ change ──▶ ( + ) ──▶ output
```

#### 4.3.3 Layer Normalization and Residual Connections

**Kid version.** Two orders are possible for "thermostat" and "shortcut". The original paper used **Post-Norm**: do the work, add the shortcut, *then* cool the result. Modern models, including everything in this repo, use **Pre-Norm**: cool the input first, do the work, then add the shortcut.

```text
   Post-Norm (the book)               Pre-Norm (modern; this repo)
   x ──┬─▶ sublayer ─▶ (+) ─▶ norm    x ──┬─▶ norm ─▶ sublayer ─▶ (+)
       └───────────────▶┘                 └──────────────────────▶┘
   out = Norm( x + F(x) )              out = x + F( Norm(x) )
```

Why Pre-Norm won: the shortcut road stays *completely clean* (nothing is applied to it), which keeps very deep stacks trainable.

**How this repo wires it (a neat trick).** Open `models/qwen3.py`. A floor does:

```python
x, residual = self.input_layernorm.forward(x, residual)
x = self.self_attn.forward(x)
x, residual = self.post_attention_layernorm.forward(x, residual)
x = self.mlp.forward(x)
return x, residual
```

*Line by line:* the floor carries **two** things: `x` (the worker's current paper) and `residual` (the shortcut road). The norm class `RMSNormFused` does *two jobs in one trip*: first adds the previous floor's work to the road (`residual += x`), then cools that road and hands a cooled copy to the next sublayer. The "add" is delayed and fused into the next thermostat so the numbers are read from memory only once.

```text
   embedding output  x0
   floor 0, norm 1 :  residual = x0            x = norm(x0)          ──▶ attention ──▶ a0
   floor 0, norm 2 :  residual = a0 + x0       x = norm(residual)    ──▶ MLP       ──▶ m0
   floor 1, norm 1 :  residual = m0 + a0 + x0  x = norm(residual)    ──▶ attention ...
```

The very first call has no road yet (`residual is None`), so it starts the road with the raw input. The last thing after the final floor, `self.norm.forward(x, residual)`, adds the last bit and does the final cool-down. So the road always equals "the embedding plus every sublayer's output so far", which is exactly Pre-Norm's `x + F(Norm(x))` chain.

**Pause and predict.** Why fuse "add" and "norm" instead of two separate steps?

<details>
<summary>Answer (try first)</summary>

On a GPU the slow part is often fetching numbers from memory, not the arithmetic. Two separate steps read and write the whole tensor twice; the fused kernel reads once and writes once. (Memory-bound work is a recurring theme from Part I chapter 2.)

</details>

### 4.4 Feed-Forward Network (Feed Forward) and Activation Functions

#### 4.4.1 Where ReLU Is Applied

**Kid version.** The private office. After the meeting, each word goes into its own office *alone* and thinks: first **spread out** (expand into a longer vector so ideas have room), then **filter** (an activation function throws away or squashes unhelpful values), then **squeeze back** to the original size.

```text
    word vector                  expanded                 filtered              back to size
   ┌─────────┐   x W1  (512→2048)  ┌──────────────┐  ReLU  ┌────────────┐  x W2  ┌─────────┐
   │  512    │ ───────────────────▶│     2048     │ ──────▶│ negatives=0 │ ──────▶│   512   │
   └─────────┘                      └──────────────┘        └────────────┘        └─────────┘
```

`FFN(x) = max(0, x W1 + b1) W2 + b2`. "Position-wise" means each word is processed on its own, with no looking at neighbours (that was the meeting room's job).

#### 4.4.2 Specific Parameter Configuration

| Quantity | Paper | Kid meaning |
|---|---|---|
| Input / output size | 512 | Size of each word's vector |
| Middle size | 2048 (4x) | The "wide office" |
| Activation | ReLU | The filter |

**Tiny numbers: where the weights live.** The two matrices hold 512 x 2048 x 2 = 2,097,152 numbers, about 2.1 million per layer. Attention's four matrices (`W^Q, W^K, W^V, W^O`) hold 4 x 512 x 512 = 1,048,576. So the "private office" is about **2/3 of a floor's weights**, attention about 1/3. In the modern gated version below the split is 8d² vs 4d², and the same 2:1 ratio holds. Keep this in mind in Part I chapter 2, where the FLOP formulas have `16 N d²` for the FFN and `8 N d²` for the projections.

#### 4.4.3 Why Use ReLU?

Cheap: the filter is `max(0, x)`, and its slope is simply 0 or 1.

#### 4.4.4 The Basic Qualities a Good Activation Function Needs

| Quality | Kid reason | Real reason |
|---|---|---|
| **Nonlinear** | If every step is just "stretch", ten steps equal one stretch | Two linear layers in a row collapse into one: `3 * (2x) = 6x` |
| **Differentiable (almost everywhere)** | We must be able to feel which way is downhill | Gradient descent needs slopes. ReLU's kink at 0 is handled with a subgradient |
| **Cheap** | It runs billions of times | Cost is paid per number, per layer, per token |

**What real LLMs use: SwiGLU (a gated filter).** Instead of one expand-filter-squeeze path, use **two** expansions: a *gate* and an *up* path. The gate decides, number by number, how much of the up path gets through:

```text
   x ──▶ gate = x Wg ──▶ silu(gate) ┐
   x ──▶ up   = x Wu ───────────────┴─▶ multiply ─▶ y ──▶ out = y Wd
```

`silu(x) = x * sigmoid(x)`: a smooth ReLU that lets small negative values leak through.

| x | -4 | -2 | -1 | 0 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|
| ReLU | 0 | 0 | 0 | 0 | 1 | 2 | 4 |
| SiLU | -0.072 | -0.238 | -0.269 | 0 | 0.731 | 1.762 | 3.928 |

In the code: `GatedMLP` in `models/utils.py` has `gate_up_proj` (gate and up merged into **one** matrix multiply, a speed trick), then `silu_and_mul` (apply silu to the first half, multiply by the second half), then `down_proj`. Three matrices of width ~`8/3 d` each give about `8 d²` weights, the same budget as the original 4x design.

**Pause and predict.** Why is it smart to merge the gate and up projections into one big matrix?

<details>
<summary>Answer (try first)</summary>

One large matrix multiply uses the GPU better than two small ones (less launch overhead, more parallel work), and the input `x` is read from memory once instead of twice.

</details>

### 4.5 Extended Reading: From This Course's Basics to the State-of-the-Art (SOTA)

**Kid version.** The book teaches the 2017 classic so you understand the ideas. Real engines run the improved versions. This table adds one column: **what the mini-sglang code actually uses** (so you know which version to expect when you read it).

| Component | Book's basic version | Modern version | In this repo (`mini-sglang`) |
|---|---|---|---|
| Overall architecture | Encoder-decoder | Decoder-only, sometimes MoE (mixture of experts) | Decoder-only; `models/qwen3_moe.py` and `layers/moe.py` add MoE |
| Positional encoding | Sinusoidal, added to inputs | RoPE (rotate Q and K) | RoPE in `layers/rotary.py`, with `llama3` and `yarn` length-extension options |
| Attention | MHA (every head has its own K,V) | MQA / GQA / MLA (share or compress K,V) | GQA: `num_qo_heads` vs `num_kv_heads` in `layers/attention.py` |
| Normalization | LayerNorm | RMSNorm (+ QK-norm) | RMSNorm, fused with the residual add; QK-norm for Qwen3 |
| Norm position | Post-Norm | Pre-Norm | Pre-Norm (the `x, residual` pattern) |
| Activation / FFN | ReLU, 4x expansion | Gated (SwiGLU / GeGLU) | Gated: `silu_and_mul` or `gelu_and_mul` in `layers/activation.py` |
| KV handling | None (recompute everything) | KV cache, paging, quantization | KV cache pool, paging and a radix cache (`kvcache/`) |
| Precision | FP32 | BF16 / FP16 / FP8 / INT4 | Set by the engine config (`dtype`) |

Terms in the book's "evolution" column, in one line each: **GQA** = several Question heads share one Key/Value head. **MLA** = compress K/V into a smaller hidden form. **NoPE** = no explicit position signal. **YaRN / NTK** = ways to stretch RoPE to longer contexts than trained. **QK-Norm** = normalise Q and K before comparing them. **MoE** = many small "private offices" ("experts"); each word visits only a few.

## 5 Key Foundational Concepts

### 5.1 Prompt Engineering

**Kid version.** Writing a good *order slip*. Say clearly what you want, with background, and the chef (model) delivers more reliably. When the slip becomes a novel stuffed with dynamic extras, quality drops.

### 5.2 Context Engineering

**Kid version.** Deciding **what goes on the chef's counter** right now. The counter (context window) is small. Putting everything on it crowds out the useful stuff and distracts the chef; and every item on the counter costs money (tokens). Prompt engineering is writing one good slip; context engineering is arranging the whole counter.

```text
   COUNTER (context window, limited)
   ┌──────────────────────────────────────────────────────────────┐
   │ rules │ past chat │ retrieved facts │ tool results │ the ask │
   └──────────────────────────────────────────────────────────────┘
   everything here is also paid for in tokens AND stored as notes (KV cache)
```

### 5.3 Skill

**Kid version.** A laminated recipe card for a specific job: how to do it and which tools to use. The chef knows *what* to make; the card says *exactly how* in your restaurant.

### 5.4 Retrieval-Augmented Generation (RAG)

**Kid version.** Before cooking, the chef looks the fact up in the **cookbook shelf** (an outside knowledge base), puts the right page on the counter, then answers. That fixes stale knowledge and reduces guessing. It's a way of doing context engineering: bring long-term memory into the short-term counter.

### 5.5 Agent

**Kid version.** A chef who doesn't just answer but **acts**: breaks the job into steps, uses tools (run code, search, write files), checks the result, and tries again. *Goal decomposition + skill use + checking.*

**Why inference people care:** an agent loops. Each loop is another request whose prompt is *the whole story so far plus the tool result*. Most of that prompt was already seen by the model one step earlier. If the engine can reuse the notes (prefix caching), the repeat costs almost nothing; if it can't, it pays again every loop. This is the motivation for RadixAttention in Part II chapter 8.

### 5.6 Harness

**Kid version.** The **restaurant's operating system around the chef**: how orders come in, which tools exist, how steps are scheduled, how memory is kept. `Model + Harness = Agent`: the chef alone can't run a restaurant.

### 5.7 Other Foundational Concepts

| Term | Kid version | Real meaning, with a number |
|---|---|---|
| **Token** | A numbered card: the smallest piece the model reads or writes | Chunks of text from the tokenizer. Roughly 0.75 English words per token. The model sees integers, not letters |
| **Embedding** | Each card gets a long list of numbers describing it | A learned vector per token ID. Qwen3-0.6B: 151,936 tokens x 1,024 numbers |
| **Parameter** | One knob/number in the recipe book | One learned floating-point number. "7B" = 7 billion of them |
| **Autoregressive** | Write one word, feed it back in, write the next | Each new token depends on all earlier ones |
| **Context window** | How many cards fit on the counter | Max tokens the model processes in one go |

**Tiny numbers: parameters become gigabytes.** Bytes = parameters x bytes per number:

| Model | bf16/fp16 (2 B) | fp8 (1 B) | int4 (0.5 B) |
|---|---|---|---|
| 7 B | 14 GB | 7 GB | 3.5 GB |
| 70 B | 140 GB | 70 GB | 35 GB |

**Tiny numbers: why a million-token window is hard.** For a 7B model with the classic multi-head layout (0.5 MiB of notes per token), **one** conversation of 1,000,000 tokens needs about 0.5 TB of notes, more than six 80 GB GPUs. Sharing K/V across heads (GQA with 8 K/V heads instead of 32) cuts it to about 0.13 TB. That is the point of every cache-saving trick in this course.

## 6 Summary and Exercises

### 6.1 Summary

**Kid version.** An LLM guesses the next token, over and over. It is built from a stack of identical floors. On each floor, words first meet (attention: Question/Label/Gift, several meetings at once, no peeking at the future), then think alone (feed-forward), with a shortcut road (residual) and a thermostat (norm). Position information comes from clock-hand fingerprints (sinusoidal) or from rotating Questions and Labels (RoPE). The things we keep from earlier words, the Labels and Gifts, become the **KV cache**, the main character of the rest of the course.

```text
   tokens ─▶ embed ─▶ [ norm ▶ attention(+road) ▶ norm ▶ MLP(+road) ] x L ─▶ norm ─▶ logits ─▶ pick
```

### 6.2 Exercises

Try each one on paper first; the hidden hint is for checking.

**1. One sentence: what is an LLM's core capability, and why does next-token prediction give rich abilities?**

<details>
<summary>Check yourself</summary>

It predicts the most likely next token given the text so far. To predict well across all kinds of text, a model with enough capacity must absorb grammar, facts, reasoning patterns, and code structure, so those skills emerge as side effects of one training objective.

</details>

**2. Why does self-attention need positional encoding? How does sinusoidal encoding provide position?**

<details>
<summary>Check yourself</summary>

Attention treats its inputs as an unordered set, so word order is invisible unless we add it. Sinusoidal encoding gives each position a fingerprint from sine/cosine pairs at many frequencies; adding it to the embedding injects "where am I". It has no learned parameters, and shifting by `k` positions is the same fixed rotation anywhere, which helps the model learn relative positions.

</details>

**3. Roles of residual connections and layer normalization.**

<details>
<summary>Check yourself</summary>

Residual: `output = input + change`, so each layer adds a correction instead of replacing the signal; this keeps information and gradients flowing through deep stacks. Normalization: recentres and rescales each word's numbers so values don't explode or vanish. Together they make deep stacks trainable.

</details>

**Extra drills (not in the book).**

4. Redo the three-word attention example by hand with the *last* word "sat" removed. Which rows change? (None of the first two, because the causal mask meant "the" and "cat" never looked at "sat". That is exactly why K/V of earlier tokens can be cached.)
5. For Qwen3-0.6B, how many bytes of notes does one token need? (`2 x 28 layers x 8 KV heads x 128 x 2 bytes = 114,688`, i.e. 112 KiB.)
6. Why does RoPE caching imply a cached prefix is only valid at the same positions?
