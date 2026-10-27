# ELI5 companion to zero-to-sglang

Personal study notes for the **English edition** of the course. Not part of the official course, so it lives outside `course-material/` and `community/` (the course CI scans those two folders).

**The deal:** every original chapter has a companion here with the **same section numbers and headings**. Read a section in the original, then the same section here, then open the matching code in `mini-sglang` (branch `eli5-comments`, where the code is commented in the same ELI5 style). Repetition between the three is intentional.

## How to read one section (the loop)

1. Read the section in the original chapter (`course-material/eng/...`).
2. Read the same section here. Each section has the same parts, in this order:

| Part | What it gives you |
|---|---|
| **Kid version** | The idea in plain words and a story, with no jargon. |
| **Picture** | An ASCII diagram. Read it slowly; the picture is the idea. |
| **Real words** | A table that maps every kid word to the real term. |
| **Tiny numbers** | The same idea with numbers small enough to check by hand. |
| **Where the story lies** | Where the analogy breaks, so you do not carry a wrong idea forward. |
| **Pause and predict** | Questions. Answer them before opening the hidden answer. |

3. Open the matching mini-sglang file and read the comments next to the real code.
4. Anything still fuzzy goes on your question list; bring it to the next session and we expand that spot.

## The Kitchen (the running story)

One story, used in every chapter, so the pieces stay connected.

| In the kitchen | In the real system |
|---|---|
| A customer with an order slip | A user sending a request (a prompt) |
| The front desk / host stand | API server (takes orders, hands back dishes) |
| The card-maker | Tokenizer: turns words into numbered cards (token IDs). The *detokenizer* turns cards back into words |
| The head waiter (expediter) | Scheduler: decides which orders get cooked next, in which batch, and whether the pantry has room |
| The chef | The model running on a GPU. The *engine* is the chef's hands |
| The giant recipe book | Model weights (billions of numbers). The chef must flip through *all of it* for every bite |
| The chef's notepad for each table | KV cache: notes about every word so far, so the chef never re-derives them |
| Pantry shelves with numbered slots | The KV cache pool, split into pages |
| A shared-beginnings tree of orders | Radix tree (prefix cache) |
| Several chefs each holding part of the recipe book | Tensor parallelism (several GPUs) |
| The kitchen itself: many cooks, small counters, a nearby pantry, a far warehouse | GPU: many SMs, registers / shared memory, L2, HBM |

Every chapter says **where the kitchen story lies**. A story that is never questioned eventually teaches a wrong idea.

## Picture legend

```text
┌──────┐   a thing: a process, a file, a block of memory
│ box  │
└──────┘
   ──▶     data or a message moving      ◀──▶   both directions
   ·····▶  a copy or a lookup (not a move)
 [ ][ ][ ] memory slots (empty)           [█][█][ ] slots (full, full, empty)
  ░▒▓█     shading: how full / how busy
  t=0 t=1  time moving left to right (or top to bottom, as labelled)
```

## Code-comment conventions (mini-sglang, branch `eli5-comments`)

Only comments are added. Behaviour never changes. Docstrings are not added (they would change the program's syntax tree).

| Tag | Meaning |
|---|---|
| `# FILE MAP` | Top of every file: what it is, who calls it, whom it calls, its kitchen role |
| `# ELI5:` | The kid-level story or analogy for this block |
| `# WHAT:` | Mechanically, in plain words, what this block does |
| `# WHY:` | Why it is done this way instead of the obvious way |
| `# SHAPE:` | Tensor shapes and dtypes at this point |
| `# GOTCHA:` | A subtle trap, an easy misreading, or a surprising detail |

Proof that nothing but comments changed (run from this repo's root):

```bash
python3 -I eli5/tools/check_comments_only.py --repo /Users/rui/code/mini-sglang
```

It compares each changed `.py` file's syntax tree against `main`, and compares CUDA/C++ files after stripping comments. Any behaviour change makes it fail.

## House rules for these notes

- English only. Same headings and numbering as the original chapter.
- Kid words first, then exact words. Never leave the exact words out.
- Other inference engines are not named (the official course forbids it; it is cheap to follow here too).
- Stub chapters (three-line "being written" pages) have nothing to expand and are skipped. Chapter 6's concepts are written from its code walkthrough and the source.
- If a book's code excerpt differs from the current source in `mini-sglang`, the notes say so ("drift") and follow the source.

## Progress ledger

Status: `todo` / `done`. Stubs are marked `stub`.

### The book

| Original (`course-material/eng/`) | Companion | Status |
|---|---|---|
| part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md | part0/Part0-Coding-Ethics-and-Open-Source-Spirit.md | done |
| part0/Part0-Deploy-Your-First-SGLang-Server.md | part0/Part0-Deploy-Your-First-SGLang-Server.md | done |
| part1/Chapter1_Introduction_to_LLM.md | part1/Chapter1_Introduction_to_LLM.md | done |
| part1/Chapter2_Introduction to Inference.md | part1/Chapter2_Introduction-to-Inference.md | done |
| part1/Chapter3_Introduction_to_GPU.md | part1/Chapter3_Introduction_to_GPU.md | done |
| part1/Chapter4_KV Cache The Core Data Structure of Inference.md | part1/Chapter4_KV-Cache-The-Core-Data-Structure-of-Inference.md | done |
| part1/Chapter5_Introduction_to_Benchmark.md | part1/Chapter5_Introduction_to_Benchmark.md | done |
| part2/Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like.md | part2/Chapter1_mini-sglang-What-an-Inference-Engine-Looks-Like.md | done |
| part2/Chapter2_Inside-SGLang.md | part2/Chapter2_Inside-SGLang.md | done |
| part2/Chapter2_Inside-SGLang_code.md | part2/Chapter2_Inside-SGLang_code.md | done |
| part2/Chapter3_Forward-Pass-and-Generation.md | part2/Chapter3_Forward-Pass-and-Generation.md | done |
| part2/Chapter3_Forward-Pass-and-Generation_code.md | part2/Chapter3_Forward-Pass-and-Generation_code.md | done |
| part2/Chapter4_KV-Cache-Optimization.md | part2/Chapter4_KV-Cache-Optimization.md | done |
| part2/Chapter4_KV-Cache-Optimization_code.md | part2/Chapter4_KV-Cache-Optimization_code.md | done |
| part2/Chapter5_HTTP-and-Concurrent-Requests.md | part2/Chapter5_HTTP-and-Concurrent-Requests.md | done |
| part2/Chapter5_HTTP-and-Concurrent-Requests_code.md | part2/Chapter5_HTTP-and-Concurrent-Requests_code.md | done (runnable toy: `tools/ch5_frontend_toy.py`) |
| part2/Chapter6_Continuous-Batching-and-the-Scheduler.md (prose is a stub) | part2/Chapter6_Continuous-Batching-and-the-Scheduler.md (written from the code chapter + source) | done (runnable toy: `tools/ch6_scheduler_toy.py`) |
| part2/Chapter6_Continuous-Batching-and-the-Scheduler_code.md | part2/Chapter6_Continuous-Batching-and-the-Scheduler_code.md | done |
| part2/Chapter7_Paged-KV-Cache-and-Memory-Management.md | part2/Chapter7_Paged-KV-Cache-and-Memory-Management.md | done (runnable toy: `tools/ch7_paging_toy.py`) |
| part2/Chapter8_RadixAttention-and-Prefix-Caching.md | part2/Chapter8_RadixAttention-and-Prefix-Caching.md | done (runnable toy: `tools/ch8_radix_toy.py`) |
| part2/Chapter8_RadixAttention-and-Prefix-Caching_code.md | part2/Chapter8_RadixAttention-and-Prefix-Caching_code.md | done |
| part2/Chapter9_Multi-process-and-Tensor-Parallelism.md | part2/Chapter9_Multi-process-and-Tensor-Parallelism.md | done (runnable toy: `tools/ch9_tp_toy.py`) |
| part2/Chapter9_Multi-process-and-Tensor-Parallelism_code.md | part2/Chapter9_Multi-process-and-Tensor-Parallelism_code.md | done |
| part2/Chapter10_Speculative-Decoding.md | n/a | stub |
| part3/Chapter1_Attention-Backends.md | part3/Chapter1_Attention-Backends.md | done (runnable toy: `tools/p3ch1_attention_toy.py`) |
| part3/Chapter2_Quantization-and-Low-Precision-Inference.md | n/a | stub |
| part3/Chapter3_Hierarchical-Caching.md | part3/Chapter3_Hierarchical-Caching.md | done (runnable toy: `tools/p3ch3_hicache_toy.py`) |
| part3/Chapter4_Scaling-Out.md | n/a | stub |
| part3/Chapter5_Prefill-Decode-Disaggregation.md | n/a | stub |
| part4/Chapter1 / Chapter2 / Chapter3 | n/a | stub |

### The code (`/Users/rui/code/mini-sglang`, branch `eli5-comments`)

Work order follows the book. `todo` until the file is commented *and* passes the checker.

| Package | Files | Status |
|---|---|---|
| `minisgl/core.py`, `env.py`, `__main__.py`, `shell.py` | 4 | done |
| `minisgl/server/` | `launch.py`, `args.py`, `api_server.py`, `__init__.py` | done |
| `minisgl/message/` | `frontend.py`, `backend.py`, `tokenizer.py`, `utils.py`, `__init__.py` | done |
| `minisgl/tokenizer/` | `server.py`, `tokenize.py`, `detokenize.py`, `__init__.py` | done |
| `minisgl/scheduler/` | `scheduler.py`, `prefill.py`, `decode.py`, `cache.py`, `table.py`, `io.py`, `config.py`, `utils.py`, `__init__.py` | done |
| `minisgl/engine/` | `engine.py`, `graph.py`, `sample.py`, `config.py`, `__init__.py` | done |
| `minisgl/kvcache/` | `base.py`, `mha_pool.py`, `naive_cache.py`, `radix_cache.py`, `__init__.py` | done |
| `minisgl/layers/` | `attention.py`, `base.py`, `linear.py`, `embedding.py`, `norm.py`, `rotary.py`, `activation.py`, `moe.py`, `__init__.py` | done |
| `minisgl/models/` | `llama.py`, `qwen2.py`, `qwen3.py`, `qwen3_moe.py`, `mistral.py`, `base.py`, `config.py`, `weight.py`, `utils.py`, `register.py`, `__init__.py` | done |
| `minisgl/attention/` | `base.py`, `fa.py`, `fi.py`, `trtllm.py`, `utils.py`, `__init__.py` | done |
| `minisgl/distributed/` | `impl.py`, `info.py`, `__init__.py` | done |
| `minisgl/moe/` | `base.py`, `__init__.py`, `fused.py` | done |
| `minisgl/llm/` | `llm.py`, `__init__.py` | done |
| `minisgl/utils/` | `mp.py`, `hf.py`, `logger.py`, `misc.py`, `arch.py`, `registry.py`, `torch_utils.py`, `__init__.py` | done |
| `minisgl/kernel/` (Python) | `index.py`, `store.py`, `radix.py`, `tensor.py`, `pynccl.py`, `moe_impl.py`, `utils.py`, `triton/fused_moe.py`, `__init__.py`, `__main__.py` | done |
| `minisgl/kernel/csrc/` (CUDA / C++) | `jit/index.cu`, `jit/store.cu`, `src/pynccl.cu`, `src/radix.cpp`, `src/tensor.cpp`, `include/minisgl/{tensor.h,utils.h,utils.cuh,warp.cuh}` (vendored `nccl227.h` skipped) | done |
| `minisgl/benchmark/`, `benchmark/` | `client.py`, `perf.py`, `offline/*`, `online/*` | done |
| `tests/` | `core/test_cache_allocate.py`, `core/test_scheduler.py`, `kernel/test_comm.py`, `kernel/test_index.py`, `kernel/test_store.py`, `kernel/test_tensor.py`, `misc/test_serialize.py` | done |

Everything above is `done` except the seven stub chapters (three-line "being written" pages in the original: Part II ch.10, Part III ch.2, 4, 5, Part IV ch.1 to 3), which have nothing to expand yet. Vendored `nccl227.h` (NVIDIA's header) is deliberately not annotated.

## Runnable toys (`tools/`)

Standard library only, so they run on this Mac. Run from the repo root with `python3 -I`.

| Script | What it lets you check by running |
|---|---|
| `build_pdf.py` (with `book.css`, `../Makefile`) | Builds the print-ready PDF of this whole book: `make pdf` |
| `check_comments_only.py` | That the commented mini-sglang code differs from `main` in comments only (AST for Python, comment-stripped text for C/C++/CUDA) |
| `ch5_frontend_toy.py` | The front desk's `ack_map` / `event_map` logic: interleaved replies, one wake-up for many replies, the cleanup gap and its `finally` fix, late replies, blocking vs awaiting, 1 KB per leaked ticket |
| `ch6_scheduler_toy.py` | The book's A/B/C scheduling trace, static vs continuous batching, the worst-case KV admission arithmetic, chunked prefill |
| `ch7_paging_toy.py` | Fragmentation arithmetic, the book's holes example, address translation, online softmax step by step (and the wrong merge), a random allocate/free simulation |
| `ch8_radix_toy.py` | Why only prefixes can be reused (a 2-layer attention toy), the A/B/C radix tree, the R1/R2 lifecycle with balanced books, why the last prompt token is excluded, page sizes, LRU eviction |
| `ch9_tp_toy.py` | Column- and row-parallel maths, why column-then-row, the bias trap, vocab-parallel embedding and LM head, KV head replication, traffic and KV capacity vs TP, shard-then-merge for fused qkv |
| `p3ch1_attention_toy.py` | The `n x n` score table at 32k, the book's 4x4 tiled example, split-KV merging, uneven decode work |
| `p3ch3_hicache_toy.py` | Restore vs recompute (assumed bandwidths), continuous reuse across tiers, layerwise overlap, prefetch and wait policies, prefix fingerprints |

## Building the PDF

```bash
make pdf              # US Letter -> eli5/build/zero-to-sglang-eli5-letter.pdf   (from the repo root or from eli5/)
make pdf PAPER=a4     # A4        -> eli5/build/zero-to-sglang-eli5-a4.pdf
make html             # only write eli5/build/book.html, to debug the layout in a browser
```

Needs Python 3.9+ and Chrome or Chromium (`CHROME=/path/to/chrome` if it is not found). The first run creates `eli5/.venv` and installs two packages (`requirements-pdf.txt`). It takes about 20 seconds.

What you get: cover, contents with page numbers, 25 chapters grouped by part, two appendices, PDF bookmarks, mirrored binding margins for double-sided printing, a running header per chapter, `(p. N)` after every cross-reference, code blocks shrunk to fit the page, and every "Answer" box printed (dashed) after its question.

On GitHub, `.github/workflows/eli5-pdf.yml` runs the same command whenever `eli5/` changes on the `eli5-notes` branch, uploads the PDFs as an artifact, and refreshes a rolling release called `eli5-pdf-latest`. (On a fork, enable Actions once in the Actions tab.)
