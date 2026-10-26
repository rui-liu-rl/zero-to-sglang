# Part 0.2 Deploy Your First SGLang Server: ELI5 companion

Original: [Part0-Deploy-Your-First-SGLang-Server.md](../../course-material/eng/part0/Part0-Deploy-Your-First-SGLang-Server.md). Same headings, same order. Every command block in the original is explained here.

## The one-minute version

**Kid version.** You are opening a tiny restaurant with exactly one chef.

1. Buy a **stove**: an NVIDIA GPU.
2. Set up the **kitchen**: a Python environment with SGLang installed.
3. Hire the **chef**: download the model's weights (Qwen3-0.6B, the smallest in its family).
4. **Open the door**: launch the server.
5. Send the first **order**: a request with `curl` or Python.

```text
  your laptop / terminal 2                       the GPU machine, terminal 1
  ┌──────────────────────┐   HTTP order slip   ┌────────────────────────────────┐
  │ curl / python client │ ──────────────────▶ │ SGLang server on port 30000    │
  │                      │ ◀────────────────── │  front desk → card-maker →     │
  └──────────────────────┘   the finished dish │  head waiter → chef (on GPU)   │
                                               └────────────────────────────────┘
```

**Your situation.** Your machine is an Apple M1 Pro Mac. The lesson says macOS is not supported for the course experiments, and `mini-sglang` is Linux + CUDA only. So everything in this lesson runs on a *rented* NVIDIA machine, not your laptop. See the last section.

## Deploying SGLang on a local GPU

### Requirements

**Kid version.** A restaurant needs a stove that is big enough and plugged in properly.

| The lesson says | Kid words | Real meaning |
|---|---|---|
| NVIDIA GPU with at least 4 GB of VRAM | A stove with enough counter space | VRAM is the GPU's own memory. The weights and notepads must fit in it |
| RTX 30/40/50 work out of the box; RTX 20 / T4 need a flag | Newer stoves know the newer tricks | Older GPU "architectures" lack features the default attention code needs |
| Linux or WSL2, not native Windows or macOS | The kitchen only fits in certain buildings | The GPU code libraries are built for Linux |
| Driver installed | The gas line is connected | The NVIDIA driver is the software layer that lets programs talk to the GPU |

**Tiny numbers: why 4 GB is enough.** Using the model's real config (28 layers, 16 query heads, 8 KV heads, head size 128, 1024 hidden size, 151,936-word vocabulary):

| Piece | Count | Size in bf16 (2 bytes each) |
|---|---|---|
| Word-embedding table (shared with the output layer) | 151,936 x 1024 = 155.6 M | 0.31 GB |
| 28 layers x 15.73 M | 440.5 M | 0.88 GB |
| **Total** | **596 M (0.6 B)** | **about 1.19 GB** |

**Drift note.** The course says "about 1.5 GB of weights". Counting from the config gives about 1.2 GB. A shared embedding table that is stored twice in the downloaded file would give 0.75 B x 2 bytes = 1.5 GB, which would explain the gap, but this was not checked against the actual file.

The other things that share the 4 GB: the **KV cache pool** (the chef's notepads), temporary working memory, and the CUDA runtime itself. For this model one token's notes cost:

```text
  K and V  x  28 layers  x  8 KV heads  x  128 numbers  x  2 bytes
    2      x     28      x      8       x      128       x    2     =  114,688 bytes = 112 KiB per token
```

So a 512-token prompt plus a 128-token answer needs 640 x 112 KiB = about 70 MiB of notes. Small, which is why a 4 GB card can serve a few dozen such requests at once.

**Checking the driver.**

```bash
nvidia-smi
```

*Line by line:* `nvidia-smi` is a tiny program from the driver that prints a table: GPU name, memory used / total, and in the top-right corner `CUDA Version: 12.x` or `13.x`. That number is the **newest CUDA the driver can run**. It decides which install commands you use below. You do not need to install the CUDA Toolkit yourself: the Python packages bring their own CUDA runtime. On WSL2 the driver lives on the Windows side.

**Where the story lies.** A "stove" is a good picture for memory and speed, but a real GPU is thousands of tiny cooks, not one big burner. Chapter 3 draws that properly.

**Pause and predict.** `nvidia-smi` says `CUDA Version: 12.4`. Which install branch do you follow, and why?

<details>
<summary>Answer (try first)</summary>

The "CUDA Version 12.x" branch. The software is built for CUDA 13 by default, but a driver that only supports 12.x cannot run it, so you swap in builds made for CUDA 12 (or upgrade the driver to 580 or newer and use the 13 commands).

</details>

### Python environment

**Kid version.** Give the restaurant its own pantry so ingredients from other kitchens do not get mixed in.

```bash
conda create -n sglang python=3.12 -y
conda activate sglang
```

*Line by line:* the first line makes a fresh, empty environment named `sglang` with Python 3.12 (`-y` = "yes to all questions"). The second line steps inside it. After it, `python` and `pip` mean *this environment's* copies.

```bash
python3 -m venv ~/sglang-env
source ~/sglang-env/bin/activate
```

*Line by line:* the no-conda way. `venv` creates a folder `~/sglang-env` holding a private Python; `source .../activate` steps into it. Open a new terminal and you must run `activate` again, because the "step inside" only lasts for that terminal.

**Pause and predict.** You install SGLang, open a new terminal, and `import sglang` fails. Why?

<details>
<summary>Answer (try first)</summary>

The new terminal is not inside the environment. Activate it again (`conda activate sglang` or `source ~/sglang-env/bin/activate`).

</details>

### Installing SGLang

**Kid version.** Buy the kitchen equipment, in the version that fits your gas line.

```bash
pip install --upgrade pip
pip install uv
uv pip install --prerelease=allow sglang
```

*Line by line:* (1) update `pip`, the standard package installer. (2) install `uv`, a much faster installer written in Rust. (3) use `uv` to install `sglang`; `--prerelease=allow` lets it accept not-yet-final versions of dependencies, which the project currently needs.

```bash
uv pip install --force-reinstall torch==2.13.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129
uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/
uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps
```

*Line by line (CUDA 12 only):* `--force-reinstall` throws away what is installed and replaces it. `--index-url .../cu129` means "get the build made for CUDA 12.9". The three lines swap in the CUDA-12 builds of (1) PyTorch and friends, (2) SGLang's own compiled GPU kernels, (3) a fast matrix-multiply library, without dragging in other dependencies (`--no-deps`).

```text
  driver can run CUDA: 13.x                        driver can run CUDA: 12.x
  ┌─────────────────────────┐                      ┌────────────────────────────────┐
  │ default install works   │                      │ default install, then swap the │
  └─────────────────────────┘                      │ GPU pieces for "cu129" builds  │
                                                   └────────────────────────────────┘
  a build made for a NEWER CUDA than your driver  ==>  "driver version is insufficient"
```

```bash
python3 -c "import sglang; print(sglang.__version__)"
```

*Line by line:* run one line of Python: import the package and print its version. Printing a version number proves the package is installed and importable (it does not prove the GPU works; launching the server does).

**Where the story lies.** Exact version numbers go stale quickly. The lesson says the official docs are the source of truth; trust them over any pinned number written here or there.

### Faster model downloads (users in mainland China)

**Kid version.** The ingredient truck is slow on one road, so take another road.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

*Line by line:* `export` sets an **environment variable** for this terminal and everything it launches. `HF_ENDPOINT` tells the Hugging Face download library to use a mirror site instead of the main one.

```bash
uv pip install modelscope
export SGLANG_USE_MODELSCOPE=true
```

*Line by line:* the alternative: install the ModelScope client and tell SGLang to fetch weights from ModelScope.

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

*Line by line:* `echo ... >> ~/.bashrc` appends that line to the file your shell reads at every start, so the setting survives new terminals. (On zsh the file is `~/.zshrc`.)

**The classic mistake:** `export` only affects *that* terminal. If you set it in one terminal and launch the server in another, the server never sees it, and the download hangs on the slow road. This is the FAQ's "download hangs" item.

### Launching the server

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --host 0.0.0.0 --port 30000
```

*Line by line:*

| Piece | Meaning |
|---|---|
| `python3 -m sglang.launch_server` | Run SGLang's launcher module (the "open the restaurant" button) |
| `--model-path Qwen/Qwen3-0.6B` | Which chef to hire. A name like this means "download from the model hub on first run" |
| `--host 0.0.0.0` | Listen on *all* network doors. `127.0.0.1` would mean only this machine. `0.0.0.0` lets other machines reach it, so treat it as open unless your network is trusted |
| `--port 30000` | Which door number to listen on |

**What happens inside** (you will read this code in Part II chapter 2): the launcher starts several processes, then loads the weights onto the GPU, sets aside the KV cache pool, and prints "The server is fired up and ready to roll!".

```text
  one `launch_server` command
        │
        ├─▶ main process   : HTTP server + tokenizer manager   (front desk + card-maker)
        ├─▶ scheduler proc : head waiter + engine on the GPU    (decides + cooks)
        └─▶ detokenizer    : cards back into words              (reads dishes aloud)
```

Closing this terminal kills the main process, which is why the lesson says to keep it open and use a *second* terminal for the orders.

### Sending requests

**Is it alive?**

```bash
curl http://localhost:30000/health
```

*Line by line:* `curl` sends an HTTP request. `GET /health` is the "are you there?" door; a success status (200) means the server is up.

**The first order.**

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Introduce yourself in one sentence."}],
    "max_tokens": 512
  }'
```

| Piece | Kid words | Real meaning |
|---|---|---|
| `curl -s` | Send it, say nothing extra | `-s` = silent (no progress bar) |
| `.../v1/chat/completions` | The "chat" order window | The OpenAI-compatible endpoint; many tools already know how to call it |
| `-H "Content-Type: application/json"` | Label on the slip | Tells the server the body is JSON |
| `-d '{...}'` | The slip itself | The request body |
| `"model"` | Which chef | Name of the model to answer with |
| `"messages"` | The conversation so far | A list of `{role, content}`. Roles are `system`, `user`, `assistant` |
| `"max_tokens": 512` | Stop after this many cards | Upper limit on how many tokens the answer may contain |

A response has this *shape* (illustrative, not captured output):

```text
  { "choices": [ { "message": { "role": "assistant", "content": "I am ..." },
                   "finish_reason": "stop" } ],
    "usage":   { "prompt_tokens": 15, "completion_tokens": 40, "total_tokens": 55 } }
```

`finish_reason` is `stop` when the model said "I'm done" and `length` when it hit `max_tokens`. `usage` counts tokens, which is what you are billed and what the KV cache stores.

**The same order from Python.**

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[{"role": "user", "content": "List 3 countries and their capitals."}],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

*Line by line:* (1) load the OpenAI client library. (2) build a client that points at *your* server, not OpenAI's: `base_url` is the address; `api_key="None"` is a placeholder because the library insists on some string while your server does not check one. (3) `create(...)` sends the same JSON order as the curl. (4) the answer sits at `choices[0].message.content`.

**Why you see `<think>...</think>`.** Qwen3 is a "thinking" model: by default it first writes its reasoning between those tags, then the answer. It is not a bug. Those thinking words are real tokens: they cost time and KV cache like any others.

```python
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
```

*Line by line:* `extra_body` lets you add fields the client library does not know about. `chat_template_kwargs` are passed to the **chat template**: a small template file shipped with the model that turns your `messages` list into the exact special-token text the model was trained on (for example `<|im_start|>user ... <|im_end|>`). `enable_thinking: False` is a variable that template reads to skip the thinking block.

Or launch with `--reasoning-parser qwen3`: the server then *splits* the thinking text out of `content` into a separate `reasoning_content` field instead of removing it.

**Pause and predict.** After disabling thinking, will the answer use fewer KV-cache slots? Why?

<details>
<summary>Answer (try first)</summary>

Usually yes: fewer tokens are generated, and every generated token adds its 112 KiB of notes (for this model). Fewer tokens also means fewer decode steps, so a lower total latency.

</details>

### FAQ

| Problem | What is going on (kid version) | Real fix |
|---|---|---|
| **OOM** (out of memory) | The stove is too small, or something else (a desktop) is using it | Lower SGLang's share and shorten the longest allowed conversation (command below) |
| **RTX 20 / T4 fails to start** | The stove is too old for the default technique | Use the `triton` attention backend |
| **"driver version is insufficient" / "no kernel image is available"** | You bought equipment built for a newer gas line than yours | Install the CUDA 12 builds (the three `--force-reinstall` lines) |
| **Download hangs** | Truck took the slow road: the env var was not set where you launched | Set `HF_ENDPOINT` / `SGLANG_USE_MODELSCOPE` in the *launching* terminal |
| **address already in use** | Someone already sits at door 30000 | Use another `--port` and change your requests to match |
| **nvidia-smi not found in WSL2** | The gas line must be connected on the Windows side | Install the Windows driver, then `wsl --shutdown` |

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --mem-fraction-static 0.6 --context-length 8192 --port 30000
```

*Line by line, the two new flags:*

- `--mem-fraction-static 0.6`: "static" memory means the things that stay for the server's whole life: **the weights and the KV cache pool**. `0.6` says "use 60% of the GPU's memory for those". The rest is left for temporary working space (and a desktop, if any). Lower it if you run out of memory during a request.
- `--context-length 8192`: the longest conversation (prompt plus answer) the server will accept. Memory planning gets easier, and for this model the worst case shrinks from 40,960 tokens x 112 KiB = 4.4 GiB of notes for one maxed-out request down to 8,192 x 112 KiB = 0.875 GiB.

**Tiny numbers: what `--mem-fraction-static` buys on a 4 GB card** (rough; the real accounting has more terms):

| Setting | Budget for weights + KV pool | Minus 1.19 GB weights | KV pool | About how many tokens of notes |
|---|---|---|---|---|
| 0.9 | 3.6 GB | 2.41 GB | 2.41 GB | ~21,000 |
| 0.6 | 2.4 GB | 1.21 GB | 1.21 GB | ~10,500 |

```bash
python3 -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --attention-backend triton --port 30000
```

*Line by line:* `--attention-backend triton` picks the attention code written in the Triton language, which works on older GPUs. The default backends use newer GPU features.

**Anything else:** post the full command and error as *text*, with `nvidia-smi` output and the SGLang version. Text can be searched and copied; screenshots cannot.

### No NVIDIA GPU

**Kid version.** No stove at home? Rent a restaurant kitchen by the hour.

The lesson lists other hardware SGLang supports (AMD Instinct, Intel Xeon CPU, Apple Silicon marked experimental) but says later lessons are not guaranteed to work there. The practical answer for a Mac: rent the cheapest hourly NVIDIA GPU machine from any cloud GPU provider (the course names one used in China). It arrives as a ready-made Linux box, so you start from the "Python environment" section. A GPU that runs Qwen3-0.6B is enough; this model is tiny.

**Where the story lies.** A rented machine is billed by the hour while it exists, not while you are using it. Stop or delete it when you are done.

## Check yourself before moving on

- Can you say what `--host`, `--port`, `--model-path`, `--mem-fraction-static` and `--context-length` each control?
- Can you explain why the first run is slow (download) and why `<think>` tokens are not free?
- Can you draw the three processes `launch_server` starts, from memory?
