# Chapter 3 Introduction to GPU: ELI5 companion

Original: [Chapter3_Introduction_to_GPU.md](../../course-material/eng/part1/Chapter3_Introduction_to_GPU.md). Same headings, same numbers. Cast and picture legend: [../README.md](../README.md).

## The one-minute version

**Kid version.** A CPU is **one brilliant master chef**: can cook anything, makes smart decisions, finishes one dish very fast. A GPU is **a stadium full of simple cooks** who all do the *same* step at the same moment on different ingredients. For a job like "multiply a billion pairs of numbers", the stadium wins by a mile. Language models are made of exactly that kind of job.

Everything else in this chapter is the *floor plan of the stadium* and *the rules for how the cooks work together*:

```text
   THE GPU (stadium)                                  HOW A JOB RUNS
   ┌──────────────────────────────────────────┐       ┌───────────────────────────────────────────┐
   │  108 COOKING STATIONS  (SMs)              │       │ you launch a KERNEL (one recipe step)      │
   │   each: 64 simple cooks  (CUDA cores)     │       │  ▼                                         │
   │         4 matrix-specialists (Tensor C.)  │       │ it is cut into BLOCKS (teams)              │
   │         a tiny private counter (shared)   │       │  ▼  each team is given to ONE station      │
   │         each cook's hands (registers)     │       │ a team is cut into WARPS (32 cooks in step)│
   │  a shared pantry shelf (L2 cache, 40 MB)  │       │  ▼                                         │
   │  a far warehouse      (HBM, 80 GB)        │       │ every cook (THREAD) handles a bit of data  │
   └──────────────────────────────────────────┘       └───────────────────────────────────────────┘
```

**The two facts you must carry out of this chapter:**

1. **Cooking is cheap, fetching is expensive.** Doing a multiply is easy; getting the numbers from the far warehouse is slow. Good GPU code is mostly about *not walking to the warehouse*.
2. **The GPU hides waiting by switching to other cooks.** It does not make the walk shorter; it just makes sure someone else is cooking while you walk.

**Gaps in the original that this companion fills** (they are only name-dropped there): how *tiling* really saves trips to the warehouse (with numbers), what *coalesced* access means, what *occupancy* is, why switching warps is free, how big the *kernel-launch tax* is, and the formula `tokens/s ≈ memory bandwidth / bytes read per token`.

## 1 Learning Objectives

| Original goal | Kid version |
|---|---|
| 1. GPU architecture: CPU vs GPU, GPC → TPC → SM → CUDA/Tensor Core | The stadium's floor plan, from whole building down to one cook |
| 2. Execution model: block / warp / thread and the memory ladder | How teams are formed and where each cook keeps things |
| 3. LLM inference on the GPU: preprocessing, compute-heavy prefill, memory-heavy decode | Why this stadium is perfect for language models, and where it still waits |

## 2 GPU Architecture Fundamentals

### 2.1 The Origin of the GPU: The Graphics Processor

**Kid version.** A 3D character in a game is built from thousands of tiny triangles. To draw one triangle you only store its 3 corner points; the pixels along its edges are *computed* each frame, using many simple multiplies and adds. That is a perfect "do the same simple thing to a million items" job. A master chef doing one pixel at a time is far too slow, so people built a chip with a huge crew of simple cooks instead.

#### 2.1.1 The Sprouting of Graphics Display (Before the 1980s)

| Year | Event | Kid gloss |
|---|---|---|
| 1981 | IBM PC's CGA card: 16 colors | A digital picture frame; the CPU did all the drawing |
| 1987 | VGA: 256 colors | Prettier frame, still no thinking inside the card |
| 1985 / 1992 | ATi founded / Mach32 adds graphics *acceleration* | First time the card did some of the work itself |

#### 2.1.2 The Melee of 3D Accelerator Cards (1990s)

| Year | Event |
|---|---|
| 1994 | 3DLabs Glint300SX: the first 3D accelerator chip for PCs |
| 1996 | 3dfx Voodoo: ordinary PCs can play 3D games |
| 1997 | First 3D geometry processor for PCs (Fujitsu); a chip with Transform and Lighting (T&L) (Mitsubishi) |

Standards were chaotic and each chip did one narrow thing, so they were called 3D accelerators, not GPUs.

#### 2.1.3 The Formal Birth of the GPU: NVIDIA (1999-2006)

In 1999 the **GeForce 256** introduced the word "GPU": it moved the 3D geometry and lighting math (T&L) off the CPU, put triangle setup, clipping, texturing and rendering on one chip, and cut the CPU's 3D load by more than 80%. By 2000 only NVIDIA and ATI (bought by AMD in 2006) were left standing.

#### 2.1.4 The Programmable Era (2001-2012)

| Phase | What changed | Kid gloss |
|---|---|---|
| 2001-2006: DirectX 8 shaders | Vertex shader and pixel shader: tiny programs | The assembly line gains the ability to follow simple instructions |
| 2006-2012: unified shaders (GeForce 8800 GTX, G80) | One kind of general worker instead of two specialised kinds; utilisation up from ~50% to over 90%; **CUDA** lets you program it in C | Workers are interchangeable, so none sit idle while others are swamped |

| Architecture | Year | Key idea |
|---|---|---|
| Tesla | 2006 | CUDA, general-purpose GPU computing |
| Fermi | 2010 | Double precision and ECC error correction |
| Kepler | 2012 | Dynamic parallelism, power efficiency |

#### 2.1.5 The General-Purpose Computing Era (2012-2018)

In 2012 researchers trained a deep network (AlexNet) on GPUs and won image recognition by a wide margin. From then on, GPUs were AI engines. NVIDIA's **CUDA ecosystem** (libraries, compilers, tools) made them easy to program, which is a big part of why it stuck.

#### 2.1.6 The Difference Between the CPU and the GPU

**Picture: what the chip area is spent on.**

```text
   CPU  (few, mighty cores)                       GPU  (very many, simple cores)
   ┌───────────────────────────────┐              ┌───────────────────────────────────┐
   │ CONTROL (branch prediction,   │              │ ctl │ ALU ALU ALU ALU ALU ALU ALU   │
   │  out-of-order, speculation)   │              │ ctl │ ALU ALU ALU ALU ALU ALU ALU   │
   │ ███████████████████           │              │ ctl │ ALU ALU ALU ALU ALU ALU ALU   │
   │ CACHE  ████████████████████   │              │ ctl │ ALU ALU ALU ALU ALU ALU ALU   │
   │ ALU ALU ALU ALU               │              │ (thousands of ALUs; little control)│
   └───────────────────────────────┘              └───────────────────────────────────┘
   Goal: finish ONE task ASAP (latency)           Goal: finish ALL tasks ASAP (throughput)
```

| | CPU | GPU |
|---|---|---|
| Core count | about 4-64 | thousands (A100: 6,912 FP32 CUDA cores) |
| Where the transistors go | Control logic and cache | Arithmetic units (ALUs) |
| Good at | Branchy, complex logic | Same simple math on huge piles of data |
| Optimises | **Latency** of one task | **Throughput** of all tasks |
| Trick for waiting | Predict, reorder, cache | Switch to other ready threads |

**Pizza example.** One oven can bake one pizza in 10 minutes (low latency for one pizza). A conveyor of 1,000 slow ovens bakes each pizza in 30 minutes but delivers 2,000 pizzas an hour. The GPU is the conveyor. For one lonely task it is slower than the CPU; for a million identical tasks it is far faster.

**Where the story lies.** The GPU does not run by itself: the CPU decides what to run and launches it. "A good GPU needs a good CPU" is literally true for inference engines (the scheduler is CPU code).

### 2.2 The Composition of the A100 GPU Core

**Kid version.** Open the box: a big circuit board, a central chip (the stadium) with memory stacks (the warehouse) sitting right next to it inside the same package, plus a power system. This section zooms from the box down to one cook.

#### 2.2.1 Product Form Factor

Per the book: A100 PCIe 80GB is a dual-slot full-height card, 267 mm long, 300 W, passively cooled (the server's airflow does the cooling), PCIe 4.0 x16 plus NVLink bridge connectors, about 1.4 kg. It has no video output; it is a pure compute card.

**Why 108 SMs and not 128?** The full chip design (GA100) has 128 SMs, but a chip with 54.2 billion transistors almost always has a few defects. NVIDIA switches off the defective parts and sells the chip as a slightly smaller product (**binning**): like a cookie sheet with a few burnt cookies; you box the good ones. The A100 PCIe 80GB ends up with 108 working SMs.

#### 2.2.2 GPU and HBM Package

The GA100 die: 54.2 billion transistors, 7 nm process, 826 mm². The memory stacks (**HBM2e**) are stacks of DRAM chips sitting beside the die and joined to it by a silicon base plate (an *interposer*). Short, very wide wires are the reason HBM bandwidth is so high.

#### 2.2.3 GA100 GPU Core Architecture

**Picture: Russian dolls.**

```text
   GPU
   └── GPC  (Graphics Processing Cluster)       full chip: 8   ; the card enables 7
        └── TPC (Texture Processing Cluster)    7-8 per GPC, 54 TPCs enabled in total
             └── SM  (Streaming Multiprocessor) 2 per TPC → 108 SMs
                  ├── 64 FP32 CUDA cores
                  ├── 64 INT32 cores
                  ├── 4 Tensor Cores
                  ├── 4 warp schedulers
                  ├── 256 KB register file
                  └── 192 KB L1 / shared memory
```

Arithmetic you can check: 54 TPCs x 2 = **108 SMs**; 108 x 64 = **6,912 FP32 cores**; 108 x 4 = **432 Tensor Cores**.

#### 2.2.4 The Internal Structure of the SM

**Kid version.** An SM is **one cooking station**: a foreman (the schedulers) who gives instructions, a crew of simple cooks, a few matrix specialists, a private counter, and everyone's hands. A *team of work* (a block) is handed to a station and stays there until done.

What the foreman does: take the thread groups it has been given, fetch instructions, and send them to the right workers: ordinary maths to CUDA cores, matrix maths to Tensor Cores.

The A100 SM is split into **4 quarters**, each with its own warp scheduler (so 4 instructions can be started per clock from 4 different warps).

#### 2.2.5 Tensor Core

**Kid version.** A Tensor Core is a **matrix specialist**: instead of one multiply at a time, it does a whole small matrix multiply-and-add (`D = A x B + C`) in a single step. That is why the numbers in the table below are so high.

**Tiny numbers: why specialists matter.**

| Unit | A100 peak | Where it comes from |
|---|---|---|
| FP32 CUDA cores | 19.5 TFLOPS | 6,912 cores x 2 FLOPs (one multiply-add) x 1.41 GHz |
| Tensor Cores (BF16/FP16) | 312 TFLOPS | 432 Tensor Cores doing about 512 FLOPs per clock each |

The matrix specialists are **16 times** faster than the generalists for matrix math. This is why "use a Tensor Core path" is the first question when someone says a kernel is slow.

**Number formats (the book names TF32; this table adds the rest).** A float has a sign bit, some exponent bits (the *range*), and some mantissa bits (the *precision*):

| Format | Sign / exponent / mantissa bits | Range | Precision | Where you see it |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | huge | high | default for old code |
| TF32 | 1 / 8 / 10 | same as FP32 | same as FP16 | A100 Tensor-Core trick for FP32 inputs |
| FP16 | 1 / 5 / 10 | small | medium | needs loss scaling |
| **BF16** | 1 / 8 / 7 | same as FP32 | low | the usual LLM format: FP32's range, so values rarely overflow |
| FP8 (E4M3 / E5M2) | 1 / 4 / 3 or 1 / 5 / 2 | small | very low | quantized inference on newer GPUs |

**Dense vs sparse (2:4).** Dense = multiply every number, zeros included. 2:4 sparse = in every group of 4 weights, at least 2 are exactly zero; the hardware stores only the 2 non-zeros plus their positions and skips the rest, so it runs about twice as fast:

```text
   weights in a group of 4:   0.7   0   -1.2   0         stored: [0.7, -1.2] + positions [0, 2]
   dense work: 4 multiplies          sparse work: 2 multiplies
```

| Precision | Dense | Sparse (2:4) |
|---|---|---|
| INT8 | 624 TOPS | 1,248 TOPS |
| BF16 / FP16 | **312 TFLOPS** | 624 TFLOPS |
| TF32 | 156 TFLOPS | 312 TFLOPS |
| FP64 | 19.5 TFLOPS | no sparse mode |

When someone quotes "312 TFLOPS" for an A100 they mean BF16/FP16 dense, which is what the roofline in Chapter 2 used.

**Pause and predict.** A kernel that does not use Tensor Cores gets stuck at FP32 CUDA-core speed. About how many times slower is it than the Tensor-Core path on the A100?

<details>
<summary>Answer (try first)</summary>

About 16x (19.5 vs 312 TFLOPS). Real kernels rarely reach either peak, but the gap is why attention and matmul kernels are written around Tensor Cores.

</details>

## 3 The GPU Execution Model

### 3.1 The Execution Flow of the SM

**Kid version.** The SM keeps **thousands of cooks** (threads) "on the books" at once, switching between groups of 32 as needed. Unlike a CPU, switching is almost free, because every cook's hands (registers) stay filled on the spot: nothing has to be saved or restored.

- **Thread scheduling:** which group of cooks goes next, and on what units.
- **Instruction pipelines:** 4 schedulers per SM, each starting one instruction per clock for some ready warp.
- **Data sharing:** a 192 KB counter (L1 / shared memory) that every cook at the station can reach quickly.

**Why lightweight switching matters** (a gap the book leaves implicit): suppose a cook asks for data from the warehouse. That takes about 290 clock cycles. A CPU would stall or have to save and restore state. The SM simply says "next!" and a *different* warp that already has its data goes. If enough warps are waiting in line, the arithmetic units never stop. This is called **latency hiding**.

```text
   time ──────────────────────────────────────────────────────────▶
   warp A:  [compute][ ......... waiting for memory ......... ][compute]
   warp B:           [compute][ ......... waiting ......... ][compute]
   warp C:                    [compute][ ........ waiting ........ ]
   warp D:                             [compute] ...
   units busy:  ████████████████████████████████████████████████   (as long as someone is ready)
```

### 3.2 A Detailed Explanation of the Core Terms of the Execution Model

**Kid version (kitchen).** You launch one **kernel** (one recipe step) with a **grid** of **blocks**. Each block is a **team** of cooks given to one station and kept there until the team is done. Inside a team, cooks move in lockstep groups of 32 called **warps**. Each single cook is a **thread**.

```text
   GRID  (one launch)
   ├── BLOCK 0 ──▶ goes to station (SM) 17     ┐ blocks are independent;
   ├── BLOCK 1 ──▶ goes to station (SM) 3      │ they cannot talk except via
   ├── BLOCK 2 ──▶ goes to station (SM) 88     ┘ the far warehouse (global memory)
   └── ...
        each BLOCK = e.g. 256 threads = 8 WARPS of 32 threads
```

| Level | Kid word | What it is | Key rule |
|---|---|---|---|
| Thread | One cook | Finest unit; runs the kernel code on its own piece of data; has private registers (up to 255) | Knows who it is via `threadIdx`, `blockIdx` |
| Warp | 32 cooks in step | The unit the SM schedules (up to 64 resident per SM) | All 32 do the same instruction at the same time |
| Block | A team | Programmer-chosen group of threads | Runs wholly on one SM; shares that SM's shared memory |
| Grid | The whole order | All blocks of one kernel launch | Blocks cannot be split across SMs |

**A real mini-kernel, line by line.** Add two vectors of 1000 numbers:

```cuda
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // my unique number across the whole grid
    if (i < n) c[i] = a[i] + b[i];                   // do my one element, if I have one
}
// launch: add<<<4, 256>>>(a, b, c, 1000);           // 4 blocks x 256 threads = 1024 threads
```

*Line by line:* `__global__` marks code that runs on the GPU and is launched from the CPU. `blockIdx.x` is "which team am I in", `blockDim.x` is "how big is each team" (256), `threadIdx.x` is "which cook am I within my team". Multiply and add to get a global number `i`. `if (i < n)` is the **guard**: 4 x 256 = 1024 threads exist but only 1000 elements, so 24 threads in the last block have nothing to do and must stay out of the way. The launch line asks for 4 blocks of 256 threads.

**Look for this pattern in the repo.** Small CUDA kernels live under `mini-sglang/python/minisgl/kernel/csrc/`. Their first lines almost always compute "which element/warp am I" from `blockIdx` and `threadIdx`.

**Under-filling the stadium.** The 4 blocks above would occupy only 4 of the 108 stations. Real kernels try to launch at least as many blocks as there are SMs, or they leave most of the chip idle. For example, an output with 4,096 columns cut into 128-wide pieces gives only 32 blocks, fewer than 108 SMs. This is one reason a tiny batch (M = 1) is awkward for GPUs.

#### 3.2.1 Warp

**Kid version.** A warp is a **bus with 32 seats**. Everyone on the bus must get off at the same stop and do the same thing. If half the passengers want a different stop, the bus makes **two trips**.

The SM can hold up to 64 warps (2,048 threads). It has 4 schedulers, each looking after 16 warps.

#### 3.2.2 Block

A programmer-chosen team (commonly 128-1024 threads). It lives on one SM and shares that SM's shared memory and registers. Because it can't be split, a block can't be bigger than one SM can hold.

#### 3.2.3 Thread

One worker; each thread runs the same program on different data, using its private registers and its ID to find its piece.

#### 3.2.4 SIMT (Single Instruction, Multiple Threads)

**Kid version.** The foreman shouts one instruction; all 32 cooks in the warp do it, each on their own ingredient. "Single Instruction, Multiple Threads". You write the program as if you were one cook; the hardware runs 32 copies in lockstep.

#### 3.2.5 The Essential Difference from SIMD

| | SIMT (GPU) | SIMD (CPU vector units, e.g. AVX) |
|---|---|---|
| Unit | Threads, each with its own program position | Lanes of one wide vector register |
| Branching | Allowed; paths run one after the other | Hard; all lanes do the same thing |
| Who manages it | Hardware scheduler with a thread mask | The compiler packs data into vectors |

**Warp divergence, with numbers.** Suppose the code says:

```cuda
if (threadIdx.x % 2 == 0) { pathA(); } else { pathB(); }
```

Even and odd threads are interleaved inside every warp, so each warp must run `pathA` (odd threads masked off), then `pathB` (even threads masked off). Time = `A + B`, utilisation = 50%. Now compare:

```cuda
if ((threadIdx.x / 32) % 2 == 0) { pathA(); } else { pathB(); }
```

The condition is the same for all 32 threads of any given warp (it depends only on the warp number), so **no divergence**: some warps take A, others take B, concurrently.

**Pause and predict.** Which of these causes divergence inside a warp? (a) `if (tid < 16)` (b) `if (tid < 32)` (c) `if (tid >= 64)`.

<details>
<summary>Answer (try first)</summary>

(a) only. In warp 0 (threads 0-31), threads 0-15 go one way and 16-31 the other. (b) and (c) split exactly on warp boundaries (32 and 64 are multiples of 32).

</details>

### 3.3 The GPU Memory Model

**Kid version.** The kitchen has **four storage places** at different distances:

```text
   your hands            registers           tiny, instant, private to one cook
   station counter       shared memory/L1    small, very quick, shared by the team
   shared pantry shelf   L2 cache (40 MB)    medium, automatic, shared by every station
   far warehouse         HBM / global (80 GB) huge, slow, the only place all teams can meet
```

```text
   fast, small, expensive                                        slow, huge, cheap
   ┌────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────────────┐
   │registers│─▶│ shared / L1   │─▶│ L2 cache  │─▶│ HBM global memory│
   └────────┘   └──────────────┘   └──────────┘   └─────────────────┘
   private         per SM            whole chip        whole chip
   ~0 cycles       ~20-33 cycles     ~200 cycles       ~290 cycles
```

**Why the central theme is distance.** A multiply takes a few cycles. Fetching from HBM takes about 290. A cook who must fetch from the warehouse for every multiply spends 99% of the time waiting. The best case is the book's: the team loads a small batch of data once into the station counter, and every cook works from there.

#### 3.3.1 Level One: Global Memory (HBM)

| Fact | Value |
|---|---|
| Where | HBM2e stacks beside the die |
| Size | 80 GB (A100 80GB) |
| Bandwidth | about 2 TB/s |
| Latency | about 290 cycles |
| Who manages | The programmer (`cudaMalloc`) |
| Who can see it | Every thread |

It holds weights, activations, the KV cache, everything persistent. It is also where CPU data arrives (over PCIe). It is about 100 times cheaper per byte than on-chip SRAM, which is why we can have 80 GB of it.

**Drift note.** Chapter 2 used 1.6 TB/s for the A100 (the 40 GB model); this chapter says about 2 TB/s (the 80 GB model). Ridge points differ accordingly: 312 / 1.6 = 195 vs 312 / 2.0 = 156.

#### 3.3.2 Level Two: L2 Cache

40 MB, on the chip, shared by all SMs, latency about 200 cycles, managed by hardware. Every access from any SM to HBM goes through the L2. Hot data (a weight block that many SMs read) gets caught here. It is also the place where SMs implicitly exchange data. Its bandwidth is not officially quoted; the book uses about 5 TB/s as an industry estimate.

#### 3.3.3 Level Three: L1 Cache / Shared Memory

192 KB per SM, split between hardware-managed L1 cache and programmer-managed **shared memory** (up to 164 KB can be shared memory). Latency in the book's measurements: 33 cycles for L1, 23 / 19 for shared loads / stores. Shared memory is how the cooks at one station exchange numbers and *reuse* data (tiling, below).

**Tiny numbers: aggregate bandwidth.** Each SM can move 128 bytes per clock from shared memory. 128 B x 108 SMs x 1.41 GHz = about 19.5 TB/s: about ten times the HBM bandwidth. Data you reuse from shared memory is about ten times cheaper to move than data you fetch again from HBM.

#### 3.3.4 Level Four: Register File

256 KB per SM (65,536 32-bit registers), private per thread, up to 255 per thread, allocated by the compiler, effectively zero latency.

**Occupancy: the registers decide how many cooks you can hire.** The register file is shared by all resident threads, so if each thread needs more, fewer threads fit:

| Registers per thread | Threads that fit in 65,536 | Warps (max 64) |
|---|---|---|
| 32 | 2,048 | 64 |
| 64 | 1,024 | 32 |
| 128 | 512 | 16 |
| 255 | 257 | 8 |

Fewer warps means less ability to hide the 290-cycle wait. That trade-off (fat threads vs more threads) is "occupancy", and it is why heavy kernels are tuned carefully.

#### 3.3.5 Why GPU Memory Is Divided into So Many Levels

Two physical reasons: (1) **Distance**: signals take time; near memory is fast, but there is little room near the cooks, far memory can be huge but slow. (2) **Locality**: data you just used you'll likely use again (*temporal*), and neighbours of data you just used you'll likely use next (*spatial*). A ladder lets each kind of reuse find a rung that suits it.

**Tiling: how shared memory saves trips (a gap the book only mentions).** Multiply two `n x n` matrices.

- *Naive:* every output element reads a row of A (n numbers) and a column of B (n numbers) straight from HBM. Total loads `2 n^3`.
- *Tiled:* a team computes a `T x T` block of outputs. It loads a `T x T` tile of A and of B into shared memory, all the cooks reuse them, then slides to the next tiles. Each loaded number is used `T` times.

Counting for `n = 4` with `T = 2`:

```text
   naive : 16 outputs x (4 + 4 loads) = 128 loads from HBM
   tiled : 4 teams x 2 phases x (2x2 + 2x2) = 64 loads from HBM        ← 2x fewer = T
   (with T = 4: 32 loads, 4x fewer)
```

In general tiling divides HBM traffic by `T`. With `T = 32` or `128` on a real matrix, that's 32-128 times fewer trips. That is the entire trick behind fast matmul kernels, and the reason bigger tiles are better (until the counter overflows).

**Coalescing: take the whole bus load in one trip (the book names it, doesn't explain it).** HBM serves memory in chunks of about 128 bytes. If 32 threads in a warp read 32 *consecutive* 4-byte floats, that is exactly one 128-byte chunk: **one trip**. If they read numbers spread far apart, each thread hits a different chunk: **up to 32 trips**.

```text
   coalesced:   thread 0..31 read addresses  0, 4, 8, 12, ... 124      → 1 chunk  (1x)
   strided:     thread 0..31 read addresses  0, 128, 256, 384, ...      → 32 chunks (32x the traffic)
```

**Pause and predict.** A matrix is stored row by row. If the 32 threads of a warp each read one element of the same *row*, is that coalesced? What if they each read one element of the same *column*?

<details>
<summary>Answer (try first)</summary>

Same row: consecutive addresses, coalesced. Same column: addresses a whole row apart, strided, up to 32x more traffic. This is why kernels transpose data or tile through shared memory.

</details>

#### 3.3.6 The Essential Difference Between GPU Memory and CPU Memory

| | GPU (A100) | CPU (Xeon) |
|---|---|---|
| Main-memory bandwidth | about 2 TB/s | several hundred GB/s |
| Fast on-chip store | Shared memory the *programmer* controls | Cache controlled automatically |
| Registers per thread | up to 255 | 16 general-purpose (x86-64) |
| Latency strategy | Hide it by switching warps | Reduce it (caches, prediction) |
| Consistency | Explicit synchronisation in a block | Hardware cache-coherence |

The GPU memory system is built for **throughput** and tolerates high latency; the CPU's is built for **latency**.

## 4 The Execution Flow of LLM Inference on the GPU

### 4.1 Why LLM Inference Is Inseparable from the GPU

**Kid version.** Four reasons, one sentence each:

1. The model is **mostly matrix multiplication**, the Tensor Cores' favourite food.
2. Each generated word needs the **whole recipe book** (e.g. 140 GB for 70 B parameters) fetched, and only HBM is fast enough.
3. Many words/requests can be cooked at once, so **latency-hiding throughput hardware** is a perfect match.
4. The **software ecosystem** (CUDA, PyTorch, FlashAttention, paged KV caches) is built around GPUs.

**The decode speed limit, as a formula.** At batch 1, one forward step reads (about) all the weights once, so the best-case speed is:

```text
   tokens per second  ≈  memory bandwidth  /  bytes of weights read per token
```

| Machine | Bandwidth | 14 GB (7B, bf16) | 7 GB (fp8) | 3.5 GB (int4) |
|---|---|---|---|---|
| Your M1 Pro Mac (unified memory) | 200 GB/s | 14 tok/s | 29 | 57 |
| A big CPU server | about 400 GB/s | 29 | 57 | 114 |
| A100 80GB | 2.0 TB/s | 143 | 286 | 571 |
| H100 | 3.35 TB/s | 239 | 479 | 957 |

(Upper bounds: ignores KV reads, kernel overheads and anything slower than weight streaming.) Two lessons: bandwidth sets the ceiling, and **quantizing weights directly raises the ceiling** because fewer bytes cross the bus.

### 4.2 The GPU Inference Execution Process

#### 4.2.1 CPU work

The CPU loads the weights from disk into GPU memory (once, at start), receives each request, tokenizes it, and prepares inputs. Then it asks the GPU to run a **kernel** (a launch call that names the grid and block sizes).

**The kernel-launch tax (a gap).** Each launch costs the CPU a few microseconds (typically around 3-10 us). One forward step of a 28-layer model launches hundreds of small kernels:

```text
   28 layers x ~15 kernels each = ~420 launches x ~5 us = ~2 ms of pure launching per step
```

If the GPU finishes each kernel faster than the CPU can launch the next, the GPU waits. That matters for **decode with small batches**, where kernels are tiny. The fix used in the engine (Part II) is **CUDA Graphs**: record the whole sequence of kernel launches once, then replay it with a single call. You will see it in the engine code (`engine/graph.py`).

#### 4.2.2 Data Transfer

Token IDs and small inputs go over PCIe 4.0 x16 (about 32 GB/s each way). A prompt of thousands of token IDs is a few KB: negligible. Loading the whole 14 GB of weights over PCIe would take at least 0.44 s; over NVLink (about 600 GB/s) 0.023 s; HBM-to-SM is faster still. Transfers between devices are many times slower than on-device memory, so avoid them in the hot loop.

#### 4.2.3 The GPU Computing Matrix Multiplication in Parallel

The book's four-step path for one tile of `C = A x B`:

```text
   HBM (global) ──1.load a tile──▶ shared memory ──2.each thread loads its piece──▶ registers
                                                                                      │
   HBM (global) ◀─4.write result back via shared─────────── 3.CUDA/Tensor Core math ◀┘
```

1. **Task decomposition:** the big matmul becomes many small independent tile jobs.
2. **Tiling:** each output tile goes to one block.
3. **Fine allocation:** inside the block, warps and threads each own a few output elements.

Then on the SM: load tile to shared memory (fewer HBM trips), threads copy their part to registers, Tensor Cores multiply-accumulate, results go back out.

**Where the story lies.** Real kernels overlap steps 1 and 3 (fetch the next tile while computing the current one) so the cooks never wait.

### 4.3 The Prefill and Decode Stages

| | Prefill | Decode |
|---|---|---|
| Math shape | Matrix x matrix (**GEMM**) | Matrix x a single vector (**GEMV**) per sequence |
| Data reuse | Each weight tile reused by hundreds or thousands of tokens | Each weight used by one token (or `B` tokens if batched) |
| Limiting unit | Tensor Cores (compute) | HBM bandwidth |
| GPU feeling | Busy stadium | Cooks waiting at the warehouse door |

**Why GEMV can't use tiling's magic:** tiling pays off when each loaded number is reused many times. In GEMV the weight matrix is read *once* per step: nothing to reuse, so the speed is just "stream 14 GB through the bus". Batching decode requests turns `B` GEMVs into one GEMM with `B` rows, which restores reuse. (This is Chapter 2's batching idea, from the hardware side.)

### 4.4 Summary

| Stage | Core task | Type | GPU bottleneck | Key optimisation |
|---|---|---|---|---|
| Prefill | Process the prompt | Compute-heavy | Tensor Core throughput | Maximise parallelism, large GEMMs |
| Decode | Generate token by token | Memory-heavy | HBM bandwidth | KV cache, quantization, continuous batching |

## 5 Summary and Exercises

### 5.1 Summary

**Kid version.** The GPU is a stadium of simple cooks, built to maximise **total output**, not the speed of any one dish. Its floor plan nests (GPU → GPC → TPC → SM → cores), and its work nests too (grid → block → warp → thread). The ladder of storage (registers → shared/L1 → L2 → HBM) trades size for speed, and programs go fast when they stay on the high rungs and move whole warps' worth of data in single trips. Compute power has grown faster than memory bandwidth (the "memory wall"), which is why decode, the memory-heavy stage, is the main challenge.

```text
   rule 1: fetch fewer bytes (tile, cache, quantize, share K/V)
   rule 2: never let the cooks wait (many warps, overlap fetch and compute, CUDA graphs)
   rule 3: keep the whole warp doing the same thing (avoid divergence, coalesce loads)
```

### 5.2 Exercises

**1. Stages of LLM inference, their characteristics, and why GPUs.**

<details><summary>Hint</summary>

Preprocessing (CPU tokenizing, launching), prefill (compute-heavy GEMM), decode (memory-heavy GEMV + KV reads). GPUs: parallel multiply-accumulate hardware, high bandwidth, latency hiding, mature software.

</details>

**2. CPU vs GPU in design; why GPUs suit matrix math.**

<details><summary>Hint</summary>

CPU: few strong cores, much area for control and cache, optimises single-task latency. GPU: thousands of simple ALUs, little control, optimises total throughput. Matrix math = enormous numbers of independent multiply-adds with regular access, exactly the GPU's sweet spot.

</details>

**3. Prefill vs decode: computation type, bottleneck, optimisations.**

<details><summary>Hint</summary>

Prefill: GEMM, Tensor Core bound; fixes: bigger well-shaped matmuls, parallelism. Decode: GEMV/memory bound; fixes: KV cache, quantization, batching.

</details>

**Extra drills (not in the book).**

4. A kernel launches 40 blocks on an A100. Roughly what fraction of the SMs have work? (About 37%: 40 of 108. At least 108 blocks are needed to touch every SM once.)
5. Why does making each thread use more registers sometimes slow a kernel down? (Fewer resident warps, so less latency hiding.)
6. A 4-byte float, 32 threads reading with a stride of 2 elements: how many 128-byte chunks does the warp touch? (Addresses span 32 x 8 = 256 bytes, so 2 chunks: 2x the traffic of the perfectly packed case.)
