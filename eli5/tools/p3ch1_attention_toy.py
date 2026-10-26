"""Toy numbers for Part III chapter 1: attention backends (standard library only).

Run:    python3 -I eli5/tools/p3ch1_attention_toy.py

  Part 1  the naive n x n score matrix at 32k tokens, and a back-of-envelope for tiled attention
  Part 2  the book's 4x4 causal example, tile by tile (Figure 5a-5c), reproduced
  Part 3  split-KV: independent partial results merged exactly
  Part 4  uneven requests: why one task per (request, head) leaves the GPU idle, and what splitting does
"""
import math
import random

GIB, MIB = 1024**3, 1024**2


def part1():
    print("== Part 1: n x n scores at 32,768 tokens (one layer, one head, 2 bytes per number) ==")
    n, d, b = 32768, 128, 2
    s_entries = n * n
    s_bytes = s_entries * b
    print(f"  S has {s_entries:,} entries = {s_bytes/GIB:.2f} GiB (= {s_bytes/1e9:.2f} GB)")
    print(f"  write S + read S = {2*s_bytes/GIB:.1f} GiB;  adding P (softmax output) write + read = {4*s_bytes/GIB:.1f} GiB   (the book's 4 GiB and the figure's 8 GiB)")
    print("  -- my own back-of-envelope for a tiled kernel (assumptions: head_dim 128, bf16, non-causal) --")
    qo = 2 * n * d * b
    kv_pass = 2 * n * d * b
    for br in (64, 128, 256):
        passes = n // br
        total = qo + kv_pass * passes
        print(f"  tile of {br:3d} query rows: K and V re-read {passes} times x {kv_pass/MIB:.0f} MiB + Q,O {qo/MIB:.0f} MiB"
              f" = {total/GIB:.2f} GiB   (and no n x n matrix is ever stored)")
    print("  => tiling does NOT make traffic vanish: K and V are re-read once per query tile. It removes the n x n storage and fuses the steps.")


def part2():
    print("== Part 2: the book's 4x4 example, V = [10, 20, 30, 40], query 4 sees all four keys ==")
    v = [10.0, 20.0, 30.0, 40.0]
    w = [1.0, 1.0, 2.0, 2.0]  # exp(score) of query 4 against keys 1..4 (the figure's numbers)
    naive = sum(wi * vi for wi, vi in zip(w, v)) / sum(w)
    print(f"  naive: weights = exp scores / {sum(w):g} = {[round(x/sum(w),4) for x in w]}; output = {naive:.4f}")
    la, oa = sum(w[:2]), sum(wi * vi for wi, vi in zip(w[:2], v[:2]))
    print(f"  after tile A (keys 1,2): total weight {la:g}, output so far = {oa/la:g}")
    lb, ob = sum(w[2:]), sum(wi * vi for wi, vi in zip(w[2:], v[2:]))
    print(f"  tile B alone  (keys 3,4): total weight {lb:g}, output = {ob/lb:g}")
    share_a = la / (la + lb)
    merged = share_a * (oa / la) + (1 - share_a) * (ob / lb)
    print(f"  merge: A's share of the total weight = {share_a:.4f} (1/3); merged output = {merged:.4f}   equals naive? {abs(merged-naive) < 1e-12}")
    wrong = (oa / la + ob / lb) / 2
    print(f"  WRONG (average the two tile outputs with no re-weighting) = {wrong:.4f}")


def merge(partials):
    m = max(p[0] for p in partials)
    l = sum(p[1] * math.exp(p[0] - m) for p in partials)
    o = sum(p[2] * math.exp(p[0] - m) for p in partials)
    return m, l, o


def part3():
    print("== Part 3: split-KV ==")
    rng = random.Random(1)
    L = 4096
    scores = [rng.uniform(-3, 3) for _ in range(L)]
    values = [rng.uniform(-1, 1) for _ in range(L)]
    mx = max(scores)
    direct = sum(math.exp(s - mx) * v for s, v in zip(scores, values)) / sum(math.exp(s - mx) for s in scores)
    for nsplit in (1, 2, 8, 16):
        size = L // nsplit
        parts = []
        for i in range(nsplit):
            s = scores[i * size:(i + 1) * size]
            v = values[i * size:(i + 1) * size]
            m = max(s)
            parts.append((m, sum(math.exp(x - m) for x in s), sum(math.exp(x - m) * y for x, y in zip(s, v))))
        m, l, o = merge(parts)
        print(f"  {nsplit:2d} segments of {size:4d} keys: merged output {o/l:+.12f}   direct {direct:+.12f}   diff {abs(o/l-direct):.1e}")
    print("  each segment is independent (can run on its own SM); the merge needs only three numbers per segment: local max, local weight total, local output")


def part4():
    print("== Part 4: one decode batch, A has 128 tokens of history, B has 8,192 ==")
    heads = 8  # KV heads, illustrative
    lens = {"A": 128, "B": 8192}
    chunk = 512
    for name, mode in (("no split", None), (f"split-KV, {chunk}-token segments", chunk)):
        tasks = []
        for r, ln in lens.items():
            if mode is None:
                tasks += [(r, ln)] * heads
            else:
                full, rem = divmod(ln, mode)
                tasks += [(r, mode)] * (full * heads) + ([(r, rem)] * heads if rem else [])
        longest = max(t[1] for t in tasks)
        total = sum(t[1] for t in tasks)
        print(f"  {name:32s}: {len(tasks):4d} tasks, longest task {longest:5d} tokens, total work {total}"
              f" -> if every task had its own SM, time ~ longest = {longest} (relative units)")
    print("  same total work; the critical path drops 16x, at the price of 16x more partial results to merge for B")


part1()
part2()
part3()
part4()
