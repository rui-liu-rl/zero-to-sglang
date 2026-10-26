"""Toy tensor parallelism for Part II chapter 9 (standard library only; "GPUs" are list slices).

Run:    python3 -I eli5/tools/ch9_tp_toy.py

  Part 1  column-parallel linear + all_gather  == the full matmul
  Part 2  row-parallel linear + all_reduce     == the full matmul
  Part 3  column then row: ONE all_reduce per block; and why the nonlinearity forces that order
  Part 4  the bias gotcha for row-parallel layers
  Part 5  vocab-parallel embedding (masked lookup + all_reduce) and LM head (all_gather + truncate)
  Part 6  attention heads and KV-head replication: what doubling TP does (and doesn't) buy
  Part 7  communication volume of the all_reduces in one forward pass
  Part 8  weight memory per GPU
  Part 9  how many tokens of KV cache fit when weights and per-token KV both shrink with TP
  Part 10 shard-then-merge vs merge-then-split for the fused qkv matrix, and KV head replication in the loader
"""
import math
import random


def matmul(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def cols(m, lo, hi):
    return [row[lo:hi] for row in m]


def rows(m, lo, hi):
    return m[lo:hi]


def add(a, b):
    return [[x + y for x, y in zip(r1, r2)] for r1, r2 in zip(a, b)]


def concat_cols(parts):
    return [sum((p[i] for p in parts), []) for i in range(len(parts[0]))]


def close(a, b):
    return all(abs(x - y) < 1e-9 for r1, r2 in zip(a, b) for x, y in zip(r1, r2))


rng = random.Random(0)


def rand(r, c):
    return [[round(rng.uniform(-1, 1), 3) for _ in range(c)] for _ in range(r)]


def part1():
    print("== Part 1: column parallel (split W by OUTPUT columns), tp = 2 ==")
    X, W = rand(2, 4), rand(4, 6)
    full = matmul(X, W)
    parts = [matmul(X, cols(W, r * 3, r * 3 + 3)) for r in range(2)]  # every GPU sees the WHOLE X
    gathered = concat_cols(parts)  # all_gather along the output dimension
    print(f"  X is 2x4 (replicated on both GPUs); W is 4x6 -> each GPU holds a 4x3 slice; each local result is 2x3")
    print(f"  all_gather(concat columns) == full X@W ? {close(gathered, full)}")


def part2():
    print("== Part 2: row parallel (split W by INPUT rows, X by columns), tp = 2 ==")
    X, W = rand(2, 6), rand(6, 4)
    full = matmul(X, W)
    partial = [matmul(cols(X, r * 3, r * 3 + 3), rows(W, r * 3, r * 3 + 3)) for r in range(2)]
    summed = add(partial[0], partial[1])  # all_reduce(sum)
    print(f"  each GPU holds X[:, 3 cols] (its local shard) and W[3 rows, :]; each local result is a 2x4 PARTIAL sum")
    print(f"  all_reduce(sum) == full X@W ? {close(summed, full)}")
    print(f"  one partial alone == full ? {close(partial[0], full)}")


def relu(m):
    return [[max(0.0, x) for x in r] for r in m]


def part3():
    print("== Part 3: column THEN row (one all_reduce), and the wrong order ==")
    X = rand(2, 4)
    W1, W2 = rand(4, 6), rand(6, 4)  # up-projection then down-projection with a ReLU in between
    full = matmul(relu(matmul(X, W1)), W2)
    # right way: column-parallel W1 (no comm), local ReLU, row-parallel W2, ONE all_reduce
    locals_ = []
    for r in range(2):
        h = relu(matmul(X, cols(W1, r * 3, r * 3 + 3)))  # local columns, local nonlinearity
        locals_.append(matmul(h, rows(W2, r * 3, r * 3 + 3)))
    ok = add(locals_[0], locals_[1])
    print(f"  column -> local relu -> row -> all_reduce == full ? {close(ok, full)}   (1 collective)")
    # wrong way: row-parallel first, apply the nonlinearity to PARTIAL sums, then sum
    Xp = [cols(X, r * 2, r * 2 + 2) for r in range(2)]
    W1r = [rows(W1, r * 2, r * 2 + 2) for r in range(2)]
    partial = [matmul(Xp[r], W1r[r]) for r in range(2)]
    wrong = add(relu(partial[0]), relu(partial[1]))
    right_h = relu(add(partial[0], partial[1]))
    print(f"  row-parallel first, relu on the partial sums, then add: equals relu(full sum)? {close(wrong, right_h)}  (relu(a)+relu(b) != relu(a+b))")
    print("  => after a row-parallel layer you must all_reduce BEFORE any nonlinearity; column-then-row needs only one at the end")


def part4():
    print("== Part 4: the bias gotcha ==")
    X, W = rand(2, 6), rand(6, 4)
    b = [[1.0, 2.0, 3.0, 4.0]] * 2
    full = add(matmul(X, W), b)
    partial = [matmul(cols(X, r * 3, r * 3 + 3), rows(W, r * 3, r * 3 + 3)) for r in range(2)]
    wrong = add(add(partial[0], b), add(partial[1], b))  # bias added on EVERY rank before the sum
    right = add(add(partial[0], partial[1]), b)           # bias added once after the sum
    print(f"  bias on every rank before all_reduce == full? {close(wrong, full)}   (bias counted tp = 2 times)")
    print(f"  bias added once after the sum        == full? {close(right, full)}")
    print("  mini-sglang's row-parallel layers are therefore created without a bias (see layers/linear.py)")


def div_ceil(a, b):
    return (a + b - 1) // b


def part5():
    print("== Part 5: vocab-parallel embedding and LM head (vocab 10, hidden 2, tp 4) ==")
    V, H, tp = 10, 2, 4
    table = [[float(v * 10 + d) for d in range(H)] for v in range(V)]  # row v = [10v, 10v+1]
    per = div_ceil(V, tp)
    print(f"  rows per GPU = ceil({V}/{tp}) = {per}; owned ranges:", [(r * per, min(r * per + per, V)) for r in range(tp)])
    ids = [7, 2, 9]
    outs = []
    for r in range(tp):
        lo, hi = r * per, min(r * per + per, V)
        outs.append([table[i] if lo <= i < hi else [0.0] * H for i in ids])
    summed = [[sum(o[t][d] for o in outs) for d in range(H)] for t in range(len(ids))]
    print(f"  token ids {ids}: per-GPU outputs (0 if not mine) summed by all_reduce = {summed}")
    print(f"  equals an ordinary lookup {[table[i] for i in ids]}? {summed == [table[i] for i in ids]}")
    # LM head: x [1, H] times table^T gives V scores; each GPU computes its slice (padded to `per` columns)
    x = [3.0, 1.0]
    slices = []
    for r in range(tp):
        lo = r * per
        sl = []
        for j in range(per):
            idx = lo + j
            row = table[idx] if idx < V else [float("nan")] * H  # padding rows are garbage (torch.empty in the real code)
            sl.append(sum(a * b for a, b in zip(x, row)))
        slices.append(sl)
    gathered = [v for sl in slices for v in sl]
    print(f"  LM head: all_gather gives {len(gathered)} columns (= {tp} x {per}); truncate to the real vocab size {V}: padding columns {[ 'nan' if math.isnan(v) else v for v in gathered[V:]]} are dropped")
    truth = [sum(a * b for a, b in zip(x, table[i])) for i in range(V)]
    print(f"  truncated logits == ordinary x @ table^T ? {gathered[:V] == truth}")


def part6():
    print("== Part 6: heads and KV replication (32 query heads, 8 KV heads, head_dim 128, 64 layers, bf16) ==")
    q, kv, hd, layers = 32, 8, 128, 64

    def div_even(a, b, allow_replicate=False):
        if allow_replicate and b > a:
            assert b % a == 0
            return 1
        assert a % b == 0
        return a // b

    print("   tp | q heads/GPU | KV heads/GPU | KV bytes per token per GPU | all GPUs together vs one copy")
    for tp in (1, 2, 4, 8, 16, 3):
        try:
            ql = div_even(q, tp)
            kl = div_even(kv, tp, allow_replicate=True)
        except AssertionError:
            print(f"   {tp:2d} | not allowed (32 query heads do not split evenly over {tp} GPUs)")
            continue
        per_tok = layers * kl * hd * 2 * 2
        total = per_tok * tp
        one = layers * kv * hd * 2 * 2
        print(f"   {tp:2d} | {ql:11d} | {kl:12d} | {per_tok/1024:8.0f} KiB               | {total/one:.1f}x")


def part7():
    print("== Part 7: all_reduce traffic in one forward pass (hidden 5120, 64 layers, bf16) ==")
    H, L = 5120, 64
    calls = 2 * L + 1  # o_proj and down_proj per layer, plus the embedding
    for name, n_tok in (("decode, 64 requests", 64), ("prefill, 8192 tokens", 8192)):
        size = n_tok * H * 2
        print(f"  {name}: one all_reduce moves a {size/1024**2:.2f} MiB tensor; {calls} all_reduce calls per forward")
        for tp in (2, 4, 8):
            per_gpu = 2 * (tp - 1) / tp * size * calls  # ring all_reduce: each GPU sends/receives 2(n-1)/n x size
            print(f"     tp={tp}: about {per_gpu/1024**3:.2f} GiB sent per GPU per forward pass (ring all_reduce estimate)")


def part8():
    print("== Part 8: weight memory per GPU ==")
    params = 3.2e10
    gb = params * 2 / 1e9
    print(f"  3.2e10 parameters x 2 bytes = {gb:.0f} GB (decimal) = {params*2/1024**3:.1f} GiB")
    for tp in (1, 2, 4, 8):
        print(f"    tp={tp}: about {gb/tp:.0f} GB of weights per GPU (the small replicated parts such as norms are ignored)")
    big = 1.6e12 * 2
    print(f"  1.6e12 parameters x 2 bytes = {big/1e12:.1f} TB; even 80 GB GPUs need at least {math.ceil(big/80e9)} of them for the weights alone")


def part9():
    print("== Part 9: KV capacity vs TP (80 GB GPUs, 32B-class model: 64 GB of bf16 weights, 64 layers, 8 KV heads, head_dim 128) ==")
    gpu, weights = 80e9, 64e9
    layers, kv, hd = 64, 8, 128
    print("   tp | weights/GPU | free/GPU | KV bytes per token per GPU | tokens of KV that fit | vs tp=1")
    base = None
    for tp in (1, 2, 4, 8, 16):
        free = gpu - weights / tp
        kv_local = 1 if tp > kv else kv // tp
        per_tok = layers * kv_local * hd * 2 * 2
        tokens = free / per_tok
        base = base or tokens
        print(f"   {tp:2d} | {weights/tp/1e9:8.0f} GB | {free/1e9:5.0f} GB | {per_tok/1024:8.0f} KiB               | {tokens:12,.0f}          | {tokens/base:5.1f}x")
    print("   (ignores activations, CUDA graph buffers and communication buffers; real capacity is lower)")


def part10():
    print("== Part 10: fused qkv_proj: shard THEN merge (what the loader does) ==")
    n = 2
    q = [f"q{i}" for i in range(4)]  # 4 query-head rows (toy: one row per head)
    k = [f"k{i}" for i in range(2)]  # 2 KV heads
    v = [f"v{i}" for i in range(2)]
    full = q + k + v
    right = []
    for r in range(n):
        qs = q[r * len(q) // n:(r + 1) * len(q) // n]
        ks = k[r * len(k) // n:(r + 1) * len(k) // n]
        vs = v[r * len(v) // n:(r + 1) * len(v) // n]
        right.append(qs + ks + vs)
    wrong = [full[r * len(full) // n:(r + 1) * len(full) // n] for r in range(n)]
    for r in range(n):
        print(f"   rank {r}: shard-then-merge {right[r]}    merge-then-split {wrong[r]}")
    print("   merge-then-split hands rank 0 ALL the query rows and rank 1 ALL the key/value rows: attention on a rank could not run")
    print("== Part 10b: KV head replication in the loader (8 KV heads of 128 rows each -> 16 GPUs) ==")
    n, kv = 16, 8
    print("   rank -> kv head index (r * kv // n):", [(r, r * kv // n) for r in range(n)])
    print("   every head appears on", n // kv, "GPUs")


part1()
part2()
part3()
part4()
part5()
part6()
part7()
part8()
part9()
part10()
