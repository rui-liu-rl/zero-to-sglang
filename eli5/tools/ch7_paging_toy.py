"""Toy numbers for Part II chapter 7: paged KV cache and online softmax (standard library only).

Run:    python3 -I eli5/tools/ch7_paging_toy.py

  Part 1  memory arithmetic: preallocation vs paging (Qwen3-0.6B KV size)
  Part 2  the book's external-fragmentation example, checked
  Part 3  token -> (logical block, offset) -> physical block -> address
  Part 4  online softmax, step by step, versus the direct formula and a wrong merge
  Part 5  a random allocate/free simulation: "memory exists but cannot be used"
"""
import math
import random

# Qwen3-0.6B: 28 layers, 8 KV heads, head_dim 128, K and V, 2 bytes (bf16)
LAYERS, KV_HEADS, HEAD_DIM, BYTES = 28, 8, 128, 2
PER_TOKEN = LAYERS * KV_HEADS * HEAD_DIM * 2 * BYTES  # bytes
KIB, MIB, GIB = 1024, 1024**2, 1024**3


def part1():
    print("== Part 1: preallocation vs paging (Qwen3-0.6B) ==")
    print(f"  KV bytes per token = {LAYERS} x {KV_HEADS} x {HEAD_DIM} x 2(K,V) x {BYTES} = {PER_TOKEN} B = {PER_TOKEN/KIB:.0f} KiB")
    max_len, used, block = 2048, 200, 16
    pre = max_len * PER_TOKEN
    need = used * PER_TOKEN
    blocks = math.ceil(used / block)
    paged = blocks * block * PER_TOKEN
    print(f"  preallocate {max_len} tokens: {pre/MIB:.1f} MiB; the request really uses {used} tokens = {need/MIB:.3f} MiB"
          f" -> {100*used/max_len:.1f}% used, {100 - 100*used/max_len:.1f}% idle")
    print(f"  paged, block {block}: ceil({used}/{block}) = {blocks} blocks = {blocks*block} token slots = {paged/MIB:.2f} MiB;"
          f" wasted slots = {blocks*block-used} ({(blocks*block-used)*PER_TOKEN/MIB:.3f} MiB, {100*(blocks*block-used)/(blocks*block):.1f}%)")
    pool = 10 * GIB
    print(f"  a 10 GiB KV pool holds {pool // pre} such requests with preallocation, {pool // paged} with paging"
          f" ({(pool // paged) / (pool // pre):.1f}x)")


def part2():
    print("== Part 2: the book's external-fragmentation example ==")
    holes = [512, 256, 1024, 768, 1024]  # MiB
    total = sum(holes)
    print(f"  holes (MiB) {holes}: sum = {total} MiB = {total/1024:.3f} GiB   (the book says 3.60 GiB)")
    need = 1536
    print(f"  a request needs {need} MiB contiguous; largest hole = {max(holes)} MiB -> contiguous allocation fails")
    block_mib = 16 * PER_TOKEN / MIB
    nb = math.ceil(need / block_mib)
    print(f"  one 16-token block = {block_mib} MiB; the request needs {nb} blocks; the holes together contain {int(total / block_mib)} blocks -> paging succeeds")


def part3():
    print("== Part 3: address translation, 50-token request, block size 16, table [12, 7, 42, 3] ==")
    b = 16
    table = [12, 7, 42, 3]
    print("   token | logical block | offset | physical block | slot address (= phys*16 + offset)")
    for t in (0, 15, 16, 31, 32, 37, 47, 48, 49):
        lb, off = divmod(t, b)
        pb = table[lb]
        print(f"   {t:5d} | {lb:13d} | {off:6d} | {pb:14d} | {pb*b+off}")
    print(f"  blocks used: {math.ceil(50/b)}; slots in the last block actually holding tokens: {50 - 3*b} of {b}")


def online_softmax(scores, values, block):
    m, l, o = -math.inf, 0.0, 0.0
    rows = []
    for j in range(0, len(scores), block):
        s = scores[j:j + block]
        v = values[j:j + block]
        mj = max(s)
        lj = sum(math.exp(x - mj) for x in s)
        oj = sum(math.exp(x - mj) * y for x, y in zip(s, v))
        m2 = max(m, mj)
        scale_old = math.exp(m - m2) if m != -math.inf else 0.0
        scale_new = math.exp(mj - m2)
        l = l * scale_old + lj * scale_new
        o = o * scale_old + oj * scale_new
        m = m2
        rows.append((j // block, mj, lj, oj, scale_old, scale_new, m, l, o))
    return o / l, rows


def part4():
    print("== Part 4: online softmax over 3 blocks of 2 ==")
    scores = [2.0, 1.0, 3.0, 0.5, 4.0, 1.5]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    mx = max(scores)
    direct = sum(math.exp(s - mx) * v for s, v in zip(scores, values)) / sum(math.exp(s - mx) for s in scores)
    result, rows = online_softmax(scores, values, 2)
    print("   block | local m | local l | local O | x old | x new | running m | running l | running O")
    for r in rows:
        print(f"   {r[0]:5d} | {r[1]:7.3f} | {r[2]:7.3f} | {r[3]:7.3f} | {r[4]:5.3f} | {r[5]:5.3f} | {r[6]:9.3f} | {r[7]:9.3f} | {r[8]:9.3f}")
    print(f"  final O/l = {result:.6f}   direct softmax-weighted sum = {direct:.6f}   difference = {abs(result-direct):.2e}")
    # wrong merge: add local l and O without moving them to a common reference
    l_bad = sum(sum(math.exp(x - max(scores[j:j+2])) for x in scores[j:j+2]) for j in range(0, 6, 2))
    o_bad = sum(sum(math.exp(x - max(scores[j:j+2])) * y for x, y in zip(scores[j:j+2], values[j:j+2])) for j in range(0, 6, 2))
    print(f"  WRONG merge (add local l and O with no rescaling): O/l = {o_bad/l_bad:.6f}  (error {abs(o_bad/l_bad-direct):.3f})")
    big = [1000.0, 1001.0, 999.0, 1002.0]
    try:
        math.exp(big[0])
    except OverflowError:
        print("  naive exp(1000) overflows:", "OverflowError")
    r2, _ = online_softmax(big, [1.0, 2.0, 3.0, 4.0], 2)
    mxb = max(big)
    d2 = sum(math.exp(s - mxb) * v for s, v in zip(big, [1.0, 2.0, 3.0, 4.0])) / sum(math.exp(s - mxb) for s in big)
    print(f"  scores around 1000: online = {r2:.6f}, max-subtracted direct = {d2:.6f}")


def part5(seed=0):
    print("== Part 5: random allocate/free, exact-size contiguous (best case) vs 4-slot blocks ==")
    rng = random.Random(seed)
    TOTAL, BLOCK = 2000, 4
    steps = 3000
    # one shared workload
    arrivals = []
    for t in range(steps):
        arrivals.append((t, rng.randint(20, 200), rng.randint(5, 60)))

    def run(kind):
        live = []  # (end_time, data)
        free_blocks = list(range(TOTAL // BLOCK))
        holes = [(0, TOTAL)]  # (start, size) contiguous free list
        served = fragfail = realfail = 0
        for t, size, life in arrivals:
            # release finished
            for item in [x for x in live if x[0] <= t]:
                live.remove(item)
                if kind == "paged":
                    free_blocks.extend(item[1])
                else:
                    holes.append(item[1])
                    holes.sort()
                    merged = []
                    for s, z in holes:
                        if merged and merged[-1][0] + merged[-1][1] == s:
                            merged[-1] = (merged[-1][0], merged[-1][1] + z)
                        else:
                            merged.append((s, z))
                    holes = merged
            if kind == "paged":
                need = math.ceil(size / BLOCK)
                if need <= len(free_blocks):
                    got, free_blocks = free_blocks[:need], free_blocks[need:]
                    live.append((t + life, got))
                    served += 1
                else:
                    realfail += 1
            else:
                free_total = sum(z for _, z in holes)
                spot = next((h for h in holes if h[1] >= size), None)
                if spot:
                    s, z = spot
                    holes.remove(spot)
                    if z > size:
                        holes.append((s + size, z - size))
                        holes.sort()
                    live.append((t + life, (s, size)))
                    served += 1
                elif free_total >= size:
                    fragfail += 1  # memory exists but is not contiguous
                else:
                    realfail += 1
        return served, fragfail, realfail

    for kind in ("contiguous", "paged"):
        served, fragfail, realfail = run(kind)
        print(f"  {kind:10s}: served {served:4d} | refused although enough TOTAL memory was free {fragfail:4d} | refused, truly full {realfail:4d}")
    print(f"  ({steps} arrivals, pool of {TOTAL} token slots, request sizes 20-200, lifetimes 5-60 steps, seed {seed}; contiguous gets the exact final size, i.e. no preallocation waste)")


part1()
part2()
part3()
part4()
part5()
