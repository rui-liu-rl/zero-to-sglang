"""Toy numbers for Part III chapter 3: hierarchical caching (standard library only).

Run:    python3 -I eli5/tools/p3ch3_hicache_toy.py

Every bandwidth, speed and latency below is an ASSUMPTION chosen to illustrate the trade-off, not a
measurement. Change the constants at the top and rerun to see where the break-even moves.

  Part 1  restore vs recompute, for a small and a large model
  Part 2  the longest CONTINUOUS reusable prefix across tiers
  Part 3  layer-by-layer restore overlapped with compute
  Part 4  prefetch while queued, and the three wait policies
  Part 5  the book's exercise 3
  Part 6  "prefix fingerprints": why the same page of text gets a different key in a different context
"""
MIB = 1024**2
GIB = 1024**3

PCIE_HOST_TO_GPU = 25e9    # bytes/s, assumed effective host->GPU
STORAGE_TO_HOST = 2e9      # bytes/s, assumed effective storage->host
GPU_EFFECTIVE_FLOPS = 100e12  # FLOP/s, assumed effective for prefill


def part1():
    print("== Part 1: restore vs recompute for a 1,024-token prefix ==")
    print(f"  assumed: host->GPU {PCIE_HOST_TO_GPU/1e9:.0f} GB/s, storage->host {STORAGE_TO_HOST/1e9:.0f} GB/s, GPU {GPU_EFFECTIVE_FLOPS/1e12:.0f} TFLOP/s effective")
    models = [
        ("Qwen3-0.6B-like", 0.6e9, 28 * 8 * 128 * 2 * 2),   # 28 layers, 8 KV heads, head_dim 128, K+V, bf16
        ("32B-class", 32e9, 64 * 8 * 128 * 2 * 2),
    ]
    tokens = 1024
    for name, params, kv_per_tok in models:
        kv = tokens * kv_per_tok
        recompute = 2 * params * tokens / GPU_EFFECTIVE_FLOPS
        l2 = kv / PCIE_HOST_TO_GPU
        l3 = kv / STORAGE_TO_HOST + kv / PCIE_HOST_TO_GPU
        better_l2 = "restore wins" if l2 < recompute else "recompute wins"
        better_l3 = "restore wins" if l3 < recompute else "recompute wins"
        breakeven = kv / recompute
        print(f"  {name:16s}: KV per token {kv_per_tok/1024:.0f} KiB -> prefix KV {kv/MIB:.0f} MiB;"
              f" recompute {recompute*1e3:7.1f} ms | L2 restore {l2*1e3:6.1f} ms ({better_l2}) | L3 read+restore {l3*1e3:7.1f} ms ({better_l3})")
        print(f"                     break-even link speed (restore = recompute): {breakeven/1e9:.1f} GB/s")


def reusable_prefix(pages):
    """pages: list of tier names or None, one per cache page, from the start of the prompt. Returns number of reusable pages."""
    n = 0
    for p in pages:
        if p is None:
            break
        n += 1
    return n


def part2():
    print("== Part 2: continuous reuse, 1,024-token prefix in pages of 256 tokens ==")
    cases = {
        "L1, L1, L2, L3 (everything available)": ["L1", "L1", "L2", "L3"],
        "L1, L1, L2, missing": ["L1", "L1", "L2", None],
        "L1, L1, missing, L3 (gap in the middle)": ["L1", "L1", None, "L3"],
        "missing, L1, L1, L1 (first page gone)": [None, "L1", "L1", "L1"],
    }
    for label, pages in cases.items():
        k = reusable_prefix(pages)
        print(f"  {label:44s} -> reusable tokens: {k*256:4d} of 1024")
    print("  a later page cannot be reused if an earlier page is missing: each page's notes were written by reading everything before it")


def pipeline(num_layers, t_transfer, t_compute):
    """Transfers happen one layer at a time on one link; compute of layer i needs its transfer and the previous layer's compute."""
    transfer_done = 0.0
    compute_done = 0.0
    for _ in range(num_layers):
        transfer_done += t_transfer
        compute_done = max(compute_done, transfer_done) + t_compute
    return compute_done


def part3():
    print("== Part 3: layer-by-layer restore overlapped with compute (4 layers) ==")
    L = 4
    for t_tr, t_co in ((3, 5), (6, 2), (4, 4)):
        serial = L * t_tr + L * t_co
        over = pipeline(L, t_tr, t_co)
        print(f"  transfer {t_tr} + compute {t_co} per layer: all-then-compute {serial:3d} | overlapped {over:4.0f} | saved {serial-over:4.0f}")
    print("  overlap hides the smaller of the two; the larger one still sets the pace (and layer 0's transfer can't be hidden)")


def part4():
    print("== Part 4: prefetch and wait policies (recompute the whole prefix would cost 300 ms) ==")
    a_compute, b_read = 120, 100
    print(f"  request A computes for {a_compute} ms while B waits; B's storage read takes {b_read} ms -> exposed wait for B: {max(0, b_read-a_compute)} ms")
    b_read2 = 180
    print(f"  if the read takes {b_read2} ms instead: exposed wait = {max(0, b_read2-a_compute)} ms")
    c = 300
    tau = 30
    print(f"  when B's turn comes, the read still needs r more ms.  Costs (ms of extra delay before B can use the cache vs recompute):")
    print("      r | wait_complete | best_effort | timeout(30)")
    for r in (10, 30, 60, 400):
        wait_complete = r
        best_effort = c
        timeout = r if r <= tau else tau + c
        print(f"    {r:3d} | {wait_complete:13d} | {best_effort:11d} | {timeout:11d}")
    print("  (best_effort here means: give up on the unfinished part and recompute it, cost c = 300. timeout gives up after 30 ms,")
    print("   so for r just above 30 it pays BOTH the wait and the recompute: the worst column.)")


def part5():
    print("== Part 5: the book's exercise 3 ==")
    recompute, host_restore, storage_read = 150, 30, 200
    l2 = host_restore
    l3 = storage_read + host_restore
    print(f"  recompute {recompute} ms | L2 hit: {l2} ms -> {'helps' if l2 < recompute else 'hurts'} | L3 hit: {storage_read} + {host_restore} = {l3} ms -> {'helps' if l3 < recompute else 'hurts'}")


def part6():
    import hashlib

    print("== Part 6: prefix fingerprints (a hash chain over pages; hashlib sha256, truncated to 8 hex digits) ==")

    def chain(pages):
        keys, prev = [], ""
        for pg in pages:
            prev = hashlib.sha256((prev + "|" + " ".join(map(str, pg))).encode()).hexdigest()[:8]
            keys.append(prev)
        return keys

    doc = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    sys_a = [[100, 101, 102, 103]]
    sys_b = [[200, 201, 202, 203]]
    a = chain(sys_a + doc)
    b = chain(sys_b + doc)
    c = chain(sys_a + doc)
    print(f"  system prompt A + document : {a}")
    print(f"  system prompt B + document : {b}")
    print(f"  system prompt A + document, other instance: {c}")
    print(f"  same document pages, different system prompt -> keys equal? {a[1:] == b[1:]}   (no reuse: the context differs)")
    print(f"  same prefix computed on another instance     -> keys equal? {a == c}   (shared storage can serve it)")


part1()
part2()
part3()
part4()
part5()
part6()
