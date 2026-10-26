"""Toy scheduler for Part II chapter 6 (no GPU, standard library only).

Run:    python3 -I eli5/tools/ch6_scheduler_toy.py

It copies the *decisions* of mini-sglang's scheduler, not its code:
  * prefill first, otherwise decode            (Scheduler._schedule_next_batch)
  * a limited number of request slots          (TableManager.available_size)
  * FIFO admission that STOPS at the first request that does not fit (PrefillManager)
  * worst-case KV reservation                  (PrefillAdder._try_allocate_one / _add_one_req)
  * chunked prefill by token budget            (PrefillAdder._add_one_req)
Every step costs "1 unit" here; real prefill and decode steps do not cost the same.

  Part 1  the book's A/B/C trace (section 5.2), reproduced
  Part 2  static vs continuous batching on one workload
  Part 3  the KV admission arithmetic, with hand numbers
  Part 4  chunked prefill: a 20-token prompt with a budget of 8
"""
from dataclasses import dataclass, field


@dataclass
class R:
    name: str
    arrival: int  # first step at which the front desk can see it
    out_len: int  # tokens it will generate (no early EOS)
    made: int = 0
    first_token_step: int | None = None
    finish_step: int | None = None


def run(requests, slots, policy, verbose=False):
    reqs = [R(r.name, r.arrival, r.out_len) for r in requests]
    waiting, running = [], []
    step = 0
    log = []
    batch_open = True  # static policy only: may we start a new batch?
    while any(r.finish_step is None for r in reqs):
        for r in reqs:
            if r.arrival == step:
                waiting.append(r)
        kind, members = None, []
        if policy == "continuous":
            free = slots - len(running)
            if waiting and free > 0:
                kind = "prefill"
                members = waiting[:free]  # FIFO; the real adder also stops at the first misfit
                waiting = waiting[free:]
                running += members
            elif running:
                kind, members = "decode", list(running)
        else:  # static: a new batch only starts when the previous one is completely done
            if not running and waiting:
                kind = "prefill"
                members = waiting[:slots]
                waiting = waiting[slots:]
                running += members
            elif running:
                kind, members = "decode", list(running)
        if kind is None:
            step += 1
            continue
        for r in members:
            r.made += 1
            if r.first_token_step is None:
                r.first_token_step = step
        done = [r for r in members if r.made == r.out_len]
        for r in done:
            r.finish_step = step
            running.remove(r)
        if verbose:
            log.append((step, kind, [f"{r.name}({r.made})" for r in members], [r.name for r in done]))
        step += 1
    return reqs, step, log


def part1():
    print("== Part 1: the book's A/B/C example (2 slots; A,B,C need 2,5,2 tokens; C arrives after step 0) ==")
    reqs = [R("A", 0, 2), R("B", 0, 5), R("C", 1, 2)]
    _, total, log = run(reqs, slots=2, policy="continuous", verbose=True)
    print("  step | batch   | tokens so far (after this step) | finished")
    for step, kind, members, done in log:
        print(f"  {step:4d} | {kind:7s} | {', '.join(members):31s} | {', '.join(done) or '-'}")
    print(f"  total steps: {total}")


def part2():
    print("== Part 2: 8 requests arrive together, 4 slots, output lengths 2 5 2 30 3 4 2 8 ==")
    lens = [2, 5, 2, 30, 3, 4, 2, 8]
    reqs = [R(f"r{i}", 0, n) for i, n in enumerate(lens)]
    useful = sum(lens)
    for policy in ("static", "continuous"):
        done, total, _ = run(reqs, slots=4, policy=policy)
        ttft = [r.first_token_step + 1 for r in done]  # steps until the first token is visible
        fin = [r.finish_step + 1 for r in done]
        slot_steps = 4 * total
        print(
            f"  {policy:10s}: {total:3d} steps | mean steps to first token {sum(ttft)/len(ttft):5.2f}"
            f" | mean steps to finish {sum(fin)/len(fin):5.2f} | useful slot-steps {useful}/{slot_steps}"
            f" = {100*useful/slot_steps:4.1f}%"
        )


def admit(queue, capacity_free, reserved, slots_free, label):
    """Mirror of PrefillAdder._try_allocate_one + _add_one_req for page_size 1, no prefix hits."""
    print(f"  {label}: free KV = {capacity_free}, reserved = {reserved}, free slots = {slots_free}")
    admitted = []
    for i, (name, input_len, output_len) in enumerate(queue):
        est = input_len + output_len  # extend_len (= input_len, nothing cached) + output_len
        if slots_free == 0:
            print(f"    {name}: no free slot -> stop")
            break
        if est + reserved > capacity_free:
            print(f"    {name}: {input_len}+{output_len} = {est}; {est} + reserved {reserved} = {est+reserved}"
                  f" > free {capacity_free} -> REJECT and STOP (FIFO)")
            for n2, i2, o2 in queue[i + 1:]:
                fits = (i2 + o2) + reserved <= capacity_free
                print(f"    {n2}: never asked (it {'WOULD have fit' if fits else 'would not fit either'}:"
                      f" {i2}+{o2} + reserved {reserved} = {i2+o2+reserved} vs free {capacity_free})")
            break
        reserved += input_len + output_len
        slots_free -= 1
        admitted.append(name)
        print(f"    {name}: {input_len}+{output_len} = {est}; fits; reserved is now {reserved}")
    return admitted


def part3():
    print("== Part 3: KV admission arithmetic ==")
    # Situation: 100-token KV pool. One running request R1 holds 30 tokens now and may still
    # produce 40 more. So 70 tokens are free, and 40 of them are already promised to R1.
    admit([("N1", 15, 20)], capacity_free=70, reserved=40, slots_free=3, label="Case A")
    admit([("N1", 15, 10), ("N2", 5, 3)], capacity_free=70, reserved=40, slots_free=3,
          label="Case B (reservation grows inside one pass)")
    admit([("BIG", 50, 20), ("small", 3, 2)], capacity_free=70, reserved=40, slots_free=3,
          label="Case C (head-of-line blocking)")


def part4():
    print("== Part 4: chunked prefill, prompt 20 tokens, token budget 8 per pass ==")
    input_len, budget = 20, 8
    cached = 0
    pass_no = 0
    while True:
        pass_no += 1
        remain = input_len - cached
        chunk = min(budget, remain)
        chunked = chunk < remain
        kind = "ChunkedReq (no token sampled, not added to decode)" if chunked else "Req (last chunk: samples the first token)"
        print(f"  pass {pass_no}: cached_len {cached:2d}, remain {remain:2d}, chunk {chunk} -> {kind}")
        if not chunked:
            break
        cached += chunk


part1()
part2()
part3()
part4()
