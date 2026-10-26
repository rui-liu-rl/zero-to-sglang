"""Toy replay of FrontendManager's ack_map / event_map logic (no GPU, no ZMQ).

Run it on any machine with Python 3.10+, standard library only:

    python3 -I eli5/tools/ch5_frontend_toy.py

Mirrors server/api_server.py: new_user, listen, wait_for_ack (copied logic, not imported).
Used to produce the traces and numbers quoted in the Part II chapter 5 ELI5 notes:

  Test 1  interleaved replies A1 B1 B2 A2 reach the right customers
  Test 2  one wake-up can carry several replies (the Event is a flag, not a counter)
  Test 3  the trailing `del` in wait_for_ack does NOT run if the caller breaks on `finished`
  Test 4  a reply for a forgotten ticket is dropped by the listener
  Test 5  an `async def` that calls a blocking function still freezes the event loop
  Test 6  size of one forgotten ticket (list + Event + 2 dict slots)
  Test 7  the same consumer, but with the tidy-up moved into try/finally: it IS cleaned up
"""
import asyncio
from dataclasses import dataclass, field


@dataclass
class Reply:
    uid: int
    text: str
    finished: bool


@dataclass
class Mgr:
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    uid_counter: int = 0
    ack_map: dict = field(default_factory=dict)
    event_map: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    def new_user(self):
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            r = await self.queue.get()
            if r.uid not in self.ack_map:
                self.log.append(f"  sorter: DROP late reply for uid {r.uid} ({r.text!r})")
                continue
            self.ack_map[r.uid].append(r)
            self.event_map[r.uid].set()

    async def wait_for_ack(self, uid):
        event = self.event_map[uid]
        while True:
            await event.wait()
            event.clear()
            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break
        del self.ack_map[uid]
        del self.event_map[uid]


async def consume(m, uid, name, out, wake_sizes):
    # like the non-streaming path: break on finished
    text = ""
    async for ack in m.wait_for_ack(uid):
        text += ack.text
        out.append((name, ack.text))
        if ack.finished:
            break
    return text


async def test_interleave():
    print("== Test 1: interleaved arrival A1 B1 B2 A2 ==")
    m = Mgr()
    sorter = asyncio.create_task(m.listen())
    a, b = m.new_user(), m.new_user()
    out = []
    ta = asyncio.create_task(consume(m, a, "A", out, []))
    tb = asyncio.create_task(consume(m, b, "B", out, []))
    await asyncio.sleep(0)
    for r in [Reply(a, "A1 ", False), Reply(b, "B1 ", False), Reply(b, "B2", True), Reply(a, "A2", True)]:
        await m.queue.put(r)
        await asyncio.sleep(0.01)
    print("  A got:", await ta)
    print("  B got:", await tb)
    print("  delivery order:", out)
    print("  leftover keys after both broke on finished -> ack_map:", sorted(m.ack_map), "event_map:", sorted(m.event_map))
    sorter.cancel()


async def test_one_wakeup_many_replies():
    print("== Test 2: three replies land before the consumer wakes ==")
    m = Mgr()
    sorter = asyncio.create_task(m.listen())
    a = m.new_user()
    wakes = []

    async def consumer():
        event = m.event_map[a]
        n_wakes = 0
        got = []
        # re-implement wait_for_ack inline so we can count wake-ups
        while True:
            await event.wait()
            event.clear()
            n_wakes += 1
            pending = m.ack_map[a]
            m.ack_map[a] = []
            got.extend(r.text for r in pending)
            wakes.append(len(pending))
            if pending and pending[-1].finished:
                break
        return n_wakes, got

    t = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    # three puts with NO await in between that lets the consumer run: sorter must run between,
    # but the consumer is only scheduled after set(); put all quickly
    for r in [Reply(a, "x", False), Reply(a, "y", False), Reply(a, "z", True)]:
        m.queue.put_nowait(r)
    n, got = await t
    print("  wake-ups:", n, " replies per wake-up:", wakes, " got:", got)
    sorter.cancel()


async def test_generator_cleanup_gap():
    print("== Test 3: does wait_for_ack's trailing `del` run when the caller breaks on finished? ==")
    m = Mgr()
    sorter = asyncio.create_task(m.listen())
    a = m.new_user()
    task = asyncio.create_task(consume(m, a, "A", [], []))
    await asyncio.sleep(0)
    await m.queue.put(Reply(a, "hi", True))
    await task
    await asyncio.sleep(0.05)  # let any async-gen finalizer run
    print("  after finished + break + 50 ms: a in ack_map?", a in m.ack_map, "| a in event_map?", a in m.event_map)

    print("  ...same, but the caller drains the generator to the end (no break):")
    b = m.new_user()

    async def drain():
        async for ack in m.wait_for_ack(b):
            pass

    t2 = asyncio.create_task(drain())
    await asyncio.sleep(0)
    await m.queue.put(Reply(b, "hi", True))
    await t2
    print("  b in ack_map?", b in m.ack_map, "| b in event_map?", b in m.event_map)
    sorter.cancel()


async def test_late_reply_dropped():
    print("== Test 4: reply arrives after the ticket was torn up ==")
    m = Mgr()
    sorter = asyncio.create_task(m.listen())
    a = m.new_user()
    del m.ack_map[a]
    del m.event_map[a]
    await m.queue.put(Reply(a, "late", False))
    await asyncio.sleep(0.01)
    print("\n".join(m.log))
    sorter.cancel()


async def test_blocking_vs_await():
    import time
    print("== Test 5: `async def` is not magic ==")

    async def handler_blocking():
        time.sleep(0.5)  # stands in for blocking_generate(): no await inside

    async def handler_awaiting():
        await asyncio.sleep(0.5)  # stands in for "wait for the backend"

    async def second_customer(t0, stamp):
        stamp.append(round(time.perf_counter() - t0, 2))

    for name, h in (("blocking", handler_blocking), ("awaiting", handler_awaiting)):
        stamp = []
        t0 = time.perf_counter()
        first = asyncio.create_task(h())
        await asyncio.sleep(0)  # let the first handler start
        second = asyncio.create_task(second_customer(t0, stamp))
        await asyncio.gather(first, second)
        print(f"  {name:9s}: second customer was first looked at after {stamp[0]} s")


async def test_leak_size():
    import tracemalloc
    print("== Test 6: size of a forgotten ticket ==")
    ack, ev = {}, {}
    tracemalloc.start()
    base = tracemalloc.take_snapshot()
    n = 10000
    for uid in range(n):
        ack[uid] = []
        ev[uid] = asyncio.Event()
    snap = tracemalloc.take_snapshot()
    total = sum(s.size_diff for s in snap.compare_to(base, "filename"))
    per = total / n
    print(f"  about {per:.0f} bytes per ticket -> {1_000_000 * per / 1e6:,.0f} MB per million requests")
    tracemalloc.stop()


async def test_finally_fix():
    print("== Test 7: tidy-up in try/finally, caller still breaks on finished ==")
    m = Mgr()
    sorter = asyncio.create_task(m.listen())

    async def wait_for_ack_fixed(uid):
        event = m.event_map[uid]
        try:
            while True:
                await event.wait()
                event.clear()
                pending = m.ack_map[uid]
                m.ack_map[uid] = []
                ack = None
                for ack in pending:
                    yield ack
                if ack and ack.finished:
                    break
        finally:
            m.ack_map.pop(uid, None)
            m.event_map.pop(uid, None)

    a = m.new_user()

    async def consumer():
        async for ack in wait_for_ack_fixed(a):
            if ack.finished:
                break

    t = asyncio.create_task(consumer())
    await asyncio.sleep(0)
    await m.queue.put(Reply(a, "hi", True))
    await t
    await asyncio.sleep(0.05)  # let the async-generator finalizer run
    print("  a in ack_map?", a in m.ack_map, "| a in event_map?", a in m.event_map)
    sorter.cancel()


async def main():
    await test_interleave()
    await test_one_wakeup_many_replies()
    await test_generator_cleanup_gap()
    await test_late_reply_dropped()
    await test_blocking_vs_await()
    await test_leak_size()
    await test_finally_fix()


asyncio.run(main())
