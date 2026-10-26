"""Toy radix prefix cache for Part II chapter 8 (standard library only, no torch).

Run:    python3 -I eli5/tools/ch8_radix_toy.py

It mirrors mini-sglang's kvcache/radix_cache.py and the cache_req / allocate / evict logic of
scheduler/cache.py with plain Python lists instead of tensors, and an integer clock instead of
time.monotonic_ns() so the output is deterministic.

  Part 0  why only a PREFIX can be reused (a 2-layer causal attention toy)
  Part 1  the book's A / B / C example: how the tree is built, split by split
  Part 2  the R1 / R2 lifecycle: match, lock, allocate, insert, unlock, evict, with the books balanced
  Part 3  why match_req leaves out the prompt's last token
  Part 4  page_size 1 vs 4: what sharing is lost when pages are bigger
  Part 5  LRU eviction order and the cascade
"""
import heapq
from dataclasses import dataclass, field


def align_down(n, a):
    return n // a * a


class Node:
    counter = 0

    def __init__(self, tree, tic=0):
        self.tree = tree
        self.key, self.value = [], []
        self.children = {}
        self.parent = None
        self.ref = 0
        self.uuid = Node.counter
        Node.counter += 1
        self.ts = tic

    @property
    def length(self):
        return len(self.key)

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return not self.children

    def set_kv(self, k, v):
        assert len(k) == len(v)
        self.key, self.value = list(k), list(v)

    def set_parent(self, p):
        self.parent = p
        p.children[self.tree.keyfn(self.key)] = self  # replaces an existing entry with the same first page

    def match_len(self, ids):
        n = 0
        for a, b in zip(self.key, ids):
            if a != b:
                break
            n += 1
        return n

    def split_at(self, pos):
        assert 0 < pos < self.length
        parent = self.parent
        new = Node(self.tree, self.ts)
        new.set_kv(self.key[:pos], self.value[:pos])
        new.set_parent(parent)
        new.ref = self.ref
        self.set_kv(self.key[pos:], self.value[pos:])
        self.set_parent(new)
        return new

    def __lt__(self, other):
        return (self.ts, self.uuid) < (other.ts, other.uuid)


@dataclass(frozen=True)
class Handle:
    cached_len: int
    node: Node

    def indices(self):
        out, n = [], self.node
        while not n.is_root():
            out.append(n.value)
            n = n.parent
        return [x for seg in reversed(out) for x in seg]


class Tree:
    def __init__(self, page_size=1):
        self.page = page_size
        self.clock = 0
        self.root = Node(self)
        self.root.ref = 1
        self.evictable = 0
        self.protected = 0

    def keyfn(self, ids):
        return ids[0] if self.page == 1 else tuple(ids[: self.page])

    def walk(self, ids):
        self.clock += 1
        tic = self.clock
        prefix, node = 0, self.root
        while prefix < len(ids):
            child = node.children.get(self.keyfn(ids[prefix:]))
            if child is None:
                return node, prefix
            node = child
            m = align_down(node.match_len(ids[prefix:]), self.page)
            prefix += m
            if m != node.length:
                node = node.split_at(m)
                node.ts = tic
                return node, prefix
            node.ts = tic
        return node, prefix

    def match(self, ids):
        node, n = self.walk(ids)
        return Handle(n, node)

    def insert(self, ids, indices):
        n_ins = align_down(len(ids), self.page)
        ids, indices = ids[:n_ins], indices[:n_ins]
        node, prefix = self.walk(ids)
        if prefix != n_ins:
            new = Node(self)
            new.ts = self.clock
            new.set_kv(ids[prefix:], indices[prefix:])
            new.set_parent(node)
            self.evictable += new.length
            node = new
        return prefix, Handle(n_ins, node)

    def lock(self, h, unlock=False):
        n = h.node
        if unlock:
            while not n.is_root():
                n.ref -= 1
                assert n.ref >= 0
                if n.ref == 0:
                    self.evictable += n.length
                    self.protected -= n.length
                n = n.parent
        else:
            while not n.is_root():
                if n.ref == 0:
                    self.evictable -= n.length
                    self.protected += n.length
                n.ref += 1
                n = n.parent

    def evict(self, size):
        if size == 0:
            return []
        assert size <= self.evictable, f"cannot evict {size}, only {self.evictable}"
        leaves, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if n.ref == 0 and not n.is_root():
                    leaves.append(n)
            else:
                stack.extend(n.children.values())
        heapq.heapify(leaves)
        got, out, order = 0, [], []
        while got < size:
            n = heapq.heappop(leaves)
            got += n.length
            out += n.value
            order.append((n.key, n.value))
            self.evictable -= n.length
            p = n.parent
            del p.children[self.keyfn(n.key)]
            if p.is_leaf() and p.ref == 0 and not p.is_root():
                heapq.heappush(leaves, p)
        self.last_evicted = order
        return out

    def show(self):
        lines = []

        def rec(n, depth):
            for c in sorted(n.children.values(), key=lambda x: x.uuid):
                lines.append("    " + "   " * depth + f"└─ key {c.key} slots {c.value}  (ref {c.ref}, t {c.ts})")
                rec(c, depth + 1)

        rec(self.root, 0)
        return "\n".join(lines) if lines else "    (empty)"


# ----------------------------------------------------------------------------------------------
def part0():
    import math
    import random

    print("== Part 0: why only a prefix can be reused (2 layers, 1 head, dim 4, random fixed weights) ==")
    rng = random.Random(0)
    D = 4

    def mat():
        return [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(D)]

    def mv(m, x):
        return [sum(a * b for a, b in zip(row, x)) for row in m]

    Wq = [mat(), mat()]
    Wk = [mat(), mat()]
    Wv = [mat(), mat()]
    Wo = [mat(), mat()]
    emb = {w: [rng.uniform(-1, 1) for _ in range(D)] for w in ("weather", "mood", "is", "fine", "bad")}
    pos = [[rng.uniform(-0.5, 0.5) for _ in range(D)] for _ in range(8)]

    def run(words):
        x = [[e + p for e, p in zip(emb[w], pos[i])] for i, w in enumerate(words)]
        kv = []
        for layer in range(2):
            ks = [mv(Wk[layer], xi) for xi in x]
            vs = [mv(Wv[layer], xi) for xi in x]
            kv.append((ks, vs))
            out = []
            for i, xi in enumerate(x):
                q = mv(Wq[layer], xi)
                sc = [sum(a * b for a, b in zip(q, ks[j])) / math.sqrt(D) for j in range(i + 1)]  # causal
                mx = max(sc)
                ex = [math.exp(v - mx) for v in sc]
                z = sum(ex)
                att = [sum(ex[j] / z * vs[j][d] for j in range(i + 1)) for d in range(D)]
                o = mv(Wo[layer], att)
                out.append([a + b for a, b in zip(xi, o)])
            x = out
        return kv

    def same(a, b, layer, i):
        return all(abs(p - q) < 1e-12 for p, q in zip(a[layer][0][i], b[layer][0][i])) and all(
            abs(p - q) < 1e-12 for p, q in zip(a[layer][1][i], b[layer][1][i])
        )

    A = run(["weather", "is", "fine"])
    B = run(["mood", "is", "fine"])        # same suffix, different left context
    C = run(["weather", "is", "bad"])      # same prefix, different last word
    print("   position of 'is' = 1, 'fine' = 2.  Is the K/V pair IDENTICAL?")
    print("   compared pair                         | layer 1 | layer 2")
    print(f"   'is'   in [weather is fine] vs [mood is fine]  | {str(same(A, B, 0, 1)):7s} | {str(same(A, B, 1, 1))}")
    print(f"   'fine' in [weather is fine] vs [mood is fine]  | {str(same(A, B, 0, 2)):7s} | {str(same(A, B, 1, 2))}")
    print(f"   'weather' (pos 0) in A vs C (same prefix)       | {str(same(A, C, 0, 0)):7s} | {str(same(A, C, 1, 0))}")
    print(f"   'is'      (pos 1) in A vs C (same prefix)       | {str(same(A, C, 0, 1)):7s} | {str(same(A, C, 1, 1))}")
    print(f"   'fine' (pos 2) vs 'bad' (pos 2), same prefix    | {str(same(A, C, 0, 2)):7s} | {str(same(A, C, 1, 2))}  (different token, expected to differ)")


def part1():
    print("== Part 1: the book's A, B, C ==")
    t = Tree()
    A = [101, 11, 12, 21, 22]
    B = [101, 11, 12, 31, 32]
    C = [101, 11, 99]
    cached, _ = t.insert(A, ["p0", "p1", "p2", "p3", "p4"])
    print(f"  after A (already in tree: {cached}):\n{t.show()}")
    cached, _ = t.insert(B, ["q0", "q1", "q2", "q3", "q4"])
    print(f"  after B (already in tree: {cached} -> B's own q0..q2 are duplicates and would be freed):\n{t.show()}")
    cached, _ = t.insert(C, ["r0", "r1", "r2"])
    print(f"  after C (already in tree: {cached}):\n{t.show()}")


# ---- a tiny stand-in for CacheManager -------------------------------------------------------
class Pool:
    def __init__(self, tree, free):
        self.tree, self.free = tree, list(free)

    def allocate(self, n):
        if n > len(self.free):
            ev = self.tree.evict(n - len(self.free))
            print(f"      evict({n - len(self.free)}) removed whole node(s) {self.tree.last_evicted} -> {len(ev)} slots back to the free list")
            self.free += ev
        got, self.free = self.free[:n], self.free[n:]
        return got

    def available(self):
        return self.tree.evictable + len(self.free)


@dataclass
class Req:
    name: str
    ids: list
    slots: list = field(default_factory=list)
    cached_len: int = 0
    handle: Handle | None = None


def state(tree, pool, tag, private=0):
    inside = tree.evictable + tree.protected
    ok = len(pool.free) + inside + private == 10
    print(f"  {tag:58s} evictable {tree.evictable} | protected {tree.protected} | free {pool.free} | books balance: {ok}")


def cache_req(tree, pool, req, finished):
    ids = req.ids[: req.cached_len]
    pages = req.slots[: req.cached_len]
    old = req.handle
    cached, new = tree.insert(ids, pages)
    tree.lock(old, unlock=True)
    dup = pages[old.cached_len: cached]
    pool.free += dup
    tail = []
    if finished:
        tail = pages[new.cached_len:]
        pool.free += tail
    else:
        req.handle = new
        tree.lock(new)
    return cached, dup, tail


def part2():
    print("== Part 2: R1 / R2 lifecycle (pool of 10 slots; slots 4,5 belong to an unrelated running order X) ==")
    t = Tree()
    S = [900, 10, 11, 12]
    t.insert(S, [0, 1, 2, 3])
    pool = Pool(t, [6, 7, 8, 9])
    state(t, pool, "start: S in the tree, nobody using it", private=2)

    # R1
    r1 = Req("R1", [900, 10, 11, 12, 31, 32])
    h = t.match(r1.ids[:-1])  # match_req leaves out the last token
    print(f"  R1 match: cached_len {h.cached_len}, slots {h.indices()}")
    t.lock(h)
    r1.handle, r1.cached_len, r1.slots = h, h.cached_len, h.indices()
    state(t, pool, "R1 locked its match (S is protected)", private=2)
    est = 6 - h.cached_len
    print(f"  R1 admission: need {est}; available = evictable {t.evictable} + free {len(pool.free)} = {pool.available()}")
    r1.slots += pool.allocate(est)
    r1.cached_len = 6  # after the prefill pass: complete_one() moved cached_len to device_len
    state(t, pool, "R1 allocated 2 slots and ran prefill", private=2 + 2)
    c, dup, tail = cache_req(t, pool, r1, finished=False)
    state(t, pool, f"R1 prefill done: cache_req(finished=False), tree already had {c}", private=2)
    print(t.show())
    c, dup, tail = cache_req(t, pool, r1, finished=True)
    state(t, pool, f"R1 finished: cache_req(finished=True), freed dup {dup} tail {tail}", private=2)

    # R2
    r2 = Req("R2", [900, 10, 11, 12, 41, 42, 43])
    h = t.match(r2.ids[:-1])
    print(f"  R2 match: cached_len {h.cached_len}, slots {h.indices()}")
    t.lock(h)
    r2.handle, r2.cached_len, r2.slots = h, h.cached_len, h.indices()
    est = 7 - h.cached_len
    print(f"  R2 admission: need {est}; available = evictable {t.evictable} + free {len(pool.free)} = {pool.available()}")
    state(t, pool, "R2 locked S (R1's old leaf is still evictable)", private=2)
    r2.slots += pool.allocate(est)
    state(t, pool, "R2 allocated 3 slots (one eviction was needed)", private=2 + 3)
    r2.cached_len = 7
    cache_req(t, pool, r2, finished=False)
    state(t, pool, "R2 prefill done: cache_req(finished=False)", private=2)
    print(t.show())


def part3():
    print("== Part 3: why match_req drops the last prompt token ==")
    t = Tree()
    t.insert([900, 10, 11, 12], [0, 1, 2, 3])
    prompt = [900, 10, 11, 12]
    h_full = t.match(prompt)
    print(f"  match on the WHOLE prompt: cached_len {h_full.cached_len} -> extend_len = {len(prompt) - h_full.cached_len}: nothing would go through the model, so no logits, no first word")
    t2 = Tree()
    t2.insert([900, 10, 11, 12], [0, 1, 2, 3])
    print(f"  tree before the real match:\n{t2.show()}")
    h = t2.match(prompt[:-1])
    print(f"  match on prompt[:-1]: cached_len {h.cached_len} -> extend_len = {len(prompt) - h.cached_len}: one token is computed")
    print(f"  tree after the match (match_prefix SPLIT the node; the docstring says it does not modify the cache):\n{t2.show()}")


def part4():
    print("== Part 4: page_size 1 vs 4 ==")
    A = [1, 2, 3, 4, 5, 6, 7, 8]
    B = [1, 2, 3, 9, 10, 11, 12, 13]
    C = [1, 2, 3, 4, 5, 6, 7, 99]
    for ps in (1, 4):
        t = Tree(ps)
        t.insert(A, list(range(8)))
        hb = t.match(B)
        hc = t.match(C)
        print(f"  page_size {ps}: B shares {hb.cached_len} tokens with A, C shares {hc.cached_len}")


def part5():
    print("== Part 5: LRU eviction order and cascade ==")
    t = Tree()
    t.insert([1, 2, 3, 4], [0, 1, 2, 3])  # X
    t.insert([1, 2, 5, 6], [4, 5, 6, 7])  # Y   (splits X at 2)
    t.insert([1, 2, 7], [8, 9, 10])       # Z   (new leaf under [1,2])
    print(f"  tree:\n{t.show()}")
    t.match([1, 2, 3, 4])  # touch X's path: it becomes the most recently used
    print(f"  after touching X's path:\n{t.show()}")
    out = t.evict(3)
    print(f"  evict(3) returned {out}; nodes removed in order: {t.last_evicted}")
    print(f"  tree now:\n{t.show()}")
    out = t.evict(3)
    print(f"  evict(3) returned {out}; nodes removed in order: {t.last_evicted}")
    print("     ^ asked for 3, got 4: whole nodes go; and the parent [1, 2] became a leaf the moment its last child left, so it joined the queue")
    print(f"  tree now:\n{t.show()}")


part0()
part1()
part2()
part3()
part4()
part5()
