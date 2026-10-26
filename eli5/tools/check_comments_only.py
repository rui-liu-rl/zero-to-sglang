#!/usr/bin/env python3
"""Prove that a branch changed COMMENTS ONLY (no behaviour change).

    python3 -I eli5/tools/check_comments_only.py --repo /Users/rui/code/mini-sglang [--base main]

For every file that differs from <base> (committed, staged, unstaged or untracked):

  .py                          compare ast.dump() of the base version and the current version
                               (comments, blank lines and line numbers are not in the AST, but
                               docstrings, names, literals and every statement are), and
                               byte-compile the current file.
  .c .cc .cpp .h .hpp .cu .cuh strip // and /* */ comments (string / char / raw-string aware),
                               collapse whitespace, compare the remaining text.
  anything else                reported as UNCHECKED: look at it by hand.

Exit status is 0 only when every changed file is OK. Untracked files count as NEW and make the
run fail unless --allow-new is given (a new file is, by definition, not a comment-only change).
"""

from __future__ import annotations

import argparse
import ast
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

C_LIKE = {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh"}


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def base_text(repo: Path, base: str, rel: str) -> str | None:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{base}:{rel}"], capture_output=True)
    return None if r.returncode != 0 else r.stdout.decode("utf-8")


def strip_c_comments(src: str) -> str:
    """Remove // and /* */ comments from C/C++/CUDA source. String, char and raw-string aware."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        nx = src[i + 1] if i + 1 < n else ""
        if c == "/" and nx == "/":  # line comment: drop up to (not including) the newline
            j = src.find("\n", i)
            i = n if j == -1 else j
        elif c == "/" and nx == "*":  # block comment: becomes one space
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
            out.append(" ")
        elif c == "R" and nx == '"' and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            # raw string literal  R"delim( ... )delim"
            paren = src.find("(", i + 2)
            delim = src[i + 2 : paren]
            end = src.find(")" + delim + '"', paren)
            end = n if end == -1 else end + len(delim) + 2
            out.append(src[i:end])
            i = end
        elif c in ('"', "'"):  # ordinary string / char literal
            quote = c
            out.append(c)
            i += 1
            while i < n:
                ch = src[i]
                out.append(ch)
                i += 1
                if ch == "\\" and i < n:
                    out.append(src[i])
                    i += 1
                elif ch == quote:
                    break
        else:
            out.append(c)
            i += 1
    return "".join(out)


def first_difference(a: str, b: str, width: int = 70) -> str:
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    lo = max(0, k - width // 2)
    return f"    base : ...{a[lo:k + width]!r}\n    now  : ...{b[lo:k + width]!r}"


def check_python(old: str, new: str, path: Path) -> str | None:
    try:
        old_dump = ast.dump(ast.parse(old))
        new_dump = ast.dump(ast.parse(new))
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    if old_dump != new_dump:
        return "AST differs from base:\n" + first_difference(old_dump, new_dump)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            py_compile.compile(str(path), cfile=os.path.join(tmp, "x.pyc"), doraise=True)
        except py_compile.PyCompileError as e:
            return f"does not compile: {e}"
    return None


def check_c_like(old: str, new: str) -> str | None:
    a = " ".join(strip_c_comments(old).split())
    b = " ".join(strip_c_comments(new).split())
    return None if a == b else "code differs once comments are stripped:\n" + first_difference(a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--base", default="main")
    ap.add_argument("--allow-new", action="store_true", help="do not fail on untracked files")
    args = ap.parse_args()
    repo: Path = args.repo.resolve()

    changed = [p for p in git(repo, "diff", "--name-only", args.base, "--").splitlines() if p]
    untracked = [p for p in git(repo, "ls-files", "--others", "--exclude-standard").splitlines() if p]

    bad = 0
    total_old = total_new = 0
    for rel in sorted(changed):
        path = repo / rel
        old = base_text(repo, args.base, rel)
        if not path.exists():
            print(f"DELETED    {rel}")
            bad += 1
            continue
        new = path.read_text(encoding="utf-8")
        if old is None:
            print(f"NEW        {rel}")
            bad += 0 if args.allow_new else 1
            continue
        err: str | None
        if path.suffix == ".py":
            err = check_python(old, new, path)
        elif path.suffix in C_LIKE:
            err = check_c_like(old, new)
        else:
            print(f"UNCHECKED  {rel}")
            bad += 1
            continue
        o, n = old.count("\n"), new.count("\n")
        total_old += o
        total_new += n
        if err:
            print(f"BROKEN     {rel}  ({o} -> {n} lines)\n{err}")
            bad += 1
        else:
            print(f"ok         {rel}  ({o} -> {n} lines)")

    for rel in sorted(untracked):
        print(f"NEW        {rel}  (untracked)")
        bad += 0 if args.allow_new else 1

    print(f"\n{len(changed)} changed file(s), {len(untracked)} untracked; "
          f"{total_old} -> {total_new} lines in checked files; {bad} problem(s).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
