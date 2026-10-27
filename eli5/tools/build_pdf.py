#!/usr/bin/env python3
"""Build the print-ready PDF of the ELI5 book.

    make pdf                  # from eli5/  (or `make pdf` from the repo root)
    make pdf PAPER=a4         # A4 instead of US Letter
    python3 tools/build_pdf.py --help

HOW IT WORKS (so you can change it)
  1. Read every companion chapter under eli5/part*/ plus the README (front matter) and
     NOTES-open-questions.md / the toys table (appendices).
  2. Render the Markdown to HTML with markdown-it-py (CommonMark + tables), adding: heading ids,
     "(p. N)" cross-references, callout classes for the bold lead-ins ("Kid version." ...),
     auto-shrunk code blocks so ASCII diagrams fit the page, and every <details> answer opened.
  3. Wrap it in a cover, colophon and table of contents, with CSS from tools/book.css plus generated
     @page rules (paper size, mirrored binding margins, a running header per chapter).
  4. Print it with headless Chrome (already on most machines and on GitHub's runners).
  5. Chrome cannot fill in "page N" in a table of contents or draw bookmarks, so we read where each
     heading landed (PDF named destinations), put the numbers into the HTML, and print again until
     nothing moves. Then pypdf adds the bookmarks and the document properties.

Only two Python packages are needed (requirements-pdf.txt): markdown-it-py and pypdf.
The only other requirement is a Chrome or Chromium binary (set CHROME=/path if it is not found).
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

try:
    from markdown_it import MarkdownIt
    from markdown_it.token import Token
    from pypdf import PdfReader, PdfWriter
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Missing Python dependency ({exc}).\nRun `make deps` (or: pip install -r requirements-pdf.txt).")

HERE = Path(__file__).resolve().parent  # eli5/tools
ROOT = HERE.parent  # eli5/
REPO = ROOT.parent  # repository root

TITLE = "Zero to SGLang"
SUBTITLE = "The ELI5 Companion"
TAGLINE = (
    "Every chapter of the course and every file of mini-sglang, explained as if you were five: "
    "with a kitchen, tiny numbers, and questions to answer before you peek."
)

# (directory, label, name, one-line note)  -- the names follow the course's own README
PARTS = [
    ("part0", "Part 0", "Before you learn", "Ethics, the open-source spirit, and your first SGLang server."),
    ("part1", "Part I", "Foundations", "Concepts only: no code, no GPU."),
    ("part2", "Part II", "Build Your Own Mini SGL", "Follow a request through mini-sglang, one mechanism at a time."),
    ("part3", "Part III", "Advanced Inference Technique", "The real SGLang: attention backends and hierarchical caching."),
]

# name -> (CSS page-size, width pt, height pt)
PAPERS = {"letter": ("8.5in 11in", 612.0, 792.0), "a4": ("210mm 297mm", 595.28, 841.89)}
# margins in pt; "inside" is the binding edge (wider)
MARGIN = {"top": 62.0, "bottom": 66.0, "inside": 68.0, "outside": 50.0}

SERIF = '"Charter", "Iowan Old Style", "Palatino", "Georgia", "Liberation Serif", "DejaVu Serif", serif'
SANS = '"Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans", sans-serif'
MONO = '"Menlo", "SF Mono", "DejaVu Sans Mono", "Liberation Mono", monospace'

MONO_ADVANCE = 0.602  # width of one monospace character, in em
CODE_PT = 8.0  # default code size
CODE_FLOOR_PT = 6.0  # never shrink below this; wrap instead

CALLOUTS = [
    ("Kid version", "kid"),
    ("Where the story lies", "lies"),
    ("Tiny numbers", "nums"),
    ("Pause and predict", "predict"),
    ("Extra drills", "drills"),
]
ALLOWED_TAGS = {
    "details", "summary", "br", "sub", "sup", "kbd", "span", "div", "p", "hr", "em", "strong", "code", "b", "i",
}

# ------------------------------------------------------------------------------------------
# data


@dataclass
class Doc:
    key: str  # unique id, also the HTML id of the chapter
    path: Path  # source file (used to resolve relative links)
    kind: str  # "chapter" | "front" | "appendix"
    part: str | None  # directory name, for chapters
    text: str = ""  # markdown to render
    title: str = ""  # clean title
    eyebrow: str = ""  # small line above the title
    toc_title: str = ""
    run_title: str = ""  # running header (right pages)
    page_name: str = ""  # CSS named page
    html: str = ""
    headings: list[tuple[int, str, str]] = field(default_factory=list)  # (level, text, id)


@dataclass
class Settings:
    paper: str = "letter"
    text_width_pt: float = 494.0
    author: str = ""
    build_note: str = ""
    code_blocks: list[tuple[str, int, float, bool]] = field(default_factory=list)  # (doc, cols, pt, wrapped)


S = Settings()

# ------------------------------------------------------------------------------------------
# small helpers


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def css_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def slugify(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", s.lower())
    return re.sub(r"[\s_]+", "-", s).strip("-")[:56] or "section"


def display_cols(line: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in line.expandtabs(4))


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


# ------------------------------------------------------------------------------------------
# Markdown -> HTML


def sanitize_html(content: str) -> str:
    """Keep a small set of tags; escape anything else (e.g. a stray <think> in prose). Open every <details>."""

    def repl(m: re.Match) -> str:
        name = m.group(2).lower()
        if name not in ALLOWED_TAGS:
            return html.escape(m.group(0))
        if name == "details" and not m.group(1):
            return '<details open class="answer">'
        return m.group(0)

    return re.sub(r"<(/?)([A-Za-z][A-Za-z0-9-]*)\b[^>]*>", repl, content)


def _render_html(self, tokens, idx, options, env):
    return sanitize_html(tokens[idx].content)


def _render_fence(self, tokens, idx, options, env):
    tok = tokens[idx]
    code = tok.content.rstrip("\n")
    lines = code.split("\n")
    cols = max((display_cols(ln) for ln in lines), default=0)
    avail = float(tok.meta.get("avail", S.text_width_pt - 24)) if tok.meta else S.text_width_pt - 24
    size, wrapped = CODE_PT, False
    if cols * MONO_ADVANCE * CODE_PT > avail:
        size = avail / (cols * MONO_ADVANCE)
        if size < CODE_FLOOR_PT:
            size, wrapped = CODE_FLOOR_PT, True
    S.code_blocks.append((str(tok.meta.get("doc", "?")) if tok.meta else "?", cols, size, wrapped))
    cls = "code" + (" wrap" if wrapped else "") + (" long" if len(lines) > 42 else "")
    return f'<pre class="{cls}" style="font-size:{size:.2f}pt"><code>{html.escape(code)}</code></pre>\n'


def make_md() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])
    md.add_render_rule("fence", _render_fence)
    md.add_render_rule("code_block", _render_fence)
    md.add_render_rule("html_block", _render_html)
    md.add_render_rule("html_inline", _render_html)
    return md


def plain_text(inline: Token) -> str:
    out = []
    for c in inline.children or []:
        if c.type in ("text", "code_inline"):
            out.append(c.content)
        elif c.type in ("softbreak", "hardbreak"):
            out.append(" ")
    return "".join(out).strip()


def render_doc(md: MarkdownIt, doc: Doc, index: dict[Path, Doc]) -> str:
    tokens = md.parse(doc.text)

    # drop the file's own H1: the book prints its own chapter header
    raw_title = ""
    for i, t in enumerate(tokens):
        if t.type == "heading_open" and t.tag == "h1":
            raw_title = plain_text(tokens[i + 1])
            del tokens[i : i + 3]
            break
    doc.title = doc.title or raw_title

    seen: dict[str, int] = {}
    in_details = 0
    out: list[Token] = []
    for i, tok in enumerate(tokens):
        if tok.type == "html_block":
            in_details += len(re.findall(r"<details\b", tok.content)) - len(re.findall(r"</details>", tok.content))
            in_details = max(in_details, 0)
        if tok.type in ("fence", "code_block"):
            tok.meta = {
                "doc": doc.key,
                "avail": S.text_width_pt - 22 - 32 * (in_details > 0) - 16 * tok.level,
            }
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            text = plain_text(tokens[i + 1])
            sid = f"{doc.key}-{slugify(text)}"
            seen[sid] = seen.get(sid, 0) + 1
            if seen[sid] > 1:
                sid = f"{sid}-{seen[sid]}"
            tok.attrSet("id", sid)
            doc.headings.append((level, text, sid))
        if tok.type == "paragraph_open" and not tok.hidden:
            nxt = tokens[i + 1]
            # markdown-it puts an empty text token before a leading **bold**; skip it
            kids = [c for c in (nxt.children or []) if not (c.type == "text" and c.content == "")]
            if len(kids) > 1 and kids[0].type == "strong_open" and kids[1].type == "text":
                for prefix, cls in CALLOUTS:
                    if kids[1].content.startswith(prefix):
                        tok.attrSet("class", cls)
                        break
        if tok.type == "inline" and tok.children:
            tok.children = rewrite_links(tok.children, doc, index)
        out.append(tok)

    return md.renderer.render(out, md.options, {})


def rewrite_links(children: list[Token], doc: Doc, index: dict[Path, Doc]) -> list[Token]:
    new: list[Token] = []
    pending_ref: str | None = None
    for c in children:
        if c.type == "link_open":
            href = str(c.attrGet("href") or "")
            if re.match(r"^(https?:|mailto:)", href):
                new.append(c)
                continue
            target = None
            if not href.startswith("#"):
                p = (doc.path.parent / unquote(href.split("#")[0])).resolve()
                target = index.get(p)
            if target is not None:
                c.attrs["href"] = "#" + target.key
                c.attrSet("class", "xref")
                pending_ref = target.key
            else:
                c.attrs.pop("href", None)  # no destination in the printed book: show as plain text
                c.attrSet("class", "nolink")
            new.append(c)
        elif c.type == "link_close" and pending_ref:
            new.append(c)
            ref = Token("html_inline", "", 0)
            ref.content = f'<span class="pageref">&nbsp;(p.&nbsp;@@PG:{pending_ref}@@)</span>'
            new.append(ref)
            pending_ref = None
        else:
            new.append(c)
    return new


# ------------------------------------------------------------------------------------------
# discovering the content


def parse_chapter_title(raw: str, part_label: str) -> tuple[str, str, str, str]:
    """-> (clean title, eyebrow, toc title, running title)"""
    t = re.sub(r":\s*ELI5 companion\s*$", "", raw).strip()
    code = bool(re.search(r"\(Code Walkthrough\)\s*$", t))
    t = re.sub(r"\s*\(Code Walkthrough\)\s*$", "", t)
    m = re.match(r"^(?:Part\s+(?:0\.(\d+)|I{1,3}|IV)\s+)?(?:Chapter\s+(\d+)\s+)?(.*)$", t)
    lesson, num, name = m.group(1), m.group(2), m.group(3)
    if lesson:
        eyebrow = f"{part_label} · Lesson {lesson}"
        short = f"0.{lesson}"
    else:
        eyebrow = f"{part_label} · Chapter {num}"
        short = f"{num}"
    if code:
        eyebrow += " · Code walkthrough"
    toc = f"{short}   {name}" + ("  (code walkthrough)" if code else "")
    run = f"{part_label} · {'Lesson ' + lesson if lesson else 'Chapter ' + str(num)}{' (code)' if code else ''} · {name}"
    if len(run) > 74:
        run = run[:73].rstrip() + "…"
    return name, eyebrow, toc, run


def split_h2(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Split markdown into (intro, [(h2 heading, body), ...]) ignoring '## ' inside code fences."""
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    fence = False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
        if not fence and line.startswith("## "):
            sections.append((line[3:].strip(), []))
            continue
        (sections[-1][1] if sections else intro).append(line)
    return "\n".join(intro), [(h, "\n".join(b)) for h, b in sections]


def discover() -> tuple[list[Doc], dict[Path, Doc], list[Doc], list[Doc]]:
    docs: list[Doc] = []
    index: dict[Path, Doc] = {}
    n = 0

    # chapters
    chapters: list[Doc] = []
    for pdir, label, _name, _note in PARTS:
        for f in sorted((ROOT / pdir).glob("*.md"), key=natural_key):
            n += 1
            first = f.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
            title, eyebrow, toc, run = parse_chapter_title(first, label)
            d = Doc(
                key=f"{pdir}-{slugify(f.stem)}",
                path=f.resolve(),
                kind="chapter",
                part=pdir,
                text=f.read_text(encoding="utf-8"),
                title=title,
                eyebrow=eyebrow,
                toc_title=toc,
                run_title=run,
                page_name=f"c{n}",
            )
            chapters.append(d)
            index[d.path] = d

    # front matter and appendices from README.md and NOTES-open-questions.md
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    intro, sections = split_h2(readme)
    intro = re.sub(r"^# .*\n", "", intro, count=1)
    about_parts = [intro]
    toys = ""
    for heading, body in sections:
        if heading.startswith(("Progress ledger", "Building the PDF")):
            continue  # status tracking and build instructions, not book content
        if heading.startswith("Runnable toys"):
            toys = body
            continue
        about_parts.append(f"## {heading}\n{body}")
    about = Doc(
        key="about", path=(ROOT / "README.md").resolve(), kind="front", part=None,
        text="\n".join(about_parts), title="About this book", eyebrow="Before you start",
        toc_title="About this book", run_title="About this book", page_name="cabout",
    )
    index[about.path] = about
    notes = Doc(
        key="appendix-a", path=(ROOT / "NOTES-open-questions.md").resolve(), kind="appendix", part=None,
        text=(ROOT / "NOTES-open-questions.md").read_text(encoding="utf-8"),
        title="Open questions and suspected upstream quirks", eyebrow="Appendix A",
        toc_title="A   Open questions and suspected upstream quirks",
        run_title="Appendix A · Open questions and suspected quirks", page_name="capxa",
    )
    index[notes.path] = notes
    toys_doc = Doc(
        key="appendix-b", path=(ROOT / "tools" / "_toys.md").resolve(), kind="appendix", part=None,
        text=toys, title="Runnable toys", eyebrow="Appendix B",
        toc_title="B   Runnable toys", run_title="Appendix B · Runnable toys", page_name="capxb",
    )
    docs = [about, *chapters, notes, toys_doc]
    return docs, index, chapters, [notes, toys_doc]


# ------------------------------------------------------------------------------------------
# page assembly


def page_rules(docs: list[Doc]) -> str:
    size, _w, page_h = PAPERS[S.paper]
    t, b, i, o = (MARGIN[k] for k in ("top", "bottom", "inside", "outside"))
    hdr = f"font: 7.4pt {SANS}; color: #6b7078; letter-spacing: 0.03em;"
    ftr = f"font: 8.6pt {SANS}; color: #444;"
    book = css_str(f"{TITLE} · {SUBTITLE}")
    r = [
        f"@page {{ size: {size}; margin: {t}pt {o}pt {b}pt {i}pt; }}",
        # left page (verso): binding edge is on the right
        f"@page :left {{ margin: {t}pt {i}pt {b}pt {o}pt; "
        f"@top-left {{ content: {book}; text-align: left; {hdr} }} "
        f"@bottom-left {{ content: counter(page); text-align: left; {ftr} }} }}",
        # right page (recto): binding edge is on the left
        f"@page :right {{ margin: {t}pt {o}pt {b}pt {i}pt; "
        f"@bottom-right {{ content: counter(page); text-align: right; {ftr} }} }}",
        "@page cover { margin: 0; }",
        f"section.cover {{ height: {page_h - 0.5:.1f}pt; }}",
        f"section.part-divider {{ height: {page_h - t - b - 2:.1f}pt; }}",
    ]
    for name in ("cover", "colophon", "divider"):
        r.append(
            f"@page {name}:left {{ @top-left {{ content: none; }} @bottom-left {{ content: none; }} }} "
            f"@page {name}:right {{ @top-right {{ content: none; }} @bottom-right {{ content: none; }} }}"
        )
    r.append(f'@page toc:right {{ @top-right {{ content: "Contents"; text-align: right; {hdr} }} }}')
    for d in docs:
        r.append(f"@page {d.page_name}:right {{ @top-right {{ content: {css_str(d.run_title)}; text-align: right; {hdr} }} }}")
    return "\n".join(r)


def stylesheet(docs: list[Doc]) -> str:
    css = (HERE / "book.css").read_text(encoding="utf-8")
    css = css.replace("%%SERIF%%", SERIF).replace("%%SANS%%", SANS).replace("%%MONO%%", MONO)
    return css + "\n" + page_rules(docs)


def cover_html() -> str:
    art = (
        "customer ─▶ front desk ─▶ card-maker ─▶ head waiter ─▶ chef\n"
        "                                                      │\n"
        "                                notepad (KV cache) ◀──┘"
    )
    return f"""<section class="cover">
<div class="band"><div class="kicker">Study notes</div>
<h1>{esc(TITLE)}</h1><div class="sub">{esc(SUBTITLE)}</div>
<div class="tag">{esc(TAGLINE)}</div></div>
<div class="art">{esc(art)}</div>
<div class="foot">Companion to the course <b>zero-to-sglang</b> and the code <b>mini-sglang</b>.<br>{esc(S.build_note)}</div>
</section>"""


def colophon_html() -> str:
    return f"""<section class="colophon">
<h2>About this PDF</h2>
<p>These are personal study notes. They are not an official part of the course or of the code they explain.
They follow the English edition of <b>zero-to-sglang</b> (Datawhale) chapter by chapter, and the commented
copy of <b>mini-sglang</b> (sgl-project), using one running story: a restaurant kitchen.</p>
<p><b>Licences.</b> The course text is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
(creativecommons.org/licenses/by-nc-sa/4.0). These notes are an adaptation of it, so if you share them, keep the same
terms and credit the original. mini-sglang has its own licence; see its repository.</p>
<p><b>Printing.</b> Print double-sided, flipping on the long edge, at <b>100% / actual size</b> (not "fit to page").
The margin on the binding side is wider on purpose. Paper: {esc(S.paper.upper())}.
Every "Answer" and "Hint" box is shown, dashed: cover it with your hand and predict first.</p>
<p><b>Re-making it.</b> <code>make pdf</code> in the repository (<code>make pdf PAPER=a4</code> for A4). The same build
runs on GitHub (see <code>.github/workflows/eli5-pdf.yml</code>).</p>
<p class="fine">{esc(S.build_note)}<br>Numbers in these notes were computed with the small programs listed in Appendix B.
Nothing here was run on a GPU.</p>
</section>"""


def part_divider_html(pdir: str, label: str, name: str, note: str, chapters: list[Doc]) -> str:
    items = "".join(f"<li>{esc(c.toc_title)}</li>" for c in chapters if c.part == pdir)
    return f"""<section class="part-divider" id="part-{pdir}">
<div class="label">{esc(label)}</div><div class="title">{esc(name)}</div>
<div class="note">{esc(note)}</div><ul>{items}</ul></section>"""


def chapter_html(d: Doc) -> str:
    return (
        f'<section class="chapter" id="{d.key}" style="page: {d.page_name}">\n'
        f'<header><div class="eyebrow">{esc(d.eyebrow)}</div><h1>{esc(d.title)}</h1></header>\n{d.html}\n</section>'
    )


def toc_row(cls: str, title: str, target: str) -> str:
    return (
        f'<div class="{cls}"><a class="row" href="#{target}"><span class="t">{esc(title)}</span>'
        f'<span class="dots"></span><span class="n">@@PG:{target}@@</span></a></div>'
    )


def build_toc(docs: list[Doc]) -> tuple[str, list[tuple[int, str, str]]]:
    """-> (toc html, bookmark entries [(level, title, id)])"""
    rows: list[str] = []
    marks: list[tuple[int, str, str]] = [(1, "Contents", "toc")]
    by_key = {d.key: d for d in docs}

    about = by_key["about"]
    rows.append(toc_row("toc-front", about.toc_title, about.key))
    marks.append((1, about.toc_title, about.key))
    for _l, text, sid in about.headings:
        marks.append((2, text, sid))

    chapters = [d for d in docs if d.kind == "chapter"]
    for pdir, label, name, _note in PARTS:
        pid = f"part-{pdir}"
        rows.append(toc_row("toc-part", f"{label} — {name}", pid))
        marks.append((1, f"{label} — {name}", pid))
        for c in (c for c in chapters if c.part == pdir):
            rows.append(toc_row("toc-ch", c.toc_title, c.key))
            marks.append((2, f"{c.eyebrow.split(' · ', 1)[1]}: {c.title}", c.key))
            for lvl, text, sid in c.headings:
                if lvl == 2:
                    rows.append(toc_row("toc-sec", text, sid))
                marks.append((lvl + 1, text, sid))
    for d in docs:
        if d.kind == "appendix":
            rows.append(toc_row("toc-front", d.toc_title, d.key))
            marks.append((1, f"{d.eyebrow}: {d.title}", d.key))
            for lvl, text, sid in d.headings:
                marks.append((lvl + 1, text, sid))
    toc = '<section class="toc" id="toc"><h1>Contents</h1>\n' + "\n".join(rows) + "\n</section>"
    return toc, marks


def assemble(docs: list[Doc], toc: str, marks: list[tuple[int, str, str]], pages: dict[str, int]) -> str:
    chapters = [d for d in docs if d.kind == "chapter"]
    body = [cover_html(), colophon_html(), toc, chapter_html(next(d for d in docs if d.kind == "front"))]
    for pdir, label, name, note in PARTS:
        body.append(part_divider_html(pdir, label, name, note, chapters))
        body.extend(chapter_html(c) for c in chapters if c.part == pdir)
    body.extend(chapter_html(d) for d in docs if d.kind == "appendix")
    # invisible links so that every heading is exported as a PDF destination (for deep bookmarks)
    anchors = "".join(f'<a href="#{sid}">.</a>' for _lvl, _t, sid in marks)
    body.append(f'<div class="dest-index" aria-hidden="true">{anchors}</div>')
    doc_html = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(TITLE)} — {esc(SUBTITLE)}</title>'
        f"<style>\n{stylesheet(docs)}\n</style></head><body>\n" + "\n".join(body) + "\n</body></html>"
    )
    return re.sub(r"@@PG:([^@]+)@@", lambda m: str(pages.get(m.group(1), "000")), doc_html)


# ------------------------------------------------------------------------------------------
# Chrome and PDF post-processing


def find_chrome(explicit: str | None) -> str:
    cands = [explicit, os.environ.get("CHROME"), os.environ.get("CHROME_BIN")]
    cands += ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    cands += [shutil.which(n) for n in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")]
    for c in cands:
        if c and Path(c).exists():
            return c
    sys.exit("No Chrome/Chromium found. Install one, or set CHROME=/path/to/chrome.")


def run_chrome(chrome: str, html_path: Path, pdf_path: Path) -> None:
    profile = tempfile.mkdtemp(prefix="eli5-chrome-")
    cmd = [
        chrome, "--headless=new", "--disable-gpu", "--disable-crash-reporter", "--no-first-run",
        "--no-default-browser-check", f"--user-data-dir={profile}", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", *shlex.split(os.environ.get("CHROME_EXTRA_FLAGS", "")), html_path.as_uri(),
    ]
    try:
        if pdf_path.exists():
            pdf_path.unlink()
        subprocess.run(cmd, check=True, timeout=1500, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"Chrome failed ({exc.returncode}):\n{exc.stderr.decode(errors='replace')[-1500:]}")
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        sys.exit("Chrome produced no PDF. Re-run with CHROME_EXTRA_FLAGS=--no-sandbox if you are in a container.")


def read_pages(pdf_path: Path) -> dict[str, int]:
    reader = PdfReader(str(pdf_path))
    return {
        str(name).lstrip("/"): reader.get_destination_page_number(dest) + 1
        for name, dest in reader.named_destinations.items()
    }


def finish_pdf(raw: Path, out: Path, marks: list[tuple[int, str, str]], pages: dict[str, int], meta: dict[str, str]) -> int:
    reader = PdfReader(str(raw))
    writer = PdfWriter(clone_from=reader)
    parents: dict[int, object] = {}
    kept = 0
    for level, title, sid in marks:
        pg = pages.get(sid)
        if pg is None:
            continue
        parent = next((parents[l] for l in range(level - 1, 0, -1) if l in parents), None)
        item = writer.add_outline_item(title, pg - 1, parent=parent)
        parents[level] = item
        for deeper in [l for l in parents if l > level]:
            del parents[deeper]
        kept += 1
    writer.add_metadata(meta)
    writer.page_mode = "/UseOutlines"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        writer.write(fh)
    return kept


# ------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the print-ready PDF of the ELI5 book.")
    ap.add_argument("--paper", choices=sorted(PAPERS), default=os.environ.get("PAPER", "letter").lower())
    ap.add_argument("--out", type=Path, default=None, help="output PDF (default build/zero-to-sglang-eli5-<paper>.pdf)")
    ap.add_argument("--chrome", default=None, help="path to Chrome/Chromium (or set CHROME)")
    ap.add_argument("--author", default=None)
    ap.add_argument("--html-only", action="store_true", help="write build/book.html and stop (for debugging)")
    ap.add_argument("--max-passes", type=int, default=5)
    args = ap.parse_args()

    S.paper = args.paper
    S.text_width_pt = PAPERS[S.paper][1] - MARGIN["inside"] - MARGIN["outside"]
    sha, date = git("rev-parse", "--short", "HEAD"), git("log", "-1", "--format=%cs")
    dirty = " + uncommitted changes" if git("status", "--porcelain", "--", "eli5") else ""
    S.author = args.author or git("config", "user.name") or "ELI5 study notes"
    S.build_note = f"Built {dt.date.today().isoformat()} from commit {sha or 'unknown'}{' (' + date + ')' if date else ''}{dirty}"
    out = args.out or (ROOT / "build" / f"zero-to-sglang-eli5-{S.paper}.pdf")
    build = ROOT / "build"
    build.mkdir(parents=True, exist_ok=True)

    docs, index, chapters, _appx = discover()
    md = make_md()
    for d in docs:
        d.html = render_doc(md, d, index)
    toc, marks = build_toc(docs)

    sizes = [s for _d, _c, s, _w in S.code_blocks]
    shrunk = sum(1 for s in sizes if s < CODE_PT - 0.01)
    wrapped = sum(1 for _d, _c, _s, w in S.code_blocks if w)
    print(
        f"{len(chapters)} chapters + front matter + 2 appendices | {sum(len(d.headings) for d in docs)} headings | "
        f"{len(sizes)} code blocks ({shrunk} shrunk to fit, {wrapped} wrapped, smallest {min(sizes, default=0):.1f} pt)"
    )

    html_path = build / "book.html"
    if args.html_only:
        html_path.write_text(assemble(docs, toc, marks, {}), encoding="utf-8")
        print(f"wrote {html_path}")
        return

    chrome = find_chrome(args.chrome)
    raw = build / "book.raw.pdf"
    pages: dict[str, int] = {}
    for attempt in range(1, args.max_passes + 1):
        html_path.write_text(assemble(docs, toc, marks, pages), encoding="utf-8")
        run_chrome(chrome, html_path, raw)
        found = read_pages(raw)
        total = len(PdfReader(str(raw)).pages)
        print(f"pass {attempt}: {total} pages, {len(found)} destinations")
        if found == pages:
            break
        pages = found
    else:
        print("warning: page numbers did not settle; the table of contents may be off by a page", file=sys.stderr)

    meta = {
        "/Title": f"{TITLE}: {SUBTITLE}", "/Author": S.author,
        "/Subject": "ELI5 study companion to the zero-to-sglang course and the mini-sglang code",
        "/Keywords": "SGLang, LLM inference, KV cache, scheduler, tensor parallelism, ELI5",
        "/Creator": "eli5/tools/build_pdf.py (headless Chrome + pypdf)",
    }
    marks_kept = finish_pdf(raw, out, marks, pages, meta)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({total} pages, {size_mb:.1f} MB, {marks_kept} bookmarks, paper {S.paper})")


if __name__ == "__main__":
    main()
