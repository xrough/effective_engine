#!/usr/bin/env python3
"""
Bind every notebook in notebooks/ into one printable PDF.

    python3 scripts/notebooks_to_pdf.py                 # -> notebooks/volatility_series.pdf
    python3 scripts/notebooks_to_pdf.py -o out.pdf      # somewhere else
    python3 scripts/notebooks_to_pdf.py --only the_cross_section,what_roughness_claims

Reading order comes from notebooks/.order. Notebooks are rendered from their
committed outputs -- nothing is executed, so the figures and verdicts in the
PDF are exactly the ones in the repo.

Pipeline: nbconvert -> HTML, inject print CSS, Chromium -> PDF per notebook,
then merge with a title page, a table of contents with real page numbers,
PDF bookmarks, and continuous page numbering.

Needs: nbconvert, playwright (chromium), pymupdf.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile

import fitz  # pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

TITLE = "Volatility Models, And The Methods That Test Them"
SUBTITLE = "The Effective Engine notebook series"

# --- density dials --------------------------------------------------------
# These four numbers set the page count. Defaults are compact: tight margins,
# small type, figures scaled down. Override from the command line
# (--body/--code/--margin/--figscale) rather than editing, or pass --loose for
# the roomier screen-reading settings.
DEFAULTS = dict(margin="0.40in 0.42in 0.48in 0.42in",
                body=8.7, code=7.2, line=1.28, figscale=0.84)
LOOSE = dict(margin="0.62in 0.55in 0.72in 0.55in",
             body=10.2, code=8.4, line=1.42, figscale=1.0)

# Print stylesheet. The important rule is the first one: nbconvert's HTML puts
# code in <pre>, which does not wrap, so in print every long line is silently
# truncated at the right margin. The rest is page-break hygiene and stripping
# the generous screen padding the lab template ships with.
PRINT_CSS = """
<style>
@page { size: Letter; margin: __MARGIN__; }

/* --- never lose a long line off the right edge ------------------------- */
pre, code, .highlight pre, .jp-RenderedText pre, .jp-OutputArea-output pre,
div.output_area pre, .CodeMirror-line, .cm-editor .cm-line {
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
    overflow: visible !important;
}
.jp-Cell, .jp-InputArea, .jp-OutputArea, .jp-CodeMirrorEditor,
.jp-InputArea-editor, .cm-editor, .cm-scroller {
    overflow: visible !important;
    max-width: 100% !important;
}

/* --- keep colour in print --------------------------------------------- */
* { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }

/* --- typography sized for paper, not a 27-inch monitor ----------------- */
body, .jp-Notebook, .jp-RenderedHTMLCommon {
    font-size: __BODYpt__ !important; line-height: __LINE__ !important; }
pre, code, .jp-RenderedText { font-size: __CODEpt__ !important;
                              line-height: 1.26 !important; }
h1 { font-size: __H1__ !important; }
h2 { font-size: __H2__ !important; }
h3 { font-size: __H3__ !important; }
h4, h5 { font-size: __H4__ !important; }

/* --- squeeze the vertical rhythm --------------------------------------- */
.jp-RenderedHTMLCommon p, .jp-RenderedHTMLCommon ul, .jp-RenderedHTMLCommon ol,
p, ul, ol, blockquote { margin: 0.28em 0 !important; }
.jp-RenderedHTMLCommon li, li { margin: 0.08em 0 !important; padding: 0 !important; }
.jp-RenderedHTMLCommon > *:first-child { margin-top: 0 !important; }
.jp-RenderedHTMLCommon > *:last-child { margin-bottom: 0 !important; }
h1 { margin: .45em 0 .22em !important; }
h2 { margin: .40em 0 .18em !important; }
h3, h4, h5 { margin: .34em 0 .14em !important; }
pre { padding: 3px 5px !important; margin: 2px 0 !important; }
.jp-Cell { padding: 0 !important; margin: 0 0 1.5px 0 !important; }
.jp-Cell-inputWrapper, .jp-Cell-outputWrapper { padding: 0 !important; margin: 0 !important; }
.jp-InputArea, .jp-OutputArea, .jp-OutputArea-child { padding: 0 !important; margin: 0 !important; }
.jp-InputArea-editor { padding: 0 !important; }
.jp-OutputArea-output { padding: 0 !important; margin: 0 !important; }
.jp-MarkdownCell, .jp-MarkdownOutput { padding: 0 !important; margin: 0 !important; }

/* --- drop the In[ ]/Out[ ] gutter --------------------------------------
   It costs ~0.7in of every line's width, and nothing in the series refers to
   a cell by execution number -- recipes are cited as R1.1, R10.2 and so on.
   Reclaiming the width is worth more than the labels, and any narrower
   gutter just truncates them to "In...". */
.jp-InputPrompt, .jp-OutputPrompt, .jp-InputArea-prompt, .jp-OutputArea-prompt {
    display: none !important; }
.jp-InputArea, .jp-OutputArea-child { padding-left: 0 !important; }

/* --- page-break hygiene ------------------------------------------------
   Figures must never be sliced, but long text and table outputs may flow
   across a page. Blanket break-avoidance is what leaves half-empty pages. */
h1, h2, h3, h4 { break-after: avoid-page; page-break-after: avoid; }
.jp-OutputArea-child, div.output_area { break-inside: auto; }
.jp-InputArea, div.input { break-inside: auto; }
.jp-RenderedImage, .jp-OutputArea-child:has(img), figure {
    break-inside: avoid; page-break-inside: avoid; }
img, svg { max-width: __FIGW__ !important; height: auto !important;
           break-inside: avoid; page-break-inside: avoid; }
table { break-inside: auto; font-size: __TBLpt__ !important; }
thead { display: table-header-group; }
tr { break-inside: avoid; page-break-inside: avoid; }
.dataframe td, .dataframe th { padding: 1px 5px !important; }

/* --- trim the chrome nbconvert adds ------------------------------------ */
.jp-Notebook { padding: 0 !important; }
body { padding: 0 !important; margin: 0 !important; background: #fff !important; }
#notebook-container { box-shadow: none !important; padding: 0 !important; }
</style>
"""


def bottom_margin_pt(margin):
    """Bottom value of a CSS margin shorthand, in points."""
    parts = margin.split()
    v = {1: parts[0], 2: parts[0], 3: parts[2], 4: parts[2]}.get(len(parts),
                                                                 parts[0])
    n = float(re.sub(r"[a-z%]+$", "", v))
    return n * {"in": 72, "cm": 28.35, "mm": 2.835, "pt": 1,
                "px": 0.75}.get(re.sub(r"^[\d.]+", "", v) or "in", 72)


def build_css(body, code, line, margin, figscale):
    return (PRINT_CSS
            .replace("__MARGIN__", margin)
            .replace("__BODYpt__", f"{body}pt")
            .replace("__CODEpt__", f"{code}pt")
            .replace("__TBLpt__", f"{code - 0.1:.1f}pt")
            .replace("__LINE__", str(line))
            .replace("__H1__", f"{body * 1.72:.1f}pt")
            .replace("__H2__", f"{body * 1.34:.1f}pt")
            .replace("__H3__", f"{body * 1.14:.1f}pt")
            .replace("__H4__", f"{body * 1.02:.1f}pt")
            .replace("__FIGW__", f"{figscale * 100:.0f}%"))

TITLE_HTML = """
<!doctype html><meta charset="utf-8"><style>
@page {{ size: Letter; margin: 0; }}
html, body {{ margin: 0; padding: 0; height: 100%; -webkit-print-color-adjust: exact; }}
body {{ font-family: Cambria, Georgia, serif; color: #191970;
        display: flex; flex-direction: column; justify-content: center;
        padding: 0 1.15in; box-sizing: border-box; }}
.rule {{ height: 7px; width: 3.1in; display: flex; margin: 0 0 0.42in 0; }}
.rule i {{ display: block; height: 100%; }}
h1 {{ font-size: 30pt; line-height: 1.16; margin: 0 0 .20in 0; font-weight: 700; }}
h2 {{ font-size: 14.5pt; font-weight: 400; color: #01796F; margin: 0 0 .55in 0;
      font-style: italic; }}
.meta {{ font-family: Calibri, Helvetica, Arial, sans-serif; font-size: 10.5pt;
         color: #5A6472; line-height: 1.68; }}
.meta b {{ color: #191970; }}
ol {{ font-family: Calibri, Helvetica, Arial, sans-serif; font-size: 10pt;
      color: #333; margin: .18in 0 0 0; padding-left: 1.15em; line-height: 1.55; }}
</style>
<div class="rule"><i style="background:#191970;flex:3"></i>
<i style="background:#01796F;flex:1"></i><i style="background:#B85042;flex:1"></i></div>
<h1>{title}</h1>
<h2>{subtitle}</h2>
<div class="meta">
<b>{n} notebooks</b> &middot; {pages} pages &middot; rendered from committed outputs<br>
Reading order as given by <code>notebooks/.order</code><br>
{date}
</div>
</div>
"""

TOC_HTML = """
<!doctype html><meta charset="utf-8"><style>
@page {{ size: Letter; margin: 0.85in 0.9in; }}
body {{ font-family: Calibri, Helvetica, Arial, sans-serif; margin: 0;
        -webkit-print-color-adjust: exact; }}
h1 {{ font-family: Cambria, Georgia, serif; color: #191970; font-size: 21pt;
      margin: 0 0 .06in 0; }}
.sub {{ color: #5A6472; font-size: 10pt; font-style: italic;
        margin: 0 0 .34in 0; }}
.part {{ font-family: Cambria, Georgia, serif; color: #01796F; font-size: 11.5pt;
         font-weight: 700; margin: .30in 0 .10in 0; }}
.part:first-of-type {{ margin-top: .10in; }}
.row {{ display: flex; align-items: baseline; font-size: 10.6pt; color: #1a1a1a;
        margin: 0 0 .105in 0; }}
.nm {{ white-space: nowrap; }}
.dots {{ flex: 1; border-bottom: 1.1px dotted #b9c0cc; margin: 0 .10in; height: .62em; }}
.pg {{ color: #191970; font-weight: 700; white-space: nowrap; }}
.f {{ color: #5A6472; font-size: 8.6pt; font-family: "Courier New", monospace; }}
</style>
<h1>Contents</h1>
<div class="sub">{subtitle} &mdash; page numbers are the printed page.</div>
{rows}
"""


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout or "").strip()[-1200:])
    return r


def reading_order(only=None):
    order_file = os.path.join(NB_DIR, ".order")
    names = []
    if os.path.exists(order_file):
        for line in open(order_file):
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    present = {f for f in os.listdir(NB_DIR) if f.endswith(".ipynb")}
    ordered = [n for n in names if n in present]
    ordered += sorted(present - set(ordered))          # anything not listed
    if only:
        keep = {o if o.endswith(".ipynb") else o + ".ipynb" for o in only}
        ordered = [n for n in ordered if n in keep]
    return ordered


def nb_title(path):
    """First markdown H1, else the filename prettified."""
    import json
    try:
        nb = json.load(open(path))
        for c in nb.get("cells", []):
            if c.get("cell_type") == "markdown":
                for ln in c.get("source", []):
                    m = re.match(r"^#\s+(.+?)\s*$", ln)
                    if m:
                        return re.sub(r"[*`_]", "", m.group(1)).strip()
    except Exception:
        pass
    return os.path.splitext(os.path.basename(path))[0].replace("_", " ").title()


PART = {
    "the_volatility_surface": "Part I — A volatility model, and whether it holds",
    "where_the_numbers_come_from": "Part III — Method in depth",
    "the_cross_section": "Part II — The cross-sectional design",
    "volatility_project_01_data_and_joins": "Volatility project workflows",
    "competition_workbench": "Workbench",
}


async def render(html_files, pdf_files):
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        for src, dst in zip(html_files, pdf_files):
            await page.goto("file://" + src, wait_until="networkidle",
                            timeout=180_000)
            await page.pdf(path=dst, format="Letter", print_background=True,
                           prefer_css_page_size=True)
        await browser.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output",
                    default=os.path.join(NB_DIR, "volatility_series.pdf"))
    ap.add_argument("--only", default=None,
                    help="comma-separated notebook names")
    ap.add_argument("--loose", action="store_true",
                    help="roomier screen-reading settings (many more pages)")
    ap.add_argument("--body", type=float, help="body text size in pt")
    ap.add_argument("--code", type=float, help="code and table size in pt")
    ap.add_argument("--line", type=float, help="body line height")
    ap.add_argument("--margin", help='CSS @page margin, e.g. "0.4in"')
    ap.add_argument("--figscale", type=float,
                    help="figure width as a fraction of the text column")
    args = ap.parse_args()

    cfg = dict(LOOSE if args.loose else DEFAULTS)
    for k in ("body", "code", "line", "margin", "figscale"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    css = build_css(**cfg)
    print(f"  body {cfg['body']}pt · code {cfg['code']}pt · "
          f"leading {cfg['line']} · margin {cfg['margin']} · "
          f"figures {cfg['figscale']:.0%}")

    only = args.only.split(",") if args.only else None
    names = reading_order(only)
    if not names:
        sys.exit("no notebooks found")

    tmp = tempfile.mkdtemp(prefix="nbpdf-")
    print(f"{len(names)} notebooks -> {args.output}")

    # --- 1. notebook -> HTML -> PDF ---------------------------------------
    htmls, pdfs, metas = [], [], []
    for i, name in enumerate(names, 1):
        src = os.path.join(NB_DIR, name)
        stem = os.path.splitext(name)[0]
        print(f"  [{i:2d}/{len(names)}] {stem}", flush=True)
        # Notebooks get renamed and resaved while this runs; one that moves
        # should cost its own section, not the whole build.
        if not os.path.exists(src):
            print("       skipped: gone since .order was read", file=sys.stderr)
            continue
        try:
            sh([sys.executable, "-m", "nbconvert", "--to", "html",
                "--template", "lab", "--output-dir", tmp, src])
        except RuntimeError as e:
            print(f"       SKIPPED, nbconvert failed:\n       {e}",
                  file=sys.stderr)
            continue
        html = os.path.join(tmp, stem + ".html")
        body = open(html, encoding="utf-8").read()
        body = body.replace("</head>", css + "</head>", 1)
        open(html, "w", encoding="utf-8").write(body)
        htmls.append(html)
        pdfs.append(os.path.join(tmp, stem + ".pdf"))
        metas.append((stem, nb_title(src)))

    print("  rendering with chromium ...", flush=True)
    asyncio.run(render(htmls, pdfs))

    # --- 2. measure, so the contents page can carry real page numbers -----
    lens = [len(fitz.open(p)) for p in pdfs]
    front = 2                                   # title + one contents page
    starts, run = [], front + 1
    for n in lens:
        starts.append(run)
        run += n
    total = run - 1

    # --- 3. front matter ---------------------------------------------------
    def html_to_pdf(html_str, dst):
        f = os.path.join(tmp, os.path.basename(dst) + ".html")
        open(f, "w", encoding="utf-8").write(html_str)
        asyncio.run(render([f], [dst]))

    title_pdf = os.path.join(tmp, "_title.pdf")
    html_to_pdf(TITLE_HTML.format(
        title=TITLE, subtitle=SUBTITLE, n=len(names), pages=total,
        date=dt.date.today().strftime("%-d %B %Y")), title_pdf)

    rows = []
    for (stem, ttl), pg in zip(metas, starts):
        if stem in PART:
            rows.append(f'<div class="part">{PART[stem]}</div>')
        rows.append(
            f'<div class="row"><span class="nm">{ttl}<br>'
            f'<span class="f">{stem}.ipynb</span></span>'
            f'<span class="dots"></span><span class="pg">{pg}</span></div>')
    toc_pdf = os.path.join(tmp, "_toc.pdf")
    html_to_pdf(TOC_HTML.format(subtitle=SUBTITLE, rows="\n".join(rows)),
                toc_pdf)

    if len(fitz.open(toc_pdf)) != 1:
        print("  note: contents ran to more than one page; renumbering",
              file=sys.stderr)
        front = 1 + len(fitz.open(toc_pdf))
        starts, run = [], front + 1
        for n in lens:
            starts.append(run)
            run += n
        total = run - 1
        rows = []
        for (stem, ttl), pg in zip(metas, starts):
            if stem in PART:
                rows.append(f'<div class="part">{PART[stem]}</div>')
            rows.append(
                f'<div class="row"><span class="nm">{ttl}<br>'
                f'<span class="f">{stem}.ipynb</span></span>'
                f'<span class="dots"></span><span class="pg">{pg}</span></div>')
        html_to_pdf(TOC_HTML.format(subtitle=SUBTITLE, rows="\n".join(rows)),
                    toc_pdf)
        html_to_pdf(TITLE_HTML.format(
            title=TITLE, subtitle=SUBTITLE, n=len(names), pages=total,
            date=dt.date.today().strftime("%-d %B %Y")), title_pdf)

    # --- 4. merge ----------------------------------------------------------
    out = fitz.open()
    out.insert_pdf(fitz.open(title_pdf))
    out.insert_pdf(fitz.open(toc_pdf))
    for p in pdfs:
        out.insert_pdf(fitz.open(p))

    # --- 5. running footer: title left, page number right ------------------
    # Sit the footer inside the bottom margin. A fixed offset collides with
    # the text block as soon as the margin is tightened.
    foot_up = max(11.0, min(22.0, bottom_margin_pt(cfg["margin"]) * 0.55))
    for i, page in enumerate(out):
        if i < front:                        # not on the front matter
            continue
        w, h = page.rect.width, page.rect.height
        which = max(k for k, s in enumerate(starts) if s <= i + 1)
        page.insert_text((bottom_margin_pt(cfg['margin']) * 0.9,
                          h - foot_up), metas[which][1][:74],
                         fontsize=7.4, fontname="helv",
                         color=(0.42, 0.46, 0.52))
        num = str(i + 1)
        page.insert_text((w - bottom_margin_pt(cfg['margin']) * 0.9
                          - fitz.get_text_length(num, fontname="hebo",
                                                 fontsize=8.4),
                          h - foot_up), num,
            fontsize=8.4, fontname="hebo", color=(0.10, 0.10, 0.44))

    # --- 6. bookmarks ------------------------------------------------------
    toc, seen = [[1, "Title", 1], [1, "Contents", 2]], set()
    for (stem, ttl), pg in zip(metas, starts):
        if stem in PART and PART[stem] not in seen:
            seen.add(PART[stem])
            toc.append([1, PART[stem], pg])
        toc.append([2 if stem in PART or seen else 1, ttl, pg])
    out.set_toc(toc)

    out.set_metadata({"title": TITLE, "author": "Xiaohao Ji",
                      "subject": SUBTITLE,
                      "keywords": "volatility, rough volatility, notebooks"})
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.save(args.output, garbage=4, deflate=True)
    out.close()
    shutil.rmtree(tmp, ignore_errors=True)

    mb = os.path.getsize(args.output) / 1e6
    print(f"\n{args.output}\n{total} pages · {mb:.1f} MB")


if __name__ == "__main__":
    main()
