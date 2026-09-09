# presentation/

Two decks of the same talk, plus a reusable component library.

| File | What it is |
|---|---|
| `talk.tex` / `talk.pdf` | The original Beamer deck. **Unchanged** — still the canonical version. |
| `effective_engine.pptx` | PowerPoint deck: the same talk (slides 1–32), then a component library (33–51). |
| `slidekit.py` | The components. Import it to build any data-analysis deck. |
| `build_deck.py` | Content for `effective_engine.pptx`. Layout maths lives in `slidekit.py`, not here. |
| `fig/` | The four result figures, recovered from git history (`demo/build/results/figures/`). |
| `references.bib` | Bibliography, shared by both decks. |

## Rebuild

```bash
python3 build_deck.py
```

Needs only `python-pptx` and `Pillow`. No LaTeX, no node.

## Writing a new deck

```python
from slidekit import *

d = Deck()
s = d.slide()
title(s, "Does the signal survive costs?", "net Sharpe against per-trade cost")
chart_with_readout(s, 1.56, "net Sharpe vs cost", [
    "Break-even is at 8bp.",
    "Live cost is 3bp, so the edge survives.",
], image="fig/cost_sweep.png")
note_bottom(s, "Caveat", "One venue, one year. Not yet a regime test.", accent=CLAY)
d.save("out.pptx")
```

Every component takes inches, origin top-left, on a 13.333 × 7.5 canvas.
`M` is the page margin, `CW` the content width, `BOTTOM` the baseline that
closing notes sit on.

## Components

**Time-split geometry** — the diagrams that show a split cannot leak.
`timebar` is the atom; the rest are named schemes built from it.

| Function | Scheme |
|---|---|
| `split_holdout` | train / validation / test on one axis |
| `split_walk_forward` | expanding window — what the retrainer actually does |
| `split_rolling` | fixed-length sliding window |
| `split_purged_kfold` | purged K-fold with an embargo band |
| `split_grouped` | whole-date blocks, for cross-sectional panels |
| `split_diagram` | any custom stack of folds |
| `time_axis`, `legend` | the furniture that makes the above readable |

**Plot holders** — layouts you design against before the figure exists.
Pass `image=` and the placeholder becomes the real chart.

`plot_holder` · `plot_grid` (small multiples) · `chart_with_readout`
(evidence left, claim right — the default analysis layout)

**Blocks**

`card` · `note` / `note_bottom` · `formula` · `kpi_row` (stat tiles) ·
`table` · `flow_row` / `flow_box` · `numbered_steps` · `verdict_block`
(claim → test → verdict) · `compare` (before / after)

**Page furniture**

`title` · `footer` · `caption` · `title_slide` · `section_slide` ·
`closing_slide` · `bullets` · `text` · `rect` · `line` · `arrow`

## Re-skinning

The palette is the first block in `slidekit.py`. Change `NAVY`, `PINE`, `CLAY`,
`INK`, `SLATE`, `RULE` and the whole deck follows — every card tint, table
header, split bar and divider derives from those six. Slide 50 of the deck
shows the current values and the type scale.

Fonts are Cambria (headings) and Calibri (body): both ship with Office and both
have metric-compatible substitutes elsewhere, so a PDF rendered on Linux keeps
its line breaks.

## Checking a build

```bash
soffice --headless --convert-to pdf --outdir qa effective_engine.pptx
```

Then read the PDF. Look for text overflowing a card, columns of unequal
height, and anything sitting below `BOTTOM`.
