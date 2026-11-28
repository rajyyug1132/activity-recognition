# Activity Recognition: Design System

**Version 3.0.** Replaces v2 (dark theme, lime `#D4FF00` accent, Inter). The v2 spec is kept in
`docs/ui_redesign_design.md` for history only.

## Principle

The interface is **navy ink on cool paper**. Chroma only ever means an activity. Buttons, tabs, focus rings and
links are ink; the six activity colours appear on the timeline, the ledger, the live board and the pictograms,
and nowhere else. A reviewer can read the colour of any pixel as data.

Register: product (a tool people use to look at results), with one brand surface, the Overview page.

## Colour (all hex lives in `app/styles/theme.css` and `.streamlit/config.toml`)

| Token | Hex | Use | Contrast |
|---|---|---|---|
| `--paper` | `#F4F6F9` | Page | n/a |
| `--surface` | `#FFFFFF` | Panels, inputs | n/a |
| `--rail` | `#E9EDF3` | Tracks, ghost bars | n/a |
| `--line` | `#DCE1E9` | Hairlines | n/a |
| `--line-strong` | `#737D92` | Input and dashed borders | 3.8:1 on paper |
| `--ink` | `#14213D` | Text, primary buttons, focus ring | 14.8:1 on paper |
| `--ink-2` | `#4A5366` | Secondary text | 7.1:1 |
| `--ink-3` | `#5C6578` | Tertiary text | 5.4:1 |

### Activity colours

Movement is cool, stillness is warm or plum. Each activity also has a pictogram and a text label, so colour is
never the only signal.

| Activity | Fill | Label colour | Label contrast |
|---|---|---|---|
| Walking | `#37C2BB` | ink | 7.3:1 |
| Walking upstairs | `#95D3FC` | ink | 9.9:1 |
| Walking downstairs | `#323EA5` | white | 8.9:1 |
| Sitting | `#EF853B` | ink | 6.1:1 |
| Standing | `#F2D855` | ink | 11.2:1 |
| Lying | `#77296A` | white | 9.1:1 |

Colour-blind check (OKLab distance between the closest pair, walking vs upstairs): 0.103 protan, 0.108 deutan,
0.092 tritan, 0.132 normal. Distinct enough with labels and pictograms; do not use colour alone.
Messages use `--err`, `--warn` and `--ok` text on tinted backgrounds, always with an icon and words.

## Type

One family, **Public Sans** (variable 400 to 800, Google Fonts), tabular figures for every number.
Fixed rem scale: 13, 15, 16, 19, 24, 36 px, plus one fluid display size (`clamp(3rem, 8vw, 6rem)`) used only
by the Overview headline, the reel and the live board. No uppercase tracked labels anywhere; section titles are
sentence case.

## Layout

Wide layout, content capped at 76rem. Every working page is **controls on the left, result on the right**
(`rail_mark()` turns the first column into the control panel); on narrow screens the columns stack and tape
labels collapse to pictograms. Phone recording, Live and Video share this shape.

## Components (`app/components/ui.py`)

| Function | Renders |
|---|---|
| `load_css()` | Injects `theme.css` (call once per page) |
| `page_header`, `section`, `headline`, `footer` | Titles and the honest accuracy note |
| `tape(runs, step=, total=)` | Proportional timeline, label inside each block |
| `ledger(rows)` | Time per activity; doubles as the legend |
| `board(key, confidence=, chip=, pop=)` | Live panel; `key=None` is idle and never names an activity |
| `facts(pairs)`, `notice(text, kind)`, `empty_state`, `setup_steps`, `confidence_chart` | Supporting pieces |
| `summarize(share)` | "Mostly walking" only when it really dominates (60%+), otherwise "A mix of six activities" |
| `act(key)` | One registry for phone keys (`WALKING_UPSTAIRS`) and camera keys (`upstairs`) |

Components emit class names; colours are CSS variables, so a colour change is one edit.
Use `st.markdown(..., unsafe_allow_html=True)` for any HTML with an SVG (`st.html` strips `<svg>`), keep each
block on one line, and never put a literal `<` followed by a letter in `theme.css` (`st.html` drops the whole
style block; write it as the CSS escape `\3c`).

## Motion

Purposeful and short, CSS only, all of it off under `prefers-reduced-motion`:

- Kinetic words (`kinetic()`): each word rises out of a mask on load.
- Overview reel: six colour wipes, one per activity, looping every 18 s, paused on hover.
- Overview scan figure: one sweep reveals the real acceleration trace, the colour bands and the labelled
  tape together while a scan line crosses; count-up numbers use a registered `--n` property.
- Tape blocks draw left to right; ledger bars fill; the live board name re-plays its entrance only when the
  activity changes (`pop` flips between two identical keyframe names).
- State only elsewhere: button and row hovers, a pulsing dot on live chips.

Reveals always start from the visible default (`animation-fill-mode: backwards`), so a paused or headless
renderer still shows complete content.

## Accessibility

Body text 16 px, never below 4.5:1. Focus ring 2.5 px ink on every control. Buttons are at least 44 px tall and
have explicit disabled styles. Timelines carry an `aria-label` that reads the whole sequence; the live board is
`aria-live="polite"`. The disabled state of a button is never the same as its active state.

## Do not

No side-stripe borders, gradient text, glass cards, neon on black, tracked uppercase eyebrows, identical card
grids, emoji as icons, or hex values outside the two token files. Do not claim "Mostly X" unless X dominates.

## File map

```
.streamlit/config.toml          Streamlit theme (light, ink primary)
app/streamlit_app.py            Navigation: Overview, Phone recording, Live, Video
app/overview.py                 Landing page: hero, reel, scan figure, entry points
app/home.py                     Phone recording
app/pages/1_Live.py, 2_Video.py Live and Video
app/components/ui.py            Shared components and the activity registry
app/styles/theme.css            Tokens, components, motion (single stylesheet)
tests/test_ui_helpers.py        Style registry and summary-line checks
```
