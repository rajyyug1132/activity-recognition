"""
app/components/ui.py
Shared presentation helpers. No ML logic.

Colours live in styles/theme.css as CSS variables; this module only emits class names and
structure. Every helper renders one HTML block through st.markdown(unsafe_allow_html=True),
because st.html strips inline <svg>. Keep the strings on one line: a blank line ends the
Markdown HTML block and the rest renders as a code block.
"""
from __future__ import annotations

import html
import itertools
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import streamlit as st

_CSS_PATH = pathlib.Path(__file__).resolve().parent.parent / "styles" / "theme.css"
esc = html.escape

PHONE_ACCURACY = 91  # % of windows right on the 9 never-seen UCI HAR people (reports/results.md)
PHONE_NOTE = (f"Phone model: {PHONE_ACCURACY}% accuracy on 9 people it never saw (UCI HAR). "
              "Real-world results vary with phone placement and sensor calibration.")


# ---------------------------------------------------------------------------
# Activities: one registry for the phone model (UPPER_CASE keys) and the camera model
# (lower-case keys). Colours are CSS classes, never hex.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Act:
    label: str  # "Walking upstairs"
    short: str  # "Upstairs"
    cls: str    # CSS class that sets --act and --on
    icon: str   # key in _ICONS


ACTIVITIES: dict[str, Act] = {
    "WALKING": Act("Walking", "Walking", "act-walking", "walk"),
    "WALKING_UPSTAIRS": Act("Walking upstairs", "Upstairs", "act-upstairs", "up"),
    "WALKING_DOWNSTAIRS": Act("Walking downstairs", "Downstairs", "act-downstairs", "down"),
    "SITTING": Act("Sitting", "Sitting", "act-sitting", "sit"),
    "STANDING": Act("Standing", "Standing", "act-standing", "stand"),
    "LAYING": Act("Lying", "Lying", "act-lying", "lie"),
    "STAIRS": Act("Stairs", "Stairs", "act-upstairs", "up"),  # camera model saved before notebook 09
}
_ALIASES = {"UPSTAIRS": "WALKING_UPSTAIRS", "DOWNSTAIRS": "WALKING_DOWNSTAIRS", "LYING": "LAYING"}
UNKNOWN = Act("Unknown", "Unknown", "act-none", "dot")


def act(key: object) -> Act:
    """Style for 'WALKING', 'walking', 'Walking upstairs' or 'upstairs'. Unknown keys get a neutral style."""
    k = str(key).upper().replace(" ", "_")
    return ACTIVITIES.get(_ALIASES.get(k, k), UNKNOWN)


# Pictograms on a 24-unit grid; stroke, caps and colour come from the .ico CSS class.
_ICONS = {
    "stand": '<circle cx="12" cy="4.5" r="2.1"/><path d="M12 7.6V15M8 12.4l4-3.2 4 3.2M12 15l-2.9 6.2M12 15l2.9 6.2"/>',
    "walk": '<circle cx="13.2" cy="4.5" r="2.1"/><path d="M12.6 7.6l-.9 7.2M12.2 9.6l-3.6 2.8M12.4 9.6l3.8 1.8M11.7 14.8l-3.4 6.4M11.7 14.8l3.9 3 .9 3.4"/>',
    "up": '<path d="M3 20.5h4.5V16H12v-4.5h4.5V7H21M4.5 10.5l5-5M9.5 5.5H6.3M9.5 5.5v3.2"/>',
    "down": '<path d="M3 7h4.5v4.5H12V16h4.5v4.5H21M14.5 4.5l5 5M19.5 9.5h-3.2M19.5 9.5V6.3"/>',
    "sit": '<circle cx="8.5" cy="4.6" r="2.1"/><path d="M8.5 7.7V15M8.5 15h7.2M15.7 15v6M8.5 10.2l4.4 2.6"/>',
    "lie": '<circle cx="4.9" cy="14.6" r="2.1"/><path d="M8 14.6h12.3l1.7 2.4M3 20h18"/>',
    "dot": '<circle cx="12" cy="12" r="3"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
}


def icon(key: str, cls: str = "ico") -> str:
    return (f'<svg class="{cls}" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            f'{_ICONS.get(key, _ICONS["dot"])}</svg>')


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def mmss(sec: float) -> str:
    """Seconds as m:ss ('2:04'). One format everywhere."""
    s = max(0, int(round(sec)))
    return f"{s // 60}:{s % 60:02d}"


def listing(words: Sequence[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    words = list(words)
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def coalesce(keys: Iterable[object]) -> list[tuple[str, int]]:
    """Consecutive equal activities -> [(key, run_length), ...] in time order."""
    return [(str(k), len(list(g))) for k, g in itertools.groupby(keys)]


_NUM = ["no", "one", "two", "three", "four", "five", "six", "seven"]


def summarize(share: Mapping[str, float], what: str = "recording") -> tuple[str, str]:
    """Headline and sub-line for a result. 'Mostly X' only when X really dominates."""
    present = sorted(((k, v) for k, v in share.items() if v > 0), key=lambda kv: -kv[1])
    total = sum(v for _, v in present)
    if not present or total <= 0:
        return "No activity found", ""
    (top, top_v), rest = present[0], present[1:]
    name, frac = act(top).label.lower(), top_v / total
    if not rest:
        return f"All {name}", f"The whole {what} was {name}."
    if frac >= 0.6:
        others = listing([act(k).label.lower() for k, _ in rest])
        return f"Mostly {name}", f"{frac:.0%} of the {what}, with {others}."
    return (f"A mix of {_NUM[min(len(present), 7)]} activities",
            f"The longest, {name}, takes {frac:.0%} of the {what}.")


def kinetic(text: str) -> str:
    """Words that rise out of a mask on load (CSS only; see .kinetic in theme.css)."""
    return " ".join(f'<span class="w" style="--i:{i}"><span>{esc(w)}</span></span>'
                    for i, w in enumerate(text.split()))


def _md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

def load_css() -> None:
    """Inject theme.css. Call once at the top of every page script."""
    st.html(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>")


def page_header(title: str, subtitle: str) -> None:
    _md(f'<header class="page-head"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></header>')


def section(title: str, meta: str = "", *, big: bool = False) -> None:
    meta_html = f"<span>{esc(meta)}</span>" if meta else ""
    _md(f'<div class="sec{" lg" if big else ""}"><h2>{esc(title)}</h2>{meta_html}</div>')


def headline(title: str, sub: str = "") -> None:
    sub_html = f"<p>{esc(sub)}</p>" if sub else ""
    _md(f'<div class="headline"><h2 class="kinetic" aria-label="{esc(title)}">{kinetic(title)}</h2>{sub_html}</div>')


def rail_mark() -> None:
    """Marks the column that holds the controls; theme.css turns it into the left panel."""
    _md('<span class="rail-mark"></span>')


def footer(note: str = PHONE_NOTE) -> None:
    _md(f'<p class="foot"><b>About these numbers.</b> {esc(note)}</p>')


def notice(text: str, kind: str = "info") -> None:
    """Inline message with a full hairline border (kind: info, warn, err, ok)."""
    _md(f'<div class="notice {esc(kind)}" role="note">{icon("info")}<p>{esc(text)}</p></div>')


def empty_state(heading: str, body: str) -> None:
    """Empty canvas that previews what will appear: a ghost timeline, then plain words."""
    ghost = "".join(f'<i style="--w:{w};--i:{i}"></i>' for i, w in enumerate((17, 16, 14, 18, 19, 16)))
    _md(f'<div class="empty"><div class="ghost" aria-hidden="true">{ghost}</div>'
        f'<h2>{esc(heading)}</h2><p>{esc(body)}</p></div>')


def setup_steps(steps: Sequence[str]) -> None:
    _md(f'<ol class="steps">{"".join(f"<li><span>{esc(s)}</span></li>" for s in steps)}</ol>')


# ---------------------------------------------------------------------------
# Data components
# ---------------------------------------------------------------------------

def tape_html(runs: Sequence[tuple[str, int]], *, step: float, total: float | None = None,
              axis: Sequence[str] | None = None) -> str:
    """Proportional timeline with the label inside each block.

    runs: [(activity_key, n_windows), ...] in time order. step: seconds per window.
    total: seconds for the axis (defaults to windows * step). axis: replaces the three tick labels.
    """
    n = sum(c for _, c in runs)
    if n <= 0:
        return ""
    blocks, spoken = [], []
    for i, (key, count) in enumerate(runs):
        a, pct = act(key), count / n * 100
        size = "" if pct >= 11 else (" min" if pct >= 5 else " tiny")
        blocks.append(f'<div class="seg {a.cls}{size}" style="--w:{pct:.2f};--i:{i}" '
                      f'title="{esc(a.label)} {mmss(count * step)}">{icon(a.icon)}<span>{esc(a.short)}</span></div>')
        spoken.append(f"{a.label} {mmss(count * step)}")
    total = n * step if total is None else total
    ticks = list(axis) if axis else [mmss(0), mmss(total / 2), mmss(total)]
    return (f'<figure class="tape" role="img" aria-label="{esc("Timeline: " + ", then ".join(spoken))}">'
            f'<div class="tape-bar">{"".join(blocks)}</div>'
            f'<div class="tape-axis">{"".join(f"<span>{esc(t)}</span>" for t in ticks)}</div></figure>')


def tape(runs: Sequence[tuple[str, int]], *, step: float, total: float | None = None,
         axis: Sequence[str] | None = None) -> None:
    markup = tape_html(runs, step=step, total=total, axis=axis)
    if markup:
        _md(markup)


def ledger(rows: Iterable[tuple[str, float, str, str]]) -> None:
    """One row per activity. rows: (activity_key, bar fraction 0-1, main text, side text)."""
    items = []
    for i, (key, frac, main, side) in enumerate(rows):
        a = act(key)
        items.append(f'<li class="{a.cls}" style="--w:{max(0.0, min(1.0, frac)):.4f};--i:{i}">'
                     f'<span class="chipico">{icon(a.icon)}</span><span class="nm">{esc(a.label)}</span>'
                     f'<span class="tm">{esc(main)}</span><span class="pc">{esc(side)}</span>'
                     f'<span class="trk"><i></i></span></li>')
    _md(f'<ul class="ledger">{"".join(items)}</ul>')


def _chip(label: str, kind: str = "idle") -> str:
    return f'<span class="chip {esc(kind)}"><i class="dot"></i>{esc(label)}</span>'


def board(key: str | None = None, *, confidence: float | None = None, headline_text: str = "",
          hint: str = "", chip: tuple[str, str] | None = None, pop: int = 0) -> None:
    """Live status panel. key=None is the idle state: it never names an activity.

    pop: bump it when the activity changes; the name re-plays its entrance (A/B animation names).
    """
    chip_html = _chip(*chip) if chip else ""
    if key is None:
        _md(f'<section class="board idle" aria-live="polite"><div class="board-top">{chip_html}</div>'
            f'<h2 class="board-name">{esc(headline_text)}</h2><p class="board-hint">{esc(hint)}</p></section>')
        return
    a = act(key)
    meter = ""
    if confidence is not None:
        meter = (f'<div class="board-meta"><span class="meter"><i style="--w:{confidence:.3f}"></i></span>'
                 f'<span>{round(confidence * 100)}% confident</span></div>')
    _md(f'<section class="board {a.cls}" aria-live="polite"><div class="board-top">{chip_html}</div>'
        f'<div class="board-main">{icon(a.icon, "ico big")}'
        f'<h2 class="board-name pop-{"ab"[pop % 2]}">{esc(a.label)}</h2></div>{meter}</section>')


def facts(pairs: Iterable[tuple[str, str]], *, col: bool = False) -> None:
    cells = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in pairs)
    _md(f'<dl class="facts{" col" if col else ""}">{cells}</dl>')


def confidence_chart(xs: Iterable[float], ys: Iterable[float], *, threshold: float = 0.6) -> None:
    """Model confidence per window over time. ys are 0-1."""
    xs, ys = list(xs), list(ys)
    if len(xs) < 2:
        return
    w, h, pad = 640, 150, 6
    x0, span = xs[0], (xs[-1] - xs[0]) or 1.0
    pts = " ".join(f"{pad + (x - x0) / span * (w - 2 * pad):.1f},{pad + (1 - y) * (h - 2 * pad):.1f}"
                   for x, y in zip(xs, ys))
    ty = pad + (1 - threshold) * (h - 2 * pad)
    svg = (f'<svg class="chart" viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img" '
           f'aria-label="Model confidence for each window over time">'
           f'<line class="grid" x1="0" x2="{w}" y1="{pad}" y2="{pad}"/>'
           f'<line class="grid" x1="0" x2="{w}" y1="{h - pad}" y2="{h - pad}"/>'
           f'<line class="thr" x1="0" x2="{w}" y1="{ty:.1f}" y2="{ty:.1f}"/>'
           f'<polyline class="ln" points="{pts}"/></svg>')
    _md(f'<figure class="plot">{svg}<figcaption><span>0:00</span>'
        f'<span>top = 100%, dashed = {threshold:.0%}</span><span>{mmss(xs[-1])}</span></figcaption></figure>')
