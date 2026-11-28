"""
app/components/ui.py
Shared UI helpers for the Activity Recognition redesign.
Implements the Essor-inspired dark design system.
All functions return HTML strings or call st.markdown — no logic, only presentation.
"""
from __future__ import annotations

import html
import pathlib

import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens (mirror of theme.css — single source is the CSS file;
# these dicts are used only to build inline styles for HTML components
# that live outside the Streamlit widget tree).
# ---------------------------------------------------------------------------

#: (bar_color, bar_text_color, display_name, icon_text)
ACTIVITY_STYLE: dict[str, tuple[str, str, str, str]] = {
    "WALKING":            ("#4ADE80", "#000000", "Walking",            "Walking"),
    "WALKING_UPSTAIRS":   ("#60A5FA", "#000000", "Walking upstairs",   "Up stairs"),
    "WALKING_DOWNSTAIRS": ("#3B82F6", "#ffffff", "Walking downstairs", "Down stairs"),
    "SITTING":            ("#FBBF24", "#000000", "Sitting",            "Sitting"),
    "STANDING":           ("#FB923C", "#000000", "Standing",           "Standing"),
    "LAYING":             ("#F87171", "#000000", "Lying",              "Lying"),
}

_FALLBACK = ("#9CA3AF", "#000000", "Unknown", "Unknown")

_CSS_PATH = pathlib.Path(__file__).parent.parent / "styles" / "theme.css"


# ---------------------------------------------------------------------------
# CSS loader
# ---------------------------------------------------------------------------

def load_css() -> None:
    """Inject theme.css and Inter font into the Streamlit app.

    Call once at the top of every page script.
    """
    css_text = _CSS_PATH.read_text(encoding="utf-8")
    st.html(f"<style>\n{css_text}\n</style>")


# ---------------------------------------------------------------------------
# mmss
# ---------------------------------------------------------------------------

def mmss(sec: float) -> str:
    """Format seconds as mm:ss string."""
    sec = max(0, int(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


# ---------------------------------------------------------------------------
# section_label
# ---------------------------------------------------------------------------

def section_label(text: str) -> None:
    """Render a small-caps eyebrow section label via st.markdown."""
    escaped = html.escape(text)
    st.markdown(
        f'<p class="section-label">{escaped}</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# activity_hero
# ---------------------------------------------------------------------------

def activity_hero(
    activity_name: str,
    confidence: float | None = None,
    duration: str | None = None,
    model: str | None = None,
) -> None:
    """Render the full-width activity hero card.

    Args:
        activity_name: Canonical activity key (e.g. ``"WALKING"``) or display
                       name. Unknown keys render as-is.
        confidence:    Float 0–1. Rendered as a percentage badge when given.
        duration:      Pre-formatted duration string (e.g. ``"02:10"``).
                       Shown in the sub-line when given.
        model:         Model identifier shown as a quiet chip in the corner.
    """
    key = activity_name.upper().replace(" ", "_")
    style = ACTIVITY_STYLE.get(key)
    if style is None:
        color = _FALLBACK[0]
        display_name = activity_name
    else:
        color, _, display_name, _ = style

    conf_badge = ""
    if confidence is not None:
        pct = int(round(confidence * 100))
        conf_badge = (
            f'<span class="confidence-badge">{pct}% confident</span>'
        )

    dur_part = ""
    if duration:
        escaped_dur = html.escape(str(duration))
        dur_part = f'<span class="hero-duration">{escaped_dur}</span>'

    model_chip = ""
    if model:
        escaped_model = html.escape(str(model))
        model_chip = f'<span class="model-chip">{escaped_model}</span>'

    sub_line = ""
    if conf_badge or dur_part:
        sub_line = f'<div class="hero-sub">{conf_badge}{dur_part}</div>'

    # Color accent strip on the left border
    act_style = f"border-left: 4px solid {color};"

    escaped_display = html.escape(display_name)
    # One line, no blank lines: an empty {model_chip} or {sub_line} used to leave a blank
    # line, which ends Markdown's HTML block and shows the indented rest as a code block.
    st.markdown(
        f'<div class="activity-hero" style="{act_style}">{model_chip}'
        f'<p class="eyebrow">CURRENT ACTIVITY</p>'
        f'<h1 class="activity-name">{escaped_display}</h1>{sub_line}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# status_chip
# ---------------------------------------------------------------------------

def status_chip(label: str, variant: str = "idle") -> None:
    """Render a pill-shaped status chip inline.

    Args:
        label:   Visible text (e.g. ``"Replay mode"``, ``"Phone connected"``).
        variant: One of ``"replay"``, ``"connected"``, ``"idle"``, ``"error"``.
    """
    safe_variant = html.escape(variant)
    safe_label = html.escape(label)
    st.markdown(
        f'<span class="status-chip {safe_variant}">{safe_label}</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# timeline_bar
# ---------------------------------------------------------------------------

def timeline_bar(
    runs: list[dict],
    total_windows: int,
    *,
    show_legend: bool = True,
) -> None:
    """Render a horizontal proportional colour-coded timeline bar.

    Args:
        runs:           List of dicts with keys ``"activity"`` (str) and
                        ``"count"`` (int — number of windows).
        total_windows:  Total window count (used to compute proportions).
        show_legend:    Whether to render a legend below the bar.
    """
    if not runs or total_windows == 0:
        st.markdown(
            '<div class="timeline-bar" style="background:#2A2B2D;height:28px;border-radius:8px;"></div>',
            unsafe_allow_html=True,
        )
        return

    segments = []
    for run in runs:
        act = run.get("activity", "UNKNOWN").upper()
        count = int(run.get("count", 0))
        pct = count / total_windows * 100
        color, text_color, display_name, _ = ACTIVITY_STYLE.get(act, _FALLBACK)
        safe_name = html.escape(display_name)
        segments.append(
            f'<div class="seg" style="width:{pct:.2f}%;background:{color};" '
            f'title="{safe_name} ({pct:.1f}%)" aria-label="{safe_name} {pct:.1f}%"></div>'
        )

    segs_html = "".join(segments)
    bar_html = f'<div class="timeline-bar">{segs_html}</div>'

    if show_legend:
        seen: set[str] = set()
        legend_items = []
        for run in runs:
            act = run.get("activity", "UNKNOWN").upper()
            if act in seen:
                continue
            seen.add(act)
            color, _, display_name, _ = ACTIVITY_STYLE.get(act, _FALLBACK)
            safe_name = html.escape(display_name)
            legend_items.append(
                f'<span class="leg-item">'
                f'<span class="leg-dot" style="background:{color};"></span>'
                f'{safe_name}'
                f'</span>'
            )
        legend_html = f'<div class="timeline-legend">{"".join(legend_items)}</div>'
    else:
        legend_html = ""

    st.markdown(
        f'<div class="timeline-wrap">{bar_html}{legend_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# activity_card
# ---------------------------------------------------------------------------

def activity_card(pairs: list[tuple[str, str]], sub: str | None = None) -> None:
    """Render a dark surface card with label/value pairs.

    Args:
        pairs: List of ``(label, value)`` tuples. Label is uppercase small-caps,
               value is large bold text.
        sub:   Optional sub-line beneath the value.
    """
    rows = []
    for label, value in pairs:
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        sub_html = f'<p class="card-sub">{html.escape(str(sub))}</p>' if sub else ""
        rows.append(
            f'<p class="card-label">{safe_label}</p>'
            f'<p class="card-value">{safe_value}</p>'
            f'{sub_html}'
        )

    st.markdown(
        f'<div class="dark-card">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# empty_state
# ---------------------------------------------------------------------------

def empty_state(icon: str, heading: str, body: str) -> None:
    """Render a centred empty-state block.

    Args:
        icon:    Text icon or short symbol (e.g. ``"+"``).
        heading: Short heading (e.g. ``"No recording yet"``).
        body:    One-to-two sentence explanation.
    """
    safe_icon = html.escape(icon)
    safe_heading = html.escape(heading)
    safe_body = html.escape(body)
    st.markdown(
        f"""
        <div class="empty-state">
            <span class="empty-icon">{safe_icon}</span>
            <p class="empty-heading">{safe_heading}</p>
            <p class="empty-body">{safe_body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# info_card
# ---------------------------------------------------------------------------

def info_card(text: str) -> None:
    """Render a dark info card with an accent left-border."""
    safe_text = html.escape(text)
    st.markdown(
        f'<div class="info-card">{safe_text}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# footer
# ---------------------------------------------------------------------------

def footer() -> None:
    """Render the standard app footer with model limitations note."""
    st.markdown(
        '<div class="app-footer">'
        "Model accuracy: 91% on the UCI HAR benchmark dataset. "
        "Real-world results vary with phone placement and sensor calibration."
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# setup_steps
# ---------------------------------------------------------------------------

def setup_steps(steps: list[str]) -> None:
    """Render numbered setup steps in a dark card."""
    items = "".join(f"<li>{html.escape(s)}</li>" for s in steps)
    st.markdown(
        f'<div class="dark-card"><ol class="setup-steps">{items}</ol></div>',
        unsafe_allow_html=True,
    )
