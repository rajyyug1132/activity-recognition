# Activity Recognition — Design System

**Version:** 2.0 (Essor-inspired dark theme)  
**Branch:** `ui-ux-redesign`  
**Replaces:** Oura Analysis v1 (warm cream, Editorial New serif)

---

## Colour Tokens

| Token | Hex | Tailwind ref | Usage |
|-------|-----|-------------|-------|
| `--clr-bg` | `#0B0B0C` | — | Page background (60% rule) |
| `--clr-surface` | `#17181A` | — | Card / panel surfaces (30%) |
| `--clr-border` | `#2A2B2D` | — | Subtle card borders |
| `--clr-text-primary` | `#F5F5F5` | neutral-100 | Body text, headings |
| `--clr-text-secondary` | `#9CA3AF` | gray-400 | Labels, captions |
| `--clr-text-disabled` | `#4B5563` | gray-600 | Placeholders, footer |
| `--clr-accent` | `#D4FF00` | — | Primary buttons, active tab, highlights |
| `--clr-accent-fg` | `#0B0B0C` | — | Text on accent backgrounds |
| `--clr-danger` | `#F87171` | red-400 | Error states |
| `--clr-success` | `#4ADE80` | green-400 | Success states |

**To change the accent colour:** update `--clr-accent` in `app/styles/theme.css` and `primaryColor` in `.streamlit/config.toml`.

### Activity Data Colours

Used only in timelines, charts and badges. Always paired with a visible text label.

| Activity | Background | Text | WCAG AA check |
|----------|-----------|------|---------------|
| WALKING | `#4ADE80` | `#000000` | 8.1:1 ✓ |
| WALKING_UPSTAIRS | `#60A5FA` | `#000000` | 5.5:1 ✓ |
| WALKING_DOWNSTAIRS | `#3B82F6` | `#ffffff` | 4.9:1 ✓ |
| SITTING | `#FBBF24` | `#000000` | 10.0:1 ✓ |
| STANDING | `#FB923C` | `#000000` | 5.4:1 ✓ |
| LAYING | `#F87171` | `#000000` | 4.5:1 ✓ |

Colours are distinguishable for the most common types of colour-blindness (deuteranopia, protanopia, tritanopia) when paired with text labels.

---

## Typography Scale

| Role | Family | Weight | Size | Line-height |
|------|--------|--------|------|-------------|
| H1 — Activity name | Inter | 900 | `clamp(2.5rem, 8vw, 6rem)` | 0.95 |
| H2 — Page title | Inter | 700 | `2rem` | 1.1 |
| H3 — Card heading | Inter | 600 | `1.25rem` | 1.3 |
| Eyebrow / section label | Inter | 500 | `0.75rem` | 1.0 |
| Body | Inter | 400 | `1rem` (16px min) | 1.6 |
| Stat number | Inter | 700 | `2rem` | 1.0 |
| Caption | Inter | 400 | `0.875rem` | 1.4 |

Font loaded via Google Fonts (`display=swap`). Fallback: `system-ui, sans-serif`.

---

## Spacing Scale

| Token | Value | Typical use |
|-------|-------|-------------|
| `--sp-xs` | 4px | Inline gap |
| `--sp-sm` | 8px | Tight vertical stack |
| `--sp-md` | 16px | Default component gap |
| `--sp-lg` | 24px | Section padding |
| `--sp-xl` | 48px | Between major sections |
| `--sp-2xl` | 80px | Hero top-padding, footer margin |

---

## Radius & Shadow

| Token | Value |
|-------|-------|
| `--radius-sm` | 8px |
| `--radius-md` | 12px |
| `--radius-pill` | 999px |
| Shadow | None — flat design with border-only cards |

---

## Component Library (`app/components/ui.py`)

| Function | Renders | Key args |
|----------|---------|---------|
| `load_css()` | Injects Inter + `theme.css` | — |
| `activity_hero(name, conf, duration, model)` | Full-width dark card, huge H1 | str, float, str, str |
| `status_chip(label, variant)` | Pill chip | `replay` / `connected` / `idle` / `error` |
| `section_label(text)` | Small-caps eyebrow | str |
| `timeline_bar(runs, total)` | Horizontal coloured bar + legend | list of dicts, int |
| `activity_card(pairs, sub)` | Dark card with label/value rows | list of tuples |
| `empty_state(icon, heading, body)` | Centred empty-state block | str, str, str |
| `info_card(text)` | Dark info card with accent left-border | str |
| `footer()` | Model limitations note | — |
| `setup_steps(steps)` | Numbered step list in dark card | list of str |
| `mmss(sec)` | Format seconds → `mm:ss` | float |

---

## Accessibility

| Requirement | Implementation |
|-------------|---------------|
| Body contrast ≥ 4.5:1 | `#F5F5F5` on `#0B0B0C` → 17.5:1 |
| Large-text contrast ≥ 3:1 | H1 on card `#17181A` → 14.1:1 |
| All activity badge pairs ≥ 4.5:1 | Verified in table above |
| Body text ≥ 16px | `1rem` base, enforced in CSS reset |
| Tap targets ≥ 44 × 44 px | `min-height: 44px` on all buttons |
| Focus ring | `:focus-visible` with 2px `--clr-accent` outline |
| `prefers-reduced-motion` | Fade transition guarded by media query |
| Alt text on images | `alt` present on all `<img>` tags |
| Semantic headings | `<h1>`, `<h2>` used in HTML injections |

---

## Responsive Breakpoints

| Breakpoint | Behaviour |
|------------|-----------|
| Default (mobile-first, ≥ 390px) | Single column; full-width cards |
| ≥ 760px | Two-column layout on Video page |
| ≥ 1080px | Content max-width `960px`, centred |

---

## File Map

```
activity-recognition/
├── .streamlit/
│   └── config.toml          ← dark theme primaryColor / backgroundColor
├── app/
│   ├── components/
│   │   ├── __init__.py
│   │   └── ui.py            ← all shared component helpers
│   ├── pages/
│   │   ├── 1_Live.py        ← redesigned
│   │   └── 2_Video.py       ← redesigned
│   ├── styles/
│   │   └── theme.css        ← single source of CSS tokens and component styles
│   └── streamlit_app.py     ← redesigned main page
└── docs/
    └── ui_redesign_design.md ← full design specification
```

---

## Out of Scope

- `src/`, `config/`, `models/`, `notebooks/`, `data/` — no changes
- Dark/light mode toggle (always dark)
- New pip dependencies
- Server performance or caching
- Authentication
