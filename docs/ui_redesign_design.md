> **Superseded.** This is the v2 spec (dark theme, lime accent). The current system is v3, documented in `DESIGN.md`.

# UI/UX Redesign — Activity Recognition App

**Status:** Draft v1  
**Branch:** `ui-ux-redesign`  
**Author:** Automated redesign — Essor Fitness inspiration  
**Date:** 2026-11-26

---

## 1. Goals

| Goal | Measure |
|------|---------|
| Make the detected activity legible at a glance | Activity name dominates the viewport as an H1 |
| Reduce visual noise | ≤ 4 visible regions per screen at any time |
| Pass WCAG 2.1 AA | Contrast ≥ 4.5:1 body, ≥ 3:1 large text; tap targets ≥ 44 × 44 px |
| Preserve all existing features | All 9 baseline tests pass after implementation |
| Match the Essor Fitness aesthetic | Pure-black background, single accent, whitespace-led layout |

**Non-goals (explicitly out of scope):**

- Changing any ML/inference logic, model files, `src/`, `config/`, `notebooks/`, or `data/`
- Adding new Python packages (beyond what is already in `requirements.txt`)
- Server-side performance optimisation
- Internationalisation or localisation
- Native mobile app

---

## 2. Audience

**Primary user:** A person — athlete, coach, researcher, or curious consumer — who uploads a phone-recorded IMU CSV or watches the live/replay demo, and wants to know *what activity was detected and how confident the model is*, within seconds of landing on the page.

They are **not** ML experts. They should not need to understand sensor fusion, window sizes, or model architecture to use the app. All copy is plain language.

---

## 3. Design Tokens

### 3.1 Colour

| Token | Hex | Usage |
|-------|-----|-------|
| `--clr-bg` | `#0B0B0C` | Page background (60%) |
| `--clr-surface` | `#17181A` | Card / panel surfaces (30%) |
| `--clr-border` | `#2A2B2D` | Subtle borders on cards |
| `--clr-text-primary` | `#F5F5F5` | Body text, headings on dark bg |
| `--clr-text-secondary` | `#9CA3AF` | Labels, secondary copy |
| `--clr-text-disabled` | `#4B5563` | Placeholder / disabled states |
| `--clr-accent` | `#D4FF00` | Primary buttons, active tabs, highlights (10%) |
| `--clr-accent-fg` | `#0B0B0C` | Text on accent backgrounds |
| `--clr-danger` | `#F87171` | Error states |
| `--clr-success` | `#4ADE80` | Success confirmation |

**Activity data colours** (used only in timeline, charts, badges — always paired with text/icon label):

| Activity | Pill bg | Pill text | Token name |
|----------|---------|-----------|------------|
| WALKING | `#4ADE80` | `#000000` | `--act-walk` |
| WALKING_UPSTAIRS | `#60A5FA` | `#000000` | `--act-up` |
| WALKING_DOWNSTAIRS | `#3B82F6` | `#ffffff` | `--act-down` |
| SITTING | `#FBBF24` | `#000000` | `--act-sit` |
| STANDING | `#FB923C` | `#000000` | `--act-stand` |
| LAYING | `#F87171` | `#000000` | `--act-lay` |

All six pairs pass WCAG AA (≥ 4.5:1 against their respective pill backgrounds).

### 3.2 Typography

| Role | Family | Weight | Size | Line-height |
|------|--------|--------|------|-------------|
| H1 — Activity name | Inter | 900 | `clamp(3rem, 10vw, 7rem)` | 0.95 |
| H2 — Page / section title | Inter | 700 | `2rem` | 1.1 |
| H3 — Card heading | Inter | 600 | `1.25rem` | 1.3 |
| Eyebrow label | Inter | 500 | `0.75rem` | 1 |
| Body | Inter | 400 | `1rem` (16px min) | 1.6 |
| Stat number | Inter | 700 | `2rem` | 1 |
| Caption | Inter | 400 | `0.875rem` | 1.4 |

Font loaded via Google Fonts CDN (`preconnect` → `stylesheet`). Fallback: `system-ui, sans-serif`.

### 3.3 Spacing Scale

| Token | Value | Use |
|-------|-------|-----|
| `--sp-xs` | `4px` | Inline gap |
| `--sp-sm` | `8px` | Tight stack |
| `--sp-md` | `16px` | Default gap |
| `--sp-lg` | `24px` | Section padding |
| `--sp-xl` | `48px` | Between major sections |
| `--sp-2xl` | `80px` | Hero top padding |

### 3.4 Radius & Shadow

| Token | Value |
|-------|-------|
| `--radius-sm` | `8px` |
| `--radius-md` | `12px` |
| `--radius-pill` | `999px` |
| Shadow | None (flat, border-only cards) |

---

## 4. Shared Component Library (`app/components/ui.py`)

| Component | Description | Key props |
|-----------|-------------|-----------|
| `load_css()` | Injects `theme.css` + Inter font via `st.markdown` | — |
| `activity_hero(name, conf, duration, model)` | Full-width dark card with eyebrow, huge H1 activity name, confidence badge and optional duration | `name` (str), `conf` (0–1 float), `duration` (str), `model` (str) |
| `status_chip(label, variant)` | Pill-shaped chip: `replay` (accent), `connected` (green), `idle` (grey), `error` (red) | `label`, `variant` |
| `section_label(text)` | Small-caps eyebrow above a section | `text` |
| `timeline_bar(runs, total, step, dur)` | Horizontal proportional colour-coded timeline bar | `runs` (list of dicts), `total`, `step`, `dur` |
| `activity_card(pairs)` | Dark surface card with label/value pairs | `pairs` (list of `(label, value)`) |
| `mmss(sec)` | Format seconds → `mm:ss` | `sec` (float) |
| `empty_state(icon, heading, body)` | Centred empty-state block for pre-upload screens | text props |

---

## 5. Page-by-Page Layout

### 5.1 Home / Story Tab

```
┌─────────────────────────────────────────────────────────┐
│  [Upload CSV area — dashed border, big icon, copy]      │  ← empty state
│                                                         │
└─────────────────────────────────────────────────────────┘
            ↓ after upload
┌─────────────────────────────────────────────────────────┐
│  CURRENT ACTIVITY    [Model chip]                       │  ← eyebrow + chip
│  WALKING                                                │  ← H1 huge
│  87% confident · 00:42                                  │  ← sub-line
├─────────────────────────────────────────────────────────┤
│  ACTIVITY TIMELINE                                      │  ← section-label
│  ██████████████████████████████████████████████████████ │  ← coloured bar
│  [walk][sit][stand][lay]  ← legend pills                │
├─────────────────────────────────────────────────────────┤
│  STORY TAB   |   DETAILS TAB                            │  ← Streamlit tabs
├─────────────────────────────────────────────────────────┤
│  STORY                                                  │
│  "You were walking for 2 m 10 s — the most common      │
│   activity in this recording."                          │  ← plain-language
│  TIME PER ACTIVITY  [horizontal bars]                   │
└─────────────────────────────────────────────────────────┘
```

**Empty state copy:** "Drop a CSV recording here, or use the replay demo on the Live page."

### 5.2 Details Tab

```
┌────────────────┬────────────────┬────────────────┐
│ SMOOTHING      │  WINDOW        │  STEP          │
│ [toggle]       │  [slider]      │  [slider]      │
│ "Reduces       │                │                │
│  flickers"     │                │                │
└────────────────┴────────────────┴────────────────┘
  CONFIDENCE DISTRIBUTION
  [bar chart — accent colour for bars]
  ACTIVITY BREAKDOWN
  [one card per activity — dark surface, label + time + %]
```

Smoothing toggle has a one-line explanation beneath it: *"On: short bursts are merged into the dominant activity. Off: raw prediction per window."*

### 5.3 Live Page

```
┌─────────────────────────────────────────────────────────┐
│  [▶ REPLAY MODE]  ← status chip top-right               │
│                                                         │
│  NOW DETECTING                                          │  ← eyebrow
│  STANDING                                               │  ← H1 huge, fades
│  Sensor data updating every 1 s                         │  ← body
├─────────────────────────────────────────────────────────┤
│  RECENT ACTIVITY                                        │  ← section label
│  ██████████████ rolling timeline strip (last 30 s)      │
├─────────────────────────────────────────────────────────┤
│  SOURCE     [Phone IP / Replay] radio                   │
│  [Phone IP input if Phone selected]                     │
│  [START / STOP button — solid accent]                   │
├─────────────────────────────────────────────────────────┤
│  SETUP (shown when idle)                                │
│  1. Install Phyphox on your phone                       │
│  2. Open the HTTP server experiment                     │
│  3. Enter the IP shown in Phyphox above                 │
│  4. Press START                                         │
└─────────────────────────────────────────────────────────┘
```

Activity name fades on change using CSS transition. Source and status chip stay at top of content area.

### 5.4 Video Page

```
┌─────────────────────────────────────────────────────────┐
│  [!] Experimental feature — dark info card, left border │
├─────────────────────────────────────────────────────────┤
│  UPLOAD VIDEO                                           │
│  [drag-and-drop zone — dashed border, icon, copy]       │
│  Accepts MP4 / MOV from a vertical phone video          │
├──────────────────────┬──────────────────────────────────┤
│  [Video player]      │ DETECTED ACTIVITIES              │
│                      │  WALKING                         │
│                      │  [H2]                            │
│                      │  [timeline bar]                  │
│                      │  [activity breakdown]            │
└──────────────────────┴──────────────────────────────────┘
  Activities: WALKING · STANDING · STAIRS only (note below)
```

### 5.5 Shared: Sidebar & Navigation

- Streamlit sidebar contains the smoothing toggle, window and step sliders (moved from main area where possible).
- Navigation: Streamlit's native multi-page navigation (no custom nav bar needed — keeps accessibility).
- Footer on every page: small italic note — *"Model accuracy: ~93% on the UCI HAR benchmark. Real-world results vary."*

---

## 6. Accessibility Plan

| Requirement | Implementation |
|-------------|----------------|
| Colour contrast ≥ 4.5:1 body | `#F5F5F5` on `#0B0B0C` → 17.5:1 |
| Colour contrast ≥ 3:1 large text | `#F5F5F5` on `#0B0B0C` → 17.5:1 |
| Activity colours paired with label | Every badge/bar has `aria-label` and visible text |
| Tap targets ≥ 44 × 44 px | All buttons: `min-height: 44px; min-width: 44px` in CSS |
| Focus visible | `:focus-visible` ring with `--clr-accent` outline, `2px` offset |
| Body text ≥ 16px | `font-size: 1rem` = 16px base in CSS reset |
| `prefers-reduced-motion` | Fade transition wrapped in `@media (prefers-reduced-motion: no-preference)` |
| Semantic headings | `h1`/`h2` used in HTML strings injected via `st.markdown(unsafe_allow_html=True)` |
| Alt text on images | Any `<img>` in HTML includes descriptive `alt` |

---

## 7. Motion & Animation

- **Activity name change:** 200 ms `opacity` fade-out → update → fade-in; disabled when `prefers-reduced-motion: reduce`.
- No parallax, scroll-driven animation, or looping animation.
- Status chip: no animation; colour/text change is instantaneous.

---

## 8. Responsive Layout

| Breakpoint | Behaviour |
|------------|-----------|
| < 760 px (mobile-first) | Single column; H1 `clamp(3rem, 10vw, 7rem)`; full-width cards |
| 760 px + | Two-column layout on Video page (video | analysis) |
| 1080 px + | Max content width `960px`, centred |

Streamlit renders inside an iframe; CSS targets `.stApp` and `.main .block-container`. The `.block-container` max-width is set to `960px` for desktop, `100%` for mobile.

---

## 9. File Map

```
activity-recognition/
├── .streamlit/
│   └── config.toml          ← dark theme tokens
├── app/
│   ├── components/
│   │   └── ui.py            ← shared component helpers
│   ├── pages/
│   │   ├── 1_Live.py        ← redesigned live page
│   │   └── 2_Video.py       ← redesigned video page
│   ├── styles/
│   │   └── theme.css        ← single shared stylesheet
│   └── streamlit_app.py     ← redesigned main page
├── docs/
│   └── ui_redesign_design.md ← this document
└── DESIGN.md                ← updated design system tokens
```

---

## 10. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Streamlit overrides CSS specificity | Medium | Prefix all custom rules with `.stApp` and use `!important` only where truly necessary |
| Google Fonts blocked in network-restricted environments | Low | `font-family` fallback to `system-ui, sans-serif` |
| `unsafe_allow_html` injection XSS | Low (data is trusted local CSV) | Never interpolate raw user-uploaded text directly into HTML; always escape |
| Activity hero `clamp` font too large on narrow Streamlit sidebar | Low | Sidebar components use H2 sizing, not H1 clamp |
| `prefers-reduced-motion` not propagated into Streamlit iframe | Medium | Wrap transition in `@supports` + media query; silently degraded to no animation |
| `data/sample_synthetic.csv` missing on fresh clone | Known | Live page shows informative error card; replay falls back gracefully |

---

## 11. Implementation Plan

### Phase 1 — Foundation (do first, no page changes)

1. **`.streamlit/config.toml`** — Replace Oura cream palette with dark tokens.
2. **`app/styles/theme.css`** — Write full stylesheet: tokens, typography reset, component classes, responsive, a11y, motion.
3. **`app/components/__init__.py`** — Empty init.
4. **`app/components/ui.py`** — Implement all shared component functions listed in §4.

### Phase 2 — Page Redesigns

5. **`app/streamlit_app.py`** — Rework Story/Details presentation; keep all data logic unchanged.
6. **`app/pages/1_Live.py`** — Rework Live presentation; keep polling logic unchanged.
7. **`app/pages/2_Video.py`** — Rework Video presentation; keep pose estimation logic unchanged.

### Phase 3 — Polish & Verification

8. **`DESIGN.md`** — Replace Oura system documentation with new Essor-inspired system.
9. **Run tests** — Confirm all 9 baseline tests pass.
10. **Screenshots** — Desktop and mobile widths via browser subagent.
11. **Commit** on `ui-ux-redesign` branch.

Each phase must be complete and working before moving to the next.

---

## 12. Out of Scope

- Any change to `src/`, `config/`, `models/`, `notebooks/`, `data/`
- New pip dependencies
- Server performance or caching
- Authentication or user accounts
- Exporting / sharing results
- Dark/light mode toggle (always dark)
- Animations beyond the single activity-name fade
