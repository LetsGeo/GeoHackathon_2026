"""
Design of the app: colours, fonts, seismic colour maps, the CSS for every
widget, card styling, and icons. app.py and wells.py import from here, so the
look can be changed in one place without touching the app's logic.

Panel draws each widget inside its own shadow DOM, so page-wide CSS doesn't
reach it: each widget gets its stylesheet passed in (stylesheets=[...]).
Colours are CSS variables (var(--bg) etc.) so light / dark mode only swaps the
variables; THEMES holds both sets.
"""
import base64

# Two colour themes. The app's CSS uses these as CSS variables (var(--bg) etc.),
# so switching theme only swaps the variables; the Bokeh plots are restyled
# from Python by style_figure().
THEMES = {
    # deep navy, with a faint violet / teal glow in the page background
    "dark": {
        "bg": "#0a0f1e", "sidebar": "#0d1426", "card": "#111a2e", "card2": "#162038",
        "border": "#1f2a45", "border-strong": "#2d3b5d",
        "text": "#e8eef9", "text2": "#aab5cc", "muted": "#7584a3",
        "accent": "#4fd1c5", "accent-ink": "#04201d", "accent-soft": "rgba(79,209,197,.13)",
        "accent-hover": "#74e3d8", "accent2": "#8b7cf6", "nav-active": "rgba(79,209,197,.10)",
        "hover": "rgba(255,255,255,.05)", "stripe": "rgba(255,255,255,.025)",
        "err": "#f4a08a", "err-soft": "rgba(244,160,138,.12)", "err-border": "#7a3b2e",
        "ok": "#4ade80", "ok-soft": "rgba(74,222,128,.12)",
        "warn": "#fbbf24", "warn-soft": "rgba(251,191,36,.12)",
        "hl-start": "#0f9f94", "hl-end": "#4f46e5",  # highlighted summary card: teal into indigo
        "shadow": "inset 0 1px 0 rgba(255,255,255,.03), 0 12px 32px -16px rgba(0,0,0,.7)",
        "bg-grad": ("radial-gradient(1100px 620px at 88% -8%, rgba(99,102,241,.16), transparent 62%), "
                    "radial-gradient(900px 520px at 18% -4%, rgba(79,209,197,.08), transparent 58%), #0a0f1e"),
        "sidebar-grad": "linear-gradient(180deg, #111a33 0%, #0d1426 55%, #0b1122 100%)",
    },
    # cool off-white: white cards that lift off the page with soft shadows
    "light": {
        "bg": "#f2f5fa", "sidebar": "#ffffff", "card": "#ffffff", "card2": "#f5f7fb",
        "border": "#e3e8f0", "border-strong": "#cbd4e1",
        "text": "#0f172a", "text2": "#3b4a63", "muted": "#64748b",
        "accent": "#0d9488", "accent-ink": "#ffffff", "accent-soft": "rgba(13,148,136,.10)",
        "accent-hover": "#0b7a70", "accent2": "#6366f1", "nav-active": "rgba(13,148,136,.09)",
        "hover": "rgba(15,23,42,.045)", "stripe": "rgba(15,23,42,.03)",
        "err": "#b4441f", "err-soft": "rgba(180,68,31,.08)", "err-border": "#e3a48c",
        "ok": "#15803d", "ok-soft": "rgba(21,128,61,.09)",
        "warn": "#b45309", "warn-soft": "rgba(180,83,9,.09)",
        "hl-start": "#0d9488", "hl-end": "#4f46e5",
        "shadow": "0 1px 2px rgba(15,23,42,.04), 0 10px 28px -14px rgba(15,23,42,.16)",
        "bg-grad": ("radial-gradient(1100px 620px at 88% -8%, rgba(99,102,241,.07), transparent 62%), "
                    "radial-gradient(900px 520px at 18% -4%, rgba(13,148,136,.05), transparent 58%), #f2f5fa"),
        "sidebar-grad": "linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%)",
    },
}


DEFAULT_THEME = "dark"


def theme_css():
    """CSS defining the theme variables: dark by default, light when the page has class "light"."""
    dark = "; ".join(f"--{k}: {v}" for k, v in THEMES["dark"].items())
    light = "; ".join(f"--{k}: {v}" for k, v in THEMES["light"].items())
    return f":root {{ {dark}; }}\nhtml.light {{ {light}; }}"


BG = "var(--bg)"


SIDEBAR_BG = "var(--sidebar)"


CARD = "var(--card)"


CARD_2 = "var(--card2)"


BORDER = "var(--border)"


BORDER_STRONG = "var(--border-strong)"


TEXT = "var(--text)"


TEXT_2 = "var(--text2)"


MUTED = "var(--muted)"


ACCENT = "var(--accent)"


ACCENT_INK = "var(--accent-ink)"


HARDEN = "#f5a524"  # amplitude increase (used by the analysis difference plots)


SOFTEN = "#4a8fe7"  # amplitude decrease


# The font (loaded from Google Fonts; the others are fall-backs if it can't load)
FONT_IMPORT_CSS = ("@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:"
                   "wght@400;500;600;700;800&display=swap');")
SANS = "'Plus Jakarta Sans', Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"


PLOT_FONT = "Plus Jakarta Sans"  # axis and title text on the Bokeh plots


MONO = "Consolas, 'Courier New', monospace"


# Seismic colour maps as (position, colour) stops, from trough (0.0, negative)
# to peak (1.0, positive). "Seismic RWB" was sampled from a reference colour bar.
SEISMIC_CMAPS = {
    "Seismic RWB": [
        (0.000, "#d8140a"),  # red
        (0.024, "#f44601"),
        (0.049, "#fc8000"),  # orange
        (0.073, "#ffbb03"),
        (0.122, "#f9f311"),  # bright yellow
        (0.171, "#eed942"),
        (0.244, "#dcbb71"),  # tan
        (0.341, "#e8d2a6"),
        (0.415, "#f1e5cb"),  # cream
        (0.490, "#fdfaf2"),  # white (zero)
        (0.537, "#edeeeb"),
        (0.585, "#d2d5d2"),
        (0.634, "#bbbbbd"),
        (0.683, "#a6a3a6"),
        (0.732, "#8a8a8a"),
        (0.780, "#716e73"),
        (0.829, "#595959"),
        (0.878, "#444144"),
        (0.927, "#282526"),
        (1.000, "#040200"),  # black
    ],
    "Seismic BW": [
        (0.0, "#ffffff"),
        (0.5, "#808080"),
        (1.0, "#000000"),
    ],
}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def cmap_palette(name, n=256):
    """Evenly spaced list of n hex colours for a colour map, ready for
    HoloViews/Bokeh plots, e.g. hv.Image(...).opts(cmap=cmap_palette(name))."""
    stops = SEISMIC_CMAPS[name]
    out = []
    for i in range(n):
        t = i / (n - 1)
        for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
            if p0 <= t <= p1:
                f = 0 if p1 == p0 else (t - p0) / (p1 - p0)
                a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
                rgb = [round(a[k] + (b[k] - a[k]) * f) for k in range(3)]
                out.append("#%02x%02x%02x" % tuple(rgb))
                break
    return out


DEFAULT_CMAP = "Seismic RWB"


# Shared stylesheet for every HTML pane (Panel renders panes in shadow DOM,
# so styles have to be passed to each pane rather than set globally).
BASE_CSS = f"""
:host {{ color: {TEXT}; font-family: {SANS}; font-size: 14px; }}
* {{ box-sizing: border-box; }}
.mono {{ font-family: {MONO}; }}
.meta {{ font-size: 12px; color: {MUTED}; }}
.label {{ font-size: 15.5px; color: {TEXT}; font-weight: 700; letter-spacing: -.01em; }}  /* card titles */
.title {{ font-weight: 700; font-size: 16px; margin: 0; letter-spacing: -.015em; }}
.eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {MUTED}; }}
.note {{ color: {MUTED}; font-size: 13px; line-height: 1.55; margin: 0; }}
.head {{ display: flex; justify-content: space-between; align-items: baseline; width: 100%; }}

/* sidebar */
.sidebar {{ padding: 22px 16px 6px; }}
.brand {{ display: flex; gap: 11px; align-items: center; padding: 0 4px 6px; }}
.brand-mark {{ width: 38px; height: 38px; border-radius: 12px; background: #045373;
               display: grid; place-items: center; flex-shrink: 0; }}
.brand-name {{ font-weight: 800; font-size: 14.5px; line-height: 1.2; letter-spacing: -.015em; }}
.brand-sub {{ color: {MUTED}; font-size: 11.5px; margin-top: 2px; }}
.nav-label {{ font-size: 10.5px; letter-spacing: .12em; color: {MUTED};
              font-weight: 700; margin: 22px 10px 8px; text-transform: uppercase; opacity: .85; }}
.nav-item {{ position: relative; display: flex; align-items: center; gap: 12px; padding: 10px 12px;
             margin: 2px 0; border-radius: 12px; color: {TEXT_2}; font-size: 14px; font-weight: 500;
             transition: background .15s ease, color .15s ease; }}
.nav-item svg {{ width: 19px; height: 19px; flex-shrink: 0; opacity: .85; }}
.nav-item .nav-dot {{ margin-left: auto; width: 7px; height: 7px; border-radius: 50%; background: {MUTED};
                      opacity: .5; transition: background .2s, opacity .2s; }}
.nav-item .nav-dot.on {{ background: {ACCENT}; opacity: 1; box-shadow: 0 0 0 3px var(--accent-soft); }}
.nav-item.active {{ color: {TEXT}; font-weight: 700; background: var(--nav-active); }}
.nav-item.active svg {{ color: {ACCENT}; opacity: 1; }}
/* accent bar on the sidebar's left edge beside the current page */
.nav-item.active::before {{ content: ""; position: absolute; left: -16px; top: 9px; bottom: 9px; width: 4px;
                            border-radius: 0 4px 4px 0;
                            background: linear-gradient(180deg, {ACCENT}, var(--accent2)); }}

/* header */
.crumbs {{ font-size: 12px; color: {MUTED}; font-weight: 600; letter-spacing: .02em; }}
.crumbs .sep {{ color: {BORDER_STRONG}; margin: 0 6px; }}
h1 {{ font-size: 30px; font-weight: 800; margin: 4px 0 0; letter-spacing: -.03em; line-height: 1.15; }}
.page-sub {{ color: {MUTED}; font-size: 14px; margin-top: 6px; line-height: 1.5; }}

/* status pills (header, overview table) */
.pill {{ display: inline-flex; align-items: center; gap: 7px; font-size: 12px; font-weight: 600;
         padding: 5px 11px; border-radius: 999px; white-space: nowrap;
         background: var(--card2); color: {TEXT_2}; border: 1px solid {BORDER}; }}
.pill::before {{ content: ""; width: 7px; height: 7px; border-radius: 50%; background: {MUTED}; }}
.pill.ok {{ background: var(--ok-soft); color: var(--ok); border-color: transparent; }}
.pill.ok::before {{ background: var(--ok); }}
.pill.accent {{ background: var(--accent-soft); color: {ACCENT}; border-color: transparent; }}
.pill.accent::before {{ background: {ACCENT}; }}
.pill.warn {{ background: var(--warn-soft); color: var(--warn); border-color: transparent; }}
.pill.warn::before {{ background: var(--warn); }}
.pill.err {{ background: var(--err-soft); color: var(--err); border-color: transparent; }}
.pill.err::before {{ background: var(--err); }}

/* overview: what's loaded, as a table */
table.overview {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
table.overview th {{ text-align: left; font-size: 11px; font-weight: 700; letter-spacing: .08em;
                     text-transform: uppercase; color: {MUTED}; padding: 4px 12px 10px;
                     border-bottom: 1px solid {BORDER}; }}
table.overview td {{ padding: 13px 12px; border-bottom: 1px solid {BORDER}; color: {TEXT_2}; vertical-align: middle; }}
table.overview tr:last-child td {{ border-bottom: none; }}
table.overview tr:hover td {{ background: var(--hover); }}
table.overview td.name {{ color: {TEXT}; font-weight: 600; }}
table.overview td.name small {{ display: block; font-weight: 500; color: {MUTED}; font-size: 12px; margin-top: 2px; }}
table.overview td.right {{ text-align: right; }}
.ov-icon {{ display: inline-grid; place-items: center; width: 32px; height: 32px; border-radius: 10px;
            background: var(--accent-soft); color: {ACCENT}; margin-right: 12px; vertical-align: middle; }}
.ov-icon svg {{ width: 17px; height: 17px; }}
.ov-name {{ display: flex; align-items: center; }}

/* empty states (pages that need data) */
.empty {{ display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
          gap: 10px; padding: 56px 24px 18px; }}
.empty-icon {{ width: 64px; height: 64px; border-radius: 20px; display: grid; place-items: center;
               background: linear-gradient(140deg, var(--accent-soft), rgba(99, 102, 241, .14));
               color: {ACCENT}; margin-bottom: 6px; }}
.empty-icon svg {{ width: 30px; height: 30px; }}
.empty h3 {{ margin: 0; font-size: 18px; font-weight: 700; letter-spacing: -.01em; color: {TEXT}; }}
.empty p {{ margin: 0; max-width: 440px; color: {MUTED}; font-size: 13.5px; line-height: 1.55; }}

/* summary cards along the top of the overview */
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; }}
.stat {{ position: relative; background: {CARD}; border: 1px solid {BORDER}; border-radius: 18px;
         padding: 18px 20px 16px; min-height: 132px; box-shadow: var(--shadow); overflow: hidden;
         transition: transform .18s ease, border-color .18s ease; }}
.stat:hover {{ transform: translateY(-2px); border-color: {BORDER_STRONG}; }}
.stat-label {{ font-size: 13px; font-weight: 600; color: {TEXT_2}; padding-right: 36px; }}
.stat-value {{ font-size: 38px; font-weight: 800; letter-spacing: -.035em; line-height: 1.1; margin: 12px 0 8px;
               color: {TEXT}; font-variant-numeric: tabular-nums; }}
.stat-value small {{ font-size: 15px; font-weight: 600; letter-spacing: 0; margin-left: 4px; color: {TEXT_2}; }}
.stat-detail {{ font-size: 12px; color: {MUTED}; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
.stat-detail .chip {{ font-size: 11px; padding: 1px 7px; border-radius: 6px; border: 1px solid {BORDER_STRONG}; }}
/* faint glow in the corner of each card */
.stat::after {{ content: ""; position: absolute; right: -40px; bottom: -60px; width: 160px; height: 160px;
                border-radius: 50%; background: radial-gradient(closest-side, var(--accent-soft), transparent);
                pointer-events: none; }}
.stat.hl::after {{ background: radial-gradient(closest-side, rgba(255, 255, 255, .18), transparent); }}
.stat-arrow {{ position: absolute; top: 16px; right: 16px; width: 30px; height: 30px; border-radius: 50%;
               border: 1px solid {BORDER_STRONG}; display: grid; place-items: center; color: {TEXT}; }}
.stat-arrow svg {{ width: 14px; height: 14px; }}
.stat.hl {{ background: linear-gradient(140deg, var(--hl-start) 0%, var(--hl-end) 100%); border-color: transparent; }}
.stat.hl .stat-label, .stat.hl .stat-value, .stat.hl .stat-value small {{ color: #ffffff; }}
.stat.hl .stat-detail {{ color: rgba(255, 255, 255, .78); }}
.stat.hl .stat-detail .chip, .stat.hl .stat-arrow {{ border-color: rgba(255, 255, 255, .45); color: #ffffff; }}

/* progress track (time-lapse, disabled)
.track {{ height: 3px; background: {BORDER}; border-radius: 2px; overflow: hidden; }}
.fill {{ height: 100%; background: {ACCENT}; }}
*/

/* colourbar */
.cbar-wrap {{ display: flex; flex-direction: column; gap: 6px; width: 100%; padding: 0 14px; }}
.cbar {{ position: relative; height: 24px; border: 1px solid #5a5a5a; }}
.cbar .div {{ position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(90, 90, 90, .85); }}
.cbar .mask {{ position: absolute; top: 0; bottom: 0; background: #4b6470; opacity: .92; }}
.cbar-ticks {{ position: relative; height: 14px; font-size: 11px; color: {TEXT_2}; font-variant-numeric: tabular-nums; }}
.cbar-ticks span {{ position: absolute; transform: translateX(-50%); white-space: nowrap; }}

/* seismic placeholders */
.survey-tag {{ font-size: 11.5px; font-weight: 600; padding: 4px 12px; border-radius: 999px;
               border: 1px solid {BORDER_STRONG}; color: {TEXT_2}; }}
.survey-tag.monitor {{ border-color: {ACCENT}; color: {ACCENT}; }}
.seis {{ height: 360px; margin-top: 12px; border: 1px dashed {BORDER_STRONG}; border-radius: 16px;
         display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
         background: repeating-linear-gradient(0deg, transparent 0 13px, var(--stripe) 13px 14px), {CARD_2};
         color: {MUTED}; text-align: center; }}
.seis svg {{ color: {TEXT_2}; opacity: .7; }}
.seis .big {{ color: {TEXT_2}; font-size: 13.5px; font-weight: 500; }}
.seis.error {{ border-color: var(--err-border); }}
.seis.error .big {{ color: var(--err); }}
.sub {{ font-size: 12.5px; color: {MUTED}; margin-top: 4px; }}
.credit {{ font-size: 11px; color: {MUTED}; line-height: 1.45; margin-top: 8px; }}
.status {{ font-size: 12.5px; color: {MUTED}; line-height: 1.5; }}
.status.ok {{ color: {ACCENT}; }}
.status.err {{ color: var(--err); }}
/* analysis */
.plot-caption {{ font-size: 12px; color: {TEXT}; padding-bottom: 2px; }}
.tiles {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
.tile {{ background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 16px; padding: 12px 14px; }}
.tile-label {{ font-size: 11px; color: {MUTED}; }}
.tile-value {{ font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; color: {TEXT}; margin-top: 4px; }}
.tile-detail {{ font-size: 11px; color: {TEXT_2}; margin-top: 2px; }}
.summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }}
.sum-col {{ background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 16px; padding: 12px 16px; }}
.sum-lang {{ font-size: 11px; letter-spacing: .08em; font-weight: 600; color: {ACCENT}; text-transform: uppercase; }}
.sum-col h4 {{ font-size: 12.5px; margin: 12px 0 4px; color: {TEXT}; }}
.sum-col p {{ font-size: 13px; line-height: 1.55; color: {TEXT_2}; margin: 0 0 6px; }}
.sum-col p.sum-note {{ font-size: 11.5px; color: {MUTED}; font-style: italic; margin-top: 10px; }}
.sum-col p.ai-placeholder {{ color: {MUTED}; font-style: italic; }}
.ai-badge {{ font-size: 10.5px; font-weight: 500; padding: 1px 8px; border-radius: 999px;
             border: 1px solid {BORDER_STRONG}; color: {MUTED}; margin-left: 6px; vertical-align: 1px; }}
/* Gemma horizons: the three suggested options */
.gopts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px;
           font-family: {SANS}; }}  /* (set here: Bokeh's default font would win inside the component) */
.gopt {{ position: relative; background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 14px;
         padding: 12px 14px 13px 16px; cursor: pointer; user-select: none;
         transition: border-color .15s ease, background-color .15s ease; }}
.gopt:hover {{ border-color: {BORDER_STRONG}; background: var(--hover); }}
.gopt:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
.gopt.sel, .gopt.sel:hover {{ border-color: {ACCENT}; background: var(--accent-soft); }}
.gopt-letter {{ display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 700; color: {MUTED};
                letter-spacing: .08em; text-transform: uppercase; padding-right: 72px; }}
.gopt-name {{ font-size: 14px; font-weight: 700; color: {TEXT}; margin-top: 7px; }}
.gopt-swatch {{ width: 18px; height: 4px; border-radius: 2px; flex-shrink: 0; }}
.gopt-desc {{ font-size: 12.5px; color: {TEXT_2}; line-height: 1.5; margin-top: 4px; }}
.gopt-tag {{ display: none; position: absolute; top: 11px; right: 13px; font-size: 10.5px; font-weight: 700;
             color: {ACCENT}; letter-spacing: .06em; text-transform: uppercase; }}
.gopt.sel .gopt-tag {{ display: block; }}
/* the Gemma horizons heading: logo and title centred on one line */
.gemma-head {{ display: flex; align-items: center; gap: 10px; }}
.gemma-logo {{ display: block; width: 24px; height: 24px; border-radius: 7px; flex-shrink: 0; }}
.morph-ends {{ display: flex; justify-content: space-between;
               font-size: 11px; color: {TEXT_2}; }}
"""


CARD_STYLES = {
    "background": CARD,
    "border": f"1px solid {BORDER}",
    "border-radius": "18px",
    "padding": "18px 20px 24px",  # a little more room under the last item
    "box-shadow": "var(--shadow)",
    "box-sizing": "border-box",
}


BTN_BASE_CSS = f"""
.bk-btn {{
  font-family: {SANS}; font-size: 13.5px; font-weight: 600;
  border-radius: 999px; padding: 9px 18px; cursor: pointer;
  background: transparent !important; color: {TEXT} !important;
  border: 1px solid {BORDER_STRONG} !important; box-shadow: none !important;
  transition: background-color .12s ease, color .12s ease;
}}
.bk-btn:hover {{ background: var(--hover) !important; }}
.bk-btn:focus {{ outline: none; }}
.bk-btn:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 2px; }}
.bk-btn[disabled] {{ opacity: .42; cursor: not-allowed; }}
.bk-btn[disabled]:hover {{ background: transparent !important; }}
"""


PRIMARY_BTN_CSS = f"""
.bk-btn {{ background: {ACCENT} !important; border-color: {ACCENT} !important; color: {ACCENT_INK} !important; }}
.bk-btn:hover {{ background: var(--accent-hover) !important; border-color: var(--accent-hover) !important; }}
.bk-btn[disabled]:hover {{ background: {ACCENT} !important; border-color: {ACCENT} !important; }}
"""


# a quiet button with no outline (sidebar theme switch, small "open" actions)
GHOST_BTN_CSS = f"""
.bk-btn {{ border-color: transparent !important; color: {TEXT_2} !important; }}
.bk-btn:hover {{ color: {TEXT} !important; border-color: transparent !important; }}
"""


# Light / dark mode switch (on = light). The knob shows a moon in dark mode and
# slides right to show a sun in light mode, with faint icons in the track.
_MOON_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M20.5 14.2A8.6 8.6 0 1 1 9.8 3.5a6.8 6.8 0 0 0 10.7 10.7z'/></svg>")
_SUN_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2.2' stroke-linecap='round'><circle cx='12' cy='12' r='4.2' fill='black'/>"
            "<path d='M12 2v2.4M12 19.6V22M4.9 4.9l1.7 1.7M17.4 17.4l1.7 1.7M2 12h2.4M19.6 12H22M4.9 19.1l1.7-1.7"
            "M17.4 6.6l1.7-1.7'/></svg>")
# The switch's parts are .body / .bar / .knob (on = :host(.active)) in Bokeh 3.4, and
# .bk-body / .bk-bar / .bk-knob (on = :host(.bk-active)) in newer versions; the CSS
# is written for the newer names and _both_names() adds the 3.4 ones.
THEME_SWITCH_CSS = """
:host { gap: 0 !important; }
.bk-label, .bk-icon { display: none !important; }
.bk-body { position: relative; width: 58px !important; height: 30px !important; cursor: pointer; }
.bk-bar {
  position: absolute !important; inset: 0 !important; top: 0 !important; height: 100% !important;
  box-sizing: border-box; border-radius: 15px !important;
  background: var(--card2) !important; border: 1px solid var(--border-strong);
  transition: background-color .25s ease, border-color .25s ease;
}
/* faint sun (right) and moon (left) in the track; the knob covers the current one */
.bk-bar::before, .bk-bar::after {
  content: ""; position: absolute; top: 50%; width: 14px; height: 14px; margin-top: -7px;
  background: var(--muted); opacity: .8;
  -webkit-mask: var(--icon) center / contain no-repeat; mask: var(--icon) center / contain no-repeat;
}
.bk-bar::before { right: 8px; --icon: url("SUN"); }
.bk-bar::after { left: 8px; --icon: url("MOON"); }
.bk-knob {
  position: absolute !important; margin: 0 !important; cursor: pointer;
  top: 3px !important; left: 3px !important; width: 24px !important; height: 24px !important;
  border-radius: 50% !important; background: var(--accent) !important; z-index: 1;
  box-shadow: 0 1px 3px rgba(0, 0, 0, .35);
  /* it slides with transform (a 'left' set here would beat the :host(...) rule below) */
  transition: transform .25s cubic-bezier(.3, .7, .4, 1), background-color .25s ease !important;
}
.bk-knob::after {
  content: ""; position: absolute; inset: 5px; background: var(--accent-ink);
  -webkit-mask: url("MOON") center / contain no-repeat; mask: url("MOON") center / contain no-repeat;
}
:host(.bk-active) .bk-bar { background: var(--accent-soft) !important; border-color: var(--accent); }
:host(.bk-active) .bk-knob { transform: translateX(28px); }
:host(.bk-active) .bk-knob::after { -webkit-mask-image: url("SUN"); mask-image: url("SUN"); }
.bk-body:hover .bk-bar { border-color: var(--accent); }
.bk-knob:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.bk-knob:focus { outline: none; }
"""


def _both_names(css):
    """Write each switch rule for both Bokeh naming schemes (see THEME_SWITCH_CSS)."""
    old = (css.replace(":host(.bk-active)", ":host(.active)")
              .replace(".bk-body", ".body").replace(".bk-bar", ".bar").replace(".bk-knob", ".knob"))
    return css + old


THEME_SWITCH_CSS = _both_names(THEME_SWITCH_CSS).replace("SUN", _SUN_SVG).replace("MOON", _MOON_SVG)


# a toggle button that opens a panel (Display, Layers): highlighted while open
PANEL_TOGGLE_CSS = f"""
.bk-btn.bk-active {{ background: var(--accent-soft) !important; color: {ACCENT} !important;
                     border-color: transparent !important; }}
"""


# Panel's download button (FileDownload) has a fixed height, no padding and its
# text pinned to the top; this makes it look and line up like the other buttons.
# Use it together with BTN_BASE_CSS and height=DOWNLOAD_BTN_HEIGHT.
DOWNLOAD_BTN_HEIGHT = 39


DOWNLOAD_BTN_CSS = f"""
.bk-btn {{
  display: flex !important; align-items: center; justify-content: center;
  width: 100%; height: {DOWNLOAD_BTN_HEIGHT}px !important; padding: 0 18px !important;
  line-height: 1 !important; box-sizing: border-box;
}}
/* the label is inside a link that Panel stretches to the button's full height
   with the text at its top: centre the text within it */
.bk-btn > a {{
  display: flex !important; align-items: center; justify-content: center;
  height: 100% !important; width: 100%; line-height: 1 !important; text-decoration: none; color: inherit;
}}
"""


# Segmented control: the options sit in one rounded track, the chosen one lit up
TOGGLE_CSS = f"""
.bk-btn-group {{ gap: 2px; padding: 3px; border-radius: 999px; background: {CARD_2};
                 border: 1px solid {BORDER}; box-sizing: border-box; }}
.bk-btn-group > .bk-btn {{
  font-family: {SANS}; font-size: 13px; font-weight: 600; padding: 6px 14px;
  background: transparent !important; color: {TEXT_2} !important;
  border: 1px solid transparent !important; box-shadow: none !important; cursor: pointer;
  border-radius: 999px !important; margin: 0 !important; white-space: nowrap;
  transition: background-color .12s ease, color .12s ease;
}}
.bk-btn-group > .bk-btn:hover {{ color: {TEXT} !important; background: var(--hover) !important; }}
.bk-btn-group > .bk-btn.bk-active {{ background: var(--accent-soft) !important; color: {ACCENT} !important;
                                     border-color: rgba(79, 209, 197, .35) !important; }}
.bk-btn:focus {{ outline: none; }}
.bk-btn:focus-visible {{ outline: 2px solid {ACCENT}; outline-offset: 1px; }}
.bk-btn[disabled] {{ opacity: .42; cursor: not-allowed; }}
"""


INPUT_CSS = f"""
:host {{ --input-bg: {CARD_2}; }}
label, .bk-input-group > label {{ color: {MUTED}; font-family: {SANS}; font-size: 12px;
                                  display: inline-block; margin: 0 0 7px 2px; }}  /* room between label and box */
.bk-input {{
  background: {CARD_2} !important; color: {TEXT} !important;
  border: 1px solid {BORDER_STRONG} !important; border-radius: 12px !important;
  font-family: {SANS}; font-size: 13px; padding: 8px 12px;
}}
.bk-input {{ transition: border-color .15s ease, box-shadow .15s ease; }}
.bk-input:hover {{ border-color: {MUTED} !important; }}
.bk-input:focus {{ border-color: {ACCENT} !important; box-shadow: 0 0 0 3px var(--accent-soft) !important;
                   outline: none; }}
.bk-input::placeholder {{ color: {MUTED}; opacity: .8; }}
.bk-input[disabled] {{ opacity: .5; cursor: not-allowed; }}
select.bk-input option {{ background: {CARD}; color: {TEXT}; }}
"""


# Range slider drawn on top of the colourbar: invisible track, two grab handles
OVERLAY_SLIDER_CSS = f"""
.bk-slider-title {{ display: none !important; }}
.noUi-target {{ height: 24px !important; background: transparent !important;
                border: none !important; box-shadow: none !important; }}
.noUi-connects, .noUi-connect {{ background: transparent !important; }}
.noUi-handle {{ width: 8px !important; height: 34px !important; right: -4px !important; top: -5px !important;
                background: #ffffff !important; border: 1px solid {BG} !important; border-radius: 3px !important;
                box-shadow: 0 0 0 1.5px {ACCENT} !important; cursor: ew-resize; }}
.noUi-handle::before, .noUi-handle::after {{ display: none !important; }}
"""


# Play uses the same teal as the Run buttons
MORPH_PLAY_CSS = f"""
.bk-btn {{ background: {ACCENT} !important; border-color: {ACCENT} !important; color: {ACCENT_INK} !important; }}
.bk-btn:hover {{ background: var(--accent-hover) !important; border-color: var(--accent-hover) !important;
                 color: {ACCENT_INK} !important; }}
"""


MORPH_BTN_CSS = f"""
.bk-btn {{
  font-family: {SANS}; font-size: 13px; font-weight: 600; min-width: 110px;
  border-radius: 999px; padding: 8px 16px; cursor: pointer;
  background: {CARD_2} !important; color: {TEXT} !important;
  border: 1px solid {BORDER_STRONG} !important; box-shadow: none !important;
}}
.bk-btn:hover {{ border-color: {ACCENT} !important; color: {ACCENT} !important; }}
.bk-btn:focus {{ outline: none; }}
"""


ICON_GRID = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>'


ICON_COMPARE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></svg>'


ICON_WELL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="7" r="3"/><path d="M12 10v11"/></svg>'


def _icon(body):
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>')


ICON_OVERVIEW = _icon('<path d="M3 13h8V3H3zM13 21h8V11h-8zM3 21h8v-6H3zM13 3v6h8V3z"/>')
ICON_VIEWER = _icon('<path d="M2 7c3 0 3-3 6-3s3 3 6 3 3-3 6-3M2 13c3 0 3-3 6-3s3 3 6 3 3-3 6-3'
                    'M2 19c3 0 3-3 6-3s3 3 6 3 3-3 6-3"/>')
ICON_ANALYSIS = _icon('<path d="M3 3v18h18"/><path d="m7 15 4-5 3 3 5-7"/>')
ICON_DATA = _icon('<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
                  '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>')
ICON_LAYERS = _icon('<path d="m12 3 9 5-9 5-9-5z"/><path d="m3 13 9 5 9-5"/>')
ICON_SHAPE = _icon('<path d="M5 4h10l4 7-6 9H6l-3-8z"/>')
ICON_UPLOAD = _icon('<path d="M12 16V4M6 10l6-6 6 6M4 20h16"/>')


NAV_CSS = """
.nav-item[data-page] { cursor: pointer; }
.nav-item[data-page]:not(.active):hover { background: var(--hover); color: var(--text); }
"""


# Display range: amplitudes outside it are greyed out on every plot, in a slate
# colour that isn't part of either seismic colour map so it can't be mistaken
# for data. Use "rgba(0, 0, 0, 0)" to hide them completely instead.
OUT_OF_RANGE_COLOR = "#4b6470"


NAN_COLOR = "rgba(0, 0, 0, 0)"  # positions with no trace


# Both survey plots draw their shapes from ONE shared data source, so a shape
# drawn or moved on either survey shows on the other (and on the morph) at once.
# Shapes are kept per view (Inline / Crossline / Time slice) because each view
# has its own axes; they stay on screen while you step through slices.
# Coordinates: Inline view = (crossline, time ms), Crossline view = (inline,
# time ms), Time slice = map view, saved as easting / northing when the SEG-Y
# headers carry coordinates (else as crossline, inline).
SHAPE_COLOUR = "#1e90ff"


FILE_INPUT_CSS = f"""
:host {{ color: {TEXT_2}; font-family: {SANS}; font-size: 12.5px; }}
input[type=file] {{ font-family: {SANS}; font-size: 12.5px; color: {TEXT_2};
                    background: transparent !important; border: none !important; padding: 0 !important; }}
input[type=file]::file-selector-button {{
  font-family: {SANS}; font-size: 13px; font-weight: 500; margin-right: 10px;
  border-radius: 8px; padding: 8px 14px; cursor: pointer;
  background: transparent; color: {TEXT}; border: 1px solid {BORDER_STRONG};
}}
input[type=file]::file-selector-button:hover {{ background: var(--hover); }}
"""


FEATURE_COLOUR = "#ff4d6d"  # uploaded points / lines and map shapes cut by a section


# hides the up/down arrow buttons; the number can still be typed, or changed
# with the keyboard's up/down arrow keys
NO_ARROWS_CSS = ".bk-spin-btn { display: none !important; }"


ICON_SEISMIC = '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 6c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M2 12c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M2 18c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/></svg>'


def svg_icon(path_d):
    """Toolbar icon (as a data URI) from an SVG path, in the toolbar's grey."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
           'stroke="#7d8f97" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           f'<path d="{path_d}"/></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# four corners pointing outwards (enter fullscreen) / inwards (leave fullscreen)
ICON_FS_ENTER = svg_icon("M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5")
# a wavy line: the "Draw horizon" tool
ICON_HORIZON = svg_icon("M3 15c3-5 5-5 8-1s5 4 10-3M3 20h18")

# Horizons: loaded ones take these colours in turn; ones drawn in the app share one
HORIZON_COLOURS = ["#f59e0b", "#a855f7", "#22c55e", "#ec4899", "#06b6d4", "#84cc16"]
DRAWN_HORIZON_COLOUR = "#facc15"
# Gemma's three suggested horizons, previewed on the seismic before one is chosen
GEMMA_OPTION_COLOURS = ["#38bdf8", "#f472b6", "#a3e635"]

# ---------------------------------------------------------------------------
# Full-screen loading overlay shown while Run is working (loading.py)
# ---------------------------------------------------------------------------
PUMPJACK_COLOUR = "#071215"   # the pumpjack silhouette
LOADING_CSS = f"""
:host {{ display: block; height: 0; }}
.rtm-loading {{
  position: fixed; inset: 0; z-index: 100000;            /* above the sidebar, plots and tooltips */
  display: none; flex-direction: column; align-items: center; justify-content: center; gap: 22px;
  background: var(--bg); opacity: 0; transition: opacity .28s ease;
  font-family: {SANS}; color: {TEXT}; cursor: progress; outline: none;
  overscroll-behavior: contain; touch-action: none; user-select: none;
}}
.rtm-loading.show {{ opacity: 1; }}
.rtm-stage {{ position: relative; width: min(480px, 84vw); aspect-ratio: 400 / 260; }}
/* ombre glow behind the pumpjack: light teal in the middle, blending out into the page
   background with no hard edge, large enough that the whole silhouette sits in the light */
.rtm-stage::before {{ content: ""; position: absolute; left: 50%; top: 50%; width: 150%; height: 135%;
                      transform: translate(-50%, -50%); border-radius: 50%; pointer-events: none;
                      background: radial-gradient(closest-side, #f2fffc 0%, #c9f3ec 30%, #8fe0d5 52%,
                                  rgba(79, 209, 197, .35) 72%, rgba(79, 209, 197, 0) 100%); }}
.rtm-jack {{ fill: {PUMPJACK_COLOUR}; stroke: {PUMPJACK_COLOUR}; }}
.rtm-oil {{ stroke: {PUMPJACK_COLOUR}; stroke-width: 5; stroke-linecap: round; fill: none; }}
.rtm-flow {{ stroke: #2dd4bf; stroke-width: 2.5; stroke-linecap: round; fill: none;
            stroke-dasharray: 3 9; animation: rtm-flow 0.9s linear infinite; }}
@keyframes rtm-flow {{ to {{ stroke-dashoffset: -12; }} }}
.rtm-title {{ font-size: 20px; font-weight: 700; letter-spacing: -.01em; }}
.rtm-dots span {{ display: inline-block; animation: rtm-dot 1.2s infinite ease-in-out; opacity: .2; }}
.rtm-dots span:nth-child(2) {{ animation-delay: .2s; }}
.rtm-dots span:nth-child(3) {{ animation-delay: .4s; }}
@keyframes rtm-dot {{ 0%, 80%, 100% {{ opacity: .2; }} 40% {{ opacity: 1; }} }}
.rtm-sub {{ font-size: 13px; color: {MUTED}; margin-top: -14px; }}
.rtm-bar {{ width: min(220px, 50vw); height: 3px; border-radius: 3px; background: var(--border); overflow: hidden; }}
.rtm-bar i {{ display: block; width: 40%; height: 100%; border-radius: 3px; background: var(--accent);
             animation: rtm-bar 1.4s ease-in-out infinite; }}
@keyframes rtm-bar {{ 0% {{ transform: translateX(-100%); }} 100% {{ transform: translateX(250%); }} }}
@media (prefers-reduced-motion: reduce) {{ .rtm-flow, .rtm-bar i, .rtm-dots span {{ animation: none; }} }}
"""


ICON_FS_EXIT = svg_icon("M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5")


# Easter egg: clicking the theme switch 10 times within 3 seconds splatters oil
# over the screen. Each splat lands, its drips run down, then the whole lot slides
# off the bottom of the screen. Runs entirely in the browser; the overlay ignores
# the mouse, so the app stays usable underneath.
#
# How it looks liquid: every splat is built from overlapping circles (body, splash
# arms as chains of shrinking circles, drips as rounded bars with a bulb), and an
# SVG "goo" filter blurs them together and sharpens the edge again, so they melt
# into one fluid shape. The same filter adds a specular highlight for the oily gloss.
OIL_SPLATTER_JS = r"""
const now = Date.now();
const clicks = (window.__oilClicks || []).filter(t => now - t < 3000);
clicks.push(now);
window.__oilClicks = clicks;
if (clicks.length >= 10 && !window.__oilActive) {
  window.__oilClicks = [];
  window.__oilActive = true;
  const NS = 'http://www.w3.org/2000/svg';
  const W = window.innerWidth, H = window.innerHeight;
  const unit = Math.min(W, H) / 900;
  const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const rand = (a, b) => a + Math.random() * (b - a);
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const play = (e, frames, opts) => { if (!still && e.animate) e.animate(frames, opts); };
  const OIL = '#0b0603';
  const blur = Math.max(3, 5.5 * unit);          // the goo radius: bigger = rounder, more merged
  const minDot = blur * 1.5;                      // anything smaller would vanish in the filter

  const svg = el('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, 'aria-hidden': 'true'});
  svg.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;overflow:visible;';
  const defs = el('defs', {}, svg);
  const f = el('filter', {id: 'rtm-oil', x: '-30%', y: '-30%', width: '160%', height: '160%',
                          'color-interpolation-filters': 'sRGB'}, defs);
  el('feGaussianBlur', {in: 'SourceGraphic', stdDeviation: blur, result: 'blur'}, f);
  el('feColorMatrix', {in: 'blur', mode: 'matrix', result: 'goo',
                       values: '1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 26 -11'}, f);
  el('feGaussianBlur', {in: 'goo', stdDeviation: blur * 0.6, result: 'bump'}, f);
  // shine only where the surface curves towards the light (a high exponent keeps flat areas dark)
  const spec = el('feSpecularLighting', {in: 'bump', surfaceScale: 5, specularConstant: 1.3,
                                         specularExponent: 70, 'lighting-color': '#ffffff', result: 'spec'}, f);
  el('feDistantLight', {azimuth: 225, elevation: 38}, spec);
  el('feComposite', {in: 'spec', in2: 'goo', operator: 'in', result: 'shine'}, f);
  el('feComposite', {in: 'shine', in2: 'goo', operator: 'arithmetic', k1: 0, k2: 0.75, k3: 1, k4: 0}, f);

  const EXIT = 3200;                              // when the splats start sliding off
  let lastEnd = 0;
  const count = Math.round(rand(5, 8));
  for (let k = 0; k < count; k++) {
    const cx = rand(0.08, 0.92) * W, cy = rand(0.06, 0.62) * H;
    const r = rand(42, 95) * unit;
    const delay = k * 130 + rand(0, 80);
    const fall = el('g', {filter: 'url(#rtm-oil)'}, svg);     // slides the whole splat off at the end
    const dripLayer = el('g', {}, fall);
    const splat = el('g', {}, fall);
    const dot = (x, y, rr, parent) => el('circle', {cx: x, cy: y, r: Math.max(rr, minDot), fill: OIL}, parent);

    // body: a big drop with overlapping lobes, so the outline is organic, not a circle
    dot(cx, cy, r, splat);
    const lobes = Math.round(rand(8, 12));
    for (let i = 0; i < lobes; i++) {
      const a = (i / lobes) * Math.PI * 2 + rand(-0.25, 0.25), d = r * rand(0.5, 0.95);
      dot(cx + Math.cos(a) * d, cy + Math.sin(a) * d, r * rand(0.3, 0.55), splat);
    }
    // a few short, thick splash arms (chains of shrinking drops that merge into rounded streaks)
    const arms = Math.round(rand(2, 4));
    for (let i = 0; i < arms; i++) {
      const a = rand(0, Math.PI * 2), len = rand(1.25, 1.75), steps = 4;
      for (let s = 0; s < steps; s++) {
        const along = r * (0.9 + (len - 0.9) * s / (steps - 1));
        dot(cx + Math.cos(a) * along, cy + Math.sin(a) * along, r * 0.3 * (1 - s * 0.17), splat);
      }
      dot(cx + Math.cos(a) * r * (len + 0.4), cy + Math.sin(a) * r * (len + 0.4), r * 0.15, splat);  // tip droplet
    }
    // loose droplets flung further out
    const drops = Math.round(rand(4, 8));
    for (let i = 0; i < drops; i++) {
      const a = rand(0, Math.PI * 2), d = r * rand(1.7, 2.6);
      dot(cx + Math.cos(a) * d, cy + Math.sin(a) * d, r * rand(0.08, 0.16), splat);
    }
    splat.style.transformBox = 'fill-box';
    splat.style.transformOrigin = 'center';
    play(splat, [{transform: 'scale(.15)', opacity: 0}, {transform: 'scale(1.1)', opacity: 1, offset: 0.55},
                 {transform: 'scale(.97)', offset: 0.8}, {transform: 'scale(1)', opacity: 1}],
         {duration: 520, delay: delay, easing: 'cubic-bezier(.2,.8,.3,1)', fill: 'both'});

    // drips: rounded bars with a heavier bulb at the end, running down under gravity
    const drips = Math.round(rand(2, 4));
    let longest = 0;
    for (let i = 0; i < drips; i++) {
      const w = r * rand(0.16, 0.27), len = r * rand(1.4, 3.6);
      const x = cx + rand(-0.6, 0.6) * r;
      longest = Math.max(longest, len);
      const drip = el('g', {}, dripLayer);
      el('rect', {x: x - w / 2, y: cy - r * 0.2, width: w, height: len + r * 0.2, rx: w / 2, fill: OIL}, drip);
      dot(x, cy + len, w * 0.8, drip);
      drip.style.transformBox = 'fill-box';
      drip.style.transformOrigin = 'top';
      const grow = rand(1700, 2800);
      play(drip, [{transform: 'scaleY(0)'}, {transform: 'scaleY(.35)', offset: 0.3}, {transform: 'scaleY(1)'}],
           {duration: grow, delay: delay + 280, easing: 'cubic-bezier(.45,0,.75,1)', fill: 'both'});
      // and they stretch further as the splat slides off
      const leave = EXIT + k * 140;
      play(drip, [{transform: 'scaleY(1)'}, {transform: 'scaleY(' + rand(1.6, 2.3).toFixed(2) + ')'}],
           {duration: 1500, delay: leave, easing: 'ease-in', fill: 'forwards'});
    }

    // slide off the bottom of the screen, speeding up like a falling drop
    const start = EXIT + k * 140 + rand(0, 260), dur = rand(1500, 2100);
    const drop = H - cy + r * 3.5 + longest * 2.4;
    play(fall, [{transform: 'translateY(0)'}, {transform: 'translateY(' + (drop * 0.08).toFixed(0) + 'px)', offset: 0.3},
                {transform: 'translateY(' + drop.toFixed(0) + 'px)'}],
         {duration: dur, delay: start, easing: 'cubic-bezier(.55,0,.85,.4)', fill: 'forwards'});
    play(splat, [{transform: 'scale(1, 1)'}, {transform: 'scale(.92, 1.25)'}],
         {duration: dur, delay: start, easing: 'ease-in', fill: 'forwards'});
    lastEnd = Math.max(lastEnd, start + dur);
  }

  document.body.appendChild(svg);
  setTimeout(() => { svg.remove(); window.__oilActive = false; }, still ? 4000 : lastEnd + 150);
}
"""


ICON_ARROW = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M7 17 17 7M8 7h9v9"/></svg>')


DIFF_STOPS = [(0.0, SOFTEN), (0.5, "#ffffff"), (1.0, HARDEN)]  # amplitude decrease -> increase


def palette_from_stops(stops, n=256):
    out = []
    for i in range(n):
        t = i / (n - 1)
        for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
            if p0 <= t <= p1:
                f = 0 if p1 == p0 else (t - p0) / (p1 - p0)
                a, b = _hex_to_rgb(c0), _hex_to_rgb(c1)
                out.append("#%02x%02x%02x" % tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3)))
                break
    return out


MARK_COLOR = "#ff4d4f"  # change-zone box and slice position on the change map


# ---------------------------------------------------------------------------
# Wells page
# ---------------------------------------------------------------------------
WELL_COLOURS = ["#ff4d6d", "#1e90ff", "#22c55e", "#a855f7", "#f59e0b",
                "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1"]


RESERVOIR_COLOUR = "#ffd400"   # picked reservoir interval (on the seismic and the log tracks)


CANDIDATE_COLOUR = "#22c55e"   # other intervals that passed the four tests (log tracks)


# (the checkbox text is a bare <span> inside the widget, not inside a <label>)
CHECKBOX_CSS = (":host, span, label { color: var(--text2) !important; font-size: 13px; } "
                "input { accent-color: var(--accent); }")


ADD_TOGGLE_CSS = """
.bk-btn.bk-active { background: var(--accent-soft) !important; color: var(--accent) !important;
                    border-color: var(--accent) !important; }
"""


TABLE_CSS = """
table.wells { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; }
table.wells th { text-align: left; font-weight: 600; font-size: 12px; color: var(--muted);
                 padding: 6px 10px; border-bottom: 1px solid var(--border-strong); }
table.wells td { padding: 9px 10px; border-bottom: 1px solid var(--border); color: var(--text2);
                 font-variant-numeric: tabular-nums; }
table.wells td:first-child { color: var(--text); font-weight: 600; }
table.wells tr:hover td { background: var(--hover); }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px;
          vertical-align: 0; }
"""
