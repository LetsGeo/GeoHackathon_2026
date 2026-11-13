"""
Reservoir Time Machine: 4D difference workspace (Panel version)

Run with:
    panel serve app.py --show

Choose the seismic files either in the app ("Data sources" card: Browse, then Run), or
when starting it, e.g.:
    panel serve app.py --show --args --baseline data/94p10ful --monitor data/10p10ful
        --baseline-label "1994 survey" --monitor-label "2010 survey"
Optional: --inline 1840 (starting inline)   --credit "Data: ..."   (paths can be absolute, or
relative to this file's folder)

Needs:  pip install -r requirements.txt   (panel, segyio, numpy, matplotlib, pillow, openai)

Analysis summary by Gemma: "Run analysis" sends the measurements to a Gemma model
behind an OpenAI-compatible server (default http://172.28.34.1:8000/v1, model
google/gemma-4-26B-A4B-it) and shows its English + Malay summary. Change the server
with --gemma-url / --gemma-model (or GEMMA_BASE_URL / GEMMA_MODEL), turn it off with
--no-gemma. If the server can't be reached the rule-based summary is shown instead.
Check the connection with:  python gemma_client.py

Fault screening ("Fault screening" card): tiles the shared area with hexagons 20 lines
across, measures attributes on every trace (coherence, fault throw, 4D change, NRMS,
time shift, envelope, instantaneous frequency, sweetness, seal contrast) and sends 5
inlines + 5 crosslines of each hexagon to Gemma, which says whether a fault crosses it
and whether it seals the CO2. Results: map, table, per-hexagon sections, CSV, PDF pages.
GEMMA_WORKERS sets how many hexagons are sent at once (default 4). The same without the
UI:  python hex_screen.py --baseline B.sgy --monitor M.sgy   (see hexscreen.py)

The first load of a file reads the whole SEG-Y once and saves it in "cache/"
(about the same size as the file), so inline, crossline and time slices are
instant afterwards. Delete the cache folder any time to free the space.

Tested with Panel 1.4 / Bokeh 3.4 on Python 3.9+.
"""
import argparse
import asyncio
import base64
import functools
import io
import os
import subprocess
import sys
import textwrap
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import panel as pn

import gemma_client
import hexscreen
from seismic_io import load_cube
import param
from bokeh.models import Button as BkButton
from bokeh.models import (
    BoxZoomTool, ColorBar, ColumnDataSource, CustomAction, CustomJS, HoverTool, LinearColorMapper, PanTool, Patches,
    Range1d, ResetTool, Span, TapTool, WheelZoomTool,
)
from bokeh.plotting import figure

LOGO_PATH = Path(__file__).parent / "assets" / "Logo.png"
LOGO_SRC = "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
# Two colour themes. The app's CSS uses these as CSS variables (var(--bg) etc.),
# so switching theme only swaps the variables; the Bokeh plots are restyled
# from Python by style_figure().
THEMES = {
    "dark": {
        "bg": "#0a1216", "sidebar": "#0c161a", "card": "#101b20", "card2": "#132026",
        "border": "#1d2b31", "border-strong": "#2a3b42",
        "text": "#e6eef0", "text2": "#a9b8bd", "muted": "#6d7f86",
        "accent": "#4fd1c5", "accent-ink": "#062b28", "accent-soft": "rgba(79,209,197,.12)",
        "accent-hover": "#6ee0d5", "nav-active": "#16262b",
        "hover": "rgba(255,255,255,.04)", "stripe": "rgba(255,255,255,.025)",
        "err": "#f0a58f", "err-border": "#7a3b2e",
    },
    "light": {
        "bg": "#eef2f4", "sidebar": "#ffffff", "card": "#ffffff", "card2": "#f4f7f8",
        "border": "#dde4e7", "border-strong": "#c3cdd2",
        "text": "#13212a", "text2": "#3e5059", "muted": "#6a7b84",
        "accent": "#0e9488", "accent-ink": "#ffffff", "accent-soft": "rgba(14,148,136,.10)",
        "accent-hover": "#0b7d73", "nav-active": "#e3eef0",
        "hover": "rgba(0,0,0,.04)", "stripe": "rgba(0,0,0,.03)",
        "err": "#b4441f", "err-border": "#e3a48c",
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
pn.extension(
    raw_css=[
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');",
        theme_css(),
        "html, body { margin: 0; background: var(--bg); transition: background .2s; }",
    ]
)

SANS = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
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

# Colourbar scale: amplitude runs from -AMP_MAX (trough) to +AMP_MAX (peak),
# split into CBAR_DIVISIONS equal sections. AMP_MAX is replaced by the data's
# 99th-percentile amplitude once a seismic file loads (32 is only the fallback).
AMP_MAX = 32
CBAR_DIVISIONS = 10

# Shared stylesheet for every HTML pane (Panel renders panes in shadow DOM,
# so styles have to be passed to each pane rather than set globally).
BASE_CSS = f"""
:host {{ color: {TEXT}; font-family: {SANS}; font-size: 14px; }}
* {{ box-sizing: border-box; }}
.mono {{ font-family: {MONO}; }}
.meta {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; }}
.label {{ font-size: 11px; letter-spacing: .08em; color: {TEXT_2}; font-weight: 600; }}
.title {{ font-weight: 600; font-size: 14px; margin: 0; }}
.note {{ color: {MUTED}; font-size: 12.5px; line-height: 1.5; margin: 0; }}
.head {{ display: flex; justify-content: space-between; align-items: baseline; width: 100%; }}

/* sidebar */
.sidebar {{ padding: 18px 14px; }}
.brand {{ display: flex; gap: 10px; align-items: center; padding: 0 4px 22px; }}
.brand-mark {{ width: 34px; height: 34px; border-radius: 9px; background: #045373;
               display: grid; place-items: center; flex-shrink: 0; }}
.brand-name {{ font-weight: 600; font-size: 13.5px; line-height: 1.2; }}
.brand-sub {{ color: {MUTED}; font-size: 11.5px; margin-top: 2px; }}
.nav-label {{ font-size: 10.5px; letter-spacing: .08em; color: {MUTED};
              font-weight: 600; margin: 18px 6px 8px; }}
.nav-item {{ display: flex; align-items: center; gap: 10px; padding: 9px 10px;
             border-radius: 8px; color: {TEXT_2}; font-size: 13.5px; }}
.nav-item svg {{ width: 16px; height: 16px; flex-shrink: 0; }}
.nav-item.active {{ background: var(--nav-active); color: {TEXT}; }}
.nav-item.active svg {{ color: {ACCENT}; }}

/* header */
.crumbs {{ font-size: 12px; color: {MUTED}; }}
.crumbs .sep {{ color: {BORDER_STRONG}; margin: 0 4px; }}
h1 {{ font-size: 20px; font-weight: 600; margin: 4px 0 0; }}

/* progress track (time-lapse, disabled)
.track {{ height: 3px; background: {BORDER}; border-radius: 2px; overflow: hidden; }}
.fill {{ height: 100%; background: {ACCENT}; }}
*/

/* colourbar */
.cbar-wrap {{ display: flex; flex-direction: column; gap: 6px; width: 100%; padding: 0 14px; }}
.cbar {{ position: relative; height: 24px; border: 1px solid #5a5a5a; }}
.cbar .div {{ position: absolute; top: 0; bottom: 0; width: 1px; background: rgba(90, 90, 90, .85); }}
.cbar .mask {{ position: absolute; top: 0; bottom: 0; background: #4b6470; opacity: .92; }}
.cbar-ticks {{ position: relative; height: 14px; font-family: {MONO}; font-size: 10.5px; color: {TEXT_2}; }}
.cbar-ticks span {{ position: absolute; transform: translateX(-50%); white-space: nowrap; }}

/* seismic placeholders */
.survey-tag {{ font-family: {MONO}; font-size: 11px; padding: 3px 8px; border-radius: 6px;
               border: 1px solid {BORDER_STRONG}; color: {TEXT_2}; }}
.survey-tag.monitor {{ border-color: {ACCENT}; color: {ACCENT}; }}
.seis {{ height: 360px; margin-top: 12px; border: 1px dashed {BORDER_STRONG}; border-radius: 10px;
         display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
         background: repeating-linear-gradient(0deg, transparent 0 13px, var(--stripe) 13px 14px), {CARD_2};
         color: {MUTED}; text-align: center; }}
.seis svg {{ color: {TEXT_2}; opacity: .7; }}
.seis .big {{ color: {TEXT_2}; font-size: 13.5px; font-weight: 500; }}
.seis.error {{ border-color: var(--err-border); }}
.seis.error .big {{ color: var(--err); }}
.sub {{ font-family: {MONO}; font-size: 11px; color: {MUTED}; margin-top: 3px; }}
.credit {{ font-size: 11px; color: {MUTED}; line-height: 1.45; margin-top: 8px; }}
.status {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; }}
.status.ok {{ color: {ACCENT}; }}
.status.err {{ color: var(--err); }}
/* analysis */
.plot-caption {{ font-size: 12px; color: {TEXT}; padding-bottom: 2px; }}
.tiles {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
.tile {{ background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 10px 12px; }}
.tile-label {{ font-size: 11px; color: {MUTED}; }}
.tile-value {{ font-family: {MONO}; font-size: 18px; font-weight: 500; color: {TEXT}; margin-top: 4px; }}
.tile-detail {{ font-size: 11px; color: {TEXT_2}; margin-top: 2px; }}
.summary {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }}
.sum-col {{ background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px 16px; }}
.sum-lang {{ font-size: 11px; letter-spacing: .08em; font-weight: 600; color: {ACCENT}; text-transform: uppercase; }}
.sum-col h4 {{ font-size: 12.5px; margin: 12px 0 4px; color: {TEXT}; }}
.sum-col p {{ font-size: 13px; line-height: 1.55; color: {TEXT_2}; margin: 0 0 6px; }}
.sum-col p.sum-note {{ font-size: 11.5px; color: {MUTED}; font-style: italic; margin-top: 10px; }}
.morph-ends {{ display: flex; justify-content: space-between; font-family: {MONO};
               font-size: 11px; color: {TEXT_2}; }}
.tiles.four {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.chip {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
         border: 1px solid currentColor; white-space: nowrap; }}
.chip.likely {{ color: #e5484d; }} .chip.possible {{ color: #e8920c; }}
.chip.fault {{ color: #6f8fb0; }} .chip.none {{ color: {MUTED}; }}
.hex-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 6px; }}
.hex-table th {{ text-align: left; font-weight: 500; color: {MUTED}; font-size: 11px; padding: 6px 8px;
                 border-bottom: 1px solid {BORDER}; }}
.hex-table td {{ padding: 6px 8px; border-bottom: 1px solid {BORDER}; color: {TEXT_2};
                 font-family: {MONO}; font-size: 12px; }}
.hex-table td.txt {{ font-family: inherit; }}
.hex-table tr.sel td {{ background: var(--accent-soft); }}
.hex-detail h3 {{ font-size: 14px; margin: 4px 0 6px; color: {TEXT}; }}
.hex-detail .lines {{ font-family: {MONO}; font-size: 11.5px; color: {MUTED}; margin-bottom: 8px; }}
.hex-imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }}
.hex-detail img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid {BORDER};
                   background: #fff; }}
.attr-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px 14px; margin-top: 8px; }}
.attr-grid div {{ font-size: 11px; color: {MUTED}; }}
.attr-grid b {{ display: block; font-family: {MONO}; font-weight: 500; font-size: 12.5px; color: {TEXT}; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 11.5px; color: {TEXT_2}; margin: 6px 0 2px; }}
.legend i {{ display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px;
             vertical-align: -1px; }}
"""

CARD_STYLES = {
    "background": CARD,
    "border": f"1px solid {BORDER}",
    "border-radius": "12px",
    "padding": "14px 16px",
    "box-sizing": "border-box",
}

BTN_BASE_CSS = f"""
.bk-btn {{
  font-family: {SANS}; font-size: 13px; font-weight: 500;
  border-radius: 8px; padding: 8px 14px; cursor: pointer;
  background: transparent !important; color: {TEXT} !important;
  border: 1px solid {BORDER_STRONG} !important; box-shadow: none !important;
}}
.bk-btn:hover {{ background: var(--hover) !important; }}
.bk-btn[disabled] {{ opacity: .45; cursor: not-allowed; }}
"""

PRIMARY_BTN_CSS = f"""
.bk-btn {{ background: {ACCENT} !important; border-color: {ACCENT} !important; color: {ACCENT_INK} !important; }}
.bk-btn:hover {{ background: var(--accent-hover) !important; }}
"""

TOGGLE_CSS = f"""
.bk-btn-group {{ gap: 0; }}
.bk-btn {{
  font-family: {SANS}; font-size: 12.5px; font-weight: 500; padding: 7px 14px;
  background: {CARD_2} !important; color: {TEXT_2} !important;
  border: 1px solid {BORDER_STRONG} !important; box-shadow: none !important; cursor: pointer;
}}
.bk-btn:hover {{ color: {TEXT} !important; }}
.bk-btn.bk-active {{ background: var(--accent-soft) !important; color: {ACCENT} !important;
                     border-color: {ACCENT} !important; }}
.bk-btn:focus {{ outline: none; }}
"""

INPUT_CSS = f"""
:host {{ --input-bg: {CARD_2}; }}
label, .bk-input-group > label {{ color: {MUTED}; font-family: {SANS}; font-size: 11.5px; }}
.bk-input {{
  background: {CARD_2} !important; color: {TEXT} !important;
  border: 1px solid {BORDER_STRONG} !important; border-radius: 8px !important;
  font-family: {MONO}; font-size: 12.5px; padding: 7px 10px;
}}
.bk-input:focus {{ border-color: {ACCENT} !important; box-shadow: none !important; }}
.bk-input::placeholder {{ color: {MUTED}; }}
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
  font-family: {SANS}; font-size: 13px; font-weight: 500; min-width: 110px;
  border-radius: 8px; padding: 8px 16px; cursor: pointer;
  background: {CARD_2} !important; color: {TEXT} !important;
  border: 1px solid {BORDER_STRONG} !important; box-shadow: none !important;
}}
.bk-btn:hover {{ border-color: {ACCENT} !important; color: {ACCENT} !important; }}
.bk-btn:focus {{ outline: none; }}
"""

# --- Time-lapse styles (disabled) ---
# STEP_CSS = f"""
# .bk-btn {{
#   width: 100%; text-align: left; white-space: pre-line; line-height: 1.45;
#   padding: 8px 12px; border-radius: 8px; cursor: pointer; box-shadow: none !important;
#   font-family: {MONO}; font-size: 12px; color: {TEXT} !important;
#   background: {CARD_2} !important; border: 1px solid {BORDER} !important;
# }}
# .bk-btn:hover {{ border-color: {BORDER_STRONG} !important; }}
# .bk-btn[disabled] {{ opacity: .45; cursor: not-allowed; }}
# .bk-btn:focus {{ outline: none; }}
# :host(.active) .bk-btn {{ border-color: {ACCENT} !important; background: rgba(79,209,197,.12) !important; }}
# """
#
# PLAY_CSS = f"""
# .bk-btn {{
#   width: 44px; height: 44px; border-radius: 50%; padding: 0;
#   background: {ACCENT} !important; border: 0 !important; color: {ACCENT_INK} !important;
#   font-size: 15px; cursor: pointer; box-shadow: none !important;
# }}
# .bk-btn:hover {{ background: #6ee0d5 !important; }}
# """


def set_label(button, text):
    """Button.name is a constant param in newer versions of param, so unlock it to relabel."""
    with param.edit_constant(button):
        button.name = text


def html(content, **kwargs):
    """HTML pane with the shared stylesheet applied."""
    kwargs.setdefault("margin", 0)
    return pn.pane.HTML(content, stylesheets=[BASE_CSS], **kwargs)


# ---------------------------------------------------------------------------
# Data (swap for real survey data later)
# ---------------------------------------------------------------------------
# MONITORS = [
#     {"id": 0, "short": "Base", "long": "Reference", "available": True},
#     {"id": 1, "short": "M1", "long": "Monitor 1", "available": True},
#     {"id": 2, "short": "M2", "long": "Monitor 2", "available": True},
#     {"id": 3, "short": "M3", "long": "Monitor 3", "available": True},
#     {"id": 4, "short": "M4", "long": "Monitor 4", "available": False},
# ]


# ---------------------------------------------------------------------------
# Data sources (nothing dataset-specific is hard-coded: these are only defaults)
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent

SLEIPNER_CREDIT = (
    "Seismic data: Sleipner CO2 Reference Dataset, © The Sleipner Group "
    "(Equinor Energy AS and partners), used under the Sleipner CO2 Reference Dataset License."
)


def parse_args():
    """Read optional settings passed with:  panel serve app.py --args --baseline ... """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline", default="data/94p10ful")
    parser.add_argument("--monitor", default="data/10p10ful")
    parser.add_argument("--baseline-label", default="1994 survey")
    parser.add_argument("--monitor-label", default="2010 survey")
    parser.add_argument("--inline", type=int, default=None)
    parser.add_argument("--credit", default=SLEIPNER_CREDIT)
    parser.add_argument("--gemma-url", default=None)    # e.g. http://172.28.34.1:8000/v1
    parser.add_argument("--gemma-model", default=None)  # e.g. google/gemma-4-26B-A4B-it
    parser.add_argument("--no-gemma", action="store_true")  # rule-based summary only
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


ARGS = parse_args()

# Which inline to display (None = the middle inline of the baseline survey; the
# monitor always shows the same inline so the two can be compared), and an
# optional time window in ms, e.g. (700, 1200) to zoom in on the reservoir.
INLINE = ARGS.inline
TIME_WINDOW_MS = None

# Gemma writes the analysis summary (see gemma_client.py). The server address and
# model can also be set with the GEMMA_BASE_URL / GEMMA_MODEL environment variables.
USE_GEMMA = not ARGS.no_gemma and os.environ.get("GEMMA_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
GEMMA_URL = ARGS.gemma_url
GEMMA_MODEL = ARGS.gemma_model


def resolve_path(text):
    """Accept absolute paths, ~ paths, or paths relative to the app's folder."""
    p = Path(text.strip().strip('"').strip("'")).expanduser()
    return p if p.is_absolute() else APP_DIR / p


# ---------------------------------------------------------------------------
# Seismic loading
# ---------------------------------------------------------------------------
# The SEG-Y reader and its cache live in seismic_io.py (load_cube).

VIEWS = ("Inline", "Crossline", "Time slice")
# per view: x-axis label, y-axis label, does y (time) increase downwards?
VIEW_AXES = {
    "Inline": ("Crossline", "Time (ms)", True),
    "Crossline": ("Inline", "Time (ms)", True),
    "Time slice": ("Crossline", "Inline", False),
}
VIEW_KEY = {"Inline": "ilines", "Crossline": "xlines", "Time slice": "times"}


def nice_round(x):
    """Round to 2 significant figures so colourbar labels stay readable."""
    if x <= 0 or not np.isfinite(x):
        return 1.0
    return float(f"{x:.2g}")


def nearest(values, v):
    return int(np.abs(np.asarray(values, dtype=np.float64) - v).argmin())


def time_index_range(times):
    if not TIME_WINDOW_MS:
        return 0, len(times)
    k = np.where((times >= TIME_WINDOW_MS[0]) & (times <= TIME_WINDOW_MS[1]))[0]
    return (int(k[0]), int(k[-1]) + 1) if len(k) else (0, len(times))


def get_slice(vol, view, value):
    """One 2D slice as a Bokeh-ready image (rows = y, columns = x).
    Inline / Crossline: x = the other line number, y = time.
    Time slice: x = crossline, y = inline (a map view)."""
    cube, il, xl, t = vol["cube"], vol["ilines"], vol["xlines"], vol["times"]
    k0, k1 = time_index_range(t)
    if view == "Inline":
        i = nearest(il, value)
        img, xs, ys, pos = np.asarray(cube[i, :, k0:k1]).T, xl, t[k0:k1], il[i]
    elif view == "Crossline":
        j = nearest(xl, value)
        img, xs, ys, pos = np.asarray(cube[:, j, k0:k1]).T, il, t[k0:k1], xl[j]
    else:
        k = nearest(t, value)
        img, xs, ys, pos = np.asarray(cube[:, :, k]), xl, il, t[k]
    return {"img": np.ascontiguousarray(img, dtype=np.float32),
            "xs": np.asarray(xs, dtype=np.float64), "ys": np.asarray(ys, dtype=np.float64),
            "pos": float(pos)}


def spacing(values):
    return float(np.median(np.diff(values))) if len(values) > 1 else 1.0


def extent(values):
    """Edges of the pixels centred on these coordinates."""
    d = spacing(values)
    return float(values[0] - d / 2), float(values[-1] + d / 2)


def image_data(sl):
    """Column data for a Bokeh image glyph from a slice."""
    x0, x1 = extent(sl["xs"])
    y0, y1 = extent(sl["ys"])
    return {"image": [sl["img"]], "x": [x0], "y": [y0], "dw": [x1 - x0], "dh": [y1 - y0]}


def load_surveys(cfg):
    """Load both surveys from a config dict. Each survey gets "vol" (or None)
    and "error" (or None). Also sets the shared colour scale (AMP_MAX)."""
    global AMP_MAX
    surveys = [
        {"role": "Baseline", "path": cfg["baseline"], "title": cfg["baseline_label"]},
        {"role": "Monitor", "path": cfg["monitor"], "title": cfg["monitor_label"]},
    ]
    for survey in surveys:
        survey["vol"], survey["error"], survey["name"] = None, None, ""
        if not survey["path"].strip():
            continue
        path = resolve_path(survey["path"])
        survey["name"] = path.name
        try:
            if not path.is_file():
                raise FileNotFoundError(f"file not found: {path}")
            survey["vol"] = load_cube(path)
        except Exception as exc:  # show the problem in the card instead of crashing
            survey["error"] = str(exc)

    amps = [s_["vol"]["amp"] for s_ in surveys if s_["vol"] is not None]
    if amps:
        AMP_MAX = nice_round(max(amps))
    return surveys


def slider_values(surveys, view):
    """Positions the slice number can take: those shared by both surveys
    (or all of one survey's if they share none)."""
    key = VIEW_KEY[view]
    arrays = [s_["vol"][key] for s_ in surveys if s_["vol"] is not None]
    if not arrays:
        return np.array([0.0])
    common = arrays[0]
    for arr in arrays[1:]:
        common = np.intersect1d(common, arr)
    return np.asarray(common if len(common) else arrays[0], dtype=np.float64)


def align2d(a, b, y_is_time, max_cells=1_500_000):
    """Put two slices on the same grid so they can be blended: keep only the
    positions both share, and (for time) the overlapping time range with the
    monitor resampled onto the baseline's sample times. Very large slices are
    thinned for the animation only."""
    cx = np.intersect1d(a["xs"], b["xs"])
    if len(cx) < 2:
        raise ValueError("the two surveys don't overlap on this slice")
    A = a["img"][:, np.searchsorted(a["xs"], cx)]
    B = b["img"][:, np.searchsorted(b["xs"], cx)]
    if y_is_time:
        keep = (a["ys"] >= max(a["ys"][0], b["ys"][0])) & (a["ys"] <= min(a["ys"][-1], b["ys"][-1]))
        if keep.sum() < 2:
            raise ValueError("the two surveys don't overlap in time")
        ys = a["ys"][keep]
        A = A[keep]
        if len(a["ys"]) == len(b["ys"]) and np.allclose(a["ys"], b["ys"]):
            B = B[keep]
        else:
            B = np.stack([np.interp(ys, b["ys"], B[:, c]) for c in range(B.shape[1])], axis=1)
    else:
        cy = np.intersect1d(a["ys"], b["ys"])
        if len(cy) < 2:
            raise ValueError("the two surveys don't overlap on this slice")
        A = A[np.searchsorted(a["ys"], cy)]
        B = B[np.searchsorted(b["ys"], cy)]
        ys = cy

    step = 1
    while (A.shape[0] // step) * (A.shape[1] // step) > max_cells:
        step += 1
    A, B = A[::step, ::step], B[::step, ::step]
    return ({"img": A.astype(np.float32), "xs": cx[::step], "ys": ys[::step]},
            (B - A).astype(np.float32))


# Per-session state for the time-lapse (disabled)
# state = {"selected": 3, "timer": None}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
ICON_GRID = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg>'
ICON_COMPARE = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 4v16"/></svg>'
ICON_WELL = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="7" r="3"/><path d="M12 10v11"/></svg>'

SIDEBAR_WIDTH = 220

sidebar = html(
    f"""
<div class="sidebar">
    <div class="brand">
    <div class="brand-mark">
        <img src="{LOGO_SRC}" alt="Reservoir Time Machine" style="width:100%; height:100%; object-fit:contain;">
    </div>
    <div>
      <div class="brand-name">Reservoir Time Machine</div>
      <div class="brand-sub">4D seismic workspace</div>
    </div>
  </div>
  <div class="nav-label">ANALYSE</div>
  <div class="nav-item active">{ICON_GRID}Workspace</div>
  <div class="nav-item">{ICON_COMPARE}Compare surveys</div>
  <div class="nav-label">FIELD</div>
  <div class="nav-item">{ICON_WELL}Wells</div>
</div>
""",
    width=SIDEBAR_WIDTH,
    # pinned to the left of the window, full height, so it stays put while the
    # page scrolls (the main column is shifted right by SIDEBAR_WIDTH to make room)
    styles={"background": SIDEBAR_BG, "border-right": f"1px solid {BORDER}",
            "position": "fixed", "top": "0", "left": "0", "height": "100vh",
            "overflow-y": "auto", "overflow-x": "hidden",  # scrolls up/down only if ever needed, never sideways
            "box-sizing": "border-box", "z-index": "50"},
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
title_block = html(
    """
<h1>4D Difference Workspace</h1>
"""
)

# Light / dark mode switch (on = light): the browser swaps the CSS variables
# instantly (jscallback), and Python restyles the plots (on_theme_change, further down).
# The knob shows a moon in dark mode and a sun in light mode.
_MOON_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
             "<path d='M20.5 14.2A8.6 8.6 0 1 1 9.8 3.5a6.8 6.8 0 0 0 10.7 10.7z'/></svg>")
_SUN_SVG = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
            "stroke='black' stroke-width='2.2' stroke-linecap='round'><circle cx='12' cy='12' r='4.2' fill='black'/>"
            "<path d='M12 2v2.4M12 19.6V22M4.9 4.9l1.7 1.7M17.4 17.4l1.7 1.7M2 12h2.4M19.6 12H22M4.9 19.1l1.7-1.7"
            "M17.4 6.6l1.7-1.7'/></svg>")
THEME_SWITCH_CSS = """
:host { gap: 0 !important; }
.bk-label, .bk-icon { display: none !important; }
.bk-body { width: 58px !important; height: 30px !important; }
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
  top: 3px !important; left: 3px !important; width: 24px !important; height: 24px !important;
  border-radius: 50% !important; background: var(--accent) !important; z-index: 1;
  box-shadow: 0 1px 3px rgba(0, 0, 0, .35);
  transition: left .25s cubic-bezier(.3, .7, .4, 1), background-color .25s ease !important;
}
.bk-knob::after {
  content: ""; position: absolute; inset: 5px; background: var(--accent-ink);
  -webkit-mask: url("MOON") center / contain no-repeat; mask: url("MOON") center / contain no-repeat;
}
:host(.bk-active) .bk-bar { background: var(--accent-soft) !important; border-color: var(--accent); }
:host(.bk-active) .bk-knob { left: 31px !important; }
:host(.bk-active) .bk-knob::after { -webkit-mask-image: url("SUN"); mask-image: url("SUN"); }
.bk-body:hover .bk-bar { border-color: var(--accent); }
.bk-knob:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
""".replace("SUN", _SUN_SVG).replace("MOON", _MOON_SVG)

theme_toggle = pn.widgets.Switch(name="", value=DEFAULT_THEME == "light",
                                 stylesheets=[THEME_SWITCH_CSS], margin=(5, 10), align="center")

# Easter egg: flipping the theme switch 10 times within 3 seconds splatters oil
# over the screen for 5 seconds. Runs entirely in the browser; the overlay
# ignores the mouse, so the app stays usable underneath.
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
  const animate = (e, frames, opts) => { if (!still && e.animate) e.animate(frames, opts); };

  // irregular closed blob: random radii round a circle, smoothed through midpoints
  const blob = (cx, cy, r, n) => {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2 + rand(-0.15, 0.15);
      const rr = r * rand(0.7, 1.15) * (Math.random() < 0.3 ? rand(1.2, 1.6) : 1);
      pts.push([cx + Math.cos(a) * rr, cy + Math.sin(a) * rr]);
    }
    const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
    const s = mid(pts[n - 1], pts[0]);
    let d = 'M' + s[0].toFixed(1) + ',' + s[1].toFixed(1);
    for (let i = 0; i < n; i++) {
      const m = mid(pts[i], pts[(i + 1) % n]);
      d += ' Q' + pts[i][0].toFixed(1) + ',' + pts[i][1].toFixed(1) + ' ' + m[0].toFixed(1) + ',' + m[1].toFixed(1);
    }
    return d + 'Z';
  };

  const svg = el('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, 'aria-hidden': 'true'});
  svg.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483647;pointer-events:none;' +
                      'transition:opacity 1s ease;filter:drop-shadow(0 4px 6px rgba(0,0,0,.35));';
  const defs = el('defs', {}, svg);
  const fill = el('radialGradient', {id: 'oil-fill', cx: '38%', cy: '32%', r: '80%'}, defs);
  [['0%', '#2b1e0d'], ['35%', '#110b05'], ['100%', '#020101']].forEach(
    ([o, c]) => el('stop', {offset: o, 'stop-color': c}, fill));
  const sheen = el('linearGradient', {id: 'oil-sheen', x1: '0', y1: '0', x2: '1', y2: '1'}, defs);
  [['0%', '#6fd3ff', '.10'], ['30%', '#000', '0'], ['55%', '#c77dff', '.08'], ['80%', '#000', '0'],
   ['100%', '#ffd166', '.10']].forEach(
    ([o, c, a]) => el('stop', {offset: o, 'stop-color': c, 'stop-opacity': a}, sheen));

  const count = Math.round(rand(6, 9));
  for (let k = 0; k < count; k++) {
    const cx = rand(0.08, 0.92) * W, cy = rand(0.08, 0.85) * H;
    const r = rand(45, 130) * unit;
    const delay = k * 110;
    const g = el('g', {}, svg);

    // drips run down from the lower half of the splat
    const drips = Math.round(rand(1, 3.4));
    for (let i = 0; i < drips; i++) {
      const w = r * rand(0.1, 0.2), len = r * rand(1.4, 3.4);
      const x = cx + rand(-0.55, 0.55) * r;
      const drip = el('g', {}, g);
      el('rect', {x: x - w / 2, y: cy, width: w, height: len, rx: w / 2, fill: 'url(#oil-fill)'}, drip);
      el('circle', {cx: x, cy: cy + len, r: w * 0.75, fill: 'url(#oil-fill)'}, drip);
      drip.style.transformBox = 'fill-box';
      drip.style.transformOrigin = 'top';
      animate(drip, [{transform: 'scaleY(0)'}, {transform: 'scaleY(1)'}],
              {duration: rand(1600, 3000), delay: delay + 150, easing: 'cubic-bezier(.4,0,.8,.7)', fill: 'both'});
    }

    // the splat itself, with a rainbow oil sheen and a glossy highlight
    const splat = el('g', {}, g);
    const d = blob(cx, cy, r, Math.round(rand(14, 20)));
    // splash arms: tapered streaks thrown out from the impact, each ending in a droplet
    const arms = Math.round(rand(6, 11));
    for (let i = 0; i < arms; i++) {
      const a = rand(0, Math.PI * 2), L = r * rand(1.15, 2.1), bw = r * rand(0.14, 0.3);
      const ux = Math.cos(a), uy = Math.sin(a), px = -uy, py = ux;
      const at = (along, side) => (cx + ux * along + px * side).toFixed(1) + ',' + (cy + uy * along + py * side).toFixed(1);
      el('path', {d: 'M' + at(r * 0.5, bw / 2) + ' Q' + at(L * 0.72, bw * 0.12) + ' ' + at(L, 0) +
                     ' Q' + at(L * 0.72, -bw * 0.12) + ' ' + at(r * 0.5, -bw / 2) + 'Z', fill: 'url(#oil-fill)'}, splat);
      el('circle', {cx: cx + ux * (L + bw * 0.15), cy: cy + uy * (L + bw * 0.15), r: bw * rand(0.3, 0.45),
                    fill: 'url(#oil-fill)'}, splat);
    }
    el('path', {d: d, fill: 'url(#oil-fill)'}, splat);
    el('path', {d: d, fill: 'url(#oil-sheen)'}, splat);
    el('ellipse', {cx: cx - r * 0.3, cy: cy - r * 0.32, rx: r * 0.28, ry: r * 0.1, fill: '#fff', opacity: '.22',
                   transform: 'rotate(-28 ' + (cx - r * 0.3) + ' ' + (cy - r * 0.32) + ')'}, splat);
    el('ellipse', {cx: cx - r * 0.05, cy: cy - r * 0.48, rx: r * 0.07, ry: r * 0.035, fill: '#fff', opacity: '.35'}, splat);

    // flung droplets around it
    const drops = Math.round(rand(6, 13));
    for (let i = 0; i < drops; i++) {
      const a = rand(0, Math.PI * 2), dist = r * rand(1.1, 2.6);
      el('circle', {cx: cx + Math.cos(a) * dist, cy: cy + Math.sin(a) * dist,
                    r: r * rand(0.025, 0.09), fill: 'url(#oil-fill)'}, splat);
    }
    splat.style.transformBox = 'fill-box';
    splat.style.transformOrigin = 'center';
    animate(splat, [{transform: 'scale(.15)', opacity: 0}, {transform: 'scale(1.08)', opacity: 1, offset: 0.6},
                    {transform: 'scale(1)', opacity: 1}],
            {duration: 380, delay: delay, easing: 'cubic-bezier(.2,.8,.3,1)', fill: 'both'});
  }

  document.body.appendChild(svg);
  setTimeout(() => { svg.style.opacity = '0'; }, 4000);  // fade over the last second
  setTimeout(() => { svg.remove(); window.__oilActive = false; }, 5000);
}
"""
theme_toggle.jscallback(value="document.documentElement.classList.toggle('light', cb_obj.active);\n" + OIL_SPLATTER_JS)

header = pn.Row(
    title_block,
    pn.layout.HSpacer(),
    pn.Row(theme_toggle, margin=0, align="center"),
    sizing_mode="stretch_width",
    margin=(0, 0, 8, 0),
)

# ---------------------------------------------------------------------------
# Difference map card goes here once the real map is ready
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Colour map switch + shared colourbar for the seismic data
# ---------------------------------------------------------------------------
cmap_select = pn.widgets.RadioButtonGroup(
    options=list(SEISMIC_CMAPS),
    value=DEFAULT_CMAP,
    stylesheets=[TOGGLE_CSS],
    margin=(0, 24, 0, 0),
    align="start",
)
colourbar = html("", sizing_mode="stretch_width")        # the bar itself
colourbar_ticks = html("", sizing_mode="stretch_width")  # the numbers under it
range_text = html("", sizing_mode="stretch_width")

# Display range: amplitudes outside it are greyed out on every plot, in a slate
# colour that isn't part of either seismic colour map so it can't be mistaken
# for data. Use "rgba(0, 0, 0, 0)" to hide them completely instead.
OUT_OF_RANGE_COLOR = "#4b6470"
NAN_COLOR = "rgba(0, 0, 0, 0)"  # positions with no trace

# The range slider sits on top of the colourbar: drag the two handles on the bar.
range_slider = pn.widgets.RangeSlider(
    start=-AMP_MAX, end=AMP_MAX, value=(-AMP_MAX, AMP_MAX), step=AMP_MAX / 100,
    format="0[.]00", show_value=False,
    sizing_mode="stretch_width", stylesheets=[OVERLAY_SLIDER_CSS], margin=(-25, 14, 0, 14),
)


def colourbar_html(name, lo=None, hi=None):
    lo = -AMP_MAX if lo is None else lo
    hi = AMP_MAX if hi is None else hi
    stops = ", ".join(f"{colour} {pos * 100:.1f}%" for pos, colour in SEISMIC_CMAPS[name])
    lines = "".join(f'<i class="div" style="left:{i / CBAR_DIVISIONS * 100:.2f}%"></i>'
                    for i in range(1, CBAR_DIVISIONS))
    # grey out the parts of the bar outside the display range
    p_lo = (lo + AMP_MAX) / (2 * AMP_MAX) * 100
    p_hi = (hi + AMP_MAX) / (2 * AMP_MAX) * 100
    masks = ""
    if p_lo > 0.01:
        masks += f'<i class="mask" style="left:0; width:{p_lo:.2f}%"></i>'
    if p_hi < 99.99:
        masks += f'<i class="mask" style="left:{p_hi:.2f}%; right:0"></i>'
    return (f'<div class="cbar-wrap"><div class="cbar" '
            f'style="background: linear-gradient(90deg, {stops});">{lines}{masks}</div></div>')


def ticks_html():
    ticks = ""
    for i in range(CBAR_DIVISIONS + 1):
        value = -AMP_MAX + 2 * AMP_MAX * i / CBAR_DIVISIONS
        label = "0" if abs(value) < 1e-12 * max(AMP_MAX, 1) else f"{value:.3g}"
        ticks += f'<span style="left:{i / CBAR_DIVISIONS * 100:.2f}%">{label}</span>'
    return f'<div class="cbar-wrap"><div class="cbar-ticks">{ticks}</div></div>'


def apply_colours(event=None):
    """Push the colour map and display range to the colourbar and every plot.
    The palette keeps its full-scale positions; values outside the display
    range get OUT_OF_RANGE_COLOR."""
    name = cmap_select.value
    lo, hi = range_slider.value
    lo, hi = max(lo, -AMP_MAX), min(hi, AMP_MAX)
    if hi <= lo:
        hi = lo + AMP_MAX / 100
    full = cmap_palette(name, 256)
    i0 = int(round((lo + AMP_MAX) / (2 * AMP_MAX) * 255))
    i1 = int(round((hi + AMP_MAX) / (2 * AMP_MAX) * 255))
    palette = full[i0:i1 + 1] if i1 > i0 else [full[i0], full[i0]]
    # Amplitudes beyond the colourbar's ends (the strongest 1 %) keep the end
    # colours; only values cut off by the display range are greyed out.
    eps = AMP_MAX * 1e-6
    low_color = full[0] if lo <= -AMP_MAX + eps else OUT_OF_RANGE_COLOR
    high_color = full[-1] if hi >= AMP_MAX - eps else OUT_OF_RANGE_COLOR
    for mapper in colour_mappers:
        mapper.update(palette=palette, low=lo, high=hi,
                      low_color=low_color, high_color=high_color, nan_color=NAN_COLOR)
    colourbar.object = colourbar_html(name, lo, hi)
    colourbar_ticks.object = ticks_html()
    range_text.object = (
        '<div class="head"><span class="label">COLOUR MAP</span>'
        f'<span class="meta">Showing {lo:.3g} to {hi:.3g} · drag the handles on the bar; '
        'values outside are greyed out</span></div>'
    )


def reset_range_slider():
    """Match the slider to the current amplitude scale (after loading new data)."""
    range_slider.param.update(start=min(range_slider.start, -AMP_MAX), end=max(range_slider.end, AMP_MAX))
    range_slider.param.update(value=(-AMP_MAX, AMP_MAX), step=AMP_MAX / 100)
    range_slider.param.update(start=-AMP_MAX, end=AMP_MAX)


colour_mappers = []  # filled in when the seismic plots are built
cmap_select.param.watch(apply_colours, "value")
range_slider.param.watch(apply_colours, "value")
display_bar = pn.Column(
    range_text,
    pn.Row(
        cmap_select,
        pn.Column(colourbar, range_slider, colourbar_ticks, sizing_mode="stretch_width", margin=0),
        sizing_mode="stretch_width", margin=(12, 0, 0, 0),
    ),
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)

# ---------------------------------------------------------------------------
# Slice controls: Inline / Crossline / Time slice, and which one
# ---------------------------------------------------------------------------
view_select = pn.widgets.RadioButtonGroup(
    options=list(VIEWS), value="Inline", stylesheets=[TOGGLE_CSS],
    margin=(0, 24, 0, 0), align="center",
)
# hides the up/down arrow buttons; the number can still be typed, or changed
# with the keyboard's up/down arrow keys
NO_ARROWS_CSS = ".bk-spin-btn { display: none !important; }"

position_input = pn.widgets.IntInput(
    start=0, end=1, step=1, value=0,
    width=170, stylesheets=[INPUT_CSS, NO_ARROWS_CSS], margin=(0, 0, 0, 0), align="center",
)
position_range = html("", align="center", margin=(0, 0, 0, 14))

slice_bar = pn.Column(
    html('<div class="head"><span class="label">SLICE</span>'
         '<span class="meta">Both surveys and the morph always show the same slice</span></div>',
         sizing_mode="stretch_width"),
    pn.Row(view_select, position_input, position_range, sizing_mode="stretch_width", margin=(6, 0, 0, 0)),
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)

# ---------------------------------------------------------------------------
# Seismic surveys side by side (placeholders)
# ---------------------------------------------------------------------------
ICON_SEISMIC = '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 6c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M2 12c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/><path d="M2 18c2 0 2-2 4-2s2 2 4 2 2-2 4-2 2 2 4 2 2-2 4-2"/></svg>'


# Bokeh's own FullscreenTool enlarges the plot's box but doesn't redraw the plot
# when it sits inside a Panel layout, so this button handles it. It puts the
# whole page in fullscreen (so hover tooltips, which Bokeh draws at page level,
# stay visible), pins the plot over the screen and resizes it to fit. Esc or the
# button again restores everything. Box zoom, hover, pan and scroll zoom all keep
# working in fullscreen.
FULLSCREEN_JS = """
const view = Bokeh.index.find_one(plot);
if (view == null) return;
if (document.fullscreenElement) { document.exitFullscreen(); return; }

const el = view.el;
const restore = {
  sizing_mode: plot.sizing_mode, width: plot.width, height: plot.height,
  style: el.getAttribute("style"), overflow: document.body.style.overflow,
};
const fit = () => {
  tool.icon = exit_icon;
  wheel.active = true;  // scroll-wheel zoom only works in fullscreen
  el.style.position = "fixed";
  el.style.inset = "0";
  el.style.zIndex = "999";  // below Bokeh's tooltips (z-index 1000)
  el.style.background = "BG_COLOUR";
  document.body.style.overflow = "hidden";
  plot.sizing_mode = "fixed";
  plot.width = window.innerWidth;
  plot.height = window.innerHeight;
};
const on_change = () => {
  if (document.fullscreenElement) {
    fit();
  } else {
    if (restore.style == null) el.removeAttribute("style"); else el.setAttribute("style", restore.style);
    document.body.style.overflow = restore.overflow;
    tool.icon = enter_icon;
    wheel.active = false;  // back to normal page scrolling
    plot.sizing_mode = restore.sizing_mode;
    plot.width = restore.width;
    plot.height = restore.height;
    document.removeEventListener("fullscreenchange", on_change);
  }
};
document.addEventListener("fullscreenchange", on_change);
document.documentElement.requestFullscreen().catch(() => {
  document.removeEventListener("fullscreenchange", on_change);
});
""".replace("BG_COLOUR", CARD)


def svg_icon(path_d):
    """Toolbar icon (as a data URI) from an SVG path, in the toolbar's grey."""
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
           'stroke="#7d8f97" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
           f'<path d="{path_d}"/></svg>')
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


# four corners pointing outwards (enter fullscreen) / inwards (leave fullscreen)
ICON_FS_ENTER = svg_icon("M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5")
ICON_FS_EXIT = svg_icon("M9 4v5H4M20 9h-5V4M15 20v-5h5M4 15h5v5")


def fullscreen_tool(plot, wheel):
    callback = CustomJS(args={"plot": plot, "wheel": wheel, "enter_icon": ICON_FS_ENTER, "exit_icon": ICON_FS_EXIT},
                        code=FULLSCREEN_JS)
    tool = CustomAction(description="Fullscreen (Esc to exit)", icon=ICON_FS_ENTER, callback=callback)
    callback.args["tool"] = tool  # so the button can swap its own icon
    return tool


def style_figure(p):
    """Colour a Bokeh figure for the current theme (called again when the theme changes)."""
    c = THEMES[ws["theme"]]
    p.background_fill_color = c["card2"]
    p.border_fill_color = c["card"]
    p.outline_line_color = c["border-strong"]
    p.grid.visible = False
    p.toolbar.logo = None
    p.title.text_color = c["text"]
    p.title.text_font_style = "normal"
    p.title.text_font_size = "12px"
    for axis in (p.xaxis, p.yaxis):
        axis.axis_label_text_color = c["text2"]
        axis.axis_label_text_font_style = "normal"
        axis.axis_label_text_font_size = "11px"
        axis.major_label_text_color = c["text2"]
        axis.major_label_text_font_size = "10.5px"
        axis.axis_line_color = c["border-strong"]
        axis.major_tick_line_color = c["border-strong"]
        axis.minor_tick_line_color = None
    for item in p.below:  # colour bars (analysis plots)
        if isinstance(item, ColorBar):
            item.background_fill_color = c["card"]
            item.major_label_text_color = c["text2"]
            item.title_text_color = c["text2"]
            item.major_tick_line_color = c["border-strong"]
            item.bar_line_color = c["border-strong"]


def make_figure(x_range, y_range, x_label, y_label, height=360, value_label="Amplitude"):
    """Dark-styled Bokeh figure for a seismic slice. All plots share the same
    x_range / y_range objects, so they always show identical axes and zoom and
    pan together."""
    p = figure(
        height=height,
        sizing_mode="stretch_width",  # the fullscreen button resizes it to the screen
        x_range=x_range,
        y_range=y_range,
        tools=[],
        toolbar_location="above",
        x_axis_label=x_label,
        y_axis_label=y_label,
    )
    # Toolbar: move | box zoom | fullscreen | reset | hover.
    # Box zoom is the default drag action: click and drag on the plot to zoom.
    # Click the move button to drag the view around instead.
    # Bokeh always orders drag tools first, then scroll tools, then buttons like
    # fullscreen/reset, so the scroll-zoom button is hidden to keep fullscreen
    # right after box zoom. Scroll-wheel zoom is off on the page (so the mouse
    # wheel scrolls the page) and switched on only in fullscreen.
    pan = PanTool()
    wheel = WheelZoomTool(visible=False)
    box_zoom = BoxZoomTool()
    p.add_tools(pan, box_zoom, wheel, fullscreen_tool(p, wheel), ResetTool())
    p.toolbar.active_drag = box_zoom
    p.toolbar.active_scroll = None
    p.add_tools(HoverTool(tooltips=[
        (x_label, "$x{0}"), (y_label, "$y{0}"), (value_label, "@image{0.000}"),
    ]))

    style_figure(p)
    ws["figs"].append(p)
    return p


def new_mapper():
    """Colour mapper on the shared scale; registered so the colour controls update it."""
    mapper = LinearColorMapper(palette=cmap_palette(cmap_select.value), low=-AMP_MAX, high=AMP_MAX)
    colour_mappers.append(mapper)
    return mapper


def credit_html(credit):
    return html(f'<div class="credit">{credit}</div>', sizing_mode="stretch_width") if credit.strip() else None


def position_text(view, pos):
    return f"time slice {pos:g} ms" if view == "Time slice" else f"{view.lower()} {pos:g}"


def message_box(title, detail="", error=False, height=360):
    return html(
        f'''<div class="seis{' error' if error else ''}" style="height:{height}px">
          {ICON_SEISMIC}
          <div class="big">{title}</div>
          <div class="meta">{detail}</div>
        </div>''',
        sizing_mode="stretch_width",
    )


def card_header(title, role, sub):
    tag_class = "survey-tag monitor" if role == "Monitor" else "survey-tag"
    head = html(
        f'<div class="head"><h2 class="title">{title}</h2>'
        f'<span class="{tag_class}">{role}</span></div>',
        sizing_mode="stretch_width",
    )
    sub_pane = html(f'<div class="sub">{sub}</div>', sizing_mode="stretch_width")
    return head, sub_pane


# The morph runs entirely in the browser: the baseline and the difference
# (monitor - baseline) are sent once, and each frame is baseline + t * difference.
# "state" holds t (0..1); the timer lives on the state object. It plays once,
# from the baseline to the monitor, and stops there.
MORPH_COMMON_JS = """
const apply = () => {
  const t = state.data.t[0];
  const img = src.data.image[0], base = src.data.base[0], diff = src.data.diff[0];
  for (let i = 0; i < img.length; i++) img[i] = base[i] + t * diff[i];
  src.change.emit();
  plot.title.text = `${a_label}  →  ${b_label}   ·   ${Math.round(t * 100)}%`;
};
const stop = () => {
  if (state._timer != null) { clearInterval(state._timer); state._timer = null; }
  play_btn.label = "▶  Play";
};
const start = () => {
  stop();
  play_btn.label = "❚❚  Pause";
  state._timer = setInterval(() => {
    if (plot.document == null) { clearInterval(state._timer); state._timer = null; return; }  // plot was replaced
    const t = Math.min(1, state.data.t[0] + 0.01);        // 100 steps of 50 ms = 5 s, forwards only
    state.data.t[0] = t;
    apply();
    if (t >= 1) stop();                                     // reached the monitor: stop, no rewind
  }, 50);
};
"""
# Play at the end starts again from the baseline
MORPH_PLAY_JS = MORPH_COMMON_JS + """
if (state._timer != null) { stop(); }
else { if (state.data.t[0] >= 1) { state.data.t[0] = 0; apply(); } start(); }
"""
MORPH_RESTART_JS = MORPH_COMMON_JS + "state.data.t[0] = 0; apply();"
MORPH_DATA_JS = MORPH_COMMON_JS + "apply();"  # new slice arrived: redraw at the current t


def morph_data(a_slice, b_slice, y_is_time):
    aligned, diff = align2d(a_slice, b_slice, y_is_time)
    data = image_data(aligned)
    data["base"] = [aligned["img"]]
    data["diff"] = [diff]
    data["image"] = [aligned["img"].copy()]
    return data


# ---------------------------------------------------------------------------
# Data sources card: choose files, then Run
# ---------------------------------------------------------------------------
def text_input(name, value, placeholder=""):
    return pn.widgets.TextInput(name=name, value=value, placeholder=placeholder,
                                sizing_mode="stretch_width", stylesheets=[INPUT_CSS])


# Browse buttons open the computer's normal file dialog. The dialog runs in a
# separate small Python process so it can't freeze the app while it's open.
PICKER_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)   # keep the dialog in front of the browser
path = filedialog.askopenfilename(title=sys.argv[1], initialdir=sys.argv[2])
root.destroy()
sys.stdout.write(path or "")
"""


def pick_file(title, start_dir):
    """Open the native file dialog and return the chosen path ("" if cancelled)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", PICKER_SCRIPT, title, str(start_dir)],
                            capture_output=True, text=True, encoding="utf-8", env=env)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "the file dialog couldn't be opened")
    return result.stdout.strip()


async def on_browse(target, title, event):
    """Browse button: pick a file and put its path in the matching text box."""
    current = resolve_path(target.value) if target.value.strip() else APP_DIR
    start_dir = current.parent if current.parent.is_dir() else APP_DIR
    browse_base.disabled = browse_mon.disabled = True
    try:
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, pick_file, title, start_dir)
        if path:
            target.value = path
            load_status.object = '<span class="status">File chosen — press Run to load it</span>'
    except Exception as exc:
        load_status.object = (f'<span class="status err">Couldn\'t open the file dialog ({exc}). '
                              'You can still type or paste the path.</span>')
    finally:
        browse_base.disabled = browse_mon.disabled = False


baseline_path = text_input("Baseline file", ARGS.baseline, "e.g. data/94p10ful or C:\\path\\to\\file.sgy")
baseline_label = text_input("Baseline label", ARGS.baseline_label)
monitor_path = text_input("Monitor file", ARGS.monitor, "e.g. data/10p10ful")
monitor_label = text_input("Monitor label", ARGS.monitor_label)
load_btn = pn.widgets.Button(name="Run", stylesheets=[BTN_BASE_CSS, PRIMARY_BTN_CSS], width=90, align="end")
# Status line beside Run: empty until there's something to say (a file couldn't
# be loaded, a file was chosen with Browse, "Loaded"). Other code writes to it,
# so keep this line even if you don't want a message showing.
load_status = html("", sizing_mode="stretch_width", align="center")
browse_base = pn.widgets.Button(name="Browse…", stylesheets=[BTN_BASE_CSS], width=96, align="end", margin=(0, 10, 9, 10))
browse_mon = pn.widgets.Button(name="Browse…", stylesheets=[BTN_BASE_CSS], width=96, align="end", margin=(0, 10, 9, 10))
browse_base.on_click(functools.partial(on_browse, baseline_path, "Choose the baseline SEG-Y file"))
browse_mon.on_click(functools.partial(on_browse, monitor_path, "Choose the monitor SEG-Y file"))

sources_card = pn.Column(
    html('<div class="head"><span class="label">DATA SOURCES</span>'
         '</div>',
         sizing_mode="stretch_width"),
    pn.Row(baseline_path, browse_base, baseline_label, sizing_mode="stretch_width", margin=(6, 0, 0, 0)),
    pn.Row(monitor_path, browse_mon, monitor_label, sizing_mode="stretch_width", margin=0),
    pn.Row(load_status, pn.layout.HSpacer(), load_btn, sizing_mode="stretch_width", margin=(4, 10, 0, 0)),
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)

content = pn.Column(sizing_mode="stretch_width", margin=0)

# Per-session workspace: loaded surveys and the live plot sources
ws = {"surveys": [], "cards": [], "morph": None, "credit": "",
      "theme": DEFAULT_THEME, "figs": [], "afigs": [], "analysis": None,
      "hfigs": [], "hex": None}


def build_views():
    """Build the two survey cards and the morph for the current view and slice."""
    colour_mappers.clear()
    ws["figs"] = []
    view, pos = view_select.value, position_input.value
    x_label, y_label, y_down = VIEW_AXES[view]
    surveys, credit = ws["surveys"], ws["credit"]
    slices = [get_slice(s_["vol"], view, pos) if s_["vol"] is not None else None for s_ in surveys]

    # one shared pair of axes covering both surveys
    loaded = [sl for sl in slices if sl is not None]
    shared_x = shared_y = None
    if loaded:
        x0 = min(extent(sl["xs"])[0] for sl in loaded)
        x1 = max(extent(sl["xs"])[1] for sl in loaded)
        y0 = min(extent(sl["ys"])[0] for sl in loaded)
        y1 = max(extent(sl["ys"])[1] for sl in loaded)
        # bounds stop zooming/panning past the data: zoom in, never out past the full view
        shared_x = Range1d(x0, x1, bounds=(x0, x1))
        shared_y = Range1d(y1, y0, bounds=(y0, y1)) if y_down else Range1d(y0, y1, bounds=(y0, y1))

    ws["cards"] = []
    cards = []
    for survey, sl, margin in zip(surveys, slices, [(8, 8, 8, 0), (8, 0, 8, 8)]):
        if sl is not None:
            sub = f'{survey["name"]} · {position_text(view, sl["pos"])}'
        else:
            sub = survey["name"] or "no file chosen"
        head, sub_pane = card_header(survey["title"], survey["role"], sub)
        if sl is not None:
            src = ColumnDataSource(image_data(sl))
            mapper = new_mapper()
            p = make_figure(shared_x, shared_y, x_label, y_label)
            p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=src, color_mapper=mapper)
            body = [pn.pane.Bokeh(p, sizing_mode="stretch_width", margin=(10, 0, 0, 0)), credit_html(credit)]
            ws["cards"].append({"survey": survey, "src": src, "sub": sub_pane})
        elif survey["error"]:
            body = [message_box(f'Couldn\'t load {survey["name"]}', survey["error"], error=True)]
        else:
            body = [message_box("No file chosen", "Add a file path under Data sources")]
        cards.append(pn.Column(head, sub_pane, *[b for b in body if b is not None],
                               sizing_mode="stretch_width", styles=CARD_STYLES, margin=margin))

    # morph card
    base, mon = surveys
    m_head, _ = card_header("Morph", "", "")
    m_head.object = ('<div class="head"><h2 class="title">Morph</h2>'
                     '<span class="meta">Baseline → Monitor, blended in amplitude</span></div>')
    ws["morph"] = None
    if slices[0] is None or slices[1] is None:
        m_body = [message_box("Load both surveys to see the morph", height=200)]
    else:
        try:
            data = morph_data(slices[0], slices[1], y_down)
        except ValueError as exc:
            data = None
            m_body = [message_box("Can't morph these surveys", str(exc), error=True, height=200)]
        if data is not None:
            src = ColumnDataSource(data)
            mapper = new_mapper()
            p = make_figure(shared_x, shared_y, x_label, y_label, height=380)
            p.title.text = f'{base["title"]}  →  {mon["title"]}   ·   0%'
            p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=src, color_mapper=mapper)
            state = ColumnDataSource({"t": [0.0]})
            play_btn = BkButton(label="▶  Play", stylesheets=[MORPH_BTN_CSS, MORPH_PLAY_CSS])
            restart_btn = BkButton(label="↺  Restart", stylesheets=[MORPH_BTN_CSS])
            js_args = {"src": src, "state": state, "plot": p, "play_btn": play_btn,
                       "a_label": base["title"], "b_label": mon["title"]}
            play_btn.js_on_event("button_click", CustomJS(args=js_args, code=MORPH_PLAY_JS))
            restart_btn.js_on_event("button_click", CustomJS(args=js_args, code=MORPH_RESTART_JS))
            src.js_on_change("data", CustomJS(args=js_args, code=MORPH_DATA_JS))
            m_body = [pn.pane.Bokeh(p, sizing_mode="stretch_width", margin=(10, 0, 0, 0)),
                      pn.Row(pn.layout.HSpacer(), play_btn, restart_btn,  # right-aligned
                             sizing_mode="stretch_width", margin=(10, 0, 0, 0)), credit_html(credit)]
            ws["morph"] = {"src": src}
    morph = pn.Column(m_head, *[b for b in m_body if b is not None],
                      sizing_mode="stretch_width", styles=CARD_STYLES, margin=(8, 0))

    content.objects = [pn.Row(*cards, sizing_mode="stretch_width", margin=0), morph]
    apply_colours()
    render_analysis()


def update_slice(event=None):
    """Changing the slice number: swap the images in place (no rebuild)."""
    if not ws["surveys"]:
        return
    view, pos = view_select.value, position_input.value
    _, _, y_down = VIEW_AXES[view]
    slices = {}
    for card in ws["cards"]:
        survey = card["survey"]
        sl = get_slice(survey["vol"], view, pos)
        slices[survey["role"]] = sl
        card["src"].data = image_data(sl)
        card["sub"].object = f'<div class="sub">{survey["name"]} · {position_text(view, sl["pos"])}</div>'
    if ws["morph"] is not None and len(slices) == 2:
        try:
            ws["morph"]["src"].data = morph_data(slices["Baseline"], slices["Monitor"], y_down)
        except ValueError:
            pass
    render_analysis()


def setup_slider(view, value=None):
    """Point the slice number box at the positions available for this view."""
    vals = slider_values(ws["surveys"], view)
    step = max(1, int(round(spacing(vals)))) if len(vals) > 1 else 1
    start, end = int(np.floor(vals[0])), int(np.ceil(vals[-1]))
    if end <= start:
        end = start + step
    if value is None:
        value = vals[len(vals) // 2]
    value = int(round(min(max(value, start), end)))
    # widen the limits first, then set the value, then narrow them, so the box
    # never passes through a state where the lower limit is above the upper one
    position_input.param.update(start=min(position_input.start, start), end=max(position_input.end, end))
    position_input.param.update(value=value, step=step)
    position_input.param.update(start=start, end=end)
    position_range.object = f'<span class="meta">{start} to {end}, step {step}</span>'


def detach_views():
    """Forget the current plots before they're replaced, so the control updates
    that follow don't try to redraw plots that are about to be removed."""
    colour_mappers.clear()
    ws["cards"], ws["morph"] = [], None
    ws["figs"] = []


def on_view_change(event):
    detach_views()
    setup_slider(event.new)
    build_views()


def current_config():
    return {
        "baseline": baseline_path.value, "baseline_label": baseline_label.value.strip() or "Baseline",
        "monitor": monitor_path.value, "monitor_label": monitor_label.value.strip() or "Monitor",
    }


def rebuild(event=None):
    """(Re)load the chosen files, then rebuild every view."""
    set_label(load_btn, "Running…")
    load_btn.disabled = True
    try:
        cfg = current_config()
        reset_analysis()
        reset_hex()
        detach_views()
        ws["surveys"] = load_surveys(cfg)
        if all(s_["vol"] is None for s_ in ws["surveys"]):
            # nothing loaded: keep the rest of the page hidden and say why
            problems = "; ".join(f'{s_["role"].lower()}: {escape(s_["error"] or "no file chosen")}'
                                 for s_ in ws["surveys"])
            load_status.object = f'<span class="status err">Couldn\'t load the files ({problems})</span>'
            workspace.visible = False
            return
        load_status.object = '<span class="status ok">Loaded</span>'
        workspace.visible = True  # show everything else now there's data
        reset_range_slider()
        first_value = INLINE if (view_select.value == "Inline" and INLINE is not None and event is None) else None
        setup_slider(view_select.value, first_value)
        build_views()
        errors = [s_ for s_ in ws["surveys"] if s_["error"]]
        if errors:
            load_status.object = f'<span class="status err">{len(errors)} file(s) couldn\'t be loaded — see the card for details</span>'
    finally:
        set_label(load_btn, "Run")
        load_btn.disabled = False


# ---------------------------------------------------------------------------
# Analysis: measure the baseline-vs-monitor differences and write a summary
# ---------------------------------------------------------------------------
# The measurements (analyse_volume / analyse_slice) are plain numbers.
# write_summary() turns them into rule-based English and Malay text straight away;
# when Gemma is enabled, "Run analysis" then sends the measurements, that draft and
# two small pictures to Gemma (gemma_client.summarise) and shows its text instead.
# If Gemma can't be reached, the rule-based text stays.

ANALYSIS_TRACES = 15_000  # traces sampled across the shared area for the whole-volume analysis
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


def read_block(vol, il_sel, xl_sel, times):
    """Traces for the chosen inlines/crosslines, resampled onto `times` if needed.
    Returns an array (n_inlines, n_crosslines, n_times); missing traces are NaN."""
    ii = np.searchsorted(vol["ilines"], il_sel)
    jj = np.searchsorted(vol["xlines"], xl_sel)
    block = np.asarray(vol["cube"][ii][:, jj], dtype=np.float32)
    vt = vol["times"]
    if len(vt) == len(times) and np.allclose(vt, times):
        return block
    pos = np.interp(times, vt, np.arange(len(vt)))
    i0 = np.clip(np.floor(pos).astype(int), 0, len(vt) - 1)
    i1 = np.clip(i0 + 1, 0, len(vt) - 1)
    w = (pos - i0).astype(np.float32)
    return block[..., i0] * (1 - w) + block[..., i1] * w


def rms(a, axis=-1):
    return np.sqrt(np.mean(np.square(a, dtype=np.float64), axis=axis))


def robust_threshold(values):
    """Median + 3 robust standard deviations: values above it count as 'significant change'."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.inf
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return med + 3 * mad if mad > 0 else med * 1.5 + 1e-12


def nrms_rating(n):
    """Repeatability categories after Lumley (NRMS in %)."""
    if not np.isfinite(n):
        return None
    for limit, key in [(10, "outstanding"), (20, "excellent"), (40, "very good"), (60, "reasonable")]:
        if n < limit:
            return key
    return "poor"


def trace_lag(b, m, max_lag):
    """Shift (in samples, sub-sample accurate) that best aligns m to b; positive = m is later."""
    b = b - b.mean()
    m = m - m.mean()
    if not np.any(b) or not np.any(m):
        return np.nan
    c = np.correlate(m, b, mode="full")
    mid = len(b) - 1
    seg = c[mid - max_lag: mid + max_lag + 1]
    k = int(np.argmax(seg))
    if 0 < k < len(seg) - 1:  # parabolic refinement of the peak
        y0, y1, y2 = seg[k - 1], seg[k], seg[k + 1]
        denom = y0 - 2 * y1 + y2
        k = k + (0.5 * (y0 - y2) / denom if denom != 0 else 0)
    return k - max_lag


def analyse_volume(bv, mv):
    """Whole-volume comparison on a regular sample of traces from the shared area."""
    il = np.intersect1d(bv["ilines"], mv["ilines"])
    xl = np.intersect1d(bv["xlines"], mv["xlines"])
    if len(il) < 2 or len(xl) < 2:
        raise ValueError("the two surveys don't share enough inlines and crosslines")
    t_lo = max(bv["times"][0], mv["times"][0])
    t_hi = min(bv["times"][-1], mv["times"][-1])
    times = bv["times"][(bv["times"] >= t_lo) & (bv["times"] <= t_hi)].astype(np.float64)
    if len(times) < 20:
        raise ValueError("the two surveys don't overlap enough in time")

    step = 1
    while (len(il) // step) * (len(xl) // step) > ANALYSIS_TRACES:
        step += 1
    il_s, xl_s = il[::step], xl[::step]
    B = read_block(bv, il_s, xl_s, times)
    M = read_block(mv, il_s, xl_s, times)
    valid = np.isfinite(B).all(-1) & np.isfinite(M).all(-1)  # traces present in both surveys
    if valid.sum() < 10:
        raise ValueError("too few traces exist in both surveys")
    B[~valid] = 0
    M[~valid] = 0
    D = M - B
    nt = len(times)
    dt = spacing(times)

    def median_nrms(k0, k1):
        num = rms(D[..., k0:k1])
        den = rms(B[..., k0:k1]) + rms(M[..., k0:k1])
        ok = valid & (den > 0)
        return float(np.median(200 * num[ok] / den[ok])) if ok.any() else float("nan")

    # repeatability in the shallow section (10-30 % of the record), assumed mostly unchanged
    k0, k1 = int(nt * 0.1), max(int(nt * 0.3), int(nt * 0.1) + 5)
    res = {
        "il_range": (float(il[0]), float(il[-1])), "xl_range": (float(xl[0]), float(xl[-1])),
        "t_range": (float(times[0]), float(times[-1])), "sample_step": step,
        "n_traces": int(valid.sum()),
        "nrms_shallow": median_nrms(k0, k1), "nrms_full": median_nrms(0, nt),
        "corr": float(np.corrcoef(B[valid].ravel(), M[valid].ravel())[0, 1]),
    }
    res["rating"] = nrms_rating(res["nrms_shallow"])

    # change map: RMS of the difference along each trace
    change = np.where(valid, rms(D), np.nan)
    thr = robust_threshold(change)
    changed = valid & (change > thr)
    res.update({"map": change.astype(np.float32), "map_il": il_s, "map_xl": xl_s,
                "threshold": float(thr), "changed_pct": float(changed.sum() / valid.sum() * 100)})
    res["significant"] = res["changed_pct"] >= 0.5
    if not changed.any():
        return res

    ci, cj = np.nonzero(changed)
    res["box"] = (float(il_s[ci.min()]), float(il_s[ci.max()]), float(xl_s[cj.min()]), float(xl_s[cj.max()]))
    pi, pj = np.unravel_index(np.nanargmax(change), change.shape)
    res["peak"] = (float(il_s[pi]), float(xl_s[pj]))

    # where (in time) the change starts on each changed trace. The top of the zone
    # is used rather than the strongest point, because reflections *below* a change
    # are often shifted in time and that makes the difference largest further down.
    Dc = np.abs(D[changed])
    kernel = np.ones(5) / 5
    env = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, Dc)
    onset_idx = np.argmax(env >= 0.5 * env.max(axis=1, keepdims=True), axis=1)
    t_on = times[onset_idx]
    on10, on50, on90 = np.percentile(t_on, [10, 50, 90])
    res["t_change"] = (float(on10), float(on50), float(on90))

    # stronger or weaker in the monitor, just around the top of the change on each trace
    rng = np.random.default_rng(0)
    pick = np.arange(len(onset_idx))
    if len(pick) > 2000:
        pick = rng.choice(pick, 2000, replace=False)
    Bc, Mc = B[changed], M[changed]
    before, after = int(round(40 / dt)), int(round(60 / dt))
    ratios = []
    for k in pick:
        w0, w1 = max(0, onset_idx[k] - before), min(nt, onset_idx[k] + after + 1)
        rb = rms(Bc[k, w0:w1])
        if rb > 0:
            ratios.append(rms(Mc[k, w0:w1]) / rb)
    res["amp_ratio"] = float(np.median(ratios)) if ratios else float("nan")

    # time shift of reflections below the change zone (velocity push-down / pull-up),
    # measured from just below the deepest top of change to the end of the record and
    # compared with the same window on unchanged traces
    s0 = min(nt - 1, int(np.searchsorted(times, on90 + 60)))
    s1 = nt
    if s1 - s0 >= 20:
        max_lag = max(2, int(round(20 / dt)))  # search up to ±20 ms

        def lags(mask):
            idx = np.argwhere(mask)
            if len(idx) > 300:
                idx = idx[rng.choice(len(idx), 300, replace=False)]
            vals = [trace_lag(B[i, j, s0:s1], M[i, j, s0:s1], max_lag) for i, j in idx]
            vals = [v for v in vals if np.isfinite(v)]
            return np.median(vals) if vals else np.nan

        changed_lag, ref_lag = lags(changed), lags(valid & ~changed)
        if np.isfinite(changed_lag) and np.isfinite(ref_lag):
            res["time_shift_ms"] = float((changed_lag - ref_lag) * dt)
    return res


def analyse_slice(a_slice, b_slice, view):
    """Comparison of the slice currently on screen."""
    _, _, y_is_time = VIEW_AXES[view]
    aligned, diff = align2d(a_slice, b_slice, y_is_time)
    A = aligned["img"]
    M = A + diff
    valid = np.isfinite(A) & np.isfinite(diff)
    if valid.sum() < 10:
        raise ValueError("the two surveys don't overlap on this slice")
    a, d, m = A[valid], diff[valid], M[valid]
    den = rms(a, None) + rms(m, None)
    res = {"view": view, "pos": float(a_slice["pos"]), "aligned": aligned, "diff": diff,
           "nrms": float(200 * rms(d, None) / den) if den > 0 else float("nan")}

    # smooth |difference| a little so single noisy samples don't count as change
    absd = np.where(valid, np.abs(diff), np.nan)
    pad = np.pad(np.nan_to_num(absd), 1, mode="edge")
    smooth = sum(pad[i:i + absd.shape[0], j:j + absd.shape[1]] for i in range(3) for j in range(3)) / 9
    smooth = np.where(valid, smooth, np.nan)
    thr = robust_threshold(smooth)
    changed = valid & (smooth > thr)
    res["changed_pct"] = float(changed.sum() / valid.sum() * 100)
    res["significant"] = res["changed_pct"] >= 0.5
    if changed.any():
        py, px = np.unravel_index(np.nanargmax(smooth), smooth.shape)
        res["peak"] = (float(aligned["xs"][px]), float(aligned["ys"][py]))
        if y_is_time:
            # top of the change in each changed column, and the amplitude change just there
            cols = np.nonzero(changed.any(axis=0))[0]
            tops = np.array([np.argmax(changed[:, c]) for c in cols])
            ys = aligned["ys"]
            res["top"] = float(np.median(ys[tops]))
            dy = spacing(ys)
            before, after = int(round(40 / dy)), int(round(60 / dy))
            band = np.zeros_like(changed)
            for c, t in zip(cols, tops):
                band[max(0, t - before): t + after + 1, c] = True
            band &= valid
            rb, rm = rms(A[band], None), rms(M[band], None)
        else:
            rb, rm = rms(A[changed], None), rms(M[changed], None)
        res["amp_ratio"] = float(rm / rb) if rb > 0 else float("nan")
    return res


# ---------------------------------------------------------------------------
# Summary text (English + Malay). Replace this function to use an AI model.
# ---------------------------------------------------------------------------
RATING_MS = {"outstanding": "luar biasa baik", "excellent": "cemerlang", "very good": "sangat baik",
             "reasonable": "sederhana", "poor": "lemah"}
VIEW_MS = {"Inline": "inline", "Crossline": "crossline", "Time slice": "keratan masa"}


def fmt(x, digits=0):
    return f"{x:,.{digits}f}"


def amp_phrases(ratio):
    """(english, malay) wording for a monitor/baseline amplitude ratio."""
    if not np.isfinite(ratio):
        return None
    pct = abs(ratio - 1) * 100
    if ratio >= 1.1:
        return (f"amplitudes are about {pct:.0f}% stronger in the monitor survey (amplitude brightening), "
                "which is typical of gas or CO₂ replacing brine in the rock",
                f"amplitud adalah kira-kira {pct:.0f}% lebih kuat dalam survei pemantauan (pencerahan amplitud), "
                "yang lazimnya menunjukkan gas atau CO₂ menggantikan air garam dalam batuan")
    if ratio <= 0.9:
        return (f"amplitudes are about {pct:.0f}% weaker in the monitor survey (amplitude dimming), "
                "which can indicate water replacing gas or oil, or a change in pressure",
                f"amplitud adalah kira-kira {pct:.0f}% lebih lemah dalam survei pemantauan (pemalapan amplitud), "
                "yang boleh menunjukkan air menggantikan gas atau minyak, atau perubahan tekanan")
    return ("amplitudes rise in some places and fall in others, with little net change",
            "amplitud meningkat di sesetengah tempat dan menurun di tempat lain, dengan perubahan bersih yang kecil")


def write_summary(vol, sl, base_label, mon_label):
    """Turn the measurements into short English and Malay paragraphs.
    Returns {"en": {"volume": [...], "slice": [...], "note": str}, "ms": {...}}."""
    en_v, ms_v, en_s, ms_s = [], [], [], []

    if vol is not None:
        il0, il1 = vol["il_range"]
        xl0, xl1 = vol["xl_range"]
        t0, t1 = vol["t_range"]
        en_v.append(
            f"Comparing {base_label} (baseline) with {mon_label} (monitor) over their shared area "
            f"(inlines {fmt(il0)}–{fmt(il1)}, crosslines {fmt(xl0)}–{fmt(xl1)}, {fmt(t0)}–{fmt(t1)} ms), "
            f"{vol['changed_pct']:.1f}% of the area shows a significant change in seismic amplitude.")
        ms_v.append(
            f"Perbandingan antara {base_label} (survei asas) dan {mon_label} (survei pemantauan) di kawasan yang "
            f"dikongsi (inline {fmt(il0)}–{fmt(il1)}, crossline {fmt(xl0)}–{fmt(xl1)}, {fmt(t0)}–{fmt(t1)} ms) "
            f"menunjukkan bahawa {vol['changed_pct']:.1f}% daripada kawasan tersebut mengalami perubahan amplitud "
            "seismik yang ketara.")

        if vol["significant"] and "box" in vol:
            b0, b1, c0, c1 = vol["box"]
            p_il, p_xl = vol["peak"]
            t10, t50, t90 = vol["t_change"]
            span_en = f"about {fmt(t50)} ms" if t90 - t10 < 20 else f"about {fmt(t50)} ms (between {fmt(t10)} and {fmt(t90)} ms across the zone)"
            span_ms = f"kira-kira {fmt(t50)} ms" if t90 - t10 < 20 else f"kira-kira {fmt(t50)} ms (antara {fmt(t10)} hingga {fmt(t90)} ms merentasi zon)"
            en_v.append(
                f"The change is concentrated between inlines {fmt(b0)}–{fmt(b1)} and crosslines {fmt(c0)}–{fmt(c1)}, "
                f"strongest near inline {fmt(p_il)} / crossline {fmt(p_xl)}. The top of the change zone is at {span_en}.")
            ms_v.append(
                f"Perubahan tertumpu antara inline {fmt(b0)}–{fmt(b1)} dan crossline {fmt(c0)}–{fmt(c1)}, paling kuat "
                f"berhampiran inline {fmt(p_il)} / crossline {fmt(p_xl)}. Bahagian atas zon perubahan berada pada {span_ms}.")
            phr = amp_phrases(vol.get("amp_ratio", np.nan))
            if phr:
                en_v.append(f"At the top of the zone, {phr[0]}.")
                ms_v.append(f"Di bahagian atas zon, {phr[1]}.")
            shift = vol.get("time_shift_ms")
            if shift is not None and abs(shift) >= 1.0:
                if shift > 0:
                    en_v.append(f"Reflections below the change zone arrive about {shift:.1f} ms later in the monitor "
                                "survey (velocity push-down), suggesting lower seismic velocity in the changed zone.")
                    ms_v.append(f"Pantulan di bawah zon perubahan tiba kira-kira {shift:.1f} ms lebih lewat dalam "
                                "survei pemantauan (velocity push-down), menunjukkan halaju seismik yang lebih rendah "
                                "di zon yang berubah.")
                else:
                    en_v.append(f"Reflections below the change zone arrive about {-shift:.1f} ms earlier in the "
                                "monitor survey (velocity pull-up), suggesting higher seismic velocity in the changed zone.")
                    ms_v.append(f"Pantulan di bawah zon perubahan tiba kira-kira {-shift:.1f} ms lebih awal dalam "
                                "survei pemantauan (velocity pull-up), menunjukkan halaju seismik yang lebih tinggi "
                                "di zon yang berubah.")
        else:
            en_v.append("No area shows an amplitude change clearly above the noise level.")
            ms_v.append("Tiada kawasan yang menunjukkan perubahan amplitud yang jelas melebihi aras hingar.")

        if vol["rating"]:
            good = vol["rating"] in ("outstanding", "excellent", "very good")
            en_v.append(
                f"In the shallow section, which is assumed to be largely unchanged, the median NRMS is "
                f"{vol['nrms_shallow']:.0f}% ({vol['rating']} repeatability), so "
                + ("the detected change is unlikely to be survey noise." if good else
                   "part of the detected change may be survey noise; interpret it with care."))
            ms_v.append(
                f"Di bahagian cetek, yang dianggap sebahagian besarnya tidak berubah, median NRMS ialah "
                f"{vol['nrms_shallow']:.0f}% (kebolehulangan {RATING_MS[vol['rating']]}), jadi "
                + ("perubahan yang dikesan berkemungkinan besar bukan hingar survei." if good else
                   "sebahagian perubahan yang dikesan mungkin hingar survei; tafsirkan dengan berhati-hati."))

    if sl is not None:
        x_lab, y_lab, y_is_time = VIEW_AXES[sl["view"]]
        where_en = f"{sl['view'].lower()} {fmt(sl['pos'])}" if sl["view"] != "Time slice" else f"time slice at {fmt(sl['pos'])} ms"
        where_ms = (f"{VIEW_MS[sl['view']]} {fmt(sl['pos'])}" if sl["view"] != "Time slice"
                    else f"keratan masa pada {fmt(sl['pos'])} ms")
        en_s.append(f"On the current {where_en}, {sl['changed_pct']:.1f}% of the section shows significant change "
                    f"(NRMS {sl['nrms']:.0f}%).")
        ms_s.append(f"Pada {where_ms} semasa, {sl['changed_pct']:.1f}% daripada keratan menunjukkan perubahan ketara "
                    f"(NRMS {sl['nrms']:.0f}%).")
        if sl["significant"] and "peak" in sl:
            px, py = sl["peak"]
            phr = amp_phrases(sl.get("amp_ratio", np.nan))
            if y_is_time:
                en_s.append(f"The change starts at about {fmt(sl['top'])} ms and is strongest at "
                            f"{x_lab.lower()} {fmt(px)}" + (f"; at the top of the change, {phr[0]}." if phr else "."))
                ms_s.append(f"Perubahan bermula pada kira-kira {fmt(sl['top'])} ms dan paling kuat pada "
                            f"{x_lab.lower()} {fmt(px)}" + (f"; di bahagian atas perubahan, {phr[1]}." if phr else "."))
            else:
                en_s.append(f"The strongest change is at crossline {fmt(px)} / inline {fmt(py)}"
                            + (f"; there, {phr[0]}." if phr else "."))
                ms_s.append(f"Perubahan paling kuat berada pada crossline {fmt(px)} / inline {fmt(py)}"
                            + (f"; di situ, {phr[1]}." if phr else "."))
        else:
            en_s.append("No part of this slice changes clearly above the noise level.")
            ms_s.append("Tiada bahagian keratan ini yang berubah dengan jelas melebihi aras hingar.")

    return {
        "en": {"volume": en_v, "slice": en_s,
               "note": "Generated automatically from the measurements (rule-based); it should be checked by an interpreter."},
        "ms": {"volume": ms_v, "slice": ms_s,
               "note": "Dijana secara automatik berdasarkan pengukuran (berasaskan peraturan); perlu disemak oleh pentafsir."},
    }


# ---------------------------------------------------------------------------
# Analysis card
# ---------------------------------------------------------------------------
MAP_PALETTE = palette_from_stops([(0.0, "#440154"), (0.25, "#3b528b"), (0.5, "#21918c"),
                                  (0.75, "#5ec962"), (1.0, "#fde725")])  # viridis
MARK_COLOR = "#ff4d4f"  # change-zone box and slice position on the change map

run_btn = pn.widgets.Button(name="Run analysis", stylesheets=[BTN_BASE_CSS, PRIMARY_BTN_CSS], width=130)
analysis_status = html('<span class="status">Compares the whole volume and the slice on screen. '
                       'Takes a few seconds on large files.</span>', sizing_mode="stretch_width", align="center")
analysis_body = pn.Column(sizing_mode="stretch_width", margin=0)


def current_slice_analysis():
    base, mon = ws["surveys"]
    if base["vol"] is None or mon["vol"] is None:
        return None
    view, pos = view_select.value, position_input.value
    try:
        return analyse_slice(get_slice(base["vol"], view, pos), get_slice(mon["vol"], view, pos), view)
    except ValueError:
        return None


def stat_tile(label, value, detail=""):
    return (f'<div class="tile"><div class="tile-label">{label}</div>'
            f'<div class="tile-value">{value}</div><div class="tile-detail">{detail}</div></div>')


def stats_html(vol):
    ratio = vol.get("amp_ratio", np.nan)
    if np.isfinite(ratio) and vol["significant"]:
        pct = (ratio - 1) * 100
        kind = "brightening" if ratio >= 1.1 else "dimming" if ratio <= 0.9 else "mixed"
        amp = stat_tile("Amplitude in change zone", f"{pct:+.0f}%", kind)
    else:
        amp = stat_tile("Amplitude in change zone", "–", "no significant change")
    if "t_change" in vol and vol["significant"]:
        t10, t50, t90 = vol["t_change"]
        zone = stat_tile("Top of change zone", f"{t50:,.0f} ms",
                         f"{t10:,.0f}–{t90:,.0f} ms across the zone" if t90 - t10 >= 20 else "")
    else:
        zone = stat_tile("Top of change zone", "–", "")
    shift = vol.get("time_shift_ms")
    if shift is not None and abs(shift) < 0.05:
        shift = 0.0  # avoid showing "-0.0 ms"
    if shift is None or not vol["significant"]:
        ts = stat_tile("Time shift below zone", "–", "not measured")
    elif abs(shift) < 1:
        ts = stat_tile("Time shift below zone", f"{shift:+.1f} ms", "none detected")
    else:
        ts = stat_tile("Time shift below zone", f"{shift:+.1f} ms", "push-down" if shift > 0 else "pull-up")
    nrms = vol["nrms_shallow"]
    rep = stat_tile("Repeatability (NRMS)", f"{nrms:.0f}%" if np.isfinite(nrms) else "–",
                    f"{vol['rating']} · shallow section" if vol["rating"] else "")
    changed = stat_tile("Changed area", f"{vol['changed_pct']:.1f}%",
                        f"of {vol['n_traces']:,} sampled traces")
    return html(f'<div class="tiles">{changed}{amp}{zone}{ts}{rep}</div>', sizing_mode="stretch_width")


def add_colorbar(p, mapper, title):
    # Horizontal, under the plot: a colour bar on the right pushes Bokeh's toolbar
    # outside the card when two plots share a row.
    cb = ColorBar(color_mapper=mapper, orientation="horizontal", height=8, title=title,
                  title_standoff=6, major_label_text_font_size="10px", title_text_font_size="10.5px",
                  title_text_font_style="normal", padding=4, margin=12)
    p.add_layout(cb, "below")


def analysis_figure(x_range, y_range, x_label, y_label, value_label, height=380):
    p = make_figure(x_range, y_range, x_label, y_label, height=height, value_label=value_label)
    ws["figs"].remove(p)          # analysis plots live in their own list,
    ws["afigs"].append(p)         # so rebuilding the survey plots doesn't drop them
    return p


def plot_with_caption(p, caption):
    """Caption above the plot (kept out of the Bokeh title row, where a long
    title would push the toolbar off the edge of a half-width plot)."""
    return pn.Column(html(f'<div class="plot-caption">{escape(caption)}</div>', sizing_mode="stretch_width"),
                     pn.pane.Bokeh(p, sizing_mode="stretch_width", margin=0),
                     sizing_mode="stretch_width", margin=(12, 0, 0, 0))


def change_map_plot(vol, sl):
    il, xl, change = vol["map_il"], vol["map_xl"], vol["map"]
    x0, x1 = extent(xl)
    y0, y1 = extent(il)
    hi = float(np.nanpercentile(change, 99.5)) if np.isfinite(change).any() else 1.0
    mapper = LinearColorMapper(palette=MAP_PALETTE, low=0, high=hi or 1.0, nan_color="rgba(0,0,0,0)")
    p = analysis_figure(Range1d(x0, x1, bounds=(x0, x1)), Range1d(y0, y1, bounds=(y0, y1)),
                        "Crossline", "Inline", "RMS change")
    p.image(image=[change], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
    if vol["significant"] and "box" in vol:
        b0, b1, c0, c1 = vol["box"]
        dx, dy = spacing(xl) / 2, spacing(il) / 2
        p.quad(left=c0 - dx, right=c1 + dx, bottom=b0 - dy, top=b1 + dy,
               fill_alpha=0, line_color=MARK_COLOR, line_width=1.5, line_dash="dashed")
    if sl is not None and sl["view"] in ("Inline", "Crossline"):
        dim = "width" if sl["view"] == "Inline" else "height"
        p.add_layout(Span(location=sl["pos"], dimension=dim, line_color=MARK_COLOR, line_width=1.5))
    add_colorbar(p, mapper, "RMS amplitude change")
    style_figure(p)
    return plot_with_caption(p, "Whole volume · RMS amplitude change (map view)")


def slice_difference_plot(sl, mon_label, base_label):
    aligned, diff = sl["aligned"], sl["diff"]
    x_label, y_label, y_down = VIEW_AXES[sl["view"]]
    x0, x1 = extent(aligned["xs"])
    y0, y1 = extent(aligned["ys"])
    lim = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 1.0
    lim = lim or 1.0
    mapper = LinearColorMapper(palette=palette_from_stops(DIFF_STOPS), low=-lim, high=lim,
                               nan_color="rgba(0,0,0,0)")
    y_range = Range1d(y1, y0, bounds=(y0, y1)) if y_down else Range1d(y0, y1, bounds=(y0, y1))
    p = analysis_figure(Range1d(x0, x1, bounds=(x0, x1)), y_range, x_label, y_label, "Change")
    p.image(image=[diff], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
    add_colorbar(p, mapper, f"Amplitude change: − weaker in {mon_label} · stronger in {mon_label} +")
    style_figure(p)
    return plot_with_caption(p, f"Current slice · difference ({mon_label} − {base_label}) · "
                                f"{position_text(sl['view'], sl['pos'])}")


def summary_html(summary):
    def column(lang, title, h_vol, h_slice):
        s = summary[lang]
        vol = "".join(f"<p>{escape(t)}</p>" for t in s["volume"]) or "<p>–</p>"
        slc = "".join(f"<p>{escape(t)}</p>" for t in s["slice"]) or "<p>–</p>"
        return (f'<div class="sum-col"><div class="sum-lang">{title}</div>'
                f'<h4>{h_vol}</h4>{vol}<h4>{h_slice}</h4>{slc}'
                f'<p class="sum-note">{escape(s["note"])}</p></div>')
    return html('<div class="summary">'
                + column("en", "English", "Whole volume", "Current slice")
                + column("ms", "Bahasa Melayu", "Keseluruhan volum", "Keratan semasa")
                + "</div>", sizing_mode="stretch_width")


def slice_key(sl):
    """Which slice a summary was written for."""
    return None if sl is None else (sl["view"], round(sl["pos"], 3))


def render_analysis():
    """(Re)draw the analysis card. The whole-volume part is kept from the last
    run; the slice part is recomputed for whatever slice is on screen now.
    Gemma's summary is shown while the slice is the one it was written for;
    after the slice changes, the rule-based text is shown until the next run."""
    an = ws["analysis"]
    if not an:
        return
    sl = current_slice_analysis()
    an["slice"] = sl
    base_label, mon_label = an["labels"]
    rule_based = write_summary(an["volume"], sl, base_label, mon_label)
    an["rule_summary"] = rule_based
    gem = an.get("gemma")
    if gem and gem["key"] == slice_key(sl):
        an["summary"] = gem["summary"]
    else:
        an["summary"] = rule_based
        if gem:
            for lang, extra in (("en", " The slice changed since Gemma wrote its summary; "
                                       "press Run analysis to ask Gemma again."),
                                ("ms", " Keratan telah berubah sejak Gemma menulis ringkasannya; "
                                       "tekan Run analysis untuk bertanya kepada Gemma semula.")):
                rule_based[lang]["note"] += extra
    ws["afigs"] = []
    plots = [change_map_plot(an["volume"], sl)]
    if sl is not None:
        plots.append(slice_difference_plot(sl, mon_label, base_label))
    analysis_body.objects = [
        stats_html(an["volume"]),
        pn.Row(*plots, sizing_mode="stretch_width", margin=0, styles={"gap": "16px"}),
        summary_html(an["summary"]),
    ]


def reset_analysis(message=None):
    ws["analysis"] = None
    ws["afigs"] = []
    analysis_body.objects = []
    pdf_btn.disabled = True
    analysis_status.object = message or ('<span class="status">Compares the whole volume and the slice on '
                                         'screen. Takes a few seconds on large files.</span>')


async def on_run(event=None):
    base, mon = ws["surveys"]
    if base["vol"] is None or mon["vol"] is None:
        reset_analysis('<span class="status err">Load both surveys first.</span>')
        return
    set_label(run_btn, "Analysing…")
    run_btn.disabled = True
    try:
        vol = analyse_volume(base["vol"], mon["vol"])
        an = {"volume": vol, "labels": (base["title"], mon["title"]),
              "files": (base["name"], mon["name"]), "credit": ws["credit"]}
        ws["analysis"] = an
        render_analysis()  # rule-based summary straight away
        pdf_btn.filename = pdf_filename()
        pdf_btn.disabled = False
        step = vol["sample_step"]
        sampled = ('<span class="status"> · whole volume sampled '
                   + ("on every line" if step == 1 else f"every {step} lines")
                   + ' · the slice part follows the slice you choose above</span>')
        if not USE_GEMMA:
            analysis_status.object = '<span class="status ok">Done</span>' + sampled
            return

        set_label(run_btn, "Asking Gemma…")
        pdf_btn.disabled = True  # until the final text is in
        analysis_status.object = ('<span class="status">Measurements done · Gemma is writing the summary '
                                  '(rule-based text shown meanwhile)…</span>')
        sl = an["slice"]
        args = (vol, sl, *an["labels"])
        kwargs = {"draft": an["rule_summary"], "slice_axes": VIEW_AXES[sl["view"]] if sl else None,
                  "base_url": GEMMA_URL, "model": GEMMA_MODEL}
        try:
            summary = await asyncio.to_thread(gemma_client.summarise, *args, **kwargs)
        except Exception as exc:
            if ws["analysis"] is an:
                pdf_btn.disabled = False
                analysis_status.object = (f'<span class="status err">Gemma unavailable ({escape(str(exc)[:160])})'
                                          ' · showing the rule-based summary</span>')
            return
        if ws["analysis"] is not an:  # new data was loaded while Gemma was writing
            return
        an["gemma"] = {"key": slice_key(sl), "summary": summary}
        render_analysis()
        pdf_btn.disabled = False
        analysis_status.object = '<span class="status ok">Done · summary by Gemma</span>' + sampled
    except Exception as exc:
        reset_analysis(f'<span class="status err">Analysis failed: {escape(str(exc))}</span>')
    finally:
        set_label(run_btn, "Run analysis")
        run_btn.disabled = False


# ---------------------------------------------------------------------------
# PDF export (matplotlib; always printed on white, whatever the app theme)
# ---------------------------------------------------------------------------
def make_pdf():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    an = ws["analysis"]
    vol, sl, summary = an["volume"], an.get("slice"), an["summary"]
    base_label, mon_label = an["labels"]
    credit = an.get("credit", "").strip()
    a4 = (8.27, 11.69)
    ink, grey = "#13212a", "#5b6b73"
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")
    diff_cmap = LinearSegmentedColormap.from_list("diff", [c for _, c in DIFF_STOPS])
    map_cmap = LinearSegmentedColormap.from_list("map", MAP_PALETTE)

    def footer(fig, page):
        if credit:
            for k, line in enumerate(textwrap.wrap(credit, 120)):
                fig.text(0.07, 0.035 - k * 0.012, line, fontsize=6.5, color=grey)
        fig.text(0.93, 0.035, f"Reservoir Time Machine · page {page}", fontsize=6.5, color=grey, ha="right")

    def text_page(fig, title, sections, note):
        fig.text(0.07, 0.94, title, fontsize=16, weight="bold", color=ink)
        fig.text(0.07, 0.918, f"{base_label} → {mon_label} · {stamp}", fontsize=9, color=grey)
        y = 0.88
        for heading, paragraphs in sections:
            fig.text(0.07, y, heading, fontsize=11.5, weight="bold", color=ink)
            y -= 0.03
            for para in paragraphs or ["–"]:
                for line in textwrap.wrap(para, 92):
                    fig.text(0.07, y, line, fontsize=10, color=ink)
                    y -= 0.019
                y -= 0.012
            y -= 0.012
        for line in textwrap.wrap(note, 100):
            fig.text(0.07, y, line, fontsize=8.5, color=grey, style="italic")
            y -= 0.016

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # page 1: key numbers and the two images
        fig = Figure(figsize=a4)
        fig.text(0.07, 0.95, "4D Seismic Difference Analysis", fontsize=18, weight="bold", color=ink)
        fig.text(0.07, 0.928, f"{base_label} (baseline) vs {mon_label} (monitor) · {stamp}", fontsize=9.5, color=grey)
        fig.text(0.07, 0.912, f"Files: {an['files'][0]} · {an['files'][1]}", fontsize=8.5, color=grey)

        rows = [("Changed area", f"{vol['changed_pct']:.1f}% of {vol['n_traces']:,} sampled traces")]
        ratio = vol.get("amp_ratio", np.nan)
        if vol["significant"] and np.isfinite(ratio):
            rows.append(("Amplitude in change zone", f"{(ratio - 1) * 100:+.0f}% "
                         + ("(brightening)" if ratio >= 1.1 else "(dimming)" if ratio <= 0.9 else "(mixed)")))
        if vol["significant"] and "t_change" in vol:
            t10, t50, t90 = vol["t_change"]
            rows.append(("Top of change zone", f"about {t50:,.0f} ms"
                         + (f" ({t10:,.0f}–{t90:,.0f} ms across the zone)" if t90 - t10 >= 20 else "")))
        if vol["significant"] and "box" in vol:
            b0, b1, c0, c1 = vol["box"]
            rows.append(("Change zone (area)", f"inlines {b0:,.0f}–{b1:,.0f}, crosslines {c0:,.0f}–{c1:,.0f}"))
        if vol["significant"] and vol.get("time_shift_ms") is not None:
            shift = vol["time_shift_ms"] if abs(vol["time_shift_ms"]) >= 0.05 else 0.0
            rows.append(("Time shift below zone", f"{shift:+.1f} ms" + (" (none detected)" if abs(shift) < 1 else "")))
        if np.isfinite(vol["nrms_shallow"]):
            rows.append(("Repeatability (NRMS)", f"{vol['nrms_shallow']:.0f}% ({vol['rating']}) in the shallow section"))
        if sl is not None:
            rows.append(("Current slice", f"{position_text(sl['view'], sl['pos'])}: {sl['changed_pct']:.1f}% changed, "
                                          f"NRMS {sl['nrms']:.0f}%"))
        y = 0.88
        for label, value in rows:
            fig.text(0.07, y, label, fontsize=9.5, color=grey)
            fig.text(0.34, y, value, fontsize=9.5, color=ink)
            y -= 0.021

        top = y - 0.02
        h = (top - 0.10) / 2 - 0.05
        ax = fig.add_axes([0.09, top - h, 0.74, h])
        il, xl, change = vol["map_il"], vol["map_xl"], vol["map"]
        (x0, x1), (y0, y1) = extent(xl), extent(il)
        hi = float(np.nanpercentile(change, 99.5)) if np.isfinite(change).any() else 1.0
        im = ax.imshow(change, origin="lower", extent=(x0, x1, y0, y1), aspect="auto",
                       cmap=map_cmap, vmin=0, vmax=hi or 1.0, interpolation="nearest")
        if vol["significant"] and "box" in vol:
            b0, b1, c0, c1 = vol["box"]
            ax.add_patch(Rectangle((c0, b0), c1 - c0, b1 - b0, fill=False, ec=MARK_COLOR, lw=1.2, ls="--"))
        if sl is not None and sl["view"] == "Inline":
            ax.axhline(sl["pos"], color=MARK_COLOR, lw=1.2)
        elif sl is not None and sl["view"] == "Crossline":
            ax.axvline(sl["pos"], color=MARK_COLOR, lw=1.2)
        ax.set_title("Whole volume · RMS amplitude change (map view)", fontsize=10, color=ink, loc="left")
        ax.set_xlabel("Crossline", fontsize=8.5)
        ax.set_ylabel("Inline", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        fig.colorbar(im, cax=fig.add_axes([0.85, top - h, 0.02, h])).ax.tick_params(labelsize=7)

        if sl is not None:
            ax2 = fig.add_axes([0.09, 0.10, 0.74, h])
            aligned, diff = sl["aligned"], sl["diff"]
            x_label, y_label, y_down = VIEW_AXES[sl["view"]]
            (x0, x1), (y0, y1) = extent(aligned["xs"]), extent(aligned["ys"])
            lim = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 1.0
            im2 = ax2.imshow(diff, origin="lower", extent=(x0, x1, y0, y1), aspect="auto",
                             cmap=diff_cmap, vmin=-(lim or 1), vmax=lim or 1, interpolation="nearest")
            if y_down:
                ax2.invert_yaxis()
            ax2.set_title(f"Current slice · difference ({mon_label} − {base_label}) · "
                          f"{position_text(sl['view'], sl['pos'])}", fontsize=10, color=ink, loc="left")
            ax2.set_xlabel(x_label, fontsize=8.5)
            ax2.set_ylabel(y_label, fontsize=8.5)
            ax2.tick_params(labelsize=7.5)
            cb = fig.colorbar(im2, cax=fig.add_axes([0.85, 0.10, 0.02, h]))
            cb.ax.tick_params(labelsize=7)
            cb.set_label("− weaker · stronger +", fontsize=7.5)
        footer(fig, 1)
        pdf.savefig(fig)

        # pages 2 and 3: the summaries
        for page, (lang, title, h_vol, h_slice) in enumerate(
                [("en", "Summary (English)", "Whole volume", "Current slice"),
                 ("ms", "Ringkasan (Bahasa Melayu)", "Keseluruhan volum", "Keratan semasa")], start=2):
            fig = Figure(figsize=a4)
            s = summary[lang]
            text_page(fig, title, [(h_vol, s["volume"]), (h_slice, s["slice"])], s["note"])
            footer(fig, page)
            pdf.savefig(fig)

        # pages 4+: fault screening, when it has been run on these surveys
        if ws.get("hex") and ws["hex"].get("done"):
            fault_pdf_pages(pdf, ws["hex"], 4, stamp)
    buf.seek(0)
    return buf


def pdf_filename():
    an = ws.get("analysis") or {}
    labels = an.get("labels", ("baseline", "monitor"))
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"{labels[0]}_vs_{labels[1]}").strip("_")
    return f"4D_analysis_{safe}.pdf"


pdf_btn = pn.widgets.FileDownload(callback=make_pdf, filename="4D_analysis.pdf", label="Export PDF",
                                  disabled=True, button_type="default", width=120,
                                  stylesheets=[BTN_BASE_CSS])
run_btn.on_click(on_run)

analysis_card = pn.Column(
    html('<div class="head"><h2 class="title">Analysis</h2>'
         '<span class="meta">Baseline vs Monitor · automatic, rule-based</span></div>',
         sizing_mode="stretch_width"),
    pn.Row(run_btn, pdf_btn, analysis_status, sizing_mode="stretch_width", margin=(8, 0, 0, 0)),
    analysis_body,
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)


# ---------------------------------------------------------------------------
# Fault screening: hexagons + trace attributes + Gemma (logic in hexscreen.py)
# ---------------------------------------------------------------------------
VERDICT_COLOURS = {"likely": "#e5484d", "possible": "#f5a524", "fault": "#6f8fb0"}
HEX_COLOUR_BY = {"Verdict": None, "Fault score": "fault_score", "Seal score": "seal_score",
                 "4D change": "diff_rms", "Throw": "throw_ms"}
GEMMA_WORKERS = gemma_client.settings()["workers"]
HEX_INTRO = ('<span class="status">Tiles the shared area with hexagons 20 lines across, measures attributes on '
             'every trace, and asks Gemma to look for sealing faults in 5 inlines and 5 crosslines per hex.</span>')

hex_run_btn = pn.widgets.Button(name="Run fault screening", stylesheets=[BTN_BASE_CSS, PRIMARY_BTN_CSS], width=170)
hex_status = html(HEX_INTRO, sizing_mode="stretch_width", align="center")
hex_colour = pn.widgets.RadioButtonGroup(options=list(HEX_COLOUR_BY), value="Verdict", stylesheets=[TOGGLE_CSS],
                                         margin=(0, 0, 0, 0))
hex_pick = pn.widgets.Select(options={}, width=340, stylesheets=[INPUT_CSS], margin=(0, 0, 0, 0))
hex_body = pn.Column(sizing_mode="stretch_width", margin=0)
hex_detail = pn.Column(sizing_mode="stretch_width", margin=0)
hex_table = html("", sizing_mode="stretch_width")


def verdict_chip(v):
    text = hexscreen.VERDICT_TEXT[v][0] if v in hexscreen.VERDICT_TEXT else "Waiting…"
    return f'<span class="chip {v if v in hexscreen.VERDICT_TEXT else "none"}">{escape(text)}</span>'


def _fmt(v, digits=2, unit=""):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{digits}f}{unit}"


def hex_fill(state, h):
    """(fill colour, alpha) for a hex under the current colouring."""
    by = HEX_COLOUR_BY[hex_colour.value]
    if by is None:
        c = VERDICT_COLOURS.get(h.get("verdict"))
        return (c, 0.55) if c else ("#000000", 0.0)
    v = h.get(by)
    if v is None or not np.isfinite(v):
        return ("#000000", 0.0)
    hi = state["scale"].get(by) or 1.0
    t = float(np.clip(v / hi, 0, 1))
    return MAP_PALETTE[int(round(t * (len(MAP_PALETTE) - 1)))], 0.65


def hex_columns(state):
    """Column data for the hexagon patches."""
    cols = {k: [] for k in ("xs", "ys", "id", "il", "xl", "fill", "alpha", "verdict", "fault", "throw",
                            "seal_c", "seal", "chg", "gemma")}
    for h in state["ctx"]["hexes"]:
        fill, alpha = hex_fill(state, h)
        g = h.get("gemma") or {}
        cols["xs"].append(h["corners_xl"])
        cols["ys"].append(h["corners_il"])
        cols["id"].append(h["id"])
        cols["il"].append(round(h["il_centre"]))
        cols["xl"].append(round(h["xl_centre"]))
        cols["fill"].append(fill)
        cols["alpha"].append(alpha)
        cols["verdict"].append(hexscreen.VERDICT_TEXT[h["verdict"]][0] if "verdict" in h else "waiting for Gemma…")
        cols["fault"].append(h["fault_score"])
        cols["throw"].append(h["throw_ms"])
        cols["seal_c"].append(h["seal_contrast"])
        cols["seal"].append(h["seal_score"])
        cols["chg"].append(h["changed_pct"])
        cols["gemma"].append(("not asked" if not g else f"error: {g['error'][:60]}" if g.get("error")
                              else f"fault {g['fault']}, sealing {g['sealing']} ({g['confidence']:.0%})"))
    return cols


def label_columns(state):
    hx = [h for h in state["ctx"]["hexes"] if h.get("verdict") in VERDICT_COLOURS]
    return {"x": [h["xl_centre"] for h in hx], "y": [h["il_centre"] for h in hx],
            "text": [str(h["id"]) for h in hx]}


def hex_map(state):
    ctx = state["ctx"]
    grid, maps = ctx["grid"], ctx["maps"]
    il, xl = grid["ilines"], grid["xlines"]
    (x0, x1), (y0, y1) = extent(xl), extent(il)
    p = make_figure(Range1d(x0, x1, bounds=(x0, x1)), Range1d(y0, y1, bounds=(y0, y1)),
                    "Crossline", "Inline", height=520, value_label="4D change")
    ws["figs"].remove(p)
    ws["hfigs"].append(p)
    p.toolbar.tools = [t for t in p.toolbar.tools if not isinstance(t, HoverTool)]

    # background: 4D change (RMS of monitor - baseline) and the traces on a fault
    chg = maps["diff_rms"]
    floor = float(np.nanmedian(chg))
    hi = float(np.nanpercentile(chg, 99.5)) if np.isfinite(chg).any() else 1.0
    mapper = LinearColorMapper(palette=MAP_PALETTE, low=floor, high=max(hi, floor * 1.01 + 1e-9),
                               nan_color="rgba(0,0,0,0)")
    p.image(image=[chg], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
    on_fault = np.isfinite(maps["coherence"]) & (1 - maps["coherence"] > ctx["stats"]["fault_threshold"])
    rgba = np.zeros(on_fault.shape, dtype=np.uint32)
    view = rgba.view(np.uint8).reshape(on_fault.shape + (4,))
    view[on_fault] = (255, 255, 255, 235)
    p.image_rgba(image=[rgba], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0)
    add_colorbar(p, mapper, "4D change (RMS of monitor − baseline) · white = traces on a fault")

    src = ColumnDataSource(hex_columns(state))
    r = p.patches("xs", "ys", source=src, fill_color="fill", fill_alpha="alpha",
                  line_color="#ffffff", line_alpha=0.55, line_width=1)
    r.selection_glyph = Patches(fill_color="fill", fill_alpha="alpha", line_color=THEMES[ws["theme"]]["accent"],
                                line_alpha=1, line_width=3.5)
    r.nonselection_glyph = r.glyph.clone()
    labels = ColumnDataSource(label_columns(state))
    p.text("x", "y", "text", source=labels, text_align="center", text_baseline="middle",
           text_color="#ffffff", text_font_size="11px", text_font_style="bold")
    p.add_tools(HoverTool(renderers=[r], tooltips=[
        ("Hex", "@id (inline @il / crossline @xl)"), ("Verdict", "@verdict"), ("Gemma", "@gemma"),
        ("Fault score", "@fault{0.00}"), ("Throw", "@throw{0.0} ms"), ("Seal contrast", "@seal_c{0.00}"),
        ("Seal score", "@seal{0.00}"), ("4D change", "@chg{0} % of traces"),
    ]), TapTool(renderers=[r]))
    src.selected.on_change("indices", lambda attr, old, new: on_hex_tap(new))
    state.update(src=src, label_src=labels, fig=p)
    style_figure(p)  # again, now the colour bar is attached
    return p


def update_scales(state):
    hexes = state["ctx"]["hexes"]
    state["scale"] = {}
    for key in HEX_COLOUR_BY.values():
        if key:
            v = np.array([h.get(key, np.nan) for h in hexes], dtype=float)
            v = v[np.isfinite(v)]
            state["scale"][key] = float(np.percentile(v, 99)) if v.size and np.percentile(v, 99) > 0 else 1.0
    state["scale"]["fault_score"] = state["scale"]["seal_score"] = 1.0


def refresh_hex_colours(event=None):
    state = ws.get("hex")
    if not state or "src" not in state:
        return
    cols = hex_columns(state)
    state["src"].data.update({k: cols[k] for k in ("fill", "alpha", "verdict", "gemma")})
    state["label_src"].data = label_columns(state)


def patch_hex(state, h):
    """Update one hex on the map as soon as its result arrives."""
    idx = state["index"][h["id"]]
    fill, alpha = hex_fill(state, h)
    g = h.get("gemma") or {}
    gem = ("not asked" if not g else f"error: {g['error'][:60]}" if g.get("error")
           else f"fault {g['fault']}, sealing {g['sealing']} ({g['confidence']:.0%})")
    state["src"].patch({"fill": [(idx, fill)], "alpha": [(idx, alpha)],
                        "verdict": [(idx, hexscreen.VERDICT_TEXT[h["verdict"]][0])], "gemma": [(idx, gem)]})
    if h["verdict"] in VERDICT_COLOURS:
        state["label_src"].stream({"x": [h["xl_centre"]], "y": [h["il_centre"]], "text": [str(h["id"])]})


def ranked_hexes(state):
    return sorted(state["ctx"]["hexes"],
                  key=lambda h: (-hexscreen.VERDICT_RANK.get(h.get("verdict"), -1), -h["seal_score"],
                                 -h["fault_score"], h["id"]))


def hex_table_html(state):
    rows = [h for h in ranked_hexes(state) if h.get("verdict") in VERDICT_COLOURS][:40]
    if not rows:
        return '<p class="status">No hexagon shows a fault.</p>'
    sel = state.get("selected")
    body = "".join(
        f'<tr class="{"sel" if h["id"] == sel else ""}"><td>{h["id"]}</td>'
        f'<td>{h["il_centre"]:,.0f} / {h["xl_centre"]:,.0f}</td><td class="txt">{verdict_chip(h["verdict"])}</td>'
        f'<td>{_fmt(h["fault_score"])}</td><td>{_fmt(h["throw_ms"], 1)}</td><td>{_fmt(h["seal_contrast"])}</td>'
        f'<td>{_fmt(h["changed_pct"], 0, "%")}</td>'
        f'<td class="txt">{escape(gemma_short(h))}</td></tr>' for h in rows)
    return ('<table class="hex-table"><tr><th>Hex</th><th>Inline / crossline</th><th>Verdict</th>'
            '<th>Fault score</th><th>Throw (ms)</th><th>Seal contrast</th><th>4D change</th><th>Gemma</th></tr>'
            + body + '</table>')


def gemma_short(h):
    g = h.get("gemma")
    if not g:
        return "not asked"
    if g.get("error"):
        return "unavailable"
    return f"fault {g['fault']}, sealing {g['sealing']}, {g['confidence']:.0%}"


def hex_images(state, h):
    """Montage PNGs for one hex (re-rendered on demand, last few kept)."""
    cache = state.setdefault("images", {})
    if h["id"] not in cache:
        base, mon = ws["surveys"]
        cache[h["id"]] = hexscreen.render_montages(state["ctx"], base["vol"], mon["vol"], h)
        while len(cache) > 6:
            cache.pop(next(iter(cache)))
    return cache[h["id"]]


def png_img(png, alt):
    return f'<img alt="{escape(alt)}" src="data:image/png;base64,{base64.b64encode(png).decode()}">'


def hex_detail_html(state, h):
    inl, xln, _, _ = hexscreen.hex_lines(state["ctx"], h)
    grid = state["ctx"]["grid"]
    il_txt = ", ".join(f"{grid['ilines'][i]:.0f}" for i in inl)
    xl_txt = ", ".join(f"{grid['xlines'][j]:.0f}" for j in xln)
    lines = f"Inlines {il_txt} · crosslines {xl_txt}"
    g = h.get("gemma")
    if not g:
        gem = '<p class="status">Gemma was not asked; the verdict comes from the attributes.</p>'
    elif g.get("error"):
        gem = (f'<p class="status err">Gemma unavailable ({escape(g["error"][:160])}); '
               'the verdict comes from the attributes.</p>')
    else:
        gem = (f'<div class="summary" style="margin-top:8px">'
               f'<div class="sum-col"><div class="sum-lang">English · Gemma</div>'
               f'<p><b>Fault:</b> {escape(g["fault"])} · <b>sealing:</b> {escape(g["sealing"])} · '
               f'<b>confidence:</b> {g["confidence"]:.0%}</p>'
               + (f'<p><b>Where:</b> {escape(g["location"])}</p>' if g.get("location") else "")
               + f'<p>{escape(g.get("evidence_en", ""))}</p></div>'
               f'<div class="sum-col"><div class="sum-lang">Bahasa Melayu · Gemma</div>'
               f'<p><b>Sesar:</b> {escape(g["fault"])} · <b>pengedap:</b> {escape(g["sealing"])} · '
               f'<b>keyakinan:</b> {g["confidence"]:.0%}</p>'
               f'<p>{escape(g.get("evidence_ms", ""))}</p></div></div>')
    attrs = [("Fault score", _fmt(h["fault_score"])), ("Throw", _fmt(h["throw_ms"], 1, " ms")),
             ("Seal contrast", _fmt(h["seal_contrast"])), ("Seal score", _fmt(h["seal_score"])),
             ("Traces on fault", f'{h["fault_traces"]}'), ("Fault trend", _fmt(h["fault_trend_deg"], 0, "°")),
             ("Coherence", _fmt(h["coherence"])), ("4D change", _fmt(h["changed_pct"], 0, "% of traces")),
             ("NRMS", _fmt(h["nrms"], 0, "%")), ("Amplitude ratio", _fmt(h["amp_ratio"])),
             ("Time shift", _fmt(h["time_shift_ms"], 1, " ms")), ("Inst. frequency", _fmt(h["inst_freq_hz"], 0, " Hz")),
             ("RMS baseline", _fmt(h["rms_base"], 3)), ("RMS monitor", _fmt(h["rms_mon"], 3)),
             ("Envelope", _fmt(h["envelope"], 3)), ("Sweetness", _fmt(h["sweetness"], 3))]
    grid_html = "".join(f"<div>{k}<b>{v}</b></div>" for k, v in attrs)
    amp, diff = hex_images(state, h)
    return (f'<div class="hex-detail"><h3>Hex {h["id"]} · inline {h["il_centre"]:,.0f} / crossline '
            f'{h["xl_centre"]:,.0f} &nbsp;{verdict_chip(h["verdict"])} '
            f'<span class="meta">{escape(h["verdict_source"])}</span></h3>'
            f'<div class="lines">{lines} · {h["n_traces"]} traces</div>'
            f'{gem}<div class="attr-grid">{grid_html}</div>'
            f'<div class="hex-imgs">{png_img(amp, "monitor amplitude sections")}'
            f'{png_img(diff, "4D difference sections")}</div></div>')


def show_hex(hex_id):
    state = ws.get("hex")
    if not state or "ctx" not in state:
        return
    h = next((x for x in state["ctx"]["hexes"] if x["id"] == hex_id), None)
    if h is None or "verdict" not in h:
        return
    state["selected"] = hex_id
    hex_detail.objects = [html(hex_detail_html(state, h), sizing_mode="stretch_width")]
    hex_table.object = hex_table_html(state)
    idx = state["index"][hex_id]
    if state["src"].selected.indices != [idx]:
        state["syncing"] = True
        state["src"].selected.indices = [idx]
        state["syncing"] = False
    if hex_pick.value != hex_id:
        hex_pick.value = hex_id


def on_hex_tap(indices):
    state = ws.get("hex")
    if not state or state.get("syncing") or not indices or not state.get("done"):
        return
    show_hex(state["ctx"]["hexes"][indices[0]]["id"])


def on_hex_pick(event):
    state = ws.get("hex")
    if state and state.get("done") and event.new is not None and event.new != state.get("selected"):
        show_hex(event.new)


def hex_tiles(state):
    counts = hexscreen.summary_counts(state["ctx"]["hexes"])
    n = len(state["ctx"]["hexes"])
    return html('<div class="tiles four">'
                + "".join(stat_tile(hexscreen.VERDICT_TEXT[v][0], f"{counts[v]}", f"of {n} hexes")
                          for v in hexscreen.VERDICTS) + '</div>', sizing_mode="stretch_width")


HEX_LEGEND = ('<div class="legend">'
              + "".join(f'<span><i style="background:{c}"></i>{hexscreen.VERDICT_TEXT[v][0]}</span>'
                        for v, c in VERDICT_COLOURS.items())
              + '<span><i style="background:transparent;border:1px solid var(--border-strong)"></i>'
                'No fault found</span><span class="meta">· click a hexagon for its sections and Gemma\'s reading'
                '</span></div>')


def reset_hex(message=None):
    ws["hex"] = None
    ws["hfigs"] = []
    hex_body.objects = []
    hex_detail.objects = []
    hex_table.object = ""
    hex_pick.options = {}
    hex_csv_btn.disabled = hex_pdf_btn.disabled = True
    hex_status.object = message or HEX_INTRO


def in_doc(doc, fn, *args):
    """Run fn on the document's next tick (with its lock). Raw Bokeh objects such as
    the hex map's data source may only be changed that way from an async callback."""
    if doc is None:
        fn(*args)
    else:
        doc.add_next_tick_callback(functools.partial(fn, *args))


async def on_hex_run(event=None):
    doc = pn.state.curdoc
    surveys = ws["surveys"]
    if len(surveys) != 2 or any(s_["vol"] is None for s_ in surveys):
        reset_hex('<span class="status err">Load both surveys first.</span>')
        return
    base, mon = surveys
    reset_hex('<span class="status">Measuring attributes on every trace…</span>')
    state = {"labels": (base["title"], mon["title"]), "files": (base["name"], mon["name"]),
             "credit": ws["credit"], "done": False, "selected": None}
    ws["hex"] = state
    set_label(hex_run_btn, "Screening…")
    hex_run_btn.disabled = True
    try:
        ctx = await asyncio.to_thread(hexscreen.prepare, base["vol"], mon["vol"])
        if ws["hex"] is not state:
            return
        if not ctx["hexes"]:
            raise ValueError("no hexagon fits inside the area the two surveys share")
        state["ctx"] = ctx
        state["index"] = {h["id"]: k for k, h in enumerate(ctx["hexes"])}
        update_scales(state)
        hex_body.objects = [
            pn.Row(html('<span class="meta">Colour hexagons by</span>', align="center", margin=(0, 10, 0, 0)),
                   hex_colour, sizing_mode="stretch_width", margin=(12, 0, 0, 0)),
            html(HEX_LEGEND, sizing_mode="stretch_width"),
            plot_with_caption(hex_map(state), "Hexagons over the 4D change map"),
        ]
        total = len(ctx["hexes"])
        use = USE_GEMMA
        client, breaker = None, hexscreen.GemmaBreaker()
        if use:
            try:
                client = gemma_client.make_client(gemma_client.settings(GEMMA_URL, GEMMA_MODEL))
            except Exception as exc:
                use, breaker.first_error = False, str(exc)
        sem = asyncio.Semaphore(GEMMA_WORKERS)
        hex_status.object = (f'<span class="status">{total} hexagons · '
                             + ("asking Gemma… 0 / " + str(total) if use else "attributes only") + '</span>')

        async def one(h):
            async with sem:
                await asyncio.to_thread(hexscreen.screen_hex, ctx, base["vol"], mon["vol"], h, use,
                                        state["labels"], client, GEMMA_URL, GEMMA_MODEL, breaker)
            return h

        done = 0
        for fut in asyncio.as_completed([one(h) for h in ctx["hexes"]]):
            h = await fut
            if ws["hex"] is not state:
                return
            done += 1
            in_doc(doc, patch_hex, state, h)
            if use:
                hex_status.object = f'<span class="status">Asking Gemma… {done} / {total} hexagons</span>'

        state["done"] = True
        hex_pick.options = {f"Hex {h['id']} · {hexscreen.VERDICT_TEXT[h['verdict']][0]}": h["id"]
                            for h in ranked_hexes(state)}
        state["selected"] = ranked_hexes(state)[0]["id"]  # shown below; set the dropdown to match
        hex_pick.value = state["selected"]
        hex_body.objects = [hex_tiles(state)] + list(hex_body.objects) + [
            html('<div class="plot-caption" style="margin-top:14px">Hexagons with a fault, strongest first</div>',
                 sizing_mode="stretch_width"),
            hex_table,
            pn.Row(html('<span class="meta">Show hexagon</span>', align="center", margin=(0, 10, 0, 0)),
                   hex_pick, sizing_mode="stretch_width", margin=(14, 0, 0, 0)),
            hex_detail,
        ]
        in_doc(doc, show_hex, ranked_hexes(state)[0]["id"])
        hex_csv_btn.filename = pdf_filename().replace("4D_analysis", "fault_screening").replace(".pdf", ".csv")
        hex_pdf_btn.filename = pdf_filename().replace("4D_analysis", "fault_screening")
        hex_csv_btn.disabled = hex_pdf_btn.disabled = False
        counts = hexscreen.summary_counts(ctx["hexes"])
        failed = sum(1 for h in ctx["hexes"] if (h.get("gemma") or {}).get("error"))
        msg = (f'<span class="status ok">Done</span><span class="status"> · {total} hexagons · '
               f'{counts["likely"]} likely and {counts["possible"]} possible sealing faults, '
               f'{counts["fault"]} other faults</span>')
        if use and failed:
            err = breaker.first_error or next(h["gemma"]["error"] for h in ctx["hexes"]
                                              if (h.get("gemma") or {}).get("error"))
            msg += (f'<br><span class="status err">Gemma unavailable for {failed} of {total} hexagons '
                    f'({escape(err[:140])}); those use the attributes only.</span>')
        elif not use and USE_GEMMA:
            msg += f'<br><span class="status err">Gemma unavailable ({escape(str(breaker.first_error)[:140])}).</span>'
        hex_status.object = msg
    except Exception as exc:
        if ws["hex"] is state:
            reset_hex(f'<span class="status err">Fault screening failed: {escape(str(exc))}</span>')
    finally:
        set_label(hex_run_btn, "Run fault screening")
        hex_run_btn.disabled = False


def make_hex_csv():
    state = ws["hex"]
    return io.BytesIO(hexscreen.to_csv(state["ctx"]["hexes"]).encode("utf-8"))


def fault_pdf_pages(pdf, state, first_page, stamp):
    """Fault screening pages for a PDF: hex map, flagged-hex table, evidence, top montages."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Polygon
    from PIL import Image

    ctx, (base_label, mon_label) = state["ctx"], state["labels"]
    credit = (state.get("credit") or "").strip()
    a4 = (8.27, 11.69)
    ink, grey = "#13212a", "#5b6b73"
    page = first_page
    hexes = ctx["hexes"]
    ranked = ranked_hexes(state)
    flagged = [h for h in ranked if h.get("verdict") in VERDICT_COLOURS]
    counts = hexscreen.summary_counts(hexes)

    def footer(fig):
        if credit:
            for k, line in enumerate(textwrap.wrap(credit, 120)):
                fig.text(0.07, 0.035 - k * 0.012, line, fontsize=6.5, color=grey)
        fig.text(0.93, 0.035, f"Reservoir Time Machine · page {page}", fontsize=6.5, color=grey, ha="right")

    # map page
    fig = Figure(figsize=a4)
    fig.text(0.07, 0.95, "Sealing Fault Screening", fontsize=18, weight="bold", color=ink)
    fig.text(0.07, 0.928, f"{base_label} (baseline) vs {mon_label} (monitor) · {stamp}", fontsize=9.5, color=grey)
    st = ctx["stats"]
    rows = [("Hexagons", f"{len(hexes)} · {st['hex_width']} lines across · 5 inlines + 5 crosslines each "
                         f"(centre {', '.join(f'{d:+d}' for d in st['line_offsets'])})"),
            ("Sealing fault likely", f"{counts['likely']}"), ("Sealing fault possible", f"{counts['possible']}"),
            ("Fault, no sealing evidence", f"{counts['fault']}"), ("No fault found", f"{counts['none']}"),
            ("Method", "trace attributes (coherence, throw, 4D change, seal contrast) + Gemma reading of the sections")]
    y = 0.89
    for label, value in rows:
        fig.text(0.07, y, label, fontsize=9.5, color=grey)
        for k, part in enumerate(textwrap.wrap(value, 78)):
            fig.text(0.34, y, part, fontsize=9.5, color=ink)
            y -= 0.021 if k == 0 else 0.017
        y -= 0.0
    ax = fig.add_axes([0.09, 0.14, 0.82, y - 0.17])
    grid, maps = ctx["grid"], ctx["maps"]
    (x0, x1), (y0, y1) = extent(grid["xlines"]), extent(grid["ilines"])
    chg = maps["diff_rms"]
    floor = float(np.nanmedian(chg))
    hi = float(np.nanpercentile(chg, 99.5))
    ax.imshow(chg, origin="lower", extent=(x0, x1, y0, y1), aspect="auto", interpolation="nearest",
              cmap=LinearSegmentedColormap.from_list("map", MAP_PALETTE), vmin=floor, vmax=max(hi, floor + 1e-9))
    on_fault = np.isfinite(maps["coherence"]) & (1 - maps["coherence"] > st["fault_threshold"])
    overlay = np.zeros(on_fault.shape + (4,))
    overlay[on_fault] = (1, 1, 1, 0.95)
    ax.imshow(overlay, origin="lower", extent=(x0, x1, y0, y1), aspect="auto", interpolation="nearest")
    for h in hexes:
        c = VERDICT_COLOURS.get(h.get("verdict"))
        ax.add_patch(Polygon(list(zip(h["corners_xl"], h["corners_il"])), closed=True,
                             fc=c or "none", alpha=0.55 if c else 1, ec="white", lw=0.6))
        if c:
            ax.text(h["xl_centre"], h["il_centre"], str(h["id"]), ha="center", va="center", fontsize=7,
                    color="white", weight="bold")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Crossline", fontsize=8.5)
    ax.set_ylabel("Inline", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    ax.set_title("Hexagons over the 4D change map (white = traces on a fault)", fontsize=10, color=ink, loc="left")
    ax.legend(handles=[Patch(fc=c, alpha=0.7, label=hexscreen.VERDICT_TEXT[v][0]) for v, c in VERDICT_COLOURS.items()],
              loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=8, frameon=False)
    footer(fig)
    pdf.savefig(fig)
    page += 1

    # table + evidence pages (English, then Malay)
    for lang, title in (("en", "Hexagons with a fault"), ("ms", "Heksagon dengan sesar")):
        fig = Figure(figsize=a4)
        fig.text(0.07, 0.95, title, fontsize=16, weight="bold", color=ink)
        fig.text(0.07, 0.93, f"{base_label} → {mon_label} · {stamp}", fontsize=9, color=grey)
        y = 0.895
        if lang == "en":
            head = ["Hex", "Inline / xline", "Verdict", "Fault", "Throw ms", "Seal c.", "4D chg", "Gemma"]
            xs = [0.07, 0.12, 0.25, 0.47, 0.54, 0.62, 0.69, 0.76]
            for x, t in zip(xs, head):
                fig.text(x, y, t, fontsize=8, color=grey)
            y -= 0.018
            for h in flagged[:18]:
                vals = [str(h["id"]), f"{h['il_centre']:.0f} / {h['xl_centre']:.0f}",
                        hexscreen.VERDICT_TEXT[h["verdict"]][0], _fmt(h["fault_score"]), _fmt(h["throw_ms"], 1),
                        _fmt(h["seal_contrast"]), _fmt(h["changed_pct"], 0, "%"), gemma_short(h)]
                for x, t in zip(xs, vals):
                    fig.text(x, y, t, fontsize=7.8, color=ink)
                y -= 0.017
            if len(flagged) > 18:
                fig.text(0.07, y, f"… and {len(flagged) - 18} more in the CSV export", fontsize=7.8, color=grey)
                y -= 0.017
            y -= 0.015
        for h in flagged[:8]:
            g = h.get("gemma") or {}
            fig.text(0.07, y, f"Hex {h['id']} · {hexscreen.VERDICT_TEXT[h['verdict']][0 if lang == 'en' else 1]}"
                              f" · inline {h['il_centre']:.0f} / crossline {h['xl_centre']:.0f}",
                     fontsize=9.5, weight="bold", color=ink)
            y -= 0.019
            if g and not g.get("error"):
                text = (g.get("evidence_en") if lang == "en" else g.get("evidence_ms")) or ""
                if g.get("location"):
                    text += (" Where: " if lang == "en" else " Lokasi: ") + g["location"].rstrip(". ") + "."
            else:
                text = ("Verdict from the attributes only (Gemma not available)." if lang == "en"
                        else "Keputusan berdasarkan atribut sahaja (Gemma tidak tersedia).")
            text += (f" Fault score {_fmt(h['fault_score'])}, throw {_fmt(h['throw_ms'], 1)} ms, "
                     f"seal contrast {_fmt(h['seal_contrast'])}.")
            for line in textwrap.wrap(text, 100):
                fig.text(0.07, y, line, fontsize=8.8, color=ink)
                y -= 0.016
            y -= 0.012
            if y < 0.1:
                break
        if not flagged:
            fig.text(0.07, y, "No hexagon shows a fault." if lang == "en" else "Tiada heksagon menunjukkan sesar.",
                     fontsize=10, color=ink)
        footer(fig)
        pdf.savefig(fig)
        page += 1

    # sections of the top hexagons
    base, mon = ws["surveys"]
    for h in flagged[:3]:
        amp, diff = hex_images(state, h)
        fig = Figure(figsize=a4)
        fig.text(0.07, 0.95, f"Hex {h['id']} · {hexscreen.VERDICT_TEXT[h['verdict']][0]}", fontsize=15,
                 weight="bold", color=ink)
        fig.text(0.07, 0.93, f"inline {h['il_centre']:.0f} / crossline {h['xl_centre']:.0f} · "
                             f"{h['verdict_source']}", fontsize=9, color=grey)
        for k, png in enumerate((amp, diff)):
            img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
            hh = 0.4
            a = fig.add_axes([0.05, 0.5 - k * 0.43, 0.9, hh])
            a.imshow(img)
            a.axis("off")
        footer(fig)
        pdf.savefig(fig)
        page += 1
    return page


def make_hex_pdf():
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fault_pdf_pages(pdf, ws["hex"], 1, datetime.now().strftime("%d %b %Y, %H:%M"))
    buf.seek(0)
    return buf


hex_csv_btn = pn.widgets.FileDownload(callback=make_hex_csv, filename="fault_screening.csv", label="Export CSV",
                                      disabled=True, button_type="default", width=120, stylesheets=[BTN_BASE_CSS])
hex_pdf_btn = pn.widgets.FileDownload(callback=make_hex_pdf, filename="fault_screening.pdf", label="Fault report",
                                      disabled=True, button_type="default", width=130, stylesheets=[BTN_BASE_CSS])
hex_run_btn.on_click(on_hex_run)
hex_colour.param.watch(refresh_hex_colours, "value")
hex_pick.param.watch(on_hex_pick, "value")

hex_card = pn.Column(
    html('<div class="head"><h2 class="title">Fault screening</h2>'
         '<span class="meta">Hexagons · trace attributes · Gemma</span></div>', sizing_mode="stretch_width"),
    pn.Row(hex_run_btn, hex_csv_btn, hex_pdf_btn, hex_status, sizing_mode="stretch_width", margin=(8, 0, 0, 0)),
    hex_body,
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)


load_btn.on_click(rebuild)


def on_theme_change(event):
    ws["theme"] = "light" if event.new else "dark"
    for p in ws["figs"] + ws["afigs"] + ws["hfigs"]:
        style_figure(p)


theme_toggle.param.watch(on_theme_change, "value")
view_select.param.watch(on_view_change, "value")
position_input.param.watch(update_slice, "value")

# ---------------------------------------------------------------------------
# Time-lapse
# ---------------------------------------------------------------------------
# pair_label = html("", sizing_mode="stretch_width")
# track = html("", sizing_mode="stretch_width", margin=(10, 0, 0, 0))
#
# step_buttons = []
# for m in MONITORS:
#     dot = "● " if m["id"] > 0 else ""
#     btn = pn.widgets.Button(
#         name=f"{dot}{m['short']}\n{m['long']}",
#         disabled=not m["available"],
#         stylesheets=[STEP_CSS],
#         sizing_mode="stretch_width",
#         margin=(0, 3),
#     )
#     btn.on_click(lambda event, i=m["id"]: on_step_click(i))
#     step_buttons.append(btn)
#
# play_btn = pn.widgets.Button(name="▶", stylesheets=[PLAY_CSS], width=44, height=44, margin=(0, 12, 0, 0))
#
# timelapse = pn.Column(
#     pair_label,
#     pn.Row(
#         play_btn,
#         pn.Column(
#             pn.Row(*step_buttons, sizing_mode="stretch_width", margin=0),
#             track,
#             sizing_mode="stretch_width",
#             margin=0,
#         ),
#         sizing_mode="stretch_width",
#         margin=(12, 0),
#         align="center",
#     ),
#     html('<p class="note">Each step is compared against the baseline survey.</p>'),
#     styles=CARD_STYLES,
#     sizing_mode="stretch_width",
#     margin=(8, 0),
# )

# --- Time-lapse rendering (disabled) ---
# def render():
#     sel = state["selected"]
#
#     # step buttons
#     for btn, m in zip(step_buttons, MONITORS):
#         btn.css_classes = ["active"] if m["id"] == sel else []
#
#     # header label + progress
#     label = "Baseline (reference)" if sel == 0 else f"{MONITORS[sel]['long']} − Baseline"
#     pair_label.object = f'<div class="head"><span class="label">TIME-LAPSE</span><span class="meta">{label}</span></div>'
#     pct = (sel + 1) / len(MONITORS) * 100
#     track.object = f'<div class="track"><div class="fill" style="width:{pct:.0f}%"></div></div>'
#
#     set_label(play_btn, "❚❚" if state["timer"] else "▶")


# --- Time-lapse playback (disabled) ---
# def select(i):
#     state["selected"] = i
#     render()
#
#
# def playable_ids():
#     return [m["id"] for m in MONITORS if m["available"] and m["id"] > 0]
#
#
# def advance():
#     ids = playable_ids()
#     cur = state["selected"]
#     nxt = ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0]
#     select(nxt)
#
#
# def stop_playback():
#     if state["timer"] is not None:
#         state["timer"].stop()
#         state["timer"] = None
#
#
# def toggle_play(event=None):
#     if state["timer"] is not None:
#         stop_playback()
#     else:
#         state["timer"] = pn.state.add_periodic_callback(advance, period=1400)
#     render()
#
#
# def on_step_click(i):
#     stop_playback()
#     select(i)
#
#
# play_btn.on_click(toggle_play)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Everything except Data sources stays hidden until Run loads some data.
workspace = pn.Column(
    # timelapse,  # time-lapse (disabled)
    slice_bar,
    display_bar,
    content,
    analysis_card,
    hex_card,
    sizing_mode="stretch_width",
    margin=0,
    visible=False,
)

main = pn.Column(
    header,
    sources_card,
    workspace,
    sizing_mode="stretch_width",
    margin=(20, 24, 32, 24 + SIDEBAR_WIDTH),  # leave room for the pinned sidebar
)

pn.Row(
    sidebar,
    main,
    sizing_mode="stretch_width",  # grows with the content, so the sidebar can reach the bottom
    styles={"background": BG, "min-height": "100vh"},
    margin=0,
).servable(title="Reservoir Time Machine")