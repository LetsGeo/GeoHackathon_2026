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

Needs:  pip install segyio numpy matplotlib   (matplotlib draws the PDF export and the picture in the shapes download)
The first load of a file reads the whole SEG-Y once and saves it in "cache/"
(about the same size as the file), so inline, crossline and time slices are
instant afterwards. Delete the cache folder any time to free the space.

Tested with Panel 1.4 / Bokeh 3.4 on Python 3.9+.
"""
import argparse
import asyncio
import base64
import functools
import gc
import hashlib
import io
import os
import re
import struct
import subprocess
import sys
import textwrap
import zipfile
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import panel as pn
import param
from bokeh.models import Button as BkButton
from bokeh.models import (
    BoxZoomTool, ColorBar, ColumnDataSource, CustomAction, CustomJS, HoverTool, InlineStyleSheet, LabelSet, LinearColorMapper,
    PanTool, PolyDrawTool, PolyEditTool, Range1d, ResetTool, Span, WheelZoomTool,
)
from bokeh.plotting import figure

sys.path.insert(0, str(Path(__file__).parent))  # so the app's own modules are found however it's started
from shapefiles import SHAPE_EXTS, closed_ring, read_dbf, read_shp, record_label, write_shapefile  # noqa: E402
from styles import *  # noqa: E402,F401,F403  (colours, fonts, CSS, icons: see styles.py)
import horizons  # noqa: E402  (horizons loaded from files or drawn on sections)
from loading import LoadingOverlay  # noqa: E402  (full-screen pumpjack overlay while Run works)
from types import SimpleNamespace  # noqa: E402

LOGO_PATH = Path(__file__).parent / "assets" / "Logo.png"
LOGO_SRC = "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()

# ---------------------------------------------------------------------------
# Page set-up: the font and the theme colours (defined in styles.py)
# ---------------------------------------------------------------------------
pn.extension(
    raw_css=[
        FONT_IMPORT_CSS,
        theme_css(),
        "html, body { margin: 0; background: var(--bg); transition: background .2s; }",
        "body { background: var(--bg-grad); background-attachment: fixed; }",
        # slim scrollbars that match the theme
        "* { scrollbar-width: thin; scrollbar-color: var(--border-strong) transparent; }",
        "::selection { background: var(--accent-soft); }",
    ]
)


# Colourbar scale: amplitude runs from -AMP_MAX (trough) to +AMP_MAX (peak),
# split into CBAR_DIVISIONS equal sections. AMP_MAX is replaced by the data's
# 99th-percentile amplitude once a seismic file loads (32 is only the fallback).
AMP_MAX = 32
CBAR_DIVISIONS = 10


def set_label(button, text):
    """Button.name is a constant param in newer versions of param, so unlock it to relabel."""
    with param.edit_constant(button):
        button.name = text


def html(content, **kwargs):
    """HTML pane with the shared stylesheet applied."""
    kwargs.setdefault("margin", 0)
    return pn.pane.HTML(content, stylesheets=[BASE_CSS], **kwargs)


APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "cache"

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
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


ARGS = parse_args()

# Which inline to display (None = the middle inline of the baseline survey; the
# monitor always shows the same inline so the two can be compared), and an
# optional time window in ms, e.g. (700, 1200) to zoom in on the reservoir.
INLINE = ARGS.inline
TIME_WINDOW_MS = None


def resolve_path(text):
    """Accept absolute paths, ~ paths, or paths relative to the app's folder."""
    p = Path(text.strip().strip('"').strip("'")).expanduser()
    return p if p.is_absolute() else APP_DIR / p


# ---------------------------------------------------------------------------
# Seismic loading
# ---------------------------------------------------------------------------
# Byte positions where inline / crossline numbers are commonly stored in SEG-Y
# trace headers. The loader tries each pair until one describes a proper 3D grid.
HEADER_CANDIDATES = [(189, 193), (9, 21), (5, 21), (17, 13), (9, 13)]
CHUNK_TRACES = 20_000  # traces read per step when building the cache

VIEWS = ("Inline", "Crossline", "Time slice")
# per view: x-axis label, y-axis label, does y (time) increase downwards?
VIEW_AXES = {
    "Inline": ("Crossline", "Time (ms)", True),
    "Crossline": ("Inline", "Time (ms)", True),
    "Time slice": ("Crossline", "Inline", False),
}
VIEW_KEY = {"Inline": "ilines", "Crossline": "xlines", "Time slice": "times"}


def cube_cache_dir(path):
    """Cache folder tied to the file's full path, size and date, so different
    datasets (even with the same file name) never share a cache."""
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{path.name}_{key}"


def robust_amp(cube, target=2_000_000):
    """99th percentile of |amplitude| from an evenly spaced sample of the cube."""
    n_il, n_xl, _ = cube.shape
    step = 1
    while (n_il // step) * (n_xl // step) * cube.shape[2] > target and step < max(n_il, n_xl):
        step += 1
    sample = np.abs(np.asarray(cube[::step, ::step, :], dtype=np.float32))
    value = np.nanpercentile(sample, 99) if np.isfinite(sample).any() else 1.0
    return float(value)


def find_line_numbers(f, step=1):
    """Inline and crossline number of every trace (or every step-th trace) in an
    open segyio file, trying each common header position (None, None if none fit)."""
    n = len(range(0, f.tracecount, step))
    for il_byte, xl_byte in HEADER_CANDIDATES:
        il_try = np.asarray(f.attributes(il_byte)[::step]).astype(np.int64)
        xl_try = np.asarray(f.attributes(xl_byte)[::step]).astype(np.int64)
        n_il, n_xl = len(np.unique(il_try)), len(np.unique(xl_try))
        unique_pairs = len(np.unique(il_try * 10_000_000 + xl_try))
        if n_il > 1 and n_xl > 1 and unique_pairs >= 0.95 * n:
            return il_try, xl_try
    return None, None


def survey_geometry(path):
    """Map coordinates of the survey grid, as a 2x3 array M with
        easting  = M[0,0] + M[0,1] * inline + M[0,2] * crossline
        northing = M[1,0] + M[1,1] * inline + M[1,2] * crossline
    fitted to the CDP (or source) X/Y of a sample of about 3000 traces. None if
    the file has no usable coordinates. Saved next to the cube cache after the first time."""
    folder = cube_cache_dir(path)
    geo_file = folder / "geometry.npz"
    if geo_file.exists():
        d = np.load(geo_file)
        return d["fwd"] if bool(d["ok"]) else None

    import segyio

    fwd = None
    with segyio.open(str(path), ignore_geometry=True) as f:
        step = max(1, f.tracecount // 3000)  # reading every header is slow on big files
        il, xl = find_line_numbers(f, step)
        if il is not None:
            sc = np.asarray(f.attributes(71)[::step], dtype=np.float64)  # coordinate scalar
            scale = np.ones_like(sc)
            scale[sc < 0] = 1.0 / -sc[sc < 0]
            scale[sc > 0] = sc[sc > 0]
            A = np.column_stack([np.ones(len(il)), il, xl]).astype(np.float64)
            for x_byte, y_byte in ((181, 185), (73, 77)):  # CDP X/Y, then source X/Y
                X = np.asarray(f.attributes(x_byte)[::step], dtype=np.float64) * scale
                Y = np.asarray(f.attributes(y_byte)[::step], dtype=np.float64) * scale
                if np.ptp(X) == 0 or np.ptp(Y) == 0:
                    continue
                cx = np.linalg.lstsq(A, X, rcond=None)[0]
                cy = np.linalg.lstsq(A, Y, rcond=None)[0]
                M = np.vstack([cx, cy])
                bin_size = min(np.hypot(M[0, 1], M[1, 1]), np.hypot(M[0, 2], M[1, 2]))
                misfit = np.hypot(A @ cx - X, A @ cy - Y)
                if bin_size > 0 and abs(np.linalg.det(M[:, 1:])) > 0 and np.percentile(misfit, 99) < 0.5 * bin_size:
                    fwd = M
                    break
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(geo_file, ok=fwd is not None, fwd=fwd if fwd is not None else np.zeros((2, 3)))
    return fwd


def lines_to_world(M, il, xl):
    il, xl = np.asarray(il, dtype=np.float64), np.asarray(xl, dtype=np.float64)
    return M[0, 0] + M[0, 1] * il + M[0, 2] * xl, M[1, 0] + M[1, 1] * il + M[1, 2] * xl


def world_to_lines(M, x, y):
    """(inline, crossline) for map coordinates: the inverse of lines_to_world."""
    rhs = np.vstack([np.asarray(x, dtype=np.float64) - M[0, 0], np.asarray(y, dtype=np.float64) - M[1, 0]])
    il, xl = np.linalg.solve(M[:, 1:], rhs)
    return il, xl


def load_cube(path):
    """Open a 3D post-stack SEG-Y file as an (inline, crossline, time) array.
    The first time, the whole file is read once and saved as cache/…/cube.npy
    (about the same size as the SEG-Y). After that the cache is memory-mapped,
    so any inline, crossline or time slice comes back instantly."""
    folder = cube_cache_dir(path)
    meta_file, cube_file = folder / "meta.npz", folder / "cube.npy"
    if meta_file.exists() and cube_file.exists():
        meta = np.load(meta_file)
        return {
            "cube": np.load(cube_file, mmap_mode="r"),
            "ilines": meta["ilines"], "xlines": meta["xlines"], "times": meta["times"],
            "amp": float(meta["amp"]),
        }

    import segyio  # imported here so the app still opens if segyio isn't installed

    folder.mkdir(parents=True, exist_ok=True)
    tmp_file = folder / "cube.partial.npy"
    with segyio.open(str(path), ignore_geometry=True) as f:
        f.mmap()
        n = f.tracecount
        il, xl = find_line_numbers(f)
        if il is None:
            raise ValueError("couldn't find inline/crossline numbers in the trace headers")

        ilines, xlines = np.unique(il), np.unique(xl)
        il_idx, xl_idx = np.searchsorted(ilines, il), np.searchsorted(xlines, xl)
        times = np.asarray(f.samples, dtype=np.float32)

        cube = np.lib.format.open_memmap(
            tmp_file, mode="w+", dtype=np.float32, shape=(len(ilines), len(xlines), len(times)))
        cube[:] = np.nan  # positions with no trace stay empty
        for a in range(0, n, CHUNK_TRACES):
            b = min(a + CHUNK_TRACES, n)
            cube[il_idx[a:b], xl_idx[a:b], :] = f.trace.raw[a:b]
        cube.flush()
        amp = robust_amp(cube)
    del cube
    gc.collect()  # release the file so it can be renamed (needed on Windows)
    tmp_file.replace(cube_file)
    np.savez(meta_file, ilines=ilines, xlines=xlines, times=times, amp=amp)
    return load_cube(path)


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


def view_range(lo, hi, flip=False):
    """Axis range for a plot. You can zoom in freely but never out past the full view
    [lo, hi], and drag the view up to 30 % beyond the edges (a range locked exactly to
    the data can't be moved at all while it shows everything, so the move tool seemed
    broken). flip=True gives a range that runs downwards (time)."""
    pad = 0.3 * (hi - lo)
    limits = {"bounds": (lo - pad, hi + pad), "max_interval": hi - lo}
    return Range1d(hi, lo, **limits) if flip else Range1d(lo, hi, **limits)


def display_res_ms():
    """Time resolution chosen under Slice (2 or 4 ms)."""
    return ws.get("res_ms", 2)


def time_stride(times):
    """How many recorded samples make one displayed sample at the chosen
    resolution: 1 at 2 ms for 2 ms data, 2 at 4 ms (never below the recording)."""
    native = spacing(times)
    return max(1, int(round(display_res_ms() / native))) if native > 0 else 1


def block_mean(a, s, axis):
    """Average each run of s neighbouring samples along an axis (a ragged end is
    dropped). Averaging rather than skipping samples keeps it from aliasing."""
    if s <= 1:
        return a
    n = (a.shape[axis] // s) * s
    a = np.take(a, np.arange(n), axis=axis)
    shape = list(a.shape)
    shape[axis:axis + 1] = [n // s, s]
    return a.reshape(shape).mean(axis=axis + 1)


def get_slice(vol, view, value):
    """One 2D slice as a Bokeh-ready image (rows = y, columns = x), at the chosen
    time resolution. Inline / Crossline: x = the other line number, y = time.
    Time slice: x = crossline, y = inline (a map view)."""
    cube, il, xl, t = vol["cube"], vol["ilines"], vol["xlines"], vol["times"]
    k0, k1 = time_index_range(t)
    s = time_stride(t)
    if view == "Inline":
        i = nearest(il, value)
        img, xs, ys, pos = block_mean(np.asarray(cube[i, :, k0:k1]).T, s, 0), xl, t[k0:k1][::s], il[i]
    elif view == "Crossline":
        j = nearest(xl, value)
        img, xs, ys, pos = block_mean(np.asarray(cube[:, j, k0:k1]).T, s, 0), il, t[k0:k1][::s], xl[j]
    else:
        k = (nearest(t, value) // s) * s  # start of the block of samples shown at this time
        img, xs, ys, pos = np.asarray(cube[:, :, k:k + s]).mean(axis=2), xl, il, t[k]
    ys = ys[:img.shape[0]]  # the ragged end dropped by block_mean
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
        survey["vol"], survey["error"], survey["name"], survey["geo"] = None, None, "", None
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
            continue
        try:
            survey["geo"] = survey_geometry(path)
        except Exception:  # no map coordinates: shapes stay in inline/crossline numbers
            survey["geo"] = None

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
    vals = np.asarray(common if len(common) else arrays[0], dtype=np.float64)
    if view == "Time slice" and len(vals) > 1:  # only times on the chosen resolution's grid
        vals = vals[::time_stride(vals)]
    return vals


# ---------------------------------------------------------------------------
# Shapefiles: reading and writing lives in shapefiles.py
# ---------------------------------------------------------------------------
# attribute table written with every saved shape
SHAPE_FIELDS = [("ID", "N", 6), ("NAME", "C", 32), ("VIEW", "C", 12), ("X_AXIS", "C", 16), ("Y_AXIS", "C", 16)]


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

SIDEBAR_WIDTH = 236
SIDEBAR_GAP = 16  # the sidebar floats this far in from the window edges, as a rounded card

class NavMenu(pn.reactive.ReactiveHTML):
    """Sidebar menu: items with a data-page attribute are clickable and set `page`.
    `loaded` lights the dot beside Data once surveys are loaded."""
    page = param.String(default="overview")
    loaded = param.Boolean(default=False)

    _template = f"""<div id="nav" class="nav">
  <div class="nav-label">Workspace</div>
  <div class="nav-item" data-page="overview">{ICON_OVERVIEW}Overview</div>
  <div class="nav-item" data-page="viewer">{ICON_VIEWER}Seismic viewer</div>
  <div class="nav-item" data-page="analysis">{ICON_ANALYSIS}Analysis</div>
  <div class="nav-label">Field</div>
  <div class="nav-item" data-page="wells">{ICON_WELL}Wells</div>
  <div class="nav-label">Setup</div>
  <div class="nav-item" data-page="data">{ICON_DATA}Data sources<span id="dot" class="nav-dot"></span></div>
</div>"""
    _scripts = {
        "render": """
          nav.querySelectorAll('[data-page]').forEach(el => { el.onclick = () => { data.page = el.dataset.page; }; });
          self.mark();
          self.loaded();
        """,
        "page": "self.mark();",
        "loaded": "dot.classList.toggle('on', data.loaded); "
                  "dot.title = data.loaded ? 'Surveys loaded' : 'No surveys loaded yet';",
        "mark": "nav.querySelectorAll('[data-page]').forEach("
                "el => el.classList.toggle('active', el.dataset.page === data.page));",
    }


nav_menu = NavMenu(
    stylesheets=[BASE_CSS, NAV_CSS], sizing_mode="stretch_width", margin=(0, 16),
)

# Light / dark mode switch (on = light), at the bottom of the sidebar: the browser
# swaps the CSS variables instantly (jscallback), and Python restyles the plots
# (on_theme_change, further down). The knob shows a moon in dark mode, a sun in light.
theme_toggle = pn.widgets.Switch(name="", value=DEFAULT_THEME == "light",
                                 stylesheets=[THEME_SWITCH_CSS], width=58, height=30, margin=0, align="center")
# (also the oil-splatter easter egg: 10 flips within 3 s, see OIL_SPLATTER_JS in styles.py)
theme_toggle.jscallback(value="document.documentElement.classList.toggle('light', cb_obj.active);\n"
                              + OIL_SPLATTER_JS)

sidebar = pn.Column(
    html(f"""
<div class="sidebar" style="padding-bottom:0">
    <div class="brand">
    <div class="brand-mark">
        <img src="{LOGO_SRC}" alt="Drill Baby Drill" style="width:100%; height:100%; object-fit:contain;">
    </div>
    <div>
      <div class="brand-name">Drill Baby Drill</div>
      <div class="brand-sub">4D seismic workspace</div>
    </div>
  </div>
</div>
""", sizing_mode="stretch_width"),
    nav_menu,
    pn.layout.VSpacer(),
    # "Appearance" on the left, the switch on the right
    pn.Row(html('<div class="eyebrow">Appearance</div>', align="center"), pn.layout.HSpacer(), theme_toggle,
           sizing_mode="stretch_width", margin=(8, 30, 22, 26)),
    width=SIDEBAR_WIDTH,
    margin=0,
    # pinned to the left of the window as a rounded card, so it stays put while
    # the page scrolls (the main column is shifted right to make room)
    styles={"background": "var(--sidebar-grad)", "border": f"1px solid {BORDER}", "border-radius": "22px",
            "box-shadow": "var(--shadow)",
            "position": "fixed", "top": f"{SIDEBAR_GAP}px", "left": f"{SIDEBAR_GAP}px",
            "height": f"calc(100vh - {2 * SIDEBAR_GAP}px)",
            "overflow-y": "auto", "overflow-x": "hidden",  # scrolls up/down only if ever needed, never sideways
            "box-sizing": "border-box", "z-index": "50"},
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
PAGES = ["overview", "viewer", "analysis", "wells", "data"]
PAGE_SECTIONS = {"overview": "Workspace", "viewer": "Workspace", "analysis": "Workspace",
                 "wells": "Field", "data": "Setup"}
PAGE_TITLES = {"overview": "Overview", "viewer": "Seismic viewer", "analysis": "Analysis",
               "wells": "Wells", "data": "Data sources"}
PAGE_SUBTITLES = {
    "overview": "What's loaded, at a glance, and where to go next.",
    "viewer": "Compare the baseline and monitor surveys slice by slice, and see what changed.",
    "analysis": "Measure the change between the surveys across the whole volume and the slice on screen.",
    "wells": "Load wells from LAS files and see them over the seismic.",
    "data": "Choose the baseline and monitor SEG-Y files, then press Run.",
}


def title_html(page):
    return (f'<div class="crumbs">Reservoir Time Machine<span class="sep">/</span>{PAGE_SECTIONS[page]}</div>'
            f'<h1>{PAGE_TITLES[page]}</h1><div class="page-sub">{PAGE_SUBTITLES[page]}</div>')


# stretches and lets the subtitle wrap: a fixed-width title would set a minimum
# page width and give narrower windows a sideways scrollbar
title_block = html(title_html("overview"), sizing_mode="stretch_width", min_width=200)
# top right: whether surveys are loaded (filled in by refresh_data_pill)
data_pill = html('<span class="pill">No surveys loaded</span>', align="start", margin=(6, 0, 0, 16))

header = pn.Row(
    title_block,
    data_pill,
    sizing_mode="stretch_width",
    margin=(0, 0, 18, 0),
)

# ---------------------------------------------------------------------------
# Colour map switch + shared colourbar for the seismic data
# ---------------------------------------------------------------------------
def build_colour_controls():
    """One set of colour-map controls: the colour map switch, the colourbar with its
    range slider on top, and the card around them. Built twice (Workspace page and
    Wells page); the two copies are kept in step further down."""
    cmap = pn.widgets.RadioButtonGroup(
        options=list(SEISMIC_CMAPS),
        value=DEFAULT_CMAP,
        stylesheets=[TOGGLE_CSS],
        margin=(0, 24, 0, 0),
        align="start",
    )
    bar = html("", sizing_mode="stretch_width")    # the bar itself
    ticks = html("", sizing_mode="stretch_width")  # the numbers under it
    text = html("", sizing_mode="stretch_width")
    # The range slider sits on top of the colourbar: drag the two handles on the bar.
    slider = pn.widgets.RangeSlider(
        start=-AMP_MAX, end=AMP_MAX, value=(-AMP_MAX, AMP_MAX), step=AMP_MAX / 100,
        format="0[.]00", show_value=False,
        sizing_mode="stretch_width", stylesheets=[OVERLAY_SLIDER_CSS], margin=(-25, 14, 0, 14),
    )
    card = pn.Column(
        text,
        pn.Row(
            cmap,
            pn.Column(bar, slider, ticks, sizing_mode="stretch_width", margin=0),
            sizing_mode="stretch_width", margin=(12, 0, 0, 0),
        ),
        sizing_mode="stretch_width",
        styles=CARD_STYLES,
        margin=(8, 0),
    )
    return {"cmap": cmap, "bar": bar, "ticks": ticks, "text": text, "slider": slider, "card": card}


_workspace_colour, _wells_colour = build_colour_controls(), build_colour_controls()
COLOUR_CONTROLS = (_workspace_colour, _wells_colour)
cmap_select, colourbar, colourbar_ticks, range_text, range_slider, display_bar = (
    _workspace_colour[k] for k in ("cmap", "bar", "ticks", "text", "slider", "card"))
wells_display_bar = _wells_colour["card"]  # shown on the Wells page (wells.py puts it above the plot)
# the Wells copy mirrors the Workspace one, both ways
cmap_select.link(_wells_colour["cmap"], value="value", bidirectional=True)
range_slider.link(_wells_colour["slider"], value="value", bidirectional=True)


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


def display_range():
    """The (low, high) amplitudes the range slider currently shows."""
    lo, hi = range_slider.value
    lo, hi = max(lo, -AMP_MAX), min(hi, AMP_MAX)
    if hi <= lo:
        hi = lo + AMP_MAX / 100
    return lo, hi


_ticks_for = [None]  # the AMP_MAX the colourbar numbers were last written for


def update_colourbar(event=None):
    """The colourbar, its greyed-out ends and the 'Showing ...' line, on both pages.
    Cheap, so it follows the slider handles live while they are dragged (the plots
    themselves are only recoloured when the handle is let go: see apply_colours)."""
    lo, hi = display_range()
    bar = colourbar_html(cmap_select.value, lo, hi)
    text = ('<div class="head"><span class="label">Colour map</span>'
            f'<span class="meta">Showing {lo:.3g} to {hi:.3g} · drag the handles on the bar; '
            'values outside are greyed out</span></div>')
    ticks = ticks_html() if _ticks_for[0] != AMP_MAX else None  # the numbers only change with the data
    for c in COLOUR_CONTROLS:
        c["bar"].object = bar
        c["text"].object = text
        if ticks is not None:
            c["ticks"].object = ticks
    _ticks_for[0] = AMP_MAX


_FULL_PALETTES = {}  # colour map name -> its 256 colours (built once)


def apply_colours(event=None):
    """Push the colour map and display range to the colourbar and every plot.
    The palette keeps its full-scale positions; values outside the display
    range get OUT_OF_RANGE_COLOR."""
    update_colourbar()
    name = cmap_select.value
    lo, hi = display_range()
    if name not in _FULL_PALETTES:
        _FULL_PALETTES[name] = cmap_palette(name, 256)
    full = _FULL_PALETTES[name]
    i0 = int(round((lo + AMP_MAX) / (2 * AMP_MAX) * 255))
    i1 = int(round((hi + AMP_MAX) / (2 * AMP_MAX) * 255))
    palette = full[i0:i1 + 1] if i1 > i0 else [full[i0], full[i0]]
    # Amplitudes beyond the colourbar's ends (the strongest 1 %) keep the end
    # colours; only values cut off by the display range are greyed out.
    eps = AMP_MAX * 1e-6
    low_color = full[0] if lo <= -AMP_MAX + eps else OUT_OF_RANGE_COLOR
    high_color = full[-1] if hi >= AMP_MAX - eps else OUT_OF_RANGE_COLOR
    for mapper in colour_mappers + extra_mappers:
        mapper.update(palette=palette, low=lo, high=hi,
                      low_color=low_color, high_color=high_color, nan_color=NAN_COLOR)


def reset_range_slider():
    """Match the sliders to the current amplitude scale (after loading new data)."""
    sliders = [c["slider"] for c in COLOUR_CONTROLS]
    # widen the limits first, then set the value, then narrow them, so the linked
    # sliders never hold a value outside their own limits
    for s in sliders:
        s.param.update(start=min(s.start, -AMP_MAX), end=max(s.end, AMP_MAX))
    for s in sliders:
        s.param.update(value=(-AMP_MAX, AMP_MAX), step=AMP_MAX / 100)
    for s in sliders:
        s.param.update(start=-AMP_MAX, end=AMP_MAX)


colour_mappers = []  # filled in when the seismic plots are built
extra_mappers = []   # plots on other pages (Wells), kept apart so the workspace rebuild leaves them alone
cmap_select.param.watch(apply_colours, "value")          # (the Wells copy follows through the link)
range_slider.param.watch(update_colourbar, "value")      # live while a handle is dragged: just the bar
for _c in COLOUR_CONTROLS:                               # the plots are recoloured once, when a handle is let go
    _c["slider"].param.watch(apply_colours, "value_throttled")

# ---------------------------------------------------------------------------
# Shapes: polygons drawn on the seismic, or loaded from a shapefile
# ---------------------------------------------------------------------------


# Shapes come from a folder: every .shp in it is read, with its .shx / .dbf
# (other files there, such as .prj, .cpg or images, are ignored).
shape_folder = pn.widgets.TextInput(name="Shapefile folder", placeholder="e.g. shapes_sleipner or C:\\path\\to\\folder",
                                    width=300, stylesheets=[INPUT_CSS], margin=(0, 10, 0, 0))
shape_browse = pn.widgets.Button(name="Browse…", stylesheets=[BTN_BASE_CSS], width=100, align="end",
                                 margin=(0, 10, 0, 0))
shape_load = pn.widgets.Button(name="Load", stylesheets=[BTN_BASE_CSS, PRIMARY_BTN_CSS], width=90,
                               align="end", margin=(0, 10, 0, 0))
shape_clear = pn.widgets.Button(name="Clear", stylesheets=[BTN_BASE_CSS], width=90, align="end",
                                disabled=True, margin=(0, 10, 0, 0))
shape_status = html("", sizing_mode="stretch_width", margin=(8, 0, 0, 0))


SECTION_POINT_TOLERANCE = 5  # map points / wells within this many lines of a section are shown on it


def empty_shapes():
    """Polygons, one entry per view; they can be drawn and edited on the plots.
    "origin" is "file" for polygons loaded from the shapefile folder and "" for
    ones drawn in the app, so loading a folder again replaces only the former."""
    return {v: {"xs": [], "ys": [], "label": [], "origin": []} for v in VIEWS}


def padded(values, n, fill=""):
    values = list(values or [])
    return (values + [fill] * n)[:n]


def empty_features():
    """Points and lines from the shapefile folder, one entry per view (shown, not edited)."""
    return {v: {"points": {"x": [], "y": [], "label": []}, "lines": {"xs": [], "ys": [], "label": []}}
            for v in VIEWS}


def view_extent(view):
    """(x0, x1, y0, y1) the loaded surveys cover in this view's plot coordinates."""
    x_key, y_key = {"Inline": ("xlines", "times"), "Crossline": ("ilines", "times"),
                    "Time slice": ("xlines", "ilines")}[view]
    vols = [s_["vol"] for s_ in ws["surveys"] if s_["vol"] is not None]
    if not vols:
        return None
    xs = np.concatenate([v[x_key] for v in vols])
    ys = np.concatenate([v[y_key] for v in vols])
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def overlaps(xs, ys, ext):
    x0, x1, y0, y1 = ext
    return min(xs) <= x1 and max(xs) >= x0 and min(ys) <= y1 and max(ys) >= y0


def shape_axes(view):
    """What a saved shape's x and y coordinates are, for this view."""
    if view == "Time slice" and ws["geo"] is not None:
        return "Easting", "Northing"
    x_label, y_label, _ = VIEW_AXES[view]
    return x_label, y_label


def finished_shapes(view):
    """(xs, ys, label) of polygons with at least three points (one still being drawn may have fewer)."""
    d = ws["shapes"][view]
    labels = d.get("label") or [""] * len(d["xs"])
    return [(xs, ys, lab) for xs, ys, lab in zip(d["xs"], d["ys"], labels) if len(xs) >= 3]


def shape_count(view):
    f = ws["features"][view]
    return len(finished_shapes(view)) + len(f["points"]["x"]) + len(f["lines"]["xs"])


def shape_file_stem(view):
    return "shapes_" + view.lower().replace(" ", "_")


def mpl_colour(value, default="none"):
    """A matplotlib RGBA colour from a CSS colour string (name, #hex, or rgb() / rgba())."""
    from matplotlib.colors import to_rgba
    try:
        return to_rgba(value)
    except ValueError:
        m = re.match(r"\s*rgba?\(([^)]*)\)", str(value))
        if m:
            parts = [p.strip() for p in m.group(1).split(",")]
            if len(parts) >= 3:
                try:
                    r, g, b = (float(p) / 255 for p in parts[:3])
                    return (r, g, b, float(parts[3]) if len(parts) > 3 else 1.0)
                except ValueError:
                    pass
        return to_rgba(default)


def make_view_png(view):
    """The two survey plots as they are on screen now, as PNG bytes: the same slice,
    colour map and display range, zoom, shapes and labels, in the app's current
    theme. Drawn with matplotlib (like the PDF export), so no browser is needed."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    if not colour_mappers:
        raise ValueError("no seismic is loaded")
    theme = THEMES[ws["theme"]]
    card, card2 = mpl_colour(theme["card"]), mpl_colour(theme["card2"])
    edge, text, text2 = (mpl_colour(theme[k]) for k in ("border-strong", "text", "text2"))
    shape_c, feature_c = mpl_colour(SHAPE_COLOUR), mpl_colour(FEATURE_COLOUR)

    # same colours as the on-screen plots (the first mapper holds the current palette and range)
    mapper = colour_mappers[0]
    cmap = ListedColormap([mpl_colour(p) for p in mapper.palette])
    cmap.set_under(mpl_colour(mapper.low_color))
    cmap.set_over(mpl_colour(mapper.high_color))
    cmap.set_bad(mpl_colour(mapper.nan_color))
    norm = Normalize(vmin=mapper.low, vmax=mapper.high)

    pos = position_input.value
    x_label, y_label, y_down = VIEW_AXES[view]
    surveys = ws["surveys"]
    slices = [get_slice(s_["vol"], view, pos) if s_["vol"] is not None else None for s_ in surveys]
    overlay = compute_overlay(view, pos, ws.get("t_range") if view != "Time slice" else None)
    ranges = ws.get("ranges")  # the shared axes, so any zoom or pan on screen carries over

    fig = Figure(figsize=(14, 6.8), facecolor=card)
    image = None
    for k, (survey, sl) in enumerate(zip(surveys, slices)):
        left = 0.065 + k * 0.5
        sub = f'{survey["name"]} · {position_text(view, sl["pos"])}' if sl is not None else survey["name"]
        fig.text(left, 0.945, survey["title"], color=text, fontsize=14, weight="bold", va="center")
        fig.text(left + 0.40, 0.945, survey["role"], color=text2, fontsize=10, ha="right", va="center")
        fig.text(left, 0.90, sub, color=text2, fontsize=9, va="center")
        ax = fig.add_axes([left, 0.25, 0.40, 0.60], facecolor=card2)
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
        if sl is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "No data loaded", color=text2, fontsize=11, ha="center", va="center",
                    transform=ax.transAxes)
            continue

        x0, x1 = extent(sl["xs"])
        y0, y1 = extent(sl["ys"])
        im = ax.imshow(sl["img"], origin="lower", extent=(x0, x1, y0, y1), aspect="auto", cmap=cmap, norm=norm)
        image = im if image is None else image

        # polygons, then everything else drawn on the plots (same styling as the screen)
        for xs, ys, _ in finished_shapes(view):
            ax.add_patch(Polygon(np.column_stack([xs, ys]), closed=True, facecolor=(*shape_c[:3], 0.15),
                                 edgecolor=shape_c, linewidth=2))
        for key, width, alpha, dash in (("marks", 1.5, 0.85, "--"), ("bars", 4, 0.9, "-"), ("lines", 2.5, 1.0, "-")):
            for xs, ys in zip(overlay[key]["xs"], overlay[key]["ys"]):
                ax.plot(xs, ys, color=feature_c, linewidth=width, alpha=alpha, linestyle=dash,
                        solid_capstyle="butt")
        if overlay["points"]["x"]:
            ax.scatter(overlay["points"]["x"], overlay["points"]["y"], s=70, marker="^", color=feature_c,
                       edgecolors="white", linewidths=1.5, zorder=5)
        lab = overlay["labels"]
        for x, y, t in zip(lab["x"], lab["y"], lab["text"]):
            ax.annotate(t, (x, y), xytext=(6, -5), textcoords="offset points", va="top", ha="left",
                        fontsize=8, weight="bold", color="#0b1418", zorder=6,
                        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5})

        if ranges is not None:
            ax.set_xlim(ranges[0].start, ranges[0].end)
            ax.set_ylim(ranges[1].start, ranges[1].end)  # already flipped for time (downwards) axes
        else:
            ax.set_xlim(x0, x1)
            ax.set_ylim((y1, y0) if y_down else (y0, y1))
        ax.set_xlabel(x_label, color=text2, fontsize=9)
        ax.set_ylabel(y_label, color=text2, fontsize=9)
        ax.tick_params(colors=text2, labelsize=8, color=edge)

    if image is not None:
        cb = fig.colorbar(image, cax=fig.add_axes([0.065, 0.085, 0.90, 0.016]), orientation="horizontal")
        cb.set_label("Amplitude", color=text2, fontsize=9)
        cb.outline.set_edgecolor(edge)
        cb.ax.tick_params(colors=text2, labelsize=8, color=edge)
    credit = (ws.get("credit") or "").strip()
    for i, line in enumerate(textwrap.wrap(credit, 170)):
        fig.text(0.065, 0.035 - i * 0.022, line, fontsize=7.5, color=text2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=card)
    return buf.getvalue()


def make_shapes_zip():
    """The current view's shapes as zipped shapefiles: polygons in shapes_<view>.*,
    points in shapes_<view>_points.*, lines in shapes_<view>_lines.* (a shapefile
    holds one kind of shape only). A picture of the two survey plots as they are
    on screen goes in too, as shapes_<view>_view.png."""
    view = view_select.value
    x_name, y_name = shape_axes(view)
    to_world = view == "Time slice" and ws["geo"] is not None  # map view: x = crossline, y = inline

    def coords(xs, ys):
        return lines_to_world(ws["geo"], il=ys, xl=xs) if to_world else (xs, ys)

    feats = ws["features"][view]
    groups = {"polygon": [], "point": [], "line": []}  # kind -> [(geometry, label)]
    for xs, ys, lab in finished_shapes(view):
        groups["polygon"].append((closed_ring(*coords(xs, ys)), lab))
    for x, y, lab in zip(feats["points"]["x"], feats["points"]["y"], feats["points"]["label"]):
        wx, wy = coords(np.array([x]), np.array([y]))
        groups["point"].append(([(float(wx[0]), float(wy[0]))], lab))
    for xs, ys, lab in zip(feats["lines"]["xs"], feats["lines"]["ys"], feats["lines"]["label"]):
        wx, wy = coords(np.asarray(xs), np.asarray(ys))
        groups["line"].append((list(zip(np.asarray(wx, float).tolist(), np.asarray(wy, float).tolist())), lab))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for kind, suffix in (("polygon", ""), ("point", "_points"), ("line", "_lines")):
            items = groups[kind]
            if not items:
                continue
            records = [{"ID": i, "NAME": lab or f"{kind.title()} {i}", "VIEW": view, "X_AXIS": x_name,
                        "Y_AXIS": y_name} for i, (_, lab) in enumerate(items, 1)]
            files = write_shapefile(kind, [g for g, _ in items], records, SHAPE_FIELDS)
            for ext, data in files.items():
                z.writestr(shape_file_stem(view) + suffix + ext, data)
        try:
            z.writestr(shape_file_stem(view) + "_view.png", make_view_png(view))
        except Exception as exc:  # the shapefiles still download if the picture can't be made
            shape_status.object = ('<span class="status err">Shapes downloaded, but the view picture '
                                   f'couldn\'t be added ({escape(str(exc))})</span>')
    buf.seek(0)
    return buf


shape_download = pn.widgets.FileDownload(callback=make_shapes_zip, filename="shapes_inline.zip",
                                         label="Download", button_type="default", width=120,
                                         disabled=True, stylesheets=[BTN_BASE_CSS, DOWNLOAD_BTN_CSS], align="end",
                                         height=DOWNLOAD_BTN_HEIGHT, margin=(0, 10, 0, 0))


def refresh_shape_controls():
    view = view_select.value
    shape_download.filename = shape_file_stem(view) + ".zip"
    shape_download.disabled = shape_count(view) == 0
    shape_clear.disabled = not any(shape_count(v) or ws["shapes"][v]["xs"] for v in VIEWS)


def on_shapes_edited(view, attr, old, new):
    """A shape was drawn, moved, edited or deleted in the browser: keep a copy here."""
    xs = [np.asarray(x, dtype=np.float64).tolist() for x in new.get("xs", [])]
    ws["shapes"][view] = {"xs": xs, "ys": [np.asarray(y, dtype=np.float64).tolist() for y in new.get("ys", [])],
                          "label": [str(v or "") for v in padded(new.get("label"), len(xs))],
                          "origin": [str(v or "") for v in padded(new.get("origin"), len(xs))]}
    if view == view_select.value:
        refresh_shape_controls()
        refresh_overlay()


def shape_columns(d):
    """Plot data for a view's polygons."""
    n = len(d["xs"])
    return {"xs": [list(x) for x in d["xs"]], "ys": [list(y) for y in d["ys"]],
            "label": padded(d.get("label"), n), "origin": padded(d.get("origin"), n)}


def show_shapes():
    """Push the current view's shapes to the plots."""
    src = ws.get("shape_src")
    if src is not None:
        d = ws["shapes"][view_select.value]
        src.data = shape_columns(d)
    refresh_shape_controls()
    refresh_overlay()


def on_clear_shapes(event=None):
    ws["shapes"], ws["features"] = empty_shapes(), empty_features()
    show_shapes()
    shape_status.object = ""


# --- what is drawn besides the editable polygons -----------------------------
def empty_overlay():
    return {"points": {"x": [], "y": [], "label": []},       # markers (map view)
            "lines": {"xs": [], "ys": [], "label": []},      # solid lines (map view)
            "bars": {"xs": [], "ys": [], "label": []},       # section: where a map polygon covers the line
            "marks": {"xs": [], "ys": [], "label": []},      # section: dashed verticals (polygon edges, wells)
            "labels": {"x": [], "y": [], "text": []}}


def crossings(xs, ys, c):
    """Where a path (closed if it's a polygon ring) crosses the line y = c: the x values, sorted."""
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    hit = (y > c) != (y2 > c)
    return np.sort(x[hit] + (c - y[hit]) * (x2[hit] - x[hit]) / (y2[hit] - y[hit]))


def compute_overlay(view, pos, t_range):
    """Points and lines of this view, plus (on inline / crossline sections) the
    map shapes cut by the section. A map polygon becomes a bar along the top of
    the section over the stretch it covers (one row per polygon, named), with
    dashed lines where its edges cross; map points / wells within
    SECTION_POINT_TOLERANCE lines and map lines become dashed verticals. The
    seismic itself is left unshaded, since outlines often cover the whole survey."""
    out = empty_overlay()
    own = ws["features"][view]
    for key in ("points", "lines"):
        for k, v in own[key].items():
            out[key][k] += list(v)
    for x, y, lab in zip(own["points"]["x"], own["points"]["y"], own["points"]["label"]):
        if lab:
            out["labels"]["x"].append(x), out["labels"]["y"].append(y), out["labels"]["text"].append(lab)
    if view == "Time slice" or t_range is None:
        for xs, ys, lab in finished_shapes(view):  # name each labelled map polygon at its centre
            if lab:
                out["labels"]["x"].append(float(np.mean(xs)))
                out["labels"]["y"].append(float(np.mean(ys)))
                out["labels"]["text"].append(lab)
        return out

    # a section: map coordinates are (x = crossline, y = inline)
    t0, t1 = min(t_range), max(t_range)
    ext = view_extent(view)
    x_lo, x_hi = (ext[0], ext[1]) if ext else (-np.inf, np.inf)
    along_is_x = view == "Inline"  # Inline section runs along crosslines (map x); Crossline along inlines (map y)

    def row_y(row):
        return t0 + (0.025 + 0.045 * row) * (t1 - t0)

    def to_section(xs, ys):
        return (np.asarray(xs, float), np.asarray(ys, float)) if along_is_x else \
               (np.asarray(ys, float), np.asarray(xs, float))

    def label(x, y, text):
        if text:
            out["labels"]["x"].append(float(x)), out["labels"]["y"].append(float(y))
            out["labels"]["text"].append(text)

    def mark(at, lab):
        out["marks"]["xs"].append([at, at]), out["marks"]["ys"].append([t0, t1]), out["marks"]["label"].append(lab)

    row = 0
    for xs, ys, lab in finished_shapes("Time slice"):
        along, across = to_section(xs, ys)
        cut = crossings(along, across, pos)
        if len(cut) < 2:
            continue
        y = row_y(row)
        row += 1
        named = False
        for a, b in zip(cut[0::2], cut[1::2]):
            out["bars"]["xs"].append([a, b]), out["bars"]["ys"].append([y, y]), out["bars"]["label"].append(lab)
            for edge in (a, b):
                if x_lo < edge < x_hi:  # edges at the survey's own limits would just trace the frame
                    mark(float(edge), "")
            lo, hi = max(a, x_lo), min(b, x_hi)  # name it where its on-screen part starts
            if not named and lo < hi:
                label(lo, y, lab)
                named = True
    feats = ws["features"]["Time slice"]
    y_points = row_y(row)

    def mark_point(at, lab):
        mark(at, lab)
        label(at, y_points, lab)
    for x, y, lab in zip(feats["points"]["x"], feats["points"]["y"], feats["points"]["label"]):
        along, across = to_section([x], [y])
        if abs(across[0] - pos) <= SECTION_POINT_TOLERANCE:
            mark_point(float(along[0]), lab)
    for xs, ys, lab in zip(feats["lines"]["xs"], feats["lines"]["ys"], feats["lines"]["label"]):
        along, across = to_section(xs, ys)
        a, c = along[:-1], across[:-1]
        a2, c2 = along[1:], across[1:]  # open path: don't join the ends
        hit = (c > pos) != (c2 > pos)
        for at in a[hit] + (pos - c[hit]) * (a2[hit] - a[hit]) / (c2[hit] - c[hit]):
            mark_point(float(at), lab)
    return out


def refresh_overlay():
    srcs = ws.get("overlay_src")
    if not srcs:
        return
    view = view_select.value
    pos = position_input.value
    data = compute_overlay(view, pos, ws.get("t_range") if view != "Time slice" else None)
    for key, src in srcs.items():
        src.data = data[key]


def place_ring(xs, ys, view, view_from_file=False):
    """Where a loaded shape goes: (view, plot xs, plot ys), or None if it lies
    outside the surveys. The coordinates are tried, in this order, as:
      1. the view saved in the file (shapefiles exported by this app record it)
      2. survey line numbers for the map view, as (crossline, inline)
      3. survey line numbers the other way round, (inline, crossline)
      4. map coordinates (easting / northing), converted to line numbers
      5. the current view's own axes (e.g. crossline and time on an inline section)
    Map shapes (2-4) go on the Time slice view."""
    if view_from_file:
        ext = view_extent(view)
        if ext is not None and overlaps(xs, ys, ext):
            return view, list(xs), list(ys)
    lines = view_extent("Time slice")  # x = crossline, y = inline
    if lines is not None:
        if overlaps(xs, ys, lines):
            return "Time slice", list(xs), list(ys)
        if overlaps(ys, xs, lines):
            return "Time slice", list(ys), list(xs)
    if ws["geo"] is not None:
        il, xl = world_to_lines(ws["geo"], xs, ys)
        if lines is not None and overlaps(xl, il, lines):
            return "Time slice", xl.tolist(), il.tolist()
    ext = view_extent(view)
    if not view_from_file and ext is not None and overlaps(xs, ys, ext):
        return view, list(xs), list(ys)
    return None


def read_shape_folder(folder):
    """{name: {".shp": bytes, ".shx": bytes, ".dbf": bytes}} for every .shp in the folder."""
    groups = {}
    for shp in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        # "._name.shp" files are macOS zip leftovers (file metadata, not shapes): skip them
        if shp.is_file() and shp.suffix.lower() == ".shp" and not shp.name.startswith("._"):
            g = {".shp": shp.read_bytes()}
            for ext in (".shx", ".dbf"):
                for cand in (shp.with_suffix(ext), shp.with_suffix(ext.upper())):
                    if cand.is_file():
                        g[ext] = cand.read_bytes()
                        break
            groups[shp.stem] = g
    return groups


def add_shapes(stem, g, current):
    """Add one shapefile's shapes. Returns (kind, number added, number outside the survey)."""
    kind, shapes = read_shp(g[".shp"])
    records = read_dbf(g[".dbf"]) if ".dbf" in g else []
    n_added = n_out = 0
    for i, parts in enumerate(shapes):
        rec = records[i] if i < len(records) else {}
        view_from_file = rec.get("VIEW") in VIEWS
        view = rec["VIEW"] if view_from_file else current
        label = record_label(rec)
        for part in parts:
            xs, ys = part[:, 0].tolist(), part[:, 1].tolist()
            if kind == "polygon" and len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
                xs, ys = xs[:-1], ys[:-1]  # plots close the ring themselves
            if len(xs) < {"polygon": 3, "line": 2, "point": 1}[kind]:
                continue
            placed = place_ring(xs, ys, view, view_from_file)
            if placed is None:
                n_out += 1
                continue
            v, px, py = placed
            if kind == "polygon":
                d = ws["shapes"][v]
                d["label"], d["origin"] = padded(d.get("label"), len(d["xs"])), padded(d.get("origin"), len(d["xs"]))
                d["xs"].append(px), d["ys"].append(py), d["label"].append(label), d["origin"].append("file")
            elif kind == "line":
                d = ws["features"][v]["lines"]
                d["xs"].append(px), d["ys"].append(py), d["label"].append(label)
            else:
                d = ws["features"][v]["points"]
                d["x"].append(px[0]), d["y"].append(py[0]), d["label"].append(label)
            n_added += 1
    return kind, n_added, n_out


def remove_loaded_shapes():
    """Forget everything loaded from a folder before (shapes drawn in the app stay)."""
    for v in VIEWS:
        d = ws["shapes"][v]
        keep = [i for i, o in enumerate(padded(d.get("origin"), len(d["xs"]))) if o != "file"]
        labels = padded(d.get("label"), len(d["xs"]))
        ws["shapes"][v] = {"xs": [d["xs"][i] for i in keep], "ys": [d["ys"][i] for i in keep],
                           "label": [labels[i] for i in keep], "origin": [""] * len(keep)}
    ws["features"] = empty_features()


def on_load_shapes(event=None):
    def say(msg, ok=True):
        shape_status.object = f'<span class="status {"ok" if ok else "err"}">{msg}</span>'

    text = shape_folder.value.strip()
    if not text:
        return say("Type or browse to the folder that holds the shapefiles", ok=False)
    folder = resolve_path(text)
    if not folder.is_dir():
        return say(f"Folder not found: {escape(str(folder))}", ok=False)
    groups = read_shape_folder(folder)
    if not groups:
        return say(f"No .shp files in {escape(str(folder))}", ok=False)

    remove_loaded_shapes()
    current = view_select.value
    shown, outside, errors = [], [], []
    plural = {"point": "points", "line": "lines", "polygon": "polygons"}
    for stem, g in groups.items():
        try:
            kind, n_added, n_out = add_shapes(stem, g, current)
        except Exception as exc:  # one bad file doesn't stop the others
            errors.append(f"{escape(stem)} ({escape(str(exc))})")
            continue
        if n_added:
            shown.append(f"{escape(stem)} ({n_added} {kind if n_added == 1 else plural[kind]})")
        if n_out:
            outside.append(f"{escape(stem)} ({n_out})")
    parts = [f"Loaded {len(groups)} shapefile{'' if len(groups) == 1 else 's'}"]
    if shown:
        parts.append("shown: " + ", ".join(shown))
    if outside:
        parts.append("outside the seismic survey area, not shown: " + ", ".join(outside))
    if errors:
        parts.append("couldn't read: " + ", ".join(errors))
    if shown and current != "Time slice" and shape_count("Time slice"):
        parts.append("map shapes show on the Time slice view, and on inline / crossline sections where they "
                     "cross the line")
    say(" · ".join(parts), ok=bool(shown) and not errors)
    show_shapes()


# Browse opens the computer's folder dialog in a separate small Python process
# (like the seismic Browse buttons), so it can't freeze the app while it's open.
FOLDER_PICKER_SCRIPT = """
import sys
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2])
root.destroy()
sys.stdout.write(path or "")
"""


def pick_folder(title, start_dir):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", FOLDER_PICKER_SCRIPT, title, str(start_dir)],
                            capture_output=True, text=True, encoding="utf-8", env=env)
    if result.returncode != 0:
        lines = result.stderr.strip().splitlines()
        raise RuntimeError(lines[-1] if lines else "the folder dialog couldn't be opened")
    return result.stdout.strip()


async def on_browse_shapes(event):
    current = resolve_path(shape_folder.value) if shape_folder.value.strip() else APP_DIR
    start_dir = current if current.is_dir() else APP_DIR
    shape_browse.disabled = True
    try:
        path = await asyncio.get_running_loop().run_in_executor(None, pick_folder,
                                                                "Choose the shapefile folder", start_dir)
        if path:
            shape_folder.value = path
            on_load_shapes()
    except Exception as exc:
        shape_status.object = (f'<span class="status err">Couldn\'t open the folder dialog ({escape(str(exc))}). '
                               'You can still type or paste the folder path.</span>')
    finally:
        shape_browse.disabled = False


shape_load.on_click(on_load_shapes)
shape_browse.on_click(on_browse_shapes)
shape_clear.on_click(on_clear_shapes)

shapes_card = pn.Column(
    html('<div class="head"><span class="label">Shapes</span></div>', sizing_mode="stretch_width"),
    # wraps onto a second line on narrow windows instead of widening the page
    pn.FlexBox(shape_folder, shape_browse, shape_load, shape_clear, shape_download, flex_wrap="wrap",
               align_items="flex-end", sizing_mode="stretch_width", gap="10px 0px",
               margin=(6, 0, 0, 0)),
    shape_status,
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)


def new_overlay_sources():
    return {k: ColumnDataSource(v) for k, v in empty_overlay().items()}


def add_shape_layer(p, image_renderer, src, overlay, editable):
    """Draw the shared shapes on a plot; on the survey plots, add the draw / edit tools.
    Everything drawn is listed in ws["shape_renderers"] so the Show toggle can hide it."""
    for hover in p.select(type=HoverTool):
        hover.renderers = [image_renderer]  # seismic tooltip only on the seismic
    drawn = ws.setdefault("shape_renderers", [])
    drawn += [
        p.multi_line("xs", "ys", source=overlay["marks"], line_color=FEATURE_COLOUR, line_width=1.5,
                     line_dash="dashed", line_alpha=0.85),
        p.multi_line("xs", "ys", source=overlay["bars"], line_color=FEATURE_COLOUR, line_width=4, line_alpha=0.9),
        p.multi_line("xs", "ys", source=overlay["lines"], line_color=FEATURE_COLOUR, line_width=2.5),
        p.scatter("x", "y", source=overlay["points"], size=10, marker="triangle", fill_color=FEATURE_COLOUR,
                  line_color="white", line_width=1.5),
    ]
    # labels sit just below and right of their point / bar, so the top one isn't cut off
    labels = LabelSet(x="x", y="y", text="text", source=overlay["labels"], x_offset=6, y_offset=-5,
                      text_baseline="top",
                      text_font_size="10.5px", text_color="#0b1418", text_font_style="bold",
                      background_fill_color="white", background_fill_alpha=0.75)
    p.add_layout(labels)
    patches = p.patches("xs", "ys", source=src, fill_color=SHAPE_COLOUR, fill_alpha=0.15,
                        line_color=SHAPE_COLOUR, line_width=2,
                        selection_fill_alpha=0.3, selection_line_color=SHAPE_COLOUR,
                        nonselection_fill_alpha=0.1, nonselection_line_alpha=0.6,
                        nonselection_line_color=SHAPE_COLOUR, nonselection_fill_color=SHAPE_COLOUR)
    drawn += [labels, patches]
    if not editable:
        return
    vertices = p.scatter("x", "y", source=ColumnDataSource({"x": [], "y": []}), size=8,
                         fill_color="white", line_color=SHAPE_COLOUR, line_width=2)
    drawn.append(vertices)
    p.add_tools(
        PolyDrawTool(renderers=[patches], vertex_renderer=vertices, drag=True, empty_value="",
                     description="Draw shape (hold to start, click to add points, hold or Esc to finish)"),
        PolyEditTool(renderers=[patches], vertex_renderer=vertices, empty_value="",
                     description="Edit shape points (click a shape, then drag its points)"),
    )


# ---------------------------------------------------------------------------
# Slice controls: Inline / Crossline / Time slice, and which one
# ---------------------------------------------------------------------------
view_select = pn.widgets.RadioButtonGroup(
    options=list(VIEWS), value="Inline", stylesheets=[TOGGLE_CSS],
    margin=(0, 24, 0, 0), align="center",
)

position_input = pn.widgets.IntInput(
    start=0, end=1, step=1, value=0,
    width=170, stylesheets=[INPUT_CSS, NO_ARROWS_CSS], margin=(0, 0, 0, 0), align="center",
)
position_range = html("", align="center", margin=(0, 24, 0, 14))

# Time resolution of every plot: 2 ms shows each recorded sample (for 2 ms data);
# 4 ms averages pairs of samples. It changes what's drawn, not the loaded data.
RESOLUTIONS = {"2 ms": 2, "4 ms": 4}
resolution_select = pn.widgets.RadioButtonGroup(options=list(RESOLUTIONS), value="2 ms", stylesheets=[TOGGLE_CSS],
                                                margin=0, align="center")


def on_resolution_change(event):
    ws["res_ms"] = RESOLUTIONS[event.new]
    if not any(s_["vol"] is not None for s_ in ws["surveys"]):
        return
    detach_views()
    setup_slider(view_select.value, position_input.value)
    build_views()
    wells_ui.on_data_loaded()  # the Wells page plot uses the same resolution


resolution_select.param.watch(on_resolution_change, "value")

# What's drawn over the seismic: the polygons (Shapes card), the horizons, or both
SHOW_OPTIONS = ["Polygons", "Horizons", "Both"]
show_select = pn.widgets.RadioButtonGroup(options=SHOW_OPTIONS, value="Both", stylesheets=[TOGGLE_CSS],
                                          margin=0, align="center")


def apply_overlay_visibility(event=None):
    show = show_select.value
    for r in ws.get("shape_renderers", []):
        r.visible = show in ("Polygons", "Both")
    horizons_ui.set_visible(show in ("Horizons", "Both"))


show_select.param.watch(apply_overlay_visibility, "value")

# The viewer's toolbar: which slice on the left; on the right, two buttons that open
# the Layers panel (shapes, horizons, what's drawn) and the Display panel (colour
# map, display range, resolution). The panels themselves are built in the Layout section.
layers_btn = pn.widgets.Toggle(name="Layers", value=False, width=104, align="center", margin=(0, 8, 0, 0),
                               stylesheets=[BTN_BASE_CSS, PANEL_TOGGLE_CSS])
display_btn = pn.widgets.Toggle(name="Display", value=False, width=104, align="center", margin=0,
                                stylesheets=[BTN_BASE_CSS, PANEL_TOGGLE_CSS])


def labelled(text, widget, right=24):
    """A small label in front of a control, kept together when the row wraps."""
    return pn.Row(html(f'<span class="meta">{text}</span>', align="center", margin=(0, 10, 0, 0)), widget,
                  margin=(0, right, 0, 0))


slice_bar = pn.Column(
    # wraps onto a second line on narrow windows instead of widening the page
    pn.FlexBox(
        pn.Row(view_select, position_input, position_range, margin=0),
        pn.Row(layers_btn, display_btn, margin=0),
        flex_wrap="wrap", align_items="center", justify_content="space-between",
        sizing_mode="stretch_width", gap="10px 12px", margin=0),
    sizing_mode="stretch_width",
    styles=dict(CARD_STYLES, padding="12px 14px 12px 16px"),
    margin=(0, 0, 8, 0),
)

# ---------------------------------------------------------------------------
# Seismic surveys side by side (placeholders)
# ---------------------------------------------------------------------------


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


def fullscreen_tool(plot, wheel):
    callback = CustomJS(args={"plot": plot, "wheel": wheel, "enter_icon": ICON_FS_ENTER, "exit_icon": ICON_FS_EXIT},
                        code=FULLSCREEN_JS)
    tool = CustomAction(description="Fullscreen (Esc to exit)", icon=ICON_FS_ENTER, callback=callback)
    callback.args["tool"] = tool  # so the button can swap its own icon
    return tool


def style_figure(p):
    """Colour a Bokeh figure for the current theme (called again when the theme changes)."""
    c = THEMES[ws["theme"]]
    for tool in p.toolbar.tools:  # hide the speech-bubble buttons; hovering still shows the tooltips
        if isinstance(tool, HoverTool):
            tool.visible = False
    # Toolbar icons are interface, not data: leave them out of browser print / Save-as-PDF.
    p.toolbar.stylesheets = [InlineStyleSheet(css="@media print { :host { display: none !important; } }")]
    p.background_fill_color = c["card2"]
    p.border_fill_color = c["card"]
    p.outline_line_color = c["border-strong"]
    p.grid.visible = False
    p.toolbar.logo = None
    p.title.text_color = c["text"]
    p.title.text_font = PLOT_FONT
    p.title.text_font_style = "normal"
    p.title.text_font_size = "12px"
    for axis in (p.xaxis, p.yaxis):
        axis.axis_label_text_color = c["text2"]
        axis.axis_label_text_font = axis.major_label_text_font = PLOT_FONT
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

# Full-screen loading overlay (loading.py). The browser shows it the instant Run is
# clicked; on_run_click hides it in a `finally` once the Run has really finished.
loading_overlay = LoadingOverlay(stylesheets=[LOADING_CSS], sizing_mode="stretch_width", margin=0)
load_btn.js_on_click(code="if (window.__rtmLoading) window.__rtmLoading.show();")
# Status line beside Run: empty until there's something to say (a file couldn't
# be loaded, a file was chosen with Browse, "Loaded"). Other code writes to it,
# so keep this line even if you don't want a message showing.
load_status = html("", sizing_mode="stretch_width", align="center")
browse_base = pn.widgets.Button(name="Browse…", stylesheets=[BTN_BASE_CSS], width=96, align="end", margin=(0, 10, 9, 10))
browse_mon = pn.widgets.Button(name="Browse…", stylesheets=[BTN_BASE_CSS], width=96, align="end", margin=(0, 10, 9, 10))
browse_base.on_click(functools.partial(on_browse, baseline_path, "Choose the baseline SEG-Y file"))
browse_mon.on_click(functools.partial(on_browse, monitor_path, "Choose the monitor SEG-Y file"))

sources_card = pn.Column(
    html('<div class="head"><span class="label">Surveys</span>'
         '<span class="meta">SEG-Y files or cached folders</span></div>',
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
      "shapes": empty_shapes(), "features": empty_features(), "shape_src": None, "overlay_src": None,
      "t_range": None, "geo": None,
      "page": "overview", "wells": [], "wfigs": [], "res_ms": 2, "shape_renderers": [], "horizons": []}

# Horizons: their card, and the horizon lines on the survey plots (horizons.py)
horizons_ui = horizons.HorizonLayer(SimpleNamespace(
    ws=ws, html=html, pick_file=pick_file, resolve_path=resolve_path, APP_DIR=APP_DIR,
    lines_to_world=lines_to_world, world_to_lines=world_to_lines,
    current=lambda: (view_select.value, position_input.value),
))


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
    ws["ranges"] = None
    if loaded:
        x0 = min(extent(sl["xs"])[0] for sl in loaded)
        x1 = max(extent(sl["xs"])[1] for sl in loaded)
        y0 = min(extent(sl["ys"])[0] for sl in loaded)
        y1 = max(extent(sl["ys"])[1] for sl in loaded)
        shared_x = view_range(x0, x1)
        shared_y = view_range(y0, y1, flip=y_down)
        ws["t_range"] = (y0, y1)
        ws["ranges"] = (shared_x, shared_y)  # kept so the download picture matches the current zoom

    # one shape source shared by every plot: draw on one survey, it shows on all
    shapes = ws["shapes"][view]
    shape_src = ColumnDataSource(shape_columns(shapes))
    shape_src.on_change("data", functools.partial(on_shapes_edited, view))
    ws["shape_src"] = shape_src
    overlay = new_overlay_sources()  # uploaded points / lines, and map shapes cut by this section
    ws["overlay_src"] = overlay
    ws["shape_renderers"] = []
    horizons_ui.new_build()

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
            img = p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=src, color_mapper=mapper)
            add_shape_layer(p, img, shape_src, overlay, editable=True)
            horizons_ui.attach(p, editable=True, view=view)
            body =[pn.pane.Bokeh(p, sizing_mode="stretch_width", margin=(10, 0, 0, 0)), credit_html(credit)]
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
    m_head.object = ('<div class="head"><h2 class="title">Morph</h2></div>')
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
            img = p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=src, color_mapper=mapper)
            add_shape_layer(p, img, shape_src, overlay, editable=False)
            horizons_ui.attach(p, editable=False, view=view)
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
    refresh_shape_controls()
    refresh_overlay()
    horizons_ui.refresh()
    apply_overlay_visibility()
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
    refresh_overlay()  # map shapes cut by the new section
    horizons_ui.refresh()
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
    ws["cards"], ws["morph"], ws["shape_src"], ws["overlay_src"], ws["ranges"] = [], None, None, None, None
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


def stat_card(label, value, detail, unit="", highlight=False):
    return (f'<div class="stat{" hl" if highlight else ""}">'
            f'<div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}{f"<small>{unit}</small>" if unit else ""}</div>'
            f'<div class="stat-detail">{detail}</div></div>')


def summary_cards_html():
    """Summary cards for the loaded surveys (the first one highlighted)."""
    vols = [s_ for s_ in ws["surveys"] if s_["vol"] is not None]
    if not vols:
        return ""
    il = np.unique(np.concatenate([s_["vol"]["ilines"] for s_ in vols]))
    xl = np.unique(np.concatenate([s_["vol"]["xlines"] for s_ in vols]))
    t = np.unique(np.concatenate([s_["vol"]["times"] for s_ in vols]))
    names = " + ".join(escape(s_["title"]) for s_ in vols)
    dt = spacing(t)
    return '<div class="stats">' + "".join([
        stat_card("Surveys loaded", len(vols), names, highlight=True),
        stat_card("Inlines", f"{len(il):,}", f'<span class="chip">{il[0]:g}</span> to '
                  f'<span class="chip">{il[-1]:g}</span>'),
        stat_card("Crosslines", f"{len(xl):,}", f'<span class="chip">{xl[0]:g}</span> to '
                  f'<span class="chip">{xl[-1]:g}</span>'),
        stat_card("Record length", f"{t[-1] - t[0]:,.0f}", f"{len(t):,} samples every {dt:g} ms", unit="ms"),
    ]) + "</div>"


stats_pane = html("", sizing_mode="stretch_width", margin=(0, 0, 8, 0))


def finish_rebuild(surveys, event=None):
    """Second half of loading: show the loaded surveys and (re)build every view."""
    ws["surveys"] = surveys
    ws["geo"] = next((s_["geo"] for s_ in ws["surveys"] if s_["geo"] is not None), None)
    if all(s_["vol"] is None for s_ in ws["surveys"]):
        # nothing loaded: keep the rest of the page hidden and say why
        problems = "; ".join(f'{s_["role"].lower()}: {escape(s_["error"] or "no file chosen")}'
                             for s_ in ws["surveys"])
        load_status.object = f'<span class="status err">Couldn\'t load the files ({problems})</span>'
        wells_ui.on_data_loaded()
        show_page(ws["page"])  # the pages that need data show their empty state again
        return
    load_status.object = '<span class="status ok">Loaded</span>'
    reset_range_slider()
    first_value = INLINE if (view_select.value == "Inline" and INLINE is not None and event is None) else None
    setup_slider(view_select.value, first_value)
    build_views()
    stats_pane.object = summary_cards_html()
    wells_ui.on_data_loaded()
    errors = [s_ for s_ in ws["surveys"] if s_["error"]]
    if errors:
        load_status.object = f'<span class="status err">{len(errors)} file(s) couldn\'t be loaded — see the card for details</span>'
    # loaded: go to the Overview (unless there were errors to read on this page)
    if errors or ws["page"] not in ("data", "overview"):
        show_page(ws["page"])
    elif nav_menu.page != "overview":
        nav_menu.page = "overview"  # (its watcher calls show_page)
    else:
        show_page("overview")


def rebuild(event=None):
    """(Re)load the chosen files, then rebuild every view (all in one go)."""
    set_label(load_btn, "Running…")
    load_btn.disabled = True
    try:
        cfg = current_config()
        reset_analysis()
        detach_views()
        finish_rebuild(load_surveys(cfg), event)
    finally:
        set_label(load_btn, "Run")
        load_btn.disabled = False
        loading_overlay.finished()


async def on_run_click(event=None):
    """The Run button: the files are read in a worker thread (so the page keeps
    animating, even through a big first load), then the views are built.
    The full-screen loading overlay is shown by the browser the moment Run is
    clicked, and hidden in `finally` below once everything has finished, whether
    it worked or failed, so it can never stay up."""
    set_label(load_btn, "Running…")
    load_btn.disabled = True
    doc = pn.state.curdoc
    surveys, error = None, None
    try:
        cfg = current_config()
        reset_analysis()
        detach_views()
        await asyncio.sleep(0.05)  # lets the browser draw the overlay before the heavy part starts
        surveys = await asyncio.get_running_loop().run_in_executor(None, load_surveys, cfg)
    except Exception as exc:
        error = exc

    def finish():
        try:
            if error is not None:
                raise error
            finish_rebuild(surveys, event)
        except Exception as exc:  # show the problem rather than failing silently
            import traceback
            traceback.print_exc()
            load_status.object = f'<span class="status err">Run failed: {escape(str(exc))}</span>'
        finally:
            set_label(load_btn, "Run")
            load_btn.disabled = False
            loading_overlay.finished()  # hides the overlay and gives the page back

    # the plots may only be changed from the page's own update loop, so finish there
    if doc is not None:
        doc.add_next_tick_callback(finish)
    else:  # no browser session (e.g. a test script)
        finish()


# ---------------------------------------------------------------------------
# Analysis: measure the baseline-vs-monitor differences and write a summary
# ---------------------------------------------------------------------------
# Everything here is rule-based. The measurements (analyse_volume /
# analyse_slice) are plain numbers; write_summary() turns them into English and
# Malay text. To use an AI model later, replace write_summary() with a call
# that sends the metrics dict to the model and returns the same {"en": [...],
# "ms": [...]} structure. Nothing else needs to change.

ANALYSIS_TRACES = 15_000  # traces sampled across the shared area for the whole-volume analysis


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
# Summary text (English + Malay): AI sections + one sentence of other findings
# ---------------------------------------------------------------------------
RATING_MS = {"outstanding": "luar biasa baik", "excellent": "cemerlang", "very good": "sangat baik",
             "reasonable": "sederhana", "poor": "lemah"}


def fmt(x, digits=0):
    return f"{x:,.{digits}f}"


# The three AI sections, in display order: (key, English heading, Malay heading).
# Their text comes from the Gemma analysis later; until then they show placeholders.
AI_SECTIONS = [
    ("polygons", "3D polygons", "Poligon 3D"),
    ("opportunity", "Opportunity score", "Skor peluang"),
    ("volume", "Hydrocarbon volume", "Isipadu hidrokarbon"),
]
AI_PLACEHOLDER = {"en": "To be provided by the AI analysis.", "ms": "Akan disediakan oleh analisis AI."}


def other_findings(vol):
    """One sentence (English, Malay) summing up what the measurements found."""
    if vol is None:
        return None, None
    nrms, rating = vol["nrms_shallow"], vol["rating"]
    rep_en = f"repeatability is {rating} (NRMS {nrms:.0f}%)" if rating else ""
    rep_ms = f"kebolehulangan {RATING_MS[rating]} (NRMS {nrms:.0f}%)" if rating else ""

    if not (vol["significant"] and "box" in vol):
        en = "No significant amplitude change was found above the noise level" + (f"; {rep_en}." if rep_en else ".")
        ms = ("Tiada perubahan amplitud ketara ditemui melebihi aras hingar"
              + (f"; {rep_ms}." if rep_ms else "."))
        return en, ms

    b0, b1, c0, c1 = vol["box"]
    top = vol["t_change"][1]
    en = (f"{vol['changed_pct']:.1f}% of the shared area changed, mainly between inlines {fmt(b0)}–{fmt(b1)} "
          f"and crosslines {fmt(c0)}–{fmt(c1)} from about {fmt(top)} ms")
    ms = (f"{vol['changed_pct']:.1f}% daripada kawasan yang dikongsi berubah, terutamanya antara inline "
          f"{fmt(b0)}–{fmt(b1)} dan crossline {fmt(c0)}–{fmt(c1)} bermula kira-kira {fmt(top)} ms")

    ratio = vol.get("amp_ratio", np.nan)
    if np.isfinite(ratio):
        pct = abs(ratio - 1) * 100
        if ratio >= 1.1:
            en += f", with amplitudes about {pct:.0f}% stronger in the monitor survey"
            ms += f", dengan amplitud kira-kira {pct:.0f}% lebih kuat dalam survei pemantauan"
        elif ratio <= 0.9:
            en += f", with amplitudes about {pct:.0f}% weaker in the monitor survey"
            ms += f", dengan amplitud kira-kira {pct:.0f}% lebih lemah dalam survei pemantauan"
        else:
            en += ", with mixed amplitude changes"
            ms += ", dengan perubahan amplitud yang bercampur"

    shift = vol.get("time_shift_ms")
    if shift is not None and abs(shift) >= 1.0:
        if shift > 0:
            en += f", and reflections below the zone arrive {shift:.1f} ms later (push-down)"
            ms += f" serta anjakan masa {shift:.1f} ms lebih lewat (push-down) di bawah zon"
        else:
            en += f", and reflections below the zone arrive {-shift:.1f} ms earlier (pull-up)"
            ms += f" serta anjakan masa {-shift:.1f} ms lebih awal (pull-up) di bawah zon"
    if rep_en:
        en += f"; {rep_en}"
        ms += f"; {rep_ms}"
    return en + ".", ms + "."


def write_summary(vol, sl, base_label, mon_label, ai=None):
    """Summary in English and Malay: the three AI sections (placeholders until
    Gemma fills them in through `ai`, e.g. {"polygons": {"en": ..., "ms": ...}}),
    plus one sentence of other findings from the measurements."""
    ai = ai or {}
    other_en, other_ms = other_findings(vol)
    out = {}
    for lang, other, note in [
        ("en", other_en, f"Baseline: {base_label}; Monitor: {mon_label}"),
        ("ms", other_ms, f"Survei asas: {base_label}; Survei pemantauan: {mon_label}"),
    ]:
        sections = []
        for key, h_en, h_ms in AI_SECTIONS:
            text = (ai.get(key) or {}).get(lang)
            sections.append((h_en if lang == "en" else h_ms, text))  # text None -> placeholder
        out[lang] = {"sections": sections, "other": other, "note": note}
    return out


# ---------------------------------------------------------------------------
# Analysis card
# ---------------------------------------------------------------------------
MAP_PALETTE = palette_from_stops([(0.0, "#440154"), (0.25, "#3b528b"), (0.5, "#21918c"),
                                  (0.75, "#5ec962"), (1.0, "#fde725")])  # viridis

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
    p = analysis_figure(view_range(x0, x1), view_range(y0, y1), "Crossline", "Inline", "RMS change")
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
    p = analysis_figure(view_range(x0, x1), view_range(y0, y1, flip=y_down), x_label, y_label, "Change")
    p.image(image=[diff], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
    add_colorbar(p, mapper, f"Amplitude change: − weaker in {mon_label} · stronger in {mon_label} +")
    style_figure(p)
    return plot_with_caption(p, f"Current slice · difference ({mon_label} − {base_label}) · "
                                f"{position_text(sl['view'], sl['pos'])}")


def summary_html(summary):
    def column(lang, title, h_other):
        s_ = summary[lang]
        parts = []
        for heading, text in s_["sections"]:
            if text:
                parts.append(f"<h4>{escape(heading)}</h4><p>{escape(text)}</p>")
            else:
                parts.append(f'<h4>{escape(heading)} <span class="ai-badge">Placeholder</span></h4>'
                             f'<p class="ai-placeholder">{AI_PLACEHOLDER[lang]}</p>')
        if s_["other"]:
            parts.append(f"<h4>{h_other}</h4><p>{escape(s_['other'])}</p>")
        note = f'<p class="sum-note">{escape(s_["note"])}</p>' if s_["note"] else ""  # skip an empty note
        return f'<div class="sum-col"><div class="sum-lang">{title}</div>{"".join(parts)}{note}</div>'
    return html('<div class="summary">'
                + column("en", "English", "Other findings")
                + column("ms", "Bahasa Melayu", "Penemuan lain")
                + "</div>", sizing_mode="stretch_width")


def render_analysis():
    """(Re)draw the analysis card. The whole-volume part is kept from the last
    run; the slice part is recomputed for whatever slice is on screen now."""
    an = ws["analysis"]
    if not an:
        return
    sl = current_slice_analysis()
    an["slice"] = sl
    base_label, mon_label = an["labels"]
    an["summary"] = write_summary(an["volume"], sl, base_label, mon_label)
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


def on_run(event=None):
    base, mon = ws["surveys"]
    if base["vol"] is None or mon["vol"] is None:
        reset_analysis('<span class="status err">Load both surveys first.</span>')
        return
    set_label(run_btn, "Analysing…")
    run_btn.disabled = True
    try:
        vol = analyse_volume(base["vol"], mon["vol"])
        ws["analysis"] = {"volume": vol, "labels": (base["title"], mon["title"]),
                          "files": (base["name"], mon["name"]), "credit": ws["credit"]}
        render_analysis()
        pdf_btn.filename = pdf_filename()
        pdf_btn.disabled = False
        step = vol["sample_step"]
        analysis_status.object = ('<span class="status ok">Done</span><span class="status"> · whole volume sampled '
                                  + ("on every line" if step == 1 else f"every {step} lines")
                                  + ' · the slice part follows the slice chosen in the Seismic viewer</span>')
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
    from matplotlib.figure import Figure

    an = ws["analysis"]
    vol, sl, summary = an["volume"], an.get("slice"), an["summary"]
    base_label, mon_label = an["labels"]
    credit = an.get("credit", "").strip()
    a4 = (8.27, 11.69)
    ink, grey = "#13212a", "#5b6b73"
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")

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
        # page 1: key numbers
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
        # (the change map and slice difference plots are shown in the app only, not in the PDF)
        footer(fig, 1)
        pdf.savefig(fig)

        # pages 2 and 3: the summaries
        for page, (lang, title, h_other) in enumerate(
                [("en", "Summary (English)", "Other findings"),
                 ("ms", "Ringkasan (Bahasa Melayu)", "Penemuan lain")], start=2):
            fig = Figure(figsize=a4)
            s = summary[lang]
            sections = [(heading, [text or AI_PLACEHOLDER[lang]]) for heading, text in s["sections"]]
            if s["other"]:
                sections.append((h_other, [s["other"]]))
            text_page(fig, title, sections, s["note"])
            footer(fig, page)
            pdf.savefig(fig)
    buf.seek(0)
    return buf


def pdf_filename():
    an = ws.get("analysis") or {}
    labels = an.get("labels", ("baseline", "monitor"))
    safe = "".join(ch if ch.isalnum() else "_" for ch in f"{labels[0]}_vs_{labels[1]}").strip("_")
    return f"4D_analysis_{safe}.pdf"


pdf_btn = pn.widgets.FileDownload(callback=make_pdf, filename="4D_analysis.pdf", label="Export PDF",
                                  disabled=True, button_type="default", width=120, height=DOWNLOAD_BTN_HEIGHT,
                                  stylesheets=[BTN_BASE_CSS, DOWNLOAD_BTN_CSS])
run_btn.on_click(on_run)

analysis_card = pn.Column(
    html('<div class="head"><h2 class="title">Baseline vs monitor</h2>'
         '<span class="meta">automatic, rule-based</span></div>',
         sizing_mode="stretch_width"),
    pn.Row(run_btn, pdf_btn, analysis_status, sizing_mode="stretch_width", margin=(8, 0, 0, 0)),
    analysis_body,
    sizing_mode="stretch_width",
    styles=CARD_STYLES,
    margin=(8, 0),
)


load_btn.on_click(on_run_click)


def on_theme_change(event):
    ws["theme"] = "light" if event.new else "dark"
    for p in ws["figs"] + ws["afigs"] + ws["wfigs"]:
        style_figure(p)


theme_toggle.param.watch(on_theme_change, "value")
view_select.param.watch(on_view_change, "value")
position_input.param.watch(update_slice, "value")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Pages (one sidebar item each): Overview, Seismic viewer, Analysis, Wells, Data
# sources. The pages that need loaded surveys show an empty state until Run has
# loaded some, with a button that goes to Data sources.

def go_to(page):
    """Switch page from Python (as clicking the sidebar would)."""
    nav_menu.page = page


def action_button(label, page, primary=False, width=None):
    btn = pn.widgets.Button(name=label, width=width or (10 * len(label) + 48), margin=(0, 10, 0, 0),
                            stylesheets=[BTN_BASE_CSS] + ([PRIMARY_BTN_CSS] if primary else []))
    btn.on_click(lambda event: go_to(page))
    return btn


def empty_state(title, text, icon=ICON_UPLOAD):
    """The card a page shows while no surveys are loaded."""
    return pn.Column(
        html(f'<div class="empty"><div class="empty-icon">{icon}</div><h3>{escape(title)}</h3>'
             f'<p>{escape(text)}</p></div>', sizing_mode="stretch_width"),
        pn.Row(action_button("Go to Data sources", "data", primary=True, width=190),
               align="center", margin=(0, 0, 40, 0)),
        sizing_mode="stretch_width", styles=CARD_STYLES, margin=0,
    )


# --- Viewer: the toolbar's two panels --------------------------------------
# Display: resolution and the colour map / display range (the colour card is
# shown inside the panel, so it loses its own card frame here)
display_bar.styles = {}
display_bar.margin = (12, 0, 0, 0)
display_panel = pn.Column(
    pn.FlexBox(
        html('<span class="label">Display</span>', align="center", margin=(0, 24, 0, 0)),
        labelled("Resolution", resolution_select, right=0),
        flex_wrap="wrap", align_items="center", justify_content="space-between",
        sizing_mode="stretch_width", gap="10px 0px", margin=0),
    display_bar,
    sizing_mode="stretch_width", styles=CARD_STYLES, margin=(0, 0, 8, 0), visible=False,
)

# Layers: what's drawn over the seismic (Show) and where it comes from (Shapes / Horizons)
for _card in (shapes_card, horizons_ui.card):
    _card.styles = {}
    _card.margin = (6, 0, 0, 0)
    _card[0].visible = False  # its own heading: the Shapes / Horizons switch names it here
layer_tabs = pn.widgets.RadioButtonGroup(options=["Shapes", "Horizons"], value="Shapes",
                                         stylesheets=[TOGGLE_CSS], margin=(0, 24, 0, 0), align="center")
horizons_ui.card.visible = False
layer_tabs.param.watch(lambda e: (setattr(shapes_card, "visible", e.new == "Shapes"),
                                  setattr(horizons_ui.card, "visible", e.new == "Horizons")), "value")
layers_panel = pn.Column(
    pn.FlexBox(
        pn.Row(html('<span class="label">Layers</span>', align="center", margin=(0, 18, 0, 0)), layer_tabs,
               margin=0),
        labelled("Show on the seismic", show_select, right=0),
        flex_wrap="wrap", align_items="center", justify_content="space-between",
        sizing_mode="stretch_width", gap="10px 0px", margin=0),
    shapes_card,
    horizons_ui.card,
    sizing_mode="stretch_width", styles=CARD_STYLES, margin=(0, 0, 8, 0), visible=False,
)
layers_btn.param.watch(lambda e: setattr(layers_panel, "visible", e.new), "value")
display_btn.param.watch(lambda e: setattr(display_panel, "visible", e.new), "value")

viewer_body = pn.Column(slice_bar, layers_panel, display_panel, content, sizing_mode="stretch_width", margin=0)
viewer_page = pn.Column(
    empty_state("No surveys loaded", "Choose the baseline and monitor SEG-Y files under Data sources and "
                "press Run. The two surveys and the morph between them appear here.", ICON_VIEWER),
    viewer_body, sizing_mode="stretch_width", margin=0, visible=False)

# --- Analysis ----------------------------------------------------------------
analysis_card.margin = 0
analysis_page = pn.Column(
    empty_state("Nothing to analyse yet", "Load both surveys under Data sources first. The analysis then "
                "measures the change across the whole volume and the slice on screen.", ICON_ANALYSIS),
    analysis_card, sizing_mode="stretch_width", margin=0, visible=False)

# --- Data sources --------------------------------------------------------------
sources_card.margin = 0
data_page = pn.Column(sources_card, sizing_mode="stretch_width", margin=0, visible=False)

# --- Overview ------------------------------------------------------------------
overview_table = html("", sizing_mode="stretch_width")
overview_analysis = pn.Column(sizing_mode="stretch_width", margin=0)
overview_body = pn.Column(
    stats_pane,
    pn.Column(
        pn.FlexBox(
            html('<div><div class="label">Workspace status</div>'
                 '<div class="meta" style="margin-top:3px">Surveys, interpretation layers, wells and analysis'
                 '</div></div>', margin=(0, 24, 0, 0)),
            pn.Row(action_button("Open viewer", "viewer", primary=True),
                   action_button("Analysis", "analysis"), action_button("Wells", "wells"), margin=0),
            flex_wrap="wrap", align_items="center", justify_content="space-between",
            sizing_mode="stretch_width", gap="12px 0px", margin=(0, 0, 14, 0)),
        overview_table,
        sizing_mode="stretch_width", styles=CARD_STYLES, margin=(8, 0),
    ),
    overview_analysis,
    sizing_mode="stretch_width", margin=0,
)
overview_page = pn.Column(
    empty_state("Welcome to Reservoir Time Machine", "Start by choosing a baseline and a monitor survey "
                "under Data sources and pressing Run. This page then sums up what's loaded.", ICON_OVERVIEW),
    overview_body, sizing_mode="stretch_width", margin=0, visible=True)


def _row(icon, name, detail, pill, pill_class, info=""):
    return (f'<tr><td class="name"><div class="ov-name"><span class="ov-icon">{icon}</span>'
            f'<div>{name}<small>{detail}</small></div></div></td>'
            f'<td>{info}</td><td class="right"><span class="pill {pill_class}">{pill}</span></td></tr>')


def overview_table_html():
    rows = []
    for s_ in ws["surveys"]:
        if s_["vol"] is not None:
            v = s_["vol"]
            info = (f'{len(v["ilines"]):,} inlines · {len(v["xlines"]):,} crosslines · '
                    f'{len(v["times"]):,} samples')
            rows.append(_row(ICON_VIEWER, escape(s_["title"]), f'{s_["role"]} · {escape(s_["name"])}',
                             "Loaded", "ok", info))
        elif s_["error"]:
            rows.append(_row(ICON_VIEWER, escape(s_["title"] or s_["role"]), s_["role"], "Error", "err",
                             escape(s_["error"])))
        else:
            rows.append(_row(ICON_VIEWER, s_["role"], "no file chosen", "Not loaded", ""))
    n_shapes = sum(shape_count(v) for v in VIEWS)
    rows.append(_row(ICON_SHAPE, "Shapes", "polygons, lines and points (Viewer → Layers)",
                     f"{n_shapes} shape{'s' if n_shapes != 1 else ''}" if n_shapes else "None",
                     "accent" if n_shapes else ""))
    n_hz = len(ws["horizons"])
    rows.append(_row(ICON_LAYERS, "Horizons", "loaded from files or drawn (Viewer → Layers)",
                     f"{n_hz} horizon{'s' if n_hz != 1 else ''}" if n_hz else "None", "accent" if n_hz else ""))
    n_wells = len(ws["wells"])
    rows.append(_row(ICON_WELL, "Wells", "from LAS files, tables or the map",
                     f"{n_wells} well{'s' if n_wells != 1 else ''}" if n_wells else "None",
                     "accent" if n_wells else ""))
    an = ws.get("analysis")
    rows.append(_row(ICON_ANALYSIS, "Analysis", "baseline vs monitor, rule-based",
                     "Done" if an else "Not run", "ok" if an else "warn",
                     f'{an["volume"]["changed_pct"]:.1f}% of the area changed' if an else ""))
    return ('<table class="overview"><thead><tr><th>Item</th><th>Details</th>'
            '<th style="text-align:right">Status</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


def refresh_overview():
    overview_table.object = overview_table_html()
    an = ws.get("analysis")
    if an:
        overview_analysis.objects = [pn.Column(
            html('<div class="head"><span class="label">Latest analysis</span>'
                 f'<span class="meta">{escape(an["labels"][0])} → {escape(an["labels"][1])}</span></div>',
                 sizing_mode="stretch_width"),
            stats_html(an["volume"]),
            sizing_mode="stretch_width", styles=CARD_STYLES, margin=(8, 0))]
    else:
        overview_analysis.objects = []


def refresh_data_pill():
    n = sum(1 for s_ in ws["surveys"] if s_["vol"] is not None)
    nav_menu.loaded = n > 0
    if n:
        data_pill.object = f'<span class="pill ok">{n} survey{"s" if n != 1 else ""} loaded</span>'
    else:
        data_pill.object = '<span class="pill">No surveys loaded</span>'


# ---------------------------------------------------------------------------
# Wells page (wells.py) and switching pages from the sidebar
# ---------------------------------------------------------------------------
class _Live:
    """Gives wells.py live access to this file's globals (values like AMP_MAX
    change after loading, so a copy taken now would go stale)."""
    def __init__(self, g):
        self.__dict__["_g"] = g

    def __getattr__(self, name):
        try:
            return self._g[name]
        except KeyError:
            raise AttributeError(name) from None


import wells  # noqa: E402

wells_ui = wells.WellsPage(_Live(globals()))


def show_page(page):
    ws["page"] = page
    title_block.object = title_html(page)
    has_data = any(s_["vol"] is not None for s_ in ws["surveys"])
    refresh_data_pill()
    if page == "overview":
        refresh_overview()
    # each data page: its empty state until surveys are loaded, then its content
    for pg, empty, body in ((overview_page, overview_page[0], overview_body),
                            (viewer_page, viewer_page[0], viewer_body),
                            (analysis_page, analysis_page[0], analysis_card)):
        empty.visible = not has_data
        body.visible = has_data
    overview_page.visible = page == "overview"
    viewer_page.visible = page == "viewer"
    analysis_page.visible = page == "analysis"
    data_page.visible = page == "data"
    wells_ui.layout.visible = page == "wells"


nav_menu.param.watch(lambda event: show_page(event.new), "page")
show_page("overview")

main = pn.Column(
    # The sidebar is pinned to the window (position: fixed), so it takes no room in
    # this column. Keeping it here rather than in a Row beside the column stops its
    # width being counted twice (once for itself, once in the left margin below),
    # which made the page wider than the window on narrower screens.
    sidebar,
    loading_overlay,  # also fixed to the window: covers everything while Run works
    header,
    overview_page,
    viewer_page,
    analysis_page,
    wells_ui.layout,
    data_page,
    sizing_mode="stretch_width",
    # top / bottom room as margins (left / right are padding on the Row below)
    margin=(SIDEBAR_GAP + 14, 0, 64, 0),
    # Panel caps a layout at its parent's height (max-height: 100%); lifted here so the
    # page grows with its content instead of being cut at one window height, which
    # left the bottom of long pages spilling out of their cards
    styles={"max-height": "none"},
)

pn.Row(
    main,
    sizing_mode="stretch_width",  # grows with the content
    # The room for the pinned sidebar is padding here rather than a margin on `main`:
    # Panel adds a child's margins to its minimum width AND the browser applies them
    # again, so a wide margin made narrower windows scroll sideways.
    # height auto: otherwise Panel pins this Row to the window's height, and every page
    # inside is cut off at one screen (long pages then spill out of their cards)
    styles={"background": "var(--bg-grad)", "background-attachment": "fixed", "min-height": "100vh",
            "height": "auto",
            "padding": f"0 32px 0 {SIDEBAR_GAP + SIDEBAR_WIDTH + 32}px",
            "box-sizing": "border-box"},
    margin=0,
).servable(title="Reservoir Time Machine")