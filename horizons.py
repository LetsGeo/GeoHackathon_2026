"""
Horizons on the seismic: loaded from a file, or drawn on an inline / crossline
section. app.py builds one HorizonLayer and hooks it into the workspace plots.

A horizon is a set of points (inline, crossline, two-way time in ms):
  - on an inline / crossline section it is drawn as a line through the points
    on (or nearest to) that section
  - on the time-slice map it is drawn as dots where the horizon sits at that time
Horizon files are text tables (Petrel / IESX style exports): with a header row
(INLINE, CROSSLINE, X, Y, TWT ...), or without one, in which case each column
is recognised from its values (inline and crossline numbers fall in the survey's
ranges, X / Y in its map coordinates, the rest is time).
"""
import asyncio
import csv
import io
import re
from html import escape
from pathlib import Path

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource, LabelSet, PolyDrawTool

from styles import (BTN_BASE_CSS, CARD_STYLES, DOWNLOAD_BTN_CSS, DOWNLOAD_BTN_HEIGHT, DRAWN_HORIZON_COLOUR,
                    HORIZON_COLOURS, ICON_HORIZON, INPUT_CSS, PRIMARY_BTN_CSS)

COLUMN_ALIASES = {
    "il": ("inline", "il", "iline", "inline_no", "line"),
    "xl": ("crossline", "xl", "xline", "crossline_no", "trace", "cdp"),
    "x": ("x", "easting", "east", "x_coord", "xcoord", "cdp_x"),
    "y": ("y", "northing", "north", "y_coord", "ycoord", "cdp_y"),
    "t": ("twt", "twt_ms", "time", "time_ms", "twt_s", "z", "value"),
}
MAP_TOLERANCE_MS = 2.0   # on the time-slice map, show horizon points within this many ms of the slice


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def read_table(text):
    """(header names or None, array of numeric rows) from a text table."""
    header, rows = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!", "@", "//")):
            continue
        tokens = re.split(r"[,\t; ]+", line)
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            if not rows:  # a non-numeric line before the numbers: the header
                header = [_norm(t) for t in tokens]
    if not rows:
        raise ValueError("no rows of numbers found")
    widths = [len(r) for r in rows]
    width = max(set(widths), key=widths.count)
    return header, np.array([r for r in rows if len(r) == width], dtype=np.float64)


def _within(values, rng, slack=0.05):
    lo, hi = rng
    pad = (hi - lo) * slack + 1
    return np.nanmin(values) >= lo - pad and np.nanmax(values) <= hi + pad


def parse_horizon(text, ranges, world_to_lines=None):
    """(inline, crossline, twt) arrays from a horizon text file.
    ranges: {"il": (lo, hi), "xl": (lo, hi), "x": (lo, hi) or None, "y": (lo, hi) or None}."""
    header, arr = read_table(text)
    cols = {}
    if header and len(header) == arr.shape[1]:
        for key, names in COLUMN_ALIASES.items():
            for j, h in enumerate(header):
                if h in names and j not in cols.values():
                    cols[key] = j
                    break
    if "t" not in cols or not (("il" in cols and "xl" in cols) or ("x" in cols and "y" in cols)):
        cols = {}  # no usable header: recognise each column from its values
        for j in range(arr.shape[1]):
            v = arr[:, j][np.isfinite(arr[:, j])]
            if len(v) == 0:
                continue
            for key in ("il", "xl", "x", "y"):
                if key not in cols and ranges.get(key) and _within(v, ranges[key]):
                    cols[key] = j
                    break
        rest = [j for j in range(arr.shape[1]) if j not in cols.values()]
        if rest:
            cols["t"] = rest[-1]  # time is usually the last column
    if "t" not in cols:
        raise ValueError("couldn't find the time column")
    t = arr[:, cols["t"]].copy()
    if "il" in cols and "xl" in cols:
        il, xl = arr[:, cols["il"]], arr[:, cols["xl"]]
    elif "x" in cols and "y" in cols and world_to_lines is not None:
        il, xl = world_to_lines(arr[:, cols["x"]], arr[:, cols["y"]])
    else:
        raise ValueError("couldn't find inline / crossline (or X / Y) columns that fit these surveys")
    t = np.abs(t)                       # some exports write time as negative (downwards)
    if np.nanmax(t) < 20:               # seconds -> ms
        t = t * 1000.0
    ok = np.isfinite(il) & np.isfinite(xl) & np.isfinite(t) & (t > 0) & (t < 20000)  # drop null values
    if not ok.any():
        raise ValueError("no valid points (all times missing or out of range)")
    return np.asarray(il[ok], float), np.asarray(xl[ok], float), np.asarray(t[ok], float)


def _half_spacing(values):
    u = np.unique(np.round(values, 3))
    return float(np.median(np.diff(u))) / 2 if len(u) > 1 else 0.5


def make_horizon(name, colour, il, xl, t, drawn_on=None):
    return {"name": name, "colour": colour, "il": il, "xl": xl, "t": t, "drawn_on": drawn_on,
            "tol_il": max(_half_spacing(il), 0.5), "tol_xl": max(_half_spacing(xl), 0.5)}


def section_points(h, view, pos):
    """(along, twt) of the horizon on this inline / crossline: the points on, or
    nearest to, the section (within half the horizon's line spacing). Gaps in
    the horizon are kept as gaps (NaN), so a line isn't drawn across them."""
    if view == "Inline":
        across, along, tol = h["il"], h["xl"], h["tol_il"]
    else:
        across, along, tol = h["xl"], h["il"], h["tol_xl"]
    d = np.abs(across - pos)
    near = d <= tol
    if not near.any():
        return None, None
    # just the nearest line (if two are equally near, e.g. halfway between grid lines, take one of them)
    nearest_line = across[near][np.argmin(d[near])]
    near &= np.abs(across - nearest_line) < 1e-6
    a, t = along[near], h["t"][near]
    order = np.argsort(a)
    a, t = a[order], t[order]
    if len(a) > 2:  # break the line where points are far apart
        step = np.median(np.diff(a))
        gaps = np.where(np.diff(a) > 3 * max(step, 1e-9))[0]
        if len(gaps):
            a = np.insert(a, gaps + 1, np.nan)
            t = np.insert(t, gaps + 1, np.nan)
    return a, t


class HorizonLayer:
    """Horizons for the workspace: the Horizons card, and the horizon lines on the survey plots.
    `app` gives access to app.py's pieces (ws, html, pick_file, ...)."""

    def __init__(self, app):
        self.app = app
        app.ws.setdefault("horizons", [])
        self.visible = True
        self.renderers = []          # everything drawn for horizons, so the Show toggle can hide it
        self.draw_tools = []         # the Draw horizon tools (hidden on time slices)
        self.src = None              # this build's plot sources
        self._updating = False       # True while the app (not the user) changes the drawn-horizon source
        self._pending = []           # horizons drawn on this section whose plot rows don't carry an id yet
        self._counter = 0

        self.path = pn.widgets.TextInput(name="Horizon file", placeholder="e.g. horizons/TOP-UTSIRA-DEMO.txt",
                                         width=300, stylesheets=[INPUT_CSS], margin=(0, 10, 0, 0))
        self.browse_btn = pn.widgets.Button(name="Browse…", stylesheets=[BTN_BASE_CSS], width=92, align="end",
                                            margin=(0, 10, 0, 0))
        self.load_btn = pn.widgets.Button(name="Load", stylesheets=[BTN_BASE_CSS, PRIMARY_BTN_CSS], width=90,
                                          align="end", margin=(0, 10, 0, 0))
        self.clear_btn = pn.widgets.Button(name="Clear", stylesheets=[BTN_BASE_CSS], width=90, align="end",
                                           margin=(0, 10, 0, 0), disabled=True)
        self.download = pn.widgets.FileDownload(callback=self.make_csv, filename="horizons.csv", label="Download",
                                                button_type="default", width=120, height=DOWNLOAD_BTN_HEIGHT,
                                                disabled=True, stylesheets=[BTN_BASE_CSS, DOWNLOAD_BTN_CSS],
                                                align="end", margin=(0, 10, 0, 0))
        self.status = app.html("", sizing_mode="stretch_width", margin=(8, 0, 0, 0))
        self.legend = app.html("", sizing_mode="stretch_width", margin=(6, 0, 0, 0))

        async def on_browse(event):  # async def, so Panel runs it (see wells.py on_browse)
            await self.on_browse()
        self.browse_btn.on_click(on_browse)
        self.load_btn.on_click(self.on_load)
        self.clear_btn.on_click(self.on_clear)

        self.card = pn.Column(
            app.html('<div class="head"><span class="label">Horizons</span></div>', sizing_mode="stretch_width"),
            # wraps onto a second line on narrow windows instead of widening the page
            pn.FlexBox(self.path, self.browse_btn, self.load_btn, self.clear_btn, self.download,
                       flex_wrap="wrap", align_items="flex-end", sizing_mode="stretch_width", gap="10px 0px",
                       margin=(6, 0, 0, 0)),
            self.legend,
            self.status,
            sizing_mode="stretch_width", styles=CARD_STYLES, margin=(8, 0),
        )

    # ------------------------------------------------------------------ data
    @property
    def horizons(self):
        return self.app.ws["horizons"]

    def ranges(self):
        """Line-number and map-coordinate ranges of the loaded surveys (for recognising columns)."""
        vols = [s["vol"] for s in self.app.ws["surveys"] if s["vol"] is not None]
        il = np.concatenate([v["ilines"] for v in vols])
        xl = np.concatenate([v["xlines"] for v in vols])
        out = {"il": (il.min(), il.max()), "xl": (xl.min(), xl.max()), "x": None, "y": None}
        geo = self.app.ws["geo"]
        if geo is not None:
            ci = np.array([il.min(), il.min(), il.max(), il.max()])
            cx = np.array([xl.min(), xl.max(), xl.min(), xl.max()])
            x, y = self.app.lines_to_world(geo, ci, cx)
            out["x"], out["y"] = (float(x.min()), float(x.max())), (float(y.min()), float(y.max()))
        return out

    def next_colour(self):
        loaded = sum(1 for h in self.horizons if h["drawn_on"] is None)
        return HORIZON_COLOURS[loaded % len(HORIZON_COLOURS)]

    def refresh_card(self):
        has = bool(self.horizons)
        self.clear_btn.disabled = self.download.disabled = not has
        self.legend.object = "".join(
            f'<span class="meta" style="margin-right:14px"><span style="display:inline-block;width:18px;height:3px;'
            f'background:{h["colour"]};vertical-align:middle;margin-right:6px;border-radius:2px"></span>'
            f'{escape(h["name"])}</span>' for h in self.horizons)

    def make_csv(self):
        geo = self.app.ws["geo"]
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(["HORIZON", "INLINE", "CROSSLINE"] + (["X", "Y"] if geo is not None else []) + ["TWT_MS"])
        for h in self.horizons:
            xs, ys = self.app.lines_to_world(geo, h["il"], h["xl"]) if geo is not None else (None, None)
            for i in range(len(h["t"])):
                row = [h["name"], f'{h["il"][i]:.2f}', f'{h["xl"][i]:.2f}']
                if geo is not None:
                    row += [f"{xs[i]:.2f}", f"{ys[i]:.2f}"]
                w.writerow(row + [f'{h["t"][i]:.2f}'])
        return io.BytesIO(out.getvalue().encode("utf-8"))

    # -------------------------------------------------------------- handlers
    def say(self, msg, ok=True):
        self.status.object = f'<span class="status {"ok" if ok else "err"}">{msg}</span>' if msg else ""

    async def on_browse(self):
        A = self.app
        current = A.resolve_path(self.path.value) if self.path.value.strip() else A.APP_DIR
        start = current.parent if current.parent.is_dir() else A.APP_DIR
        self.browse_btn.disabled = True
        try:
            chosen = await asyncio.get_running_loop().run_in_executor(None, A.pick_file,
                                                                      "Choose a horizon file", start)
            if chosen:
                self.path.value = chosen
        except Exception as exc:
            self.say(f"Couldn't open the file dialog ({escape(str(exc))}); type or paste the path.", ok=False)
        finally:
            self.browse_btn.disabled = False

    def on_load(self, event=None):
        A = self.app
        if not any(s["vol"] is not None for s in A.ws["surveys"]):
            return self.say("Load the surveys first (Data sources, then Run)", ok=False)
        text = self.path.value.strip()
        if not text:
            return self.say("Type or browse to a horizon file", ok=False)
        path = A.resolve_path(text)
        if not path.is_file():
            return self.say(f"File not found: {escape(str(path))}", ok=False)
        try:
            geo = A.ws["geo"]
            to_lines = (lambda x, y: A.world_to_lines(geo, x, y)) if geo is not None else None
            il, xl, t = parse_horizon(path.read_text(errors="ignore"), self.ranges(), to_lines)
        except Exception as exc:
            return self.say(f"Couldn't read {escape(path.name)}: {escape(str(exc))}", ok=False)
        name = path.stem
        self.horizons[:] = [h for h in self.horizons if h["name"] != name]  # loading again replaces it
        self.horizons.append(make_horizon(name, self.next_colour(), il, xl, t))
        self.say("Loaded")
        self.refresh_card()
        self.refresh()

    def on_clear(self, event=None):
        self.horizons.clear()
        self._pending = []
        self.say("")
        self.refresh_card()
        self.refresh()

    # --------------------------------------------------------------- drawing
    def new_build(self):
        """Fresh plot sources for a rebuild of the workspace plots (call before attach)."""
        self.renderers = []
        self.draw_tools = []
        self._pending = []
        self.src = {
            "lines": ColumnDataSource({"xs": [], "ys": [], "name": [], "colour": []}),
            "drawn": ColumnDataSource({"xs": [], "ys": [], "hid": []}),   # editable: this section's drawn ones
            "dots": ColumnDataSource({"x": [], "y": [], "colour": []}),
            "labels": ColumnDataSource({"x": [], "y": [], "text": []}),
        }
        self.src["drawn"].on_change("data", self.on_drawn_edit)

    def attach(self, p, editable, view):
        """Draw the horizons on one plot; on the survey sections, add the Draw horizon tool."""
        s = self.src
        rs = [
            p.multi_line("xs", "ys", source=s["lines"], line_color="colour", line_width=2.5, line_alpha=0.95),
            p.multi_line("xs", "ys", source=s["drawn"], line_color=DRAWN_HORIZON_COLOUR, line_width=2.5),
            p.scatter("x", "y", source=s["dots"], size=3, color="colour", alpha=0.9),
        ]
        labels = LabelSet(x="x", y="y", text="text", source=s["labels"], x_offset=4, y_offset=4,
                          text_font_size="10.5px", text_color="#0b1418", text_font_style="bold",
                          background_fill_color="white", background_fill_alpha=0.75)
        p.add_layout(labels)
        rs.append(labels)
        if editable:
            vertices = p.scatter("x", "y", source=ColumnDataSource({"x": [], "y": []}), size=7,
                                 fill_color="white", line_color=DRAWN_HORIZON_COLOUR, line_width=2)
            rs.append(vertices)
            # always added, but only shown on inline / crossline sections (the app keeps the
            # same plots when the view changes, so the tool is switched, not re-created)
            tool = PolyDrawTool(renderers=[rs[1]], vertex_renderer=vertices, drag=True, empty_value="",
                                icon=ICON_HORIZON, visible=view != "Time slice",
                                description="Draw horizon (hold to start, click along the reflector, "
                                            "hold or Esc to finish; click it + Backspace to delete)")
            p.add_tools(tool)
            self.draw_tools.append(tool)
        for r in rs:
            r.visible = self.visible
        self.renderers += rs

    def set_view(self, view):
        """The view changed on the same plots: show the Draw horizon tool on sections only."""
        for tool in self.draw_tools:
            tool.visible = view != "Time slice"

    def set_visible(self, visible):
        self.visible = visible
        for r in self.renderers:
            r.visible = visible

    def refresh(self):
        """Redraw the horizons for the current view and slice."""
        if self.src is None:
            return
        view, pos = self.app.current()
        lines = {"xs": [], "ys": [], "name": [], "colour": []}
        drawn = {"xs": [], "ys": [], "hid": []}
        dots = {"x": [], "y": [], "colour": []}
        labels = {"x": [], "y": [], "text": []}
        for i, h in enumerate(self.horizons):
            if view == "Time slice":  # map: dots where the horizon is at this time (x = crossline, y = inline)
                near = np.abs(h["t"] - pos) <= MAP_TOLERANCE_MS
                dots["x"] += h["xl"][near].tolist()
                dots["y"] += h["il"][near].tolist()
                dots["colour"] += [h["colour"]] * int(near.sum())
                continue
            if h["drawn_on"] == (view, pos):  # drawn on this very section: editable
                along = h["xl"] if view == "Inline" else h["il"]
                drawn["xs"].append(along.tolist()), drawn["ys"].append(h["t"].tolist())
                drawn["hid"].append(str(id(h)))
                a, t = along, h["t"]
            else:
                a, t = section_points(h, view, pos)
                if a is None:
                    continue
                if np.isfinite(a).sum() == 1:  # crosses this section at one point (drawn on another line)
                    k = np.where(np.isfinite(a))[0][0]
                    dots["x"].append(float(a[k])), dots["y"].append(float(t[k])), dots["colour"].append(h["colour"])
                else:
                    lines["xs"].append(a.tolist()), lines["ys"].append(t.tolist())
                    lines["name"].append(h["name"]), lines["colour"].append(h["colour"])
            k = int(np.nanargmin(a)) if np.isfinite(a).any() else None
            if k is not None:
                labels["x"].append(float(a[k])), labels["y"].append(float(t[k])), labels["text"].append(h["name"])
        self._updating = True
        try:
            self.src["lines"].data = lines
            self.src["drawn"].data = drawn
            self.src["dots"].data = dots
            self.src["labels"].data = labels
        finally:
            self._updating = False
        self._pending = []

    def on_drawn_edit(self, attr, old, new):
        """A horizon was drawn, moved or deleted on this section: update the stored horizons.
        (Nothing is written back to the plot here, so a line being drawn isn't disturbed.)"""
        if self._updating:
            return
        view, pos = self.app.current()
        if view == "Time slice":
            return
        by_id = {str(id(h)): h for h in self.horizons if h["drawn_on"] == (view, pos)}
        keep_ids, new_rows = set(), []
        for xs, ys, hid in zip(new.get("xs", []), new.get("ys", []), new.get("hid", [])):
            xs, ys = np.asarray(xs, float), np.asarray(ys, float)
            if len(xs) < 2:
                continue  # still being started
            if hid and hid in by_id:
                h = by_id[hid]
                self._set_points(h, view, pos, xs, ys)
                keep_ids.add(hid)
            else:
                new_rows.append((xs, ys))
        # rows without an id: match them, in order, to horizons drawn since the last redraw
        # (horizons hold arrays, so they're compared by identity, never with == / in / remove)
        def stored(h):
            return any(x is h for x in self.horizons)

        pending = []
        for k, (xs, ys) in enumerate(new_rows):
            if k < len(self._pending) and stored(self._pending[k]):
                h = self._pending[k]
            else:
                self._counter += 1
                h = make_horizon(f"Horizon {self._counter}", DRAWN_HORIZON_COLOUR, xs, xs, ys, drawn_on=(view, pos))
                self.horizons.append(h)
            self._set_points(h, view, pos, xs, ys)
            pending.append(h)
        self.horizons[:] = [h for h in self.horizons
                            if h["drawn_on"] != (view, pos) or str(id(h)) in keep_ids
                            or any(h is p for p in pending)]
        self._pending = pending
        self.refresh_card()

    @staticmethod
    def _set_points(h, view, pos, along, t):
        order = np.argsort(along)
        along, t = along[order], t[order]
        fixed = np.full(len(along), float(pos))
        h["il"], h["xl"] = (fixed, along) if view == "Inline" else (along, fixed)
        h["t"] = t
        h["tol_il"] = h["tol_xl"] = 0.5
