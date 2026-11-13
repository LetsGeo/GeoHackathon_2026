"""
Wells page: well paths drawn over the seismic, in a map view (time slice) and
in inline / crossline sections. app.py builds it with WellsPage(app), where
`app` gives live access to app.py's helpers and per-session state (ws).

Wells come from a .csv / .txt file or from clicking on the map. Each well is
kept in survey line numbers (inline, crossline) plus two-way time in ms when
known, so it can be drawn on any view:
    {"name": str, "il": array, "xl": array, "twt": array or None, "colour": str}
A well with one row is a surface location only (drawn as a dashed vertical
line on sections); several rows make a path.

Wells can also be loaded from a LAS file + checkshot ("Load well from LAS"):
welllogs.py converts the logs to two-way time and finds the reservoir; such a
well also carries "logs" (curves against TWT, for the log tracks beside the
seismic) and "reservoir" (the picked top / base TWT, drawn on the seismic).
"""
import asyncio
import csv
import io
import re
from html import escape
from pathlib import Path

import numpy as np
import panel as pn
from bokeh.events import Tap
from bokeh.layouts import gridplot
from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool, LabelSet, LinearColorMapper, Range1d
from bokeh.plotting import figure

import welllogs
from styles import (ADD_TOGGLE_CSS, CANDIDATE_COLOUR, CHECKBOX_CSS, RESERVOIR_COLOUR, TABLE_CSS,
                    WELL_COLOURS)

WELL_EXTS = (".csv", ".txt")
# accepted column names (compared after lower-casing and turning spaces /
# brackets into "_", so "TWT (ms)" matches "twt_ms")
COLUMN_ALIASES = {
    "name": ("well", "well_name", "wellname", "name", "uwi", "well_id"),
    "x": ("x", "easting", "east", "x_loc", "xcoord", "x_coord", "utm_x", "cdp_x"),
    "y": ("y", "northing", "north", "y_loc", "ycoord", "y_coord", "utm_y", "cdp_y"),
    "il": ("inline", "il", "iline", "inline_no"),
    "xl": ("crossline", "xl", "xline", "crossline_no"),
    "twt": ("twt", "twt_ms", "time", "time_ms", "twt_msec"),
    "depth": ("tvdss", "tvd_ss", "tvd", "tvd_m", "depth", "depth_m", "z"),
}
PAGE_VIEWS = {"Map": "Time slice", "Inline": "Inline", "Crossline": "Crossline"}
SURVEY_ROLE = "Baseline"   # the Wells page always draws the wells over the baseline survey
LOG_STEP_MS = 1.0              # log tracks show 1 ms averages (a LAS has ~10x more samples than needed)
PATH_STEP_MS = 4.0             # well path drawn with a point every 4 ms
CHECKSHOT_DEPTH_TYPES = {"TVDSS (below sea level)": "TVDSS", "TVD below KB": "TVD_KB", "MD from KB": "MD"}
CHECKSHOT_TIME_TYPES = {"Two-way time (TWT)": "TWT", "One-way time (OWT)": "OWT"}


ROW_GAP = 18  # space above each row of labelled boxes, so a label isn't squashed against the box above it


def wrap_row(*objects, margin=(10, 0, 0, 0)):
    """A row of controls that wraps onto the next line when the window is narrow.
    (A pn.Row of fixed-width widgets sets a minimum page width instead, even
    while its page is hidden, which gives the whole app a horizontal scrollbar.)"""
    return pn.FlexBox(*objects, flex_wrap="wrap", align_items="flex-end", sizing_mode="stretch_width",
                      gap=f"{ROW_GAP}px 0px", margin=margin)  # FlexBox's own gap: space between wrapped lines


def norm(header):
    return re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")


def read_table(text):
    """Rows of a delimited text file (comma, semicolon, tab or spaces)."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    first = lines[0]
    if any(d in first for d in ",;\t"):
        delim = max(",;\t", key=first.count)
        return [[c.strip() for c in row] for row in csv.reader(lines, delimiter=delim)]
    return [ln.split() for ln in lines]


def parse_wells(text, geo, velocity, default_name):
    """Wells from a table. Returns (wells, notes). Raises ValueError if the
    columns needed for positions aren't there."""
    rows = read_table(text)
    if len(rows) < 2:
        raise ValueError("the file needs a header row and at least one data row")
    header = [norm(h) for h in rows[0]]
    col = {}
    for key, names in COLUMN_ALIASES.items():
        for i, h in enumerate(header):
            if h in names:
                col[key] = i
                break
    has_xy, has_lines = "x" in col and "y" in col, "il" in col and "xl" in col
    if not has_xy and not has_lines:
        raise ValueError("no position columns found: add X and Y (easting / northing) "
                         "or INLINE and CROSSLINE")
    use_xy = has_xy and not has_lines
    if use_xy and geo is None:
        raise ValueError("these surveys have no map coordinates in their headers, "
                         "so give INLINE and CROSSLINE columns instead of X / Y")

    def num(row, key):
        try:
            return float(row[col[key]]) if key in col else np.nan
        except (ValueError, IndexError):
            return np.nan

    groups, skipped = {}, 0
    for row in rows[1:]:
        name = row[col["name"]].strip() if "name" in col and col["name"] < len(row) else default_name
        a, b = (num(row, "x"), num(row, "y")) if use_xy else (num(row, "il"), num(row, "xl"))
        if not (np.isfinite(a) and np.isfinite(b)):
            skipped += 1
            continue
        z = num(row, "twt")
        if not np.isfinite(z) and "depth" in col:
            d = num(row, "depth")
            z = 2000.0 * abs(d) / velocity if np.isfinite(d) else np.nan  # depth (m) -> two-way time (ms)
        groups.setdefault(name or default_name, []).append((a, b, z))

    wells = []
    for name, pts in groups.items():
        arr = np.asarray(pts, dtype=np.float64)
        if use_xy:
            il, xl = geo_to_lines(geo, arr[:, 0], arr[:, 1])
        else:
            il, xl = arr[:, 0], arr[:, 1]
        twt = arr[:, 2]
        if np.isfinite(twt).all():
            order = np.argsort(twt, kind="stable")
            il, xl, twt = il[order], xl[order], twt[order]
        else:
            twt = None
        wells.append({"name": name, "il": np.asarray(il, dtype=np.float64),
                      "xl": np.asarray(xl, dtype=np.float64), "twt": twt})
    notes = [f"{skipped} row{'s' if skipped != 1 else ''} without a usable position skipped"] if skipped else []
    return wells, notes


def geo_to_lines(M, x, y):
    rhs = np.vstack([np.asarray(x, dtype=np.float64) - M[0, 0], np.asarray(y, dtype=np.float64) - M[1, 0]])
    il, xl = np.linalg.solve(M[:, 1:], rhs)
    return il, xl


def lines_to_geo(M, il, xl):
    il, xl = np.asarray(il, dtype=np.float64), np.asarray(xl, dtype=np.float64)
    return M[0, 0] + M[0, 1] * il + M[0, 2] * xl, M[1, 0] + M[1, 1] * il + M[1, 2] * xl


class WellsPage:
    def __init__(self, app):
        self.app = app
        ws = app.ws
        ws.setdefault("wells", [])
        ws.setdefault("wfigs", [])
        self.plot = None  # live plot pieces: {"img_src", "path_src", ...}
        A = app

        # --- wells card: upload / download / clear -------------------------
        self.upload = pn.widgets.FileInput(accept=",".join(WELL_EXTS), multiple=True, width=330,
                                           stylesheets=[A.FILE_INPUT_CSS], align="end", margin=(0, 10, 0, 0))
        self.velocity = pn.widgets.IntInput(name="Velocity for depth → time (m/s)", value=2000, start=500,
                                            end=8000, step=50, width=210, stylesheets=[A.INPUT_CSS],
                                            margin=(0, 10, 0, 0))
        self.download = pn.widgets.FileDownload(callback=self.make_csv, filename="wells.csv",
                                                label="Download wells", button_type="default", width=160,
                                                height=A.DOWNLOAD_BTN_HEIGHT, disabled=True,
                                                stylesheets=[A.BTN_BASE_CSS, A.DOWNLOAD_BTN_CSS], align="end",
                                                margin=(0, 10, 0, 0))
        self.clear_btn = pn.widgets.Button(name="Remove all", stylesheets=[A.BTN_BASE_CSS], width=110,
                                           align="end", disabled=True, margin=(0, 10, 0, 0))
        self.status = A.html("", sizing_mode="stretch_width", margin=(8, 0, 0, 0))
        self.upload.param.watch(self.on_upload, "value")
        self.clear_btn.on_click(self.on_clear)

        # "From a table": accepted columns are WELL, X / EASTING, Y / NORTHING (or INLINE,
        # CROSSLINE), and optionally TWT (ms) or a depth (TVD / TVDSS, m) converted with the velocity box
        self.table_row = wrap_row(self.upload, self.velocity, margin=(14, 0, 0, 0))

        # --- load a well from LAS + checkshot ----------------------------------
        # These boxes stretch to fill their column (min_width keeps them usable), so each
        # row below always keeps its columns side by side instead of wrapping.
        def path_box(name, placeholder):
            # small min_width: the rows never wrap, so their minimum sets how narrow the page can get
            return pn.widgets.TextInput(name=name, placeholder=placeholder, sizing_mode="stretch_width",
                                        min_width=60, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))

        def small_box(name, placeholder=""):
            return pn.widgets.TextInput(name=name, placeholder=placeholder, sizing_mode="stretch_width",
                                        min_width=50, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))

        def with_browse(target, title):
            """A path box and its Browse button: one column of a row."""
            btn = pn.widgets.Button(name="Browse…", stylesheets=[A.BTN_BASE_CSS], width=92, align="end",
                                    margin=0)
            # must be an `async def`: Panel only awaits handlers it can see are async (a lambda
            # around on_browse would just create the task and never run it, so nothing would open)
            async def on_click(event):
                await self.on_browse(btn, target, title)
            btn.on_click(on_click)
            return pn.Row(target, btn, sizing_mode="stretch_width", margin=(0, 24, 0, 0))

        def column(*objects):
            """One equal-width column of a row."""
            return pn.Row(*objects, sizing_mode="stretch_width", margin=(0, 24, 0, 0))

        def columns_row(*cols, top=ROW_GAP):
            """A row split into equal columns that never wrap onto a second line."""
            return pn.Row(*cols, sizing_mode="stretch_width", margin=(top, 0, 0, 0))

        self.las_path = path_box("LAS file", "e.g. wells/SLEIPNER-DEMO-1.las")
        self.cs_path = path_box("Checkshot file", "e.g. wells/SLEIPNER-DEMO-1_Checkshot.txt")
        self.dev_path = path_box("Deviation survey (optional)", "empty = vertical well")
        self.loc_x = small_box("Surface X (easting)", "from LAS header")
        self.loc_y = small_box("Surface Y (northing)", "from LAS header")
        self.loc_il = small_box("Inline")
        self.loc_xl = small_box("Crossline")
        self.kb_box = small_box("KB elevation (m)", "from LAS header")
        self.use_gemma = pn.widgets.Checkbox(name="Ask Gemma to pick the reservoir", value=True,
                                             stylesheets=[CHECKBOX_CSS], align="end", margin=(0, 18, 10, 0))
        self.cs_settings_btn = pn.widgets.Toggle(name="Checkshot format ▾", value=False, width=170,
                                                 stylesheets=[A.BTN_BASE_CSS, ADD_TOGGLE_CSS], align="end",
                                                 margin=(0, 10, 0, 0))
        self.load_las_btn = pn.widgets.Button(name="Load well", stylesheets=[A.BTN_BASE_CSS, A.PRIMARY_BTN_CSS],
                                              width=120, align="end", margin=(0, 10, 0, 0))
        self.las_status = A.html("", sizing_mode="stretch_width", margin=(8, 0, 0, 0))
        # how the checkshot file is laid out (hidden until "Checkshot format" is pressed)
        d = welllogs.CHECKSHOT_DEFAULTS
        self.cs_depth_col = pn.widgets.IntInput(name="Depth column (1 = first)", value=d["depth_col"] + 1, start=1,
                                                end=50, width=170, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.cs_time_col = pn.widgets.IntInput(name="Time column", value=d["time_col"] + 1, start=1, end=50,
                                               width=120, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.cs_depth_type = pn.widgets.Select(name="Depth is", options=CHECKSHOT_DEPTH_TYPES, value=d["depth_type"],
                                               width=200, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.cs_time_type = pn.widgets.Select(name="Time is", options=CHECKSHOT_TIME_TYPES, value=d["time_type"],
                                              width=190, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.cs_time_unit = pn.widgets.Select(name="Time unit", options=["ms", "s"], value=d["time_unit"],
                                              width=100, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.cs_shift = pn.widgets.FloatInput(name="Time shift (ms)", value=d["shift_ms"], width=130,
                                              stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.cs_negative = pn.widgets.Checkbox(name="Depths are negative", value=d["depth_sign"] < 0,
                                               stylesheets=[CHECKBOX_CSS], align="end", margin=(0, 10, 10, 0))
        self.cs_row = wrap_row(self.cs_depth_col, self.cs_time_col, self.cs_depth_type, self.cs_time_type,
                               self.cs_time_unit, self.cs_shift, self.cs_negative)
        self.cs_row.visible = False
        self.cs_settings_btn.param.watch(lambda e: setattr(self.cs_row, "visible", e.new), "value")
        self.load_las_btn.on_click(self.on_load_las)

        # LAS file | Checkshot file
        # Deviation survey | KB elevation
        # Surface X | Surface Y | Inline | Crossline
        self.las_section = pn.Column(
            columns_row(with_browse(self.las_path, "Choose the LAS file"),
                        with_browse(self.cs_path, "Choose the checkshot file"), top=14),
            # KB sits in the right-hand column, half as wide as the column, under the checkshot box
            columns_row(with_browse(self.dev_path, "Choose the deviation survey"),
                        column(self.kb_box, pn.Spacer(sizing_mode="stretch_width"))),
            columns_row(column(self.loc_x, self.loc_y), column(self.loc_il, self.loc_xl)),
            wrap_row(self.cs_settings_btn, self.use_gemma, self.load_las_btn, margin=(ROW_GAP + 4, 0, 0, 0)),
            self.cs_row,
            self.las_status,
            sizing_mode="stretch_width", margin=0,
        )
        self.table_section = pn.Column(self.table_row, self.status, sizing_mode="stretch_width", margin=0,
                                       visible=False)

        # one card for adding wells: LAS files only
        self.add_card = pn.Column(
            pn.FlexBox(
                A.html('<div class="head"><span class="label">Add wells</span></div>', align="center",
                       margin=(0, 18, 0, 0)),
                pn.Row(self.clear_btn, self.download, margin=(0, 0, 0, 0)),  # these act on every well
                flex_wrap="wrap", justify_content="space-between", align_items="center",
                sizing_mode="stretch_width", gap="10px 0px", margin=0),
            self.las_section,
            self.status,
            sizing_mode="stretch_width", styles=A.CARD_STYLES, margin=(8, 0),
        )

        # --- view card ------------------------------------------------------
        self.view = pn.widgets.RadioButtonGroup(options=list(PAGE_VIEWS), value="Map", stylesheets=[A.TOGGLE_CSS],
                                                margin=(0, 18, 0, 0), align="center")
        self.position = pn.widgets.IntInput(start=0, end=1, step=1, value=0, width=130,
                                            stylesheets=[A.INPUT_CSS, A.NO_ARROWS_CSS], align="center", margin=0)
        self.position_range = A.html("", align="center", margin=(0, 18, 0, 10))
        self.tolerance = pn.widgets.IntInput(name="Show wells within (lines)", value=10, start=0, end=500,
                                             width=170, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.well_select = pn.widgets.Select(name="Well", options=[], width=200, stylesheets=[A.INPUT_CSS],
                                             margin=(0, 10, 0, 0), disabled=True)  # enabled once a well is added
        self.goto_btn = pn.widgets.Button(name="Go to well", stylesheets=[A.BTN_BASE_CSS], width=110,
                                          align="end", disabled=True)
        self.remove_btn = pn.widgets.Button(name="Remove", stylesheets=[A.BTN_BASE_CSS], width=90,
                                            align="end", disabled=True)
        self.add_toggle = pn.widgets.Toggle(name="＋ Add wells by clicking the map", value=False, width=250,
                                            stylesheets=[A.BTN_BASE_CSS, ADD_TOGGLE_CSS], align="end",
                                            margin=(0, 10, 0, 0))
        self.new_name = pn.widgets.TextInput(name="New well name", placeholder="e.g. 15/9-A-16", width=190,
                                             stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))

        self.view.param.watch(self.on_view_change, "value")
        self.position.param.watch(self.on_position, "value")
        self.tolerance.param.watch(lambda e: self.update_overlay(), "value")
        self.well_select.param.watch(self.on_select_well, "value")
        self.goto_btn.on_click(self.on_goto)
        self.remove_btn.on_click(self.on_remove)
        self.add_toggle.param.watch(self.on_add_toggle, "value")

        self.view_card = pn.Column(
            A.html('<div class="head"><span class="label">View</span></div>', sizing_mode="stretch_width"),
            # where to look, then: pick a well (go to it / remove it) | add wells by clicking the map
            wrap_row(self.view, self.position, self.position_range, margin=(6, 0, 0, 0)),
            pn.FlexBox(
                pn.Row(self.well_select, self.goto_btn, self.remove_btn, margin=(0, 24, 0, 0)),
                pn.Row(self.new_name, self.add_toggle, self.tolerance, margin=0),
                flex_wrap="wrap", align_items="flex-end", justify_content="space-between",
                sizing_mode="stretch_width", gap=f"{ROW_GAP}px 0px", margin=(ROW_GAP, 0, 0, 0)),
            sizing_mode="stretch_width", styles=A.CARD_STYLES, margin=(8, 0),
        )

        # --- plot + table -----------------------------------------------------
        self.plot_card = pn.Column(sizing_mode="stretch_width", styles=A.CARD_STYLES, margin=(8, 0))
        self.table = A.html("", sizing_mode="stretch_width")
        self.table_card = pn.Column(
            A.html('<div class="head"><span class="label">Well list</span></div>', sizing_mode="stretch_width"),
            self.table, sizing_mode="stretch_width", styles=A.CARD_STYLES, margin=(8, 0),
        )
        self.empty = A.message_box("No seismic loaded", "Load the surveys under Data sources (in the sidebar, "
                                   "then Run) and the wells are drawn over the seismic here", height=300)
        # the colour map controls (a copy of the Workspace ones, kept in step by app.py) sit above the plot
        self.body = pn.Column(self.view_card, A.wells_display_bar, self.plot_card, self.table_card,
                              sizing_mode="stretch_width", margin=0)
        self.layout = pn.Column(self.add_card, self.empty, self.body, sizing_mode="stretch_width",
                                margin=0, visible=False)
        self.body.visible = False

    # ------------------------------------------------------------------ data
    @property
    def wells(self):
        return self.app.ws["wells"]

    def loaded_surveys(self):
        return [s for s in self.app.ws["surveys"] if s["vol"] is not None]

    def unique_name(self, name):
        names = {w["name"] for w in self.wells}
        if name not in names:
            return name
        k = 2
        while f"{name} ({k})" in names:
            k += 1
        return f"{name} ({k})"

    def add_wells(self, wells):
        for w in wells:
            w["name"] = self.unique_name(w["name"])
            w["colour"] = WELL_COLOURS[len(self.wells) % len(WELL_COLOURS)]
            self.wells.append(w)
        self.after_wells_changed(select=wells[-1]["name"] if wells else None)

    def after_wells_changed(self, select=None):
        names = [w["name"] for w in self.wells]
        current = select if select in names else (self.well_select.value if self.well_select.value in names else None)
        self.well_select.options = names
        self.well_select.value = current if current else (names[0] if names else None)
        self.update_overlay()
        self.refresh_table()
        self.refresh_controls()
        self.app.refresh_overlay()  # the hexagon grid on the viewer's time slices sits on a well

    def refresh_controls(self):
        has, chosen = bool(self.wells), self.well_select.value is not None
        self.well_select.disabled = not has  # nothing to choose until a well is added
        self.download.disabled = not has
        self.clear_btn.disabled = not has
        self.goto_btn.disabled = self.remove_btn.disabled = not (has and chosen)

    def make_csv(self):
        geo = self.app.ws["geo"]
        out = io.StringIO()
        w = csv.writer(out, lineterminator="\n")
        w.writerow(["WELL"] + (["X", "Y"] if geo is not None else []) + ["INLINE", "CROSSLINE", "TWT_MS"])
        for well in self.wells:
            xs, ys = lines_to_geo(geo, well["il"], well["xl"]) if geo is not None else (None, None)
            for i in range(len(well["il"])):
                row = [well["name"]]
                if geo is not None:
                    row += [f"{xs[i]:.2f}", f"{ys[i]:.2f}"]
                row += [f"{well['il'][i]:.2f}", f"{well['xl'][i]:.2f}",
                        f"{well['twt'][i]:.1f}" if well["twt"] is not None else ""]
                w.writerow(row)
        return io.BytesIO(out.getvalue().encode("utf-8"))

    # -------------------------------------------------------------- handlers
    def set_status(self, msg, ok=True):
        self.status.object = f'<span class="status {"ok" if ok else "err"}">{msg}</span>' if msg else ""

    def on_upload(self, event):
        names, blobs = self.upload.filename, self.upload.value
        if not names:
            return
        if isinstance(names, str):
            names, blobs = [names], [blobs]
        bad = [n for n in names if Path(n).suffix.lower() not in WELL_EXTS]
        if bad:
            return self.set_status(f'Only .csv and .txt well files can be uploaded (not {escape(", ".join(bad))}).',
                                   ok=False)
        added, notes, outside = [], [], 0
        for name, blob in zip(names, blobs):
            try:
                text = blob.decode("utf-8-sig") if isinstance(blob, bytes) else str(blob)
            except UnicodeDecodeError:
                text = blob.decode("latin-1")
            try:
                wells, file_notes = parse_wells(text, self.app.ws["geo"], float(self.velocity.value), Path(name).stem)
            except ValueError as exc:
                return self.set_status(f"Couldn't read {escape(name)}: {escape(str(exc))}", ok=False)
            notes += file_notes
            for w in wells:
                if self.inside_survey(w):
                    added.append(w)
                else:
                    outside += 1
        self.add_wells(added)
        parts = [f"Added {len(added)} well{'s' if len(added) != 1 else ''}"] if added else ["No wells added"]
        if outside:
            parts.append(f"{outside} outside the survey area (check the coordinates match these surveys)")
        parts += notes
        self.set_status(escape(" · ".join(parts)), ok=bool(added) and not outside)

    def inside_survey(self, well):
        vols = [s["vol"] for s in self.loaded_surveys()]
        if not vols:
            return True
        il = np.concatenate([v["ilines"] for v in vols])
        xl = np.concatenate([v["xlines"] for v in vols])
        return bool(np.any((well["il"] >= il.min()) & (well["il"] <= il.max())
                           & (well["xl"] >= xl.min()) & (well["xl"] <= xl.max())))

    def on_clear(self, event=None):
        self.wells.clear()
        self.set_status("")
        self.after_wells_changed()

    def selected_well(self):
        return next((w for w in self.wells if w["name"] == self.well_select.value), None)

    def on_remove(self, event=None):
        w = self.selected_well()
        if w is not None:
            self.wells.remove(w)
            self.after_wells_changed()

    def on_goto(self, event=None):
        """Show the section through the chosen well (map view -> its inline)."""
        w = self.selected_well()
        if w is None:
            return
        if self.view.value == "Map":
            self._goto = w
            self.view.value = "Inline"  # rebuilds; the position is set in on_view_change
            return
        value = w["il"][0] if self.view.value == "Inline" else w["xl"][0]
        self.position.value = int(round(min(max(value, self.position.start), self.position.end)))

    def on_add_toggle(self, event):
        if event.new and self.view.value != "Map":
            self.view.value = "Map"
        self.set_status("Click on the map to place a well (it is added at the surface location)" if event.new else "")

    def on_tap(self, event):
        if not self.add_toggle.value or self.view.value != "Map":
            return
        name = self.new_name.value.strip() or f"Well {len(self.wells) + 1}"
        self.add_wells([{"name": name, "il": np.array([event.y]), "xl": np.array([event.x]), "twt": None}])
        self.new_name.value = ""
        self.set_status(f"Added {escape(self.wells[-1]['name'])} at inline {event.y:.0f}, crossline {event.x:.0f}")

    # ------------------------------------------------- load a well from LAS
    async def on_browse(self, btn, target, title):
        """Browse button: the computer's file dialog, run outside the app so it can't freeze it."""
        A = self.app
        current = A.resolve_path(target.value) if target.value.strip() else A.APP_DIR
        start = current.parent if current.parent.is_dir() else A.APP_DIR
        btn.disabled = True
        try:
            path = await asyncio.get_running_loop().run_in_executor(None, A.pick_file, title, start)
            if path:
                target.value = path
        except Exception as exc:
            self.set_las_status(f"Couldn't open the file dialog ({escape(str(exc))}); type or paste the path.",
                                ok=False)
        finally:
            btn.disabled = False

    def set_las_status(self, msg, ok=True):
        self.las_status.object = f'<span class="status {"ok" if ok else "err"}">{msg}</span>' if msg else ""

    @staticmethod
    def _number(box, label):
        text = box.value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{label} must be a number") from None

    async def on_load_las(self, event=None):
        A = self.app
        try:
            las = A.resolve_path(self.las_path.value) if self.las_path.value.strip() else None
            cs = A.resolve_path(self.cs_path.value) if self.cs_path.value.strip() else None
            dev = A.resolve_path(self.dev_path.value) if self.dev_path.value.strip() else None
            if las is None or cs is None:
                raise ValueError("choose both the LAS file and the checkshot file")
            for p in (las, cs, dev):
                if p is not None and not p.is_file():
                    raise ValueError(f"file not found: {p}")
            kb = self._number(self.kb_box, "KB elevation")
            loc = {k: self._number(box, label) for k, box, label in (
                ("x", self.loc_x, "Surface X"), ("y", self.loc_y, "Surface Y"),
                ("il", self.loc_il, "Inline"), ("xl", self.loc_xl, "Crossline"))}
        except ValueError as exc:
            return self.set_las_status(escape(str(exc)), ok=False)
        settings = dict(depth_col=self.cs_depth_col.value - 1, time_col=self.cs_time_col.value - 1,
                        depth_type=self.cs_depth_type.value, depth_sign=-1 if self.cs_negative.value else 1,
                        time_type=self.cs_time_type.value, time_unit=self.cs_time_unit.value,
                        shift_ms=float(self.cs_shift.value or 0))

        A.set_label(self.load_las_btn, "Loading…")
        self.load_las_btn.disabled = True
        self.set_las_status("Reading the logs" + (" and asking Gemma…" if self.use_gemma.value else "…"))
        doc = pn.state.curdoc
        use_gemma = self.use_gemma.value
        try:  # the slow part (reading, converting, asking Gemma) runs in the background
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: welllogs.process_well(las, cs, dev, kb=kb, checkshot=settings, use_gemma=use_gemma))
            error = None
        except Exception as exc:
            result, error = None, exc
        # the plots may only be changed from the page's own update loop, so finish there
        finish = lambda: self.finish_load(result, error, loc, dev is not None)  # noqa: E731
        if doc is not None:
            doc.add_next_tick_callback(finish)
        else:  # no browser session (e.g. a test script)
            finish()

    def finish_load(self, result, error, loc, has_deviation):
        A = self.app
        A.set_label(self.load_las_btn, "Load well")
        self.load_las_btn.disabled = False
        try:
            if error is not None:
                raise error
            well = self.well_from_result(result, loc, has_deviation)
        except Exception as exc:  # show what went wrong instead of failing silently
            self.set_las_status(f"Couldn't load the well: {escape(str(exc))}", ok=False)
            return

        well.pop("_notes", None)
        self.add_wells([well])
        self.set_las_status("Loaded")  # the reservoir shows on the seismic and in the well list

    def well_from_result(self, r, loc, has_deviation):  # noqa: C901
        """A well for this page (survey line numbers + TWT) from welllogs.process_well's result."""
        geo, c, d = self.app.ws["geo"], r["curves"], r["logs"]
        notes = []
        # wellhead: inline / crossline typed in, else X / Y typed in, else X / Y from the LAS header
        if loc["il"] is not None and loc["xl"] is not None:
            il0, xl0 = loc["il"], loc["xl"]
            x0, y0 = lines_to_geo(geo, [il0], [xl0]) if geo is not None else (None, None)
            x0, y0 = (float(x0[0]), float(y0[0])) if geo is not None else (None, None)
        else:
            x0 = loc["x"] if loc["x"] is not None else r["header"]["x"]
            y0 = loc["y"] if loc["y"] is not None else r["header"]["y"]
            if x0 is None or y0 is None:
                raise ValueError("the LAS header has no surface X / Y: type the well's location "
                                 "(X and Y, or inline and crossline)")
            if geo is None:
                raise ValueError("these surveys have no map coordinates, so give the well's inline and crossline")
            il0, xl0 = (float(v[0]) for v in geo_to_lines(geo, [x0], [y0]))

        # the path, one point every PATH_STEP_MS, starting at the surface
        ok = d.TWT_MS.notna()
        p = d[ok].groupby((d.TWT_MS[ok] // PATH_STEP_MS)).first()
        twt = np.r_[0.0, p.TWT_MS.values.clip(min=0)]
        north, east = np.r_[0.0, p.NORTH.values], np.r_[0.0, p.EAST.values]
        if geo is not None and x0 is not None:
            il, xl = geo_to_lines(geo, x0 + east, y0 + north)
        else:
            il, xl = np.full(len(twt), il0), np.full(len(twt), xl0)
            if has_deviation:
                notes.append("no map coordinates, so the deviation survey couldn't be used: drawn vertical")

        # curves for the log tracks: 1 ms averages
        cols = {"GR": c["gr"], "NPHI": c["nphi"], "RHOB": c["rhob"], "RT": c["rt"], "MD": c["depth"],
                "GR_CUTOFF": "GR_CUTOFF", "NPHI_CUTOFF": "NPHI_CUTOFF", "RHOB_CUTOFF": "RHOB_CUTOFF",
                "RT_CUTOFF": "RT_CUTOFF"}
        g = d[ok].groupby((d.TWT_MS[ok] // LOG_STEP_MS))
        logs = {k: g[v].mean().values for k, v in cols.items()}
        logs["TWT"] = g["TWT_MS"].mean().values
        logs["PAY"] = g["PAY"].mean().values
        cand = r["candidates"]
        return {"name": r["name"], "il": np.asarray(il, float), "xl": np.asarray(xl, float), "twt": twt,
                "logs": logs, "reservoir": r["pick"], "cutoffs": r["cutoffs"],
                "candidates": cand.to_dict("records") if len(cand) else [], "_notes": notes}

    def on_select_well(self, event):
        self.refresh_controls()
        if self.view.value != "Map" and self.plot is not None:
            self.build_plot()  # swap the log tracks for the chosen well

    def on_view_change(self, event):
        self.setup_position()
        goto = getattr(self, "_goto", None)
        if goto is not None:
            self._goto = None
            self.position.param.update(value=int(round(min(max(goto["il"][0], self.position.start),
                                                           self.position.end))))
        if event.new != "Map" and self.add_toggle.value:
            self.add_toggle.value = False
        self.build_plot()

    def on_position(self, event):
        if self.plot is None:
            return
        sl = self.current_slice()
        if sl is None:
            return
        self.plot["img_src"].data = self.app.image_data(sl)
        self.plot["sub"].object = self.subtitle(sl)
        self.update_overlay()

    # --------------------------------------------------------------- drawing
    def on_data_loaded(self):
        """Called by app.py after Run loads the surveys."""
        loaded = bool(self.loaded_surveys())
        self.empty.visible = not loaded
        self.body.visible = loaded
        if not loaded:
            return
        self.setup_position()
        self.build_plot()
        self.refresh_table()
        self.refresh_controls()

    def survey_obj(self):
        """The survey the wells are drawn over: always the baseline (the monitor only if no baseline is loaded)."""
        loaded = self.loaded_surveys()
        return next((s for s in loaded if s["role"] == SURVEY_ROLE), loaded[0] if loaded else None)

    def setup_position(self):
        A = self.app
        view = PAGE_VIEWS[self.view.value]
        vals = A.slider_values(A.ws["surveys"], view)
        step = max(1, int(round(A.spacing(vals)))) if len(vals) > 1 else 1
        start, end = int(np.floor(vals[0])), int(np.ceil(vals[-1]))
        end = max(end, start + step)
        value = int(round(vals[len(vals) // 2]))
        self.position.param.update(start=min(self.position.start, start), end=max(self.position.end, end))
        self.position.param.update(value=value, step=step)
        self.position.param.update(start=start, end=end)
        unit = " ms" if view == "Time slice" else ""
        self.position_range.object = f'<span class="meta">{start} to {end}{unit}, step {step}</span>'

    def current_slice(self):
        s = self.survey_obj()
        if s is None:
            return None
        return self.app.get_slice(s["vol"], PAGE_VIEWS[self.view.value], self.position.value)

    def subtitle(self, sl):
        s = self.survey_obj()
        return f'<div class="sub">{escape(s["name"])} · {self.app.position_text(PAGE_VIEWS[self.view.value], sl["pos"])}</div>'

    def build_plot(self):
        A = self.app
        sl = self.current_slice()
        if sl is None:
            return
        view = PAGE_VIEWS[self.view.value]
        x_label, y_label, y_down = A.VIEW_AXES[view]
        x0, x1 = A.extent(sl["xs"])
        y0, y1 = A.extent(sl["ys"])
        xr = A.view_range(x0, x1)
        yr = A.view_range(y0, y1, flip=y_down)
        height = 640 if view == "Time slice" else 560
        p = A.make_figure(xr, yr, x_label, y_label, height=height)
        if p in A.ws["figs"]:
            A.ws["figs"].remove(p)  # the workspace rebuild clears that list; keep this one separately
        A.ws["wfigs"] = [p]

        mapper = LinearColorMapper(palette=A.cmap_palette(A.cmap_select.value), low=-A.AMP_MAX, high=A.AMP_MAX)
        A.extra_mappers[:] = [mapper]
        img_src = ColumnDataSource(A.image_data(sl))
        img = p.image(image="image", x="x", y="y", dw="dw", dh="dh", source=img_src, color_mapper=mapper)
        for hover in p.select(type=HoverTool):
            hover.renderers = [img]

        path_src = ColumnDataSource({"xs": [], "ys": [], "name": [], "colour": []})
        stub_src = ColumnDataSource({"xs": [], "ys": [], "name": [], "colour": []})
        head_src = ColumnDataSource({"x": [], "y": [], "name": [], "colour": [], "info": []})
        hit_src = ColumnDataSource({"x": [], "y": [], "name": [], "colour": [], "info": []})
        res_src = ColumnDataSource({"xs": [], "ys": [], "name": [], "info": []})     # reservoir on sections
        res_pt_src = ColumnDataSource({"x": [], "y": [], "name": [], "info": []})    # reservoir on the map
        res_lbl_src = ColumnDataSource({"x": [], "y": [], "text": []})               # "top 849 ms" labels
        # dark casing under each path so it reads on both bright and dark seismic
        p.multi_line("xs", "ys", source=path_src, line_color="#111111", line_width=6, line_alpha=0.55)
        # the reservoir as a thick band along the path, under the path's own colour
        res = p.multi_line("xs", "ys", source=res_src, line_color=RESERVOIR_COLOUR, line_width=12, line_alpha=0.9)
        res_pts = p.scatter("x", "y", source=res_pt_src, marker="star", size=20, fill_color=RESERVOIR_COLOUR,
                            line_color="#111111", line_width=1)
        p.add_layout(LabelSet(x="x", y="y", text="text", source=res_lbl_src, x_offset=12, y_offset=0,
                              text_baseline="middle", text_font_size="10.5px", text_color="#0b1418",
                              background_fill_color=RESERVOIR_COLOUR, background_fill_alpha=0.85))
        paths = p.multi_line("xs", "ys", source=path_src, line_color="colour", line_width=3)
        stubs = p.multi_line("xs", "ys", source=stub_src, line_color="colour", line_width=2.5, line_dash="dashed")
        heads = p.scatter("x", "y", source=head_src, size=11, fill_color="colour", line_color="white",
                          line_width=2)
        hits = p.scatter("x", "y", source=hit_src, marker="diamond", size=16, fill_color="colour",
                         line_color="white", line_width=2)
        p.add_layout(LabelSet(x="x", y="y", text="name", source=head_src, x_offset=8, y_offset=6,
                              text_font_size="11px", text_color="#0b1418", text_font_style="bold",
                              background_fill_color="white", background_fill_alpha=0.8))
        p.add_tools(HoverTool(renderers=[paths, stubs, heads, hits, res, res_pts],
                              tooltips=[("Well", "@name"), ("", "@info")],
                              visible=False))  # tooltips only: no extra toolbar button
        p.on_event(Tap, self.on_tap)

        role = self.survey_obj()["role"]
        head = A.html('<div class="head"><h2 class="title">Wells on the seismic</h2>'
                      f'<span class="survey-tag{" monitor" if role == "Monitor" else ""}">{role}</span></div>',
                      sizing_mode="stretch_width")
        sub = A.html(self.subtitle(sl), sizing_mode="stretch_width")
        credit = A.credit_html(A.ws["credit"]) if A.ws.get("credit") else None
        # on sections, the chosen LAS well's logs sit beside the seismic on the same time axis
        well = self.selected_well()
        if view != "Time slice" and well is not None and "logs" in well:
            tracks = self.log_tracks(well, yr, height)
            for t in tracks:  # same toolbar settings as the seismic, so the merged toolbar has one value
                t.toolbar.active_drag, t.toolbar.active_scroll = p.toolbar.active_drag, p.toolbar.active_scroll
            A.ws["wfigs"] += tracks
            plot_obj = gridplot([[p, *tracks]], toolbar_location="above", merge_tools=True,
                                sizing_mode="stretch_width")
            # the seismic takes the spare width; each log track keeps its own width
            plot_obj.cols = ["1fr"] + [f"{t.width}px" for t in tracks]
            plot_obj.toolbar.stylesheets = list(p.toolbar.stylesheets)  # same print-hiding rule
        else:
            plot_obj = p
        self.plot_card.objects = [o for o in (head, sub, pn.pane.Bokeh(plot_obj, sizing_mode="stretch_width",
                                                                       margin=(10, 0, 0, 0)), credit) if o is not None]
        self.plot = {"img_src": img_src, "path_src": path_src, "stub_src": stub_src, "head_src": head_src,
                     "hit_src": hit_src, "res_src": res_src, "res_pt_src": res_pt_src, "res_lbl_src": res_lbl_src,
                     "sub": sub, "y_range": (y0, y1)}
        A.apply_colours()
        self.update_overlay()

    def log_tracks(self, w, y_range, height):
        """GR | density + neutron | resistivity against TWT, sharing the seismic's time axis.
        The picked reservoir is shaded yellow, other candidate intervals green."""
        A, L = self.app, w["logs"]
        t = L["TWT"]

        def track(label, x_range, x_type="linear", width=150):
            f = figure(width=width, height=height, y_range=y_range, x_range=x_range, x_axis_type=x_type,
                       tools="", toolbar_location=None, x_axis_label=label, min_border_left=6, min_border_right=6)
            A.style_figure(f)
            f.yaxis.visible = False
            for c in w["candidates"]:
                if w.get("reservoir") and c["id"] in (w["reservoir"].get("candidate_ids") or []):
                    continue
                f.add_layout(BoxAnnotation(bottom=c["top_twt_ms"], top=c["base_twt_ms"], fill_color=CANDIDATE_COLOUR,
                                           fill_alpha=0.15, line_alpha=0))
            pick = w.get("reservoir")
            if pick:
                f.add_layout(BoxAnnotation(bottom=pick["top_twt_ms"], top=pick["base_twt_ms"],
                                           fill_color=RESERVOIR_COLOUR, fill_alpha=0.3, line_color=RESERVOIR_COLOUR,
                                           line_width=1.5))
            return f

        gr = track("Gamma Ray (gAPI)", Range1d(0, 150))
        gr.xaxis.ticker = [0, 50, 100, 150]
        gr.line(L["GR"], t, color="#22c55e", line_width=1.3)
        gr.line(L["GR_CUTOFF"], t, color="#22c55e", line_width=1, line_dash="dashed", line_alpha=0.7)

        # density on its own scale; neutron drawn on the same track by mapping its usual
        # reversed scale (0.45 -> -0.15) onto the density scale (1.95 -> 2.95)
        nd = track("Density g/cc (red) · Neutron (blue)", Range1d(1.95, 2.95), width=190)
        nd.xaxis.ticker = [2.0, 2.25, 2.5, 2.75]
        nd.line(L["RHOB"], t, color="#ef4444", line_width=1.3)
        nd.line(1.95 + (0.45 - np.asarray(L["NPHI"])) / 0.60, t, color="#3b82f6", line_width=1.3)

        rt = track("Resistivity (ohm·m)", Range1d(0.2, 200), x_type="log")
        rt.line(L["RT"], t, color="#8b9aa6", line_width=1.3)  # grey reads on both light and dark themes
        rt.line(L["RT_CUTOFF"], t, color="#f59e0b", line_width=1, line_dash="dashed")
        return [gr, nd, rt]

    def update_overlay(self):
        if self.plot is None:
            return
        view, pos, tol = PAGE_VIEWS[self.view.value], self.position.value, self.tolerance.value
        y0, y1 = self.plot["y_range"]
        paths = {"xs": [], "ys": [], "name": [], "colour": []}  # well paths (solid)
        stubs = {"xs": [], "ys": [], "name": [], "colour": []}  # location-only wells on sections (dashed)
        heads = {"x": [], "y": [], "name": [], "colour": [], "info": []}
        hits = {"x": [], "y": [], "name": [], "colour": [], "info": []}
        res = {"xs": [], "ys": [], "name": [], "info": []}
        res_pts = {"x": [], "y": [], "name": [], "info": []}
        res_lbls = {"x": [], "y": [], "text": []}

        def add(d, **kv):
            for k, v in kv.items():
                d[k].append(v)

        for w in self.wells:
            il, xl, twt, c, n = w["il"], w["xl"], w["twt"], w["colour"], w["name"]
            where = f"inline {il[0]:.0f}, crossline {xl[0]:.0f}"
            pick = w.get("reservoir") if twt is not None and len(twt) > 1 else None
            res_info = (f'reservoir {pick["top_twt_ms"]:.0f}–{pick["base_twt_ms"]:.0f} ms TWT '
                        f'({pick.get("source", "")})') if pick else ""
            if view == "Time slice":  # map: x = crossline, y = inline
                if len(il) > 1:
                    add(paths, xs=xl.tolist(), ys=il.tolist(), name=n, colour=c)
                add(heads, x=float(xl[0]), y=float(il[0]), name=n, colour=c, info=f"top at {where}")
                if pick:  # a star where the well is in the middle of its reservoir
                    mid = (pick["top_twt_ms"] + pick["base_twt_ms"]) / 2
                    add(res_pts, x=float(np.interp(mid, twt, xl)), y=float(np.interp(mid, twt, il)), name=n,
                        info=res_info)
                if twt is not None and twt[0] <= pos <= twt[-1]:
                    hx, hy = float(np.interp(pos, twt, xl)), float(np.interp(pos, twt, il))
                    add(hits, x=hx, y=hy, name=n, colour=c,
                        info=f"crosses {pos} ms at inline {hy:.0f}, crossline {hx:.0f}")
                continue
            # sections: Inline view -> x = crossline, distance in inlines; Crossline view -> the reverse
            along, across = (xl, il) if view == "Inline" else (il, xl)
            dist = np.abs(across - pos)
            if dist.min() > tol:
                continue
            off = f"{dist.min():.0f} line{'s' if round(dist.min()) != 1 else ''} from this section"
            if twt is not None and len(twt) > 1:
                add(paths, xs=along.tolist(), ys=twt.tolist(), name=n, colour=c)
                add(heads, x=float(along[0]), y=float(twt[0]), name=n, colour=c, info=off)
                if pick:  # the reservoir as a band along the path, with its top and base labelled
                    tt = np.linspace(pick["top_twt_ms"], pick["base_twt_ms"], 24)
                    seg = np.interp(tt, twt, along)
                    add(res, xs=seg.tolist(), ys=tt.tolist(), name=n, info=res_info)
                    add(res_lbls, x=float(seg[0]), y=float(tt[0]), text=f'top {pick["top_twt_ms"]:.0f} ms')
                    add(res_lbls, x=float(seg[-1]), y=float(tt[-1]), text=f'base {pick["base_twt_ms"]:.0f} ms')
            else:
                top = float(twt[0]) if twt is not None else min(y0, y1)
                add(stubs, xs=[float(along[0])] * 2, ys=[top, max(y0, y1)], name=n, colour=c)
                add(heads, x=float(along[0]), y=top, name=n, colour=c,
                    info=off + ("" if twt is not None else " · no time data: dashed line shows the location"))
        self.plot["path_src"].data = paths
        self.plot["stub_src"].data = stubs
        self.plot["head_src"].data = heads
        self.plot["hit_src"].data = hits
        self.plot["res_src"].data = res
        self.plot["res_pt_src"].data = res_pts
        self.plot["res_lbl_src"].data = res_lbls

    def refresh_table(self):
        if not self.wells:
            self.table.object = ('<p class="note" style="margin-top:8px">No wells yet: upload a file or turn on '
                                 '“Add wells by clicking the map”.</p>')
            return
        geo = self.app.ws["geo"]
        rows = ""
        for w in self.wells:
            loc = ""
            if geo is not None:
                x, y = lines_to_geo(geo, w["il"][:1], w["xl"][:1])
                loc = f"{x[0]:,.1f} E · {y[0]:,.1f} N"
            twt = f'{w["twt"][0]:.0f}–{w["twt"][-1]:.0f} ms' if w["twt"] is not None else "—"
            pick = w.get("reservoir")
            res = (f'{pick["top_twt_ms"]:.0f}–{pick["base_twt_ms"]:.0f} ms · {escape(pick.get("source", ""))}'
                   if pick else ("none found" if "logs" in w else "—"))
            rows += (f'<tr><td><span class="swatch" style="background:{w["colour"]}"></span>{escape(w["name"])}</td>'
                     f'<td>{w["il"][0]:.0f}</td><td>{w["xl"][0]:.0f}</td><td>{loc or "—"}</td>'
                     f'<td>{twt}</td><td>{res}</td><td>{len(w["il"])}</td></tr>')
        self.table.object = (
            f'<style>{TABLE_CSS}</style><table class="wells"><thead><tr><th>Well</th><th>Inline</th>'
            '<th>Crossline</th><th>Surface location</th><th>Time range</th><th>Reservoir (TWT)</th><th>Points</th>'
            f'</tr></thead><tbody>{rows}</tbody></table>')