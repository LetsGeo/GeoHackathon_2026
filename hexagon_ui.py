"""
The Hexagon analysis card on the Analysis page (the engine is hexagon_analysis.py).

Run puts the 19 hexagons round the well chosen on the Wells page (at its reservoir top),
measures the traces inside each in the reservoir window, asks Gemma to score their sections
and estimates the hydrocarbon volume. The results fill the report's three AI sections
("3D polygons", "Opportunity score", "Hydrocarbon volume"), so they appear in the summary,
its translations and the PDF; the card shows the map, the table and the best hexagon's
sections, and downloads the hexagon table (sealing_faults.py's format) and the details.

The app passes itself in (`app`), the same way the Wells page does.
"""
import asyncio
import base64
from html import escape

import numpy as np
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, Range1d

import hexagon_analysis as ha

OPP_PALETTE = ["#3b0f70", "#8c2981", "#de4968", "#fe9f6d", "#fcfdbf"]  # low -> high opportunity
DIFF_STOPS = [(0.0, "#2f6fd0"), (0.5, "#f7f7f7"), (1.0, "#d7301f")]


def _palette(stops, n=256):
    def rgb(h):
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))
    out = []
    for k in range(n):
        t = k / (n - 1)
        for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
            if p0 <= t <= p1:
                f = 0 if p1 == p0 else (t - p0) / (p1 - p0)
                a, b = rgb(c0), rgb(c1)
                out.append("#%02x%02x%02x" % tuple(round(a[i] + (b[i] - a[i]) * f) for i in range(3)))
                break
    return out


def _f(v, d=0, unit=""):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}{unit}"


class HexagonCard:
    def __init__(self, app):
        self.app = app
        A = app
        A.ws.setdefault("hexagons", None)
        A.ws.setdefault("hfigs", [])
        self.run_btn = pn.widgets.Button(name="Run hexagon analysis", width=190,
                                         stylesheets=[A.BTN_BASE_CSS, A.PRIMARY_BTN_CSS])
        self.sections = pn.widgets.IntInput(name="Sections per hexagon", value=ha.SETTINGS["sections"], start=1,
                                            end=5, width=150, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.sw = pn.widgets.FloatInput(name="Water saturation Sw", value=ha.SETTINGS["sw"], start=0, end=1,
                                        step=0.05, width=150, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.bo = pn.widgets.FloatInput(name="Formation volume factor Bo", value=ha.SETTINGS["bo"], start=0.5,
                                        end=5, step=0.05, width=170, stylesheets=[A.INPUT_CSS], margin=(0, 10, 0, 0))
        self.table_dl = pn.widgets.FileDownload(callback=self.table_file, filename="hexagon_table.csv",
                                                label="Hexagon table", button_type="default", width=140,
                                                height=A.DOWNLOAD_BTN_HEIGHT, disabled=True, align="end",
                                                stylesheets=[A.BTN_BASE_CSS, A.DOWNLOAD_BTN_CSS], margin=(0, 10, 0, 0))
        self.details_dl = pn.widgets.FileDownload(callback=self.details_file, filename="hexagon_details.csv",
                                                  label="Details", button_type="default", width=100,
                                                  height=A.DOWNLOAD_BTN_HEIGHT, disabled=True, align="end",
                                                  stylesheets=[A.BTN_BASE_CSS, A.DOWNLOAD_BTN_CSS])
        self.intro = ('<span class="status">Puts the 19 hexagons round the well chosen on the Wells page (at its '
                      'reservoir top), measures the traces inside each in the reservoir window, asks Gemma to score '
                      'their sections and estimates the volume. Fills the report\'s three AI sections below.</span>')
        self.status = A.html(self.intro, sizing_mode="stretch_width", margin=(8, 0, 0, 0))
        self.body = pn.Column(sizing_mode="stretch_width", margin=0)
        self.run_btn.on_click(self.on_run)
        self.layout = pn.Column(
            A.html('<div class="head"><h2 class="title">Hexagon analysis</h2>'
                   '<span class="meta">3D polygons · opportunity score · hydrocarbon volume</span></div>',
                   sizing_mode="stretch_width"),
            pn.Row(self.run_btn, self.sections, self.sw, self.bo, self.table_dl, self.details_dl,
                   sizing_mode="stretch_width", margin=(8, 0, 0, 0), styles={"flex-wrap": "wrap", "gap": "6px"}),
            self.status,
            self.body,
            sizing_mode="stretch_width", styles=A.CARD_STYLES, margin=(8, 0),
        )

    # ------------------------------------------------------------------ running
    def reset(self, message=None):
        """New surveys: the old results no longer apply."""
        self.app.ws["hexagons"] = None
        self.app.ws["hfigs"] = []
        self.body.objects = []
        self.table_dl.disabled = self.details_dl.disabled = True
        self.status.object = message or self.intro

    async def on_run(self, event=None):
        A = self.app
        ws = A.ws
        surveys = ws["surveys"]
        if len(surveys) != 2 or any(s_["vol"] is None for s_ in surveys):
            self.status.object = '<span class="status err">Load both surveys first (Data sources).</span>'
            return
        well = A.hexagon_well()
        if well is None:
            self.status.object = ('<span class="status err">Load a well on the Wells page first: the hexagons go '
                                  'round it, and its reservoir pick sets the time window.</span>')
            return
        base, mon = surveys
        settings = {"sections": int(self.sections.value or 1), "sw": float(self.sw.value), "bo": float(self.bo.value)}
        doc = pn.state.curdoc
        A.set_label(self.run_btn, "Running…")
        self.run_btn.disabled = True
        self.status.object = '<span class="status">Starting…</span>'

        def say(text):  # progress from the worker thread, shown on the page's own update loop
            msg = f'<span class="status">{escape(text)}</span>'
            if doc is not None:
                doc.add_next_tick_callback(lambda: setattr(self.status, "object", msg))

        result, error = None, None
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: ha.run(base["vol"], mon["vol"], well, list(ws.get("horizons") or []), ws.get("geo"),
                                     settings, say))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            error = exc

        def finish():
            try:
                if error is not None:
                    self.status.object = f'<span class="status err">Hexagon analysis failed: {escape(str(error))}</span>'
                    return
                ws["hexagons"] = result
                self.show(result)
                self.table_dl.disabled = self.details_dl.disabled = False
                # the report: run the baseline-vs-monitor analysis too if it hasn't been, then redraw it
                if ws.get("analysis") is None:
                    A.on_run()
                else:
                    A.render_analysis(force=True)
                A.refresh_overview()
            finally:
                A.set_label(self.run_btn, "Run hexagon analysis")
                self.run_btn.disabled = False

        if doc is not None:
            doc.add_next_tick_callback(finish)
        else:
            finish()

    # ------------------------------------------------------------------ showing
    def show(self, res):
        A = self.app
        rows = [r for r in res["rows"] if r["n_traces"]]
        opp = [r for r in rows if np.isfinite(r["opportunity_pct"])]
        best = max(opp, key=lambda r: r["opportunity_pct"]) if opp else None
        likely = [r for r in rows if r.get("verdict") == "likely"]
        v = res["volume"]
        t0, t1 = res["window"]
        tiles = "".join([
            A.stat_card("Hexagons", f"{len(rows)}", f"of 19 round {escape(res['well_name'])}", highlight=True),
            A.stat_card("Window", f"{t0:,.0f}–{t1:,.0f}", escape(res["window_source"]), unit="ms"),
            A.stat_card("Best opportunity", f"{best['opportunity_pct']:.0f}" if best else "–",
                        f"hexagon {best['hexagon']}" if best else "Gemma not reached", unit="%" if best else ""),
            A.stat_card("Likely sealing faults", f"{len(likely)}",
                        ("hexagon " + ", ".join(str(r["hexagon"]) for r in likely)) if likely else
                        f"check: {escape((res.get('sealing') or {}).get('source', 'not run'))}"),
            A.stat_card("In place", _f(v["stoiip_mmbbl"], 1), "risked " + _f(v["risked_mmbbl"], 1) + " MMbbl"
                        if v["risked"] else "not risked (no opportunity score)", unit="MMbbl"),
        ])
        notes = "".join(f'<div class="status err">{escape(n)}</div>' for n in res["notes"])
        source = (f'<span class="status ok">Done</span><span class="status"> · centre inline '
                  f'{res["centre"][0]:,.0f} / crossline {res["centre"][1]:,.0f} · sections scored by '
                  f'{"Gemma" if res["gemma_used"] else "nobody (Gemma not reached)"} · report text '
                  f'{escape(res["texts_source"])}</span>')
        self.status.object = source + notes
        ws = A.ws
        ws["hfigs"] = []
        self.body.objects = [
            A.html(f'<div class="stats" style="margin-top:12px">{tiles}</div>', sizing_mode="stretch_width"),
            A.plot_with_caption(self.map_plot(res), "Hexagons round the well over RMS(base) − RMS(mon) in the "
                                                    "window (fill = opportunity score, or sealing fault "
                                                    "probability without Gemma)"),
            A.html(self.table_html(res), sizing_mode="stretch_width", margin=(14, 0, 0, 0)),
            self.pictures_pane(res, best or (max(rows, key=lambda r: r["sealing_fault_probability"]) if rows else None)),
        ]

    def map_plot(self, res):
        A = self.app
        m = res["map"]
        il, xl, d = m["ilines"], m["xlines"], m["drms"]
        (x0, x1), (y0, y1) = A.extent(xl), A.extent(il)
        lim = float(np.nanpercentile(np.abs(d), 99)) if np.isfinite(d).any() else 1.0
        p = A.make_figure(Range1d(x0, x1, bounds=(x0, x1)), Range1d(y0, y1, bounds=(y0, y1)), "Crossline", "Inline",
                          height=520, value_label="RMS(base) − RMS(mon)")
        A.ws["figs"].remove(p)
        A.ws["hfigs"].append(p)
        mapper = LinearColorMapper(palette=_palette(DIFF_STOPS)[::-1], low=-lim or -1, high=lim or 1,
                                   nan_color="rgba(0,0,0,0)")
        p.image(image=[np.ascontiguousarray(d, np.float32)], x=x0, y=y0, dw=x1 - x0, dh=y1 - y0, color_mapper=mapper)
        A.add_colorbar(p, mapper, "RMS(base) − RMS(mon) (red = monitor stronger)")
        use_opp = res["gemma_used"] and any(np.isfinite(r["opportunity_pct"]) for r in res["rows"])
        pal = _palette([(i / (len(OPP_PALETTE) - 1), c) for i, c in enumerate(OPP_PALETTE)])
        rows = [r for r in res["rows"] if r["n_traces"]]
        cols = {k: [] for k in ("xs", "ys", "fill", "id", "il", "xl", "rms", "seal", "trace", "verdict", "opp",
                                "res", "vol", "x", "y")}
        for r in rows:
            val = r["opportunity_pct"] if use_opp else r["sealing_fault_probability"]
            cols["xs"].append(r["corners_xl"])
            cols["ys"].append(r["corners_il"])
            cols["fill"].append(pal[int(np.clip(val / 100, 0, 1) * 255)] if np.isfinite(val) else "#888888")
            cols["id"].append(str(r["hexagon"]))
            cols["il"].append(r["centre_il"])
            cols["xl"].append(r["centre_xl"])
            cols["rms"].append(r["rms_diff"])
            cols["seal"].append(r["sealing_fault_probability"])
            cols["trace"].append(r["trace_pct"])
            cols["verdict"].append(r.get("verdict") or "–")
            cols["opp"].append(r["opportunity_pct"])
            cols["res"].append(r["reservoir_pct"])
            cols["vol"].append(r["stoiip_m3"] * ha.M3_TO_BBL / 1e6 if np.isfinite(r["stoiip_m3"]) else float("nan"))
            cols["x"].append(float(np.mean(r["corners_xl"])))
            cols["y"].append(float(np.mean(r["corners_il"])))
        src = ColumnDataSource(cols)
        patches = p.patches("xs", "ys", source=src, fill_color="fill", fill_alpha=0.35, line_color="#ffffff",
                            line_width=1.2)
        p.text("x", "y", "id", source=src, text_align="center", text_baseline="middle", text_color="#ffffff",
               text_font_size="12px", text_font_style="bold")
        p.add_tools(HoverTool(renderers=[patches], tooltips=[
            ("Hexagon", "@id (inline @il / crossline @xl)"), ("RMS(base) − RMS(mon)", "@rms{0.000}"),
            ("Sealing fault", "@seal{0}% (trace @trace{0.0}%) · @verdict"), ("Opportunity", "@opp{0}%"),
            ("Reservoir (Gemma)", "@res{0}%"), ("In place", "@vol{0.0} MMbbl")]))
        A.style_figure(p)
        return p

    def table_html(self, res):
        A = self.app
        head = ["Hexagon", "Ring", "Centre IL", "Centre CL", "Sealing fault", "Trace", "Evidence", "Verdict",
                "RMS(base) − RMS(mon)", "Reservoir", "Opportunity", "In place (MMbbl)", "Risked (MMbbl)"]
        body = []
        for r in sorted([r for r in res["rows"] if r["n_traces"]],
                        key=lambda r: (-(r["opportunity_pct"] if np.isfinite(r["opportunity_pct"]) else -1),
                                       -r["sealing_fault_probability"])):
            body.append("<tr>" + "".join(f"<td>{c}</td>" for c in [
                r["hexagon"], r["ring"], f'{r["centre_il"]:,}', f'{r["centre_xl"]:,}',
                _f(r["sealing_fault_probability"], 0, "%"), _f(r["trace_pct"], 1, "%"), escape(r["seal_evidence"]),
                escape(r.get("verdict") or "–"), _f(r["rms_diff"], 3), _f(r["reservoir_pct"], 0, "%"),
                _f(r["opportunity_pct"], 0, "%"), _f(r["stoiip_m3"] * ha.M3_TO_BBL / 1e6, 1),
                _f(r["risked_m3"] * ha.M3_TO_BBL / 1e6, 1)]) + "</tr>")
        p = res["props"]
        foot = (f'<p class="meta" style="margin-top:8px">Volume: area × thickness × N/G {p["ntg"]:.2f} × porosity '
                f'{p["phi"]:.2f} ({escape(p["props_source"])}) × (1 − Sw {res["settings"]["sw"]:.2f}) ÷ Bo '
                f'{res["settings"]["bo"]:.2f}; thickness {_f(p["thickness_m"], 0)} m at the well'
                + (", per hexagon from the horizons" if any(r.get("thickness_ms") for r in res["rows"]) else "")
                + '. Risked = in place × opportunity.</p>')
        return (f'<style>{A.TABLE_CSS}</style><table class="wells"><thead><tr>'
                + "".join(f"<th>{h}</th>" for h in head) + f'</tr></thead><tbody>{"".join(body)}</tbody></table>'
                + foot)

    def pictures_pane(self, res, row):
        A = self.app
        if row is None or not res["pictures"].get(row["hexagon"]):
            return A.html("", sizing_mode="stretch_width")
        kind, line, b, d = res["pictures"][row["hexagon"]][0]

        def img(png, alt):
            return (f'<img alt="{alt}" style="width:100%;border-radius:10px;border:1px solid var(--border);'
                    f'background:#fff" src="data:image/png;base64,{base64.b64encode(png).decode()}">')
        why = escape(row.get("opportunity_reason") or row.get("verdict_reason") or "")
        return A.html(f'<div class="plot-caption" style="margin-top:14px">Hexagon {row["hexagon"]}: '
                      f'{"inline" if kind == "IL" else "crossline"} {line}, baseline and 4D difference '
                      f'(what Gemma saw){" — " + why if why else ""}</div>'
                      f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px">'
                      f'{img(b, "baseline")}{img(d, "difference")}</div>', sizing_mode="stretch_width")

    # ------------------------------------------------------------------ files
    def table_file(self):
        import io
        return io.BytesIO(ha.table_csv(self.app.ws["hexagons"]).encode("utf-8"))

    def details_file(self):
        import io
        return io.BytesIO(ha.details_csv(self.app.ws["hexagons"]).encode("utf-8"))

    def overview_row(self):
        """(name, detail, pill, pill_class, info) for the Overview status table."""
        res = self.app.ws.get("hexagons")
        if not res:
            return "Hexagon analysis", "3D polygons, opportunity, volume (Analysis page)", "Not run", "warn", ""
        rows = [r for r in res["rows"] if r["n_traces"]]
        opp = [r for r in rows if np.isfinite(r["opportunity_pct"])]
        info = (f'{len(rows)} hexagons round {escape(res["well_name"])}'
                + (f' · best opportunity {max(r["opportunity_pct"] for r in opp):.0f}%' if opp else "")
                + (f' · {res["volume"]["stoiip_mmbbl"]:.1f} MMbbl in place' if res["props"]["thickness_m"] else ""))
        return "Hexagon analysis", "3D polygons, opportunity, volume (Analysis page)", "Done", "ok", info

    def pdf_pages(self, pdf, footer, first_page, stamp):
        """Map + table pages for the PDF (after the summary pages). Returns the next page number."""
        res = self.app.ws.get("hexagons")
        if not res:
            return first_page
        from matplotlib.figure import Figure
        from matplotlib.patches import Polygon
        ink, grey = "#13212a", "#5b6b73"
        rows = [r for r in res["rows"] if r["n_traces"]]
        fig = Figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.95, "3D polygons: hexagon analysis", fontsize=17, weight="bold", color=ink)
        t0, t1 = res["window"]
        fig.text(0.07, 0.928, f"Well {res['well_name']} · {t0:,.0f}–{t1:,.0f} ms ({res['window_source']}) · {stamp}",
                 fontsize=9, color=grey)
        ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
        m = res["map"]
        lim = float(np.nanpercentile(np.abs(m["drms"]), 99)) or 1.0
        ax.imshow(m["drms"], origin="lower", aspect="auto", cmap="seismic_r", vmin=-lim, vmax=lim,
                  extent=(m["xlines"][0], m["xlines"][-1], m["ilines"][0], m["ilines"][-1]), interpolation="nearest")
        use_opp = any(np.isfinite(r["opportunity_pct"]) for r in rows)
        import matplotlib
        cmap = matplotlib.colormaps["magma"]
        for r in rows:
            val = r["opportunity_pct"] if use_opp else r["sealing_fault_probability"]
            ax.add_patch(Polygon(list(zip(r["corners_xl"], r["corners_il"])), closed=True,
                                 fc=cmap(np.clip(val / 100, 0, 1)) if np.isfinite(val) else "none",
                                 alpha=0.45, ec="white", lw=0.8))
            ax.text(np.mean(r["corners_xl"]), np.mean(r["corners_il"]), str(r["hexagon"]), ha="center",
                    va="center", fontsize=8, color="white", weight="bold")
        ax.set_xlim(m["xlines"].min(), m["xlines"].max())
        ax.set_ylim(m["ilines"].min(), m["ilines"].max())
        ax.set_xlabel("Crossline", fontsize=8.5)
        ax.set_ylabel("Inline", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_title("RMS(base) − RMS(mon) in the window; hexagon fill = "
                     + ("opportunity score" if use_opp else "sealing fault probability"), fontsize=9.5, loc="left")
        y = 0.45
        head = ["Hex", "Centre IL/CL", "Seal %", "Trace %", "Verdict", "Opp %", "In place", "Risked"]
        xs = [0.07, 0.13, 0.30, 0.38, 0.46, 0.58, 0.66, 0.78]
        for x, h in zip(xs, head):
            fig.text(x, y, h, fontsize=8, color=grey)
        y -= 0.016
        for r in sorted(rows, key=lambda r: -(r["opportunity_pct"] if np.isfinite(r["opportunity_pct"]) else -1)):
            vals = [str(r["hexagon"]), f'{r["centre_il"]:,}/{r["centre_xl"]:,}', _f(r["sealing_fault_probability"]),
                    _f(r["trace_pct"], 1), r.get("verdict") or "–", _f(r["opportunity_pct"]),
                    _f(r["stoiip_m3"] * ha.M3_TO_BBL / 1e6, 1), _f(r["risked_m3"] * ha.M3_TO_BBL / 1e6, 1)]
            for x, v_ in zip(xs, vals):
                fig.text(x, y, v_, fontsize=7.8, color=ink)
            y -= 0.0155
        v = res["volume"]
        fig.text(0.07, y - 0.01, f"Total in place {_f(v['stoiip_mmbbl'], 1)} MMbbl"
                 + (f", risked {_f(v['risked_mmbbl'], 1)} MMbbl" if v["risked"] else "") + " (volumes in MMbbl).",
                 fontsize=8.5, color=ink)
        footer(fig, first_page)
        pdf.savefig(fig)
        return first_page + 1
