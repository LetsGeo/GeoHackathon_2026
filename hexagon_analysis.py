"""
Hexagon analysis for the Analysis page: fills the report's three AI sections
("3D polygons", "Opportunity score", "Hydrocarbon volume").

Pipeline (follows the app's: Data sources -> Wells -> Seismic viewer -> Analysis):
  1. The well chosen on the Wells page; its reservoir pick (top / base TWT) is the time
     window (with a margin). Without a pick, the first two horizons at the well, else the
     whole trace.
  2. The 19-hexagon grid (polygon.py: 1 in the middle, rings of 6 and 12) centred on the
     well where it crosses the reservoir top: the 3D polygons are these hexagons between
     the window's top and base.
  3. 3D polygons, per hexagon: the traces inside it, RMS(base) - RMS(mon) in the window
     (hex_rms.py), the fault attribute -> trace % and sealing fault probability %
     (hex_table.py), and sealing_faults.py's verdict (well + horizons; Gemma or rule engine).
  4. Opportunity score: the central inline(s) and crossline(s) of each hexagon, baseline
     beside the 4D difference, go to Gemma (gemma_ilcl.py's prompt) -> reservoir, change and
     opportunity 0-100 %; averaged per hexagon.
  5. Hydrocarbon volume, per hexagon: area x thickness x N/G x porosity x (1 - Sw) / Bo,
     with N/G and porosity from the well's picked interval, thickness from the pick (or
     from the horizons in each hexagon, converted with the well's velocity), Sw and Bo from
     the settings. In place, and risked by the opportunity score.
  6. The three sections' text in English and Malay: Gemma writes it from the numbers, or
     (off the hackathon network) the app writes it itself.
Everything Gemma does falls back without it: no opportunity score, rule-engine verdicts,
texts written from the numbers.
"""
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import ai
import gemma_ilcl
import hex_table
import polygon
import sealing_faults
from hex_rms import traces_in_hexagon
from styles import SEISMIC_CMAPS
from volume3d import Volume3DSeismic, align_3d_volumes

SETTINGS = {
    "window_margin_ms": 20,     # added above and below the well's reservoir pick
    "picture_pad_ms": 150,      # shown above and below the window in the pictures (seal, overburden)
    "sections": 1,              # inlines and crosslines per hexagon sent to Gemma (1 = through the centre)
    "workers": int(os.environ.get("GEMMA_WORKERS", "8")),  # Gemma requests at the same time
    "sw": 0.30,                 # water saturation (not in the logs)
    "bo": 1.20,                 # formation volume factor, reservoir m3 per stock-tank m3
    "default_ntg": 0.70,        # used only when the well has no picked interval to measure
    "default_phi": 0.25,
    "bin_m": 12.5,              # line spacing when the survey has no map coordinates
    "use_gemma": True,
}
M3_TO_BBL = 6.28981
SECTION_KEYS = ("polygons", "opportunity", "volume")


# ---------------------------------------------------------------------------
# Inputs: volumes, well, window, geometry
# ---------------------------------------------------------------------------
def app_volume(vol):
    """The app's survey dict -> Volume3DSeismic (no copy: the cube stays memory-mapped)."""
    times = np.asarray(vol["times"], dtype=np.float64)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    return Volume3DSeismic(np.asarray(vol["ilines"]), np.asarray(vol["xlines"]), times, dt, vol["cube"])


def well_at_time(well, t):
    """(inline, crossline) of the well at time t (followed down a deviated well)."""
    il, xl, twt = np.asarray(well["il"], float), np.asarray(well["xl"], float), well.get("twt")
    if twt is None or len(twt) < 2:
        return float(il[0]), float(xl[0])
    return float(np.interp(t, twt, il)), float(np.interp(t, twt, xl))


def _horizon_near(h, il, xl, radius):
    near = (np.abs(np.asarray(h["il"]) - il) <= radius) & (np.abs(np.asarray(h["xl"]) - xl) <= radius)
    return float(np.mean(np.asarray(h["t"])[near])) if near.any() else None


def reservoir_window(well, horizons, twt, margin_ms):
    """(t0, t1, top, base, where it came from): the reservoir pick at the well, else the first
    two horizons at the well, else the whole trace; t0 / t1 include the margin."""
    t_lo, t_hi = float(twt[0]), float(twt[-1])
    pick = (well or {}).get("reservoir")
    if pick and pick.get("top_twt_ms") is not None and pick.get("base_twt_ms") is not None:
        top, base = float(pick["top_twt_ms"]), float(pick["base_twt_ms"])
        src = f"reservoir pick at {well['name']} ({pick.get('source', 'well')})"
    elif well and len(horizons) >= 2:
        il, xl = well_at_time(well, 0.5 * (t_lo + t_hi))
        ts = [_horizon_near(h, il, xl, 0.5 * polygon.SCALE) for h in horizons[:2]]
        if None in ts:
            return t_lo, t_hi, None, None, "whole trace (the horizons don't reach the well)"
        top, base = min(ts), max(ts)
        src = f"horizons {horizons[0]['name']} / {horizons[1]['name']} at {well['name']}"
    else:
        return t_lo, t_hi, None, None, "whole trace (no reservoir pick or horizons)"
    return max(t_lo, top - margin_ms), min(t_hi, base + margin_ms), top, base, src


def cell_area_m2(geo, ilines, xlines, bin_m):
    """Map area of one trace's cell (m2), and the inline spacing (m)."""
    il_step = float(np.median(np.diff(ilines))) if len(ilines) > 1 else 1.0
    xl_step = float(np.median(np.diff(xlines))) if len(xlines) > 1 else 1.0
    if geo is not None:
        M = np.asarray(geo)
        return abs(float(np.linalg.det(M[:, 1:]))) * il_step * xl_step, float(np.hypot(M[0, 1], M[1, 1])) * il_step
    return bin_m * bin_m * il_step * xl_step, bin_m * il_step  # bin_m = distance between neighbouring line numbers


# ---------------------------------------------------------------------------
# Pictures of a hexagon's sections (baseline and 4D difference)
# ---------------------------------------------------------------------------
def _chord(corners, value, axis):
    """Where the line `axis` = value crosses the hexagon: (lo, hi) along the other axis, or None."""
    c = np.asarray(corners, float)
    centre = c.mean(0)
    c = c[np.argsort(np.arctan2(c[:, 0] - centre[0], c[:, 1] - centre[1]))]
    fixed, free = c[:, axis], c[:, 1 - axis]
    hits = []
    for k in range(len(c)):
        a0, a1, b0, b1 = fixed[k], fixed[(k + 1) % len(c)], free[k], free[(k + 1) % len(c)]
        if a0 != a1:
            t = (value - a0) / (a1 - a0)
            if 0 <= t <= 1:
                hits.append(b0 + t * (b1 - b0))
    return (min(hits), max(hits)) if hits else None


def _lines(corners, lines, axis, n):
    """Indices of n lines spread across the hexagon (1 = through the centre), inside the survey."""
    c = np.asarray(corners, float)[:, axis]
    lo, hi = c.min(), c.max()
    wanted = [0.5 * (lo + hi)] if n <= 1 else np.linspace(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo), n)
    lines = np.asarray(lines)
    idx = [int(np.abs(lines - w).argmin()) for w in wanted]
    return [i for i in dict.fromkeys(idx) if lo <= lines[i] <= hi]


def _png(section, x0, x1, t0, t1, cmap, lim, title, xlabel):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111)
    ax.imshow(np.asarray(section, np.float32).T, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim,
              extent=(x0, x1, t1, t0), interpolation="bilinear")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("TWT (ms)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    return buf.getvalue()


def hexagon_pictures(name, corners, base, mon, k0, k1, n, amp, dif):
    """[(kind, line, baseline PNG, difference PNG)] for a hexagon's central section(s)."""
    from matplotlib.colors import LinearSegmentedColormap
    rwb = LinearSegmentedColormap.from_list("rwb", [(p, c) for p, c in SEISMIC_CMAPS["Seismic RWB"]])
    il, xl, twt = base.ilines, base.xlines, base.twt_ms
    t0, t1 = float(twt[k0]), float(twt[k1 - 1])
    out = []
    for axis, kind, lines, other in ((0, "IL", il, xl), (1, "CL", xl, il)):
        for i in _lines(corners, lines, axis, n):
            ch = _chord(corners, lines[i], axis)
            if ch is None:
                continue
            lo, hi = int(np.searchsorted(other, ch[0], "left")), int(np.searchsorted(other, ch[1], "right"))
            if hi - lo < 2:
                continue
            if axis == 0:
                b, m = base.data[i, lo:hi, k0:k1], mon.data[i, lo:hi, k0:k1]
            else:
                b, m = base.data[lo:hi, i, k0:k1], mon.data[lo:hi, i, k0:k1]
            b = np.asarray(b, np.float32)
            m = np.asarray(m, np.float32)
            d = np.where(np.isfinite(b) & np.isfinite(m), m - b, np.nan)  # blank where either isn't finite
            b = np.where(np.isfinite(b), b, np.nan)
            what = "Inline" if axis == 0 else "Crossline"
            xlab = "Crossline" if axis == 0 else "Inline"
            x0, x1 = float(other[lo]), float(other[hi - 1])
            out.append((kind, int(lines[i]),
                        _png(b, x0, x1, t0, t1, rwb, amp, f"{name} · {what} {lines[i]}", xlab),
                        _png(d, x0, x1, t0, t1, "seismic", dif, f"{name} · {what} {lines[i]} · monitor - baseline",
                             xlab)))
    return out


def ask_opportunity(name, kind, line, base_png, diff_png):
    """Gemma's reservoir / change / opportunity scores for one section (one retry if unusable)."""
    what = "Inline" if kind == "IL" else "Crossline"
    url = gemma_ilcl.pair_url(io.BytesIO(base_png), io.BytesIO(diff_png))
    keys = ("reservoir", "change", "opportunity")
    text = (f"Hexagon {name}, {what} {line}: baseline (left) and monitor - baseline (right). "
            "Score reservoir, change and opportunity. Reply with the JSON only.")
    try:
        return gemma_ilcl.parse_reply(ai.gemma_chat_image(gemma_ilcl.OPPORTUNITY_PROMPT, text, url, check=False), keys)
    except (ValueError, json.JSONDecodeError) as exc:
        text += f" Your previous reply could not be used ({exc}); reply with only the JSON object."
        return gemma_ilcl.parse_reply(ai.gemma_chat_image(gemma_ilcl.OPPORTUNITY_PROMPT, text, url, check=False), keys)


# ---------------------------------------------------------------------------
# Volumetrics
# ---------------------------------------------------------------------------
def reservoir_properties(well, settings):
    """N/G, porosity, gross thickness (m) and interval velocity (m/s) from the well's pick."""
    pick = (well or {}).get("reservoir") or {}
    cands = (well or {}).get("candidates") or []
    ids = set(pick.get("candidate_ids") or [])
    used = [c for c in cands if c.get("id") in ids]
    top, base = pick.get("top_twt_ms"), pick.get("base_twt_ms")
    if not used and top is not None:  # the candidates the pick overlaps
        used = [c for c in cands if c.get("base_twt_ms", -1) >= top and c.get("top_twt_ms", 1e9) <= base]
    out = {"ntg": settings["default_ntg"], "phi": settings["default_phi"], "props_source": "assumed (no interval)",
           "thickness_m": None, "velocity_ms": None}
    if used:
        w = np.array([max(float(c.get("thickness_ms", 1)), 1e-6) for c in used])
        out["ntg"] = float(np.average([float(c["net_fraction"]) for c in used], weights=w))
        out["phi"] = float(np.average([float(c["phi_density"]) for c in used], weights=w))
        out["props_source"] = "well interval " + ", ".join(str(c.get("id")) for c in used)
    if top is not None and base is not None:
        logs = (well or {}).get("logs")
        h = None
        try:
            if logs is not None and "TWT_MS" in logs and "TVD_KB" in logs:
                d = logs.sort_values("TWT_MS")
                tv = np.interp([top, base], d.TWT_MS.values, d.TVD_KB.values)
                h = float(tv[1] - tv[0])
        except Exception:
            h = None
        if (h is None or h <= 0) and pick.get("top_md_m") is not None and pick.get("base_md_m") is not None:
            h = float(pick["base_md_m"]) - float(pick["top_md_m"])
        if h and h > 0:
            out["thickness_m"] = h
            out["velocity_ms"] = 2 * h / ((float(base) - float(top)) / 1000.0)
    return out


def volumetrics(rows, props, area_m2, settings):
    """Per-hexagon and total volumes (fills rows in place)."""
    sw, bo = settings["sw"], settings["bo"]
    tot = {"grv_m3": 0.0, "stoiip_m3": 0.0, "risked_m3": 0.0, "risked": False}
    for r in rows:
        r["area_km2"] = r["n_traces"] * area_m2 / 1e6
        h = props["thickness_m"]
        th_ms = r.get("thickness_ms")
        if th_ms is not None and np.isfinite(th_ms) and th_ms > 0 and props["velocity_ms"]:
            h = props["velocity_ms"] * th_ms / 2000.0   # this hexagon's thickness from the horizons
        r["thickness_m"] = h
        if not h or not r["n_traces"]:
            r["stoiip_m3"] = r["risked_m3"] = float("nan")
            continue
        grv = r["n_traces"] * area_m2 * h
        stoiip = grv * props["ntg"] * props["phi"] * (1 - sw) / bo
        r["stoiip_m3"] = stoiip
        opp = r.get("opportunity_pct")
        r["risked_m3"] = stoiip * opp / 100 if opp is not None and np.isfinite(opp) else float("nan")
        tot["grv_m3"] += grv
        tot["stoiip_m3"] += stoiip
        if np.isfinite(r["risked_m3"]):
            tot["risked_m3"] += r["risked_m3"]
            tot["risked"] = True
    tot["stoiip_mmbbl"] = tot["stoiip_m3"] * M3_TO_BBL / 1e6
    tot["risked_mmbbl"] = tot["risked_m3"] * M3_TO_BBL / 1e6
    return tot


# ---------------------------------------------------------------------------
# The sections' text
# ---------------------------------------------------------------------------
def _num(v, d=0):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:,.{d}f}"


def rule_texts(res):
    """The three sections in English and Malay, written from the numbers (no Gemma needed)."""
    rows = [r for r in res["rows"] if r["n_traces"]]
    w, t0, t1 = res["well_name"], res["window"][0], res["window"][1]
    n = len(rows)
    area = np.mean([r["area_km2"] for r in rows]) if rows else 0
    chg = [r for r in rows if r.get("rms_diff") is not None and np.isfinite(r["rms_diff"])]
    strong = max(chg, key=lambda r: abs(r["rms_diff"]), default=None)
    seal = max(rows, key=lambda r: r["sealing_fault_probability"], default=None)
    likely = [r for r in rows if r.get("verdict") == "likely"]
    en = (f"{n} hexagons (about {area:.2f} km² each) around well {w}, between {_num(t0)} and {_num(t1)} ms "
          f"({res['window_source']}). ")
    ms = (f"{n} heksagon (kira-kira {area:.2f} km² setiap satu) di sekitar telaga {w}, antara {_num(t0)} dan "
          f"{_num(t1)} ms. ")
    if strong is not None:
        en += (f"The largest 4D change is in hexagon {strong['hexagon']} "
               f"(RMS(base) − RMS(mon) = {strong['rms_diff']:.3g}). ")
        ms += (f"Perubahan 4D terbesar berada di heksagon {strong['hexagon']} "
               f"(RMS(asas) − RMS(pemantauan) = {strong['rms_diff']:.3g}). ")
    if seal is not None and seal["sealing_fault_probability"] > 0:
        en += (f"Sealing fault probability is highest in hexagon {seal['hexagon']} "
               f"({seal['sealing_fault_probability']:.0f}%, fault on {seal['trace_pct']:.0f}% of its traces, "
               f"{seal['seal_evidence']}). ")
        ms += (f"Kebarangkalian sesar pengedap paling tinggi di heksagon {seal['hexagon']} "
               f"({seal['sealing_fault_probability']:.0f}%, sesar pada {seal['trace_pct']:.0f}% surihnya). ")
    else:
        en += "No hexagon shows a clear sealing fault. "
        ms += "Tiada heksagon menunjukkan sesar pengedap yang jelas. "
    if res.get("sealing"):
        if likely:
            ids = ", ".join(str(r["hexagon"]) for r in likely)
            en += f"The sealing-fault check ({res['sealing']['source']}) rates hexagon(s) {ids} as likely."
            ms += f"Semakan sesar pengedap menilai heksagon {ids} sebagai berkemungkinan."
        else:
            en += f"The sealing-fault check ({res['sealing']['source']}) rates no hexagon as likely."
            ms += "Semakan sesar pengedap tidak menilai mana-mana heksagon sebagai berkemungkinan."
    out = {"polygons": {"en": en.strip(), "ms": ms.strip()}}

    opp = [r for r in rows if r.get("opportunity_pct") is not None and np.isfinite(r["opportunity_pct"])]
    if opp:
        best = sorted(opp, key=lambda r: -r["opportunity_pct"])[:3]
        mean = np.mean([r["opportunity_pct"] for r in opp])
        en = (f"Gemma scored {len(opp)} hexagons from their sections (baseline beside the 4D difference): "
              f"average opportunity {mean:.0f}%. Best: "
              + ", ".join(f"hexagon {r['hexagon']} ({r['opportunity_pct']:.0f}%)" for r in best) + ". ")
        ms = (f"Gemma menilai {len(opp)} heksagon daripada keratannya (asas di sebelah perbezaan 4D): "
              f"purata peluang {mean:.0f}%. Terbaik: "
              + ", ".join(f"heksagon {r['hexagon']} ({r['opportunity_pct']:.0f}%)" for r in best) + ". ")
        if best[0].get("opportunity_reason"):
            en += f"Hexagon {best[0]['hexagon']}: {best[0]['opportunity_reason']}"
        out["opportunity"] = {"en": en.strip(), "ms": ms.strip()}
    else:
        out["opportunity"] = {"en": "No opportunity score: Gemma could not be reached to read the hexagons' sections.",
                              "ms": "Tiada skor peluang: Gemma tidak dapat dihubungi untuk membaca keratan heksagon."}

    v, p = res["volume"], res["props"]
    if p["thickness_m"]:
        en = (f"In place {v['stoiip_mmbbl']:.1f} MMbbl ({v['stoiip_m3'] / 1e6:.2f} million m³) over the "
              f"hexagons, from N/G {p['ntg']:.2f} and porosity {p['phi']:.2f} ({p['props_source']}), "
              f"{p['thickness_m']:.0f} m gross at the well, Sw {res['settings']['sw']:.2f} and "
              f"Bo {res['settings']['bo']:.2f}. ")
        ms = (f"Isipadu di tempat {v['stoiip_mmbbl']:.1f} MMbbl ({v['stoiip_m3'] / 1e6:.2f} juta m³) bagi "
              f"heksagon, daripada N/G {p['ntg']:.2f} dan keliangan {p['phi']:.2f}, ketebalan kasar "
              f"{p['thickness_m']:.0f} m di telaga, Sw {res['settings']['sw']:.2f} dan Bo {res['settings']['bo']:.2f}. ")
        if v["risked"]:
            en += f"Risked by the opportunity score: {v['risked_mmbbl']:.1f} MMbbl."
            ms += f"Selepas risiko skor peluang: {v['risked_mmbbl']:.1f} MMbbl."
        out["volume"] = {"en": en.strip(), "ms": ms.strip()}
    else:
        out["volume"] = {"en": "No volume: the well has no reservoir pick with a thickness to use.",
                         "ms": "Tiada isipadu: telaga tidak mempunyai pilihan reservoir dengan ketebalan."}
    return out


TEXT_SYSTEM = ("You are a senior reservoir geoscientist writing a short report on a 4D seismic study of a "
               "CO2 storage / hydrocarbon site. Use ONLY the numbers given; never invent values.")


def gemma_texts(res, fallback):
    """The three sections written by Gemma (English + Bahasa Melayu); raises if unusable."""
    rows = [{k: (round(v, 3) if isinstance(v, float) and np.isfinite(v) else v) for k, v in r.items()
             if k in ("hexagon", "ring", "centre_il", "centre_xl", "rms_diff", "sealing_fault_probability",
                      "trace_pct", "seal_evidence", "verdict", "opportunity_pct", "reservoir_pct", "stoiip_m3")}
            for r in res["rows"] if r["n_traces"]]
    facts = {"well": res["well_name"], "window_ms": [round(res["window"][0]), round(res["window"][1])],
             "window_source": res["window_source"], "hexagons": rows,
             "reservoir_properties": res["props"], "settings": {k: res["settings"][k] for k in ("sw", "bo")},
             "totals": {k: round(v, 3) if isinstance(v, float) else v for k, v in res["volume"].items()},
             "sealing_fault_check": (res.get("sealing") or {}).get("summary", ""),
             "draft": {k: fallback[k]["en"] for k in SECTION_KEYS}}
    user = ("Facts (JSON):\n" + json.dumps(facts, ensure_ascii=False, default=float) + "\n\n"
            "Write the report's three sections, 2-4 sentences each, in English and Bahasa Melayu:\n"
            "- polygons: the 3D polygons (hexagons round the well, in the window): where the 4D change and "
            "sealing faults are\n- opportunity: the opportunity scores, the best hexagons and why\n"
            "- volume: the hydrocarbon volume, in place and risked, and its main assumptions\n"
            'Reply with ONLY this JSON: {"polygons": {"en": "...", "ms": "..."}, '
            '"opportunity": {"en": "...", "ms": "..."}, "volume": {"en": "...", "ms": "..."}}')
    data = ai.parse_json_reply(ai.gemma_chat(TEXT_SYSTEM, user, max_tokens=1400))
    out = {}
    for k in SECTION_KEYS:
        sec = data.get(k) or {}
        if not str(sec.get("en", "")).strip() or not str(sec.get("ms", "")).strip():
            raise ValueError(f"Gemma's reply has no '{k}' section")
        out[k] = {"en": str(sec["en"]).strip(), "ms": str(sec["ms"]).strip()}
    return out


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------
def run(base_vol, mon_vol, well, horizons=(), geo=None, settings=None, progress=None):
    """The hexagon analysis. base_vol / mon_vol are the app's survey dicts; well and horizons
    the app's records. progress(text) is called as it goes. Returns a dict (see the keys below)."""
    cfg = dict(SETTINGS, **(settings or {}))
    say = progress or (lambda text: None)
    if well is None:
        raise ValueError("load a well on the Wells page first: the hexagons go round it")
    notes = []

    say("Aligning the surveys…")
    base, mon = align_3d_volumes(app_volume(base_vol), app_volume(mon_vol))
    twt = base.twt_ms
    t0, t1, top, bottom, src = reservoir_window(well, list(horizons), twt, cfg["window_margin_ms"])
    k0, k1 = int(np.searchsorted(twt, t0, "left")), int(np.searchsorted(twt, t1, "right"))
    if k1 - k0 < 8:
        raise ValueError(f"the window {t0:.0f}-{t1:.0f} ms is outside the surveys' {twt[0]:.0f}-{twt[-1]:.0f} ms")
    c_il, c_xl = well_at_time(well, top if top is not None else 0.5 * (t0 + t1))
    centers, hexes, ring = polygon.hexagon_rings(c_il, c_xl)

    say("Measuring attributes on every trace…")
    dt = float(np.median(np.diff(twt)))
    disc, drms = hex_table.trace_attributes(base, mon, k0, k1, max(1, int(round(hex_table.MAX_DIP_LAG_MS / dt))))
    z, flag, p_trace, changed = hex_table.reference_levels(disc, drms)
    line_step = float(np.median(np.diff(base.ilines))) if len(base.ilines) > 1 else 1.0
    rows = [hex_table.hexagon_row(k, rg, c, crn, base, disc, z, flag, p_trace, changed, drms, line_step)
            for k, (c, crn, rg) in enumerate(zip(centers, hexes, ring))]
    for r, crn in zip(rows, hexes):
        r["corners_il"], r["corners_xl"] = crn[:, 0].tolist(), crn[:, 1].tolist()
        r.update({"verdict": None, "verdict_confidence": None, "verdict_reason": "", "thickness_ms": None,
                  "opportunity_pct": float("nan"), "reservoir_pct": float("nan"), "change_pct": float("nan"),
                  "opportunity_reason": ""})
    inside = [r for r in rows if r["n_traces"]]
    if not inside:
        raise ValueError("none of the hexagons round the well is inside the surveys")

    use_gemma = cfg["use_gemma"] and ai.gemma_reachable()
    if cfg["use_gemma"] and not use_gemma:
        notes.append(f"Gemma isn't reachable ({ai.GEMMA_URL}): no opportunity score; verdicts and texts "
                     "come from the rule engine and the numbers")

    say("Checking the sealing faults against the well and horizons…")
    area_m2, spacing_m = cell_area_m2(geo, base.ilines, base.xlines, cfg["bin_m"])
    sealing = None
    try:
        sf_rows = [{"hexagon": r["hexagon"], "il": float(r["centre_il"]), "xl": float(r["centre_xl"]),
                    "probability": float(r["sealing_fault_probability"]), "trace_pct": float(r["trace_pct"]),
                    "source": "hexagon analysis"} for r in inside]
        sealing = sealing_faults.find_sealing_faults(sf_rows, well=well, horizons=list(horizons),
                                                     use_gemma=use_gemma, line_spacing_m=spacing_m)
        by_id = {c["hexagon"]: c for c in sealing["hexagons"]}
        for r in inside:
            c = by_id.get(r["hexagon"], {})
            r["verdict"], r["verdict_confidence"] = c.get("verdict"), c.get("confidence")
            r["verdict_reason"] = c.get("reason") or ""
            th = [v for k, v in c.items() if k.startswith("thickness_") and k.endswith("_ms")]
            r["thickness_ms"] = th[0] if th and th[0] is not None else None
        if sealing.get("note"):
            notes.append(sealing["note"])
    except Exception as exc:
        notes.append(f"sealing-fault check skipped ({type(exc).__name__}: {str(exc)[:160]})")

    say("Drawing the hexagons' sections…")
    sample = np.asarray(base.data[::max(1, len(base.ilines) // 60), ::max(1, len(base.xlines) // 60), k0:k1],
                        np.float32)
    msample = np.asarray(mon.data[::max(1, len(base.ilines) // 60), ::max(1, len(base.xlines) // 60), k0:k1],
                         np.float32)
    def p99(a):
        a = np.abs(a[np.isfinite(a)])
        v = float(np.percentile(a, 99)) if a.size else 0.0
        return v if v > 0 else None
    amp = p99(sample) or 1.0
    dif = p99(msample - sample) or amp  # NaN / inf samples (in either survey) are left out
    bad = 1 - float(np.mean(np.isfinite(msample))) if msample.size else 0.0
    if bad > 0.001:
        notes.append(f"the monitor has {100 * bad:.1f}% non-finite samples (NaN / inf) in the window: "
                     "those samples are left out of the comparison")
    pk0 = int(np.searchsorted(twt, max(twt[0], t0 - cfg["picture_pad_ms"]), "left"))
    pk1 = int(np.searchsorted(twt, min(twt[-1], t1 + cfg["picture_pad_ms"]), "right"))
    pictures = {}
    for r, crn in zip(rows, hexes):
        if r["n_traces"]:
            pictures[r["hexagon"]] = hexagon_pictures(f"hex{r['hexagon']:02d}", crn, base, mon, pk0, pk1,
                                                      cfg["sections"], amp, dif)

    if use_gemma:
        jobs = [(hid, kind, line, b, d) for hid, pics in pictures.items() for kind, line, b, d in pics]
        answers, failed, done = {}, [], 0
        say(f"Gemma is scoring the hexagons… 0 / {len(jobs)} sections")
        with ThreadPoolExecutor(max_workers=max(1, cfg["workers"])) as pool:
            futs = {pool.submit(ask_opportunity, f"hex{hid:02d}", kind, line, b, d): (hid, kind, line)
                    for hid, kind, line, b, d in jobs}
            for fut in as_completed(futs):
                hid = futs[fut][0]
                done += 1
                try:
                    answers.setdefault(hid, []).append(fut.result())
                except Exception as exc:
                    failed.append(f"{type(exc).__name__}: {str(exc)[:100]}")
                say(f"Gemma is scoring the hexagons… {done} / {len(jobs)} sections")
        for r in inside:
            a = answers.get(r["hexagon"]) or []
            if a:
                r["reservoir_pct"] = float(np.mean([x["reservoir"] for x in a]))
                r["change_pct"] = float(np.mean([x["change"] for x in a]))
                r["opportunity_pct"] = float(np.mean([x["opportunity"] for x in a]))
                r["opportunity_reason"] = max(a, key=lambda x: x["opportunity"])["reason"]
        if failed:
            notes.append(f"Gemma didn't answer for {len(failed)} of {len(jobs)} sections ({failed[0]})")

    props = reservoir_properties(well, cfg)
    if props["thickness_m"] is None:
        notes.append("the well has no reservoir pick with a thickness: no volume")
    volume = volumetrics(rows, props, area_m2, cfg)

    res = {"well_name": well["name"], "centre": (c_il, c_xl), "window": (t0, t1), "reservoir": (top, bottom),
           "window_source": src, "rows": rows, "sealing": sealing, "props": props, "volume": volume,
           "settings": cfg, "pictures": pictures, "notes": notes, "gemma_used": use_gemma,
           "area_m2_per_trace": area_m2,
           "map": {"drms": drms, "ilines": base.ilines, "xlines": base.xlines}}
    say("Writing the report sections…")
    texts = rule_texts(res)
    res["texts_source"] = "written from the numbers"
    if use_gemma:
        try:
            texts = gemma_texts(res, texts)
            res["texts_source"] = f"Gemma ({ai.GEMMA_MODEL})"
        except Exception as exc:
            notes.append(f"Gemma didn't write the sections ({type(exc).__name__}: {str(exc)[:120]}); "
                         "they're written from the numbers")
    res["texts"] = texts
    return res


TABLE_COLUMNS = ["hexagon", "centre_il", "centre_xl", "sealing_fault_probability", "trace_pct"]


def table_csv(res):
    """The hexagon table (sealing_faults.py's input format: numbers only)."""
    lines = [",".join(TABLE_COLUMNS)]
    for r in res["rows"]:
        if r["n_traces"]:
            lines.append(",".join(str(round(r[c], 2) if isinstance(r[c], float) else r[c]) for c in TABLE_COLUMNS))
    return "\n".join(lines) + "\n"


DETAIL_COLUMNS = TABLE_COLUMNS + ["ring", "n_traces", "area_km2", "rms_diff", "fault_traces", "fault_probability",
                                  "seal_contrast", "seal_evidence", "verdict", "verdict_confidence", "verdict_reason",
                                  "reservoir_pct", "change_pct", "opportunity_pct", "opportunity_reason",
                                  "thickness_m", "stoiip_m3", "risked_m3"]


def details_csv(res):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(DETAIL_COLUMNS)
    for r in res["rows"]:
        w.writerow([round(r.get(c), 4) if isinstance(r.get(c), float) else r.get(c, "") for c in DETAIL_COLUMNS])
    return buf.getvalue()
