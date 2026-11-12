"""
Well logs in seismic time, and the reservoir in them (used by the Wells page).

process_well() does everything for one well:
  1. reads the LAS file (curves + header: well name, KB elevation, surface X/Y)
  2. converts every log sample from depth to two-way time (TWT, ms):
       MD (from KB) -> TVD below KB (deviation survey, or vertical)
                    -> TVDSS (below sea level) -> TWT via the checkshot
     The deviation survey also gives each sample's north / east offset from
     the wellhead, so deviated wells are drawn along their real path.
  3. flags every sample that passes all four tests (in time):
       Gamma Ray low      -> sandy, not shale
       Bulk density low   -> porous
       Neutron low        -> below the shale level: hydrocarbon (especially gas)
       Resistivity high   -> hydrocarbon, not water
     and groups them into candidate intervals, ranked by a simple score.
  4. picks the reservoir's top and base TWT: Gemma chooses from the candidates
     when it can be reached (see ai.py), otherwise the top-ranked candidate.

Based on the team's well_reservoir_twt script, reorganised so the app can call
it: no pop-up plots, and problems come back as messages instead of prints.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Imported here, at start-up, not inside ask_gemma: when the app is started with
# `panel serve app.py`, the project folder is only on Python's search path while
# app.py first runs, so an import made later (on "Load well") can't find ai.py.
from ai import gemma_chat, parse_json_reply

# Curve names are found automatically; these are tried in order (upper case).
CURVE_ALIASES = {
    "depth": ("DEPTH", "DEPT", "MD", "DEPTH_M"),
    "gr": ("GR_P", "GR", "GRC", "SGR", "CGR", "GR_EDTC", "GAMMA"),
    "nphi": ("NEUT_P", "NPHI", "TNPH", "NPOR", "NEUT", "CNC", "NPHI_LS"),
    "rhob": ("DENS_P", "RHOB", "RHOZ", "DEN", "DENS", "ZDEN"),
    "rt": ("RT_P", "RT", "RD", "ILD", "LLD", "AT90", "RDEP", "RESD", "HDRS"),
}
CURVE_LABELS = {"gr": "Gamma Ray", "nphi": "Neutron porosity", "rhob": "Bulk density", "rt": "Resistivity"}

# How the checkshot file is laid out (the Wells page lets you change these)
CHECKSHOT_DEFAULTS = dict(
    depth_col=0,          # column number (0 = first) holding depth
    time_col=1,           # column number holding time
    depth_type="TVDSS",   # "TVDSS" (below sea level), "TVD_KB" (below KB) or "MD" (from KB)
    depth_sign=1,         # -1 if depths are written as negative numbers
    time_type="TWT",      # "TWT" (two-way) or "OWT" (one-way)
    time_unit="ms",       # "ms" or "s"
    shift_ms=0.0,         # bulk shift if the checkshot datum differs from the seismic datum
)
DEVIATION_COLS = dict(md=0, inc=1, azi=2)

# Reservoir cut-offs, all applied in TWT (sensible starting points; tune per field)
RESERVOIR_DEFAULTS = dict(
    baseline_mode="rolling",   # "rolling" (follows the log down) or "constant"
    baseline_window_ms=120,    # window for "rolling" baselines
    vsh_max=0.50,              # GR low: shale volume from GR below this
    rho_matrix=2.65, rho_fluid=1.0,
    phi_min=0.15,              # RHOB low: density porosity above this (-> RHOB < ~2.40)
    nphi_margin=0.04,          # NPHI low: at least this far below the shale neutron baseline
    rt_factor=2.0,             # RT high: above rt_factor x shale resistivity background
    min_thick_ms=2.0,          # ignore intervals thinner than this
    merge_gap_ms=1.0,          # join intervals closer than this
)


# ----------------------------------------------------------------- READERS
def _las_header_value(line):
    """(mnemonic, value) from a LAS header line 'MNEM.UNIT  VALUE : DESCRIPTION'."""
    if "." not in line:
        return None, None
    mnem, rest = line.split(".", 1)
    rest = rest.split(":", 1)[0]
    parts = rest.split(None, 1)  # first token is the unit (may be absent: then it's the value)
    if rest[:1].isspace() or not parts:
        value = rest.strip()
    else:
        value = parts[1].strip() if len(parts) > 1 else ""
    return mnem.strip().upper(), value


def read_las(path):
    """LAS 2.0/3.0 file -> (curves DataFrame, header dict of mnemonic -> text value)."""
    lines = Path(path).read_text(errors="ignore").splitlines()
    try:
        c_start = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("~C"))
        a_start = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("~A"))
    except StopIteration:
        raise ValueError("it doesn't look like a LAS file (no ~Curve or ~ASCII section)") from None
    header = {}
    for l in lines[:c_start]:
        if l.strip() and not l.lstrip().startswith(("#", "~")):
            mnem, value = _las_header_value(l)
            if mnem:
                header[mnem] = value
    names = [l.split(".")[0].strip() for l in lines[c_start + 1:a_start]
             if l.strip() and not l.lstrip().startswith(("#", "~"))]
    null = -999.25
    try:
        null = float(header.get("NULL", null))
    except ValueError:
        pass
    df = pd.read_csv(path, sep=r"\s+", skiprows=a_start + 1, header=None, names=names,
                     comment="#", engine="python")
    return df.replace(null, np.nan), header


def detect_curves(columns):
    """Which column is depth / GR / NPHI / RHOB / RT (raises if one is missing)."""
    upper = {c.upper(): c for c in columns}
    found, missing = {}, []
    for key, names in CURVE_ALIASES.items():
        col = next((upper[n] for n in names if n in upper), None)
        if col is None and key != "depth":  # e.g. "GR_CAL" when no exact match
            col = next((c for c in columns if c.upper().startswith(names[1])), None)
        if col is None:
            missing.append(CURVE_LABELS.get(key, "depth"))
        found[key] = col
    if missing:
        raise ValueError(f"no {', '.join(missing)} curve found (curves in the file: {', '.join(columns)})")
    return found


def _header_float(header, *keys):
    for k in keys:
        try:
            return float(str(header.get(k, "")).split()[0])
        except (ValueError, IndexError):
            continue
    return None


def header_info(header, fallback_name):
    """Well name, KB elevation and surface X / Y from the LAS header (None where absent)."""
    name = (header.get("WELL") or header.get("UWI") or "").strip() or fallback_name
    return dict(
        name=name,
        kb=_header_float(header, "EKB", "EREF", "KB", "EDF", "ELEV"),
        x=_header_float(header, "XCOORD", "X", "XWELL", "EAST", "EASTING", "LOCX", "X_LOC"),
        y=_header_float(header, "YCOORD", "Y", "YWELL", "NORTH", "NORTHING", "LOCY", "Y_LOC"),
    )


def read_numeric_table(path):
    """Rows of a text table where every token is a number (headers / comments skipped)."""
    rows = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        tokens = line.replace(",", " ").replace("\t", " ").replace(";", " ").split()
        if not tokens:
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no rows of numbers in {Path(path).name}")
    widths = [len(r) for r in rows]
    width = max(set(widths), key=widths.count)
    return np.array([r for r in rows if len(r) == width])


# ----------------------------------------------------------------- MD -> TVD, offsets
def minimum_curvature(md, inc_deg, azi_deg):
    """TVD, north and east offsets along a deviation survey (minimum-curvature method)."""
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tvd, north, east = (np.zeros_like(md, dtype=float) for _ in range(3))
    for i in range(1, len(md)):
        cos_dl = (np.cos(inc[i] - inc[i - 1])
                  - np.sin(inc[i - 1]) * np.sin(inc[i]) * (1 - np.cos(azi[i] - azi[i - 1])))
        dl = np.arccos(np.clip(cos_dl, -1, 1))
        rf = 1.0 if dl < 1e-9 else (2 / dl) * np.tan(dl / 2)
        half = (md[i] - md[i - 1]) / 2 * rf
        tvd[i] = tvd[i - 1] + half * (np.cos(inc[i - 1]) + np.cos(inc[i]))
        north[i] = north[i - 1] + half * (np.sin(inc[i - 1]) * np.cos(azi[i - 1]) + np.sin(inc[i]) * np.cos(azi[i]))
        east[i] = east[i - 1] + half * (np.sin(inc[i - 1]) * np.sin(azi[i - 1]) + np.sin(inc[i]) * np.sin(azi[i]))
    return tvd, north, east


class Deviation:
    """MD -> (TVD below KB, north offset, east offset). Vertical if no survey file."""

    def __init__(self, path=None):
        self.path = path
        if path is None:
            return
        t = read_numeric_table(path)
        md, inc, azi = (t[:, DEVIATION_COLS[k]] for k in ("md", "inc", "azi"))
        if md[0] > 0:  # start the survey at the wellhead
            md, inc, azi = np.r_[0, md], np.r_[inc[0], inc], np.r_[azi[0], azi]
        self.md = md
        self.tvd, self.north, self.east = minimum_curvature(md, inc, azi)

    def __call__(self, md):
        md = np.asarray(md, dtype=float)
        if self.path is None:
            return md.copy(), np.zeros_like(md), np.zeros_like(md)
        return (np.interp(md, self.md, self.tvd), np.interp(md, self.md, self.north),
                np.interp(md, self.md, self.east))


# ----------------------------------------------------------------- CHECKSHOT
def load_checkshot(path, kb, deviation, settings=None):
    """Checkshot -> (DataFrame of TVDSS, TWT_MS, interval velocity, list of warnings)."""
    s = dict(CHECKSHOT_DEFAULTS, **(settings or {}))
    t = read_numeric_table(path)
    if max(s["depth_col"], s["time_col"]) >= t.shape[1]:
        raise ValueError(f"the checkshot has only {t.shape[1]} columns; check the depth / time column numbers")
    depth = s["depth_sign"] * t[:, s["depth_col"]]
    time = t[:, s["time_col"]] * (1000.0 if s["time_unit"] == "s" else 1.0)
    if s["time_type"].upper() == "OWT":
        time = 2 * time
    time = time + s["shift_ms"]
    if s["depth_type"] == "MD":
        depth = deviation(depth)[0] - kb
    elif s["depth_type"] == "TVD_KB":
        depth = depth - kb

    cs = pd.DataFrame({"TVDSS": depth, "TWT_MS": time}).dropna()
    cs = cs.sort_values("TVDSS").drop_duplicates("TVDSS")
    if (cs.TVDSS <= 0).sum() == 0:  # sea level = time zero
        cs = pd.concat([pd.DataFrame({"TVDSS": [0.0], "TWT_MS": [0.0]}), cs])
    cs = cs.reset_index(drop=True)
    if len(cs) < 2:
        raise ValueError("the checkshot needs at least two depth-time pairs")
    cs["VINT_MPS"] = np.r_[np.nan, 2000 * np.diff(cs.TVDSS) / np.diff(cs.TWT_MS)]
    warnings = []
    if not np.all(np.diff(cs.TWT_MS.values) > 0):
        warnings.append("checkshot time doesn't increase with depth everywhere: check the column numbers "
                        "and the 'depths are negative' setting")
    bad = cs.VINT_MPS[(cs.VINT_MPS < 1400) | (cs.VINT_MPS > 6000)]
    if len(bad):
        warnings.append(f"{len(bad)} checkshot interval velocities are outside 1400-6000 m/s "
                        "(usually a wrong column, unit or depth-type setting)")
    return cs, warnings


def tvdss_to_twt(tvdss, cs):
    """Linear inside the checkshot; outside, extended with the end interval velocities."""
    z, t = cs.TVDSS.values, cs.TWT_MS.values
    tvdss = np.asarray(tvdss, dtype=float)
    twt = np.interp(tvdss, z, t)
    v_top = 2000 * (z[1] - z[0]) / (t[1] - t[0])
    v_bot = 2000 * (z[-1] - z[-2]) / (t[-1] - t[-2])
    twt = np.where(tvdss < z[0], t[0] - 2000 * (z[0] - tvdss) / v_top, twt)
    twt = np.where(tvdss > z[-1], t[-1] + 2000 * (tvdss - z[-1]) / v_bot, twt)
    return twt


# ----------------------------------------------------------------- RESERVOIR IN TIME
def flag_reservoir_twt(d, c, cfg):
    """Adds baselines, cut-offs and the four test flags (+ PAY) to d. Returns the cut-offs used."""
    GR, NPHI, RHOB, RT = c["gr"], c["nphi"], c["rhob"], c["rt"]
    logged = d[[GR, NPHI, RHOB, RT]].notna().all(axis=1)
    gr = d[GR].where(logged)
    if cfg["baseline_mode"] == "rolling":
        step = np.nanmedian(np.diff(d.TWT_MS))
        win = max(int(cfg["baseline_window_ms"] / step), 11)
        roll = gr.rolling(win, center=True, min_periods=win // 4)
        d["GR_SHALE_BASE"], d["GR_SAND_BASE"] = roll.quantile(0.95), roll.quantile(0.05)
        shale = gr > roll.quantile(0.75)

        def med(s):
            return s.where(shale).rolling(win, center=True, min_periods=5).median()
        d["NPHI_SHALE_BASE"], d["RT_SHALE_BASE"] = med(d[NPHI]), med(d[RT])
    else:
        shale = gr > gr.quantile(0.75)
        d["GR_SHALE_BASE"], d["GR_SAND_BASE"] = gr.quantile(0.95), gr.quantile(0.05)
        d["NPHI_SHALE_BASE"] = d.loc[shale, NPHI].median()
        d["RT_SHALE_BASE"] = d.loc[shale, RT].median()
    for col in ["GR_SHALE_BASE", "GR_SAND_BASE", "NPHI_SHALE_BASE", "RT_SHALE_BASE"]:
        d[col] = d[col].ffill().bfill()

    d["GR_CUTOFF"] = d.GR_SAND_BASE + cfg["vsh_max"] * (d.GR_SHALE_BASE - d.GR_SAND_BASE)
    d["RHOB_CUTOFF"] = cfg["rho_matrix"] - cfg["phi_min"] * (cfg["rho_matrix"] - cfg["rho_fluid"])
    d["NPHI_CUTOFF"] = d.NPHI_SHALE_BASE - cfg["nphi_margin"]
    d["RT_CUTOFF"] = cfg["rt_factor"] * d.RT_SHALE_BASE
    d["PAY"] = (logged & (d[GR] < d.GR_CUTOFF) & (d[RHOB] < d.RHOB_CUTOFF)
                & (d[NPHI] < d.NPHI_CUTOFF) & (d[RT] > d.RT_CUTOFF))
    z = d[logged]
    if z.empty:
        raise ValueError("no depth has all four curves (GR, neutron, density, resistivity)")

    def rng(s, n=2):
        lo, hi = round(float(s.min()), n), round(float(s.max()), n)
        return lo if lo == hi else f"{lo} to {hi}"
    return dict(baseline_mode=cfg["baseline_mode"], gr_cutoff_api=rng(z.GR_CUTOFF, 1),
                rhob_cutoff_gcc=round(float(z.RHOB_CUTOFF.iloc[0]), 3),
                nphi_cutoff_vv=rng(z.NPHI_CUTOFF, 3), rt_cutoff_ohmm=rng(z.RT_CUTOFF, 2))


def group_intervals_twt(d, c, cfg):
    """Runs of PAY samples -> candidate intervals, best first (ids C1, C2, ...)."""
    d = d.sort_values("TWT_MS")
    t, pay = d.TWT_MS.values, d.PAY.values
    runs, start = [], None
    for i, p in enumerate(pay):
        if p and start is None:
            start = i
        if start is not None and (not p or i == len(pay) - 1):
            runs.append([t[start], t[i if p else i - 1]])
            start = None
    merged = []
    for top, base in runs:
        if merged and top - merged[-1][1] <= cfg["merge_gap_ms"]:
            merged[-1][1] = base
        else:
            merged.append([top, base])

    rows = []
    for top, base in merged:
        if base - top < cfg["min_thick_ms"]:
            continue
        z = d[(d.TWT_MS >= top) & (d.TWT_MS <= base)]
        rows.append(dict(
            top_twt_ms=round(top, 1), base_twt_ms=round(base, 1), thickness_ms=round(base - top, 1),
            top_md_m=round(z[c["depth"]].min(), 1), base_md_m=round(z[c["depth"]].max(), 1),
            net_fraction=round(z.PAY.mean(), 2),
            gr_api=round(z[c["gr"]].mean(), 1), rhob_gcc=round(z[c["rhob"]].mean(), 3),
            nphi_vv=round(z[c["nphi"]].mean(), 3), rt_ohmm=round(z[c["rt"]].mean(), 2),
            rt_max_ohmm=round(z[c["rt"]].max(), 2),
            phi_density=round(((cfg["rho_matrix"] - z[c["rhob"]])
                               / (cfg["rho_matrix"] - cfg["rho_fluid"])).mean(), 3)))
    cand = pd.DataFrame(rows)
    if len(cand):
        cand["score"] = (cand.thickness_ms * cand.net_fraction * cand.phi_density
                         * np.log(cand.rt_ohmm.clip(lower=1.01))).round(3)
        cand = cand.sort_values("score", ascending=False).reset_index(drop=True)
        cand.insert(0, "id", [f"C{i + 1}" for i in range(len(cand))])
    return cand


# ----------------------------------------------------------------- GEMMA
GEMMA_SYSTEM = ("You are a senior petrophysicist. A reservoir needs ALL of: low gamma ray (sand), "
                "low bulk density (porous), low neutron porosity relative to shale (hydrocarbon, "
                "especially gas), and high resistivity (hydrocarbon, not water). "
                "Use ONLY the numbers given; never invent values.")


def ask_gemma(well, cand, params):
    """Gemma's pick of the best reservoir (dict). Raises if Gemma can't be reached or replies badly."""
    user = f"""Well {well}. All depths are in seismic two-way time (TWT, ms).
Cut-offs used: {json.dumps(params)}

Candidate intervals that passed all four tests, ranked by a simple score:
{cand.to_csv(index=False)}
Choose the SINGLE best reservoir and give its two bounding values in TWT.
Reply with ONLY this JSON:
{{"top_twt_ms": <number>, "base_twt_ms": <number>, "candidate_ids": ["C1", ...],
  "fluid": "gas" | "oil" | "uncertain", "confidence": "High" | "Medium" | "Low",
  "reason": "<two sentences citing GR, RHOB, NPHI and RT values>",
  "risks": "<one sentence>"}}"""
    pick = parse_json_reply(gemma_chat(GEMMA_SYSTEM, user, max_tokens=800))
    top, base = float(pick["top_twt_ms"]), float(pick["base_twt_ms"])
    if not base > top:
        raise ValueError("Gemma's base isn't below its top")
    pick["top_twt_ms"], pick["base_twt_ms"] = top, base
    return pick


# ----------------------------------------------------------------- ONE WELL
def process_well(las_path, checkshot_path, deviation_path=None, kb=None, checkshot=None,
                 reservoir=None, use_gemma=True):
    """Everything for one well. Returns a dict:
        name, header (well name / kb / x / y from the LAS), curves (column names),
        logs (DataFrame: MD, TVD_KB, TVDSS, TWT_MS, NORTH, EAST, the four curves,
              baselines, cut-offs, PAY), checkshot, candidates, cutoffs,
        pick (top/base TWT + MD, source, and Gemma's reasons when it answered),
        notes (warnings and what happened, as short sentences)."""
    cfg = dict(RESERVOIR_DEFAULTS, **(reservoir or {}))
    logs, header = read_las(las_path)
    info = header_info(header, Path(las_path).stem)
    c = detect_curves(list(logs.columns))
    if c["depth"] is None:
        c["depth"] = logs.columns[0]  # LAS files put depth first
    notes = []
    kb = kb if kb is not None else info["kb"]
    if kb is None:
        raise ValueError("no KB elevation in the LAS header (EKB / EREF): type it in the KB elevation box")
    dev = Deviation(deviation_path)
    if deviation_path is None:
        notes.append("no deviation survey: the well is treated as vertical")
    cs, cs_warnings = load_checkshot(checkshot_path, kb, dev, checkshot)
    notes += cs_warnings

    logs = logs[logs[c["depth"]].notna()].sort_values(c["depth"]).reset_index(drop=True)
    tvd, north, east = dev(logs[c["depth"]].values)
    logs["TVD_KB"], logs["NORTH"], logs["EAST"] = tvd, north, east
    logs["TVDSS"] = logs.TVD_KB - kb
    logs["TWT_MS"] = tvdss_to_twt(logs.TVDSS.values, cs)

    params = flag_reservoir_twt(logs, c, cfg)
    cand = group_intervals_twt(logs, c, cfg)
    pick = None
    if len(cand) and use_gemma:
        try:
            pick = ask_gemma(info["name"], cand, params)
            pick["source"] = "Gemma"
        except ImportError as exc:
            if getattr(exc, "name", None) == "openai":
                notes.append("Gemma not asked: the openai package isn't installed in the Python running the app "
                             "(python -m pip install openai, then restart the app)")
            else:  # some other import failed: say which, rather than blaming openai
                notes.append(f"Gemma not asked: an import failed ({type(exc).__name__}: {str(exc)[:160]})")
        except Exception as exc:
            notes.append(f"Gemma couldn't be used ({type(exc).__name__}: {str(exc)[:120]}), "
                         "so the top-ranked interval was taken")
    if pick is None and len(cand):
        b = cand.iloc[0]
        pick = dict(top_twt_ms=float(b.top_twt_ms), base_twt_ms=float(b.base_twt_ms),
                    candidate_ids=[b.id], source="rule engine")
    if pick is not None:
        order = np.argsort(logs.TWT_MS.values)
        t_sorted, md_sorted = logs.TWT_MS.values[order], logs[c["depth"]].values[order]
        pick["top_md_m"] = round(float(np.interp(pick["top_twt_ms"], t_sorted, md_sorted)), 1)
        pick["base_md_m"] = round(float(np.interp(pick["base_twt_ms"], t_sorted, md_sorted)), 1)
    else:
        notes.append("no interval passed all four tests (low GR, low density, low neutron, high resistivity)")
    return dict(name=info["name"], header=info, curves=c, logs=logs, checkshot=cs, candidates=cand,
                cutoffs=params, pick=pick, notes=notes)


if __name__ == "__main__":  # quick check from the command line: python welllogs.py WELL.las CHECKSHOT.txt
    import sys
    if len(sys.argv) < 3:
        sys.exit("usage: python welllogs.py WELL.las CHECKSHOT.txt [DEVIATION.txt]")
    r = process_well(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None,
                     use_gemma=os.environ.get("USE_GEMMA", "1") != "0")
    print(r["name"], r["header"], r["cutoffs"], sep="\n")
    print(r["candidates"].to_string(index=False) if len(r["candidates"]) else "no candidates")
    print(json.dumps(r["pick"], indent=2))
    print("\n".join(r["notes"]))
