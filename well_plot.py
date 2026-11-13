"""
Well log reservoir finder: plots the logs, then lets Gemma pick the reservoir.

  1. Reads a LAS file (or a .txt copy of one).
  2. Plots Gamma Ray, Neutron Porosity and Density overlaid in one track
     (plus resistivity) so a human can read the well.
  3. A simple rule engine (plain Python, no AI) flags every depth that is:
       - sandy, not shale       -> low Gamma Ray
       - porous                 -> density porosity high enough
       - hydrocarbon, not water -> resistivity well above the shale background
     and groups those depths into candidate intervals.
  4. Sends the candidate table + a 2 m summary of the log + the plot to Gemma,
     which picks the TOP and BASE depth of the best reservoir and explains why.
  5. Draws Gemma's pick on the plot and checks it against the petrophysicist's
     interpreted curves (VOIL / VGAS / SWE) if the file has them.

Why not send the raw LAS straight to Gemma? It is ~18,000 rows. Language
models are poor at scanning long number tables, so Python does the counting
and Gemma does the reasoning and the explanation.
"""
import base64
import io
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- SETTINGS
LAS_FILE = "/data/epic20/APGCE/input_data/challenge3/well_data/las/Well-1.las"          # .las or a .txt copy of one
WELL = "WELL-1"

# Curve names in this file (change if another well uses different names)
DEPTH, GR, NPHI, RHOB, RT = "DEPTH", "GR_P", "NEUT_P", "DENS_P", "RT_P"

# Rock/fluid constants for density porosity
RHO_MATRIX, RHO_FLUID = 2.65, 1.0   # sandstone matrix, water/mud filtrate

# Cut-offs (sensible starting points; tune per field)
VSH_MAX = 0.50        # shale volume from Gamma Ray must be below this -> "sandy"
PHI_MIN = 0.15        # density porosity must be above this            -> "porous"
RT_FACTOR = 2.0       # resistivity must be > RT_FACTOR x shale background -> "hydrocarbon"
MIN_THICK_M = 2.0     # ignore intervals thinner than this

# Shale / sand baselines on the Gamma Ray (and the shale resistivity background)
#   "constant": one baseline for the whole well
#   "rolling" : baseline follows the log down the well (handles GR drift with depth)
SHALE_BASELINE_MODE = "rolling"
ROLLING_WINDOW_M = 150        # window length for "rolling" mode
GR_SHALE_PCT, GR_SAND_PCT = 0.95, 0.05   # percentiles used to auto-pick the baselines
GR_SHALE_OVERRIDE = None      # e.g. 105 -> use your own shale baseline (gAPI) instead
GR_SAND_OVERRIDE = None       # e.g. 45  -> use your own sand baseline (gAPI) instead
MERGE_GAP_M = 1.0     # join intervals separated by less than this

GEMMA_URL = "http://172.28.34.1:8000/v1"
GEMMA_MODEL = "google/gemma-4-26B-A4B-it"
SEND_SUMMARY_LOG = True   # also send a 2 m-step summary of the whole log


# ----------------------------------------------------------------- 1. READ LAS
def read_las(path):
    """Minimal LAS 2.0/3.0 reader: curve names from ~Curve, data from ~Ascii."""
    lines = open(path, errors="ignore").read().splitlines()
    c_start = next(i for i, l in enumerate(lines) if l.upper().startswith("~C"))
    a_start = next(i for i, l in enumerate(lines) if l.upper().startswith("~A"))
    names = [l.split(".")[0].strip() for l in lines[c_start + 1:a_start]
             if l.strip() and not l.lstrip().startswith("#")]
    null = -9999.0
    for l in lines[:c_start]:
        if l.strip().upper().startswith("NULL"):
            null = float(l.split(".", 1)[1].split(":")[0].split()[0])
    df = pd.read_csv(path, sep=r"\s+", skiprows=a_start + 1, header=None, names=names)
    return df.replace(null, np.nan)


# ----------------------------------------------------------------- 2. RULE ENGINE
def add_baselines(d):
    """Shale baseline, sand baseline and GR cut-off, plus shale resistivity background.
    Stored as curves so they can be constant or vary with depth."""
    logged = d[[GR, RHOB, RT]].notna().all(axis=1)
    gr = d[GR].where(logged)
    if SHALE_BASELINE_MODE == "rolling":
        win = max(int(ROLLING_WINDOW_M / d[DEPTH].diff().median()), 11)
        roll = gr.rolling(win, center=True, min_periods=win // 4)
        d["GR_SHALE_BASE"] = roll.quantile(GR_SHALE_PCT)
        d["GR_SAND_BASE"] = roll.quantile(GR_SAND_PCT)
        shale = gr > roll.quantile(0.75)
        d["RT_SHALE_BASE"] = d[RT].where(shale).rolling(win, center=True, min_periods=5).median()
    else:
        d["GR_SHALE_BASE"] = gr.quantile(GR_SHALE_PCT)
        d["GR_SAND_BASE"] = gr.quantile(GR_SAND_PCT)
        d["RT_SHALE_BASE"] = d.loc[gr > gr.quantile(0.75), RT].median()
    if GR_SHALE_OVERRIDE is not None:
        d["GR_SHALE_BASE"] = float(GR_SHALE_OVERRIDE)
    if GR_SAND_OVERRIDE is not None:
        d["GR_SAND_BASE"] = float(GR_SAND_OVERRIDE)
    for c in ["GR_SHALE_BASE", "GR_SAND_BASE", "RT_SHALE_BASE"]:
        d[c] = d[c].ffill().bfill()
    d["GR_CUTOFF"] = d["GR_SAND_BASE"] + VSH_MAX * (d["GR_SHALE_BASE"] - d["GR_SAND_BASE"])
    d["RT_CUTOFF"] = RT_FACTOR * d["RT_SHALE_BASE"]
    return d, logged


def _rng(series, nd=1):
    lo, hi = round(float(series.min()), nd), round(float(series.max()), nd)
    return lo if lo == hi else f"{lo} to {hi} (varies with depth)"


def flag_reservoir(df):
    d, logged = add_baselines(df.copy())

    d["VSH_GR"] = ((d[GR] - d["GR_SAND_BASE"]) /
                   (d["GR_SHALE_BASE"] - d["GR_SAND_BASE"])).clip(0, 1)
    d["PHI_D"] = ((RHO_MATRIX - d[RHOB]) / (RHO_MATRIX - RHO_FLUID)).clip(0, 0.45)

    d["SANDY"] = d["VSH_GR"] < VSH_MAX            # same as GR < GR_CUTOFF
    d["POROUS"] = d["PHI_D"] > PHI_MIN
    d["HC"] = d[RT] > d["RT_CUTOFF"]
    d["PAY"] = logged & d["SANDY"] & d["POROUS"] & d["HC"]

    z = d[logged]
    params = dict(baseline_mode=SHALE_BASELINE_MODE,
                  gr_shale_baseline=_rng(z["GR_SHALE_BASE"]),
                  gr_sand_baseline=_rng(z["GR_SAND_BASE"]),
                  gr_cutoff=_rng(z["GR_CUTOFF"]),
                  rt_shale_background=_rng(z["RT_SHALE_BASE"], 2),
                  rt_cutoff=_rng(z["RT_CUTOFF"], 2),
                  vsh_max=VSH_MAX, phi_min=PHI_MIN)
    return d, params


def group_intervals(d):
    depth, pay = d[DEPTH].values, d["PAY"].values
    runs, start = [], None
    for i, p in enumerate(pay):
        if p and start is None:
            start = i
        if start is not None and (not p or i == len(pay) - 1):
            end = i if p else i - 1
            runs.append([depth[start], depth[end]])
            start = None
    merged = []                                   # join small gaps
    for top, base in runs:
        if merged and top - merged[-1][1] <= MERGE_GAP_M:
            merged[-1][1] = base
        else:
            merged.append([top, base])

    rows = []
    for top, base in merged:
        if base - top < MIN_THICK_M:
            continue
        z = d[(d[DEPTH] >= top) & (d[DEPTH] <= base)]
        rows.append(dict(
            top_m=round(top, 1), base_m=round(base, 1), thickness_m=round(base - top, 1),
            net_pay_fraction=round(z["PAY"].mean(), 2),
            gr_api=round(z[GR].mean(), 1), vsh=round(z["VSH_GR"].mean(), 2),
            phi_density=round(z["PHI_D"].mean(), 3),
            nphi=round(z[NPHI].mean(), 3) if z[NPHI].notna().any() else None,
            rhob=round(z[RHOB].mean(), 3), rt_ohmm=round(z[RT].mean(), 2),
            rt_max_ohmm=round(z[RT].max(), 2)))
    cand = pd.DataFrame(rows)
    if len(cand):
        # simple ranking hint for Gemma: thick, porous, clean, resistive
        cand["score"] = (cand.thickness_m * cand.net_pay_fraction * cand.phi_density
                         * (1 - cand.vsh) * np.log(cand.rt_ohmm)).round(3)
        cand = cand.sort_values("score", ascending=False).reset_index(drop=True)
        cand.insert(0, "id", [f"C{i + 1}" for i in range(len(cand))])
    return cand


# ----------------------------------------------------------------- 3. PLOT
def plot_logs(d, cand=None, pick=None, depth_range=None, title=None):
    if depth_range is None:
        ok = d[[NPHI, RHOB]].notna().all(axis=1)
        depth_range = (d.loc[ok, DEPTH].min() - 20, d.loc[ok, DEPTH].max() + 20)
    z = d[(d[DEPTH] >= depth_range[0]) & (d[DEPTH] <= depth_range[1])]
    y = z[DEPTH]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 14), sharey=True,
                                   gridspec_kw=dict(width_ratios=[2.2, 1]))

    # Track 1: GR + NPHI + RHOB overlaid, each on its own top axis
    ax1.plot(z[GR], y, color="green", lw=0.7, label="Gamma Ray")
    ax1.plot(z["GR_SHALE_BASE"], y, color="dimgray", lw=1.8, ls="--", label="Shale baseline")
    ax1.plot(z["GR_SAND_BASE"], y, color="darkgoldenrod", lw=1.8, ls=":", label="Sand baseline")
    ax1.plot(z["GR_CUTOFF"], y, color="darkorange", lw=1.2, ls="-.",
             label=f"GR cut-off (Vsh = {VSH_MAX})")
    ax1.fill_betweenx(y, z[GR], z["GR_CUTOFF"], where=z[GR] < z["GR_CUTOFF"],
                      color="sandybrown", alpha=0.35, lw=0, label="Sand (GR < cut-off)")
    ax1.set_xlim(0, 150)
    ax1.set_xlabel("Gamma Ray (gAPI)  0 → 150", color="green")
    ax1.tick_params(axis="x", colors="green")

    axn = ax1.twiny()
    axn.plot(z[NPHI], y, color="blue", lw=0.7)
    axn.set_xlim(0.45, -0.15)                     # standard reversed neutron scale
    axn.set_xlabel("Neutron porosity (v/v)  0.45 → -0.15", color="blue")
    axn.tick_params(axis="x", colors="blue")

    axd = ax1.twiny()
    axd.spines["top"].set_position(("outward", 40))
    axd.plot(z[RHOB], y, color="red", lw=0.7)
    axd.set_xlim(1.95, 2.95)                      # standard density scale
    axd.set_xlabel("Bulk density (g/cc)  1.95 → 2.95", color="red")
    axd.tick_params(axis="x", colors="red")

    # Shade neutron-density crossover (density left of neutron = light fluid / gas sign)
    nphi_on_rho = 1.95 + (0.45 - z[NPHI]) / 0.60 * 1.0   # neutron mapped onto density axis
    axd.fill_betweenx(y, z[RHOB], nphi_on_rho, where=z[RHOB] < nphi_on_rho,
                      color="gold", alpha=0.5, label="N-D crossover")

    # Track 2: resistivity (log scale)
    ax2.semilogx(z[RT], y, color="black", lw=0.7)
    ax2.plot(z["RT_SHALE_BASE"], y, color="dimgray", lw=1.8, ls="--", label="Shale RT background")
    ax2.plot(z["RT_CUTOFF"], y, color="darkorange", lw=1.2, ls="-.",
             label=f"HC cut-off ({RT_FACTOR}x shale)")
    ax2.fill_betweenx(y, z["RT_CUTOFF"], z[RT], where=z[RT] > z["RT_CUTOFF"],
                      color="tomato", alpha=0.35, lw=0)
    ax2.legend(loc="lower right", fontsize=7)
    ax2.set_xlim(0.2, 200)
    ax2.set_xlabel("Resistivity RT (ohm.m)")
    ax2.grid(True, which="both", alpha=0.3)

    # Candidate intervals from the rule engine (light green bands)
    if cand is not None and len(cand):
        inside = cand[(cand.base_m >= depth_range[0]) & (cand.top_m <= depth_range[1])]
        for _, c in inside.iterrows():
            for a in (ax1, ax2):
                a.axhspan(c.top_m, c.base_m, color="lightgreen", alpha=0.35, lw=0)
            ax2.text(150, (c.top_m + c.base_m) / 2, c.id, fontsize=7, va="center")

    # Gemma's pick (bold box)
    if pick:
        for a in (ax1, ax2):
            a.axhspan(pick["top_m"], pick["base_m"], facecolor="none",
                      edgecolor="magenta", lw=2.5)
            a.axhline(pick["top_m"], color="magenta", lw=1.5, ls="--")
            a.axhline(pick["base_m"], color="magenta", lw=1.5, ls="--")
        ax2.text(0.25, pick["top_m"], f" top {pick['top_m']} m", color="magenta",
                 fontsize=9, va="bottom", weight="bold")
        ax2.text(0.25, pick["base_m"], f" base {pick['base_m']} m", color="magenta",
                 fontsize=9, va="top", weight="bold")

    ax1.set_ylim(depth_range[1], depth_range[0])
    ax1.set_ylabel("Depth (m MD)")
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = axd.get_legend_handles_labels()
    axd.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7)
    fig.suptitle(title or f"{WELL}: GR / Neutron / Density overlay + Resistivity", y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.10, right=0.97, wspace=0.08)
    return fig


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------- 4. GEMMA
SYSTEM = """You are a senior petrophysicist. You identify hydrocarbon reservoirs from well logs.
Rules of thumb you must apply:
- Sand (reservoir rock) has LOW gamma ray; shale has HIGH gamma ray.
- Porous rock has LOW bulk density; neutron-density crossover (density reading
  lighter than neutron suggests) points to light hydrocarbon, especially gas.
- Hydrocarbon-bearing rock has HIGH resistivity; water-bearing sand has LOW
  resistivity similar to or below the shale background.
- A good reservoir is all three at once: sandy, porous, and resistive.
Use ONLY the numbers given. Never invent depths or values."""


def ask_gemma(cand, params, d, plot_b64):
    import openai
    client = openai.OpenAI(base_url=GEMMA_URL, api_key="token-not-needed")

    summary = ""
    if SEND_SUMMARY_LOG:
        logged = d[d[[GR, RHOB, RT]].notna().all(axis=1)]
        bins = (logged[DEPTH] // 2) * 2
        s = logged.groupby(bins)[[GR, NPHI, RHOB, RT]].mean().round(3)
        summary = "\nWHOLE-WELL SUMMARY (2 m averages; depth, GR, NPHI, RHOB, RT):\n" + \
                  s.to_csv(header=False)

    user = f"""Well: {WELL}. Depths are measured depth in metres.

Cut-offs our rule engine used: {json.dumps(params)}

CANDIDATE INTERVALS that passed all three tests (sandy + porous + resistive),
ranked by a simple score (higher = better):
{cand.to_csv(index=False)}
{summary}
The attached image shows the logs (GR green, neutron blue, density red, crossover
shaded gold, resistivity black). Grey dashed lines are the shale baselines (GR and
resistivity), orange dash-dot lines are the cut-offs, sand is shaded brown on the GR
track and resistive intervals are shaded red. Candidates are green bands.

Task: choose the SINGLE best hydrocarbon reservoir in this well and give its top
and base depth. You may tighten or merge adjacent candidates if the logs justify it.

Reply with ONLY this JSON, no other text:
{{"top_m": <number>, "base_m": <number>, "candidate_ids": ["C1", ...],
  "fluid": "gas" | "oil" | "uncertain",
  "confidence": "High" | "Medium" | "Low",
  "why_sandy": "<one sentence citing GR>",
  "why_porous": "<one sentence citing density / neutron>",
  "why_hydrocarbon_not_water": "<one sentence citing resistivity>",
  "risks": "<one sentence>",
  "other_zones_worth_testing": ["<id: short reason>", ...]}}"""

    resp = client.chat.completions.create(
        model=GEMMA_MODEL, temperature=0.1, max_tokens=1200,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [
                      {"type": "text", "text": user},
                      {"type": "image_url",
                       "image_url": {"url": f"data:image/png;base64,{plot_b64}"}}]}])
    text = resp.choices[0].message.content
    match = re.search(r"\{.*\}", text, re.S)
    return json.loads(match.group(0)), text


# ----------------------------------------------------------------- 5. VALIDATE
def validate(d, pick):
    """Compare the AI pick with the petrophysicist's interpreted curves, if present."""
    if not {"VOIL", "VGAS", "SWE"}.issubset(d.columns):
        return None
    z = d[(d[DEPTH] >= pick["top_m"]) & (d[DEPTH] <= pick["base_m"])]
    hc = (z["VOIL"].fillna(0) + z["VGAS"].fillna(0)) > 0.01
    return dict(interpreted_hc_fraction=round(hc.mean(), 2),
                mean_SWE=round(z["SWE"].mean(), 2),
                mean_PHIE=round(z["PHIE"].mean(), 3),
                mean_VGAS=round(z["VGAS"].mean(), 3),
                mean_VOIL=round(z["VOIL"].mean(), 3))


# ----------------------------------------------------------------- RUN
if __name__ == "__main__":
    df = read_las(LAS_FILE)
    d, params = flag_reservoir(df)
    cand = group_intervals(d)
    print("Rule-engine cut-offs:", params)
    print(f"\n{len(cand)} candidate intervals:\n", cand.to_string(index=False))

    fig = plot_logs(d, cand)
    fig.savefig(f"{WELL}_logs.png", dpi=150)
    plot_b64 = fig_to_b64(fig)
    plt.show()

    try:
        pick, raw = ask_gemma(cand, params, d, plot_b64)
        source = "Gemma"
    except Exception as e:                      # Gemma unreachable or bad JSON
        print(f"\nGemma step skipped ({type(e).__name__}: {e}).")
        print("Falling back to the top-ranked rule-engine candidate.")
        best = cand.iloc[0]
        pick = dict(top_m=float(best.top_m), base_m=float(best.base_m),
                    candidate_ids=[best.id], confidence="n/a (no AI)")
        source = "rule engine"

    print(f"\n=== Reservoir pick ({source}) ===")
    print(json.dumps(pick, indent=2))
    json.dump(pick, open(f"{WELL}_reservoir_pick.json", "w"), indent=2)

    check = validate(d, pick)
    if check:
        print("\nCheck against petrophysicist's interpretation in the same interval:")
        print(json.dumps(check, indent=2))

    zoom = (pick["top_m"] - 60, pick["base_m"] + 60)
    fig2 = plot_logs(d, cand, pick, depth_range=zoom,
                     title=f"{WELL}: reservoir picked by {source} "
                           f"({pick['top_m']}–{pick['base_m']} m)")
    fig2.savefig(f"{WELL}_reservoir_pick.png", dpi=150)
    plt.show()


"""
Well logs in SEISMIC TIME: checkshot depth->TWT conversion + reservoir finder in TWT.

Self-contained: no other project files needed.

  1. Converts every log sample from depth to two-way time (TWT, ms):
       MD (log depth, from KB) -> TVD below KB (deviation survey, or vertical)
                               -> TVDSS (below sea level) -> TWT via the checkshot
  2. Finds the reservoir IN TIME. A sample counts as reservoir when ALL FOUR hold:
       - Gamma Ray LOW            -> sandy, not shale
       - Bulk Density LOW         -> porous
       - Neutron Porosity LOW     -> lower than the shale/water level: hydrocarbon
                                     (especially gas) makes the neutron read low
       - Resistivity HIGH         -> hydrocarbon, not water
     Samples that pass are grouped into intervals; the best one gives the two
     bounding values: TOP TWT and BASE TWT (ms).
  3. Optionally asks Gemma to choose/justify the final top and base (falls back
     to the top-ranked interval if Gemma is unreachable).

Outputs
  <WELL>_checkshot_qc.png           checkshot curve + interval velocity (TWT down the side)
  <WELL>_reservoir_twt.png          logs vs TWT with cut-offs, candidates and the pick
  <WELL>_logs_twt.csv               every log sample with MD, TVD_KB, TVDSS, TWT_MS + flags
  <WELL>_reservoir_candidates_twt.csv
  <WELL>_reservoir_pick_twt.json    the two bounding values (top/base TWT) + details
"""
import base64
import io
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- SETTINGS
WELL = "WELL-2"
LAS_FILE = "/data/epic20/APGCE/input_data/challenge3/well_data/las/Well-2.las"
DEPTH, GR, NPHI, RHOB, RT = "DEPTH", "GR_P", "NEUT_P", "DENS_P", "RT_P"

# --- checkshot
CHECKSHOT_FILE = "/data/epic20/APGCE/input_data/challenge3/well_data/checkshots/Well-2_Checkshot.txt"
CS_DEPTH_COL = 0          # column number (0 = first) holding depth
CS_TIME_COL = 1           # column number holding time
CS_DEPTH_TYPE = "TVDSS"   # "TVDSS" (below sea level), "TVD_KB" (below KB) or "MD" (from KB)
CS_DEPTH_SIGN = 1         # -1 if depths are written as negative numbers
CS_TIME_TYPE = "TWT"      # "TWT" (two-way) or "OWT" (one-way)
CS_TIME_UNIT = "ms"       # "ms" or "s"
TIME_SHIFT_MS = 0.0       # bulk shift if checkshot datum differs from seismic datum
DEVIATION_FILE = None     # e.g. ".../deviations/Well-2_Deviation.txt"; None = vertical
DEV_MD_COL, DEV_INC_COL, DEV_AZI_COL = 0, 1, 2
KB_ELEVATION = None       # metres above sea level; None = read EKB from the LAS header

# --- reservoir cut-offs (all applied in TIME)
BASELINE_MODE = "rolling"     # "rolling" (follows the log down) or "constant"
BASELINE_WINDOW_MS = 120      # window for "rolling" baselines
VSH_MAX = 0.50                # GR low:   shale volume from GR below this
RHO_MATRIX, RHO_FLUID = 2.65, 1.0
PHI_MIN = 0.15                # RHOB low: density porosity above this (-> RHOB < ~2.40)
NPHI_MARGIN = 0.04            # NPHI low: at least this far below the shale neutron baseline
RT_FACTOR = 2.0               # RT high:  above RT_FACTOR x shale resistivity background
MIN_THICK_MS = 2.0            # ignore intervals thinner than this (ms TWT)
MERGE_GAP_MS = 1.0            # join intervals separated by less than this (ms)

# --- Gemma (only reachable on the hackathon server)
USE_GEMMA = True
GEMMA_URL = "http://172.28.34.1:8000/v1"
GEMMA_MODEL = "google/gemma-4-26B-A4B-it"


# ----------------------------------------------------------------- READERS
def read_las(path):
    """Minimal LAS 2.0/3.0 reader: curve names from ~Curve, data from ~Ascii."""
    lines = open(path, errors="ignore").read().splitlines()
    c_start = next(i for i, l in enumerate(lines) if l.upper().startswith("~C"))
    a_start = next(i for i, l in enumerate(lines) if l.upper().startswith("~A"))
    names = [l.split(".")[0].strip() for l in lines[c_start + 1:a_start]
             if l.strip() and not l.lstrip().startswith("#")]
    null = -9999.0
    for l in lines[:c_start]:
        if l.strip().upper().startswith("NULL"):
            null = float(l.split(".", 1)[1].split(":")[0].split()[0])
    df = pd.read_csv(path, sep=r"\s+", skiprows=a_start + 1, header=None, names=names)
    return df.replace(null, np.nan)


def kb_elevation_from_las(path):
    for line in open(path, errors="ignore"):
        if line.upper().startswith("~A"):
            break
        if line.strip().upper().startswith(("EKB", "EREF")) and "." in line:
            try:
                return float(line.split(".", 1)[1].split(":")[0].split()[-1])
            except (ValueError, IndexError):
                pass
    raise ValueError("No EKB/EREF in LAS header; set KB_ELEVATION manually.")


def read_numeric_table(path, show=8):
    """Keep only rows where every token is a number (skips headers/comments)."""
    rows, header = [], []
    for line in open(path, errors="ignore"):
        tokens = line.replace(",", " ").replace("\t", " ").split()
        if not tokens:
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            if not rows:
                header.append(line.rstrip())
    widths = [len(r) for r in rows]
    width = max(set(widths), key=widths.count)
    table = np.array([r for r in rows if len(r) == width])
    if show:
        print(f"\n--- {os.path.basename(path)}: {len(table)} rows x {width} columns")
        for h in header[-3:]:
            print("   header:", h)
        print("   columns:  " + "  ".join(f"[{i}]".rjust(10) for i in range(width)))
        for r in table[:show]:
            print("             " + "  ".join(f"{v:10.3f}" for v in r))
    return table


# ----------------------------------------------------------------- MD -> TVD
def minimum_curvature(md, inc_deg, azi_deg):
    inc, azi = np.radians(inc_deg), np.radians(azi_deg)
    tvd = np.zeros_like(md)
    for i in range(1, len(md)):
        cos_dl = (np.cos(inc[i] - inc[i - 1])
                  - np.sin(inc[i - 1]) * np.sin(inc[i]) * (1 - np.cos(azi[i] - azi[i - 1])))
        dl = np.arccos(np.clip(cos_dl, -1, 1))
        rf = 1.0 if dl < 1e-9 else (2 / dl) * np.tan(dl / 2)
        tvd[i] = tvd[i - 1] + (md[i] - md[i - 1]) / 2 * (np.cos(inc[i - 1]) + np.cos(inc[i])) * rf
    return tvd


def md_to_tvd_kb(md):
    md = np.asarray(md, dtype=float)
    if DEVIATION_FILE is None:
        return md.copy()                                   # vertical well
    t = read_numeric_table(DEVIATION_FILE, show=3)
    s_md, s_inc, s_azi = t[:, DEV_MD_COL], t[:, DEV_INC_COL], t[:, DEV_AZI_COL]
    if s_md[0] > 0:
        s_md, s_inc, s_azi = np.r_[0, s_md], np.r_[s_inc[0], s_inc], np.r_[s_azi[0], s_azi]
    return np.interp(md, s_md, minimum_curvature(s_md, s_inc, s_azi))


# ----------------------------------------------------------------- CHECKSHOT
def load_checkshot(kb):
    t = read_numeric_table(CHECKSHOT_FILE)
    depth = CS_DEPTH_SIGN * t[:, CS_DEPTH_COL]
    time = t[:, CS_TIME_COL] * (1000.0 if CS_TIME_UNIT == "s" else 1.0)
    if CS_TIME_TYPE.upper() == "OWT":
        time = 2 * time
    time = time + TIME_SHIFT_MS
    if CS_DEPTH_TYPE == "MD":
        depth = md_to_tvd_kb(depth) - kb
    elif CS_DEPTH_TYPE == "TVD_KB":
        depth = depth - kb

    cs = pd.DataFrame({"TVDSS": depth, "TWT_MS": time}).dropna()
    cs = cs.sort_values("TVDSS").drop_duplicates("TVDSS")
    if (cs.TVDSS <= 0).sum() == 0:                         # sea level = time zero
        cs = pd.concat([pd.DataFrame({"TVDSS": [0.0], "TWT_MS": [0.0]}), cs])
    cs = cs.reset_index(drop=True)
    cs["VINT_MPS"] = np.r_[np.nan, 2000 * np.diff(cs.TVDSS) / np.diff(cs.TWT_MS)]
    if not np.all(np.diff(cs.TWT_MS.values) > 0):
        print("WARNING: time does not increase with depth everywhere. "
              "Check CS_DEPTH_COL / CS_TIME_COL / CS_DEPTH_SIGN.")
    bad = cs.VINT_MPS[(cs.VINT_MPS < 1400) | (cs.VINT_MPS > 6000)]
    if len(bad):
        print(f"WARNING: {len(bad)} interval velocities outside 1400-6000 m/s "
              "(usually a wrong column, unit or depth-type setting).")
    return cs


def tvdss_to_twt(tvdss, cs):
    """Linear inside the checkshot; outside, extend with the end interval velocities."""
    z, t = cs.TVDSS.values, cs.TWT_MS.values
    tvdss = np.asarray(tvdss, dtype=float)
    twt = np.interp(tvdss, z, t)
    v_top = 2000 * (z[1] - z[0]) / (t[1] - t[0])
    v_bot = 2000 * (z[-1] - z[-2]) / (t[-1] - t[-2])
    twt = np.where(tvdss < z[0], t[0] - 2000 * (z[0] - tvdss) / v_top, twt)
    twt = np.where(tvdss > z[-1], t[-1] + 2000 * (tvdss - z[-1]) / v_bot, twt)
    return twt


def plot_checkshot_qc(cs, logs):
    """TWT on the vertical axis (time increasing downwards), depth / velocity across."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    a1.plot(cs.TVDSS, cs.TWT_MS, "o", ms=4, label="checkshot")
    zz = np.linspace(0, max(cs.TVDSS.max(), np.nanmax(logs.TVDSS)), 300)
    a1.plot(zz, tvdss_to_twt(zz, cs), "-", lw=1, label="interpolated")
    a1.set_xlabel("TVDSS (m)"); a1.set_ylabel("TWT (ms)")
    a1.grid(alpha=0.3); a1.legend()
    a2.step(cs.VINT_MPS, cs.TWT_MS, where="pre", color="purple")
    a2.set_xlabel("Interval velocity (m/s)"); a2.grid(alpha=0.3)
    a1.invert_yaxis()
    fig.suptitle(f"{WELL}: checkshot time-depth QC")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- RESERVOIR IN TIME
def add_baselines(d):
    """Shale/sand GR baselines, shale neutron baseline and shale resistivity background,
    all computed along TWT (constant or rolling)."""
    logged = d[[GR, NPHI, RHOB, RT]].notna().all(axis=1)
    gr = d[GR].where(logged)
    if BASELINE_MODE == "rolling":
        step = np.nanmedian(np.diff(d.TWT_MS))
        win = max(int(BASELINE_WINDOW_MS / step), 11)
        roll = gr.rolling(win, center=True, min_periods=win // 4)
        d["GR_SHALE_BASE"], d["GR_SAND_BASE"] = roll.quantile(0.95), roll.quantile(0.05)
        shale = gr > roll.quantile(0.75)
        med = lambda s: s.where(shale).rolling(win, center=True, min_periods=5).median()
        d["NPHI_SHALE_BASE"], d["RT_SHALE_BASE"] = med(d[NPHI]), med(d[RT])
    else:
        shale = gr > gr.quantile(0.75)
        d["GR_SHALE_BASE"], d["GR_SAND_BASE"] = gr.quantile(0.95), gr.quantile(0.05)
        d["NPHI_SHALE_BASE"] = d.loc[shale, NPHI].median()
        d["RT_SHALE_BASE"] = d.loc[shale, RT].median()
    for c in ["GR_SHALE_BASE", "GR_SAND_BASE", "NPHI_SHALE_BASE", "RT_SHALE_BASE"]:
        d[c] = d[c].ffill().bfill()

    d["GR_CUTOFF"] = d.GR_SAND_BASE + VSH_MAX * (d.GR_SHALE_BASE - d.GR_SAND_BASE)
    d["RHOB_CUTOFF"] = RHO_MATRIX - PHI_MIN * (RHO_MATRIX - RHO_FLUID)
    d["NPHI_CUTOFF"] = d.NPHI_SHALE_BASE - NPHI_MARGIN
    d["RT_CUTOFF"] = RT_FACTOR * d.RT_SHALE_BASE
    return logged


def flag_reservoir_twt(d):
    logged = add_baselines(d)
    d["GR_LOW"] = d[GR] < d.GR_CUTOFF
    d["RHOB_LOW"] = d[RHOB] < d.RHOB_CUTOFF
    d["NPHI_LOW"] = d[NPHI] < d.NPHI_CUTOFF
    d["RT_HIGH"] = d[RT] > d.RT_CUTOFF
    d["PAY"] = logged & d.GR_LOW & d.RHOB_LOW & d.NPHI_LOW & d.RT_HIGH
    z = d[logged]
    rng = lambda s, n=2: (round(float(s.min()), n) if round(float(s.min()), n) == round(float(s.max()), n)
                          else f"{s.min():.{n}f} to {s.max():.{n}f}")
    return dict(baseline_mode=BASELINE_MODE, gr_cutoff_api=rng(z.GR_CUTOFF, 1),
                rhob_cutoff_gcc=round(float(z.RHOB_CUTOFF.iloc[0]), 3),
                nphi_cutoff_vv=rng(z.NPHI_CUTOFF, 3), rt_cutoff_ohmm=rng(z.RT_CUTOFF, 2))


def group_intervals_twt(d):
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
        if merged and top - merged[-1][1] <= MERGE_GAP_MS:
            merged[-1][1] = base
        else:
            merged.append([top, base])

    rows = []
    for top, base in merged:
        if base - top < MIN_THICK_MS:
            continue
        z = d[(d.TWT_MS >= top) & (d.TWT_MS <= base)]
        rows.append(dict(
            top_twt_ms=round(top, 1), base_twt_ms=round(base, 1),
            thickness_ms=round(base - top, 1),
            top_md_m=round(z[DEPTH].min(), 1), base_md_m=round(z[DEPTH].max(), 1),
            net_fraction=round(z.PAY.mean(), 2),
            gr_api=round(z[GR].mean(), 1), rhob_gcc=round(z[RHOB].mean(), 3),
            nphi_vv=round(z[NPHI].mean(), 3), rt_ohmm=round(z[RT].mean(), 2),
            rt_max_ohmm=round(z[RT].max(), 2),
            phi_density=round(((RHO_MATRIX - z[RHOB]) / (RHO_MATRIX - RHO_FLUID)).mean(), 3)))
    cand = pd.DataFrame(rows)
    if len(cand):
        cand["score"] = (cand.thickness_ms * cand.net_fraction * cand.phi_density
                         * np.log(cand.rt_ohmm.clip(lower=1.01))).round(3)
        cand = cand.sort_values("score", ascending=False).reset_index(drop=True)
        cand.insert(0, "id", [f"C{i + 1}" for i in range(len(cand))])
    return cand


def plot_logs_twt(d, cand=None, pick=None, twt_range=None, title=None):
    if twt_range is None:
        ok = d[[NPHI, RHOB]].notna().all(axis=1)
        twt_range = (d.loc[ok, "TWT_MS"].min() - 10, d.loc[ok, "TWT_MS"].max() + 10)
    z = d[(d.TWT_MS >= twt_range[0]) & (d.TWT_MS <= twt_range[1])].sort_values("TWT_MS")
    y = z.TWT_MS

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 14), sharey=True,
                                   gridspec_kw=dict(width_ratios=[2.2, 1]))
    # Track 1: GR + NPHI + RHOB overlaid, each with its cut-off
    ax1.plot(z[GR], y, color="green", lw=0.8, label="Gamma Ray")
    ax1.plot(z.GR_CUTOFF, y, color="green", lw=1.2, ls="-.", alpha=0.7, label="GR cut-off")
    ax1.fill_betweenx(y, z[GR], z.GR_CUTOFF, where=z[GR] < z.GR_CUTOFF,
                      color="sandybrown", alpha=0.3, lw=0, label="GR low (sand)")
    ax1.set_xlim(0, 150)
    ax1.set_xlabel("Gamma Ray (gAPI)  0 → 150", color="green")
    ax1.tick_params(axis="x", colors="green")

    axn = ax1.twiny()
    axn.plot(z[NPHI], y, color="blue", lw=0.8, label="Neutron")
    axn.plot(z.NPHI_CUTOFF, y, color="blue", lw=1.2, ls="-.", alpha=0.7, label="NPHI cut-off")
    axn.set_xlim(0.45, -0.15)
    axn.set_xlabel("Neutron porosity (v/v)  0.45 → -0.15", color="blue")
    axn.tick_params(axis="x", colors="blue")

    axd = ax1.twiny()
    axd.spines["top"].set_position(("outward", 40))
    axd.plot(z[RHOB], y, color="red", lw=0.8, label="Density")
    axd.plot(z.RHOB_CUTOFF, y, color="red", lw=1.2, ls="-.", alpha=0.7, label="RHOB cut-off")
    axd.set_xlim(1.95, 2.95)
    axd.set_xlabel("Bulk density (g/cc)  1.95 → 2.95", color="red")
    axd.tick_params(axis="x", colors="red")
    nphi_on_rho = 1.95 + (0.45 - z[NPHI]) / 0.60
    axd.fill_betweenx(y, z[RHOB], nphi_on_rho, where=z[RHOB] < nphi_on_rho,
                      color="gold", alpha=0.5, lw=0, label="N-D crossover")

    # Track 2: resistivity
    ax2.semilogx(z[RT], y, color="black", lw=0.8)
    ax2.plot(z.RT_SHALE_BASE, y, color="dimgray", lw=1.5, ls="--", label="Shale RT background")
    ax2.plot(z.RT_CUTOFF, y, color="darkorange", lw=1.2, ls="-.", label="RT cut-off")
    ax2.fill_betweenx(y, z.RT_CUTOFF, z[RT], where=z[RT] > z.RT_CUTOFF,
                      color="tomato", alpha=0.35, lw=0)
    ax2.set_xlim(0.2, 200)
    ax2.set_xlabel("Resistivity RT (ohm.m)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="lower right", fontsize=7)

    if cand is not None and len(cand):
        inside = cand[(cand.base_twt_ms >= twt_range[0]) & (cand.top_twt_ms <= twt_range[1])]
        for _, c in inside.iterrows():
            for a in (ax1, ax2):
                a.axhspan(c.top_twt_ms, c.base_twt_ms, color="lightgreen", alpha=0.35, lw=0)
            ax2.text(150, (c.top_twt_ms + c.base_twt_ms) / 2, c.id, fontsize=7, va="center")
    if pick:
        for a in (ax1, ax2):
            a.axhspan(pick["top_twt_ms"], pick["base_twt_ms"], facecolor="none",
                      edgecolor="magenta", lw=2.5)
        ax2.text(0.25, pick["top_twt_ms"], f" top {pick['top_twt_ms']} ms",
                 color="magenta", fontsize=9, va="bottom", weight="bold")
        ax2.text(0.25, pick["base_twt_ms"], f" base {pick['base_twt_ms']} ms",
                 color="magenta", fontsize=9, va="top", weight="bold")

    ax1.set_ylim(twt_range[1], twt_range[0])
    ax1.set_ylabel("TWT (ms)")
    ax1.grid(True, alpha=0.3)
    handles = sum((a.get_legend_handles_labels()[0] for a in (ax1, axn, axd)), [])
    axd.legend(handles=handles, loc="lower left", fontsize=7)
    fig.suptitle(title or f"{WELL}: logs in seismic time (TWT)", y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.10, right=0.97, wspace=0.08)
    return fig


# ----------------------------------------------------------------- GEMMA
def ask_gemma(cand, params, fig):
    import openai
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80)
    img = base64.b64encode(buf.getvalue()).decode()
    system = ("You are a senior petrophysicist. A reservoir needs ALL of: low gamma ray (sand), "
              "low bulk density (porous), low neutron porosity relative to shale (hydrocarbon, "
              "especially gas), and high resistivity (hydrocarbon, not water). "
              "Use ONLY the numbers given; never invent values.")
    user = f"""Well {WELL}. All depths are in seismic two-way time (TWT, ms).
Cut-offs used: {json.dumps(params)}

Candidate intervals that passed all four tests, ranked by a simple score:
{cand.to_csv(index=False)}
The image shows the logs against TWT with cut-offs (dash-dot) and candidates (green).

Choose the SINGLE best reservoir and give its two bounding values in TWT.
Reply with ONLY this JSON:
{{"top_twt_ms": <number>, "base_twt_ms": <number>, "candidate_ids": ["C1", ...],
  "fluid": "gas" | "oil" | "uncertain", "confidence": "High" | "Medium" | "Low",
  "reason": "<two sentences citing GR, RHOB, NPHI and RT values>",
  "risks": "<one sentence>"}}"""
    client = openai.OpenAI(base_url=GEMMA_URL, api_key="token-not-needed")
    r = client.chat.completions.create(
        model=GEMMA_MODEL, temperature=0.1, max_tokens=800,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": [
                      {"type": "text", "text": user},
                      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}])
    return json.loads(re.search(r"\{.*\}", r.choices[0].message.content, re.S).group(0))


# ----------------------------------------------------------------- RUN
if __name__ == "__main__":
    # 1. depth -> time
    kb = KB_ELEVATION if KB_ELEVATION is not None else kb_elevation_from_las(LAS_FILE)
    print(f"KB elevation above sea level: {kb} m")
    cs = load_checkshot(kb)

    logs = read_las(LAS_FILE)
    logs["TVD_KB"] = md_to_tvd_kb(logs[DEPTH].values)
    logs["TVDSS"] = logs.TVD_KB - kb
    logs["TWT_MS"] = tvdss_to_twt(logs.TVDSS.values, cs)

    fig_qc = plot_checkshot_qc(cs, logs)
    fig_qc.savefig(f"{WELL}_checkshot_qc.png", dpi=150)
    plt.show()

    # 2. reservoir in time
    params = flag_reservoir_twt(logs)
    cand = group_intervals_twt(logs)
    logs.to_csv(f"{WELL}_logs_twt.csv", index=False)
    cand.to_csv(f"{WELL}_reservoir_candidates_twt.csv", index=False)
    print("\nCut-offs:", params)
    print(f"\n{len(cand)} candidate intervals (in TWT):")
    print(cand.to_string(index=False))

    # 3. pick the two bounding values (Gemma, or fallback to the top-ranked candidate)
    fig_all = plot_logs_twt(logs, cand)
    pick, source = None, "rule engine"
    if USE_GEMMA and len(cand):
        try:
            pick, source = ask_gemma(cand, params, fig_all), "Gemma"
        except Exception as e:
            print(f"\nGemma step skipped ({type(e).__name__}: {e}).")
    plt.close(fig_all)
    if pick is None and len(cand):
        b = cand.iloc[0]
        pick = dict(top_twt_ms=float(b.top_twt_ms), base_twt_ms=float(b.base_twt_ms),
                    candidate_ids=[b.id])
    if pick is None:
        raise SystemExit("No interval passed all four tests. Loosen the cut-offs in SETTINGS.")

    pick["source"] = source
    at = lambda t: float(np.interp(t, logs.TWT_MS, logs[DEPTH]))    # TWT -> MD for reference
    pick["top_md_m"], pick["base_md_m"] = round(at(pick["top_twt_ms"]), 1), round(at(pick["base_twt_ms"]), 1)
    json.dump(pick, open(f"{WELL}_reservoir_pick_twt.json", "w"), indent=2)
    print(f"\n=== Reservoir ({source}) ===")
    print(f"TOP  = {pick['top_twt_ms']} ms TWT   (≈ {pick['top_md_m']} m MD)")
    print(f"BASE = {pick['base_twt_ms']} ms TWT   (≈ {pick['base_md_m']} m MD)")
    print(json.dumps(pick, indent=2))

    zoom = (pick["top_twt_ms"] - 50, pick["base_twt_ms"] + 50)
    fig = plot_logs_twt(logs, cand, pick, twt_range=zoom,
                        title=f"{WELL}: reservoir in TWT ({source}): "
                              f"{pick['top_twt_ms']}–{pick['base_twt_ms']} ms")
    fig.savefig(f"{WELL}_reservoir_twt.png", dpi=150)
    plt.show()


"""
Smooth well-log curves for cleaner display (separate from the reservoir finder).

Two steps per curve:
  1. Despike : a short running median removes single-sample spikes
               (bad hole, tool glitches) without blurring bed boundaries much.
  2. Smooth  : np.convolve with a smoothing kernel ("hanning", "gaussian" or
               "boxcar") over a window given in METRES, so it behaves the same
               whatever the log step.
Resistivity is smoothed in log10 space (it varies by orders of magnitude).

Self-contained: no other project files needed.

Outputs
  <WELL>_smoothed_logs.png   raw curves (faint) with smoothed curves (bold) on top
  <WELL>_logs_smoothed.csv   all curves, with the smoothed ones replacing the originals

To use the smoothed curves in the reservoir finder as well, run this file first
(or its cell), then in well_reservoir_gemma.py right after `df = read_las(LAS_FILE)`:
    df = smooth_logs(df)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_las(path):
    """Minimal LAS 2.0/3.0 reader: curve names from ~Curve, data from ~Ascii."""
    lines = open(path, errors="ignore").read().splitlines()
    c_start = next(i for i, l in enumerate(lines) if l.upper().startswith("~C"))
    a_start = next(i for i, l in enumerate(lines) if l.upper().startswith("~A"))
    names = [l.split(".")[0].strip() for l in lines[c_start + 1:a_start]
             if l.strip() and not l.lstrip().startswith("#")]
    null = -9999.0
    for l in lines[:c_start]:
        if l.strip().upper().startswith("NULL"):
            null = float(l.split(".", 1)[1].split(":")[0].split()[0])
    df = pd.read_csv(path, sep=r"\s+", skiprows=a_start + 1, header=None, names=names)
    return df.replace(null, np.nan)

# ----------------------------------------------------------------- SETTINGS
WELL = "WELL-1"
LAS_FILE = "/data/epic20/APGCE/input_data/challenge3/well_data/las/Well-1.las"
DEPTH, GR, NPHI, RHOB, RT = "DEPTH", "GR_P", "NEUT_P", "DENS_P", "RT_P"
CURVES = [GR, NPHI, RHOB, RT]
LOG_SCALE_CURVES = [RT]          # smoothed in log10 space

KERNEL = "hanning"               # "hanning", "gaussian" or "boxcar" (plain moving average)
WINDOW_M = 5.0                   # smoothing window (metres). Bigger = smoother
DESPIKE_SAMPLES = 5              # running-median length (odd number of samples)

DEPTH_RANGE = None               # e.g. (1550, 1700) to zoom; None = where N + D exist


# ----------------------------------------------------------------- SMOOTHING
def convolve_smooth(x, win, kernel):
    """Weighted moving average with np.convolve.
    Edges are re-normalised so the ends of the log don't droop towards zero."""
    x = np.asarray(x, dtype=float)
    win = min(win, len(x) if len(x) % 2 else len(x) - 1)   # window can't exceed segment
    if win < 3:
        return x
    if kernel == "boxcar":                      # plain moving average
        k = np.ones(win)
    elif kernel == "hanning":                   # cosine bell: smooth, no ringing
        k = np.hanning(win + 2)[1:-1]
    elif kernel == "gaussian":                  # bell curve, window = +/- 3 sigma
        n = np.arange(win) - (win - 1) / 2
        k = np.exp(-0.5 * (n / (win / 6)) ** 2)
    else:
        raise ValueError(f"unknown kernel {kernel!r}")
    k = k / k.sum()
    return np.convolve(x, k, mode="same") / np.convolve(np.ones_like(x), k, mode="same")


def smooth_curve(depth, values, window_m=WINDOW_M, kernel=KERNEL, log=False):
    step = np.nanmedian(np.diff(depth))
    win = max(int(round(window_m / step)) | 1, 3)          # odd, at least 3 samples
    v = pd.Series(values, dtype=float)
    if log:
        v = np.log10(v.where(v > 0))
    v = v.rolling(DESPIKE_SAMPLES, center=True, min_periods=1).median()   # despike

    out = np.full(len(v), np.nan)
    ok = v.notna().values
    # smooth each continuous stretch separately, so gaps are not bridged
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        out[start:stop] = convolve_smooth(v.values[start:stop], win, kernel)
    return 10 ** out if log else out


def smooth_logs(df, curves=CURVES, window_m=WINDOW_M, kernel=KERNEL):
    """Return a copy of df with the chosen curves smoothed (same column names)."""
    s = df.copy()
    for c in curves:
        if c in s:
            s[c] = smooth_curve(s[DEPTH].values, s[c].values, window_m, kernel,
                                log=c in LOG_SCALE_CURVES)
    return s


# ----------------------------------------------------------------- PLOT
def plot_raw_vs_smooth(raw, sm, depth_range=None):
    if depth_range is None:
        ok = raw[[NPHI, RHOB]].notna().all(axis=1)
        depth_range = (raw.loc[ok, DEPTH].min(), raw.loc[ok, DEPTH].max())
    m = (raw[DEPTH] >= depth_range[0]) & (raw[DEPTH] <= depth_range[1])
    r, s, y = raw[m], sm[m], raw.loc[m, DEPTH]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 14), sharey=True,
                                   gridspec_kw=dict(width_ratios=[2.2, 1]))
    faint = dict(lw=0.5, alpha=0.25)
    bold = dict(lw=1.1)

    ax1.plot(r[GR], y, color="green", **faint)
    ax1.plot(s[GR], y, color="green", **bold, label="Gamma Ray")
    ax1.set_xlim(0, 150)
    ax1.set_xlabel("Gamma Ray (gAPI)  0 → 150", color="green")
    ax1.tick_params(axis="x", colors="green")

    axn = ax1.twiny()
    axn.plot(r[NPHI], y, color="blue", **faint)
    axn.plot(s[NPHI], y, color="blue", **bold, label="Neutron")
    axn.set_xlim(0.45, -0.15)
    axn.set_xlabel("Neutron porosity (v/v)  0.45 → -0.15", color="blue")
    axn.tick_params(axis="x", colors="blue")

    axd = ax1.twiny()
    axd.spines["top"].set_position(("outward", 40))
    axd.plot(r[RHOB], y, color="red", **faint)
    axd.plot(s[RHOB], y, color="red", **bold, label="Density")
    axd.set_xlim(1.95, 2.95)
    axd.set_xlabel("Bulk density (g/cc)  1.95 → 2.95", color="red")
    axd.tick_params(axis="x", colors="red")
    nphi_on_rho = 1.95 + (0.45 - s[NPHI]) / 0.60
    axd.fill_betweenx(y, s[RHOB], nphi_on_rho, where=s[RHOB] < nphi_on_rho,
                      color="gold", alpha=0.5, lw=0, label="N-D crossover")

    ax2.semilogx(r[RT], y, color="black", **faint)
    ax2.semilogx(s[RT], y, color="black", **bold)
    ax2.set_xlim(0.2, 200)
    ax2.set_xlabel("Resistivity RT (ohm.m)")
    ax2.grid(True, which="both", alpha=0.3)

    ax1.set_ylim(depth_range[1], depth_range[0])
    ax1.set_ylabel("Depth (m MD)")
    ax1.grid(True, alpha=0.3)
    handles = sum((a.get_legend_handles_labels()[0] for a in (ax1, axn, axd)), [])
    axd.legend(handles=handles, loc="lower left", fontsize=8)
    fig.suptitle(f"{WELL}: smoothed logs ({KERNEL}, {WINDOW_M} m window); "
                 "faint = raw", y=0.99)
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.10, right=0.97, wspace=0.08)
    return fig


# ----------------------------------------------------------------- RUN
if __name__ == "__main__":
    raw = read_las(LAS_FILE)
    sm = smooth_logs(raw)
    sm.to_csv(f"{WELL}_logs_smoothed.csv", index=False)

    for c in CURVES:
        noise_before = raw[c].diff().abs().median()
        noise_after = sm[c].diff().abs().median()
        print(f"{c:8s} sample-to-sample jitter: {noise_before:.4g} -> {noise_after:.4g}")

    fig = plot_raw_vs_smooth(raw, sm, DEPTH_RANGE)
    fig.savefig(f"{WELL}_smoothed_logs.png", dpi=150)
    plt.show()