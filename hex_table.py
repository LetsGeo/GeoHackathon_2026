"""
The final hexagon table: attributes calculated on every trace, summarised per hexagon as

    hexagon | centre IL | centre CL | sealing fault probability % | trace %

(the input format of sealing_faults.py), plus a details file with everything behind it.

    Jupyter:   %run hex_table.py
    Terminal:  python hex_table.py

Per trace (in WINDOW):
  - fault attribute (discontinuity): 1 - how well the baseline trace matches its next inline and
    next crossline neighbour, best over a small shift (MAX_DIP_LAG_MS, so gentle dip doesn't
    count). A trace more than FAULT_Z robust standard deviations above the survey's median is
    "on a fault".
  - 4D change: RMS(base) - RMS(mon), as in hex_rms.py.
Per hexagon:
  - trace %: share of the hexagon's traces that are on a fault (how much of it the fault covers)
  - fault probability: how strongly those traces flag x how well they line up as one fault
  - sealing fault probability %: fault probability x the 4D evidence across the fault:
        change on one side only  -> x seal contrast (high)
        change on both sides     -> x seal contrast (low: the CO2 crosses it)
        no 4D change near it     -> x NO_4D_WEIGHT (can't be tested)

Writes to OUT_DIR:
    hexagon_table.csv          the table above (numbers only, so sealing_faults.py can read it)
    hexagon_table_details.csv  + ring, traces, fault probability, seal contrast, evidence, mean
                               RMS(base) - RMS(mon), fault trend, and Gemma's scores if
                               gemma_ilcl.py has been run (GEMMA_SCORES)
The grid and hexagon IDs are the same as polygon_ilcl_pics.py / hex_rms.py (0 = middle).
"""
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # pasted into a notebook cell: run it from the repo folder
    sys.path.insert(0, str(Path.cwd()))
from hex_rms import (BASE_PATH, CENTER_ILINE, CENTER_XLINE, MONITOR_PATH, N_RINGS, SCALE,  # noqa: E402
                     hexagon_grid, rms, traces_in_hexagon)
from volume3d import load_cache_or_load_and_align_3d_pair  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
OUT_DIR = "/data/epic20/APGCE/apgce_team_07/friday/hex_table"
GEMMA_SCORES = "/data/epic20/APGCE/apgce_team_07/friday/ilcl_pics/gemma/scores.csv"  # merged in if it exists

WINDOW = None          # (t0_ms, t1_ms) to analyse, e.g. the reservoir (800, 1100); None = whole trace
MAX_DIP_LAG_MS = 8     # neighbouring traces may be shifted this much (dip) before it counts as a break
FAULT_Z = 4.0          # discontinuity this many robust SDs above the survey median = trace on a fault
MIN_FAULT_TRACES = 10  # fewer fault traces than this in a hexagon = no fault
SIDE_GAP_LINES = 2     # traces this close to the fault line belong to neither side
NO_4D_WEIGHT = 0.25    # share of the fault probability kept when no 4D change reaches the fault
CHANGE_Z = 3.0         # |RMS(base) - RMS(mon)| this many robust SDs above its median = a trace that changed
MIN_CHANGED_SHARE = 0.10  # a side of the fault with fewer changed traces than this has no 4D change
ID_START = 0           # first hexagon number (0 = middle, as in the picture names); 1 to start at 1

TABLE_COLUMNS = ["hexagon", "centre_il", "centre_xl", "sealing_fault_probability", "trace_pct"]


# ---------------------------------------------------------------------------
# Per-trace attributes
# ---------------------------------------------------------------------------
def _best_corr(X, Y, max_lag):
    """Normalised correlation of traces X and Y (..., n_t), best over shifts -max_lag..max_lag."""
    n = X.shape[-1]
    L = max(0, min(max_lag, n // 4))
    best = None
    with np.errstate(divide="ignore", invalid="ignore"):
        for lag in range(-L, L + 1):
            x = X[..., L:n - L]
            y = Y[..., L + lag:n - L + lag]
            c = np.sum(x * y, -1) / np.sqrt(np.sum(x * x, -1) * np.sum(y * y, -1))
            best = c if best is None else np.fmax(best, c)
    return best


def trace_attributes(base_vol, mon_vol, k0, k1, max_lag, block=32):
    """Per-trace maps (n_il, n_xl): discontinuity (1 - worst neighbour match, baseline) and
    RMS(base) - RMS(mon), over samples k0:k1. Dead (all-zero) traces are NaN."""
    n_il, n_xl = len(base_vol.ilines), len(base_vol.xlines)
    disc = np.full((n_il, n_xl), np.nan, np.float32)
    drms = np.full((n_il, n_xl), np.nan, np.float32)
    for a in range(0, n_il, block):
        b = min(a + block, n_il)
        bx = min(b + 1, n_il)  # one more inline for the inline neighbour
        raw = np.asarray(base_vol.data[a:bx, :, k0:k1], dtype=np.float32)
        M = np.asarray(mon_vol.data[a:b, :, k0:k1], dtype=np.float32)
        rows = b - a
        # compare only samples that are finite in both surveys (some exports have NaN / inf
        # samples inside live traces); a trace needs half its window usable
        ok = np.isfinite(raw[:rows]) & np.isfinite(M)
        n_ok = ok.sum(-1)
        bz, mz = np.where(ok, raw[:rows], 0), np.where(ok, M, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            rb = np.sqrt(np.sum(np.square(bz, dtype=np.float64), -1) / n_ok)
            rm = np.sqrt(np.sum(np.square(mz, dtype=np.float64), -1) / n_ok)
        live = (n_ok >= 0.5 * ok.shape[-1]) & np.any(bz != 0, axis=-1)
        drms[a:b] = np.where(live, rb - rm, np.nan)
        raw = np.where(np.isfinite(raw), raw, 0)  # the fault attribute uses the baseline only
        B = raw - raw.mean(-1, keepdims=True)
        c_x = np.full((rows, n_xl), np.nan, np.float32)
        c_x[:, :-1] = _best_corr(B[:rows, :-1], B[:rows, 1:], max_lag)
        c_i = np.full((rows, n_xl), np.nan, np.float32)
        if bx - a > 1:
            r = min(rows, bx - a - 1)
            c_i[:r] = _best_corr(B[:r], B[1:r + 1], max_lag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # edge traces with one neighbour
            worst = np.nanmin(np.stack([c_x, c_i]), axis=0)
        disc[a:b] = np.where(live, 1 - worst, np.nan)
    return disc, drms


def reference_levels(disc, drms):
    """Survey-wide levels from the per-trace maps: robust z of the discontinuity, which traces
    are on a fault, a per-trace fault probability, and which traces clearly changed in 4D."""
    d = disc[np.isfinite(disc)]
    med = float(np.median(d)) if d.size else 0.0
    mad = (float(np.median(np.abs(d - med)) * 1.4826) if d.size else 0.0) or 1e-6
    z = (disc - med) / mad
    flag = np.isfinite(z) & (z > FAULT_Z)
    with np.errstate(over="ignore", invalid="ignore"):
        p_trace = 1 / (1 + np.exp(-(z - FAULT_Z)))       # per-trace fault probability
    ad = np.abs(drms)
    noise = float(np.nanmedian(ad)) if np.isfinite(ad).any() else 0.0  # typical |RMS change| = non-repeatability
    spread = float(np.nanmedian(np.abs(ad - noise)) * 1.4826) if np.isfinite(ad).any() else 0.0
    changed = np.isfinite(ad) & (ad > noise + CHANGE_Z * max(spread, 1e-12))  # traces that clearly changed
    return z, flag, p_trace, changed


# ---------------------------------------------------------------------------
# Per-hexagon table
# ---------------------------------------------------------------------------
def hexagon_row(hid, ring, centre, corners, base_vol, disc, z, flag, p_trace, changed, drms, line_step):
    il, xl = base_vol.ilines, base_vol.xlines
    ii, jj = traces_in_hexagon(corners, il, xl)
    keep = np.isfinite(disc[ii, jj])
    ii, jj = ii[keep], jj[keep]
    row = {"hexagon": hid, "centre_il": round(float(centre[0])), "centre_xl": round(float(centre[1])),
           "sealing_fault_probability": 0.0, "trace_pct": 0.0, "ring": int(ring), "n_traces": int(len(ii)),
           "fault_traces": 0, "fault_probability": 0.0, "seal_contrast": float("nan"), "seal_evidence": "no fault",
           "rms_diff": float("nan"), "fault_trend_deg": float("nan"),
           "changed_share_side_a": float("nan"), "changed_share_side_b": float("nan")}
    if not len(ii):
        row["seal_evidence"] = "outside the survey"
        return row
    row["rms_diff"] = float(np.nanmean(drms[ii, jj]))
    f = flag[ii, jj]
    nf = int(f.sum())
    row["fault_traces"] = nf
    row["trace_pct"] = 100.0 * nf / len(ii)
    if nf < MIN_FAULT_TRACES:
        return row

    # do the fault traces line up? (principal direction of their positions)
    pts = np.column_stack([il[ii[f]], xl[jj[f]]]).astype(float)
    c0 = pts.mean(0)
    _, sv, vt = np.linalg.svd(pts - c0, full_matrices=False)
    linearity = float(np.clip(1 - sv[1] / sv[0], 0, 1)) if sv[0] > 0 else 0.0
    fault_p = float(np.mean(p_trace[ii[f], jj[f]])) * linearity
    u = vt[0]
    row["fault_probability"] = fault_p
    row["fault_trend_deg"] = float(np.degrees(np.arctan2(u[0], u[1])) % 180)  # from the crossline axis

    # 4D change on each side of the fault line: share of traces that clearly changed
    normal = np.array([-u[1], u[0]])
    side = (np.column_stack([il[ii], xl[jj]]) - c0) @ normal
    gap = SIDE_GAP_LINES * line_step
    ch = changed[ii, jj]
    a, b = ch[side > gap], ch[side < -gap]
    if len(a) < 20 or len(b) < 20:
        row["seal_evidence"] = "fault at the hexagon edge"
        row["sealing_fault_probability"] = 100 * NO_4D_WEIGHT * fault_p
        return row
    fa, fb = float(a.mean()), float(b.mean())
    row["changed_share_side_a"], row["changed_share_side_b"] = fa, fb
    if max(fa, fb) < MIN_CHANGED_SHARE:
        row["seal_evidence"] = "no 4D change near the fault"
        row["sealing_fault_probability"] = 100 * NO_4D_WEIGHT * fault_p
        return row
    contrast = abs(fa - fb) / (fa + fb)
    row["seal_contrast"] = contrast
    row["seal_evidence"] = "change on one side only" if contrast >= 0.5 else "change on both sides"
    row["sealing_fault_probability"] = 100 * fault_p * contrast
    return row


def read_gemma_scores(path):
    """{hexagon number: {reservoir_pct, change_pct, opportunity_pct}} from gemma_ilcl.py's scores.csv."""
    p = Path(path) if path else None
    if not p or not p.is_file():
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                hid = int(str(r.get("id", "")).strip())
            except ValueError:
                continue

            def num(k):
                v = (r.get(k) or "").strip()
                return float(v) if v else float("nan")
            out[hid] = {"reservoir_pct": num("score_pct"), "gemma_change_pct": num("change_pct"),
                        "opportunity_pct": num("opportunity_pct")}
    return out


def main(base_vol=None, mon_vol=None, out_dir=OUT_DIR, window=WINDOW, center=(CENTER_ILINE, CENTER_XLINE),
         scale=SCALE, n_rings=N_RINGS, gemma_scores=GEMMA_SCORES):
    if base_vol is None or mon_vol is None:
        base_vol, mon_vol = load_cache_or_load_and_align_3d_pair(BASE_PATH, MONITOR_PATH)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    twt = np.asarray(base_vol.twt_ms, float)
    k0, k1 = (0, len(twt)) if window is None else (int(np.searchsorted(twt, window[0], "left")),
                                                 int(np.searchsorted(twt, window[1], "right")))
    if k1 - k0 < 8:
        raise ValueError(f"window {window} ms has too few samples in the data ({twt[0]:.0f}-{twt[-1]:.0f} ms)")
    dt = float(np.median(np.diff(twt)))
    max_lag = max(1, int(round(MAX_DIP_LAG_MS / dt)))
    print(f"Trace attributes over {twt[k0]:.0f}-{twt[k1 - 1]:.0f} ms "
          f"({len(base_vol.ilines)} x {len(base_vol.xlines)} traces)...")
    disc, drms = trace_attributes(base_vol, mon_vol, k0, k1, max_lag)

    z, flag, p_trace, changed = reference_levels(disc, drms)  # survey-wide reference levels
    line_step = float(np.median(np.diff(base_vol.ilines))) if len(base_vol.ilines) > 1 else 1.0
    print(f"  {flag.sum()} traces on a fault ({100 * flag.mean():.2f}% of the survey); "
          f"{changed.sum()} traces with clear 4D change ({100 * changed.mean():.2f}%)")

    centers, corners, ring = hexagon_grid(*center, scale=scale, n_rings=n_rings)
    gemma = read_gemma_scores(gemma_scores)
    rows = []
    for k, (c, crn, rg) in enumerate(zip(centers, corners, ring)):
        r = hexagon_row(k + ID_START, rg, c, crn, base_vol, disc, z, flag, p_trace, changed, drms, line_step)
        g = gemma.get(k)  # gemma_ilcl.py ids follow the grid order from 0
        r.update(g if g else {"reservoir_pct": float("nan"), "gemma_change_pct": float("nan"),
                              "opportunity_pct": float("nan")})
        rows.append(r)

    def fmt(v):
        return round(v, 2) if isinstance(v, float) else v

    with open(out / "hexagon_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TABLE_COLUMNS)
        for r in rows:
            if r["n_traces"]:
                w.writerow([fmt(r[c]) for c in TABLE_COLUMNS])
    with open(out / "hexagon_table_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})

    print(f"\n{'hex':>4} {'centre IL':>9} {'centre CL':>9} {'sealing %':>9} {'trace %':>8}   evidence")
    for r in rows:
        if not r["n_traces"]:
            print(f"{r['hexagon']:>4} {r['centre_il']:>9} {r['centre_xl']:>9} {'-':>9} {'-':>8}   outside the survey")
            continue
        print(f"{r['hexagon']:>4} {r['centre_il']:>9} {r['centre_xl']:>9} {r['sealing_fault_probability']:>8.1f}% "
              f"{r['trace_pct']:>7.1f}%   {r['seal_evidence']}")
    print(f"Saved {out}/hexagon_table.csv (for sealing_faults.py) and hexagon_table_details.csv"
          + ("" if gemma else " (no Gemma scores found to merge)"))
    try:
        import pandas as pd
        return pd.DataFrame(rows)
    except ImportError:
        return rows


if __name__ == "__main__":
    table = main()
