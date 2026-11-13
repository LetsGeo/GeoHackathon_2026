"""
RMS difference per hexagon: for every hexagon of the 19-hexagon grid, take all the traces
inside it and, in each time window, compute

    RMS(baseline) - RMS(monitor)

(the RMS over all samples of all the hexagon's traces in that window). Negative = the monitor
is stronger there (amplitude brightening, e.g. CO2), positive = weaker (dimming).

    Jupyter:   %run hex_rms.py
    Terminal:  python hex_rms.py

Windows: set WINDOWS to your own time ranges, e.g. [(800, 950), (950, 1100)] (ms), or leave it
None to step a WINDOW_MS window down the whole trace. Writes to OUT_DIR:
    hex_rms.csv          one row per hexagon and window: rms_base, rms_mon, rms_diff (base - mon), %
    hex_rms_summary.csv  one row per hexagon: the window with the biggest change, and its value
    hex_rms_map.png      RMS(base) - RMS(mon) per trace in the window with the most change, with the
                         hexagons and IDs (red = monitor stronger, as in the _diff pictures)
    hex_rms_profile.png  rms_diff against time, one line per hexagon

The grid (centre, SCALE, rings) and the hexagon IDs are the same as in polygon_ilcl_pics.py:
0 = middle, 1-6 = ring 1, 7-18 = ring 2.
"""
import csv
import sys
from pathlib import Path

import numpy as np

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # pasted into a notebook cell: run it from the repo folder
    sys.path.insert(0, str(Path.cwd()))
from volume3d import load_cache_or_load_and_align_3d_pair  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
BASE_PATH = "/data/epic20/APGCE/input_data/challenge3/volumes/Challenge3_Baseline_FilteredScaled_FullStack.sgy"
MONITOR_PATH = "/data/epic20/APGCE/input_data/challenge3/volumes/Challenge3_Monitor_FilteredScaled_FullStack.sgy"
OUT_DIR = "/data/epic20/APGCE/apgce_team_07/friday/hex_rms"

CENTER_ILINE, CENTER_XLINE = 1006, 2412  # middle hexagon (line numbers), as in polygon_ilcl_pics.py
SCALE = 110                              # hexagon radius in lines, as in polygon_ilcl_pics.py
N_RINGS = 2                              # 2 -> 1 + 6 + 12 = 19 hexagons

WINDOWS = None      # your windows in ms, e.g. [(800, 950), (950, 1100)]; None = step WINDOW_MS down the trace
WINDOW_MS = 100     # length of the stepped windows (used when WINDOWS is None)

s32 = np.sqrt(3) / 2


# ---------------------------------------------------------------------------
# Hexagon grid (same layout and order as polygon.py / polygon_ilcl_pics.py)
# ---------------------------------------------------------------------------
def hexagon_grid(center_iline, center_xline, scale=SCALE, n_rings=N_RINGS):
    """centers (N, 2) [iline, xline], corners (N, 6, 2) [iline, xline], ring (N,)."""
    hex_shape = np.array([[1, 0], [0.5, -s32], [-0.5, -s32], [-1, 0], [-0.5, s32], [0.5, s32]]) * scale
    cells = []
    for q in range(-n_rings, n_rings + 1):
        for r in range(-n_rings, n_rings + 1):
            k = max(abs(q), abs(r), abs(q + r))
            if k <= n_rings:
                cells.append((k, q, r))
    cells.sort(key=lambda c: (c[0], np.arctan2(c[2] + c[1] / 2, c[1])))  # by ring, then around the ring
    ring = np.array([k for k, _, _ in cells])
    q = np.array([c[1] for c in cells])
    r = np.array([c[2] for c in cells])
    centers = np.column_stack([center_iline + 1.5 * scale * q,
                               center_xline + 2 * s32 * scale * (r + q / 2)])
    corners = centers[:, None, :] + hex_shape[None, :, :]
    return centers, corners, ring


def traces_in_hexagon(corners, ilines, xlines):
    """(il_idx, xl_idx) of every trace whose (iline, xline) lies inside the hexagon."""
    from matplotlib.path import Path as Polygon
    c = np.asarray(corners, float)
    i_box = np.where((ilines >= c[:, 0].min()) & (ilines <= c[:, 0].max()))[0]
    j_box = np.where((xlines >= c[:, 1].min()) & (xlines <= c[:, 1].max()))[0]
    if not len(i_box) or not len(j_box):
        return np.array([], int), np.array([], int)
    I, J = np.meshgrid(i_box, j_box, indexing="ij")
    pts = np.column_stack([ilines[I.ravel()], xlines[J.ravel()]])
    inside = Polygon(c).contains_points(pts, radius=1e-9)
    return I.ravel()[inside], J.ravel()[inside]


# ---------------------------------------------------------------------------
# RMS in windows
# ---------------------------------------------------------------------------
def time_windows(twt, windows=WINDOWS, window_ms=WINDOW_MS):
    """[(t0, t1, k0, k1)]: each window in ms and the sample slice k0:k1 it covers."""
    twt = np.asarray(twt, float)
    if windows is None:
        edges = np.arange(twt[0], twt[-1] + window_ms, window_ms)
        windows = [(a, min(a + window_ms, twt[-1])) for a in edges if a < twt[-1]]
    out = []
    for t0, t1 in windows:
        k0, k1 = int(np.searchsorted(twt, t0, "left")), int(np.searchsorted(twt, t1, "right"))
        if k1 - k0 >= 2:
            out.append((float(t0), float(t1), k0, k1))
        else:
            print(f"  [!] window {t0}-{t1} ms is outside the data ({twt[0]:.0f}-{twt[-1]:.0f} ms): skipped")
    return out


def rms(a, axis=None):
    return np.sqrt(np.mean(np.square(a, dtype=np.float64), axis=axis))


def hexagon_rms(base_vol, mon_vol, corners, windows):
    """[(t0, t1, rms_base, rms_mon)] over all the hexagon's traces, and the number of traces."""
    ii, jj = traces_in_hexagon(corners, base_vol.ilines, base_vol.xlines)
    if not len(ii):
        return [], 0
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1  # read only the box round the hexagon
    rows = []
    for t0, t1, k0, k1 in windows:
        b = np.asarray(base_vol.data[i0:i1, j0:j1, k0:k1], dtype=np.float32)[ii - i0, jj - j0]
        m = np.asarray(mon_vol.data[i0:i1, j0:j1, k0:k1], dtype=np.float32)[ii - i0, jj - j0]
        rows.append((t0, t1, float(rms(b)), float(rms(m))))
    return rows, len(ii)


def rms_diff_map(base_vol, mon_vol, k0, k1, block=64):
    """RMS(base) - RMS(mon) per trace in samples k0:k1 (for the map), read a block of inlines at a time."""
    n_il, n_xl = len(base_vol.ilines), len(base_vol.xlines)
    out = np.empty((n_il, n_xl), np.float32)
    for a in range(0, n_il, block):
        b = min(a + block, n_il)
        out[a:b] = rms(base_vol.data[a:b, :, k0:k1], axis=-1) - rms(mon_vol.data[a:b, :, k0:k1], axis=-1)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(base_vol=None, mon_vol=None, out_dir=OUT_DIR, windows=WINDOWS, window_ms=WINDOW_MS,
         center=(CENTER_ILINE, CENTER_XLINE), scale=SCALE, n_rings=N_RINGS):
    if base_vol is None or mon_vol is None:
        base_vol, mon_vol = load_cache_or_load_and_align_3d_pair(BASE_PATH, MONITOR_PATH)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    centers, corners, ring = hexagon_grid(*center, scale=scale, n_rings=n_rings)
    wins = time_windows(base_vol.twt_ms, windows, window_ms)
    print(f"{len(centers)} hexagons, {len(wins)} windows: "
          + ", ".join(f"{t0:.0f}-{t1:.0f}" for t0, t1, _, _ in wins[:8]) + (" ..." if len(wins) > 8 else "") + " ms")

    rows, summary = [], []
    for hid, (c, crn, k) in enumerate(zip(centers, corners, ring)):
        name = f"hex{hid:02d}"
        res, n = hexagon_rms(base_vol, mon_vol, crn, wins)
        if not n:
            print(f"{name} (ring {k}): no traces inside the survey: skipped")
            continue
        best = None
        for t0, t1, rb, rm in res:
            d = rb - rm
            pct = 100 * d / rb if rb > 0 else float("nan")
            rows.append({"hex": name, "ring": int(k), "centre_il": round(float(c[0])), "centre_xl": round(float(c[1])),
                         "n_traces": n, "t0_ms": t0, "t1_ms": t1, "rms_base": rb, "rms_mon": rm,
                         "rms_diff": d, "rms_diff_pct": pct})
            if best is None or abs(d) > abs(best[2]):
                best = (t0, t1, d, pct)
        summary.append({"hex": name, "ring": int(k), "centre_il": round(float(c[0])), "centre_xl": round(float(c[1])),
                        "n_traces": n, "biggest_change_window_ms": f"{best[0]:.0f}-{best[1]:.0f}",
                        "rms_diff": best[2], "rms_diff_pct": best[3]})
        print(f"{name} (ring {k}): {n} traces, biggest change {best[0]:.0f}-{best[1]:.0f} ms: "
              f"RMS(base) - RMS(mon) = {best[2]:.4g} ({best[3]:+.1f}%)")

    _write_csv(out / "hex_rms.csv", rows)
    _write_csv(out / "hex_rms_summary.csv", summary)
    if rows:
        _plots(base_vol, mon_vol, corners, rows, wins, out)
    print(f"Saved {out}/hex_rms.csv, hex_rms_summary.csv, hex_rms_map.png and hex_rms_profile.png")
    try:
        import pandas as pd
        return {"windows": pd.DataFrame(rows), "summary": pd.DataFrame(summary)}
    except ImportError:
        return {"windows": rows, "summary": summary}


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def _plots(base_vol, mon_vol, corners, rows, wins, out):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    # map: per-trace RMS difference in the window where the hexagons change most overall
    by_win = {}
    for r in rows:
        by_win.setdefault((r["t0_ms"], r["t1_ms"]), []).append(abs(r["rms_diff"]))
    t0, t1 = max(by_win, key=lambda w: np.mean(by_win[w]))
    k0, k1 = next((a, b) for s, e, a, b in wins if (s, e) == (t0, t1))
    dmap = rms_diff_map(base_vol, mon_vol, k0, k1)
    lim = float(np.nanpercentile(np.abs(dmap), 99)) or 1.0
    il, xl = base_vol.ilines, base_vol.xlines
    fig = Figure(figsize=(9, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(dmap, origin="lower", aspect="auto", cmap="seismic_r", vmin=-lim, vmax=lim,
                   extent=(xl[0], xl[-1], il[0], il[-1]), interpolation="nearest")
    for hid, crn in enumerate(corners):  # corners are [iline, xline]; the map is x = xline, y = iline
        ax.add_patch(Polygon(crn[:, ::-1], closed=True, fill=False, ec="black", lw=0.8))
        cx, cy = crn[:, 1].mean(), crn[:, 0].mean()
        if xl.min() <= cx <= xl.max() and il.min() <= cy <= il.max():
            ax.text(cx, cy, f"{hid:02d}", ha="center", va="center", fontsize=8, weight="bold")
    ax.set_xlim(xl.min(), xl.max())
    ax.set_ylim(il.min(), il.max())
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Inline")
    ax.set_title(f"RMS(base) - RMS(mon), {t0:.0f}-{t1:.0f} ms (red = monitor stronger, blue = weaker)")
    fig.colorbar(im, ax=ax, label="RMS(base) - RMS(mon)")
    fig.tight_layout()
    fig.savefig(out / "hex_rms_map.png", dpi=120)

    # profile: rms_diff against window centre time, one line per hexagon
    fig = Figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    hexes = sorted({r["hex"] for r in rows})
    strongest = sorted(hexes, key=lambda h: -max(abs(r["rms_diff"]) for r in rows if r["hex"] == h))[:5]
    for h in hexes:
        rr = [r for r in rows if r["hex"] == h]
        t = [(r["t0_ms"] + r["t1_ms"]) / 2 for r in rr]
        d = [r["rms_diff"] for r in rr]
        if h in strongest:
            ax.plot(d, t, marker="o", lw=1.8, label=h)
        else:
            ax.plot(d, t, color="0.75", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("RMS(base) - RMS(mon)")
    ax.set_ylabel("Window centre, TWT (ms)")
    ax.set_title("RMS difference per hexagon and window (5 biggest changes coloured)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "hex_rms_profile.png", dpi=120)


if __name__ == "__main__":
    results = main()
