"""
Hexagon fault screening for the Reservoir Time Machine.

1. The area the two surveys share is tiled with flat-topped hexagons, 20 lines
   across (measured in line numbers: 20 crosslines wide, about 17 inlines tall).
2. Attributes are computed on every trace (whole trace) and summarised per hex:
   RMS amplitude, 4D difference, NRMS, amplitude ratio, time shift, envelope,
   instantaneous frequency, sweetness, coherence and fault throw, plus a "seal
   contrast": how differently the 4D change behaves on the two sides of the
   strongest discontinuity in the hex (CO2 change on one side only = a fault
   that may be sealing).
3. For each hex, 5 inlines and 5 crosslines spread evenly across it (centre -8,
   -4, 0, +4, +8 lines) are drawn as two montages: monitor amplitude and
   4D difference (monitor - baseline). These go to Gemma (gemma_client.classify_fault).
4. combine() merges Gemma's reading with the attribute evidence into one verdict
   per hex: "likely", "possible" or "unlikely" sealing fault.

Everything here is plain numpy + matplotlib, so it runs the same in the app and
in the command-line script (hex_screen.py).
"""
import csv
import io
import math
import warnings

import numpy as np

HEX_WIDTH = 20                    # lines across a hexagon (flat side to flat side is ~17)
LINE_OFFSETS = (-8, -4, 0, 4, 8)  # the 5 inlines / crosslines used per hex, relative to its centre
MIN_FILL = 0.25                   # keep edge hexes with at least this share of a full hex's traces
COH_WINDOW_MS = 128               # window length for coherence / throw
COH_MAX_LAG_MS = 32               # largest shift searched between neighbouring traces
SHIFT_MAX_LAG_MS = 20             # largest baseline-vs-monitor time shift searched
SQRT3 = math.sqrt(3.0)

# verdict per hex, strongest first
VERDICTS = ("likely", "possible", "fault", "none")
VERDICT_RANK = {"likely": 3, "possible": 2, "fault": 1, "none": 0}
VERDICT_TEXT = {
    "likely": ("Sealing fault likely", "Sesar pengedap berkemungkinan besar"),
    "possible": ("Sealing fault possible", "Sesar pengedap mungkin"),
    "fault": ("Fault, no sealing evidence", "Sesar, tiada bukti pengedap"),
    "none": ("No fault found", "Tiada sesar dikesan"),
}

# per-hex columns written to the CSV / shown in the UI, with short descriptions
HEX_FIELDS = [
    ("id", "Hex number"),
    ("il_centre", "Inline at the hex centre"),
    ("xl_centre", "Crossline at the hex centre"),
    ("n_traces", "Traces inside the hex"),
    ("rms_base", "Mean RMS amplitude, baseline"),
    ("rms_mon", "Mean RMS amplitude, monitor"),
    ("diff_rms", "Mean RMS of the 4D difference"),
    ("diff_rms_max", "Largest RMS of the 4D difference"),
    ("changed_pct", "% of traces with significant 4D change"),
    ("nrms", "Median NRMS (%)"),
    ("amp_ratio", "Median monitor/baseline RMS ratio"),
    ("time_shift_ms", "Median monitor-vs-baseline time shift (ms, + = later)"),
    ("envelope", "Mean envelope (monitor)"),
    ("inst_freq_hz", "Mean instantaneous frequency (monitor, Hz)"),
    ("sweetness", "Mean sweetness (monitor)"),
    ("coherence", "Median coherence (monitor, 1 = continuous)"),
    ("discontinuity", "Discontinuity, 98th percentile (1 - coherence)"),
    ("fault_traces", "Traces on a fault (discontinuity above the fault threshold)"),
    ("throw_ms", "Apparent fault throw, median on fault traces (ms)"),
    ("fault_score", "Fault score from attributes (0-1)"),
    ("seal_contrast", "4D change contrast across the discontinuity (0-1)"),
    ("seal_score", "Sealing score from attributes (0-1)"),
    ("fault_trend_deg", "Trend of the discontinuity (degrees from the crossline axis)"),
]


# ---------------------------------------------------------------------------
# Shared grid
# ---------------------------------------------------------------------------
def shared_grid(bv, mv):
    """Inlines, crosslines and times common to both surveys."""
    il = np.intersect1d(bv["ilines"], mv["ilines"])
    xl = np.intersect1d(bv["xlines"], mv["xlines"])
    if len(il) < 3 or len(xl) < 3:
        raise ValueError("the two surveys don't share enough inlines and crosslines")
    t_lo = max(bv["times"][0], mv["times"][0])
    t_hi = min(bv["times"][-1], mv["times"][-1])
    times = bv["times"][(bv["times"] >= t_lo) & (bv["times"] <= t_hi)].astype(np.float64)
    if len(times) < 20:
        raise ValueError("the two surveys don't overlap enough in time")
    dt = float(np.median(np.diff(times)))
    return {"ilines": il, "xlines": xl, "times": times, "dt": dt,
            "dil": float(np.median(np.diff(il))), "dxl": float(np.median(np.diff(xl)))}


def read_block(vol, il_sel, xl_sel, times):
    """Traces (n_il, n_xl, n_t) for the chosen lines, resampled onto `times` if needed."""
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


def il_at(grid, i):
    """Inline number at (fractional) inline index i of the shared grid."""
    return grid["ilines"][0] + np.asarray(i) * grid["dil"]


def xl_at(grid, j):
    return grid["xlines"][0] + np.asarray(j) * grid["dxl"]


# ---------------------------------------------------------------------------
# Per-trace attributes
# ---------------------------------------------------------------------------
def _rms(a):
    return np.sqrt(np.mean(np.square(a, dtype=np.float64), axis=-1))


def _analytic(x):
    """Analytic signal along the last axis (FFT Hilbert transform)."""
    n = x.shape[-1]
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def _lag_ms(B, M, max_lag, dt):
    """Per-trace shift (ms) that best aligns M to B; positive = monitor later."""
    nt = B.shape[-1]
    L = max(1, min(max_lag, nt // 4))
    core = B[..., L:nt - L]
    C = np.stack([np.sum(core * M[..., L + k:nt - L + k], axis=-1) for k in range(-L, L + 1)])
    k = np.argmax(C, axis=0)
    kk = np.clip(k, 1, 2 * L - 1)
    y0 = np.take_along_axis(C, (kk - 1)[None], 0)[0]
    y1 = np.take_along_axis(C, kk[None], 0)[0]
    y2 = np.take_along_axis(C, (kk + 1)[None], 0)[0]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where((k == kk) & (denom != 0), 0.5 * (y0 - y2) / denom, 0.0)
    return ((k + frac) - L) * dt


def _pair_coherence(X, Y, W, Lc):
    """Windowed normalised correlation between neighbouring traces X and Y.
    Returns per-window coherence (best within +-1 sample, so gentle dip doesn't
    count as a break), per-window |best shift| (samples, up to +-Lc) and X energy."""
    nt = X.shape[-1]
    nw = max(1, nt // W)
    W = min(W, nt)
    T = nw * W
    shape = X.shape[:-1] + (nw, W)
    Xw = X[..., :T].reshape(shape)
    ex = np.sum(Xw * Xw, axis=-1)
    pad = [(0, 0)] * (Y.ndim - 1) + [(Lc, Lc)]
    Yp = np.pad(Y, pad)
    corrs = []
    for k in range(-Lc, Lc + 1):
        Yk = Yp[..., Lc + k:Lc + k + T].reshape(shape)
        num = np.sum(Xw * Yk, axis=-1)
        ey = np.sum(Yk * Yk, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            corrs.append(num / np.sqrt(ex * ey))
    C = np.nan_to_num(np.stack(corrs), nan=0.0)
    coh = C[max(0, Lc - 1):Lc + 2].max(axis=0)
    lag = np.abs(np.argmax(C, axis=0) - Lc).astype(np.float32)
    return coh, lag, ex


def _window_stat(values, energy, keep_frac=0.05):
    """Median over the windows that carry signal (energy above 5% of the median)."""
    ref = np.nanmedian(energy[energy > 0]) if np.any(energy > 0) else 0.0
    mask = energy > keep_frac * ref
    v = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN
        return np.nanmedian(v, axis=-1)


ATTR_KEYS = ("rms_base", "rms_mon", "diff_rms", "nrms", "amp_ratio", "time_shift_ms", "envelope",
             "inst_freq_hz", "sweetness", "coherence", "throw_ms")


def trace_attributes(bv, mv, grid, chunk=16, progress=None):
    """Attribute maps (n_il, n_xl) over the shared grid, whole trace."""
    il, xl, times, dt = grid["ilines"], grid["xlines"], grid["times"], grid["dt"]
    n_il, n_xl, nt = len(il), len(xl), len(times)
    maps = {k: np.full((n_il, n_xl), np.nan, np.float32) for k in ATTR_KEYS}
    W = max(8, int(round(COH_WINDOW_MS / dt)))
    Lc = max(2, int(round(COH_MAX_LAG_MS / dt)))
    Ls = max(1, int(round(SHIFT_MAX_LAG_MS / dt)))
    amp_samples, diff_samples = [], []
    for a in range(0, n_il, chunk):
        b = min(a + chunk, n_il)
        b_ext = min(b + 1, n_il)  # one extra inline for the inline-direction neighbour
        B = read_block(bv, il[a:b_ext], xl, times)
        M = read_block(mv, il[a:b_ext], xl, times)
        valid = np.isfinite(B).all(-1) & np.isfinite(M).all(-1)
        B = np.nan_to_num(B)
        M = np.nan_to_num(M)
        n = b - a
        Bc, Mc, vc = B[:n], M[:n], valid[:n]
        D = Mc - Bc
        rb, rm, rd = _rms(Bc), _rms(Mc), _rms(D)
        with np.errstate(divide="ignore", invalid="ignore"):
            maps["rms_base"][a:b] = rb
            maps["rms_mon"][a:b] = rm
            maps["diff_rms"][a:b] = rd
            maps["nrms"][a:b] = 200 * rd / (rb + rm)
            maps["amp_ratio"][a:b] = rm / rb
        maps["time_shift_ms"][a:b] = _lag_ms(Bc, Mc, Ls, dt)

        an = _analytic(Mc)
        env = np.abs(an)
        dphi = np.angle(an[..., 1:] * np.conj(an[..., :-1]))
        freq = np.clip(dphi / (2 * np.pi * dt / 1000.0), 0, None)
        w = env[..., 1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            f_mean = np.sum(w * freq, -1) / np.sum(w, -1)
            maps["envelope"][a:b] = env.mean(-1)
            maps["inst_freq_hz"][a:b] = f_mean
            maps["sweetness"][a:b] = env.mean(-1) / np.sqrt(f_mean)

        # coherence + throw against the next crossline and the next inline (monitor)
        coh_x = np.full((n, n_xl), np.nan, np.float32)
        thr_x = np.full((n, n_xl), np.nan, np.float32)
        c, lg, e = _pair_coherence(Mc[:, :-1], Mc[:, 1:], W, Lc)
        coh_x[:, :-1] = _window_stat(c, e)
        thr_x[:, :-1] = _window_stat(lg, e) * dt
        coh_i = np.full((n, n_xl), np.nan, np.float32)
        thr_i = np.full((n, n_xl), np.nan, np.float32)
        rows = min(n, M.shape[0] - 1)
        if rows > 0:
            c, lg, e = _pair_coherence(M[:rows], M[1:rows + 1], W, Lc)
            coh_i[:rows] = _window_stat(c, e)
            thr_i[:rows] = _window_stat(lg, e) * dt
        use_i = np.isnan(coh_x) | (coh_i < coh_x)
        maps["coherence"][a:b] = np.where(use_i, coh_i, coh_x)
        maps["throw_ms"][a:b] = np.where(use_i, thr_i, thr_x)

        for k in ATTR_KEYS:
            maps[k][a:b][~vc] = np.nan
        step = max(1, int(np.sqrt(Mc.size / 200_000)))
        amp_samples.append(np.abs(Mc[vc][::step]).ravel())
        diff_samples.append(np.abs(D[vc][::step]).ravel())
        if progress:
            progress(b / n_il)

    amp = np.concatenate(amp_samples) if amp_samples else np.array([1.0])
    dif = np.concatenate(diff_samples) if diff_samples else np.array([1.0])
    clips = {"amp": float(np.percentile(amp, 99)) or 1.0, "diff": float(np.percentile(dif, 99)) or 1.0}
    return maps, clips


def robust_threshold(values):
    """Median + 3 robust standard deviations."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return np.inf
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826
    return float(med + 3 * mad) if mad > 0 else float(med * 1.5 + 1e-12)


# ---------------------------------------------------------------------------
# Hexagon grid
# ---------------------------------------------------------------------------
def _cube_round(q, r):
    x, z = q, r
    y = -x - z
    rx, ry, rz = np.round(x), np.round(y), np.round(z)
    dx, dy, dz = np.abs(rx - x), np.abs(ry - y), np.abs(rz - z)
    fix_x = (dx > dy) & (dx > dz)
    fix_y = ~fix_x & (dy > dz)
    rx = np.where(fix_x, -ry - rz, rx)
    rz = np.where(~fix_x & ~fix_y, -rx - ry, rz)
    return rx.astype(int), rz.astype(int)


def hex_corners(cx, cy, size):
    """Corners of a flat-topped hexagon in index units (x = crossline, y = inline)."""
    ang = np.radians(np.arange(6) * 60.0)
    return cx + size * np.cos(ang), cy + size * np.sin(ang)


def build_hexes(grid, maps, width=HEX_WIDTH):
    """Tile the shared grid with flat-topped hexagons `width` lines across and
    summarise the attribute maps inside each one."""
    n_il, n_xl = len(grid["ilines"]), len(grid["xlines"])
    size = width / 2.0
    I, J = np.meshgrid(np.arange(n_il), np.arange(n_xl), indexing="ij")
    # axial coordinates of every trace (x = crossline index, y = inline index)
    q = (2.0 / 3.0 * J) / size
    r = (-1.0 / 3.0 * J + SQRT3 / 3.0 * I) / size
    hq, hr = _cube_round(q, r)
    cells, cell_of = np.unique(np.stack([hq.ravel(), hr.ravel()], 1), axis=0, return_inverse=True)
    cell_of = cell_of.reshape(hq.shape)
    full_area = 1.5 * SQRT3 * size * size

    disc = 1.0 - maps["coherence"]
    g_disc = disc[np.isfinite(disc)]
    d50 = float(np.median(g_disc)) if g_disc.size else 0.0
    mad = float(np.median(np.abs(g_disc - d50)) * 1.4826) if g_disc.size else 0.0
    fault_thr = d50 + max(5 * mad, 0.05)  # traces this discontinuous count as "on a fault"
    on_fault = np.isfinite(disc) & (disc > fault_thr)
    chg_thr = robust_threshold(maps["diff_rms"])
    chg_floor = float(np.nanmedian(maps["diff_rms"]))  # 4D noise level (non-repeatability)
    changed = maps["diff_rms"] > chg_thr

    hexes = []
    for k, (aq, ar) in enumerate(cells):
        sel = cell_of == k
        if sel.sum() < MIN_FILL * full_area:
            continue
        aq, ar = int(aq), int(ar)
        cx = size * 1.5 * aq
        cy = size * SQRT3 * (ar + aq / 2.0)
        ii, jj = I[sel], J[sel]
        ok = np.isfinite(maps["rms_mon"][sel])
        if ok.sum() < 10:
            continue

        def mean(name):
            v = maps[name][sel][ok]
            v = v[np.isfinite(v)]
            return float(np.mean(v)) if v.size else float("nan")

        def pct(name, p):
            v = maps[name][sel][ok]
            v = v[np.isfinite(v)]
            return float(np.percentile(v, p)) if v.size else float("nan")

        h = {
            "axial": (aq, ar), "cx": float(cx), "cy": float(cy),
            "il_centre": float(il_at(grid, cy)), "xl_centre": float(xl_at(grid, cx)),
            "n_traces": int(ok.sum()),
            "rms_base": mean("rms_base"), "rms_mon": mean("rms_mon"),
            "diff_rms": mean("diff_rms"), "diff_rms_max": pct("diff_rms", 100),
            "changed_pct": float(np.mean(changed[sel][ok]) * 100),
            "nrms": pct("nrms", 50), "amp_ratio": pct("amp_ratio", 50),
            "time_shift_ms": pct("time_shift_ms", 50),
            "envelope": mean("envelope"), "inst_freq_hz": mean("inst_freq_hz"), "sweetness": mean("sweetness"),
            "coherence": pct("coherence", 50),
        }
        hd = disc[sel][ok]
        hd = hd[np.isfinite(hd)]
        h["discontinuity"] = float(np.percentile(hd, 98)) if hd.size else float("nan")

        # fault: traces above the fault threshold that line up. A fault crossing the
        # hex lights up ~width traces in a line; scattered noise doesn't line up.
        pts = on_fault[sel]
        n_f = int(pts.sum())
        h["fault_traces"] = n_f
        h["fault_score"], h["throw_ms"] = 0.0, 0.0
        h["seal_contrast"], h["fault_trend_deg"] = 0.0, float("nan")
        if n_f >= 4:
            P = np.column_stack([jj[pts], ii[pts]]).astype(float)
            c0 = P.mean(0)
            _, sv, vt = np.linalg.svd(P - c0, full_matrices=False)
            linearity = float(1 - sv[1] / sv[0]) if sv[0] > 0 else 0.0
            h["fault_score"] = float(min(1.0, n_f / (0.5 * width)) * max(0.0, linearity))
            thr = maps["throw_ms"][sel][pts]
            thr = thr[np.isfinite(thr)]
            h["throw_ms"] = float(np.median(thr)) if thr.size else 0.0
            u = vt[0]
            nrm = np.array([-u[1], u[0]])
            side = (np.column_stack([jj, ii]) - c0) @ nrm
            chg = maps["diff_rms"][sel]
            A = chg[(side > 1) & np.isfinite(chg)]
            Bs = chg[(side < -1) & np.isfinite(chg)]
            if A.size >= 5 and Bs.size >= 5:
                ma, mb = float(A.mean()), float(Bs.mean())
                ea, eb = max(ma - chg_floor, 0.0), max(mb - chg_floor, 0.0)  # change above the noise
                contrast = abs(ea - eb) / (ea + eb + 1e-12)
                h["seal_contrast"] = float(contrast * min(1.0, max(ma, mb) / chg_thr)) if np.isfinite(chg_thr) else 0.0
                h["fault_trend_deg"] = float(np.degrees(np.arctan2(u[1], u[0])) % 180)
        h["seal_score"] = float(h["fault_score"] * h["seal_contrast"])
        xs, ys = hex_corners(cx, cy, size)
        h["corners_xl"] = [float(v) for v in xl_at(grid, xs)]
        h["corners_il"] = [float(v) for v in il_at(grid, ys)]
        hexes.append(h)

    hexes.sort(key=lambda h: (h["cy"], h["cx"]))
    for n, h in enumerate(hexes, start=1):
        h["id"] = n
    stats = {"disc_median": d50, "fault_threshold": fault_thr, "change_threshold": chg_thr,
             "hex_width": width, "line_offsets": list(LINE_OFFSETS)}
    return hexes, stats


def prepare(bv, mv, width=HEX_WIDTH, progress=None):
    """Everything the screening needs: shared grid, attribute maps, hexes."""
    grid = shared_grid(bv, mv)
    maps, clips = trace_attributes(bv, mv, grid, progress=progress)
    hexes, stats = build_hexes(grid, maps, width)
    return {"grid": grid, "maps": maps, "clips": clips, "hexes": hexes, "stats": stats, "width": width}


# ---------------------------------------------------------------------------
# Sections and montages
# ---------------------------------------------------------------------------
def hex_lines(ctx, h):
    """The 5 inlines and 5 crosslines used for a hex, with the span each covers."""
    grid, size = ctx["grid"], ctx["width"] / 2.0
    n_il, n_xl = len(grid["ilines"]), len(grid["xlines"])
    half_h = int(math.ceil(SQRT3 / 2 * size))
    j0, j1 = max(0, int(round(h["cx"] - size))), min(n_xl - 1, int(round(h["cx"] + size)))
    i0, i1 = max(0, int(round(h["cy"] - half_h))), min(n_il - 1, int(round(h["cy"] + half_h)))
    inl = sorted({int(np.clip(round(h["cy"] + d), 0, n_il - 1)) for d in LINE_OFFSETS})
    xln = sorted({int(np.clip(round(h["cx"] + d), 0, n_xl - 1)) for d in LINE_OFFSETS})
    return inl, xln, (j0, j1), (i0, i1)


def hex_sections(ctx, bv, mv, h):
    """[(kind, line number, x numbers, baseline section, monitor section)], sections are (n_x, n_t)."""
    grid = ctx["grid"]
    il, xl, times = grid["ilines"], grid["xlines"], grid["times"]
    inl, xln, (j0, j1), (i0, i1) = hex_lines(ctx, h)
    out = []
    xs = xl[j0:j1 + 1]
    for i in inl:
        b = read_block(bv, il[[i]], xs, times)[0]
        m = read_block(mv, il[[i]], xs, times)[0]
        out.append(("IL", int(il[i]), xs, b, m))
    ys = il[i0:i1 + 1]
    for j in xln:
        b = read_block(bv, ys, xl[[j]], times)[:, 0]
        m = read_block(mv, ys, xl[[j]], times)[:, 0]
        out.append(("XL", int(xl[j]), ys, b, m))
    return out


def _colormaps():
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    return {"amp": matplotlib.colormaps["gray_r"],
            "diff": LinearSegmentedColormap.from_list("diff", ["#2f6fd0", "#ffffff", "#f08a24"])}


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _nice_step(span, target=5):
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def render_montages(ctx, bv, mv, h, panel_w=170, panel_h=300):
    """Two PNGs (bytes): monitor amplitude and 4D difference (monitor - baseline).
    Top row: the 5 inlines, bottom row: the 5 crosslines, whole trace, time down."""
    from PIL import Image, ImageDraw

    secs = hex_sections(ctx, bv, mv, h)
    times = ctx["grid"]["times"]
    clips = {"amp": ctx["clips"]["amp"], "diff": ctx["clips"]["diff"]}
    cmaps = _colormaps()
    f_title, f_head, f_tick = _font(16), _font(13), _font(11)
    left, top, gap, head, foot, right = 54, 40, 10, 20, 30, 12
    row_h = head + panel_h + foot
    W = left + 5 * panel_w + 4 * gap + right
    H = top + 2 * row_h + 8
    t0, t1 = float(times[0]), float(times[-1])
    step = _nice_step(t1 - t0)
    ticks = np.arange(math.ceil(t0 / step) * step, t1 + 1e-9, step)
    titles = {"amp": "monitor amplitude",
              "diff": "4D difference, monitor - baseline (blue = weaker, orange = stronger)"}
    pngs = []
    for kind in ("amp", "diff"):
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        d.text((left, 10), f"Hex {h['id']} (inline {h['il_centre']:.0f} / crossline {h['xl_centre']:.0f}): "
                           f"{titles[kind]}", fill="black", font=f_title)
        col = {"IL": 0, "XL": 0}
        for line_kind, num, xs, b, m in secs:
            r = 0 if line_kind == "IL" else 1
            c = col[line_kind]
            col[line_kind] += 1
            x0 = left + c * (panel_w + gap)
            y0 = top + r * row_h + head
            data = (m if kind == "amp" else m - b).T  # (time, x)
            norm = np.clip((data / clips[kind] + 1) / 2, 0, 1)
            rgb = cmaps[kind](np.nan_to_num(norm, nan=0.5), bytes=True)[..., :3]
            panel = Image.fromarray(rgb).resize((panel_w, panel_h), Image.BILINEAR)
            img.paste(panel, (x0, y0))
            d.rectangle([x0 - 1, y0 - 1, x0 + panel_w, y0 + panel_h], outline="#555555")
            name = "Inline" if line_kind == "IL" else "Crossline"
            d.text((x0, y0 - head + 2), f"{name} {num}", fill="black", font=f_head)
            other = "xl" if line_kind == "IL" else "il"
            d.text((x0, y0 + panel_h + 4), f"{other} {xs[0]:.0f}", fill="#333333", font=f_tick)
            lbl = f"{xs[-1]:.0f}"
            d.text((x0 + panel_w - 7 * len(lbl), y0 + panel_h + 4), lbl, fill="#333333", font=f_tick)
            if c == 0:
                for t in ticks:
                    y = y0 + (t - t0) / (t1 - t0) * panel_h
                    d.line([x0 - 5, y, x0 - 1, y], fill="black")
                    d.text((4, y - 6), f"{t:.0f}", fill="black", font=f_tick)
                d.text((4, y0 - head + 2), "ms", fill="#333333", font=f_tick)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        pngs.append(buf.getvalue())
    return pngs[0], pngs[1]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def attribute_evidence(h):
    return {k: (round(h[k], 3) if isinstance(h.get(k), float) and np.isfinite(h[k]) else h.get(k))
            for k, _ in HEX_FIELDS if k in h}


def combine(h, gemma):
    """One verdict per hex from the attribute scores and Gemma's reading.
    gemma is None (not asked / failed) or {"fault", "sealing", "confidence", ...}.
    Returns (verdict, source) with verdict one of VERDICTS."""
    seal = h.get("seal_score", 0.0)
    fault = h.get("fault_score", 0.0)
    if not gemma or gemma.get("error"):
        if seal >= 0.5:
            return "likely", "attributes only"
        if seal >= 0.25:
            return "possible", "attributes only"
        return ("fault" if fault >= 0.5 else "none"), "attributes only"
    conf = float(gemma.get("confidence") or 0)
    g_fault, g_seal = gemma.get("fault"), gemma.get("sealing")
    if g_seal == "yes":
        verdict = "likely" if (conf >= 0.6 or seal >= 0.25) else "possible"
    elif g_seal == "unsure":
        verdict = "possible" if (seal >= 0.25 or g_fault == "yes") else ("fault" if fault >= 0.5 else "none")
    elif seal >= 0.5:  # Gemma says no / not testable, but the attributes disagree strongly
        verdict = "possible"
    else:
        verdict = "fault" if (g_fault == "yes" or fault >= 0.5) else "none"
    return verdict, "Gemma + attributes"


def gemma_context(ctx):
    """Survey-wide reference numbers sent with every hex."""
    st = ctx["stats"]
    return {"fault_threshold_discontinuity": round(st["fault_threshold"], 3),
            "change_threshold_diff_rms": round(st["change_threshold"], 4),
            "survey_median_diff_rms": round(float(np.nanmedian(ctx["maps"]["diff_rms"])), 4),
            "hex_width_lines": st["hex_width"], "lines_used_offsets": st["line_offsets"]}


class GemmaBreaker:
    """Stops asking Gemma once it is clearly unreachable (3 failures before any
    success), so a down server doesn't cost a timeout per hex."""

    def __init__(self, limit=3):
        import threading
        self.limit, self.ok, self.failed, self.first_error = limit, 0, 0, None
        self._lock = threading.Lock()

    @property
    def open(self):
        return self.ok == 0 and self.failed >= self.limit

    def record(self, error=None):
        with self._lock:
            if error is None:
                self.ok += 1
            else:
                self.failed += 1
                self.first_error = self.first_error or error


def screen_hex(ctx, bv, mv, h, use_gemma=True, labels=("Baseline", "Monitor"), client=None,
               base_url=None, model=None, breaker=None, keep_images=False):
    """Montages + (optionally) Gemma + verdict for one hex. Fills h in place and returns it.
    The montage PNGs are kept on h only with keep_images=True (they are large)."""
    h["gemma"] = None
    if use_gemma and not (breaker and breaker.open):
        import gemma_client
        amp_png, diff_png = render_montages(ctx, bv, mv, h)
        if keep_images:
            h["amp_png"], h["diff_png"] = amp_png, diff_png
        try:
            h["gemma"] = gemma_client.classify_fault(
                attribute_evidence(h), amp_png, diff_png, labels=labels, context=gemma_context(ctx),
                base_url=base_url, model=model, client=client)
            if breaker:
                breaker.record()
        except Exception as exc:  # keep going: this hex falls back to the attributes
            h["gemma"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            if breaker:
                breaker.record(h["gemma"]["error"])
    elif use_gemma:
        h["gemma"] = {"error": "skipped: Gemma did not answer the first requests"}
    elif keep_images:
        h["amp_png"], h["diff_png"] = render_montages(ctx, bv, mv, h)
    h["verdict"], h["verdict_source"] = combine(h, h["gemma"])
    return h


def screen_all(ctx, bv, mv, use_gemma=True, labels=("Baseline", "Monitor"), workers=4,
               base_url=None, model=None, progress=None, keep_images=False):
    """Screen every hex, `workers` at a time. progress(done, total, hex) after each one."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    client = None
    if use_gemma:
        import gemma_client
        client = gemma_client.make_client(gemma_client.settings(base_url, model))
    hexes = ctx["hexes"]
    breaker = GemmaBreaker()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(screen_hex, ctx, bv, mv, h, use_gemma, labels, client, base_url, model,
                               breaker, keep_images)
                   for h in hexes]
        for n, fut in enumerate(as_completed(futures), start=1):
            h = fut.result()
            if progress:
                progress(n, len(hexes), h)
    return hexes


def summary_counts(hexes):
    counts = {v: 0 for v in VERDICTS}
    for h in hexes:
        if h.get("verdict") in counts:
            counts[h["verdict"]] += 1
    return counts


def to_csv(hexes):
    """CSV text: one row per hex with its attributes and verdict."""
    buf = io.StringIO()
    cols = [k for k, _ in HEX_FIELDS] + ["verdict", "verdict_source", "gemma_fault", "gemma_sealing",
                                         "gemma_confidence", "gemma_location", "gemma_evidence_en",
                                         "gemma_evidence_ms", "gemma_error"]
    w = csv.writer(buf)
    w.writerow(cols)
    for h in hexes:
        g = h.get("gemma") or {}
        row = []
        for k in cols:
            if k.startswith("gemma_"):
                v = g.get(k[6:], "")
            else:
                v = h.get(k, "")
            if isinstance(v, float):
                v = "" if not np.isfinite(v) else round(v, 4)
            row.append(v)
        w.writerow(row)
    return buf.getvalue()
