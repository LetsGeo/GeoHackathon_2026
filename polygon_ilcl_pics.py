import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import segyio
except ImportError as exc:
    raise ImportError("Please install segyio: pip install segyio") from exc

# ---------------------------------------------------------------------------
# Data Structure
# ---------------------------------------------------------------------------

@dataclass
class Volume3DSeismic:
    """Container for a full 3D seismic volume."""
    ilines: np.ndarray          # shape: (n_ilines,), int64
    xlines: np.ndarray          # shape: (n_xlines,), int64
    twt_ms: np.ndarray          # shape: (n_samples,), float64
    dt_ms: float                # sample interval in ms
    data: np.ndarray            # shape: (n_ilines, n_xlines, n_samples), float32 (to save memory)


# ---------------------------------------------------------------------------
# 1. Fault-Tolerant 3D Loader
# ---------------------------------------------------------------------------

def _extract_time_axis(f: segyio.SegyFile, default_dt_ms: float = 4.0) -> Tuple[np.ndarray, float]:
    """Safely extracts the Two-Way Time array and sample rate."""
    raw_samples = getattr(f, "samples", None)

    try:
        dt_us = segyio.tools.dt(f)
        dt_ms = float(dt_us) / 1000.0 if dt_us and dt_us > 0 else default_dt_ms
    except Exception:
        dt_ms = default_dt_ms

    if raw_samples is not None and len(raw_samples) > 0:
        twt_ms = np.asarray(raw_samples, dtype=np.float64)
        if len(twt_ms) >= 2:
            inferred_dt = float(twt_ms[1] - twt_ms[0])
            if inferred_dt > 0:
                dt_ms = inferred_dt
    else:
        n_samples = f.trace.shape if isinstance(f.trace.shape, int) else len(f.trace[0])
        twt_ms = np.arange(n_samples, dtype=np.float64) * dt_ms

    return twt_ms, dt_ms


def read_3d_volume(
    segy_path: Union[str, Path],
    iline_byte: int = 189,
    xline_byte: int = 193,
) -> Volume3DSeismic:
    """
    Reads a full 3D SEG-Y volume into memory.
    Uses segyio's fast cube if regular, falls back to manual memory mapping if irregular.
    """
    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {segy_path}")

    try:
        # Attempt 1: Fast 3D regular grid mode
        with segyio.open(
            str(segy_path), mode="r", iline=iline_byte, xline=xline_byte, strict=True
        ) as f:
            if f.ilines is None or f.xlines is None:
                raise RuntimeError("Irregular grid detected. Falling back to trace mapping.")

            ilines = np.asarray(f.ilines, dtype=np.int64)
            xlines = np.asarray(f.xlines, dtype=np.int64)
            twt_ms, dt_ms = _extract_time_axis(f)

            # Load full cube. Cast to float32 immediately to save 50% RAM.
            data = segyio.tools.cube(f).astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            return Volume3DSeismic(ilines, xlines, twt_ms, dt_ms, data)

    except Exception as e:
        # Attempt 2: Unstructured Fallback (Trace-by-Trace Mapping)
        print(f"      [!] Fast 3D load failed ({type(e).__name__}). Constructing 3D grid manually...")

        with segyio.open(str(segy_path), mode="r", ignore_geometry=True) as f:
            twt_ms, dt_ms = _extract_time_axis(f)

            # 1. Read all headers to determine the exact grid boundaries
            all_ilines = np.asarray(f.attributes(iline_byte)[:], dtype=np.int64)
            all_xlines = np.asarray(f.attributes(xline_byte)[:], dtype=np.int64)

            ilines = np.unique(all_ilines)
            xlines = np.unique(all_xlines)

            if ilines.size == 0 or xlines.size == 0:
                raise ValueError(f"Could not find valid headers at bytes IL={iline_byte}, XL={xline_byte}")

            # 2. ???
            il_idx = np.searchsorted(ilines, all_ilines)
            xl_idx = np.searchsorted(xlines, all_xlines)

            # 3. Pre-allocate the 3D volume filled with zeros (using float32 to protect RAM)
            n_il, n_xl, n_samp = len(ilines), len(xlines), len(twt_ms)
            data3d = np.zeros((n_il, n_xl, n_samp), dtype=np.float32)

            # 4. Stream traces into their exact 3D coordinate slot
            # (Dead/missing traces in the survey simply remain as 0.0)
            print(f"      Mapping {f.tracecount} traces to [{n_il} IL x {n_xl} XL] grid...")
            data3d[il_idx, xl_idx] = f.trace.raw[:]

            data3d = np.nan_to_num(data3d, nan=0.0, posinf=0.0, neginf=0.0)

            return Volume3DSeismic(ilines, xlines, twt_ms, dt_ms, data3d)


# ---------------------------------------------------------------------------
# 2. 3D Volume Alignment (Intersection)
# ---------------------------------------------------------------------------

def align_3d_volumes(
    base_vol: Volume3DSeismic,
    mon_vol: Volume3DSeismic,
) -> Tuple[Volume3DSeismic, Volume3DSeismic]:
    """
    Slices both 3D matrices so they share the exact same Inline, Crossline, and Time limits.
    """
    # Find intersecting Inlines and Crosslines
    common_ilines, base_il_idx, mon_il_idx = np.intersect1d(base_vol.ilines, mon_vol.ilines, return_indices=True)
    common_xlines, base_xl_idx, mon_xl_idx = np.intersect1d(base_vol.xlines, mon_vol.xlines, return_indices=True)

    if len(common_ilines) == 0 or len(common_xlines) == 0:
        raise ValueError("Baseline and Monitor have ZERO spatial overlap! Check header byte positions.")

    # Find common time samples (truncate to the shortest trace)
    n_samples = min(len(base_vol.twt_ms), len(mon_vol.twt_ms))
    twt_ms = base_vol.twt_ms[:n_samples]

    # Slice Baseline Data: axis 0 (ilines), then axis 1 (xlines), then axis 2 (time)
    b_data = base_vol.data[base_il_idx, :, :]
    b_data = b_data[:, base_xl_idx, :]
    b_data = b_data[:, :, :n_samples]

    aligned_base = Volume3DSeismic(
        ilines=common_ilines,
        xlines=common_xlines,
        twt_ms=twt_ms,
        dt_ms=base_vol.dt_ms,
        data=b_data,
    )

    # Slice Monitor Data
    m_data = mon_vol.data[mon_il_idx, :, :]
    m_data = m_data[:, mon_xl_idx, :]
    m_data = m_data[:, :, :n_samples]

    aligned_mon = Volume3DSeismic(
        ilines=common_ilines,
        xlines=common_xlines,
        twt_ms=twt_ms,
        dt_ms=mon_vol.dt_ms,
        data=m_data,
    )

    return aligned_base, aligned_mon


# ---------------------------------------------------------------------------
# 3. Core Loader Entrypoint
# ---------------------------------------------------------------------------

CACHE = "/data/epic20/APGCE/apgce_team_07/dtw/qc_images"
def load_and_align_3d_pair(
    base_path: Union[str, Path],
    monitor_path: Union[str, Path],
    iline_byte: int = 189,
    xline_byte: int = 193,
) -> Tuple[Volume3DSeismic, Volume3DSeismic]:
    """
    Reads whole Baseline and Monitor volumes, aligns them, and returns two matching 3D matrices.
    """
    print(f"[1/3] Reading Baseline 3D Volume: {base_path}")
    base_vol = read_3d_volume(base_path, iline_byte, xline_byte)

    print(f"[2/3] Reading Monitor 3D Volume: {monitor_path}")
    mon_vol = read_3d_volume(monitor_path, iline_byte, xline_byte)

    print(f"[3/3] Aligning 3D geometries...")
    base_aligned, mon_aligned = align_3d_volumes(base_vol, mon_vol)

    n_il = len(base_aligned.ilines)
    n_xl = len(base_aligned.xlines)
    n_samp = len(base_aligned.twt_ms)

    print(f"      -> Success! Aligned 3D Volumes to shape: [{n_il} IL x {n_xl} XL x {n_samp} Samples]")

    np.save(f'{CACHE}/base_aligned.npy', base_aligned)
    np.save(f'{CACHE}/mon_aligned.npy', mon_aligned)

    return base_aligned, mon_aligned

def load_cache_or_load_and_align_3d_pair(
    base_path: Union[str, Path],
    monitor_path: Union[str, Path],
    iline_byte: int = 189,
    xline_byte: int = 193,
) -> Tuple[Volume3DSeismic, Volume3DSeismic]:
    try:
        base_aligned = np.load(f'{CACHE}/base_aligned.npy', allow_pickle=True).item()
        mon_aligned = np.load(f'{CACHE}/mon_aligned.npy', allow_pickle=True).item()
        return base_aligned, mon_aligned
    except:
        return load_and_align_3d_pair(base_path, monitor_path, iline_byte, xline_byte)


# In[2]:


base_vol, mon_vol = load_cache_or_load_and_align_3d_pair(
    base_path="/data/epic20/APGCE/input_data/challenge3/volumes/Challenge3_Baseline_FilteredScaled_FullStack.sgy",
    monitor_path="/data/epic20/APGCE/input_data/challenge3/volumes/Challenge3_Monitor_FilteredScaled_FullStack.sgy"
)

import numpy as np

s32 = np.sqrt(3) / 2
SCALE = 110 # Number of ilines/xlines for desired size radius, hardcoded to 10 for ~250m, ignoring non-square il/cl
HEX = np.array([[1, 0], [0.5, -s32], [-0.5, -s32], [-1, 0], [-0.5, s32], [0.5, s32]]) * SCALE

def hexagon(center_iline: int, center_xline: int):
    """
    center_iline: the actual iline number (NOT INDEX in Volume3DSeismic.ilines)
    center_xline: the actual xline number (NOT INDEX in Volume3DSeismic.xlines)
    """
    return np.add(HEX, [center_iline, center_xline])

# Hexagons to picture: ID -> (centre inline, centre crossline), in line numbers (NOT indices).
# The ID names the pictures: hex<ID>_IL_<inline>.png and hex<ID>_CL_<crossline>.png
HEXAGONS = {
    1: (1006, 2412),
}
N_LINES = 10  # inlines and crosslines per hexagon, spread evenly across it

import matplotlib
from matplotlib.colors import LinearSegmentedColormap, Normalize

seismic_rwb_data = [
    (0.000, "#d8140a"),  # red
    (0.024, "#f44601"),
    (0.049, "#fc8000"),  # orange
    (0.073, "#ffbb03"),
    (0.122, "#f9f311"),  # bright yellow
    (0.171, "#eed942"),
    (0.244, "#dcbb71"),  # tan
    (0.341, "#e8d2a6"),
    (0.415, "#f1e5cb"),  # cream
    (0.490, "#fdfaf2"),  # white (zero)
    (0.537, "#edeeeb"),
    (0.585, "#d2d5d2"),
    (0.634, "#bbbbbd"),
    (0.683, "#a6a3a6"),
    (0.732, "#8a8a8a"),
    (0.780, "#716e73"),
    (0.829, "#595959"),
    (0.878, "#444144"),
    (0.927, "#282526"),
    (1.000, "#040200"),  # black
]

seismic_rwb = LinearSegmentedColormap.from_list("Seismic RWB", seismic_rwb_data, N=256)

# Optional: register so it can be used by name, e.g. cmap="Seismic RWB"
if "Seismic RWB" not in matplotlib.colormaps:  # re-running the cell would otherwise raise an error
    matplotlib.colormaps.register(cmap=seismic_rwb)

import matplotlib.pyplot as plt

DIR = '/data/epic20/APGCE/apgce_team_07/friday/ilcl_pics'
NORM = Normalize(vmin=-1e5, vmax=1e5, clip=True)


def hex_chord(corners, value, axis):
    """Where one line crosses the hexagon.
    corners: (6, 2) array of [iline, xline] corners.
    axis=0: the inline numbered `value`    -> (first xline, last xline) inside the hexagon
    axis=1: the crossline numbered `value` -> (first iline, last iline) inside the hexagon
    Returns None if the line misses the hexagon."""
    c = np.asarray(corners, float)
    centre = c.mean(axis=0)
    c = c[np.argsort(np.arctan2(c[:, 0] - centre[0], c[:, 1] - centre[1]))]  # corners in order
    fixed, free = c[:, axis], c[:, 1 - axis]
    hits = []
    for k in range(len(c)):
        a0, a1 = fixed[k], fixed[(k + 1) % len(c)]
        b0, b1 = free[k], free[(k + 1) % len(c)]
        if a0 == a1:  # edge runs along the line
            if a0 == value:
                hits += [b0, b1]
            continue
        t = (value - a0) / (a1 - a0)
        if 0 <= t <= 1:
            hits.append(b0 + t * (b1 - b0))
    return (min(hits), max(hits)) if hits else None


def index_range(lines, lo, hi):
    """start, stop for slicing the lines numbered lo..hi (inclusive), clipped to the survey,
    so a hexagon hanging off the edge can't go out of bounds."""
    lines = np.asarray(lines)
    if lines[0] > lines[-1]:  # numbers decreasing along the array
        s, e = index_range(lines[::-1], lo, hi)
        return len(lines) - e, len(lines) - s
    return int(np.searchsorted(lines, lo, "left")), int(np.searchsorted(lines, hi, "right"))


def lines_across(corners, lines, axis, n=N_LINES):
    """Indices of n lines spread evenly across the hexagon (axis=0: inlines, axis=1: crosslines),
    keeping off its very tips (only a few traces wide) and keeping only lines inside the survey."""
    c = np.asarray(corners, float)[:, axis]
    lo, hi = c.min(), c.max()
    margin = 0.1 * (hi - lo)
    lines = np.asarray(lines)
    idx = [int(np.abs(lines - w).argmin()) for w in np.linspace(lo + margin, hi - margin, n)]
    return [i for i in dict.fromkeys(idx) if lo <= lines[i] <= hi]


def hex_name(hex_id):
    return f"hex{hex_id:02d}" if isinstance(hex_id, (int, np.integer)) else f"hex{hex_id}"


saved = 0
for hex_id, (center_iline, center_xline) in HEXAGONS.items():
    corners = hexagon(center_iline, center_xline)
    name = hex_name(hex_id)
    print(f"{name}: centre inline {center_iline} / crossline {center_xline}")
    if not lines_across(corners, base_vol.ilines, axis=0) and not lines_across(corners, base_vol.xlines, axis=1):
        print("  no part of this hexagon is inside the survey: skipped")
        continue

    for il_idx in lines_across(corners, base_vol.ilines, axis=0):
        iline = base_vol.ilines[il_idx]
        chord = hex_chord(corners, iline, axis=0)
        if chord is None:
            continue
        j0, j1 = index_range(base_vol.xlines, *chord)  # the crosslines inside the hexagon on this inline
        if j1 - j0 < 2:
            continue
        plt.clf()
        plt.imshow(
            base_vol.data[il_idx, j0:j1, :].T,
            cmap='Seismic RWB',
            norm=NORM,
            # aspect='auto',
            extent=(base_vol.xlines[j0], base_vol.xlines[j1 - 1], base_vol.twt_ms.max(), base_vol.twt_ms.min())
        )
        plt.gca().invert_xaxis()  # crosslines decreasing to the right, as before
        plt.title(f'{name} · Inline {iline}')
        path = f'{DIR}/{name}_IL_{iline}.png'
        plt.savefig(path)
        saved += 1
        print(f'  saved {path}')

    for xl_idx in lines_across(corners, base_vol.xlines, axis=1):
        xline = base_vol.xlines[xl_idx]
        chord = hex_chord(corners, xline, axis=1)
        if chord is None:
            continue
        i0, i1 = index_range(base_vol.ilines, *chord)  # the inlines inside the hexagon on this crossline
        if i1 - i0 < 2:
            continue
        plt.clf()
        plt.imshow(
            base_vol.data[i0:i1, xl_idx, :].T,
            cmap='Seismic RWB',
            norm=NORM,
            # aspect='auto',
            extent=(base_vol.ilines[i0], base_vol.ilines[i1 - 1], base_vol.twt_ms.max(), base_vol.twt_ms.min())
        )
        plt.title(f'{name} · Crossline {xline}')
        path = f'{DIR}/{name}_CL_{xline}.png'
        plt.savefig(path)
        saved += 1
        print(f'  saved {path}')

print(f"Saved {saved} pictures of {len(HEXAGONS)} hexagon(s) in {DIR}")
