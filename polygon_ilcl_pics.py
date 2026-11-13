import sys
from pathlib import Path

import numpy as np

# The 3D SEG-Y loader / aligner lives in volume3d.py (same folder as this file).
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:  # pasted into a notebook cell: run it from the repo folder
    sys.path.insert(0, str(Path.cwd()))
from volume3d import Volume3DSeismic, load_cache_or_load_and_align_3d_pair  # noqa: E402


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


def hexagon_rings(center_iline: float, center_xline: float, n_rings: int = 2):
    """One hexagon in the middle with n_rings rings of hexagons around it, all sharing edges
    (same layout and order as hexagon_rings in polygon.py, but with this file's SCALE).
    n_rings = 2 -> 1 + 6 + 12 = 19 hexagons, ordered middle first, then ring 1, then ring 2.
    Returns (centers, ring): centers (N, 2) [iline, xline] of each centre, ring (N,) its ring."""
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
    centers = np.column_stack([
        center_iline + 1.5 * SCALE * q,
        center_xline + 2 * s32 * SCALE * (r + q / 2),
    ])
    return centers, ring


# All the hexagons in the polygon: 19 = 1 in the middle + a ring of 6 + a ring of 12, centred
# on the middle hexagon below (line numbers, NOT indices).
# IDs follow the hexagon_rings order: 0 = middle, 1-6 = ring 1, 7-18 = ring 2.
# The ID names the pictures: hex<ID>_IL_<inline>.png and hex<ID>_CL_<crossline>.png
CENTER_ILINE, CENTER_XLINE = 1006, 2412
N_RINGS = 2
centers, rings = hexagon_rings(CENTER_ILINE, CENTER_XLINE, N_RINGS)
HEXAGONS = {hex_id: (float(il), float(xl)) for hex_id, (il, xl) in enumerate(centers)}
RING = {hex_id: int(k) for hex_id, k in enumerate(rings)}
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

# 4D difference (monitor - baseline) pictures of the same sections, saved next to the
# baseline ones as hex<ID>_IL_<inline>_diff.png / hex<ID>_CL_<crossline>_diff.png
DIFF = True
DIFF_CMAP = 'seismic'   # blue = weaker in the monitor, red = stronger, white = no change
DIFF_VMAX = None        # colour range -DIFF_VMAX..DIFF_VMAX; None = 99th percentile of |difference|


def difference_vmax(base_vol, mon_vol, target=2_000_000):
    """99th percentile of |monitor - baseline| from an evenly spread sample of the volume."""
    n_il, n_xl, n_t = base_vol.data.shape
    step = max(1, int(np.sqrt(n_il * n_xl * n_t / target)))
    d = np.asarray(mon_vol.data[::step, ::step, :], dtype=np.float32) - \
        np.asarray(base_vol.data[::step, ::step, :], dtype=np.float32)
    a = np.abs(d)
    v = float(np.percentile(a, 99))
    if v <= 0:  # the change covers under 1% of the samples: scale on the samples that changed
        a = a[a > 0]
        v = float(np.percentile(a, 99)) if a.size else 1.0
    return v


def save_section(section, x_first, x_last, title, path, cmap, norm, invert_x=False):
    plt.clf()
    plt.imshow(
        np.asarray(section, dtype=np.float32).T,
        cmap=cmap,
        norm=norm,
        # aspect='auto',
        extent=(x_first, x_last, base_vol.twt_ms.max(), base_vol.twt_ms.min())
    )
    if invert_x:
        plt.gca().invert_xaxis()
    plt.title(title)
    plt.savefig(path)


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


if DIFF:
    if base_vol.data.shape != mon_vol.data.shape:
        raise ValueError("base_vol and mon_vol must be aligned (same lines and times) for difference pictures")
    diff_vmax = DIFF_VMAX or difference_vmax(base_vol, mon_vol)
    DIFF_NORM = Normalize(vmin=-diff_vmax, vmax=diff_vmax, clip=True)
    print(f"Difference pictures: monitor - baseline, colour range +-{diff_vmax:.4g}")

saved, skipped = 0, []
for hex_id, (center_iline, center_xline) in HEXAGONS.items():
    corners = hexagon(center_iline, center_xline)
    name = hex_name(hex_id)
    print(f"{name} (ring {RING.get(hex_id, '-')}): centre inline {center_iline:.0f} / crossline {center_xline:.0f}")
    if not lines_across(corners, base_vol.ilines, axis=0) and not lines_across(corners, base_vol.xlines, axis=1):
        print("  no part of this hexagon is inside the survey: skipped")
        skipped.append(name)
        continue

    for il_idx in lines_across(corners, base_vol.ilines, axis=0):
        iline = base_vol.ilines[il_idx]
        chord = hex_chord(corners, iline, axis=0)
        if chord is None:
            continue
        j0, j1 = index_range(base_vol.xlines, *chord)  # the crosslines inside the hexagon on this inline
        if j1 - j0 < 2:
            continue
        x0, x1 = base_vol.xlines[j0], base_vol.xlines[j1 - 1]
        base = base_vol.data[il_idx, j0:j1, :]
        path = f'{DIR}/{name}_IL_{iline}.png'
        # invert_x: crosslines decreasing to the right, as before
        save_section(base, x0, x1, f'{name} · Inline {iline}', path, 'Seismic RWB', NORM, invert_x=True)
        saved += 1
        if DIFF:
            diff = np.asarray(mon_vol.data[il_idx, j0:j1, :], dtype=np.float32) - np.asarray(base, dtype=np.float32)
            save_section(diff, x0, x1, f'{name} · Inline {iline} · monitor - baseline',
                         path.replace('.png', '_diff.png'), DIFF_CMAP, DIFF_NORM, invert_x=True)
            saved += 1
        print(f'  saved {path}' + (' + _diff' if DIFF else ''))

    for xl_idx in lines_across(corners, base_vol.xlines, axis=1):
        xline = base_vol.xlines[xl_idx]
        chord = hex_chord(corners, xline, axis=1)
        if chord is None:
            continue
        i0, i1 = index_range(base_vol.ilines, *chord)  # the inlines inside the hexagon on this crossline
        if i1 - i0 < 2:
            continue
        x0, x1 = base_vol.ilines[i0], base_vol.ilines[i1 - 1]
        base = base_vol.data[i0:i1, xl_idx, :]
        path = f'{DIR}/{name}_CL_{xline}.png'
        save_section(base, x0, x1, f'{name} · Crossline {xline}', path, 'Seismic RWB', NORM)
        saved += 1
        if DIFF:
            diff = np.asarray(mon_vol.data[i0:i1, xl_idx, :], dtype=np.float32) - np.asarray(base, dtype=np.float32)
            save_section(diff, x0, x1, f'{name} · Crossline {xline} · monitor - baseline',
                         path.replace('.png', '_diff.png'), DIFF_CMAP, DIFF_NORM)
            saved += 1
        print(f'  saved {path}' + (' + _diff' if DIFF else ''))

print(f"Saved {saved} pictures of {len(HEXAGONS) - len(skipped)} of {len(HEXAGONS)} hexagons in {DIR}"
      + (f" (outside the survey: {', '.join(skipped)})" if skipped else ""))
# Next: score each hexagon with Gemma (ID + % likelihood of a reservoir):
#   %run gemma_ilcl.py          (reads the pictures in DIR, writes DIR/gemma/scores.csv)
