import numpy as np

s32 = np.sqrt(3) / 2
SCALE = 10  # Number of ilines/xlines for desired size, hardcoded to 10 for ~250m, ignoring non-square il/xl
HEX = np.array([[1, 0], [0.5, -s32], [-0.5, -s32], [-1, 0], [-0.5, s32], [0.5, s32]]) * SCALE


def hexagon(center_iline: float, center_xline: float):
    """
    center_iline: the actual iline number (NOT INDEX in Volume3DSeismic.ilines)
    center_xline: the actual xline number (NOT INDEX in Volume3DSeismic.xlines)
    """
    return np.add(HEX, [center_iline, center_xline])


def hexagon_rings(center_iline: float, center_xline: float, n_rings: int = 2):
    """
    One hexagon in the middle with n_rings rings of hexagons around it, all sharing edges.
    n_rings = 2 -> 1 + 6 + 12 = 19 hexagons.

    center_iline / center_xline: centre of the middle hexagon, in actual line numbers.
    Returns (centers, hexagons, ring):
        centers  -> array of shape (N, 2)    [iline, xline] of each centre
        hexagons -> array of shape (N, 6, 2) the 6 corners of each hexagon
        ring     -> array of shape (N,)      which ring each hexagon is in (0 = middle)
    Hexagons are ordered middle first, then ring 1, then ring 2, ...
    A hexagon's ID is its position in this order: 0 = middle, 1-6 = ring 1, 7-18 = ring 2
    (save_hexagon_sections names its images with it).
    """
    # Axial hex coordinates (q, r): a cell is in ring k when max(|q|, |r|, |q + r|) == k
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

    # Flat-topped layout: neighbouring columns are 1.5 * SCALE apart, rows sqrt(3) * SCALE apart
    centers = np.column_stack([
        center_iline + 1.5 * SCALE * q,
        center_xline + 2 * s32 * SCALE * (r + q / 2),
    ])
    hexagons = np.array([hexagon(il, xl) for il, xl in centers])
    return centers, hexagons, ring


def centered_hexagon_rings(ilines, xlines, n_rings: int = 2):
    """
    Same as hexagon_rings, but puts the middle hexagon in the middle of the survey.

    ilines / xlines: the actual line numbers (e.g. Volume3DSeismic.ilines / .xlines)
    """
    center_il = (np.min(ilines) + np.max(ilines)) / 2
    center_xl = (np.min(xlines) + np.max(xlines)) / 2
    return hexagon_rings(center_il, center_xl, n_rings)


# ---------------------------------------------------------------------------
# Inline / crossline pictures of each hexagon
# ---------------------------------------------------------------------------
LINE_OFFSETS = (-8, -4, 0, 4, 8)  # the 5 inlines / crosslines per hexagon, in lines from its centre


def hex_chord(corners, value, axis):
    """Where one line crosses a hexagon.
    corners: (6, 2) array of [iline, xline] corners (any order).
    axis=0: the inline numbered `value`   -> (first xline, last xline) inside the hexagon
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
    so a hexagon hanging off the edge can never index out of bounds."""
    lines = np.asarray(lines)
    if lines[0] > lines[-1]:  # numbers decreasing along the array
        s, e = index_range(lines[::-1], lo, hi)
        return len(lines) - e, len(lines) - s
    return int(np.searchsorted(lines, lo, "left")), int(np.searchsorted(lines, hi, "right"))


def lines_in_hex(corners, lines, axis, offsets=LINE_OFFSETS):
    """Indices of the lines at `offsets` from the hexagon's centre, keeping only those
    inside both the hexagon and the survey (axis=0: inlines, axis=1: crosslines)."""
    c = np.asarray(corners, float)[:, axis]
    lines = np.asarray(lines)
    step = np.median(np.diff(lines))
    wanted = c.mean() + np.asarray(offsets) * abs(step)
    idx = np.clip(np.round((wanted - lines[0]) / step).astype(int), 0, len(lines) - 1)
    return [int(i) for i in dict.fromkeys(idx) if c.min() <= lines[i] <= c.max()]


def save_hexagon_sections(data, ilines, xlines, twt, hexagons, out_dir, ids=None, offsets=LINE_OFFSETS,
                          vmax=1e5, cmap="seismic", dpi=150, verbose=True):
    """Save 5 inline and 5 crossline pictures of every hexagon, each cut to the part of the
    line inside the hexagon, named after the hexagon's ID:
        hex07_IL_1840.png, hex07_CL_1102.png, ...

    data:     the seismic cube, shape (n_ilines, n_xlines, n_samples), e.g. Volume3DSeismic.data
    ilines, xlines, twt: its line numbers and times (ms), e.g. .ilines, .xlines, .twt_ms
    hexagons: (N, 6, 2) corners in [iline, xline] line numbers, e.g. from hexagon_rings()
    ids:      one ID per hexagon; default 0..N-1, the hexagon_rings order (0 = middle)
    vmax:     colour scale is -vmax..vmax
    Returns the list of saved file paths."""
    from pathlib import Path

    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure  # no pyplot: nothing piles up in a notebook

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ilines, xlines, twt = np.asarray(ilines), np.asarray(xlines), np.asarray(twt)
    ids = list(range(len(hexagons))) if ids is None else list(ids)
    if len(ids) != len(hexagons):
        raise ValueError(f"{len(ids)} IDs for {len(hexagons)} hexagons")
    norm = Normalize(vmin=-vmax, vmax=vmax, clip=True)
    t_top, t_bottom = float(np.min(twt)), float(np.max(twt))
    width = max(2, len(str(max(ids, key=lambda v: len(str(v))))))  # hex07, hex18, ... (or hex123)
    saved = []

    def save(section, x0, x1, title, xlabel, name):
        fig = Figure(figsize=(4, 6))
        ax = fig.add_subplot(111)
        ax.imshow(np.asarray(section, dtype=np.float32).T, cmap=cmap, norm=norm, aspect="auto",
                  extent=(float(x0), float(x1), t_bottom, t_top))  # time increases downwards
        ax.set(title=title, xlabel=xlabel, ylabel="TWT (ms)")
        path = out_dir / name
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved.append(path)
        if verbose:
            print(f"  saved {path.name}")

    for hid, corners in zip(ids, hexagons):
        tag = f"hex{hid:0{width}d}" if isinstance(hid, (int, np.integer)) else f"hex{hid}"
        il_idx = lines_in_hex(corners, ilines, axis=0, offsets=offsets)
        xl_idx = lines_in_hex(corners, xlines, axis=1, offsets=offsets)
        if verbose:
            print(f"{tag}: {len(il_idx)} inlines, {len(xl_idx)} crosslines")
        for i in il_idx:
            chord = hex_chord(corners, ilines[i], axis=0)
            if chord is None:
                continue
            j0, j1 = index_range(xlines, *chord)
            if j1 - j0 >= 2:
                save(data[i, j0:j1, :], xlines[j0], xlines[j1 - 1], f"{tag} · Inline {ilines[i]}", "Crossline",
                     f"{tag}_IL_{ilines[i]}.png")
        for j in xl_idx:
            chord = hex_chord(corners, xlines[j], axis=1)
            if chord is None:
                continue
            i0, i1 = index_range(ilines, *chord)
            if i1 - i0 >= 2:
                save(data[i0:i1, j, :], ilines[i0], ilines[i1 - 1], f"{tag} · Crossline {xlines[j]}", "Inline",
                     f"{tag}_CL_{xlines[j]}.png")
    if verbose:
        print(f"Saved {len(saved)} pictures of {len(hexagons)} hexagons in {out_dir}")
    return saved
