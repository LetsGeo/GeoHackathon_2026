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
