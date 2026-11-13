"""
3D SEG-Y volume loader & aligner (shared by the notebook scripts, e.g. polygon_ilcl_pics.py).

Reads full 3D volumes into memory, handles irregular grids, and aligns Baseline and
Monitor on the same inlines, crosslines and times.

    from volume3d import load_cache_or_load_and_align_3d_pair
    base_vol, mon_vol = load_cache_or_load_and_align_3d_pair(base_path, monitor_path)
    base_vol.data[il_idx, xl_idx, :]      # (n_ilines, n_xlines, n_samples), float32

Cache: the aligned pair is saved under CACHE in a folder named after the two input
files (path, size, date and header bytes), as plain .npy arrays. Loading it again is
instant (memory-mapped, so it doesn't fill RAM), and pointing at different files can
never return an old pair. Delete the folder any time to free the space.
"""
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np

try:
    import segyio
except ImportError as exc:
    raise ImportError("Please install segyio: pip install segyio") from exc

CACHE = "/data/epic20/APGCE/apgce_team_07/dtw/qc_images"
CHUNK_TRACES = 50_000  # traces read per step on the irregular-grid path (keeps memory down)


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
def _extract_time_axis(f: "segyio.SegyFile", default_dt_ms: float = 4.0) -> Tuple[np.ndarray, float]:
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
    Uses segyio's fast cube if regular, falls back to manual trace mapping if irregular.
    """
    segy_path = Path(segy_path)
    if not segy_path.exists():
        raise FileNotFoundError(f"SEG-Y file not found: {segy_path}")

    try:
        # Attempt 1: Fast 3D regular grid mode
        with segyio.open(str(segy_path), mode="r", iline=iline_byte, xline=xline_byte, strict=True) as f:
            if f.ilines is None or f.xlines is None:
                raise RuntimeError("Irregular grid detected. Falling back to trace mapping.")
            ilines = np.asarray(f.ilines, dtype=np.int64)
            xlines = np.asarray(f.xlines, dtype=np.int64)
            twt_ms, dt_ms = _extract_time_axis(f)
            data = segyio.tools.cube(f)
            if f.sorting == segyio.TraceSortingFormat.CROSSLINE_SORTING:
                data = data.transpose(1, 0, 2)  # cube() is (xline, iline, t) for crossline-sorted files
            data = np.ascontiguousarray(data, dtype=np.float32)  # float32 to save 50% RAM
            np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            return Volume3DSeismic(ilines, xlines, twt_ms, dt_ms, data)

    except Exception as e:
        # Attempt 2: Unstructured Fallback (trace-by-trace mapping)
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

        # 2. Grid position (index) of every trace
        il_idx = np.searchsorted(ilines, all_ilines)
        xl_idx = np.searchsorted(xlines, all_xlines)

        # 3. Pre-allocate the 3D volume filled with zeros (float32 to protect RAM)
        n_il, n_xl, n_samp = len(ilines), len(xlines), len(twt_ms)
        data3d = np.zeros((n_il, n_xl, n_samp), dtype=np.float32)

        # 4. Stream traces into their 3D slot, a large block at a time
        # (dead / missing traces simply remain 0.0)
        print(f"      Mapping {f.tracecount} traces to [{n_il} IL x {n_xl} XL] grid...")
        f.mmap()
        for a in range(0, f.tracecount, CHUNK_TRACES):
            b = min(a + CHUNK_TRACES, f.tracecount)
            data3d[il_idx[a:b], xl_idx[a:b]] = f.trace.raw[a:b]
        np.nan_to_num(data3d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return Volume3DSeismic(ilines, xlines, twt_ms, dt_ms, data3d)


# ---------------------------------------------------------------------------
# 2. 3D Volume Alignment (Intersection)
# ---------------------------------------------------------------------------
def _take(a, idx, axis):
    """a[idx] along axis; a view (no copy) when idx is a contiguous run."""
    idx = np.asarray(idx)
    if len(idx) and np.all(np.diff(idx) == 1):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(int(idx[0]), int(idx[-1]) + 1)
        return a[tuple(sl)]
    return np.take(a, idx, axis=axis)


def align_3d_volumes(
    base_vol: Volume3DSeismic,
    mon_vol: Volume3DSeismic,
) -> Tuple[Volume3DSeismic, Volume3DSeismic]:
    """
    Slices both 3D matrices so they share the exact same Inline, Crossline and Time samples.
    Times are matched by value, so surveys that start at different times still line up;
    different sample intervals are refused (they would need resampling).
    """
    # Find intersecting Inlines and Crosslines
    common_ilines, base_il_idx, mon_il_idx = np.intersect1d(base_vol.ilines, mon_vol.ilines, return_indices=True)
    common_xlines, base_xl_idx, mon_xl_idx = np.intersect1d(base_vol.xlines, mon_vol.xlines, return_indices=True)
    if len(common_ilines) == 0 or len(common_xlines) == 0:
        raise ValueError("Baseline and Monitor have ZERO spatial overlap! Check header byte positions.")

    # Common time samples, matched by value
    if not np.isclose(base_vol.dt_ms, mon_vol.dt_ms):
        raise ValueError(f"Baseline and Monitor have different sample intervals ({base_vol.dt_ms} vs "
                         f"{mon_vol.dt_ms} ms); resample one of them first.")
    tol = 0.01 * base_vol.dt_ms
    t_lo = max(base_vol.twt_ms[0], mon_vol.twt_ms[0])
    t_hi = min(base_vol.twt_ms[-1], mon_vol.twt_ms[-1])
    b_t = np.where((base_vol.twt_ms >= t_lo - tol) & (base_vol.twt_ms <= t_hi + tol))[0]
    m_t = np.where((mon_vol.twt_ms >= t_lo - tol) & (mon_vol.twt_ms <= t_hi + tol))[0]
    n = min(len(b_t), len(m_t))
    b_t, m_t = b_t[:n], m_t[:n]
    if n == 0 or not np.allclose(base_vol.twt_ms[b_t], mon_vol.twt_ms[m_t], atol=tol):
        raise ValueError("Baseline and Monitor time samples don't line up (their sample times are offset "
                         "by a fraction of a sample); resample one of them first.")
    twt_ms = base_vol.twt_ms[b_t]

    def aligned(vol, il_idx, xl_idx, t_idx):
        d = _take(_take(_take(vol.data, il_idx, 0), xl_idx, 1), t_idx, 2)
        return Volume3DSeismic(ilines=common_ilines, xlines=common_xlines, twt_ms=twt_ms,
                               dt_ms=vol.dt_ms, data=d)

    return aligned(base_vol, base_il_idx, base_xl_idx, b_t), aligned(mon_vol, mon_il_idx, mon_xl_idx, m_t)


# ---------------------------------------------------------------------------
# 3. Cache
# ---------------------------------------------------------------------------
def _cache_dir(cache, base_path, monitor_path, iline_byte, xline_byte):
    """Folder named after both input files (full path, size, date) and the header bytes,
    so different inputs never share a cache."""
    parts = []
    for p in (Path(base_path), Path(monitor_path)):
        st = p.stat()
        parts.append(f"{p.resolve()}|{st.st_size}|{st.st_mtime}")
    key = hashlib.sha1(("||".join(parts) + f"|{iline_byte}|{xline_byte}").encode()).hexdigest()[:12]
    return Path(cache) / f"aligned_{Path(base_path).stem}__{Path(monitor_path).stem}_{key}"


def _save_vol(folder, name, vol):
    np.save(folder / f"{name}_data.npy", vol.data)
    np.savez(folder / f"{name}_meta.npz", ilines=vol.ilines, xlines=vol.xlines, twt_ms=vol.twt_ms,
             dt_ms=np.float64(vol.dt_ms))


def _load_vol(folder, name, mmap=True):
    meta = np.load(folder / f"{name}_meta.npz")
    data = np.load(folder / f"{name}_data.npy", mmap_mode="r" if mmap else None)
    return Volume3DSeismic(meta["ilines"], meta["xlines"], meta["twt_ms"], float(meta["dt_ms"]), data)


# ---------------------------------------------------------------------------
# 4. Core Loader Entrypoint
# ---------------------------------------------------------------------------
def load_and_align_3d_pair(
    base_path: Union[str, Path],
    monitor_path: Union[str, Path],
    iline_byte: int = 189,
    xline_byte: int = 193,
    cache: Union[str, Path, None] = CACHE,
) -> Tuple[Volume3DSeismic, Volume3DSeismic]:
    """
    Reads whole Baseline and Monitor volumes, aligns them, and returns two matching 3D matrices
    (and saves them in the cache, unless cache=None).
    """
    print(f"[1/3] Reading Baseline 3D Volume: {base_path}")
    base_vol = read_3d_volume(base_path, iline_byte, xline_byte)

    print(f"[2/3] Reading Monitor 3D Volume: {monitor_path}")
    mon_vol = read_3d_volume(monitor_path, iline_byte, xline_byte)

    print("[3/3] Aligning 3D geometries...")
    base_aligned, mon_aligned = align_3d_volumes(base_vol, mon_vol)
    del base_vol, mon_vol

    n_il, n_xl, n_samp = len(base_aligned.ilines), len(base_aligned.xlines), len(base_aligned.twt_ms)
    print(f"      -> Success! Aligned 3D Volumes to shape: [{n_il} IL x {n_xl} XL x {n_samp} Samples]")

    if cache:
        folder = _cache_dir(cache, base_path, monitor_path, iline_byte, xline_byte)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _save_vol(folder, "base", base_aligned)
            _save_vol(folder, "mon", mon_aligned)
            (folder / "complete").write_text("ok")  # written last: a half-written cache is never used
            print(f"      Cached in {folder}")
        except OSError as exc:  # keep the loaded volumes even if the cache can't be written
            print(f"      [!] Could not write the cache ({exc}); continuing without it.")
    return base_aligned, mon_aligned


def load_cache_or_load_and_align_3d_pair(
    base_path: Union[str, Path],
    monitor_path: Union[str, Path],
    iline_byte: int = 189,
    xline_byte: int = 193,
    cache: Union[str, Path, None] = CACHE,
    mmap: bool = True,
) -> Tuple[Volume3DSeismic, Volume3DSeismic]:
    """The aligned pair from the cache if these exact files were loaded before, otherwise
    read + align (and cache). mmap=True leaves the cached data on disk and reads only what
    is used; mmap=False loads it all into RAM."""
    if cache:
        folder = _cache_dir(cache, base_path, monitor_path, iline_byte, xline_byte)
        if (folder / "complete").exists():
            try:
                base_aligned, mon_aligned = _load_vol(folder, "base", mmap), _load_vol(folder, "mon", mmap)
                print(f"Loaded the aligned pair from the cache: {folder}")
                return base_aligned, mon_aligned
            except (OSError, ValueError, KeyError) as exc:
                print(f"      [!] Cache unreadable ({type(exc).__name__}: {exc}); reading the SEG-Y files again.")
    return load_and_align_3d_pair(base_path, monitor_path, iline_byte, xline_byte, cache)
