"""
SEG-Y loading with a disk cache, shared by the app (app.py) and the command-line
fault screening (hex_screen.py).

The first load of a file reads the whole SEG-Y once and saves it in "cache/"
(about the same size as the file); later loads memory-map that cache.
"""
import gc
import hashlib
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).parent
CACHE_DIR = APP_DIR / "cache"

# Byte positions where inline / crossline numbers are commonly stored in SEG-Y
# trace headers. The loader tries each pair until one describes a proper 3D grid.
HEADER_CANDIDATES = [(189, 193), (9, 21), (5, 21), (17, 13), (9, 13)]
CHUNK_TRACES = 20_000  # traces read per step when building the cache


def cube_cache_dir(path):
    """Cache folder tied to the file's full path, size and date, so different
    datasets (even with the same file name) never share a cache."""
    st = path.stat()
    key = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime}".encode()).hexdigest()[:12]
    return CACHE_DIR / f"{path.name}_{key}"


def robust_amp(cube, target=2_000_000):
    """99th percentile of |amplitude| from an evenly spaced sample of the cube."""
    n_il, n_xl, _ = cube.shape
    step = 1
    while (n_il // step) * (n_xl // step) * cube.shape[2] > target and step < max(n_il, n_xl):
        step += 1
    sample = np.abs(np.asarray(cube[::step, ::step, :], dtype=np.float32))
    value = np.nanpercentile(sample, 99) if np.isfinite(sample).any() else 1.0
    return float(value)


def load_cube(path):
    """Open a 3D post-stack SEG-Y file as an (inline, crossline, time) array.
    The first time, the whole file is read once and saved as cache/…/cube.npy
    (about the same size as the SEG-Y). After that the cache is memory-mapped,
    so any inline, crossline or time slice comes back instantly."""
    folder = cube_cache_dir(path)
    meta_file, cube_file = folder / "meta.npz", folder / "cube.npy"
    if meta_file.exists() and cube_file.exists():
        meta = np.load(meta_file)
        return {
            "cube": np.load(cube_file, mmap_mode="r"),
            "ilines": meta["ilines"], "xlines": meta["xlines"], "times": meta["times"],
            "amp": float(meta["amp"]),
        }

    import segyio  # imported here so the app still opens if segyio isn't installed

    folder.mkdir(parents=True, exist_ok=True)
    tmp_file = folder / "cube.partial.npy"
    with segyio.open(str(path), ignore_geometry=True) as f:
        f.mmap()
        n = f.tracecount
        il = xl = None
        for il_byte, xl_byte in HEADER_CANDIDATES:
            il_try = np.asarray(f.attributes(il_byte)[:]).astype(np.int64)
            xl_try = np.asarray(f.attributes(xl_byte)[:]).astype(np.int64)
            n_il, n_xl = len(np.unique(il_try)), len(np.unique(xl_try))
            unique_pairs = len(np.unique(il_try * 10_000_000 + xl_try))
            if n_il > 1 and n_xl > 1 and unique_pairs >= 0.95 * n:
                il, xl = il_try, xl_try
                break
        if il is None:
            raise ValueError("couldn't find inline/crossline numbers in the trace headers")

        ilines, xlines = np.unique(il), np.unique(xl)
        il_idx, xl_idx = np.searchsorted(ilines, il), np.searchsorted(xlines, xl)
        times = np.asarray(f.samples, dtype=np.float32)

        cube = np.lib.format.open_memmap(
            tmp_file, mode="w+", dtype=np.float32, shape=(len(ilines), len(xlines), len(times)))
        cube[:] = np.nan  # positions with no trace stay empty
        for a in range(0, n, CHUNK_TRACES):
            b = min(a + CHUNK_TRACES, n)
            cube[il_idx[a:b], xl_idx[a:b], :] = f.trace.raw[a:b]
        cube.flush()
        amp = robust_amp(cube)
    del cube
    gc.collect()  # release the file so it can be renamed (needed on Windows)
    tmp_file.replace(cube_file)
    np.savez(meta_file, ilines=ilines, xlines=xlines, times=times, amp=amp)
    return load_cube(path)
