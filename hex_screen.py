"""
Sealing-fault screening from the command line (same steps as the app's
"Fault screening" card, without the UI).

    python hex_screen.py --baseline data/94p10ful --monitor data/10p10ful
    python hex_screen.py --baseline B.sgy --monitor M.sgy --limit 10      # quick test: 10 strongest hexes
    python hex_screen.py --baseline B.sgy --monitor M.sgy --no-gemma      # attributes only

Writes to --out (default fault_screening/):
    hexes.csv        one row per hexagon: attributes, Gemma's answer, verdict
    hexes.json       the same as JSON
    hex_map.png      hexagons over the 4D change map, coloured by verdict
    images/          the two montages sent to Gemma for each hexagon (flagged ones,
                     or all with --save-images)

Gemma settings as for the app: --gemma-url / --gemma-model or GEMMA_BASE_URL /
GEMMA_MODEL; --workers (or GEMMA_WORKERS) requests at the same time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import gemma_client
import hexscreen
from seismic_io import APP_DIR, load_cube

VERDICT_COLOURS = {"likely": "#e5484d", "possible": "#f5a524", "fault": "#6f8fb0"}


def resolve(text):
    p = Path(text).expanduser()
    return p if p.is_absolute() else APP_DIR / p


def save_map(ctx, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Polygon

    grid, maps, st = ctx["grid"], ctx["maps"], ctx["stats"]
    il, xl = grid["ilines"], grid["xlines"]
    ext = (xl[0] - grid["dxl"] / 2, xl[-1] + grid["dxl"] / 2, il[0] - grid["dil"] / 2, il[-1] + grid["dil"] / 2)
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    chg = maps["diff_rms"]
    floor = float(np.nanmedian(chg))
    im = ax.imshow(chg, origin="lower", extent=ext, aspect="auto", cmap="viridis", interpolation="nearest",
                   vmin=floor, vmax=max(float(np.nanpercentile(chg, 99.5)), floor + 1e-9))
    on_fault = np.isfinite(maps["coherence"]) & (1 - maps["coherence"] > st["fault_threshold"])
    overlay = np.zeros(on_fault.shape + (4,))
    overlay[on_fault] = (1, 1, 1, 0.95)
    ax.imshow(overlay, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    for h in ctx["hexes"]:
        c = VERDICT_COLOURS.get(h.get("verdict"))
        ax.add_patch(Polygon(list(zip(h["corners_xl"], h["corners_il"])), closed=True,
                             fc=c or "none", alpha=0.55 if c else 1, ec="white", lw=0.6))
        ax.text(h["xl_centre"], h["il_centre"], str(h["id"]), ha="center", va="center", fontsize=6,
                color="white", weight="bold" if c else "normal")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Inline")
    ax.set_title("Hexagons over the 4D change map (white = traces on a fault)")
    fig.colorbar(im, ax=ax, label="4D change (RMS of monitor - baseline)")
    ax.legend(handles=[Patch(fc=c, alpha=0.7, label=hexscreen.VERDICT_TEXT[v][0]) for v, c in VERDICT_COLOURS.items()],
              loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser(description="Hexagon sealing-fault screening (attributes + Gemma).")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--monitor", required=True)
    ap.add_argument("--baseline-label", default="Baseline")
    ap.add_argument("--monitor-label", default="Monitor")
    ap.add_argument("--out", default="fault_screening")
    ap.add_argument("--width", type=int, default=hexscreen.HEX_WIDTH, help="hexagon width in lines (default 20)")
    ap.add_argument("--limit", type=int, default=None,
                    help="send only the N hexagons with the strongest fault/seal attributes to Gemma")
    ap.add_argument("--no-gemma", action="store_true", help="attributes only")
    ap.add_argument("--gemma-url", default=None)
    ap.add_argument("--gemma-model", default=None)
    ap.add_argument("--workers", type=int, default=None, help="Gemma requests at the same time (GEMMA_WORKERS)")
    ap.add_argument("--save-images", action="store_true", help="save the montages of every hexagon")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print("Loading surveys (the first load builds the cache)…")
    bv, mv = load_cube(resolve(args.baseline)), load_cube(resolve(args.monitor))
    print(f"  baseline {bv['cube'].shape}, monitor {mv['cube'].shape}")

    def attr_progress(f):
        print(f"\r  attributes {f:5.0%}", end="", flush=True)

    print("Measuring attributes on every trace…")
    ctx = hexscreen.prepare(bv, mv, width=args.width, progress=attr_progress)
    hexes = ctx["hexes"]
    print(f"\n  {len(hexes)} hexagons, {args.width} lines across ({time.perf_counter() - t0:.0f}s so far)")

    use_gemma = not args.no_gemma
    gcfg = gemma_client.settings(args.gemma_url, args.gemma_model)
    workers = args.workers or gcfg["workers"]
    todo = hexes
    if use_gemma and args.limit:
        todo = sorted(hexes, key=lambda h: (-h["seal_score"], -h["fault_score"]))[:args.limit]
        for h in hexes:
            if h not in todo:
                hexscreen.screen_hex(ctx, bv, mv, h, use_gemma=False)
    if use_gemma:
        print(f"Asking Gemma ({gcfg['model']} at {gcfg['base_url']}) about {len(todo)} hexagons, "
              f"{workers} at a time…")
    labels = (args.baseline_label, args.monitor_label)
    t1 = time.perf_counter()

    def progress(done, total, h):
        g = h.get("gemma") or {}
        what = (f"Gemma: fault {g['fault']}, sealing {g['sealing']} ({g['confidence']:.0%})" if g and not g.get("error")
                else f"Gemma error: {g['error'][:80]}" if g.get("error") else "attributes only")
        print(f"  {done:4d}/{total} hex {h['id']:4d}: {hexscreen.VERDICT_TEXT[h['verdict']][0]:28s} {what}")
        keep = args.save_images or h["verdict"] in VERDICT_COLOURS
        if keep:
            amp, diff = h.pop("amp_png", None), h.pop("diff_png", None)
            if amp is None:
                amp, diff = hexscreen.render_montages(ctx, bv, mv, h)
            (out / "images" / f"hex_{h['id']:04d}_amplitude.png").write_bytes(amp)
            (out / "images" / f"hex_{h['id']:04d}_difference.png").write_bytes(diff)
        h.pop("amp_png", None)
        h.pop("diff_png", None)

    sub = dict(ctx, hexes=todo)
    hexscreen.screen_all(sub, bv, mv, use_gemma=use_gemma, labels=labels, workers=workers,
                         base_url=args.gemma_url, model=args.gemma_model, progress=progress, keep_images=use_gemma)
    if use_gemma and todo:
        print(f"  Gemma took {time.perf_counter() - t1:.0f}s ({len(todo) / max(time.perf_counter() - t1, 1e-9) * 60:.1f} "
              "hexagons/min)")

    (out / "hexes.csv").write_text(hexscreen.to_csv(hexes), encoding="utf-8")
    clean = [{k: v for k, v in h.items() if k not in ("amp_png", "diff_png")} for h in hexes]
    (out / "hexes.json").write_text(json.dumps({"stats": ctx["stats"], "hexes": clean}, indent=1, default=float),
                                    encoding="utf-8")
    save_map(ctx, out / "hex_map.png")
    counts = hexscreen.summary_counts(hexes)
    failed = sum(1 for h in hexes if (h.get("gemma") or {}).get("error"))
    print("\nResult: " + ", ".join(f"{counts[v]} {hexscreen.VERDICT_TEXT[v][0].lower()}" for v in hexscreen.VERDICTS))
    if failed:
        print(f"Gemma failed for {failed} hexagons (they use the attributes only); see gemma_error in hexes.csv.")
    print(f"Saved {out}/hexes.csv, hexes.json, hex_map.png and images/ ({time.perf_counter() - t0:.0f}s in total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
