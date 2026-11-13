"""
Sealing faults: which hexagons around a well most likely hold a sealing fault.

Input: one or more hexagon text files, one row per hexagon, with
    hexagon number, centre inline, centre crossline, sealing fault probability %, trace %
(with a header row using names like the ones in HEXAGON_COLUMNS, or without a
header in exactly that column order). Several files are merged by hexagon number;
if a hexagon appears in more than one file, each file's values are kept.

Each hexagon is then compared with:
  - the well: how far the hexagon is from it, and whether the horizon in the
    hexagon is shallower (up-dip) or deeper (down-dip) than at the well
  - each horizon: its two-way time inside the hexagon, how rough it is there
    once the general dip is taken out (a fault breaks the surface), and the
    biggest jump in time to a neighbouring hexagon (a fault offsets it)
and everything is sent to Gemma, which says for each hexagon whether a sealing
fault is likely, possible or unlikely, and why. If Gemma can't be reached (or its
reply doesn't fit), a simple rule engine gives the answer instead, and the result
says which one was used.

Use it from the app:
    hexes = read_hexagon_files(["hexes.txt"])
    result = find_sealing_faults(hexes, well, horizons)
where well and horizons are the app's own records (ws["wells"][i], and the
horizon dicts from horizons.py: {"name", "il", "xl", "t"}).

Or from the command line:
    python sealing_faults.py hexes.txt --well-il 1854 --well-xl 1150 \
        --horizon horizons/TOP-UTSIRA-DEMO.txt --horizon horizons/BASE-UTSIRA-DEMO.txt
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

import polygon
from ai import GEMMA_MODEL, GEMMA_URL, gemma_chat, parse_json_reply

# Accepted header names for each column (compared after lower-casing and turning
# anything that isn't a letter or digit into "_"). Headerless files must use this order.
HEXAGON_COLUMNS = {
    "hexagon": ("hexagon", "hexagon_no", "hexagon_number", "hex", "hex_no", "hex_id", "id", "no", "number"),
    "il": ("inline", "center_inline", "centre_inline", "inline_center", "inline_centre", "il", "iline",
           "center_il", "centre_il"),
    "xl": ("crossline", "center_crossline", "centre_crossline", "crossline_center", "crossline_centre", "xl",
           "xline", "center_xl", "centre_xl"),
    "probability": ("sealing_fault_probability", "sealing_fault_probability_pct", "sealing_probability",
                    "seal_probability", "probability", "prob", "sealing_fault", "sealing_fault_pct", "sfp"),
    "trace_pct": ("trace", "trace_pct", "trace_percent", "trace_percentage", "traces", "traces_pct", "trace_pc"),
}
COLUMN_ORDER = ("hexagon", "il", "xl", "probability", "trace_pct")

VERDICTS = ("likely", "possible", "unlikely")
NEIGHBOUR_DIST = 1.2 * 2 * polygon.s32 * polygon.SCALE  # centres closer than this share an edge
GEMMA_TIMEOUT_S = 120  # a reply for ~20 hexagons takes longer than the app's usual 30 s


# --------------------------------------------------------------------- reading
def _norm(text):
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _number(token):
    return float(token.strip().rstrip("%"))


def read_hexagon_text(text, source="hexagons"):
    """Hexagon rows from one text file: a list of
    {"hexagon", "il", "xl", "probability", "trace_pct", "source"} (percentages as 0-100)."""
    header, rows = None, []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!", "//")):
            continue
        tokens = [t for t in re.split(r"[,\t;| ]+", line) if t]
        try:
            rows.append([_number(t) for t in tokens])
        except ValueError:
            if not rows and header is None:
                header = [_norm(t) for t in re.split(r"[,\t;|]+", line) if t.strip()]
                if len(header) < 5:  # space-separated header: split on spaces instead
                    header = [_norm(t) for t in tokens]
    if not rows:
        raise ValueError(f"{source}: no rows of numbers found")

    cols = {}
    if header:
        for key, names in HEXAGON_COLUMNS.items():
            for j, h in enumerate(header):
                if j not in cols.values() and (h in names or any(h.startswith(n) for n in names if len(n) > 3)):
                    cols[key] = j
                    break
    if len(cols) < len(COLUMN_ORDER):  # no (usable) header: the columns are in the documented order
        if header:
            missing = [k for k in COLUMN_ORDER if k not in cols]
            raise ValueError(f"{source}: couldn't find these columns in the header: {', '.join(missing)} "
                             f"(header was: {', '.join(header)})")
        cols = {key: j for j, key in enumerate(COLUMN_ORDER)}

    out = []
    for r in rows:
        if len(r) <= max(cols.values()):
            continue
        rec = {key: r[j] for key, j in cols.items()}
        rec["hexagon"] = int(round(rec["hexagon"]))
        for key in ("probability", "trace_pct"):
            if 0 <= rec[key] <= 1 and all(0 <= x[cols[key]] <= 1 for x in rows if len(x) > cols[key]):
                rec[key] *= 100  # written as fractions (0-1): turn into %
        rec["source"] = source
        out.append(rec)
    if not out:
        raise ValueError(f"{source}: no complete rows")
    return out


def read_hexagon_files(paths):
    """All hexagon rows from several files (see read_hexagon_text)."""
    rows = []
    for p in paths:
        p = Path(p)
        rows += read_hexagon_text(p.read_text(encoding="utf-8-sig", errors="replace"), p.name)
    return rows


HORIZON_COLUMNS = {  # same names the app's Horizons card accepts (horizons.py)
    "il": ("inline", "il", "iline", "inline_no", "line"),
    "xl": ("crossline", "xl", "xline", "crossline_no", "trace", "cdp"),
    "t": ("twt", "twt_ms", "time", "time_ms", "twt_s", "z", "value"),
}


def read_horizon_file(path):
    """A horizon from a text file, as {"name", "il", "xl", "t"}. With a header
    (INLINE, CROSSLINE, ..., TWT), the columns are found by name; without one,
    the first two columns are inline and crossline and the last is time.
    Kept separate from horizons.py so this file runs without the app's packages."""
    header, rows = None, []
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "!", "@", "//")):
            continue
        tokens = [t for t in re.split(r"[,\t; ]+", line) if t]
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            if not rows:
                header = [_norm(t) for t in tokens]
    if not rows:
        raise ValueError(f"{path}: no rows of numbers found")
    widths = [len(r) for r in rows]
    width = max(set(widths), key=widths.count)
    arr = np.array([r for r in rows if len(r) == width], dtype=np.float64)
    cols = {}
    if header and len(header) == width:
        for key, names in HORIZON_COLUMNS.items():
            j = next((j for j, h in enumerate(header) if h in names and j not in cols.values()), None)
            if j is not None:
                cols[key] = j
    if len(cols) < 3:
        cols = {"il": 0, "xl": 1, "t": width - 1}
    il, xl, t = arr[:, cols["il"]], arr[:, cols["xl"]], np.abs(arr[:, cols["t"]])
    if np.nanmax(t) < 20:  # seconds -> ms
        t = t * 1000.0
    ok = np.isfinite(il) & np.isfinite(xl) & np.isfinite(t) & (t > 0) & (t < 20000)  # drop null values
    return {"name": Path(path).stem, "il": il[ok], "xl": xl[ok], "t": t[ok]}


# ------------------------------------------------------------------- comparing
def _well_position(well, t=None):
    """(inline, crossline) of the well at time t (top of the well if t is None or it has no times)."""
    il, xl, twt = np.atleast_1d(well["il"]), np.atleast_1d(well["xl"]), well.get("twt")
    if t is None or twt is None or len(twt) < 2:
        return float(il[0]), float(xl[0])
    return float(np.interp(t, twt, il)), float(np.interp(t, twt, xl))


def _points_in_hexagon(h, c_il, c_xl):
    """Mask of the horizon's points that lie inside the hexagon centred at (c_il, c_xl)."""
    r = polygon.SCALE
    box = (np.abs(h["il"] - c_il) <= r) & (np.abs(h["xl"] - c_xl) <= r)
    mask = np.zeros(len(h["il"]), bool)
    if box.any():
        # a hexagon is convex: a point is inside when it's on the same side of all six edges
        v = polygon.hexagon(c_il, c_xl)
        e = np.roll(v, -1, axis=0) - v
        p_il, p_xl = h["il"][box], h["xl"][box]
        cross = e[:, 0, None] * (p_xl[None, :] - v[:, 1, None]) - e[:, 1, None] * (p_il[None, :] - v[:, 0, None])
        mask[box] = np.all(cross >= -1e-9, axis=0) | np.all(cross <= 1e-9, axis=0)
    return mask


def _horizon_at(h, il, xl, radius=2.0):
    """Mean horizon time within `radius` lines of (il, xl), or None if it has no points there."""
    near = (np.abs(h["il"] - il) <= radius) & (np.abs(h["xl"] - xl) <= radius)
    return float(np.mean(h["t"][near])) if near.any() else None


def _surface_stats(il, xl, t):
    """How the horizon behaves inside one hexagon: mean / min / max time, the dip of
    a plane fitted through it (ms per line), and how far the points stray from that
    plane (a fault breaks the surface: big residuals; a smooth dip: small ones)."""
    out = {"points": int(len(t)), "twt_mean_ms": float(np.mean(t)), "twt_min_ms": float(np.min(t)),
           "twt_max_ms": float(np.max(t))}
    if len(t) >= 6:
        A = np.column_stack([np.ones(len(t)), il - il.mean(), xl - xl.mean()])
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        resid = t - A @ coef
        out["dip_ms_per_line"] = float(np.hypot(coef[1], coef[2]))
        out["roughness_ms"] = float(np.std(resid))          # typical misfit to a plane
        out["max_step_ms"] = float(np.max(np.abs(resid)))   # the worst one: an offset inside the hexagon
    return out


def compare_hexagons(hexes, well=None, horizons=(), line_spacing_m=None):
    """The hexagons with everything they're compared on: distance and direction
    from the well, ring around it, and per horizon the time inside the hexagon,
    its roughness and the biggest jump to a neighbouring hexagon.
    Several rows for the same hexagon (from several files) are kept as one
    hexagon, with each file's probability / trace % listed under "inputs"."""
    by_id = {}
    for r in hexes:
        h = by_id.setdefault(r["hexagon"], {"hexagon": r["hexagon"], "il": r["il"], "xl": r["xl"], "inputs": []})
        h["inputs"].append({"source": r["source"], "sealing_fault_probability_pct": round(r["probability"], 2),
                            "trace_pct": round(r["trace_pct"], 2)})
    cells = sorted(by_id.values(), key=lambda c: c["hexagon"])
    for c in cells:
        c["sealing_fault_probability_pct"] = round(float(np.mean([i["sealing_fault_probability_pct"]
                                                                   for i in c["inputs"]])), 2)
        c["trace_pct"] = round(float(np.mean([i["trace_pct"] for i in c["inputs"]])), 2)

    # where the well is: at the first horizon's time if there is one (a deviated well moves with depth)
    well_info = None
    if well is not None:
        ref_t = None
        if horizons:
            w_il0, w_xl0 = _well_position(well)
            ref_t = _horizon_at(horizons[0], w_il0, w_xl0, radius=polygon.SCALE / 2)
        w_il, w_xl = _well_position(well, ref_t)
        well_info = {"name": well.get("name", "well"), "inline": round(w_il, 2), "crossline": round(w_xl, 2),
                     "position_taken_at_twt_ms": None if ref_t is None else round(ref_t, 1)}
        pick = well.get("reservoir")
        if pick:
            well_info["reservoir_twt_ms"] = [round(pick["top_twt_ms"], 1), round(pick["base_twt_ms"], 1)]
        well_info["horizon_twt_at_well_ms"] = {
            h["name"]: (None if (v := _horizon_at(h, w_il, w_xl, radius=polygon.SCALE / 2)) is None else round(v, 1))
            for h in horizons}
        hex_step = 2 * polygon.s32 * polygon.SCALE  # distance between neighbouring centres
        for c in cells:
            d = float(np.hypot(c["il"] - w_il, c["xl"] - w_xl))
            c["distance_from_well_lines"] = round(d, 1)
            if line_spacing_m:
                c["distance_from_well_m"] = round(d * line_spacing_m, 0)
            c["ring_around_well"] = int(round(d / hex_step))
            ang = np.degrees(np.arctan2(c["xl"] - w_xl, c["il"] - w_il))  # 0 = towards higher inlines
            c["direction_from_well_deg"] = None if d < 1 else round(float(ang % 360), 0)

    # horizons inside each hexagon
    for h in horizons:
        for c in cells:
            mask = _points_in_hexagon(h, c["il"], c["xl"])
            stats = _surface_stats(h["il"][mask], h["xl"][mask], h["t"][mask]) if mask.any() else {"points": 0}
            if well_info and stats.get("points") and well_info["horizon_twt_at_well_ms"].get(h["name"]) is not None:
                # negative: the horizon is shallower here than at the well (up-dip, where fluids would migrate)
                stats["twt_minus_well_ms"] = round(stats["twt_mean_ms"] - well_info["horizon_twt_at_well_ms"][h["name"]], 2)
            c.setdefault("horizons", {})[h["name"]] = stats
        for c in cells:  # biggest jump in mean time to a neighbouring hexagon
            mine = c["horizons"][h["name"]].get("twt_mean_ms")
            jumps = [abs(mine - o["horizons"][h["name"]]["twt_mean_ms"]) for o in cells
                     if o is not c and mine is not None and o["horizons"][h["name"]].get("twt_mean_ms") is not None
                     and np.hypot(o["il"] - c["il"], o["xl"] - c["xl"]) <= NEIGHBOUR_DIST]
            c["horizons"][h["name"]]["max_jump_to_neighbour_ms"] = round(max(jumps), 2) if jumps else None
    if len(horizons) >= 2:  # thickness between the first two horizons (e.g. reservoir top and base)
        a, b = horizons[0]["name"], horizons[1]["name"]
        for c in cells:
            ta, tb = c["horizons"][a].get("twt_mean_ms"), c["horizons"][b].get("twt_mean_ms")
            c[f"thickness_{a}_to_{b}_ms"] = None if ta is None or tb is None else round(tb - ta, 2)

    for c in cells:  # tidy numbers for the report / prompt
        c["il"], c["xl"] = round(float(c["il"]), 2), round(float(c["xl"]), 2)
        for stats in c.get("horizons", {}).values():
            for k, v in stats.items():
                if isinstance(v, float):
                    stats[k] = round(v, 2)
    return cells, well_info


# ---------------------------------------------------------------------- judging
def rule_engine(cells):
    """A plain score for each hexagon, used when Gemma isn't available: the fault
    probability counts most, then how much of the hexagon the fault covers (trace %),
    then how strongly the horizons are broken or offset there."""
    out = []
    for c in cells:
        p, tr = c["sealing_fault_probability_pct"] / 100, c["trace_pct"] / 100
        breaks = [max(s.get("max_jump_to_neighbour_ms") or 0, s.get("max_step_ms") or 0)
                  for s in c.get("horizons", {}).values()]
        offset = min(max(breaks) / 10.0, 1.0) if breaks else 0.0  # 10 ms or more counts as a clear offset
        score = 0.55 * p + 0.25 * tr + 0.20 * offset
        verdict = "likely" if score >= 0.6 else "possible" if score >= 0.4 else "unlikely"
        reason = (f"probability {c['sealing_fault_probability_pct']:.0f}%, trace {c['trace_pct']:.0f}%"
                  + (f", horizon offset up to {max(breaks):.1f} ms" if breaks else ""))
        out.append({"hexagon": c["hexagon"], "verdict": verdict, "confidence": round(score, 2), "reason": reason})
    return out


SYSTEM_PROMPT = """You are a structural geologist assessing fault seal around a well from 3D seismic.
The area around the well is split into hexagons (about HEX_ACROSS inlines across). For each hexagon you get:
- sealing_fault_probability_pct: the fault attribute's probability (0-100) that a sealing fault is in it
- trace_pct: the percentage of the hexagon's seismic traces where that fault is picked up (how much of it the fault covers)
- distance and direction from the well, and its ring around the well (0 = the well's own hexagon)
- for each horizon: mean two-way time (ms), twt_minus_well_ms (negative = shallower than at the well, i.e. up-dip),
  roughness_ms and max_step_ms (how broken the surface is once its general dip is taken out),
  and max_jump_to_neighbour_ms (offset of the horizon against the next hexagons: a fault throw)
A sealing fault is most believable where the probability is high, it covers much of the hexagon,
and the horizons are clearly offset or broken there. A high probability with a smooth, unbroken
horizon, or a low trace %, is weaker evidence. Faults up-dip of the well matter most for trapping.
Judge every hexagon. Reply with one JSON object only, no other text:
{"hexagons": [{"hexagon": <number>, "verdict": "likely" | "possible" | "unlikely",
  "confidence": <0 to 1>, "reason": "<one short sentence with the numbers that decided it>"}],
 "summary": "<two or three sentences: where the sealing fault(s) most likely are relative to the well, and how sure you are>"}"""
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("HEX_ACROSS", f"{2 * polygon.SCALE:g}")  # hexagon size from polygon.py


def ask_gemma(cells, well_info, horizon_names):
    """Gemma's verdict per hexagon. Raises if Gemma can't be reached or its reply doesn't fit."""
    user = json.dumps({"well": well_info, "horizons": list(horizon_names),
                       "hexagons": [{k: v for k, v in c.items() if k != "inputs"} | (
                           {"inputs_per_file": c["inputs"]} if len(c["inputs"]) > 1 else {}) for c in cells]},
                      ensure_ascii=False)
    reply = gemma_chat(SYSTEM_PROMPT, user, max_tokens=300 + 90 * len(cells), temperature=0.1,
                       timeout_s=GEMMA_TIMEOUT_S)
    out = parse_json_reply(reply)
    got = {int(h["hexagon"]): h for h in out.get("hexagons", []) if "hexagon" in h}
    missing = [c["hexagon"] for c in cells if c["hexagon"] not in got]
    if missing:
        raise ValueError(f"Gemma's reply left out hexagon(s) {missing}")
    verdicts = []
    for c in cells:
        h = got[c["hexagon"]]
        verdict = str(h.get("verdict", "")).strip().lower()
        if verdict not in VERDICTS:
            raise ValueError(f"Gemma gave hexagon {c['hexagon']} an unknown verdict '{verdict}'")
        try:
            conf = min(max(float(h.get("confidence", 0)), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = None
        verdicts.append({"hexagon": c["hexagon"], "verdict": verdict, "confidence": conf,
                         "reason": str(h.get("reason", "")).strip()})
    return verdicts, str(out.get("summary", "")).strip()


def find_sealing_faults(hexes, well=None, horizons=(), use_gemma=True, line_spacing_m=None):
    """Compare the hexagons with the well and horizons, then judge each one.
    Returns {"source": "Gemma" | "rule engine", "summary", "note", "well", "hexagons": [...]},
    each hexagon with its inputs, comparisons and "verdict" / "confidence" / "reason",
    sorted with the most likely sealing faults first."""
    cells, well_info = compare_hexagons(hexes, well, horizons, line_spacing_m)
    source, summary, note = "rule engine", "", ""
    verdicts = None
    if use_gemma:
        try:
            verdicts, summary = ask_gemma(cells, well_info, [h["name"] for h in horizons])
            source = f"Gemma ({GEMMA_MODEL})"
        except Exception as exc:  # unreachable, no openai package, or a reply that doesn't fit
            note = f"Gemma wasn't used: {exc}"
    if verdicts is None:
        verdicts = rule_engine(cells)
        likely = [v["hexagon"] for v in verdicts if v["verdict"] == "likely"]
        summary = (f"Hexagon(s) {', '.join(map(str, likely))} most likely hold a sealing fault (rule engine)."
                   if likely else "No hexagon scores high enough for a likely sealing fault (rule engine).")
    by_id = {v["hexagon"]: v for v in verdicts}
    for c in cells:
        c.update({k: by_id[c["hexagon"]][k] for k in ("verdict", "confidence", "reason")})
    rank = {v: i for i, v in enumerate(VERDICTS)}
    cells.sort(key=lambda c: (rank[c["verdict"]], -(c["confidence"] or 0), c["hexagon"]))
    return {"source": source, "summary": summary, "note": note, "well": well_info, "hexagons": cells}


# ------------------------------------------------------------------ command line
def write_csv(result, path):
    import csv
    names = sorted({n for c in result["hexagons"] for n in c.get("horizons", {})})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hexagon", "verdict", "confidence", "center_inline", "center_crossline",
                    "sealing_fault_probability_pct", "trace_pct", "distance_from_well_lines", "ring_around_well"]
                   + [f"{n}_{k}" for n in names for k in ("twt_mean_ms", "twt_minus_well_ms", "max_step_ms",
                                                          "max_jump_to_neighbour_ms")] + ["reason"])
        for c in result["hexagons"]:
            hz = c.get("horizons", {})
            w.writerow([c["hexagon"], c["verdict"], c["confidence"], c["il"], c["xl"],
                        c["sealing_fault_probability_pct"], c["trace_pct"], c.get("distance_from_well_lines"),
                        c.get("ring_around_well")]
                       + [hz.get(n, {}).get(k) for n in names for k in ("twt_mean_ms", "twt_minus_well_ms",
                                                                         "max_step_ms", "max_jump_to_neighbour_ms")]
                       + [c["reason"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find sealing faults in hexagons around a well (with Gemma).")
    ap.add_argument("hexagon_files", nargs="+", help="text files: hexagon, centre inline, centre crossline, "
                                                     "sealing fault probability %%, trace %%")
    ap.add_argument("--well-il", type=float, help="well inline")
    ap.add_argument("--well-xl", type=float, help="well crossline")
    ap.add_argument("--well-name", default="well")
    ap.add_argument("--reservoir", type=float, nargs=2, metavar=("TOP_MS", "BASE_MS"),
                    help="reservoir top and base two-way time at the well")
    ap.add_argument("--horizon", action="append", default=[], help="horizon text file (repeat for more)")
    ap.add_argument("--line-spacing", type=float, help="metres per inline / crossline, to give distances in m")
    ap.add_argument("--no-gemma", action="store_true", help="use the rule engine only")
    ap.add_argument("--out", default="sealing_faults", help="output name: writes <out>.json and <out>.csv")
    args = ap.parse_args(argv)

    hexes = read_hexagon_files(args.hexagon_files)
    well = None
    if args.well_il is not None and args.well_xl is not None:
        well = {"name": args.well_name, "il": np.array([args.well_il]), "xl": np.array([args.well_xl]), "twt": None}
        if args.reservoir:
            well["reservoir"] = {"top_twt_ms": args.reservoir[0], "base_twt_ms": args.reservoir[1]}
    horizons = [read_horizon_file(p) for p in args.horizon]
    print(f"{len(hexes)} hexagon rows, {len(horizons)} horizon(s), well: {'yes' if well else 'no'}. "
          f"Asking {'the rule engine' if args.no_gemma else f'Gemma at {GEMMA_URL}'}...")
    result = find_sealing_faults(hexes, well, horizons, use_gemma=not args.no_gemma,
                                 line_spacing_m=args.line_spacing)
    Path(f"{args.out}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(result, f"{args.out}.csv")
    print(f"\nJudged by: {result['source']}")
    if result["note"]:
        print(result["note"])
    print(f"\n{result['summary']}\n")
    for c in result["hexagons"]:
        conf = "" if c["confidence"] is None else f" ({c['confidence']:.2f})"
        print(f"  hexagon {c['hexagon']:>3}: {c['verdict']:<9}{conf}  {c['reason']}")
    print(f"\nWritten: {args.out}.json, {args.out}.csv")


if __name__ == "__main__":
    main()
