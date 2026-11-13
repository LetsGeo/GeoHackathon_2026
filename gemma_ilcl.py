"""
Send the hexagon inline / crossline pictures (from polygon_ilcl_pics.py) to Gemma and get,
for every hexagon, its ID, the % likelihood that it contains a reservoir and, using the 4D
difference pictures, an opportunity score (0-100 %).

    Jupyter:   %run gemma_ilcl.py
               or: import gemma_ilcl; gemma_ilcl.main(pic_dir="/path/to/ilcl_pics")
    Terminal:  python gemma_ilcl.py --dir /path/to/ilcl_pics --workers 8

Reads every hex<ID>_IL_<inline>.png / hex<ID>_CL_<crossline>.png in the folder. When the
section's difference picture (..._diff.png) is there too, both go to Gemma as ONE image side
by side (baseline left, monitor - baseline right; one image per request, WORKERS at a time),
and Gemma gives three 0-100 scores:
    reservoir    how likely the section crosses a reservoir (from the baseline)
    change       how strong and organised the 4D change is (from the difference)
    opportunity  Gemma's own overall judgement of the location for the CO2 storage project,
                 weighing both, with its reason
Without a difference picture, only the reservoir score is asked for. Writes to <folder>/gemma/:
    scores.csv     THE OUTPUT: one row per hexagon -> id, score_pct (reservoir), change_pct,
                   opportunity_pct (averages over its sections), highest opportunity first
    pictures.csv   the scores and Gemma's reason for every section
    results.json   everything above

Needs: pip install openai pillow
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings (change here, or pass to main(...), or use the --options)
# ---------------------------------------------------------------------------
PIC_DIR = '/data/epic20/APGCE/apgce_team_07/friday/ilcl_pics'  # where polygon_ilcl_pics.py saved them
BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://172.28.34.1:8000/v1")
MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-26B-A4B-it")
API_KEY = "token-not-needed"
WORKERS = int(os.environ.get("GEMMA_WORKERS", "8"))  # requests at the same time (see gemma_speed_test.py)
TIMEOUT_S = 180
MAX_TOKENS = 600
CROP_WHITE = True   # trim the white margins round each picture, so Gemma sees more of the seismic
USE_DIFF = True     # send the _diff picture with its baseline section and ask for the opportunity score

NAME_RE = re.compile(r"^hex(?P<hex>.+?)_(?P<kind>IL|CL)_(?P<line>-?\d+)\.(png|jpg|jpeg)$", re.I)

SYSTEM_PROMPT = """You are an experienced seismic interpreter screening a CO2 storage site
(Sleipner-type setting: sandstone reservoir under a shale seal) for reservoirs.

You get ONE seismic section from a 3D survey (baseline, amplitude): an inline (horizontal
axis = crossline numbers) or a crossline (horizontal axis = inline numbers). The vertical
axis is two-way time in ms, increasing downwards. Colours: red / yellow = one polarity,
white = about zero, grey / black = the other polarity.

Score from 0 to 100 how likely it is that this section crosses a RESERVOIR: a porous sand
body that can hold fluids or CO2, sealed above. Signs that raise the score:
- strong, laterally continuous reflections from a sand / shale package, or bright amplitudes
- a continuous strong reflector above it that could act as a seal
- a trap: structural high, dome or anticline, or a fault-bounded closure
Signs that lower it: weak, chaotic or noisy reflections, no clear sealing layer, no trap.
Use the whole range: 0-20 unlikely, 40-60 possible, 80-100 very likely.
Never invent line numbers or times that are not on the axes.

Reply with ONE JSON object and nothing else:
{"score": whole number from 0 to 100,
 "reason": "one short sentence: what in the section supports the score"}"""


OPPORTUNITY_PROMPT = """You are an experienced 4D seismic interpreter screening a CO2 storage
site (Sleipner-type setting: sandstone reservoir under a shale seal).

You get ONE image with two panels of the same seismic section (an inline or a crossline;
vertical axis = two-way time in ms, increasing downwards):
- LEFT: baseline amplitude. Red / yellow = one polarity, white = about zero, grey / black =
  the other polarity.
- RIGHT: 4D difference, monitor minus baseline. Blue = weaker in the monitor, red =
  stronger, white = no change.

Give three scores from 0 to 100:
- "reservoir": how likely the section crosses a RESERVOIR (porous sand under a seal, ideally
  in a trap: structural high, dome, fault-bounded closure), judged from the LEFT panel.
- "change": how strong and organised the 4D change is in the RIGHT panel: coherent events
  that follow the layering score high; weak, patchy or random noise scores low.
- "opportunity": your own overall judgement of how attractive this location is for the CO2
  storage project, weighing BOTH panels. Decide what matters most here and say it, e.g.
  CO2 clearly arriving in good reservoir and staying below the seal (proven storage),
  good reservoir that the CO2 has not reached yet (unused capacity), or risks such as change
  above the seal (possible leakage) or poor reservoir.
Use the whole range: 0-20 low, 40-60 medium, 80-100 high.
Never invent line numbers or times that are not on the axes.

Reply with ONE JSON object and nothing else:
{"reservoir": whole number 0-100, "change": whole number 0-100, "opportunity": whole number 0-100,
 "reason": "one or two short sentences: what drove the opportunity score"}"""


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------
def find_pictures(pic_dir):
    """[(hex_id, kind, line, path)] for every hexagon picture in the folder, sorted."""
    out = []
    for f in sorted(Path(pic_dir).iterdir()):
        m = NAME_RE.match(f.name)
        if m and f.is_file():
            out.append((m["hex"], m["kind"].upper(), int(m["line"]), f))
    key = lambda t: (int(t[0]) if t[0].isdigit() else t[0], t[1], t[2])
    try:
        return sorted(out, key=key)
    except TypeError:  # mixed numeric / text IDs
        return sorted(out, key=lambda t: (str(t[0]), t[1], t[2]))


def diff_path(path):
    """The section's 4D difference picture (hex07_IL_676.png -> hex07_IL_676_diff.png), or None."""
    p = Path(path)
    d = p.with_name(f"{p.stem}_diff{p.suffix}")
    return d if d.is_file() else None


def _cropped(path, crop=CROP_WHITE):
    from PIL import Image, ImageChops
    img = Image.open(path).convert("RGB")
    if crop:
        bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
        if bbox:
            pad = 6
            img = img.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                            min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad)))
    return img


def _png_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def picture_url(path, crop=CROP_WHITE):
    try:
        return _png_url(_cropped(path, crop))
    except Exception:  # send it as it is
        return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def pair_url(base_path, diff_path_, crop=CROP_WHITE):
    """Baseline (left) and 4D difference (right) side by side in ONE image, with a label over
    each, so the request has a single image whatever the server's per-request image limit."""
    from PIL import Image, ImageDraw, ImageFont
    left, right = _cropped(base_path, crop), _cropped(diff_path_, crop)
    h = max(left.height, right.height)
    if left.height != h:
        left = left.resize((round(left.width * h / left.height), h))
    if right.height != h:
        right = right.resize((round(right.width * h / right.height), h))
    gap, band = 24, 28
    out = Image.new("RGB", (left.width + gap + right.width, h + band), "white")
    out.paste(left, (0, band))
    out.paste(right, (left.width + gap, band))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    draw.text((4, 5), "LEFT: baseline amplitude", fill="black", font=font)
    draw.text((left.width + gap + 4, 5), "RIGHT: 4D difference (monitor - baseline)", fill="black", font=font)
    draw.line([(left.width + gap // 2, band), (left.width + gap // 2, h + band)], fill="#999999", width=2)
    return _png_url(out)


# ---------------------------------------------------------------------------
# Gemma
# ---------------------------------------------------------------------------
def make_client(base_url=BASE_URL, timeout=TIMEOUT_S):
    import openai
    try:
        import httpx
        http = httpx.Client(limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
                            timeout=httpx.Timeout(timeout, connect=10.0))
    except ImportError:
        http = None
    return openai.OpenAI(base_url=base_url, api_key=API_KEY, timeout=timeout, max_retries=0, http_client=http)


def _pct(data, key):
    """A 0-100 score from the reply (a 0-1 fraction is read as a percentage)."""
    raw = data.get(key)
    try:
        v = float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number from 0 to 100 (got {raw!r})")
    if 0 < v <= 1 and "." in str(raw):  # answered as a fraction (0.7) instead of a percentage
        v *= 100
    if not 0 <= v <= 100:
        raise ValueError(f"'{key}' must be from 0 to 100 (got {raw!r})")
    return round(v)


def parse_reply(text, keys=("score",)):
    """The scores (0-100) and reason from Gemma's reply; raises ValueError if unusable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON in the reply")
    data = json.loads(text[a:b + 1])
    out = {k: _pct(data, k) for k in keys}
    out["reason"] = str(data.get("reason", "")).strip()
    return out


def ask_gemma(client, model, hex_id, kind, line, path, diff=None):
    """One section -> Gemma's answer (dict). With its difference picture: reservoir, change and
    opportunity scores; without: the reservoir score only. One retry if the reply isn't usable."""
    what = "Inline" if kind == "IL" else "Crossline"
    if diff is not None:
        keys = ("reservoir", "change", "opportunity")
        text = (f"Hexagon {hex_id}, {what} {line}: baseline (left) and monitor - baseline (right). "
                "Score reservoir, change and opportunity. Reply with the JSON only.")
        system, url = OPPORTUNITY_PROMPT, pair_url(path, diff)
    else:
        keys = ("score",)
        text = (f"Hexagon {hex_id}, {what} {line} (baseline amplitude). "
                "How likely is it that this section crosses a reservoir? Reply with the JSON only.")
        system, url = SYSTEM_PROMPT, picture_url(path)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": text},
                                         {"type": "image_url", "image_url": {"url": url}}]}]

    def call(m):
        r = client.chat.completions.create(model=model, messages=m, temperature=0.1, max_tokens=MAX_TOKENS)
        return (r.choices[0].message.content or "") if r.choices else ""

    reply = call(msgs)
    try:
        ans = parse_reply(reply, keys)
    except (ValueError, json.JSONDecodeError) as exc:
        retry = msgs + [{"role": "assistant", "content": reply[:2000]},
                        {"role": "user", "content": f"That reply could not be used ({exc}). "
                                                    "Reply again with only the JSON object."}]
        ans = parse_reply(call(retry), keys)
    if "reservoir" in ans:  # same column as the baseline-only answers
        ans["score"] = ans.pop("reservoir")
    return ans


# ---------------------------------------------------------------------------
# Per-hexagon summary
# ---------------------------------------------------------------------------
def hexagon_scores(rows):
    """One row per hexagon: its ID, score_pct (% likelihood of a reservoir), change_pct and
    opportunity_pct = the averages over its sections; highest opportunity first (then reservoir)."""
    def mean(v):
        return round(sum(v) / len(v)) if v else None

    by_hex = {}
    for r in rows:
        by_hex.setdefault(r["hex"], []).append(r)
    out = []
    for hex_id, rs in by_hex.items():
        ok = [r for r in rs if not r["error"]]
        scores = [r["score"] for r in ok if r["score"] is not None]
        change = [r["change"] for r in ok if r.get("change") is not None]
        opp = [r["opportunity"] for r in ok if r.get("opportunity") is not None]
        out.append({
            "id": hex_id,
            "score_pct": mean(scores),          # None: Gemma didn't answer for this hexagon
            "change_pct": mean(change),         # None: no difference pictures
            "opportunity_pct": mean(opp),
            "min_pct": min(scores) if scores else None,
            "max_pct": max(scores) if scores else None,
            "opportunity_min": min(opp) if opp else None,
            "opportunity_max": max(opp) if opp else None,
            "pictures": len(rs), "answered": len(ok),
        })
    out.sort(key=lambda h: (h["opportunity_pct"] is None and h["score_pct"] is None,
                            -(h["opportunity_pct"] if h["opportunity_pct"] is not None else -1),
                            -(h["score_pct"] or 0)))
    return out


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(pic_dir=PIC_DIR, base_url=BASE_URL, model=MODEL, workers=WORKERS, hexes=None, limit=None,
         use_diff=USE_DIFF):
    """hexes: only these hexagon IDs (e.g. ["00", "07"]); limit: only the first N pictures (a quick test);
    use_diff: send each section's _diff picture with it and get the opportunity score."""
    pics = find_pictures(pic_dir)
    if hexes:
        wanted = {str(h).lstrip("0") or "0" for h in hexes}
        pics = [p for p in pics if (p[0].lstrip("0") or "0") in wanted]
    if limit:
        pics = pics[:limit]
    if not pics:
        print(f"No hex<ID>_IL_<n>.png / hex<ID>_CL_<n>.png pictures in {pic_dir}. "
              "Run polygon_ilcl_pics.py first, or check PIC_DIR.")
        return None
    out_dir = Path(pic_dir) / "gemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_hex = len({p[0] for p in pics})
    n_diff = sum(1 for p in pics if diff_path(p[3])) if use_diff else 0
    print(f"{len(pics)} sections of {n_hex} hexagons in {pic_dir}; "
          + (f"{n_diff} with a difference picture (-> reservoir, change and opportunity scores)"
             if n_diff else "no difference pictures used (-> reservoir score only)"))
    print(f"Gemma: {model} at {base_url}, {workers} at a time")

    client = make_client(base_url)
    state = {"ok": 0, "failed": 0, "first_error": None}
    lock = threading.Lock()

    def one(item):
        hex_id, kind, line, path = item
        diff = diff_path(path) if use_diff else None
        row = {"hex": hex_id, "kind": kind, "line": line, "file": path.name,
               "diff_file": diff.name if diff else "", "score": None, "change": None, "opportunity": None,
               "reason": "", "seconds": 0.0, "error": ""}
        if state["ok"] == 0 and state["failed"] >= 3:  # Gemma clearly not answering: don't wait on every picture
            row["error"] = "skipped: Gemma did not answer the first requests"
            return row
        t0 = time.perf_counter()
        try:
            row.update(ask_gemma(client, model, hex_id, kind, line, path, diff))
            with lock:
                state["ok"] += 1
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            with lock:
                state["failed"] += 1
                state["first_error"] = state["first_error"] or row["error"]
        row["seconds"] = round(time.perf_counter() - t0, 2)
        return row

    rows = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(one, p) for p in pics]
        for k, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            rows.append(r)
            if r["error"]:
                what = f"ERROR {r['error'][:70]}"
            elif r["opportunity"] is not None:
                what = (f"reservoir {r['score']:3d}%  change {r['change']:3d}%  opportunity {r['opportunity']:3d}%  "
                        f"{r['reason'][:50]}")
            else:
                what = f"reservoir {r['score']:3d}%  {r['reason'][:70]}"
            print(f"  {k:4d}/{len(pics)}  hex{r['hex']} {r['kind']} {r['line']:<6} {what}")
    took = time.perf_counter() - t0

    order = {(p[0], p[1], p[2]): i for i, p in enumerate(pics)}
    rows.sort(key=lambda r: order[(r["hex"], r["kind"], r["line"])])
    hex_rows = hexagon_scores(rows)
    write_csv(out_dir / "scores.csv", [{"id": h["id"], "score_pct": h["score_pct"], "change_pct": h["change_pct"],
                                        "opportunity_pct": h["opportunity_pct"]} for h in hex_rows])
    write_csv(out_dir / "pictures.csv", rows)
    (out_dir / "results.json").write_text(json.dumps({"model": model, "base_url": base_url, "scores": hex_rows,
                                                      "pictures": rows}, indent=1, ensure_ascii=False),
                                          encoding="utf-8")

    rate = f" ({state['ok'] / max(took, 1e-9) * 60:.0f} pictures/min)" if state["ok"] else ""
    print(f"\nDone in {took:.0f}s{rate}. "
          f"Answered {state['ok']} of {len(pics)}"
          + (f"; failed {len(pics) - state['ok']} (first error: {state['first_error']})" if state["first_error"] else ""))
    def pct(v):
        return "   -" if v is None else f"{v:3d}%"

    print("\n   ID  reservoir  change  opportunity   (averages over the hexagon's sections)")
    for h in hex_rows:
        if h["score_pct"] is None:
            print(f"{h['id']:>5}         -       -            -   (Gemma did not answer for this hexagon)")
        else:
            rng = (f"opportunity {h['opportunity_min']}-{h['opportunity_max']}%"
                   if h["opportunity_pct"] is not None else f"reservoir {h['min_pct']}-{h['max_pct']}%")
            print(f"{h['id']:>5}     {pct(h['score_pct'])}    {pct(h['change_pct'])}         {pct(h['opportunity_pct'])}"
                  f"   ({rng}, {h['answered']} of {h['pictures']} sections)")
    print(f"Saved {out_dir}/scores.csv (ID, reservoir %, change %, opportunity %), pictures.csv and results.json")
    try:
        import pandas as pd
        return {"scores": pd.DataFrame(hex_rows), "pictures": pd.DataFrame(rows)}
    except ImportError:
        return {"scores": hex_rows, "pictures": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Send the hexagon IL/CL pictures to Gemma, one per request.")
    ap.add_argument("--dir", default=PIC_DIR, help="folder with the hex<ID>_IL_/CL_ pictures")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--hex", nargs="+", default=None, help="only these hexagon IDs, e.g. --hex 00 07")
    ap.add_argument("--limit", type=int, default=None, help="only the first N pictures (quick test)")
    ap.add_argument("--no-diff", action="store_true", help="ignore the _diff pictures: reservoir score only")
    args, _ = ap.parse_known_args()  # parse_known_args: also works under %run in Jupyter
    results = main(pic_dir=args.dir, base_url=args.base_url, model=args.model, workers=args.workers,
                   hexes=args.hex, limit=args.limit, use_diff=not args.no_diff and USE_DIFF)
