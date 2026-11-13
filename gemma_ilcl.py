"""
Send the hexagon inline / crossline pictures (from polygon_ilcl_pics.py) to Gemma and get,
for every hexagon, its ID and a score: the % likelihood that it contains a reservoir.

    Jupyter:   %run gemma_ilcl.py
               or: import gemma_ilcl; gemma_ilcl.main(pic_dir="/path/to/ilcl_pics")
    Terminal:  python gemma_ilcl.py --dir /path/to/ilcl_pics --workers 8

Reads every hex<ID>_IL_<inline>.png / hex<ID>_CL_<crossline>.png in the folder, sends each
picture to Gemma (one picture per request, WORKERS at a time; Gemma scores 0-100 % how likely
the section crosses a reservoir), and writes to <folder>/gemma/:
    scores.csv     THE OUTPUT: one row per hexagon -> id, score_pct (average of its pictures)
    pictures.csv   the score and Gemma's one-line reason for every picture
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


def picture_url(path, crop=CROP_WHITE):
    data = Path(path).read_bytes()
    if crop:
        try:
            from PIL import Image, ImageChops
            img = Image.open(io.BytesIO(data)).convert("RGB")
            bbox = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
            if bbox:
                pad = 6
                bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                        min(img.width, bbox[2] + pad), min(img.height, bbox[3] + pad))
                buf = io.BytesIO()
                img.crop(bbox).save(buf, format="PNG")
                data = buf.getvalue()
        except Exception:
            pass  # send it as it is
    return "data:image/png;base64," + base64.b64encode(data).decode()


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


def parse_reply(text):
    """{"score": 0-100, "reason": str} from Gemma's reply; raises ValueError if unusable."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON in the reply")
    data = json.loads(text[a:b + 1])
    raw = data.get("score")
    try:
        score = float(str(raw).strip().rstrip("%"))
    except (TypeError, ValueError):
        raise ValueError(f"'score' must be a number from 0 to 100 (got {raw!r})")
    if 0 < score <= 1 and "." in str(raw):  # answered as a fraction (0.7) instead of a percentage
        score *= 100
    if not 0 <= score <= 100:
        raise ValueError(f"'score' must be from 0 to 100 (got {raw!r})")
    return {"score": round(score), "reason": str(data.get("reason", "")).strip()}


def ask_gemma(client, model, hex_id, kind, line, path):
    """One picture -> Gemma's answer (dict). One retry if the reply isn't usable JSON."""
    what = "Inline" if kind == "IL" else "Crossline"
    text = (f"Hexagon {hex_id}, {what} {line} (baseline amplitude). "
            "How likely is it that this section crosses a reservoir? Reply with the JSON only.")
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": text},
                                         {"type": "image_url", "image_url": {"url": picture_url(path)}}]}]

    def call(m):
        r = client.chat.completions.create(model=model, messages=m, temperature=0.1, max_tokens=MAX_TOKENS)
        return (r.choices[0].message.content or "") if r.choices else ""

    reply = call(msgs)
    try:
        return parse_reply(reply)
    except (ValueError, json.JSONDecodeError) as exc:
        retry = msgs + [{"role": "assistant", "content": reply[:2000]},
                        {"role": "user", "content": f"That reply could not be used ({exc}). "
                                                    "Reply again with only the JSON object."}]
        return parse_reply(call(retry))


# ---------------------------------------------------------------------------
# Per-hexagon summary
# ---------------------------------------------------------------------------
def hexagon_scores(rows):
    """One row per hexagon: its ID and score (% likelihood of a reservoir) = the average of
    its pictures' scores, highest first."""
    by_hex = {}
    for r in rows:
        by_hex.setdefault(r["hex"], []).append(r)
    out = []
    for hex_id, rs in by_hex.items():
        scores = [r["score"] for r in rs if not r["error"]]
        out.append({
            "id": hex_id,
            "score_pct": round(sum(scores) / len(scores)) if scores else None,  # None: Gemma didn't answer
            "min_pct": min(scores) if scores else None,
            "max_pct": max(scores) if scores else None,
            "pictures": len(rs), "answered": len(scores),
        })
    out.sort(key=lambda h: (h["score_pct"] is None, -(h["score_pct"] or 0)))
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
def main(pic_dir=PIC_DIR, base_url=BASE_URL, model=MODEL, workers=WORKERS, hexes=None, limit=None):
    """hexes: only these hexagon IDs (e.g. ["00", "07"]); limit: only the first N pictures (a quick test)."""
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
    print(f"{len(pics)} pictures of {n_hex} hexagons in {pic_dir}")
    print(f"Gemma: {model} at {base_url}, {workers} at a time")

    client = make_client(base_url)
    state = {"ok": 0, "failed": 0, "first_error": None}
    lock = threading.Lock()

    def one(item):
        hex_id, kind, line, path = item
        row = {"hex": hex_id, "kind": kind, "line": line, "file": path.name, "score": None, "reason": "",
               "seconds": 0.0, "error": ""}
        if state["ok"] == 0 and state["failed"] >= 3:  # Gemma clearly not answering: don't wait on every picture
            row["error"] = "skipped: Gemma did not answer the first requests"
            return row
        t0 = time.perf_counter()
        try:
            row.update(ask_gemma(client, model, hex_id, kind, line, path))
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
            what = f"ERROR {r['error'][:70]}" if r["error"] else f"{r['score']:3d}%  {r['reason'][:70]}"
            print(f"  {k:4d}/{len(pics)}  hex{r['hex']} {r['kind']} {r['line']:<6} {what}")
    took = time.perf_counter() - t0

    order = {(p[0], p[1], p[2]): i for i, p in enumerate(pics)}
    rows.sort(key=lambda r: order[(r["hex"], r["kind"], r["line"])])
    hex_rows = hexagon_scores(rows)
    write_csv(out_dir / "scores.csv", [{"id": h["id"], "score_pct": h["score_pct"]} for h in hex_rows])
    write_csv(out_dir / "pictures.csv", rows)
    (out_dir / "results.json").write_text(json.dumps({"model": model, "base_url": base_url, "scores": hex_rows,
                                                      "pictures": rows}, indent=1, ensure_ascii=False),
                                          encoding="utf-8")

    rate = f" ({state['ok'] / max(took, 1e-9) * 60:.0f} pictures/min)" if state["ok"] else ""
    print(f"\nDone in {took:.0f}s{rate}. "
          f"Answered {state['ok']} of {len(pics)}"
          + (f"; failed {len(pics) - state['ok']} (first error: {state['first_error']})" if state["first_error"] else ""))
    print("\n   ID   score   (% likelihood of a reservoir; range over its pictures)")
    for h in hex_rows:
        if h["score_pct"] is None:
            print(f"{h['id']:>5}       -   (Gemma did not answer for this hexagon)")
        else:
            print(f"{h['id']:>5}   {h['score_pct']:4d}%   ({h['min_pct']}-{h['max_pct']}%, "
                  f"{h['answered']} of {h['pictures']} pictures)")
    print(f"Saved {out_dir}/scores.csv (ID + score), pictures.csv and results.json")
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
    args, _ = ap.parse_known_args()  # parse_known_args: also works under %run in Jupyter
    results = main(pic_dir=args.dir, base_url=args.base_url, model=args.model, workers=args.workers,
                   hexes=args.hex, limit=args.limit)
