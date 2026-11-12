"""
Gemma summaries for the Reservoir Time Machine analysis card.

The app measures the baseline-vs-monitor differences itself (analyse_volume /
analyse_slice in app.py). This module sends those measurements, the app's own
rule-based draft and two small pictures (the whole-volume change map and the
difference on the current slice) to a Gemma model served behind an
OpenAI-compatible API (e.g. vLLM), and asks it to write the English and Malay
summary. It returns the same structure as write_summary() in app.py:

    {"en": {"volume": [...], "slice": [...], "note": str},
     "ms": {"volume": [...], "slice": [...], "note": str}}

Settings (environment variables, or the --gemma-* options of app.py):
    GEMMA_BASE_URL   default http://172.28.34.1:8000/v1
    GEMMA_MODEL      default google/gemma-4-26B-A4B-it
    GEMMA_API_KEY    default "token-not-needed" (the client needs some value)
    GEMMA_TIMEOUT    seconds, default 120
    GEMMA_IMAGES     1 (default) sends the two pictures, 0 sends numbers only

Check the connection from the machine that runs the app:
    python gemma_client.py
"""
import base64
import io
import json
import math
import os
import re

DEFAULT_BASE_URL = "http://172.28.34.1:8000/v1"
DEFAULT_MODEL = "google/gemma-4-26B-A4B-it"

NOTE_EN = ("Written by Gemma ({model}) from the app's measurements; "
           "it should be checked by an interpreter.")
NOTE_MS = ("Ditulis oleh Gemma ({model}) berdasarkan pengukuran aplikasi; "
           "perlu disemak oleh pentafsir.")

SYSTEM_PROMPT = """You are an experienced 4D (time-lapse) seismic interpreter.
You write short, factual summaries of the difference between a baseline and a
monitor seismic survey for engineers and managers.

Rules:
- Use ONLY the numbers in the measurements. Never invent locations, times,
  percentages or volumes. Keep line numbers and times exactly as given.
- Explain what the numbers mean physically (e.g. amplitude brightening with a
  velocity push-down is typical of gas or CO2 replacing brine; dimming can mean
  water replacing hydrocarbons or a pressure change), but say "suggests" or
  "is consistent with" - never state a cause as certain.
- Mention survey repeatability (NRMS) and whether the change is likely real.
- The pictures, when present, show the same data as the numbers: a map of RMS
  amplitude change (bright = more change, dashed box = change zone) and the
  monitor-minus-baseline difference on the current slice (blue = weaker in the
  monitor, orange = stronger). Use them for shape and extent only.
- Write the same content twice: in English and in Bahasa Melayu (Malaysian
  Malay). Keep technical terms such as inline, crossline, NRMS, push-down.
- 2 to 4 sentences per paragraph, at most 4 paragraphs per section.
- If there is no slice data, return an empty list for "slice".

Reply with ONE JSON object and nothing else, in exactly this shape:
{"en": {"volume": ["paragraph", ...], "slice": ["paragraph", ...]},
 "ms": {"volume": ["perenggan", ...], "slice": ["perenggan", ...]}}"""


class GemmaError(RuntimeError):
    """Gemma could not be reached or gave an unusable answer."""


def settings(base_url=None, model=None):
    return {
        "base_url": base_url or os.environ.get("GEMMA_BASE_URL", DEFAULT_BASE_URL),
        "model": model or os.environ.get("GEMMA_MODEL", DEFAULT_MODEL),
        "api_key": os.environ.get("GEMMA_API_KEY", "token-not-needed"),
        "timeout": float(os.environ.get("GEMMA_TIMEOUT", "120")),
        "images": os.environ.get("GEMMA_IMAGES", "1").strip().lower() not in ("0", "false", "no", "off"),
    }


def make_client(cfg):
    try:
        import openai
    except ImportError as exc:
        raise GemmaError("the 'openai' package is not installed (pip install openai)") from exc
    return openai.OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                         timeout=cfg["timeout"], max_retries=0)


# ---------------------------------------------------------------------------
# Measurements -> plain JSON
# ---------------------------------------------------------------------------
def _clean(value):
    """Plain JSON value, or None for things the model doesn't need (arrays, NaN)."""
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return None
        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:
        pass
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        items = [_clean(v) for v in value]
        return items if all(i is not None for i in items) else None
    return None


def _scalars(d, skip=()):
    out = {}
    for k, v in (d or {}).items():
        if k in skip:
            continue
        c = _clean(v)
        if c is not None:
            out[k] = c
    return out


FIELD_GUIDE = {
    "il_range / xl_range / t_range": "shared area: first and last inline, crossline, and time (ms)",
    "changed_pct": "percentage of the area (or slice) with significant amplitude change",
    "significant": "true if the change is clearly above the noise level",
    "box": "change zone: [first inline, last inline, first crossline, last crossline]",
    "peak": "volume: [inline, crossline] of the strongest change; slice: [x, y] in the slice's axes",
    "t_change": "top of the change zone in ms: [10th, 50th, 90th percentile] across the zone",
    "top": "slice: median time (ms) where the change starts",
    "amp_ratio": "monitor / baseline RMS amplitude at the top of the change (1.1+ brightening, 0.9- dimming)",
    "time_shift_ms": "delay of reflections below the zone, monitor vs baseline (+ push-down, - pull-up)",
    "nrms_shallow": "median NRMS (%) in the shallow, assumed unchanged section; lower = more repeatable",
    "rating": "repeatability rating from the shallow NRMS",
    "nrms": "NRMS (%) of the current slice",
    "sample_step": "whole-volume analysis used every n-th line",
}


# ---------------------------------------------------------------------------
# Pictures (small PNGs, so the request stays light)
# ---------------------------------------------------------------------------
def _png_data_url(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def change_map_png(vol):
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    il, xl, change = vol["map_il"], vol["map_xl"], vol["map"]
    hi = float(np.nanpercentile(change, 99.5)) if np.isfinite(change).any() else 1.0
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(change, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=hi or 1.0,
                   extent=[xl[0], xl[-1], il[0], il[-1]])
    if vol.get("significant") and "box" in vol:
        b0, b1, c0, c1 = vol["box"]
        ax.add_patch(Rectangle((c0, b0), c1 - c0, b1 - b0, fill=False, ec="red", ls="--", lw=1.5))
    ax.set_xlabel("Crossline")
    ax.set_ylabel("Inline")
    ax.set_title("Whole volume: RMS amplitude change (map view)")
    fig.colorbar(im, ax=ax, label="RMS change")
    fig.tight_layout()
    return _png_data_url(fig)


def slice_diff_png(sl, axes):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure

    aligned, diff = sl["aligned"], sl["diff"]
    x_label, y_label, y_down = axes
    xs, ys = aligned["xs"], aligned["ys"]
    lim = float(np.nanpercentile(np.abs(diff), 99)) if np.isfinite(diff).any() else 1.0
    cmap = LinearSegmentedColormap.from_list("diff", ["#3b7dd8", "#ffffff", "#f39c32"])
    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(diff, origin="lower", aspect="auto", cmap=cmap, vmin=-(lim or 1), vmax=lim or 1,
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    if y_down:
        ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    where = f"time slice {sl['pos']:.0f} ms" if sl["view"] == "Time slice" else f"{sl['view'].lower()} {sl['pos']:.0f}"
    ax.set_title(f"Current slice ({where}): monitor - baseline")
    fig.colorbar(im, ax=ax, label="- weaker | stronger +")
    fig.tight_layout()
    return _png_data_url(fig)


# ---------------------------------------------------------------------------
# Request / answer
# ---------------------------------------------------------------------------
def build_messages(vol, sl, base_label, mon_label, draft, slice_axes=None, images=True):
    facts = {
        "baseline_survey": base_label,
        "monitor_survey": mon_label,
        "whole_volume": _scalars(vol, skip=("map", "map_il", "map_xl")) if vol else None,
        "current_slice": _scalars(sl, skip=("aligned", "diff")) if sl else None,
        "field_meanings": FIELD_GUIDE,
    }
    draft_en = {"volume": draft["en"]["volume"], "slice": draft["en"]["slice"]} if draft else None
    text = ("Measurements (JSON):\n" + json.dumps(facts, ensure_ascii=False, indent=1)
            + "\n\nRule-based draft from the app (correct but mechanical; rewrite it into a clearer "
              "interpretation, keeping every number consistent with the measurements):\n"
            + json.dumps(draft_en, ensure_ascii=False, indent=1)
            + "\n\nWrite the JSON summary now.")
    content = [{"type": "text", "text": text}]
    if images:
        try:
            if vol is not None and "map" in vol:
                content.append({"type": "image_url", "image_url": {"url": change_map_png(vol)}})
            if sl is not None and slice_axes is not None and "diff" in sl:
                content.append({"type": "image_url", "image_url": {"url": slice_diff_png(sl, slice_axes)}})
        except Exception:
            pass  # pictures are a bonus; the numbers alone are enough
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content if len(content) > 1 else text}]


def _paragraphs(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(p).strip() for p in value if str(p).strip()]


def parse_answer(text):
    """Pull the JSON object out of the model's reply (tolerates ```json fences and
    text or <think> blocks around it) and check it has the expected shape."""
    if not text:
        raise GemmaError("empty answer")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise GemmaError("the answer contained no JSON")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise GemmaError(f"the answer was not valid JSON ({exc.msg})") from exc
    out = {}
    for lang in ("en", "ms"):
        part = data.get(lang) if isinstance(data, dict) else None
        if not isinstance(part, dict):
            raise GemmaError(f"the answer has no '{lang}' section")
        out[lang] = {"volume": _paragraphs(part.get("volume")), "slice": _paragraphs(part.get("slice"))}
        if not out[lang]["volume"] and not out[lang]["slice"]:
            raise GemmaError(f"the '{lang}' section is empty")
    return out


def summarise(vol, sl, base_label, mon_label, draft=None, slice_axes=None, base_url=None, model=None):
    """Ask Gemma for the English + Malay summary. Raises GemmaError on any problem,
    so the caller can fall back to the rule-based text."""
    cfg = settings(base_url, model)
    client = make_client(cfg)
    messages = build_messages(vol, sl, base_label, mon_label, draft, slice_axes, cfg["images"])

    def ask(msgs):
        try:
            resp = client.chat.completions.create(model=cfg["model"], messages=msgs,
                                                  temperature=0.2, max_tokens=1500)
        except Exception as exc:
            raise GemmaError(f"{type(exc).__name__}: {exc}") from exc
        return (resp.choices[0].message.content or "") if resp.choices else ""

    try:
        result = parse_answer(ask(messages))
    except GemmaError as exc:
        if "JSON" not in str(exc) and "section" not in str(exc):
            raise
        # one retry, numbers only, in case the pictures or the format tripped it up
        text_only = build_messages(vol, sl, base_label, mon_label, draft, slice_axes, images=False)
        result = parse_answer(ask(text_only))
    result["en"]["note"] = NOTE_EN.format(model=cfg["model"])
    result["ms"]["note"] = NOTE_MS.format(model=cfg["model"])
    return result


def ping(base_url=None, model=None):
    """One short request, to check the server is reachable. Returns the reply text."""
    cfg = settings(base_url, model)
    resp = make_client(cfg).chat.completions.create(
        model=cfg["model"], temperature=0.2, max_tokens=60,
        messages=[{"role": "user", "content": "Reply with one short sentence: are you ready?"}])
    return resp.choices[0].message.content


if __name__ == "__main__":
    cfg = settings()
    print(f"Gemma server: {cfg['base_url']}  model: {cfg['model']}")
    try:
        print("Reply:", ping())
    except Exception as exc:
        raise SystemExit(f"Could not reach Gemma: {type(exc).__name__}: {exc}")
