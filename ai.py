"""
AI descriptions of a point on the seismic ("AI at cursor" in the workspace).

app.py calls describe_position(context) when the mouse rests on a seismic plot.
For now it returns a placeholder, marked [PH]. To use Gemma, replace the body
of describe_position() with a call to the model, sending build_prompt(context)
and returning its reply as plain text. Nothing else in the app needs to change.

context holds (None where unknown):
    view            "Inline", "Crossline" or "Time slice"
    inline          inline number under the cursor
    crossline       crossline number under the cursor
    time_ms         two-way time in ms
    baseline        baseline amplitude there
    monitor         monitor amplitude there
    difference      monitor - baseline
    baseline_label  e.g. "1994 survey"
    monitor_label   e.g. "2010 survey"
    amp_scale       a typical strong amplitude for these data (99th percentile),
                    to judge whether a value is weak or strong
"""

import json
import os
import re

PLACEHOLDER_TAG = "[PH]"

# Gemma server (OpenAI-compatible API on the hackathon network). Set the
# GEMMA_URL / GEMMA_MODEL environment variables to use a different one.
GEMMA_URL = os.environ.get("GEMMA_URL", "http://172.28.34.1:8000/v1")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-26B-A4B-it")
GEMMA_TIMEOUT_S = 30  # give up after this long, so the app doesn't hang when the server is unreachable


def gemma_reachable(timeout_s=3):
    """Quick check that something is listening at GEMMA_URL. Off the hackathon
    network the address doesn't answer at all, and without this check the app
    would wait the full GEMMA_TIMEOUT_S before giving up."""
    import socket
    from urllib.parse import urlparse
    url = urlparse(GEMMA_URL)
    try:
        socket.create_connection((url.hostname, url.port or (443 if url.scheme == "https" else 80)),
                                 timeout=timeout_s).close()
        return True
    except OSError:
        return False


def gemma_chat(system, user, max_tokens=800, temperature=0.1):
    """Ask Gemma once and return its reply text. Needs `pip install openai`;
    raises if the package is missing or the server can't be reached."""
    import openai
    if not gemma_reachable():
        raise ConnectionError(f"the Gemma server ({GEMMA_URL}) isn't reachable from this computer")
    client = openai.OpenAI(base_url=GEMMA_URL, api_key="token-not-needed",
                           timeout=GEMMA_TIMEOUT_S, max_retries=0)
    reply = client.chat.completions.create(
        model=GEMMA_MODEL, temperature=temperature, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return reply.choices[0].message.content


def parse_json_reply(text):
    """The JSON object inside a model reply (models sometimes add text around it)."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("the reply had no JSON in it")
    return json.loads(match.group(0))


def translate_texts(texts, language):
    """Translate a dict of English report texts into `language` (e.g. "Simplified
    Chinese", "Tamil") with Gemma. Returns a dict with the same keys; a value that
    is a list comes back as a list of the same length. Raises if Gemma can't be
    reached or its reply doesn't fit."""
    system = (
        f"You translate short 4D seismic report texts from English into {language}. "
        "Keep every number, unit (ms, %), inline / crossline number, survey name and the "
        "abbreviation NRMS exactly as written. Use the usual technical term in "
        f"{language} where one exists; otherwise keep the English term (e.g. push-down) in brackets. "
        "Reply with a JSON object only: the same keys, each value translated, lists kept as lists "
        "of the same length."
    )
    reply = gemma_chat(system, json.dumps(texts, ensure_ascii=False), max_tokens=2000)
    out = parse_json_reply(reply)
    for key, value in texts.items():
        got = out.get(key)
        if isinstance(value, list):
            if not isinstance(got, list) or len(got) != len(value):
                raise ValueError(f"the reply didn't translate '{key}' properly")
            out[key] = [str(g) for g in got]
        elif value and not got:
            raise ValueError(f"the reply was missing '{key}'")
        else:
            out[key] = str(got or "")
    return out


def suggest_horizons(context):
    """Horizons Gemma suggests for one inline / crossline section: the "Draw
    horizons with Gemma" button in the Layers area calls this, and the app lets
    the user pick one of the options it returns.

    PLACEHOLDER: returns 3 made-up options marked [PH] so the UI can be tried.
    Replace the body with the real Gemma call; nothing else in the app needs to change.

    context:
        view            "Inline" or "Crossline"
        position        the inline / crossline number on screen
        along           (first, last) crossline (on an inline) or inline (on a crossline)
        twt_range       (top, bottom) two-way time shown, in ms
        baseline_label  e.g. "1994 survey"
    Return a list of options (3 are shown), each:
        {"name": str, "description": str,
         "along": [line numbers along the section], "twt": [two-way time in ms, same length]}
    """
    import numpy as np
    a0, a1 = context["along"]
    t0, t1 = context["twt_range"]
    along = np.linspace(a0, a1, 40)
    s = (along - a0) / max(a1 - a0, 1)
    options = []
    for k, (frac, name) in enumerate([(0.42, "Option A"), (0.5, "Option B"), (0.58, "Option C")]):
        twt = t0 + (t1 - t0) * frac + 18 * np.sin(s * np.pi * (1.5 + k * 0.5)) - 25 * np.exp(-((s - 0.45) / 0.15) ** 2)
        options.append({"name": f"{PLACEHOLDER_TAG} {name}",
                        "description": f"{PLACEHOLDER_TAG} Gemma's reason for this pick will appear here.",
                        "along": along.round(1).tolist(), "twt": twt.round(1).tolist()})
    return options


def _fmt(value, digits=3):
    return "unknown" if value is None else f"{value:.{digits}g}"


def build_prompt(context):
    """Text to send to Gemma for this position."""
    c = context
    return (
        "You are helping a geoscientist read 4D (time-lapse) seismic. Describe briefly, in one or two "
        "sentences, what is likely at this position and whether it has changed between the surveys.\n"
        f"Position: inline {_fmt(c.get('inline'), 6)}, crossline {_fmt(c.get('crossline'), 6)}, "
        f"two-way time {_fmt(c.get('time_ms'), 6)} ms ({c.get('view')} view).\n"
        f"Amplitude in {c.get('baseline_label') or 'the baseline'}: {_fmt(c.get('baseline'))}; "
        f"in {c.get('monitor_label') or 'the monitor'}: {_fmt(c.get('monitor'))}; "
        f"change: {_fmt(c.get('difference'))}. A strong amplitude for these data is about "
        f"{_fmt(c.get('amp_scale'))}."
    )


def describe_position(context):
    """What the AI says about this position (plain text). Placeholder for now."""
    c = context
    return (f"{PLACEHOLDER_TAG} Gemma's description of inline {_fmt(c.get('inline'), 6)}, "
            f"crossline {_fmt(c.get('crossline'), 6)} at {_fmt(c.get('time_ms'), 6)} ms will appear here.")
