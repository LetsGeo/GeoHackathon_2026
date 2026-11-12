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
