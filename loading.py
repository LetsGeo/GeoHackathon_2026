"""
Full-screen loading overlay with an animated oil pumpjack, shown while Run works.

How it's tied to the real Run (no timers decide when it ends):
  - show: the browser shows it the instant Run is clicked
    (app.py: load_btn.js_on_click calls window.__rtmLoading.show())
  - hide: app.py calls overlay.finished() in a `finally` when the Run has completely
    finished (worked or failed). That bumps `done`, and the browser hides it.

While it's up it covers the whole window above everything else, and blocks
scrolling (mouse wheel, touchpad, touch) and the keyboard, so nothing underneath
can be clicked, typed into or started.

The pumpjack moves with CSS animations only (no script runs per frame), so it
stays smooth even while the browser is busy drawing the new plots. Each moving
part is its own layer: the crank and counterweight turn, the walking beam rocks
on the samson post, the pitman arm (connecting rod) follows the crank pin, and
the polished rod rises and falls into the wellhead. Their poses are worked out
here from the real linkage (a rigid pitman arm between the crank pin and the
beam's rear end) at 36 steps per turn; one turn is one stroke, so it loops
seamlessly.
"""
import math

import panel as pn
import param

PERIOD_S = 2.6                           # one crank turn = one pump stroke
VIEW_W, VIEW_H = 400, 260                # drawing size (all coordinates below)
PIVOT = (190.0, 92.0)                    # top of the samson post: the beam rocks about it
CRANK = (315.0, 188.0)                   # crank shaft centre
CRANK_R = 20.0                           # crank pin radius
PITMAN = 95.0                            # pitman arm length (rigid)
BEAM_TAIL = (285.0, 101.0)               # where the pitman meets the beam, at rest
HORSEHEAD = (72.0, 132.0)                # bottom of the horsehead, at rest: the polished rod hangs here
ROD_LENGTH = 67.5                        # its lower end always stays hidden inside the stuffing box / wellhead
STEPS = 36


def _rotate(point, deg, centre):
    a = math.radians(deg)
    dx, dy = point[0] - centre[0], point[1] - centre[1]
    return (centre[0] + dx * math.cos(a) - dy * math.sin(a), centre[1] + dx * math.sin(a) + dy * math.cos(a))


def linkage_poses():
    """Poses over one turn: [(crank deg, beam deg, pin (x, y), pitman angle deg, rod top y)]."""
    beam_r = math.dist(PIVOT, BEAM_TAIL)
    a0 = math.atan2(BEAM_TAIL[1] - PIVOT[1], BEAM_TAIL[0] - PIVOT[0])
    poses = []
    for k in range(STEPS + 1):
        crank_deg = 360.0 * k / STEPS
        pin = _rotate((CRANK[0], CRANK[1] - CRANK_R), crank_deg, CRANK)
        # the beam's rear end is on its circle about the pivot AND exactly PITMAN from the pin
        dx, dy = pin[0] - PIVOT[0], pin[1] - PIVOT[1]
        d = math.hypot(dx, dy)
        a = (beam_r ** 2 - PITMAN ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(max(0.0, beam_r ** 2 - a ** 2))
        bx, by = PIVOT[0] + a * dx / d, PIVOT[1] + a * dy / d
        tail = min([(bx - h * dy / d, by + h * dx / d), (bx + h * dy / d, by - h * dx / d)], key=lambda p: p[1])
        beam_deg = math.degrees(math.atan2(tail[1] - PIVOT[1], tail[0] - PIVOT[0]) - a0)
        pitman_deg = math.degrees(math.atan2(tail[1] - pin[1], tail[0] - pin[0]))
        head = _rotate(HORSEHEAD, beam_deg, PIVOT)
        poses.append((crank_deg, beam_deg, pin, pitman_deg, head[1]))
    return poses


def _pct(value, size):
    return f"{value / size * 100:.4f}%"


def keyframes_css():
    """CSS animations for the moving layers, from the linkage poses."""
    poses = linkage_poses()
    beam = pitman = rod = ""
    for k, (_, beam_deg, pin, pitman_deg, rod_top) in enumerate(poses):
        at = f"{100 * k / STEPS:.4f}%"
        beam += f"{at} {{ transform: rotate({beam_deg:.3f}deg); }}\n"
        pitman += (f"{at} {{ transform: translate({_pct(pin[0], VIEW_W)}, {_pct(pin[1], VIEW_H)}) "
                   f"rotate({pitman_deg:.3f}deg); }}\n")
        rod += f"{at} {{ transform: translateY({_pct(rod_top, VIEW_H)}); }}\n"
    return f"""
.rtm-layer {{ position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; will-change: transform; }}
.rtm-crank  {{ transform-origin: {_pct(CRANK[0], VIEW_W)} {_pct(CRANK[1], VIEW_H)};
              animation: rtm-crank {PERIOD_S}s linear infinite; }}
.rtm-beam   {{ transform-origin: {_pct(PIVOT[0], VIEW_W)} {_pct(PIVOT[1], VIEW_H)};
              animation: rtm-beam {PERIOD_S}s linear infinite; }}
.rtm-pitman {{ transform-origin: 0 0; animation: rtm-pitman {PERIOD_S}s linear infinite; }}
.rtm-rod    {{ animation: rtm-rod {PERIOD_S}s linear infinite; }}
@keyframes rtm-crank  {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
@keyframes rtm-beam   {{ {beam} }}
@keyframes rtm-pitman {{ {pitman} }}
@keyframes rtm-rod    {{ {rod} }}
@media (prefers-reduced-motion: reduce) {{ .rtm-layer {{ animation-play-state: paused; }} }}
"""


def _layer(cls, body):
    return f'<svg class="rtm-layer {cls}" viewBox="0 0 {VIEW_W} {VIEW_H}" aria-hidden="true">{body}</svg>'


# Back to front. Moving layers are animated by keyframes_css(); the rod's bottom end
# is always covered by the stuffing box and wellhead drawn in the front layer.
PUMPJACK_LAYERS = "".join([
    _layer("", '<g class="rtm-jack">'
               '<rect x="18" y="228" width="364" height="5" rx="2"/>'
               '<rect x="118" y="217" width="236" height="11" rx="2"/>'
               '<line x1="164" y1="218" x2="190" y2="94" stroke-width="9" stroke-linecap="round"/>'
               '<line x1="216" y1="218" x2="190" y2="94" stroke-width="9" stroke-linecap="round"/>'
               '<line x1="171" y1="174" x2="209" y2="174" stroke-width="5"/>'
               '<line x1="177" y1="140" x2="203" y2="140" stroke-width="4"/>'
               '<rect x="296" y="196" width="40" height="22" rx="3"/>'
               '<rect x="300" y="188" width="32" height="10" rx="2"/></g>'),
    _layer("rtm-crank", '<g class="rtm-jack">'
                        '<rect x="309" y="162" width="12" height="36" rx="5"/>'
                        '<path d="M293,196 h44 a8,8 0 0 1 8,8 v12 a8,8 0 0 1 -8,8 h-44 a8,8 0 0 1 -8,-8 v-12 '
                        'a8,8 0 0 1 8,-8 z"/><circle cx="315" cy="168" r="5"/></g>'),
    _layer("", '<circle class="rtm-jack" cx="315" cy="188" r="7"/>'),
    _layer("rtm-pitman", f'<line class="rtm-jack" x1="0" y1="0" x2="{PITMAN}" y2="0" stroke-width="6" '
                         'stroke-linecap="round"/>'),
    _layer("rtm-rod", f'<line class="rtm-jack" x1="72" y1="0" x2="72" y2="{ROD_LENGTH}" stroke-width="3"/>'),
    _layer("rtm-beam", '<g class="rtm-jack">'
                       '<rect x="96" y="85" width="196" height="13" rx="3"/>'
                       '<rect x="272" y="96" width="22" height="8" rx="2"/>'
                       '<path d="M102,80 L86,80 Q57,93 68,134 L80,131 Q73,102 102,99 Z"/></g>'),
    _layer("", '<g class="rtm-jack">'
               '<circle cx="190" cy="92" r="7"/>'
               '<rect x="68" y="170" width="8" height="32" rx="2"/>'     # stuffing box: hides the rod's end
               '<rect x="64" y="200" width="16" height="28" rx="2"/>'    # wellhead
               '<rect x="56" y="207" width="32" height="8" rx="2"/></g>'
               '<path class="rtm-oil" d="M56,211 L30,211 L30,227"/>'
               '<path class="rtm-flow" d="M56,211 L30,211 L30,227"/>'),
])

LOCK_JS = """
// block scrolling and the keyboard everywhere while the overlay is up
state.block = (e) => { e.preventDefault(); e.stopPropagation(); };
state.lock = () => {
  state.saved = [document.documentElement.style.overflow, document.body.style.overflow];
  document.documentElement.style.overflow = 'hidden';
  document.body.style.overflow = 'hidden';
  for (const type of ['wheel', 'touchmove', 'keydown', 'keypress', 'keyup'])
    window.addEventListener(type, state.block, {capture: true, passive: false});
  if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
};
state.unlock = () => {
  for (const type of ['wheel', 'touchmove', 'keydown', 'keypress', 'keyup'])
    window.removeEventListener(type, state.block, {capture: true});
  document.documentElement.style.overflow = state.saved ? state.saved[0] : '';
  document.body.style.overflow = state.saved ? state.saved[1] : '';
};
"""


class LoadingOverlay(pn.reactive.ReactiveHTML):
    """The full-screen Run overlay. Call finished() when the Run is over (in a finally)."""

    done = param.Integer(default=0, doc="Bumped by finished(); the browser hides the overlay when it changes")

    _template = (
        '<div id="overlay" class="rtm-loading" role="alert" aria-busy="true" aria-live="polite">'
        '<div class="rtm-stage">' + PUMPJACK_LAYERS + '</div>'
        '<div class="rtm-title">Pumping things into motion'
        '<span class="rtm-dots"><span>.</span><span>.</span><span>.</span></span></div>'
        '<div class="rtm-sub">Loading the surveys and building the views</div>'
        '<div class="rtm-bar"><i></i></div>'
        '</div>'
    )
    _scripts = {
        "render": LOCK_JS + """
          // the Run button (in the browser) calls window.__rtmLoading.show() the moment it's clicked
          window.__rtmLoading = {show: () => self.show(), hide: () => self.hide()};
        """,
        "show": """
          clearTimeout(state.hideTimer);
          if (!state.running) {
            state.running = true;
            state.lock();
            overlay.style.display = 'flex';                                // starts the CSS animations
          }
          void overlay.offsetWidth;                                       // apply display first, so the
          overlay.classList.add('show');                                  // opacity change fades in
        """,
        "hide": """
          if (!state.running) return;
          overlay.classList.remove('show');                              // fades out (CSS)
          clearTimeout(state.hideTimer);
          state.hideTimer = setTimeout(() => {                           // after the fade: remove it
            if (overlay.classList.contains('show')) return;              // shown again meanwhile
            overlay.style.display = 'none';                              // stops the CSS animations
            state.running = false;
            state.unlock();
          }, 300);
        """,
        "done": "self.hide();",
    }

    def __init__(self, **params):
        # the moving parts' keyframes come from the linkage maths above
        params["stylesheets"] = list(params.get("stylesheets", [])) + [keyframes_css()]
        super().__init__(**params)

    def finished(self):
        """The Run is over: tell the browser to hide the overlay."""
        self.done += 1
