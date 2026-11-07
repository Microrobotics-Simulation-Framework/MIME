"""A view in a web page: one self-contained HTML file holding the GLB (``mime.viz.gltf``) as a data URI and Google's
``<model-viewer>`` 4.3.1 to draw it, with play and pause, a speed, a slider over the recording, its simulated time,
the scene's size, what the view leaves out and every set's terms.

The viewer is inlined (a copy shipped in the wheel, ``_viewer/model-viewer.min.js``, with the licence of every part
its bundle holds), so the page draws with no network, from a file or a notebook alike; or, smaller by 1.4 MB, loaded
from jsDelivr and held to its hash (``cdn=True``). A notebook's output is the page inside an ``<iframe srcdoc>``
(:class:`WebPage`), so its module script runs as it does in a browser. GitHub's notebook preview runs no script, so
there the view is blank."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import override

from mime._lineage import Origin, origin_of_folder
from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors._frames import warn_at_script
from mime.things._handles import did_you_mean
from mime.viz._export import ArrowKey, BodyTrack, GltfResult, gltf, length_text

VIEWER_VERSION = "4.3.1"
VIEWER_FILE = "model-viewer.min.js"
#: The viewer's bytes, pinned: its sha256, and the integrity jsDelivr's copy is held to (sha384, as SRI writes it).
VIEWER_SHA256 = "283b0672384614b4847636c306fc93fe4b1fcadc76d668b4e47f0ca76bcf033b"
VIEWER_SRI = "sha384-cprcVQt7wbUl0xngF3PGP6yBB7n4/t+4AoAMG9biiMCGFiWOdzUH10Ie2COTqFNW"
VIEWER_CDN = f"https://cdn.jsdelivr.net/npm/@google/model-viewer@{VIEWER_VERSION}/dist/{VIEWER_FILE}"
#: Above this many bytes a notebook's output is a large share of the notebook's own file (every display is saved in it):
#: ``display`` warns, and says how to make it smaller. Not a limit of Jupyter's: its output rate limit counts a kernel's
#: stream output only, so a displayed value of any size is not dropped by it.
OUTPUT_WARNING_BYTES = 1_000_000
#: The speeds offered, display seconds per second.
SPEEDS = (0.25, 0.5, 1.0, 2.0)

CREDIT = (
    'Drawn with <a href="https://modelviewer.dev/">model-viewer</a> 4.3.1 (Apache-2.0), whose bundle holds '
    '<a href="https://threejs.org/">three.js</a> (MIT), <a href="https://lit.dev/">lit</a> (BSD-3-Clause), fflate '
    '(MIT) and gainmap-js (MIT); converted from USD by MIME, its converter derived from usd2gltf 0.3.5 (MIT).')


def viewer_bytes() -> bytes:
    """The inlined viewer's bytes, held to their pinned sha256."""
    data = files("mime.viz").joinpath("_viewer", VIEWER_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != VIEWER_SHA256:
        raise ValueError(f"mime/viz/_viewer/{VIEWER_FILE} is not the viewer MIME pins")
    return data


def _time_unit(simulated_s: float) -> tuple[str, float, int]:
    if simulated_s >= 1.0:
        return "s", 1.0, 3
    if simulated_s >= 1e-3:
        return "ms", 1e3, 2
    return "µs", 1e6, 1


def _viewer_tag(cdn: bool) -> str:
    if cdn:
        return f'<script type="module" src="{VIEWER_CDN}" integrity="{VIEWER_SRI}" crossorigin="anonymous"></script>'
    encoded = base64.b64encode(viewer_bytes()).decode("ascii")
    return f'<script type="module" src="data:text/javascript;base64,{encoded}"></script>'


_STYLE = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --panel: #f3f4f6; --edge: #d1d9e0; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --panel: #161b22; --edge: #3d444d; }
}
:root[data-theme="dark"] { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --panel: #161b22; --edge: #3d444d; }
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, -apple-system, sans-serif;
  display: flex; flex-direction: column; overflow: hidden; }
.stage { position: relative; flex: 1 1 auto; min-height: 160px; background: var(--panel); }
model-viewer { position: absolute; inset: 0; display: block; width: 100%; height: 100%; background: var(--panel); }
.overlay { position: absolute; z-index: 1; pointer-events: none; font-size: 12px; color: var(--fg);
  background: color-mix(in srgb, var(--bg) 80%, transparent); border: 1px solid var(--edge); border-radius: 6px;
  padding: 4px 8px; }
#scale-bar { left: 12px; bottom: 12px; }
#scale-line { height: 6px; width: 0; border: 2px solid var(--fg); border-top: none; }
#scale-label { font-variant-numeric: tabular-nums; }
#legend { right: 12px; top: 12px; max-width: min(34em, 60%); }
#legend ul { list-style: none; margin: 2px 0 0; padding: 0; }
#legend li { margin: 2px 0; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
.toggles { margin: 4px 0; padding-left: 20px; }
.grip { flex: 0 0 auto; height: 12px; cursor: ns-resize; touch-action: none; display: flex; align-items: center;
  justify-content: center; background: var(--panel); border-top: 1px solid var(--edge); }
.grip::before { content: ""; width: 48px; height: 4px; border-radius: 2px; background: var(--muted); opacity: 0.6; }
.grip:focus-visible { outline: 2px solid var(--muted); outline-offset: -2px; }
.controls { flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; padding: 10px 16px;
  border-bottom: 1px solid var(--edge); }
.controls input[type=range] { flex: 1 1 200px; min-width: 0; }
.controls button, .controls select { font: inherit; color: var(--fg); background: var(--panel);
  border: 1px solid var(--edge); border-radius: 6px; padding: 2px 10px; }
#clock { font-variant-numeric: tabular-nums; }
.notes { flex: 0 0 auto; overflow: auto; }
.notes-body { padding: 8px 16px 16px; max-width: 72em; }
.notes p { margin: 6px 0; }
.muted { color: var(--muted); }
a { color: inherit; }
"""

_LOGIC = """
// The view's arithmetic, apart from the page so that it can be run and tested without a browser.
// The metres one pixel spans at the point the camera orbits: the view's height there is 2 r tan(fov / 2) glTF units.
function metresPerPixel(unitMetres, radius, fovDegrees, heightPixels) {
  return unitMetres * 2 * radius * Math.tan(fovDegrees * Math.PI / 360) / heightPixels;
}
// The largest of 1, 2 and 5 times a power of ten that is not more than metres: a round length.
function niceLength(metres) {
  if (!(metres > 0) || !isFinite(metres)) return 0;
  const base = Math.pow(10, Math.floor(Math.log10(metres)));
  const mantissa = metres / base;
  return (mantissa >= 5 ? 5 : mantissa >= 2 ? 2 : 1) * base;
}
const PREFIXES = [[1, "m"], [1e-3, "mm"], [1e-6, "\\u00b5m"], [1e-9, "nm"], [1e-12, "pm"]];
function lengthLabel(metres) {
  let chosen = PREFIXES[PREFIXES.length - 1];
  for (const prefix of PREFIXES) { if (metres >= prefix[0] * 0.999999) { chosen = prefix; break; } }
  return String(Number((metres / chosen[0]).toPrecision(6))) + " " + chosen[1];
}
// The scale bar for a view: a round length and the pixels it spans, aiming at targetPixels.
function scaleBar(unitMetres, radius, fovDegrees, heightPixels, targetPixels) {
  const perPixel = metresPerPixel(unitMetres, radius, fovDegrees, heightPixels);
  const metres = niceLength(perPixel * targetPixels);
  return { metres: metres, pixels: metres / perPixel, label: metres > 0 ? lengthLabel(metres) : "" };
}
// A recorded position at display time t: the samples joined by straight lines, held before the first and after the last.
function followAt(times, positions, t) {
  const last = times.length - 1;
  if (last < 1 || t <= times[0]) return positions[0];
  if (t >= times[last]) return positions[last];
  let low = 0, high = last;
  while (high - low > 1) { const mid = (low + high) >> 1; if (times[mid] <= t) low = mid; else high = mid; }
  const weight = (t - times[low]) / (times[high] - times[low]);
  return positions[low].map((v, k) => v + (positions[high][k] - v) * weight);
}
"""

_SCRIPT = """
const view = document.getElementById("view");
const config = CONFIG;
// The wheel (a mouse's notches, a touchpad's scroll) zooms here, not in the viewer: by a factor in proportion to how far
// it turned, toward the point under the pointer, with the view's angle kept, as close as config.closest. The viewer's own
// zoom moves in equal steps toward a limit and narrows the view as it goes, so a body a millionth of its scene could not
// be reached in comfort, and a touchpad's many small events each jumped to the limit. A pinch keeps the viewer's.
// The handle under the view: dragging it down gives the view the room the notes had, down to the controls alone at the
// bottom; up gives the notes back, up to their whole length. A double-click (or Enter) folds or unfolds the notes.
const grip = document.getElementById("grip");
const notes = document.getElementById("notes");
let unfolded = null;
function notesHeight(px) { notes.style.height = Math.max(0, Math.min(px, notes.scrollHeight)) + "px"; }
function shown() { return notes.getBoundingClientRect().height; }
function fold() {
  if (shown() > 0) { unfolded = shown(); notesHeight(0); }
  else notesHeight(unfolded ?? Math.min(notes.scrollHeight, window.innerHeight * 0.35));
}
if (grip && notes) {
  notesHeight(Math.min(notes.scrollHeight, window.innerHeight * 0.35));
  grip.addEventListener("pointerdown", (event) => {
    const fromY = event.clientY, from = shown();
    grip.setPointerCapture(event.pointerId);
    const move = (e) => notesHeight(from - (e.clientY - fromY));
    const up = () => { grip.removeEventListener("pointermove", move); grip.removeEventListener("pointerup", up); };
    grip.addEventListener("pointermove", move);
    grip.addEventListener("pointerup", up);
  });
  grip.addEventListener("dblclick", fold);
  grip.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") notesHeight(shown() - 24);
    else if (event.key === "ArrowUp") notesHeight(shown() + 24);
    else if (event.key === "Enter" || event.key === " ") fold();
    else return;
    event.preventDefault();
  });
}
const ZOOM_PER_PIXEL = 0.0025;
if (config.closest !== null) {
    // The viewer draws nothing nearer its camera than its near plane, which it sets once, when it frames the scene, at a
  // thousandth of its far one: twice the larger of the camera's distance then and the scene's reach. The camera may come
  // half as far again as that, never nearer, so what it is pointed at is never cut away; config.closest is the floor
  // a viewer with a moving near plane could reach.
  let closest = config.closest;
  view.addEventListener("load", () => {
    const framed = Math.max(view.getCameraOrbit().radius, config.reach);
    closest = Math.max(config.closest, 1.5 * (2 * framed / 1000));
    view.minCameraOrbit = "auto auto " + closest + "m";
  });
  view.addEventListener("wheel", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1);
    const orbit = view.getCameraOrbit();
    const radius = Math.max(closest, orbit.radius * Math.exp(pixels * ZOOM_PER_PIXEL));
    if (radius < orbit.radius) {
      const hit = view.positionAndNormalFromPoint(event.clientX, event.clientY);  // the page's coordinates, as it takes them
      if (hit) {
        const target = view.getCameraTarget(), towards = 1 - radius / orbit.radius;
        view.cameraTarget = [target.x + (hit.position.x - target.x) * towards, target.y + (hit.position.y - target.y) * towards,
                             target.z + (hit.position.z - target.z) * towards].map((v) => v + "m").join(" ");
      }
    }
    view.cameraOrbit = orbit.theta + "rad " + orbit.phi + "rad " + radius + "m";
  }, { capture: true, passive: false });
}
const play = document.getElementById("play");
const speed = document.getElementById("speed");
const time = document.getElementById("time");
const clock = document.getElementById("clock");
function show() {
  if (!clock) return;
  const simulated = view.currentTime / config.slow * config.factor;
  clock.textContent = "t = " + simulated.toFixed(config.digits) + " " + config.unit + " simulated" + config.shown;
}
function label() { if (play) play.textContent = view.paused ? "Play" : "Pause"; }
// The scale bar is true at the point the camera orbits, and is redrawn as the camera moves.
const barLine = document.getElementById("scale-line");
const barLabel = document.getElementById("scale-label");
function drawBar() {
  if (!barLine) return;
  const shown = scaleBar(config.unit_m, view.getCameraOrbit().radius, view.getFieldOfView(), view.clientHeight, config.bar_pixels);
  barLine.style.width = shown.pixels + "px";
  barLabel.textContent = shown.label;
}
// A followed body keeps the camera's target on its recorded position at the display time the animation is at.
const followBox = document.getElementById("follow");
function follow() {
  if (!config.follow || (followBox && !followBox.checked)) return;
  const at = followAt(config.follow.times, config.follow.positions, view.currentTime);
  view.cameraTarget = at.map((v) => v + "m").join(" ");
  view.jumpCameraToGoal();
}
function tick() {
  if (time && !view.paused) time.value = String(view.currentTime);
  show();
  follow();
  drawBar();
  requestAnimationFrame(tick);
}
view.addEventListener("load", () => {
  document.body.dataset.loaded = "1";
  // Focus: the camera's target on the body's first position, and its distance a few times the body's extent. The
  // viewer clamps both before it loads, so they are set here, after the closest the camera may come is.
  if (config.focus) {
    view.cameraTarget = config.focus.target.map((v) => v.toFixed(9) + "m").join(" ");
    if (config.focus.radius !== null) {
      const orbit = view.getCameraOrbit();
      view.cameraOrbit = orbit.theta + "rad " + orbit.phi + "rad " + config.focus.radius + "m";
    }
    view.jumpCameraToGoal();
  }
  document.body.dataset.duration = String(view.duration);
  document.body.dataset.animations = JSON.stringify(view.availableAnimations);
  if (time) time.max = String(view.duration);
  label();
  requestAnimationFrame(tick);
});
view.addEventListener("error", (event) => { document.body.dataset.error = JSON.stringify(event.detail ?? null); });
if (play) play.addEventListener("click", () => { if (view.paused) view.play(); else view.pause(); label(); });
if (speed) speed.addEventListener("change", () => { view.timeScale = Number(speed.value); });
if (time) time.addEventListener("input", () => {
  view.pause();
  view.currentTime = Number(time.value);
  label();
  show();
});
"""


#: The closest the camera may come, as a fraction of the scene's diagonal: a millionth, so that a micrometre's body in a
#: metre's scene (a microrobot in a vessel, a vessel in a body) can be approached. The viewer's own limit stops the
#: camera outside the whole scene's bounds, which leaves a millimetre's helix beside a robot arm out of reach.
CLOSEST = 1e-6


def closest_radius(result: GltfResult) -> float | None:
    """How close the camera may come at the most, in the glTF's own unit: :data:`CLOSEST` of the scene's diagonal;
    none for a scene with no extent, which keeps the viewer's own limit."""
    diagonal = result.across_m / result.unit_m
    return diagonal * CLOSEST if diagonal > 0.0 else None


#: A body is framed at this many times its largest extent from the camera when a view is focused on it.
FOCUS_FACTOR = 5.0
#: The scale bar aims at this many pixels.
BAR_PIXELS = 110
_VECTOR_KINDS = {"T": "magnetic flux density", "A/m": "magnetic field", "N": "force", "N m": "torque",
                 "m/s": "velocity", "rad/s": "angular velocity"}


def _body(result: GltfResult, name: str, argument: str) -> BodyTrack:
    """The track of the body ``name``, or a refusal naming the nearest bodies."""
    if name in result.bodies:
        return result.bodies[name]
    raise ArgumentRefused([Refusal(
        "visuals.unknown-body", RefusalClass.USER,
        f"{argument}={name!r} names no body of {result.name}'s scene. {did_you_mean(name, result.bodies)}",
        (f"name one of its bodies: {argument}={sorted(result.bodies)[0]!r} (mime visuals web <target> --{argument} <body>)"
         if result.bodies else f"the scene has no bodies, so leave `{argument}=None` (`mime visuals web <target>`)"))])


def _legend_line(key: ArrowKey) -> str:
    kind = _VECTOR_KINDS.get(key.unit)
    what = f"{key.name}: {kind} ({key.unit})" if kind else f"{key.name} ({key.unit})" if key.unit else key.name
    line = f"{what}, drawn {length_text(key.metres_per_unit)} per {key.unit or 'unit'}"
    if key.limit_m is not None:
        line += (f"; never longer than {length_text(key.limit_m)} (a vector of {key.limit_m / key.metres_per_unit:.3g} "
                 f"{key.unit} or more is drawn at that length, its direction kept)")
    if key.cap is not None:
        line += (f"; capped in this view: its longest is drawn {length_text(key.cap.drawn_m)}, not "
                 f"{length_text(key.cap.would_be_m)}")
    return line


def _legend(result: GltfResult) -> str:
    if not result.arrows:
        return ""
    items = "".join(
        f'<li data-prim="{html.escape(key.prim)}"><span class="swatch" style="background: rgb('
        f'{round(key.colour[0] * 255)}, {round(key.colour[1] * 255)}, {round(key.colour[2] * 255)})"></span>'
        f"{html.escape(_legend_line(key))}</li>" for key in result.arrows)
    return f'<div class="overlay" id="legend"><strong>Arrows</strong><ul>{items}</ul></div>'


def _toggle(name: str, on: bool, said: str, how: str) -> str:
    return f'<li data-toggle="{name}" data-state="{"on" if on else "off"}">{html.escape(said)} <span class="muted">({html.escape(how)})</span></li>'


def page(result: GltfResult, *, cdn: bool = False, caption: str | None = None, scale_bar: bool = True,
         legend: bool = True, focus: str | None = None, follow: str | None = None) -> str:
    """The page of ``result``: the viewer inlined, or from jsDelivr with ``cdn``; ``caption`` above the notes.

    ``scale_bar`` draws a round length in SI units that follows the zoom; ``legend`` says what each arrow is, its colour
    and the scale it is drawn at, with its cap; ``focus`` (a body's name) frames that body at the start; ``follow`` keeps
    the camera's target on that body along the recording. A name the scene does not hold is refused
    (``visuals.unknown-body``). Whether the motor and magnet of an apparatus are in the view was decided when the
    result was made (``gltf(..., apparatus_shapes=)``), and the notes say it."""
    focused = _body(result, focus, "focus") if focus is not None else None
    followed = _body(result, follow, "follow") if follow is not None else None
    glb = base64.b64encode(result.glb).decode("ascii")
    animated = result.samples > 1 and result.simulated_s > 0.0
    unit, factor, digits = _time_unit(result.simulated_s)
    if result.slow == 1.0:
        shown = " (shown at its own speed)"
    elif result.slow > 1.0:
        shown = f" (shown {result.slow:,.4g} times slower)"
    else:
        shown = f" (shown {1.0 / result.slow:,.4g} times faster)"
    settings: dict[str, object] = {
        "slow": result.slow, "unit": unit, "factor": factor, "digits": digits, "shown": shown,
        "closest": closest_radius(result), "reach": result.across_m / result.unit_m / 2.0,
        "unit_m": result.unit_m, "bar_pixels": BAR_PIXELS, "scale_bar": scale_bar, "legend": legend and bool(result.arrows),
        "follow": {"name": follow, "times": [float(f"{t:.9g}") for t in followed.times],
                   "positions": [[float(f"{v:.9g}") for v in p] for p in followed.positions]}
        if followed is not None and follow is not None else None,
        "focus": {"name": focus, "target": [float(f"{v:.9g}") for v in focused.positions[0]],
                  "radius": FOCUS_FACTOR * focused.extent if focused.extent else None}
        if focused is not None and focus is not None else None}
    config = json.dumps(settings)
    title = html.escape(result.name)
    play = ' autoplay animation-name="run"' if animated else ""
    parts = [
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">', f"<title>{title}</title>",
        f"<style>{_STYLE}</style>", _viewer_tag(cdn), "</head>", "<body>",
        '<div class="stage" id="stage">',
        f'<model-viewer id="view" src="data:model/gltf-binary;base64,{glb}" alt="{title}, drawn from its USD stage" '
        f'camera-controls{play} interaction-prompt="none" shadow-intensity="0" exposure="1"></model-viewer>']
    if scale_bar:
        parts.append('<div class="overlay" id="scale-bar" aria-label="scale bar"><div id="scale-line"></div>'
                     '<span id="scale-label"></span></div>')
    if legend:
        parts.append(_legend(result))
    parts += ['</div>',
              '<div class="grip" id="grip" role="separator" aria-orientation="horizontal" tabindex="0" '
              'aria-label="Resize the notes below the view" '
              'title="Drag down for more view, up for more notes; double-click to fold or unfold the notes"></div>']
    if animated:
        options = "".join(f'<option value="{s:g}"{" selected" if s == 1.0 else ""}>{s:g}x</option>' for s in SPEEDS)
        start = f"t = {0:.{digits}f} {unit} simulated{shown}"
        following = (f'<label><input id="follow" type="checkbox" checked> follow {html.escape(follow)}</label>'
                     if follow is not None else "")
        parts.append(
            '<div class="controls">'
            '<button id="play" type="button">Pause</button>'
            f'<label>speed <select id="speed">{options}</select></label>{following}'
            f'<input id="time" type="range" min="0" max="{result.simulated_s * result.slow:.6g}" step="any" value="0" '
            'aria-label="the recording\'s time">'
            f'<span id="clock">{html.escape(start)}</span></div>')
    notes = [f"<p>{html.escape(caption)}</p>"] if caption else []
    span = (f"{result.samples} samples over {result.simulated_s * factor:.{digits}f} {unit} simulated"
            if animated else "a scene with no recording")
    notes.append(f'<p class="muted">{html.escape(result.name)}: about {html.escape(result.size_text())} across '
                 f"over its whole motion; {html.escape(span)}.</p>")
    notes.append('<p class="muted">Scroll to zoom toward the pointer, as close as the view can draw (about a hundredth of the scene); click a part to orbit around it; right-drag or two fingers to pan.</p>')
    drawn = ", ".join(sorted({shape.name for shape in result.shapes})) or "none in this scene"
    toggles = [
        _toggle("scale_bar", scale_bar, "Scale bar: " + ("on. A round length in SI units, true at the point the camera "
                "orbits, redrawn as you zoom." if scale_bar else "off."),
                "scale_bar=False or --no-scale-bar hides it" if scale_bar else "scale_bar=True shows it"),
        _toggle("legend", legend, "Arrow legend: " + (("on. What each arrow is, its colour, the scale it is drawn at and "
                "its cap." if result.arrows else "on, but this scene draws no arrows.") if legend else "off."),
                "legend=False or --no-legend hides it" if legend else "legend=True shows it"),
        _toggle("apparatus_shapes", result.apparatus_shapes, "Apparatus shapes: " + (
                f"on. The motor and magnet of an apparatus are drawn for this view alone, never in the stage ({drawn})."
                if result.apparatus_shapes else "off."),
                "apparatus_shapes=False or --no-apparatus-shapes leaves them out, set where the view is made: gltf, "
                "display, web" if result.apparatus_shapes else "apparatus_shapes=True draws them"),
        _toggle("focus", focus is not None, f"Focus: {focus}. The camera starts framed on it." if focus is not None else
                "Focus: off. The camera frames the whole motion.",
                "focus=None or no --focus turns it off" if focus is not None else "focus='<body>' or --focus <body>"),
        _toggle("follow", follow is not None, f"Follow: {follow}. The camera's target keeps on its recorded position"
                f"{', with a checkbox to release it' if animated else ' (a scene with no recording has one position, so the camera is held there and cannot be released)'}."
                if follow is not None else
                "Follow: off. The camera stays where you put it.",
                "follow=None or no --follow turns it off" if follow is not None else "follow='<body>' or --follow <body>")]
    notes.append(f'<ul class="toggles" id="toggles">{"".join(toggles)}</ul>')
    notes += [f'<p class="muted">{html.escape(line)}</p>' for line in result.lines()]
    where = ("The viewer is loaded from jsDelivr, held to its hash; the scene is inside this page." if cdn else
             "The viewer and the scene are both inside this page: it needs no network.")
    notes.append(f'<p class="muted">{CREDIT} {where}</p>')
    parts += ['<div class="notes" id="notes"><div class="notes-body">', *notes, "</div></div>",
              f'<script type="module">{_LOGIC}{_SCRIPT.replace("CONFIG", config)}</script>', "</body>", "</html>"]
    return "\n".join(parts) + "\n"


@dataclass(frozen=True)
class WebPage:
    """A view as a notebook shows it: the page in an ``<iframe srcdoc>``. A cell whose last expression it is shows it;
    :meth:`save` writes the page as a file. Its ``repr`` is one line, not the page."""

    html: str
    result: GltfResult
    height: int = 720
    #: The declarations the run came from, for a view of a run made in this process (``mime._lineage``).
    origin: Origin | None = None

    @property
    def stale(self) -> str | None:
        """Why this view no longer matches its experiment, or ``None``: the replacements and removals made since the
        build its run came from, and what to do. Shown wherever the view is shown."""
        return self.origin.stale("this view", "it shows a run of the declarations before: build and run again") \
            if self.origin is not None else None

    @override
    def __repr__(self) -> str:
        """One line, not the page: Jupyter shows ``text/plain`` beside the HTML, and the page is megabytes."""
        stale = self.stale
        return (f"<WebPage {self.result.name!r}, {len(self.html.encode('utf-8')) / 1e3:.0f} kB: a notebook cell shows it "
                f"when it is the cell's last expression; elsewhere .save('view.html') writes the page, to open in a "
                f"browser" + (f"; STALE, {stale}" if stale is not None else "") + ">")

    def _repr_html_(self) -> str:
        stale = self.stale
        banner = f'<p style="color: #b3261e; font-weight: bold">STALE: {html.escape(stale)}</p>' if stale else ""
        return (banner + f'<iframe srcdoc="{html.escape(self.html, quote=True)}" style="width: 100%; height: {self.height}px; '
                f'border: 0" title="{html.escape(self.result.name)}"></iframe>')

    def save(self, path: str | Path) -> Path:
        written = Path(path)
        written.write_text(self.html, encoding="utf-8")
        return written


def _stale_caption(origin: Origin | None, caption: str | None) -> str | None:
    """``caption`` with the mark of a view of a run that predates a replacement or a removal (warned about, too), where
    the run's folder is one this process wrote; the caption alone elsewhere (a copy of the folder, one another process
    wrote, cannot be known stale and is shown with no mark)."""
    stale = origin.stale("this view", "it shows a run of the declarations before: build and run again") \
        if origin is not None else None
    if stale is None:
        return caption
    warn_at_script(stale, UserWarning)
    return f"STALE: {stale}. {caption}" if caption else f"STALE: {stale}"


def display(target: str | Path, *, offline: bool = False, caption: str | None = None, slow: float | None = None,
            scale_bar: bool = True, legend: bool = True, apparatus_shapes: bool = True, focus: str | None = None,
            follow: str | None = None) -> WebPage:
    """``target`` (a stage or recording file, a run folder or a saved folder) as a notebook output: the view's page,
    the viewer from jsDelivr (``offline=True`` inlines it, 1.4 MB more), ``caption`` above its notes. Warns above
    ``OUTPUT_WARNING_BYTES``, which a notebook saves into its file with every display.

    Four things in the view are switched here, each in the page's notes: ``scale_bar`` and ``legend`` (on by default:
    they cost the page a few hundred bytes and answer "how big?" and "what is this arrow?"); ``apparatus_shapes`` (on:
    the motor and magnet of an apparatus, drawn for the view alone); ``focus`` and ``follow`` (off until a body is
    named: ``focus="umr"`` frames it at the start, ``follow="umr"`` keeps the camera on it through the recording)."""
    result = gltf(target, slow=slow, apparatus_shapes=apparatus_shapes)
    origin = origin_of_folder(target)
    held = page(result, cdn=not offline, caption=_stale_caption(origin, caption), scale_bar=scale_bar, legend=legend,
                focus=focus, follow=follow)
    size = len(held.encode("utf-8"))
    if size > OUTPUT_WARNING_BYTES:
        warn_at_script(
            f"the view of {target} is {size / 1e6:.1f} MB as a notebook output, and every display saves it into the "
            f"notebook's file: record more coarsely (stage_every=), bake meshes to fewer triangles (TriangleBudget), "
            f"{'leave the viewer to jsDelivr (offline=False), ' if offline else ''}or write a page instead "
            f"(mime visuals web {target})", UserWarning)
    return WebPage(held, result, origin=origin)


def held_by(path: Path) -> Path | None:
    """The saved or run folder ``path`` would be written inside, where it is inside one."""
    for folder in path.resolve().parents:
        if (folder / "mime.lock").is_file() or (folder / "experiment" / "mime.lock").is_file() \
                or (folder / "run.json").is_file() or (folder / "recording.usdc").is_file():
            return folder
    return None


def web(target: str | Path, out: str | Path | None = None, *, cdn: bool = False, caption: str | None = None,
        slow: float | None = None, glb: str | Path | None = None, scale_bar: bool = True, legend: bool = True,
        apparatus_shapes: bool = True, focus: str | None = None, follow: str | None = None) -> tuple[Path, GltfResult]:
    """Write the view of ``target`` as one page, ``out`` (``<name>.html`` in the working folder by default), the
    viewer inlined unless ``cdn``; and its GLB alone where ``glb`` names a file. A view is never written into a saved
    or a run folder (``visuals.web-out``). The switches are :func:`display`'s, and so is the mark: a page of a run this
    process wrote, before a replacement or a removal, says STALE in its notes."""
    result = gltf(target, slow=slow, apparatus_shapes=apparatus_shapes)
    caption = _stale_caption(origin_of_folder(target), caption)
    written = Path(out) if out is not None else Path(f"{result.name}.html")
    for path in (written, *((Path(glb),) if glb is not None else ())):
        folder = held_by(path)
        if folder is not None:
            raise ArgumentRefused([Refusal(
                "visuals.web-out", RefusalClass.USER,
                f"{path} is inside {folder}, a saved or run folder, which never holds presentation: a view is written "
                f"beside it, never into it",
                f"name a file outside it: mime visuals web {target} --out {result.name}.html")])
    text = page(result, cdn=cdn, caption=caption, scale_bar=scale_bar, legend=legend, focus=focus, follow=follow)
    written.write_text(text, encoding="utf-8")
    if glb is not None:
        Path(glb).write_bytes(result.glb)
    return written, result


__all__ = ["BAR_PIXELS", "CLOSEST", "CREDIT", "FOCUS_FACTOR", "OUTPUT_WARNING_BYTES", "VIEWER_CDN", "VIEWER_FILE", "VIEWER_SHA256", "VIEWER_SRI",
           "VIEWER_VERSION", "WebPage", "closest_radius", "display", "held_by", "page", "viewer_bytes", "web"]
