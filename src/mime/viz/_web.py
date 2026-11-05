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

from mime.build._refusals import ArgumentRefused, Refusal, RefusalClass
from mime.errors._frames import warn_at_script
from mime.viz._export import GltfResult, gltf

VIEWER_VERSION = "4.3.1"
VIEWER_FILE = "model-viewer.min.js"
#: The viewer's bytes, pinned: its sha256, and the integrity jsDelivr's copy is held to (sha384, as SRI writes it).
VIEWER_SHA256 = "283b0672384614b4847636c306fc93fe4b1fcadc76d668b4e47f0ca76bcf033b"
VIEWER_SRI = "sha384-cprcVQt7wbUl0xngF3PGP6yBB7n4/t+4AoAMG9biiMCGFiWOdzUH10Ie2COTqFNW"
VIEWER_CDN = f"https://cdn.jsdelivr.net/npm/@google/model-viewer@{VIEWER_VERSION}/dist/{VIEWER_FILE}"
#: Above this many bytes a notebook's output may be dropped (Jupyter's default limit on a kernel's output rate is a
#: million bytes a second): ``display`` warns, and says how to make it smaller.
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
model-viewer { display: block; width: 100%; flex: 1 1 auto; min-height: 160px; background: var(--panel); }
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
function tick() {
  if (time && !view.paused) time.value = String(view.currentTime);
  show();
  requestAnimationFrame(tick);
}
view.addEventListener("load", () => {
  document.body.dataset.loaded = "1";
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


def page(result: GltfResult, *, cdn: bool = False, caption: str | None = None) -> str:
    """The page of ``result``: the viewer inlined, or from jsDelivr with ``cdn``; ``caption`` above the notes."""
    glb = base64.b64encode(result.glb).decode("ascii")
    animated = result.samples > 1 and result.simulated_s > 0.0
    unit, factor, digits = _time_unit(result.simulated_s)
    if result.slow == 1.0:
        shown = " (shown at its own speed)"
    elif result.slow > 1.0:
        shown = f" (shown {result.slow:,.4g} times slower)"
    else:
        shown = f" (shown {1.0 / result.slow:,.4g} times faster)"
    config = json.dumps({"slow": result.slow, "unit": unit, "factor": factor, "digits": digits, "shown": shown,
                         "closest": closest_radius(result),
                         "reach": result.across_m / result.unit_m / 2.0})
    title = html.escape(result.name)
    parts = [
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">', f"<title>{title}</title>",
        f"<style>{_STYLE}</style>", _viewer_tag(cdn), "</head>", "<body>",
        f'<model-viewer id="view" src="data:model/gltf-binary;base64,{glb}" alt="{title}, drawn from its USD stage" '
        f'camera-controls {"autoplay animation-name=" + chr(34) + "run" + chr(34) if animated else ""} '
        'interaction-prompt="none" shadow-intensity="0" exposure="1"></model-viewer>',
        '<div class="grip" id="grip" role="separator" aria-orientation="horizontal" tabindex="0" '
        'aria-label="Resize the notes below the view" '
        'title="Drag down for more view, up for more notes; double-click to fold or unfold the notes"></div>']
    if animated:
        options = "".join(f'<option value="{s:g}"{" selected" if s == 1.0 else ""}>{s:g}x</option>' for s in SPEEDS)
        start = f"t = {0:.{digits}f} {unit} simulated{shown}"
        parts.append(
            '<div class="controls">'
            '<button id="play" type="button">Pause</button>'
            f'<label>speed <select id="speed">{options}</select></label>'
            f'<input id="time" type="range" min="0" max="{result.simulated_s * result.slow:.6g}" step="any" value="0" '
            'aria-label="the recording\'s time">'
            f'<span id="clock">{html.escape(start)}</span></div>')
    notes = [f"<p>{html.escape(caption)}</p>"] if caption else []
    span = (f"{result.samples} samples over {result.simulated_s * factor:.{digits}f} {unit} simulated"
            if animated else "a scene with no recording")
    notes.append(f'<p class="muted">{html.escape(result.name)}: about {html.escape(result.size_text())} across '
                 f"over its whole motion; {html.escape(span)}.</p>")
    notes.append('<p class="muted">Scroll to zoom toward the pointer, as close as the view can draw (about a hundredth of the scene); click a part to orbit around it; right-drag or two fingers to pan.</p>')
    notes += [f'<p class="muted">{html.escape(line)}</p>' for line in result.lines()]
    where = ("The viewer is loaded from jsDelivr, held to its hash; the scene is inside this page." if cdn else
             "The viewer and the scene are both inside this page: it needs no network.")
    notes.append(f'<p class="muted">{CREDIT} {where}</p>')
    parts += ['<div class="notes" id="notes"><div class="notes-body">', *notes, "</div></div>",
              f'<script type="module">{_SCRIPT.replace("CONFIG", config)}</script>', "</body>", "</html>"]
    return "\n".join(parts) + "\n"


@dataclass(frozen=True)
class WebPage:
    """A view as a notebook shows it: the page in an ``<iframe srcdoc>``. The last expression of a cell shows it;
    :meth:`save` writes the page as a file."""

    html: str
    result: GltfResult
    height: int = 720

    @override
    def __repr__(self) -> str:
        """One line, not the page: Jupyter shows ``text/plain`` beside the HTML, and the page is megabytes."""
        return (f"WebPage({self.result.name!r}, {len(self.html.encode('utf-8')) / 1e3:.0f} kB): the last expression of a "
                f"cell shows it; .save('view.html') writes it as a page to open in a browser")

    def _repr_html_(self) -> str:
        return (f'<iframe srcdoc="{html.escape(self.html, quote=True)}" style="width: 100%; height: {self.height}px; '
                f'border: 0" title="{html.escape(self.result.name)}"></iframe>')

    def save(self, path: str | Path) -> Path:
        written = Path(path)
        written.write_text(self.html, encoding="utf-8")
        return written


def display(target: str | Path, *, offline: bool = False, caption: str | None = None,
            slow: float | None = None) -> WebPage:
    """``target`` (a stage or recording file, a run folder or a saved folder) as a notebook output: the view's page,
    the viewer from jsDelivr (``offline=True`` inlines it, 1.4 MB more), ``caption`` above its notes. Warns above
    ``OUTPUT_WARNING_BYTES``, which a notebook may drop."""
    result = gltf(target, slow=slow)
    held = page(result, cdn=not offline, caption=caption)
    size = len(held.encode("utf-8"))
    if size > OUTPUT_WARNING_BYTES:
        warn_at_script(
            f"the view of {target} is {size / 1e6:.1f} MB as a notebook output, which Jupyter may drop at its default "
            f"output rate: record more coarsely (stage_every=), bake meshes to fewer triangles (TriangleBudget), "
            f"{'leave the viewer to jsDelivr (offline=False), ' if offline else ''}or write a page instead "
            f"(mime visuals web {target})", UserWarning)
    return WebPage(held, result)


def _held_by(path: Path) -> Path | None:
    """The saved or run folder ``path`` would be written inside, where it is inside one."""
    for folder in path.resolve().parents:
        if (folder / "mime.lock").is_file() or (folder / "experiment" / "mime.lock").is_file() \
                or (folder / "run.json").is_file() or (folder / "recording.usdc").is_file():
            return folder
    return None


def web(target: str | Path, out: str | Path | None = None, *, cdn: bool = False, caption: str | None = None,
        slow: float | None = None, glb: str | Path | None = None) -> tuple[Path, GltfResult]:
    """Write the view of ``target`` as one page, ``out`` (``<name>.html`` in the working folder by default), the
    viewer inlined unless ``cdn``; and its GLB alone where ``glb`` names a file. A view is never written into a saved
    or a run folder (``visuals.web-out``)."""
    result = gltf(target, slow=slow)
    written = Path(out) if out is not None else Path(f"{result.name}.html")
    for path in (written, *((Path(glb),) if glb is not None else ())):
        folder = _held_by(path)
        if folder is not None:
            raise ArgumentRefused([Refusal(
                "visuals.web-out", RefusalClass.USER,
                f"{path} is inside {folder}, a saved or run folder, which never holds presentation: a view is written "
                f"beside it, never into it",
                f"name a file outside it: mime visuals web {target} --out {result.name}.html")])
    written.write_text(page(result, cdn=cdn, caption=caption), encoding="utf-8")
    if glb is not None:
        Path(glb).write_bytes(result.glb)
    return written, result


__all__ = ["CLOSEST", "CREDIT", "OUTPUT_WARNING_BYTES", "VIEWER_CDN", "VIEWER_FILE", "VIEWER_SHA256", "VIEWER_SRI",
           "VIEWER_VERSION", "WebPage", "closest_radius", "display", "page", "viewer_bytes", "web"]
