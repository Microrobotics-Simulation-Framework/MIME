"""The compiled graph as a page you can pan and zoom: one self-contained HTML file holding the Mermaid text and the
library that draws it, and the notebook fragment that shows that page in an ``<iframe srcdoc>``.

A notebook that put the diagram straight into its own output page left the drawing to the front end's script handling,
and Colab showed the text. A page in an ``<iframe srcdoc>`` is a document of its own, as the 3-D viewer's is
(:class:`mime.viz._web.WebPage`): its scripts run in order as in a browser, and nothing of the front end (a module
loader it defines, the order it inserts scripts in) is in their way.

**Where the library comes from.** Mermaid 11.17.2's single-file build (a classic script: ``integrity`` holds every byte it
runs, which the ES-module build's further files would escape), either inlined from the copy shipped in the wheel
(``_mermaid_lib/mermaid.min.js``, with the licence of every package its bundle can hold, pinned by sha256) or from
jsDelivr, held to its sha384 (``cdn=True``). A page written to a file is inlined by default (3.6 MB, 4.8 MB as the data
URI that carries it) and needs no network. A notebook's output is the page with the library from jsDelivr by default, as
the 3-D view's is (:mod:`mime.viz._web`), and for two reasons that hold: an inlined page is a 4.8 MB output, which a
notebook saves into its own file with every display, so a notebook that shows a few graphs weighs tens of megabytes; and
one word means one thing across both views (``offline=True`` inlines, ``offline=False`` loads from jsDelivr). It is not
because Jupyter would drop the output: a server's output rate limit counts only a kernel's stream output (``print``),
not a displayed value, so that is not claimed.

**Pan and zoom** are a short script of this page's own rather than a second library: drag to pan, wheel or pinch to zoom
toward the pointer, buttons for zoom in, zoom out, fit and 1:1, and keys. It moves one element by a CSS transform, so
the diagram stays vector at every zoom. The page opens fitted when the whole graph fits the frame at a readable size,
and otherwise at that readable size from the graph's top left corner, so a large graph is read by panning, not shrunk
to nothing; Fit shows all of it.

**The frame's height** is a field of the page object (:class:`mime.experiment._mermaid.GraphPage`, 720 pixels as the 3-D
view's), not computed from the graph: the graph's size in pixels is known only after the browser has laid it out, and a
height guessed from the number of nodes would be wrong for a wide graph and for a tall one.

**What the page can load.** A content security policy in the page (:data:`POLICY`) lets it run its own script and the
library, and nothing else: no image, stylesheet, font, frame or request from the graph's text. A label a person wrote by
hand (``Mermaid(str)``) can hold an ``<img src=...>`` that Mermaid's own sanitiser keeps, and without the policy an
offline page would fetch that URL.

**Determinism.** The page is built from constants and the graph's text and holds no random id, no time and no counter;
Mermaid is given a fixed id for its drawing. The same text gives the same page byte for byte."""

from __future__ import annotations

import base64
import hashlib
import html
from importlib.resources import files
from urllib.parse import urlsplit

#: The library, pinned. The sha256 is of the shipped file; the sha384 (as SRI writes it) is what jsDelivr's copy is held to,
#: and to check a bump: ``curl -sL <MERMAID_CDN> | openssl dgst -sha384 -binary | openssl base64 -A``.
MERMAID_VERSION = "11.17.2"
MERMAID_FILE = "mermaid.min.js"
MERMAID_SHA256 = "581ed7d74bd9048d0e3a91363927d72ef22942d7722546b27f7cc29e35390eb8"
MERMAID_SRI = "sha384-EOXBFmc3gx5mb+vn0vPvvGqACToJD24hhacX5Yx+8NUUQrHIle/Qi5Bg9o3zKwW2"
MERMAID_CDN = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/{MERMAID_FILE}"
#: Where to draw Mermaid text that a front end shows as text.
RENDERER = "https://mermaid.live"
#: The frame's height in a notebook, in pixels: the 3-D view's (``WebPage.height``).
FRAME_HEIGHT = 720
#: Above this many bytes a notebook's output is a large share of the notebook's own file (every display is saved in it): a
#: display warns. The same size as the 3-D view's, which is released. Not a limit of Jupyter's: its output rate limit
#: counts a kernel's stream output only, so a displayed value of any size is not dropped by it.
OUTPUT_WARNING_BYTES = 1_000_000

CREDIT = ('Drawn with <a href="https://mermaid.js.org/">Mermaid</a> 11.17.2 (MIT) and the packages it bundles (MIT, ISC, '
          "BSD-3-Clause, Apache-2.0, Unlicense), whose licences ship with MIME beside the library.")

#: What the notebook shows under the frame: read where the frame is blank (a front end that runs no script, such as a
#: static preview of the notebook, draws nothing of the page) and under the diagram where it is drawn.
NOTE = ("Compiled graph: drag to pan, scroll or pinch to zoom. If no diagram is drawn here, your front end does not run "
        f"scripts: print this object for the Mermaid text, or paste that into {RENDERER}.")


def library_bytes() -> bytes:
    """The shipped library's bytes, held to their pinned sha256."""
    data = files("mime.experiment").joinpath("_mermaid_lib", MERMAID_FILE).read_bytes()
    if hashlib.sha256(data).hexdigest() != MERMAID_SHA256:
        raise ValueError(f"mime/experiment/_mermaid_lib/{MERMAID_FILE} is not the library MIME pins")
    return data


def library_tag(cdn: bool) -> str:
    """The script element that loads the library: a data URI of the shipped bytes, or jsDelivr's file held to its hash."""
    if cdn:
        return f'<script src="{MERMAID_CDN}" integrity="{MERMAID_SRI}" crossorigin="anonymous"></script>'
    encoded = base64.b64encode(library_bytes()).decode("ascii")
    return f'<script src="data:text/javascript;base64,{encoded}"></script>'


STYLE = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --panel: #f3f4f6; --edge: #d1d9e0; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --panel: #161b22; --edge: #3d444d; }
}
:root[data-theme="dark"] { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --panel: #161b22; --edge: #3d444d; }
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.5 system-ui, -apple-system, sans-serif;
  display: flex; flex-direction: column; overflow: hidden; }
.bar { flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; padding: 6px 12px;
  background: var(--panel); border-bottom: 1px solid var(--edge); }
.bar button { font: inherit; color: var(--fg); background: var(--bg); border: 1px solid var(--edge); border-radius: 6px;
  padding: 1px 10px; min-width: 32px; cursor: pointer; }
.bar button:hover { border-color: var(--muted); }
.bar button:focus-visible, #stage:focus-visible { outline: 2px solid var(--muted); outline-offset: -2px; }
#zoom { font-variant-numeric: tabular-nums; min-width: 4.5em; }
.muted { color: var(--muted); font-size: 12px; }
#hint { margin-left: auto; }
#stage { position: relative; flex: 1 1 auto; min-height: 0; overflow: hidden; cursor: grab; touch-action: none;
  user-select: none; -webkit-user-select: none; background: var(--bg); }
#stage.dragging { cursor: grabbing; }
#world { position: absolute; left: 0; top: 0; transform-origin: 0 0; }
#world svg { display: block; max-width: none; }
body[data-state="drawing"] pre.mermaid { visibility: hidden; }
body[data-state="text"] #stage { overflow: auto; cursor: auto; touch-action: auto; user-select: text; -webkit-user-select: text; }
body[data-state="text"] #world { position: static; }
body[data-state="text"] .bar { display: none; }
.notes { flex: 0 0 auto; padding: 4px 12px; }
.notes p { margin: 4px 0; }
.stale { color: #b3261e; font-weight: bold; }
pre.mermaid { margin: 12px; white-space: pre-wrap; font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }
#status { flex: 0 0 auto; margin: 0; padding: 4px 12px; border-top: 1px solid var(--edge); background: var(--panel); }
#status[data-kind="error"] { color: #b3261e; }
.credit { flex: 0 0 auto; margin: 0; padding: 4px 12px; border-top: 1px solid var(--edge); }
a { color: inherit; }
"""

#: The page's script. ``mermaid`` is the library's global; the text is read back from the element that holds it. Pan and
#: zoom move ``#world`` by ``translate(x, y) scale(k)``; ``data-state`` and ``data-scale`` on ``<body>`` say what the page
#: is doing, for a reader of the DOM (the manual proof reads them).
SCRIPT = """
(async function () {
  const READABLE = 0.75, MARGIN = 16, MIN_VISIBLE = 64, MAX_SCALE = 8;
  const body = document.body, stage = document.getElementById("stage"), world = document.getElementById("world");
  const status = document.getElementById("status"), zoomLabel = document.getElementById("zoom");
  const source = document.getElementById("source");
  body.dataset.state = "drawing";
  const say = function (text, kind) { status.textContent = text; status.dataset.kind = kind || "note"; status.hidden = !text; };
  const fail = function (text) { body.dataset.state = "text"; say(text, "error"); };
  if (typeof mermaid === "undefined") {
    fail("The Mermaid library did not load (no network, or the file was not the one MIME pins), so the graph is shown as " +
         "its text. A page with the library inlined needs no network: diagram.display(offline=True) in a notebook, or diagram.save(path).");
    return;
  }
  // The diagram's theme is chosen once, at load: the page's own colours follow a change of the browser's setting, the drawing does not.
  const forced = document.documentElement.dataset.theme;
  const dark = forced ? forced === "dark" : window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  mermaid.initialize({
    startOnLoad: false, securityLevel: "strict", theme: dark ? "dark" : "default", deterministicIds: true,
    deterministicIDSeed: "mime", maxTextSize: 50000000, maxEdges: 100000, flowchart: {useMaxWidth: false, htmlLabels: true}});
  let drawn;
  try {
    drawn = await mermaid.render("mime-graph", source.textContent);
  } catch (error) {
    fail("Mermaid could not draw this graph (" + (error && error.message ? error.message : error) + "), so it is shown as its text.");
    return;
  }
  world.innerHTML = drawn.svg;
  const svg = world.querySelector("svg");
  const box = svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width ? svg.viewBox.baseVal : svg.getBBox();
  const width = Math.ceil(box.width), height = Math.ceil(box.height);
  svg.removeAttribute("style");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  world.style.width = width + "px";
  world.style.height = height + "px";
  const view = {k: 1, x: 0, y: 0};
  let fitted = true, minScale = 0.05;
  const size = function () { const r = stage.getBoundingClientRect(); return {w: r.width, h: r.height}; };
  const fitScale = function () { const s = size(); return Math.min((s.w - 2 * MARGIN) / width, (s.h - 2 * MARGIN) / height); };
  const clamp = function () {
    const s = size();
    view.k = Math.min(MAX_SCALE, Math.max(minScale, view.k));
    view.x = Math.min(s.w - MIN_VISIBLE, Math.max(MIN_VISIBLE - width * view.k, view.x));
    view.y = Math.min(s.h - MIN_VISIBLE, Math.max(MIN_VISIBLE - height * view.k, view.y));
  };
  const apply = function () {
    clamp();
    world.style.transform = "translate(" + view.x.toFixed(2) + "px, " + view.y.toFixed(2) + "px) scale(" + view.k.toFixed(5) + ")";
    zoomLabel.textContent = Math.round(view.k * 100) + "%";
    body.dataset.scale = view.k.toFixed(5);
  };
  const zoomAt = function (px, py, factor) {
    fitted = false;
    const k = Math.min(MAX_SCALE, Math.max(minScale, view.k * factor));
    view.x = px - (px - view.x) * (k / view.k);
    view.y = py - (py - view.y) * (k / view.k);
    view.k = k;
    apply();
  };
  const centre = function () { const s = size(); return {x: s.w / 2, y: s.h / 2}; };
  const fit = function () {
    const s = size(), k = fitScale();
    minScale = Math.min(0.05, k / 2);
    view.k = Math.min(k, 1);
    view.x = (s.w - width * view.k) / 2;
    view.y = (s.h - height * view.k) / 2;
    fitted = true;
    apply();
    say("");
  };
  const open = function () {
    const s = size(), k = fitScale();
    minScale = Math.min(0.05, k / 2);
    if (k >= READABLE) { fit(); return; }
    view.k = READABLE;
    view.x = MARGIN;
    view.y = Math.max(MARGIN, (s.h - height * view.k) / 2);
    fitted = false;
    apply();
    say("This graph is larger than the frame at a readable size, so it opens at " + Math.round(READABLE * 100) +
        "% from its top left: drag to pan, Fit shows all of it.");
  };
  const reset = function () {
    const c = centre();
    zoomAt(c.x, c.y, 1 / view.k);
  };
  document.getElementById("zoom-in").addEventListener("click", function () { const c = centre(); zoomAt(c.x, c.y, 1.25); });
  document.getElementById("zoom-out").addEventListener("click", function () { const c = centre(); zoomAt(c.x, c.y, 0.8); });
  document.getElementById("zoom-fit").addEventListener("click", fit);
  document.getElementById("zoom-reset").addEventListener("click", reset);
  const full = document.getElementById("zoom-full");
  if (document.fullscreenEnabled) {
    full.hidden = false;
    full.addEventListener("click", function () {
      if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen();
    });
  }
  stage.addEventListener("wheel", function (event) {
    event.preventDefault();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1;
    const r = stage.getBoundingClientRect();
    zoomAt(event.clientX - r.left, event.clientY - r.top, Math.exp(-event.deltaY * unit * (event.ctrlKey ? 0.01 : 0.0015)));
  }, {passive: false});
  const pointers = new Map();
  let pinch = null;
  const spread = function () {
    const p = Array.from(pointers.values());
    return {d: Math.hypot(p[0].x - p[1].x, p[0].y - p[1].y), x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2};
  };
  stage.addEventListener("pointerdown", function (event) {
    stage.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, {x: event.clientX, y: event.clientY});
    stage.classList.add("dragging");
    pinch = pointers.size === 2 ? spread() : null;
    stage.focus({preventScroll: true});
  });
  stage.addEventListener("pointermove", function (event) {
    const held = pointers.get(event.pointerId);
    if (!held) return;
    const before = {x: held.x, y: held.y};
    held.x = event.clientX;
    held.y = event.clientY;
    fitted = false;
    if (pointers.size === 1) {
      view.x += held.x - before.x;
      view.y += held.y - before.y;
      apply();
    } else if (pointers.size === 2 && pinch) {
      const now = spread(), r = stage.getBoundingClientRect();
      view.x += now.x - pinch.x;
      view.y += now.y - pinch.y;
      zoomAt(now.x - r.left, now.y - r.top, now.d / pinch.d);
      pinch = now;
    }
  });
  const release = function (event) {
    pointers.delete(event.pointerId);
    pinch = pointers.size === 2 ? spread() : null;
    if (!pointers.size) stage.classList.remove("dragging");
  };
  stage.addEventListener("pointerup", release);
  stage.addEventListener("pointercancel", release);
  stage.addEventListener("keydown", function (event) {
    const step = 60, c = centre();
    const moves = {ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step]};
    if (moves[event.key]) { fitted = false; view.x += moves[event.key][0]; view.y += moves[event.key][1]; apply(); }
    else if (event.key === "+" || event.key === "=") zoomAt(c.x, c.y, 1.25);
    else if (event.key === "-" || event.key === "_") zoomAt(c.x, c.y, 0.8);
    else if (event.key === "0") fit();
    else if (event.key === "1") reset();
    else return;
    event.preventDefault();
  });
  new ResizeObserver(function () { if (fitted) fit(); else apply(); }).observe(stage);
  open();
  body.dataset.state = "drawn";
  body.dataset.size = width + "x" + height;
})();
"""


def script_hash() -> str:
    """The page's own script as a content security policy names it: ``'sha256-...'`` of the text between its tags."""
    return "'sha256-" + base64.b64encode(hashlib.sha256(SCRIPT.encode("utf-8")).digest()).decode("ascii") + "'"


def policy(cdn: bool) -> str:
    """What the page may load, as a content security policy: its own script (held by hash, so no other inline script runs,
    and no ``on...`` attribute), the library (a data URI inlined, or jsDelivr's origin), its own inline style (Mermaid
    sets styles on what it draws), and no image, no font, no frame and no request. A label a person wrote
    by hand can hold an ``<img src=...>`` that Mermaid's sanitiser keeps; this is what stops the page fetching it."""
    origin = "{0.scheme}://{0.netloc}".format(urlsplit(MERMAID_CDN))
    library = origin if cdn else "data:"
    return (f"default-src 'none'; script-src {script_hash()} {library}; style-src 'unsafe-inline'; base-uri 'none'; "
            "form-action 'none'")


def page(text: str, *, cdn: bool = False, caption: str | None = None, stale: str | None = None) -> str:
    """The page of the Mermaid ``text``: the library inlined, or from jsDelivr with ``cdn``. The text is in the page once,
    in the element the library draws, so where nothing draws (no script, no library) it is shown. ``caption`` is a line
    above the diagram; ``stale`` (why the graph no longer matches its experiment) is a line above that, as the 3-D page's
    notes lead with it."""
    notes = ([f'<p class="stale" role="alert">STALE: {html.escape(stale, quote=False)}</p>'] if stale else []) \
        + ([f"<p>{html.escape(caption, quote=False)}</p>"] if caption else [])
    parts = [
        "<!doctype html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        f'<meta http-equiv="Content-Security-Policy" content="{html.escape(policy(cdn), quote=True)}">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">', "<title>Compiled graph</title>",
        f"<style>{STYLE}</style>", "</head>", "<body>",
        *([f'<div class="notes">{"".join(notes)}</div>'] if notes else []),
        '<div class="bar" role="toolbar" aria-label="Pan and zoom">'
        '<button id="zoom-in" type="button" aria-label="Zoom in" title="Zoom in (+)">+</button>'
        '<button id="zoom-out" type="button" aria-label="Zoom out" title="Zoom out (-)">−</button>'
        '<button id="zoom-fit" type="button" title="Fit the whole graph in the frame (0)">Fit</button>'
        '<button id="zoom-reset" type="button" title="One graph pixel to one screen pixel (1)">1:1</button>'
        '<button id="zoom-full" type="button" title="Full screen" hidden>Full screen</button>'
        '<span id="zoom" class="muted" aria-live="polite"></span>'
        '<span id="hint" class="muted">Drag to pan, scroll or pinch to zoom; keys: arrows, + - 0 1</span></div>',
        '<div id="stage" tabindex="0" role="group" aria-label="The compiled graph; drag to pan, scroll to zoom">'
        f'<div id="world"><pre class="mermaid" id="source">{html.escape(text, quote=False)}</pre></div></div>',
        '<p id="status" class="muted" role="status" hidden></p>',
        f'<p class="credit muted">{CREDIT} '
        + ("The library is loaded from jsDelivr, held to its hash." if cdn else
           "The library is inside this page: it needs no network.") + "</p>",
        library_tag(cdn), f"<script>{SCRIPT}</script>", "</body>", "</html>"]
    return "\n".join(parts) + "\n"


def frame(held: str, *, height: int = FRAME_HEIGHT, stale: str | None = None) -> str:
    """The notebook's fragment for the page ``held``: the page in an ``<iframe srcdoc>`` as the 3-D view's is (full width,
    ``height`` pixels, no border), with the red ``STALE`` line above it where the graph is stale, and the note under it.
    ``sandbox="allow-scripts"`` runs the page's script and no more: the page is a document of its own with an opaque
    origin, and no access to the notebook around it (the label text of a hand-built graph is a person's)."""
    banner = f'<p style="color: #b3261e; font-weight: bold">STALE: {html.escape(stale)}</p>\n' if stale else ""
    return (
        banner + f'<iframe srcdoc="{html.escape(held, quote=True)}" sandbox="allow-scripts" allowfullscreen '
        f'style="width: 100%; height: {height}px; border: 0" title="The compiled graph"></iframe>\n'
        f'<p style="margin: 2px 0 8px; font: 12px/1.4 system-ui, sans-serif; opacity: 0.75"><em>{html.escape(NOTE, quote=False)}</em></p>\n')


__all__ = ["CREDIT", "FRAME_HEIGHT", "MERMAID_CDN", "MERMAID_FILE", "MERMAID_SHA256", "MERMAID_SRI", "MERMAID_VERSION", "NOTE",
           "OUTPUT_WARNING_BYTES", "RENDERER", "frame", "library_bytes", "library_tag", "page", "policy", "script_hash"]
