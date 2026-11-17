"""The view as one web page (``mime.viz.page``), as the object a notebook shows (``mime.viz.web_page``) and as what
a notebook is told to show (``mime.viz.display``):

- the page holds the GLB as a data URI, byte for byte, and the viewer either inlined, its bytes the pinned file's
  (sha256 below, and the sha384 that jsDelivr's copy is held to), or from jsDelivr with that integrity;
- the offline page holds no ``http(s)`` URL but in its credit line, and the jsDelivr page none but the viewer's and the
  credit's (the control: a URL planted outside the credit is found);
- ``web_page`` returns the page inside an ``<iframe srcdoc>`` that unescapes to it, from jsDelivr unless ``offline``,
  and warns above a million bytes (the control: the ball's own page, far smaller, does not);
- ``display`` shows the page itself, wherever in a cell it is called (in an ``if``, a loop, a function), exactly once
  when it is also a cell's last line, and returns ``None``; outside IPython it is refused
  (``visuals.display-outside-ipython``). IPython is not a test dependency, so the notebook is simulated where MIME meets
  it, as ``tests/cli/test_notebook_display.py`` does: ``get_ipython`` in the builtins and a stand-in for
  ``IPython.display.display`` that records a ``display_data`` bundle (``text/plain`` and ``text/html``, from the
  object's ``_repr_html_``) as IPython does, and a cell run as IPython runs one (its statements, then its last
  expression shown unless ``None``). The control is the old behaviour, the call that returns the page: inside an
  ``if`` it shows nothing;
- the camera may come within a millionth of the scene's diagonal, the wheel zooming by a factor toward the pointer,
  so a body far smaller than its scene can be approached (the control: a scene with no extent keeps the viewer's);
- a page is never written into a saved or a run folder (``visuals.web-out``).

That the page draws, with no network, is shown by hand in a browser: ``scripts/offline_view_proof.py``, which is not a
test (CI has no WebGL)."""

import base64
import hashlib
import builtins
import html
import re
import threading
import warnings

import pytest
from gltf_cases import ball, recorded, with_pose
from notebook_cells import Cells, install

import mime.viz._web as web_module
from mime.build import ArgumentRefused
from mime.viz import display, gltf, page, web, web_page
from mime.viz._web import CLOSEST, CREDIT, VIEWER_CDN, VIEWER_SRI, closest_radius, viewer_bytes

#: The inlined viewer: model-viewer 4.3.1's dist/model-viewer.min.js, 1,068,903 bytes (jsDelivr's, unpkg's and the npm
#: tarball's alike).
VIEWER_SHA256 = "283b0672384614b4847636c306fc93fe4b1fcadc76d668b4e47f0ca76bcf033b"
URL = re.compile(r"https?://[^\s\"'<>)]+")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ball") / "run"
    recorded(ball().build(), folder, 5, lambda k, start: with_pose(start, "ball", [0, 0, -1e-6 * k, 1, 0, 0, 0]))
    return folder


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


def data_uri(text: str, kind: str) -> bytes:
    (encoded,) = re.findall(rf'src="data:{re.escape(kind)};base64,([A-Za-z0-9+/=]+)"', text)
    return base64.b64decode(encoded)


def urls_outside_the_credit(text: str) -> list[str]:
    held = re.sub(r'src="data:[^"]+"', "", text).replace(CREDIT, "")
    return URL.findall(held)


def test_the_inlined_viewer_is_the_pinned_file():
    data = viewer_bytes()
    assert len(data) == 1_068_903 and hashlib.sha256(data).hexdigest() == VIEWER_SHA256
    assert VIEWER_SRI == "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()


def test_the_offline_page_holds_the_glb_and_the_viewer_and_no_url_but_its_credits(run):
    result = gltf(run)
    text = page(result)
    assert data_uri(text, "model/gltf-binary") == result.glb
    assert data_uri(text, "text/javascript") == viewer_bytes()
    assert urls_outside_the_credit(text) == []
    assert URL.findall(CREDIT), "the credit's links are its own"


def test_the_jsdelivr_page_names_the_viewer_with_its_integrity(run):
    text = page(gltf(run), cdn=True)
    assert f'src="{VIEWER_CDN}" integrity="{VIEWER_SRI}" crossorigin="anonymous"' in text
    assert urls_outside_the_credit(text) == [VIEWER_CDN] and "text/javascript;base64" not in text


def test_control_a_url_planted_outside_the_credit_is_found(run):
    text = page(gltf(run)).replace("<body>", '<body><img src="https://example.org/x.png">')
    assert urls_outside_the_credit(text) == ["https://example.org/x.png"]


def test_the_page_says_the_caption_the_size_and_the_slowing(run):
    result = gltf(run)
    text = page(result, caption="a ball <in> water")
    assert "a ball &lt;in&gt; water" in text and f"about {result.size_text()} across" in text
    assert f"shown {result.slow:,.4g} times slower" in text


def test_web_page_is_the_page_in_an_iframe_from_jsdelivr_unless_offline(run):
    shown = web_page(run)
    (srcdoc,) = re.findall(r'srcdoc="([^"]*)"', shown._repr_html_())
    assert html.unescape(srcdoc) == shown.html == page(shown.result, cdn=True)
    with pytest.warns(UserWarning, match="leave the viewer to jsDelivr"):
        assert "text/javascript;base64" in web_page(run, offline=True).html


def test_web_page_warns_above_a_million_bytes(run, monkeypatch):
    result = gltf(run)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        web_page(run)  # the control: the ball's page is far smaller
    monkeypatch.setattr(web_module, "gltf", lambda target, slow=None, **_: type(result)(
        **{**result.__dict__, "glb": result.glb + b"\0" * 900_000}))
    with pytest.warns(UserWarning, match="every display saves it") as caught:
        web_page(run)
    assert [w.filename for w in caught] == [__file__], "at the script's line, not web_page's"


#: What Jupyter holds of a cell's value beside its HTML is its ``text/plain``, ``repr``: one line, not the page.
PLAIN_TEXT_BYTES = 300


def test_the_plain_text_of_a_web_page_is_one_short_line_saying_what_it_is_and_how_to_open_it(run, monkeypatch):
    shown = web_page(run)
    plain = repr(shown)
    assert len(plain.encode("utf-8")) <= PLAIN_TEXT_BYTES and "\n" not in plain
    assert shown.result.name in plain and f"{len(shown.html.encode('utf-8')) / 1e3:.0f} kB" in plain
    assert "browser" in plain
    assert shown.html not in plain and "<iframe" not in plain and "<html" not in plain
    assert shown._repr_html_().startswith("<iframe srcdoc=") and html.unescape(shown._repr_html_()).count(shown.html) == 1
    assert plain.startswith("<WebPage ") and plain.endswith(">"), "a description, not a constructor call to evaluate"
    assert "last expression" in plain and ".save(" in plain, "how a notebook shows it, and how anywhere else does"


def test_control_the_dataclass_repr_it_replaces_held_the_whole_page(run):
    shown = web_page(run)
    generated = type(shown).__dataclass_fields__  # the fields the generated repr would print
    assert "html" in generated
    plain = f"WebPage(html={shown.html!r}, result=..., height={shown.height})"
    assert len(plain.encode("utf-8")) > 100 * PLAIN_TEXT_BYTES, "so a bound of 300 bytes fails on it"


def test_a_page_is_never_written_into_a_saved_or_run_folder(run, tmp_path):
    with pytest.raises(ArgumentRefused, match="visuals.web-out"):
        web(run, run / "view.html")
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / "mime.lock").write_text("{}")
    with pytest.raises(ArgumentRefused, match="visuals.web-out"):
        web(run, tmp_path / "out.html", glb=saved / "deep" / "scene.glb")
    written, result = web(run, tmp_path / "out.html", glb=tmp_path / "scene.glb")
    assert written.read_text(encoding="utf-8") == page(result)
    assert (tmp_path / "scene.glb").read_bytes() == result.glb


def config_of(text: str) -> dict:
    import json

    (held,) = re.findall(r"const config = (\{.*?\});", text)
    return json.loads(held)


def test_the_camera_may_come_within_a_millionth_of_the_scene(run):
    result = gltf(run)
    text = page(result)
    closest = config_of(text)["closest"]
    assert closest == pytest.approx(result.across_m / result.unit_m * CLOSEST) and closest > 0.0
    assert config_of(text)["reach"] == pytest.approx(result.across_m / result.unit_m / 2.0)
    assert "2 * framed / 1000" in text  # on load, never nearer than the viewer's own near plane
    assert "view.minCameraOrbit" in text and "positionAndNormalFromPoint" in text  # the wheel zooms toward the pointer
    assert "Scroll to zoom toward the pointer, as close as the view can draw" in text


def test_control_a_scene_with_no_extent_keeps_the_viewers_own_limit(run):
    import dataclasses

    flat = dataclasses.replace(gltf(run), across_m=0.0)
    assert closest_radius(flat) is None and config_of(page(flat))["closest"] is None
    (tag,) = re.findall(r"<model-viewer [^>]*>", page(flat))
    assert "min-camera-orbit" not in tag


def test_a_handle_under_the_view_folds_the_notes_down_to_the_controls(run):
    text = page(gltf(run))
    order = [text.index(marker) for marker in ("<model-viewer ", 'id="grip"', 'class="controls"', 'id="notes"')]
    assert order == sorted(order)  # the view, the handle, the controls, then the notes the handle folds
    assert 'role="separator"' in text and "notesHeight(0)" in text and '"dblclick"' in text


# -- ``display`` shows, wherever it is called ----------------------------------------------------------------------------


@pytest.fixture
def notebook(monkeypatch) -> Cells:
    return install(monkeypatch)


def html_outputs(cells: Cells) -> list[str]:
    return cells.html


def test_display_inside_an_if_shows_the_view_as_display_data_with_html(run, notebook):
    notebook.namespace.update(display=display, folder=str(run), wanted=True)
    notebook.run("if wanted:\n    display(folder)\n")
    (shown,) = html_outputs(notebook)
    assert shown.startswith("<iframe srcdoc=") and web_page(run)._repr_html_() == shown, "the page the view is"
    assert len(notebook.outputs) == 1 and "WebPage" in notebook.outputs[0]["text/plain"]


def test_display_in_a_loop_and_a_function_shows_each_call(run, notebook):
    notebook.namespace.update(display=display, folder=str(run))
    notebook.run("def show(caption):\n    display(folder, caption=caption)\nfor caption in ('one', 'two'):\n    show(caption)\n")
    shown = html_outputs(notebook)
    assert len(shown) == 2 and "one" in html.unescape(shown[0]) and "two" in html.unescape(shown[1])


def test_control_the_call_that_returns_the_page_shows_nothing_inside_an_if(run, notebook):
    """What ``display`` did before: it returned the page, so inside an ``if`` the cell shows nothing and raises
    nothing. The harness sees that, which is what makes the passing tests above evidence."""
    notebook.namespace.update(web_page=web_page, folder=str(run), wanted=True)
    notebook.run("if wanted:\n    web_page(folder)\n")
    assert notebook.outputs == []
    notebook.run("web_page(folder)")
    assert len(html_outputs(notebook)) == 1, "as the cell's last line it is shown, by the cell's rule"


def test_display_as_a_cells_last_line_shows_exactly_once(run, notebook):
    notebook.namespace.update(display=display, folder=str(run))
    notebook.run("display(folder)")
    assert len(notebook.outputs) == 1 and len(html_outputs(notebook)) == 1
    assert display(run) is None


def test_display_after_its_cell_has_returned_still_shows_from_a_thread_and_from_a_post_execute_hook(run, notebook):
    """IPython puts ``get_ipython`` among the builtins only while a cell runs, so a display called later (a thread the
    cell started, a ``post_execute`` hook, a widget callback) finds the shell by ``IPython.get_ipython()`` instead.
    The control, in the same test: the harness has no builtin then, so a test of the builtin alone would refuse."""
    notebook.namespace.update(display=display, folder=str(run), threading=threading, go=threading.Event())
    notebook.run("shown = threading.Thread(target=lambda: (go.wait(5), display(folder)))\nshown.start()")
    assert not hasattr(builtins, "get_ipython"), "the cell has returned: the builtin is gone, as in IPython"
    assert notebook.outputs == []
    notebook.namespace["go"].set()
    notebook.namespace["shown"].join(10)
    assert len(html_outputs(notebook)) == 1
    notebook.post_execute.append(lambda: display(run, caption="hook"))
    notebook.run("pass")
    assert len(html_outputs(notebook)) == 2 and "hook" in html.unescape(html_outputs(notebook)[1])


def test_display_outside_ipython_is_refused_before_the_page_is_made(run, monkeypatch):
    monkeypatch.delattr(builtins, "get_ipython", raising=False)
    monkeypatch.setattr(web_module, "web_page", lambda *_, **__: pytest.fail("the page was made"))
    with pytest.raises(ArgumentRefused) as refused:
        display(run)
    (refusal,) = refused.value.refusals
    assert refusal.code == "visuals.display-outside-ipython"
    assert "mime.viz.web_page(" in refusal.action and f"mime visuals web {run}" in refusal.action
