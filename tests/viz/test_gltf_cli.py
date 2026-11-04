"""``mime visuals web``: the page written where ``--out`` says (the viewer inlined, or from jsDelivr with ``--cdn``),
the GLB beside it with ``--glb``, opened in the default browser with ``--open``; refused inside a run folder; and the
command, with its flags, in the completion scripts, which are written from the parser."""

import base64
import webbrowser

import pytest
from gltf_cases import ball, recorded, with_pose

from mime.cli import build_parser, main
from mime.cli._completion import describe
from mime.viz import page


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ball") / "run"
    recorded(ball().build(), folder, 3, lambda k, start: with_pose(start, "ball", [0, 0, -1e-6 * k, 1, 0, 0, 0]))
    return folder


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


def test_the_command_writes_the_page_and_the_glb_and_opens_it(run, tmp_path, monkeypatch, capsys):
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    out, glb = tmp_path / "ball.html", tmp_path / "ball.glb"
    assert main(["visuals", "web", str(run), "--out", str(out), "--glb", str(glb), "--open"]) == 0
    text = out.read_text(encoding="utf-8")
    encoded = text.split("data:model/gltf-binary;base64,")[1].split('"')[0]
    assert "text/javascript;base64" in text and glb.read_bytes() == base64.b64decode(encoded)
    assert opened == [out.resolve().as_uri()] and "the viewer inlined" in capsys.readouterr().out


def test_cdn_leaves_the_viewer_to_jsdelivr(run, tmp_path):
    out = tmp_path / "ball.html"
    assert main(["visuals", "web", str(run), "--out", str(out), "--cdn", "--slow-motion", "100"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "text/javascript;base64" not in text and "shown 100 times slower" in text


def test_a_page_inside_the_run_folder_is_refused(run, capsys):
    assert main(["visuals", "web", str(run), "--out", str(run / "view.html")]) == 2
    assert "visuals.web-out" in capsys.readouterr().err and not (run / "view.html").exists()


def test_the_completion_scripts_hold_the_command_and_its_flags():
    tree = describe(build_parser())
    (visuals,) = [node for node in tree.children if node.words == ("visuals",)]
    (held,) = [node for node in visuals.children if node.words == ("visuals", "web")]
    assert set(held.flags) >= {"--out", "--glb", "--slow-motion", "--caption", "--cdn", "--open"}
    assert held.files, "its target completes as a file or a folder"


def test_the_page_the_command_writes_is_the_librarys(run, tmp_path):
    from mime.viz import gltf

    out = tmp_path / "ball.html"
    main(["visuals", "web", str(run), "--out", str(out), "--caption", "a ball"])
    assert out.read_text(encoding="utf-8") == page(gltf(run), caption="a ball")
