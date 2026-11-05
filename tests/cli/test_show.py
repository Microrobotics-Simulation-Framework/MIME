"""``mime show``: a saved experiment's topology as a tree (``--tree``, the default) and the graph it compiles to as
Mermaid text (``--graph``), without running it.

**What this bounds.** That the command prints what the Python calls print (``Experiment.tree()``, ``built.to_mermaid()``)
for a saved folder, byte for byte when its output is not going to a terminal; that on a terminal the tree is drawn
with every word of the plain tree; that ``--json`` is the tree as data; and that a missing folder and an option pair
that do not go together are refused with an exit code. The tree and the Mermaid are judged in
``tests/build/test_graph_and_tree.py``; this judges the command around them.

**Windows.** The terminal test uses a pseudo-terminal, which Windows has not (``needs_pty``), so the drawn tree is not
tested there; the piped and ``--json`` tests, which are what a program reads, run on every platform.

**Controls.** The committed text of the full toy's tree and Mermaid is the oracle of the piped output, so a command that
printed the tree drawn (colour codes and guide lines) into a pipe differs from it; the terminal test plants a colour code
in the oracle's comparison to show the escape check can fail; and ``--plain`` and ``NO_COLOR`` on a terminal are held to
the piped text."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import jax
import pytest

from _platform_support import needs_pty
from _terminal_support import run_on_terminal, stripped
from mime.cli import main as mime_main
from mime.testing.toys import full_toy

pytestmark = pytest.mark.own_child_cache

GOLDEN = Path(__file__).resolve().parents[1] / "build" / "golden"
TREE = (GOLDEN / "full_toy.tree").read_text()
MERMAID = (GOLDEN / "full_toy.mmd").read_text()
FRAME = re.compile(r"[─-╿]")
ESCAPE = re.compile("\x1b")
COLUMNS = 160


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    folder = tmp_path_factory.mktemp("toy")
    full_toy().save(folder)
    jax.config.update("jax_enable_x64", held)
    return folder


@pytest.fixture(autouse=True)
def restored(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)


def environment(home: Path, **extra: str) -> dict[str, str]:
    held = {key: value for key, value in os.environ.items() if key not in ("MIME_TRACEBACK", "NO_COLOR", "FORCE_COLOR",
                                                                            "TERM")}
    return {**held, "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
            "MIME_TRACEBACK_DIR": str(home / "tracebacks"), "TERM": "xterm-256color", "JAX_PLATFORMS": "cpu",
            "MIME_OFFLINE": "1", **extra}


def piped(tmp_path, *words) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, "-m", "mime.cli", *words], capture_output=True, text=True, check=False,
                          env=environment(home), cwd=tmp_path)


def on_a_terminal(tmp_path, *words, env=None, translated=True) -> tuple[str, int]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return run_on_terminal([sys.executable, "-m", "mime.cli", *words], env=environment(home, **(env or {})),
                           cwd=tmp_path, columns=COLUMNS, translated=translated)


def tokens(text: str) -> Counter[str]:
    return Counter(word for word in FRAME.sub(" ", text).split() if word)


def test_piped_the_tree_is_the_committed_text_byte_for_byte_and_the_default(saved, tmp_path):
    for words in (("show", str(saved)), ("show", str(saved), "--tree")):
        done = piped(tmp_path, *words)
        assert done.returncode == 0, done.stderr
        assert done.stdout == TREE
        assert not ESCAPE.search(done.stdout) and not FRAME.search(done.stdout)


def test_piped_the_graph_is_the_committed_mermaid_byte_for_byte(saved, tmp_path):
    done = piped(tmp_path, "show", str(saved), "--graph")
    assert done.returncode == 0, done.stderr
    assert done.stdout == MERMAID


def test_both_views_are_the_tree_a_blank_line_and_the_graph(saved, capsys):
    assert mime_main(["show", str(saved), "--tree", "--graph"]) == 0
    assert capsys.readouterr().out == TREE + "\n" + MERMAID


def test_json_is_the_tree_as_data_and_the_same_every_time(saved, capsys):
    assert mime_main(["show", str(saved), "--json"]) == 0
    first = capsys.readouterr().out
    assert mime_main(["show", str(saved), "--tree", "--json"]) == 0
    assert capsys.readouterr().out == first
    held = json.loads(first)
    assert held["format"] == "mime.tree/1" and held["experiment"] == "full_toy"
    assert [row["id"] for row in held["laws"]] == ["toy.couple:probe", "toy.drag:probe", "toy.hold:probe",
                                                   "toy.lift:probe", "toy.spring:probe"]
    assert first.endswith("}\n") and first == json.dumps(held, indent=2) + "\n"


def test_the_json_carries_every_name_of_the_plain_tree(saved, capsys):
    mime_main(["show", str(saved), "--json"])
    held = json.loads(capsys.readouterr().out)
    names = {row["name"] for key in ("media", "bodies", "apparatus") for row in held[key]}
    ids = {row["id"] for key in ("laws", "observations") for row in held[key]}
    for word in names | ids:
        assert word in TREE


def test_a_document_stands_for_its_folder(saved, capsys):
    assert mime_main(["show", str(saved / "experiment.mime.yaml")]) == 0
    assert capsys.readouterr().out == TREE
    assert mime_main(["show", str(saved / "experiment.mime.yaml"), "--graph"]) == 0
    assert capsys.readouterr().out == MERMAID


def test_a_folder_that_is_not_there_is_refused_naming_the_code(tmp_path, capsys):
    assert mime_main(["show", str(tmp_path / "nowhere")]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "load.missing-folder" in captured.err


def test_json_and_graph_do_not_go_together(saved, capsys):
    with pytest.raises(SystemExit) as stopped:
        mime_main(["show", str(saved), "--json", "--graph"])
    assert stopped.value.code == 2
    assert "--json" in capsys.readouterr().err


def test_show_is_listed_in_the_help_under_run_and_has_its_example(capsys):
    with pytest.raises(SystemExit):
        mime_main(["--help"])
    out = capsys.readouterr().out
    run_group = out.split("Run")[1].split("Share")[0]
    assert re.search(r"^\s+show\s+what a saved experiment is made of", run_group, re.MULTILINE)
    with pytest.raises(SystemExit):
        mime_main(["show", "--help"])
    assert "mime show my_experiment --graph" in capsys.readouterr().out


@needs_pty
def test_on_a_terminal_the_tree_is_drawn_with_every_word_of_the_plain_tree(saved, tmp_path):
    raw, code = on_a_terminal(tmp_path, "show", str(saved))
    drawn = stripped(raw)
    assert code == 0, drawn
    assert FRAME.search(drawn), "the tree is drawn with guide lines"
    assert not (tokens(TREE) - tokens(drawn)), f"the drawing dropped {dict(tokens(TREE) - tokens(drawn))}"
    assert [line.strip(" │├└─") for line in drawn.splitlines() if line.strip()][0] == \
        TREE.splitlines()[0], "the experiment is its root"


@needs_pty
def test_control_the_check_for_a_dropped_word_can_fail(saved, tmp_path):
    raw, _ = on_a_terminal(tmp_path, "show", str(saved))
    drawn = stripped(raw).replace("toy.lift:probe", "")
    assert tokens(TREE) - tokens(drawn)


@needs_pty
@pytest.mark.parametrize("how", ["--plain", "NO_COLOR", "TERM=dumb"])
def test_a_terminal_asked_for_plain_text_gets_the_piped_text_byte_for_byte(saved, tmp_path, how):
    if how == "--plain":
        raw, code = on_a_terminal(tmp_path, "--plain", "show", str(saved), translated=False)
    else:
        name, _, value = how.partition("=")
        raw, code = on_a_terminal(tmp_path, "show", str(saved), env={name: value or "1"}, translated=False)
    assert code == 0
    assert raw == TREE and not ESCAPE.search(raw)


@needs_pty
def test_control_the_terminal_without_a_request_for_plain_text_is_not_the_piped_text(saved, tmp_path):
    raw, _ = on_a_terminal(tmp_path, "show", str(saved), translated=False)
    assert raw != TREE


@needs_pty
def test_the_graph_on_a_terminal_is_still_the_mermaid_text(saved, tmp_path):
    raw, code = on_a_terminal(tmp_path, "show", str(saved), "--graph", translated=False)
    assert code == 0 and raw == MERMAID
