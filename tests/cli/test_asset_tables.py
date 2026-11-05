"""The tables the asset commands draw on a terminal (``mime assets list``, ``mime assets sources``, ``mime assets
licences``, ``mime visuals list``), ``mime pending``'s lines to paste set apart as blocks that are never wrapped, and what
``mime cache clear`` says it freed; through a real pseudo-terminal (POSIX; ``_platform_support.needs_pty``), and the other
half of the rule: output that is not going to a terminal, or that a terminal asked not to be styled for, is byte for byte
what the commands printed before (written out here from the objects the commands hold, as ``test_command_tables.py``
does).

What a table may not do is drop a word: each holds every word of the plain output. The controls force the drawing on a
captured stream and show the oracle would notice, on every run."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from _platform_support import needs_pty
from _terminal_support import stripped, terminal
from chain_cases import AR4_VISUALS

from mime import assets
from mime.assets import AssetRef
from mime.cli import main
from mime.cli import _display
from mime.cli._display import show_pending
from mime.terms import CustomLicence

FRAME = re.compile(r"[─-╿]")
COLUMNS = 160


@pytest.fixture(autouse=True)
def plain_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("MIME_OFFLINE", "1")
    monkeypatch.setenv("COLUMNS", str(COLUMNS))
    yield
    _display.set_plain(False)


def tokens(text: str) -> Counter[str]:
    return Counter(word.rstrip(":") for word in FRAME.sub(" ", text).split() if word.rstrip(":"))


def holds_every_word_of(plain: str, drawn: str) -> bool:
    missing = tokens(plain) - tokens(drawn)
    assert not missing, f"the table dropped {dict(missing)}"
    return True


def on_a_terminal(monkeypatch: pytest.MonkeyPatch, *words: str) -> tuple[str, int]:
    """``mime <words>`` run in this process with a terminal for standard output: what it drew, and its exit code."""
    with terminal(COLUMNS) as held, monkeypatch.context() as inside:
        inside.setattr(sys, "stdout", held.stream)
        code = main(list(words))
        return held.transcript(), code


def piped(capsys: pytest.CaptureFixture[str], *words: str) -> tuple[str, str, int]:
    code = main(list(words))
    captured = capsys.readouterr()
    return captured.out, captured.err, code


@pytest.fixture
def ar4_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The AR4's manifest as an installed package would ship it, so that ``mime visuals list`` has a set to list."""
    held = assets.read_manifest(AR4_VISUALS)
    monkeypatch.setattr("mime.scene.visual_sets", lambda: {held.name: held})


# --- mime assets list ------------------------------------------------------------------------------------------------------


def old_assets_list() -> str:
    return "".join(f"{ref.name}  {assets.size_text(ref.size)}  {'cached' if assets.is_cached(ref) else 'missing'}  "
                   f"{ref.licence}\n" for ref in assets.known().values())


def test_piped_assets_list_is_what_it_was(capsys: pytest.CaptureFixture[str]):
    out, err, code = piped(capsys, "assets", "list")
    assert (out, err, code) == (old_assets_list(), "", 0) and out


@needs_pty
def test_assets_list_on_a_terminal_is_a_table_of_name_size_cached_licence_and_tier(monkeypatch: pytest.MonkeyPatch):
    drawn, code = on_a_terminal(monkeypatch, "assets", "list")
    text = stripped(drawn)
    assert code == 0 and re.search(r"name\s+size\s+cached\s+licence\s+tier", text)
    assert holds_every_word_of(old_assets_list(), text)
    for ref in assets.known().values():
        assert re.search(rf"{re.escape(ref.name)}\s+{re.escape(assets.size_text(ref.size))}\s+\S+\s+{re.escape(ref.licence)}"
                         rf"\s+{ref.tier}", text), ref.name


def test_plain_and_no_color_keep_assets_list_as_it_was(monkeypatch: pytest.MonkeyPatch):
    with terminal(COLUMNS) as held, monkeypatch.context() as inside:
        inside.setattr(sys, "stdout", held.stream)
        assert main(["assets", "list", "--plain"]) == 0
        assert stripped(held.transcript()) == old_assets_list() and "\x1b" not in held.transcript()
    monkeypatch.setenv("NO_COLOR", "1")
    with terminal(COLUMNS) as held, monkeypatch.context() as inside:
        inside.setattr(sys, "stdout", held.stream)
        assert main(["assets", "list"]) == 0
        assert stripped(held.transcript()) == old_assets_list()


def test_control_the_table_forced_on_a_pipe_is_caught_by_the_oracle(monkeypatch: pytest.MonkeyPatch,
                                                                    capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(_display, "styled", lambda stream: True)
    monkeypatch.setattr("mime.cli.styled", lambda stream: True)
    out, _, _ = piped(capsys, "assets", "list")
    assert out != old_assets_list() and FRAME.search(out), "a table on a pipe is not what the pipe got before"


# --- mime assets sources ---------------------------------------------------------------------------------------------------


def old_sources(ref: AssetRef) -> str:
    custom = ref.custom[ref.licence]
    lines = [f"{ref.name}  {assets.size_text(ref.size)}  not cached  sha256 {ref.sha256}",
             f"  terms: licence {ref.licence} ({custom.title}, {custom.url}); tier referenced (not permitted, referenced); "
             f"role presentation; kind authored; credit {ref.credit}",
             "  its bytes never travel inside a bundle, a push or a package: only this record does",
             "  locations, in the order a fetch tries them:",
             f"    {ref.locations[0]}: not looked at: not looked at: MIME_OFFLINE is set"]
    return "\n".join(lines) + "\n"


def test_piped_sources_is_what_it_was(capsys: pytest.CaptureFixture[str]):
    held = assets.read_manifest(AR4_VISUALS).assets[0]
    out, err, code = piped(capsys, "assets", "sources", str(AR4_VISUALS))
    expected = old_sources(held)
    assert code == 0 and err == "" and out.startswith(expected) and out.count("sha256 ") == 18
    assert out.endswith(old_sources(assets.read_manifest(AR4_VISUALS).assets[-1]))


@needs_pty
def test_sources_on_a_terminal_is_a_table_with_how_many_locations_answered_and_the_lines_under_it(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    plain, _, _ = piped(capsys, "assets", "sources", str(AR4_VISUALS))
    drawn, code = on_a_terminal(monkeypatch, "assets", "sources", str(AR4_VISUALS))
    text = stripped(drawn)
    assert code == 0 and re.search(r"name\s+size\s+cached\s+licence\s+tier\s+reachable", text)
    assert re.search(r"ar4_mk5\.Link_Base_Aluminum\.STL\s+13 MB\s+not cached\s+LicenseRef-Annin-Robotics-NonCommercial\s+"
                     r"referenced\s+not looked at", text)
    assert holds_every_word_of(plain, text), "every line of the plain output, the locations and their state among them"


def test_a_reachable_count_is_how_many_of_the_locations_answered(monkeypatch: pytest.MonkeyPatch):
    from mime.assets._availability import SourceLocation, SourceView

    ref = assets.read_manifest(AR4_VISUALS).assets[0]
    where = (SourceLocation("a", "", True, "ok"), SourceLocation("b", "", False, "no"), SourceLocation("c", "", None, "?"))
    assert SourceView(ref, False, where).reachable == "1 of 2"
    assert SourceView(ref, False, where[2:]).reachable == "not looked at"
    assert SourceView(ref, False, where[:1]).reachable == "1 of 1"


# --- mime visuals list -----------------------------------------------------------------------------------------------------


def old_visuals_list() -> str:
    held = assets.read_manifest(AR4_VISUALS)
    return f"{held.name}  {len(held.files)} files  not fetched  {held.licence} ({held.verdict.tier}, {held.role})\n"


def test_piped_visuals_list_is_what_it_was(ar4_sets: None, capsys: pytest.CaptureFixture[str]):
    import mime.scene

    assert mime.scene.visual_sets()
    out, err, code = piped(capsys, "visuals", "list")
    assert (out, err, code) == (old_visuals_list(), "", 0)


@needs_pty
def test_visuals_list_on_a_terminal_is_a_table(ar4_sets: None, monkeypatch: pytest.MonkeyPatch,
                                               tmp_path: Path):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    drawn, code = on_a_terminal(monkeypatch, "visuals", "list")
    text = stripped(drawn)
    assert code == 0 and re.search(r"set\s+files\s+size\s+baked\s+licence\s+tier\s+role", text)
    assert re.search(r"ar4_mk5\s+18\s+50 MB\s+not fetched\s+LicenseRef-Annin-Robotics-NonCommercial\s+referenced\s+"
                     r"presentation", text)
    assert holds_every_word_of(old_visuals_list().replace("files", "").replace("(", "").replace(",", "").replace(")", ""),
                               text)


def test_visuals_list_with_nothing_says_what_it_always_said(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                            capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    out, _, code = piped(capsys, "visuals", "list")
    assert code == 0 and out.startswith("no set: a description's manifest is beside it (visuals.json)")


# --- mime assets licences -------------------------------------------------------------------------------------------------------


@needs_pty
def test_licences_on_a_terminal_is_a_table_and_piped_a_line_each(monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture[str]):
    from mime.assets import Licence, _acceptances

    licence = Licence("LicenseRef-Maker-NonCommercial", "Maker Licence 1.1", "https://example.org/licence",
                      ("non-commercial use only",))
    _acceptances.remember([licence])
    date = _acceptances.read_acceptances()[0].date
    out, _, code = piped(capsys, "assets", "licences")
    assert code == 0 and out == f"{licence.id}  {licence.title}  {licence.url}  accepted {date}\n"
    drawn, code = on_a_terminal(monkeypatch, "assets", "licences")
    text = stripped(drawn)
    assert code == 0 and re.search(r"licence\s+title\s+link\s+accepted", text)
    assert licence.id in text and licence.url in text and date in text, "the id is whole: it is what --revoke takes"


@needs_pty
def test_licences_table_keeps_an_id_whole_where_the_terminal_has_room_for_it_however_long_the_other_cells_are(
        monkeypatch: pytest.MonkeyPatch):
    """The id is what ``--revoke`` takes, and a folded id cannot be pasted: with a long title and a long link competing
    for the width, the id (which fits in half the terminal) is still one piece."""
    from mime.assets import Licence, _acceptances

    long_id = "LicenseRef-" + "Maker-Of-Meshes-" * 3 + "NonCommercial"
    assert len(long_id) + 2 <= COLUMNS // 2
    licence = Licence(long_id, "A Very Long Licence Title, Version 1.1, " * 3, "https://example.org/" + "licence/" * 14, ())
    _acceptances.remember([licence])
    drawn, code = on_a_terminal(monkeypatch, "assets", "licences")
    text = stripped(drawn)
    assert code == 0 and long_id in text, "the id is on one line, whole"


@needs_pty
def test_licences_table_draws_a_strangers_title_as_plain_characters(monkeypatch: pytest.MonkeyPatch):
    """An acceptance holds the title and link a manifest gave: no escape sequence of theirs reaches the terminal, and a
    newline in a title is not a row of the table's of its own."""
    from mime.assets import Licence, _acceptances

    _acceptances.remember([Licence("LicenseRef-Maker", "Maker \x1b[2J\x1b]0;pwned\x07 Licence\nI accept everything\n",
                                   "https://example.org/\x1b[99mlicence", ())])
    drawn, code = on_a_terminal(monkeypatch, "assets", "licences")
    assert code == 0
    for planted in ("\x1b[2J", "\x1b]0;", "\x07", "\x1b[99m"):
        assert planted not in drawn, planted
    assert "I accept everything" not in [" ".join(FRAME.sub(" ", row).split()) for row in stripped(drawn).splitlines()]


def test_licences_with_none_accepted_says_so(capsys: pytest.CaptureFixture[str]):
    out, _, code = piped(capsys, "assets", "licences")
    assert code == 0 and out.startswith("no licence accepted on this machine (")


# --- mime pending: the lines to paste are never wrapped --------------------------------------------------------------------


PENDING = (
    "profile exploratory: 1 item pending, each recorded in the saved form and stamped on every output. To seal the "
    "experiment, give each its reason in place of pending() (in the document, in place of {pending: true}) and build with "
    "profile=\"research\", which refuses a pending reason.\n"
    "In Python, the lines need, at the top of the script: from mime.validity import pending.\n"
    "1. drag.mesh, numerical default: the cell size is a default the build took for a setting the experiment did not give\n"
    "   in Python, in the call that makes the drag:\n"
    "       drag = Drag(rung=\"mrs\", cells_per_radius=pending(\"the reason this value is right for the experiment, "
    "written as a sentence a reader can check against the numbers\"), settings=dict(a=1, b=2, c=3, d=4, e=5, f=6, g=7))\n"
    "   in the document (experiment.mime.yaml), merged in:\n"
    "       interactions:\n"
    "         - name: drag\n"
    "           settings: {cells_per_radius: {pending: true}}\n")


@needs_pty
def test_the_lines_to_paste_are_blocks_written_whole_and_the_rest_is_the_plain_output(monkeypatch: pytest.MonkeyPatch):
    pasted = [line for line in PENDING.splitlines() if line.startswith("       ")]
    assert max(len(line) for line in pasted) > 150, "a line longer than the terminal is wide"
    with terminal(100) as held:
        show_pending(PENDING, held.stream)
        drawn = stripped(held.transcript())
    lines = drawn.splitlines()
    for line in pasted:
        assert line in lines, "a line to paste is one line of the output, with its indentation, never cut"
    assert holds_every_word_of(PENDING, drawn)
    first = next(index for index, line in enumerate(lines) if line == pasted[0])
    assert lines[first - 1] == "" and lines[first + 1].startswith("   in the document") is False
    assert "\n".join(lines[first:first + 4]).count("\n") == 3


@needs_pty
def test_control_a_wrapped_line_would_be_caught(monkeypatch: pytest.MonkeyPatch):
    """The check above can fail: the same long line drawn as an ordinary line of the output (wrapped, as rich wraps
    one) is not one line of it."""
    long = next(line for line in PENDING.splitlines() if len(line) > 150)
    with terminal(100) as held:
        _display._console_of(held.stream).print(_display._text(long))
        drawn = stripped(held.transcript())
    assert long not in drawn.splitlines() and holds_every_word_of(long, drawn)


def test_pending_piped_is_the_text_as_it_was(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]):
    import json

    from mime.testing.toys import full_toy

    folder = tmp_path / "toy"
    full_toy().save(folder)
    from mime.experiment import Experiment

    expected = Experiment.load(folder).build().report.profile.text()  # type: ignore[union-attr]
    out, err, code = piped(capsys, "pending", str(folder))
    assert out == expected and code == 1, (err, json.dumps(out)[:200])


# --- mime cache clear -----------------------------------------------------------------------------------------------------


def test_cache_clear_says_how_much_it_freed_on_a_terminal_and_what_it_always_said_elsewhere(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path):
    root = tmp_path / "xla"
    monkeypatch.setattr("mime.env.user_xla_cache_root", lambda: root)
    monkeypatch.setattr("mime.env.clear_user_xla_cache", lambda path: 3_000_000)
    out, err, code = piped(capsys, "cache", "clear")
    assert (out, code) == (f"removed 3000000 bytes of XLA's compilation cache from {root}\n", 0)
    with terminal(COLUMNS) as held, monkeypatch.context() as inside:
        inside.setattr(sys, "stdout", held.stream)
        assert main(["cache", "clear"]) == 0
        text = stripped(held.transcript())
    assert "freed 3.0 MB" in text and str(root) in text and "?" not in text, "it says how much it freed, and asks nothing"
    monkeypatch.setattr("mime.env.clear_user_xla_cache", lambda path: 0)
    out, _, _ = piped(capsys, "cache", "clear")
    assert out == f"no XLA compilation cache to remove at {root}\n"
