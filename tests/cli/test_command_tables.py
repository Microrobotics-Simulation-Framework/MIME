"""The tables ``mime verify``, ``mime env check`` and ``mime registry`` draw on a terminal, through a real pseudo-terminal
(POSIX; ``_platform_support.needs_pty`` says what does not run on Windows), and the other half of the rule: output that
is not going to a terminal, or that a terminal asked not to be styled for (``--plain``, ``NO_COLOR``, ``TERM=dumb``),
is byte for byte what the commands printed before the tables, on every platform.

What a table may not do is drop a word: each test of a table holds that every word of the plain output is in it. The
plain output's oracle is written out here from the objects the commands hold (the load report, the verification, the
environment report, the registry's export), as ``test_pipes.py`` does for ``mime run``; the controls plant a colour code
in piped output, force the table on a captured stream, and plant a fault in the rerun, each to show the check that
passes can fail."""

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

import mime.cli
import mime.env
import mime.experiment._experiment as experiment_module
from _platform_support import needs_pty
from _terminal_support import run_on_terminal, stripped, terminal
from mime.cli import main as mime_main
from mime.cli._display import set_plain, show_registry
from mime.docs._export import export_json
from mime.env import EnvReport
from mime.env._check import AS_PINNED, DistributionRow
from mime.experiment import Experiment
from mime.saved import read_folder
from mime.testing.toys import full_toy

pytestmark = pytest.mark.own_child_cache

FRAME = re.compile(r"[─-╿]")
ESCAPE = re.compile("\x1b")
#: Wide enough that no path or hash in a cell is folded: a folded word is whole to a reader and split to a test.
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
    monkeypatch.delenv("XLA_FLAGS", raising=False)
    for name in ("NO_COLOR", "FORCE_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    held = jax.config.jax_enable_x64
    yield
    jax.config.update("jax_enable_x64", held)
    set_plain(False)


def tokens(text: str) -> Counter[str]:
    """The words of a text, frames and the colons that end a label dropped: a table sets the same words in cells."""
    return Counter(word.rstrip(":") for word in FRAME.sub(" ", text).split() if word.rstrip(":"))


def holds_every_word_of(plain: str, drawn: str) -> bool:
    missing = tokens(plain) - tokens(drawn)
    assert not missing, f"the table dropped {dict(missing)}"
    return True


def child_environment(home: Path, **extra: str) -> dict[str, str]:
    held = {key: value for key, value in os.environ.items() if key not in ("MIME_TRACEBACK", "NO_COLOR", "FORCE_COLOR",
                                                                            "TERM")}
    return {**held, "HOME": str(home), "USERPROFILE": str(home), "XDG_CACHE_HOME": str(home / ".cache"),
            "MIME_TRACEBACK_DIR": str(home / "tracebacks"), "TERM": "xterm-256color", "JAX_PLATFORMS": "cpu",
            "MIME_OFFLINE": "1", **extra}


def in_child_on_a_terminal(tmp_path, *words, env=None, translated=True) -> tuple[str, int]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return run_on_terminal([sys.executable, "-m", "mime.cli", *words], env=child_environment(home, **(env or {})),
                           cwd=tmp_path, columns=COLUMNS, translated=translated)


def in_child_piped(tmp_path, *words) -> subprocess.CompletedProcess[str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run([sys.executable, "-m", "mime.cli", *words], capture_output=True, text=True, check=False,
                          env=child_environment(home), cwd=tmp_path)


def old_verify_text(folder: Path) -> tuple[str, str]:
    """What ``mime verify`` printed before the table, written out from the objects it holds: the load report, the
    verification (and the verification's verdict line)."""
    from mime.env import apply_lock_precision

    apply_lock_precision(folder)
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    report = loaded.build().verify()
    return f"{loaded.load_report.text()}\n{report.text()}\n", report.text().splitlines()[0]


def old_registry_text() -> str:
    found = json.loads(export_json())
    return "".join(f"{e['kind']}  {e['name']}  {e['version']}\n" for e in found["entries"]) + "".join(
        f"not loaded: {name} ({reason})\n" for name, reason in found["not_loaded"].items())


ROWS = (DistributionRow("jax", "0.12.0", "0.12.0", "the cpu pins", AS_PINNED),
        DistributionRow("a-distribution-with-a-long-name", "2026.11.5.post1", "2026.11.5.post1", "the lab's pins",
                        AS_PINNED))


def an_env_report(**changes) -> EnvReport:
    held = dict(variant="cpu", distributions=len(ROWS), python="3.14.5", system="Test OS 1.0, glibc 2.39",
                devices=("cpu: cpu",), x64=True, matmul_precision="highest", xla_cache=None, fingerprint="f" * 64,
                shadows_checked=True, rows=ROWS)
    return EnvReport(**{**held, **changes})


def old_env_text(report: EnvReport) -> str:
    return report.text() + "\n"


# --- the tables on a terminal ---------------------------------------------------------------------------------------


@needs_pty
def test_verify_on_a_terminal_is_a_table_of_its_checks_with_every_word_of_the_plain_output_and_the_verdict_under_it(
        saved, tmp_path):
    old, verdict = old_verify_text(saved)
    raw, code = in_child_on_a_terminal(tmp_path, "verify", str(saved))
    text = stripped(raw)
    assert code == 0, text
    assert holds_every_word_of(old, text)
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[-1] == verdict, "the verdict is one line, under the table"
    assert not any(line.startswith("verify:") for line in lines[:-1])
    header = next(line for line in lines if "check" in line and "found" in line)
    for check in ("saved form", "input hash", "environment", "derived graph", "rerun", "held to", "probe.x", "probe.v"):
        assert any(line.strip().startswith(check) for line in lines[lines.index(header):]), f"{check} has no row"
    held = [key for key in read_folder(saved).reference if key.startswith("history/")]
    assert len(held) > 1 and text.count("equal, bit for bit") == len(held), "a row for each field the reference holds"


@needs_pty
def test_verify_with_a_field_moved_names_the_numbers_that_moved_and_the_tolerance_in_words(saved, monkeypatch):
    """The control is the run without the plant: the same command says nothing moved, so a table that marked every row
    'moved' (or none) would fail one of the two."""
    import numpy as np

    real = experiment_module.verify_history

    def moved(form, history):
        history = {node: dict(fields) for node, fields in history.items()}
        history["probe"]["x"] = np.asarray(history["probe"]["x"]) + 1e-6
        history["probe"]["v"] = np.asarray(history["probe"]["v"]) * 1.5
        return real(form, history)

    def draw(planted: bool) -> tuple[str, int]:
        with monkeypatch.context() as patch:
            if planted:
                patch.setattr(experiment_module, "verify_history", moved)
            with terminal(columns=COLUMNS) as tty:
                patch.setattr(sys, "stdout", tty.stream)
                code = mime_main(["verify", str(saved)])
                return stripped(tty.transcript()), code

    unplanted, code = draw(False)
    assert code == 0 and "moved" not in unplanted
    text, code = draw(True)
    assert code == 1
    rows = {line.split()[0]: line for line in text.splitlines() if line.split()[:1] and line.split()[0].startswith("probe.")}
    assert "moved: max |difference| 1.000e-06, first at step 0" in rows["probe.x"]
    assert "moved: max |difference| 2.472e-01, first at step 0" in rows["probe.v"]
    assert text.count("held to bit for bit") == 2, "the tolerance, in words, beside each failure"
    assert [line for line in text.splitlines() if line.strip()][-1] == \
        "verify: 2 reference output(s) moved over 8 steps on cpu, float64"
    assert "probe.x: max |difference|" not in text, "the old line is a row, not repeated"


@needs_pty
def test_env_check_on_a_terminal_is_the_pins_against_what_is_installed_a_row_each_and_the_other_findings_as_lines(
        monkeypatch, capsys):
    report = an_env_report()
    monkeypatch.setattr(mime.env, "check", lambda *args, **kwargs: report)
    with terminal(columns=COLUMNS) as tty:
        monkeypatch.setattr(sys, "stdout", tty.stream)
        assert mime_main(["env", "check", "--variant", "cpu"]) == 0
        text = stripped(tty.transcript())
    assert holds_every_word_of(old_env_text(report), text)
    lines = text.splitlines()
    head = next(line for line in lines if "distribution" in line and "verdict" in line)
    rows = [line.split() for line in lines[lines.index(head) + 2:] if line.split()]
    assert [row[:2] for row in rows[:2]] == [["jax", "0.12.0"], ["a-distribution-with-a-long-name", "2026.11.5.post1"]]
    assert all("as" in row and "pinned" in row for row in rows[:2])
    own = report.text().splitlines()
    assert [line for line in lines if line.strip()][-len(own):] == own, "the report's own lines, one each, under the table"
    assert capsys.readouterr().out == ""


@needs_pty
def test_registry_on_a_terminal_is_one_table_grouped_by_kind_with_every_entry_a_row(tmp_path):
    old = old_registry_text()
    raw, code = in_child_on_a_terminal(tmp_path, "registry")
    text = stripped(raw)
    assert code == 0, text
    kinds = list(dict.fromkeys(line.split("  ")[0] for line in old.splitlines() if not line.startswith("not loaded:")))
    once = "".join(line.split("  ", 1)[1] + "\n" if not line.startswith("not loaded:") else line + "\n"
                   for line in old.splitlines()) + "\n".join(kinds)
    assert holds_every_word_of(once, text), "every word, the kind once for its group and not again for each entry"
    rows = [line.split() for line in text.splitlines() if line.strip()]
    for line in old.splitlines():
        if line.startswith("not loaded:"):
            assert line in text
            continue
        kind, name, version = line.split("  ")
        assert any(name in row and version in row for row in rows), f"{line!r} has no row"
        assert text.count(kind) >= 1
    first_cells = [line[1:].lstrip() for line in text.splitlines() if line.strip()]
    for kind in kinds:
        assert sum(cell.startswith(kind + " ") for cell in first_cells) == 1, f"{kind} is named once, for its group"


@needs_pty
def test_a_long_cell_wraps_and_is_never_cut():
    declares = f"{'a' * 30}, {'b' * 30}"
    with terminal(columns=40) as tty:
        show_registry([("geometry", "screw", "0.3.0", declares)], {"labx": "needs the verify extra"}, tty.stream)
        text = stripped(tty.transcript())
    lines = text.splitlines()
    rule = next(i for i, line in enumerate(lines) if "──" in line)
    last_cells = [line.split()[-1] for line in lines[rule + 1:] if line.split() and not line.startswith("not loaded")]
    assert "".join(last_cells) == "a" * 30 + "," + "b" * 30, "every character of the cell, over as many lines as it needs"
    assert max(len(line) for line in lines[:rule + 1 + len(last_cells)]) <= 40, "the table fits the terminal"
    assert lines[-1] == "not loaded: labx (needs the verify extra)"


# --- not a terminal, or a terminal asked not to be drawn on: what the commands always printed -----------------------


def registry_oracle_matches(printed: str) -> bool:
    return printed == old_registry_text()


@pytest.mark.parametrize("command", ["verify", "registry", "env"])
def test_piped_output_is_exactly_the_old_commands_and_holds_no_escape(command, saved, capsys, monkeypatch):
    if command == "env":
        report = an_env_report()
        monkeypatch.setattr(mime.env, "check", lambda *args, **kwargs: report)
        assert mime_main(["env", "check", "--variant", "cpu"]) == 0
        expected = old_env_text(report)
    elif command == "registry":
        assert mime_main(["registry"]) == 0
        expected = old_registry_text()
    else:
        expected, _ = old_verify_text(saved)
        capsys.readouterr()
        assert mime_main(["verify", str(saved)]) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == expected
    assert ESCAPE.search(captured.out) is None


def test_piped_registry_json_is_untouched(capsys):
    assert mime_main(["registry", "--json"]) == 0
    assert capsys.readouterr().out == export_json() + "\n"


@pytest.mark.parametrize("command", ["verify", "registry"])
@pytest.mark.parametrize(("words", "env"), [(["--plain"], {}), ([], {"NO_COLOR": "1"}), ([], {"TERM": "dumb"})],
                         ids=["--plain", "NO_COLOR", "TERM=dumb"])
@needs_pty
def test_a_terminal_asked_to_be_plain_gets_exactly_what_a_pipe_gets(command, words, env, saved, tmp_path):
    words_of = ["verify", str(saved)] if command == "verify" else ["registry"]
    piped = in_child_piped(tmp_path, *words_of)
    assert piped.returncode == 0 and piped.stderr == ""
    raw, code = in_child_on_a_terminal(tmp_path, *words, *words_of, env=env, translated=False)
    assert code == 0 and ESCAPE.search(raw) is None and "─" not in raw, "nothing is drawn"
    assert raw == piped.stdout


@pytest.mark.parametrize(("setting", "value"), [("NO_COLOR", "1"), ("TERM", "dumb"), ("plain", True)],
                         ids=["NO_COLOR", "TERM=dumb", "--plain"])
@needs_pty
def test_env_check_on_a_terminal_asked_to_be_plain_prints_the_report_text(setting, value, monkeypatch):
    report = an_env_report()
    monkeypatch.setattr(mime.env, "check", lambda *args, **kwargs: report)
    words = ["--plain"] if setting == "plain" else []
    if setting != "plain":
        monkeypatch.setenv(setting, value)
    with terminal(columns=COLUMNS) as tty:
        monkeypatch.setattr(sys, "stdout", tty.stream)
        assert mime_main([*words, "env", "check", "--variant", "cpu"]) == 0
        raw = tty.transcript()
    assert raw.replace("\r\n", "\n") == old_env_text(report) and ESCAPE.search(raw) is None


# --- the controls: each fails the check it stands behind ------------------------------------------------------------


def test_the_control_a_colour_code_planted_in_piped_output_is_found_in_each_command(saved, capsys, monkeypatch):
    """The equality above is only evidence if a regression it guards against would break it: plant the colour in the
    text each command prints and the same comparison fails, and the escape is found."""
    from mime.saved import VerifyReport

    report = an_env_report()
    expected_env = old_env_text(report)
    monkeypatch.setattr(mime.env, "check", lambda *args, **kwargs: report)
    original_env, original_verify = EnvReport.text, VerifyReport.text
    monkeypatch.setattr(EnvReport, "text", lambda self: f"\x1b[1m{original_env(self)}\x1b[0m")
    assert mime_main(["env", "check", "--variant", "cpu"]) == 0
    printed = capsys.readouterr().out
    assert ESCAPE.search(printed) and printed != expected_env
    monkeypatch.undo()

    monkeypatch.setattr(VerifyReport, "text", lambda self: f"\x1b[1m{original_verify(self)}\x1b[0m")
    expected, _ = old_verify_text(saved)
    capsys.readouterr()
    assert mime_main(["verify", str(saved)]) == 0
    printed = capsys.readouterr().out
    assert printed != expected and "\\x1b[1mverify: reproduced" in printed, \
        "verify writes a control character out as its escape (mime._terminal), so the plant shows as that text"
    monkeypatch.undo()

    real = export_json()
    monkeypatch.setattr("mime.docs._export.export_json", lambda: real.replace('"arm_tracking"', '"\\u001b[1marm_tracking"'))
    assert mime_main(["registry"]) == 0
    printed = capsys.readouterr().out
    assert ESCAPE.search(printed) and not registry_oracle_matches(printed)


def test_the_control_the_tables_forced_on_a_captured_stream_are_not_the_old_commands(saved, capsys, monkeypatch):
    """The same commands with the display forced on: what they print differs from the old text, so the equalities above
    are tests that a pipe is not drawn for."""
    report = an_env_report()
    monkeypatch.setattr(mime.env, "check", lambda *args, **kwargs: report)
    monkeypatch.setattr(mime.cli, "styled", lambda stream: True)
    expected_verify, _ = old_verify_text(saved)
    capsys.readouterr()
    for words, expected in ((["env", "check", "--variant", "cpu"], old_env_text(report)),
                            (["registry"], old_registry_text()), (["verify", str(saved)], expected_verify)):
        assert mime_main(words) in (0, 1)
        printed = capsys.readouterr().out
        assert "──" in printed and printed != expected, words[0]


def test_a_check_that_fails_is_still_a_refusal_block_and_no_table(capsys, monkeypatch, tmp_path):
    """A failing ``mime env check`` is drawn by the refusal's own block (a row for each distribution that differs, under
    its code); the table is the passing check's."""
    from mime.env import _check

    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path))
    refusals = _check.distribution_refusals({"extra": ["1.0.0"]}, {})
    monkeypatch.setattr(mime.env, "check", lambda *a, **k: (_ for _ in ()).throw(_check.EnvRefused(refusals)))
    assert mime_main(["env", "check", "--variant", "cpu"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "env.unpinned" in captured.err

