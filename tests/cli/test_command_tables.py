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
from _platform_support import needs_pty, needs_readable_cpu_flags
from _terminal_support import run_on_terminal, stripped, terminal
from mime.cli import main as mime_main
from mime.cli._display import set_plain, show_registry
from mime.docs._export import export_json
from mime.env import EnvReport
from mime.env._check import AS_PINNED, DistributionRow, EnvRefused, Expected, distribution_refusals, distribution_rows
from mime.experiment import Experiment
from mime.saved import VerifyReport, read_folder
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


def registry_entries_of_a_child(tmp_path) -> dict[str, object]:
    """The registry as a fresh ``mime registry --json`` process holds it: the process that draws the table. This test
    process holds what earlier tests registered in it (toy entries a child never sees), so it is no oracle for the table."""
    done = in_child_piped(tmp_path, "registry", "--json")
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def old_registry_text(found: dict[str, object] | None = None) -> str:
    """What ``mime registry`` printed before the table, from the registry's export (this process's, unless ``found``)."""
    held = found if found is not None else json.loads(export_json())
    return "".join(f"{e['kind']}  {e['name']}  {e['version']}\n" for e in held["entries"]) + "".join(
        f"not loaded: {name} ({reason})\n" for name, reason in held["not_loaded"].items())


def parsed_table(text: str, *heads: str) -> list[list[str]]:
    """The rows of the table whose header holds ``heads`` in order, each row as its cells by column (the header's columns
    are where a cell starts; rich pads every line of a table to its width, and a blank line ends it, or is a rule
    between groups when a row follows). A cell is read by its column, so a word in the wrong column is a different cell."""
    lines = text.splitlines()
    for top, line in enumerate(lines):
        starts, at = [], 0
        for head in heads:
            at = line.find(head, at)
            if at < 0:
                break
            starts.append(at)
            at += len(head)
        if len(starts) == len(heads) and line.strip(" │").split()[:1] == [heads[0]]:
            break
    else:
        raise AssertionError(f"no table with the columns {heads} in\n{text}")
    width, rows, blank = len(lines[top + 1]), [], False
    for line in lines[top + 2:]:
        empty = not line.strip(" │")
        if len(line) != width or (empty and blank):
            break
        blank = empty
        if not empty:
            ends = [*starts[1:], width]
            rows.append([line[a:b].strip(" │") for a, b in zip(starts, ends, strict=True)])
    return rows


ROWS = (DistributionRow("jax", "0.12.0", "0.12.0", "the cpu pins", AS_PINNED),
        DistributionRow("a-distribution-with-a-long-name", "2026.11.5.post1", "2026.11.5.post1", "the lab's pins",
                        AS_PINNED))

#: Rows whose five cells all differ from each other and whose verdicts are not 'as pinned', so a table that put one
#: column's text in another, or said 'as pinned' whatever the row said, is found by the cell it got wrong.
DIFFERING = (DistributionRow("numpy", "2.4.0", "2.3.5", "the cpu pins", "differs from the pin"),
             DistributionRow("scipy", "1.16.0", None, "the lab's pins", "not installed"),
             DistributionRow("jax", "0.12.0", "0.12.0", "the pinned MIME wheel", AS_PINNED))


def an_env_report(**changes) -> EnvReport:
    held = dict(variant="cpu", distributions=len(ROWS), python="3.14.5", system="Test OS 1.0, glibc 2.39",
                devices=("cpu: cpu",), x64=True, matmul_precision="highest", xla_cache=None, fingerprint="f" * 64,
                shadows_checked=True, rows=ROWS)
    return EnvReport(**{**held, **changes})


def old_env_text(report: EnvReport) -> str:
    return report.text() + "\n"


# --- the tables on a terminal ---------------------------------------------------------------------------------------


def verify_in_process(saved, monkeypatch, *, planted: bool = False, other_environment: bool = False) -> tuple[str, int]:
    """``mime verify`` in this process on a pseudo-terminal: its transcript without control sequences, and its code.
    ``planted`` moves two fields of the rerun, ``other_environment`` makes the load and the rerun find this process unlike
    the one the lock was written in (on any platform, as every platform but Linux does: its CPU flags are unreadable)."""
    import numpy as np

    real = experiment_module.verify_history

    def moved(form, history, scales=None):
        history = {node: dict(fields) for node, fields in history.items()}
        history["probe"]["x"] = np.asarray(history["probe"]["x"]) + 1e-6
        history["probe"]["v"] = np.asarray(history["probe"]["v"]) * 1.5
        return real(form, history, scales)

    with monkeypatch.context() as patch:
        if planted:
            patch.setattr(experiment_module, "verify_history", moved)
        if other_environment:
            patch.setattr("mime.saved._folder.environment_differences", lambda lock: ["device: cpu here, gpu in the lock"])
        with terminal(columns=COLUMNS) as tty:
            patch.setattr(sys, "stdout", tty.stream)
            code = mime_main(["verify", str(saved)])
            return stripped(tty.transcript()), code


@needs_pty
@needs_readable_cpu_flags
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
    cells = {row[0]: row[1] for row in parsed_table(text, "check", "found")}
    for check in ("saved form", "input hash", "environment", "derived graph", "rerun", "held to", "probe.x", "probe.v"):
        assert check in cells, f"{check} has no row"
    assert cells["rerun"] == "the first 8 steps, on cpu, float64"
    assert cells["held to"] == "bit for bit: the environment the lock was written in"
    held = [key.removeprefix("history/").replace("/", ".", 1) for key in read_folder(saved).reference
            if key.startswith("history/")]
    assert len(held) > 1
    assert [cells[label] for label in held] == ["equal, bit for bit"] * len(held), "a row for each field the reference holds"


@needs_pty
@needs_readable_cpu_flags
def test_verify_with_a_field_moved_names_the_numbers_that_moved_and_the_tolerance_in_words(saved, monkeypatch):
    """The control is the run without the plant: the same command says nothing moved, so a table that marked every row
    'moved' (or none) would fail one of the two."""
    unplanted, code = verify_in_process(saved, monkeypatch)
    assert code == 0 and "moved" not in unplanted
    text, code = verify_in_process(saved, monkeypatch, planted=True)
    assert code == 1
    cells = {row[0]: row[1] for row in parsed_table(text, "check", "found")}
    assert cells["probe.x"] == "moved: max |difference| 1.000e-06, first at step 0; held to bit for bit"
    assert cells["probe.v"] == "moved: max |difference| 2.472e-01, first at step 0; held to bit for bit"
    assert [line for line in text.splitlines() if line.strip()][-1] == \
        "verify: 2 reference output(s) moved over 8 steps on cpu, float64"
    assert "probe.x: max |difference|" not in text, "the old line is a row, not repeated"


@needs_pty
def test_verify_in_another_environment_says_within_the_tolerance_and_never_equal_or_bit_for_bit(saved, monkeypatch):
    """Both arms of the words: a field within the tolerance is not equal, and a table that said 'equal' (or 'bit for bit',
    or the tolerance's two numbers the wrong way round, or the environment's sentence the wrong way round) is found
    here. The first run is the control: the same environment says 'equal, bit for bit'. The folder is of the current
    form, float64, so its tolerance is 2e-14 of each field's source scale (``mime.saved._scales``)."""
    tolerance = "within rtol 2e-14 of each field's source scale"
    text, code = verify_in_process(saved, monkeypatch, other_environment=True)
    assert code == 0, text
    cells = {row[0]: row[1] for row in parsed_table(text, "check", "found")}
    assert cells["held to"] == f"{tolerance}: another environment"
    assert cells["probe.x"] == cells["probe.v"] == tolerance
    assert "equal" not in text.replace("the environment the lock", "") and "bit for bit" not in text
    assert [line for line in text.splitlines() if line.strip()][-1].startswith(f"verify: reproduced 8 steps {tolerance} on ")
    text, code = verify_in_process(saved, monkeypatch, planted=True, other_environment=True)
    cells = {row[0]: row[1] for row in parsed_table(text, "check", "found")}
    assert code == 1
    # the cell runs past the column and folds onto the row below: its first line is read here
    assert cells["probe.v"].startswith("moved: largest difference 5.000e-01 of its source scale, its own largest "
                                       "magnitude, 4.944e-01, first at step 0; held to")
    assert cells["probe.x"].startswith("moved: largest difference ") and "first at step 0; held to" in cells["probe.x"]
    assert " ".join(text.split()).count(f"held to {tolerance}") == 3, "the held-to row and the two moved fields"


@needs_pty
def test_the_same_environment_arm_is_equal_bit_for_bit_the_control_of_the_other_environment_arm(saved, monkeypatch):
    """Where this platform can read its CPU, the load finds the lock's environment (the arm above is then the contrast);
    elsewhere it does not, and the contrast is made by a lock found to be this environment."""
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: [])
    with terminal(columns=COLUMNS) as tty:
        monkeypatch.setattr(sys, "stdout", tty.stream)
        assert mime_main(["verify", str(saved)]) == 0
        cells = {row[0]: row[1] for row in parsed_table(stripped(tty.transcript()), "check", "found")}
    assert cells["held to"] == "bit for bit: the environment the lock was written in"
    assert cells["probe.x"] == cells["probe.v"] == "equal, bit for bit"


def a_verification(**changes) -> VerifyReport:
    held = dict(steps=8, same_environment=True, rtol=1e-6, atol=1e-9, environment="cpu, float64",
                moved=("a.b: max |difference| 1.000e-03, first at step 2",), compared=("a.b", "a.c"))
    return VerifyReport(**{**held, **changes})


def drawn_verification(report: VerifyReport) -> dict[str, str]:
    from mime.cli._display import show_verification

    with terminal(columns=COLUMNS) as tty:
        show_verification(None, report, tty.stream)
        return {row[0]: row[1] for row in parsed_table(stripped(tty.transcript()), "check", "found")}


@needs_pty
def test_a_moved_line_whose_field_is_not_among_those_compared_is_a_row_of_its_own_so_no_numbers_are_dropped():
    cells = drawn_verification(a_verification(compared=("a.c",)))
    assert cells["a.b"] == "moved: max |difference| 1.000e-03, first at step 2; held to bit for bit"
    assert cells["a.c"] == "equal, bit for bit"
    assert drawn_verification(a_verification(compared=()))["a.b"].startswith("moved: max |difference| 1.000e-03")
    nested = drawn_verification(a_verification(moved=("a: b: max |difference| 1.000e-03, first at step 2",),
                                               compared=("a", "a: b")))
    assert nested["a: b"].startswith("moved: max |difference|") and nested["a"] == "equal, bit for bit", \
        "the longest label a line starts with is its own"


@needs_pty
def test_a_field_missing_or_reshaped_is_moved_and_no_tolerance_is_said_to_hold_it():
    cells = drawn_verification(a_verification(moved=("a.b: missing or of another shape",), same_environment=False))
    assert cells["a.b"] == "moved: missing or of another shape"
    assert cells["a.c"] == "within rtol 1e-06, atol 1e-09"


def test_a_verification_lists_every_field_it_compared_among_them_one_the_rerun_lacks_or_gave_in_another_shape(
        monkeypatch):
    """``compared`` is every field of the reference and of the rerun, the missing one too: a row for each."""
    import numpy as np
    from types import SimpleNamespace

    from mime.saved import verify_history

    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: [])
    lock = {"reference": {"steps": 3, "rtol": 1e-6, "atol": 1e-9}, "environment": {"device": "cpu"}}
    reference = {"history/n/kept": np.zeros(3), "history/n/lost": np.zeros(3), "history/n/reshaped": np.zeros(3)}
    form = SimpleNamespace(lock=lock, reference=reference, version=9)
    history = {"n": {"kept": np.zeros(3), "reshaped": np.zeros((3, 2)), "extra": np.zeros(3)}}
    report = verify_history(form, history)
    assert report.compared == ("n.extra", "n.kept", "n.lost", "n.reshaped")
    assert report.moved == ("n.extra: missing or of another shape", "n.lost: missing or of another shape",
                            "n.reshaped: missing or of another shape")


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
    rows = parsed_table(text, "distribution", "wanted", "found", "verdict", "pinned by")
    assert rows == [["jax", "0.12.0", "0.12.0", "as pinned", "the cpu pins"],
                    ["a-distribution-with-a-long-name", "2026.11.5.post1", "2026.11.5.post1", "as pinned", "the lab's pins"]]
    lines = text.splitlines()
    own = report.text().splitlines()
    assert [line for line in lines if line.strip()][-len(own):] == own, "the report's own lines, one each, under the table"
    assert capsys.readouterr().out == ""


@needs_pty
def test_the_pins_table_puts_each_cell_in_its_column_and_says_what_each_row_says():
    from mime.cli._display import show_environment

    with terminal(columns=COLUMNS) as tty:
        show_environment(an_env_report(rows=DIFFERING), tty.stream)
        rows = parsed_table(stripped(tty.transcript()), "distribution", "wanted", "found", "verdict", "pinned by")
    assert rows == [["numpy", "2.4.0", "2.3.5", "differs from the pin", "the cpu pins"],
                    ["scipy", "1.16.0", "-", "not installed", "the lab's pins"],
                    ["jax", "0.12.0", "0.12.0", "as pinned", "the pinned MIME wheel"]]


LONG = DistributionRow("jsonschema-specifications", "0.0.1", "2025.9.1", "the lab's pins", "differs from the pin")


@needs_pty
@pytest.mark.parametrize("columns", [80, 100])
def test_the_pins_table_on_a_narrow_terminal_keeps_every_name_and_version_whole_and_says_who_pins_in_a_line(columns):
    """At 80 columns five columns do not fit: a name and a version are never folded, and who pins is a line under the
    table (the control is 100 columns, where it is a column)."""
    from mime.cli._display import show_environment

    rows = (LONG, *(DistributionRow(f"dist-{i:02d}", "0.0.1", "1.2.3", "the cpu pins", "differs from the pin")
                    for i in range(20)))
    with terminal(columns=columns) as tty:
        show_environment(an_env_report(rows=rows), tty.stream)
        text = stripped(tty.transcript())
    heads = ("distribution", "wanted", "found", "verdict") + (("pinned by",) if columns == 100 else ())
    table = parsed_table(text, *heads)
    assert [row[:3] for row in table] == [[r.name, r.wanted, r.found] for r in rows], "every name and version whole"
    assert len(text.splitlines()) < 2 * len(rows) + 14, "a row is about a line"
    assert ("pinned by: the lab's pins (1), the cpu pins (20)" in text) == (columns == 80)
    if columns == 100:
        assert {row[4] for row in table} == {"the lab's pins", "the cpu pins"}


def refusal_of_a_differing_environment() -> EnvRefused:
    """What ``mime.env.check`` raises, built from the check's own two functions over an environment that differs in every
    way a pinned distribution can: two at other versions, one missing, one pinned and right, and one nothing pins."""
    expected = {"numpy": Expected("2.4.0", "the cpu pins"), "pandas": Expected("3.0.0", "the lab's pins"),
                "scipy": Expected("1.16.0", "the lab's pins"), "jax": Expected("0.12.0", "the cpu pins")}
    installed = {"numpy": ("2.3.5",), "pandas": ("2.9.0",), "jax": ("0.12.0",), "stray": ("1.0",)}
    return EnvRefused(distribution_refusals(installed, expected),
                      pin_rows=[row for row in distribution_rows(installed, expected) if not row.agrees])


@needs_pty
def test_a_failing_env_check_on_a_terminal_opens_with_the_pins_that_differ_and_keeps_every_other_reason(
        monkeypatch, capsys, tmp_path):
    refusal = refusal_of_a_differing_environment()
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path))
    monkeypatch.setattr(mime.env, "check", lambda *a, **k: (_ for _ in ()).throw(refusal))
    with terminal(columns=COLUMNS) as tty:
        monkeypatch.setattr(sys, "stderr", tty.stream)
        assert mime_main(["env", "check", "--variant", "cpu"]) == 2
        text = stripped(tty.transcript())
    rows = parsed_table(text, "distribution", "wanted", "found", "verdict", "pinned by")
    assert rows == [["numpy", "2.4.0", "2.3.5", "differs from the pin", "the cpu pins"],
                    ["pandas", "3.0.0", "2.9.0", "differs from the pin", "the lab's pins"],
                    ["scipy", "1.16.0", "-", "not installed", "the lab's pins"]], "only the distributions that differ"
    assert capsys.readouterr().out == ""
    for code in ("env.pin", "env.missing", "env.unpinned"):
        assert f"docs for {code}:" in text, f"{code} keeps its block"
    assert sum(line.strip(" │").startswith("numpy ") for line in text.splitlines()) == 1, "a row is not said twice"
    stray = [line for line in text.splitlines() if line.strip(" │").startswith("stray")]
    assert stray, "a reason the table does not hold keeps its row"
    assert text.count("what to do:") == 3 and "2 installed distributions differ from their pins:" in text


def test_a_failing_env_check_that_is_not_drawn_is_the_refusal_s_plain_text_with_no_table(monkeypatch, capsys, tmp_path):
    refusal = refusal_of_a_differing_environment()
    monkeypatch.setenv("MIME_TRACEBACK_DIR", str(tmp_path))
    monkeypatch.setattr(mime.env, "check", lambda *a, **k: (_ for _ in ()).throw(refusal))
    assert mime_main(["env", "check", "--variant", "cpu"]) == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "the pins that differ" not in captured.err
    assert all(code in captured.err for code in ("env.pin", "env.missing", "env.unpinned"))


@needs_pty
def test_registry_on_a_terminal_is_one_table_grouped_by_kind_with_every_entry_a_row(tmp_path):
    held = registry_entries_of_a_child(tmp_path)
    old = old_registry_text(held)
    raw, code = in_child_on_a_terminal(tmp_path, "registry")
    text = stripped(raw)
    assert code == 0, text
    kinds = list(dict.fromkeys(line.split("  ")[0] for line in old.splitlines() if not line.startswith("not loaded:")))
    once = "".join(line.split("  ", 1)[1] + "\n" if not line.startswith("not loaded:") else line + "\n"
                   for line in old.splitlines()) + "\n".join(kinds)
    assert holds_every_word_of(once, text), "every word, the kind once for its group and not again for each entry"
    for line in old.splitlines():
        if line.startswith("not loaded:"):
            assert line in text
    drawn: list[list[str]] = []
    for kind, name, version, declares in parsed_table(text, "kind", "name", "version", "declares"):
        if not name and drawn:  # a cell that wrapped: the rest of the row above
            drawn[-1][3] += declares
        else:
            drawn.append([kind or drawn[-1][0], name, version, declares])
    expected = [[e["kind"], e["name"], e["version"], "".join((", ".join(e["settings"]) or "-").split())]
                for e in held["entries"]]
    assert [[*row[:3], "".join(row[3].split())] for row in drawn] == expected, \
        "every entry is a row of its kind, name, version and what it declares, in the order the plain output prints"
    first_cells = [row[0] for row in parsed_table(text, "kind", "name", "version", "declares")]
    for kind in kinds:
        assert first_cells.count(kind) == 1, f"{kind} is named once, for its group"
    assert any(e["settings"] for e in held["entries"]), "the control: some entry declares settings, so that cell is read"


@needs_pty
def test_the_registry_table_puts_each_cell_in_its_column():
    entries = [("geometry", "screw", "0.3.0", "pitch, radius"), ("geometry", "tube", "0.4.1", ""),
               ("dynamics", "rigid_body", "1.2.0", "mass")]
    with terminal(columns=COLUMNS) as tty:
        show_registry(entries, {}, tty.stream)
        rows = parsed_table(stripped(tty.transcript()), "kind", "name", "version", "declares")
    assert rows == [["geometry", "screw", "0.3.0", "pitch, radius"], ["", "tube", "0.4.1", "-"],
                    ["dynamics", "rigid_body", "1.2.0", "mass"]]


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

