"""The error corpus: planted mistakes of a script's user, each a small script in ``corpus/`` run as its user runs it, in
a fresh process. Adding a case is one file.

A ``.py`` case is a script; its header names the codes it must be refused with (``# corpus: codes <code>...``) and
one of its lines carries ``# corpus: here``, the line the error must name. A ``.cli`` case is one ``mime`` command
line after its header, which names the text the error must name instead of a line (``# corpus: names <text>``).

By default each must give a coded MIME error, in at most :data:`MOST_LINES` lines, naming the script's file and the
marked line (or the command's argument), with no frame of MIME's, JAX's, XLA's or MADDENING's, and its full chain
written to its file. **The control:** with ``MIME_TRACEBACK=full`` (the CLI's ``--traceback``), the library's frames
appear, so the short form's lack of them is the filter's doing and not the error's."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

CORPUS = Path(__file__).parent / "corpus"
#: The most lines a short error may take, frames and footer included.
MOST_LINES = 25
#: A frame line of a file in a library: MIME's own, an installed package's, JAX's or MADDENING's, under either
#: separator (the platforms job runs the corpus on Windows).
LIBRARY_FRAME = re.compile(r'File "[^"]*[/\\](?:site-packages|src[/\\]mime|jax|jaxlib|maddening)[/\\]')
#: What JAX, XLA or MADDENING print of their own, which a short error never shows.
HEADER = re.compile(r"# corpus: (codes|names) (.+)")
BELOW = re.compile(r"JAX has removed its internal frames|jax\._src|jaxlib|XlaRuntimeError|maddening[/\\]")


@dataclass(frozen=True)
class Case:
    path: Path
    codes: tuple[str, ...]
    #: The line the error names, for a script; ``None`` for a command line.
    here: int | None
    #: The text the error names in place of a line, for a command line.
    names: str | None
    #: The command line, for a ``.cli`` case.
    words: tuple[str, ...] = ()

    @property
    def cli(self) -> bool:
        return self.path.suffix == ".cli"


def read_case(path: Path) -> Case:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = {found[1]: found[2] for found in map(HEADER.fullmatch, lines) if found}
    codes = tuple(header["codes"].split())
    if path.suffix == ".cli":
        command = [line for line in lines if line.strip() and not line.startswith("#")]
        assert len(command) == 1, f"{path.name}: a .cli case is one command line"
        return Case(path, codes, None, header["names"], tuple(command[0].split()))
    here = [number for number, line in enumerate(lines, start=1) if line.rstrip().endswith("# corpus: here")]
    assert len(here) == 1, f"{path.name}: one line carries '# corpus: here'"
    return Case(path, codes, here[0], None)


CASES = [read_case(path) for path in sorted([*CORPUS.glob("*.py"), *CORPUS.glob("*.cli")])]
#: The cases that read a record MIME packages (``mime.data``'s ``records/water.json``). On the platforms job's Windows
#: runner the record is refused before the planted mistake is reached (``record.file``: its pinned sha256, most likely
#: of a checkout that wrote the file with CRLF line ends), so there they test nothing of the corpus's.
READS_A_PACKAGED_RECORD = frozenset({"record_condition_unchecked", "record_without_conditions"})


def skip_where_untestable(case: Case) -> None:
    if sys.platform == "win32" and case.path.stem in READS_A_PACKAGED_RECORD:
        pytest.skip("on Windows the packaged record is refused as record.file before the planted mistake is reached")


def run_case(case: Case, tmp_path: Path, *, full: bool) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "MIME_TRACEBACK_DIR": str(tmp_path / "tracebacks")}
    environment.pop("MIME_TRACEBACK", None)
    if full and not case.cli:
        environment["MIME_TRACEBACK"] = "full"
    command = [sys.executable, "-m", "mime.cli", *case.words, *(["--traceback"] if full else [])] if case.cli \
        else [sys.executable, str(case.path)]
    return subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300,
                          check=False)


def test_the_corpus_holds_the_cases_the_brief_names():
    held = {case.path.stem for case in CASES}
    assert {"spacing_missing", "law_traced_branch", "wrong_unit", "misspelled_body", "unfed_input", "nan_large_dt",
            "record_without_conditions", "series_not_recorded", "law_wrong_shape", "bem_settings_missing",
            "gravity_none", "inertia_axes", "record_condition_unchecked", "value_unknown", "run_duration_bare",
            "load_missing_folder"} <= held


@pytest.mark.parametrize("case", CASES, ids=[case.path.name for case in CASES])
def test_a_planted_mistake_gives_a_short_coded_error_at_the_users_line(case: Case, tmp_path: Path):
    skip_where_untestable(case)
    done = run_case(case, tmp_path, full=False)
    shown = done.stderr
    assert done.returncode != 0, shown
    for code in case.codes:
        assert re.search(rf"\] {re.escape(code)}: ", shown), (code, shown)
    assert len(shown.strip().splitlines()) <= MOST_LINES, shown
    assert not LIBRARY_FRAME.search(shown) and not BELOW.search(shown), shown
    assert "error.uncoded" not in shown and "''s" not in shown, shown
    if case.cli:
        assert case.names is not None and case.names in shown and 'File "' not in shown, shown
    else:
        frame, at = f'File "{case.path}", line {case.here}', f"at: {case.path}, line {case.here}"
        assert frame in shown or at in shown, shown
        assert all(line.strip().startswith(f'File "{case.path}"') for line in shown.splitlines()
                   if line.strip().startswith('File "')), shown
    written = sorted((tmp_path / "tracebacks").glob("mime-*.txt"))
    assert len(written) == 1 and str(written[0]) in shown, shown
    chain = written[0].read_text(encoding="utf-8")
    assert LIBRARY_FRAME.search(chain) and all(code in chain for code in case.codes), chain


@pytest.mark.parametrize("case", CASES, ids=[case.path.name for case in CASES])
def test_the_control_with_the_full_traceback_the_frames_appear(case: Case, tmp_path: Path):
    skip_where_untestable(case)
    done = run_case(case, tmp_path, full=True)
    shown = done.stderr
    assert done.returncode != 0, shown
    assert LIBRARY_FRAME.search(shown), shown
    for code in case.codes:
        assert code in shown, (code, shown)
