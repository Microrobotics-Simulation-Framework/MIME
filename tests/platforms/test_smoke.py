"""What a user on macOS or native Windows meets, with nothing assumed about the platform: the `mime` command as the
installed console script starts it, its output through pipes (a redirected stream, where Python before 3.15 does not
default to UTF-8 on Windows), a saved folder in a directory whose name holds a space and a character outside ASCII, and
every text file a save or a run writes read as bytes.

The platforms workflow (``.github/workflows/platforms.yml``) runs this file with the selection in ``tests/platforms.txt``
on macOS and Windows; the default lane runs it on Linux. It builds free-space experiments only, so it needs no asset."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from mime.env._host import cpu_record
from mime.experiment import Experiment
from mime.results import RunResult
from mime.saved import DOCUMENT_FILE, LOCK_FILE, read_folder
from mime.testing.toys import full_toy

#: A folder name a user's machine can hold: a space, a letter the Windows ANSI code page (the default encoding of a
#: redirected stream) can encode (a micro sign) and one it cannot (a Greek capital omega).
FOLDER = "toy experiment \u00b5 \u03a9"
#: What a text file is, by suffix: the files a save or a run writes whose bytes a hash or a diff reads.
TEXT = (".json", ".yaml", ".yml", ".lock", ".usda", ".txt", ".urdf")


#: Whether this machine can read the CPU's flags (Linux), which a load needs to call an environment the same one.
FLAGS_READABLE = cpu_record()["flags_sha256"] is not None


def mime_command(*arguments: str, cwd: Path, **environment: str) -> subprocess.CompletedProcess[bytes]:
    """The `mime` console script installed beside this interpreter, run as a user's shell runs it: the environment is
    this process's, so no UTF-8 mode and no encoding is set for it that a user's machine would not set (``environment``
    adds a variable on purpose, for a test that plants one)."""
    script = shutil.which("mime", path=str(Path(sys.executable).parent))
    assert script is not None, f"no `mime` console script beside {sys.executable}"
    held = {**os.environ, "MIME_OFFLINE": "1", "MIME_ASSETS_DIR": str(cwd / "assets"), **environment}
    return subprocess.run([script, *arguments], capture_output=True, cwd=cwd, env=held, check=False)


def said(run: subprocess.CompletedProcess[bytes]) -> str:
    """What a run printed, for an assertion's message."""
    return f"exit {run.returncode}\nstdout:\n{run.stdout.decode('utf-8', 'replace')}\nstderr:\n{run.stderr.decode('utf-8', 'replace')}"


def text_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix in TEXT)


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> Path:
    folder = tmp_path_factory.mktemp("platform smoke") / FOLDER
    full_toy().save(folder)
    return folder


def test_the_console_script_runs_and_lists_the_registry_through_a_pipe(tmp_path):
    run = mime_command("registry", cwd=tmp_path)
    assert run.returncode == 0 and run.stdout, said(run)
    assert b"Traceback" not in run.stderr, said(run)


def test_the_environment_check_names_only_what_nothing_pins(tmp_path):
    """The check is for a runner's exact environment, so a development install is refused for its tools (pytest, the
    lab fixture): every reason must be that one. Another reason (a pin that does not apply here, a marker that does not
    evaluate) is what this test is for."""
    run = mime_command("env", "check", "--variant", "cpu", cwd=tmp_path)
    output = run.stdout.decode("utf-8", "replace")
    assert run.returncode in (0, 2) and b"Traceback" not in run.stderr, said(run)
    reasons = [line for line in output.splitlines() if line.startswith("[")]
    assert all(line.startswith("[environment] env.unpinned:") for line in reasons), said(run)


def test_a_saved_experiment_loads_and_verifies_in_the_library_and_in_the_command(saved, tmp_path):
    """Reproduced everywhere; bit for bit only where the environment can be called the same one, which needs the CPU's
    flags (``test_off_linux_the_only_difference_from_the_environment_a_save_was_made_in_is_the_unreadable_cpu_flags``)."""
    experiment = Experiment.load(saved)
    assert experiment.build().verify().reproduced
    run = mime_command("verify", str(saved), cwd=tmp_path)
    assert run.returncode == 0 and b"reproduced" in run.stdout, said(run)
    assert (b"bit for bit" in run.stdout) is FLAGS_READABLE, said(run)


def test_off_linux_the_only_difference_from_the_environment_a_save_was_made_in_is_the_unreadable_cpu_flags(saved):
    """The diagnosis, held rather than assumed: the save and the load are one process and one machine, so the
    environment differs in nothing a load can read, and in the flags only where they cannot be read."""
    report = Experiment.load(saved).load_report
    assert report is not None
    if FLAGS_READABLE:
        assert report.same_environment and report.environment == ()
    else:
        assert not report.same_environment
        assert len(report.environment) == 1 and report.environment[0].startswith("cpu.flags_sha256:"), report.environment
        assert "unreadable" in report.environment[0]


def test_a_run_through_the_command_writes_a_folder_the_library_reads_and_the_command_verifies(saved, tmp_path):
    dt = read_folder(saved).lock["resolved"]["run"]["dt"]
    out = tmp_path / FOLDER / "run"
    run = mime_command("run", str(saved), "--duration", repr(4 * dt), "--out", str(out), "--history", cwd=tmp_path)
    assert run.returncode == 0, said(run)
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["steps"] == 4
    assert RunResult.read(out).steps == 4
    again = mime_command("verify", str(out / "experiment"), cwd=tmp_path)
    assert again.returncode == 0 and b"reproduced" in again.stdout, said(again)


def test_mime_pending_prints_what_is_pending_through_the_command_for_a_folder_and_its_document(saved, tmp_path):
    """`mime pending` through the console script and a pipe, for the saved toy's folder and for its document as it is:
    what the library prints, through the one formatter, and the exit code that says whether anything is pending (the
    toy's entries declare no roles, so their written defaults are)."""
    report = Experiment.load(saved).build().report
    expected, code = report.pending(), 1 if report.profile is not None and report.profile.pending else 0
    for target in (saved, saved / "experiment.mime.yaml"):
        run = mime_command("pending", str(target), cwd=tmp_path)
        assert run.returncode == code and b"Traceback" not in run.stderr, said(run)
        assert run.stdout.decode("utf-8").replace("\r\n", "\n") == expected, said(run)


def test_a_save_writes_the_same_bytes_on_every_platform(saved):
    """A saved folder is hashed, packed and compared as bytes, and a bundle made on one machine is read on another. A
    text file written through the platform's newline translation holds a carriage return on Windows and none on
    Linux, so one experiment saved in two places would be two different folders."""
    files = text_files(saved)
    assert {DOCUMENT_FILE, LOCK_FILE} <= {path.name for path in files}
    assert [path.name for path in files if b"\r" in path.read_bytes()] == []


def test_a_run_writes_the_same_bytes_on_every_platform(saved, tmp_path):
    dt = read_folder(saved).lock["resolved"]["run"]["dt"]
    out = tmp_path / "run"
    assert mime_command("run", str(saved), "--duration", repr(2 * dt), "--out", str(out), cwd=tmp_path).returncode == 0
    files = text_files(out)
    assert {"run.json", "outputs.json"} <= {path.name for path in files}
    assert [path.relative_to(out).as_posix() for path in files if b"\r" in path.read_bytes()] == []


def test_a_report_that_a_narrow_stream_cannot_encode_is_written_with_the_escape_not_raised(saved, tmp_path):
    """The folder's name is in the load report, and its omega is not in the ANSI code page: on Windows the command died
    after its work with a UnicodeEncodeError. Here the narrow stream is planted on every platform (ASCII), so this fails
    before the fix wherever it runs."""
    run = mime_command("verify", str(saved), cwd=tmp_path, PYTHONIOENCODING="ascii", PYTHONUTF8="0")
    assert run.returncode == 0 and b"UnicodeEncodeError" not in run.stderr, said(run)
    assert b"\\u03a9" in run.stdout, said(run)


def test_a_path_on_another_drive_is_shown_as_it_is(monkeypatch, tmp_path):
    """Windows has no relative path between drives and raises: the temporary directory is on C: and the checkout on D: on
    GitHub's runners. The raise is planted, so this fails before the fix on every platform."""
    from mime.env._modal_script import _shown

    def across_drives(path, start=None):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(os.path, "relpath", across_drives)
    script = tmp_path / "toy-modal" / "modal_app.py"
    assert _shown(script) == str(script)
