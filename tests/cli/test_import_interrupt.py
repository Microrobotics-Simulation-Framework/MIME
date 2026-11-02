"""Ctrl+C while the ``mime`` command is still importing MIME (JAX and MADDENING take about a second) ends it quietly, 130,
where it used to be Python's traceback of whatever import was under way (``mime/_import_guard.py``).

The press is made to land at a known point, not timed: the child starts with a ``sitecustomize`` that raises SIGINT in
itself the moment ``mime.units`` begins to import, which is after ``mime/__init__`` has installed the guard and before JAX
and MADDENING, so it is the same on a loaded machine and a fast one. The control is a process that is not the command
(``python -c "import mime"``) given the same press at the same point: it gets Python's traceback. POSIX only: Windows
delivers Ctrl+C to a console's process group and not to one process (``_platform_support.needs_posix_sigint``)."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from _platform_support import needs_posix_sigint

pytestmark = needs_posix_sigint

#: A process started in the background by a non-interactive shell inherits SIGINT ignored, and Python then leaves it so; the
#: line below is what Python's start gives a process run from a terminal, so that the control means the same everywhere.
SITECUSTOMIZE = '''
import signal, sys

signal.signal(signal.SIGINT, signal.default_int_handler)


class PressAt:
    def find_spec(self, name, path=None, target=None):
        if name == "mime.units":
            sys.meta_path.remove(self)
            signal.raise_signal(signal.SIGINT)
        return None

sys.meta_path.insert(0, PressAt())
'''


def pressed_during_import(tmp_path: Path, *words: str) -> subprocess.CompletedProcess[bytes]:
    (tmp_path / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    environment = {**os.environ, "JAX_PLATFORMS": "cpu",
                   "PYTHONPATH": os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH")]))}
    return subprocess.run([sys.executable, *words], capture_output=True, env=environment, timeout=120, check=False)


def test_ctrl_c_while_the_command_imports_ends_it_quietly_with_130(tmp_path):
    done = pressed_during_import(tmp_path, "-m", "mime.cli", "--help")
    assert done.returncode == 130, done.stderr.decode("utf-8", "replace")[-800:]
    assert done.stdout == b"" and done.stderr == b""


def test_the_control_the_same_press_in_a_process_that_is_not_the_command_is_pythons_traceback(tmp_path):
    done = pressed_during_import(tmp_path, "-c", "import mime")
    assert done.returncode != 130 and b"KeyboardInterrupt" in done.stderr, done.stderr.decode("utf-8", "replace")[-800:]


# What the guard is installed for: a process the interpreter was asked to run as the command, and nothing else.

#: A script that imports MIME and cleans up on Ctrl+C, as a user's would; it prints whether the guard was installed.
USER_SCRIPT = '''
import signal, sys, time
import mime
from mime import _import_guard
print("guard:", signal.getsignal(signal.SIGINT) is _import_guard._leave, flush=True)
try:
    print("ready", flush=True)  # inside the try: a press that lands as soon as this is read is already in its handling
    time.sleep(30)
except KeyboardInterrupt:
    print("CLEANUP RAN", flush=True)
'''


def python_handles_ctrl_c() -> None:
    """In a child about to start: Ctrl+C as Python's start gives a process run from a terminal (a job started in the
    background by a non-interactive shell inherits it ignored, and Python then leaves it so)."""
    signal.signal(signal.SIGINT, signal.default_int_handler)


def ran_and_pressed(argv: list[str], tmp_path: Path, *, press: bool = True) -> tuple[int, str]:
    import time

    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "HOME": str(tmp_path)}
    child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, cwd=tmp_path,
                             preexec_fn=python_handles_ctrl_c)
    assert child.stdout is not None
    held, deadline = "", time.monotonic() + 120
    while "ready" not in held and time.monotonic() < deadline:
        held += child.stdout.readline().decode("utf-8", "replace")
    if press:
        child.send_signal(signal.SIGINT)
    rest, _ = child.communicate(timeout=60)
    return child.returncode, held + rest.decode("utf-8", "replace")


def test_a_users_script_that_happens_to_be_called_mime_keeps_its_own_handling_of_ctrl_c(tmp_path):
    (tmp_path / "mime").write_text(USER_SCRIPT, encoding="utf-8")
    code, said = ran_and_pressed([sys.executable, str(tmp_path / "mime")], tmp_path)
    assert "guard: False" in said and "CLEANUP RAN" in said and code == 0, said


def test_the_same_script_given_the_command_s_words_as_its_own_arguments_keeps_it_too(tmp_path):
    (tmp_path / "tool.py").write_text(USER_SCRIPT, encoding="utf-8")
    code, said = ran_and_pressed([sys.executable, str(tmp_path / "tool.py"), "-m", "mime.cli"], tmp_path)
    assert "guard: False" in said and "CLEANUP RAN" in said and code == 0, said


def test_the_control_the_installed_command_s_script_and_python_dash_m_do_install_it(tmp_path):
    """The two tests above can fail: a file called ``mime`` whose text names ``mime.cli``, as the entry point's does, is the
    command, and the guard stands in it (``-m mime.cli`` is the first test of this file)."""
    probe = ("import signal, sys\nimport mime\nfrom mime import _import_guard\n"
             "print('guard:', signal.getsignal(signal.SIGINT) is _import_guard._leave, flush=True)\n"
             "print('ready', flush=True)\n")
    (tmp_path / "mime").write_text("# the console script's import, which is what names it\n# from mime.cli import main\n" + probe,
                                   encoding="utf-8")
    code, said = ran_and_pressed([sys.executable, str(tmp_path / "mime")], tmp_path, press=False)
    assert "guard: True" in said, said


@pytest.mark.parametrize(("held", "asked"), [
    (["python", "-m", "mime.cli", "run"], ("module", "mime.cli")),
    (["python", "-mmime.cli", "run"], ("module", "mime.cli")),
    (["python", "-I", "-X", "utf8", "-W", "ignore", "-m", "mime.cli"], ("module", "mime.cli")),
    (["python", "-Im", "mime.cli"], ("module", "mime.cli")),
    (["python", "-Wignore", "-Xdev", "-m", "mime.cli"], ("module", "mime.cli")),
    (["python", "tool.py", "-m", "mime.cli"], ("script", "tool.py")),
    (["python", "-u", "tool.py", "-m", "mime.cli"], ("script", "tool.py")),
    (["python", "-c", "import mime", "-m", "mime.cli"], ("code", "")),
    (["python", "--", "tool.py", "-m", "mime.cli"], ("script", "tool.py")),
    (["python", "-W", "-m", "x.py"], ("script", "x.py")),
    (["python"], None),
    (["python", "-i"], None),
])
def test_the_interpreters_own_options_are_told_from_the_programs_arguments(held, asked):
    from mime._import_guard import _asked_of_the_interpreter

    assert _asked_of_the_interpreter(held) == asked


def test_a_spawned_child_of_the_command_is_not_the_command(tmp_path):
    """A child process of ``multiprocessing`` inherits its parent's ``sys.argv`` (a script named ``mime``), but the
    interpreter was asked to run its own bootstrap, not the script: the guard is not installed in it."""
    (tmp_path / "mime").write_text('''
# from mime.cli import main
import multiprocessing as mp, signal


def child():
    import mime
    from mime import _import_guard
    print("child guard:", signal.getsignal(signal.SIGINT) is _import_guard._leave, flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    process = mp.Process(target=child)
    process.start()
    process.join()
''', encoding="utf-8")
    done = subprocess.run([sys.executable, str(tmp_path / "mime")], capture_output=True, text=True, timeout=120, check=False,
                          env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert "child guard: False" in done.stdout, done.stdout + done.stderr


def under_a_handler(handler):
    """``guard()`` and ``release()`` in this process with SIGINT holding ``handler`` first, as the command's start finds it;
    what it held while installed and after, the process's own handler put back."""
    from mime import _import_guard

    kept = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, handler)
    try:
        _import_guard.is_the_command, real = (lambda: True), _import_guard.is_the_command
        try:
            _import_guard.guard()
            installed = signal.getsignal(signal.SIGINT)
        finally:
            _import_guard.is_the_command = real
        _import_guard.release()
        return installed, signal.getsignal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, kept)


def test_the_guard_never_replaces_a_ctrl_c_that_was_ignored_and_release_leaves_it_so():
    installed, after = under_a_handler(signal.SIG_IGN)
    assert installed is signal.SIG_IGN and after is signal.SIG_IGN


def test_release_puts_back_exactly_the_handler_there_was():
    from mime import _import_guard

    def mine(signum, frame):
        pass

    installed, after = under_a_handler(mine)
    assert installed is _import_guard._leave and after is mine
    installed, after = under_a_handler(signal.default_int_handler)
    assert installed is _import_guard._leave and after is signal.default_int_handler


def test_a_command_started_with_ctrl_c_ignored_keeps_ignoring_it_through_main(tmp_path):
    """The real command's process: ``trap '' INT`` as a job started in the background by a script has, the script the
    console script's own text, ``main`` called for a command that installs no handler of its own. The control is the same
    start with Ctrl+C not ignored, where the guard stands before ``main`` and Python's handler after it."""
    (tmp_path / "mime").write_text('''
import signal, sys
from mime.cli import main
before = signal.getsignal(signal.SIGINT)
code = main(["cache", "clear"])
after = signal.getsignal(signal.SIGINT)
print("before ignored:", before is signal.SIG_IGN, "| before is the guard:", getattr(before, "__name__", "") == "_leave",
      "| after ignored:", after is signal.SIG_IGN, "| after is Python's:", after is signal.default_int_handler, file=sys.stderr)
''', encoding="utf-8")
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "HOME": str(tmp_path), "XDG_CACHE_HOME": str(tmp_path / ".cache"),
                   "USERPROFILE": str(tmp_path)}
    ignored = subprocess.run(["sh", "-c", f"trap '' INT; exec '{sys.executable}' '{tmp_path / 'mime'}'"], capture_output=True,
                             text=True, env=environment, timeout=120, check=False)
    assert ("before ignored: True | before is the guard: False | after ignored: True | after is Python's: False"
            in ignored.stderr), ignored.stderr
    heard = subprocess.run([sys.executable, str(tmp_path / "mime")], capture_output=True, text=True, env=environment,
                           timeout=120, check=False, preexec_fn=python_handles_ctrl_c)
    assert ("before ignored: False | before is the guard: True | after ignored: False | after is Python's: True"
            in heard.stderr), heard.stderr
