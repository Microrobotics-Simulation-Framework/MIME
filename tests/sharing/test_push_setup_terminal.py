"""``mime push <folder> --setup`` on a real pseudo-terminal (POSIX; ``_platform_support.needs_pty``): the walk asked
through the display's own question path, each answer typed at the terminal, an invalid ORCID iD said and asked again,
the file written and the defaults saved in the config directory. ``test_push_setup.py`` holds the walk's rules; this
holds that a person at a terminal is asked them, and answers them there.

With standard output and standard error both redirected to a file (``> log 2>&1``), the walk asks on the controlling
terminal, and no question goes into the file (the review of #112, F6).

The control: the same command with no terminal for its standard input is refused (``push.setup``) and writes nothing.
"""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import tomllib

import pytest
from _platform_support import needs_pty
from _terminal_support import stripped
from homes import isolate
from push_cases import copy_of, saved_toy

ORCID = "0000-0002-1825-0097"
MISTYPED = "0000-0002-1825-0098"
#: What is waited for on the screen, and what is typed then: a new mime.toml, every field but the creators, the
#: keywords and the uploader skipped, an ORCID iD mistyped first, the defaults saved.
STEPS = (('use "full_toy"? [Y/n]', ""), ("description: a paragraph", ""), ('use "CC-BY-4.0"? [Y/n]', ""),
         ("creator 1, family name", "Carberry"), ("creator 1, given name", "Josiah"),
         ("creator 1, ORCID iD", MISTYPED), ("the ORCID iD's check digit is 8", ORCID),
         ("creator 1, affiliation", "Brown University"), ("creator 1, the affiliation's ROR", ""),
         ("creator 2, family name", ""), ("contributor 1, family name", ""),
         ("keywords, separated by commas", "microrobots, helices"), ("related work 1", ""),
         ("work to cite 1", ""), ("uploader, your name", "Carberry, Josiah"), ("uploader, your institution", "Brown University"),
         ("funder 1, its name", ""), ("conference, its name", ""), ("the record, public or restricted", ""),
         ("its files, public or restricted", ""), ("Save these as your defaults", "y"))


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return saved_toy(tmp_path_factory.mktemp("toy"))


def environment(home: Path) -> dict[str, str]:
    """The child's environment: this test's private home (``homes.isolate``'s variables), the CPU."""
    with pytest.MonkeyPatch.context() as patch:
        isolate(patch, home)
        held = dict(os.environ)
    held.update({"JAX_PLATFORMS": "cpu", "PYTHONIOENCODING": "utf-8"})
    for name in ("NO_COLOR", "TERM", "MIME_TRACEBACK"):
        held.pop(name, None)
    held["TERM"] = "xterm-256color"
    return held


#: Run first in the child, a session of its own: its standard input, the terminal, made its controlling terminal
#: (``/dev/tty``), then the command itself in its place. (A ``preexec_fn`` would run Python in a fork of this
#: multithreaded process.)
_CONTROLLED = ("import fcntl, os, sys, termios; fcntl.ioctl(0, termios.TIOCSCTTY, 0); "
               "os.execv(sys.argv[1], sys.argv[1:])")


def driven(argv: Sequence[str], *, env: dict[str, str], cwd: Path, steps: Sequence[tuple[str, str]],
           columns: int = 100, timeout: float = 180.0, output: Path | None = None) -> tuple[str, int, int]:
    """``argv`` on a terminal, each of ``steps``' answers typed once its words are on the screen after the last one's:
    what it wrote, its exit code, and how many answers were typed. With ``output``, standard output and standard error
    both go into that file, and the terminal is the child's controlling terminal and its standard input alone."""
    import pty

    from _terminal_support import _size  # pyright: ignore[reportPrivateUsage]

    master, slave = pty.openpty()
    _size(master, columns, 30)
    if output is None:
        child = subprocess.Popen(list(argv), stdin=slave, stdout=slave, stderr=slave, env=env, cwd=cwd, close_fds=True)
    else:
        with output.open("wb") as log:
            child = subprocess.Popen([sys.executable, "-c", _CONTROLLED, *argv], stdin=slave, stdout=log,
                                     stderr=subprocess.STDOUT, env=env, cwd=cwd, close_fds=True, start_new_session=True)
    os.close(slave)
    held, step, after, deadline = b"", 0, 0, time.monotonic() + timeout
    try:
        while True:
            if time.monotonic() > deadline:
                child.kill()
                raise TimeoutError(f"no {steps[step][0]!r} on the screen; it wrote:\n{stripped(held.decode())}")
            if select.select([master], [], [], 0.05)[0]:
                try:
                    more = os.read(master, 65536)
                except OSError:
                    break
                if not more:
                    break
                held += more
            elif child.poll() is not None:
                break
            text = stripped(held.decode("utf-8", "replace"))
            while step < len(steps) and steps[step][0] in text[after:]:
                after = text.index(steps[step][0], after) + len(steps[step][0])
                time.sleep(0.05)
                os.write(master, (steps[step][1] + "\r").encode("utf-8"))
                step += 1
        return held.decode("utf-8", "replace"), child.wait(), step
    finally:
        os.close(master)


@needs_pty
def test_setup_on_a_terminal_asks_each_field_there_and_writes_the_file_and_the_defaults(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    home = tmp_path / "child-home"
    env = environment(home)
    raw, code, typed = driven([sys.executable, "-m", "mime.cli", "push", str(folder), "--setup"], env=env,
                              cwd=tmp_path, steps=STEPS)
    screen = stripped(raw)
    assert code == 0 and typed == len(STEPS), screen
    held = tomllib.loads((folder / "mime.toml").read_text(encoding="utf-8"))["record"]
    assert held["creators"] == [{"family": "Carberry", "given": "Josiah", "orcid": ORCID,
                                 "affiliation": "Brown University"}]
    assert held["keywords"] == ["microrobots", "helices"] and held["title"] == "full_toy"
    defaults = home / ".config" / "mime" / "defaults.toml"
    assert defaults.is_file() and ORCID in defaults.read_text(encoding="utf-8")
    assert "a digit is mistyped" in screen and f"wrote {folder / 'mime.toml'}" in screen.replace("\n", "")


@needs_pty
def test_control_with_no_terminal_to_answer_at_setup_is_refused_and_writes_nothing(saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    done = subprocess.run([sys.executable, "-m", "mime.cli", "push", str(folder), "--setup"],
                          env=environment(tmp_path / "child-home"), cwd=tmp_path, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=180, check=False)
    assert done.returncode == 2 and "push.setup" in done.stderr and "--init" in done.stderr
    assert not (folder / "mime.toml").exists()


@needs_pty
def test_setup_with_both_outputs_in_a_file_asks_on_the_terminal_and_puts_no_question_in_the_file(
        saved: Path, tmp_path: Path):
    folder = copy_of(saved, tmp_path)
    home = tmp_path / "child-home"
    log = tmp_path / "log"
    raw, code, typed = driven([sys.executable, "-m", "mime.cli", "push", str(folder), "--setup"],
                              env=environment(home), cwd=tmp_path, steps=STEPS, output=log)
    screen, written = stripped(raw), log.read_text(encoding="utf-8")
    assert code == 0 and typed == len(STEPS), f"{screen}\n--- the file:\n{written}"
    assert 'use "full_toy"? [Y/n]' in screen and "Save these as your defaults" in screen
    assert "[Y/n]" not in written and "family name" not in written and "[y/N]" not in written, written
    assert f"wrote {folder / 'mime.toml'}" in written.replace("\n", "")
    held = tomllib.loads((folder / "mime.toml").read_text(encoding="utf-8"))["record"]
    assert held["creators"][0]["orcid"] == ORCID
