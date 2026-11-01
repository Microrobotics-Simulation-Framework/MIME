"""The live tests' cleanup (``process_groups.stop_group``) stops a server whose launcher has already exited: the
launcher's process group outlives it while any member lives, and the whole group is killed. The control: before the
cleanup the launcher has exited and its child still runs, so a cleanup that looked at the launcher alone would have
left it."""

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest
from process_groups import stop_group

#: A guard against a hang, never a measure of speed.
HANG = 600


def alive(pid: int) -> bool:
    """Whether ``pid`` runs. One look at ``/proc``, because two looks (``os.kill(pid, 0)``, then the file) leave a gap
    in which the reaper takes a zombie away and the second look finds nothing: a loaded machine opens that gap."""
    try:
        with open(f"/proc/{pid}/stat") as held:
            state = held.read().rpartition(") ")[2]
    except (FileNotFoundError, ProcessLookupError):  # reaped, or gone while it was read
        return False
    return not state.startswith("Z")  # a zombie, exited and not yet reaped, is not alive


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc; the live tests run on Linux")
def test_a_group_whose_launcher_exited_is_stopped_whole():
    program = ("import subprocess, sys; child = subprocess.Popen([sys.executable, '-c', 'import time; "
               "time.sleep(10 ** 6)']); print(child.pid, flush=True)")
    launcher = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
    assert launcher.stdout is not None
    child = int(launcher.stdout.readline())
    try:
        launcher.wait(timeout=HANG)
        assert launcher.poll() is not None and alive(child)  # the control: the launcher gone, its child serving on
        stop_group(launcher)
        deadline = time.monotonic() + HANG
        while alive(child):
            assert time.monotonic() < deadline, "the launcher's child outlived the cleanup"
            time.sleep(0.05)
    finally:
        with contextlib.suppress(ProcessLookupError):  # gone between the look and the kill
            if alive(child):
                os.kill(child, signal.SIGKILL)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc; the live tests run on Linux")
def test_alive_tells_a_running_an_exited_unreaped_and_a_reaped_process_apart():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10 ** 6)"])
    try:
        assert alive(process.pid)
        process.kill()
        deadline = time.monotonic() + HANG
        while alive(process.pid):  # exited and not reaped (a zombie of ours) is not alive
            assert time.monotonic() < deadline, "a killed process stayed alive"
            time.sleep(0.01)
        assert os.path.exists(f"/proc/{process.pid}")  # a zombie still: the state answered, not an absence
        process.wait(timeout=HANG)
        assert not alive(process.pid)  # reaped
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=HANG)
