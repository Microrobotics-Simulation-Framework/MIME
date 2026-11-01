"""The live tests' cleanup (``process_groups.stop_group``) stops a server whose launcher has already exited: the
launcher's process group outlives it while any member lives, and the whole group is killed. The control: before the
cleanup the launcher has exited and its child still runs, so a cleanup that looked at the launcher alone would have
left it."""

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
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    with open(f"/proc/{pid}/stat") as held:  # a zombie, exited and not yet reaped, is not alive
        return not held.read().split(") ")[-1].startswith("Z")


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
        if alive(child):
            os.kill(child, signal.SIGKILL)
