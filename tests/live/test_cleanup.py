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
from process_groups import running, stop_group

#: A guard against a hang, never a measure of speed.
HANG = 600


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
        assert launcher.poll() is not None and running(child)  # the control: the launcher gone, its child serving on
        stop_group(launcher)
        deadline = time.monotonic() + HANG
        while running(child):
            assert time.monotonic() < deadline, "the launcher's child outlived the cleanup"
            time.sleep(0.05)
    finally:
        with contextlib.suppress(ProcessLookupError):  # gone between the look and the kill
            if running(child):
                os.kill(child, signal.SIGKILL)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc; the live tests run on Linux")
def test_running_tells_a_running_an_exited_unreaped_and_a_reaped_process_apart():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10 ** 6)"])
    try:
        assert running(process.pid)
        process.kill()
        deadline = time.monotonic() + HANG
        while running(process.pid):  # exited and not reaped (a zombie of ours) is not alive
            assert time.monotonic() < deadline, "a killed process stayed alive"
            time.sleep(0.01)
        assert os.path.exists(f"/proc/{process.pid}")  # a zombie still: the state answered, not an absence
        process.wait(timeout=HANG)
        assert not running(process.pid)  # reaped
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=HANG)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc; the live tests run on Linux")
def test_running_never_raises_for_a_process_that_exits_and_is_reaped_while_it_is_asked():
    """The control for the gap between two looks at ``/proc``: a thread reaps each process as the main thread asks
    ``running`` of it without pause, so the reap falls between any two syscalls of the question at some point."""
    import threading

    for _ in range(100):
        process = subprocess.Popen([sys.executable, "-c", "pass"])
        reaper = threading.Thread(target=process.wait)
        reaper.start()
        while reaper.is_alive():
            running(process.pid)  # raising is the failure
        reaper.join()
        assert not running(process.pid)
