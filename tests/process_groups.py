"""Processes a test starts in a process group of their own, the stop that kills the whole group (the live tests'
servers, ``tests/live/conftest.py``'s ``launch``), and the question whether a process still runs."""

import os
import signal
import subprocess
import sys


def stop_group(process: subprocess.Popen) -> None:
    """Kill ``process`` and every process of its group, and reap it: the group's other members too when ``process``
    itself has exited (a launcher killed while its server serves on), since the group outlives its leader while any
    member lives."""
    if sys.platform == "win32":
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # the whole group is gone
            pass
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:  # a hang guard: SIGKILL is not refused
        pass


def running(pid: int) -> bool:
    """Whether the process ``pid`` still runs: an exited process that is not yet reaped (a zombie) does not, and nor
    does one that is gone. On Linux this is one look at ``/proc``: two looks (``os.kill(pid, 0)``, then the file) leave
    a gap in which the reaper takes a zombie away and the second look finds nothing, and a loaded machine opens that
    gap. It never raises for a process that exited."""
    if sys.platform == "win32":
        done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False)
        return str(pid) in done.stdout
    try:
        with open(f"/proc/{pid}/stat") as held:
            state = held.read().rpartition(") ")[2]
    except (FileNotFoundError, ProcessLookupError):  # reaped, or gone while it was read
        return False
    return not state.startswith("Z")
