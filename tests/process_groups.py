"""Processes a test starts in a process group of their own, and the stop that kills the whole group: the live tests'
servers (``tests/live/conftest.py``'s ``launch``)."""

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
