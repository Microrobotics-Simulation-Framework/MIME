"""``tests/process_groups.py``'s ``running``, which the live tests use to wait for a server to be gone: it answers for the
platform it runs on. On macOS it once answered ``False`` for every pid (it read ``/proc``, which is not there), so a
test of a server that must stop when its launcher is gone asserted before the server had written its run folder. The
macOS lane runs this file: its first assertion, a live process running, fails there with the old answer."""

import os
import subprocess
import sys
import time

from process_groups import running, stop_group


def test_running_says_a_live_process_runs_and_a_killed_one_unreaped_or_reaped_does_not():
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10 ** 6)"])
    try:
        assert running(sleeper.pid) and running(os.getpid())  # the control: a live process (``False`` for every pid fails here)
        sleeper.kill()
        deadline = time.monotonic() + 60
        while running(sleeper.pid):  # killed and not yet reaped: a zombie is not running
            assert time.monotonic() < deadline, "a killed process still runs"
            time.sleep(0.01)
        sleeper.wait(timeout=60)
        assert not running(sleeper.pid)  # reaped: gone
    finally:
        stop_group(sleeper)
