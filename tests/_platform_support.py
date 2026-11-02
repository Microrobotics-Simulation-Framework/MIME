"""What a test needs of its platform, said once, so that a skip names what it needs and ``docs/platforms.md`` can list it.

A test skipped here is a test that does not run on that platform; the platforms workflow reports every skip
(``-ra``), and the default lane, on Linux, skips none."""

import sys

import pytest

from mime.env._host import cpu_record

#: Off Linux the CPU's flags are unreadable (``/proc/cpuinfo``), and a device field either run could not read is never
#: equal to anything, so a load reports the environment as different from the one the lock was written in and a
#: verification cannot claim bits (``mime.saved._folder.environment_differences``).
needs_readable_cpu_flags = pytest.mark.skipif(
    cpu_record()["flags_sha256"] is None,
    reason="needs a readable CPU flags hash (read from /proc/cpuinfo, so Linux only): without one a load reports the "
           "environment as different from the lock's, and nothing is bit for bit; platforms route R1")

#: A stand-in for a program, written as a script with a shebang line and put first on ``PATH``.
needs_posix_script_launch = pytest.mark.skipif(
    sys.platform == "win32",
    reason="needs a script with a shebang line to start as a program (Windows starts only an .exe); platforms route R2")

#: A SIGINT sent to one child process. Windows delivers Ctrl+C to a console's process group (CTRL_C_EVENT), not to a
#: process, so the tests of ``mime run``'s Ctrl+C that send one to a child are POSIX only; the machinery they test is
#: also tested in process with ``signal.raise_signal``, which runs everywhere.
needs_posix_sigint = pytest.mark.skipif(
    sys.platform == "win32",
    reason="sends SIGINT to a child process, which Windows delivers to a console's process group and not to one process")
