"""Every test file under ``tests/`` is collected. pytest skips some folder names by default (``build`` among them),
and a test file that is never collected looks, in a summary line, like one that passed."""

import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).parent


def test_every_test_file_is_collected():
    collected = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "", "-p",
                                "no:cacheprovider", str(TESTS)],
                               capture_output=True, text=True, cwd=TESTS.parent, check=False).stdout
    files = {line.split("::")[0] for line in collected.splitlines() if "::" in line}
    expected = {str(path.relative_to(TESTS.parent)) for path in TESTS.rglob("test_*.py")}
    assert expected and expected <= files, sorted(expected - files)
