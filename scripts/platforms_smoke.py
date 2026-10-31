"""The tests the platforms workflow runs on macOS and Windows: the files `tests/platforms.txt` names, one `path  # reason`
a line, run by pytest with any further arguments passed on. The list is the one place the selection is written
(`tests/test_platforms_selection.py` holds that every file it names exists), and this script is the only reader of it,
so a workflow step is the same on every platform's shell.

    uv run --no-sync python scripts/platforms_smoke.py -n auto --dist worksteal
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("tests/platforms.txt")


def selection(root: Path = ROOT) -> list[str]:
    """The test files named in the list, in its order; a line that is blank or a comment is skipped, and a trailing
    `# reason` is removed."""
    lines = (root / SELECTION).read_text(encoding="utf-8").splitlines()
    return [line.split("#", 1)[0].strip() for line in lines if line.split("#", 1)[0].strip()]


def main(arguments: list[str]) -> int:
    import pytest

    return int(pytest.main([*selection(), *arguments]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
