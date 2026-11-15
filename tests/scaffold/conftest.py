"""``mime scaffold``'s tests run in a private home (the sharing tests' ``homes.isolate``): the defaults it reads and writes,
the tokens it searches for and the acceptances it must not touch are in a folder of the test's own."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sharing"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from homes import isolate  # noqa: E402


@pytest.fixture(autouse=True)
def private_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = isolate(monkeypatch, tmp_path / "home")
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "default-cache"))
    yield home
