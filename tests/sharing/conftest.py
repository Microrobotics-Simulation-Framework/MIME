"""The sharing tests build on the assets tests' local server (``tests/assets/assets_rig.py``), and keep the real home
and config directories out of reach: ``pack`` reads the stored tokens and the home directory, and ``pull`` writes the
cache."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assets"))

from assets_rig import Rig  # noqa: E402

from mime import auth  # noqa: E402


@pytest.fixture(autouse=True)
def private_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """``HOME`` and the config directory in a temporary folder, no token in the environment, and the assets cache and the
    network switches at their defaults (a test that wants the cache takes the ``rig``)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
    for variable in auth.SERVICES.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.delenv("MIME_OFFLINE", raising=False)
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "default-cache"))
    assert home in auth.credentials_path().parents
    yield home


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()
