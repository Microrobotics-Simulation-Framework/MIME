"""The sharing tests build on the assets tests' local server (``tests/assets/assets_rig.py``), and keep the real home,
config, cache, data and state directories out of reach (``homes.py``): ``pack`` reads the stored tokens and the home
directory, ``pull`` writes the cache, and ``push`` keeps its records in the state directory, which a shell's
``XDG_STATE_HOME`` would otherwise name. Each module's tests, and its module-scoped fixtures, run in a home of the
module's; each test in a home of its own, apart from it; and a test that leaves a file in the module's state directory
fails, so a test whose isolation is lost is caught where it writes."""

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assets"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets_rig import Rig  # noqa: E402
from homes import Outside, isolate, outside  # noqa: E402

from mime import credentials  # noqa: E402
from mime.push._state import state_root  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def outside_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Outside]:
    """The module's home: where its module-scoped fixtures run, and where a test would write if its own isolation
    failed."""
    yield from outside(tmp_path_factory.mktemp("outside"))


@pytest.fixture(autouse=True)
def private_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outside_home: Outside) -> Iterator[Path]:
    """Every user directory in a temporary folder (``homes.isolate``), no token in the environment, and the assets
    cache and the network switches at their defaults (a test that wants the cache takes the ``rig``). After the test,
    the module's state directory must hold nothing: push's records were written in the test's own."""
    home = isolate(monkeypatch, tmp_path / "home")
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "default-cache"))
    assert home in credentials.credentials_path().parents
    assert home in state_root().parents
    yield home
    written = outside_home.written()
    assert not written, f"a test wrote outside its own state directory, into its module's: {written}"


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()
