"""The suite cannot write the developer's own configuration: a test that saves defaults or a credential through the public
functions, with no home of its own, writes under the test's folder (``user_config_and_state_in_the_tests_own_folder`` in
``tests/conftest.py``), never under the home the shell has. An acceptance guard of the harness, not a law of MIME.

The "real" home here is a sentinel: a temporary folder set as ``HOME`` for the module, before the per-test fixture runs,
with every XDG variable out of the environment, so that without the fixture the config directory is the sentinel's
``.config``, exactly as on a developer's machine whose shell sets none. Linux and Windows only: ``platformdirs`` reads
``~/Library`` on macOS, which no variable but ``HOME`` moves."""

import os
import secrets
import sys
from collections.abc import Iterator
from pathlib import Path

import platformdirs
import pytest

from mime.credentials import _store, _token
from mime.push import _defaults, _state

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="platformdirs reads ~/Library on macOS; no variable moves it")

_XDG = ("XDG_CONFIG_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME")


@pytest.fixture(scope="module", autouse=True)
def the_real_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """A sentinel for the developer's home, set before any function-scoped fixture of the suite runs."""
    home = tmp_path_factory.mktemp("real-home")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HOME", str(home))
        patch.setenv("USERPROFILE", str(home))
        for variable in _XDG:
            patch.delenv(variable, raising=False)
        patch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
        for override in ("WIN_PD_OVERRIDE_APPDATA", "WIN_PD_OVERRIDE_LOCAL_APPDATA"):
            patch.setenv(override, str(home / "AppData"))
        yield home


def files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file()) if root.exists() else []


def test_defaults_and_credentials_written_through_the_public_functions_leave_the_real_home_untouched(the_real_home: Path):
    token = "tok" + secrets.token_urlsafe(30)
    written = [_defaults.write_defaults(licences={"code": "MIT"}, namespace="somebody"),
               _store.store("zenodo", _token.Token("zenodo", token))]
    assert files_under(the_real_home) == [], "a test wrote into the real home"
    assert not any(the_real_home in path.parents for path in written), f"{written} is under the real home"


def test_the_state_directory_is_the_tests_own_too(the_real_home: Path):
    assert the_real_home not in _state.state_root().parents


def test_the_config_directory_is_read_from_the_variable_the_fixture_set():
    """The resolution is still platformdirs': the fixture moves the variable, not the logic."""
    if sys.platform == "win32":
        pytest.skip("platformdirs reads Windows' overrides, not XDG's variables")
    assert Path(platformdirs.user_config_dir("mime")) == Path(os.environ["XDG_CONFIG_HOME"]) / "mime"
    assert _defaults.defaults_path() == Path(os.environ["XDG_CONFIG_HOME"]) / "mime" / _defaults.DEFAULTS_FILE
    assert _store.credentials_path() == Path(os.environ["XDG_CONFIG_HOME"]) / "mime" / _store.CREDENTIALS_FILE


def test_the_cache_directory_is_left_where_it_was(the_real_home: Path):
    """The XLA compilation cache is a deliberate speed-up in the user's cache directory; the fixture does not move it."""
    if sys.platform == "win32":
        pytest.skip("Windows keeps the cache under the one application-data folder the fixture moves")
    assert os.environ["XDG_CACHE_HOME"] == str(the_real_home / ".cache")
    assert Path(platformdirs.user_cache_dir("mime")) == the_real_home / ".cache" / "mime"
