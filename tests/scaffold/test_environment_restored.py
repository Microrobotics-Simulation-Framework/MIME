"""The scaffold tests' module-scoped fixtures leave the process's environment as they found it.

``clean_plugin_base`` and ``clean_plugin`` are first requested *inside* a test (``request.getfixturevalue``), after the
autouse ``private_home`` has set that test's own ``MIME_ASSETS_DIR``; the monkeypatch each opens takes that as the value
to restore. Their module-level teardown then put the first test's assets cache (an empty folder) back for every later
test of the pytest-xdist worker, and the first test that read the cache failed with ``asset.missing``: in CI, only in a
shard where one such test came after the module (``tests/saved/test_lock_assets.py``, in the shard of the usability bundle's
pull request; the module's tests had passed in every earlier shard layout).

The check runs the real ``conftest.py`` in a subprocess session with a stub ``scaffold_rig`` (what the fixtures build is not
the question), a module that requests the fixture in a test's body, and a later module in another folder that reads the
variable. **A control that fails every run:** the same session with the restoring fixture removed from the conftest's text
leaves the variable changed, deterministically, so the check can tell the two apart."""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFTEST = (HERE / "conftest.py").read_text(encoding="utf-8")
RESTORING = "module_environment_restored"

STUB = """
class PluginBase: ...
class PluginClean: ...
def plugin_base(root): return PluginBase()
def plugin_clean(base): return PluginClean()
"""
REQUESTS_IT = """
def test_first(request):
    request.getfixturevalue("clean_plugin_base")
"""
READS_IT = """
import os
def test_later():
    assert os.environ.get("MIME_ASSETS_DIR") == os.environ["EXPECTED_ASSETS_DIR"], os.environ.get("MIME_ASSETS_DIR")
"""


def session(root: Path, conftest: str) -> subprocess.CompletedProcess[str]:
    (root / "sharing").mkdir(parents=True)
    (root / "sharing" / "homes.py").write_text((HERE.parent / "sharing" / "homes.py").read_text(encoding="utf-8"),
                                               encoding="utf-8")
    for name, text in (("a_scaffold/conftest.py", conftest), ("a_scaffold/scaffold_rig.py", STUB),
                       ("a_scaffold/test_first.py", REQUESTS_IT), ("z_other/test_later.py", READS_IT)):
        (root / name).parent.mkdir(exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")
    held = {**os.environ, "MIME_ASSETS_DIR": str(root / "the-cache"), "EXPECTED_ASSETS_DIR": str(root / "the-cache"),
            "JAX_PLATFORMS": "cpu"}
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist",
                           "--rootdir", str(root), str(root / "a_scaffold"), str(root / "z_other")],
                          cwd=root, env=held, capture_output=True, text=True, check=False)


def without_the_restoring_fixture(text: str) -> str:
    start = text.index('@pytest.fixture(scope="module", autouse=True)')
    end = text.index("@pytest.fixture(autouse=True)")
    assert RESTORING in text[start:end]
    return text[:start] + text[end:]


def test_a_later_test_sees_the_environment_the_session_started_with(tmp_path: Path):
    done = session(tmp_path, CONFTEST)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_same_session_without_the_restoring_fixture_leaves_the_assets_cache_changed(tmp_path: Path):
    """The control: the check above can fail, and by the leak (the later test names the cache it found)."""
    done = session(tmp_path, without_the_restoring_fixture(CONFTEST))
    assert done.returncode != 0, "the planted leak was not seen: the check above cannot tell the two apart"
    assert "default-cache" in done.stdout, done.stdout
