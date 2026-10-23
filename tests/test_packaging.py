"""The built wheel carries what a consumer relies on: the PEP 561 marker (without it a checker following PEP 561
must ignore every annotation in MIME), the ``mime`` command, and the entry point by which a load finds the toy
namespace. The claim is about the distribution, so the test builds a real wheel and reads its members."""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> zipfile.ZipFile:
    try:
        import hatchling  # noqa: F401
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("hatchling, the declared build backend, is not installed; the dev group provides it")
        pytest.skip("hatchling is not installed: uv sync installs the dev group")
    out = tmp_path_factory.mktemp("wheel")
    done = subprocess.run([sys.executable, "-c", "import sys; from hatchling.build import build_wheel; "
                                                 "print(build_wheel(sys.argv[1]))", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr
    (built,) = out.glob("*.whl")
    return zipfile.ZipFile(built)


def test_the_wheel_ships_an_empty_py_typed(wheel):
    assert (ROOT / "src" / "mime" / "py.typed").read_bytes() == b""
    assert wheel.read("mime/py.typed") == b""


def test_the_wheel_ships_the_package_and_not_only_the_marker(wheel):
    modules = [name for name in wheel.namelist() if name.startswith("mime/") and name.endswith(".py")]
    assert len(modules) > 40 and "mime/saved/_schema.py" in modules and "mime/testing/_fresh_load.py" in modules


def test_the_wheel_declares_its_licence_by_expression_and_ships_both_licence_files(wheel):
    """PEP 639: the licence is an SPDX expression and its files are listed and shipped, with no licence classifier
    beside them."""
    (metadata,) = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    headers = wheel.read(metadata).decode().partition("\n\n")[0].splitlines()
    assert "License-Expression: Apache-2.0" in headers
    assert [line for line in headers if line.startswith("License-File:")] == ["License-File: LICENSE",
                                                                              "License-File: NOTICE"]
    assert not [line for line in headers if line.startswith(("License:", "Classifier: License ::"))]
    licences = f"{metadata.rpartition('/')[0]}/licenses/"
    assert {name.removeprefix(licences) for name in wheel.namelist() if name.startswith(licences)} == {"LICENSE",
                                                                                                     "NOTICE"}
    assert wheel.read(licences + "LICENSE") == (ROOT / "LICENSE").read_bytes()


def test_the_wheel_declares_the_command_and_the_toy_namespace(wheel):
    (entry_points,) = [name for name in wheel.namelist() if name.endswith(".dist-info/entry_points.txt")]
    text = wheel.read(entry_points).decode()
    assert "[console_scripts]\nmime = mime.cli:main" in text
    assert "[mime.entries]\ntoy = mime.testing.toys" in text
