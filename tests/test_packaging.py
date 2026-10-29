"""The built wheel carries what a consumer relies on: the PEP 561 marker (without it a checker following PEP 561
must ignore every annotation in MIME), the refusal table (``import mime`` reads it), the ``mime`` command, and the entry
point by which a load finds the toy namespace. The claim is about the distribution, so the test builds a real wheel and
reads its members."""

import hashlib
import os
import subprocess
import sys
import tarfile
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


@pytest.fixture(scope="module")
def sdist(tmp_path_factory: pytest.TempPathFactory) -> tarfile.TarFile:
    try:
        import hatchling  # noqa: F401
    except ImportError:
        if os.environ.get("CI"):
            pytest.fail("hatchling, the declared build backend, is not installed; the dev group provides it")
        pytest.skip("hatchling is not installed: uv sync installs the dev group")
    out = tmp_path_factory.mktemp("sdist")
    done = subprocess.run([sys.executable, "-c", "import sys; from hatchling.build import build_sdist; "
                                                 "print(build_sdist(sys.argv[1]))", str(out)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert done.returncode == 0, done.stderr
    (built,) = out.glob("*.tar.gz")
    return tarfile.open(built)


#: The surrogate's weights: in the 0.2 line a root ``.gitignore`` rule kept them out of every distribution.
WEIGHTS = "mime/data/mlp_cholesky_weights_v2.npz"
PINNED = "ac1fca5c109fa3e2f27520cbd0ee1c0ce5e85ac31a0b1e93e7b8085c68f0a345"


def test_the_wheel_ships_the_surrogate_weights_with_their_pinned_hash(wheel):
    assert hashlib.sha256(wheel.read(WEIGHTS)).hexdigest() == PINNED


def test_the_sdist_ships_the_surrogate_weights_with_their_pinned_hash(sdist):
    (member,) = [m for m in sdist.getmembers() if m.name.endswith(f"/src/{WEIGHTS}")]
    held = sdist.extractfile(member)
    assert held is not None and hashlib.sha256(held.read()).hexdigest() == PINNED


#: The wall tables' references: data the asset commands read, so the wheel and the sdist must carry them.
WALL_TABLE_REFERENCES = ("mime/data/assets/wall_tables.json", "mime/data/assets/unit_tube_tables.json")


@pytest.mark.parametrize("references", WALL_TABLE_REFERENCES)
def test_the_wheel_ships_the_wall_table_references(wheel, references):
    assert wheel.read(references) == (ROOT / "src" / references).read_bytes()


@pytest.mark.parametrize("references", WALL_TABLE_REFERENCES)
def test_the_sdist_ships_the_wall_table_references(sdist, references):
    (member,) = [m for m in sdist.getmembers() if m.name.endswith(f"/src/{references}")]
    held = sdist.extractfile(member)
    assert held is not None and held.read() == (ROOT / "src" / references).read_bytes()


#: The environment's pins and the Modal script's template: what ``mime env check`` and ``mime modal`` read.
ENVIRONMENT_FILES = ("mime/data/env/requirements-cpu.txt", "mime/data/env/requirements-cuda12.txt",
                     "mime/data/env/bases.json", "mime/data/env/uv.json", "mime/env/modal_app.txt")


@pytest.mark.parametrize("member", ENVIRONMENT_FILES)
def test_the_wheel_and_the_sdist_ship_the_environments_pins_byte_for_byte(wheel, sdist, member):
    assert wheel.read(member) == (ROOT / "src" / member).read_bytes()
    (held,) = [m for m in sdist.getmembers() if m.name.endswith(f"/src/{member}")]
    extracted = sdist.extractfile(held)
    assert extracted is not None and extracted.read() == (ROOT / "src" / member).read_bytes()


def test_the_wheel_ships_an_empty_py_typed(wheel):
    assert (ROOT / "src" / "mime" / "py.typed").read_bytes() == b""
    assert wheel.read("mime/py.typed") == b""


def test_the_wheel_ships_the_refusal_table_that_import_mime_reads(wheel):
    table = "mime/build/refusal_table.txt"
    assert wheel.read(table) == (ROOT / "src" / table).read_bytes()


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
