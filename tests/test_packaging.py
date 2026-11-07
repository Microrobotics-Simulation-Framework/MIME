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
                     "mime/data/env/requirements-optional.txt", "mime/data/env/bases.json", "mime/data/env/uv.json", "mime/env/modal_app.txt")


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


#: The licence files the wheel lists: MIME's, the copy of usd2gltf's (MIT), the inlined viewer's parts' and the one file
#: that holds the licences of the inlined Mermaid and of the packages its bundle can hold.
LICENCE_FILES = ["LICENSE", "NOTICE", "src/mime/experiment/_mermaid_lib/LICENSE.mermaid", "src/mime/viz/_gltf/LICENSE",
                 *(f"src/mime/viz/_viewer/LICENSE.{part}" for part in ("fflate", "gainmap-js", "lit", "model-viewer",
                                                                      "three"))]


def test_the_wheel_declares_its_licence_by_expression_and_ships_every_licence_file(wheel):
    """PEP 639: the licence is an SPDX expression (MIME's Apache-2.0, the MIT of the code derived from usd2gltf and of
    the viewer's three.js, fflate and gainmap-js, the BSD-3-Clause of its lit, the ISC and Unlicense of the parts of the
    inlined Mermaid's bundle) and its files are listed and shipped,
    with no licence classifier beside them."""
    (metadata,) = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
    headers = wheel.read(metadata).decode().partition("\n\n")[0].splitlines()
    assert "License-Expression: Apache-2.0 AND MIT AND BSD-3-Clause AND ISC AND Unlicense" in headers
    assert [line for line in headers if line.startswith("License-File:")] == [f"License-File: {name}"
                                                                              for name in LICENCE_FILES]
    assert not [line for line in headers if line.startswith(("License:", "Classifier: License ::"))]
    licences = f"{metadata.rpartition('/')[0]}/licenses/"
    assert {name.removeprefix(licences) for name in wheel.namelist() if name.startswith(licences)} == set(
        LICENCE_FILES)
    for name in LICENCE_FILES:
        assert wheel.read(licences + name) == (ROOT / name).read_bytes(), name
    for name in ("mime/viz/_gltf/LICENSE", "mime/viz/_viewer/model-viewer.min.js", "mime/viz/_viewer/LICENSE.lit",
                 "mime/experiment/_mermaid_lib/mermaid.min.js", "mime/experiment/_mermaid_lib/LICENSE.mermaid"):
        assert wheel.read(name) == (ROOT / "src" / name).read_bytes(), name


def test_the_wheel_declares_the_command_and_the_toy_namespace(wheel):
    (entry_points,) = [name for name in wheel.namelist() if name.endswith(".dist-info/entry_points.txt")]
    text = wheel.read(entry_points).decode()
    assert "[console_scripts]\nmime = mime.cli:main" in text
    assert "[mime.entries]\ntoy = mime.testing.toys" in text


#: The AR4's example description, shipped in the sdist alone: the URDF byte for byte as the 0.2 line's A0 read it
#: (its sha256 the one ``tests/differential`` holds after the header fix it documents), its MIT licence, and its
#: meshes' manifest. No mesh is in either distribution.
AR4_FOLDER = "examples/descriptions/ar4"
AR4_URDF_SHA256 = "851d2f4ba29f4a49709f0f286c91eabb3fa2d84af59624ff3581d322309ddbc4"
#: The sha256 of ``LICENSE`` in Annin Robotics' ``ar4_ros_driver`` at commit 6a3ebb11cedab12cd1d29b41d63aa270a008ab8b,
#: read from the upstream file: the URDF's MIT notice ships as the upstream text and copyright line, byte for byte.
AR4_LICENSE_SHA256 = "717977043020350ff6fbb8b1be5fd2b604dfe5056d74cebaaee580d028073d09"


def _sdist_member(sdist: tarfile.TarFile, path: str) -> bytes:
    (member,) = [m for m in sdist.getmembers() if m.name.endswith("/" + path)]
    held = sdist.extractfile(member)
    assert held is not None
    return held.read()


def test_the_sdist_ships_the_ar4s_description_with_its_mit_notice_and_the_wheel_does_not(wheel, sdist):
    urdf = _sdist_member(sdist, f"{AR4_FOLDER}/ar4.urdf")
    assert hashlib.sha256(urdf).hexdigest() == AR4_URDF_SHA256
    licence = _sdist_member(sdist, f"{AR4_FOLDER}/LICENSE").decode()
    assert hashlib.sha256(licence.encode()).hexdigest() == AR4_LICENSE_SHA256, "the upstream LICENSE, byte for byte"
    assert hashlib.sha256((ROOT / AR4_FOLDER / "LICENSE").read_bytes()).hexdigest() == AR4_LICENSE_SHA256
    assert licence.startswith("MIT License\n\nCopyright (c) 2021 Dexter Ong\n")
    assert "The above copyright notice and this permission notice shall be included" in licence
    assert b'"format": "mime.asset_manifest/2"' in _sdist_member(sdist, f"{AR4_FOLDER}/visuals.json")
    assert not [name for name in wheel.namelist() if "ar4" in name.lower()]


def test_neither_distribution_holds_a_file_that_may_not_be_redistributed_or_an_unrecorded_mesh(wheel, sdist,
                                                                                             tmp_path, monkeypatch):
    from mime import assets

    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    for built in (wheel.filename, sdist.name):
        assert built is not None
        report = assets.audit(str(built), manifests=[ROOT / AR4_FOLDER / "visuals.json"])
        assert report.findings == (), report.text()
        assert report.files > 100


def test_control_a_planted_copy_in_the_wheel_fails_the_guard(wheel, tmp_path, monkeypatch):
    """A made STL planted in a copy of the wheel, once with a record that forbids its redistribution and once with
    none: the guard finds both."""
    import json
    import struct

    from mime import assets

    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    made = b"made".ljust(80, b"\0") + struct.pack("<I", 2) + bytes(100)
    held = json.loads((ROOT / AR4_FOLDER / "visuals.json").read_text(encoding="utf-8"))
    held["assets"][0]["sha256"], held["assets"][0]["size"] = hashlib.sha256(made).hexdigest(), len(made)
    (tmp_path / "visuals.json").write_text(json.dumps(held), encoding="utf-8")
    planted = tmp_path / "planted.whl"
    assert wheel.filename is not None
    with zipfile.ZipFile(wheel.filename) as source, zipfile.ZipFile(planted, "w") as copy:
        for info in source.infolist():
            copy.writestr(info, source.read(info))
        copy.writestr("mime/data/visuals/link.STL", made)
    found = assets.audit(planted, manifests=[tmp_path / "visuals.json"]).findings
    assert [(each.path, each.kind) for each in found] == [("mime/data/visuals/link.STL", "referenced")]
    assert [each.kind for each in assets.audit(planted).findings] == ["unrecorded mesh"]
