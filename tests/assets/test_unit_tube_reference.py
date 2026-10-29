"""The reference MIME ships for the corrected table of the unit tube (MIME-ANO-117): it parses, it is pinned to the
revision it was uploaded at by a resolve URL, and its sha256 and size are the uploaded file's, held to a record kept
apart from the package (``uploaded_unit_tube.json``, the file as the coordinator verified it against Hugging Face's
``x-linked-etag`` and ``x-linked-size``). A wrong size is the costlier miss: ``locate`` would refuse a correct
download for ever.

The shipped file also holds the table at the earlier truncation (1667 modes), superseded: its reciprocity misses 2.0e-7
at the wall's edge knots. It is kept as the record, pinned the same way, and read by nothing (a test searches the package).

The controls: a changed digit of the hash, a size one byte off and another revision, each planted, are found every run.
"""

import importlib.resources
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mime import assets

NAME, SUPERSEDED = "unit_tube_R0.997", "unit_tube_R0.997_1667_modes"
FILE = "unit_tube_tables.json"
RESOLVE = "https://huggingface.co/datasets/NickERoy/mime-wall-tables/resolve/"


def uploaded() -> dict[str, dict[str, str | int]]:
    return json.loads((Path(__file__).resolve().parent / "uploaded_unit_tube.json").read_text(encoding="utf-8"))["files"]


def mismatches(ref: assets.AssetRef) -> list[str]:
    """Every difference between the shipped reference and the uploaded file's record of its name."""
    (record,) = [(path, held) for path, held in uploaded().items() if held["name"] == ref.name]
    path, held = record
    found: list[str] = []
    expected = f"{RESOLVE}{held['revision']}/{path}"
    if list(ref.locations) != [expected]:
        return [f"{ref.locations} is not the one resolve URL {expected}, pinned to its revision"]
    if ref.sha256 != held["sha256"]:
        found.append(f"{path}: {ref.sha256}, and the uploaded file has {held['sha256']}")
    if ref.size != held["size"]:
        found.append(f"{path}: size {ref.size}, and the uploaded file has {held['size']}")
    return found


def shipped(name: str = NAME) -> assets.AssetRef:
    return assets.reference(name)


def changed(name: str = NAME, **change: object) -> assets.AssetRef:
    return assets.AssetRef.from_record({**shipped(name).to_record(), **change})


@pytest.mark.parametrize("name", [NAME, SUPERSEDED])
def test_the_unit_tube_references_are_the_uploaded_files_pinned_to_their_revisions(name):
    ref = shipped(name)
    assert ref.licence == "CC0-1.0" and ref.locations[0].endswith("/unit_tube_R0.997.npz")
    assert mismatches(ref) == []


def test_the_table_the_wall_kind_reads_is_the_rebuild_and_the_superseded_one_is_the_record():
    assert shipped().sha256 == "d2b4dbe4bf19caac61c74c5d4d2d77e57c6318c4f96a664a22221fa75151741c"
    assert "unit_tube_2026-10-29b/" in shipped().locations[0] and "unit_tube_2026-10-29/" in shipped(SUPERSEDED).locations[0]
    assert shipped().sha256 != shipped(SUPERSEDED).sha256


def test_nothing_in_the_package_reads_the_superseded_table():
    root = Path(__file__).resolve().parents[2] / "src" / "mime"
    readers = [path.name for path in root.rglob("*.py") if SUPERSEDED in path.read_text(encoding="utf-8")]
    assert readers == []


def a_digit_changes() -> list[str]:
    digest = shipped().sha256
    return mismatches(changed(sha256=("0" if digest[0] != "0" else "1") + digest[1:]))


def a_size_changes_by_one_byte() -> list[str]:
    return mismatches(changed(size=shipped().size + 1))


def another_revision() -> list[str]:
    revision = next(held["revision"] for held in uploaded().values() if held["name"] == NAME)
    return mismatches(changed(locations=[shipped().locations[0].replace(str(revision), "1" * 40)]))


@pytest.mark.parametrize("plant", [a_digit_changes, a_size_changes_by_one_byte, another_revision])
def test_every_plant_is_found_every_run(plant: Callable[[], list[str]]):
    for _ in range(20):
        assert plant(), plant.__name__


def test_the_package_ships_it_beside_the_eleven_and_labels_the_eleven_wrong():
    folder = importlib.resources.files("mime.data") / "assets"
    record = json.loads((folder / FILE).read_text(encoding="utf-8"))
    old = json.loads((folder / "wall_tables.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in record["assets"]] == [NAME, SUPERSEDED]
    assert "MIME-ANO-117" in old["about"] and "read by nothing" in old["about"]
