"""The reference MIME ships for the corrected table of the unit tube (MIME-ANO-117): it parses, it is pinned to the
revision it was uploaded at by a resolve URL, and its sha256 and size are the uploaded file's, held to a record kept
apart from the package (``uploaded_unit_tube.json``, the file as the coordinator verified it against Hugging Face's
``x-linked-etag`` and ``x-linked-size``). A wrong size is the costlier miss: ``locate`` would refuse a correct
download for ever.

The controls: a changed digit of the hash, a size one byte off and another revision, each planted, are found every run.
"""

import importlib.resources
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from mime import assets

NAME = "unit_tube_R0.997"
FILE = "unit_tube_tables.json"


def uploaded() -> tuple[str, dict[str, dict[str, str | int]]]:
    listing = json.loads((Path(__file__).resolve().parent / "uploaded_unit_tube.json").read_text(encoding="utf-8"))
    return listing["revision"], listing["files"]


def mismatches(ref: assets.AssetRef) -> list[str]:
    """Every difference between the shipped reference and the uploaded file's record."""
    revision, files = uploaded()
    resolve = f"https://huggingface.co/datasets/NickERoy/mime-wall-tables/resolve/{revision}/"
    found: list[str] = []
    if len(ref.locations) != 1 or not ref.locations[0].startswith(resolve):
        return [f"{ref.locations} is not the one resolve URL pinned to {revision}"]
    path = ref.locations[0].removeprefix(resolve)
    if path not in files:
        return [f"{path} is not the uploaded file, {sorted(files)}"]
    if ref.sha256 != files[path]["sha256"]:
        found.append(f"{path}: {ref.sha256}, and the uploaded file has {files[path]['sha256']}")
    if ref.size != files[path]["size"]:
        found.append(f"{path}: size {ref.size}, and the uploaded file has {files[path]['size']}")
    return found


def shipped() -> assets.AssetRef:
    return assets.reference(NAME)


def changed(**change: object) -> assets.AssetRef:
    return assets.AssetRef.from_record({**shipped().to_record(), **change})


def test_the_unit_tube_reference_is_the_uploaded_file_pinned_to_its_revision():
    ref = shipped()
    assert ref.name == NAME and ref.licence == "CC0-1.0" and ref.locations[0].endswith("/unit_tube_R0.997.npz")
    assert mismatches(ref) == []


def a_digit_changes() -> list[str]:
    digest = shipped().sha256
    return mismatches(changed(sha256=("0" if digest[0] != "0" else "1") + digest[1:]))


def a_size_changes_by_one_byte() -> list[str]:
    return mismatches(changed(size=shipped().size + 1))


def another_revision() -> list[str]:
    revision, _ = uploaded()
    return mismatches(changed(locations=[shipped().locations[0].replace(revision, "1" * 40)]))


@pytest.mark.parametrize("plant", [a_digit_changes, a_size_changes_by_one_byte, another_revision])
def test_every_plant_is_found_every_run(plant: Callable[[], list[str]]):
    for _ in range(20):
        assert plant(), plant.__name__


def test_the_package_ships_it_beside_the_eleven_and_labels_the_eleven_wrong():
    folder = importlib.resources.files("mime.data") / "assets"
    record = json.loads((folder / FILE).read_text(encoding="utf-8"))
    old = json.loads((folder / "wall_tables.json").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in record["assets"]] == [NAME]
    assert "MIME-ANO-117" in old["about"] and "read by nothing" in old["about"]
