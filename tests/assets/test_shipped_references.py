"""The references MIME ships for the eleven wall tables: they parse, and they are the manifest the dataset was uploaded
with, hash for hash and path for path, pinned to one revision of the dataset. Nothing reads them yet (the tube's wall
kind will).

They are held to two records kept apart: the manifest packaged beside them, and ``uploaded_revision.json``, the uploaded
revision's listing (each file's sha256 and byte count) as the coordinator verified it. The package's two files could be
edited together and agree; the listing is a third copy, so a changed digit in either file, in both, or a changed size is
found. A wrong size is the costlier: ``locate`` would refuse a correct download for ever.

The controls: each of those edits is planted and must be found, every run.
"""

import importlib.resources
import json
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from mime import assets
from mime.build import BuildRefused

REVISION = "1f7b5a4e5a66c679ecd2a0b2c50cb57eb5a25cd6"
RESOLVE = f"https://huggingface.co/datasets/NickERoy/mime-wall-tables/resolve/{REVISION}/"
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ("src/mime/data/assets/wall_tables.json", "src/mime/data/assets/wall_tables.SHA256SUMS")


def manifest_text() -> str:
    return (importlib.resources.files("mime.data") / "assets" / "wall_tables.SHA256SUMS").read_text(encoding="utf-8")


def manifest(text: str | None = None) -> dict[str, str]:
    """The dataset's ``SHA256SUMS.txt`` as it was uploaded, by path in the dataset."""
    lines = (manifest_text() if text is None else text).splitlines()
    return {path: digest for digest, path in (line.split() for line in lines)}


def uploaded() -> dict[str, dict[str, str | int]]:
    """The uploaded revision's listing, recorded apart from the package: each file's sha256 (the dataset's
    ``SHA256SUMS.txt``, checked with ``sha256sum -c`` against the files that were uploaded) and its size in bytes. A
    change to the package's data is a change against this record, so one edit there cannot pass alone."""
    text = (Path(__file__).resolve().parent / "uploaded_revision.json").read_text(encoding="utf-8")
    listing = json.loads(text)
    assert listing["revision"] == REVISION
    return listing["files"]


def mismatches(refs: dict[str, assets.AssetRef], manifest_by_path: dict[str, str] | None = None) -> list[str]:
    """Every difference between the shipped references, the packaged manifest and the uploaded revision's listing."""
    found: list[str] = []
    expected = manifest() if manifest_by_path is None else manifest_by_path
    listing = uploaded()
    by_path = {ref.locations[0].removeprefix(RESOLVE): ref for ref in refs.values()}
    for label, paths in (("the manifest", set(expected)), ("the uploaded revision", set(listing))):
        if set(by_path) != paths:
            found.append(f"paths differ from {label}: {sorted(set(by_path) ^ paths)}")
    for path, ref in by_path.items():
        if path in expected and ref.sha256 != expected[path]:
            found.append(f"{path}: {ref.sha256}, and the manifest says {expected[path]}")
        if path in listing and ref.sha256 != listing[path]["sha256"]:
            found.append(f"{path}: {ref.sha256}, and the uploaded revision has {listing[path]['sha256']}")
        if path in listing and ref.size != listing[path]["size"]:
            found.append(f"{path}: size {ref.size}, and the uploaded revision has {listing[path]['size']}")
    for path, digest in expected.items():
        if path in listing and digest != listing[path]["sha256"]:
            found.append(f"{path}: the manifest says {digest}, and the uploaded revision has {listing[path]['sha256']}")
    return found


def wall_tables() -> dict[str, assets.AssetRef]:
    return dict(assets.known())


def changed(refs: dict[str, assets.AssetRef], name: str, **change: object) -> dict[str, assets.AssetRef]:
    return {**refs, name: assets.AssetRef.from_record({**refs[name].to_record(), **change})}


def test_eleven_references_parse_and_match_the_manifest_and_the_uploaded_revision():
    refs = wall_tables()
    assert len(refs) == 11 and len(manifest()) == 11 and len(uploaded()) == 11
    assert mismatches(refs) == []


NAME = "cylinder_wall_r40_R1.526"
PATH = "r40/wall_R1.526.npz"


def flipped(digest: str) -> str:
    return ("0" if digest[0] != "0" else "1") + digest[1:]


def the_json_alone_changes_a_digit() -> list[str]:
    refs = wall_tables()
    return mismatches(changed(refs, NAME, sha256=flipped(refs[NAME].sha256)))


def the_manifest_alone_changes_a_digit() -> list[str]:
    expected = manifest()
    expected[PATH] = flipped(expected[PATH])
    return mismatches(wall_tables(), expected)


def both_change_the_same_digit() -> list[str]:
    """The coordinated edit that a comparison of the package with itself cannot see."""
    refs = wall_tables()
    expected = manifest()
    expected[PATH] = flipped(expected[PATH])
    return mismatches(changed(refs, NAME, sha256=expected[PATH]), expected)


def a_size_changes_by_one_byte() -> list[str]:
    refs = wall_tables()
    return mismatches(changed(refs, NAME, size=refs[NAME].size + 1))


def a_table_is_missing() -> list[str]:
    refs = wall_tables()
    del refs["cylinder_wall_r30_R6.667"]
    return mismatches(refs)


@pytest.mark.parametrize("plant", [the_json_alone_changes_a_digit, the_manifest_alone_changes_a_digit,
                                   both_change_the_same_digit, a_size_changes_by_one_byte, a_table_is_missing])
def test_every_plant_is_found_every_run(plant: Callable[[], list[str]]):
    """The control: each plant fails the comparison, the coordinated edit and the size among them."""
    for _ in range(20):
        found = plant()
        assert found, plant.__name__
    assert "uploaded revision" in " ".join(both_change_the_same_digit())
    assert "size" in " ".join(a_size_changes_by_one_byte())


def test_the_packaged_data_is_the_uploaded_listing_to_the_byte():
    text = (importlib.resources.files("mime.data") / "assets" / "wall_tables.json").read_text(encoding="utf-8")
    sizes = {record["name"]: record["size"] for record in json.loads(text)["assets"]}
    assert sorted(sizes.values()) == sorted(entry["size"] for entry in uploaded().values())


def test_every_table_is_pinned_to_the_one_revision_by_a_resolve_url_with_one_location():
    for name, ref in wall_tables().items():
        assert name == ref.name and ref.licence == "CC0-1.0"
        assert len(ref.locations) == 1 and ref.locations[0].startswith(RESOLVE), ref.locations
        assert ref.locations[0].endswith(".npz") and ref.size > 500_000_000


def test_the_names_say_the_radial_knots_and_the_tube_radius():
    refs = wall_tables()
    for ref in refs.values():
        knots, radius = ref.name.removeprefix("cylinder_wall_").split("_")
        assert knots in {"r30", "r40"} and ref.locations[0].endswith(f"/{knots}/wall_{radius}.npz")


def test_an_unknown_name_is_refused_with_the_known_ones():
    with pytest.raises(BuildRefused) as refused:
        assets.reference("no_such_table")
    assert "asset.unknown" in str(refused.value) and "cylinder_wall_r40_R1.526" in str(refused.value)


@pytest.mark.parametrize("path", DATA)
def test_git_tracks_the_data_and_ignores_it_nowhere(path: str):
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", path], capture_output=True,
                             check=False)
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-v", "--no-index", path],
                             capture_output=True, check=False)
    assert tracked.returncode == 0 and ignored.returncode == 1, path
