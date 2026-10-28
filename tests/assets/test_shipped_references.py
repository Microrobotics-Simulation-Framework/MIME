"""The references MIME ships for the eleven wall tables: they parse, and they are the manifest the dataset was uploaded
with, hash for hash and path for path, pinned to one revision of the dataset. Nothing reads them yet (the tube's wall
kind will); the sizes are the files' byte counts, which Hugging Face's listing of that revision also gives.

The control: a reference whose hash differs by one digit from the manifest's is found, so the comparison can fail.
"""

import importlib.resources
import shutil
import subprocess
from pathlib import Path

import pytest

from mime import assets
from mime.build import BuildRefused

REVISION = "1f7b5a4e5a66c679ecd2a0b2c50cb57eb5a25cd6"
RESOLVE = f"https://huggingface.co/datasets/NickERoy/mime-wall-tables/resolve/{REVISION}/"
ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ("src/mime/data/assets/wall_tables.json", "src/mime/data/assets/wall_tables.SHA256SUMS")


def manifest() -> dict[str, str]:
    """The dataset's ``SHA256SUMS.txt`` as it was uploaded, by path in the dataset."""
    text = (importlib.resources.files("mime.data") / "assets" / "wall_tables.SHA256SUMS").read_text(encoding="utf-8")
    return {path: digest for digest, path in (line.split() for line in text.splitlines())}


def mismatches(refs: dict[str, assets.AssetRef]) -> list[str]:
    """Every difference between the shipped references and the manifest."""
    found: list[str] = []
    expected = manifest()
    by_path = {ref.locations[0].removeprefix(RESOLVE): ref for ref in refs.values()}
    if set(by_path) != set(expected):
        found.append(f"paths differ: {sorted(set(by_path) ^ set(expected))}")
    for path, digest in expected.items():
        ref = by_path.get(path)
        if ref is not None and ref.sha256 != digest:
            found.append(f"{path}: {ref.sha256}, and the manifest says {digest}")
    return found


def wall_tables() -> dict[str, assets.AssetRef]:
    return dict(assets.known())


def test_eleven_references_parse_and_match_the_manifest():
    refs = wall_tables()
    assert len(refs) == 11 and len(manifest()) == 11
    assert mismatches(refs) == []


def test_the_control_finds_a_changed_digit_and_a_missing_table():
    refs = wall_tables()
    name = "cylinder_wall_r40_R1.526"
    refs[name] = assets.AssetRef.from_record({**refs[name].to_record(), "sha256": "0" + refs[name].sha256[1:]})
    assert len(mismatches(refs)) == 1 and "r40/wall_R1.526.npz" in mismatches(refs)[0]
    del refs["cylinder_wall_r30_R6.667"]
    assert any("paths differ" in found for found in mismatches(refs))


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
