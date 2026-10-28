"""``mime assets fetch <folder>`` stores exactly the assets an experiment folder's lock references.

Nothing in 0.3 records an asset in a lock yet, so the lock here is a real one (a saved toy experiment) with the
references added under the key the reading looks for; the day a build writes that key, the reading needs no change.
The lock with no key is a real lock as it is: no asset.
"""

import json
from pathlib import Path

import pytest

from mime import assets
from mime.assets import _known
from mime.build import BuildRefused
from mime.cli import main
from mime.saved import LOCK_FILE
from mime.testing.toys import coupled_toy
from assets_rig import Rig, reference


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()


@pytest.fixture(scope="module")
def saved_toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("toy")
    coupled_toy().build().save(folder)
    return folder


def with_assets(saved: Path, target: Path, *refs: assets.AssetRef) -> Path:
    """A copy of a saved folder's lock in ``target`` with ``refs`` recorded as the references."""
    lock = json.loads((saved / LOCK_FILE).read_text(encoding="utf-8"))
    lock["assets"] = [ref.to_record() for ref in refs]
    target.mkdir()
    (target / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True), encoding="utf-8")
    return target


def test_the_lock_file_name_is_the_saved_forms():
    assert assets.folder_assets.__module__ == "mime.assets._lock"
    from mime.assets._lock import LOCK_FILE as read

    assert read == LOCK_FILE


def test_a_lock_with_no_asset_key_references_none(saved_toy: Path):
    assert "assets" not in json.loads((saved_toy / LOCK_FILE).read_text(encoding="utf-8"))
    assert assets.folder_assets(saved_toy) == ()


def test_the_references_a_lock_names_are_read_in_order_and_once_each(saved_toy: Path, tmp_path: Path):
    first, second = reference(b"first" * 50, "https://example.org/a", name="first"), reference(
        b"second" * 50, "https://example.org/b", name="second")
    folder = with_assets(saved_toy, tmp_path / "folder", first, second, first)
    assert assets.folder_assets(folder) == (first, second)


@pytest.mark.parametrize("break_it", ["not json", "[]", '{"assets": 3}', '{"assets": [3]}',
                                      '{"assets": [{"name": "x"}]}'])
def test_an_unreadable_or_malformed_lock_is_refused(tmp_path: Path, break_it: str):
    (tmp_path / LOCK_FILE).write_text(break_it, encoding="utf-8")
    with pytest.raises(BuildRefused) as refused:
        assets.folder_assets(tmp_path)
    assert refused.value.codes() in (["asset.lock"], ["asset.reference"])


def test_a_folder_without_a_lock_is_refused(tmp_path: Path):
    with pytest.raises(BuildRefused) as refused:
        assets.folder_assets(tmp_path)
    assert refused.value.codes() == ["asset.lock"]


def test_fetching_a_folder_stores_exactly_its_assets(saved_toy: Path, tmp_path: Path, rig: Rig,
                                                    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    needed = b"needed by the folder" * 100
    other = b"known, but not needed" * 100
    rig.server.files.update({"/needed": needed, "/other": other})
    wanted = reference(needed, rig.server.url("/needed"), name="needed")
    unwanted = reference(other, rig.server.url("/other"), name="unwanted")
    held = {ref.name: ref for ref in (wanted, unwanted)}
    monkeypatch.setattr(assets, "known", lambda: held)
    monkeypatch.setattr(_known, "known", lambda: held)
    folder = with_assets(saved_toy, tmp_path / "fl9_water", wanted)
    assert main(["assets", "fetch", str(folder)]) == 0
    assert rig.server.requests == ["GET /needed"]
    assert assets.is_cached(wanted) and not assets.is_cached(unwanted)
    assert main(["assets", "fetch", "unwanted"]) == 0 and assets.is_cached(unwanted)
    assert "needed: hashed in full" in capsys.readouterr().out


def test_fetching_a_folder_that_references_nothing_says_so_and_asks_nothing(saved_toy: Path, rig: Rig,
                                                                           capsys: pytest.CaptureFixture[str]):
    assert main(["assets", "fetch", str(saved_toy)]) == 0
    assert "references no asset" in capsys.readouterr().out and rig.requests == 0


def test_a_folder_given_to_the_command_that_cannot_be_read_exits_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["assets", "fetch", str(tmp_path)]) == 2
    assert "asset.lock" in capsys.readouterr().err
