"""A file a person copied into the cache directory (an air-gapped cluster's administrator, a lab moving tables by hand) is
accepted under the asset's own name, or the file name its location serves, when its full hash is the reference's.

Under the asset's own name (``cylinder_wall_r40_R2.500``, with or without the extension its locations' files have) the
file is this asset's alone, so any difference is refused as ``asset.placed-file``. Under the name a location serves
(``wall_R2.500.npz``) it may be another asset's, since ``r30/`` and ``r40/`` both hold that name: it is accepted when
its hash matches, refused when it has the right size and other bytes, and otherwise left alone. No file is deleted,
overwritten or renamed. Each scenario is a function of the rig; the controls run it under a plant twenty times and each
time it must fail.
"""

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
from assets_rig import Rig, reference, run_with_rig

from mime import assets
from mime.assets import _fetch
from mime.build import BuildRefused

PAYLOAD = b"a wall table, in miniature" * 1000
OTHER = bytes(reversed(PAYLOAD))
NAME = "cylinder_wall_r40_R2.500"
SERVED = "wall_R2.500.npz"


def ref_for(rig: Rig, payload: bytes = PAYLOAD, name: str = NAME, served: str = SERVED) -> assets.AssetRef:
    """The reference whose location serves ``served``; the server has the right bytes there (a request is a failure of
    the scenario, if it expects none)."""
    rig.server.files[f"/r40/{served}"] = payload
    return reference(payload, rig.server.url(f"/r40/{served}"), name=name)


def place(rig: Rig, name: str, payload: bytes) -> Path:
    rig.cache.mkdir(parents=True, exist_ok=True)
    path = rig.cache / name
    path.write_bytes(payload)
    return path


def untouched(path: Path, payload: bytes, inode: int) -> None:
    assert path.is_file() and path.read_bytes() == payload and path.stat().st_ino == inode, f"{path.name} was touched"


def codes_of(action: Callable[[], object]) -> list[str]:
    try:
        action()
    except BuildRefused as refused:
        return refused.codes()
    raise AssertionError("not refused")


# --- scenarios -------------------------------------------------------------------------------------------------------


def a_file_under_each_accepted_name_is_taken_without_a_request(rig: Rig) -> None:
    for name in (SERVED, NAME + ".npz", NAME):
        ref = ref_for(rig)
        path = place(rig, name, PAYLOAD)
        inode = path.stat().st_ino
        taken = assets.fetch(ref)
        assert taken == rig.cache / ref.sha256 and taken.read_bytes() == PAYLOAD
        assert rig.requests == 0, f"{name}: a request was made for a file that was there"
        assert assets.locate(ref) == taken and assets.is_cached(ref) and assets.check(ref) == "ok"
        untouched(path, PAYLOAD, inode)
        (rig.cache / ref.sha256).unlink()
        (rig.cache / (ref.sha256 + ".verified")).unlink()
        path.unlink()


def a_wrong_file_under_the_assets_own_name_is_refused_and_left(rig: Rig) -> None:
    for name in (NAME, NAME + ".npz"):
        for wrong in (OTHER, PAYLOAD[:-1]):  # the right size with other bytes; a copy still in progress
            ref = ref_for(rig)
            path = place(rig, name, wrong)
            inode = path.stat().st_ino
            assert codes_of(lambda: assets.fetch(ref)) == ["asset.placed-file"]
            assert codes_of(lambda: assets.resolve(ref)) == ["asset.placed-file"]
            assert rig.requests == 0 and not (rig.cache / ref.sha256).exists()
            untouched(path, wrong, inode)
            path.unlink()


def a_wrong_file_under_the_served_name_of_the_right_size_is_refused_and_left(rig: Rig) -> None:
    ref = ref_for(rig)
    path = place(rig, SERVED, OTHER)
    inode = path.stat().st_ino
    assert codes_of(lambda: assets.fetch(ref)) == ["asset.placed-file"]
    assert rig.requests == 0
    untouched(path, OTHER, inode)


def a_file_of_another_size_under_the_served_name_belongs_to_another_asset(rig: Rig) -> None:
    """``r30/wall_R2.500.npz`` and ``r40/wall_R2.500.npz`` are two assets; the file there is one of them."""
    r40 = ref_for(rig)
    other = b"the r30 table" * 300
    path = place(rig, SERVED, other)
    inode = path.stat().st_ino
    assert assets.fetch(r40).read_bytes() == PAYLOAD, "the other asset's file blocked this one"
    assert rig.requests == 1
    untouched(path, other, inode)


def an_offline_run_takes_a_placed_file(rig: Rig, patch: pytest.MonkeyPatch) -> None:
    patch.setenv("MIME_OFFLINE", "1")
    ref = ref_for(rig)
    place(rig, SERVED, PAYLOAD)
    assert assets.fetch(ref).read_bytes() == PAYLOAD and rig.requests == 0


def the_sha256_name_is_still_taken_first(rig: Rig) -> None:
    ref = ref_for(rig)
    path = place(rig, ref.sha256, PAYLOAD)
    place(rig, NAME, OTHER)  # a wrong file under the asset's name is never looked at once the sha256 file is right
    assert assets.fetch(ref) == path and rig.requests == 0


SCENARIOS: list[Callable[[Rig], None]] = [
    a_file_under_each_accepted_name_is_taken_without_a_request,
    a_wrong_file_under_the_assets_own_name_is_refused_and_left,
    a_wrong_file_under_the_served_name_of_the_right_size_is_refused_and_left,
    a_file_of_another_size_under_the_served_name_belongs_to_another_asset,
    the_sha256_name_is_still_taken_first,
]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    held = Rig(tmp_path, monkeypatch)
    yield held
    held.server.close()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.__name__)
def test_scenario(rig: Rig, scenario: Callable[[Rig], None]):
    scenario(rig)


def test_an_offline_run_takes_a_placed_file(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    an_offline_run_takes_a_placed_file(rig, monkeypatch)


def test_a_file_in_a_symlink_is_taken_and_the_link_left(rig: Rig):
    ref = ref_for(rig)
    real = rig.cache.parent / "elsewhere.npz"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_bytes(PAYLOAD)
    rig.cache.mkdir(parents=True, exist_ok=True)
    (rig.cache / SERVED).symlink_to(real)
    assert assets.fetch(ref).read_bytes() == PAYLOAD and rig.requests == 0
    assert (rig.cache / SERVED).is_symlink() and real.read_bytes() == PAYLOAD


def test_a_name_that_is_a_control_file_is_not_a_place_to_look(rig: Rig):
    """A location serving ``x.verified`` does not make MIME's own record file a candidate."""
    ref = reference(PAYLOAD, "https://example.org/r40/wall.verified", name=NAME)
    assert (SERVED, False) not in _fetch._names_a_person_may_use(ref)
    assert [name for name, _ in _fetch._names_a_person_may_use(ref)] == [NAME]


def test_the_names_looked_for(rig: Rig):
    ref = ref_for(rig)
    assert _fetch._names_a_person_may_use(ref) == [(NAME, True), (NAME + ".npz", True), (SERVED, False)]


# --- the controls ----------------------------------------------------------------------------------------------------


@contextmanager
def hash_not_checked() -> Iterator[None]:
    """Every file found under a name is taken as the reference's."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "_adopt_named_file", _lenient_adopt)
        yield


def _lenient_adopt(ref: assets.AssetRef) -> bool:
    for name, _ in _fetch._names_a_person_may_use(ref):
        path = _fetch.cache_dir() / name
        if path.is_file():
            _fetch._link_into_place(ref, path)
            _fetch.write_record(ref, _fetch.PLACED)
            return True
    return False


@contextmanager
def a_placed_file_replaced() -> Iterator[None]:
    """A wrong file under the asset's name is deleted and the download goes ahead."""
    with pytest.MonkeyPatch.context() as patch:
        def replacing(ref: assets.AssetRef) -> bool:
            for name, _ in _fetch._names_a_person_may_use(ref):
                (_fetch.cache_dir() / name).unlink(missing_ok=True)
            return False

        patch.setattr(_fetch, "_adopt_named_file", replacing)
        yield


@contextmanager
def only_the_sha256_name() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_fetch, "_names_a_person_may_use", lambda ref: [])
        yield


RUNS = 20


def planted_runs(tmp_path: Path, scenario: Callable[[Rig], None], plant: Callable[[], object]) -> int:
    detected = 0
    for run in range(RUNS):
        try:
            with cast("Iterator[None]", plant()):
                run_with_rig(scenario, tmp_path / str(run))
        except (AssertionError, BuildRefused, FileNotFoundError):
            detected += 1
    return detected


def test_the_scenarios_pass_unplanted_twenty_times(tmp_path: Path):
    for run in range(RUNS):
        for number, scenario in enumerate(SCENARIOS):
            run_with_rig(scenario, tmp_path / f"{run}-{number}")


def test_control_a_hash_not_checked_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_wrong_file_under_the_assets_own_name_is_refused_and_left, hash_not_checked) == RUNS


def test_control_a_wrong_file_replaced_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_wrong_file_under_the_assets_own_name_is_refused_and_left, a_placed_file_replaced) == RUNS


def test_control_no_name_but_the_sha256_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_file_under_each_accepted_name_is_taken_without_a_request, only_the_sha256_name) == RUNS


def test_control_the_served_name_taken_whatever_its_bytes_is_detected_every_run(tmp_path: Path):
    assert planted_runs(tmp_path, a_file_of_another_size_under_the_served_name_belongs_to_another_asset,
                        hash_not_checked) == RUNS
