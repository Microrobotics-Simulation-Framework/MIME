"""``mime pack``'s verdicts (designs #198, #199): **a file that may not be redistributed never travels inside a bundle,
whatever its size**. It travels as a reference to its origin, which the lock names; a file that must and gives no
origin is refused (``bundle.no-location``); a declared permission that grants redistribution lets it travel; a file
over the size a bundle carries travels as a reference where its terms give an origin.

The property draws saved folders with files of every verdict, at real sizes (a few hundred bytes to two megabytes, the
range of a robot's visual meshes) and with real SPDX ids, **some of them copies of an earlier file's bytes at another
path** (with no terms, with the original's, or with terms of their own), and holds the bundle's stored bytes to
``terms_cases``' table: every file that table says may not travel is absent from the bundle, by sha256 and by its bytes,
and named as a reference; every other file is carried, byte for byte. Terms bind by path and a permission by sha256, so
two copies declared with different terms are refused at save (``terms.file``), and a carried copy of bytes that
travel as a reference is refused by ``pack`` (``bundle.referenced-copy``), whatever terms the copy has. The reference
outputs of the toy are run once and reused by every save here (``same_reference_run``): they are a run of the toy's
graph, which no layer, licence or terms changes, and nothing here reads them. Every example is a real save and a real
pack, which is why the property is on the duration allowlist rather than cut: its examples are the folders it draws.

The control plants the error the property guards against, a pack that carries every file, and the scenario must catch
it on every one of twenty runs; another plants a save that writes no reference into the lock, which ``pack`` must catch
by judging the terms itself; a third, a pack that never compares a carried file's bytes with a referenced file's, must
be caught by the copy cases on every run; a fourth, a pack that trusts the licence and locations a lock states for a
reference's asset (the lock is outside the hashed record), must be caught by a lock rewritten to call a restricted
file's asset CC0-1.0 or to move it to another location, on every run.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from terms_cases import (
    CONDITIONAL_IDS,
    NO_REDISTRIBUTION,
    OPEN_IDS,
    UNKNOWN_IDS,
    layered,
    same_reference_run,
    write_mesh,
)

from mime.assets import known
from mime.build import BuildRefused
from mime.saved import _bundle, _record, _terms, pack, unpack
from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

ORIGIN = "https://example.org/upstream/6a3ebb1/{}"
UNKNOWN_CUSTOM = CustomLicence("LicenseRef-Somebody-Else", "Somebody Else's Terms", "https://example.org/else")


@dataclass(frozen=True)
class Spec:
    """One file of a drawn folder: its kind of licence (``yours`` for none), the licence, its size, whether its terms
    give an origin, whether a permission granting redistribution names its bytes, and, for a copy, the index of the
    earlier file whose bytes it holds and whose terms it was given (``same``), or not (its own, or none)."""

    kind: str
    licence: str | CustomLicence | None
    size: int
    origin: bool
    permitted: bool
    copy_of: int | None = None
    same_terms: bool = False


def original(files: list[Spec], index: int) -> int:
    """The index of the file whose bytes ``files[index]`` holds: itself, or the first of its copies' chain."""
    while files[index].copy_of is not None:
        index = cast(int, files[index].copy_of)
    return index


def terms_of(files: list[Spec], index: int, paths: list[Path]) -> Terms | None:
    """The terms ``files[index]`` is declared with: a copy given the same terms takes its original's, origin and all."""
    spec = files[index]
    if spec.same_terms and spec.copy_of is not None:
        return terms_of(files, spec.copy_of, paths)
    if spec.licence is None:
        return None
    return Terms(spec.licence, "Robot Maker Ltd", "presentation",
                 (ORIGIN.format(paths[index].name),) if spec.origin else ())


def travels(files: list[Spec], index: int) -> bool:
    """This file's own reading of whether a file's bytes may travel in a bundle: its terms let them, or a permission
    names them (bound by sha256, so it covers every copy declared with terms)."""
    spec = files[index]
    if spec.same_terms and spec.copy_of is not None:
        return travels(files, spec.copy_of)
    permitted = any(files[each].permitted for each in range(len(files))
                    if original(files, each) == original(files, index))
    return spec.kind in ("yours", "open", "conditional") or (permitted and spec.licence is not None)


def expected_refusals(files: list[Spec], paths: list[Path]) -> tuple[str, set[str]]:
    """What this file's reading says a save and a pack of the drawn folder refuse: ``("save", codes)`` where two copies
    of one file's bytes carry different terms; else ``("pack", codes)``, empty where the pack goes through."""
    groups: dict[int, list[int]] = {}
    for index in range(len(files)):
        groups.setdefault(original(files, index), []).append(index)
    for members in groups.values():
        declared = [terms_of(files, each, paths) for each in members]
        if len({each for each in declared if each is not None}) > 1:
            return "save", {"terms.file"}
    codes: set[str] = set()
    for index in range(len(files)):
        terms = terms_of(files, index, paths)
        if not travels(files, index) and terms is not None and not terms.origin:
            codes.add("bundle.no-location")
    for members in groups.values():
        moving = [travels(files, each) for each in members]
        if any(moving) and not all(moving):
            codes.add("bundle.referenced-copy")
    return "pack", codes


licences = st.one_of(
    st.tuples(st.just("open"), st.sampled_from(OPEN_IDS)),
    st.tuples(st.just("conditional"), st.sampled_from(sorted(CONDITIONAL_IDS))),
    st.tuples(st.just("unknown"), st.sampled_from(UNKNOWN_IDS)),
    st.tuples(st.just("unknown"), st.just(UNKNOWN_CUSTOM)),
    st.tuples(st.just("not permitted"), st.just(NO_REDISTRIBUTION)),
    st.tuples(st.just("yours"), st.none()))
#: Real sizes, log-uniform from a stub layer to a large single mesh.
sizes = st.integers(min_value=8, max_value=21).map(lambda power: 2 ** power)


#: How a copy of an earlier file's bytes is declared: with no terms (the author's own, as a copied part usually is),
#: with its original's terms, or with terms of its own.
COPY_TERMS = ("none", "same", "own")


@st.composite
def specs(draw: st.DrawFn) -> list[Spec]:
    """One to four files, each after the first a copy of an earlier one's bytes about half the time, declared one of
    the ways of ``COPY_TERMS``, each as likely."""
    count = draw(st.integers(min_value=1, max_value=4))
    found: list[Spec] = []
    for index in range(count):
        copy_of = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=index - 1))) if index else None
        declared = draw(st.sampled_from(COPY_TERMS)) if copy_of is not None else "own"
        kind, licence = ("yours", None) if declared == "none" else draw(licences)
        restricted = kind in ("unknown", "not permitted")
        found.append(Spec(kind, licence, draw(sizes), draw(st.booleans()),
                          draw(st.booleans()) if restricted and copy_of is None else False, copy_of,
                          declared == "same"))
    return found


def saved(root: Path, files: list[Spec], *, licence: str | None = "CC-BY-4.0") -> tuple[Path | None, list[Path]]:
    """The full toy with ``files`` as mesh layers, saved; the folder (``None`` where the save is refused, as
    ``expected_refusals`` says it must be) and each layer's source file."""
    paths: list[Path] = []
    for index, spec in enumerate(files):
        path = write_mesh(root / "sources", f"Mesh{index}", spec.size)
        if spec.copy_of is not None:
            path.write_bytes(paths[original(files, index)].read_bytes())
        paths.append(path)
    layers = [(path, terms_of(files, index, paths)) for index, path in enumerate(paths)]
    permissions = [Permission("Ada Maker, Robot Maker Ltd", "A. Lab", (hashlib.sha256(path.read_bytes()).hexdigest(),),
                              Scope(True, True, True), "2026-11-20",
                              Evidence("public_statement", "https://example.org/statement"))
                   for path, spec in zip(paths, files, strict=True) if spec.permitted]
    folder = root / "folder"
    built = layered("drawn", layers, licence=licence, permissions=permissions).build()
    stage, codes = expected_refusals(files, paths)
    if stage == "save":
        try:
            built.save(folder)
        except BuildRefused as refused:
            assert set(refused.codes()) == codes, (files, refused.codes())
            return None, paths
        raise AssertionError((files, "two copies of one file's bytes with different terms were saved"))
    built.save(folder)
    return folder, paths


def pack_or_codes(folder: Path, out: Path) -> tuple[Path | None, set[str]]:
    try:
        return pack(folder, out=out), set()
    except BuildRefused as refused:
        return None, set(refused.codes())


def pack_holds_the_tiers(root: Path, files: list[Spec]) -> None:
    """The scenario: save, pack, and hold the bundle's bytes to the table; then unpack, and hold the folder to it."""
    folder, paths = saved(root, files)
    if folder is None:
        return
    _, expected = expected_refusals(files, paths)
    bundle, codes = pack_or_codes(folder, root / "drawn.mimebundle")
    assert codes == expected, (files, codes, expected)
    if bundle is None:
        assert not (root / "drawn.mimebundle").exists()
        return
    stored = bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    member_hashes = {hashlib.sha256(data).hexdigest() for data in members.values()}
    for index, path in enumerate(paths):
        layer = f"scene/layers/{path.name}"
        data = path.read_bytes()
        spec = files[index]
        if travels(files, index):
            assert members.get(f"experiment/{layer}") == data, (spec, "a file that may travel was not carried")
            assert layer not in manifest["referenced"]
        else:
            assert hashlib.sha256(data).hexdigest() not in member_hashes, (spec, "a file that may not travel is inside")
            assert data not in stored, (spec, "the bytes of a file that may not travel are in the bundle")
            assert manifest["referenced"][layer]["sha256"] == hashlib.sha256(data).hexdigest()
    unpacked = unpack(bundle, root / "unpacked")
    absent = sorted(f"scene/layers/{path.name}" for index, path in enumerate(paths) if not travels(files, index))
    assert sorted(unpacked.missing) == absent and unpacked.placed == ()
    assert all(not (root / "unpacked" / each).exists() for each in absent)


@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(specs())
def test_no_file_that_may_not_be_redistributed_travels_in_a_bundle(tmp_path_factory: pytest.TempPathFactory,
                                                                    files: list[Spec]):
    root = tmp_path_factory.mktemp("drawn")
    with same_reference_run():
        pack_holds_the_tiers(root, files)
    shutil.rmtree(root, ignore_errors=True)


def test_a_tiny_file_that_may_not_be_redistributed_is_a_reference_and_a_large_open_one_is_carried(tmp_path: Path):
    with same_reference_run():
        pack_holds_the_tiers(tmp_path, [Spec("not permitted", NO_REDISTRIBUTION, 300, True, False),
                                        Spec("open", "CC0-1.0", 2 ** 21, False, False)])


def test_a_permission_lets_a_file_travel_and_only_the_bytes_it_names(tmp_path: Path):
    with same_reference_run():
        pack_holds_the_tiers(tmp_path, [Spec("not permitted", NO_REDISTRIBUTION, 4000, True, True),
                                        Spec("not permitted", NO_REDISTRIBUTION, 5000, True, False)])


def test_a_file_that_must_be_a_reference_and_has_no_origin_is_refused(tmp_path: Path):
    with same_reference_run():
        pack_holds_the_tiers(tmp_path, [Spec("unknown", "MPL-2.0", 4000, False, False)])


def test_a_large_file_with_an_origin_travels_as_a_reference_and_one_without_is_refused(tmp_path: Path,
                                                                                       monkeypatch: pytest.MonkeyPatch):
    limit = 200_000  # above every file MIME writes for the toy, so that the layer alone is over it
    monkeypatch.setattr(_terms, "INLINE_LIMIT_BYTES", limit)
    monkeypatch.setattr(_bundle, "INLINE_LIMIT_BYTES", limit)
    with same_reference_run():
        folder, (path,) = saved(tmp_path / "with", [Spec("open", "CC0-1.0", 2 * limit, True, False)])
        bundle = pack(folder, out=tmp_path / "with.mimebundle")
        manifest = json.loads(zipfile.ZipFile(bundle).read("manifest.json"))
        assert set(manifest["referenced"]) == {f"scene/layers/{path.name}"}
        assert path.read_bytes() not in bundle.read_bytes()
        folder, _ = saved(tmp_path / "without", [Spec("open", "CC0-1.0", 2 * limit, False, False)])
        with pytest.raises(BuildRefused) as refused:
            pack(folder, out=tmp_path / "without.mimebundle")
    assert refused.value.codes() == ["bundle.file-too-large"]


@pytest.mark.parametrize("files, code", [
    ([Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False), Spec("yours", None, 4000, False, False, 0)],
     "bundle.referenced-copy"),
    ([Spec("yours", None, 4000, False, False), Spec("unknown", "MPL-2.0", 4000, True, False, 0)],
     "bundle.referenced-copy"),
    ([Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False), Spec("open", "MIT", 4000, False, False, 0)],
     "terms.file"),
    ([Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False), Spec("yours", None, 4000, False, False, 0, True)],
     None),
    ([Spec("not permitted", NO_REDISTRIBUTION, 4000, True, True), Spec("unknown", "MPL-2.0", 4000, False, False, 0)],
     "terms.file"),
])
def test_the_bytes_of_a_file_that_travels_as_a_reference_never_travel_at_another_path(
        tmp_path: Path, files: list[Spec], code: str | None):
    with same_reference_run():
        pack_holds_the_tiers(tmp_path, files)
    stage, codes = expected_refusals(files, [tmp_path / "sources" / f"Mesh{index}.usda" for index in range(len(files))])
    assert codes == ({code} if code is not None else set()), (stage, codes)


def test_two_copies_with_the_same_terms_travel_as_one_asset_and_load(tmp_path: Path):
    files = [Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False),
             Spec("yours", None, 4000, False, False, 0, True)]
    with same_reference_run():
        folder, paths = saved(tmp_path, files)
    assert folder is not None
    lock = json.loads((folder / "mime.lock").read_text())
    assert lock["referenced_files"] == {f"scene/layers/{paths[0].name}": paths[0].name,
                                        f"scene/layers/{paths[1].name}": paths[0].name}
    assert [held["name"] for held in lock["assets"]] == [paths[0].name]


def edited_lock(folder: Path, change: Callable[[dict[str, object]], None]) -> None:
    lock = json.loads((folder / "mime.lock").read_text())
    change(lock)
    (folder / "mime.lock").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")


def relicensed(lock: dict[str, object]) -> None:
    lock["assets"][0]["licence"] = "CC0-1.0"  # type: ignore[index]


def relocated(lock: dict[str, object]) -> None:
    lock["assets"][0]["locations"] = ["https://zenodo.org/records/1/files/Mesh0.usda"]  # type: ignore[index]


def a_lock_that_restates_a_references_terms_is_refused(root: Path, change: Callable[[dict[str, object]], None]) -> None:
    """The scenario: a restricted file saved as a reference; its asset's licence or locations rewritten in the lock,
    which no hash covers; ``pack`` must re-derive them from the terms, which the input hash covers, and refuse."""
    folder, _ = saved(root, [Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False)])
    assert folder is not None
    edited_lock(folder, change)
    bundle, codes = pack_or_codes(folder, root / "x.mimebundle")
    assert bundle is None and codes == {"bundle.lock-terms"}, codes


@pytest.mark.parametrize("change", [relicensed, relocated])
def test_a_lock_that_restates_a_references_licence_or_locations_is_refused(tmp_path: Path,
                                                                         change: Callable[[dict[str, object]], None]):
    with same_reference_run():
        a_lock_that_restates_a_references_terms_is_refused(tmp_path, change)


def test_a_lock_that_restates_a_shipped_assets_licence_is_refused(tmp_path: Path):
    """An asset MIME ships, named in a lock (an entry's table), is held to the licence MIME ships it under."""
    ref = next(iter(known().values()))
    with same_reference_run():
        folder, _ = saved(tmp_path, [Spec("yours", None, 900, False, False)])
    assert folder is not None

    def with_shipped(licence: str) -> Callable[[dict[str, object]], None]:
        def change(lock: dict[str, object]) -> None:
            lock["assets"] = [{**ref.to_record(), "licence": licence}]
        return change

    edited_lock(folder, with_shipped(ref.licence))
    assert pack_or_codes(folder, tmp_path / "true.mimebundle")[1] == set()
    edited_lock(folder, with_shipped("MIT"))
    assert pack_or_codes(folder, tmp_path / "false.mimebundle") == (None, {"bundle.lock-terms"})


def test_a_lock_that_lists_a_reference_no_terms_give_an_origin_is_refused(tmp_path: Path):
    """D8: a file travels as a reference only where its terms give an origin; ``save`` writes no other, and ``pack``
    refuses a lock that says otherwise, however large the file."""
    with same_reference_run():
        folder, (path,) = saved(tmp_path, [Spec("yours", None, 4000, False, False)])
    assert folder is not None
    data = path.read_bytes()

    def listed(lock: dict[str, object]) -> None:
        lock["assets"] = [{"name": "mesh", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                           "licence": "CC-BY-4.0", "locations": ["https://example.org/mesh.usda"]}]
        lock["referenced_files"] = {f"scene/layers/{path.name}": "mesh"}

    edited_lock(folder, listed)
    assert pack_or_codes(folder, tmp_path / "x.mimebundle") == (None, {"bundle.lock-terms"})


def test_a_lock_edited_to_carry_a_file_that_may_not_travel_is_refused(tmp_path: Path):
    with same_reference_run():
        folder, _ = saved(tmp_path, [Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False)])
    lock = json.loads((folder / "mime.lock").read_text())
    del lock["referenced_files"]
    (folder / "mime.lock").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    with pytest.raises(BuildRefused) as refused:
        pack(folder, out=tmp_path / "x.mimebundle")
    assert refused.value.codes() == ["bundle.lock-terms"]


def test_a_lock_edited_to_reference_a_file_its_terms_let_travel_is_refused(tmp_path: Path):
    with same_reference_run():
        folder, (path,) = saved(tmp_path, [Spec("open", "MIT", 4000, True, False)])
    lock = json.loads((folder / "mime.lock").read_text())
    data = path.read_bytes()
    lock["assets"] = [{"name": "mesh", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                       "licence": "MIT", "locations": [ORIGIN.format(path.name)]}]
    lock["referenced_files"] = {f"scene/layers/{path.name}": "mesh"}
    (folder / "mime.lock").write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    with pytest.raises(BuildRefused) as refused:
        pack(folder, out=tmp_path / "x.mimebundle")
    assert refused.value.codes() == ["bundle.lock-terms"]


def test_the_lock_names_each_reference_and_its_asset_as_the_save_writes_them(tmp_path: Path):
    with same_reference_run():
        folder, (path,) = saved(tmp_path, [Spec("not permitted", NO_REDISTRIBUTION, 4000, True, False)])
    lock = json.loads((folder / "mime.lock").read_text())
    data = path.read_bytes()
    assert lock["schema"] == "mime.lock/4"
    assert lock["referenced_files"] == {f"scene/layers/{path.name}": path.name}
    assert lock["assets"] == [{"name": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
                               "licence": NO_REDISTRIBUTION.id, "locations": [ORIGIN.format(path.name)]}]


# --- the controls ------------------------------------------------------------------------------------------------

RUNS = 20
CASES = [[Spec("not permitted", NO_REDISTRIBUTION, 300, True, False)],
         [Spec("unknown", "MPL-2.0", 70_000, True, False), Spec("open", "MIT", 900, True, False)],
         [Spec("unknown", UNKNOWN_CUSTOM, 2 ** 20, True, False), Spec("yours", None, 4000, False, False)],
         [Spec("unknown", "EPL-2.0", 5000, True, False)]]


@contextmanager
def pack_carries_everything() -> Iterator[None]:
    """The plant: a pack whose judgement lets every file travel, and a save that writes no reference."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "referenced_members", lambda judged, lock: ({}, []))
        patch.setattr(_record, "referenced_assets", lambda document, files, taken: [])
        yield


@contextmanager
def save_writes_no_reference() -> Iterator[None]:
    """The plant: a save that writes no reference, which ``pack``'s own judgement must catch."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_record, "referenced_assets", lambda document, files, taken: [])
        yield


def planted_runs(tmp_path: Path, plant: Callable[[], object], cases: list[list[Spec]] = CASES) -> int:
    """How many of twenty runs catch the plant: the scenario's assertion fails."""
    caught = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        with plant():  # type: ignore[attr-defined]
            try:
                pack_holds_the_tiers(root, cases[run % len(cases)])
            except AssertionError:
                caught += 1
        shutil.rmtree(root, ignore_errors=True)
    return caught


def lock_refusals(root: Path, files: list[Spec]) -> set[str]:
    folder, _ = saved(root, files)
    assert folder is not None
    return pack_or_codes(folder, root / "x.mimebundle")[1]


def test_control_a_pack_that_carries_every_file_is_caught_every_run(tmp_path: Path):
    with same_reference_run():
        assert planted_runs(tmp_path, pack_carries_everything) == RUNS


def test_control_a_save_that_writes_no_reference_is_caught_by_the_pack_every_run(tmp_path: Path):
    """``pack`` judges the terms itself: a lock with no reference for a file that may not travel is refused, by the
    code that names the disagreement, on every run."""
    caught = 0
    with same_reference_run():
        for run in range(RUNS):
            root = tmp_path / str(run)
            root.mkdir()
            with save_writes_no_reference():
                caught += lock_refusals(root, CASES[run % len(CASES)]) == {"bundle.lock-terms"}
            shutil.rmtree(root, ignore_errors=True)
    assert caught == RUNS


#: Folders with a copy of a referenced file's bytes that would be carried: the cases the copy check must refuse.
COPY_CASES = [[Spec("not permitted", NO_REDISTRIBUTION, 300, True, False), Spec("yours", None, 300, False, False, 0)],
              [Spec("yours", None, 5000, False, False), Spec("unknown", "EPL-2.0", 5000, True, False, 0)],
              [Spec("open", "MIT", 900, True, False), Spec("unknown", UNKNOWN_CUSTOM, 70_000, True, False),
               Spec("yours", None, 70_000, False, False, 1)],
              [Spec("unknown", "MPL-2.0", 2 ** 20, True, False), Spec("yours", None, 2 ** 20, False, False, 0)]]


@contextmanager
def pack_ignores_copies() -> Iterator[None]:
    """The plant: a pack that never compares a carried file's bytes with a referenced file's, as one judging by path
    alone would."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_copy_problems", lambda judged, listed: [])
        yield


@contextmanager
def pack_trusts_the_lock() -> Iterator[None]:
    """The plant: a pack that takes the licence and locations a lock states for a reference's asset as they are."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_bundle, "_restated", lambda held, file, publication=None: False)
        yield


def test_control_a_pack_that_ignores_copies_is_caught_every_run(tmp_path: Path):
    with same_reference_run():
        assert planted_runs(tmp_path, pack_ignores_copies, COPY_CASES) == RUNS


def test_control_a_pack_that_trusts_the_locks_terms_is_caught_every_run(tmp_path: Path):
    caught = 0
    with same_reference_run():
        for run in range(RUNS):
            root = tmp_path / str(run)
            root.mkdir()
            with pack_trusts_the_lock():
                try:
                    a_lock_that_restates_a_references_terms_is_refused(root, (relicensed, relocated)[run % 2])
                except AssertionError:
                    caught += 1
            shutil.rmtree(root, ignore_errors=True)
    assert caught == RUNS


def test_the_control_cases_pass_unplanted(tmp_path: Path):
    with same_reference_run():
        for index, files in enumerate([*CASES, *COPY_CASES]):
            pack_holds_the_tiers(tmp_path / str(index), files)
        for index, change in enumerate((relicensed, relocated)):
            a_lock_that_restates_a_references_terms_is_refused(tmp_path / f"lock{index}", change)
