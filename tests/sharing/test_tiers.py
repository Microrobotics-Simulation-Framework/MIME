"""``mime pack``'s verdicts (designs #198, #199): **a file that may not be redistributed never travels inside a bundle,
whatever its size**. It travels as a reference to its origin, which the lock names; a file that must and gives no
origin is refused (``bundle.no-location``); a declared permission that grants redistribution lets it travel; a file
over the size a bundle carries travels as a reference where its terms give an origin.

The property draws saved folders with files of every verdict, at real sizes (a few hundred bytes to two megabytes, the
range of a robot's visual meshes) and with real SPDX ids, and holds the bundle's stored bytes to ``terms_cases``'
table: every file that table says may not travel is absent from the bundle, by sha256 and by its bytes, and named as a
reference; every other file is carried, byte for byte. The reference outputs of the toy are run once and reused by
every save here (``same_reference_run``): they are a run of the toy's graph, which no layer, licence or terms changes,
and nothing here reads them. Every example is a real save and a real pack, which is why the property is on the
duration allowlist rather than cut: its examples are the folders it draws.

The control plants the error the property guards against, a pack that carries every file, and the scenario must catch
it on every one of twenty runs; another plants a save that writes no reference into the lock, which ``pack`` must catch
by judging the terms itself.
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

from mime.build import BuildRefused
from mime.saved import _bundle, _record, _terms, pack, unpack
from mime.terms import CustomLicence, Evidence, Permission, Scope, Terms

ORIGIN = "https://example.org/upstream/6a3ebb1/{}"
UNKNOWN_CUSTOM = CustomLicence("LicenseRef-Somebody-Else", "Somebody Else's Terms", "https://example.org/else")


@dataclass(frozen=True)
class Spec:
    """One file of a drawn folder: its kind of licence (``yours`` for none), the licence, its size, whether its terms
    give an origin, and whether a permission granting redistribution names it."""

    kind: str
    licence: str | CustomLicence | None
    size: int
    origin: bool
    permitted: bool

    def travels(self) -> bool:
        """This file's own reading of whether the bytes may travel in a bundle."""
        return self.kind in ("yours", "open", "conditional") or self.permitted


licences = st.one_of(
    st.tuples(st.just("open"), st.sampled_from(OPEN_IDS)),
    st.tuples(st.just("conditional"), st.sampled_from(sorted(CONDITIONAL_IDS))),
    st.tuples(st.just("unknown"), st.sampled_from(UNKNOWN_IDS)),
    st.tuples(st.just("unknown"), st.just(UNKNOWN_CUSTOM)),
    st.tuples(st.just("not permitted"), st.just(NO_REDISTRIBUTION)),
    st.tuples(st.just("yours"), st.none()))
#: Real sizes, log-uniform from a stub layer to a large single mesh.
sizes = st.integers(min_value=8, max_value=21).map(lambda power: 2 ** power)


@st.composite
def specs(draw: st.DrawFn) -> list[Spec]:
    count = draw(st.integers(min_value=1, max_value=4))
    found: list[Spec] = []
    for _ in range(count):
        kind, licence = draw(licences)
        restricted = kind in ("unknown", "not permitted")
        found.append(Spec(kind, licence, draw(sizes), draw(st.booleans()), draw(st.booleans()) if restricted else False))
    return found


def saved(root: Path, files: list[Spec], *, licence: str | None = "CC-BY-4.0") -> tuple[Path, list[Path]]:
    """The full toy with ``files`` as mesh layers, saved; the folder and each layer's source file."""
    paths = [write_mesh(root / "sources", f"Mesh{index}", spec.size) for index, spec in enumerate(files)]
    layers = [(path, None if spec.licence is None else Terms(
        spec.licence, "Robot Maker Ltd", "presentation", (ORIGIN.format(path.name),) if spec.origin else ()))
        for path, spec in zip(paths, files, strict=True)]
    permissions = [Permission("Ada Maker, Robot Maker Ltd", "A. Lab", (hashlib.sha256(path.read_bytes()).hexdigest(),),
                              Scope(True, True, True), "2026-11-20",
                              Evidence("public_statement", "https://example.org/statement"))
                   for path, spec in zip(paths, files, strict=True) if spec.permitted]
    folder = root / "folder"
    layered("drawn", layers, licence=licence, permissions=permissions).build().save(folder)
    return folder, paths


def pack_holds_the_tiers(root: Path, files: list[Spec]) -> None:
    """The scenario: save, pack, and hold the bundle's bytes to the table; then unpack, and hold the folder to it."""
    folder, paths = saved(root, files)
    must_reference = [spec for spec in files if not spec.travels()]
    if any(not spec.origin for spec in must_reference):
        with pytest.raises(BuildRefused) as refused:
            pack(folder, out=root / "drawn.mimebundle")
        assert "bundle.no-location" in refused.value.codes()
        assert not (root / "drawn.mimebundle").exists()
        return
    bundle = pack(folder, out=root / "drawn.mimebundle")
    stored = bundle.read_bytes()
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    member_hashes = {hashlib.sha256(data).hexdigest() for data in members.values()}
    for path, spec in zip(paths, files, strict=True):
        layer = f"scene/layers/{path.name}"
        data = path.read_bytes()
        if spec.travels():
            assert members.get(f"experiment/{layer}") == data, (spec, "a file that may travel was not carried")
            assert layer not in manifest["referenced"]
        else:
            assert hashlib.sha256(data).hexdigest() not in member_hashes, (spec, "a file that may not travel is inside")
            assert data not in stored, (spec, "the bytes of a file that may not travel are in the bundle")
            assert manifest["referenced"][layer]["sha256"] == hashlib.sha256(data).hexdigest()
    unpacked = unpack(bundle, root / "unpacked")
    expected = sorted(f"scene/layers/{path.name}" for path, spec in zip(paths, files, strict=True)
                      if not spec.travels())
    assert sorted(unpacked.missing) == expected and unpacked.placed == ()
    assert all(not (root / "unpacked" / each).exists() for each in expected)


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


def planted_runs(tmp_path: Path, plant: Callable[[], object], caught_by: type[Exception]) -> int:
    """How many of twenty runs catch the plant, by ``caught_by``: the scenario's assertion, or a refusal of ``pack``
    with the code that names the disagreement."""
    caught = 0
    for run in range(RUNS):
        root = tmp_path / str(run)
        root.mkdir()
        with plant():  # type: ignore[attr-defined]
            try:
                pack_holds_the_tiers(root, CASES[run % len(CASES)])
            except caught_by as error:
                caught += not isinstance(error, BuildRefused) or error.codes() == ["bundle.lock-terms"]
        shutil.rmtree(root, ignore_errors=True)
    return caught


def test_control_a_pack_that_carries_every_file_is_caught_every_run(tmp_path: Path):
    with same_reference_run():
        assert planted_runs(tmp_path, pack_carries_everything, AssertionError) == RUNS


def test_control_a_save_that_writes_no_reference_is_caught_by_the_pack_every_run(tmp_path: Path):
    with same_reference_run():
        assert planted_runs(tmp_path, save_writes_no_reference, BuildRefused) == RUNS


def test_the_control_cases_pass_unplanted(tmp_path: Path):
    with same_reference_run():
        for index, files in enumerate(CASES):
            pack_holds_the_tiers(tmp_path / str(index), files)
