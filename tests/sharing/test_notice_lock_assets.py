"""A bundle's NOTICE and availability statement read the lock's assets as well as the folder's files.

An experiment whose result reads an asset it holds no file of (a wall's table, a lab's licensed record) names it in the
lock's ``assets``. A bundle never carries such an asset's bytes, so its NOTICE must name the asset's licence, and its
availability must not say "Reproducible exactly from this bundle alone". Before the fix both read only the folder's
files: a folder whose only licensed thing was a lab's record said "Licences: None named" and "from this bundle alone".

The kinds a lock can hold, each tested: a lab's record under its own licence (referenced; the bytes are at the lab); a
conditional one (Creative Commons non-commercial: its conditions are said); one MIME's own data holds (the tube table:
what a reuser needs is that MIME at the version that saved the folder, and the table's licence). The real tube
experiment is saved and packed in ``tests/saved/test_lock_assets.py``, which reads the cached table (this folder's
fixtures give every test an empty assets cache).

What is held unchanged: a folder with no lock assets has the NOTICE it had (a frozen text, from a hand-built folder
terms so that no environment's hashes are in it), and no bytes travel that did not before (the bundle's members are the
folder's files, as ever).

The controls plant the defect itself, the lock's assets left out of the folder's terms, and every scenario must fail
under it on every run: the planted code is deterministic, so a run is the same as twenty, and each kind is run under it
once.
"""

import json
import shutil
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, cast

import pytest
from terms_cases import layered, same_reference_run

from mime import assets
from mime.assets import AssetRef
from mime.nodes.hydro.mrs._tube import UNIT_TUBE
from mime.saved import LOCK_FILE, _terms, card_text, notice, pack
from mime.saved._card import availability, notice_text
from mime.saved._schema import ExperimentDocument, Lock
from mime.saved._terms import FileTerms, FolderTerms
from mime.terms import CustomLicence, Verdict

GOLDEN = Path(__file__).with_name("golden_notice_no_lock_assets.txt")
LAB_LICENCE = CustomLicence("LicenseRef-lab-NC", "The lab's data licence 0.1", "https://example.org/lab-licence",
                            "unknown")
LAB = AssetRef("lab_clot", "ab" * 32, 179, LAB_LICENCE.id, ("https://example.org/lab/clot.json",),
               credit="the lab", role="results", kind="measured", licences=(LAB_LICENCE,))
CONDITIONAL = AssetRef("lab_scan", "cd" * 32, 4_000_000, "CC-BY-NC-4.0", ("https://example.org/lab/scan.npz",),
                       credit="the lab", role="results", kind="measured")
TUBE = assets.reference(UNIT_TUBE)
HERE = "from this bundle alone"


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The toy, saved with no asset in its lock."""
    root = tmp_path_factory.mktemp("lock_assets") / "plain"
    with same_reference_run():
        layered("plain", [], licence="CC-BY-4.0").build().save(root)
    assert "assets" not in json.loads((root / LOCK_FILE).read_text(encoding="utf-8"))
    return root


def with_lock_asset(plain: Path, target: Path, ref: AssetRef) -> Path:
    """A copy of the folder whose lock names ``ref``, as a build that reads it writes it (``to_record``)."""
    shutil.copytree(plain, target)
    lock = json.loads((target / LOCK_FILE).read_text(encoding="utf-8"))
    lock["assets"] = [ref.to_record()]
    (target / LOCK_FILE).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return target


def saved_by(folder: Path) -> str:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))["environment"]["mime"]


def licences_section(text: str) -> str:
    return text.split("\nLicences\n--------\n")[1].split("\n\nAvailability")[0]


def says_what_a_bundle_needs(folder: Path, ref: AssetRef, *, verdict: str, conditions: tuple[str, ...] = ()) -> None:
    text = notice(folder)
    assert "Assets the lock names" in text, "the NOTICE has no heading for the lock's assets"
    block = text.split("Assets the lock names\n---------------------\n")[1].split("\n\nPermissions")[0]
    assert f"\n{ref.name}\n  sha256 {ref.sha256}," in block
    assert f"licence {ref.licence}" in block and f"verdict: {verdict}, by its licence; not carried in a bundle" in block
    assert f"available from: {ref.locations[0]}" in block
    named = licences_section(text)
    for identifier in ref.licence.split(" OR "):
        assert identifier in named, f"{identifier} is not among the licences named:\n{named}"
    assert "None named." not in named
    said = text.split("Availability\n------------\n")[1]
    assert HERE not in said, said
    assert "given 1 asset the lock names that the bundle does not carry" in said and ref.name in said
    assert f"licence {ref.licence}" in said and f"available from {ref.locations[0]}" in said


def lab_record(folder: Path) -> None:
    says_what_a_bundle_needs(folder, LAB, verdict="unknown, referenced")
    assert f"{LAB_LICENCE.id}: {LAB_LICENCE.title}, {LAB_LICENCE.url}" in notice(folder)


def conditional_record(folder: Path) -> None:
    says_what_a_bundle_needs(folder, CONDITIONAL, verdict="permitted, conditional (non-commercial)")
    assert "CC-BY-NC-4.0: https://spdx.org/licenses/CC-BY-NC-4.0.html" in notice(folder)


def mime_own_record(folder: Path) -> None:
    says_what_a_bundle_needs(folder, TUBE, verdict="permitted, open")
    text = notice(folder)
    version = saved_by(folder)
    assert "MIME's own record; licence CC0-1.0" in text
    assert f"needs: MIME {version}, the version that saved the folder (`pip install mime-engine=={version}`), " \
           f"besides the bytes" in text
    assert f"MIME's own record, which needs MIME {version}, the version that saved the folder" in text


KINDS: dict[str, tuple[AssetRef, Callable[[Path], None]]] = {
    "a lab's record under its own licence": (LAB, lab_record),
    "a conditional one": (CONDITIONAL, conditional_record),
    "one MIME's own data holds": (TUBE, mime_own_record)}


@pytest.mark.parametrize("kind", KINDS)
def test_the_notice_and_availability_name_each_kind_of_asset_the_lock_holds(plain: Path, tmp_path: Path, kind: str):
    ref, scenario = KINDS[kind]
    scenario(with_lock_asset(plain, tmp_path / "held", ref))


def test_a_lab_assets_bundle_says_so_in_its_own_notice_and_carries_none_of_its_bytes(plain: Path, tmp_path: Path):
    folder = with_lock_asset(plain, tmp_path / "held", LAB)
    bundle = pack(folder, out=tmp_path / "held.mimebundle")
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        held = archive.read("NOTICE").decode()
        manifest = json.loads(archive.read("manifest.json"))
    assert held == notice(folder), "the bundle's NOTICE is not the folder's"
    assert "Licences\n--------\nCC-BY-4.0: https://spdx.org/licenses/CC-BY-4.0.html\n" + LAB_LICENCE.id in held
    assert HERE not in held and "None named." not in held
    assert manifest["assets"] == [LAB.to_record()], "the bundle carries the record, as ever"
    assert not any(member.endswith(LAB.name) for member in names), "no bytes travel that did not before"


def test_the_card_names_a_lab_assets_licence_with_its_conditions_and_link(plain: Path, tmp_path: Path):
    from mime.saved import _card

    folder = with_lock_asset(plain, tmp_path / "held", CONDITIONAL)
    judged, document, lock = _card.judged_folder(folder)
    assert [held.ref.name for held in judged.assets] == ["lab_scan"]
    row = _card._asset_row(judged.assets[0])
    assert row == "| `lab_scan` | an asset the lock names | CC-BY-NC-4.0 | permitted, **non-commercial** | referenced | the lab |"
    assert availability(judged).startswith("Reproducible exactly, given 1 asset the lock names")
    assert document["experiment"] == "plain" and lock["assets"] == [CONDITIONAL.to_record()]


def test_an_asset_a_referenced_file_stands_for_is_the_files_and_not_counted_twice():
    """A file that travels as a reference is in the lock's ``assets`` too (its asset): the NOTICE says it once, under
    Files, and the asset the lock names for no file is the one counted here. Two records of one file's bytes are one."""
    lock = cast(Lock, {"assets": [LAB.to_record(), CONDITIONAL.to_record(), CONDITIONAL.to_record()],
                       "referenced_files": {"scene/layers/lab.usda": LAB.name}})
    found = _terms._lock_assets(lock, {})
    assert [held.ref.name for held in found] == [CONDITIONAL.name]


# --- what is held unchanged ----------------------------------------------------------------------------------------


def hand_built() -> tuple[FolderTerms, ExperimentDocument, Lock]:
    """A folder's terms built by hand, so that the frozen text has no environment's hashes: the author's own file, a
    third party's MIT file that travels, and a presentation mesh that travels as a reference, no lock asset."""
    own = Verdict("permitted", "open", (), "the author's own")
    files = (FileTerms("mime.lock", "ab" * 32, 7000, "yours", "CC-BY-4.0", own),
             FileTerms("scene/layers/arm.usda", "cd" * 32, 6000, "third party", "MIT", Verdict("permitted", "open"),
                       "Copyright (c) 2021 A. Driver", "results", (), ()),
             FileTerms("scene/layers/big.usda", "ef" * 32, 3_000_000, "third party", "LicenseRef-x",
                       Verdict("unknown", "referenced", (), "its licence"), "A maker", "presentation",
                       ("https://example.org/big.usda",), (), "big.usda"))
    return (FolderTerms(files, "CC-BY-4.0", {}, (), 4), cast(ExperimentDocument, {"experiment": "golden"}),
            cast(Lock, {"environment": {"mime": "0.3.0"}}))


def test_a_folder_with_no_lock_assets_has_the_notice_it_always_had():
    assert notice_text(*hand_built()) == GOLDEN.read_text(encoding="utf-8")


def test_a_folder_with_no_lock_assets_says_from_this_bundle_alone_where_it_carries_every_file(plain: Path):
    text = notice(plain)
    assert "Assets the lock names" not in text
    assert text.endswith(f"Availability\n------------\nReproducible exactly {HERE}.\n")
    assert availability(_terms_of(plain)) == f"Reproducible exactly {HERE}."


def _terms_of(folder: Path) -> FolderTerms:
    from mime.saved import _card

    return _card.judged_folder(folder)[0]


# --- the controls ---------------------------------------------------------------------------------------------------


@contextmanager
def lock_assets_left_out() -> Iterator[None]:
    """The planted defect: the folder's terms read no asset of the lock."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_terms, "_lock_assets", lambda lock, derived: ())
        yield


@pytest.mark.parametrize("kind", KINDS)
def test_control_the_scenario_fails_under_the_planted_defect_on_every_run(plain: Path, tmp_path: Path, kind: str):
    ref, scenario = KINDS[kind]
    folder = with_lock_asset(plain, tmp_path / "held", ref)
    with lock_assets_left_out(), pytest.raises(AssertionError):
        scenario(folder)
    scenario(folder)


def test_control_the_frozen_text_catches_an_empty_assets_heading():
    """A NOTICE that gains a heading for no asset is not the one it always was."""
    from mime.saved import _card

    judged, document, lock = hand_built()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_card, "availability", lambda held: availability(held) + "\n\nAssets the lock names\n---")
        assert notice_text(judged, document, lock) != GOLDEN.read_text(encoding="utf-8")
