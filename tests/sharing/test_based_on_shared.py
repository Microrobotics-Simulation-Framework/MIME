"""``Experiment.based_on`` from what ``mime pull`` takes, and the pack and push of an experiment based on another.

A bundle file, an address and a DOI are pulled as ``mime pull`` pulls them (the network is the pull tests' local
server and Zenodo's record API faked on it, as there), into ``./<name>/``, and the new experiment is based on that
folder, which is then never written. A DOI's record is named by the lock ``mime push`` packed into it: the new
experiment records that DOI, and its own lock names no record (pushing it makes a record of its own).

**Pack and push.** An experiment based on another references the parent's files on their own terms, as any experiment
does (#198's tiers): the same files at the same paths, with the same licences, verdicts and tiers, a file that may not
be redistributed still a reference; the parent's licence is the new experiment's until it declares another. What
``mime push`` sends is unchanged: its description is the card without the line naming the parent, and its related
identifiers are MIME's own and ``sharing.yaml``'s, with no relation to the parent (that relation is a new kind of
Zenodo action, rehearsed on the sandbox first)."""

import dataclasses
import json
import warnings
import zipfile
from pathlib import Path

import pytest
from assets_rig import Rig, reference
from bundles import lock_with_assets, write_bundle
from fake_zenodo import DOI, zenodo
from folder_state import untouched
from push_cases import copy_of, with_token, write_sharing
from terms_cases import NO_REDISTRIBUTION, layered, write_mesh
from zenodo_records import FakeRecords

from mime import assets, push
from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.push import _plan
from mime.saved import DOCUMENT_FILE, LOCK_FILE, card_text, pack
from mime.saved._card import judged_folder
from mime.terms import Terms
from mime.testing.toys import full_toy

ORIGIN = "https://example.org/arm_driver/6a3ebb1/meshes/arm_visual.usda"


@pytest.fixture(scope="module")
def saved_toy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("toy") / "full_toy"
    full_toy().save(folder)
    return folder


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def based_on_record(folder: Path) -> dict:
    import yaml

    return yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))["based_on"]


def published(saved_toy: Path, root: Path) -> Path:
    """The toy as ``mime push`` publishes it: its lock naming the record it is, outside the input hash."""
    folder = copy_of(saved_toy, root)
    lock = lock_of(folder)
    lock["publication"] = {"service": "zenodo", "record": "4242", "doi": DOI, "concept": "4241", "version": 1}
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return folder


# --- the sources pull takes --------------------------------------------------------------------------------------------


def test_a_bundle_file_is_pulled_into_the_working_directory_and_the_experiment_based_on_that_folder(
        saved_toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    monkeypatch.chdir(tmp_path)
    child = Experiment.based_on(bundle, name="child")
    pulled = tmp_path / "full_toy"
    assert (pulled / LOCK_FILE).read_bytes() == (saved_toy / LOCK_FILE).read_bytes()
    with untouched(pulled):
        child.save(tmp_path / "child")
    assert based_on_record(tmp_path / "child") == {"experiment": "full_toy",
                                                   "input_hash": lock_of(saved_toy)["hashes"]["input"],
                                                   "mime": lock_of(saved_toy)["environment"]["mime"]}


def test_based_on_the_same_bundle_again_is_refused_naming_the_call_on_the_folder_pulled_before(
        saved_toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    monkeypatch.chdir(tmp_path)
    Experiment.based_on(bundle, name="child")
    with untouched(tmp_path / "full_toy"), pytest.raises(BuildRefused) as refused:
        Experiment.based_on(bundle, name="child")
    (refusal,) = refused.value.refusals
    assert refusal.code == "pull.destination" and "Experiment.based_on(\"<the folder>\", name=...)" in refusal.action
    assert Experiment.based_on(tmp_path / "full_toy", name="child").name == "child", "the call it names works"


def test_an_address_is_pulled_through_the_asset_cache(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    rig.server.files["/b/full_toy.mimebundle"] = pack(saved_toy, out=tmp_path / "b.mimebundle").read_bytes()
    monkeypatch.chdir(tmp_path)
    child = Experiment.based_on(rig.server.url("/b/full_toy.mimebundle"), name="child")
    assert rig.requests == 1 and child.document()["based_on"]["input_hash"] == lock_of(saved_toy)["hashes"]["input"]


def test_a_doi_is_pulled_and_its_record_named_by_the_lock_push_packed_is_recorded(saved_toy: Path, rig: Rig,
                                                                                tmp_path: Path,
                                                                                monkeypatch: pytest.MonkeyPatch):
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        child = Experiment.based_on(DOI, name="child")
    with untouched(work / "full_toy"):
        child.save(work / "child")
    assert based_on_record(work / "child") == {"experiment": "full_toy",
                                               "input_hash": lock_of(saved_toy)["hashes"]["input"],
                                               "mime": lock_of(saved_toy)["environment"]["mime"], "doi": DOI}
    assert "publication" not in lock_of(work / "child"), "the child is no record yet: its push makes its own"
    assert f"- **Based on full_toy (DOI {DOI})**" in card_text(work / "child").splitlines()


def test_a_pulled_sources_assets_pass_the_licence_gate_and_are_accepted_by_name(saved_toy: Path, rig: Rig,
                                                                               tmp_path: Path,
                                                                               monkeypatch: pytest.MonkeyPatch):
    """A bundle whose lock names an asset under a non-commercial licence: with nobody to ask, ``based_on`` is refused
    as ``pull`` is, naming the licence and ``accept_licences``; accepted by name, the asset is fetched and the new
    experiment made. ``yes`` and ``force`` accept no licence."""
    payload = b"a table under a non-commercial licence" * 30
    rig.server.files["/t/table.bin"] = payload
    ref = dataclasses.replace(reference(payload, rig.server.url("/t/table.bin"), name="table"),
                              licence="CC-BY-NC-4.0")
    files = {path.relative_to(saved_toy).as_posix(): path.read_bytes() for path in saved_toy.rglob("*")
             if path.is_file()}
    bundle = write_bundle(tmp_path / "full_toy.mimebundle", files, "full_toy", lock_with_assets(files, ref))
    for attempt, given in enumerate(({}, {"yes": True, "force": True})):
        work = tmp_path / f"refused{attempt}"
        work.mkdir()
        monkeypatch.chdir(work)
        with assets.asking(None), pytest.raises(BuildRefused) as refused:
            Experiment.based_on(bundle, name="child", **given)
        assert refused.value.codes() == ["asset.licence-not-accepted"]
        assert "accept_licences={'CC-BY-NC-4.0'}" in refused.value.refusals[0].action
        assert not assets.is_cached(ref)
    work = tmp_path / "accepted"
    work.mkdir()
    monkeypatch.chdir(work)
    with assets.asking(None):
        child = Experiment.based_on(bundle, name="child", accept_licences={"CC-BY-NC-4.0"})
    assert assets.is_cached(ref) and child.document()["based_on"]["experiment"] == "full_toy"


# --- pack and push -------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arm_parent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A presentation mesh under a rights holder's licence that forbids redistribution (with its origin) beside an
    MIT description of the arm's kinematics, in a folder licensed CC-BY-4.0: the referenced tier and the open one."""
    root = tmp_path_factory.mktemp("arm")
    visual = write_mesh(root / "meshes", "arm_visual", 6000)
    kinematics = write_mesh(root / "meshes", "arm_kinematics", 1200)
    layers = [(visual, Terms(NO_REDISTRIBUTION, "Robot Maker Ltd (the arm's maker)", "presentation", (ORIGIN,))),
              (kinematics, Terms("MIT", "Copyright (c) 2021 A. Driver", "results"))]
    layered("arm_like", layers, licence="CC-BY-4.0").save(root / "arm_like")
    return root / "arm_like"


def third_party(folder: Path) -> list[tuple[object, ...]]:
    judged, _, _ = judged_folder(folder)
    return [(file.path, file.sha256, file.licence, file.credit, file.role, file.origin, file.verdict.status,
             file.verdict.tier, file.referenced is not None) for file in judged.files if file.owner == "third party"]


def test_the_parents_files_keep_their_terms_and_tiers_and_pack_carries_the_same(arm_parent: Path, tmp_path: Path):
    with untouched(arm_parent):
        Experiment.based_on(arm_parent, name="arm_variant").save(tmp_path / "arm_variant")
    child = tmp_path / "arm_variant"
    assert third_party(child) == third_party(arm_parent)
    assert [row[-1] for row in third_party(child)] == [False, True], "the mesh that may not travel is a reference"
    assert judged_folder(child)[0].licence == "CC-BY-4.0", "the parent's licence, until the child declares another"
    manifests = []
    for folder in (arm_parent, child):
        with zipfile.ZipFile(pack(folder, out=tmp_path / f"{folder.name}.mimebundle")) as archive:
            manifests.append(json.loads(archive.read("manifest.json")))
    parent_files, child_files = ({name: held for name, held in manifest["files"].items() if name.startswith("scene/layers/")}
                                 for manifest in manifests)
    assert child_files == parent_files and manifests[1]["referenced"] == manifests[0]["referenced"] != {}


def test_the_licence_taken_from_the_parent_is_replaced_by_one_declared_again(arm_parent: Path, tmp_path: Path):
    child = Experiment.based_on(arm_parent, name="arm_variant")
    with pytest.warns(UserWarning, match="took from 'arm_like'"):
        child.licence("CC0-1.0")
    child.save(tmp_path / "arm_variant")
    judged = judged_folder(tmp_path / "arm_variant")[0]
    assert judged.licence == "CC0-1.0" and third_party(tmp_path / "arm_variant") == third_party(arm_parent)


def test_push_sends_what_it_sent_no_relation_and_no_line_naming_the_parent(arm_parent: Path, tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Experiment.based_on(arm_parent, name="arm_variant").save(tmp_path / "arm_variant")
    plans = {}
    for folder in (copy_of(arm_parent, tmp_path / "parent"), tmp_path / "arm_variant"):
        write_sharing(folder)
        fake = FakeRecords(with_token(monkeypatch))
        plans[folder.name] = _plan.make_plan(copy_of(folder, tmp_path / "push" / folder.name), fake.service(),
                                             push.read_sharing(folder), tmp_path / "staged" / folder.name)
    parent, child = plans["arm_like"], plans["arm_variant"]
    metadata = child.payload["metadata"]
    assert metadata["related_identifiers"] == parent.payload["metadata"]["related_identifiers"]  # type: ignore[index]
    assert [each["relation_type"]["id"] for each in metadata["related_identifiers"]] == ["requires", "iscompiledby"]  # type: ignore[index]
    assert "Based on" not in str(metadata["description"]) and "Based on" not in child.card  # type: ignore[index]
    assert "Based on arm_like" in card_text(tmp_path / "arm_variant"), "mime card names it"
    assert child.target(None) == "a new draft, its DOI reserved" and child.second_concept(None) is None
