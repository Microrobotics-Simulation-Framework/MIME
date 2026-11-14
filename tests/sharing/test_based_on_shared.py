"""``Experiment.based_on`` from what ``mime pull`` takes, and the pack and push of an experiment based on another.

A bundle file, an address and a DOI are pulled as ``mime pull`` pulls them (the network is the pull tests' local
server and Zenodo's record API faked on it, as there), into ``./<name>/``, and the new experiment is based on that
folder, which is then never written. A folder there already that holds the same experiment (a notebook's cell run
again) is used as it is; one that holds anything else is refused, naming the call on it.

**The DOI.** Pulled by a DOI, the new experiment records the record it was pulled from (the record's own DOI, where a
concept DOI resolved to its latest version): a bundle uploaded by hand, whose lock names no record, is named by it too,
and a lock that names another record is warned of, naming both. A bundle file and a folder record the DOI their lock
names. The new experiment's own lock names no record (pushing it makes a record of its own).

**Pack and push.** An experiment based on another references the parent's files on their own terms, as any experiment
does (#198's tiers): the same files at the same paths, with the same licences, verdicts and tiers, a file that may not
be redistributed still a reference; the parent's licence is the new experiment's until it declares another. What
``mime push`` sends is unchanged: its description is the card without the line naming the parent, and its related
identifiers are MIME's own and ``[record]``'s, with no relation to the parent (that relation is a new kind of
Zenodo action, rehearsed on the sandbox first)."""

import builtins
import dataclasses
import hashlib
import json
import shutil
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from assets_rig import Rig, reference
from bundles import lock_with_assets, write_bundle
from fake_zenodo import DOI, RECORD, zenodo
from folder_state import untouched
from push_cases import copy_of, with_token, write_manifest
from terms_cases import NO_REDISTRIBUTION, layered, write_mesh
from zenodo_records import FakeRecords

from mime import assets, push
from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.push import _plan
from mime.saved import DOCUMENT_FILE, LOCK_FILE, card_text, pack
from mime.saved import _pull
from mime.saved._card import judged_folder
from mime.terms import Terms
from mime.testing.toys import full_toy
from mime.testing.toys._entries import ToySpring
from mime.units import N, m

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


def published(saved_toy: Path, root: Path, doi: str = DOI) -> Path:
    """The toy as ``mime push`` publishes it: its lock naming the record it is (``doi``), outside the input hash."""
    folder = copy_of(saved_toy, root)
    lock = lock_of(folder)
    lock["publication"] = {"service": "zenodo", "record": doi.rpartition(".")[2], "doi": doi, "concept": "4241",
                           "version": 1}
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return folder


def pulled_by_doi(bundle: Path, rig: Rig, work: Path, monkeypatch: pytest.MonkeyPatch, source: str = DOI,
                  **record: str) -> tuple[dict, list[str]]:
    """``bundle`` published as the fake Zenodo's record and based on by ``source`` from ``work``, saved there: the
    record the child's document holds, and the warnings ``based_on`` gave."""
    work.mkdir()
    monkeypatch.chdir(work)
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}, **record), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        child = Experiment.based_on(source, name="child")
    child.save(work / "child")
    return based_on_record(work / "child"), [str(each.message) for each in caught if "pulled from" in str(each.message)]


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


def test_based_on_the_same_bundle_again_uses_the_folder_pulled_before_and_writes_nothing_in_it(
        saved_toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    monkeypatch.chdir(tmp_path)
    first = Experiment.based_on(bundle, name="child").document()["based_on"]
    before = sorted(path.name for path in tmp_path.iterdir())
    with untouched(tmp_path / "full_toy"):
        again = Experiment.based_on(bundle, name="child").document()["based_on"]
    assert again == first and sorted(path.name for path in tmp_path.iterdir()) == before, "nothing unpacked again"


def stiffer_toy_saved_as_full_toy(saved_toy: Path, folder: Path) -> Path:
    """Another experiment under the toy's name: its spring stiffer, so its input hash is another."""
    experiment = Experiment.based_on(saved_toy, name="full_toy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        experiment.attach(ToySpring(k=N(3.4) / m(1.0)), body="probe")
    experiment.save(folder)
    return folder


@pytest.mark.parametrize("held", ["another experiment of its name", "a file of the user's"])
def test_a_folder_there_that_holds_anything_else_is_refused_naming_the_call_on_it_and_left_as_it_was(
        saved_toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, held: str):
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    there = tmp_path / "full_toy"
    if held == "another experiment of its name":
        stiffer_toy_saved_as_full_toy(saved_toy, there)
    else:
        there.mkdir()
        (there / "notes.txt").write_text("mine", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with untouched(there), pytest.raises(BuildRefused) as refused:
        Experiment.based_on(bundle, name="child")
    (refusal,) = refused.value.refusals
    assert refusal.code == "pull.destination" and "--into" not in refusal.action
    assert "uses a folder there only where it holds the same experiment" in refusal.action
    assert "Experiment.based_on(\"<the folder>\", name=...)" in refusal.action
    if held == "another experiment of its name":
        recorded = Experiment.based_on(there, name="child").document()["based_on"]
        assert recorded["input_hash"] == lock_of(there)["hashes"]["input"] != lock_of(saved_toy)["hashes"]["input"], \
            "the call it names bases the new experiment on what the folder holds"


NOTEBOOK_CELL = """from mime.experiment import Experiment
from mime.testing.toys._entries import ToySpring
from mime.units import N, m

child = Experiment.based_on("{source}", name="stiffer_toy")
child.attach(ToySpring(k=N(3.4) / m(1.0)), body="probe")
"""


def test_a_notebook_cell_that_bases_an_experiment_on_a_doi_runs_again(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch):
    """The same cell run twice in a notebook, as the notebook tests run a cell again: the second run uses the folder
    the first pulled, writes nothing in it, and gives the same experiment."""
    monkeypatch.setattr(builtins, "get_ipython", lambda: None, raising=False)
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    cell, namespace = NOTEBOOK_CELL.format(source=DOI), {}
    documents = []
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        for number, label in enumerate(("", ", run again")):
            with pytest.warns(UserWarning, match="took from 'full_toy'"), \
                    (untouched(work / "full_toy") if number else warnings.catch_warnings()):
                exec(compile(cell, f"<code cell 1{label}>", "exec"), namespace)
            documents.append(namespace["child"].document())
    assert documents[0] == documents[1] and documents[1]["based_on"]["doi"] == DOI
    assert [path.name for path in work.iterdir()] == ["full_toy"]


# --- a cell run again does not download its bundle again ----------------------------------------------------------------


def bundle_downloads(rig: Rig) -> int:
    """How many times the fake record's bundle bytes were requested (its ``content`` address) so far."""
    return sum(1 for request in rig.server.requests if request.endswith("/files/full_toy.mimebundle/content"))


def record_reads(rig: Rig) -> int:
    """How many times the fake record's metadata (its API address) was requested so far."""
    return sum(1 for request in rig.server.requests if request.endswith(f"/zenodo/api/records/{RECORD}"))


def fresh_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> Path:
    work = tmp_path / name
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def test_a_doi_run_again_reads_the_record_and_downloads_no_bundle(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch):
    """Measured on the fake repository: the first call reads the record and downloads the bundle once; the same call run
    again reads the record (its checksum is what says the bundle is the one held) and takes the bundle from the cache."""
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    work = fresh_work(tmp_path, monkeypatch, "work")
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        first = Experiment.based_on(DOI, name="child").document()
        assert (bundle_downloads(rig), record_reads(rig) > 0) == (1, True)
        reads = record_reads(rig)
        with untouched(work / "full_toy"):
            again = Experiment.based_on(DOI, name="child").document()
        assert bundle_downloads(rig) == 1, "no bundle bytes the second time"
        assert record_reads(rig) > reads, "the record is read again: its checksum names the bundle"
    assert again == first and [path.name for path in work.iterdir()] == ["full_toy"]


def test_a_changed_record_is_downloaded(saved_toy: Path, rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The record now lists another file (a new version: another checksum): it is not the cached bundle, so it is fetched
    (here its experiment is the folder's name with other contents, which is refused as before)."""
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    stiffer = pack(stiffer_toy_saved_as_full_toy(saved_toy, tmp_path / "stiffer" / "full_toy"),
                   out=tmp_path / "stiffer.mimebundle")
    fresh_work(tmp_path, monkeypatch, "work")
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        Experiment.based_on(DOI, name="child")
    assert bundle_downloads(rig) == 1
    with zenodo(rig, {"full_toy.mimebundle": stiffer.read_bytes()}), pytest.raises(BuildRefused) as refused:
        Experiment.based_on(DOI, name="child")
    assert bundle_downloads(rig) == 2 and refused.value.codes() == ["pull.destination"]


@pytest.mark.parametrize("cleared", ["the whole cache", "the bundle alone, its checksum's note left",
                                     "the bundle truncated, its note left"])
def test_a_cache_without_the_bundle_downloads_it_again(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch, cleared: str):
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    work = fresh_work(tmp_path, monkeypatch, "work")
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        first = Experiment.based_on(DOI, name="child").document()
        if cleared == "the whole cache":
            shutil.rmtree(rig.cache)
        elif cleared.startswith("the bundle alone"):
            (rig.cache / hashlib.sha256(bundle.read_bytes()).hexdigest()).unlink()
        else:
            held = rig.cache / hashlib.sha256(bundle.read_bytes()).hexdigest()
            held.write_bytes(held.read_bytes()[:-1])
        with untouched(work / "full_toy"):
            again = Experiment.based_on(DOI, name="child").document()
    assert bundle_downloads(rig) == 2 and again == first


def test_control_a_folder_holding_another_experiment_is_refused_though_the_bundle_comes_from_the_cache(
        saved_toy: Path, rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The cache knows the record's checksum, so no bundle bytes are fetched; the folder's input hash is still compared
    with the bundle's, so a folder there that holds another experiment is refused and left as it was. The checksum is a
    shortcut to the bytes and never to the verdict."""
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    fresh_work(tmp_path, monkeypatch, "first")
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        Experiment.based_on(DOI, name="child")
        assert bundle_downloads(rig) == 1
        planted = stiffer_toy_saved_as_full_toy(saved_toy, fresh_work(tmp_path, monkeypatch, "second") / "full_toy")
        assert lock_of(planted)["hashes"]["input"] != lock_of(saved_toy)["hashes"]["input"]
        with untouched(planted), pytest.raises(BuildRefused) as refused:
            Experiment.based_on(DOI, name="child")
    assert refused.value.codes() == ["pull.destination"] and bundle_downloads(rig) == 1


def test_a_doi_run_again_is_gated_on_its_assets_as_the_first_run_was(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch):
    """The bundle comes from the cache the second time, and its assets pass the same licence gate: refused again, naming
    the licence, until it is accepted by name."""
    payload = b"a table under a non-commercial licence" * 30
    rig.server.files["/t/table.bin"] = payload
    ref = dataclasses.replace(reference(payload, rig.server.url("/t/table.bin"), name="table"),
                              licence="CC-BY-NC-4.0")
    files = {path.relative_to(saved_toy).as_posix(): path.read_bytes() for path in saved_toy.rglob("*")
             if path.is_file()}
    bundle = write_bundle(tmp_path / "full_toy.mimebundle", files, "full_toy", lock_with_assets(files, ref))
    work = fresh_work(tmp_path, monkeypatch, "work")
    with zenodo(rig, {"full_toy.mimebundle": bundle.read_bytes()}):
        for run in range(2):
            with assets.asking(None), pytest.raises(BuildRefused) as refused, \
                    (untouched(work / "full_toy") if run else warnings.catch_warnings()):
                Experiment.based_on(DOI, name="child")
            assert refused.value.codes() == ["asset.licence-not-accepted"] and not assets.is_cached(ref)
        with assets.asking(None), untouched(work / "full_toy"):
            child = Experiment.based_on(DOI, name="child", accept_licences={"CC-BY-NC-4.0"})
    assert bundle_downloads(rig) == 1, "the bundle was downloaded once, by the first run"
    assert assets.is_cached(ref) and child.document()["based_on"]["experiment"] == "full_toy"


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


def test_a_bundle_uploaded_by_hand_and_pulled_by_its_doi_records_that_doi(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch):
    """Its lock names no record (``mime push`` did not publish it): the DOI it was pulled by names it."""
    bundle = pack(saved_toy, out=tmp_path / "full_toy.mimebundle")
    assert "publication" not in lock_of(saved_toy)
    recorded, warned = pulled_by_doi(bundle, rig, tmp_path / "work", monkeypatch)
    assert recorded["doi"] == DOI and warned == []
    assert f"- **Based on full_toy (DOI {DOI})**" in card_text(tmp_path / "work" / "child").splitlines()


def test_a_bundle_whose_lock_names_another_record_records_the_one_it_was_pulled_from_and_warns_naming_both(
        saved_toy: Path, rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    other = "10.5281/zenodo.9999"
    bundle = pack(published(saved_toy, tmp_path / "published", doi=other), out=tmp_path / "full_toy.mimebundle")
    recorded, warned = pulled_by_doi(bundle, rig, tmp_path / "work", monkeypatch)
    assert recorded["doi"] == DOI
    assert len(warned) == 1 and f"pulled from the record '{DOI}'" in warned[0] and f"names the record '{other}'" in warned[0]


def test_a_concept_doi_records_the_version_it_resolved_to(saved_toy: Path, rig: Rig, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch):
    """A concept DOI names every version and resolves to the latest: the record pulled is that version, whose DOI
    the record states, and that is the one recorded (here the lock names it too, so nothing is warned of)."""
    concept = f"10.5281/zenodo.{RECORD - 1}"
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    recorded, warned = pulled_by_doi(bundle, rig, tmp_path / "work", monkeypatch, source=concept, concept=concept,
                                     version="2")
    assert recorded["doi"] == DOI != concept and warned == []


def test_the_record_doi_is_the_one_the_record_states_where_it_is_a_dois_shape_and_else_the_one_pulled_by():
    def repository(stated: object) -> SimpleNamespace:
        return SimpleNamespace(api_response={"doi": stated})

    assert _pull._record_doi(DOI, repository("10.5281/zenodo.4243")) == "10.5281/zenodo.4243"  # pyright: ignore[reportPrivateUsage]
    for stated in (None, "", "not a doi", "10.5281/zenodo.1 x", 4243):
        assert _pull._record_doi(DOI, repository(stated)) == DOI, stated  # pyright: ignore[reportPrivateUsage]
    assert _pull._record_doi(DOI, object()) == DOI  # pyright: ignore[reportPrivateUsage]


def test_a_bundle_file_records_the_doi_its_lock_names(saved_toy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle = pack(published(saved_toy, tmp_path / "published"), out=tmp_path / "full_toy.mimebundle")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert Experiment.based_on(bundle, name="child").document()["based_on"]["doi"] == DOI


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


def test_a_cell_run_again_after_its_licence_was_refused_is_refused_again_until_the_licence_is_accepted(
        saved_toy: Path, rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The first run unpacks the folder and stops at the licence; the same call again uses that folder, and its assets
    pass the same gate: refused again as the first was, and fetched once the licence is accepted by name."""
    payload = b"a table under a non-commercial licence" * 30
    rig.server.files["/t/table.bin"] = payload
    ref = dataclasses.replace(reference(payload, rig.server.url("/t/table.bin"), name="table"),
                              licence="CC-BY-NC-4.0")
    files = {path.relative_to(saved_toy).as_posix(): path.read_bytes() for path in saved_toy.rglob("*")
             if path.is_file()}
    bundle = write_bundle(tmp_path / "full_toy.mimebundle", files, "full_toy", lock_with_assets(files, ref))
    monkeypatch.chdir(tmp_path)
    for run in range(2):
        with assets.asking(None), pytest.raises(BuildRefused) as refused, \
                (untouched(tmp_path / "full_toy") if run else warnings.catch_warnings()):
            Experiment.based_on(bundle, name="child")
        assert refused.value.codes() == ["asset.licence-not-accepted"] and not assets.is_cached(ref)
    with assets.asking(None), untouched(tmp_path / "full_toy"):
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
        write_manifest(folder)
        fake = FakeRecords(with_token(monkeypatch))
        plans[folder.name] = _plan.make_plan(copy_of(folder, tmp_path / "push" / folder.name), fake.service(),
                                             push.read_manifest(folder).record, tmp_path / "staged" / folder.name)
    parent, child = plans["arm_like"], plans["arm_variant"]
    metadata = child.payload["metadata"]
    assert metadata["related_identifiers"] == parent.payload["metadata"]["related_identifiers"]  # type: ignore[index]
    assert [each["relation_type"]["id"] for each in metadata["related_identifiers"]] == ["requires", "iscompiledby"]  # type: ignore[index]
    assert "Based on" not in str(metadata["description"]) and "Based on" not in child.card  # type: ignore[index]
    assert "Based on arm_like" in card_text(tmp_path / "arm_variant"), "mime card names it"
    assert child.target(None) == "a new draft, its DOI reserved" and child.second_concept(None) is None
