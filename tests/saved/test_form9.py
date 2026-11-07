"""Saved form 9: the resolved record holds the full inertia tensor a body's parts give (nine numbers, row-major), where
an earlier form held its diagonal; a declared inertia keeps the form it was declared in. No key is added. A folder of
an earlier form loads as it was: its document names ``rigid_inertial`` 0.3.0 (or 0.2.0), kept registered, which builds
its node from the diagonal as before, and the rebuilt record is written back into the folder's form (the parts' tensor
as its diagonal; a declaration's source naming the parts' tensor as the three numbers it named, with the words where
the products were dropped) for its input hash. A folder that rested on exploratory's waiver of the products is refused
at load by the kept version's own words, never reproduced with them carried.

The folders of earlier forms, each written by MIME at the last commit of its form, float64, one BLAS thread, four
usable CPUs (``taskset -c 0-3``), pinned by their sha256:

- ``form8/rigid_parts`` (9a8036c): two free rigid bodies at 1 ms, no gravity and no law: ``ball``, a 1 mm sphere of
  density 1000 kg/m^3 whose inertia its parts give, at the origin moving at 1 mm/s along x and turning at (30, -20,
  10) rad/s; and ``umr``, FL-9's screw of density 1410 kg/m^3 at 10 mm along x turning at (40, 30, -20) rad/s, its
  inertia declared as its parts' diagonal, (1.3043197461318855e-10, 4.631130043564001e-10, 4.570054968582633e-10)
  kg m^2. Run six steps chunked by one step and saved, so its lock names the one-step program and records both
  counts: ``built.run(s(6e-3), history=True, hook=lambda chunk: None, chunk=s(1e-3)); built.save(folder)``.

- ``form8/dropped_products`` (9a8036c): FL-9's parts alone, ``profile_cases.parts_screw("exploratory")`` built and
  saved: its inertia the parts' diagonal, the products dropped and stamped pending.
- ``form7/based_on_child`` (f2cdac6): the 0.3.0.dev1 wheel's ``form6/timed_toy``, based on and its spring made stiffer
  (``test_based_on.based_on_edited_saved``), saved: a document that records its parent.
- ``form8/chain_screw``: the current folder as form 8 wrote it.

Each loads and verifies under form 9 with its form's keys as they are (form 8's reference program and counts, form 7's
``based_on``). A reader of version 8 refuses a folder of version 9 as ``load.newer-form``, the document and the lock
each once. A folder holding a 3 x 3 inertia round-trips: saved, loaded in a fresh process, rebuilt, the same first
steps.

**The controls.** With the write-back removed the form-8 folder whose ball's inertia its parts give is refused as
``load.input-hash``, naming that value, on every run (it is a hash: the same input, the same refusal). The form left
unstepped (the folder's files stating version 8) is read by version 8's reader without a ``load.newer-form``, so the
refusal is the step's."""

import hashlib
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.nodes.body import RigidInertialMidpointNode
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, read_folder
from mime.saved import _folder, _versions
from mime.saved._document import document_hash, parse_yaml
from mime.saved._record import input_hash
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, INERTIA_TENSOR_ADDED, LOCK_SCHEMAS, resolved_in_form
from mime.testing import round_trip_failures

SAVED = Path(__file__).parent
FORM8_CHAIN = SAVED / "form8" / "chain_screw"
CURRENT_FOLDER = SAVED / "current" / "chain_screw"
RIGID8 = SAVED / "form8" / "rigid_parts"
DROPPED8 = SAVED / "form8" / "dropped_products"
CHILD7 = SAVED / "form7" / "based_on_child"
PINNED = {
    RIGID8: {"experiment.mime.yaml": "584e310df4a31d69a88a46c19d8a7d646c8bfdda5a6c39e613d27d334b8624a7",
             "mime.lock": "5ee10a50929b60c3d6f905b4f969fcef5efc322dd5309675898ff6d721dfa71d",
             "reference.npz": "3ff9bd9ade7bcd24088de67d5176553108ded79ffd61cdefad4d597d6d6985b7",
             "scene/scene.usda": "0fb2c8cb1512efd34423f839555ca777ffd974010e923afcace163882ed817ae"},
    CHILD7: {"experiment.mime.yaml": "a940d76217134e69d6f2b3163e7a25e47c9b84a46f797febe58fea448968e600",
             "mime.lock": "885df88f3ce35b7304acf4a5ca9b1de47549a4af7d07a2eaa1589f4c2131c4ea",
             "reference.npz": "182d164ee7aa9e4d3c708c83c04eaaaf7c8c1058174035afd265698bd6a4a79d",
             "scene/scene.usda": "78a93ae11ce3de31021b76e56c8f7d849d2ee1c1a292bdf78f83a7b136132039"},
}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM8_READS = 8


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def document_of(folder: Path) -> dict:
    return parse_yaml((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))  # type: ignore[return-value]


def test_version_nine_holds_the_parts_tensor_and_this_mime_writes_it(tmp_path: Path):
    from profile_cases import parts_screw

    assert CURRENT == INERTIA_TENSOR_ADDED == 9
    assert DOCUMENT_SCHEMAS[CURRENT] == "mime.document/9" and LOCK_SCHEMAS[CURRENT] == "mime.lock/9"
    parts_screw("exploratory").build().save(tmp_path / "screw")
    lock = lock_of(tmp_path / "screw")
    held = lock["resolved"]["values"]["umr.inertia"]
    assert lock["schema"] == "mime.lock/9" and held["source"] == "derived from its parts" and len(held["value"]) == 9
    assert held["value"][5] == held["value"][7] != 0.0, "the product of inertia, in both places"


@pytest.mark.parametrize("folder", [FORM8_CHAIN, CURRENT_FOLDER], ids=["form8", "current"])
def test_a_committed_folders_lock_holds_the_hashes_its_document_makes(folder: Path):
    document = document_of(folder)
    lock = lock_of(folder)
    assert lock["hashes"]["document"] == document_hash(document)  # type: ignore[arg-type]
    assert lock["hashes"]["input"] == input_hash(document, lock["resolved"], lock["entries"])  # type: ignore[arg-type]


def test_saving_again_under_form_nine_moves_the_document_and_input_hashes_and_never_the_others():
    eight, nine = lock_of(FORM8_CHAIN)["hashes"], lock_of(CURRENT_FOLDER)["hashes"]
    assert [key for key in ("document", "input", "derived", "observations") if eight[key] != nine[key]] == \
        ["document", "input"]


@pytest.mark.parametrize("folder", list(PINNED), ids=["form8_rigid_parts", "form7_based_on_child"])
def test_the_earlier_forms_folders_are_the_folders_their_commits_wrote_byte_for_byte(folder: Path):
    found = {path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(folder.rglob("*")) if path.is_file()}
    assert found == PINNED[folder]


def verify_asks(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """The programs ``verify`` reruns, as ``_history_on`` is asked for them."""
    from mime.experiment import _experiment

    asked: list[object] = []
    real = _experiment.BuiltExperiment._history_on

    def recorded(self, graph, steps, program):
        asked.append(program)
        return real(self, graph, steps, program)

    monkeypatch.setattr(_experiment.BuiltExperiment, "_history_on", recorded)
    return asked


def test_a_form_eight_folder_with_inertia_from_parts_loads_on_the_kept_version_and_verifies_by_its_program(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Its ball's inertia, its parts', and FL-9's source naming its parts' tensor are written back into form 8, so the
    input hash is reproduced; ``rigid_inertial`` 0.3.0 builds the node it built from the diagonal, so the derived graph
    is the saved one; ``verify`` reruns the one-step program its lock names, and the lock's counts are read as they
    are."""
    folder = Path(shutil.copytree(RIGID8, tmp_path / "rigid_parts"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    lines = loaded.load_report.text().splitlines()
    assert lines[1:3] == [f"saved form: mime.document/8, read as version {CURRENT}", "input hash: reproduced exactly"]
    assert any(line.startswith("derived graph: the same graph") for line in lines), lines
    asked = verify_asks(monkeypatch)
    built = loaded.build()
    for body in ("ball", "umr"):
        assert type(built.graph.get_node(body)) is RigidInertialMidpointNode
        assert len(built.graph.get_node(body).constants()["inertia"]) == 3
    assert built.verify().reproduced and asked == ["one-step"]
    environment = read_folder(folder).lock["environment"]
    assert (environment["usable_cpus"], environment["blas_threads"]) == (4, 1)


def test_a_form_seven_based_on_child_loads_keeps_its_parent_and_verifies(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch):
    folder = Path(shutil.copytree(CHILD7, tmp_path / "based_on_child"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/7, read as version {CURRENT}", "input hash: reproduced exactly"]
    assert loaded.document()["based_on"] == document_of(CHILD7)["based_on"]
    assert document_of(CHILD7)["based_on"]["experiment"] == "timed_toy"
    asked = verify_asks(monkeypatch)
    assert loaded.build().verify().reproduced and asked == [None], "no program named: the history scan"


@contextmanager
def write_back_removed() -> Iterator[None]:
    """The planted step: the rebuilt record left in form 9 for an earlier form's lock."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_versions, "_inertia_in_form", lambda path, record: record)
        yield


def test_control_with_the_write_back_removed_the_form_eight_folder_is_refused_naming_the_parts_inertia(
        tmp_path: Path):
    folder = Path(shutil.copytree(RIGID8, tmp_path / "rigid_parts"))
    with write_back_removed(), pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    (found,) = refused.value.refusals
    assert found.code == "load.input-hash"
    assert "ball.inertia" in found.message and "umr.inertia" in found.message, found.message


def test_the_write_back_writes_the_parts_tensor_as_its_diagonal_and_a_source_as_the_three_it_named():
    nine = [1.0e-10, 0.0, 0.0, 0.0, 4.6e-10, 5.0e-13, 0.0, 5.0e-13, 4.5e-10]
    named = ("declared in the experiment, in place of the value its parts give, (1e-10, 0, 0, 0, 4.6e-10, 5e-13, 0, "
             "5e-13, 4.5e-10) kg m^2")
    plain = "declared in the experiment, in place of the value its parts give, (1e-10, 0, 0, 0, 2e-10, 0, 0, 0, 3e-10) kg m^2"
    record = {"values": {
        "a.inertia": {"value": nine, "source": "derived from its parts"},
        "b.inertia": {"value": [1e-10, 2e-10, 3e-10], "source": named},
        "c.inertia": {"value": [1e-10, 2e-10, 3e-10], "source": plain},
        "a.mass": {"value": 1e-6, "source": "derived from its parts"},
        "d.inertia": {"value": nine, "source": "declared in the experiment"}}}
    written = resolved_in_form(record, 8)["values"]
    assert written["a.inertia"]["value"] == [1.0e-10, 4.6e-10, 4.5e-10]
    assert written["b.inertia"]["source"] == ("declared in the experiment, in place of the value its parts give, (1e-10, "
                                              "4.6e-10, 4.5e-10) kg m^2, its products of inertia dropped")
    assert written["c.inertia"]["source"] == ("declared in the experiment, in place of the value its parts give, (1e-10, "
                                              "2e-10, 3e-10) kg m^2")
    assert written["a.mass"] == record["values"]["a.mass"] and written["d.inertia"] == record["values"]["d.inertia"], \
        "a declared tensor kept the form it was declared in"
    assert resolved_in_form(record, CURRENT) == record


def test_a_form_eight_folder_that_rested_on_the_waiver_of_its_products_is_refused_by_the_kept_versions_words(
        tmp_path: Path):
    """Its document names ``rigid_inertial`` 0.3.0, which takes three moments; its parts' tensor has a product above
    what 0.3.0 took as none, and 0.3.0 dropped it only through exploratory's waiver, which went with 0.4.0."""
    assert [item["target"] for item in lock_of(DROPPED8)["resolved"]["profile"]["pending"]] == ["umr.inertia"]
    folder = Path(shutil.copytree(DROPPED8, tmp_path / "dropped_products"))
    with pytest.raises(BuildRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["value.products-of-inertia"], refused.value
    (found,) = refused.value.refusals
    assert "rigid_inertial 0.3.0" in found.message and "products of inertia" in found.message
    assert "exploratory's waiver" in found.message and "refused at load" in found.message
    assert "re-create the experiment from its script" in found.action and "exp.declare" not in found.action, \
        "the folder's route, not a script's"


def test_a_script_that_builds_the_kept_version_with_products_gets_the_scripts_route_under_the_same_code():
    """The load's wording is the load's: the same refusal from a script's build names how to build a body."""
    from mime.dynamics.body import RigidInertialMoments

    found = RigidInertialMoments().value_problems(
        {"inertia": (1e-10, 2e-11, 0.0, 2e-11, 1e-10, 0.0, 0.0, 0.0, 1e-10)})
    assert [each.code for each in found] == ["value.products-of-inertia"]
    assert "exp.declare" in found[0].action and "exploratory" not in found[0].message


@contextmanager
def reader_of(newest: int) -> Iterator[None]:
    """The loader reading the forms up to ``newest``: version 8's, this MIME before the step."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_folder, "DOCUMENT_SCHEMAS", {v: DOCUMENT_SCHEMAS[v] for v in range(1, newest + 1)})
        patch.setattr(_folder, "LOCK_SCHEMAS", {v: LOCK_SCHEMAS[v] for v in range(1, newest + 1)})
        yield


def codes_of(folder: Path) -> list[str]:
    try:
        read_folder(folder)
    except LoadRefused as refused:
        return refused.codes()
    return []


def restated(folder: Path, out: Path, version: int) -> Path:
    copy = Path(shutil.copytree(folder, out))
    path = copy / DOCUMENT_FILE
    path.write_text(re.sub(r"^schema: .*$", f"schema: {DOCUMENT_SCHEMAS[version]}", path.read_text(encoding="utf-8"),
                           count=1, flags=re.MULTILINE), encoding="utf-8")
    lock = lock_of(copy)
    lock["schema"] = LOCK_SCHEMAS[version]
    (copy / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    return copy


@pytest.fixture(scope="module")
def current_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from mime.testing.toys import full_toy

    root = tmp_path_factory.mktemp("form9")
    full_toy().save(root / "full_toy")
    return root / "full_toy"


def test_version_eights_reader_refuses_a_folder_of_version_nine_as_newer_each_file_once(current_folder: Path):
    assert codes_of(current_folder) == [], "this MIME reads it"
    with reader_of(FORM8_READS):
        assert codes_of(current_folder) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_read_by_version_eights_reader_with_no_newer_form(current_folder: Path,
                                                                                             tmp_path: Path):
    unstepped = restated(current_folder, tmp_path / "unstepped", FORM8_READS)
    with reader_of(FORM8_READS):
        assert NEWER not in codes_of(unstepped)


def spinning_screw() -> Experiment:
    """FL-9's parts alone, free and turning about no principal axis, so that its products of inertia act in the first
    steps (no confinement: the containment watch reads a body the centre-held contact holds, and none is here)."""
    from mime import Body, State
    from mime.dynamics import RigidInertial
    from mime.scene import Scene, Screw, Z
    from mime.things import Part
    from mime.units import dimensionless, kg_per_m3, m_per_s, m_per_s2, mm, ms, rad_per_s

    exp = Experiment("spinning_screw", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1.0))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((40.0, 30.0, -20.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.parts.shell.density", kg_per_m3(1410.0), reason="FL-9 resin plus magnets, mean density")
    return exp


@pytest.mark.parametrize("declared", [False, True], ids=["from_its_parts", "declared"])
def test_a_folder_holding_a_full_inertia_tensor_round_trips_in_a_fresh_process(tmp_path: Path, declared: bool):
    """FL-9's parts' tensor, as the parts give it or declared as its nine numbers, on 0.4.0's node."""
    from mime.units import kg_m2

    experiment = spinning_screw()
    if declared:
        tensor = experiment.build().record.resolved["values"]["umr.inertia"]["value"]
        experiment.declare("umr.inertia", kg_m2(tuple(float(c) for c in tensor)), reason="the parts' tensor, as given")
    assert len(np.asarray(experiment.build().graph.get_node("umr").constants()["inertia"])) == 9
    assert round_trip_failures(experiment, tmp_path / "saved") == []
