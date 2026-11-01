"""The profile's record in the saved form: what a build in exploratory leaves pending is recorded in a canonical order
(by target, then code), so the order a script declares things in never reaches the hashed record, and a folder saved
with several pending declarations loads, ``mime pending``s and ``mime verify``s. The walk's T1, written as a new user
writes it with no paperwork (its declarations out of alphabetical order, the MRS drag's settings left to
their defaults), is the case; the same declarations in reverse order give the same lock, and in alphabetical order
(the review's control, which loaded before this fix too) the same again.

A default is recorded by its kind, value and entry, never its words: a reworded basis loads every folder that took
the default, and a changed value (the control) refuses it. A load's report says the saved form's version and the
profile it is read under. A difference in MIME's own record of the profile, under the MIME that wrote the lock, is
MIME's defect, refused as internal, never as the installed entries'."""

import dataclasses
import json
from pathlib import Path

import pytest

from mime import Body, Experiment, Medium, State
from mime.build import BuildRefused, RefusalClass
from mime.cli import main as mime_main
from mime.dynamics import RigidInertial
from mime.interactions import Drag, Gravity, hydrodynamic
from mime.observations import SwimSpeed
from mime.registry import registration_of
from mime.saved import LOCK_FILE
from mime.saved._record import input_hash  # noqa: PLC2701 - the control rewrites a lock as a defect would
from mime.scene import Scene, Sphere, Z
from mime.things import Part
from mime.units import kg_per_m3, m_per_s, m_per_s2, mPa_s, ms, rad_per_s, s, um

#: T1's declarations, in the order the walk wrote them (not alphabetical).
WALK_ORDER = ("water.density", "water.viscosity", "ball.parts.sphere.density")
VALUES = {"water.density": kg_per_m3(998.0), "water.viscosity": mPa_s(1.0),
          "ball.parts.sphere.density": kg_per_m3(7850.0)}


def t1_bare(order: tuple[str, ...] = WALK_ORDER) -> Experiment:
    """The walk's T1 with its paperwork taken out: no declaration's reason, the drag's settings left out."""
    exp = Experiment("t1", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.05))
    water = exp.add_medium(Medium("water"))
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"sphere": Part(Sphere(radius=um(50.0)))}))
    for path in order:
        exp.declare(path, VALUES[path])
    exp.attach(Drag.MRS(), body=ball, medium=water)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.observe(SwimSpeed(ball, along=-Z, after=s(0.04)))
    return exp


@pytest.fixture(scope="module")
def folders(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """T1 bare saved three ways: the walk's order, reversed, and alphabetical."""
    held = {}
    for name, order in (("walk", WALK_ORDER), ("reversed", WALK_ORDER[::-1]), ("sorted", tuple(sorted(WALK_ORDER)))):
        folder = tmp_path_factory.mktemp(name) / "saved"
        t1_bare(order).build().save(folder, steps=2)
        held[name] = folder
    return held


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text())


def test_t1_bare_saves_a_folder_that_loads_pends_and_verifies(folders, capsys: pytest.CaptureFixture[str]):
    folder = folders["walk"]
    loaded = Experiment.load(folder)
    pending = [item.target for item in loaded.build().report.profile.pending]
    assert pending == ["ball.parts.sphere.density", "drag:ball.blob", "drag:ball.spacing", "water.density",
                       "water.viscosity"]
    assert mime_main(["pending", str(folder)]) == 1
    assert "5 items pending" in capsys.readouterr().out
    assert mime_main(["verify", str(folder)]) == 0


def test_the_declarations_order_never_reaches_the_lock(folders):
    """The same declarations in reverse order, and in alphabetical order (the control), give the same lock."""
    hashes = {name: lock_of(folder)["hashes"]["input"] for name, folder in folders.items()}
    assert len(set(hashes.values())) == 1, hashes
    records = {name: lock_of(folder)["resolved"]["profile"] for name, folder in folders.items()}
    assert records["walk"] == records["reversed"] == records["sorted"]
    for folder in folders.values():
        Experiment.load(folder)


def test_a_difference_in_mime_s_own_record_is_internal_and_an_entry_s_is_the_environment(folders, tmp_path: Path):
    """The lock rewritten as a defect of MIME's ordering would write it (its pending items reversed, the hash taken
    over them) is refused as internal, naming the record; the same rewrite of what the entries resolved (the nodes'
    order), the installed entries' (the environment)."""
    for part, expected in ((("profile", "pending"), RefusalClass.INTERNAL),
                           (None, RefusalClass.ENVIRONMENT)):
        folder = tmp_path / str(expected.value)
        folder.mkdir()
        for path in folders["walk"].rglob("*"):
            if path.is_file():
                target = folder / path.relative_to(folders["walk"])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        lock = lock_of(folder)
        if part is not None:
            lock["resolved"]["profile"]["pending"].reverse()
        else:
            lock["resolved"]["order"].reverse()
        document = json.loads(json.dumps(Experiment.from_document(folder / "experiment.mime.yaml").document()))
        lock["hashes"]["input"] = input_hash(document, lock["resolved"], lock["entries"])
        (folder / LOCK_FILE).write_text(json.dumps(lock))
        with pytest.raises(BuildRefused) as refused:
            Experiment.load(folder)
        (found,) = [r for r in refused.value.refusals if r.code == "load.input-hash"]
        assert found.category is expected, (expected, found)
        if part is not None:
            assert "resolved.profile.pending" in found.message and "MIME" in found.message


def test_a_reworded_default_loads_and_a_changed_value_is_refused(folders, monkeypatch: pytest.MonkeyPatch):
    """K5: the blob's basis reworded in the code, its entry's version unchanged: the folder loads. The control: the
    blob's default value changed is refused."""
    roles = registration_of(Drag.MRS()).roles
    monkeypatch.setitem(roles, "blob", dataclasses.replace(roles["blob"], basis="the same claim, in other words"))
    report = Experiment.load(folders["walk"]).load_report
    assert report is not None and report.text().splitlines()[2] == "input hash: reproduced exactly"
    monkeypatch.setattr(hydrodynamic, "DEFAULT_BLOB", 0.6)
    with pytest.raises(BuildRefused) as refused:
        Experiment.load(folders["walk"])
    assert "load.input-hash" in refused.value.codes()


def test_the_load_report_says_the_form_and_the_profile(folders):
    """K8 (D7 stated on load): the current form's version and profile; a folder of the previous form, read as
    exploratory, and why."""
    report = Experiment.load(folders["walk"]).load_report
    assert report is not None
    assert report.text().splitlines()[1] == "saved form: mime.document/3, built and read under the exploratory profile"
    previous = Experiment.load(Path(__file__).parent / "previous" / "magnetic_ball").load_report
    assert previous is not None
    line = previous.text().splitlines()[1]
    assert line.startswith("saved form: mime.document/1, the form before profiles, read under the exploratory profile")
