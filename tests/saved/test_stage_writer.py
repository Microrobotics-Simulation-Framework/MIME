"""The stage writer's versions (``mime.scene.STAGE_WRITER``, the lock's ``stage.writer``): a folder whose stage the
first writer wrote, before dev1 phase 7n placed every chain frame in the world and drew visuals, vectors and proxies
within their budget, loads and verifies, its stage rebuilt by the first writer and held to its lock; a stage that
matches no writer is refused, as before.

``previous/chain_screw/`` was written by MIME on ``v0.3`` before phase 7n (4bc1ada, the saved form's version 4), in
float64 with one BLAS thread as the saved tests run, from :func:`chain_screw` below: a screw body in a tube under a
motor whose rotor carries a magnet, so that each of the second writer's changes is in it (the motor's two frames, the
field the magnetic torque reads at the body, and a screw drawn over the proxy budget at the first writer's
resolution). Its lock records no writer. In another environment its stage need not be the same bytes (a drawn
proxy's points are the CPU's trigonometry), so the environment-free checks compare the stage's structure outside its
derived prims; the checks of the refusals run on a folder this test writes in the first writer's form, in the
environment it loads in.

The controls: the fixture's stage is not the second writer's; the same first-writer folder with its lock naming the
second writer is refused (``load.stage-hash``), with a writer this MIME does not have it is refused
(``load.stage-writer``), and a stage neither writer writes is refused; a live session of the first writer's folder,
whose stage has no prims for the motor's frames, is refused (``live.stage-writer``), and of the same experiment saved
again it is not."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mime import Apparatus, Body, Experiment, State, Structure
from mime.build import ArgumentRefused, stage_spec
from mime.dynamics import Motor, RigidInertial
from mime.fields import PointDipole
from mime.interactions import Magnetic
from mime.saved import DOCUMENT_FILE, LOCK_FILE
from mime.saved._folder import LoadRefused
from mime.scene import STAGE_FILE, STAGE_WRITER, Cylinder, Scene, Screw, Tube, X, Z, sha256_hex, stage_edits, \
    stage_text
from mime.things import Part, PermanentMagnet
from mime.units import A_m2, dimensionless, kg, kg_m2, m, m_per_s, m_per_s2, mm, ms, rad, rad_per_s

FIXTURE = Path(__file__).parent / "previous" / "chain_screw"


def chain_screw() -> Experiment:
    """The fixture's experiment, as it was written for ``v0.3`` at 4bc1ada."""
    exp = Experiment("chain_screw", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1.0))
    exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                      length=mm(100.0))))
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={"shell": Part(Screw(
            radius=mm(1.56), length=mm(1.6), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2))}))
    exp.declare("umr.mass", kg(8.5e-5), reason="a millimetre body, round numbers of its order")
    exp.declare("umr.inertia", kg_m2((1.0e-10, 1.2e-10, 0.8e-10)), reason="a millimetre body, its order")
    exp.declare("umr.magnetic_moment", A_m2((0.0, 1.7e-3, 0.0)), reason="a millimetre body's magnet, its order")
    exp.add_apparatus(Apparatus(
        "spin", chain={"motor": Motor.Kinematic(axis=Z, prescribed="angle", initial_angle=rad(0.0),
                                                at=m((0.0, 0.0, 0.1)))},
        parts={"magnet": Part(Cylinder(radius=m(17.5e-3), length=m(20e-3), axis=(1.0, 0.0, 0.0)), mount="motor.rotor")},
        magnets={"magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))},
        fields={"magnetic": PointDipole(part="magnet")}, commands={"motor.angle": rad(0.25)}))
    exp.declare("spin.magnetic_moment", A_m2(18.89), reason="de Jongh's rotor magnet")
    exp.attach(Magnetic.Torque(), body="umr", field="spin.magnetic")
    return exp


def copied(tmp_path: Path) -> Path:
    folder = tmp_path / "chain_screw"
    shutil.copytree(FIXTURE, folder)
    return folder


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text())


def write_lock(folder: Path, lock: dict) -> None:
    (folder / LOCK_FILE).write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")


def test_the_fixture_is_a_first_writer_folder():
    assert yaml.safe_load((FIXTURE / DOCUMENT_FILE).read_text())["schema"] == "mime.document/4"
    assert "writer" not in lock_of(FIXTURE)["stage"]


def test_the_first_writer_rebuilds_the_fixtures_stage_and_the_second_does_not():
    plans = chain_screw().plan()
    held = (FIXTURE / STAGE_FILE).read_text()
    assert not stage_edits(held, stage_text(stage_spec(plans, writer=1))).outside
    # The control: the second writer's stage differs outside its derived prims (the frames, the vector).
    assert stage_edits(held, stage_text(stage_spec(plans, writer=STAGE_WRITER))).outside


def test_the_fixture_loads_and_verifies(tmp_path):
    folder = copied(tmp_path)
    experiment = Experiment.load(folder)
    report = experiment.load_report
    assert report is not None and report.text().splitlines()[2] == "input hash: reproduced exactly"
    if report.same_environment:
        assert report.stage_writer == 1 and "stage: written by stage writer 1" in report.text()
    done = subprocess.run([sys.executable, "-m", "mime.cli", "verify", str(folder)], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify: reproduced 8 steps" in done.stdout


@pytest.fixture(scope="module")
def first_writer(tmp_path_factory) -> Path:
    """The experiment saved here, then its stage and lock put in the first writer's form, as the first writer would have
    written them in this environment."""
    folder = tmp_path_factory.mktemp("first") / "chain_screw"
    built = chain_screw().build()
    built.save(folder)
    text = stage_text(stage_spec(chain_screw().plan(), writer=1))
    (folder / STAGE_FILE).write_text(text, encoding="utf-8", newline="\n")
    lock = lock_of(folder)
    assert lock["stage"]["writer"] == STAGE_WRITER
    del lock["stage"]["writer"]
    lock["stage"]["sha256"] = sha256_hex(text.encode("utf-8"))
    write_lock(folder, lock)
    return folder


def test_a_first_writer_folder_loads_in_the_environment_it_was_written_in(first_writer):
    report = Experiment.load(first_writer).load_report
    assert report is not None and report.same_environment and report.stage_writer == 1
    saved_again = Experiment.load(first_writer)
    assert saved_again.build().record.stage_hash != lock_of(first_writer)["stage"]["sha256"]


def test_the_control_a_lock_naming_another_writer_or_a_stage_neither_writes_is_refused(first_writer, tmp_path):
    named = tmp_path / "named"
    shutil.copytree(first_writer, named)
    lock = lock_of(named)
    write_lock(named, {**lock, "stage": {**lock["stage"], "writer": STAGE_WRITER}})
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(named)
    assert refused.value.codes() == ["load.stage-hash"]
    write_lock(named, {**lock, "stage": {**lock["stage"], "writer": STAGE_WRITER + 1}})
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(named)
    assert "load.stage-writer" in refused.value.codes()
    neither = tmp_path / "neither"
    shutil.copytree(first_writer, neither)
    text = (neither / STAGE_FILE).read_text() + "\n"  # one byte more than either writer writes
    (neither / STAGE_FILE).write_text(text, encoding="utf-8", newline="\n")
    lock = lock_of(neither)
    write_lock(neither, {**lock, "stage": {**lock["stage"], "sha256": sha256_hex(text.encode("utf-8"))}})
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(neither)
    assert refused.value.codes() == ["load.stage-hash"]


def test_a_live_session_of_a_first_writer_folder_with_a_chain_is_refused_and_one_saved_again_is_not(first_writer,
                                                                                                    tmp_path):
    with pytest.raises(ArgumentRefused, match="live.stage-writer"):
        Experiment.load(first_writer).build().live(pace=None)
    Experiment.load(first_writer).save(tmp_path / "again")
    assert lock_of(tmp_path / "again")["stage"]["writer"] == STAGE_WRITER
    session = Experiment.load(tmp_path / "again").build().live(pace=None)
    assert session.hello()["ok"]
