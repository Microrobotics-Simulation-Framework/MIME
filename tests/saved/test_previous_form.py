"""A folder saved before phase 7l-a (``mime.document/1``, ``mime.lock/1``) loads under the current schema, held to its
own lock (design #190 item 7, ``mime.saved._versions``).

``previous/magnetic_ball/`` was written by MIME at the commit before 7l-a (e36c017), in float64 with one BLAS thread as
the saved tests run, from the experiment below, with nothing of 7l-a's API in it: a resin ball with an N45 magnet inside,
the magnet's remanence read from a vendor's record, accepted with a text and its nominal value declared, water's density
read from its record, the resin's density declared with its reason. (It has no MRS drag: ``test_form_v2.py`` holds a version 2 folder with
one.) It loads with its profile read as exploratory (the reason is in ``_versions``' docstring), its input hash reproduced
exactly in the previous form, and it verifies. The controls: the same folder with its document edited after its lock
is refused, and a previous document holding a key version 2 added is refused, as is a previous document beside a
current lock.

The experiment the folder was saved from, as written then::

    exp = Experiment("magnetic_ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1.0),
                     conditions={"temperature": degC(20.0, plus_minus=0.5)})
    water = exp.add_medium(Medium("water", material=Record(
        "water", packaged_record("openmaterica_example_water"), require={"test_method": ("reference_formulation",)})))
    n45 = Record("n45", packaged_record("openmaterica_example_ndfeb_n45"),
                 accept={"magnetic.remanence": Acceptance(
                     ("stated_without_origin", "condition_unstated:temperature", "condition_unstated:specimen_shape",
                      "condition_unstated:test_method"), reason="a vendor's grade limits, the only source at hand")},
                 nominal={"magnetic.remanence": Nominal(T(1.345), reason="the middle of the grade's limits")})
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts={
        "shell": Part(Sphere(radius=mm(1.0))),
        "magnet": Part(Box(size=mm((0.5, 0.5, 0.5))), material=n45, at=mm((0.0, 0.0, 0.0)), inside="shell")},
        magnets={"magnet": PermanentMagnet(direction=(0.0, 1.0, 0.0))}))
    exp.declare("ball.parts.shell.density", kg_per_m3(1410.0), reason="a printed resin's density, by hand")
    field = exp.add_apparatus(Apparatus("field", fields={
        "magnetic": UniformRotating(rotation_axis=(1.0, 0.0, 0.0), initial_direction=(0.0, 1.0, 0.0))},
        commands={"frequency": Hz(5.0), "amplitude": mT(1.0)}))
    exp.attach(Magnetic.Torque(), body=ball, field=field.magnetic)
    exp.attach(Gravity(), body=ball, medium=water)
    exp.observe(SwimSpeed(ball, along=-Z, after=s(0.004)))
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from profile_cases import Paperwork, settling_ball

from mime.experiment import Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE
from mime.saved._folder import LoadRefused
from mime.saved._versions import (
    PREVIOUS_PROFILE,
    document_in_form,
    resolved_in_form,
    upgraded_document,
    upgraded_lock,
)
from mime.units import mm
from mime.validity import chosen, pending

FIXTURE = Path(__file__).parent / "previous" / "magnetic_ball"


def copied(tmp_path: Path) -> Path:
    folder = tmp_path / "magnetic_ball"
    shutil.copytree(FIXTURE, folder)
    return folder


def test_the_fixture_is_a_previous_folder():
    assert yaml.safe_load((FIXTURE / DOCUMENT_FILE).read_text())["schema"] == "mime.document/1"
    assert json.loads((FIXTURE / LOCK_FILE).read_text())["schema"] == "mime.lock/1"


def test_a_previous_folder_loads_as_exploratory(tmp_path: Path):
    experiment = Experiment.load(copied(tmp_path))
    assert experiment._settings.profile == PREVIOUS_PROFILE == "exploratory"
    assert experiment.load_report is not None
    assert experiment.load_report.text().splitlines()[2] == "input hash: reproduced exactly"
    assert experiment.load_report.text().splitlines()[1].startswith(
        "saved form: mime.document/1, the form before profiles, read under the exploratory profile")
    built = experiment.build()
    assert built.report.profile is not None and built.report.profile.pending == ()


def test_mime_verify_reproduces_a_previous_folder(tmp_path: Path):
    """In a fresh process, as a user runs it: ``mime verify`` applies the lock's precision before it loads, and its
    process starts with the one BLAS thread the tests' processes set (``tests/conftest.py``), as the fixture was
    written; a test process started its BLAS before that, so its last bits are another process's."""
    done = subprocess.run([sys.executable, "-m", "mime.cli", "verify", str(copied(tmp_path))], capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "input hash: reproduced exactly" in done.stdout
    assert "verify: reproduced 8 steps" in done.stdout


def test_the_control_a_previous_document_edited_after_its_lock_is_refused(tmp_path: Path):
    folder = copied(tmp_path)
    text = (folder / DOCUMENT_FILE).read_text()
    assert "a printed resin's density, by hand" in text
    (folder / DOCUMENT_FILE).write_text(text.replace("a printed resin's density, by hand", "a printed resin's density"))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["load.input-hash"]


def test_the_control_a_previous_document_holding_a_key_of_version_two_is_refused(tmp_path: Path):
    folder = copied(tmp_path)
    document = yaml.safe_load((folder / DOCUMENT_FILE).read_text())
    (folder / DOCUMENT_FILE).write_text(yaml.safe_dump({**document, "profile": "research"}, sort_keys=False))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert "load.document-shape" in refused.value.codes()


def test_the_control_a_previous_document_beside_a_current_lock_is_refused(tmp_path: Path):
    folder = copied(tmp_path)
    lock = json.loads((folder / LOCK_FILE).read_text())
    (folder / LOCK_FILE).write_text(json.dumps(upgraded_lock(lock)[0]))
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert "load.lock-shape" in refused.value.codes()


def test_the_upgrade_and_the_previous_form_are_inverse():
    document = yaml.safe_load((FIXTURE / DOCUMENT_FILE).read_text())
    upgraded, version, renamed = upgraded_document(document)
    assert version == 1 and renamed == () and upgraded["profile"] == "exploratory" and upgraded["sources"] == {}
    assert upgraded["schema"] == "mime.document/3"
    assert document_in_form(upgraded, 1) == document
    lock = json.loads((FIXTURE / LOCK_FILE).read_text())
    upgraded_held, version = upgraded_lock(lock)
    assert version == 1 and resolved_in_form(upgraded_held["resolved"], 1) == lock["resolved"]
    current = {**document, "schema": "mime.document/3"}
    assert upgraded_document(current) == (current, 3, ())


def test_a_previous_folder_saved_again_is_of_the_current_form(tmp_path: Path):
    experiment = Experiment.load(copied(tmp_path))
    experiment.save(tmp_path / "again")
    assert yaml.safe_load((tmp_path / "again" / DOCUMENT_FILE).read_text())["schema"] == "mime.document/3"
    assert Experiment.load(tmp_path / "again").load_report is not None


def test_the_current_form_round_trips_reasons_sources_choices_and_pending(tmp_path: Path):
    """Version 2 keeps every reason beside its value, each ``exp.source`` and every pending reason: what is saved
    loads to the same input hash and the same items pending."""
    experiment = settling_ball(Paperwork(acceptance=pending(), radius_reason=chosen("the radius a study sweeps"),
                                         spacing=mm(0.2, reason="converged at half of it"),
                                         sources={"dt": "the step the drag is converged at",
                                                  "drag:ball.blob": pending()}))
    built = experiment.build()
    built.save(tmp_path / "saved")
    document = yaml.safe_load((tmp_path / "saved" / DOCUMENT_FILE).read_text())
    assert document["sources"] == {"drag:ball.blob": {"pending": True}, "dt": "the step the drag is converged at"}
    assert document["declared"]["ball.parts.shell.density"]["reason"] == {"pending": True}
    shell = document["things"]["ball"]["parts"]["shell"]["solid"]["settings"]["radius"]
    assert shell["reason"] == {"chosen": "the radius a study sweeps"}
    loaded = Experiment.load(tmp_path / "saved")
    assert loaded.document() == experiment.document()
    again = loaded.build()
    assert [item.line() for item in again.report.profile.items] == [item.line() for item in built.report.profile.items]
