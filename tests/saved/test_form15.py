"""Saved form 15: an apparatus's ports in the document (``things.<apparatus>.ports``), the controller's ``believed``
slot, the resolved record's ``ports``, a controller read's class ``measurement``; and a kinematic arm's servo moved
onto its apparatus's ports. A form-14 document's arm ``servo`` is read as a first-order lag on each of the arm's
position commands, which the arm's own node computes as it did: the derived graph is unchanged, and the document and the
resolved record are written back into form 14 for the lock's hashes (the ports taken off, the arm's ``servo`` put back).

The folder of the fourteenth form, ``form14/servo_arm``, is written by ``write_form14.py`` under MIME at the last commit
of form 14 (82219938), float64, four usable CPUs (``taskset -c 0-3``), one BLAS thread, and pinned by its sha256: the
two-joint arm of ``tests/descriptions/c_arm.urdf``, kinematic, its servo 20 ms, commanded past its limits. It holds what
the step moves, as form 14's own code wrote it.

**The controls.** The folder loads, rebuilds and verifies with its derived hash the lock's (design section 9,
control 4); with the servo dropped by the upgrade instead of moved onto ports, the load refuses it
(``load.input-hash``). A form-14 document holding the ports or a form-14 lock holding the resolved ports is refused
(``load.document-shape``, ``load.lock-shape``), and with the step's check removed it is read as declaring them; the
unedited folder reads with none. Version 14's reader refuses a folder of version 15 as ``load.newer-form``, the
document and the lock each once, and the form left unstepped is read by it with none."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from test_form11 import codes_of, reader_of, restated

from mime.experiment import BuiltExperiment, Experiment
from mime.saved import DOCUMENT_FILE, LOCK_FILE, LoadRefused, _folder, _versions, read_folder
from mime.saved._versions import CURRENT, DOCUMENT_SCHEMAS, LOCK_SCHEMAS, PORTS_ADDED, SERVO_REASON

SAVED = Path(__file__).parent
SERVO14 = SAVED / "form14" / "servo_arm"
PINNED = {"descriptions/5a99c19cd9e8821a-c_arm.urdf": "5a99c19cd9e8821a42ee25c1631b6eed71126c466742d1b888b090c42aa949b2",
          "experiment.mime.yaml": "43eefc80809c9100ca8bcd3477205ec28a671f577faee50781bad254b76a2521",
          "mime.lock": "02a2aa163e3cef46d2a5a2407ed49ec09161e2def841ac6e2c72d27e5e253856",
          "reference.npz": "151c5f74354db6cf6fe1c9106efad1b608946322e1d3c090839f0d043860c6cb",
          "scene/scene.usda": "c783351b7f6a77e88e68f67666be2525c81e239fe5aa2529ba6994e76cc4dbea"}
NEWER = "load.newer-form"
#: The form this MIME read before the step.
FORM14_READS = 14
LAG = {"name": "response.first_order_lag", "version": "0.1.0", "settings": {"tau": {"value": 0.02, "unit": "s"}}}
POSITION_COMMANDS = ("arm.joint_displacements", "arm.joint_positions")


def document_of(folder: Path) -> dict:
    return yaml.safe_load((folder / DOCUMENT_FILE).read_text(encoding="utf-8"))


def lock_of(folder: Path) -> dict:
    return json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))


def test_the_form_fourteen_folder_is_the_folder_its_commit_wrote_byte_for_byte():
    found = {path.relative_to(SERVO14).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(SERVO14.rglob("*")) if path.is_file()}
    assert found == PINNED


def test_the_form_fourteen_folder_holds_an_arm_with_a_servo_and_no_port():
    """The premise of every test below: the arm's servo in the document's settings and in the resolved record's, and
    nothing version 15 added."""
    document, lock = document_of(SERVO14), lock_of(SERVO14)
    assert document["schema"] == "mime.document/14" and lock["schema"] == "mime.lock/14"
    (arm,) = [s for s in document["things"]["carm"]["chain"] if s["name"] == "arm"]
    assert arm["settings"]["servo"] == {"value": 0.02, "unit": "s"}
    assert lock["resolved"]["entries"]["carm.arm"]["settings"]["servo"] == {"unit": "s", "value": 0.02}
    assert _versions.step_15_held(document, lock=False) == _versions.step_15_held(lock, lock=True) == []


def test_the_step_reads_the_servo_as_a_lag_on_each_position_command_and_writes_it_back():
    document, lock = document_of(SERVO14), lock_of(SERVO14)
    upgraded, version, _ = _versions.upgraded_document(document)
    assert version == FORM14_READS
    apparatus = upgraded["things"]["carm"]
    (arm,) = [s for s in apparatus["chain"] if s["name"] == "arm"]
    assert "servo" not in arm["settings"]
    assert apparatus["ports"] == {command: {"port": "command", "response": LAG, "reason": SERVO_REASON}
                                  for command in POSITION_COMMANDS}
    assert _versions.document_in_form(upgraded, FORM14_READS) == document
    assert _versions.resolved_in_form(lock["resolved"], FORM14_READS) == lock["resolved"]
    assert _versions.physics_in_form(lock["derived"]["physics"], FORM14_READS) == lock["derived"]["physics"]


def test_the_form_fourteen_folder_loads_rebuilds_and_verifies_its_program_unchanged(tmp_path: Path):
    """Design section 9, control 4: the servo read as its ports' lag builds the program form 14 built, the lag computed
    by the arm's node."""
    folder = Path(shutil.copytree(SERVO14, tmp_path / "servo_arm"))
    loaded = Experiment.load(folder)
    assert loaded.load_report is not None
    assert loaded.load_report.text().splitlines()[1:3] == [
        f"saved form: mime.document/14, read as version {CURRENT}", "input hash: reproduced exactly"]
    built = loaded.build()
    assert built.report.derived_hash == lock_of(SERVO14)["hashes"]["derived"]
    assert all(line.endswith("response.first_order_lag 0.1.0 (computed by its receiver)")
               for line in built.report.ports) and len(built.report.ports) == 2
    assert built.verify().reproduced


def test_control_the_servo_dropped_by_the_upgrade_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The load above reproduces because the servo is read as a lag, and for no other reason: with the upgrade dropping
    the servo instead, the rebuild resolves no lag and the load refuses it."""
    read = _versions._servo_read  # noqa: SLF001

    def dropped(apparatus: dict) -> dict:
        moved = read(apparatus)
        return {key: value for key, value in moved.items() if key != _versions.PORTS_KEY}

    monkeypatch.setattr(_versions, "_servo_read", dropped)
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(Path(shutil.copytree(SERVO14, tmp_path / "servo_arm")))
    assert "load.input-hash" in refused.value.codes()


#: Each place the step adds to, as an edit of the form-14 folder: a port in the document, a resolved port in the lock.
PORT = {"port": "command", "latency": {"value": 0.001, "unit": "s"}, "reason": "a link one step late"}
EDITS = {"document: ports": (DOCUMENT_FILE, lambda d: d["things"]["carm"].update(ports={"arm.joint_positions": PORT})),
         "lock: resolved ports": (LOCK_FILE, lambda d: d["resolved"].update(ports={}))}
HELD_AT = {"document: ports": ("load.document-shape", "a mime.document/14 document holds things.carm.ports"),
           "lock: resolved ports": ("load.lock-shape", "a mime.lock/14 lock holds resolved.ports")}


def edited(tmp_path: Path, edit: str) -> Path:
    folder = Path(shutil.copytree(SERVO14, tmp_path / "edited"))
    name, change = EDITS[edit]
    data = document_of(folder) if name == DOCUMENT_FILE else lock_of(folder)
    change(data)
    (folder / name).write_text(yaml.safe_dump(data, sort_keys=False) if name == DOCUMENT_FILE else
                               json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return folder


@pytest.mark.parametrize("edit", EDITS)
def test_a_form_fourteen_folder_holding_what_form_fifteen_added_is_refused(tmp_path: Path, edit: str):
    assert codes_of(SERVO14) == [], "the control: the folder as its commit wrote it"
    with pytest.raises(LoadRefused) as refused:
        read_folder(edited(tmp_path, edit))
    code, held = HELD_AT[edit]
    assert refused.value.codes()[0] == code
    assert f"{held}, which version {PORTS_ADDED} of the saved form added" in str(refused.value)


@pytest.mark.parametrize("edit", EDITS)
def test_control_without_the_steps_check_an_edited_form_fourteen_folder_is_read(tmp_path: Path, edit: str):
    """The refusal above is the step's check, and nothing else's: with it removed, the edited folder is read."""
    folder = edited(tmp_path, edit)
    with pytest.MonkeyPatch.context() as patch:
        for module in (_folder, _versions):
            patch.setattr(module, "step_15_held", lambda data, *, lock: [])
        assert read_folder(folder).version == FORM14_READS


def test_what_version_fifteen_added_is_found_where_it_is_held():
    assert _versions.step_15_held({"controller": {"name": "c", "believed": {}}}, lock=False) == ["controller.believed"]
    assert _versions.step_15_held({"things": {"a": {"kind": "apparatus", "ports": {}},
                                              "b": {"kind": "body", "ports": {}}}}, lock=False) == ["things.a.ports"]
    assert _versions.step_15_held({"resolved": {"ports": {}}}, lock=True) == ["resolved.ports"]


def ported(name: str = "ported") -> Experiment:
    """The coil sphere with every rung of both stages: a tripped, slewed and late command port, a measurement port on
    its pose and a noisy, quantised one on its swim speed."""
    from port_cases import coil_sphere

    from mime.observations import SwimSpeed
    from mime.scene import X
    from mime.things import CommandPort, Gaussian, Measurement
    from mime.units import A, m_per_s, ms, s

    ports = {"currents": CommandPort(rate=ms(2), latency=ms(3), bounds=A(3.0), on_bound="trip", rest=A(0.5),
                                     slew=A(2000.0) / s(1.0), reason="a supply's datasheet"),
             "tracker": Measurement(of="sphere.pose", rate=ms(2), latency=ms(1), reason="a tracker's datasheet"),
             "speed": Measurement(of="swim_speed:sphere.speed", latency=ms(1), noise=Gaussian(sigma=m_per_s(1e-6)),
                                  quantum=m_per_s(1e-7), reason="a tracker's datasheet")}
    experiment = coil_sphere(ports, name=name)
    experiment.observe(SwimSpeed("sphere", along=X))
    return experiment


@pytest.fixture(scope="module")
def ported_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """:func:`ported`, saved in this form."""
    root = tmp_path_factory.mktemp("form15")
    ported().build().save(root / "ported")
    return root / "ported"


def test_version_fifteen_writes_ports_and_a_measurement_read_and_reads_them_back(ported_folder: Path, tmp_path: Path):
    assert PORTS_ADDED == CURRENT == 15
    assert DOCUMENT_SCHEMAS[CURRENT] == "mime.document/15" and LOCK_SCHEMAS[CURRENT] == "mime.lock/15"
    document, lock = document_of(ported_folder), lock_of(ported_folder)
    ports = document["things"]["coils"]["ports"]
    assert set(ports) == {"currents", "speed", "tracker"}
    assert ports["speed"]["noise"] == {"sigma": {"value": 1e-06, "unit": "m/s"}}
    assert ports["currents"]["on_bound"] == "trip" and ports["currents"]["rest"] == {"value": 0.5, "unit": "A"}
    assert ports["tracker"] == {"port": "measurement", "of": "sphere.pose", "rate": {"value": 0.002, "unit": "s"},
                                "latency": {"value": 0.001, "unit": "s"}, "reason": "a tracker's datasheet"}
    assert lock["resolved"]["ports"]["coils.currents"]["staged"] is True
    built = lock["hashes"]["derived"]
    loaded = Experiment.load(Path(shutil.copytree(ported_folder, tmp_path / "ported")))
    assert loaded.load_report is not None and "input hash: reproduced exactly" in loaded.load_report.text()
    again = loaded.build()
    assert again.report.derived_hash == built and again.verify().reproduced


def test_version_fourteens_reader_refuses_a_folder_of_version_fifteen_as_newer_each_file_once(ported_folder: Path):
    assert codes_of(ported_folder) == [], "this MIME reads it"
    with reader_of(FORM14_READS):
        assert codes_of(ported_folder) == [NEWER, NEWER]


def test_control_the_form_left_unstepped_is_read_by_version_fourteens_reader_with_no_newer_form(
        ported_folder: Path, tmp_path: Path):
    unstepped = restated(ported_folder, tmp_path / "unstepped", FORM14_READS)
    with reader_of(FORM14_READS):
        assert NEWER not in codes_of(unstepped)


def fed_back() -> BuiltExperiment:
    """The coil sphere's loop closed through its ports by a controller reading a measurement port and its age."""
    from port_cases import fed_back_sphere

    from mime.things import CommandPort, Measurement
    from mime.units import ms

    return fed_back_sphere({"currents": CommandPort(latency=ms(1), reason="a supply's datasheet"),
                            "tracker": Measurement(of="sphere.pose", latency=ms(1), reason="a tracker's datasheet")},
                           mode="graph", name="fed_back_15").build()


def test_a_controllers_read_of_a_measurement_port_is_recorded_as_a_measurement():
    reads = fed_back().record.resolved["controller"]["reads"]
    assert {label: reads[label]["read_class"] for label in ("pose", "age")} == {"pose": "measurement",
                                                                                 "age": "measurement"}


def test_a_believed_model_is_refused_by_name_until_a_mime_reads_it(tmp_path: Path):
    """``controller.believed`` is a slot of the form that nothing writes yet: a document holding it is refused, never
    built as though its controller had no model of what it drives; the document without it builds."""
    from mime.saved import dump_document

    document = fed_back().record.document
    assert document is not None
    path = tmp_path / DOCUMENT_FILE
    path.write_text(dump_document(document), encoding="utf-8")
    Experiment.from_document(path).build()
    path.write_text(dump_document({**document, "controller": {**document["controller"], "believed": {}}}),
                    encoding="utf-8")
    with pytest.raises(LoadRefused) as refused:
        Experiment.from_document(path)
    assert refused.value.codes() == ["load.document-shape"] and "controller.believed" in str(refused.value)
