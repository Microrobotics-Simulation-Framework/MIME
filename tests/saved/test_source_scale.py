"""``mime verify`` where it claims no bits, from the tenth saved form: each field's largest difference over the reference
steps judged against its source scale (``mime.saved._scales``), the tolerance per precision; and the two levels of
claim, bit for bit in the environment the lock was written in except for a reference made on a GPU with XLA's
deterministic mode off, which is held to the tolerance everywhere and says so.

**The data.** ``artefacts/tube_demo_counts.npz`` holds the tube demo's reference outputs (``examples/sphere_settling_tube``,
its history arrays: the MRS drag with the tube's wall, the hull contact, gravity, the swim speed) saved at 24, 4 and 2
usable CPUs in float32 and float64 on one laptop (``artefacts/tube_demo_counts.json`` says how). Two counts are two
environments with other last bits (MIME-ANO-140), which is what a rerun elsewhere meets. The GPU spread is not
reproducible here, so it is planted: :data:`T4_SPREAD` is, per array, the largest difference over its own largest
magnitude over the eight pairs of runs measured on a T4 and its CPU (2026-11-07, jax 0.11.1, the same demo), planted on
the float32 reference at that size. The twist's is planted in its angular part: on the CPU its difference across counts
is angular but for the speed's rounding (``test_across_cpu_counts_the_twists_difference_is_angular_but_for_rounding``),
and that the T4's is too is an inference, which the linear-part control bounds: planted there, the same size is a
change of the ball's speed and is reported.

**The sliding ball** (:func:`sliding_ball`) is the light case with a force: a 10 um steel sphere on the floor of a 60 um
tube of water under gravity, held by the hull contact, sliding along the tube at 1 mm/s and turning at 20 rad/s. One
of its contact's eight rows takes the weight at every step; the others take none and hold wherever their search
ended (MIME-ANO-144).

**The controls** each fail on every run: they are planted at every step (and, for the twist, at every angular
component), never drawn."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mime.nodes.body import GravityNode, RigidInertialTensorNode, SwimSpeedNode
from mime.nodes.hydro.mrs import MRSDragNode
from mime.registry import Hull
from mime.saved import LOCK_FILE, read_folder, verify_history
from mime.saved._folder import DETERMINISTIC_OFF
from mime.saved._scales import FLOAT32_RTOL, FLOAT64_RTOL, OWN, SourceScale, source_scales
from mime.scene import Sphere
from mime.units import um
from _platform_support import needs_readable_cpu_flags

ARTEFACTS = Path(__file__).parent / "artefacts"
TUBE = ARTEFACTS / "tube_demo_counts.npz"
TUBE_SHA256 = "474f4670acfe96febfad9a7bcfbc920d38eb7fc816d5709b6a88d1d802354b93"
#: The tube demo's nodes, as its lock's derived graph names their types, and the swim speed's observation node.
TUBE_NODES = {"ball": RigidInertialTensorNode, "drag:ball": MRSDragNode, "gravity:ball": GravityNode,
              "swim_speed:ball": SwimSpeedNode}
#: The ball's hull, as the tube demo's parts give it: a 10 um sphere centred on the body's origin.
BALL = Hull("ball", "steel", Sphere(radius=um(10.0)), {"radius": 1e-5}, (0.0, 0.0, 0.0),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
#: The T4's spread per array (the largest over its eight pairs of runs of each array's largest difference over its own
#: largest magnitude), the Colab verify-spread run of 2026-11-07; every other array was identical in every pair.
T4_SPREAD = {"ball/contact_points": 0.319, "ball/pose": 5.33e-08, "ball/twist": 0.0137,
             "drag:ball/asymmetry": 0.0127, "drag:ball/least_eigenvalue": 1.07e-06,
             "drag:ball/resistance": 5.92e-07, "drag:ball/wall_gap": 1.46e-07,
             "drag:ball/wall_gap_spacings": 9.54e-08, "swim_speed:ball/distance": 3.32e-07,
             "swim_speed:ball/mean_speed": 3.39e-07, "swim_speed:ball/speed": 4.46e-07}
PRECISIONS = {"float32": FLOAT32_RTOL, "float64": FLOAT64_RTOL}
STEPS = 8


def test_the_tube_demos_arrays_are_the_ones_measured():
    assert hashlib.sha256(TUBE.read_bytes()).hexdigest() == TUBE_SHA256
    record = json.loads((ARTEFACTS / "tube_demo_counts.json").read_text(encoding="utf-8"))
    assert record["nodes"] == {"ball": "RigidInertialTensorNode", "drag:ball": "MRSDragNode",
                               "gravity:ball": "GravityNode"}
    assert sorted(record["arms"]) == [f"{p}/{c}" for p in PRECISIONS for c in (2, 24, 4)]


def tube(precision: str, cpus: int) -> dict[str, np.ndarray]:
    prefix = f"{precision}/{cpus}/"
    with np.load(TUBE) as held:
        return {key.removeprefix(prefix): np.asarray(held[key]) for key in held.files if key.startswith(prefix)}


def history_of(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    held: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        node, name = key.removeprefix("history/").split("/")
        held.setdefault(node, {})[name] = value
    return held


def tube_scales(arrays: dict[str, np.ndarray]) -> dict[str, SourceScale]:
    nodes = {name: object.__new__(cls) for name, cls in TUBE_NODES.items()}
    fields = {node: tuple(names) for node, names in history_of(arrays).items()}
    return source_scales(nodes, fields, {"ball": BALL})


def form_of(arrays: dict[str, np.ndarray], *, version: int = 10, device: str = "cpu", deterministic: bool = False,
            rtol: float | None = None, atol: float = 0.0, x64: bool = False) -> SimpleNamespace:
    """A saved form as ``verify_history`` reads one, its lock's environment and reference as written."""
    held = rtol if rtol is not None else (FLOAT64_RTOL if x64 else FLOAT32_RTOL)
    if version < 10 and rtol is None:
        held, atol = 1e-6, 1e-9
    lock = {"reference": {"steps": STEPS, "rtol": held, "atol": atol},
            "environment": {"device": device, "deterministic": deterministic, "x64": x64}}
    return SimpleNamespace(lock=lock, reference=arrays, version=version)


@pytest.fixture
def same_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: [])


@pytest.fixture
def other_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: ["usable_cpus: built under 24"])


def moved_labels(report) -> set[str]:
    return {line.split(": ")[0] for line in report.moved}


def test_the_tube_demos_fields_take_their_source_scales_from_their_types():
    scales = tube_scales(tube("float32", 24))
    assert scales["ball.twist"].kind.value == "twist" and scales["ball.twist"].size == pytest.approx(1e-5, rel=1e-6)
    assert scales["drag:ball.asymmetry"].kind.value == "ratio"
    assert scales["ball.contact_points"] == SourceScale(scales["ball.contact_points"].kind,
                                                        forces="ball.contact_forces")
    assert scales["ball.contact_points"].kind.value == "held points"
    assert {label for label, scale in scales.items() if scale != OWN} == {"ball.twist", "drag:ball.asymmetry",
                                                                           "ball.contact_points"}


def test_across_cpu_counts_the_twists_difference_is_angular_but_for_rounding():
    """In float32 its angular part moves by 2e-6 rad/s, 0.5 % of its largest turning, and its linear part by 2.3e-10
    m/s, 2.3e-7 of the ball's speed: a few units of the resolution."""
    for precision, resolution in (("float32", np.finfo(np.float32).eps), ("float64", np.finfo(np.float64).eps)):
        a, b = tube(precision, 24)["history/ball/twist"], tube(precision, 2)["history/ball/twist"]
        linear = np.max(np.abs(a[:, :3].astype(np.float64) - b[:, :3]))
        angular = np.max(np.abs(a[:, 3:].astype(np.float64) - b[:, 3:]))
        speed, turning = np.max(np.linalg.norm(a[:, :3], axis=1)), np.max(np.linalg.norm(a[:, 3:], axis=1))
        assert linear < 4 * resolution * speed and angular > 1e3 * linear, (precision, linear, angular)
        assert angular / turning > 1e4 * resolution, (precision, angular, turning)


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize(("saved", "found"), [(24, 2), (24, 4), (4, 2)], ids=["24_2", "24_4", "4_2"])
def test_across_cpu_counts_every_field_is_within_its_precisions_tolerance_on_its_source_scale(
        other_environment, precision: str, saved: int, found: int):
    reference, rerun = tube(precision, saved), tube(precision, found)
    report = verify_history(form_of(reference, x64=precision == "float64"), history_of(rerun), tube_scales(rerun))
    print(f"\n{report.text()}")
    assert report.reproduced and report.scaled and not report.bit_for_bit and report.warning is None
    assert report.tolerance == f"within rtol {PRECISIONS[precision]} of each field's source scale"


def test_today_across_cpu_counts_the_float32_twist_is_moved_element_by_element(other_environment):
    """The control of the arm above, and #250's false "moved": the same pair under a form-9 lock."""
    reference, rerun = tube("float32", 24), tube("float32", 2)
    report = verify_history(form_of(reference, version=9), history_of(rerun), tube_scales(rerun))
    print(f"\n{report.text()}")
    assert not report.reproduced and not report.scaled and "ball.twist" in moved_labels(report)
    assert report.tolerance == "within rtol 1e-06, atol 1e-09"


def planted_spread(reference: dict[str, np.ndarray], *, twist_part: slice = slice(3, 6)) -> dict[str, np.ndarray]:
    """Each array of :data:`T4_SPREAD` moved by its spread times its own largest magnitude, every element, in turn up
    and down; the twist's in ``twist_part`` alone (its angular part by default)."""
    found = dict(reference)
    for name, spread in T4_SPREAD.items():
        key = f"history/{name}"
        held = reference[key].astype(np.float64)
        step = spread * float(np.max(np.abs(held)))
        signs = np.where(np.arange(held.size).reshape(held.shape) % 2 == 0, 1.0, -1.0)
        moved = held + step * signs
        if name == "ball/twist":
            moved = held.copy()
            moved[:, twist_part] += step * signs[:, twist_part]
        found[key] = moved.astype(reference[key].dtype)
    return found


def test_no_false_moved_the_t4_spread_planted_per_field_verifies_within_tolerance_with_the_warning(same_environment):
    reference = tube("float32", 24)
    found = planted_spread(reference)
    report = verify_history(form_of(reference, device="gpu"), history_of(found), tube_scales(found))
    print(f"\n{report.text()}")
    assert report.reproduced and report.same_environment and not report.bit_for_bit
    assert report.text().splitlines() == [
        f"verify: reproduced {STEPS} steps within rtol 1e-05 of each field's source scale on {report.environment}",
        DETERMINISTIC_OFF]


@pytest.mark.parametrize("environment", ["same", "other"])
def test_today_the_same_plant_is_moved(environment: str, monkeypatch: pytest.MonkeyPatch):
    """Under a form-9 lock (today's rule: bit for bit in the environment the lock was written in, element by element
    in another), the same plant is reported moved."""
    differences = [] if environment == "same" else ["device: built under 'gpu'"]
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: differences)
    reference = tube("float32", 24)
    found = planted_spread(reference)
    for bits in (True, False):
        report = verify_history(form_of(reference, version=9, device="gpu", deterministic=bits), history_of(found),
                                tube_scales(found))
        print(f"\n{report.text()}")
        assert {"ball.twist", "drag:ball.asymmetry", "ball.contact_points"} <= moved_labels(report)


def test_control_the_t4_twist_spread_in_its_linear_part_is_moved(same_environment):
    reference = tube("float32", 24)
    found = planted_spread(reference, twist_part=slice(0, 3))
    report = verify_history(form_of(reference, device="gpu"), history_of(found), tube_scales(found))
    print(f"\n{report.text()}")
    assert moved_labels(report) == {"ball.twist"}


def test_a_deterministic_folder_verified_in_the_same_environment_is_bit_for_bit(same_environment):
    reference = tube("float32", 24)
    report = verify_history(form_of(reference, device="gpu", deterministic=True), history_of(reference),
                            tube_scales(reference))
    assert report.reproduced and report.bit_for_bit and report.warning is None
    assert report.text() == f"verify: reproduced {STEPS} steps bit for bit on {report.environment}"
    found = dict(reference)
    twist = reference["history/ball/twist"].copy()
    twist[3, 4] = np.nextafter(twist[3, 4], np.float32(np.inf))
    found["history/ball/twist"] = twist
    moved = verify_history(form_of(reference, device="gpu", deterministic=True), history_of(found), tube_scales(found))
    assert moved_labels(moved) == {"ball.twist"} and "first at step 3" in moved.moved[0]


def twist_plant(reference: np.ndarray, tolerance: float, step: int, component: int) -> np.ndarray:
    """``w`` moved at ``step`` along ``component`` so that ``L |dw|`` is ten times the tolerance times the surface
    points' speed, the reference's largest ``|v| + L |w|``."""
    held = reference.astype(np.float64)
    speed = float(np.max(np.linalg.norm(held[:, :3], axis=1) + 1e-5 * np.linalg.norm(held[:, 3:], axis=1)))
    moved = held.copy()
    moved[step, 3 + component] += 10.0 * tolerance * speed / 1e-5
    return moved.astype(reference.dtype)


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_a_twists_angular_part_changed_at_ten_times_the_tolerance_is_moved_at_every_step_and_component(
        other_environment, precision: str):
    """At float32's tolerance the plant is 0.01 rad/s on a ball whose largest turning is 3.7e-4 rad/s: its surface
    points' speed moves by 1e-4 of the ball's, which the surface-point scale does not hide."""
    reference = tube(precision, 24)
    tolerance = PRECISIONS[precision]
    for step in range(STEPS):
        for component in range(3):
            found = dict(reference)
            found["history/ball/twist"] = twist_plant(reference["history/ball/twist"], tolerance, step, component)
            report = verify_history(form_of(reference, x64=precision == "float64"), history_of(found),
                                    tube_scales(found))
            assert moved_labels(report) == {"ball.twist"}, (step, component, report.text())
            assert f"first at step {step}" in report.moved[0]


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_an_asymmetry_of_1e_4_of_the_resistances_scale_is_moved_at_every_step(other_environment, precision: str):
    reference = tube(precision, 24)
    for step in range(STEPS):
        found = dict(reference)
        held = reference["history/drag:ball/asymmetry"].copy()
        held[step] += 1e-4
        found["history/drag:ball/asymmetry"] = held
        report = verify_history(form_of(reference, x64=precision == "float64"), history_of(found), tube_scales(found))
        assert moved_labels(report) == {"drag:ball.asymmetry"}, (step, report.text())
        assert "of its source scale, one, the quantity it is a share of" in report.moved[0]


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize("name", ["drag:ball/least_eigenvalue", "ball/pose", "swim_speed:ball/speed"])
def test_a_default_scale_field_changed_at_ten_times_the_tolerance_is_moved_at_every_step(other_environment,
                                                                                         precision: str, name: str):
    reference = tube(precision, 24)
    key = f"history/{name}"
    for step in range(STEPS):
        found = dict(reference)
        held = reference[key].astype(np.float64)
        moved = held.copy()
        flat = moved[step:step + 1].reshape(-1)
        assert np.shares_memory(flat, moved)
        flat[int(np.argmax(np.abs(flat)))] += 10.0 * PRECISIONS[precision] * float(np.max(np.abs(held)))
        found[key] = moved.astype(reference[key].dtype)
        report = verify_history(form_of(reference, x64=precision == "float64"), history_of(found), tube_scales(found))
        assert moved_labels(report) == {name.replace("/", ".", 1)}, (step, report.text())


def sliding_ball():
    """A 10 um steel sphere on the floor of a 60 um tube of water, held by the hull contact 1.5 um off the wall under
    its weight, sliding along the tube at 1 mm/s and turning at 20 rad/s about the tube's y axis."""
    from mime import Body, Experiment, Medium, State, Structure
    from mime.dynamics import RigidInertial
    from mime.interactions import Contact, Gravity
    from mime.scene import Scene, Tube, X, Z
    from mime.things import Part
    from mime.units import kg_per_m3, m_per_s2, mm_per_s, mPa_s, ms, rad_per_s

    exp = Experiment("sliding_ball", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(1.0))
    tube = exp.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=um(60.0), wall=um(10.0),
                                                             length=um(480.0))))
    water = exp.add_medium(Medium("water", fills=tube.lumen))
    exp.declare("water.density", kg_per_m3(998.0), reason="water at 20 degC")
    exp.declare("water.viscosity", mPa_s(1.0), reason="water at 20 degC")
    ball = exp.add_body(Body("ball", dynamics=RigidInertial(), initial=State(
        position=um((0.0, 0.0, -48.5)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=mm_per_s((1.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 20.0, 0.0))), parts={"steel": Part(Sphere(radius=um(10.0)))}))
    exp.declare("ball.parts.steel.density", kg_per_m3(7850.0), reason="steel")
    exp.attach(Gravity(), body=ball, medium=water)
    exp.attach(Contact.Projection(margin=um(1.5), spacing=um(1.5)), body=ball, surface=tube.inner_surface)
    return exp


@pytest.fixture(scope="module")
def sliding(tmp_path_factory: pytest.TempPathFactory):
    """The sliding ball built and saved: its folder, its build, its reference outputs as a rerun on this machine gives
    them back (a rerun in the environment the lock was written in is bit for bit), and their source scales."""
    built = sliding_ball().build()
    folder = tmp_path_factory.mktemp("sliding") / "ball"
    built.save(folder)
    form = read_folder(folder)
    history = history_of({key: value for key, value in form.reference.items() if key.startswith("history/")})
    return SimpleNamespace(folder=folder, built=built, form=form, history=history,
                           scales=built.source_scales(history))


def test_the_sliding_balls_twist_is_sized_by_its_hulls_reach_and_one_row_takes_its_weight(sliding):
    twist = sliding.scales["ball.twist"]
    assert twist.size == pytest.approx(1e-5, rel=1e-6) and "largest reach of its hull" in twist.size_from
    forces = sliding.history["ball"]["contact_forces"]
    acting = np.flatnonzero(np.all(forces != 0, axis=0))
    assert acting.size == 1 and np.all(np.delete(forces, acting, axis=1) == 0), forces


def with_points(history: dict, points: np.ndarray) -> dict:
    return {**history, "ball": {**history["ball"], "contact_points": points}}


def test_a_held_point_of_a_row_that_takes_a_force_moved_at_ten_times_the_tolerance_is_moved_at_every_step(
        sliding, other_environment):
    points = sliding.history["ball"]["contact_points"]
    row = int(np.flatnonzero(np.all(sliding.history["ball"]["contact_forces"] != 0, axis=0))[0])
    size = 10.0 * sliding.form.lock["reference"]["rtol"] * float(np.max(np.abs(points.astype(np.float64))))
    for step in range(STEPS):
        moved = points.astype(np.float64).copy()
        moved[step, row, 0] += size
        report = verify_history(sliding.form, with_points(sliding.history, moved.astype(points.dtype)), sliding.scales)
        assert moved_labels(report) == {"ball.contact_points"}, (step, report.text())
        assert f"first at step {step}" in report.moved[0] and "on the rows whose force acts" in report.moved[0]


def test_a_held_point_of_a_row_that_takes_no_force_jumping_two_millimetres_is_not_reported(sliding, other_environment):
    """And the control: the same jump judged against the field's own largest magnitude, with no pairing, is moved."""
    points = sliding.history["ball"]["contact_points"]
    far = int(np.flatnonzero(np.all(sliding.history["ball"]["contact_forces"] == 0, axis=0))[0])
    for step in range(STEPS):
        moved = points.astype(np.float64).copy()
        moved[step, far] += 2e-3
        found = with_points(sliding.history, moved.astype(points.dtype))
        assert verify_history(sliding.form, found, sliding.scales).reproduced, step
        unpaired = {**sliding.scales, "ball.contact_points": OWN}
        assert moved_labels(verify_history(sliding.form, found, unpaired)) == {"ball.contact_points"}, step


@needs_readable_cpu_flags
def test_the_sliding_ball_verifies_bit_for_bit_where_its_lock_was_written(sliding):
    """The real rerun, in this environment: the CPU path as it was. The plants above take the reference outputs as the
    rerun on that ground."""
    from mime.experiment import Experiment

    report = Experiment.load(sliding.folder).build().verify()
    print(f"\n{report.text()}")
    assert report.same_environment and report.reproduced and report.bit_for_bit and report.warning is None


def test_a_gpu_reference_made_with_deterministic_mode_off_reruns_within_tolerance_with_the_warning(
        sliding, same_environment, tmp_path: Path):
    """A real rerun of the folder whose lock says a GPU made it with the mode off: held to the tolerance in the
    environment the lock was written in, and saying why."""
    import shutil

    from mime.experiment import Experiment

    folder = Path(shutil.copytree(sliding.folder, tmp_path / "gpu"))
    lock = json.loads((folder / LOCK_FILE).read_text(encoding="utf-8"))
    lock["environment"].update(device="gpu", deterministic=False)
    (folder / LOCK_FILE).write_text(json.dumps(lock), encoding="utf-8")
    report = Experiment.load(folder).build().verify()
    print(f"\n{report.text()}")
    assert report.reproduced and report.same_environment and not report.bit_for_bit
    assert report.text().splitlines()[-1] == DETERMINISTIC_OFF
    assert f"within rtol {lock['reference']['rtol']} of each field's source scale" in report.text()
