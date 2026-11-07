"""``mime verify`` where it claims no bits, from the tenth saved form (design #266): every saved field split into blocks
by its declared units, each block judged at every reference step against ``max(rtol * its largest magnitude, k * its
measured resolution)``, the resolution being each block's divergence over the reference steps run again at save from a
start one ulp off (``mime.saved._scales``); held points only on the rows whose force acts. And the two levels of claim:
bit for bit in the environment the lock was written in, except for a reference made on a GPU with XLA's deterministic
mode off, which is held to the tolerance everywhere and says so.

**The data.** ``artefacts/tube_demo_counts.npz`` holds the tube demo's reference outputs (``examples/sphere_settling_tube``:
the MRS drag with the tube's wall, the hull contact, gravity, the swim speed) saved at 24, 4 and 2 usable CPUs in float32
and float64 on one laptop (``artefacts/tube_demo_counts.json`` says how). Two counts are two environments with other last
bits (MIME-ANO-140), which is what a rerun elsewhere meets. ``artefacts/tube_demo_resolution.json`` holds the resolution
the demo's save measured, by field and block, on the same laptop at 24 CPUs, and the units its build declares
(:data:`TUBE_UNITS`; ``tests/examples/test_sphere_settling_tube.py`` holds the demo's own save to them). The GPU
spread is not reproducible here, so it is planted: :data:`T4_SPREAD` is, per array, the largest difference over its own
largest magnitude over the eight pairs of runs measured on a T4 and its CPU (2026-11-07, jax 0.11.1, the same demo), and
:func:`t4_per_block` spreads it over a field's blocks as the CPU's spread across counts is spread (the twist's in its
angular velocity, the pose's in its quaternion): an inference, which the per-array number bounds from above and the
control planting the twist's in its velocity tests.

**The review's majors**, each planted at every step and caught on every run: M1, the ball's position moved 8e-6 m in
float32; M2, its angular velocity zeroed, flipped and doubled; M3, the resistance's rotational block times 100.

**The sliding ball** (:func:`sliding_ball`) is the light case with a force, built and saved here: a 10 um steel sphere
on the floor of a 60 um tube of water under gravity, held by the hull contact, sliding along the tube at 1 mm/s and
turning at 20 rad/s. One of its contact's eight rows takes the weight at every step; the others take none and hold
wherever their search ended (MIME-ANO-144)."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mime.saved import LOCK_FILE, read_folder, verify_history
from mime.saved._folder import DETERMINISTIC_OFF
from mime.saved._scales import FLOAT32_RTOL, FLOAT64_RTOL, OWN, PART, RESOLUTION_FACTOR, UNDECLARED, FieldUnits, \
    blocks, judge, unit_parts
from mime.scene import Sphere
from mime.units import um
from _platform_support import needs_readable_cpu_flags

ARTEFACTS = Path(__file__).parent / "artefacts"
TUBE = ARTEFACTS / "tube_demo_counts.npz"
TUBE_SHA256 = "474f4670acfe96febfad9a7bcfbc920d38eb7fc816d5709b6a88d1d802354b93"
RESOLUTION = ARTEFACTS / "tube_demo_resolution.json"
RESOLUTION_SHA256 = "0a2087710ede834b695ca02a5caacfb3baf9348ee6e9d9f70de726686e273500"
#: The units the tube demo's build declares for each field it declares them for, as the artefact records them (its
#: nodes' symbol tables, the edges that read them, the stage's binding, the observations' outputs, the flags); every
#: other field is one block. ``tests/examples/test_sphere_settling_tube.py`` holds the demo's own save to them.
TUBE_UNITS: dict[str, str] = json.loads(RESOLUTION.read_text(encoding="utf-8"))["units"]
#: The T4's spread per array (the largest over its eight pairs of runs of each array's largest difference over its own
#: largest magnitude), the Colab verify-spread run of 2026-11-07; every other array was identical in every pair. The held
#: points' is on rows that take no force: no row of the tube demo's reference takes one.
T4_SPREAD = {"ball/contact_points": 0.319, "ball/pose": 5.33e-08, "ball/twist": 0.0137,
             "drag:ball/asymmetry": 0.0127, "drag:ball/least_eigenvalue": 1.07e-06,
             "drag:ball/resistance": 5.92e-07, "drag:ball/wall_gap": 1.46e-07,
             "drag:ball/wall_gap_spacings": 9.54e-08, "swim_speed:ball/distance": 3.32e-07,
             "swim_speed:ball/mean_speed": 3.39e-07, "swim_speed:ball/speed": 4.46e-07}
PRECISIONS = {"float32": FLOAT32_RTOL, "float64": FLOAT64_RTOL}
STEPS = 8


def test_the_tube_demos_arrays_and_resolution_are_the_ones_measured():
    assert hashlib.sha256(TUBE.read_bytes()).hexdigest() == TUBE_SHA256
    assert hashlib.sha256(RESOLUTION.read_bytes()).hexdigest() == RESOLUTION_SHA256
    record = json.loads((ARTEFACTS / "tube_demo_counts.json").read_text(encoding="utf-8"))
    assert record["nodes"] == {"ball": "RigidInertialTensorNode", "drag:ball": "MRSDragNode",
                               "gravity:ball": "GravityNode"}
    assert sorted(record["arms"]) == [f"{p}/{c}" for p in PRECISIONS for c in (2, 24, 4)]
    assert sorted(json.loads(RESOLUTION.read_text(encoding="utf-8"))["resolution"]) == sorted(PRECISIONS)


def tube(precision: str, cpus: int) -> dict[str, np.ndarray]:
    prefix = f"{precision}/{cpus}/"
    with np.load(TUBE) as held:
        return {key.removeprefix(prefix): np.asarray(held[key]) for key in held.files if key.startswith(prefix)}


def tube_resolution(precision: str) -> dict[str, dict[str, float | None]]:
    return json.loads(RESOLUTION.read_text(encoding="utf-8"))["resolution"][precision]


def history_of(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    held: dict[str, dict[str, np.ndarray]] = {}
    for key, value in arrays.items():
        node, name = key.removeprefix("history/").split("/")
        held.setdefault(node, {})[name] = value
    return held


def label_of(name: str) -> str:
    return name.replace("/", ".", 1)


def tube_fields(arrays: dict[str, np.ndarray]) -> dict[str, FieldUnits]:
    labels = [label_of(key.removeprefix("history/")) for key in arrays if key.startswith("history/")]
    return {label: FieldUnits(TUBE_UNITS.get(label), "ball.contact_forces" if label == "ball.contact_points" else None)
            for label in labels}


def form_of(arrays: dict[str, np.ndarray], *, version: int = 10, device: str = "cpu", deterministic: bool = False,
            x64: bool = False, resolution: dict | None = None) -> SimpleNamespace:
    """A saved form as ``verify_history`` reads one, its lock's environment and reference as written: from the tenth
    form its precision's tolerance with the tube demo's measured resolution (or ``resolution``), before it 1e-6 and
    1e-9."""
    reference: dict = {"steps": STEPS, "rtol": FLOAT64_RTOL if x64 else FLOAT32_RTOL, "atol": 0.0}
    if version < 10:
        reference.update(rtol=1e-6, atol=1e-9)
    else:
        reference.update(resolution=tube_resolution("float64" if x64 else "float32") if resolution is None
                         else resolution, resolution_factor=RESOLUTION_FACTOR)
    lock = {"reference": reference, "environment": {"device": device, "deterministic": deterministic, "x64": x64}}
    return SimpleNamespace(lock=lock, reference=arrays, version=version)


def verified(reference: dict[str, np.ndarray], found: dict[str, np.ndarray], **form):
    return verify_history(form_of(reference, **form), history_of(found), tube_fields(found))


@pytest.fixture
def same_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: [])


@pytest.fixture
def other_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: ["usable_cpus: built under 24"])


def moved_labels(report) -> set[str]:
    return {line.split(": ")[0] for line in report.moved}


def block_mask(units: str | None, block: str, shape: tuple[int, ...]) -> np.ndarray:
    """``block``'s elements over every step of an array of ``shape``."""
    return np.broadcast_to(dict(blocks(units, shape[1:]))[block][None], shape)


def bound_of(reference: np.ndarray, label: str, block: str, precision: str) -> float:
    """A block's bound as the rule states it, computed here from the arrays: ``max(rtol * its largest magnitude,
    k * its resolution)``."""
    mask = block_mask(TUBE_UNITS.get(label), block, reference.shape)
    magnitude = float(np.max(np.where(mask, np.abs(reference.astype(np.float64)), 0.0)))
    floor = tube_resolution(precision).get(label, {}).get(block)
    return max(PRECISIONS[precision] * magnitude, RESOLUTION_FACTOR * floor if floor else 0.0)


# --- blocks by unit -------------------------------------------------------------------------------------------------


def names_and_sizes(units: str | None, shape: tuple[int, ...]) -> dict[str, int]:
    return {name: int(mask.sum()) for name, mask in blocks(units, shape)}


def test_a_pose_twist_and_wrench_split_into_their_first_three_and_the_rest():
    assert names_and_sizes("m; 1", (7,)) == {"m": 3, "1": 4}
    assert names_and_sizes("m/s; rad/s", (6,)) == {"m/s": 3, "rad/s": 3}
    assert names_and_sizes("N; N m", (6,)) == {"N": 3, "N m": 3}
    poses = dict(blocks("m; 1", (4, 7)))
    assert poses["m"][:, :PART].all() and not poses["m"][:, PART:].any(), "each row of frame poses alike"


def test_a_resistance_splits_into_its_translational_coupling_and_rotational_blocks():
    held = dict(blocks("N s/m; N s; N m s", (6, 6)))
    assert {name: int(mask.sum()) for name, mask in held.items()} == {"N s/m": 9, "N s": 18, "N m s": 9}
    assert held["N s/m"][:3, :3].all() and held["N m s"][3:, 3:].all()
    assert held["N s"][:3, 3:].all() and held["N s"][3:, :3].all(), "both coupling blocks share their unit"


def test_one_unit_is_one_block_and_so_is_a_field_whose_units_are_not_declared_or_do_not_fit():
    assert names_and_sizes("T/m", (3, 3)) == {"T/m": 9}
    assert names_and_sizes(None, (8, 3)) == {UNDECLARED: 24}
    assert names_and_sizes("m; 1", (3,)) == {"m; 1": 3}, "a composite that does not fit its shape is whole"
    assert names_and_sizes("m; m", (6,)) == {"m": 6}, "two parts of one unit are one block"


def test_every_composite_interface_of_the_vocabulary_splits_into_one_block_per_unit():
    """The vocabulary's composites are the units MIME's builds declare: each splits on its shape (a count of rows, -1,
    taken as two) into as many blocks as it names distinct units."""
    from mime.vocabulary import lookup, names

    held_names = [lookup(name) for name in names()]
    composites = [held for held in held_names if held is not None and ";" in held.units and held.shape is not None]
    assert {held.name for held in composites} >= {"body.pose", "body.twist", "body.resistance", "body.applied_wrench"}
    for held in composites:
        shape = tuple(2 if size == -1 else size for size in held.shape)
        assert len(blocks(held.units, shape)) == len(set(unit_parts(held.units))), held.name


# --- across CPU counts ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize(("saved", "found"), [(24, 2), (24, 4), (4, 2)], ids=["24_2", "24_4", "4_2"])
def test_across_cpu_counts_every_block_is_within_its_bound(other_environment, precision: str, saved: int, found: int):
    report = verified(tube(precision, saved), tube(precision, found), x64=precision == "float64")
    print(f"\n{report.text()}")
    assert report.reproduced and report.scaled and not report.bit_for_bit and report.warning is None
    assert report.tolerance == (f"within rtol {PRECISIONS[precision]} of each block's largest magnitude, or "
                                f"{RESOLUTION_FACTOR:g} times its measured resolution")


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_control_without_its_resolution_the_angular_velocity_across_counts_is_moved(other_environment, precision: str):
    """The floor is what holds the angular velocity across counts: the same pair with no resolution recorded reports
    the twist moved, in its block in rad/s."""
    report = verified(tube(precision, 24), tube(precision, 2), x64=precision == "float64", resolution={})
    assert any(line.startswith("ball.twist: its block in rad/s") for line in report.moved), report.text()


def test_today_across_cpu_counts_the_float32_twist_is_moved_element_by_element(other_environment):
    """#250's false "moved": the same pair under a form-9 lock."""
    report = verified(tube("float32", 24), tube("float32", 2), version=9)
    assert not report.reproduced and not report.scaled and "ball.twist" in moved_labels(report)
    assert report.tolerance == "within rtol 1e-06, atol 1e-09"


# --- the review's majors, each caught at every step ------------------------------------------------------------------


def planted(reference: dict[str, np.ndarray], name: str, change) -> dict[str, np.ndarray]:
    """``reference`` with ``change`` made to a float64 copy of ``name``'s array, written back in its dtype."""
    found = dict(reference)
    held = reference[f"history/{name}"].astype(np.float64)
    change(held)
    found[f"history/{name}"] = held.astype(reference[f"history/{name}"].dtype)
    return found


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_m1_the_balls_position_moved_8e_6_m_is_moved_at_every_step(other_environment, precision: str):
    """80 % of the ball's radius, half its whole run: in float32 the quaternion's 1.0 hid it, the pose one block."""
    reference = tube(precision, 24)
    for step in [*range(STEPS), None]:
        def shift(held, step=step):
            held[slice(None) if step is None else step, 2] += 8e-6

        report = verified(reference, planted(reference, "ball/pose", shift), x64=precision == "float64")
        assert moved_labels(report) == {"ball.pose"}, (step, report.text())
        assert report.moved[0].startswith("ball.pose: its block in m:"), report.moved[0]
        assert report.moved[0].endswith(f"first at step {step or 0}"), report.moved[0]


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize("change", ["zeroed", "flipped", "doubled"])
def test_m2_the_balls_angular_velocity_zeroed_flipped_or_doubled_is_moved(other_environment, precision: str,
                                                                           change: str):
    """The tube ball turns at 3.6e-4 rad/s at most: by the surface points' speed the whole of it was inside the
    tolerance. Judged on its own block, on its resolution, each change is reported."""
    reference = tube(precision, 24)

    def turned(held):
        held[:, 3:] = {"zeroed": 0.0, "flipped": -held[:, 3:], "doubled": 2.0 * held[:, 3:]}[change]

    report = verified(reference, planted(reference, "ball/twist", turned), x64=precision == "float64")
    assert moved_labels(report) == {"ball.twist"}, report.text()
    assert report.moved[0].startswith("ball.twist: its block in rad/s:"), report.moved[0]


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize(("block", "where"), [("N m s", (slice(None), slice(3, 6), slice(3, 6))),
                                              ("N s", (slice(None), slice(0, 3), slice(3, 6)))],
                         ids=["rotational", "coupling"])
def test_m3_a_resistances_rotational_or_coupling_block_times_100_is_moved(other_environment, precision: str,
                                                                          block: str, where):
    reference = tube(precision, 24)

    def scaled(held):
        held[where] *= 100.0

    report = verified(reference, planted(reference, "drag:ball/resistance", scaled), x64=precision == "float64")
    assert moved_labels(report) == {"drag:ball.resistance"}, report.text()
    assert report.moved[0].startswith(f"drag:ball.resistance: its block in {block}:"), report.moved[0]


@pytest.mark.parametrize("precision", list(PRECISIONS))
@pytest.mark.parametrize("factor", [0.5, 2.0], ids=["halved", "doubled"])
def test_the_asymmetry_halved_or_doubled_is_moved(other_environment, precision: str, factor: float):
    """The review's minor m1: judged against one, the drag's asymmetry (1.95e-5 at most) could halve unseen."""
    reference = tube(precision, 24)

    def scaled(held):
        held *= factor

    report = verified(reference, planted(reference, "drag:ball/asymmetry", scaled), x64=precision == "float64")
    assert moved_labels(report) == {"drag:ball.asymmetry"}, report.text()


@pytest.mark.parametrize("precision", list(PRECISIONS))
def test_a_torque_where_the_reference_holds_none_is_moved(other_environment, precision: str):
    """Gravity's torque on the ball is zero and so is its resolution: no floor, so a torque of 1e-12 N m beside a force
    of 2.8e-10 N is reported, which one block judged against the force."""
    reference = tube(precision, 24)

    def torque(held):
        held[:, 3:] = 1e-12

    report = verified(reference, planted(reference, "gravity:ball/wrench", torque), x64=precision == "float64")
    assert moved_labels(report) == {"gravity:ball.wrench"}, report.text()
    assert report.moved[0].startswith("gravity:ball.wrench: its block in N m:"), report.moved[0]


# --- the review's edge plants: just beyond each bound, at every step, both signs, both precisions ------------------


#: Each plant: the field, its element at a step, and the block it is in; the review's twelve.
EDGES = [*(("ball/twist", (component,), "m/s" if component < 3 else "rad/s") for component in range(6)),
         ("drag:ball/asymmetry", (), "1"), ("ball/pose", (6,), "1"),
         ("drag:ball/least_eigenvalue", (), "1"), ("drag:ball/wall_gap", (), "m"),
         ("swim_speed:ball/speed", (), "m/s"), ("swim_speed:ball/distance", (), "m")]


@pytest.mark.parametrize("factor", [1.1, 1.5])
def test_a_plant_just_beyond_a_blocks_bound_is_caught_at_every_step_with_either_sign(other_environment, factor: float):
    """384 plants for each factor: 12 elements, 8 steps, 2 signs, 2 precisions, each ``factor`` times its block's bound
    as the rule states it (:func:`bound_of`), and none missed."""
    missed, count = [], 0
    for precision in PRECISIONS:
        reference = tube(precision, 24)
        for name, element, block in EDGES:
            bound = bound_of(reference[f"history/{name}"], label_of(name), block, precision)
            for step in range(STEPS):
                for sign in (1.0, -1.0):
                    def moved(held, step=step, sign=sign, element=element, bound=bound):
                        held[(step, *element)] += sign * factor * bound

                    count += 1
                    if verified(reference, planted(reference, name, moved), x64=precision == "float64").reproduced:
                        missed.append((precision, name, element, step, sign))
    assert count == 384 and missed == []


# --- no false moved: the T4's spread ---------------------------------------------------------------------------------


def cpu_spread_per_block(name: str) -> dict[str, float]:
    """``name``'s largest difference between any two CPU counts in float32, in each of its blocks."""
    arms = [tube("float32", count)[f"history/{name}"].astype(np.float64) for count in (24, 4, 2)]
    units = TUBE_UNITS.get(label_of(name))
    return {block: max(float(np.max(np.where(block_mask(units, block, a.shape), np.abs(a - b), 0.0)))
                       for a in arms for b in arms)
            for block, _ in blocks(units, arms[0].shape[1:])}


def t4_per_block(reference: dict[str, np.ndarray], name: str) -> dict[str, float]:
    """The T4's absolute spread of ``name`` over its blocks as the CPU's spread across counts is spread: the field's
    largest absolute difference on the T4, times each block's CPU spread over the field's largest CPU spread."""
    total = T4_SPREAD[name] * float(np.max(np.abs(reference[f"history/{name}"].astype(np.float64))))
    spread = cpu_spread_per_block(name)
    largest = max(spread.values())
    if largest == 0:
        return {block: total if len(spread) == 1 else 0.0 for block in spread}
    return {block: total * value / largest for block, value in spread.items()}


def planted_spread(reference: dict[str, np.ndarray], *, linear: bool = False) -> dict[str, np.ndarray]:
    """Each array of :data:`T4_SPREAD` moved by its T4 spread in each block (:func:`t4_per_block`), every element, in
    turn up and down; ``linear``, the twist's angular spread moved into its velocity instead. The held points move by
    their own on every row, none of which takes a force."""
    found = dict(reference)
    for name in T4_SPREAD:
        key = f"history/{name}"
        held = reference[key].astype(np.float64)
        signs = np.where(np.arange(held.size).reshape(held.shape) % 2 == 0, 1.0, -1.0)
        if name == "ball/contact_points":
            found[key] = (held + T4_SPREAD[name] * float(np.max(np.abs(held))) * signs).astype(reference[key].dtype)
            continue
        per = t4_per_block(reference, name)
        if linear and name == "ball/twist":
            per = {"m/s": per["rad/s"], "rad/s": 0.0}
        units = TUBE_UNITS.get(label_of(name))
        moved = held.copy()
        for block, value in per.items():
            moved += np.where(block_mask(units, block, held.shape), value * signs, 0.0)
        found[key] = moved.astype(reference[key].dtype)
    return found


def test_no_false_moved_the_t4_spread_planted_per_block_verifies_within_tolerance_with_the_warning(same_environment):
    reference = tube("float32", 24)
    report = verified(reference, planted_spread(reference), device="gpu")
    print(f"\n{report.text()}")
    assert report.reproduced and report.same_environment and not report.bit_for_bit
    assert report.text().splitlines() == [
        f"verify: reproduced {STEPS} steps within rtol 1e-05 of each block's largest magnitude, or "
        f"{RESOLUTION_FACTOR:g} times its measured resolution on {report.environment}", DETERMINISTIC_OFF]


#: How far inside its bound the T4's spread sits, at least, in each block its ``rtol`` term does not cover: the margin
#: ``tests/saved/test_tolerance_ratchet.py`` states for k.
INSIDE = 3.0


def test_no_false_moved_is_not_on_the_edge_the_t4_spread_sits_well_inside_each_floor():
    reference = tube("float32", 24)
    floored = 0
    for name in T4_SPREAD:
        if name == "ball/contact_points":
            continue
        for block, spread in t4_per_block(reference, name).items():
            held = reference[f"history/{name}"]
            magnitude = float(np.max(np.where(block_mask(TUBE_UNITS.get(label_of(name)), block, held.shape),
                                              np.abs(held.astype(np.float64)), 0.0)))
            if spread > FLOAT32_RTOL * magnitude:
                floored += 1
                assert INSIDE * spread <= bound_of(held, label_of(name), block, "float32"), (name, block)
    assert floored >= 2, "the angular velocity and the asymmetry rest on their floors"


@pytest.mark.parametrize("environment", ["same", "other"])
def test_today_the_same_plant_is_moved(environment: str, monkeypatch: pytest.MonkeyPatch):
    """Under a form-9 lock (bit for bit in the environment the lock was written in, element by element in another),
    the same plant is reported moved."""
    differences = [] if environment == "same" else ["device: built under 'gpu'"]
    monkeypatch.setattr("mime.saved._folder.environment_differences", lambda lock: differences)
    reference = tube("float32", 24)
    found = planted_spread(reference)
    for bits in (True, False):
        report = verified(reference, found, version=9, device="gpu", deterministic=bits)
        assert {"ball.twist", "drag:ball.asymmetry", "ball.contact_points"} <= moved_labels(report)


def test_control_the_t4_angular_spread_in_the_balls_velocity_is_moved(same_environment):
    reference = tube("float32", 24)
    report = verified(reference, planted_spread(reference, linear=True), device="gpu")
    assert moved_labels(report) == {"ball.twist"} and report.moved[0].startswith("ball.twist: its block in m/s:")


def test_a_deterministic_folder_verified_in_the_same_environment_is_bit_for_bit(same_environment):
    reference = tube("float32", 24)
    report = verified(reference, reference, device="gpu", deterministic=True)
    assert report.reproduced and report.bit_for_bit and report.warning is None
    assert report.text() == f"verify: reproduced {STEPS} steps bit for bit on {report.environment}"

    def nudged(held):
        held[3, 4] = float(np.nextafter(np.float32(held[3, 4]), np.float32(np.inf)))

    moved = verified(reference, planted(reference, "ball/twist", nudged), device="gpu", deterministic=True)
    assert moved_labels(moved) == {"ball.twist"} and "first at step 3" in moved.moved[0]


# --- a floor that is no floor ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("recorded", [0.0, None, math.inf, -1.0, True])
def test_a_resolution_that_is_zero_or_not_a_finite_positive_number_gives_no_floor(recorded):
    saved = np.zeros((2, 3))
    found = saved.copy()
    found[1, 0] = 1e-30
    judged = judge(saved, found, FieldUnits("m"), 1e-5, 0.0, resolutions={"m": recorded})
    assert not judged.within and judged.first == 1
    assert judge(saved, found, FieldUnits("m"), 1e-5, 0.0, resolutions={"m": 1e-31}).within, "the control: a floor"


# --- the sliding ball, built and saved ---------------------------------------------------------------------------------


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
    them back (a rerun in the environment the lock was written in is bit for bit), and their units."""
    built = sliding_ball().build()
    folder = tmp_path_factory.mktemp("sliding") / "ball"
    built.save(folder)
    form = read_folder(folder)
    history = history_of({key: value for key, value in form.reference.items() if key.startswith("history/")})
    return SimpleNamespace(folder=folder, built=built, form=form, history=history, fields=built.field_units(history))


def test_the_sliding_balls_fields_take_their_units_from_its_build_and_its_lock_records_each_blocks_resolution(
        sliding):
    fields = sliding.fields
    assert (fields["ball.pose"], fields["ball.twist"]) == (FieldUnits("m; 1"), FieldUnits("m/s; rad/s"))
    assert fields["gravity:ball.wrench"] == FieldUnits("N; N m")
    assert fields["ball.contact_points"] == FieldUnits("m", "ball.contact_forces"), "held points pair with the forces"
    assert fields["ball.contact_forces"] == FieldUnits("N"), "the typed state's newtons, not the touching flag's 1"
    reference = sliding.form.lock["reference"]
    resolution = reference["resolution"]
    assert sorted(resolution["ball.pose"]) == ["1", "m"] and sorted(resolution["ball.twist"]) == ["m/s", "rad/s"]
    assert sorted(resolution["gravity:ball.wrench"]) == ["N", "N m"]
    assert sorted(resolution["ball.contact_points"]) == ["m"]
    assert reference["resolution_factor"] == RESOLUTION_FACTOR
    assert resolution["ball.pose"]["m"] > 0 and resolution["ball.twist"]["rad/s"] > 0, \
        "the start one ulp off moved the body: the rerun measured something"
    forces = sliding.history["ball"]["contact_forces"]
    acting = np.flatnonzero(np.all(forces != 0, axis=0))
    assert acting.size == 1 and np.all(np.delete(forces, acting, axis=1) == 0), forces


def with_points(history: dict, points: np.ndarray) -> dict:
    return {**history, "ball": {**history["ball"], "contact_points": points}}


def test_a_held_point_of_a_row_that_takes_a_force_moved_at_ten_times_its_bound_is_moved_at_every_step(
        sliding, other_environment):
    points = sliding.history["ball"]["contact_points"]
    row = int(np.flatnonzero(np.all(sliding.history["ball"]["contact_forces"] != 0, axis=0))[0])
    reference = sliding.form.lock["reference"]
    magnitude = float(np.max(np.abs(points[:, row].astype(np.float64))))
    floor = reference["resolution"]["ball.contact_points"]["m"] or 0.0
    size = 10.0 * max(reference["rtol"] * magnitude, RESOLUTION_FACTOR * floor)
    for step in range(STEPS):
        moved = points.astype(np.float64).copy()
        moved[step, row, 0] += size
        report = verify_history(sliding.form, with_points(sliding.history, moved.astype(points.dtype)), sliding.fields)
        assert moved_labels(report) == {"ball.contact_points"}, (step, report.text())
        assert f"first at step {step}" in report.moved[0] and "on the rows whose force acts" in report.moved[0]


def test_a_held_point_of_a_row_that_takes_no_force_jumping_two_millimetres_is_not_reported(sliding, other_environment):
    """And the control: the same jump judged with no pairing, the field whole, is moved."""
    points = sliding.history["ball"]["contact_points"]
    far = int(np.flatnonzero(np.all(sliding.history["ball"]["contact_forces"] == 0, axis=0))[0])
    for step in range(STEPS):
        moved = points.astype(np.float64).copy()
        moved[step, far] += 2e-3
        found = with_points(sliding.history, moved.astype(points.dtype))
        assert verify_history(sliding.form, found, sliding.fields).reproduced, step
        unpaired = {**sliding.fields, "ball.contact_points": OWN}
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
    assert f"within rtol {lock['reference']['rtol']} of each block's largest magnitude" in report.text()
