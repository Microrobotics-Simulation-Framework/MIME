"""A coil array on a frame of its own apparatus's chain (``CoilArray(mount="<segment>.<frame>")``), on the rig of
``mount_cases``: Navion_2 on the rotor of A0's kinematic AR4, the rotor turning, a still sphere at the array's centre.

- **The array placed at the frame's pose.** At each step of the moving chain, the mounted array's field and gradient
  at the sphere against the host's ``actuation`` (float64) of the same array placed in the world at the rotor's pose
  that step: the pose the chain published (its forward kinematics, itself checked against ``frame_pose`` of the AR4's
  description turned by the rotor's angle), composed with the array's placement on the host. In float64 and float32,
  within ``SLACK`` times the rounding floor of each quantity's peak: ``eps`` times the sum of the magnitudes the
  quantity is summed from (``mount_cases.term_sums``: each source's terms, over the sources, the coils through the
  currents' magnitudes, turned into the world), over the quantity's peak. **This is a first-order running-error
  bound** (rounding at ``eps`` in a sum costs at most ``eps`` times the sum of the magnitudes, once through the
  evaluation's own roundings), and ``SLACK`` is ``tests/oracles/test_calibrated_multipole.py``'s factor over it, **an
  empirical factor, not a derived one**; the host's evaluation and the node's are two programs, each rounding, so the
  difference is both. **Its measured margin:** on a scan of 72 random chain poses (12 seeds of 6: the arm's joints
  within 0.25 rad of home, the rotor's angle and speed, the array's placement and the sphere within 2 cm of its
  centre all drawn; 10 steps each, the oracle at the host's pose of the rotor at the published angle) the largest miss
  is 2.3 floors in float64 and 1.9 in float32, so 4.3 and 5.2 times under ``SLACK``. The floor this test first took,
  ``eps / c`` with ``c`` the worst field cancellation of Navion_2's sources per coil, **was not a bound for these poses**: the
  scan's largest miss against it was 9.4 times (0.94 of ``10 eps / c``; float32 4.3), while against the term floor it
  is 2.3 and 1.9. ``c`` is one coil's sources' cancellation as a vector norm; the term floor also counts the cancellation
  between the coils through the currents, which are of both signs, and between a source's own terms (the two were not
  separated). The pose's rounding is not the cause: a perturbation of the pose by ``eps`` of its size moves the field
  by under 1e-15 of itself, against floors of 1e-14. The scan ran on the laptop, not a runner, and the margin is the 72 poses'
  and not a bound over all of them. ``tests/fields/test_mounted_array.py::test_the_mounted_field_at_the_worst_pose_of_a_scan_of_random_chain_poses``
  holds the scan's worst pose. Agreement to rounding, not bit for bit: the two are different programs. The
  control, the placement composed the other way round (the frame's pose inside the placement), misses at every step by
  more than a hundred times the tolerance: the comparison tells the two apart.
- **The forward model takes the mount's pose** (``x.mount_pose``): at each step's published rotor pose it gives the
  node's field and gradient, to rounding.
- **The exact allocation holds each tick** along the moving chain: the controller (``MountedFieldControl``, a callback,
  reading the rotor's pose and the sphere's) allocates the field ``WANTED`` at the sphere through ``ctx.allocator`` on
  its model of the mounted array; at each tick the currents written make ``WANTED`` at the pose it read, by the host's
  ``actuation`` there, within the rounding of the field's terms, and are the host's own exact allocation
  (``allocate()``) at that pose. The field delivered a step later is the array's at the pose the chain has moved to,
  which no allocation from the last read can know.
- **Refusals** (``mount.provider``), each with its control: a mount that names no frame of the chain (a frame of it
  builds); a frame of a segment that takes loads, the dynamic arm (the kinematic arm plans); ``actuation()`` of a
  mounted array (placed in the world, it gives one); and a belief mounted elsewhere than its truth
  (``controller.believed``; the same mount builds).
- **Unused, nothing moves**: an array placed in the world holds no ``mount`` in its record, and builds the node it
  built before (no mount input, no mount constant)."""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Iterator, Sequence
from functools import cache
from typing import Any

import jax
import numpy as np
import pytest

from believed_cases import model_of
from chain_cases import AR4, AR4_BASE, AR4_HOME
from mount_cases import CURRENTS, LIMIT, ROTOR, ROTOR_AT, SPEED, TURN, WANTED, MountedFieldControl, arm_mounted, \
    coils, mounted_rig, term_sums

from mime import Study
from mime.build import BuildRefused
from mime.build._plan import _Planner as Planner
from mime.core.multibody import frame_pose, read_urdf
from mime.assets import asset
from mime.experiment import Experiment
from mime.fields import CalibratedMultipole, allocate
from mime.fields._coil_array import calibration_of
from mime.observations import Exposure
from mime.saved._document import all_settings
from mime.study import grid
from mime.units import A, T, m, ms

jnp: Any = jax.numpy
#: The factor over the rounding floor ``eps / c``: ``tests/oracles/test_calibrated_multipole.py``'s.
SLACK = 10.0
STEPS = 40
#: The plant a control's detection rate is measured with (``AP3B_PLANT``): ``mount``, the planner's refusal of a mount
#: on a segment that takes loads removed (as if a mounted array's mass and reaction were modelled); ``compose``, the
#: node's placement composed the other way round (the frame's pose inside the placement).
PLANT = os.environ.get("AP3B_PLANT", "")
if PLANT == "compose":  # a fresh process per run, planted before any step is traced
    from mime.nodes.actuation import _multipole

    _compose = _multipole.compose
    _multipole.compose = lambda outer, inner: _compose(inner, outer)


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = q[1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + q[0] * v)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """``inner``, a pose ``(x, y, z, qw, qx, qy, qz)`` in ``outer``'s frame, in ``outer``'s parent, float64."""
    return np.concatenate([outer[:3] + rotate(outer[3:], inner[:3]), multiply(outer[3:], inner[3:])])


PLACEMENT = np.asarray((*ROTOR_AT, *TURN))


def placed(world: np.ndarray) -> CalibratedMultipole:
    """The array placed in the world at ``world``, its quaternion normalised in float64."""
    turn = world[3:] / np.linalg.norm(world[3:])
    return CalibratedMultipole(calibration=asset("navion_2"), at=m(tuple(world[:3])), orientation=tuple(turn))


def worst_cancellation(worlds: list[np.ndarray], point: np.ndarray) -> float:
    """The worst cancellation of Navion_2's sources at ``point`` read by the array at each of ``worlds``."""
    locals_ = []
    for world in worlds:
        turn = world[3:] / np.linalg.norm(world[3:])
        matrix = np.array([[1 - 2 * (turn[2] ** 2 + turn[3] ** 2), 2 * (turn[1] * turn[2] - turn[0] * turn[3]),
                            2 * (turn[1] * turn[3] + turn[0] * turn[2])],
                           [2 * (turn[1] * turn[2] + turn[0] * turn[3]), 1 - 2 * (turn[1] ** 2 + turn[3] ** 2),
                            2 * (turn[2] * turn[3] - turn[0] * turn[1])],
                           [2 * (turn[1] * turn[3] - turn[0] * turn[2]), 2 * (turn[2] * turn[3] + turn[0] * turn[1]),
                            1 - 2 * (turn[1] ** 2 + turn[2] ** 2)]])
        locals_.append(matrix.T @ (point - world[:3]))
    return calibration_of(asset("navion_2")).cancellation(np.asarray(locals_))


def miss(found: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(found - expected)) / np.max(np.abs(expected)))


@cache
def ar4_tree() -> Any:
    return read_urdf(str(AR4)).tree


def rotor_pose(angle: float, joints: Sequence[float] = AR4_HOME) -> np.ndarray:
    """The rotor's world pose on the host, float64: ``frame_pose`` of the AR4's flange at ``joints``, placed at its
    base, turned about the flange's z by ``angle``."""
    with in_dtype("float64"):
        flange = np.asarray(frame_pose(ar4_tree(), jnp.asarray(joints, dtype=jnp.float64), "link_6"),
                            dtype=np.float64)
    base = compose(np.asarray((*AR4_BASE, 1.0, 0.0, 0.0, 0.0)), flange)
    return compose(base, np.asarray((0.0, 0.0, 0.0, math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))))


@cache
def mounted_history(dtype: str) -> dict[str, dict[str, np.ndarray]]:
    with in_dtype(dtype):
        held = mounted_rig().build().run(ms(STEPS), history=True).history
    return {node: {field: np.asarray(value, dtype=np.float64) for field, value in fields.items()}
            for node, fields in held.items()}


# -- the array placed at the frame's pose -------------------------------------------------------------------------

def miss_ratios(history: dict[str, dict[str, np.ndarray]], point: np.ndarray, placement: np.ndarray, dtype: str, *,
                joints: Sequence[float] | None = None) -> tuple[list[float], list[float]]:
    """At each step of the run, in units of the rounding floor ``eps`` times the sum of the magnitudes the quantity is
    summed from over its peak (``mount_cases.term_sums``): the mounted array's field and then its gradient at ``point``
    against the host's ``actuation`` (float64) of the array placed at the rotor's pose composed with ``placement``; and,
    the control, the field against the placement composed the other way round (the frame's pose inside it). The
    rotor's pose is the frame the chain published, or, given the arm's ``joints``, the host's own pose of the rotor
    at the angle the motor published (a pose a rounding off the published one: the oracle's evaluation, a sum that
    cancels, changes by about the floor with it)."""
    eps = float(np.finfo(np.dtype(dtype)).eps)
    frames = history["rig.motor"]["frame_poses"][:, ROTOR]
    if joints is not None:
        frames = np.asarray([rotor_pose(float(angle), joints) for angle in history["rig.motor"]["angle"]])
    field = history["rig.coils"]["flux_density_magnetic_force_sphere"]
    gradient = history["rig.coils"]["gradient_magnetic_force_sphere"]
    currents = np.asarray(CURRENTS)
    ratios, wrong = [], []
    for step, frame in enumerate(frames):
        world = compose(frame, placement)
        made = placed(world).actuation(m(tuple(point)))
        expected = made.field @ currents, made.gradient @ currents
        floors = [eps * np.max(terms) / np.max(np.abs(value)) for terms, value in zip(term_sums(world, point), expected,
                                                                                      strict=True)]
        ratios += [miss(field[step], expected[0]) / floors[0], miss(gradient[step], expected[1]) / floors[1]]
        other = placed(compose(placement, frame)).actuation(m(tuple(point)))
        wrong.append(miss(field[step], other.field @ currents) / max(floors))
    return ratios, wrong


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_mounted_field_is_the_array_placed_at_the_frames_pose_along_the_moving_chain(dtype):
    history = mounted_history(dtype)
    frames = history["rig.motor"]["frame_poses"][:, ROTOR]
    eps = float(np.finfo(np.dtype(dtype)).eps)
    for step, angle in enumerate(history["rig.motor"]["angle"]):
        expected = rotor_pose(float(angle))
        assert np.max(np.abs(frames[step, :3] - expected[:3])) <= 256 * eps, step
        assert min(np.max(np.abs(frames[step, 3:] - expected[3:])), np.max(np.abs(frames[step, 3:] + expected[3:]))) \
            <= 256 * eps, step
    assert np.ptp(history["rig.motor"]["angle"]) > 0.05, "the chain moves: the rotor turns"
    assert np.all(history["sphere"]["pose"][:, :3] == 0.0), "the sphere stays where the field is read"
    point = np.zeros(3)
    worlds = [compose(frame, PLACEMENT) for frame in frames]
    ratios, wrong = miss_ratios(history, point, PLACEMENT, dtype)
    assert max(ratios) <= SLACK, (max(ratios), SLACK)
    assert min(wrong) > 100 * SLACK, "the control: the placement composed the other way round is told apart"
    moved = np.linalg.norm(worlds[-1][:3] - worlds[0][:3])
    assert moved > 1e-3, "the array's centre moves along the chain"


#: The pose of a scan of 72 random chain poses (12 seeds of 6; the joints within 0.25 rad of home, the rotor's angle and
#: speed, the array's placement and the sphere within 2 cm of its centre all drawn) at which float64's miss was the
#: largest: the seed-3 draw's first.
WORST = {"joints": (-0.23478541642818782, 0.10736525329804986, -0.6241127673968017, 1.597191018032184,
                    1.3433443211201996, -0.0334365298817631),
         "angle": -0.1316245757259975, "speed": -2.0415665121775284,
         "at": (-0.10554228485907854, -0.11513279800785967, -0.010877180950433794),
         "turn": (-0.2655717040726008, -0.21184190425601945, -0.5031160334901614, -0.7946501962765034),
         "local": (0.018250690193443945, -0.008631953450048342, 0.005941888283193003)}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_mounted_field_at_the_worst_pose_of_a_scan_of_random_chain_poses(dtype):
    """The pose of the scan where the array's field and gradient missed ``actuation`` by most in float64, run in both
    dtypes, against the oracle at the rotor's published pose and at the host's own pose at the published angle (the
    way the scan took it): the miss in units of the floor of the sums the field is made of
    (``mount_cases.term_sums``), under the ``SLACK`` of 10. On the laptop that miss was 9.4 times ``eps / c`` at the
    host's own pose (the scan's worst, 0.94 of the tolerance ``10 eps / c`` the test once had, which ``c``, the
    sources' cancellation per coil, does not bound: the coils cancel through the currents too, and each source's terms)
    and 2.0 times the term floor; at the published pose 4.6 and 1.3 (float32: 4.4 and 1.1). A planted error, the
    placement composed the other way round, misses by more than a hundred ``SLACK``."""
    pose = rotor_pose(WORST["angle"], WORST["joints"])
    placement = np.asarray((*WORST["at"], *WORST["turn"]))
    world = compose(pose, placement)
    point = world[:3] + rotate(world[3:] / np.linalg.norm(world[3:]), np.asarray(WORST["local"]))
    with in_dtype(dtype):
        experiment = mounted_rig("mounted_worst_pose", array=coils(at=WORST["at"], orientation=WORST["turn"]),
                                 speed=WORST["speed"], joints=WORST["joints"], angle=WORST["angle"],
                                 sphere_at=tuple(point))
        held = experiment.build().run(ms(10), history=True).history
    history = {node: {field: np.asarray(value, dtype=np.float64) for field, value in fields.items()}
               for node, fields in held.items()}
    assert np.all(history["sphere"]["pose"][:, :3] == np.asarray(point, dtype=np.float32 if dtype == "float32"
                                                                 else np.float64)), "the sphere stays where it starts"
    for joints in (None, WORST["joints"]):
        ratios, wrong = miss_ratios(history, point, placement, dtype, joints=joints)
        assert max(ratios) <= SLACK, (joints is None, max(ratios), SLACK)
        assert min(wrong) > 100 * SLACK, "the control: the placement composed the other way round is told apart"


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_forward_model_takes_the_mounts_pose_and_gives_the_nodes_field(dtype):
    history = mounted_history(dtype)
    eps = float(np.finfo(np.dtype(dtype)).eps)
    with in_dtype(dtype):
        model = model_of(mounted_rig(), "rig.coils")
        assert set(model.inputs) >= {"point", "mount_pose"} and model.affine_in == {"currents"}
        frames = history["rig.motor"]["frame_poses"][:, ROTOR]
        worlds = [compose(frame, PLACEMENT) for frame in frames]
        dtype_ = jnp.zeros(()).dtype
        evaluate = jax.jit(lambda pose: model({"currents": jnp.asarray(CURRENTS, dtype=dtype_)},
                                              {"point": jnp.zeros(3, dtype=dtype_), "mount_pose": pose}))
        for step in range(0, STEPS, 7):
            made = evaluate(jnp.asarray(frames[step], dtype=dtype_))
            for quantity, slot, terms in zip(("flux_density", "gradient"),
                                             ("flux_density_magnetic_force_sphere",
                                              "gradient_magnetic_force_sphere"),
                                             term_sums(worlds[step], np.zeros(3)), strict=True):
                expected = history["rig.coils"][slot][step]
                floor = eps * np.max(terms) / np.max(np.abs(expected))
                assert miss(np.asarray(made[quantity], dtype=np.float64), expected) <= SLACK * floor, (quantity, step)


# -- the exact allocation each tick -------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_the_exact_allocation_holds_at_each_tick_of_the_moving_chain(dtype):
    """A tick at step ``k`` reads the state as it stands before step ``k`` (a callback, latency zero): the rotor's pose
    published at step ``k - 1``, or its pose at the start for the first tick."""
    with in_dtype(dtype):
        built = mounted_rig("mounted_allocation", controller=MountedFieldControl()).build()
        history = built.run(ms(20), history=True).history
        start = next(p for p in built.plans.entries if p.owner == "rig.coils").mount
    assert start is not None
    frames = np.asarray(history["rig.motor"]["frame_poses"], dtype=np.float64)[:, ROTOR]
    written = np.asarray(history["ports:rig"]["written_currents"], dtype=np.float64)
    eps = float(np.finfo(np.dtype(dtype)).eps)
    wanted = np.asarray(WANTED)
    ticks = list(range(0, 20, 2))
    reads = [np.asarray(start.initial_pose) if tick == 0 else frames[tick - 1] for tick in ticks]
    worlds = [compose(read, PLACEMENT) for read in reads]
    floor = eps / worst_cancellation(worlds, np.zeros(3))
    moved = []
    for tick, world in zip(ticks, worlds, strict=True):
        made = placed(world).actuation(m((0.0, 0.0, 0.0)))
        terms = np.abs(made.field) @ np.abs(written[tick])
        assert np.all(np.abs(made.field @ written[tick] - wanted) <= SLACK * floor * np.max(terms)), tick
        exact = np.asarray(allocate(made, limits=A(LIMIT), field=T(WANTED)).currents.value)
        condition = float(np.linalg.cond(made.field))
        assert np.max(np.abs(written[tick] - exact)) <= SLACK * condition * floor * np.max(np.abs(exact)), tick
        moved.append(written[tick])
    assert np.ptp(np.asarray(moved), axis=0).max() > 1e-3, "the currents follow the chain: the allocation moves"
    assert np.max(np.abs(np.asarray(moved))) < LIMIT, "within the port's bounds: the exact solution is reached"


def test_inside_the_compiled_step_the_controller_reading_the_chains_frames_is_refused_not_given_a_late_value():
    """In mode 1 the controller's node runs first in the step and each read edge must carry the state at the step's
    start; nothing the controller writes reaches the motor, so the schedule runs the motor before it and its read of
    the rotor's pose would carry the end of the step. The build refuses it, with its code, and names the read."""
    with pytest.raises(BuildRefused) as caught:
        mounted_rig("mounted_in_graph", controller=MountedFieldControl(), mode="graph").build()
    found = [r for r in caught.value.refusals if r.code == "controller.unreached-read"]
    assert found and "reads rig.motor.frame_poses, which nothing it writes reaches" in found[0].message


# -- refusals -----------------------------------------------------------------------------------------------------

def refused(experiment) -> BuildRefused:
    with pytest.raises(BuildRefused) as caught:
        experiment.plan()
    return caught.value


@pytest.fixture
def plant(monkeypatch):
    if PLANT == "mount":
        monkeypatch.setattr(Planner, "load_port", lambda self, plan, interface=None: None)


@pytest.mark.parametrize("mount, held", [("arm.link_9", True), ("arm.link_6", False)],
                         ids=["no_such_frame", "control_a_frame_of_the_chain"])
def test_a_mount_that_names_no_frame_of_the_chain_is_refused(mount, held):
    experiment = arm_mounted(mount=mount)
    if not held:
        experiment.plan()
        return
    found = [r for r in refused(experiment).refusals if r.code == "mount.provider"]
    assert found and "no frame of a segment of its apparatus's chain" in found[0].message
    assert "arm.link_6" in found[0].action


@pytest.mark.parametrize("rung", ["dynamic", "kinematic"], ids=["dynamic_arm", "control_kinematic_arm"])
def test_a_mount_on_a_segment_that_takes_loads_is_refused_on_both_arm_rungs(plant, rung):
    experiment = arm_mounted(rung=rung)
    if rung == "kinematic":
        plans = experiment.plan()
        assert next(p for p in plans.entries if p.owner == "rig.coils").mount is not None
        return
    found = [r for r in refused(experiment).refusals if r.code == "mount.provider"]
    assert found, "a mounted array on a dynamic arm is refused"
    assert "mass on the segment and the reaction on it are not modelled" in found[0].message
    assert "kinematic" in found[0].action


@pytest.mark.parametrize("mounted", [True, False], ids=["mounted", "control_placed_in_the_world"])
def test_a_mounted_arrays_host_actuation_is_refused(mounted):
    array = coils() if mounted else coils(mount=None)
    if not mounted:
        assert array.actuation(m((0.0, 0.0, 0.0))).coils == 3
        return
    from mime.build import ArgumentRefused

    with pytest.raises(ArgumentRefused) as caught:
        array.actuation(m((0.0, 0.0, 0.0)))
    assert [r.code for r in caught.value.refusals] == ["mount.provider"]


@pytest.mark.parametrize("mount, code", [("arm.link_6", "controller.believed"), ("motor.rotor", None)],
                         ids=["mounted_elsewhere", "control_the_same_mount"])
def test_a_belief_mounted_elsewhere_than_its_truth_is_refused(mount, code):
    experiment = mounted_rig(controller=MountedFieldControl(), believed={"rig.coils": coils(mount=mount)})
    if code is None:
        experiment.plan()
        return
    found = [r for r in refused(experiment).refusals if r.code == code]
    assert found and "is mounted on 'arm.link_6', and the part on 'motor.rotor'" in found[0].message


# -- a study over the mounted rig --------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_study_sweeping_a_mounted_rigs_currents_gives_each_member_its_own_runs_values(dtype):
    """Two members, the rig's currents at two values (one group, batched: the mounted array's pose is read from the
    chain at each step of each member): each member's final state is its own experiment's run's, built alone, within
    the rounding of the field's terms (the two are different compiled programs, a batched step and a single one:
    ``SLACK eps`` times the field's term sum, ``mount_cases.term_sums``, of each quantity's scale; the rate of change,
    a difference of two steps' fields over ``dt``, by that over ``dt``). The control: the second member against the
    first's own run misses by more than a hundred bounds, so a study that gave every member the first one's currents
    is told apart."""
    values = [A(CURRENTS), A((4.0, -2.0, 1.0))]
    eps = float(np.finfo(np.dtype(dtype)).eps)

    def rig() -> Experiment:
        return mounted_rig("mounted_study", observe=Exposure("sphere", field="rig.coils"))

    with in_dtype(dtype):
        study = Study(rig()).sweep(grid({rig().handle("rig").commands.currents: values}), duration=ms(20))
        plan = study.plan()
        assert len(plan.groups) == 1 and len(plan.points) == 2
        result = study.run()
        owns = [Experiment.declared_from(point.declared.settings, point.declared.declarations).build()
                .run(ms(20), history=True) for point in plan.points]
    floors = []
    for own, current in zip(owns, values, strict=True):
        frames = np.asarray(own.history["rig.motor"]["frame_poses"], dtype=np.float64)[:, ROTOR]
        world, currents = compose(frames[-1], PLACEMENT), np.asarray(current.value)
        made = placed(world).actuation(m((0.0, 0.0, 0.0)))
        floors.append(max(eps * np.max(terms) / np.max(np.abs(value)) for terms, value in zip(
            term_sums(world, np.zeros(3), currents), (made.field @ currents, made.gradient @ currents), strict=True)))

    def worst_miss(member: Any, own: Any, floor: float) -> float:
        """The largest miss of ``member`` against ``own``, in units of its bound."""
        peak, worst = float(own.final["exposure:sphere"]["max_flux_density"]), 0.0
        for node, fields in own.final.items():
            for name, held in fields.items():
                found, expected = np.asarray(member.final[node][name], dtype=np.float64), np.asarray(held, np.float64)
                bound = 2 * SLACK * floor * peak / 1e-3 if name.endswith("rate_of_change") \
                    else SLACK * floor * np.max(np.abs(expected))
                worst = max(worst, float(np.max(np.abs(found - expected)) / max(bound, SLACK * eps * np.max(np.abs(expected)))))
        return worst

    for index, own in enumerate(owns):
        assert worst_miss(result.member(index), own, floors[index]) <= 1.0, index
    assert worst_miss(result.member(1), owns[0], floors[0]) > 100, "the control: a member given the other's currents"


# -- unused -------------------------------------------------------------------------------------------------------

def test_an_array_placed_in_the_world_keeps_its_record_and_its_node():
    assert "mount" not in all_settings(coils(mount=None), "coils")
    assert all_settings(coils(), "coils")["mount"] == "motor.rotor"
    built = mounted_rig(array=coils(mount=None, at=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))).build()
    node = built.graph.get_node("rig.coils")
    assert "mount_poses" not in node.boundary_input_spec()
    assert not {"mount_index", "mount_count"} & set(node.constants())
    mounted = mounted_rig().build().graph.get_node("rig.coils")
    assert "mount_poses" in mounted.boundary_input_spec() and mounted.constants()["mount_index"] == ROTOR
    assert SPEED > 0.0
