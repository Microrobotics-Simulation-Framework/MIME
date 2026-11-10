"""The MRS rung's separated quadrature near a wall (drag 0.4.0, MIME-ANO-130; phase 7m-b's fix round), each with a
control that fails on every run (no draw: the runs are deterministic).

- **A run whose resistance turns indefinite stops at that step, before the body takes it** (the review's script A
  case: FL-9 on the floor of the 1/4 in tube under gravity, the contact and the field, at the screw's defaults). It
  builds, definite at its start, and its resistance turns indefinite at step 10, where the node replaces it by NaN and
  the run refuses with ``drag.mrs-indefinite``, naming the step, the pose, the gap and the way out; before it every
  step's resistance is definite and the drag takes energy out. The control, the stop removed, runs on through the
  indefinite resistance, and the drag puts energy in.
- **The body's distance from the wall** is reported at build (with the force spacing the experiment's closest approach
  needs) and at each step (``near_wall``, within one force spacing), and the confined resistance's asymmetry beside it
  (``asymmetric``); the gap the node reports is the tube's own distance (``WallKind.distance``, 7m-a's) at its
  quadrature, which a planted distance fails.
- **The build refuses an indefinite resistance at the start, read inside the lumen** (the surrogate generator's own
  pose, its body 0.2 mm from the wall at its nearest point): every read accepted and the eigenvalue a negative number.
  The control: the pose the test took before, from the mean radius, puts the body through the wall, where the reads
  are refused.
- **A quadrature below its blob's rule is refused where refining it moves the resistance** (``drag.mrs-quadrature``):
  the point forces at blob 0.1 miss the shipped free-space reference by 72%, symmetric to rounding, which the
  asymmetry limit passes; the move from them to the rule's quadrature predicts that miss. The control: with the limit
  lifted the same build passes.
- **A body held on the wall is refused by name** (``drag.mrs-on-wall``): a contact's margin of 0 leaves no force spacing
  that clears it, and the spacing's rule takes none there. The control plants the change out, and the body plans on
  its family's spacing with nothing said.
- **The contact that holds the hull (0.4.0) declares its margin** (``holds_off``), so a body it holds takes its default
  force spacing from the margin, CONFINED_GAP force spacings or more off the wall. The control takes the declaration
  off, and the plan falls back to the family's spacing at every margin.

Float64 (``tests/oracles/conftest.py``); measured in float32 too, where the run stops at the same step."""

import contextlib
import ctypes
import dataclasses
import gc
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from mrs_cases import drag_node, free_space, resistance_at
from script_a import fl9

import mime.nodes.hydro.mrs._node as mrs_node
from mime import Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused, RunFailed
from mime.core import quaternion
from mime.dynamics import RigidInertial
from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
from mime.nodes.hydro.mrs import _quadrature
from mime.nodes.hydro.mrs._quadrature import capped_cells
from mime.nodes.hydro.mrs._tube import TUBE_CORRECTION
from mime.scene import Scene, Tube, Z
from mime.surrogates.generator import design_named, resting_offset
from mime.things import Part
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s, s
from mime.walls import TubeWall

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(__file__).parent / "references" / "helix_fl9_free_space.json"
TUBE_RADIUS = 3.175e-3
#: Script A's base step and the run's length: the review measured the stop at step 10 (t = 5 ms), in float32 and in
#: float64 alike (unchecked, the state went non-finite at step 13 and 15); twenty-four steps leave room for a runner
#: whose rounding moves it.
DT = 0.5e-3
STEPS = 24
#: The node's gap against the generator's nominal one (the body's nearest surface point): the quadrature's points lie
#: on the surface, so its least distance is the nominal one to within the quadrature's resolution of the crest;
#: measured at most 1.3 um on FL-9 at 0.6 and 0.829 mm (96 poses, eight rolls), 0.03 um on the sphere.
GAP_RESOLUTION = 5e-6


def operating_point():
    """``scripts/sim2real/operating_point.py``, loaded as the register's pins test loads it."""
    path = ROOT / "scripts" / "sim2real" / "operating_point.py"
    spec = importlib.util.spec_from_file_location("_near_wall_operating_point", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OP = operating_point()


@pytest.fixture(autouse=True)
def memory_given_back():
    """Each test here builds FL-9 in the tube, and JAX's caches keep each build's programs and the node they close over
    alive in the worker: 1.5 GiB more per worker over this file, which on CI's 16 GB runner, beside what a shard's other
    tests hold, lost the runner (run 41362062284's and 41366046326's shard 2, and this file in shard 4 after it moved).
    So what a test made goes when it ends, as a ``memory_heavy`` test's free memory does (``tests/conftest.py``)."""
    yield
    cast(Any, jax).clear_caches()
    gc.collect()
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def floored(monkeypatch, start_mm=(0.0, 0.0, -1.0), wall_refinement: int | None = 1) -> Experiment:
    """Script A's experiment on the MRS rung (``operating_point.experiment(rung="bem")``) with its laws and kind
    unpinned, so that it is drag 0.4.0 at the screw's defaults at 0.829 mm (222 force points, blob 0.25, the
    quadrature eight times finer), FL-9 resting on the floor at the contact's 0.1 mm margin; the review's case, with the
    wall's image at the force points (``wall_refinement`` 1) unless another is given."""
    monkeypatch.setattr(OP, "pinned", lambda geometry: geometry)
    monkeypatch.setattr(OP, "pinned_law", lambda law: dataclasses.replace(law, wall_refinement=wall_refinement,
                                                                          lubrication=False)
                        if law.name == "drag" else law)
    return OP.experiment(rung="bem", spacing_mm=0.829, blob=0.25, start_mm=start_mm)


def ran(built):
    """The run of ``STEPS`` with history, and how it ended: the result and the refusal, if any."""
    try:
        return built.run(s(STEPS * DT), history=True), None
    except RunFailed as failed:
        return failed.result, failed


def drag_powers(history) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each step's resistance, its least eigenvalue as the node judged it, and the drag's power on the body,
    ``-v^T R v`` (above zero: the drag puts energy in)."""
    drag, body = history[OP.DRAG_NODE], history[OP.BODY_NODE]
    resistance = np.asarray(drag["resistance"], dtype=np.float64)
    twist = np.asarray(body["twist"], dtype=np.float64)
    power = np.einsum("ki,kij,kj->k", twist, resistance, twist) * -1.0
    return resistance, np.asarray(drag["least_eigenvalue"], dtype=np.float64), power


@pytest.fixture(scope="module")
def floored_run():
    """Script A's floored screw at the defaults, built and run once for the tests that read it."""
    with pytest.MonkeyPatch.context() as patched:
        built = floored(patched).build()
    result, failed = ran(built)
    return built, result, failed


def test_script_as_floored_screw_builds_and_its_run_stops_at_the_first_indefinite_step_before_the_body_takes_it(
        floored_run):
    _, result, failed = floored_run
    assert failed is not None and failed.codes() == ["drag.mrs-indefinite"], "the run stops on the drag's own code"
    drag = result.history[OP.DRAG_NODE]
    resistance, least, power = drag_powers(result.history)
    raised = np.asarray(drag["indefinite"]) > 0
    first = int(np.argmax(raised))
    print(f"\nstopped at step {first + 1}: least eigenvalue before it {np.round(least[:first], 3)}, drag power "
          f"{power[:first].max():.2e} at most; {failed.refusals[0].message}")
    assert raised.any() and first > 0, "definite at the start, as the build found it, and indefinite later"
    assert np.all(np.isfinite(resistance[:first])) and np.all(least[:first] > 0.0)
    assert np.all(power[:first] < 0.0), "the drag takes energy out at every step before the stop"
    assert np.all(np.isnan(resistance[first])), "the indefinite resistance never reaches the body"
    body_pose = np.asarray(result.history[OP.BODY_NODE]["pose"], dtype=np.float64)
    assert np.allclose(np.asarray(drag["stopped_pose"])[-1], body_pose[first - 1], rtol=0.0, atol=1e-12), \
        "the pose recorded is the one the drag read, the body's at the step's start"
    message, action = failed.refusals[0].message, failed.refusals[0].action
    gap = float(np.asarray(drag["stopped_gap"])[-1])
    assert f"step {first + 1} " in message and "quaternion" in message and f"{gap * 1e3:.4g} mm" in message
    finer = re.search(r"Drag\.MRS\(spacing=mm\(([0-9.]+)\)\)", action)
    assert finer is not None and float(finer.group(1)) == pytest.approx(1e3 * gap / mrs_node.CONFINED_GAP, rel=1e-2)
    assert "quadrature_refinement=1) are no way out" in action


def test_control_the_stop_removed_lets_the_indefinite_resistance_through_and_the_drag_puts_energy_in(monkeypatch):
    monkeypatch.setattr(mrs_node, "stops_where", lambda accepted, least: jnp.zeros((), dtype=bool))
    result, failed = ran(floored(monkeypatch).build())
    assert failed is None or failed.codes() != ["drag.mrs-indefinite"]
    resistance, least, power = drag_powers(result.history)
    finite = np.all(np.isfinite(resistance), axis=(1, 2))
    taken = finite & ~(least > 0.0)
    print(f"\nthe stop removed: indefinite resistances taken at steps {np.flatnonzero(taken) + 1}, the drag's power "
          f"above zero at {np.flatnonzero(finite & (power > 0.0)) + 1}")
    assert taken.any(), "an indefinite resistance reaches the body"
    assert np.any(finite & (power > 0.0)), "and the drag puts energy in"


def test_the_body_near_the_wall_is_reported_at_build_and_at_each_step_with_the_spacing_its_closest_approach_needs(
        floored_run):
    built, result, _ = floored_run
    lines = built.report.checked[OP.DRAG_NODE]
    print("\n" + "\n".join(lines))
    assert "0.121 of the force spacing (0.829 mm)" in lines[0]
    assert "closest approach is 0.1 mm" in lines[1]
    assert f"at most {0.1 / mrs_node.CONFINED_GAP:.4g} mm" in lines[1] and "222 force points" in lines[2]
    needed = re.search(r"at about ([0-9.]+) GiB a run", lines[1])
    assert needed is not None and float(needed.group(1)) == pytest.approx(
        mrs_node.run_gib(222 * (0.829 * mrs_node.CONFINED_GAP / 0.1) ** 2), rel=0.05)
    assert f"about {mrs_node.run_gib(222):.2g} GiB for the run" in lines[2] and mrs_node.STEP_MEASURED in lines[2]
    assert mrs_node.RUN_GIB_BASIS in lines[1] and mrs_node.RUN_GIB_BASIS in lines[2], "the memory stated as a fit"
    assert f"at most {0.1 / mrs_node.INTEGRATED_GAP:.4g} mm" in lines[3] and "16 GB of memory or less" in lines[3]
    assert mrs_node.WALL_IMAGE_MEMORY in lines[3] and "a step's time alone, not its memory" in lines[3]
    drag = result.history[OP.DRAG_NODE]
    first = int(np.argmax(np.asarray(drag["indefinite"]) > 0))
    assert np.all(np.asarray(drag["near_wall"])[:first + 1] > 0), "near_wall at every step on the floor"
    assert np.all(np.asarray(drag["asymmetric"])[:first + 1] > 0), "and the confined asymmetry past its limit"
    assert {flag.flag.field for flag in result.flags if flag.raised} >= {"near_wall", "asymmetric", "indefinite"}


def test_control_from_the_axis_near_wall_is_raised_only_where_the_body_comes_within_a_force_spacing(monkeypatch):
    """The same experiment started on the tube's axis: the run goes on, never indefinite, and ``near_wall`` is down at
    the start (the body 1.1 mm, 1.33 force spacings, from the wall) and up exactly at the steps where the node's gap
    is under one force spacing (the body, turning under the field, tilts its ends towards the wall)."""
    result, failed = ran(floored(monkeypatch, start_mm=(0.0, 0.0, 0.0)).build())
    drag = result.history[OP.DRAG_NODE]
    gap, near = np.asarray(drag["wall_gap"], dtype=np.float64), np.asarray(drag["near_wall"]) > 0
    print(f"\nfrom the axis: gap {gap[0] * 1e3:.4f} to {gap.min() * 1e3:.4f} mm; near_wall at steps "
          f"{np.flatnonzero(near) + 1}")
    assert failed is None and not np.any(np.asarray(drag["indefinite"]) > 0)
    assert np.all(np.asarray(drag["least_eigenvalue"]) > 0.0)
    assert not near[0] and gap[0] > 0.829e-3
    assert np.array_equal(near, gap < 0.829e-3)


def tube_screw(law, offset: float = 0.0, roll: float = 0.0) -> Experiment:
    """FL-9 in the walk's 1/4 in tube along +x, its centre ``offset`` (m) below the axis, turned ``roll`` about its own
    axis, with ``law`` and nothing else."""
    e = Experiment("near_the_floor", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    e.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.175), wall=mm(0.5),
                                                    length=mm(100.0))))
    e.add_medium(Medium("water", fills="tube.lumen"))
    e.declare("water.viscosity", mPa_s(1.0), reason="water")
    e.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(fl9())}, initial=State(
        position=mm((0.0, 0.0, -offset * 1e3)), orientation=(math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0),
        velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    e.declare("body.mass", kg(1e-5), reason="a millimetre body's order")
    e.declare("body.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    e.attach(law, body="body", medium="water")
    return e


def generators_pose(gap: float, roll: float = 0.0) -> float:
    """The surrogate generator's resting offset for FL-9: its nearest surface point ``gap`` (m) from the wall."""
    return resting_offset(design_named("FL-9"), TUBE_RADIUS, gap, roll)


def built_unchecked(experiment: Experiment, monkeypatch) -> MRSDragNode:
    with monkeypatch.context() as patched:
        patched.setattr(MRSDragNode, "built_problems", lambda self, start=None: [])
        return drag_node(experiment.build(), "body")


def pose_at(offset: float, roll: float = 0.0) -> np.ndarray:
    return np.array((0.0, 0.0, -offset, math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0))


def tube_distance_of(node: MRSDragNode, pose: np.ndarray) -> float:
    """The least of the tube wall kind's own distance (``WallKind.distance``, minus the cylinder's signed distance:
    7m-a's) over the node's quadrature at ``pose``, independent of the correction's constants."""
    body = node.body({key: jnp.asarray(node.constants()[key]) for key in node.body_leaf_names()})
    turn = np.asarray(quaternion.rotation_matrix(jnp.asarray(pose[3:])), dtype=np.float64)
    points = pose[:3] + np.asarray(body.points, dtype=np.float64) @ turn.T
    parameters = {"centre": jnp.zeros(3), "axis": jnp.asarray((1.0, 0.0, 0.0)), "radius": TUBE_RADIUS}
    wall = TubeWall()
    return float(jnp.min(jax.vmap(lambda point: wall.distance(parameters, point).distance)(jnp.asarray(points))))


#: The generator's case (blob 0.5, the quadrature four times finer, 0.6 mm), as the PR's refusal test took it, with the
#: wall's image at the force points, as the review measured it there.
GENERATORS = Drag.MRS(lubrication=False, spacing=mm(0.6), blob=0.5, quadrature_refinement=4, wall_refinement=1, wall_kind="tube")


def test_the_build_refuses_an_indefinite_resistance_at_the_generators_own_pose_its_reads_accepted(monkeypatch):
    """FL-9 0.2 mm above the floor at its nearest point (the generator's ``resting_offset``): every read inside the
    table and the eigenvalue a negative number (the review measured -6.80), refused by name at the start."""
    offset = generators_pose(0.2e-3)
    with pytest.raises(BuildRefused) as refused:
        tube_screw(GENERATORS, offset).build()
    assert refused.value.codes() == ["drag.mrs-resistance"]
    message = refused.value.refusals[0].message
    found = re.search(r"scaled by its diagonal, is (-?[0-9.e+-]+)", message)
    assert "starting pose" in message and found is not None
    judged = built_unchecked(tube_screw(GENERATORS, offset), monkeypatch).judged_at(pose_at(offset))
    print(f"\nat the generator's pose: {message}")
    assert judged.accepted and math.isfinite(judged.least) and judged.least < 0.0
    assert float(found.group(1)) == pytest.approx(judged.least, rel=1e-2)
    assert "quadrature_refinement=1) are no way out" in refused.value.refusals[0].action


def test_control_the_mean_radius_pose_puts_the_body_through_the_wall_where_no_read_is_accepted(monkeypatch):
    """The pose the refusal test took before, lowered by the gap from the mean radius (1.56 mm) rather than from the
    outer reach: the body lies past the wall, every read is refused, and an assertion of accepted reads fails it."""
    offset = TUBE_RADIUS - float(fl9().radius.value) - 0.2e-3
    judged = built_unchecked(tube_screw(GENERATORS, offset), monkeypatch).judged_at(pose_at(offset))
    assert not judged.accepted and judged.gap is not None and judged.gap < 0.0


def test_the_gap_the_node_reports_is_the_tube_wall_kinds_own_distance_at_its_quadrature(monkeypatch):
    offset = generators_pose(0.1e-3, math.pi / 8)
    node = built_unchecked(tube_screw(Drag.MRS(lubrication=False, spacing=mm(0.829), wall_kind="tube"), offset, math.pi / 8),
                           monkeypatch)
    judged = node.judged_at(pose_at(offset, math.pi / 8))
    expected = tube_distance_of(node, pose_at(offset, math.pi / 8))
    print(f"\nthe node's gap {judged.gap * 1e3:.6f} mm, the wall kind's {expected * 1e3:.6f} mm, nominal 0.1 mm")
    assert judged.gap == pytest.approx(expected, rel=1e-9)
    assert abs(judged.gap - 0.1e-3) < GAP_RESOLUTION


def test_control_a_distance_from_the_axis_in_the_walls_place_fails_the_gap(monkeypatch):
    from mime.nodes.hydro.mrs import _walls

    planted = dataclasses.replace(TUBE_CORRECTION, distance=lambda points, parameters: jnp.sqrt(
        jnp.sum(points[:, 1:] ** 2, axis=-1)))
    monkeypatch.setitem(_walls._corrections, (TUBE_CORRECTION.name, TUBE_CORRECTION.version), planted)
    offset = generators_pose(0.1e-3, math.pi / 8)
    node = built_unchecked(tube_screw(Drag.MRS(lubrication=False, spacing=mm(0.829), wall_kind="tube"), offset, math.pi / 8),
                           monkeypatch)
    judged = node.judged_at(pose_at(offset, math.pi / 8))
    assert judged.gap != pytest.approx(tube_distance_of(node, pose_at(offset, math.pi / 8)), rel=1e-3)


def speed_miss(built) -> float:
    point = json.loads(REFERENCE.read_text(encoding="utf-8"))["reference"]["speed_m_per_s"]
    r = resistance_at(built, "body", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    return float(-r[0, 3] * 2.0 * math.pi * 10.0 / r[0, 0]) / point - 1.0


POINT_FORCES = Drag.MRS(spacing=mm(0.829), blob=0.1, quadrature_refinement=1, wall_kind="free_space")


def test_a_quadrature_below_its_rule_that_refining_moves_is_refused_and_the_asymmetry_limit_would_pass_it():
    """FL-9's point forces at blob 0.1 (a refinement of 1, the rule's 20): refused by name, the move the message
    states predicting the miss; at a refinement of 4 (0.8% from the rule's) it builds."""
    with pytest.raises(BuildRefused) as refused:
        free_space("point_forces", {"body": fl9()}, POINT_FORCES).build()
    assert refused.value.codes() == ["drag.mrs-quadrature"]
    found = re.search(r"moves by ([0-9.]+)% of itself", refused.value.refusals[0].message)
    print(f"\n{refused.value.refusals[0].message}")
    assert found is not None and float(found.group(1)) > 50.0
    free_space("refined", {"body": fl9()}, Drag.MRS(spacing=mm(0.829), blob=0.1, quadrature_refinement=4,
                                                     wall_kind="free_space")).build()


def test_control_with_the_limit_lifted_the_point_forces_build_symmetric_to_rounding_and_72_per_cent_off(monkeypatch):
    monkeypatch.setattr(mrs_node, "QUADRATURE_MOVE_LIMIT", math.inf)
    built = free_space("point_forces", {"body": fl9()}, POINT_FORCES).build()
    node = drag_node(built, "body")
    miss = speed_miss(built)
    print(f"\nthe limit lifted: asymmetry {node.free_asymmetry:.1e}, speed {miss:+.1%} from the reference")
    assert node.free_asymmetry < 1e-12 < mrs_node.ASYMMETRY_LIMIT
    assert miss < -0.5


# -- the wall's image integrated over each cell (W2, design #204) ---------------------------------------------------

def test_the_tube_image_between_a_point_set_and_itself_is_the_tube_correction(monkeypatch):
    """The wall kind's kernel between arbitrary point sets (``tube_image``) with the targets the sources is the
    correction the node took at the force points (``tube_correction``): one blend of the same reads, made both ways."""
    from mime.nodes.hydro.mrs._tube import tube_correction, tube_image

    offset = generators_pose(0.3e-3)
    node = built_unchecked(tube_screw(Drag.MRS(lubrication=False, spacing=mm(0.829), wall_kind="tube"), offset), monkeypatch)
    parameters = wall_parameters(node)
    points = jnp.asarray(world_force_points(node, pose_at(offset))[::4])  # a quarter of them: every pair is read twice
    pairs = tube_image(points, points, parameters, 1e-3).matrix
    count = 3 * points.shape[0]
    found = np.asarray(pairs.transpose(0, 2, 1, 3).reshape(count, count))
    expected = np.asarray(tube_correction(points, parameters, 1e-3).matrix)
    print(f"\nthe kernel against the correction: {np.max(np.abs(found - expected)) / np.max(np.abs(expected)):.1e}")
    assert np.max(np.abs(found - expected)) <= 1e-12 * np.max(np.abs(expected))


def wall_parameters(node: MRSDragNode) -> dict:
    from mime.nodes.hydro.mrs import WALL

    held = node._step_constants(node.params_pytree())
    return {name: held[f"{WALL}{name}"] for name in TUBE_CORRECTION.parameters}


def world_force_points(node: MRSDragNode, pose: np.ndarray) -> np.ndarray:
    body = node.body({key: jnp.asarray(node.constants()[key]) for key in node.body_leaf_names()})
    turn = np.asarray(quaternion.rotation_matrix(jnp.asarray(pose[3:])), dtype=np.float64)
    return pose[:3] + np.asarray(body.force, dtype=np.float64) @ turn.T


def capped_cell(mine: np.ndarray, weights: np.ndarray, most: int) -> tuple[np.ndarray, np.ndarray]:
    """One cell's wall points, ``mine`` in the quadrature's order, under the cap, written again: ``most`` runs of nearly
    equal count, each its middle point with its run's weights added."""
    groups = min(len(mine), most)
    run = np.arange(len(mine)) * groups // len(mine)
    members = [np.flatnonzero(run == group) for group in range(groups)]
    return (np.array([mine[(held[0] + held[-1]) // 2] for held in members]),
            np.array([weights[mine[held]].sum() for held in members]))


def wall_cap(owners: np.ndarray, count: int) -> int:
    return min(int(np.bincount(owners, minlength=count).max()), math.ceil(mrs_node.WALL_CELL_CAP * len(owners) / count))


def traced_step(node: MRSDragNode, pose: np.ndarray) -> dict:
    """The node's step traced afresh, so that a planted stop is read, with its ``params`` arguments of the program as the
    compiled step's are: eagerly, a scan's body takes the wall's table as a constant of its program, some gigabytes in
    a compile on CI, whose persistent cache serialises it too."""
    step = jax.jit(lambda params, state, at: node.fresh_update(state, {"pose": at}, 1e-3, params=params))
    return step(node.params_pytree(), node.initial_state(), jnp.asarray(pose))


def rebuilt(node: MRSDragNode, pose: np.ndarray, most: int | None = None) -> np.ndarray:
    """The node's step with a wall written again from its pieces, the image summed cell by cell with no chunks: the
    held free matrix turned, plus the tube's correction at the force points, each column times its cell's area, and for
    each of the cells nearest the wall within the node's budget its column moved towards the tube's kernel summed over
    the cell's wall points under the cap (:func:`capped_cell`, ``most`` to a cell, the node's where not given; each
    weight scaled so the cell's sum to its area; every cell where the node integrates them all) by the smoothstep of
    its force point's distance from the wall in force spacings, one within half the reach and zero beyond it; then
    ``W^T A^-1 M``, its symmetric part (the review's s5 rebuild, here of the image integrated near the wall)."""
    from mime.nodes.hydro.mrs import HELD
    from mime.nodes.hydro.mrs._quadrature import separated_resistance, turned
    from mime.nodes.hydro.mrs._tube import tube_correction, tube_distance, tube_image

    theta = {key: jnp.asarray(node.constants()[key]) for key in node.body_leaf_names()}
    body = node.body(theta)
    turn = quaternion.rotation_matrix(jnp.asarray(pose[3:]))
    force = jnp.asarray(pose[:3]) + body.force @ turn.T
    centroid = jnp.asarray(pose[:3]) + body.centroid @ turn.T
    counts, owners = node._wall_quadrature
    cells = node._quadrature_samples({**theta, "viscosity": theta["blob"]}, counts)
    points = jnp.asarray(pose[:3]) + cells.points @ turn.T
    weights, area = np.asarray(cells.weights), np.asarray(body.area)
    parameters = wall_parameters(node)
    count = node.sample_count
    image = np.asarray(tube_correction(force, parameters, 1e-3).matrix) * np.repeat(area, 3)[None, :]
    gap = np.asarray(tube_distance(force, parameters)) / node.constants()["force_spacing"]
    reach = mrs_node.WALL_REACH
    widest = wall_cap(owners, count) if most is None else most
    kernel = jax.jit(lambda targets, sources, wall: tube_image(targets, sources, wall, 1e-3).matrix)
    chosen = range(count) if node.wall_every else np.argsort(gap, kind="stable")[:node.wall_budget]
    for cell in chosen:
        t = min(max((reach - gap[cell]) / (0.5 * reach), 0.0), 1.0)
        blend = t * t * (3.0 - 2.0 * t)
        if blend == 0.0:
            continue
        mine, summed = capped_cell(np.flatnonzero(owners == cell), weights, widest)
        scaled = np.zeros(widest)
        scaled[:len(mine)] = summed * area[cell] / summed.sum()
        sources = jnp.asarray(np.asarray(points)[np.concatenate([mine, np.full(widest - len(mine), mine[0])])])
        integrated = np.einsum("tsij,s->tij", np.asarray(kernel(force, sources, parameters)), scaled).reshape(3 * count, 3)
        columns = slice(3 * cell, 3 * cell + 3)
        image[:, columns] += blend * (integrated - image[:, columns])
    matrix = turned(jnp.asarray(node.params_pytree()[HELD]), turn) / 1e-3 + jnp.asarray(image)
    found = np.asarray(separated_resistance(matrix, force, body.area, centroid, jnp.asarray(pose[:3])))
    return (found + found.T) / 2.0


def integrated_miss(node: MRSDragNode, pose: np.ndarray, most: int | None = None) -> float:
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(mrs_node, "stops_where", lambda accepted, least: jnp.zeros((), dtype=bool))
        out = traced_step(node, pose)
    found = np.asarray(out["resistance"], dtype=np.float64)
    expected = rebuilt(node, pose, most)
    scale = np.sqrt(np.outer(np.abs(np.diag(expected)), np.abs(np.diag(expected))))
    return float(np.max(np.abs(found - expected) / scale))


def coarse_screw_on_the_floor(monkeypatch) -> tuple[MRSDragNode, np.ndarray]:
    offset = generators_pose(0.2e-3, math.pi / 8)
    law = Drag.MRS(lubrication=False, spacing=mm(1.2), blob=0.5, quadrature_refinement=4, wall_refinement=2, wall_kind="tube")
    return built_unchecked(tube_screw(law, offset, math.pi / 8), monkeypatch), pose_at(offset, math.pi / 8)


@pytest.mark.parametrize("share", [1.0, 0.2], ids=["every cell", "the budget's"])
def test_the_integrated_image_is_its_sum_over_each_cell_written_again(monkeypatch, share):
    """A budget of every cell holds the wall quadrature, and every cell is integrated and blended; at a budget of a
    fifth the cells nearest the wall are chosen each step."""
    monkeypatch.setattr(mrs_node, "WALL_CELL_SHARE", share)
    node, pose = coarse_screw_on_the_floor(monkeypatch)
    assert node.wall_every == (share == 1.0)
    fill = np.bincount(node._wall_quadrature[1], minlength=node.sample_count)
    assert fill.max() > wall_cap(node._wall_quadrature[1], node.sample_count), "the cap thins a cell here"
    miss = integrated_miss(node, pose)
    print(f"\nthe node's integrated image against its sum cell by cell: {miss:.1e} ({node.sample_count} force points, "
          f"{len(node._wall_quadrature[1])} wall points, {len(node.wall_capped[0])} read, at most "
          f"{node.wall_table.shape[1]} to a cell)")
    assert miss < 1e-9


def test_control_the_cells_weights_left_unscaled_to_their_areas_are_caught_by_the_sum(monkeypatch):
    monkeypatch.setattr(mrs_node, "on_the_cells", lambda weights, owners, area: weights)
    node, pose = coarse_screw_on_the_floor(monkeypatch)
    assert integrated_miss(node, pose) > 1e-6


def test_control_the_node_uncapped_misses_the_sum_under_the_cap(monkeypatch):
    """With the cap lifted the node integrates every wall point, and the oracle, summing under the cap it had, misses
    it: the sum sees the cap."""
    monkeypatch.setattr(mrs_node, "WALL_CELL_CAP", 1e6)
    uncapped, pose = coarse_screw_on_the_floor(monkeypatch)
    owners = uncapped._wall_quadrature[1]
    monkeypatch.setattr(mrs_node, "WALL_CELL_CAP", 2.0)
    most = wall_cap(owners, uncapped.sample_count)
    assert uncapped.wall_table.shape[1] > most
    assert integrated_miss(uncapped, pose, most) > 1e-6


def wall_read(node: MRSDragNode, pose: np.ndarray) -> tuple[bool, np.ndarray]:
    """The step's reads accepted and its resistance at ``pose``, the stop removed, so that an indefinite one is read."""
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(mrs_node, "stops_where", lambda accepted, least: jnp.zeros((), dtype=bool))
        out = traced_step(node, pose)
    return bool(out["wall_accepted"]), np.asarray(out["resistance"], dtype=np.float64)


#: Far along the tube from the world's origin, past the table's axial reach of it (a read there was refused at 15 mm).
ALONG_THE_TUBE = 30e-3


@pytest.mark.parametrize("share", [1.0, 0.2], ids=["every cell", "the budget's"])
def test_the_integrated_image_is_the_same_all_along_the_tube(monkeypatch, share):
    """The tube is the same along its axis, so a body's resistance does not move with it: a swimmer goes far from the
    world's origin, and a chunk's padding must read nothing there."""
    monkeypatch.setattr(mrs_node, "WALL_CELL_SHARE", share)
    node, pose = coarse_screw_on_the_floor(monkeypatch)
    along = pose + np.array((ALONG_THE_TUBE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    (here_accepted, here), (there_accepted, there) = wall_read(node, pose), wall_read(node, along)
    assert here_accepted and there_accepted
    scale = np.sqrt(np.outer(np.abs(np.diag(here)), np.abs(np.diag(here))))
    print(f"\nat the origin against {ALONG_THE_TUBE * 1e3} mm along the tube: {np.max(np.abs(there - here) / scale):.1e}")
    assert np.max(np.abs(there - here) / scale) < 1e-9


def test_control_a_chunks_padding_at_the_worlds_origin_refuses_the_body_along_the_tube(monkeypatch):
    def at_the_origin(points, weights):
        return (jnp.concatenate([points, jnp.zeros((1, 3), dtype=points.dtype)]),
                jnp.concatenate([weights, jnp.zeros((1,), dtype=weights.dtype)]))

    monkeypatch.setattr(_quadrature, "padded", at_the_origin)
    monkeypatch.setattr(mrs_node, "padded", at_the_origin)
    node, pose = coarse_screw_on_the_floor(monkeypatch)
    along = pose + np.array((ALONG_THE_TUBE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert wall_read(node, pose)[0], "accepted at the origin"
    assert not wall_read(node, along)[0], "and refused along the tube, for its padding alone"


def cap_faults(owners: np.ndarray, weights: np.ndarray, cells: int, most: int, kept: np.ndarray,
               runs: np.ndarray) -> list[str]:
    """Which of the cap's properties ``(kept, runs)`` breaks on the quadrature ``owners``, ``weights``: each cell keeps
    ``min(count, most)`` points; every point adds to a kept point of its own cell, each kept point standing for its own
    run; the weights summed as the step sums them (``segment_sum`` over ``runs``, an index past ``kept`` discarded) are
    each cell's own; a cell's runs lie together in the quadrature's order, of nearly equal count, each kept point its
    run's middle; a cell within the cap keeps every point."""
    counts = np.bincount(owners, minlength=cells)
    faults: list[str] = []
    if not np.array_equal(np.bincount(owners[kept], minlength=cells), np.minimum(counts, most)):
        faults.append("a cell keeps other than min(count, most) points")
    inside = (runs >= 0) & (runs < len(kept))
    if not (inside.all() and np.array_equal(owners[kept[runs]], owners)):
        faults.append("a point adds to no kept point, or to another cell's")
    if not np.array_equal(runs[kept], np.arange(len(kept))):
        faults.append("a kept point stands for another's run")
    summed = np.bincount(runs[inside], weights[inside], minlength=len(kept))
    if not np.allclose(np.bincount(owners[kept], summed, minlength=cells), np.bincount(owners, weights, minlength=cells),
                       rtol=1e-12, atol=0.0):
        faults.append("a cell's weight is not its own")
    for cell in range(cells):
        mine = np.flatnonzero(owners == cell)
        theirs = runs[mine]
        if np.any(np.diff(theirs) < 0) or not np.all(inside[mine]):
            faults.append("a cell's runs do not lie together in the quadrature's order")
            continue
        found = np.unique(theirs)
        sizes = np.array([np.count_nonzero(theirs == run) for run in found])
        if sizes.max() - sizes.min() > 1:
            faults.append("a cell's runs are not of nearly equal count")
        if counts[cell] <= most and len(found) != counts[cell]:
            faults.append("a cell within the cap does not keep every point")
        if any(kept[run] != members[(len(members) - 1) // 2] for run in found
               for members in (mine[theirs == run],)):
            faults.append("a kept point is not its run's middle")
    return list(dict.fromkeys(faults))


def drawn_quadrature(counts: list[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A quadrature of ``len(counts)`` cells with those counts, its points in a drawn order, and drawn weights."""
    rng = np.random.default_rng(seed)
    owners = rng.permutation(np.repeat(np.arange(len(counts)), counts))
    return owners, rng.uniform(1e-3, 1.0, len(owners))


@settings(deadline=None)
@given(counts=st.lists(st.integers(1, 120), min_size=1, max_size=50), most=st.integers(1, 40),
       seed=st.integers(0, 2 ** 32 - 1))
def test_a_cell_under_the_cap_keeps_its_weight_on_at_most_the_caps_points(counts, most, seed):
    owners, weights = drawn_quadrature(counts, seed)
    faults = cap_faults(owners, weights, len(counts), most, *capped_cells(owners, len(counts), most))
    assert not faults, faults


def first_of_each_run(owners: np.ndarray, cells: int, most: int) -> tuple[np.ndarray, np.ndarray]:
    """The cap planted with each run's first point kept, not its middle; its runs and weights the real cap's."""
    kept, runs = capped_cells(owners, cells, most)
    order = np.argsort(runs, kind="stable")
    return order[np.searchsorted(runs[order], np.arange(len(kept)))].astype(np.int32), runs


def wide_cells_thinned(owners: np.ndarray, cells: int, most: int) -> tuple[np.ndarray, np.ndarray]:
    """The cap planted with a wide cell's points beyond those it keeps dropped, as a stride through them would: their
    runs point past ``kept``, which the step's sum discards, so the cell's weight falls short of its own."""
    kept, runs = capped_cells(owners, cells, most)
    dropped = np.ones(len(owners), dtype=bool)
    dropped[kept] = False
    return kept, np.where(dropped, len(kept), runs).astype(np.int32)


@st.composite
def with_a_wide_cell(draw: st.DrawFn) -> tuple[list[int], int]:
    """The property test's cells and cap, with one cell at least three times the cap inserted among them: its runs
    hold three points or more, so a run's first point is not its middle and the cap drops points there."""
    most = draw(st.integers(1, 40))
    counts = draw(st.lists(st.integers(1, 120), min_size=0, max_size=49))
    counts.insert(draw(st.integers(0, len(counts))), draw(st.integers(3 * most, 3 * most + 120)))
    return counts, most


@settings(deadline=None)
@given(drawn=with_a_wide_cell(), seed=st.integers(0, 2 ** 32 - 1))
def test_control_a_cap_that_keeps_a_runs_first_point_or_drops_points_fails_the_property_on_every_draw(drawn, seed):
    """Two defects planted in the cap, each caught on every draw that has a cell three times wider than the cap (the
    real cap passes the same draws): each run's first point kept rather than its middle, which conserves every weight;
    and a wide cell's points beyond those it keeps dropped, as a stride through them would."""
    counts, most = drawn
    owners, weights = drawn_quadrature(counts, seed)
    assert not cap_faults(owners, weights, len(counts), most, *capped_cells(owners, len(counts), most))
    first = cap_faults(owners, weights, len(counts), most, *first_of_each_run(owners, len(counts), most))
    thinned = cap_faults(owners, weights, len(counts), most, *wide_cells_thinned(owners, len(counts), most))
    assert first == ["a kept point is not its run's middle"], first
    assert "a cell's weight is not its own" in thinned, thinned


def default_spacing(experiment: Experiment) -> float:
    taken = {d.path: d.value for d in experiment.plan().profile.defaults}
    return float(taken["drag:body.spacing"].value) * 1e3


def held_off(margin_mm: float | None, wall: bool = True, wall_refinement: int | None = None,
             spacing_mm: float | None = None) -> Experiment:
    """FL-9 in the tube (or in free space, with no wall), a contact holding it ``margin_mm`` off the tube's inner
    surface (none where ``None``), and ``Drag.MRS()`` with every numerical setting its default but
    ``wall_refinement`` and ``spacing_mm``, where given."""
    from mime.interactions import Contact

    e = Experiment("held_off", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    e.add_structure(Structure("tube", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(3.175), wall=mm(0.5),
                                                    length=mm(100.0))))
    e.add_medium(Medium("water", fills="tube.lumen") if wall else Medium("water"))
    e.declare("water.viscosity", mPa_s(1.0), reason="water")
    e.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(fl9())}, initial=State(
        position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    e.declare("body.mass", kg(1e-5), reason="a millimetre body's order")
    e.declare("body.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    e.attach(Drag.MRS(lubrication=False, wall_kind="tube" if wall else "free_space", wall_refinement=wall_refinement,
                      spacing=mm(spacing_mm) if spacing_mm is not None else None), body="body", medium="water")
    if margin_mm is not None:
        e.attach(Contact.Projection(margin=mm(margin_mm)), body="body", surface="tube.inner_surface")
    return e


def test_a_confined_body_held_off_the_wall_takes_the_force_spacing_its_closest_approach_needs():
    family = default_spacing(held_off(None))
    found = default_spacing(held_off(0.2))
    print(f"\nthe family's {family} mm; held 0.2 mm off the wall {found} mm")
    assert found <= 0.2 / mrs_node.CONFINED_GAP < family
    assert found == pytest.approx(0.2 / mrs_node.CONFINED_GAP, rel=1e-2)
    taken = {d.path for d in held_off(0.2).plan().profile.defaults}
    assert "drag:body.wall_refinement" not in taken, "the image stays at the force points unless one is written"


def test_the_integrated_image_when_written_takes_its_own_gap():
    found = default_spacing(held_off(0.1, wall_refinement=2))
    print(f"\nheld 0.1 mm off the wall with wall_refinement 2: {found} mm")
    assert found <= 0.1 / mrs_node.INTEGRATED_GAP
    assert found == pytest.approx(0.1 / mrs_node.INTEGRATED_GAP, rel=1e-2)


def test_at_its_gap_the_point_image_is_definite_at_every_roll_where_the_floor_of_that_spacing_is_not(monkeypatch):
    """FL-9 at the default's 0.829 mm with the image at the force points: every roll of eight definite with its nearest
    surface point :data:`CONFINED_GAP` force spacings from the wall, and the same node on the floor, 0.1 mm from it,
    indefinite at some roll (the control, the gap's whole effect: a resistance the gap makes definite)."""
    node = built_unchecked(tube_screw(Drag.MRS(lubrication=False, spacing=mm(0.829), wall_kind="tube"), generators_pose(0.5e-3)),
                           monkeypatch)
    assert node.wall_table is None, "the image at the force points"

    def least(gap: float) -> list[float]:
        rolls = [k * math.pi / 8 for k in range(8)]
        judged = [node.judged_at(pose_at(generators_pose(gap, roll), roll)) for roll in rolls]
        assert all(one.accepted for one in judged)
        return [one.least for one in judged]

    at_its_gap, floored_there = least(mrs_node.CONFINED_GAP * 0.829e-3), least(0.1e-3)
    print(f"\nat {mrs_node.CONFINED_GAP} spacings the least eigenvalues {np.round(at_its_gap, 2)}; on the floor "
          f"{np.round(floored_there, 2)}")
    assert min(at_its_gap) > 0.0
    assert min(floored_there) < 0.0


def test_control_in_free_space_or_held_far_off_the_body_keeps_its_familys_force_spacing():
    family = default_spacing(held_off(None))
    assert default_spacing(held_off(0.2, wall=False)) == family
    assert default_spacing(held_off(1.0)) == family


#: The margins the merged contact's default plan is read at: each one's force spacing below the family's 0.829 mm, and
#: planned only (the 0.1 mm margin's 0.181 mm would build 4,816 force points).
MARGINS_MM = (0.3, 0.2)


def default_plan_misses() -> list[str]:
    """Each margin at which the drag's default force spacing beside ``Contact.Projection(margin=...)`` does not come
    from the margin: the margin over :data:`CONFINED_GAP` to the rule's three figures and never coarser, so that the
    body's closest approach is CONFINED_GAP force spacings or more, where the family's spacing would leave it nearer."""
    family = default_spacing(held_off(None))
    found = []
    for margin in MARGINS_MM:
        spacing = default_spacing(held_off(margin))
        if not (spacing <= margin / mrs_node.CONFINED_GAP < family
                and spacing == pytest.approx(margin / mrs_node.CONFINED_GAP, rel=1e-2)):
            found.append(f"held {margin} mm off the wall: {spacing} mm, {margin / spacing:.3g} force spacings off it")
    return found


def test_the_contact_that_holds_the_hull_declares_its_margin_and_the_default_plan_comes_from_it():
    """The contact 0.5.0 (the body's hull points held, its orientation read, since 0.2.0) names its margin as how far it
    holds the body off the surface (``Interaction.holds_off``), so the planner's ``PlanContext.approach`` reads it and
    the MRS rung's default force spacing keeps FL-9 CONFINED_GAP (0.55) force spacings or more off the tube's wall."""
    from mime.interactions import Contact

    assert (Contact.version, Contact.holds_off) == ("0.5.0", "margin")
    assert default_plan_misses() == []


def test_control_without_the_declaration_the_plan_falls_back_to_the_familys_force_spacing(monkeypatch):
    """The declaration taken off the contact: the planner finds no closest approach, so the drag takes its family's
    0.829 mm at every margin, as it did before the contact declared it, and the test above fails on every run (the
    plan draws nothing)."""
    from mime.interactions import Contact

    family = default_spacing(held_off(None))
    monkeypatch.setattr(Contact, "holds_off", None)
    assert [default_spacing(held_off(margin)) for margin in MARGINS_MM] == [family] * len(MARGINS_MM)
    assert len(default_plan_misses()) == len(MARGINS_MM)


@pytest.mark.parametrize(("wall_refinement", "spacing_mm"), [(None, None), (2, None), (None, 0.181)],
                         ids=["the default", "the image integrated", "a spacing written"])
def test_a_body_held_on_the_wall_is_refused_by_name_since_no_force_spacing_clears_it(wall_refinement, spacing_mm):
    """A contact's margin of 0 holds FL-9 on the tube's wall, so its closest approach is the wall itself, within every
    share of every force spacing: the plan refuses it by name (``drag.mrs-on-wall``) and alone, whatever the spacing
    and the image, and its action is a margin above 0; the spacing's rule takes none there, not the family's. Before
    the second fix round the rule took the family's and the build died on an uncoded ``ZeroDivisionError``."""
    with pytest.raises(BuildRefused) as refused:
        held_off(0.0, wall_refinement=wall_refinement, spacing_mm=spacing_mm).plan()
    assert refused.value.codes() == ["drag.mrs-on-wall"]
    (refusal,) = refused.value.refusals
    assert "margin of 0" in refusal.message and "margin above 0" in refusal.action
    assert "drag:body.spacing" not in {d.path for d in refused.value.report.profile.defaults}


def test_control_the_refusal_lifted_a_body_held_on_the_wall_plans_silently_on_its_familys_spacing(monkeypatch):
    """The second fix round's change planted out: 0.3.0's plan problems in drag 0.4.0's place, and the spacing's rule
    taking the family's at a closest approach of 0, as before it. The body held on the wall then plans, on the family's
    0.829 mm, with nothing said: what the test above refuses."""
    from mime.interactions.hydrodynamic import PointForceDrag

    family = default_spacing(held_off(None))
    rule = Drag.default_spacing
    monkeypatch.setattr(Drag, "_mrs_problems", PointForceDrag._mrs_problems)
    monkeypatch.setattr(Drag, "default_spacing", lambda self, ctx: PointForceDrag.default_spacing(self, ctx)
                        if ctx.approach(of="body") == 0.0 else rule(self, ctx))
    assert default_spacing(held_off(0.0)) == family


def test_a_report_given_a_closest_approach_of_0_says_no_force_spacing_clears_it(monkeypatch):
    """The build report's own reading of the closest approach (the dynamics' confinements, not the planner's) at 0:
    a line that no force spacing clears it, never a division by it."""
    node = built_unchecked(tube_screw(Drag.MRS(lubrication=False, spacing=mm(0.829), wall_kind="tube")), monkeypatch)
    lines = node.built_lines({"pose": pose_at(0.0)}, 0.0)
    assert any("holds the body on the wall" in line and "no force spacing clears" in line for line in lines)



@pytest.mark.parametrize(("written", "taken"), [(2, 3), (4, 4)], ids=["promoted", "as written"])
def test_the_build_report_states_the_wall_refinement_taken_and_says_where_it_is_not_the_one_written(monkeypatch,
                                                                                                     written, taken):
    """FL-9's grid at 1.2 mm leaves a cell with no wall point at wall_refinement 2 (as at the default's 0.829 mm), so
    the build takes the next finer that leaves none, 3, and the report's cost line says both; at 4, taken as written,
    it names 4 alone (the control: no promotion stated where there is none)."""
    law = Drag.MRS(lubrication=False, spacing=mm(1.2), blob=0.5, quadrature_refinement=4, wall_kind="tube", wall_refinement=written)
    text = built_unchecked(tube_screw(law), monkeypatch)._wall_text()
    assert f"a wall quadrature {taken} times finer than the force spacing" in text
    assert (f"wall_refinement={written} written" in text) == (written != taken)
