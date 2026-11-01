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

Float64 (``tests/oracles/conftest.py``); measured in float32 too, where the run stops at the same step."""

import dataclasses
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mrs_cases import drag_node, free_space, resistance_at
from script_a import fl9

import mime.nodes.hydro.mrs._node as mrs_node
from mime import Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused, RunFailed
from mime.core import quaternion
from mime.dynamics import RigidInertial
from mime.interactions import Drag
from mime.nodes.hydro.mrs import MRSDragNode
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


def floored(monkeypatch, start_mm=(0.0, 0.0, -1.0), wall_refinement: int | None = 1) -> Experiment:
    """Script A's experiment on the MRS rung (``operating_point.experiment(rung="bem")``) with its laws and kind
    unpinned, so that it is drag 0.4.0 at the screw's defaults at 0.829 mm (222 force points, blob 0.25, the
    quadrature eight times finer), FL-9 resting on the floor at the contact's 0.1 mm margin; the review's case, with the
    wall's image at the force points (``wall_refinement`` 1) unless another is given."""
    monkeypatch.setattr(OP, "pinned", lambda geometry: geometry)
    monkeypatch.setattr(OP, "pinned_law", lambda law: dataclasses.replace(law, wall_refinement=wall_refinement)
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
GENERATORS = Drag.MRS(spacing=mm(0.6), blob=0.5, quadrature_refinement=4, wall_refinement=1, wall_kind="tube")


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
    node = built_unchecked(tube_screw(Drag.MRS(spacing=mm(0.829), wall_kind="tube"), offset, math.pi / 8),
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
    node = built_unchecked(tube_screw(Drag.MRS(spacing=mm(0.829), wall_kind="tube"), offset, math.pi / 8),
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
    node = built_unchecked(tube_screw(Drag.MRS(spacing=mm(0.829), wall_kind="tube"), offset), monkeypatch)
    parameters = wall_parameters(node)
    points = jnp.asarray(world_force_points(node, pose_at(offset)))
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


def rebuilt(node: MRSDragNode, pose: np.ndarray) -> np.ndarray:
    """The node's step with a wall written again from its pieces, the image summed cell by cell with no chunks: the
    held free matrix turned, plus for each force point's cell the tube's kernel at every force point from each of the
    cell's wall points, its weight scaled so the cell's weights sum to its area; then ``W^T A^-1 M``, its symmetric
    part (the review's s5 rebuild, here of the integrated image)."""
    from mime.nodes.hydro.mrs import HELD
    from mime.nodes.hydro.mrs._quadrature import separated_resistance, turned
    from mime.nodes.hydro.mrs._tube import tube_image

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
    widest = int(np.bincount(owners).max())
    kernel = jax.jit(lambda sources: tube_image(force, sources, parameters, 1e-3).matrix)
    columns = []
    for cell in range(node.sample_count):
        mine = np.flatnonzero(owners == cell)
        scaled = np.zeros(widest)
        scaled[:len(mine)] = weights[mine] * area[cell] / weights[mine].sum()
        sources = jnp.asarray(np.asarray(points)[np.concatenate([mine, np.full(widest - len(mine), mine[0])])])
        columns.append(np.einsum("tsij,s->tij", np.asarray(kernel(sources)), scaled))
    image = np.stack(columns, axis=1).transpose(0, 2, 1, 3).reshape(3 * node.sample_count, 3 * node.sample_count)
    matrix = turned(jnp.asarray(node.params_pytree()[HELD]), turn) / 1e-3 + jnp.asarray(image)
    found = np.asarray(separated_resistance(matrix, force, body.area, centroid, jnp.asarray(pose[:3])))
    return (found + found.T) / 2.0


def integrated_miss(node: MRSDragNode, pose: np.ndarray) -> float:
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(mrs_node, "stops_where", lambda accepted, least: jnp.zeros((), dtype=bool))
        out = node.fresh_update(node.initial_state(), {"pose": jnp.asarray(pose)}, 1e-3, params=node.params_pytree())
    found = np.asarray(out["resistance"], dtype=np.float64)
    expected = rebuilt(node, pose)
    scale = np.sqrt(np.outer(np.abs(np.diag(expected)), np.abs(np.diag(expected))))
    return float(np.max(np.abs(found - expected) / scale))


def coarse_screw_on_the_floor(monkeypatch) -> tuple[MRSDragNode, np.ndarray]:
    offset = generators_pose(0.2e-3, math.pi / 8)
    law = Drag.MRS(spacing=mm(1.2), blob=0.5, quadrature_refinement=4, wall_refinement=2, wall_kind="tube")
    return built_unchecked(tube_screw(law, offset, math.pi / 8), monkeypatch), pose_at(offset, math.pi / 8)


def test_the_integrated_image_is_its_sum_over_each_cell_written_again(monkeypatch):
    node, pose = coarse_screw_on_the_floor(monkeypatch)
    miss = integrated_miss(node, pose)
    print(f"\nthe node's integrated image against its sum cell by cell: {miss:.1e} ({node.sample_count} force points, "
          f"{len(node._wall_quadrature[1])} wall points)")
    assert miss < 1e-9


def test_control_the_cells_weights_left_unscaled_to_their_areas_are_caught_by_the_sum(monkeypatch):
    monkeypatch.setattr(mrs_node, "on_the_cells", lambda weights, owners, area: weights)
    node, pose = coarse_screw_on_the_floor(monkeypatch)
    assert integrated_miss(node, pose) > 1e-6


def default_spacing(experiment: Experiment) -> float:
    taken = {d.path: d.value for d in experiment.plan().profile.defaults}
    return float(taken["drag:body.spacing"].value) * 1e3


def held_off(margin_mm: float | None, wall: bool = True) -> Experiment:
    """FL-9 in the tube (or in free space, with no wall), a contact holding it ``margin_mm`` off the tube's inner
    surface (none where ``None``), and ``Drag.MRS()`` with every numerical setting its default."""
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
    e.attach(Drag.MRS(wall_kind="tube" if wall else "free_space"), body="body", medium="water")
    if margin_mm is not None:
        e.attach(Contact.Projection(margin=mm(margin_mm)), body="body", surface="tube.inner_surface")
    return e


def test_a_confined_body_held_off_the_wall_takes_the_force_spacing_its_closest_approach_needs():
    family = default_spacing(held_off(None))
    found = default_spacing(held_off(0.2))
    print(f"\nthe family's {family} mm; held 0.2 mm off the wall {found} mm")
    assert found <= 0.2 / mrs_node.CONFINED_GAP < family
    assert found == pytest.approx(0.2 / mrs_node.CONFINED_GAP, rel=1e-2)


def test_control_in_free_space_or_held_far_off_the_body_keeps_its_familys_force_spacing():
    family = default_spacing(held_off(None))
    assert default_spacing(held_off(0.2, wall=False)) == family
    assert default_spacing(held_off(1.0)) == family
