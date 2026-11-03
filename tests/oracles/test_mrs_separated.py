"""The MRS rung's separated quadrature (drag 0.4.0, phase 7m-b, design #195): its oracles, each with a control that
fails on every run, at the rung's defaults unless a case says otherwise.

- **The sphere against the exact regularised drag** (``regularised_sphere``: the continuous regularised problem at the
  node's own blob, an independent computation in numpy): the difference is the discretisation's error alone, the force
  points' constant tractions and the quadrature, which the separated quadrature is built to make small; and against
  Stokes's law, which adds the regularisation's error, the blob's.
- **The prolate spheroid** (semi-axes 2, 1, 1, the ellipsoid kind) against Kim & Karrila's closed form.
- **The confined sphere** on the axis of the tube against Haberman & Sayre, at a fifth and two fifths of the radius.
- **FL-9 in free space** against the shipped helix reference (``references/helix_fl9_free_space.json``), at the rung's
  default for a screw, and the same count of samples on the point-force method of 0.3.0 as the control.
- **The plants**, each caught by the oracle beside it: the map's weights wrong (each cell's area its force point's own
  weight from the sampling, not the sum of its quadrature points'); the quadrature held in the wrong frame (the held
  quantity read as if it were the world's, not turned to the pose); the self-interaction made with stale positions (the
  held value read after a study moved the body's radius). The plants are deterministic, so a plant caught is caught on
  every run.

Every tolerance is a measured value with room (labelled decisions of phase 7m-b, in the pull request), stated beside
it with what it was measured on; float64 throughout (``tests/oracles/conftest.py``)."""

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mrs_cases import PILL, drag_node, free_space, prolate, resistance_at
import separated_oracle
from regularised_sphere import regularised_drag, regularised_rotational_drag
from script_a import fl9
from tube_cases import haberman_sayre

import mime.interactions.hydrodynamic as hydrodynamic
import mime.nodes.hydro.mrs._node as mrs_node
from mime import Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused
from mime.build._maddening import parameter_leaves
from mime.core import quaternion
from mime.dynamics import RigidInertial
from mime.interactions import Drag
from mime.interactions.hydrodynamic import PointForceDrag
from mime.nodes.hydro.mrs import HELD, MRSDragNode
from mime.nodes.hydro.mrs._quadrature import Chunks, separated_matrix
from mime.scene import Ellipsoid, Scene, Sphere, Tube, Z
from mime.things import Part
from mime.units import kg, kg_m2, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

MU = 1e-3
BALL = Sphere(radius=mm(1.0))
SPHEROID = Ellipsoid(semi_axes=mm((2.0, 1.0, 1.0)))
IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
REFERENCE = Path(__file__).parent / "references" / "helix_fl9_free_space.json"
#: The sphere at the defaults against its exact regularised drag: measured 4.2e-7 in translation and -1.8e-4 in
#: rotation (the force points' constant tractions and the quadrature); twice the larger.
DISCRETISATION = 4e-4
#: And against Stokes's law, which adds the regularisation's error: measured +0.50% and +1.45%; the larger with a tenth.
REGULARISED = 0.016
#: The prolate spheroid's diagonal against its closed form at the compact defaults: measured +0.50% to +1.68%.
SPHEROID_MISS = 0.02
#: The confined sphere against Haberman & Sayre: measured +0.80% at a fifth of the radius and +1.72% at two fifths.
CONFINED = {0.2: 0.01, 0.4: 0.02}
#: FL-9 at the screw's default in free space against the shipped reference: measured -2.6%; drag 0.3.0 at the same
#: count misses by more than three times this.
SCREW_MISS = 0.035
#: The held value turned to a pose against a node built with its part so turned: rounding (measured 2e-15).
FRAME = 1e-11
#: The node against its formula written again in NumPy: rounding.
FORMULA = 1e-9


def free(kind, law=None, name="separated"):
    return free_space(name, {"body": kind}, law if law is not None else Drag.MRS(wall_kind="free_space"))


def eps_of(node: MRSDragNode, leaves) -> float:
    held = node.body({key: leaves[key] for key in node.body_leaf_names()})
    return float(np.sqrt(held.eps_squared))


def sphere_misses(built) -> tuple[np.ndarray, np.ndarray]:
    """The sphere's drag and rotational drag over Stokes's law, and over the exact regularised values at its blob."""
    r = resistance_at(built, "body", IDENTITY)
    node = drag_node(built, "body")
    eps = eps_of(node, parameter_leaves(built.graph)["drag:body"]) / 1e-3
    stokes = np.array([np.mean(np.diag(r)[:3]) / (6 * math.pi * MU * 1e-3),
                       np.mean(np.diag(r)[3:]) / (8 * math.pi * MU * 1e-9)])
    exact = np.array([regularised_drag(eps), regularised_rotational_drag(eps)])
    return stokes - 1, stokes / exact - 1


@pytest.fixture(scope="module")
def ball():
    return free(BALL).build()


def test_the_rungs_defaults_resolve_by_family():
    """The sphere's 338 force points at a tenth, its quadrature twenty times finer; FL-9's 222 at a quarter, eight
    times finer; and 0.3.0 kept, on its point forces."""
    sphere = drag_node(free(BALL).build(), "body")
    screw = drag_node(free(fl9()).build(), "body")
    kept = drag_node(free(BALL, PointForceDrag.MRS(spacing=mm(0.25), blob=0.5, wall_kind="free_space")).build(), "body")
    found = [(n.sample_count, n.constants()["blob"], n.separated) for n in (sphere, screw, kept)]
    print(f"\n(force points, blob, separated): {found}; quadratures {sphere.quadrature_count}, {screw.quadrature_count}")
    assert found == [(338, 0.1, True), (222, 0.25, True), (338, 0.5, False)]


def test_the_sphere_meets_the_exact_regularised_drag_and_stokes_law_within_its_blob(ball):
    stokes, discretisation = sphere_misses(ball)
    print(f"\nagainst Stokes {np.round(stokes, 6)}, against the exact regularised problem {np.round(discretisation, 7)}")
    assert np.all(np.abs(discretisation) < DISCRETISATION)
    assert np.all((0.0 < stokes) & (stokes < REGULARISED))


#: The pill's spacing for the formula's check: its force points few enough for NumPy's pairs.
PILL_SPACING = 0.4e-3
PILL_SETTINGS = {"radius": 5e-4, "height": 2e-3, "axis": (0.0, 1.0, 0.0)}


def formula_miss(built) -> float:
    """How far the node's resistance is from its formula written again in NumPy (``separated_oracle``), the pill at a
    pose moved and turned, each entry on the scale of its row's and column's diagonal."""
    node = drag_node(built, "body")
    blob = node.constants()["blob"]
    refinement = math.ceil(2.0 / blob - 1e-9)
    pose = (1e-3, -2e-3, 0.5e-3, *TURNED)
    found = resistance_at(built, "body", pose)
    expected = separated_oracle.resistance(
        *separated_oracle.discretisation(PILL, PILL_SETTINGS, PILL_SPACING, refinement, tuple(PILL.patches)),
        blob, MU, pose)
    scale = np.sqrt(np.outer(np.abs(np.diag(expected)), np.abs(np.diag(expected))))
    return float(np.max(np.abs(found - expected) / scale))


def pill_free():
    return free(PILL, Drag.MRS(spacing=mm(PILL_SPACING * 1e3), wall_kind="free_space"))


def test_the_node_is_its_formula_written_again():
    miss = formula_miss(pill_free().build())
    print(f"\nthe node against its formula in NumPy: {miss:.1e}")
    assert miss < FORMULA


def pill_wall(built):
    """The tube's correction as the node holds it, as a function of the force points in the world."""
    from mime.nodes.hydro.mrs import WALL
    from mime.nodes.hydro.mrs._tube import TUBE_CORRECTION

    node = drag_node(built, "body")
    held = node._step_constants(parameter_leaves(built.graph)["drag:body"])
    parameters = {name: held[f"{WALL}{name}"] for name in TUBE_CORRECTION.parameters}
    return lambda x: np.asarray(TUBE_CORRECTION.correction(jnp.asarray(x), parameters, MU).matrix)


def wall_formula_miss(built, with_wall: bool = True) -> float:
    """``formula_miss`` in the tube: the pill moved off the axis and turned, the wall's image at its force points in
    the formula, each column times its cell's area."""
    node = drag_node(built, "body")
    blob = node.constants()["blob"]
    refinement = math.ceil(2.0 / blob - 1e-9)
    pose = (0.6e-3, -0.3e-3, 0.5e-3, *TURNED)
    found = resistance_at(built, "body", pose)
    expected = separated_oracle.resistance(
        *separated_oracle.discretisation(PILL, PILL_SETTINGS, PILL_SPACING, refinement, tuple(PILL.patches)),
        blob, MU, pose, pill_wall(built) if with_wall else None)
    scale = np.sqrt(np.outer(np.abs(np.diag(expected)), np.abs(np.diag(expected))))
    return float(np.max(np.abs(found - expected) / scale))


def test_the_node_with_a_wall_is_its_formula_written_again_off_the_axis():
    """With the wall's image at the force points (``wall_refinement`` 1); the image integrated over each cell has its
    own rebuild (``test_mrs_near_wall.py``)."""
    miss = wall_formula_miss(pill_in_a_tube(wall_refinement=1).build())
    print(f"\nthe node in the tube, off its axis, against its formula in NumPy: {miss:.1e}")
    assert miss < FORMULA


def test_control_the_formula_without_the_wall_misses_the_node_in_the_tube():
    miss = wall_formula_miss(pill_in_a_tube(wall_refinement=1).build(), with_wall=False)
    print(f"\nthe formula without the wall: {miss:.1e}")
    assert miss > 1e6 * FORMULA


def test_control_the_maps_weights_made_uniform_are_caught_by_the_formula(monkeypatch):
    """Each quadrature point's weight replaced by their mean in the node (the formula keeps the kind's)."""
    real = MRSDragNode._quadrature_samples

    def uniform(self, p, counts=None):
        held = real(self, p, counts)
        return dataclasses.replace(held, weights=jnp.full_like(held.weights, jnp.mean(held.weights)))

    monkeypatch.setattr(MRSDragNode, "_quadrature_samples", uniform)
    miss = formula_miss(pill_free().build())
    print(f"\nthe map's weights uniform: {miss:.1e}")
    assert miss > 1e6 * FORMULA


def test_control_each_quadrature_point_in_its_second_nearest_force_points_cell_is_caught_by_the_formula(monkeypatch):
    """The map's partition wrong: each quadrature point in the cell of the force point second nearest it. (Giving every
    cell to the next force point by index would be no control: it relabels the unknowns, and ``W^T A^-1 M`` is
    unchanged by a relabelling, so only the partition the map makes is the discretisation.)"""
    from scipy.spatial import cKDTree

    from mime.scene._sampling import quadrature_arrays, sampled_points

    def second(kind, settings, sampling, counts):
        force = sampled_points(kind, settings, sampling)
        points, _, _ = quadrature_arrays(kind, settings, sampling.patches, counts)
        owners = np.asarray(cKDTree(force).query(points, k=2)[1][:, 1], dtype=np.int32)
        for empty in np.flatnonzero(np.bincount(owners, minlength=len(force)) == 0):  # none left empty: a partition
            owners[int(np.argmin(np.linalg.norm(points - force[empty], axis=1)))] = empty
        return owners

    monkeypatch.setattr(hydrodynamic, "quadrature_owners", second)
    with pytest.raises(BuildRefused) as refused:
        pill_free().build()
    assert refused.value.codes() == ["drag.mrs-resistance"], "the build's asymmetry check catches it first"
    monkeypatch.setattr(MRSDragNode, "built_problems", lambda self, start=None: [])
    miss = formula_miss(pill_free().build())
    print(f"\nthe map's partition by the second nearest force point, the build's check bypassed: {miss:.1e}")
    assert miss > 1e6 * FORMULA


def test_the_map_puts_each_quadrature_point_in_its_nearest_force_points_cell(ball):
    """Against a brute-force nearest search: each point's force point is at its least distance (to the tie rule's
    rounding), the cells cover the quadrature, and none is empty."""
    node = drag_node(ball, "body")
    leaves = parameter_leaves(ball.graph)["drag:body"]
    held = node.body({key: leaves[key] for key in node.body_leaf_names()})
    force, points = np.asarray(held.force), np.asarray(held.points)
    owners = np.asarray(node.constants()["quadrature_owners"])
    for start in range(0, len(points), 4096):
        distance = np.linalg.norm(points[start:start + 4096, None, :] - force[None, :, :], axis=-1)
        own = distance[np.arange(len(distance)), owners[start:start + 4096]]
        assert np.all(own <= distance.min(axis=1) * (1 + 1e-9)), start
    assert np.array_equal(np.unique(owners), np.arange(len(force)))


def test_the_prolate_spheroid_meets_its_closed_form():
    r = resistance_at(free(SPHEROID).build(), "body", IDENTITY)
    miss = np.diag(r) / prolate(2e-3, 1e-3, MU) - 1
    print(f"\nthe spheroid against the closed form: {np.round(miss, 5)}")
    assert np.all(np.abs(miss) < SPHEROID_MISS)


def tube_ball(law, ratio: float, radius_mm: float = 1.0) -> Experiment:
    e = Experiment("confined", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    e.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(radius_mm), wall=mm(0.1),
                                                    length=mm(40.0))))
    e.add_medium(Medium("water", fills="tube.lumen"))
    e.declare("water.viscosity", mPa_s(1.0), reason="water")
    e.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(Sphere(radius=mm(ratio * radius_mm)))},
                    initial=State(position=mm((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                  velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    e.declare("body.mass", kg(1e-7), reason="a millimetre body's order")
    e.declare("body.inertia", kg_m2((1e-14, 1e-14, 1e-14)), reason="a millimetre body's order")
    e.attach(law, body="body", medium="water")
    return e


@pytest.mark.parametrize("ratio", [0.2, 0.4])
def test_the_confined_sphere_drags_as_haberman_and_sayre_say(ratio):
    r = resistance_at(tube_ball(Drag.MRS(wall_kind="tube"), ratio).build(), "body", IDENTITY)
    factor = r[2, 2] / (6 * math.pi * MU * ratio * 1e-3)
    miss = factor / haberman_sayre(ratio) - 1
    print(f"\nat {ratio} of the radius: K {factor:.5f} against {haberman_sayre(ratio):.5f}, {miss:+.4%}")
    assert abs(miss) < CONFINED[ratio]


def test_control_free_spaces_correction_in_the_tubes_place_fails_the_confined_sphere(monkeypatch):
    from mime.nodes.hydro.mrs import UNBOUNDED, _walls
    from mime.nodes.hydro.mrs._tube import TUBE_CORRECTION

    planted = dataclasses.replace(TUBE_CORRECTION, correction=lambda points, parameters, viscosity:
                                  UNBOUNDED.correction(points, {}, viscosity))
    monkeypatch.setitem(_walls._corrections, (TUBE_CORRECTION.name, TUBE_CORRECTION.version), planted)
    r = resistance_at(tube_ball(Drag.MRS(wall_kind="tube"), 0.2).build(), "body", IDENTITY)
    miss = r[2, 2] / (6 * math.pi * MU * 0.2e-3) / haberman_sayre(0.2) - 1
    assert abs(miss) > 10 * CONFINED[0.2]


def axial_speed(built) -> float:
    r = resistance_at(built, "body", IDENTITY)
    return float(-r[0, 3] * 2.0 * math.pi * 10.0 / r[0, 0])


def test_fl9_in_free_space_at_the_default_comes_within_its_stated_miss_of_the_helix_reference_and_0_3_does_not():
    point = json.loads(REFERENCE.read_text(encoding="utf-8"))["reference"]["speed_m_per_s"]
    built = free(fl9()).build()
    count = drag_node(built, "body").sample_count
    separated = axial_speed(built) / point - 1
    kept_built = free(fl9(), PointForceDrag.MRS(wall_kind="free_space")).build()
    assert drag_node(kept_built, "body").sample_count == count
    kept = axial_speed(kept_built) / point - 1
    print(f"\nFL-9 at {count} force points: {separated:+.3%}; 0.3.0 at the same count: {kept:+.3%}")
    assert abs(separated) < SCREW_MISS
    assert abs(kept) > 3 * SCREW_MISS


# -- the frame and the held value -----------------------------------------------------------------------------------

TURNED = (math.cos(0.4), 0.3 * math.sin(0.4), 0.5 * math.sin(0.4), math.sqrt(0.66) * math.sin(0.4))


def fresh(node: MRSDragNode, rotation: np.ndarray) -> MRSDragNode:
    """``node`` constructed again with its part turned by ``rotation`` (its owner map is the kind's, and stays)."""
    held = dict(node.constants())
    part = np.asarray(held.pop("part_rotation")).reshape(3, 3)
    centre = np.asarray(held.pop("part_centre"))
    return MRSDragNode(node.name, node.delta_t, part_rotation=tuple((rotation @ part).ravel()),
                       part_centre=tuple(rotation @ centre), **held)


def held_at(node: MRSDragNode, pose) -> np.ndarray:
    """The node's resistance at ``pose`` through its held value, its step traced afresh (so that a plant is seen)."""
    out = node.fresh_update(node.initial_state(), {"pose": jnp.asarray(pose, dtype=jnp.float64)}, 1e-3,
                            params=node.params_pytree())
    return np.asarray(out["resistance"], dtype=np.float64)


def frame_miss(built) -> float:
    """How far the resistance at a turned pose is from that of the node built with its part turned so, at rest:
    the held value must be the body frame's and turned to the pose, with or without a wall."""
    node = drag_node(built, "body")
    rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(TURNED)), dtype=np.float64)
    position = np.asarray(node_pose(built)[:3])
    at_pose = held_at(node, (*position, *TURNED))
    built_turned = held_at(fresh(node, rotation), (*position, 1.0, 0.0, 0.0, 0.0))
    scale = np.sqrt(np.outer(np.abs(np.diag(built_turned)), np.abs(np.diag(built_turned))))
    return float(np.max(np.abs(at_pose - built_turned) / scale))


def node_pose(built) -> np.ndarray:
    from mime.build._maddening import node_state
    return np.asarray(node_state(built.graph, "body")["pose"], dtype=np.float64)


def pill_in_a_tube(wall_refinement: int | None = None) -> Experiment:
    e = Experiment("confined_pill", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    e.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(3.0), wall=mm(0.1),
                                                    length=mm(40.0))))
    e.add_medium(Medium("water", fills="tube.lumen"))
    e.declare("water.viscosity", mPa_s(1.0), reason="water")
    e.add_body(Body("body", dynamics=RigidInertial(), parts={"shell": Part(PILL)},
                    initial=State(position=mm((0.6, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                  velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    e.declare("body.mass", kg(1e-7), reason="a millimetre body's order")
    e.declare("body.inertia", kg_m2((1e-14, 1e-14, 1e-14)), reason="a millimetre body's order")
    e.attach(Drag.MRS(spacing=mm(PILL_SPACING * 1e3), wall_kind="tube", wall_refinement=wall_refinement), body="body",
             medium="water")
    return e


@pytest.fixture(scope="module")
def pills():
    return {"free space": free(PILL, Drag.MRS(spacing=mm(0.4), wall_kind="free_space")).build(),
            "the tube": pill_in_a_tube().build()}


@pytest.mark.parametrize("where", ["free space", "the tube"])
def test_the_held_value_is_the_body_frames_and_turns_with_the_body(pills, where):
    miss = frame_miss(pills[where])
    print(f"\nin {where}: {miss:.2e}")
    assert miss < FRAME


@pytest.mark.parametrize("where", ["free space", "the tube"])
def test_control_the_held_value_read_in_the_wrong_frame_is_caught(pills, where, monkeypatch):
    monkeypatch.setattr(mrs_node, "turned", lambda matrix, rotation: matrix)
    monkeypatch.setattr(mrs_node, "turned_resistance", lambda resistance, rotation: resistance)
    miss = frame_miss(pills[where])
    print(f"\nin {where}, the held value not turned: {miss:.2e}")
    assert not miss < 1e3 * FRAME, "NaN where the plant leaves the confined resistance indefinite, which the step stops"


def moved_radius(built, factor: float, *, with_held: bool = True) -> np.ndarray:
    node = drag_node(built, "body")
    leaves = dict(parameter_leaves(built.graph)["drag:body"])
    leaves["shape_radius"] = leaves["shape_radius"] * factor
    if not with_held:
        leaves.pop(HELD)
    out = node.fresh_update(node.initial_state(), {"pose": jnp.asarray(IDENTITY)}, 1e-3, params=leaves)
    return np.asarray(out["resistance"], dtype=np.float64)


def test_a_moved_radius_makes_the_held_value_again_from_the_moved_positions(ball):
    moved = moved_radius(ball, 1.01)
    made = moved_radius(ball, 1.01, with_held=False)
    at_rest = moved_radius(ball, 1.0)
    print(f"\nmoved against made {np.max(np.abs(moved - made) / np.abs(made).max()):.1e}; drag ratio "
          f"{moved[0, 0] / at_rest[0, 0]:.6f}")
    scale = np.sqrt(np.outer(np.abs(np.diag(made)), np.abs(np.diag(made))))
    assert np.max(np.abs(moved - made) / scale) < 1e-12
    assert moved[0, 0] / at_rest[0, 0] == pytest.approx(1.01, rel=1e-6)


def test_control_the_held_value_read_with_stale_positions_is_caught(ball, monkeypatch):
    monkeypatch.setattr(mrs_node, "held_or_made", lambda make, built: lambda theta, held: held)
    moved = moved_radius(ball, 1.01)
    made = moved_radius(ball, 1.01, with_held=False)
    assert abs(moved[0, 0] / made[0, 0] - 1) > 5e-3


@pytest.mark.parametrize("where", ["free space", "the tube"])
def test_the_held_values_derivative_in_the_radius_is_the_made_values_and_the_differences(where, pills, ball):
    built = ball if where == "free space" else tube_ball(Drag.MRS(wall_kind="tube"), 0.2).build()
    node = drag_node(built, "body")
    leaves = dict(parameter_leaves(built.graph)["drag:body"])
    radius = leaves["shape_radius"]

    def drag(r, held=True):
        params = {**leaves, "shape_radius": r}
        if not held:
            params.pop(HELD)
        out = node.update(node.initial_state(), {"pose": jnp.asarray(IDENTITY)}, 1e-3, params=params)
        return out["resistance"][2, 2]

    through_held = float(jax.jvp(drag, (radius,), (jnp.ones_like(radius),))[1])
    through_made = float(jax.jvp(lambda r: drag(r, held=False), (radius,), (jnp.ones_like(radius),))[1])
    h = 1e-5 * float(radius)
    difference = (float(drag(radius + h)) - float(drag(radius - h))) / (2 * h)
    print(f"\nin {where}: held {through_held:.10e}, made {through_made:.10e}, difference {difference:.10e}")
    assert through_held == pytest.approx(through_made, rel=1e-12)
    assert through_held == pytest.approx(difference, rel=1e-6)


# -- the assembly's memory ------------------------------------------------------------------------------------------

def assembly_temporaries(force_count: int, quadrature_count: int) -> tuple[int, int]:
    """XLA's temporaries for the separated matrix of ``force_count`` force points on ``quadrature_count`` quadrature
    points (random, on a unit sphere), and the bytes one array of every pair's nine entries would take."""
    rng = np.random.default_rng(7)
    force = rng.normal(size=(force_count, 3))
    points = rng.normal(size=(quadrature_count, 3))
    owners = np.sort(rng.integers(0, force_count, size=quadrature_count)).astype(np.int32)
    owners[:force_count] = np.arange(force_count)
    owners = np.sort(owners)
    chunks = Chunks.of(owners, force_count)
    compiled = jax.jit(lambda f, p, w: separated_matrix(f, p, w, chunks, 0.01)).lower(
        jnp.asarray(force), jnp.asarray(points), jnp.ones(quadrature_count)).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes), 9 * 8 * force_count * quadrature_count


def test_the_assembly_never_holds_every_pair_at_once():
    temporaries, pairs = assembly_temporaries(400, 40_000)
    print(f"\ntemporaries {temporaries / 2**20:.0f} MiB against every pair's {pairs / 2**20:.0f} MiB")
    assert temporaries < pairs / 4


def reverse_temporaries(force_count: int, quadrature_count: int) -> tuple[int, int]:
    """``assembly_temporaries`` for the reverse mode: XLA's temporaries of the gradient of a sum of the matrix in the
    force points' and the quadrature's positions, which the checkpoint makes keep each chunk's inputs, not its pairs."""
    rng = np.random.default_rng(7)
    force = rng.normal(size=(force_count, 3))
    points = rng.normal(size=(quadrature_count, 3))
    owners = np.sort(rng.integers(0, force_count, size=quadrature_count)).astype(np.int32)
    owners[:force_count] = np.arange(force_count)
    owners = np.sort(owners)
    chunks = Chunks.of(owners, force_count)
    gradient = jax.grad(lambda f, p, w: jnp.sum(separated_matrix(f, p, w, chunks, 0.01)), argnums=(0, 1))
    compiled = jax.jit(gradient).lower(jnp.asarray(force), jnp.asarray(points), jnp.ones(quadrature_count)).compile()
    return int(compiled.memory_analysis().temp_size_in_bytes), 9 * 8 * force_count * quadrature_count


def test_the_reverse_mode_never_holds_every_pair_at_once():
    """Measured 189 MiB against every pair's 1,099 MiB at this size (the review's W8)."""
    temporaries, pairs = reverse_temporaries(400, 40_000)
    print(f"\nthe reverse mode's temporaries {temporaries / 2**20:.0f} MiB against every pair's {pairs / 2**20:.0f} MiB")
    assert temporaries < pairs / 4


def test_control_one_chunk_of_the_whole_quadrature_holds_every_pair_in_the_reverse_mode_too(monkeypatch):
    import mime.nodes.hydro.mrs._quadrature as quadrature

    monkeypatch.setattr(quadrature, "CHUNK_BYTES", 1 << 40)
    temporaries, pairs = reverse_temporaries(400, 40_000)
    assert temporaries >= pairs


def test_the_regularised_spheres_oracle_meets_stokes_law_as_its_blob_vanishes():
    """``regularised_sphere``'s own control, its limit ``eps -> 0``, Stokes's law: at eps / a = 0.002 its drag and
    rotational drag are 1 to the regularisation's first order (the review measured 1.00050 and 1.00150), and they
    fall linearly towards it (slopes 0.251 and 0.751 per eps / a); its own quadrature has converged there (doubled
    panels and points round move it by 1e-14)."""
    small, smaller = (regularised_drag(0.004), regularised_rotational_drag(0.004)), (
        regularised_drag(0.002), regularised_rotational_drag(0.002))
    slopes = [(a - b) / 0.002 for a, b in zip(small, smaller, strict=True)]
    fine = regularised_drag(0.002, per_panel=96, around=64)
    print(f"\nat eps/a 0.002: {smaller}; slopes {np.round(slopes, 3)}; doubled panels {fine - smaller[0]:.1e}")
    assert abs(smaller[0] - 1) < 1e-3 and abs(smaller[1] - 1) < 2e-3
    assert slopes[0] == pytest.approx(0.25, abs=0.01) and slopes[1] == pytest.approx(0.75, abs=0.01)
    assert abs(fine - smaller[0]) < 1e-10


def test_control_one_chunk_of_the_whole_quadrature_holds_every_pair(monkeypatch):
    import mime.nodes.hydro.mrs._quadrature as quadrature

    monkeypatch.setattr(quadrature, "CHUNK_BYTES", 1 << 40)
    temporaries, pairs = assembly_temporaries(400, 40_000)
    assert temporaries >= pairs


# -- the build's refusals -------------------------------------------------------------------------------------------

def test_a_force_point_whose_cell_is_empty_is_refused_by_name(monkeypatch):
    """The map planted with force point 0's quadrature points given to force point 1: its cell is empty and the matrix
    singular, so the build refuses it (``drag.empty-cell``) rather than solving."""
    real = hydrodynamic.quadrature_owners

    def emptied(kind, settings, sampling, counts):
        owners = real(kind, settings, sampling, counts)
        return np.where(owners == 0, 1, owners).astype(np.int32)

    monkeypatch.setattr(hydrodynamic, "quadrature_owners", emptied)
    with pytest.raises(BuildRefused) as refused:
        free(BALL).build()
    assert refused.value.codes() == ["drag.empty-cell"]


def test_a_free_resistance_symmetric_only_past_the_limit_is_refused_by_name(monkeypatch):
    """The check fires at its limit: with the limit set below the sphere's own asymmetry (a few in a million), the
    build refuses it (``drag.mrs-resistance``), naming the share."""
    monkeypatch.setattr(mrs_node, "ASYMMETRY_LIMIT", 1e-9)
    with pytest.raises(BuildRefused) as refused:
        free(BALL).build()
    assert refused.value.codes() == ["drag.mrs-resistance"]
    assert "symmetric only to" in refused.value.refusals[0].message


def test_a_free_resistance_that_is_not_positive_definite_is_refused_by_name(monkeypatch):
    real = mrs_node.separated_resistance
    monkeypatch.setattr(mrs_node, "separated_resistance", lambda *args: -real(*args))
    with pytest.raises(BuildRefused) as refused:
        free(BALL).build()
    assert refused.value.codes() == ["drag.mrs-resistance"]
    assert "not positive definite" in refused.value.refusals[0].message
