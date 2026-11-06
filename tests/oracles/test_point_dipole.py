"""The point dipole's field and gradient against quantities derived another way (float64, per module), drawn by
Hypothesis over moments of a magnet's order and points from a centimetre to a metre from it:

- **On its axis**, ``B = mu_0 2 m / (4 pi r^3)``, along the moment, at any distance; **across it**, ``-mu_0 m / (4 pi
  r^3)``.
- **The general form**, ``B = mu_0 / (4 pi) (3 (m . r) r / r^2 - m) / r^3``, written again in NumPy.
- **The gradient** is the field differentiated by ``jax.jacfwd``, and it is **divergence-free and curl-free**: its
  trace is zero and it is symmetric, to the rounding of its entries.
- **Two bodies on one provider**, with different levers, moments and orientations: each body's wrench from the
  built graph is the one at its own magnet, with the gradient and field of the dipole differentiated by autodiff. The
  control: the provider reading every slot's offset from the first row fails it on every run (below).
- **The provider in a build** places the dipole at its part, fixed in the world, and evaluates it at the reader's
  point; where the read names a point fixed in the body (``magnet_centre``, ``Capability.BODY_POINTS``), at the pose's
  position plus its orientation applied to that point, on every drawn pose. The control: the point taken unturned
  fails on every draw that turns it. Its node, with such a slot, passes MADDENING's battery.

A closed form written in the distance alone is judged on the scale the point was placed on: its coordinates round
on their own scale, up to a metre, beside a distance down to a centimetre. The control: a field planted with a factor
of two fails the on-axis closed form, on every draw. The near field is not
tested: there the point dipole is not the magnet's field (MIME-ANO-100), which no closed form here measures."""

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import MAGNET_MOMENT, fixed_magnet
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mime import Apparatus, Body, Experiment, State
from mime.build._maddening import _held, fed_inputs, node_state
from mime.dynamics import RigidInertial
from mime.fields import PointDipole
from mime.interactions import Magnetic
from mime.nodes.actuation import point_dipole
from mime.nodes.actuation._point_dipole import PointDipoleNode
from mime.scene import Cylinder, Scene, Z
from mime.testing import node_results
from mime.things import Part, PermanentMagnet
from mime.units import VACUUM_PERMEABILITY, A_m2, kg, kg_m2, m, m_per_s, m_per_s2, mm, rad_per_s, s

K = VACUUM_PERMEABILITY / (4 * np.pi)
EPS = float(np.finfo(np.float64).eps)
#: How many float64 roundings of its scale each comparison allows: each side computes in a few dozen operations.
ROUNDINGS = 64
PROPERTY = settings(deadline=None)

field_and_gradient = jax.jit(point_dipole)
gradient_by_autodiff = jax.jit(jax.jacfwd(lambda moment, centre, point: point_dipole(moment, centre, point)[0],
                                          argnums=2))


def directions() -> st.SearchStrategy[np.ndarray]:
    """A unit vector, drawn from a box and scaled, never near zero length."""
    return st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))).map(np.asarray).filter(
        lambda v: float(np.linalg.norm(v)) > 0.1).map(lambda v: v / np.linalg.norm(v))


def magnets() -> st.SearchStrategy[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """A moment from a millimetre magnet's to a centimetre magnet's (1e-4 to 50 A m^2), its centre within a metre of
    the origin, and a point a centimetre to a metre from it."""
    return st.tuples(directions(), st.floats(1e-4, 50.0), st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))),
                     directions(), st.floats(1e-2, 1.0)).map(
        lambda d: (d[0] * d[1], np.asarray(d[2]), np.asarray(d[2]) + d[4] * d[3]))


def close(found: Any, expected: Any, roundings: float = ROUNDINGS) -> bool:
    found, expected = np.asarray(found, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    return bool(np.all(np.abs(found - expected) <= roundings * EPS * float(np.max(np.abs(expected)))))


def placement_spread(centre: np.ndarray, point: np.ndarray) -> float:
    """How much a point placed at a distance from a centre rounds, against that distance: its coordinates round on
    their own scale, and a closed form written in the distance alone does not see it. The field goes as the distance's
    inverse cube, so its rounding grows by this factor."""
    return 1.0 + 3.0 * (float(np.linalg.norm(centre)) + float(np.linalg.norm(point))) / float(
        np.linalg.norm(point - centre))


def on_axis_misfit(field: Any, moment: np.ndarray, centre: np.ndarray, distance: float, sign: float) -> bool:
    point = centre + sign * distance * moment / np.linalg.norm(moment)
    found = np.asarray(field(jnp.asarray(moment), jnp.asarray(centre), jnp.asarray(point))[0])
    return close(found, K * 2 * moment / distance**3, ROUNDINGS * placement_spread(centre, point))


@PROPERTY
@given(magnets(), st.sampled_from([1.0, -1.0]))
def test_on_its_axis_the_field_is_twice_the_moment_over_the_cube_of_the_distance(drawn, sign):
    moment, centre, point = drawn
    assert on_axis_misfit(field_and_gradient, moment, centre, float(np.linalg.norm(point - centre)), sign)


@PROPERTY
@given(magnets())
def test_across_its_axis_the_field_is_the_moment_reversed_over_the_cube_of_the_distance(drawn):
    moment, centre, point = drawn
    distance = float(np.linalg.norm(point - centre))
    across = np.cross(moment, point - centre)
    if float(np.linalg.norm(across)) <= 1e-3 * float(np.linalg.norm(moment)) * distance:
        across = np.cross(moment, (1.0, 0.0, 0.0)) if abs(moment[0]) < 0.9 * np.linalg.norm(moment) else \
            np.cross(moment, (0.0, 1.0, 0.0))
    target = centre + distance * across / np.linalg.norm(across)
    found = np.asarray(field_and_gradient(jnp.asarray(moment), jnp.asarray(centre), jnp.asarray(target))[0])
    assert close(found, -K * moment / distance**3, ROUNDINGS * placement_spread(centre, target))


def general_form(moment: np.ndarray, centre: np.ndarray, point: np.ndarray) -> np.ndarray:
    r = point - centre
    distance = np.linalg.norm(r)
    return K * (3 * np.dot(moment, r) * r / distance**2 - moment) / distance**3


@PROPERTY
@given(magnets())
def test_the_field_is_the_general_form(drawn):
    moment, centre, point = drawn
    found = field_and_gradient(jnp.asarray(moment), jnp.asarray(centre), jnp.asarray(point))[0]
    assert close(found, general_form(moment, centre, point))


@PROPERTY
@given(magnets())
def test_the_gradient_is_the_field_differentiated_and_is_divergence_and_curl_free(drawn):
    moment, centre, point = drawn
    gradient = np.asarray(field_and_gradient(jnp.asarray(moment), jnp.asarray(centre), jnp.asarray(point))[1])
    differentiated = np.asarray(gradient_by_autodiff(jnp.asarray(moment), jnp.asarray(centre), jnp.asarray(point)))
    scale = float(np.max(np.abs(gradient)))
    assert close(gradient, differentiated)
    assert abs(float(np.trace(gradient))) <= ROUNDINGS * EPS * scale
    assert float(np.max(np.abs(gradient - gradient.T))) <= ROUNDINGS * EPS * scale


def doubled(moment: Any, centre: Any, point: Any) -> tuple[Any, Any]:
    field, gradient = point_dipole(moment, centre, point)
    return 2.0 * field, gradient


@PROPERTY
@given(magnets(), st.sampled_from([1.0, -1.0]))
def test_control_a_field_off_by_a_factor_of_two_fails_the_closed_form_on_every_draw(drawn, sign):
    moment, centre, point = drawn
    assert not on_axis_misfit(jax.jit(doubled), moment, centre, float(np.linalg.norm(point - centre)), sign)


def test_the_provider_places_its_dipole_at_its_part_and_reads_it_at_the_body():
    """A magnet fixed 10 cm above the origin, magnetised along z; the body 1 mm below the origin: on the magnet's
    axis, below it, so the field there is the on-axis field along +z."""
    built = fixed_magnet(at=m((0.0, 0.0, 0.1))).build()
    node = built.graph.get_node("stand.magnetic")
    slot = next(name for name in node_state(built.graph, "stand.magnetic") if name.startswith("flux_density_"))
    stepped = node.update(node_state(built.graph, "stand.magnetic"),
                          {f"points_{slot.removeprefix('flux_density_')}": jnp.asarray(
                              (0.0, 0.0, -1e-3, 1.0, 0.0, 0.0, 0.0))}, 1e-3)
    distance = 0.1 + 1e-3
    assert close(stepped[slot], (0.0, 0.0, K * 2 * MAGNET_MOMENT / distance**3))
    assert close(stepped["centre"], (0.0, 0.0, 0.1))


#: A body's magnet 0.8 mm off its centre of mass, along no axis.
LEVER = np.array([0.5e-3, -0.4e-3, 0.48e-3])


def unit(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q) if np.linalg.norm(q) > 1e-3 else np.array([1.0, 0.0, 0.0, 0.0])


def turned(q, v):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]) @ v


@pytest.fixture(scope="module")
def offset_slot():
    """The fixed magnet's provider as the force 0.2.0 reads it, from a body whose magnet is ``LEVER`` off its centre of
    mass: its node, and the slot that reads at the magnet's centroid."""
    built = fixed_magnet("offset_slot", law=Magnetic.Force(), magnet_centre=m(tuple(LEVER))).build()
    node = built.graph.get_node("stand.magnetic")
    (slot,) = node.constants()["offset_slots"]
    return node, slot


#: The body's pose: within 3 mm of the origin, turned any way.
BODY_POSES = st.tuples(st.tuples(*(st.floats(-3e-3, 3e-3) for _ in range(3))),
                       st.tuples(*(st.floats(-1.0, 1.0) for _ in range(4))).map(unit))


def at_the_centroid(node, slot, position, orientation) -> np.ndarray:
    pose = jnp.asarray(np.concatenate([position, orientation]))
    return np.asarray(node.update(node.initial_state(), {f"points_{slot}": pose}, 1e-3)[f"gradient_{slot}"])


@PROPERTY
@given(BODY_POSES)
def test_a_slot_read_at_a_point_fixed_in_the_body_is_evaluated_there(offset_slot, drawn):
    node, slot = offset_slot
    position, orientation = np.asarray(drawn[0]), np.asarray(drawn[1])
    expected = np.asarray(field_and_gradient(jnp.asarray((0.0, 0.0, MAGNET_MOMENT)), jnp.asarray((0.0, 0.0, 0.1)),
                                             jnp.asarray(position + turned(orientation, LEVER)))[1])
    found = at_the_centroid(node, slot, position, orientation)
    assert np.max(np.abs(found - expected)) <= ROUNDINGS * EPS * np.max(np.abs(expected))


@PROPERTY
@given(BODY_POSES)
def test_control_the_point_taken_unturned_fails_on_every_draw_that_turns_it(offset_slot, drawn):
    node, slot = offset_slot
    position, orientation = np.asarray(drawn[0]), np.asarray(drawn[1])
    assume(float(np.linalg.norm(turned(orientation, LEVER) - LEVER)) > 1e-2 * float(np.linalg.norm(LEVER)))
    unturned = np.asarray(field_and_gradient(jnp.asarray((0.0, 0.0, MAGNET_MOMENT)), jnp.asarray((0.0, 0.0, 0.1)),
                                             jnp.asarray(position + LEVER))[1])
    found = at_the_centroid(node, slot, position, orientation)
    assert np.max(np.abs(found - unturned)) > ROUNDINGS * EPS * np.max(np.abs(unturned))


def outside_the_fixed_magnet(given):
    """Each drawn reader's pose with its quaternion of unit length and its position moved out of the fixed magnet's 5 cm
    sphere (centred 10 cm up), along its direction from the magnet's centre; a batch of poses pose by pose."""
    centre = np.array([0.0, 0.0, 0.1])

    def moved(pose: np.ndarray) -> np.ndarray:
        offset = pose[:3] - centre
        distance = float(np.linalg.norm(offset))
        position = pose[:3] if distance >= 0.05 else centre + 0.05 * (
            offset / distance if distance > 0 else np.array([1.0, 0.0, 0.0]))
        return np.concatenate([position, unit(pose[3:])])

    held = dict(given)
    for key, value in given.items():
        if not key.startswith("points_"):
            continue
        poses = np.asarray(value, dtype=np.float64)
        flat = poses.reshape(-1, 7)
        held[key] = np.stack([moved(pose) for pose in flat]).reshape(poses.shape).astype(np.asarray(value).dtype)
    return held


def test_the_provider_node_with_a_slot_at_a_body_point_passes_maddenings_battery(offset_slot):
    node, _ = offset_slot
    results = node_results(node, {"boundary_bounds": {name: (-1.0, 1.0) for name in node.boundary_input_spec()},
                                  "constrain_boundary": outside_the_fixed_magnet})
    print("\n" + "\n".join(str(r) for r in results.values()))
    assert all(r.passed for r in results.values())


#: Two bodies on one point dipole, each with a lever, a moment and a pose of its own: (position, quaternion, moment,
#: magnet_centre), in SI. The levers differ in length and direction, so that no slot's offset is another's.
TWO_BODIES = {
    "b1": ((0.003, 0.0, 0.0), (1.0, 0.2, 0.1, 0.3), (0.0, 1.68e-3, 0.0), (0.0, 0.0, 1e-3)),
    "b2": ((-0.004, 0.002, 0.001), (0.5, 0.4, -0.3, 0.7), (1e-3, 0.5e-3, 0.2e-3), (1e-3, -0.7e-3, 0.4e-3)),
}
SOURCE_AT, SOURCE_DIRECTION = np.array([0.01, -0.02, 0.15]), unit((0.2, -0.3, 1.0))


def two_bodies_on_one_dipole():
    exp = Experiment("two_bodies", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=s(1e-4))
    for name, (position, orientation, moment, lever) in TWO_BODIES.items():
        exp.add_body(Body(name, dynamics=RigidInertial(), initial=State(
            position=m(position), orientation=tuple(float(c) for c in unit(orientation)),
            velocity=m_per_s((0.0, 0.0, 0.0)), angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
        exp.declare(f"{name}.mass", kg(1e-4), reason="a millimetre body, its order")
        exp.declare(f"{name}.inertia", kg_m2((1e-10, 1e-10, 1e-10)), reason="a millimetre body, its order")
        exp.declare(f"{name}.magnetic_moment", A_m2(moment), reason="a millimetre body's magnet, its order")
        exp.declare(f"{name}.magnet_centre", m(lever), reason="a lever of its own")
    exp.add_apparatus(Apparatus(
        "stand", parts={"magnet": Part(Cylinder(radius=mm(4.0), length=mm(6.0), axis=tuple(SOURCE_DIRECTION)),
                                       at=m(tuple(SOURCE_AT)))},
        magnets={"magnet": PermanentMagnet(direction=tuple(SOURCE_DIRECTION))}, fields={"magnetic": PointDipole(
            part="magnet")}))
    exp.declare("stand.magnetic_moment", A_m2(MAGNET_MOMENT), reason="de Jongh's rotor magnet")
    for name in TWO_BODIES:
        exp.attach(Magnetic.Torque(), body=name, field="stand.magnetic")
        exp.attach(Magnetic.Force(), body=name, field="stand.magnetic")
    return exp


def wrench_at_the_magnet(name: str) -> np.ndarray:
    """``[F; tau]`` about the centre of mass of the body ``name`` in the dipole: ``F = G^T m`` and ``tau = m x B +
    (R l) x F`` at the magnet's centroid ``p = x + R l``, with ``B`` and ``G`` from the field's own formula and
    ``jax.jacfwd`` of it, and not from the node's slots."""
    position, orientation, moment, lever = (np.asarray(v, dtype=np.float64) for v in TWO_BODIES[name])
    rotation = turned_matrix(unit(orientation))
    moment, lever = rotation @ moment, rotation @ lever
    source = jnp.asarray(MAGNET_MOMENT * SOURCE_DIRECTION)
    point = jnp.asarray(position + lever)
    field = np.asarray(point_dipole(source, jnp.asarray(SOURCE_AT), point)[0])
    gradient = np.asarray(gradient_of_the_field(source, jnp.asarray(SOURCE_AT), point))
    force = gradient.T @ moment
    return np.concatenate([force, np.cross(moment, field) + np.cross(lever, force)])


def turned_matrix(q: np.ndarray) -> np.ndarray:
    return np.stack([turned(q, e) for e in np.eye(3)], axis=1)


gradient_of_the_field = jax.jit(jax.jacfwd(lambda moment, centre, point: point_dipole(moment, centre, point)[0],
                                           argnums=2))


def wrenches_from_the_graph() -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph = two_bodies_on_one_dipole().build().graph
    graph.reset_state()
    _, history = _held(graph).run_scan_with_history(1, fed_inputs(graph, None))
    return {name: sum(np.asarray(history[f"magnetic.{law}:{name}"]["wrench"])[0] for law in ("torque", "force"))
            for name in TWO_BODIES}


def test_two_bodies_with_different_levers_on_one_provider_each_get_the_wrench_at_their_own_magnet():
    found = wrenches_from_the_graph()
    for name in TWO_BODIES:
        expected = wrench_at_the_magnet(name)
        assert close(found[name][:3], expected[:3]) and close(found[name][3:], expected[3:]), name


class FirstRowOnly(tuple):
    """The slots that read at a body point, answering every slot's index with the first row of ``offsets``."""

    def index(self, value, *args):
        return 0


def test_control_a_provider_that_reads_every_slots_offset_from_the_first_row_fails_the_second_body(monkeypatch):
    """The plant: every slot reads the first row of offsets. The first row is the first body's force slot (slots are sorted by name),
    so the first body is unmoved and the second reads at the wrong lever, by far more than the bound allows. The
    bodies and levers are fixed, so the control fails on every run."""
    real = PointDipoleNode._offset_slots
    monkeypatch.setattr(PointDipoleNode, "_offset_slots", lambda self: FirstRowOnly(real(self)))
    found = wrenches_from_the_graph()
    first, second = (wrench_at_the_magnet(name) for name in TWO_BODIES)
    assert close(found["b1"][:3], first[:3]) and close(found["b1"][3:], first[3:])
    assert not close(found["b2"][:3], second[:3]) and not close(found["b2"][3:], second[3:])
