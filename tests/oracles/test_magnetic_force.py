"""``Magnetic.Force`` beyond its family's suite (float64, per module):

- **The pair law**, on the interaction alone, with no arm: two point dipoles, each the source of the field the other's
  force reads. The force on the second from the first's gradient, and on the first from the second's, sum to zero, to
  the rounding of the forces, on every draw. The control: a gradient missing its ``r m^T`` term (no longer the field's
  derivative, nor symmetric) breaks the law on every draw, which are drawn away from the configurations where its
  missing part vanishes.
- **Sign**: a moment aligned with the field on a magnet's axis is pulled toward the magnet, where the field is
  stronger; turned against it, pushed away.
- **A uniform field** pulls on nothing: its gradient is zero, and so is the force, exactly.
- **The torque's second support row**: ``Magnetic.Torque`` bound to a point dipole reads the field at the body's
  centre of mass, ``m x B(x)``."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from chain_cases import BODY_MOMENT, MAGNET_MOMENT, add_body, experiment, fixed_magnet
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mime.build._maddening import node_state, run_scan
from mime.drives import Constant
from mime.fields import UniformRotatingField
from mime.interactions import Magnetic
from mime.nodes.actuation import DIPOLE_CONSTANT, point_dipole
from mime.nodes.magnetic import MagneticForceNode
from mime.things import Apparatus
from mime.units import Hz, mT

EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 64
PROPERTY = settings(deadline=None)
IDENTITY_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
FORCE = MagneticForceNode("force", 1e-3, moment=(0.0, 0.0, 0.0), magnet_centre=(0.0, 0.0, 0.0))


def directions() -> st.SearchStrategy[np.ndarray]:
    return st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))).map(np.asarray).filter(
        lambda v: float(np.linalg.norm(v)) > 0.1).map(lambda v: v / np.linalg.norm(v))


def pairs() -> st.SearchStrategy[tuple[np.ndarray, ...]]:
    """Two dipoles a centimetre to a metre apart: moments from a millimetre magnet's to a centimetre magnet's, each
    centre within a metre of the origin."""
    return st.tuples(directions(), st.floats(1e-4, 50.0), directions(), st.floats(1e-4, 50.0),
                     st.tuples(*(st.floats(-1.0, 1.0) for _ in range(3))), directions(), st.floats(1e-2, 1.0)).map(
        lambda d: (d[0] * d[1], np.asarray(d[4]), d[2] * d[3], np.asarray(d[4]) + d[6] * d[5]))


def force_on(target_moment: np.ndarray, gradient: Any) -> np.ndarray:
    """What the law delivers on a body with ``target_moment`` at its centre of mass, unturned, in ``gradient``."""
    p = {"moment": jnp.asarray(target_moment), "magnet_centre": jnp.zeros(3)}
    out = FORCE.advance(p, {}, {"pose": jnp.asarray(IDENTITY_POSE), "gradient": jnp.asarray(gradient)}, 1e-3)
    return np.asarray(out["wrench"])[:3]


def without_its_r_m_term(moment: Any, centre: Any, point: Any) -> Any:
    """The gradient with its ``r m^T`` term planted away."""
    _, gradient = point_dipole(moment, centre, point)
    offset = point - centre
    return gradient - 3.0 * DIPOLE_CONSTANT * jnp.outer(offset, moment) / jnp.sum(offset * offset) ** 2.5


def pair_misfit(gradient_of, first, at_first, second, at_second) -> float:
    on_second = force_on(second, gradient_of(jnp.asarray(first), jnp.asarray(at_first), jnp.asarray(at_second)))
    on_first = force_on(first, gradient_of(jnp.asarray(second), jnp.asarray(at_second), jnp.asarray(at_first)))
    scale = max(float(np.max(np.abs(on_second))), float(np.max(np.abs(on_first))))
    return float(np.max(np.abs(on_first + on_second))) / (EPS * scale)


def generic(first, at_first, second, at_second) -> bool:
    """A pair whose planted defect shows: ``(m1 . r) m2 - (m2 . r) m1`` at least a tenth of ``|m1| |m2|``."""
    r = (at_second - at_first) / np.linalg.norm(at_second - at_first)
    left = np.dot(first, r) * second - np.dot(second, r) * first
    return float(np.linalg.norm(left)) >= 0.1 * float(np.linalg.norm(first) * np.linalg.norm(second))


def the_gradient(moment: Any, centre: Any, point: Any) -> Any:
    return point_dipole(moment, centre, point)[1]


@PROPERTY
@given(pairs())
def test_two_point_dipoles_pull_on_each_other_equally_and_oppositely(drawn):
    assert pair_misfit(jax.jit(the_gradient), *drawn) <= ROUNDINGS


@PROPERTY
@given(pairs())
def test_control_a_gradient_missing_a_term_breaks_the_pair_law_on_every_draw(drawn):
    assume(generic(*drawn))
    assert pair_misfit(jax.jit(without_its_r_m_term), *drawn) > ROUNDINGS


def body_force(built, moment: tuple[float, float, float]) -> np.ndarray:
    """The force the law delivers on a body at its start pose with ``moment``, from the field as the graph reads it."""
    graph = built.graph
    graph.reset_state()
    state = run_scan(graph, 1, {"nodes": {"magnetic.force:umr": {"moment": jnp.asarray(moment)}}})
    return np.asarray(state["magnetic.force:umr"]["wrench"])[:3]


def test_an_aligned_moment_is_pulled_toward_the_stronger_field_and_a_reversed_one_pushed_away():
    """The magnet 10 cm above the origin, magnetised along +z: below it on its axis the field points up and grows
    toward it. A body there with its moment along +z is pulled up, toward the magnet; along -z, pushed down. On the
    axis the force is along it: ``3 mu_0 m M / (2 pi r^4)``."""
    built = fixed_magnet().build()
    distance = 0.1 + 1e-3
    expected = 3 * DIPOLE_CONSTANT * 2 * MAGNET_MOMENT * 1e-3 / distance**4
    up, down = body_force(built, (0.0, 0.0, 1e-3)), body_force(built, (0.0, 0.0, -1e-3))
    assert up[2] > 0 > down[2]
    assert np.allclose(up, (0.0, 0.0, expected), rtol=1e-12, atol=1e-12 * expected)
    assert np.allclose(down, -up, rtol=0.0, atol=1e-15 * expected)


def test_a_uniform_field_pulls_on_nothing():
    exp = experiment("uniform_force")
    exp.add_apparatus(Apparatus("coils", fields={
        "magnetic": UniformRotatingField(rotation_axis=(0.0, 0.0, 1.0), initial_direction=(1.0, 0.0, 0.0))},
        commands={"frequency": Constant(Hz(5.0)), "amplitude": Constant(mT(1.0))}))
    add_body(exp)
    exp.attach(Magnetic.Force(), body="umr", field="coils.magnetic")
    graph = exp.build().graph
    graph.reset_state()
    state = run_scan(graph, 3)
    assert np.array_equal(np.asarray(state["magnetic.force:umr"]["wrench"]), np.zeros(6))


def test_the_torque_reads_a_dipoles_field_at_the_bodys_centre_of_mass():
    """``Magnetic.Torque`` bound to the fixed magnet, by its second support row: ``m x B`` with ``B`` the dipole's
    field at the body's start, 1 mm below the origin."""
    built = fixed_magnet("torque_at_centre", law=Magnetic.Torque()).build()
    reads = {edge.target[1]: edge for edge in built.report.lags if edge.target[0] == "stand.magnetic"}
    assert any(field.startswith("points_") for field in reads), "the field is read at the body's points"
    graph = built.graph
    graph.reset_state()
    state = run_scan(graph, 1)
    field = np.asarray(point_dipole(jnp.asarray((0.0, 0.0, MAGNET_MOMENT)), jnp.asarray((0.0, 0.0, 0.1)),
                                    jnp.asarray((0.0, 0.0, -1e-3)))[0])
    expected = np.cross(np.asarray(BODY_MOMENT), field)
    torque = np.asarray(state["magnetic.torque:umr"]["wrench"])[3:]
    assert np.allclose(torque, expected, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(expected))))
    assert node_state(graph, "magnetic.torque:umr") is not None
