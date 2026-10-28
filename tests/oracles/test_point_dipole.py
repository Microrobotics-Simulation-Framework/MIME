"""The point dipole's field and gradient against quantities derived another way (float64, per module), drawn by
Hypothesis over moments of a magnet's order and points from a centimetre to a metre from it:

- **On its axis**, ``B = mu_0 2 m / (4 pi r^3)``, along the moment, at any distance; **across it**, ``-mu_0 m / (4 pi
  r^3)``.
- **The general form**, ``B = mu_0 / (4 pi) (3 (m . r) r / r^2 - m) / r^3``, written again in NumPy.
- **The gradient** is the field differentiated by ``jax.jacfwd``, and it is **divergence-free and curl-free**: its
  trace is zero and it is symmetric, to the rounding of its entries.
- **The provider in a build** places the dipole at its part, fixed in the world, and evaluates it at the reader's
  point.

A closed form written in the distance alone is judged on the scale the point was placed on: its coordinates round
on their own scale, up to a metre, beside a distance down to a centimetre. The control: a field planted with a factor
of two fails the on-axis closed form, on every draw. The near field is not
tested: there the point dipole is not the magnet's field (MIME-ANO-100), which no closed form here measures."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from chain_cases import MAGNET_MOMENT, fixed_magnet
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.build._maddening import node_state
from mime.nodes.actuation import point_dipole
from mime.units import VACUUM_PERMEABILITY, m

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
