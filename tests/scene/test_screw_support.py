"""The screw's support query, refined on its own surface (``Screw.support``; review 7o's F2, option B): from one of its
samples at a contact's default spacing, Newton's steps on the lateral surface reach the body's farthest point along the
direction, which moves continuously as the direction turns, and its derivatives, in the direction and in the kind's
settings, are the implicit function's, carried by the one step taken with the gradients flowing.

The control: the same refinement with that last step's gradients stopped too (its point held as the stopped steps
leave it) has no derivative in the settings, and fails the derivative's check."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hull_cases import FL9_SETTINGS, kind_points

from mime.registry import Support
from mime.scene import Screw
from mime.units import dimensionless, mm

FL9 = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33), modulation=dimensionless(0.33),
            starts=2)
#: The contact's default spacing on FL-9, the margin, whose samples the refinement starts from.
SPACING = 1e-4
#: The directions: across the axis, tilted towards it by up to 60 degrees, at turns round it that put the farthest
#: point on a crest in the middle and at the ends.
DIRECTIONS = np.asarray([(math.sin(t), math.cos(t) * math.cos(r), math.cos(t) * math.sin(r))
                         for t in (0.0, 0.1, 0.4, 1.0) for r in (0.3, 1.7, 2.9, 4.4)])


def parameters(**changed):
    return {**{name: jnp.asarray(value, dtype=jnp.float64) if isinstance(value, float) else value
               for name, value in FL9_SETTINGS.items()}, **changed}


def farthest_samples() -> np.ndarray:
    points = kind_points(FL9, FL9_SETTINGS, SPACING)
    return points[np.argmax(points @ DIRECTIONS.T, axis=0)]


@pytest.fixture(scope="module", autouse=True)
def x64():
    with jax.enable_x64(True):
        yield


def test_from_a_sample_at_the_margins_spacing_it_reaches_the_bodys_farthest_point():
    """Against a dense quadrature (a 300th of its crests' radius apart): never short of it by more than the dense
    quadrature's own gap, and from every one of its nearby samples the same point to rounding."""
    near = farthest_samples()
    found = np.asarray(FL9.support(parameters(), jnp.asarray(DIRECTIONS), jnp.asarray(near)).point)
    dense = kind_points(FL9, FL9_SETTINGS, 2.0748e-3 / 300)
    brute = (dense @ DIRECTIONS.T).max(axis=0)
    reached = np.einsum("ij,ij->i", found, DIRECTIONS)
    gap = 8.0 * (2.0748e-3 / 300) ** 2 / 2.0748e-3
    assert np.all(reached >= brute - 1e-12), (reached - brute).min()
    assert np.all(reached <= brute + gap), (reached - brute).max()
    points = kind_points(FL9, FL9_SETTINGS, SPACING)
    for i, d in enumerate(DIRECTIONS):
        nearby = points[np.argsort(-(points @ d))[:6]]
        again = np.asarray(FL9.support(parameters(), jnp.asarray(np.tile(d, (6, 1))), jnp.asarray(nearby)).point)
        assert np.max(np.abs(again - found[i])) < 1e-12, (i, np.max(np.abs(again - found[i])))


def test_the_point_moves_continuously_as_the_direction_turns_about_the_axis():
    """A turn of the direction about the axis by a thousandth of a radian moves the point along the crest by about the
    crest's radius times that, never by a spacing: the touching point slides, where a sample's jumps."""
    angles = np.linspace(0.0, 0.05, 51)
    directions = np.stack([np.full_like(angles, 0.1), np.cos(angles), np.sin(angles)], axis=1)
    points = kind_points(FL9, FL9_SETTINGS, SPACING)
    near = points[np.argmax(points @ directions[0])]
    found = []
    for d in directions:
        found.append(np.asarray(FL9.support(parameters(), jnp.asarray(d), jnp.asarray(near)).point))
        near = found[-1]
    moves = np.linalg.norm(np.diff(np.asarray(found), axis=0), axis=1)
    assert moves.max() < 10.0 * 2.0748e-3 * 1e-3, moves.max()


def settings_slope(support) -> tuple[np.ndarray, np.ndarray]:
    """The point's derivative in the radius and the modulation, by forward mode and by central difference."""
    near = jnp.asarray(farthest_samples())
    d = jnp.asarray(DIRECTIONS)

    def point(radius, modulation):
        return support(parameters(radius=radius, modulation=modulation), d, near).point

    radius, modulation = jnp.asarray(1.56e-3), jnp.asarray(0.33)
    forward = np.stack([np.asarray(jax.jacfwd(point, argnums=k)(radius, modulation)) for k in (0, 1)], axis=-1)
    steps = (1e-9, 1e-7)
    central = np.stack([
        (np.asarray(point(radius + steps[0], modulation)) - np.asarray(point(radius - steps[0], modulation))) / (2 * steps[0]),
        (np.asarray(point(radius, modulation + steps[1])) - np.asarray(point(radius, modulation - steps[1]))) / (2 * steps[1]),
    ], axis=-1)
    return forward, central


def test_its_derivatives_in_the_settings_are_the_implicit_functions():
    forward, central = settings_slope(FL9.support)
    assert np.allclose(forward, central, rtol=0, atol=1e-6 * np.abs(central).max()), np.abs(forward - central).max()


def test_its_derivative_in_the_direction_is_the_points_central_difference():
    near = jnp.asarray(farthest_samples())
    found = FL9.support(parameters(), jnp.asarray(DIRECTIONS), near)
    step = 1e-7
    central = np.stack([(np.asarray(FL9.support(parameters(), jnp.asarray(DIRECTIONS + step * np.eye(3)[k]), near).point)
                         - np.asarray(FL9.support(parameters(), jnp.asarray(DIRECTIONS - step * np.eye(3)[k]),
                                                  near).point)) / (2 * step) for k in range(3)], axis=-1)
    assert np.allclose(np.asarray(found.jacobian), central, rtol=0, atol=1e-6 * np.abs(central).max())


def test_control_with_its_last_step_stopped_the_derivative_in_the_settings_is_lost(monkeypatch):
    """The refinement with its last step's gradients stopped too (its place held as the stopped steps leave it): the
    point keeps its value, and loses the part of its derivative that only the flowing step carries, the move of the
    farthest point along the surface as the settings change."""
    settled = Screw._settled

    def stopped(self, parameters_, d, theta, eta, scale):
        held = settled(self, parameters_, d, theta, eta, scale)
        moved = jax.lax.stop_gradient(held.point)
        fixed = self._lateral(parameters_, d, theta, eta)[0]
        return Support(moved + fixed - jax.lax.stop_gradient(fixed), held.jacobian)

    monkeypatch.setattr(Screw, "_settled", stopped)
    forward, central = settings_slope(FL9.support)
    assert not np.allclose(forward, central, rtol=0, atol=1e-6 * np.abs(central).max())
