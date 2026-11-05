"""The solid kinds of 7d against what they are: the de Jongh helix (de Jongh et al. 2025, Eq. 1), the sphere and the
capsule.

**The oracle.** The helix's volume, ``pi R^2 L (1 + eps^2 / 2)``, equals script A's declared volume (FL-9, the 0.2
line's ``(410 + 1000) kg/m^3`` body's) to two units in the last place. The screw's signed distance, from its root find,
equals a brute-force minimum (a dense grid of the surface, then a local refinement in float64 with SciPy) at points
drawn round FL-9 and at random screws, out to three radii from the axis, and at the 0.2 line's own surface points,
which lie on it: the 0.2 body moved half its length. The screw's turn at a point, carried to twice the working
precision, is the exact turn's (exact arithmetic on the float inputs) to a unit in the last place, in float64 and
float32, out to a thousand times FL-9's length; taken plainly in the working precision it rounds at its size (the
control).

The plan's refusals of a body value its hull kind gives, of an extent no hull gives, and of a drag on a body that is
not the surrogate's family (another kind, or the helix turned or moved in the body's frame) are named here, with the
drag's refusal of a tube whose shape has an axis but not a circular tube's parameters."""

import math
from fractions import Fraction
from pathlib import Path

import jax
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from pxr import Usd, UsdGeom
from scipy.optimize import minimize, root
from script_a import VOLUME, fl9, script_a

import mime.interactions.hydrodynamic as hydrodynamic
from mime import Body, Experiment, Medium, State, Structure, asset
from mime.build import BuildRefused
from mime.core import SCREW, Cylinder, jnp
from mime.core._shapes import _exact_product as exact_product
from mime.core._shapes import _exact_sum as exact_sum
from mime.data import asset as data_asset
from mime.registry import Problem, Region
from mime.dynamics import RigidInertial
from mime.interactions import KEPT, Contact, Drag
from mime.scene import Capsule, Ellipsoid, Scene, Screw, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import dimensionless, kg_m2, kg_per_m3, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

R, L = 1.56e-3, 7.47e-3


def settings_of(nu=2.33, epsilon=0.33):
    return {"radius": R, "length": L, "wavenumber": nu, "modulation": epsilon}


def codes(experiment) -> set[str]:
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    return set(refused.value.codes())


def test_the_helixs_volume_is_script_as_declared_volume_to_two_ulps():
    found = fl9().volume({"radius": mm(1.56).value, "length": mm(7.47).value, "wavenumber": 2.33, "modulation": 0.33})
    declared = VOLUME.value
    assert abs(found - declared) <= 2 * math.ulp(declared), (found, declared)
    assert found == pytest.approx(math.pi * R * R * L * (1 + 0.33 ** 2 / 2), rel=1e-15)


def test_its_reach_and_family_are_its_settings():
    helix = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.33), starts=2)
    assert helix.radial_reach(settings_of()) == R * 1.33
    assert helix.family_parameters(settings_of()) == {"radius": R, "length": L, "wavenumber": 2.33, "modulation": 0.33,
                                                      "starts": 2}


@pytest.mark.parametrize(("change", "problem"), [
    ({"modulation": 0.0}, "not above 0"), ({"modulation": 0.6}, "at most 0.5"), ({"wavenumber": 4.5}, "within +-4"),
    ({"radius": -1e-3}, "radius is -0.001 m"),
])
def test_settings_outside_what_the_root_find_was_checked_over_are_refused(change, problem):
    found = fl9().setting_problems({**settings_of(), **change})
    assert any(problem in line for line in found), found


@pytest.mark.parametrize("starts", [1, 5, True])
def test_starts_other_than_two_to_four_are_refused(starts):
    helix = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.33), starts=starts)
    assert any("starts is" in line for line in helix.setting_problems(settings_of()))


# -- the screw's distance against a brute-force minimum -------------------------------------------------------------

def surface(theta, zeta, nu, epsilon, starts, radius=R, length=L):
    """De Jongh's Eq. 1 with the axis along x and the body centred on the origin: the 0.2 line's surface, its zeta
    from 0 to L, moved L / 2 along the axis."""
    rho = radius * (1 + epsilon * np.sin(starts * theta))
    alpha = theta + nu * (zeta + length / 2) / radius
    return np.stack([zeta, rho * np.cos(alpha), rho * np.sin(alpha)], -1)


def stationarity(x, point, nu, epsilon, starts, radius):
    """Half the gradient of the squared distance from ``point`` to the surface point ``(theta, zeta / radius) = x``,
    in the reference radius: the residual's components along the surface's two tangents, which vanish where the
    surface point is the nearest one."""
    theta, zeta = x[0], x[1] * radius
    rho = radius * (1 + epsilon * np.sin(starts * theta))
    alpha = theta + nu * (zeta + L / 2) / radius
    c, s = np.cos(alpha), np.sin(alpha)
    residual = np.array([zeta, rho * c, rho * s]) - point
    drho = radius * epsilon * starts * np.cos(starts * theta)
    along_theta = np.array([0.0, drho * c - rho * s, drho * s + rho * c])
    along_zeta = np.array([1.0, -rho * s * nu / radius, rho * c * nu / radius])
    return np.array([residual @ along_theta / radius ** 2, residual @ along_zeta / radius])


#: How many unit roundoffs of its scale ``stationarity`` may remain at (``stationarity_floor``): the largest excess over
#: one that 6,900 draws gave, the strategy's corner (|nu| near 4, the smallest radius, the farthest point) among them, was
#: 1.5, so the factor leaves a margin of ten.
STATIONARITY_ROUNDINGS = 16
UNIT_ROUNDOFF = float(np.finfo(float).eps) / 2


def stationarity_floor(x, nu, epsilon, radius, distance) -> float:
    """What ``stationarity`` can remain at when the solve has done all it can, from the evaluation's own rounding: a unit
    roundoff times three factors, with no part of its own (each is at least one).

    - ``(1 + |nu|)^2 (1 + epsilon)^2``: the stationarity's change per unit change of the coordinate along the axis,
      about ``1 + nu^2 rho^2 / R^2``; the solve cannot place ``x[1]`` closer than the spacing of floats at its size, so
      it stops at that change.
    - ``max(1, |x[1]|, |alpha|)``: the size of the numbers it rounds: the coordinate along the axis, and the turn
      ``alpha = theta + nu (zeta + L / 2) / R``, at most ``|theta| + |nu| (|x[1]| + L / 2R)``. The turn's rounding
      swings the surface point about the axis by as much, and the stationarity's change per unit turn is at most the
      tangents' product.
    - ``1 + d / R``: the length of the residual, in radii, which that swing moves along the tangents. ``distance`` is
      ``d``, in metres.

    Judged this way, the corner of the strategy (nu near -4, the radius near 5e-4, the point near the end of the
    length and three radii out, where ``|x[1]|`` is about 21) is not mistaken for a failure of the solve: an absolute
    1e-13 was exceeded by about 2% of its draws."""
    turn = abs(x[0]) + abs(nu) * (abs(x[1]) + L / 2 / radius)
    scale = (1 + abs(nu)) ** 2 * (1 + epsilon) ** 2 * (1 + distance / radius) * max(1.0, abs(x[1]), turn)
    return STATIONARITY_ROUNDINGS * UNIT_ROUNDOFF * scale


def brute_force(point, nu, epsilon, starts, radius=R) -> float:
    """The signed distance from ``point`` to the unbounded screw: the nearest of a 720 x 721 grid of the surface over
    three radii either side of the point's cross-section, refined by BFGS, then solved to the stationarity conditions
    themselves (analytic, so no step of a finite difference leaves a bias); negative inside the cross-section.

    The solve is what makes the oracle's resolution a distance's and not a squared distance's. BFGS on the squared
    distance with finite-difference gradients stops where the forward difference's bias, about the step (1.5e-8)
    over two, leaves the minimiser: an error of about 1e-16 in the squared distance, in radii squared, which the
    square root turns into 1e-16 / (2 d) in the distance, d in radii: 6% of a distance of 3e-8 radii."""
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    zeta = point[0] + np.linspace(-3 * radius, 3 * radius, 721)
    grid_t, grid_z = np.meshgrid(theta, zeta, indexing="ij")
    distance = np.linalg.norm(surface(grid_t, grid_z, nu, epsilon, starts, radius) - point, axis=-1)
    i = np.unravel_index(np.argmin(distance), distance.shape)

    def squared(x):
        return np.sum((surface(x[0], x[1] * radius, nu, epsilon, starts, radius) - point) ** 2) / radius ** 2

    seed = minimize(squared, [grid_t[i], grid_z[i] / radius], method="BFGS", options={"gtol": 1e-14})
    arguments = (point, nu, epsilon, starts, radius)
    solved = root(stationarity, seed.x, args=arguments, method="hybr", tol=1e-15)
    nearest = squared(solved.x)
    # the solve moves the seed a hair (BFGS's bias) and does not leave its basin, nor end on a saddle or a maximum: its
    # distance has not risen above the seed's, beyond the rounding of a residual (each of its three components is a
    # difference of numbers of the point's or the surface's size, rounded at most twice, so the distance, in radii,
    # moves by at most four unit roundoffs of that size), and it is stationary to the evaluation's own rounding. Its own
    # success flag is not asked: it reports "not making good progress" once the solve has reached that rounding
    size = max(1.0, abs(solved.x[1]), float(np.max(np.abs(point))) / radius, 1 + epsilon)
    assert math.sqrt(nearest) <= math.sqrt(seed.fun) * (1 + 5e-10) + 4 * UNIT_ROUNDOFF * size, \
        (seed.x, seed.fun, solved.x, nearest)
    found = np.max(np.abs(stationarity(solved.x, *arguments)))
    floor = stationarity_floor(solved.x, nu, epsilon, radius, math.sqrt(nearest) * radius)
    assert found <= floor, (found, floor, stationarity(solved.x, *arguments))
    across = math.atan2(point[2], point[1]) - nu * (point[0] + L / 2) / radius
    inside = math.hypot(point[1], point[2]) < radius * (1 + epsilon * math.sin(starts * across))
    return (-1.0 if inside else 1.0) * math.sqrt(nearest) * radius


def screw(nu, epsilon, starts, radius=R):
    return {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": radius, "length": L,
            "wavenumber": nu, "modulation": epsilon, "starts": starts}


@settings(deadline=None)
@given(nu=st.floats(-4.0, 4.0), epsilon=st.floats(0.02, 0.5), starts=st.sampled_from((2, 3, 4)),
       radius=st.floats(5e-4, 3e-3), across=st.floats(0.0, 3.0), angle=st.floats(0.0, 2 * math.pi),
       along=st.floats(-1.0, 1.0))
# 6e-11 m (3e-8 radii) inside the surface: the oracle's BFGS on the squared distance stopped 7e-9 radians short, 6% of
# the distance off, where the screw's value (-5.88352562e-11 m, from a 60-digit solve) was right
@example(nu=5.08e-07, epsilon=0.0625, starts=2, radius=0.001953125, across=1.0, angle=0.0, along=-0.625)
# the corner of the strategy (|nu| near 4, radius near 5e-4, a point near the end of the length and three radii out),
# where an absolute 1e-13 on the stationarity was exceeded by about 2% of the draws: the rounding of its numbers, up to 21
# radii long and turned 80 radians
@example(nu=-3.915023864900873, epsilon=0.42663211825561725, starts=2, radius=0.000530892216212288,
         across=2.7548954049342114, angle=4.019461499491126, along=0.9880332153980829)
def test_the_screws_distance_is_the_brute_force_minimums(nu, epsilon, starts, radius, across, angle, along):
    point = np.array([along * L, across * radius * math.cos(angle), across * radius * math.sin(angle)])
    expected = brute_force(point, nu, epsilon, starts, radius)
    found = float(SCREW.signed_distance(screw(nu, epsilon, starts, radius), jnp.asarray(point)))
    assert abs(found - expected) <= 1e-9 * radius, (found, expected)


@pytest.mark.parametrize("offset", [-0.3, -1e-4, -3e-8, 0.0, 3e-8, 1e-7, 1e-4, 0.5])
def test_the_distance_oracle_resolves_a_point_a_hair_from_the_surface(offset):
    """The control for the oracle: with no modulation the screw is a circular cylinder, whose signed distance is
    exact, ``hypot(y, z) - radius``. The oracle returns it within 1e-9 radii at an offset of 3e-8 radii, where BFGS on
    the squared distance alone, with finite-difference gradients, was off by 2.6e-9 radii (1.3e-8 on the surface), and
    failed this bound at three of the eight offsets."""
    radius, point = 1.56e-3, np.array([0.3 * L, 0.0, 0.0])
    point[1:] = (radius * (1 + offset)) * np.array([math.cos(0.7), math.sin(0.7)])
    assert abs(brute_force(point, 1.3, 0.0, 3, radius) - radius * offset) <= 1e-9 * radius


@pytest.mark.parametrize("offset", [-3e-8, 3e-8, 1e-7])
@pytest.mark.parametrize(("nu", "epsilon", "starts"), [(2.33, 0.33, 2), (-3.7, 0.45, 3), (1.1, 0.2, 4)])
def test_the_distance_oracle_resolves_a_point_a_hair_from_a_modulated_surface(nu, epsilon, starts, offset):
    """The control for the oracle's modulation terms, which the cylinder's has none of: a point ``offset`` radii from
    the surface along its normal (from central differences of the surface, independent of the oracle's tangents) is
    ``offset`` radii from it, and the oracle returns that within 1e-9 radii. The squared-distance BFGS alone, with
    finite-difference gradients, was off by 1.7e-9 to 2.3e-7 radii at the worst of six feet in each of these nine cases
    and failed this bound at eight of them (measured against the oracle it replaced)."""
    radius, step = 1.56e-3, 1e-5
    rng = np.random.default_rng(5)
    for theta, zeta in zip(rng.uniform(0, 2 * np.pi, 6), rng.uniform(-3, 3, 6) * radius + rng.uniform(-0.5, 0.5, 6) * L):
        def foot(t, z):
            return surface(t, z, nu, epsilon, starts, radius)
        along_theta = (foot(theta + step, zeta) - foot(theta - step, zeta)) / (2 * step)
        along_zeta = (foot(theta, zeta + step * radius) - foot(theta, zeta - step * radius)) / (2 * step * radius)
        normal = np.cross(along_theta, along_zeta)
        point = foot(theta, zeta) + offset * radius * normal / np.linalg.norm(normal)
        assert abs(abs(brute_force(point, nu, epsilon, starts, radius)) - abs(offset) * radius) <= 1e-9 * radius


def test_the_02_lines_surface_points_moved_half_its_length_lie_on_the_screw():
    """The 0.2 line's ``dejongh_umr_surface`` vertices for FL-9 (its zeta from 0 to L), each moved -L/2 along the axis:
    the kind is that body, re-centred, with no turn about its axis."""
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    zeta = np.linspace(0, L, 12)
    t, z = np.meshgrid(theta, zeta, indexing="ij")
    rho = R * (1 + 0.33 * np.sin(2 * t))
    alpha = 2.33 * z / R + t
    vertices = np.stack([z - L / 2, rho * np.cos(alpha), rho * np.sin(alpha)], -1).reshape(-1, 3)
    distance = jax.vmap(lambda p: SCREW.signed_distance(screw(2.33, 0.33, 2), p))(jnp.asarray(vertices))
    assert float(jnp.max(jnp.abs(distance))) <= 1e-17


#: Pi to fifty digits, for the exact turn's reduction by whole turns, far beyond float64's.
PI = Fraction("3.14159265358979323846264338327950288419716939937510")
#: The turn's cosine and sine against the exact turn's, in units in the last place of one: measured at most 0.5 (float64)
#: and 0.81 (float32) over wide draws of points, lengths and wavenumbers; the bound is twice a unit, since the
#: reference's reduced turn is itself rounded once before its cosine.
TURN_ULPS = 2


def exact_turn(nu, length, radius, along) -> tuple[float, float]:
    """The cosine and sine of ``nu (along + length / 2) / radius`` in exact arithmetic on the float inputs, reduced by
    whole turns before it is rounded."""
    turn = Fraction(float(nu)) * (Fraction(float(along)) + Fraction(float(length)) / 2) / Fraction(float(radius))
    turn -= round(turn / (2 * PI)) * 2 * PI
    return math.cos(float(turn)), math.sin(float(turn))


def turn_errors(turned, dtype, factor) -> list[float]:
    """The largest error of ``turned(parameters, along)`` (a cosine and sine) against the exact turn, in units in the
    last place of one, for each wavenumber at FL-9's radius and ``factor`` times its length, at both ends and 99 points
    drawn along it with every bit of the working precision (seeded): points on the grid of the length's last place,
    as a ``linspace`` gives, make ``along + length / 2`` exact and hide the sum's rounding."""
    length = dtype(L * factor)
    draws = np.random.default_rng(0).uniform(-0.5, 0.5, 99) * float(length)
    along = np.concatenate([[-length / 2, length / 2], draws]).astype(dtype)
    found = []
    for nu in (-4.0, 2.33, 4.0):
        held = {"radius": dtype(R), "length": length, "wavenumber": dtype(nu)}
        cos, sin = (np.asarray(part) for part in jax.jit(lambda a, held=held: turned(held, a))(jnp.asarray(along)))
        exact = np.array([exact_turn(held["wavenumber"], length, held["radius"], a) for a in along])
        found.append(float(np.max(np.abs(np.stack([cos, sin], axis=-1) - exact))) / float(np.finfo(dtype).eps))
    return found


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("factor", [1, 10, 1000])
def test_the_screws_turn_rounds_at_its_own_scale_however_far_along_the_axis(dtype, factor):
    """The cross-section's turn at a point, carried to twice the working precision, is its exact turn's to about one
    unit in the last place, at FL-9's length and a thousand times it (where the turn is some 19,000 radians)."""
    assert max(turn_errors(SCREW.turn, dtype, factor)) <= TURN_ULPS


def carried_turn(parameters, along, *, compensated_sum=True, second_order=True):
    """The screw's turn as it is written, with the sum ``along + length / 2``'s low part, or the low part's cosine to
    second order, left out: the controls' defects. Nothing else differs from ``Screw.turn``."""
    dtype = jnp.result_type(along, parameters["radius"], parameters["wavenumber"], parameters["length"])
    along = jnp.asarray(along, dtype=dtype)
    radius, nu = jnp.asarray(parameters["radius"], dtype=dtype), jnp.asarray(parameters["wavenumber"], dtype=dtype)
    rate = nu / radius
    product, error = exact_product(rate, radius)
    rate_low = ((nu - product) - error) / radius
    offset, offset_low = exact_sum(along, jnp.asarray(parameters["length"], dtype=dtype) / 2.0)
    if not compensated_sum:
        offset_low = jnp.zeros_like(offset_low)
    turn, error = exact_product(rate, offset)
    low = error + (rate * offset_low + rate_low * offset)
    cos, sin = jnp.cos(turn), jnp.sin(turn)
    kept = 1.0 - low * low / 2.0 if second_order else jnp.ones_like(low)
    return cos * kept - sin * low, sin * kept + cos * low


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_the_controls_turn_with_nothing_left_out_is_the_screws(dtype):
    """The controls below leave a part out of a copy of the turn; this keeps the copy the screw's own, bit for bit."""
    length = dtype(L * 1000)
    held = {"radius": dtype(R), "length": length, "wavenumber": dtype(2.33)}
    along = jnp.asarray((np.random.default_rng(1).uniform(-0.5, 0.5, 64) * float(length)).astype(dtype))
    copy = jax.jit(lambda a: carried_turn(held, a))(along)
    screws = jax.jit(lambda a: SCREW.turn(held, a))(along)
    for found, expected in zip(copy, screws):
        assert np.array_equal(np.asarray(found), np.asarray(expected))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_control_the_turn_taken_plainly_rounds_at_its_size(dtype):
    """The turn as the screw took it before, ``wavenumber (along + length / 2) / radius`` in the working precision:
    it rounds at the turn's size, and fails, on every run (the draws are seeded)."""

    def plain(held, along):
        turn = held["wavenumber"] * (along + held["length"] / 2) / held["radius"]
        return jnp.cos(turn), jnp.sin(turn)

    assert max(turn_errors(plain, dtype, 1000)) > 100 * TURN_ULPS


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_control_the_turn_without_the_sums_low_part_fails(dtype):
    """The sum ``along + length / 2`` taken without its rounding error: a point off the grid of the length's last place
    loses it, and the turn errs by thousands of units at a thousand times FL-9's length (5,127 float64, 5,118 float32
    measured), on every run. The normals check cannot see it, since the helix's samples and the shape share the turn."""
    uncompensated = lambda held, along: carried_turn(held, along, compensated_sum=False)  # noqa: E731
    assert max(turn_errors(uncompensated, dtype, 1000)) > 100 * TURN_ULPS


def test_control_the_turn_without_the_low_parts_second_order_cosine_fails():
    """``cos`` of the low part to first order only: in float32 at some 19,000 radians the low part is 1e-3, and its
    square, 5e-7, is above a unit in the last place (11.1 measured, on every run). Float64's is too small to show."""
    first_order = lambda held, along: carried_turn(held, along, second_order=False)  # noqa: E731
    assert max(turn_errors(first_order, np.float32, 1000)) > 2 * TURN_ULPS


def test_the_helixs_tangent_is_its_thread_across_its_normal():
    helix = fl9()
    points = helix.samples(settings_of(), "lateral", (24, 6))
    tangents = np.asarray(jax.jit(jax.vmap(lambda p: helix.tangent(settings_of(), p)))(points.points))
    normals = np.asarray(points.normals)
    assert np.all(np.abs(np.linalg.norm(tangents, axis=1) - 1) < 1e-14)
    assert np.all(np.abs(np.sum(tangents * normals, axis=1)) < 1e-14) and np.all(tangents[:, 0] > 0)


# -- the plan's refusals ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["extent", "radius", "length", "helix_wavenumber"])
def test_a_body_value_its_hull_kind_gives_is_refused(name):
    experiment = script_a()
    experiment.declare(f"umr.{name}", mm(1.0) if name != "helix_wavenumber" else dimensionless(2.0), reason="again")
    assert "value.from-kind" in codes(experiment)


CentreContact, _ = KEPT["contact", "0.1.0"]


def experiment_with_parts(parts, *, extent: bool = False, drag: bool = False):
    """A body of ``parts`` (each outside every other at the test's density) in A's tube, held by the contact 0.1.0,
    the reader of a body's extent (0.2.0 holds the hull's points and reads none), with the surrogate drag where
    ``drag``."""
    experiment = Experiment("parts", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    experiment.add_medium(Medium("water", fills=tube.lumen))
    experiment.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0))), parts=parts))
    for key in parts:
        if parts[key].inside is None:
            experiment.declare(f"umr.parts.{key}.density", kg_per_m3(1410.0), reason="the test's")
    if extent:
        experiment.declare("umr.extent", mm(2.07), reason="the test's")
    experiment.attach(CentreContact.Projection(margin=mm(0.1)), body="umr", surface=tube.inner_surface)
    if drag:
        experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
        experiment.attach(Drag.Surrogate(weights=data_asset("dejongh_mlp_v2")), body="umr", medium="water")
    return experiment


def test_a_hull_of_a_kind_with_no_reach_takes_a_declared_extent_and_refuses_none():
    ellipsoid = {"shell": Part(Ellipsoid(semi_axes=mm((1.56, 1.56, 1.56))))}
    experiment_with_parts(ellipsoid, extent=True).plan()
    assert "parts.extent" in codes(experiment_with_parts(ellipsoid))


def test_two_parts_outside_one_another_give_no_extent():
    two = {"a": Part(Sphere(radius=mm(1.0)), at=mm((-1.0, 0.0, 0.0))),
           "b": Part(Sphere(radius=mm(1.0)), at=mm((1.0, 0.0, 0.0)))}
    assert "parts.extent" in codes(experiment_with_parts(two))


def test_the_extent_of_a_sphere_or_a_capsule_is_its_radius():
    for kind in (Sphere(radius=mm(1.2)), Capsule(radius=mm(1.2), height=mm(3.0), axis=(1.0, 0.0, 0.0))):
        plans = experiment_with_parts({"shell": Part(kind)}).plan()
        (extent,) = [line for line in plans.values if line.path == "umr.extent"]
        assert extent.value == pytest.approx(1.2e-3, rel=1e-15) and extent.source == "derived from its parts"


@pytest.mark.parametrize("kind", [Sphere(radius=mm(1.56)), Capsule(radius=mm(1.56), height=mm(4.0),
                                                                    axis=(1.0, 0.0, 0.0))])
def test_the_de_jongh_surrogate_refuses_a_body_of_any_other_kind(kind):
    assert "drag.body-kind" in codes(experiment_with_parts({"shell": Part(kind)}, drag=True))


def test_the_de_jongh_surrogate_refuses_a_helix_of_another_modulation_or_count_of_starts():
    other = Screw(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.3), starts=3)
    found = codes(experiment_with_parts({"shell": Part(other)}, drag=True))
    assert found == {"drag.outside-training"}


def test_the_de_jongh_surrogate_refuses_a_helix_moved_in_the_body():
    """The surrogate takes the helix's centroid to be the body's origin: a helix 1 mm along the body's x is refused."""
    assert "drag.body-kind" in codes(experiment_with_parts({"shell": Part(fl9(), at=mm((1.0, 0.0, 0.0)))}, drag=True))


def turned_shell(path: Path, radians: float) -> Path:
    """An asset of a body prim ``/UMR`` holding a part ``Shell``, turned ``radians`` about the body's x."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/UMR")
    stage.SetDefaultPrim(body.GetPrim())
    UsdGeom.Xformable(UsdGeom.Xform.Define(stage, "/UMR/Shell")).AddRotateXOp().Set(math.degrees(radians))
    path.write_text(stage.GetRootLayer().ExportToString(), encoding="utf-8")
    return path


def bound_helix(path: Path) -> Experiment:
    """A body bound to the asset's prim, its shell FL-9's helix, in A's tube with the surrogate drag."""
    scene = Scene.new(up=Z, units="m")
    scene.reference(asset(path), at="/World/UMR")
    experiment = Experiment("turned", scene=scene, gravity=m_per_s2(9.81), dt=ms(0.5))
    tube = experiment.add_structure(Structure("tube", geometry=Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6),
                                                                    length=mm(100.0))))
    experiment.add_medium(Medium("water", fills=tube.lumen))
    umr = experiment.add_body(Body(prim=scene["/World/UMR"], dynamics=RigidInertial(), parts={"Shell": Part(fl9())},
                                   initial=State(position=mm((0.0, 0.0, -1.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                                 velocity=m_per_s((0.0, 0.0, 0.0)),
                                                 angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare(f"{umr.name}.parts.Shell.density", kg_per_m3(1410.0), reason="the test's")
    experiment.declare(f"{umr.name}.inertia", kg_m2((1e-10, 1e-10, 1e-10)),
                       reason="the test's: FL-9's helix has a product of inertia across its axis")
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.attach(Contact.Projection(margin=mm(0.1)), body=umr, surface=tube.inner_surface)
    experiment.attach(Drag.Surrogate(weights=data_asset("dejongh_mlp_v2")), body=umr, medium="water")
    return experiment


def test_the_de_jongh_surrogate_refuses_a_helix_turned_in_the_body(tmp_path):
    """The surrogate takes the helix's axis to be the body's x: a helix turned 0.3 rad about it by the asset's own
    transform is refused; unturned, the same body is not."""
    assert "drag.body-kind" in codes(bound_helix(turned_shell(tmp_path / "turned.usda", 0.3)))
    bound_helix(turned_shell(tmp_path / "unturned.usda", 0.0)).plan()


class Tapered(Cylinder):
    """A shape with an axis and a parameter a circular tube does not have: a lab's, say."""

    parameter_names = ("centre", "axis", "radius", "taper")


def test_the_drag_refuses_a_tube_whose_shape_has_an_axis_but_not_a_circular_tubes_parameters(monkeypatch):
    """The drag's node reads the tube's shape through ``vessel_<name>`` for each of its parameters, which are a
    circular tube's: a shape with an axis and other parameters is refused when planned. Its control: the cylinder's
    own parameters pass."""
    parameters = {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0), "radius": 3.175e-3, "taper": 0.0}
    monkeypatch.setattr(hydrodynamic, "shape_named", lambda name: Tapered() if name == "tapered" else Cylinder())
    found = hydrodynamic.uniform_tube(Region("tube.lumen", "tapered", parameters), "water")
    assert isinstance(found, Problem) and found.code == "drag.no-axis" and "not a circular tube's" in found.message
    tube = hydrodynamic.uniform_tube(Region("tube.lumen", "cylinder", parameters), "water")
    assert not isinstance(tube, Problem) and tube.radius == pytest.approx(3.175e-3, rel=1e-15)

