"""The solid kinds of 7d against what they are: the de Jongh helix (de Jongh et al. 2025, Eq. 1), the sphere and the
capsule.

**The oracle.** The helix's volume, ``pi R^2 L (1 + eps^2 / 2)``, equals script A's declared volume (FL-9, the 0.2
line's ``(410 + 1000) kg/m^3`` body's) to two units in the last place. The screw's signed distance, from its root find,
equals a brute-force minimum (a dense grid of the surface, then a local refinement in float64 with SciPy) at points
drawn round FL-9 and at random screws, out to three radii from the axis, and at the 0.2 line's own surface points,
which lie on it: the 0.2 body moved half its length.

The plan's refusals of a body value its hull kind gives, of an extent no hull gives, and of a drag on a body that is
not the surrogate's family are named here."""

import math

import jax
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize
from script_a import VOLUME, fl9, script_a

from mime import Body, Experiment, Medium, State, Structure
from mime.build import BuildRefused
from mime.core import DEJONGH_SCREW, jnp
from mime.data import asset
from mime.dynamics import RigidInertial
from mime.interactions import Contact, Drag
from mime.scene import Capsule, DeJonghHelix, Scene, SolidEllipsoid, Sphere, Tube, X, Z
from mime.things import Part
from mime.units import dimensionless, kg_per_m3, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

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
    helix = DeJonghHelix(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.33))
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
    helix = DeJonghHelix(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.33), starts=starts)
    assert any("starts is" in line for line in helix.setting_problems(settings_of()))


# -- the screw's distance against a brute-force minimum -------------------------------------------------------------

def surface(theta, zeta, nu, epsilon, starts, radius=R, length=L):
    """De Jongh's Eq. 1 with the axis along x and the body centred on the origin: the 0.2 line's surface, its zeta
    from 0 to L, moved L / 2 along the axis."""
    rho = radius * (1 + epsilon * np.sin(starts * theta))
    alpha = theta + nu * (zeta + length / 2) / radius
    return np.stack([zeta, rho * np.cos(alpha), rho * np.sin(alpha)], -1)


def brute_force(point, nu, epsilon, starts, radius=R) -> float:
    """The signed distance from ``point`` to the unbounded screw: the nearest of a 720 x 721 grid of the surface over
    three radii either side of the point's cross-section, refined by BFGS; negative inside the cross-section."""
    theta = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    zeta = point[0] + np.linspace(-3 * radius, 3 * radius, 721)
    grid_t, grid_z = np.meshgrid(theta, zeta, indexing="ij")
    distance = np.linalg.norm(surface(grid_t, grid_z, nu, epsilon, starts, radius) - point, axis=-1)
    i = np.unravel_index(np.argmin(distance), distance.shape)

    def squared(x):
        return np.sum((surface(x[0], x[1] * radius, nu, epsilon, starts, radius) - point) ** 2) / radius ** 2

    found = minimize(squared, [grid_t[i], grid_z[i] / radius], method="BFGS", options={"gtol": 1e-14})
    across = math.atan2(point[2], point[1]) - nu * (point[0] + L / 2) / radius
    inside = math.hypot(point[1], point[2]) < radius * (1 + epsilon * math.sin(starts * across))
    return (-1.0 if inside else 1.0) * math.sqrt(found.fun) * radius


def screw(nu, epsilon, starts, radius=R):
    return {"centre": (0.0, 0.0, 0.0), "orientation": (1.0, 0.0, 0.0, 0.0), "radius": radius, "length": L,
            "wavenumber": nu, "modulation": epsilon, "starts": starts}


@settings(deadline=None)
@given(nu=st.floats(-4.0, 4.0), epsilon=st.floats(0.02, 0.5), starts=st.sampled_from((2, 3, 4)),
       radius=st.floats(5e-4, 3e-3), across=st.floats(0.0, 3.0), angle=st.floats(0.0, 2 * math.pi),
       along=st.floats(-1.0, 1.0))
def test_the_screws_distance_is_the_brute_force_minimums(nu, epsilon, starts, radius, across, angle, along):
    point = np.array([along * L, across * radius * math.cos(angle), across * radius * math.sin(angle)])
    expected = brute_force(point, nu, epsilon, starts, radius)
    found = float(DEJONGH_SCREW.signed_distance(screw(nu, epsilon, starts, radius), jnp.asarray(point)))
    assert abs(found - expected) <= 1e-9 * radius, (found, expected)


def test_the_02_lines_surface_points_moved_half_its_length_lie_on_the_screw():
    """The 0.2 line's ``dejongh_umr_surface`` vertices for FL-9 (its zeta from 0 to L), each moved -L/2 along the axis:
    the kind is that body, re-centred, with no turn about its axis."""
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    zeta = np.linspace(0, L, 12)
    t, z = np.meshgrid(theta, zeta, indexing="ij")
    rho = R * (1 + 0.33 * np.sin(2 * t))
    alpha = 2.33 * z / R + t
    vertices = np.stack([z - L / 2, rho * np.cos(alpha), rho * np.sin(alpha)], -1).reshape(-1, 3)
    distance = jax.vmap(lambda p: DEJONGH_SCREW.signed_distance(screw(2.33, 0.33, 2), p))(jnp.asarray(vertices))
    assert float(jnp.max(jnp.abs(distance))) <= 1e-17


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


def experiment_with_parts(parts, *, extent: bool = False, drag: bool = False):
    """A body of ``parts`` (each outside every other at the test's density) in A's tube, held by the contact, with the
    surrogate drag where ``drag``."""
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
    experiment.attach(Contact.Projection(margin=mm(0.1)), body="umr", surface=tube.inner_surface)
    if drag:
        experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
        experiment.attach(Drag.Surrogate(weights=asset("dejongh_mlp_v2")), body="umr", medium="water")
    return experiment


def test_a_hull_of_a_kind_with_no_reach_takes_a_declared_extent_and_refuses_none():
    ellipsoid = {"shell": Part(SolidEllipsoid(semi_axes=mm((1.56, 1.56, 1.56))))}
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
    other = DeJonghHelix(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.3), starts=3)
    found = codes(experiment_with_parts({"shell": Part(other)}, drag=True))
    assert found == {"drag.outside-training"}
