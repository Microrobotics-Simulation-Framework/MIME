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
from hypothesis import given, settings
from hypothesis import strategies as st
from pxr import Usd, UsdGeom
from scipy.optimize import minimize
from script_a import VOLUME, fl9, script_a

import mime.interactions.hydrodynamic as hydrodynamic
from mime import Body, Experiment, Medium, State, Structure, asset
from mime.build import BuildRefused
from mime.core import DEJONGH_SCREW, Cylinder, jnp
from mime.data import asset as data_asset
from mime.registry import Problem, Region
from mime.dynamics import RigidInertial
from mime.interactions import Contact, Drag
from mime.scene import Capsule, DeJonghHelix, Scene, SolidEllipsoid, Sphere, Tube, X, Z
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


#: Pi to fifty digits, for the exact turn's reduction by whole turns, far beyond float64's.
PI = Fraction("3.14159265358979323846264338327950288419716939937510")
#: The turn's cosine and sine against the exact turn's, in units in the last place of one: about one each for the
#: carried turn and the reference's own rounding (its reduced turn rounded once, then a cosine).
TURN_ULPS = 4


def exact_turn(nu, length, radius, along) -> tuple[float, float]:
    """The cosine and sine of ``nu (along + length / 2) / radius`` in exact arithmetic on the float inputs, reduced by
    whole turns before it is rounded."""
    turn = Fraction(float(nu)) * (Fraction(float(along)) + Fraction(float(length)) / 2) / Fraction(float(radius))
    turn -= round(turn / (2 * PI)) * 2 * PI
    return math.cos(float(turn)), math.sin(float(turn))


def turn_errors(turned, dtype, factor) -> list[float]:
    """The largest error of ``turned(parameters, along)`` (a cosine and sine) against the exact turn, in units in the
    last place of one, for each wavenumber at FL-9's radius and ``factor`` times its length, at 101 points along it."""
    length = dtype(L * factor)
    found = []
    for nu in (-4.0, 2.33, 4.0):
        held = {"radius": dtype(R), "length": length, "wavenumber": dtype(nu)}
        along = np.linspace(-length / 2, length / 2, 101, dtype=dtype)
        cos, sin = (np.asarray(part) for part in jax.jit(lambda a, held=held: turned(held, a))(jnp.asarray(along)))
        exact = np.array([exact_turn(held["wavenumber"], length, held["radius"], a) for a in along])
        found.append(float(np.max(np.abs(np.stack([cos, sin], axis=-1) - exact))) / float(np.finfo(dtype).eps))
    return found


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("factor", [1, 10, 1000])
def test_the_screws_turn_rounds_at_its_own_scale_however_far_along_the_axis(dtype, factor):
    """The cross-section's turn at a point, carried to twice the working precision, is its exact turn's to about one
    unit in the last place, at FL-9's length and a thousand times it (where the turn is some 19,000 radians)."""
    assert max(turn_errors(DEJONGH_SCREW.turn, dtype, factor)) <= TURN_ULPS


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_control_the_turn_taken_plainly_rounds_at_its_size(dtype):
    """The turn as the screw took it before, ``wavenumber (along + length / 2) / radius`` in the working precision:
    it rounds at the turn's size, and fails, on every run (nothing is drawn)."""

    def plain(held, along):
        turn = held["wavenumber"] * (along + held["length"] / 2) / held["radius"]
        return jnp.cos(turn), jnp.sin(turn)

    assert max(turn_errors(plain, dtype, 1000)) > 100 * TURN_ULPS


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
        experiment.attach(Drag.Surrogate(weights=data_asset("dejongh_mlp_v2")), body="umr", medium="water")
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

