"""Shapes and the tube: the cylinder's signed distance and outward normal by their closed forms, finite with finite
derivatives on its axis, where the normal is not defined, and their derivatives off it against central differences;
on any axis, within the cylinder's resolution, the values and derivatives its record states; the tube's inner surface
as the cylinder of its inner radius; a geometry kind checked against the shape protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import find, given, settings
from hypothesis import strategies as st

from mime.core import AXIS_ULPS, CYLINDER, Shape, shape_named
from mime.core import _geometry as geometry
from mime.scene import Tube, X
from mime.units import mm


@pytest.fixture(autouse=True, scope="module")
def x64():
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


AXIS = np.array([0.0, 0.6, 0.8])
SURFACE = {"centre": (1e-3, -2e-3, 0.5e-3), "axis": tuple(AXIS * 2.0), "radius": 3e-3}


def test_the_cylinders_distance_and_normal_are_their_closed_forms():
    point = np.array([4e-3, 1e-3, -1e-3])
    offset = point - np.asarray(SURFACE["centre"])
    radial = offset - (offset @ AXIS) * AXIS
    distance = float(CYLINDER.signed_distance(SURFACE, jnp.asarray(point)))
    normal = np.asarray(CYLINDER.outward_normal(SURFACE, jnp.asarray(point)))
    assert abs(distance - (np.linalg.norm(radial) - 3e-3)) < 1e-18
    assert np.allclose(normal, radial / np.linalg.norm(radial), atol=1e-15)
    assert np.allclose(np.asarray(CYLINDER.axis(SURFACE)), AXIS, atol=1e-15)


def test_on_its_axis_the_cylinder_is_finite_with_finite_derivatives():
    centre = jnp.asarray(SURFACE["centre"])
    assert float(CYLINDER.signed_distance(SURFACE, centre)) == -3e-3
    assert np.array_equal(np.asarray(CYLINDER.outward_normal(SURFACE, centre)), np.zeros(3))
    for jacobian in (jax.grad(lambda x: CYLINDER.signed_distance(SURFACE, x))(centre),
                     jax.jacfwd(lambda x: CYLINDER.outward_normal(SURFACE, x))(centre)):
        assert np.all(np.isfinite(np.asarray(jacobian)))


def test_the_tubes_inner_surface_is_the_cylinder_of_its_inner_radius():
    tube = Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6), length=mm(100))
    settings = {"axis": (1.0, 0.0, 0.0), "inner_radius": 3.175e-3, "wall": 1.6e-3, "length": 0.1,
                "centre": (0.0, 0.0, 0.0)}
    assert Tube.surfaces == {"inner_surface": "cylinder"} and shape_named("cylinder") is CYLINDER
    assert tube.surface_parameters("inner_surface", settings) == {"centre": (0.0, 0.0, 0.0), "axis": (1.0, 0.0, 0.0),
                                                                  "radius": 3.175e-3}
    assert tube.direction("axis", settings) == (1.0, 0.0, 0.0)
    assert tube.setting_problems(settings) == []
    assert isinstance(CYLINDER, Shape)


def test_the_tubes_lumen_is_the_cylinder_with_its_axis_and_a_point_on_it():
    """The region a medium fills, as a law reads it through the shape protocol: the axis, a point on it, and the
    radius as minus the signed distance there."""
    tube = Tube(axis=+X, inner_radius=mm(3.175), wall=mm(1.6), length=mm(100))
    settings = {"axis": (1.0, 0.0, 0.0), "inner_radius": 3.175e-3, "wall": 1.6e-3, "length": 0.1,
                "centre": (0.0, 0.0, 0.0)}
    assert Tube.region_shapes == {"lumen": "cylinder"}
    lumen = tube.region_parameters("lumen", settings)
    point = CYLINDER.axis_point(lumen)
    assert np.array_equal(np.asarray(point), np.zeros(3))
    assert float(CYLINDER.signed_distance(lumen, point)) == -3.175e-3
    assert np.allclose(np.asarray(CYLINDER.axis_point(SURFACE)), SURFACE["centre"], atol=0.0)
    with pytest.raises(KeyError):
        tube.region_parameters("wall", settings)


# -- the tube's derivative, off its axis ------------------------------------------------------------------------------
#
# The tube's switch on its axis is the cylinder's (cylinder.axis): on the axis the distance from it has a cone
# point and the normal no limit, so neither has a derivative there; off the axis both are smooth, and their
# derivatives exact. The property compares JAX's derivatives of the signed distance and the outward normal, with
# respect to the point and to each parameter, with central finite differences, in float64, at points drawn off the
# axis by 1e-4 to 2 radii. Each finite difference moves the geometry by a step ``H`` times the point's distance from
# the axis, so it is judged on the change that move makes (the distance's, or the normal's over that distance). Its
# error is the rounding of the values, about eps times the drawn lengths over that move, plus the difference's own
# truncation, H squared: ``H`` is chosen for each draw to make their sum least, and the resolution is a multiple of
# that sum, from 7e-10 to 2.2e-5 of the derivative over the drawn range (8e-8 at the median draw; 1.6e-3 with the
# fixed H = 1e-2 this replaced). The control: a derivative off by two and a half times the draw's resolution, of either
# function, planted in a copy of the cylinder, fails on every draw.

#: The finite differences' resolution, in units of their rounding and truncation at the step chosen: over 20000 draws
#: the exact derivative's worst difference was 1.8 of these units, a ninth of it.
RESOLUTION = 16.0
#: The control's plant: a derivative off by this many times the draw's resolution. A derivative's error moves the
#: largest difference by at least 1/sqrt(3) of itself (the normal's, across the axis), so an error above about twice
#: the resolution is resolved.
PLANTED = 2.5
VARIABLES = ("point", "centre", "axis", "radius")


@st.composite
def tube_and_point(draw: st.DrawFn) -> dict[str, np.ndarray]:
    """A tube at physical scales (radius 0.1 to 10 mm, centre and point within 1 cm of the origin, any axis direction
    with a length from 0.5 to 2), and a point off its axis by 1e-4 to 2 radii, on any side."""
    radius = draw(st.floats(1e-4, 1e-2))
    centre = np.array([draw(st.floats(-1e-2, 1e-2)) for _ in range(3)])
    polar, azimuth, side = draw(st.floats(0.0, np.pi)), draw(st.floats(0.0, 2 * np.pi)), draw(st.floats(0.0, 2 * np.pi))
    unit = np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])
    first = np.cross(unit, np.eye(3)[int(np.argmin(np.abs(unit)))])
    first /= np.linalg.norm(first)
    across = np.cos(side) * first + np.sin(side) * np.cross(unit, first)
    offset = radius * 10 ** draw(st.floats(-4.0, np.log10(2.0)))
    point = centre + draw(st.floats(-1e-2, 1e-2)) * unit + offset * across
    return {"point": point, "centre": centre, "axis": unit * draw(st.floats(0.5, 2.0)), "radius": np.array(radius)}


def off_by_the_plant(function):
    """``function`` of the parameters and the point, with its derivative off by the fraction ``parameters["plant"]``:
    the control's plant."""
    planted = jax.custom_jvp(function)

    @planted.defjvp
    def _(primals, tangents):
        value, tangent = jax.jvp(function, primals, tangents)
        return value, (1.0 + primals[0]["plant"]) * tangent

    return planted


class Planted:
    """The cylinder with the derivative of one of its functions planted off."""

    def __init__(self, which: str) -> None:
        self.signed_distance = off_by_the_plant(CYLINDER.signed_distance) if which == "distance" else \
            CYLINDER.signed_distance
        self.outward_normal = off_by_the_plant(CYLINDER.outward_normal) if which == "normal" else \
            CYLINDER.outward_normal


def evaluated(shape):
    """Both functions of ``shape`` and their derivatives with respect to each variable, compiled once; the plant's
    fraction is an argument, not a variable."""

    def both(x, plant):
        parameters = {"centre": x["centre"], "axis": x["axis"], "radius": x["radius"], "plant": plant}
        return shape.signed_distance(parameters, x["point"]), shape.outward_normal(parameters, x["point"])

    return jax.jit(jax.vmap(both, in_axes=(0, None))), jax.jit(jax.jacfwd(both))


SHAPES = {"exact": evaluated(CYLINDER), "distance": evaluated(Planted("distance")),
          "normal": evaluated(Planted("normal"))}


def worst(shape: str, x: dict[str, np.ndarray]) -> float:
    """The largest difference between JAX's derivative and the central finite difference over every variable's
    components, for both functions, in units of the resolution: above 1 is a wrong derivative."""
    values, jacobians = SHAPES[shape]
    a = x["axis"] / np.linalg.norm(x["axis"])
    relative = x["point"] - x["centre"]
    along = float(relative @ a)
    distance = float(np.linalg.norm(relative - along * a))
    lengths = max(float(np.linalg.norm(x["point"])), float(np.linalg.norm(x["centre"])), float(x["radius"]))
    rounding = float(np.finfo(np.float64).eps) * lengths / distance
    h = (rounding / 2) ** (1 / 3)
    resolution = RESOLUTION * (rounding / h + h * h)
    plant = jnp.asarray(PLANTED * resolution)
    steps = {"point": h * distance, "centre": h * distance, "radius": h * distance,
             "axis": h * distance * float(np.linalg.norm(x["axis"])) / max(abs(along), distance)}
    moved = []
    for name in VARIABLES:
        for i in range(x[name].size):
            for sign in (1.0, -1.0):
                held = {key: jnp.asarray(value) for key, value in x.items()}
                held[name] = jnp.asarray((x[name].ravel() + sign * steps[name] * np.eye(x[name].size)[i])
                                         .reshape(x[name].shape))
                moved.append(held)
    batch = {key: jnp.stack([m[key] for m in moved]) for key in x}
    distances, normals = (np.asarray(v) for v in values(batch, plant))
    derivative = jacobians({key: jnp.asarray(value) for key, value in x.items()}, plant)
    found, row = 0.0, 0
    for name in VARIABLES:
        for i in range(x[name].size):
            for function, scale, at in ((distances, h * distance, 0), (normals, h, 1)):
                difference = (function[row] - function[row + 1]) / 2
                predicted = np.asarray(derivative[at][name]).reshape(function.shape[1:] + (x[name].size,))[..., i] \
                    * steps[name]
                found = max(found, float(np.linalg.norm(difference - predicted)) / scale / resolution)
            row += 2
    return found


@settings(deadline=None)
@given(tube_and_point())
def test_off_its_axis_the_tubes_derivatives_are_those_of_its_values(x):
    assert worst("exact", x) <= 1.0


@pytest.mark.parametrize("which", ["distance", "normal"])
@settings(deadline=None)
@given(x=tube_and_point())
def test_a_derivative_planted_two_and_a_half_resolutions_off_fails_on_every_draw(which, x):
    assert worst(which, x) > 1.0


# -- the tube's axis: the switch --------------------------------------------------------------------------------------
#
# On the axis the distance from it has a cone point and the normal no limit, so neither has a derivative there. The
# cylinder takes a point within AXIS_ULPS units in the last place of its radius of the axis as on it, whatever the
# axis's direction, and returns there what the record says: the distance as minus the radius, its derivative zero but
# in the radius, and the normal as zero, with the derivative of the offset from the axis. A point put on an axis off the
# coordinate axes computes an offset of a few units in the last place of its distance from the centre, not zero. So the
# property draws such axes (the review's (1, 1, 1) and (0.3, -0.5, 0.8), and any direction), with points put on them
# and points within half the resolution of them. The control: the cylinder with its resolution removed, on points put on
# tilted axes, departs from the record.

TILTED = ((1.0, 1.0, 1.0), (0.3, -0.5, 0.8))
EPS = float(np.finfo(np.float64).eps)


@st.composite
def tube_and_point_on_its_axis(draw: st.DrawFn, within: bool = True) -> dict[str, np.ndarray]:
    """A tube as :func:`tube_and_point` draws one, its axis one of ``TILTED`` or any direction, and a point put on its
    axis (exactly, but for rounding), or, with ``within``, off it by up to half the cylinder's resolution."""
    radius = draw(st.floats(1e-4, 1e-2))
    centre = np.array([draw(st.floats(-1e-2, 1e-2)) for _ in range(3)])
    polar, azimuth, side = draw(st.floats(0.0, np.pi)), draw(st.floats(0.0, 2 * np.pi)), draw(st.floats(0.0, 2 * np.pi))
    drawn = np.array([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)])
    direction = draw(st.sampled_from([np.array(axis) for axis in TILTED] + [drawn]))
    unit = direction / np.linalg.norm(direction)
    first = np.cross(unit, np.eye(3)[int(np.argmin(np.abs(unit)))])
    first /= np.linalg.norm(first)
    across = np.cos(side) * first + np.sin(side) * np.cross(unit, first)
    fraction = draw(st.one_of(st.just(0.0), st.floats(1e-3, 0.5))) if within else 0.0
    point = centre + draw(st.floats(-1e-2, 1e-2)) * unit + fraction * AXIS_ULPS * EPS * radius * across
    return {"point": point, "centre": centre, "axis": unit * draw(st.floats(0.5, 2.0)), "radius": np.array(radius)}


def on_the_axis(shape):
    """Both functions of ``shape`` and their derivatives with respect to each variable."""

    def both(x):
        parameters = {"centre": x["centre"], "axis": x["axis"], "radius": x["radius"]}
        return shape.signed_distance(parameters, x["point"]), shape.outward_normal(parameters, x["point"])

    return jax.jit(both), jax.jit(jax.jacfwd(both))


ON_THE_AXIS = on_the_axis(CYLINDER)


def departures(evaluate, x: dict[str, np.ndarray]) -> list[str]:
    """Where the values and derivatives at ``x`` depart from the record's on the axis: the distance minus the radius,
    its derivative zero but in the radius (minus one); the normal zero, its derivative the offset's from the axis, ``P``
    in the point, ``-P`` in the centre, ``-(a o^T P + (o . a) P) / |axis|`` in the axis (``P = I - a a^T``, ``o`` the
    point's offset from the centre), zero in the radius, to its rounding."""
    values, jacobians = evaluate
    held = {key: jnp.asarray(value) for key, value in x.items()}
    distance, normal = (np.asarray(v) for v in values(held))
    d_distance, d_normal = jacobians(held)
    length = float(np.linalg.norm(x["axis"]))
    a = x["axis"] / length
    offset = x["point"] - x["centre"]
    projection = np.eye(3) - np.outer(a, a)
    expected = {"point": projection, "centre": -projection, "radius": np.zeros(3),
                "axis": -(np.outer(a, offset @ projection) + float(offset @ a) * projection) / length}
    rounding = 64 * EPS * max(1.0, float(np.linalg.norm(offset)) / length)
    found = []
    if float(distance) != -float(x["radius"]):
        found.append(f"distance {float(distance)!r}, not minus the radius {float(x['radius'])!r}")
    if np.any(normal != 0.0):
        found.append(f"normal {normal}, not zero")
    for name in VARIABLES:
        held_distance = np.asarray(d_distance[name])
        if not np.array_equal(held_distance, -np.ones_like(held_distance) if name == "radius" else
                              np.zeros_like(held_distance)):
            found.append(f"d distance / d {name} = {held_distance}")
        held_normal = np.asarray(d_normal[name]).reshape(expected[name].shape)
        if float(np.max(np.abs(held_normal - expected[name]))) > rounding:
            found.append(f"d normal / d {name} = {held_normal}, not {expected[name]}")
    return found


@settings(deadline=None)
@given(tube_and_point_on_its_axis())
def test_within_its_resolution_of_any_axis_the_cylinder_returns_what_the_record_says(x):
    assert departures(ON_THE_AXIS, x) == []


@pytest.mark.parametrize("axis", [(0.0, 0.0, 2.0), *TILTED])
def test_on_its_axis_neither_function_has_a_derivative_and_the_code_returns_zero_and_the_projection(axis):
    """The switch the record declares, on a tube along a coordinate axis and the review's two tilted axes, a point
    put on each: the distance's one-sided differences across the axis are +1 and -1 at every step (a cone point) and
    the normal's central difference grows as one over the step (no limit). The code returns the distance's derivative
    as zero and the normal's as the projection across the axis."""
    surface = {"centre": (1e-3, -2e-3, 5e-4), "axis": axis, "radius": 3.175e-3}
    a = np.asarray(axis) / np.linalg.norm(axis)
    point = jnp.asarray(np.asarray(surface["centre"]) + 4e-3 * a)
    across = np.cross(a, (0.6, 0.8, 0.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0))
    across /= np.linalg.norm(across)
    distance = float(CYLINDER.signed_distance(surface, point))
    assert distance == -3.175e-3
    for step in (1e-6, 1e-9, 1e-12):
        rounding = 16 * EPS * (3.175e-3 + float(np.linalg.norm(point))) / step
        moved = [float(CYLINDER.signed_distance(surface, point + sign * step * across)) for sign in (1.0, -1.0)]
        assert (moved[0] - distance) / step == pytest.approx(1.0, rel=rounding, abs=0.0)
        assert (distance - moved[1]) / step == pytest.approx(-1.0, rel=rounding, abs=0.0)
        normals = [np.asarray(CYLINDER.outward_normal(surface, point + sign * step * across)) for sign in (1.0, -1.0)]
        assert np.linalg.norm(normals[0] - normals[1]) / (2 * step) == pytest.approx(1.0 / step, rel=1e-6)
    assert np.array_equal(np.asarray(jax.grad(lambda x: CYLINDER.signed_distance(surface, x))(point)), np.zeros(3))
    assert np.allclose(np.asarray(jax.jacfwd(lambda x: CYLINDER.outward_normal(surface, x))(point)),
                       np.eye(3) - np.outer(a, a), rtol=0.0, atol=16 * EPS)


def test_without_its_resolution_the_cylinder_departs_from_the_record_on_a_tilted_axis(monkeypatch):
    """The control: with ``AXIS_ULPS`` zero, as the cylinder was, a point put on the review's tilted axes computes an
    offset of rounding, not zero, and the cylinder returns a normal of that rounding's direction and a distance whose
    derivative is not zero; and Hypothesis finds such a point on a drawn tilted axis. The plant is traced while the
    constant is patched, so the cylinder itself is evaluated, and compiled, before it."""
    centre = np.asarray((1e-3, -2e-3, 5e-4))
    points = [{"point": centre + 4e-3 * np.asarray(axis) / np.linalg.norm(axis), "centre": centre,
               "axis": np.asarray(axis), "radius": np.asarray(3.175e-3)} for axis in TILTED]
    assert [departures(ON_THE_AXIS, x) for x in points] == [[], []]
    monkeypatch.setattr(geometry, "AXIS_ULPS", 0)
    planted = on_the_axis(geometry.Cylinder())
    assert all(departures(planted, x) for x in points)
    found = find(tube_and_point_on_its_axis(within=False), lambda x: bool(departures(planted, x)),
                 settings=settings(deadline=None, database=None))
    print(f"\n{departures(planted, found)}")
