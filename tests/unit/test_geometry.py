"""Shapes and the tube: the cylinder's signed distance and outward normal by their closed forms, finite with finite
derivatives on its axis, where the normal is not defined, and exact derivatives off it; the tube's inner surface as the
cylinder of its inner radius; a geometry kind checked against the shape protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import CYLINDER, Shape, shape_named
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
# Tube is declared piecewise, its switch the axis (its gradient record): on the axis the distance from it has a cone
# point and the normal no limit, so neither has a derivative there; off the axis both are smooth, and their
# derivatives exact. The property compares JAX's derivatives of the signed distance and the outward normal, with
# respect to the point and to each parameter, with central finite differences, in float64, at points drawn off the
# axis by 1e-4 to 2 radii. Each finite difference moves the geometry by the step ``H`` times the point's distance from
# the axis, so it is judged on the change that move makes (the distance's, or the normal's over that distance); its
# resolution is the rounding of the values, about eps times the drawn lengths over that move, plus the difference's
# own truncation, H squared. The control: a derivative 10% off, of either function, planted in a copy of the
# cylinder, fails on every draw.

H = 1e-2
#: The finite differences' resolution, in units of their rounding and truncation: over 2000 draws the exact derivative's
#: worst difference was below a tenth of it, and the plants' least was forty times it.
RESOLUTION = 16.0
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


def off_by_a_tenth(function):
    """``function`` with its derivative 10% off: the control's plant."""
    planted = jax.custom_jvp(function)

    @planted.defjvp
    def _(primals, tangents):
        value, tangent = jax.jvp(function, primals, tangents)
        return value, 1.1 * tangent

    return planted


class Planted:
    """The cylinder with the derivative of one of its functions planted 10% off."""

    def __init__(self, which: str) -> None:
        self.signed_distance = off_by_a_tenth(CYLINDER.signed_distance) if which == "distance" else \
            CYLINDER.signed_distance
        self.outward_normal = off_by_a_tenth(CYLINDER.outward_normal) if which == "normal" else CYLINDER.outward_normal


def evaluated(shape):
    """Both functions of ``shape`` and their derivatives with respect to each variable, compiled once."""

    def both(x):
        parameters = {"centre": x["centre"], "axis": x["axis"], "radius": x["radius"]}
        return shape.signed_distance(parameters, x["point"]), shape.outward_normal(parameters, x["point"])

    return jax.jit(jax.vmap(both)), jax.jit(jax.jacfwd(both))


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
    resolution = RESOLUTION * (float(np.finfo(np.float64).eps) * lengths / (H * distance) + H * H)
    steps = {"point": H * distance, "centre": H * distance, "radius": H * distance,
             "axis": H * distance * float(np.linalg.norm(x["axis"])) / max(abs(along), distance)}
    moved = []
    for name in VARIABLES:
        for i in range(x[name].size):
            for sign in (1.0, -1.0):
                held = {key: jnp.asarray(value) for key, value in x.items()}
                held[name] = jnp.asarray((x[name].ravel() + sign * steps[name] * np.eye(x[name].size)[i])
                                         .reshape(x[name].shape))
                moved.append(held)
    batch = {key: jnp.stack([m[key] for m in moved]) for key in x}
    distances, normals = (np.asarray(v) for v in values(batch))
    derivative = jacobians({key: jnp.asarray(value) for key, value in x.items()})
    found, row = 0.0, 0
    for name in VARIABLES:
        for i in range(x[name].size):
            for function, scale, at in ((distances, H * distance, 0), (normals, H, 1)):
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
def test_a_derivative_planted_a_tenth_off_fails_on_every_draw(which, x):
    assert worst(which, x) > 1.0


def test_on_its_axis_neither_function_has_a_derivative_and_the_code_returns_zero_and_the_projection():
    """The switch the record declares, on a tube along a coordinate axis, where a centred point's offset computes to
    zero exactly: the distance's one-sided differences across the axis are +1 and -1 at every step (a cone point) and
    the normal's central difference grows as one over the step (no limit). The code returns the distance's derivative
    as zero and the normal's as the projection across the axis."""
    surface = {"centre": (1e-3, -2e-3, 5e-4), "axis": (0.0, 0.0, 2.0), "radius": 3.175e-3}
    point = jnp.asarray((1e-3, -2e-3, 4.5e-3))
    across = np.array([0.6, 0.8, 0.0])
    distance = float(CYLINDER.signed_distance(surface, point))
    for step in (1e-6, 1e-9, 1e-12):
        moved = [float(CYLINDER.signed_distance(surface, point + sign * step * across)) for sign in (1.0, -1.0)]
        assert (moved[0] - distance) / step == pytest.approx(1.0, rel=1e-6)
        assert (distance - moved[1]) / step == pytest.approx(-1.0, rel=1e-6)
        normals = [np.asarray(CYLINDER.outward_normal(surface, point + sign * step * across)) for sign in (1.0, -1.0)]
        assert np.linalg.norm(normals[0] - normals[1]) / (2 * step) == pytest.approx(1.0 / step, rel=1e-6)
    assert np.array_equal(np.asarray(jax.grad(lambda x: CYLINDER.signed_distance(surface, x))(point)), np.zeros(3))
    assert np.array_equal(np.asarray(jax.jacfwd(lambda x: CYLINDER.outward_normal(surface, x))(point)),
                          np.diag([1.0, 1.0, 0.0]))
