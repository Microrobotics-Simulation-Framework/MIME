"""Shapes and the tube: the cylinder's signed distance and outward normal by their closed forms, finite with finite
derivatives on its axis, where the normal is not defined; the tube's inner surface as the cylinder of its inner
radius; a geometry kind checked against the shape protocol."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
