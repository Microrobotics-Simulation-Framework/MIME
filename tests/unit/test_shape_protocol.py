"""One ``Shape`` protocol (``mime.core.shapes``): a shape gives its exact signed distance, and the rest is derived from it
unless it gives its own. The extensibility test's toy plugin shape, ``toy.capsule`` (``mime.testing.toys``), gives
only a distance; each derived form it is evaluated by is held to MIME's capsule in closed form (``mime.core.CAPSULE``),
which gives its own:

- **the protocol**: the distance is the one method required; a shape registered without subclassing ``Shape`` (the
  soft-tissue plugin's) keeps its own methods and takes the derived forms of those it lacks; a derived normal declares
  its switch, ``<name>.kink``, when the shape is registered;
- **the derived normal** and its derivative against the capsule's closed forms, at drawn points off its segment (the
  medial set), to the dtype's rounding; zero on the segment;
- **contact**: a body the rigid body's containment holds inside the toy capsule (projected back along the outward
  normal, its outward velocity removed) moves as one held inside MIME's capsule, to rounding;
- **the viewer**: the toy kind's contoured drawing is closed and faces out, its vertices within the contour's
  derived line of the capsule's surface, ``3 h^2 / (8 (r - sqrt(3) h))`` (an exact distance's interpolation along a cell
  edge of length at most ``sqrt(3) h``, its second derivative at most ``1 / (r - sqrt(3) h)`` there), and its volume
  converges on the capsule's at second order.

**Controls**, each by ``SHAPE_PLANT``: ``normal`` turns the derived normal by a milliradian about a fixed axis (a
direction error, which no scale on the distance can show), and must fail the normal and the contact tests on every run;
``sign`` (the camera's, ``tests/sensing/test_toy_shape_camera.py``) offsets the derived sign function."""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.core._geometry as geometry
from mime.core import CAPSULE, shape_named, shape_switches
from mime.core.shapes import Shape, register_shape
from mime.nodes.body import RigidInertialMidpointNode
from mime.sensing._sampled import made_kind
from mime.testing._kind_conformance import _mesh_volume, drawing_checks, drawing_problems
from mime.testing.toys import ToyCapsule

PLANT = os.environ.get("SHAPE_PLANT", "")
#: The toy capsule's parameters: tilted, off the origin, at a micro-robot's scale.
PARAMETERS = {"centre": (1e-4, -2e-4, 3e-4), "axis": (0.3, -0.5, 0.8), "radius": 2.5e-4, "height": 6e-4}
#: The normal's line, in units in the last place: both normals are a few operations from the point's coordinates.
NORMAL_ULPS = 64
#: The plant: a milliradian about this axis.
TURN, TURN_AXIS = 1e-3, np.asarray((1.0, 2.0, 3.0)) / math.sqrt(14.0)


@pytest.fixture(params=["float32", "float64"])
def dtype(request: pytest.FixtureRequest):
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", request.param == "float64")
    yield request.param
    jax.config.update("jax_enable_x64", held)


@pytest.fixture(autouse=True)
def planted(monkeypatch: pytest.MonkeyPatch) -> None:
    if PLANT == "normal":
        found = geometry.derived_normal
        c, s = math.cos(TURN), math.sin(TURN)
        k = TURN_AXIS
        cross = np.asarray([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        turn = np.eye(3) + s * cross + (1.0 - c) * cross @ cross

        def turned(*args, **kwargs):
            normal = found(*args, **kwargs)
            return normal @ jnp.asarray(turn.T, dtype=normal.dtype)

        monkeypatch.setattr(geometry, "derived_normal", turned)


# -- the protocol --------------------------------------------------------------------------------------------------------


class OnlyADistance(Shape):
    parameter_names = ("radius",)

    def signed_distance(self, parameters, point):
        return jnp.sqrt(jnp.sum(point * point) + 1e-30) - parameters["radius"]


def test_a_shape_must_give_its_signed_distance_and_need_give_nothing_else():
    class Nothing(Shape):
        parameter_names = ()

    with pytest.raises(TypeError, match="signed_distance"):
        Nothing()  # pyright: ignore[reportAbstractUsage]
    shape = OnlyADistance()
    point = jnp.asarray((3e-4, 0.0, 4e-4))
    assert np.allclose(shape.outward_normal({"radius": 1e-4}, point), (0.6, 0.0, 0.8), rtol=0, atol=1e-6)
    assert shape.axis({"radius": 1e-4}) is None and shape.axis_point({"radius": 1e-4}) is None


def test_a_shape_registered_without_subclassing_keeps_its_own_methods_and_takes_the_derived_rest():
    class Plugin:
        """The soft-tissue plugin's form: its own normal and axis, no sign function, not a subclass."""

        parameter_names = ("radius",)
        takes_many = True

        def signed_distance(self, parameters, point):
            return jnp.sqrt(jnp.sum(point * point, axis=-1) + 1e-30) - parameters["radius"]

        def outward_normal(self, parameters, point):
            return jnp.zeros_like(point) + 7.0

        def axis(self, parameters):
            return jnp.asarray((0.0, 0.0, 1.0))

    plugin = Plugin()
    assert register_shape("test_protocol.plugin", plugin) is plugin
    assert register_shape("test_protocol.plugin", plugin) is plugin
    held = shape_named("test_protocol.plugin")
    point = jnp.asarray((3e-4, 0.0, 4e-4))
    assert isinstance(held, Shape) and held.takes_many
    assert float(held.outward_normal({"radius": 1e-4}, point)[0]) == 7.0
    assert held.axis_point({}) is None and float(held.axis({})[2]) == 1.0
    found = held.sign_function({"radius": 1e-4}, (point[None, 0], point[None, 1], point[None, 2]))
    assert np.allclose(found.value, 4e-4) and np.all(found.slope == 1.0)
    assert [d.owner for d in shape_switches("test_protocol.plugin")] == []
    with pytest.raises(ValueError, match="already registered"):
        register_shape("test_protocol.plugin", Plugin())


def test_a_derived_normal_declares_its_switch_when_registered():
    assert [d.owner for d in shape_switches("toy.capsule")] == ["toy.capsule.end", "toy.capsule.kink"]
    assert shape_named("toy.capsule").kink_owner == "toy.capsule.kink"
    assert all(not d.owner.endswith(".kink") for d in shape_switches("capsule"))


# -- the derived normal ---------------------------------------------------------------------------------------------------


def _placed(dtype: str, along: float, radial: float, angle: float) -> jnp.ndarray:
    """A point ``along`` the capsule's axis (in half-heights, beyond one past an end) and ``radial`` radii off it."""
    axis = np.asarray(PARAMETERS["axis"]) / np.linalg.norm(PARAMETERS["axis"])
    first = np.cross(axis, (1.0, 0.0, 0.0))
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    offset = PARAMETERS["radius"] * radial * (math.cos(angle) * first + math.sin(angle) * second)
    point = np.asarray(PARAMETERS["centre"]) + along * PARAMETERS["height"] / 2.0 * axis + offset
    return jnp.asarray(point, dtype=dtype)


#: Along the axis in half-heights: on the segment, or past an end, never within a hundredth of an end's plane (the
#: switch ``capsule.end``, where the normal's derivative is a tie's, not defined: there the two forms take different
#: means of the two sides, as the switch module says a tie does).
ALONG = st.one_of(st.floats(-0.99, 0.99), st.floats(1.01, 1.6), st.floats(-1.6, -1.01))


# The dtype fixture sets the run's dtype once for every example, which is what the examples need.
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(along=ALONG, radial=st.floats(1e-3, 3.0), angle=st.floats(0.0, 2.0 * math.pi))
def test_the_derived_normal_and_its_derivative_are_the_capsules_closed_forms(dtype, along, radial, angle):
    """Judged on the rounding of the point's offset from the segment: a direction from a point ``rho`` off it, whose
    coordinates are rounded at the scale ``L`` of the inputs, turns by about ``eps L / rho``, and its derivative, of
    size ``1 / rho``, by that over ``rho``."""
    point = _placed(dtype, along, radial, angle)
    toy = shape_named("toy.capsule")
    eps = float(jnp.finfo(point.dtype).eps)
    scale = float(jnp.max(jnp.abs(point))) + PARAMETERS["radius"] + PARAMETERS["height"]
    off = max(radial, abs(along) - 1.0 if abs(along) > 1.0 else 0.0) * PARAMETERS["radius"]
    line = NORMAL_ULPS * eps * (1.0 + scale / off)
    derived, analytic = toy.outward_normal(PARAMETERS, point), CAPSULE.outward_normal(PARAMETERS, point)
    assert float(jnp.max(jnp.abs(derived - analytic))) <= line
    turn = jnp.asarray((0.2, -0.7, 0.4), dtype=point.dtype)
    _, d_derived = jax.jvp(lambda p: toy.outward_normal(PARAMETERS, p), (point,), (turn,))
    _, d_analytic = jax.jvp(lambda p: CAPSULE.outward_normal(PARAMETERS, p), (point,), (turn,))
    assert float(jnp.max(jnp.abs(d_derived - d_analytic))) <= line / off
    distance = toy.signed_distance(PARAMETERS, point)
    assert abs(float(distance - CAPSULE.signed_distance(PARAMETERS, point))) <= NORMAL_ULPS * eps * scale


def test_on_the_segment_the_derived_normal_is_zero_and_its_derivative_finite(dtype):
    toy = shape_named("toy.capsule")
    point = jnp.asarray(PARAMETERS["centre"], dtype=dtype)
    normal, tangent = jax.jvp(lambda p: toy.outward_normal(PARAMETERS, p), (point,), (jnp.ones_like(point),))
    assert np.all(np.asarray(normal) == 0.0) and np.all(np.isfinite(np.asarray(tangent)))
    assert np.all(np.isfinite(np.asarray(jax.grad(lambda p: jnp.sum(toy.outward_normal(PARAMETERS, p)))(point))))


def test_the_derived_sign_function_is_the_distance_with_slope_one(dtype):
    toy = shape_named("toy.capsule")
    points = jnp.stack([_placed(dtype, a, r, t) for a, r, t in ((0.3, 1.2, 0.4), (1.4, 0.4, 2.0), (-1.2, 2.0, 5.0))])
    found = toy.sign_function(PARAMETERS, (points[:, 0], points[:, 1], points[:, 2]))
    assert np.array_equal(np.asarray(found.value), np.asarray(toy.signed_distance(PARAMETERS, points)))
    assert np.all(np.asarray(found.slope) == 1.0)


# -- contact --------------------------------------------------------------------------------------------------------------

EXTENT, MARGIN = 0.4e-4, 0.1e-4


def _held(shape: str) -> RigidInertialMidpointNode:
    return RigidInertialMidpointNode("probe", 5e-4, mass=1e-9, inertia=(1e-17, 1e-17, 1e-17),
                                     initial_pose=(0, 0, 0, 1, 0, 0, 0), initial_twist=(0.0,) * 6, surface_shape=shape,
                                     extent=EXTENT, margin=MARGIN, velocity_rule="remove_outward",
                                     **{f"surface_{k}": v for k, v in PARAMETERS.items()})


#: Starts beside the side, past a cap and at an end's plane, each moving out through the wall.
STARTS = [(0.2, 0.85, 0.3, (0.0, 0.0, 0.2)), (1.6, 0.3, 4.0, (0.1, 0.0, 0.0)), (1.0, 0.75, 1.0, (0.0, 0.2, 0.0))]


@pytest.mark.parametrize("start", STARTS)
def test_a_body_held_inside_the_toy_capsule_moves_as_one_held_inside_mimes(start, dtype):
    along, radial, angle, velocity = start
    position = _placed("float64", along, radial, angle)
    toy, capsule = _held("toy.capsule"), _held("capsule")
    state = {"pose": jnp.concatenate([jnp.asarray(position, dtype=dtype), jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=dtype)]),
             "twist": jnp.asarray((*velocity, 0.0, 0.0, 0.0), dtype=dtype)}
    given_wrench = {"applied_wrench": jnp.zeros(6, dtype=dtype)}
    held = {"toy": state, "capsule": state}
    eps = float(jnp.finfo(jnp.dtype(dtype)).eps)
    for _ in range(3):
        held = {"toy": toy.update(held["toy"], given_wrench, 5e-4), "capsule": capsule.update(held["capsule"],
                                                                                               given_wrench, 5e-4)}
        apart = float(jnp.max(jnp.abs(held["toy"]["pose"][:3] - held["capsule"]["pose"][:3])))
        speed = float(jnp.max(jnp.abs(held["toy"]["twist"] - held["capsule"]["twist"])))
        assert apart <= NORMAL_ULPS * eps * (PARAMETERS["height"] + 2.0 * PARAMETERS["radius"])
        assert speed <= NORMAL_ULPS * eps * 0.2
    inside = float(CAPSULE.signed_distance(PARAMETERS, held["toy"]["pose"][:3]))
    assert inside <= -(EXTENT + MARGIN) * (1.0 - 1e-3)


# -- the viewer -----------------------------------------------------------------------------------------------------------

SETTINGS = {"radius": 0.25e-3, "height": 0.8e-3, "axis": (1.0, 0.3, 0.4)}


def test_the_toy_kinds_contoured_drawing_lies_within_its_line_of_the_capsule_and_converges():
    kind = made_kind(ToyCapsule, SETTINGS)
    assert drawing_checks(kind) == ("closed", "near the surface")
    assert drawing_problems(kind, SETTINGS, 32) == []
    region = kind.region_parameters("solid", SETTINGS)
    radius, errors = SETTINGS["radius"], []
    for cells in (16, 32, 64):
        drawing = kind.drawing(SETTINGS, cells)
        assert "drawing" in drawing.representation
        points = np.asarray(drawing.points)
        off = np.asarray(jax.vmap(lambda x: CAPSULE.signed_distance(region, x))(jnp.asarray(points)))
        cell = 2.0 * kind.enclosing_radius(SETTINGS) / cells
        assert float(np.max(np.abs(off))) <= 3.0 * cell * cell / (8.0 * (radius - math.sqrt(3.0) * cell))
        volume = _mesh_volume(points, np.asarray(drawing.face_vertex_counts), np.asarray(drawing.face_vertex_indices))
        errors.append(abs(volume - kind.volume(SETTINGS)) / kind.volume(SETTINGS))
    assert errors[1] <= errors[0] / 3.0 and errors[2] <= errors[1] / 3.0
