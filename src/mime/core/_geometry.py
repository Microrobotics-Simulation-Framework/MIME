"""Shapes as a node evaluates them: pure JAX functions of a shape's parameters in SI, so that the parameters are
``params`` leaves of the node that reads them and a study can move them.

A geometry kind (``mime.scene``) declares a thing's geometry in a script; each of its surfaces names the shape its free
side is evaluated with and gives that shape's parameters. A node holds the shape's registered name, a string, and its
parameters, and evaluates the shape through :func:`shape_named`. A lab's geometry kind registers its own shapes here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from jax import Array, lax
from jaxtyping import Float

from mime.core._node import jnp
from mime.core._quaternion import safe_norm

type Point = Float[Array, "3"]
type Parameters = Mapping[str, Any]

#: A point whose offset from a shape's axis is below this many units in the last place of the shape's radius counts as
#: on the axis, where the direction from it is not defined: 1.6 micrometres in A's 3.175 mm tube in float32, 3e-15 m
#: in float64. Well above the rounding of a point put on an axis off the coordinate axes, or on a turned world's (an
#: offset of up to about 3 units in the last place of its coordinates), so that such a point is on it while its
#: coordinates are within about a thousand radii.
AXIS_ULPS = 4096


@runtime_checkable
class Shape(Protocol):
    """A region of space, evaluated at a point: its signed distance (negative inside, zero on its boundary), its outward
    normal (the unit gradient of the signed distance, pointing out of the region), and its axis where it has one, as
    a direction and a point on it. Every function is traceable, and finite with finite derivatives at every point,
    including where the normal is not defined (it may be zero there). A region with an axis is taken to be uniform
    along it: its signed distance at a point on the axis is minus its radius there and everywhere."""

    parameter_names: tuple[str, ...]

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]: ...  # noqa: F722 - a jaxtyping shape

    def outward_normal(self, parameters: Parameters, point: Point) -> Point: ...

    def axis(self, parameters: Parameters) -> Point | None: ...

    def axis_point(self, parameters: Parameters) -> Point | None: ...


def _unit(v: Any) -> Array:
    """``v`` scaled to unit length; zero for a zero vector."""
    length = safe_norm(v)
    return v / jnp.where(length > 0, length, 1.0)


class Cylinder:
    """The inside of an unbounded circular cylinder: ``centre`` (a point on its axis), ``axis`` (its direction,
    normalised here), ``radius``. Its outward normal is the radial direction, which is not defined on the axis. A point
    within ``AXIS_ULPS`` units in the last place of the radius of the axis is on it, whatever the axis's direction:
    there the signed distance is minus the radius (its derivative zero, but minus one in the radius), and the normal is
    zero, with the derivative of the offset from the axis (in the point, the projection across the axis)."""

    parameter_names: tuple[str, ...] = ("centre", "axis", "radius")

    @staticmethod
    def _radial(parameters: Parameters, point: Point) -> tuple[Array, Array, Array]:
        """The point's offset from the axis, its length (zero on the axis), and whether the point is off the axis."""
        axis = _unit(jnp.asarray(parameters["axis"]))
        offset = point - jnp.asarray(parameters["centre"])
        radial = offset - jnp.dot(offset, axis) * axis
        squared = jnp.sum(radial * radial)
        off = squared > (AXIS_ULPS * float(jnp.finfo(radial.dtype).eps) * parameters["radius"]) ** 2
        return radial, jnp.where(off, jnp.sqrt(jnp.where(off, squared, 1.0)), 0.0), off

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        return self._radial(parameters, point)[1] - parameters["radius"]

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        radial, length, off = self._radial(parameters, point)
        return jnp.where(off, radial / jnp.where(off, length, 1.0), radial - lax.stop_gradient(radial))

    def axis(self, parameters: Parameters) -> Point | None:
        return _unit(jnp.asarray(parameters["axis"]))

    def axis_point(self, parameters: Parameters) -> Point | None:
        return jnp.asarray(parameters["centre"])


_shapes: dict[str, Shape] = {}


def register_shape(name: str, shape: Shape) -> Shape:
    """Make ``shape`` evaluable by nodes under ``name``. Registering the same object again changes nothing."""
    held = _shapes.get(name)
    if held is not None and held is not shape:
        raise ValueError(f"shape {name!r} is already registered as {held!r}")
    _shapes[name] = shape
    return shape


def shape_named(name: str) -> Shape:
    if name not in _shapes:
        raise KeyError(f"no shape {name!r} is registered; the shapes are {sorted(_shapes)}")
    return _shapes[name]


def shape_names() -> list[str]:
    return sorted(_shapes)


CYLINDER = register_shape("cylinder", Cylinder())

#: How a confinement's velocity rule changes the velocity of a body it has just projected back inside, given that
#: velocity and the outward normal where it was projected to.
type VelocityRule = Callable[[Point, Point], Point]


def remove_outward(velocity: Point, normal: Point) -> Point:
    """The velocity with its outward component removed; the inward and tangential components are kept."""
    return velocity - jnp.maximum(jnp.dot(velocity, normal), 0.0) * normal


VELOCITY_RULES: dict[str, VelocityRule] = {"remove_outward": remove_outward}
