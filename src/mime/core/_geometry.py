"""Shapes as a node evaluates them: pure JAX functions of a shape's parameters in SI, so that the parameters are
``params`` leaves of the node that reads them and a study can move them.

A geometry kind (``mime.scene``) declares a thing's geometry in a script; each of its surfaces names the shape its free
side is evaluated with and gives that shape's parameters. A node holds the shape's registered name, a string, and its
parameters, and evaluates the shape through :func:`shape_named`. A lab's geometry kind registers its own shapes here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from jax import Array, lax
from jaxtyping import Float

from mime.core import switch
from mime.core._node import jnp

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
    along it: its signed distance at a point on the axis is minus its radius there and everywhere.

    A shape whose functions also take an array of points ``(..., 3)``, evaluating each (its distances ``(...)``, its
    normals ``(..., 3)``), declares ``takes_many = True``: a containment contact holds a body's hull points against it,
    many at once. A shape that declares nothing takes one point at a time; a lab's shape need not take many."""

    parameter_names: tuple[str, ...]

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]: ...  # noqa: F722 - a jaxtyping shape

    def outward_normal(self, parameters: Parameters, point: Point) -> Point: ...

    def axis(self, parameters: Parameters) -> Point | None: ...

    def axis_point(self, parameters: Parameters) -> Point | None: ...


def _unit(v: Any, *, owner: str) -> Array:
    """``v`` scaled to unit length; zero for a zero vector, where its length is the caller's switch ``owner``."""
    length = switch.norm(v, owner=owner, resolution=0.0, indicator=lambda: jnp.sqrt(jnp.sum(v * v)))
    return v / switch.where(length > 0, length, 1.0, owner=owner, resolution=0.0, indicator=length)


class Cylinder:
    """The inside of an unbounded circular cylinder: ``centre`` (a point on its axis), ``axis`` (its direction,
    normalised here), ``radius``. Its outward normal is the radial direction, which is not defined on the axis. A point
    within ``AXIS_ULPS`` units in the last place of the radius of the axis is on it, whatever the axis's direction:
    there the signed distance is minus the radius (its derivative zero, but minus one in the radius), and the normal is
    zero, with the derivative of the offset from the axis (in the point, the projection across the axis). It takes an
    array of points as well as one (``takes_many``), each evaluated as it would be alone."""

    parameter_names: tuple[str, ...] = ("centre", "axis", "radius")
    takes_many = True

    @staticmethod
    def _radial(parameters: Parameters, point: Point) -> tuple[Array, Array, Array]:
        """The point's offset from the axis, its length (zero on the axis), and whether the point is off the axis;
        for an array of points, each one's."""
        axis = _unit(jnp.asarray(parameters["axis"]), owner="cylinder.axis_norm")
        offset = point - jnp.asarray(parameters["centre"])
        radial = offset - jnp.matmul(offset, axis)[..., None] * axis
        squared = jnp.sum(radial * radial, axis=-1)
        resolution = AXIS_ULPS * float(jnp.finfo(radial.dtype).eps) * parameters["radius"]
        off = squared > resolution ** 2
        length = switch.where(off, jnp.sqrt(switch.domain_guard(off, squared, 1.0)), 0.0, owner="cylinder.axis",
                              resolution=resolution, indicator=lambda: jnp.sqrt(squared))
        return radial, length, off

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        return self._radial(parameters, point)[1] - parameters["radius"]

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        radial, length, off = self._radial(parameters, point)
        return switch.where(off[..., None], radial / switch.domain_guard(off, length, 1.0)[..., None],
                            radial - lax.stop_gradient(radial), owner="cylinder.axis",
                            resolution=lambda: AXIS_ULPS * float(jnp.finfo(radial.dtype).eps) * parameters["radius"],
                            indicator=lambda: jnp.sqrt(jnp.sum(radial * radial, axis=-1)))

    def axis(self, parameters: Parameters) -> Point | None:
        return _unit(jnp.asarray(parameters["axis"]), owner="cylinder.axis_norm")

    def axis_point(self, parameters: Parameters) -> Point | None:
        return jnp.asarray(parameters["centre"])


_shapes: dict[str, Shape] = {}
_switches: dict[str, tuple[switch.Declaration, ...]] = {}


def register_shape(name: str, shape: Shape, *, switches: tuple[switch.Declaration, ...] = ()) -> Shape:
    """Make ``shape`` evaluable by nodes under ``name``, with the ``switches`` its functions evaluate, declared beside
    its class (orders per capability: ``distance``, ``normal``). Registering the same object again changes nothing."""
    held = _shapes.get(name)
    if held is not None and held is not shape:
        raise ValueError(f"shape {name!r} is already registered as {held!r}")
    _shapes[name] = shape
    _switches[name] = switches
    return shape


def shape_switches(name: str) -> tuple[switch.Declaration, ...]:
    """The switches the shape registered as ``name`` declares: an entry whose surfaces or regions it evaluates
    inherits them."""
    shape_named(name)
    return _switches[name]


def shape_named(name: str) -> Shape:
    if name not in _shapes:
        raise KeyError(f"no shape {name!r} is registered; the shapes are {sorted(_shapes)}")
    return _shapes[name]


def shape_names() -> list[str]:
    return sorted(_shapes)


#: The cylinder's switches: the axis, where the distance has a cone and the normal no limit, and the length of the
#: axis it is given, which it scales to unit length.
CYLINDER_SWITCHES = (
    switch.Declaration(
        owner="cylinder.axis",
        indicator="the point's distance from the axis, m; on the axis within AXIS_ULPS units in the last place of the "
                  "radius (a band)",
        orders={"distance": 0, "normal": -1},
        route="none: on the axis the distance is a cone and the normal has no limit, so neither has a derivative to "
              "compute; within the band the distance is minus the radius and the normal zero, with the offset's "
              "derivative"),
    switch.Declaration(
        owner="cylinder.axis_norm",
        indicator="the length of the axis as given (a direction, any length)",
        orders={"distance": -1, "normal": -1},
        route="none needed: an axis of zero length is no direction, and the plan refuses one, so the switch is never "
              "live"),
)
CYLINDER = register_shape("cylinder", Cylinder(), switches=CYLINDER_SWITCHES)

class VelocityRule(Protocol):
    """How a confinement's velocity rule changes the velocity of a body it has just projected back inside, given that
    velocity and the outward normal where it was projected to; a switch it evaluates is the caller's ``owner``."""

    def __call__(self, velocity: Point, normal: Point, *, owner: str) -> Point: ...


def remove_outward(velocity: Point, normal: Point, *, owner: str) -> Point:
    """The velocity with its outward component removed; the inward and tangential components are kept. Whether there
    is one is the caller's switch ``owner``."""
    outward = jnp.dot(velocity, normal)
    return velocity - switch.maximum(outward, 0.0, owner=owner, resolution=lambda: switch.rounding(
        jnp.sqrt(jnp.sum(velocity * velocity))), indicator=outward) * normal


VELOCITY_RULES: dict[str, VelocityRule] = {"remove_outward": remove_outward}
