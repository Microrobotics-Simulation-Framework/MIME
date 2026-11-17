"""Shapes as a node evaluates them: pure JAX functions of a shape's parameters in SI, so that the parameters are
``params`` leaves of the node that reads them and a study can move them. A shape gives its exact signed distance;
everything else is derived from it unless the shape gives its own (:class:`Shape`).

A geometry kind (``mime.scene``) declares a thing's geometry in a script; each of its surfaces names the shape its free
side is evaluated with and gives that shape's parameters. A node holds the shape's registered name, a string, and its
parameters, and evaluates the shape through :func:`shape_named`. A lab's geometry kind registers its own shapes here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple, Protocol, cast, override

import jax
from jax import Array
from jaxtyping import Float

from mime.core import switch
from mime.core._node import jnp

#: ``jax.grad`` and ``jax.vmap``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_grad: Any = cast(Any, jax).grad
_vmap: Any = cast(Any, jax).vmap

type Point = Float[Array, "3"]
type Parameters = Mapping[str, Any]

#: A point whose offset from a shape's axis is below this many units in the last place of the shape's radius counts as
#: on the axis, where the direction from it is not defined: 1.6 micrometres in A's 3.175 mm tube in float32, 3e-15 m
#: in float64. Well above the rounding of a point put on an axis off the coordinate axes, or on a turned world's (an
#: offset of up to about 3 units in the last place of its coordinates), so that such a point is on it while its
#: coordinates are within about a thousand radii.
AXIS_ULPS = 4096


#: The length of a signed distance's gradient below which a shape's derived normal counts a point as on its medial set
#: (where the nearest boundary point is not unique and the distance has a kink): an exact distance's gradient has
#: length one wherever it is differentiable, and at a kink a distance written to the contract below gives a mean of its
#: one-sided gradients (``switch.maximum``'s half of each at a tie), or zero where it guards a length, so its length
#: there is below one. Where it is at or above this, the kink's normal is that mean's direction; below it, zero.
KINK = 0.5


class SignValue(NamedTuple):
    """A region's cheap sign function at points (``Shape.sign_function``, ``Geometry.sign_function``): its ``value``,
    below zero inside the region, zero on its boundary and above zero outside it, and its gradient's length there
    (``slope``), so that ``value / slope`` is its first-order distance from the boundary, in metres."""

    value: Any
    slope: Any


#: The capabilities a shape derives from its signed distance where it gives none of its own (its analytic overrides).
DERIVED = ("outward_normal", "axis", "axis_point", "sign_function")


class Shape(ABC):
    """A region of space, evaluated at a point by its exact signed distance (negative inside, zero on its boundary):
    the one method a shape must give, with ``parameter_names``. Every function is traceable, and finite with finite
    derivatives at every point, including where the normal is not defined (a length is guarded at zero, a switch
    taken through ``mime.core.switch``).

    **Derived by default**, each an override a shape may give for speed or exactness (MIME's closed forms give them):

    - ``outward_normal``: the unit gradient of the distance by autodiff, pointing out of the region; zero on the medial
      set, where the gradient's length is below ``KINK`` (the switch ``<name>.kink``, declared when the shape is
      registered as ``name``).
    - ``axis`` and ``axis_point``: none. A region with an axis gives them, a direction and a point on it, and is taken
      to be uniform along it: its signed distance at a point on the axis is minus its radius there and everywhere.
    - ``sign_function``: the camera's sign test on arrays of points' coordinates (:class:`SignValue`), the distance
      itself with slope one, so a shape with only a distance is imaged by the camera's sampled route.

    The display mesh is derived from the distance too, by a geometry kind whose regions are shapes
    (``mime.registry.Geometry.drawing``, contoured by :func:`contour`): a drawing, marked as one. A capability added
    later comes as one more method here with its derived default, so a shape that gives only its distance gains it
    unchanged.

    **Declarations.** A shape whose functions also take an array of points ``(..., 3)``, evaluating each (its distances
    ``(...)``, its normals ``(..., 3)``), declares ``takes_many = True``: a containment contact holds a body's hull
    points against it, many at once; a derived capability evaluates a shape that declares nothing one point at a time.
    Such a shape gives ``curvature(parameters, points)``, its outward normal's derivative in the point
    ``(..., 3, 3)``, which the contact finds the body's deepest point with; and one whose region is convex declares
    ``convex = True``, so that a contact needs only the points of each group of a body's samples that lie on that
    group's convex hull (the wall's signed distance from inside a convex region is a convex function, whose largest
    value over a set is at a vertex of the set's hull). A shape that declares nothing is not taken to be convex. The
    declaration is the shape's word: the shape suite refutes a wrong one where two drawn points show it
    (``mime.testing.convexity_problem``: the signed distance at their midpoint above their mean), and no check proves
    one, so a non-convex region declared convex would cut a group's deepest point away wherever no draw happened to
    show it.

    A lab's shape subclasses this class. One registered without subclassing it (giving the same methods by name) is
    evaluated with the derived forms of those it lacks."""

    parameter_names: tuple[str, ...] = ()
    takes_many: bool = False
    convex: bool = False
    #: The owner of the derived normal's switch: ``<name>.kink``, set when the shape is registered as ``name``.
    kink_owner: str = "shape.kink"

    @abstractmethod
    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        """The exact signed distance at ``point``, m."""

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        """The unit gradient of the signed distance by autodiff; zero where its length is below ``KINK``."""
        return derived_normal(self, parameters, point, owner=self.kink_owner)

    def axis(self, parameters: Parameters) -> Point | None:
        return None

    def axis_point(self, parameters: Parameters) -> Point | None:
        return None

    def sign_function(self, parameters: Parameters, point: tuple[Any, Any, Any]) -> SignValue:
        """The camera's sign test at points given as three arrays of their coordinates: the signed distance, slope
        one."""
        x, y, z = jnp.broadcast_arrays(*(jnp.asarray(c) for c in point))
        value = each_point(self, self.signed_distance, parameters, jnp.stack([x, y, z], axis=-1))
        return SignValue(value, jnp.ones_like(value))


def each_point(shape: Any, function: Callable[[Parameters, Any], Any], parameters: Parameters, points: Any) -> Any:
    """``function(parameters, point)`` at each point of ``points`` (``(..., 3)``): at once where the shape takes many,
    one at a time where it does not."""
    if points.ndim == 1 or getattr(shape, "takes_many", False):
        return function(parameters, points)
    def one(point: Any) -> Any:
        return function(parameters, point)

    found = _vmap(one)(points.reshape(-1, 3))
    return found.reshape(*points.shape[:-1], *found.shape[1:])


def derived_normal(shape: Any, parameters: Parameters, point: Point, *, owner: str) -> Point:
    """The unit gradient of ``shape``'s signed distance at each point, by autodiff, and zero where the gradient's
    length is below ``KINK``, the switch ``owner``. Its derivative is the distance's second derivative's, which the
    shape's contract keeps finite."""
    def slope(held: Parameters, at: Any) -> Any:
        return _grad(lambda q: shape.signed_distance(held, q))(at)  # pyright: ignore[reportUnknownLambdaType]

    gradient = each_point(None, slope, parameters, jnp.asarray(point))
    squared = jnp.sum(gradient * gradient, axis=-1)
    sharp = squared > KINK * KINK
    length = jnp.sqrt(switch.domain_guard(sharp, squared, 1.0))
    return switch.where(sharp[..., None], gradient / length[..., None], jnp.zeros_like(gradient), owner=owner,
                        resolution=lambda: switch.rounding(jnp.asarray(KINK, dtype=gradient.dtype)),
                        indicator=lambda: jnp.sqrt(squared) - KINK)


def kink_switch(name: str) -> switch.Declaration:
    """The derived normal's switch of the shape registered as ``name``: its medial set, where the gradient's length
    falls below ``KINK``."""
    return switch.Declaration(
        owner=f"{name}.kink",
        indicator="the length of the signed distance's gradient less KINK (one half), dimensionless: below zero on the "
                  "shape's medial set, where the distance has a kink",
        orders={"normal": -1},
        route="none: on the medial set the nearest boundary point is not unique and the normal has no limit; the "
              "derived normal is zero there, with the derivative zero")


def derives(shape: Shape, capability: str) -> bool:
    """Whether ``shape`` takes ``capability`` (one of ``DERIVED``) from its signed distance, giving none of its own."""
    return capability not in vars(shape) and getattr(type(shape), capability) is getattr(Shape, capability)


class _Adopted(Shape):
    """A shape registered without subclassing :class:`Shape`: its own methods by name, and the derived forms of those
    it lacks; every other attribute (``curvature``, ``tangent``) is its own."""

    def __init__(self, own: Any) -> None:
        self.own = own
        self.parameter_names = tuple(own.parameter_names)
        for name in (*DERIVED, "takes_many", "convex"):
            if hasattr(own, name):
                setattr(self, name, getattr(own, name))

    @override
    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        return self.own.signed_distance(parameters, point)

    def __getattr__(self, name: str) -> Any:
        if name == "own":
            raise AttributeError(name)
        return getattr(self.own, name)


_given: dict[str, object] = {}
_shapes: dict[str, Shape] = {}
_switches: dict[str, tuple[switch.Declaration, ...]] = {}


def register_shape[S](name: str, shape: S, *, switches: tuple[switch.Declaration, ...] = ()) -> S:
    """Make ``shape`` evaluable by nodes under ``name``, with the ``switches`` its functions evaluate, declared beside
    its class (orders per capability: ``distance``, ``normal``). A shape that takes its normal from its distance
    declares that normal's switch, ``<name>.kink``, here. A shape need only subclass :class:`Shape`; one that gives
    the same methods by name without subclassing it is taken as it is, with the derived forms of the methods it
    lacks. Registering the same object again changes nothing."""
    held = _given.get(name)
    if held is shape:
        return shape
    if held is not None:
        raise ValueError(f"shape {name!r} is already registered as {held!r}")
    if not callable(getattr(shape, "signed_distance", None)):
        raise TypeError(f"shape {name!r} gives no signed_distance: it is the one method a shape must give")
    evaluated = shape if isinstance(shape, Shape) else _Adopted(shape)
    if derives(evaluated, "outward_normal"):
        evaluated.kink_owner = f"{name}.kink"
        switches = (*switches, kink_switch(name))
    _given[name], _shapes[name], _switches[name] = shape, evaluated, switches
    return shape


def shape_switches(name: str) -> tuple[switch.Declaration, ...]:
    """The switches the shape registered as ``name`` declares: an entry whose surfaces or regions it evaluates
    inherits them."""
    shape_named(name)
    return _switches[name]


def shape_named(name: str) -> Shape:
    """The shape registered as ``name``, with every capability: its own and the derived."""
    if name not in _shapes:
        raise KeyError(f"no shape {name!r} is registered; the shapes are {sorted(_shapes)}")
    return _shapes[name]


def shape_names() -> list[str]:
    return sorted(_shapes)


def shape_package(name: str) -> str:
    """The top-level package whose class the shape registered as ``name`` is: whose shape it is."""
    shape_named(name)
    return type(_given[name]).__module__.split(".")[0]


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
