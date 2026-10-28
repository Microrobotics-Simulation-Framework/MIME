"""The shapes of the solid geometry kinds: a ball, a capsule, a slab and de Jongh's screw, each a :class:`Shape` with
its switches declared beside it.

Every shape's medial set (the centre, the segment, the mid-plane, the screw's seam) is where the nearest point of its
boundary is not unique: there the distance has a kink and the normal no limit. A point within ``AXIS_ULPS`` units in
the last place of the shape's radius (or half-thickness) of a centre, segment or mid-plane is on it, whatever its
direction: there the signed distance is its value on the set and the normal zero, each with the derivative zero in the
point (the derivative of the value computed there)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import jax
from jax import Array
from jaxtyping import Float

from mime.core import _quaternion as quaternion
from mime.core import switch
from mime.core._geometry import AXIS_ULPS, Parameters, Point, register_shape
from mime.core._node import jnp

#: ``jax.jvp`` and ``jax.lax``, whose hints name a device type from jaxlib's compiled core, which ships no stubs.
_jvp: Any = cast(Any, jax).jvp
lax: Any = cast(Any, jax).lax


def _band(parameters: Parameters, scale: str, dtype: Any) -> Any:
    """The medial band's width: ``AXIS_ULPS`` units in the last place of the shape's ``scale`` parameter."""
    return AXIS_ULPS * float(jnp.finfo(dtype).eps) * parameters[scale]


def _off_the_set(offset: Array, resolution: Any, *, owner: str) -> tuple[Array, Array]:
    """The length of ``offset`` (the point's offset from a medial set) and whether the point is off the set, where
    the set is the switch ``owner``: zero within ``resolution`` of it."""
    squared = jnp.sum(offset * offset)
    off = squared > resolution ** 2
    length = switch.where(off, jnp.sqrt(switch.domain_guard(off, squared, 1.0)), 0.0, owner=owner,
                          resolution=resolution, indicator=lambda: jnp.sqrt(squared))
    return length, off


def _direction(offset: Array, length: Array, off: Array, resolution: Any, *, owner: str) -> Array:
    """``offset`` over its length off the medial set, zero on it: the switch ``owner``."""
    return switch.where(off, offset / switch.domain_guard(off, length, 1.0), jnp.zeros_like(offset), owner=owner,
                        resolution=resolution, indicator=lambda: jnp.sqrt(jnp.sum(offset * offset)))


def _unit_axis(parameters: Parameters, dtype: Any) -> Array:
    """The shape's ``axis`` scaled to unit length: a direction, which the plan refuses at zero."""
    axis = jnp.asarray(parameters["axis"], dtype=dtype)
    return axis / switch.nonzero_norm(axis)


class Sphere:
    """A ball: ``centre`` and ``radius``. Its normal is radial, with no limit at the centre."""

    parameter_names: tuple[str, ...] = ("centre", "radius")

    def _offset(self, parameters: Parameters, point: Point) -> tuple[Array, Array, Array, Any]:
        offset = point - jnp.asarray(parameters["centre"], dtype=point.dtype)
        resolution = _band(parameters, "radius", point.dtype)
        length, off = _off_the_set(offset, resolution, owner="sphere.centre")
        return offset, length, off, resolution

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        return self._offset(parameters, point)[1] - parameters["radius"]

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        offset, length, off, resolution = self._offset(parameters, point)
        return _direction(offset, length, off, resolution, owner="sphere.centre")

    def axis(self, parameters: Parameters) -> Point | None:
        return None

    def axis_point(self, parameters: Parameters) -> Point | None:
        return None


class Capsule:
    """A capsule: a cylinder of ``radius`` and ``height`` (between the centres of its caps, as USD's ``Capsule``
    counts it) along ``axis`` through ``centre``, capped by two hemispheres. Its distance is the distance from its
    segment less the radius: across the planes through the segment's ends only the curvature changes, and on the
    segment the normal has no limit."""

    parameter_names: tuple[str, ...] = ("centre", "axis", "radius", "height")

    def _offset(self, parameters: Parameters, point: Point) -> tuple[Array, Array, Array, Any]:
        axis = _unit_axis(parameters, point.dtype)
        offset = point - jnp.asarray(parameters["centre"], dtype=point.dtype)
        along = jnp.dot(offset, axis)
        half = parameters["height"] / 2.0
        nearest = switch.clip(along, -half, half, owner="capsule.end", resolution=lambda: switch.rounding(half),
                              indicator=along)
        radial = offset - nearest * axis
        resolution = _band(parameters, "radius", point.dtype)
        length, off = _off_the_set(radial, resolution, owner="capsule.segment")
        return radial, length, off, resolution

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        return self._offset(parameters, point)[1] - parameters["radius"]

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        radial, length, off, resolution = self._offset(parameters, point)
        return _direction(radial, length, off, resolution, owner="capsule.segment")

    def axis(self, parameters: Parameters) -> Point | None:
        return None

    def axis_point(self, parameters: Parameters) -> Point | None:
        return None


class Slab:
    """The region between two parallel planes: ``centre`` midway between them, ``axis`` normal to them and
    ``half_thickness`` from the centre to each. Its normal is the axis on one side of the mid-plane and its opposite on
    the other, with no limit on the mid-plane."""

    parameter_names: tuple[str, ...] = ("centre", "axis", "half_thickness")

    def _offset(self, parameters: Parameters, point: Point) -> tuple[Array, Array, Array, Any]:
        axis = _unit_axis(parameters, point.dtype)
        offset = jnp.dot(point - jnp.asarray(parameters["centre"], dtype=point.dtype), axis) * axis
        resolution = _band(parameters, "half_thickness", point.dtype)
        length, off = _off_the_set(offset, resolution, owner="slab.mid")
        return offset, length, off, resolution

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        return self._offset(parameters, point)[1] - parameters["half_thickness"]

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        offset, length, off, resolution = self._offset(parameters, point)
        return _direction(offset, length, off, resolution, owner="slab.mid")

    def axis(self, parameters: Parameters) -> Point | None:
        return None

    def axis_point(self, parameters: Parameters) -> Point | None:
        return None


#: The screw's root find: Newton steps on the squared distance (in the reference radius squared) from each start,
#: each made a descent by shifting the Hessian by about its most negative eigenvalue (a softplus of width
#: ``1 / SCREW_SHARPNESS``) plus ``SCREW_FLOOR``, and shortened smoothly to about ``SCREW_TRUST`` (in radians and
#: radii) at most: near a minimum the shift is the floor and the steps Newton's.
SCREW_ITERATIONS = 40
SCREW_SHARPNESS = 20.0
SCREW_FLOOR = 1e-3
SCREW_TRUST = 0.3
#: Starts per lobe of the cross-section, spread evenly about the point's own angle: two on each lobe and two in each
#: groove between, so that each flank of a groove has a start of its own. The first start is at the point's own angle
#: and the last opposite it, so that the seam's first and last roots are the two that tie on the axis.
SCREW_STARTS_PER_LOBE = 4


class DeJonghScrew:
    """The inside of de Jongh's screw surface (de Jongh et al. 2025, Eq. 1), unbounded along its axis: a cross-section
    ``rho(theta) = R (1 + modulation sin(starts theta))`` turned along the axis by ``wavenumber / R`` radians per
    metre. ``orientation`` (a unit quaternion) turns the screw's frame onto the world's, the axis its x; ``centre`` is
    its frame's origin; the cross-section's angle is measured from the frame's y towards its z at ``x = -length / 2``,
    where the de Jongh body's end is, so that the body of length ``length`` centred on the origin is the 0.2 line's
    body moved half its length along the axis.

    The distance is the distance from the nearest point of the surface, found by a root find on the squared distance
    over the surface's two coordinates (the cross-section's angle and the axial offset) from ``starts *
    SCREW_STARTS_PER_LOBE`` starts about the point's own angle, its gradients stopped (``starts``, an integer, is
    structural: a step is traced with it); the nearest of the roots is chosen, the switch ``dejongh_screw.seam`` (the
    surface's medial set: where two roots are equally near, the axis among them), and one Newton step from it with the gradients flowing gives the nearest point's derivatives. The
    signed distance is the offset from the nearest point along the normal there, the normal the surface's own there,
    and the tangent the thread's: the surface's direction along the axis at constant angle. The modulation is above
    zero: at zero every point of the circle is nearest the axis, and the nearest point there has no derivative."""

    parameter_names: tuple[str, ...] = ("centre", "orientation", "radius", "length", "wavenumber", "modulation",
                                        "starts")

    def _frame(self, parameters: Parameters, point: Point) -> tuple[Array, Array, Array]:
        """The screw's frame as columns, the point in it, and the turn of the cross-section at the point's x."""
        frame = quaternion.rotation_matrix(jnp.asarray(parameters["orientation"], dtype=point.dtype))
        local = frame.T @ (point - jnp.asarray(parameters["centre"], dtype=point.dtype))
        turn = parameters["wavenumber"] * (local[0] + parameters["length"] / 2.0) / parameters["radius"]
        return frame, local, turn

    @staticmethod
    def _residual(nu: Any, epsilon: Any, starts: int, u: Any, v: Any) -> Callable[[Any, Any], Array]:
        """The offset, in the reference radius, from the point (at ``u, v`` in the cross-section through it, turned so
        that the angle there is the surface's own) to the surface point at angle ``theta`` and ``tau`` radii along
        the axis."""

        def residual(theta: Any, tau: Any) -> Array:
            radius = 1.0 + epsilon * jnp.sin(starts * theta)
            turned = theta + nu * tau
            return jnp.stack([tau, radius * jnp.cos(turned) - u, radius * jnp.sin(turned) - v])

        return residual

    @staticmethod
    def _derivatives(residual: Callable[[Any, Any], Array], theta: Any, tau: Any) -> tuple[Array, Array]:
        """The gradient and the Hessian of half the residual's square, elementwise over the starts: ``(2, n)`` and
        ``(2, 2, n)``."""
        ones, zeros = jnp.ones_like(theta), jnp.zeros_like(theta)

        def along(direction: tuple[Any, Any]) -> Callable[[Any, Any], Array]:
            return lambda a, b: _jvp(residual, (a, b), direction)[1]

        r, r_theta = _jvp(residual, (theta, tau), (ones, zeros))
        r_tau = along((zeros, ones))(theta, tau)
        r_tt = _jvp(along((ones, zeros)), (theta, tau), (ones, zeros))[1]
        r_ts = _jvp(along((ones, zeros)), (theta, tau), (zeros, ones))[1]
        r_ss = _jvp(along((zeros, ones)), (theta, tau), (zeros, ones))[1]
        jac = jnp.stack([r_theta, r_tau])
        gradient = jnp.sum(jac * r[None], axis=1)
        gauss = jnp.sum(jac[:, None] * jac[None, :], axis=2)
        curvature = jnp.stack([jnp.stack([jnp.sum(r * r_tt, axis=0), jnp.sum(r * r_ts, axis=0)]),
                               jnp.stack([jnp.sum(r * r_ts, axis=0), jnp.sum(r * r_ss, axis=0)])])
        return gradient, gauss + curvature

    @staticmethod
    def _solve(matrix: Array, vector: Array) -> Array:
        """``matrix^-1 vector`` for 2 x 2 matrices, elementwise over the last axis."""
        a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        determinant = a * d - b * c
        return jnp.stack([d * vector[0] - b * vector[1], a * vector[1] - c * vector[0]]) / determinant

    def _root(self, parameters: Parameters, local: Array, turn: Array) -> tuple[Array, Array]:
        """The nearest surface point's angle and axial offset (in the reference radius), its gradients stopped."""
        starts = int(parameters["starts"])
        held = lax.stop_gradient((local, turn, parameters["radius"], parameters["wavenumber"],
                                  parameters["modulation"]))
        x_local, turned, radius, nu, epsilon = held
        across = jnp.stack([jnp.cos(turned), jnp.sin(turned)])
        u = (across[0] * x_local[1] + across[1] * x_local[2]) / radius
        v = (across[0] * x_local[2] - across[1] * x_local[1]) / radius
        count = starts * SCREW_STARTS_PER_LOBE
        own = jnp.arctan2(v, u)
        order = [k for k in range(count) if k != count // 2] + [count // 2]
        theta = own + 2.0 * math.pi * jnp.asarray(order, dtype=local.dtype) / count
        tau = jnp.zeros_like(theta)
        residual = self._residual(nu, epsilon, starts, u, v)
        shift = jnp.eye(2, dtype=local.dtype)[:, :, None]

        def descend(_: int, held: tuple[Array, Array]) -> tuple[Array, Array]:
            gradient, hessian = self._derivatives(residual, *held)
            a, b, d = hessian[0, 0], hessian[0, 1], hessian[1, 1]
            lowest = (a + d) / 2.0 - jnp.sqrt(((a - d) / 2.0) ** 2 + b ** 2)
            lift = jnp.logaddexp(0.0, -SCREW_SHARPNESS * lowest) / SCREW_SHARPNESS + SCREW_FLOOR
            step = self._solve(hessian + lift * shift, gradient)
            step = step * lax.rsqrt(1.0 + jnp.sum(step * step, axis=0) / SCREW_TRUST ** 2)
            return held[0] - step[0], held[1] - step[1]

        theta, tau = lax.fori_loop(0, SCREW_ITERATIONS, descend, (theta, tau))
        squared = jnp.sum(residual(theta, tau) ** 2, axis=0)
        roots = jnp.stack([theta, tau, squared])

        def gap() -> Array:
            """How much nearer the nearest root is than the nearest other one, in metres: zero on the seam."""
            distance = jnp.sqrt(squared)
            best = jnp.argmin(distance)
            apart = jnp.abs(jnp.angle(jnp.exp(1j * (theta - theta[best])))) + jnp.abs(tau - tau[best])
            other = jnp.min(jnp.where(apart > 1e-6, distance, jnp.inf))
            return (other - distance[best]) * radius

        chosen = roots[:, 0]
        for start in range(1, count):
            chosen = switch.where(roots[2, start] < chosen[2], roots[:, start], chosen, owner="dejongh_screw.seam",
                                  resolution=lambda: switch.rounding(radius), indicator=gap)
        return chosen[0], chosen[1]

    def _nearest(self, parameters: Parameters, point: Point) -> tuple[Array, Array, Array, Array, Array]:
        """The point in the screw's turned frame (in the reference radius), the nearest surface point and its normal
        and thread there, in that frame, and the frame back to the world's (columns)."""
        frame, local, turn = self._frame(parameters, point)
        radius, nu = parameters["radius"], parameters["wavenumber"]
        epsilon, starts = parameters["modulation"], int(parameters["starts"])
        cos, sin = jnp.cos(turn), jnp.sin(turn)
        u = (cos * local[1] + sin * local[2]) / radius
        v = (cos * local[2] - sin * local[1]) / radius
        theta, tau = self._root(parameters, local, turn)
        residual = self._residual(nu, epsilon, starts, u, v)
        gradient, hessian = self._derivatives(residual, theta[None], tau[None])
        step = self._solve(hessian, gradient)[:, 0]
        theta, tau = theta - step[0], tau - step[1]
        rho = 1.0 + epsilon * jnp.sin(starts * theta)
        slope = epsilon * starts * jnp.cos(starts * theta)
        turned = theta + nu * tau
        c, s = jnp.cos(turned), jnp.sin(turned)
        surface = jnp.stack([tau, rho * c, rho * s])
        normal = jnp.stack([nu * rho * slope, rho * c + slope * s, rho * s - slope * c])
        thread = jnp.stack([jnp.ones_like(rho), -nu * rho * s, nu * rho * c])
        back = frame @ jnp.stack([jnp.stack([jnp.ones_like(cos), jnp.zeros_like(cos), jnp.zeros_like(cos)]),
                                  jnp.stack([jnp.zeros_like(cos), cos, -sin]),
                                  jnp.stack([jnp.zeros_like(cos), sin, cos])])
        return jnp.stack([jnp.zeros_like(u), u, v]), surface, normal / switch.nonzero_norm(normal), \
            thread / switch.nonzero_norm(thread), back

    def signed_distance(self, parameters: Parameters, point: Point) -> Float[Array, ""]:  # noqa: F722 - a jaxtyping shape
        at, surface, normal, _, _ = self._nearest(parameters, point)
        return jnp.dot(at - surface, normal) * parameters["radius"]

    def outward_normal(self, parameters: Parameters, point: Point) -> Point:
        _, _, normal, _, back = self._nearest(parameters, point)
        return back @ normal

    def tangent(self, parameters: Parameters, point: Point) -> Point:
        """The thread's direction at the nearest surface point: along the surface at constant angle, towards +x."""
        _, _, _, thread, back = self._nearest(parameters, point)
        return back @ thread

    def axis(self, parameters: Parameters) -> Point | None:
        return None

    def axis_point(self, parameters: Parameters) -> Point | None:
        return None


def _medial(owner: str, what: str) -> switch.Declaration:
    return switch.Declaration(
        owner=owner,
        indicator=f"the point's distance from {what}, m; on it within AXIS_ULPS units in the last place of the shape's "
                  f"size (a band)",
        orders={"distance": 0, "normal": -1},
        route=f"none: on {what} the distance is a cone and the normal has no limit, so neither has a derivative to "
              f"compute; within the band the distance is its value there and the normal zero, each with the "
              f"derivative zero in the point")


#: Each shape's switches: its medial set, and the capsule's end planes, across which only the curvature changes.
SPHERE_SWITCHES = (_medial("sphere.centre", "the centre"),)
CAPSULE_SWITCHES = (
    _medial("capsule.segment", "the segment between the caps' centres"),
    *(switch.Declaration(
        owner=f"capsule.end.{end}",
        indicator=f"the point's distance along the axis inside the plane through the segment's {side} end, m",
        orders={"distance": 1, "normal": 0},
        route="none needed: the distance from a segment is once differentiable off it, and the normal continuous; "
              "only the second derivative jumps there") for end, side in (("low", "lower"), ("high", "upper"))),
)
SLAB_SWITCHES = (_medial("slab.mid", "the mid-plane"),)
SCREW_SWITCHES = (
    switch.Declaration(
        owner="dejongh_screw.seam",
        indicator="how much nearer the nearest root of the surface is than the nearest other one, m: zero on the "
                  "surface's medial set (its axis among it), the seam of the frame read at a point",
        orders={"distance": 0, "normal": -1, "tangent": -1},
        route="none: on the seam two surface points are equally near, so the distance has a kink and the normal and "
              "the thread jump"),
)

SPHERE = register_shape("sphere", Sphere(), switches=SPHERE_SWITCHES)
CAPSULE = register_shape("capsule", Capsule(), switches=CAPSULE_SWITCHES)
SLAB = register_shape("slab", Slab(), switches=SLAB_SWITCHES)
DEJONGH_SCREW = DeJonghScrew()
register_shape("dejongh_screw", DEJONGH_SCREW, switches=SCREW_SWITCHES)


def nearest_point(shape: Any, parameters: Parameters, point: Point) -> Point:
    """The nearest point of ``shape``'s boundary: the point less its signed distance along its outward normal, exact
    off the shape's medial set, where the normal is zero and the point itself is given."""
    return point - shape.signed_distance(parameters, point) * shape.outward_normal(parameters, point)


__all__ = ["CAPSULE", "CAPSULE_SWITCHES", "DEJONGH_SCREW", "SCREW_SWITCHES", "SLAB", "SLAB_SWITCHES", "SPHERE",
           "SPHERE_SWITCHES", "Capsule", "DeJonghScrew", "Slab", "Sphere", "nearest_point"]
