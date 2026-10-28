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
#: Newton steps polishing a root after the root find, each lifted by ``w (1 + softplus(-lowest / w))``, ``w =
#: |gradient|^(2/3)`` the curvature a quartic valley has at that gradient: where the Hessian at the nearest point
#: vanishes (the focal set) the root find's fixed lift stalls, and the polish still converges; elsewhere its lift
#: vanishes with the gradient and its steps are Newton's.
SCREW_POLISH = 16
#: Starts per lobe of the cross-section, spread evenly about the point's own angle: two on each lobe and two in each
#: groove between, so that each flank of a groove has a start of its own. The first start is at the point's own angle
#: and the last opposite it, so that the seam's first and last roots are the two that tie on the axis.
SCREW_STARTS_PER_LOBE = 4
#: The low bits of a significand that :func:`_halves` clears, by the width in bytes of the float: 27 of float64's 52
#: stored bits and 12 of float32's 23, so that the product of two high halves, or of a high and a low half, is exact.
_LOW_BITS = {8: (jnp.uint64, 27), 4: (jnp.uint32, 12)}


def _halves(a: Array) -> tuple[Array, Array]:
    """``a`` as ``high + low`` exactly, ``high`` the leading half of its significand. The low bits are cleared by a
    mask rather than by Veltkamp's product, which a multiply-add contraction can change (XLA's compiled code
    contracts)."""
    unsigned, low_bits = _LOW_BITS[a.dtype.itemsize]
    mask = unsigned(~((1 << low_bits) - 1) & ((1 << (8 * a.dtype.itemsize)) - 1))
    high = lax.bitcast_convert_type(lax.bitcast_convert_type(a, unsigned) & mask, a.dtype)
    return high, a - high


def _exact_product(a: Array, b: Array) -> tuple[Array, Array]:
    """``a b`` as its rounded value and its rounding error (Dekker's product), whose sum is ``a b`` to the rounding of
    the two low halves' product, about ``eps^2`` of it."""
    product = a * b
    (a_high, a_low), (b_high, b_low) = _halves(a), _halves(b)
    return product, ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low


def _exact_sum(a: Array, b: Array) -> tuple[Array, Array]:
    """``a + b`` as its rounded value and its rounding error, whose sum is ``a + b`` exactly (Knuth's sum)."""
    total = a + b
    back = total - a
    return total, (a - (total - back)) + (b - back)


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
    surface's medial set: where two roots are equally near, the axis among them), polished, and one Newton step from
    it with the gradients flowing gives the nearest point's derivatives, except on the focal set where the seam ends
    (the switch ``dejongh_screw.focus``), where that step is singular. The signed distance is the offset from the nearest point along the normal there, the normal the surface's own there,
    and the tangent the thread's: the surface's direction along the axis at constant angle. The modulation is above
    zero: at zero every point of the circle is nearest the axis, and the nearest point there has no derivative."""

    parameter_names: tuple[str, ...] = ("centre", "orientation", "radius", "length", "wavenumber", "modulation",
                                        "starts")

    @staticmethod
    def turn(parameters: Parameters, along: Array) -> tuple[Array, Array]:
        """The cosine and sine of the cross-section's turn at ``along`` (axial positions in the screw's frame),
        ``wavenumber (along + length / 2) / radius``, each to the rounding of one however far along the axis. Taken in
        the working precision the turn rounds at its own size, ``eps |turn|`` (``eps wavenumber length / radius`` at a
        body's far end, turning the cross-section there by that), and so does a station chosen in it, a float whose
        own turn rounds alike; so the turn is carried to twice the working precision (a compensated sum and product),
        and its station is the nearest quarter turn, which ``cos`` and ``sin`` reduce it to exactly: from either
        neighbouring station the result is the same to their rounding. Its derivatives are the turn's: the correction
        carries none."""
        dtype = jnp.result_type(along, parameters["radius"], parameters["wavenumber"], parameters["length"])
        along = jnp.asarray(along, dtype=dtype)
        radius, nu = jnp.asarray(parameters["radius"], dtype=dtype), jnp.asarray(parameters["wavenumber"], dtype=dtype)
        rate = nu / radius
        product, error = _exact_product(rate, radius)
        rate_low = ((nu - product) - error) / radius
        offset, offset_low = _exact_sum(along, jnp.asarray(parameters["length"], dtype=dtype) / 2.0)
        turn, error = _exact_product(rate, offset)
        low = error + (rate * offset_low + rate_low * offset)
        cos, sin = jnp.cos(turn), jnp.sin(turn)
        kept = 1.0 - low * low / 2.0  # cos(low) to second order: in float32 the low part is 1e-3 at 20,000 rad
        return cos * kept - sin * low, sin * kept + cos * low

    def _frame(self, parameters: Parameters, point: Point) -> tuple[Array, Array, tuple[Array, Array]]:
        """The screw's frame as columns, the point in it, and the cosine and sine of the cross-section's turn at the
        point's x."""
        frame = quaternion.rotation_matrix(jnp.asarray(parameters["orientation"], dtype=point.dtype))
        local = frame.T @ (point - jnp.asarray(parameters["centre"], dtype=point.dtype))
        return frame, local, self.turn(parameters, local[0])

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
    def _derivatives(residual: Callable[[Any, Any], Array], theta: Any, tau: Any) -> tuple[Array, Array, Array]:
        """The gradient and the Hessian of half the residual's square, elementwise over the starts: ``(2, n)`` and
        ``(2, 2, n)``; and the surface's second derivatives in ``theta theta``, ``theta tau`` and ``tau tau``,
        ``(3, 3, n)``."""
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
        return gradient, gauss + curvature, jnp.stack([r_tt, r_ts, r_ss])

    @staticmethod
    def _solve(matrix: Array, vector: Array) -> Array:
        """``matrix^-1 vector`` for 2 x 2 matrices, elementwise over the last axis."""
        a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        determinant = a * d - b * c
        return jnp.stack([d * vector[0] - b * vector[1], a * vector[1] - c * vector[0]]) / determinant

    @staticmethod
    def _lowest(hessian: Array) -> Array:
        """The smallest eigenvalue of each symmetric 2 x 2 matrix, elementwise over the last axis."""
        a, b, d = hessian[0, 0], hessian[0, 1], hessian[1, 1]
        return (a + d) / 2.0 - jnp.sqrt(((a - d) / 2.0) ** 2 + b ** 2)

    def _lifted(self, held: tuple[Array, Array], gradient: Array, matrix: Array) -> tuple[Array, Array]:
        """One step from ``held`` solving with the lifted Hessian ``matrix``, shortened smoothly to about
        ``SCREW_TRUST`` at most."""
        step = self._solve(matrix, gradient)
        step = step * lax.rsqrt(1.0 + jnp.sum(step * step, axis=0) / SCREW_TRUST ** 2)
        return held[0] - step[0], held[1] - step[1]

    def _polish(self, residual: Callable[[Any, Any], Array], theta: Array, tau: Array) -> tuple[Array, Array]:
        """``SCREW_POLISH`` steps from the roots ``theta, tau`` (elementwise), each lifted in proportion to the
        gradient to the two-thirds: a root the fixed lift left short in a valley with no curvature converges."""
        shift = jnp.eye(2, dtype=theta.dtype)[:, :, None]
        tiny = float(jnp.finfo(theta.dtype).tiny)

        def polish(_: int, held: tuple[Array, Array]) -> tuple[Array, Array]:
            gradient, hessian = self._derivatives(residual, *held)[:2]
            width = jnp.cbrt(jnp.sum(gradient * gradient, axis=0) + tiny)
            lift = (1.0 + jnp.logaddexp(0.0, -self._lowest(hessian) / width)) * width
            return self._lifted(held, gradient, hessian + lift * shift)

        return lax.fori_loop(0, SCREW_POLISH, polish, (theta, tau))

    def _root(self, parameters: Parameters, local: Array, turned: tuple[Array, Array]) -> tuple[Array, Array]:
        """The nearest surface point's angle and axial offset (in the reference radius), its gradients stopped;
        ``turned`` the cosine and sine of the cross-section's turn at the point."""
        starts = int(parameters["starts"])
        held = lax.stop_gradient((local, turned, parameters["radius"], parameters["wavenumber"],
                                  parameters["modulation"]))
        x_local, across, radius, nu, epsilon = held
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
            gradient, hessian = self._derivatives(residual, *held)[:2]
            lift = jnp.logaddexp(0.0, -SCREW_SHARPNESS * self._lowest(hessian)) / SCREW_SHARPNESS + SCREW_FLOOR
            return self._lifted(held, gradient, hessian + lift * shift)

        theta, tau = lax.fori_loop(0, SCREW_ITERATIONS, descend, (theta, tau))
        squared = jnp.sum(residual(theta, tau) ** 2, axis=0)
        roots = jnp.stack([theta, tau, squared])

        def gap() -> Array:
            """How much nearer the nearest root is than the nearest other one, in metres, each root polished: zero on
            the seam."""
            polished = self._polish(residual, theta, tau)
            distance = jnp.sqrt(jnp.sum(residual(*polished) ** 2, axis=0))
            best = jnp.argmin(distance)
            apart = jnp.abs(jnp.angle(jnp.exp(1j * (polished[0] - polished[0][best])))) + \
                jnp.abs(polished[1] - polished[1][best])
            other = jnp.min(jnp.where(apart > 1e-6, distance, jnp.inf))
            return (other - distance[best]) * radius

        chosen = roots[:, 0]
        for start in range(1, count):
            chosen = switch.where(roots[2, start] < chosen[2], roots[:, start], chosen, owner="dejongh_screw.seam",
                                  resolution=lambda: switch.rounding(radius), indicator=gap)
        theta, tau = self._polish(residual, chosen[0][None], chosen[1][None])
        return theta[0], tau[0]

    def _newton_step(self, parameters: Parameters, residual: Callable[[Any, Any], Array], theta: Array, tau: Array,
                     dtype: Any) -> Array:
        """One Newton step from the stopped root ``theta, tau`` with the gradients flowing: the nearest point's
        derivatives, the implicit function's. Its Hessian, of half the squared distance in the reference radius, is
        ``H = I - s II`` (the surface's first and second fundamental forms, ``s`` the point's signed distance), singular
        where the point lies at a centre of curvature of its nearest surface point: the switch ``dejongh_screw.focus``,
        whose set is where the medial set ends, each crest's and each trough's centre of curvature turned along the
        axis. There the set is the cusp of the caustic, so a point ``delta`` from it along the normal (``delta``, in
        the radius, the smaller root of ``det(u II - H) = det(II) u^2 + b u + det H``) has a nearest point analytic
        within only about ``R delta^(3/2)`` of it across the normal: that is the indicator. Within ``AXIS_ULPS`` units
        in the last place of the radius of the set in the indicator, the step is not taken, and the root keeps its
        value, polished to rounding: there ``delta`` is at most that band to the two-thirds, ``w``, about the turn a
        point's own rounding gives its nearest point's centre of curvature, and ``|det H| <= w |b|`` (to first order in
        ``w`` over the other centre's distance)."""
        gradient, hessian, second = self._derivatives(residual, theta[None], tau[None])
        nu, epsilon, starts = parameters["wavenumber"], parameters["modulation"], int(parameters["starts"])
        rho = 1.0 + epsilon * jnp.sin(starts * theta)
        slope = epsilon * starts * jnp.cos(starts * theta)
        turned = theta + nu * tau
        c, s = jnp.cos(turned), jnp.sin(turned)
        normal = jnp.stack([nu * rho * slope, rho * c + slope * s, rho * s - slope * c])
        squared = jnp.sum(normal * normal)
        form = jnp.sum(second[:, :, 0] * normal[None, :], axis=1)  # II_tt, II_ts, II_ss, times |normal|
        a, b, d = hessian[0, 0, 0], hessian[0, 1, 0], hessian[1, 1, 0]
        determinant = a * d - b * b
        linear = -(form[0] * d + form[2] * a - 2.0 * form[1] * b)  # b, times |normal|
        width = (AXIS_ULPS * float(jnp.finfo(dtype).eps)) ** (2.0 / 3.0)
        off = determinant * determinant * squared > (width * linear) ** 2
        g = gradient[:, 0]
        newton = jnp.stack([d * g[0] - b * g[1], a * g[1] - b * g[0]]) / switch.domain_guard(off, determinant, 1.0)

        def nearer_centre() -> Array:
            quadratic = (form[0] * form[2] - form[1] * form[1]) / squared
            unit = linear / jnp.sqrt(squared)
            root = jnp.sqrt(jnp.maximum(unit * unit - 4.0 * quadratic * determinant, 0.0))
            q = -(unit + jnp.where(unit < 0.0, -root, root)) / 2.0
            return parameters["radius"] * jnp.minimum(jnp.abs(q / quadratic), jnp.abs(determinant / q)) ** 1.5

        return switch.where(off, newton, jnp.zeros_like(newton), owner="dejongh_screw.focus",
                            resolution=lambda: _band(parameters, "radius", dtype), indicator=nearer_centre)

    def _nearest(self, parameters: Parameters, point: Point) -> tuple[Array, Array, Array, Array, Array]:
        """The point in the screw's turned frame (in the reference radius), the nearest surface point and its normal
        and thread there, in that frame, and the frame back to the world's (columns)."""
        frame, local, (cos, sin) = self._frame(parameters, point)
        radius, nu = parameters["radius"], parameters["wavenumber"]
        epsilon, starts = parameters["modulation"], int(parameters["starts"])
        u = (cos * local[1] + sin * local[2]) / radius
        v = (cos * local[2] - sin * local[1]) / radius
        theta, tau = self._root(parameters, local, (cos, sin))
        step = self._newton_step(parameters, self._residual(nu, epsilon, starts, u, v), theta, tau, point.dtype)
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


#: Each shape's switches: its medial set, the capsule's end planes, across which only the curvature changes, and the
#: screw's focal set. An order of 1 or more is the theory's (a distance from a convex set is once differentiable off
#: it): the switch harness measures a jump against anything continuous, so it checks such an order as at least 0.
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
    switch.Declaration(
        owner="dejongh_screw.focus",
        indicator="R (delta / R)^(3/2), m, delta the distance along the normal from the point to the nearer centre "
                  "of curvature of its nearest surface point: how far across the normal the nearest point stays "
                  "analytic, the set being the caustic's cusp; zero on the focal set, where the seam ends (each "
                  "crest's and each trough's centre of curvature, turned along the axis: a line); within AXIS_ULPS "
                  "units in the last place of the radius (a band)",
        orders={"distance": 1, "normal": 0, "tangent": 0},
        route="none: on the focal set the nearest point moves without bound as the point does, so the normal and the "
              "thread are continuous with no derivative there, and the distance once differentiable; within the band "
              "the Newton step, singular there, is not taken, so the distance's derivative is its partial with the "
              "nearest point held (the whole derivative at a stationary root), and the normal's and the thread's "
              "are theirs with the nearest point held",
        unmeasured="its branches take the root, polished to rounding, with the Newton step or without it: one value, "
                   "differing in its derivative only, which no output's value shows (nearer the set than about a "
                   "fiftieth of the radius the step's rounding over the vanishing Hessian shows in the normal's last "
                   "digits, a noise and no order)"),
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
