"""The prescribed inlet of an open vessel (the maintainer's inflow decision (A), 2026-11-12): an ``Inflow``'s opening holds
its velocity, not its flux alone. Its normal velocity is the developed profile of a round section scaled to the law's
flux at the instant asked, its tangential velocity zero; the open grid holds the plane (``mime.nodes.flow.fvm._open``'s
prescribed plane: the pressure Neumann there).

**The profile** (the open-ports spike's velocity inlet, its q5b). For the law's ``Q(t) = Q_0 + A cos(omega t + phi)``
into the field, the face ``w = n_in (Q_0 S_0 + Re[A e^{i (omega t + phi)} S_1])``, ``n_in`` +1 on the low plane and -1 on
the high: ``S_0`` Poiseuille's shape ``(1 - r^2 / R^2)`` and ``S_1`` Womersley's ``1 - J0(b r) / J0(b R)``
(``mime.nodes.flow._womersley.profile``, its series and its validity line), each weighted by the faces' fluid share
``1 - chi`` and divided by its own sum over the plane times ``h^2``, so that the faces' flux is the law's to rounding at
any resolution and wall position. ``r`` is the distance from the section's centre on the plane, ``R`` its radius:
the tube's, or the region builder's run, which reaches the plane straight along the axis at its cut's radius, so the
section a law's opening has in a built region is round. A negative frequency takes the conjugate shape. **Past the
series' line** (its dtype's, ``VALIDITY``: 50 in float64, 25 in float32) the harmonic's shape is the Stokes layer's,
``1 - (r / R)^{-1/2} e^{-(1 + i)(R - r) / delta}``, ``delta = R sqrt(2) / alpha`` (``r`` held at least ``R / alpha``),
within 9.5e-5 of the exact shape's peak at alpha 25 and 2.4e-5 at 50 once both are scaled to the law's flux (against
SciPy's Bessel functions on the section, measured); each form is evaluated inside its own range, so neither's
derivative is a NaN.

**What it is not** (stated in the open vessel's validity text, as measured): the developed profile of a straight round
tube, not of the discrete field next to the plane (its band, its resolution, convection), so the field adjusts to it
over a development length past the opening; and the profile's own convective term is not zero on the grid."""

from __future__ import annotations

from typing import Any, NamedTuple

from jax import Array

import numpy as np

from mime.core import jnp
from mime.nodes.flow._womersley import VALIDITY, profile


class InletProfile(NamedTuple):
    """The prescribed plane's profile: its steady shape and the harmonic's (its real and imaginary parts), each in
    1/m^2 (unit flux over the plane), the law's row (steady and amplitude in m^3/s, frequency in Hz, phase in rad) and
    the sign of the inward normal along z."""

    steady: Array
    harmonic_re: Array
    harmonic_im: Array
    parameters: Array
    inward: float

    def at(self, instant: Any, flux: Any = None) -> Array:
        """The plane's ``w`` (m/s, ``nx x ny``) at ``instant`` (s): its flux into the field ``flux`` (m^3/s), the law's
        as its row states it at the instant, or by default its waveform's ``Q(instant)``; the harmonic's share on its
        own shape, the rest (the steady part, and whatever the law's flux holds beyond its waveform) on the steady one."""
        steady, amplitude, frequency, phase = (self.parameters[k] for k in range(4))
        theta = 2.0 * jnp.pi * frequency * instant + phase
        harmonic = jnp.cos(theta) * self.harmonic_re - jnp.sin(theta) * self.harmonic_im
        held = steady if flux is None else flux - amplitude * jnp.cos(theta)
        return self.inward * (held * self.steady + amplitude * harmonic)

    def rate(self, instant: Any) -> Array:
        """The plane's ``dw/dt`` (m/s^2) at ``instant``: what a scheme that starts from a consistent pressure reads."""
        _, amplitude, frequency, phase = (self.parameters[k] for k in range(4))
        omega = 2.0 * jnp.pi * frequency
        theta = omega * instant + phase
        return -self.inward * amplitude * omega * (jnp.sin(theta) * self.harmonic_re + jnp.cos(theta) * self.harmonic_im)


def inlet_profile(points: Array, centre: Array, radius: Any, fluid: Array, viscosity: Any, parameters: Array,
                  spacing: Any, terms: int, inward: float) -> InletProfile:
    """The profile on a plane's faces at ``points`` (m, ``nx x ny x 2``, their x and y), the section centred at
    ``centre`` (m, its x and y) of ``radius`` (m), the faces' fluid share ``fluid`` (``nx x ny``), the kinematic
    ``viscosity`` (m^2/s) and the law's row ``parameters``; ``terms`` the Womersley series' count (its dtype's)."""
    offset = points - centre
    rho_squared = jnp.sum(offset * offset, axis=-1) / (radius * radius)
    area = spacing * spacing
    steady = fluid * jnp.clip(1.0 - rho_squared, 0.0, None)
    steady = steady / (area * jnp.sum(steady))
    frequency = parameters[2]
    alpha_squared = radius * radius * 2.0 * jnp.pi * jnp.abs(frequency) / viscosity
    line, _ = VALIDITY[np.dtype(points.dtype).name]
    inside = jnp.clip(rho_squared, 0.0, 1.0)
    series = _scaled(fluid, profile(inside, jnp.minimum(alpha_squared, line * line), terms), area)
    layer = _scaled(fluid, _stokes_layer(inside, jnp.sqrt(jnp.maximum(alpha_squared, line * line))), area)
    held = alpha_squared <= line * line
    harmonic_re = jnp.where(held, series[0], layer[0])
    harmonic_im = jnp.where(held, series[1], layer[1])
    harmonic_im = jnp.where(frequency < 0.0, -harmonic_im, harmonic_im)
    return InletProfile(steady, harmonic_re, harmonic_im, parameters, inward)


def _stokes_layer(rho_squared: Array, alpha: Any) -> tuple[Array, Array]:
    """The harmonic's shape past the series' line, ``1 - (r / R)^{-1/2} e^{-(1 + i) a}``, ``a = alpha (1 - r / R) /
    sqrt(2)``, ``r / R`` held at least ``1 / alpha`` (where the layer's term is under ``e^{-alpha / sqrt(2)}``), as
    (re, im)."""
    held = jnp.sqrt(jnp.maximum(rho_squared, 1.0 / (alpha * alpha)))
    a = alpha * (1.0 - held) / jnp.sqrt(2.0)
    amplitude = jnp.exp(-a) / jnp.sqrt(held)
    return 1.0 - amplitude * jnp.cos(a), amplitude * jnp.sin(a)


def _scaled(fluid: Array, shape: tuple[Array, Array], area: Any) -> tuple[Array, Array]:
    """``fluid`` times the complex ``shape``, divided by its own sum times ``area``: unit flux over the plane."""
    weighted_re, weighted_im = fluid * shape[0], fluid * shape[1]
    total_re, total_im = area * jnp.sum(weighted_re), area * jnp.sum(weighted_im)
    size = total_re * total_re + total_im * total_im
    return (weighted_re * total_re + weighted_im * total_im) / size, (weighted_im * total_re - weighted_re * total_im) / size


__all__ = ["InletProfile", "inlet_profile"]
