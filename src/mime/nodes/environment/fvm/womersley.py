"""Analytical pulsatile flow solutions used for FVM verification.

Two geometries are supported:

- :func:`channel_velocity` — fully developed pulsatile flow between
  parallel plates at ``y = ±h``, driven by a body force
  ``f_x(t) = f_steady + f_osc · cos(ωt)``. The dimensionless
  parameter is the Womersley number ``Wo = h √(ω/ν)``. Closed-form
  solution uses ``cosh`` of a complex argument.
- :func:`pipe_velocity` — fully developed pulsatile flow in a circular
  tube of radius ``R``, the original Womersley (1955) solution. Closed
  form uses ``J₀`` of a complex argument. Used for the cylindrical
  pipe IBM validation in M2.

References
----------
- Womersley (1955) "Method for the calculation of velocity, rate of
  flow and viscous drag in arteries when the pressure gradient is
  known," J. Physiol. 127:553-563.
- White (2006) Viscous Fluid Flow, 3rd ed., §3-4 (channel form).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def channel_velocity(
    y: np.ndarray,
    t: float,
    *,
    h: float,
    nu: float,
    omega: float,
    f_steady: float = 0.0,
    f_osc: float = 0.0,
) -> np.ndarray:
    """Analytical fully developed pulsatile channel velocity ``u(y, t)``.

    Channel between ``y = -h`` and ``y = +h``. Flow is driven by a
    spatially uniform body force per unit mass

        f_x(t) = f_steady + f_osc * cos(ωt).

    Returns the streamwise velocity. Shape follows ``y`` (broadcastable).
    """
    y = np.asarray(y, dtype=np.float64)

    # Steady Poiseuille component
    u_steady = (f_steady / (2.0 * nu)) * (h ** 2 - y ** 2)

    # Oscillatory (Womersley) component
    if f_osc == 0.0 or omega == 0.0:
        return u_steady

    # Complex amplitude: U(y) = −i (f_osc / ω) H(y), where
    # H(y) = 1 − cosh(α y/h) / cosh(α) and α = Wo·exp(iπ/4).
    # u_osc(y, t) = Re{ U(y) · exp(iωt) }. Verified to <1% RMS against
    # a 401-point Crank-Nicolson PDE solve at Wo=7 over 8 cycles.
    Wo = h * np.sqrt(omega / nu)
    alpha = Wo * np.exp(1j * np.pi / 4)
    H = 1.0 - np.cosh(alpha * y / h) / np.cosh(alpha)
    U = -1j * (f_osc / omega) * H
    u_osc = np.real(U * np.exp(1j * omega * t))
    return u_steady + u_osc


def channel_centerline_amplitude_phase(
    *,
    h: float,
    nu: float,
    omega: float,
    f_osc: float,
    y: float = 0.0,
) -> tuple[float, float]:
    """Return (amplitude, phase_lag) of the oscillatory velocity at ``y``.

    The oscillatory component at any ``y`` can be written as
    ``A(y) cos(ωt − φ(y))``. ``phase_lag = φ(y)`` (radians, positive
    means the local velocity *lags* the driving force).
    """
    Wo = h * np.sqrt(omega / nu)
    alpha = Wo * np.exp(1j * np.pi / 4)
    H = 1.0 - np.cosh(alpha * y / h) / np.cosh(alpha)
    U = -1j * (f_osc / omega) * H
    amp = float(np.abs(U))
    phase = float(-np.angle(U))   # phase lag (positive = lags driving)
    return amp, phase


def pipe_velocity(
    r: np.ndarray,
    t: float,
    *,
    R: float,
    nu: float,
    omega: float,
    f_steady: float = 0.0,
    f_osc: float = 0.0,
) -> np.ndarray:
    """Analytical fully developed pulsatile pipe velocity ``u_z(r, t)``.

    Cylindrical tube of radius ``R``. Body-force-driven (force per unit
    mass). Steady part is Poiseuille; oscillatory part uses the
    Womersley solution involving ``J_0`` of a complex argument.
    """
    from scipy.special import jv  # lazy: only needed for pipe variant

    r = np.asarray(r, dtype=np.float64)

    u_steady = (f_steady / (4.0 * nu)) * (R ** 2 - r ** 2)
    if f_osc == 0.0 or omega == 0.0:
        return u_steady

    Wo = R * np.sqrt(omega / nu)
    alpha = Wo * np.exp(3j * np.pi / 4)            # = i^{3/2} Wo
    j0_alpha = jv(0, alpha)
    j0_alphar = jv(0, alpha * r / R)
    H = 1.0 - j0_alphar / j0_alpha
    U = -1j * (f_osc / omega) * H
    return u_steady + np.real(U * np.exp(1j * omega * t))
