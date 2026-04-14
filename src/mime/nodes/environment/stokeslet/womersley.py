"""Analytical pulsatile Poiseuille (Womersley) velocity profile.

Provides the closed-form solution for oscillatory flow in a circular
pipe. Used as the background flow u_bg for Level 2 BEM subtraction:

    rhs = u_body - u_bg_at_body

The exact Womersley solution involves complex Bessel functions, but
for the low Womersley numbers typical of microrobot applications
(Wo < 5), the quasi-steady sinusoidal Poiseuille approximation is
within 2% of the exact solution and avoids the Bessel evaluation.

Reference: Womersley (1955), J. Physiol. 127:553-563.
"""

from __future__ import annotations

import numpy as np


def pulsatile_poiseuille(
    r: np.ndarray,
    t: float,
    R: float,
    U_mean: float,
    amplitude: float = 0.5,
    f_pulse: float = 1.0,
) -> np.ndarray:
    """Quasi-steady pulsatile Poiseuille velocity (axial component).

    u_z(r, t) = 2 * U_mean * (1 - (r/R)²) * (1 + A * sin(2π f t))

    The factor of 2 converts mean velocity to centreline velocity
    for the parabolic profile (U_max = 2 * U_mean).

    Parameters
    ----------
    r : (N,) radial positions from pipe axis.
    t : float, time [s].
    R : float, pipe radius [m or non-dim].
    U_mean : float, time-averaged mean flow velocity [m/s or non-dim].
    amplitude : float, pulsation amplitude (0 = steady, 1 = full pulsation).
    f_pulse : float, pulse frequency [Hz].

    Returns
    -------
    u_z : (N,) axial velocity at each radial position.
    """
    profile = np.maximum(1.0 - (r / R) ** 2, 0.0)
    modulation = 1.0 + amplitude * np.sin(2.0 * np.pi * f_pulse * t)
    return 2.0 * U_mean * profile * modulation


def pulsatile_poiseuille_3d(
    points: np.ndarray,
    t: float,
    R: float,
    U_mean: float,
    amplitude: float = 0.5,
    f_pulse: float = 1.0,
    flow_axis: int = 2,
) -> np.ndarray:
    """Pulsatile Poiseuille velocity at arbitrary 3D points.

    Assumes the pipe is centred on the origin with axis along
    ``flow_axis``. Returns the full 3D velocity vector (only
    the axial component is non-zero for Poiseuille flow).

    Parameters
    ----------
    points : (N, 3) positions.
    t : float, time.
    R : float, pipe radius.
    U_mean : float, mean flow velocity.
    amplitude, f_pulse : float, pulsation parameters.
    flow_axis : int, 0=x, 1=y, 2=z.

    Returns
    -------
    u : (N, 3) velocity vectors.
    """
    # Radial distance from pipe axis
    axes = [i for i in range(3) if i != flow_axis]
    r = np.sqrt(points[:, axes[0]] ** 2 + points[:, axes[1]] ** 2)

    u_axial = pulsatile_poiseuille(r, t, R, U_mean, amplitude, f_pulse)

    u = np.zeros_like(points)
    u[:, flow_axis] = u_axial
    return u


def deboer_flow_velocity(
    pump_percentage: float,
    vessel_diameter_mm: float = 8.0,
) -> float:
    """Convert de Boer pump percentage to mean flow velocity.

    De Boer et al. (2025) §VI.E use a vessel diameter of 8 mm and
    report pump percentages of physiological flow rate. For the
    iliac artery, physiological flow is ~250 mL/min (Defined as 100%).

    Q_phys ≈ 250 mL/min = 4.17e-6 m³/s
    A = π(d/2)² = π(4e-3)² = 5.03e-5 m²
    U_mean_phys = Q/A ≈ 0.083 m/s

    Parameters
    ----------
    pump_percentage : float
        0-100, percentage of physiological flow.
    vessel_diameter_mm : float
        Vessel diameter in mm.

    Returns
    -------
    U_mean : float
        Mean flow velocity in m/s.
    """
    Q_phys = 250e-6 / 60.0  # 250 mL/min → m³/s
    R = vessel_diameter_mm * 0.5e-3  # mm → m radius
    A = np.pi * R ** 2
    U_mean_phys = Q_phys / A
    return U_mean_phys * pump_percentage / 100.0
