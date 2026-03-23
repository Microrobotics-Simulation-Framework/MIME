"""UMR ODE model — 3-state reduced-order model for untethered magnetic robot.

Pure JAX module (NOT a MimeNode). Implements forward Euler integration of:

    State: [theta, Omega, U]
    dtheta/dt = omega_field - Omega
    I_eff * dOmega/dt = n_mag * m * B * sin(theta) - C_rot * Omega
    m_eff * dU/dt     = C_prop * Omega - C_trans * U

Where:
    theta   = phase lag between field and body rotation [rad]
    Omega   = body angular velocity [rad/s]
    U       = translational speed [m/s]

The module also provides an *averaged* speed-curve model that uses time-scale
separation: when the rotational settling time tau_rot = I_eff/C_rot is much
shorter than the translational settling time tau_trans = m_eff/C_trans, the
rotation can be treated as quasi-steady. The mean angular velocity follows
the Adler equation result:

    Below step-out:  <Omega> = omega_field
    Above step-out:  <Omega> = omega_field - sqrt(omega_field^2 - omega_so^2)

where omega_so = n*m*B / C_rot is the step-out angular frequency. The
translational dynamics then reduce to a first-order ODE:

    m_eff * dU/dt = C_prop * <Omega> - C_trans * U

Reference: de Boer et al. (2025), Wireless mechanical thrombus fragmentation.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Parameter packing / unpacking
# ---------------------------------------------------------------------------

_PARAM_KEYS = (
    "omega_field", "n_mag", "m_single", "B",
    "I_eff", "m_eff", "C_rot", "C_prop", "C_trans",
)


def pack_params(
    omega_field: float,
    n_mag: float,
    m_single: float,
    B: float,
    I_eff: float,
    m_eff: float,
    C_rot: float,
    C_prop: float,
    C_trans: float,
) -> jnp.ndarray:
    """Pack parameters into a flat JAX array (length 9) for JIT safety."""
    return jnp.array([
        omega_field, n_mag, m_single, B,
        I_eff, m_eff, C_rot, C_prop, C_trans,
    ])


def unpack_params(p: jnp.ndarray) -> dict[str, Any]:
    """Unpack flat parameter array to a dict."""
    return {k: p[i] for i, k in enumerate(_PARAM_KEYS)}


def params_dict_to_array(d: dict) -> jnp.ndarray:
    """Convert a params dict to a flat array."""
    return jnp.array([float(d[k]) for k in _PARAM_KEYS])


def params_array_to_dict(p: jnp.ndarray) -> dict[str, float]:
    """Convert flat array back to dict (with Python floats)."""
    return {k: float(p[i]) for i, k in enumerate(_PARAM_KEYS)}


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def umr_ode_rhs(state: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Compute d/dt [theta, Omega, U] from state and packed params.

    Parameters
    ----------
    state : jnp.ndarray, shape (3,)
        [theta, Omega, U]
    params : jnp.ndarray, shape (9,)
        Packed parameter array from :func:`pack_params`.

    Returns
    -------
    jnp.ndarray, shape (3,)
        Time derivatives [dtheta/dt, dOmega/dt, dU/dt].
    """
    theta, Omega, U = state[0], state[1], state[2]
    p = unpack_params(params)

    omega_field = p["omega_field"]
    n_mag = p["n_mag"]
    m_single = p["m_single"]
    B = p["B"]
    I_eff = p["I_eff"]
    m_eff = p["m_eff"]
    C_rot = p["C_rot"]
    C_prop = p["C_prop"]
    C_trans = p["C_trans"]

    dtheta_dt = omega_field - Omega
    dOmega_dt = (n_mag * m_single * B * jnp.sin(theta) - C_rot * Omega) / I_eff
    dU_dt = (C_prop * Omega - C_trans * U) / m_eff

    return jnp.array([dtheta_dt, dOmega_dt, dU_dt])


# ---------------------------------------------------------------------------
# Euler integration via jax.lax.scan
# ---------------------------------------------------------------------------

def umr_euler_integrate(
    params: jnp.ndarray, dt: float, n_steps: int,
) -> jnp.ndarray:
    """Forward Euler integration of UMR ODE.

    Parameters
    ----------
    params : jnp.ndarray, shape (9,)
        Packed parameters.
    dt : float
        Timestep in seconds.
    n_steps : int
        Number of integration steps.

    Returns
    -------
    jnp.ndarray, shape (n_steps + 1, 3)
        State history including initial condition [0, 0, 0].
    """
    state0 = jnp.zeros(3)

    def step(state, _):
        d = umr_ode_rhs(state, params)
        new_state = state + dt * d
        return new_state, new_state

    _, history = jax.lax.scan(step, state0, None, length=n_steps)
    # Prepend initial state
    return jnp.concatenate([state0[None, :], history], axis=0)


# ---------------------------------------------------------------------------
# Speed curve utility
# ---------------------------------------------------------------------------

def umr_speed_curve(
    params: jnp.ndarray, dt: float, t_final: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrate UMR ODE and return (t_array, U_array).

    Parameters
    ----------
    params : jnp.ndarray, shape (9,)
        Packed parameters.
    dt : float
        Timestep in seconds.
    t_final : float
        Final time in seconds.

    Returns
    -------
    t_array : jnp.ndarray, shape (n_steps + 1,)
    U_array : jnp.ndarray, shape (n_steps + 1,)
    """
    n_steps = int(t_final / dt)
    history = umr_euler_integrate(params, dt, n_steps)
    t_array = jnp.arange(n_steps + 1) * dt
    U_array = history[:, 2]  # translational speed
    return t_array, U_array


# ---------------------------------------------------------------------------
# Averaged (time-scale separated) speed curve
# ---------------------------------------------------------------------------

def _mean_angular_velocity(omega_field: jnp.ndarray, omega_so: jnp.ndarray) -> jnp.ndarray:
    """Mean angular velocity from Adler equation (overdamped rotational limit).

    Below step-out (omega_field <= omega_so):  <Omega> = omega_field
    Above step-out (omega_field > omega_so):   <Omega> = omega_field - sqrt(omega_field^2 - omega_so^2)

    Parameters
    ----------
    omega_field : angular frequency of the rotating field [rad/s]
    omega_so : step-out angular frequency = n*m*B / C_rot [rad/s]
    """
    # Smoothly handle both regimes
    ratio = omega_field / jnp.maximum(omega_so, 1e-30)
    # Below step-out: return omega_field
    # Above step-out: return omega_field - sqrt(omega_field^2 - omega_so^2)
    omega_above = omega_field - jnp.sqrt(jnp.maximum(omega_field**2 - omega_so**2, 0.0))
    return jnp.where(ratio <= 1.0, omega_field, omega_above)


def umr_averaged_speed_curve(
    params: jnp.ndarray, dt: float, t_final: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Speed curve using time-scale-separated model (averaged rotation).

    Assumes tau_rot << tau_trans so rotation reaches quasi-steady state
    nearly instantaneously. Integrates only the translational ODE:

        m_eff * dU/dt = C_prop * <Omega> - C_trans * U

    where <Omega> is the mean angular velocity from the Adler equation.

    Parameters
    ----------
    params : jnp.ndarray, shape (9,)
        Packed parameters.
    dt : float
        Timestep in seconds.
    t_final : float
        Final time in seconds.

    Returns
    -------
    t_array : jnp.ndarray, shape (n_steps + 1,)
    U_array : jnp.ndarray, shape (n_steps + 1,)
    """
    p = unpack_params(params)
    omega_field = p["omega_field"]
    n_mag = p["n_mag"]
    m_single = p["m_single"]
    B = p["B"]
    m_eff = p["m_eff"]
    C_rot = p["C_rot"]
    C_prop = p["C_prop"]
    C_trans = p["C_trans"]

    # Step-out angular frequency
    omega_so = n_mag * m_single * B / C_rot
    Omega_avg = _mean_angular_velocity(omega_field, omega_so)

    n_steps = int(t_final / dt)

    def step(U, _):
        dU_dt = (C_prop * Omega_avg - C_trans * U) / m_eff
        U_new = U + dt * dU_dt
        return U_new, U_new

    U0 = jnp.array(0.0)
    _, U_history = jax.lax.scan(step, U0, None, length=n_steps)
    U_all = jnp.concatenate([jnp.array([0.0]), U_history])
    t_array = jnp.arange(n_steps + 1) * dt
    return t_array, U_all


# ---------------------------------------------------------------------------
# Step-out frequency
# ---------------------------------------------------------------------------

def compute_step_out_frequency(
    n_mag: float, m: float, B: float, C_rot: float,
) -> float:
    """Analytical step-out frequency: f_step = n*m*B / (2*pi*C_rot).

    Parameters
    ----------
    n_mag : float
        Number of magnets.
    m : float
        Magnetic moment per magnet [A*m^2].
    B : float
        Field strength [T].
    C_rot : float
        Rotational drag coefficient [N.m.s].

    Returns
    -------
    float
        Step-out frequency [Hz].
    """
    return n_mag * m * B / (2.0 * math.pi * C_rot)


# ---------------------------------------------------------------------------
# Frequency sweep
# ---------------------------------------------------------------------------

def sweep_frequency(
    params_template: jnp.ndarray,
    freqs: jnp.ndarray,
    dt: float,
    t_settle: float,
) -> jnp.ndarray:
    """Sweep over field frequencies and return steady-state speed at each.

    Parameters
    ----------
    params_template : jnp.ndarray, shape (9,)
        Template parameters (omega_field will be overwritten).
    freqs : jnp.ndarray, shape (N,)
        Frequencies to sweep [Hz].
    dt : float
        Timestep for integration.
    t_settle : float
        Settling time (seconds) — integrates this long, takes final U.

    Returns
    -------
    jnp.ndarray, shape (N,)
        Steady-state translational speed at each frequency.
    """
    n_steps = int(t_settle / dt)

    def run_one(freq):
        omega = 2.0 * jnp.pi * freq
        p = params_template.at[0].set(omega)
        history = umr_euler_integrate(p, dt, n_steps)
        return history[-1, 2]  # final U

    return jax.vmap(run_one)(freqs)


# ---------------------------------------------------------------------------
# UMR geometry and inertia computation
# ---------------------------------------------------------------------------

def _compute_geometry_params(n_mag: int = 1) -> dict:
    """Compute I_eff and m_eff from UMR geometry and added-mass.

    UMR geometry (de Boer 2025):
        - Cylinder body: diameter 1.74 mm, length 4.1 mm
        - Cone tip: length 1.9 mm, end diameter 0.51 mm
        - Total length: 6.0 mm
        - Magnets: n_mag x 1 mm^3 NdFeB

    Returns dict with I_eff, m_eff, and intermediate values.
    """
    # Material densities
    rho_resin = 1150.0      # kg/m^3  (SLA resin)
    rho_NdFeB = 7500.0      # kg/m^3
    rho_water = 1000.0      # kg/m^3

    # Geometry (SI units)
    D_cyl = 1.74e-3         # cylinder diameter [m]
    R_cyl = D_cyl / 2.0     # cylinder radius [m]
    L_cyl = 4.1e-3          # cylinder length [m]
    L_cone = 1.9e-3         # cone length [m]
    D_cone_end = 0.51e-3    # cone end diameter [m]
    R_cone_end = D_cone_end / 2.0

    # Volumes
    V_cyl = math.pi * R_cyl**2 * L_cyl
    # Truncated cone volume: (pi/3)*h*(R1^2 + R1*R2 + R2^2)
    V_cone = (math.pi / 3.0) * L_cone * (
        R_cyl**2 + R_cyl * R_cone_end + R_cone_end**2
    )
    V_body = V_cyl + V_cone

    # Magnet volume
    V_mag_single = 1e-9     # 1 mm^3 = 1e-9 m^3
    V_mag = n_mag * V_mag_single

    # Masses
    m_body = rho_resin * V_body
    m_mag = rho_NdFeB * V_mag

    # Added mass (cylinder approximation): m_added = rho_water * pi * R^2 * L
    L_total = L_cyl + L_cone
    m_added = rho_water * math.pi * R_cyl**2 * L_total

    # Effective mass
    m_eff = m_body + m_mag + m_added

    # Effective moment of inertia (about long axis)
    # Body + magnets: 0.5 * (m_body + m_mag) * R^2
    # Added inertia (cylinder): 0.5 * rho_water * pi * R^4 * L
    I_body = 0.5 * (m_body + m_mag) * R_cyl**2
    I_added = 0.5 * rho_water * math.pi * R_cyl**4 * L_total
    I_eff = I_body + I_added

    return {
        "I_eff": I_eff,
        "m_eff": m_eff,
        "m_body": m_body,
        "m_mag": m_mag,
        "m_added": m_added,
        "V_body": V_body,
        "R_cyl": R_cyl,
        "L_total": L_total,
    }


# ---------------------------------------------------------------------------
# Drag coefficient fitting
# ---------------------------------------------------------------------------

def fit_drag_coefficients(
    digitised_data: np.ndarray,
    n_mag: int = 1,
    m_single: float = 1.07e-3,
    B: float = 3e-3,
    f_step: float = 128.0,
) -> dict:
    """Fit drag coefficients from digitised speed-vs-time data.

    Strategy:
        1. I_eff, m_eff from geometry + added mass (computed, not fitted)
        2. C_rot from step-out constraint: C_rot = n*m*B / (2*pi*f_step)
        3. C_prop/C_trans from steady-state speed ratio: U_ss / omega_field
        4. C_trans from translational time constant fitted to data transient:
           tau_trans = m_eff / C_trans, so C_trans = m_eff / tau_trans.
           C_prop = ratio * C_trans.

    The translational time constant tau_trans is estimated by fitting
    U(t) = U_ss * (1 - exp(-t/tau)) to the digitised data.

    Parameters
    ----------
    digitised_data : np.ndarray, shape (N, 2)
        Columns [t_s, speed_m_per_s].
    n_mag : int
        Number of magnets.
    m_single : float
        Moment per magnet [A*m^2].
    B : float
        Field strength [T].
    f_step : float
        Step-out frequency [Hz].

    Returns
    -------
    dict
        Parameter dictionary with all 9 ODE params.
    """
    from scipy.optimize import curve_fit

    geom = _compute_geometry_params(n_mag)

    # Operating at step-out frequency
    omega_field = 2.0 * math.pi * f_step

    # C_rot from step-out constraint
    C_rot = n_mag * m_single * B / (2.0 * math.pi * f_step)

    # Steady-state speed from data (last data point)
    t_data = digitised_data[:, 0]
    U_data = digitised_data[:, 1]
    U_ss = float(U_data[-1])

    # Fit exponential transient to extract tau_trans
    def _exp_model(t, tau):
        return U_ss * (1.0 - np.exp(-t / tau))

    popt, _ = curve_fit(_exp_model, t_data, U_data, p0=[0.1])
    tau_trans = float(popt[0])

    # C_trans from translational time constant: tau_trans = m_eff / C_trans
    C_trans = geom["m_eff"] / tau_trans

    # Ratio: at steady state, U_ss = (C_prop / C_trans) * omega_field
    ratio = U_ss / omega_field

    # C_prop from ratio
    C_prop = ratio * C_trans

    return {
        "omega_field": omega_field,
        "n_mag": float(n_mag),
        "m_single": m_single,
        "B": B,
        "I_eff": geom["I_eff"],
        "m_eff": geom["m_eff"],
        "C_rot": C_rot,
        "C_prop": C_prop,
        "C_trans": C_trans,
    }
