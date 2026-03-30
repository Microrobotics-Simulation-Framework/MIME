"""Controller for UMR confinement experiment.

Provides external inputs (field frequency and strength) to the graph
each step. Supports two modes:

- "steady": constant frequency at F_STEADY_FRAC * safe_max_freq
- "stepout": linear ramp from F_RAMP_START to F_RAMP_END (fractions
  of the Mach-safe maximum frequency) over RAMP_STEPS steps.

The maximum safe frequency is derived from the Mach number constraint:
  omega_lattice * R_fin_lu * sqrt(3) < Ma_target
  f_max_hz = omega_max_lattice / (2*pi*dt_physical)
"""


def get_external_inputs(params: dict, step: int) -> dict:
    """Return external inputs for the current step.

    Parameters
    ----------
    params : dict
        Experiment parameters from params.py.
    step : int
        Current simulation step number.

    Returns
    -------
    dict
        External inputs for GraphManager.step().
    """
    import jax.numpy as jnp
    import math

    N = params["RESOLUTION"]
    tau = params["TAU"]
    b_field = params["B_FIELD"]
    cs = 1.0 / math.sqrt(3)

    # Compute safe frequency from Mach constraint
    dx_mm = params["VESSEL_DIAMETER_MM"] / N
    dx_physical = params["VESSEL_DIAMETER_MM"] * 1e-3 / N
    nu_lattice = (tau - 0.5) / 3.0
    nu_physical = params["FLUID_VISCOSITY"] / params["FLUID_DENSITY"]
    dt_physical = nu_lattice * dx_physical ** 2 / nu_physical

    geom_lu = {k: v / dx_mm for k, v in params["UMR_GEOM_MM"].items()}
    R_fin_lu = geom_lu["fin_outer_radius"]

    # Ma_target = 0.05 (safe margin below 0.1 limit)
    Ma_target = 0.05
    omega_max_lu = Ma_target * cs / R_fin_lu
    f_max_hz = omega_max_lu / (2.0 * math.pi * dt_physical)

    mode = params.get("MODE", "steady")

    if mode == "stepout":
        ramp_steps = params.get("RAMP_STEPS", 20000)
        f_start_frac = params.get("F_RAMP_START", 0.3)
        f_end_frac = params.get("F_RAMP_END", 1.0)
        t_frac = min(step / max(ramp_steps, 1), 1.0)
        f_current = f_max_hz * (f_start_frac + (f_end_frac - f_start_frac) * t_frac)
    else:
        f_steady_frac = params.get("F_STEADY_FRAC", 0.8)
        f_current = f_max_hz * f_steady_frac

    return {
        "ext_field": {
            "frequency_hz": jnp.float32(f_current),
            "field_strength_mt": jnp.float32(b_field * 1e3),
        },
    }
