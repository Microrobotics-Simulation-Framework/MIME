"""Controller for UMR confinement experiment.

Provides external inputs (field frequency and strength) to the graph
each step. For the confinement sweep, these are constant. For the
step-out demo, frequency ramps linearly.
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
    ratio = params["CONFINEMENT_RATIO"]
    cs = 1.0 / math.sqrt(3)
    dx_mm = params["VESSEL_DIAMETER_MM"] / N
    dx_physical = params["VESSEL_DIAMETER_MM"] * 1e-3 / N
    nu_lattice = (tau - 0.5) / 3.0
    nu_physical = params["FLUID_VISCOSITY"] / params["FLUID_DENSITY"]
    dt_physical = nu_lattice * dx_physical ** 2 / nu_physical

    geom_lu = {k: v / dx_mm for k, v in params["UMR_GEOM_MM"].items()}
    R_fin_lu = geom_lu["fin_outer_radius"]
    omega_lu = 0.05 * cs / R_fin_lu
    omega_physical = omega_lu / dt_physical
    f_field_hz = omega_physical / (2.0 * math.pi)

    return {
        "ext_field": {
            "frequency_hz": jnp.float32(f_field_hz),
            "field_strength_mt": jnp.float32(params["B_FIELD"] * 1e3),
        },
    }
