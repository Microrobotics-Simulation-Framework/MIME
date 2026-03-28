"""Controller for UMR confinement experiment.

Provides external inputs (field frequency and strength) to the graph
each step. Supports two modes:

- "steady": constant frequency at F_STEADY_FRAC * F_STEP_UNCONFINED
  (synchronized rotation with forward propulsion)
- "stepout": linear ramp from F_RAMP_START to F_RAMP_END (fractions
  of F_STEP_UNCONFINED) over RAMP_STEPS steps, then holds at F_RAMP_END.
  Shows transition from synchronous rotation to step-out.
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

    mode = params.get("MODE", "steady")
    f_step = params["F_STEP_UNCONFINED"]
    b_field = params["B_FIELD"]

    if mode == "stepout":
        ramp_steps = params.get("RAMP_STEPS", 20000)
        f_start_frac = params.get("F_RAMP_START", 0.5)
        f_end_frac = params.get("F_RAMP_END", 1.3)
        t_frac = min(step / max(ramp_steps, 1), 1.0)
        f_current = f_step * (f_start_frac + (f_end_frac - f_start_frac) * t_frac)
    else:
        # Steady mode: constant frequency below step-out
        f_steady_frac = params.get("F_STEADY_FRAC", 0.8)
        f_current = f_step * f_steady_frac

    return {
        "ext_field": {
            "frequency_hz": jnp.float32(f_current),
            "field_strength_mt": jnp.float32(b_field * 1e3),
        },
    }
