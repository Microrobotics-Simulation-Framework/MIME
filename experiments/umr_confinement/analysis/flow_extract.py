"""Hook: flow_extractor — velocity magnitude cross-section.

Uses FluidFieldProvider protocol when available, falls back to
direct LBM extraction during the transition period.
"""

from __future__ import annotations

import numpy as np


def extract_flow(ctx):
    """Return (nx, ny) velocity magnitude at the solver's midplane.

    Parameters
    ----------
    ctx : HookContext

    Returns
    -------
    np.ndarray or None
    """
    # Prefer protocol-based access (solver-agnostic)
    provider = ctx.state.get("_fluid_field_provider")
    if provider is not None:
        return provider.get_midplane_velocity(resolution=(64, 64))

    # REMOVE: once IBLBMFluidNode implements FluidFieldProvider (Step 9)
    lbm = ctx.state.get("lbm_fluid")
    if lbm is None or "f" not in lbm:
        return None
    from mime.nodes.environment.lbm.d3q19 import compute_macroscopic
    _, velocity = compute_macroscopic(lbm["f"])
    vel_np = np.asarray(velocity)
    nz = vel_np.shape[2]
    return np.linalg.norm(vel_np[:, :, nz // 2, :], axis=-1)
