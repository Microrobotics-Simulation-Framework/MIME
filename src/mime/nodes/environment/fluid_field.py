"""FluidFieldProvider protocol — solver-agnostic flow field access.

Fluid nodes (LBM, BEM, spectral NS) implement this protocol to
expose velocity data to the visualization and analysis pipeline
without callers needing to know solver internals.

The protocol is intentionally narrow: get_midplane_velocity returns
data at the solver's natural midplane. When BEM arrives with
arbitrary-plane capability, the protocol will be extended with
get_velocity_at_plane(origin, normal, resolution).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class FluidFieldProvider(Protocol):
    """Protocol for fluid nodes that can provide velocity field data."""

    def get_midplane_velocity(
        self,
        resolution: tuple[int, int],
    ) -> np.ndarray | None:
        """Return (nx, ny) velocity magnitude at the solver's midplane.

        The midplane definition is solver-specific (Z-midplane for LBM,
        solver-dependent for others). Returns None if data is unavailable.

        Parameters
        ----------
        resolution : (nx, ny)
            Requested output resolution. Implementations downsample
            if the native resolution is higher.

        Returns
        -------
        np.ndarray or None
            (nx, ny) float32 velocity magnitude, or None.
        """
        ...
