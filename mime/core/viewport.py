"""USDViewport protocol — swappable rendering backend.

Defines the minimal interface that decouples MIME simulation code from
the rendering backend. Three implementations serve three contexts:
- PyVistaViewport: local development
- HydraStormViewport: production headless (cloud streaming)
- MICROBOTICAViewport: MICROBOTICA desktop integration (stub)

See ARCHITECTURE_PLAN.md Section 15 for the full design.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class USDViewport(Protocol):
    """Minimal viewport protocol for rendering a live USD stage."""

    def render(self, stage: "Usd.Stage", camera: str = "/Camera") -> np.ndarray:
        """Render the current stage state.

        Parameters
        ----------
        stage : pxr.Usd.Stage
            Live USD stage — the same stage the simulation writes to.
        camera : str
            USD prim path of the camera to render from.

        Returns
        -------
        np.ndarray
            HxWx3 uint8 RGB image array.
        """
        ...

    def close(self) -> None:
        """Release GPU/window resources."""
        ...
