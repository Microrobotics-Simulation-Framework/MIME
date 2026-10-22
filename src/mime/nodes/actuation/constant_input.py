"""ConstantInputNode — emit a fixed array on a graph edge.

A pure graph-plumbing node (no physics, no anatomy), so — like
``LBMFarFieldNode`` / ``StokesletFluidNode`` / ``DefectCorrectionFluidNode``
— it extends MADDENING's ``SimulationNode`` directly rather than ``MimeNode``;
there is no domain metadata to carry.

It backs the :class:`~mime.effects.sources.ConstantInput` source provider:
when a source's drive (e.g. a fixed magnet pose, a held field strength) is a
constant, the provider materialises one of these and wires its ``value``
output into the consuming node's input.
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.node import BoundaryFluxSpec, SimulationNode


class ConstantInputNode(SimulationNode):
    """Emit a fixed value on the ``value`` output every step.

    Parameters
    ----------
    name : str
    timestep : float
        Unused by the constant (kept for the SimulationNode contract).
    value : array-like
        The constant to emit; its shape defines the output shape.
    """

    def __init__(self, name: str, timestep: float, value, **kwargs):
        super().__init__(name, timestep, **kwargs)
        self._value = jnp.asarray(value)
        self._shape = tuple(self._value.shape)

    def initial_state(self) -> dict:
        return {"value": self._value}

    def boundary_input_spec(self) -> dict:
        return {}

    def boundary_flux_spec(self) -> dict[str, BoundaryFluxSpec]:
        return {
            "value": BoundaryFluxSpec(
                shape=self._shape,
                description="Constant value",
                output_units="SI",
            ),
        }

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return {"value": self._value}

    def compute_boundary_fluxes(
        self, state: dict, boundary_inputs: dict, dt: float,
    ) -> dict:
        return {"value": state["value"]}
