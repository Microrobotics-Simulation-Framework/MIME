"""HydrodynamicModel family — the v0.2 EffectModel pilot.

ADR-2026-EFFECT-MODEL §7 item 1: v1.0 ships ``HydrodynamicModel`` with four
swappable backends (LBM / FVM / Stokeslet / DefectCorrection). This module
is the *pilot* — it proves the Protocol surface against the one family that
already shares a contract (``FLUID_NODE_CONTRACT.md``): each backend wraps an
existing fluid node and materialises it plus its body-coupling edges onto a
GraphManager, so one backend can be swapped for another across the same
graph edges.

The fluid nodes emit ``drag_force`` / ``drag_torque`` on the contract names;
the wrapper wires them into the rigid body. The LBM backend carries the
``lbm_to_si_*`` edge transforms (its outputs are in lattice units); the BEM
backends are already SI. No node is rewritten — this is an adapter layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from mime.effects.protocol import (
    BaseEffectModel,
    EffectHandle,
    HydrodynamicRegime,
)
from mime.effects.registry import register_effect

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import SimulationNode

    from mime.effects.body_medium import Body, Medium


class _HydrodynamicEffect(BaseEffectModel):
    """Common adapter: wrap a fluid node + an edge-builder into an EffectModel.

    Parameters
    ----------
    node : SimulationNode
        A pre-constructed fluid node (IBLBM / FVM / Stokeslet /
        DefectCorrection). Backend-specific construction params belong on the
        node itself (ADR decision #6: parameters live in __init__).
    edge_builder : callable | None
        ``(fluid_name, body_name) -> list[EdgeSpec]`` for backends with a
        bespoke wiring helper (e.g. the LBM unit transforms). When None, the
        generic SI ``drag_force`` / ``drag_torque`` additive edges are used —
        valid for the SI backends (FVM / Stokeslet / DefectCorrection).
    """

    def __init__(
        self,
        node: "SimulationNode",
        *,
        edge_builder: Optional[Callable[[str, str], list]] = None,
        re_range: tuple[float, float] = (0.0, 1.0),
    ):
        self._node = node
        self._edge_builder = edge_builder
        self._re_range = re_range

    def applicable_regime(self) -> HydrodynamicRegime:
        return HydrodynamicRegime(self._re_range)

    def required_medium_properties(self) -> set[str]:
        return {"density", "viscosity"}

    def build(self, gm: "GraphManager", *, body: "Body", medium: "Medium") -> EffectHandle:
        gm.add_node(self._node)
        fluid_name = self._node.name
        if self._edge_builder is not None:
            for e in self._edge_builder(fluid_name, body.name):
                gm.add_edge(
                    e.source_node, e.target_node,
                    e.source_field, e.target_field,
                    transform=getattr(e, "transform", None),
                    additive=getattr(e, "additive", False),
                )
        else:
            # Generic SI hydrodynamic load → body (the interchangeable subset
            # of FLUID_NODE_CONTRACT.md).
            gm.add_edge(fluid_name, body.name, "drag_force", "drag_force",
                        additive=True)
            gm.add_edge(fluid_name, body.name, "drag_torque", "drag_torque",
                        additive=True)
        return EffectHandle(node_names=(fluid_name,))


class HydrodynamicModel:
    """Namespace of swappable hydrodynamic backends (ADR §7 item 1)."""

    @register_effect("HydrodynamicModel.LBM")
    class LBM(_HydrodynamicEffect):
        """IB-LBM backend. Carries the lattice→SI drag edge transforms."""

        def __init__(
            self,
            node: "SimulationNode",
            *,
            dx_physical: float,
            dt_physical: float,
            fluid_density: float = 1060.0,
            re_range: tuple[float, float] = (0.0, 1.0),
        ):
            from mime.nodes.environment.lbm.fluid_node import (
                make_iblbm_rigid_body_edges,
            )

            def _edges(fluid_name: str, body_name: str) -> list:
                return make_iblbm_rigid_body_edges(
                    fluid_name, body_name, dx_physical, dt_physical,
                    fluid_density,
                )

            super().__init__(node, edge_builder=_edges, re_range=re_range)

    @register_effect("HydrodynamicModel.Stokeslet")
    class Stokeslet(_HydrodynamicEffect):
        """Regularised-Stokeslet BEM backend (SI; bespoke edge helper)."""

        def __init__(self, node: "SimulationNode", *, re_range=(0.0, 1.0)):
            from mime.nodes.environment.stokeslet.fluid_node import (
                make_stokeslet_rigid_body_edges,
            )

            def _edges(fluid_name: str, body_name: str) -> list:
                return make_stokeslet_rigid_body_edges(fluid_name, body_name)

            super().__init__(node, edge_builder=_edges, re_range=re_range)

    @register_effect("HydrodynamicModel.FVM")
    class FVM(_HydrodynamicEffect):
        """Finite-volume + IBM backend (SI; generic drag edges)."""

        def __init__(self, node: "SimulationNode", *, re_range=(0.0, 1.0)):
            super().__init__(node, edge_builder=None, re_range=re_range)

    @register_effect("HydrodynamicModel.DefectCorrection")
    class DefectCorrection(_HydrodynamicEffect):
        """BEM/LBM defect-correction backend (SI; generic drag edges)."""

        def __init__(self, node: "SimulationNode", *, re_range=(0.0, 1.0)):
            super().__init__(node, edge_builder=None, re_range=re_range)
