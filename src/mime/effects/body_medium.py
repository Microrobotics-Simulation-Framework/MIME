"""Body and Medium descriptors for the EffectModel contract.

ADR-2026-EFFECT-MODEL decision log #2/#3: Body and Medium are *fat
dataclasses* (not Protocols) at v1.0 scale — simpler, with a written
refactor tripwire (if any family needs >3 mutually-exclusive body sections,
move to per-physics Protocols).

``Body`` names the rigid-body node an effect couples force/torque into, and
carries per-physics property *sections* (e.g. ``"hydrodynamic"``). ``Medium``
carries the surrounding-fluid properties. ``build()`` reads what it declared
as required in :meth:`EffectModel.required_body_properties` /
``required_medium_properties`` (validated in pass 3 before any subgraph
is materialised).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.node import SimulationNode


@dataclass
class Body:
    """The rigid body environment effects act on.

    Parameters
    ----------
    name : str
        The body's node name in the graph (effects wire ``drag_force`` /
        ``drag_torque`` / ``magnetic_*`` edges into it).
    node : SimulationNode | None
        The body node itself. ``Experiment.build`` adds it to the graph
        once; effects only reference it by ``name``.
    properties : dict[str, dict]
        Per-physics property sections, e.g.
        ``{"hydrodynamic": {"semi_major_axis_m": ...}}``.
    """

    name: str
    node: "SimulationNode | None" = None
    properties: dict[str, dict] = field(default_factory=dict)

    def has_section(self, section: str) -> bool:
        return section in self.properties


@dataclass
class Medium:
    """The surrounding medium (fluid) environment effects propagate through.

    Parameters
    ----------
    properties : dict[str, Any]
        e.g. ``{"density": 1000.0, "viscosity": 1e-3,
        "reynolds_number": 1e-3}``.
    """

    properties: dict[str, Any] = field(default_factory=dict)

    def has(self, key: str) -> bool:
        return key in self.properties
