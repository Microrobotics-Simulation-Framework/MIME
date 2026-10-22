"""SourceInputProvider — where a SourcedEffectModel's drive comes from.

ADR-2026-EFFECT-MODEL §4 item 6 / MAGNETIC_NODE_AUDIT.md Item 1. A
:class:`~mime.effects.protocol.SourcedEffectModel` (magnetic, acoustic,
electric) is driven by external *sources*. Each per-source input — a magnet's
world pose, a coil current, a held field strength — is provided one of three
ways, unified behind :class:`SourceInputProvider`:

* :class:`ConstantInput` — a fixed value (materialised as a
  ``ConstantInputNode`` wired into the consumer).
* :class:`NodeFieldRef` — driven by another graph node's output field (e.g. a
  ``MotorNode``'s ``rotor_pose_world``); materialised as an edge.
* :class:`ExternalInputRef` — fed at ``step()`` time by the caller (the RPM
  command, a closed-loop controller); materialised as a graph-external input.

Each provider knows how to ``resolve_into`` a ``GraphManager`` for one
target node + field, so an EffectModel's ``build()`` resolves all of a
source's inputs uniformly without caring which kind each is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from mime.effects.protocol import Source

if TYPE_CHECKING:  # pragma: no cover
    from maddening.core.graph_manager import GraphManager


class SourceInputProvider(ABC):
    """How a single source input is provided (constant / node / external)."""

    @abstractmethod
    def resolve_into(
        self,
        gm: "GraphManager",
        target_node: str,
        target_field: str,
        *,
        timestep: float = 1.0,
    ) -> list[str]:
        """Materialise this input onto ``gm`` driving ``target_node``'s
        ``target_field``. ``timestep`` is used only when the provider must add
        a node (the constant case). Returns the names of any *new* nodes added
        (empty for edge / external-input providers), so a caller can keep an
        accurate record of the subgraph it built."""


@dataclass(frozen=True)
class ConstantInput(SourceInputProvider):
    """A fixed value, materialised as a ``ConstantInputNode``.

    Parameters
    ----------
    value : array-like
        The constant to hold.
    name : str | None
        Optional explicit name for the constant node; defaults to a name
        derived from the target (``"<node>__<field>__const"``).
    """

    value: Any
    name: Optional[str] = None

    def resolve_into(self, gm, target_node, target_field, *, timestep=1.0):
        from mime.nodes.actuation.constant_input import ConstantInputNode

        node_name = self.name or f"{target_node}__{target_field}__const"
        const = ConstantInputNode(node_name, timestep, value=self.value)
        gm.add_node(const)
        gm.add_edge(node_name, target_node, "value", target_field)
        return [node_name]


@dataclass(frozen=True)
class NodeFieldRef(SourceInputProvider):
    """Driven by another graph node's output field, materialised as an edge.

    The referenced ``node`` must already exist in the graph (the EffectModel
    that owns the source, or the experiment, is responsible for adding it).
    """

    node: str
    field: str

    def resolve_into(self, gm, target_node, target_field, *, timestep=1.0):
        gm.add_edge(self.node, target_node, self.field, target_field)
        return []


@dataclass(frozen=True)
class ExternalInputRef(SourceInputProvider):
    """Fed at ``step()`` time by the caller, materialised as a graph-external
    input (the RPM drive command, a closed-loop controller output)."""

    shape: tuple = ()
    dtype: Any = None

    def resolve_into(self, gm, target_node, target_field, *, timestep=1.0):
        if self.dtype is None:
            gm.add_external_input(target_node, target_field, shape=self.shape)
        else:
            gm.add_external_input(target_node, target_field,
                                  shape=self.shape, dtype=self.dtype)
        return []


@dataclass(frozen=True)
class MagneticSource(Source):
    """An actuated magnetic dipole / field source for the MagneticModel family.

    ``inputs`` maps a consuming node's input-field name to the provider that
    drives it (e.g. ``{"magnet_pose_world": NodeFieldRef("motor",
    "rotor_pose_world")}``). The EffectModel resolves every entry in ``build()``.
    """

    inputs: dict[str, SourceInputProvider] = field(default_factory=dict)

    def resolve_all(self, gm, target_node: str, *, timestep: float = 1.0) -> list[str]:
        """Resolve every declared input into ``target_node``. Returns the names
        of any new nodes added (e.g. constant-input nodes)."""
        added: list[str] = []
        for target_field, provider in self.inputs.items():
            added.extend(provider.resolve_into(
                gm, target_node, target_field, timestep=timestep))
        return added
