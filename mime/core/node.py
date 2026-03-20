"""MimeNode ABC — extends MADDENING's SimulationNode with domain concerns.

MimeNode is the base class for all MIME physics nodes. It adds:
- mime_meta ClassVar for domain-specific metadata
- observable_fields() / commandable_fields() for control integration
- validate_mime_consistency() for metadata consistency checks
- Default requires_halo = False (most MIME nodes are pointwise)
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Optional

from maddening.core.node import SimulationNode
from maddening.core.compliance.metadata import NodeMeta

from mime.core.metadata import MimeNodeMeta


class MimeNode(SimulationNode, ABC):
    """Abstract base class for all MIME physics nodes.

    Subclasses must set both ``meta`` (NodeMeta) and ``mime_meta``
    (MimeNodeMeta) as ClassVars. The ``meta`` field is consumed by
    MADDENING's harvester; ``mime_meta`` is consumed by MIME's own
    tooling, MimeAssetSchema, and the MICROBOTICA registry.
    """

    meta: ClassVar[Optional[NodeMeta]] = None
    mime_meta: ClassVar[Optional[MimeNodeMeta]] = None

    @property
    def requires_halo(self) -> bool:
        """Whether this node's update() accesses spatial neighbours.

        Most MIME nodes are pointwise (rigid body, ODE-based actuation)
        and return False. Spatially-resolved nodes (CSF flow, diffusion)
        must override and return True.
        """
        return False

    def observable_fields(self) -> list[str]:
        """State fields visible to a ControlPolicy via UncertaintyModel.

        Default: all fields from initial_state(). Override to restrict
        visibility.
        """
        return self.state_fields()

    def commandable_fields(self) -> list[str]:
        """Boundary inputs a ControlPolicy may set at runtime.

        Derived from ActuationMeta.commandable_fields if present.
        Must be a subset of boundary_input_spec() keys.
        """
        if self.mime_meta and self.mime_meta.actuation:
            return list(self.mime_meta.actuation.commandable_fields)
        return []

    def validate_mime_consistency(self) -> list[str]:
        """Check internal consistency of MIME metadata.

        Returns a list of error strings (empty = consistent).
        """
        errors: list[str] = []
        cls_name = type(self).__name__

        if self.meta is None:
            errors.append(f"{cls_name}: meta (NodeMeta) is not set")
        if self.mime_meta is None:
            errors.append(f"{cls_name}: mime_meta (MimeNodeMeta) is not set")
            return errors  # Can't check further without mime_meta

        # commandable_fields must be subset of boundary_input_spec keys
        bi_keys = set(self.boundary_input_spec().keys())
        for cf in self.commandable_fields():
            if cf not in bi_keys:
                errors.append(
                    f"{cls_name}: commandable field '{cf}' not in "
                    f"boundary_input_spec() keys: {bi_keys}"
                )

        # observable_fields must be subset of state_fields
        sf = set(self.state_fields())
        for of in self.observable_fields():
            if of not in sf:
                errors.append(
                    f"{cls_name}: observable field '{of}' not in "
                    f"state_fields(): {sf}"
                )

        # Role-specific checks
        from mime.core.metadata import NodeRole
        role = self.mime_meta.role

        if role == NodeRole.ROBOT_BODY and self.mime_meta.biocompatibility is None:
            errors.append(
                f"{cls_name}: robot_body role requires BiocompatibilityMeta"
            )

        if role == NodeRole.SENSING and self.mime_meta.sensing is None:
            errors.append(
                f"{cls_name}: sensing role requires SensingMeta"
            )

        if role == NodeRole.THERAPEUTIC and self.mime_meta.therapeutic is None:
            errors.append(
                f"{cls_name}: therapeutic role requires TherapeuticMeta"
            )

        if role == NodeRole.EXTERNAL_APPARATUS and self.mime_meta.actuation is None:
            errors.append(
                f"{cls_name}: external_apparatus role requires ActuationMeta"
            )

        return errors
