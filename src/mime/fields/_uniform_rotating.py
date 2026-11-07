from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import IntegratingRotatingNode
from mime.registry import DESIGN
from mime.registry import (
    BuildContext,
    Command,
    Direction,
    DirectionLike,
    FieldProvider,
    register_field_provider,
)
from mime.units import FREQUENCY, MAGNETIC_FLUX_DENSITY, Magnitude
from mime.vocabulary import FLUX_DENSITY, GRADIENT, Capability

if TYPE_CHECKING:
    from mime.build import ProviderBuilder, Slot


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class UniformRotating(FieldProvider):
    """A uniform magnetic field turning right-handed about ``rotation_axis`` (``tube.axis``): ``B = amplitude
    (cos(phi) e1 + sin(phi) axis x e1)``, where ``e1`` is ``initial_direction`` projected onto the plane of rotation,
    the field's direction at ``t = 0``, and the phase ``phi`` is ``2 pi`` times the integral of the commanded
    frequency, each step's frequency held over the step. Its commands are ``frequency`` and ``amplitude``, and either
    may change during a run (a ramp, ``drives.Ramp``, or a controller). The phase is held in turns as a pair
    of floating numbers to which each step adds ``f h`` exactly, so its error does not grow with the run (a constant
    frequency's field differs from version 0.1.0's ``2 pi f t`` by that version's rounding). Being uniform, a reader's slot takes no
    points, and its gradient is zero.

    Version 0.1.0 made the phase as the present frequency times the time since the start, so a frequency that changed
    in a run turned the field at the wrong rate and jumped its angle (MIME-ANO-139). It stays registered
    (``mime.fields.KEPT``) for a folder saved with it, and refuses any writer of its frequency but a constant drive."""

    name = "uniform_rotating"
    version = "0.2.0"
    setting_roles = {"rotation_axis": DESIGN, "initial_direction": DESIGN}
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": (IntegratingRotatingNode,)}
    flags = frozenset({Capability.PROVISION_EVALUATED, Capability.UNIFORM})
    publishes = {FLUX_DENSITY.name: "flux_density", GRADIENT.name: "gradient"}
    commands = {"frequency": Command("frequency", FREQUENCY), "amplitude": Command("amplitude", MAGNETIC_FLUX_DENSITY)}
    turns_at = "frequency"
    rotation_axis: Annotated[DirectionLike, Direction]
    initial_direction: Annotated[DirectionLike, Direction]

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        axis, start = settings["rotation_axis"], settings["initial_direction"]
        if isinstance(axis, tuple) and isinstance(start, tuple) and \
                abs(sum(a * s for a, s in zip(axis, start, strict=True))) > 1.0 - 1e-6:
            return [f"initial_direction {start} lies along the rotation axis {axis}: the field's direction at t = 0 "
                    f"needs a part in the plane of rotation"]
        return []

    @override
    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        b.add_node(IntegratingRotatingNode(ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots],
                                           rotation_axis=ctx.vector("rotation_axis"),
                                           initial_direction=ctx.vector("initial_direction")))
