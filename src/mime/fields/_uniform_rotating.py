from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import UniformRotatingFieldNode
from mime.registry import BuildContext, Command, Direction, DirectionLike, FieldProvider, register_field_provider
from mime.units import FREQUENCY, MAGNETIC_FLUX_DENSITY, Magnitude
from mime.vocabulary import FLUX_DENSITY, GRADIENT, Capability

if TYPE_CHECKING:
    from mime.build import ProviderBuilder, Slot


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class UniformRotatingField(FieldProvider):
    """A uniform magnetic field turning right-handed about ``rotation_axis`` (``tube.axis``): ``B(t) = amplitude
    (cos(2 pi f t) e1 + sin(2 pi f t) axis x e1)``, where ``e1`` is ``initial_direction`` projected onto the plane of
    rotation, the field's direction at ``t = 0``. Its commands are ``frequency`` and ``amplitude``; its time is its
    step count times its timestep. Being uniform, a reader's slot takes no points, and its gradient is zero."""

    name = "uniform_rotating_field"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": (UniformRotatingFieldNode,)}
    flags = frozenset({Capability.PROVISION_EVALUATED, Capability.UNIFORM})
    publishes = {FLUX_DENSITY.name: "flux_density", GRADIENT.name: "gradient"}
    commands = {"frequency": Command("frequency", FREQUENCY), "amplitude": Command("amplitude", MAGNETIC_FLUX_DENSITY)}
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
        b.add_node(UniformRotatingFieldNode(ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots],
                                            rotation_axis=ctx.vector("rotation_axis"),
                                            initial_direction=ctx.vector("initial_direction")))
