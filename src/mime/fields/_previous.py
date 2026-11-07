"""The versions of field providers kept registered after a later version changed what they compute, so that a folder
saved with one still loads and reproduces: the uniform rotating field 0.1.0, whose phase is the present frequency times
the time since the start (MIME-ANO-139). A script that writes ``UniformRotating(...)`` gets the current version; a
load of a document that names 0.1.0 gets it from here (``mime.fields.KEPT``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from mime.fields._uniform_rotating import UniformRotating
from mime.nodes.actuation import UniformRotatingNode
from mime.registry import BuildContext, register_field_provider

if TYPE_CHECKING:
    from mime.build import ProviderBuilder, Slot


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class ClockRotating(UniformRotating):
    """``uniform_rotating`` 0.1.0: the phase is ``2 pi f t``, the frequency commanded at a step times the time since
    the start, not the integral of the frequency. For a constant frequency that is the integral, and 0.2.0 computes the
    same to the last bit; for one that changes in a run it turns the field at ``2 pi (f + t df/dt)`` and jumps its angle
    where the frequency steps (MIME-ANO-139). So it refuses, when the experiment is built, any drive of its frequency but
    the constant and a controller that writes it, naming 0.2.0 as the route."""

    version = "0.1.0"
    builds = {"": (UniformRotatingNode,)}
    constant_commands = {"frequency": (
        "its phase is the present frequency times the time since the start, not the integral of the frequency, so a "
        "frequency that changes in a run would turn the field at the wrong rate and jump its angle (MIME-ANO-139)",
        "build the field's current version, uniform_rotating 0.2.0, `UniformRotating(...)`, whose phase is the "
        "integral of its frequency")}

    @override
    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        b.add_node(UniformRotatingNode(ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots],
                                       rotation_axis=ctx.vector("rotation_axis"),
                                       initial_direction=ctx.vector("initial_direction")))


__all__ = ["ClockRotating"]
