"""Drives: in-graph command generators, each a node whose numbers are ``params`` leaves with one exclusive edge into
the apparatus command it writes. A quantity written for a command is short for :class:`Constant`; a controller (phase
7) replaces the drive of each command it writes. A drive's id is ``drive:<apparatus>.<command>``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, override

from mime.gradients import GradientStatus
from mime.nodes.actuation import ConstantNode
from mime.registry import DESIGN
from mime.registry import BuildContext, Commanded, Drive, register_drive
from mime.units import Quantity

if TYPE_CHECKING:
    from mime.build import DriveBuilder


@register_drive
@dataclass(frozen=True, kw_only=True)
class Constant(Drive):
    """The command's value, the same every step: ``Constant(Hz(10))``."""

    name = "drive.constant"
    version = "0.1.0"
    setting_roles = {"value": DESIGN}
    differentiability = GradientStatus.EXACT
    references = ()
    builds = {"": (ConstantNode,)}
    value: Annotated[Quantity, Commanded] = field(kw_only=False)

    @override
    def build(self, ctx: BuildContext, b: DriveBuilder) -> None:
        value = ctx.settings["value"]
        node = b.add_node(ConstantNode(ctx.node_name(), ctx.timestep, value=value))
        b.command(node["value"])


def as_drive(value: object) -> Drive:
    """A command as its drive: a drive as written, a quantity as :class:`Constant`."""
    if isinstance(value, Drive):
        return value
    if isinstance(value, Quantity):
        return Constant(value)
    raise TypeError(f"a command takes a drive or a quantity (short for drives.Constant), not {value!r}")


__all__ = ["Constant", "as_drive"]
