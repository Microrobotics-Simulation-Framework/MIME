from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mime.registry import Controller, Interaction, Observation
from mime.things import Thing, TypedPath
from mime.units import standard_gravity
from mime.validity import DEFAULT_PROFILE

#: An experiment's gravity when it states none: standard gravity, a convention default (design #191 item 4). The build
#: knows it by identity, so a script that writes ``standard_gravity()`` states it, and the document writes it.
CONVENTIONAL_GRAVITY = standard_gravity()


@dataclass(frozen=True)
class RunSettings:
    """The experiment's settings as written, before phase 1 checks them: a script is not type-checked, so each may
    hold anything until then. ``up`` is held only to be refused: up is declared on the scene. ``conditions`` are the
    operating conditions every record is read at; ``profile`` the profile it is built under (``exploratory`` when
    not given)."""

    experiment: Any
    scene: Any
    gravity: Any
    dt: Any
    seed: Any
    up: Any = None
    conditions: Any = None
    profile: Any = DEFAULT_PROFILE.value


@dataclass(frozen=True)
class ThingDeclaration:
    thing: Thing


@dataclass(frozen=True)
class Attachment:
    law: Interaction
    bindings: Mapping[str, Any]


@dataclass(frozen=True)
class ObservationDeclaration:
    observation: Observation


@dataclass(frozen=True)
class ValueDeclaration:
    """A value of a thing stated in the experiment, with its reason: ``path`` is ``<thing>.<value>``. A reason not
    given (``None``) is pending, unless the quantity carries its own."""

    path: TypedPath | str
    value: Any
    reason: Any = None


@dataclass(frozen=True)
class SourceDeclaration:
    """A setting's source, stated apart from its value (``exp.source("tube.inner_radius", reason=...)``): for a value
    that is not a quantity, or a source added when an experiment is sealed."""

    path: Any
    reason: Any


@dataclass(frozen=True)
class ControllerDeclaration:
    """The experiment's controller as ``set_controller`` declares it: the entry; what each of its read labels reads
    and each write label writes, as written (a path or its string); its mode, rate and latency."""

    controller: Controller
    reads: Mapping[str, Any]
    writes: Mapping[str, Any]
    mode: Any
    rate: Any = None
    latency: Any = None


Declaration = (ThingDeclaration | Attachment | ObservationDeclaration | ValueDeclaration | ControllerDeclaration
               | SourceDeclaration)
