from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mime.registry import Interaction, Observation
from mime.things import Thing, TypedPath


@dataclass(frozen=True)
class RunSettings:
    """The experiment's settings as written, before phase 1 checks them: a script is not type-checked, so each may
    hold anything until then. ``up`` is held only to be refused: up is declared on the scene."""

    experiment: Any
    scene: Any
    gravity: Any
    dt: Any
    seed: Any
    up: Any = None


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
    """A value of a thing stated in the experiment, with its reason: ``path`` is ``<thing>.<value>``."""

    path: TypedPath | str
    value: Any
    reason: Any


Declaration = ThingDeclaration | Attachment | ObservationDeclaration | ValueDeclaration
