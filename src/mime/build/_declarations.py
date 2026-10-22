from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mime.registry import Interaction, Observation
from mime.things import Thing


@dataclass(frozen=True)
class RunSettings:
    """The experiment's settings as written, before phase 1 checks them."""

    experiment: str
    up: Any
    gravity: Any
    dt: Any
    seed: int


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


Declaration = ThingDeclaration | Attachment | ObservationDeclaration
