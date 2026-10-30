"""A surrogate as a type: an asset record naming its evaluator, what it outputs, its accuracy per use and its provenance.

A surrogate's weights are an asset (``mime.data``'s, by name and sha256). What makes them a model is what the record
says of them: the **evaluator** that reads the bytes and computes from them (registered by name and version, as a wall
kind's correction is), the **outputs** it delivers (a resistance always; a flow-load operator, the load per unit of an
ambient flow, where it was trained for one), the **features** it reads, in order, its **accuracy per use** (each a
measurement with what it bounds or estimates, or ``None`` where nothing measured it), and its **provenance**: what it
was trained on, on which of the body's patches, about which reference point, and the shipped derivation record
(``mime.assets.derivation_of``) that holds its builder, inputs by hash, parameters and checks. ``staleness`` compares
that record's inputs with what plays their roles now: a surrogate trained through a table that has since been replaced
is stale, and says so.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

#: What a surrogate can deliver: its resistance always, and the operator taking an ambient flow to a load where trained.
OUTPUTS = ("resistance", "flow_load")
#: The reference points a surrogate's resistance can be trained about: the end of the body's axis (``zeta = 0`` of de
#: Jongh's Eq. 1, the 0.2 line's) or the body's centroid.
REFERENCE_POINTS = ("end", "centre")


@dataclass(frozen=True)
class Accuracy:
    """A surrogate's accuracy for one ``use`` (``resistance``, ``swim_speed``): the ``measure`` taken, its ``value``
    (``None`` where nothing has measured it: unknown, not bounded), ``against`` what, ``over`` which cases, and the
    ``claim``, what the value bounds or estimates and what it does not; the ``anomaly`` that records it, if any."""

    use: str
    measure: str
    value: float | None
    against: str
    over: str
    claim: str
    anomaly: str | None = None


@dataclass(frozen=True)
class Provenance:
    """Where a surrogate's weights came from: what they were ``trained_on``, the body's ``patches`` its training
    sampled, the ``reference_point`` its resistance is about, and ``derivation``, the asset whose shipped derivation
    record holds the rest. ``stale`` is the declared reason it is out of date (the anomaly that says so), or ``None``;
    ``staleness`` must find the same."""

    trained_on: str
    patches: tuple[str, ...]
    reference_point: str
    derivation: str
    stale: str | None = None


@dataclass(frozen=True)
class Evaluator:
    """How a surrogate's bytes become a model: ``read`` takes the asset's bytes to a network, ``resistance`` takes the
    network and the features (in the dtype of the run) to the 6 x 6 non-dimensional resistance, and ``flow_load``, where
    the evaluator has one, to the flow-load operator."""

    name: str
    version: str
    read: Callable[[bytes], Any]
    resistance: Callable[[Any, Any], Any]
    flow_load: Callable[[Any, Any], Any] | None = None

    def label(self) -> str:
        return f"{self.name} {self.version}"


_evaluators: dict[str, Evaluator] = {}


def register_evaluator(evaluator: Evaluator) -> Evaluator:
    held = _evaluators.get(evaluator.label())
    if held is not None and held is not evaluator:
        raise ValueError(f"an evaluator {evaluator.label()} is already registered")
    _evaluators[evaluator.label()] = evaluator
    return evaluator


def evaluator_named(label: str) -> Evaluator:
    """The evaluator registered as ``"<name> <version>"``; ``KeyError`` naming those registered otherwise."""
    if label not in _evaluators:
        raise KeyError(f"no surrogate evaluator {label!r} is registered; those registered are {evaluators()}")
    return _evaluators[label]


def evaluators() -> list[str]:
    return sorted(_evaluators)


@dataclass(frozen=True, kw_only=True)
class Surrogate:
    """A surrogate's record: the ``asset`` holding its weights, the ``evaluator`` (``"<name> <version>"``) that reads
    them, its ``outputs``, the ``features`` it reads in order, its ``accuracy`` per use, and its ``provenance``."""

    asset: str
    evaluator: str
    features: tuple[str, ...]
    accuracy: tuple[Accuracy, ...]
    provenance: Provenance
    outputs: tuple[str, ...] = ("resistance",)

    def __post_init__(self) -> None:
        if "resistance" not in self.outputs or not set(self.outputs) <= set(OUTPUTS):
            raise ValueError(f"{self.asset}: a surrogate outputs a resistance, and optionally a flow load: "
                             f"{self.outputs!r}")
        if self.provenance.reference_point not in REFERENCE_POINTS:
            raise ValueError(f"{self.asset}: the reference point {self.provenance.reference_point!r} is not one of "
                             f"{REFERENCE_POINTS}")
        uses = [each.use for each in self.accuracy]
        if len(uses) != len(set(uses)):
            raise ValueError(f"{self.asset}: one accuracy per use, not {uses}")

    def accuracy_for(self, use: str) -> Accuracy | None:
        found = [each for each in self.accuracy if each.use == use]
        return found[0] if found else None

    def resolve(self) -> Evaluator:
        """The registered evaluator this record names, refused where it does not deliver the record's outputs."""
        held = evaluator_named(self.evaluator)
        if "flow_load" in self.outputs and held.flow_load is None:
            raise ValueError(f"{self.asset} outputs a flow load, and its evaluator {held.label()} computes none")
        return held


def current_inputs() -> Mapping[str, Any]:
    """What plays each role a derivation can name, now: ``wall:<kind>`` for each wall kind's correction that reads a
    shipped table, the reference of that table."""
    from mime import assets
    from mime.nodes.hydro.bem import wall_correction_named, wall_corrections

    found: dict[str, Any] = {}
    for label in wall_corrections():
        name, version = label.split(" ")
        correction = wall_correction_named(name, version)
        for asset in correction.assets:
            found[f"wall:{name}"] = assets.reference(asset)
    return MappingProxyType(found)


def staleness(surrogate: Surrogate, current: Mapping[str, Any] | None = None) -> list[str]:
    """Why ``surrogate`` is stale, from its derivation record's inputs against ``current`` (what plays each role now,
    ``current_inputs()`` by default); none when every input it names is the one in use."""
    from mime import assets

    record = assets.derivation_of(surrogate.provenance.derivation)
    return assets.stale_inputs(record, current_inputs() if current is None else current)


__all__ = ["OUTPUTS", "REFERENCE_POINTS", "Accuracy", "Evaluator", "Provenance", "Surrogate", "current_inputs",
           "evaluator_named", "evaluators", "register_evaluator", "staleness"]
