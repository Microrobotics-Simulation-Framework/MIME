"""The solved-field contract's declarations (F0's SF-1 to SF-11): what a field kind a medium carries states about
itself, so that the build and the conformance suite (SF-12, ``mime.testing.field_conformance``) read it rather than
the kind's internals. **The standing check**: nothing here names a solver, a projector or a fluid's operator; the one
fluid-specific word is the quantity a kind provides, which the vocabulary names.

Three rows are open in the record (SF-2's time level, SF-4's exchange interface, SF-11's rates) and may be declared
:data:`NOT_DECLARED`; every other row is declared by every kind."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

#: The value an open row takes until its decision is made.
NOT_DECLARED = "not declared"
#: The rows the record leaves open, the only ones :data:`NOT_DECLARED` is accepted on.
OPEN_ROWS = ("time_level", "exchange", "rates")
#: How a medium's field is hosted: solved by the field kind's own scheme, or prescribed by an analytic solution.
HOSTINGS = ("solved", "prescribed")
#: How a field couples at its openings (SF-6).
PORTS = ("none", "implicit", "lagged")


@dataclass(frozen=True, kw_only=True)
class SolvedField:
    """One field kind's declarations, by the contract's part:

    SF-1 ``hosting`` (``"solved"`` or ``"prescribed"``) and ``role`` (the medium's key, ``"flow"``); SF-2 ``provides``
    (the vocabulary's interfaces a reading gives, ``field.flow.velocity``) and ``time_level`` (the time a reading
    belongs to: open); SF-3 ``own_contribution`` (what the field provides of its response to given sources: ``"none"``
    until a kind does); SF-4 ``accepts`` (the interfaces it is driven through), ``exchange`` (the exchange law's
    interface: open) and ``outside`` (it takes no delivery from anything that reads it, F0-1 M3); SF-5
    ``domain_kinds``; SF-6 ``ports``; SF-7 ``properties`` (what it samples of its material); SF-8 ``settings`` (its
    settings, grouped by owner, as their names); SF-9 ``ledger_route`` (2, its step identity by formula; 1, storage and
    dissipation only; ``None``, with ``ledger_reason``); SF-10 ``steady_solve``, ``affine_response``, ``stores`` (what
    it stores, the fluid's inertia or ``"nothing"``), ``batches`` and ``dtypes``; SF-11 ``rates`` (open)."""

    hosting: str
    role: str
    provides: tuple[str, ...]
    time_level: str
    own_contribution: str
    accepts: tuple[str, ...]
    exchange: str
    outside: bool
    domain_kinds: tuple[str, ...]
    ports: str
    properties: tuple[str, ...]
    settings: tuple[str, ...]
    ledger_route: int | None
    ledger_reason: str
    steady_solve: bool
    affine_response: bool
    stores: str
    batches: bool
    dtypes: tuple[str, ...]
    rates: str

    contract: ClassVar[str] = "SF-1 to SF-11"

    def problems(self) -> list[str]:
        """What is wrong with the declarations as written: an open row's value on a decided row, an unknown hosting
        or port coupling, a route with no reason or a reason with a route, nothing provided."""
        found: list[str] = []
        for item in fields(self):
            value = getattr(self, item.name)
            if value == NOT_DECLARED and item.name not in OPEN_ROWS:
                found.append(f"{item.name} is a decided row of the contract and must be declared")
        if self.hosting not in HOSTINGS:
            found.append(f"hosting is one of {HOSTINGS}, not {self.hosting!r}")
        if self.ports not in PORTS:
            found.append(f"ports is one of {PORTS}, not {self.ports!r}")
        if not self.provides:
            found.append("a field kind provides at least one quantity")
        if self.ledger_route not in (1, 2, None):
            found.append(f"ledger_route is 1, 2 or None, not {self.ledger_route!r}")
        if (self.ledger_route is None) != bool(self.ledger_reason):
            found.append("a kind with no ledger route states why (ledger_reason), and only then")
        if self.outside and self.exchange not in ("none", NOT_DECLARED):
            found.append("a kind outside takes no exchange delivery")
        if not self.dtypes:
            found.append("a kind runs in at least one dtype")
        return found


@dataclass(frozen=True)
class ExchangeFrame:
    """What a field that takes exchange deliveries tells its partners (SF-4, its exchange interface): the node that
    holds it, the grid kind and its cells and spacing (m), and where the grid's origin sits in the world (m), so that a
    partner's points are carried into the grid's coordinates by a translation; the state fields holding each face set
    and the inputs taking each face set's force density. The kernel's weights are the grid kind's own."""

    node: str
    kind: str
    cells: tuple[int, int, int]
    spacing: float
    origin: tuple[float, float, float]
    faces: tuple[str, str, str]
    exchanges: tuple[str, str, str]
    #: The time scheme the field declares (its kind's name), or ``None`` where it chooses one from the run's values.
    scheme: str | None = None

    def grid_point(self, world: tuple[float, ...]) -> tuple[float, float, float]:
        """A world point in the grid's coordinates."""
        return (world[0] - self.origin[0], world[1] - self.origin[1], world[2] - self.origin[2])


__all__ = ["HOSTINGS", "NOT_DECLARED", "OPEN_ROWS", "PORTS", "ExchangeFrame", "SolvedField"]
