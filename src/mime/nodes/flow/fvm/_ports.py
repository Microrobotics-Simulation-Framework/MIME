"""The laws held at a flow's openings (port laws) and the step's port system (F0's B6/B7, decisions 7 to 9 and 11; the
record's "F2's rows, 2", A2's hybrid rows).

**A port law's rows.** A law binds one opening or several (:attr:`PortLaw.openings`, the record's "F2's rows, 2", item
1) and states a row for each over the stage, affine in the rest (A2): **pressure rows**, its openings' pressures affine
in their outward fluxes, ``p = a + B q`` with ``B`` a block over its openings (a Windkessel's ``1 x 1``; a shared
distal resistance's ``2 x 2``, whose off-diagonals couple its openings), or **flux rows**, its fluxes given, ``q = a``
(an inflow; a closed valve, ``q = 0``). A one-opening law gives ``a`` and ``b`` as scalars, a law over several as a
vector and a matrix. The field calls :meth:`PortLaw.affine` for the stage, solves its openings' rows together with its
own affine response, and calls :meth:`PortLaw.advanced` with the stage's solved fluxes and pressures (scalars for one
opening, vectors for several); the law books its own terms (:meth:`PortLaw.books`). A hybrid form is guaranteed only
for standard (power-conserving) relations: the stated limit that stays.

**The port system** (:func:`port_system`). The field sees each opening's pressure relative to one reference opening
(the gauge, decision 9: an incompressible field's effort is defined up to a constant, the laws see absolute pressures);
its fluxes are affine in those differences through its unit fields' response ``R``: ``q = q0 + R m``, ``m`` the
differences' move from those the step was solved at; the laws' blocks are placed in one block ``B`` over the openings. The rows are solved for the moves first, the reference's
pressure (the level) eliminated by its own row, then for the level: so a move is formed from the rows' own terms and
never as the difference of two absolute pressures (whose rounding, ``eps |level|``, would grow with a downstream
resistance many times the vessel's), save where two pressure laws' own pressures differ. **Well-posed** where some row
is a pressure row; with none, the flux rows (which then cover every opening) sum to zero, so one is dropped for the
gauge's reference row (the reference's pressure set to zero): what the build checks before it builds
(:func:`well_posed`).

**Passivity** (CI check (a)): the block ``B`` of the rows the solver solves (:meth:`PortLaw.affine`'s) has a positive
semidefinite symmetric part at every step a run can take, its limit ``dt -> 0`` included, which keeps the port system
well posed and is invisible to the law's own identity (which books whatever the law does). A source (a given flux or
pressure: it supplies work) is exempt (:attr:`PortLaw.source`)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple, override

from jax import Array

from mime.core import jnp
from mime.core import ledger as books
from mime.core.ledger import TermOrder


class RowKind(StrEnum):
    """What an opening's law states over a stage: its pressure, affine in its flux, or its flux."""

    PRESSURE = "pressure"
    FLUX = "flux"


#: The parameters a law takes, a row an opening (padded with zeros), and its own state, a row an opening.
PARAMETERS, STATES = 4, 1


@dataclass(frozen=True)
class LawBooks:
    """A law's own books over one step (J): its storage at the step's start and end, what its resistances lost, and
    its own scheme's numerical term (zero for a law that stores nothing or steps exactly)."""

    storage_start: Array
    storage_end: Array
    loss: Array
    numerical: Array


def _waveform(parameters: Array, time: Any) -> Array:
    """``steady + amplitude cos(2 pi frequency t + phase)`` from a law's first four parameters."""
    return parameters[0] + parameters[1] * jnp.cos(2.0 * jnp.pi * parameters[2] * time + parameters[3])


class PortLaw:
    """A registered port law (the module's interface): how many openings it binds, its row kind, its parameters (SI, a
    row a law), whether it stores energy, whether it keeps its own books in the ledger (a storing law does, and a law
    over several openings, whose books see its off-diagonals), whether it is a source (exempt from passivity), its class
    in the ledger, and its numerical terms' orders (``mime.testing.term_orders``)."""

    name: ClassVar[str]
    version: ClassVar[str]
    row: ClassVar[RowKind]
    parameters: ClassVar[tuple[str, ...]]
    openings: ClassVar[int] = 1
    stores: ClassVar[bool] = False
    own_books: ClassVar[bool] = False
    source: ClassVar[bool] = False
    energy_class: ClassVar[str] = books.UNMETERED
    #: The law's numerical term's expected order (``mime.core.ledger.TermOrder``), by its name.
    orders: ClassVar[Mapping[str, TermOrder]] = MappingProxyType({})

    def row_of(self, settings: Mapping[str, object]) -> tuple[float, ...]:
        """The law's row of parameters (SI, padded to :data:`PARAMETERS`) from its settings by their names, as an
        experiment's law at an opening gives them (a setting it leaves out is zero)."""
        values = [float(held) if isinstance(held := settings.get(name), (int, float)) else 0.0
                  for name in self.parameters]
        return tuple(values + [0.0] * (PARAMETERS - len(values)))

    def affine(self, parameters: Array, state: Array, time: Any, dt: Any) -> tuple[Array, Array]:
        """The stage's rows: ``(a, b)`` with ``p = a + b q`` for pressure rows, ``(q, 0)`` for flux rows (``q``
        outward), at the stage's instant ``time``: scalars for a law at one opening, a vector and a matrix over its
        openings for one over several."""
        raise NotImplementedError

    def advanced(self, parameters: Array, state: Array, flux: Array, pressure: Array, dt: Any) -> Array:
        """The law's own state after the stage, given its solved flux and pressure."""
        return state

    def books(self, parameters: Array, state: Array, ended: Array, flux: Array, pressure: Array, dt: Any) -> LawBooks:
        """The law's own books over the stage: ``dt p . q`` it received equals its storage's change, its loss and its
        numerical term."""
        zero = jnp.zeros((), dtype=flux.dtype)
        return LawBooks(zero, zero, zero, zero)


class Windkessel(PortLaw):
    """The RCR Windkessel stepped by backward Euler at the flow's instant (decision 8): the proximal resistance
    ``R_p``, the compliance ``C`` and the distal resistance ``R_d`` to a distal pressure of zero. Over a stage,
    ``Pc' = (C Pc + dt q) / (C + dt / R_d)``, so ``p = Pc' + R_p q = a + b q`` with ``a = C Pc / (C + dt / R_d)`` and
    ``b = R_p + dt / (C + dt / R_d)``. Its books: ``dt p q = (C/2)(Pc'^2 - Pc^2) + dt (R_p q^2 + Pc'^2 / R_d) +
    N_WK``, ``N_WK = (C/2)(Pc' - Pc)^2`` its own first-order numerical dissipation (the spike's: the exchange
    workless to about 1e-16, ``N_WK`` halving with dt)."""

    name = "windkessel"
    version = "0.1.0"
    row = RowKind.PRESSURE
    parameters = ("proximal", "compliance", "distal")
    stores = True
    energy_class = books.DISSIPATIVE
    #: ``N_WK``, a step's capacitor change squared: ``O(dt^2)`` a step.
    orders = MappingProxyType({"windkessel": TermOrder(dt=2)})

    @staticmethod
    def _denominator(parameters: Array, dt: Any) -> Array:
        return parameters[1] + dt / parameters[2]

    @override
    def affine(self, parameters: Array, state: Array, time: Any, dt: Any) -> tuple[Array, Array]:
        held = self._denominator(parameters, dt)
        return parameters[1] * state[0] / held, parameters[0] + dt / held

    @override
    def advanced(self, parameters: Array, state: Array, flux: Array, pressure: Array, dt: Any) -> Array:
        held = self._denominator(parameters, dt)
        return jnp.stack([(parameters[1] * state[0] + dt * flux) / held])

    @override
    def books(self, parameters: Array, state: Array, ended: Array, flux: Array, pressure: Array, dt: Any) -> LawBooks:
        capacitance = parameters[1]
        before, after = state[0], ended[0]
        loss = dt * (parameters[0] * flux * flux + after * after / parameters[2])
        return LawBooks(0.5 * capacitance * before * before, 0.5 * capacitance * after * after, loss,
                        0.5 * capacitance * (after - before) ** 2)


class Inflow(PortLaw):
    """A flux pinned at an opening (decision 7: the evolved face whose pressure pins its flux; the spike's flux to
    4e-15): ``Q(t) = steady + amplitude cos(2 pi frequency t + phase)`` into the field (m^3/s), so the opening's
    outward flux is ``-Q(t)``. A source: the work it supplies is the pressure the field returns times its flux."""

    name = "inflow"
    version = "0.1.0"
    row = RowKind.FLUX
    parameters = ("steady", "amplitude", "frequency", "phase")
    source = True

    @override
    def affine(self, parameters: Array, state: Array, time: Any, dt: Any) -> tuple[Array, Array]:
        return -_waveform(parameters, time), jnp.zeros((), dtype=parameters.dtype)


class Valve(PortLaw):
    """A closed valve at an opening: a flux row, ``q = 0`` (A2: what impedance rows alone cannot state). Workless: the
    pressure the field returns does no work on a flux of zero; its row's block is zero, passive."""

    name = "valve"
    version = "0.1.0"
    row = RowKind.FLUX
    parameters = ()
    energy_class = books.WORKLESS

    @override
    def affine(self, parameters: Array, state: Array, time: Any, dt: Any) -> tuple[Array, Array]:
        zero = jnp.zeros((), dtype=parameters.dtype)
        return zero, zero


class SharedDistal(PortLaw):
    """Two openings each through its own resistance into one shared distal resistance to a pressure of zero (the record's
    amendment 5, R-4): ``p_k = R_k q_k + R_d (q_1 + q_2)``, pressure rows over both with the block ``[[R_1 + R_d, R_d],
    [R_d, R_2 + R_d]]``, symmetric and positive definite for positive resistances. Its books: ``dt p . q = dt (R_1 q_1^2
    + R_2 q_2^2 + R_d (q_1 + q_2)^2)``, all loss, exactly, so a lagged or dropped off-diagonal shows in them."""

    name = "shared_distal"
    version = "0.1.0"
    row = RowKind.PRESSURE
    parameters = ("first", "second", "distal")
    openings = 2
    own_books = True
    energy_class = books.DISSIPATIVE

    @override
    def affine(self, parameters: Array, state: Array, time: Any, dt: Any) -> tuple[Array, Array]:
        distal = parameters[2]
        block = jnp.stack([jnp.stack([parameters[0] + distal, distal]), jnp.stack([distal, parameters[1] + distal])])
        return jnp.zeros(2, dtype=parameters.dtype), block

    @override
    def books(self, parameters: Array, state: Array, ended: Array, flux: Array, pressure: Array, dt: Any) -> LawBooks:
        zero = jnp.zeros((), dtype=flux.dtype)
        total = flux[0] + flux[1]
        loss = dt * (parameters[0] * flux[0] * flux[0] + parameters[1] * flux[1] * flux[1]
                     + parameters[2] * total * total)
        return LawBooks(zero, zero, loss, zero)


class GivenPressure(PortLaw):
    """A pressure given at an opening, ``p(t) = steady + amplitude cos(2 pi frequency t + phase)`` (Pa): a pressure row
    with no block. A source. The node level's (the open tube's gates against the periodic tube); no experiment's law
    builds it yet."""

    name = "pressure"
    version = "0.1.0"
    row = RowKind.PRESSURE
    parameters = ("steady", "amplitude", "frequency", "phase")
    source = True

    @override
    def affine(self, parameters: Array, state: Array, time: Any, dt: Any) -> tuple[Array, Array]:
        return _waveform(parameters, time), jnp.zeros((), dtype=parameters.dtype)


PORT_LAWS: MappingProxyType[str, PortLaw] = MappingProxyType({
    Windkessel.name: Windkessel(), Inflow.name: Inflow(), Valve.name: Valve(), SharedDistal.name: SharedDistal(),
    GivenPressure.name: GivenPressure()})


def port_law_named(name: str) -> PortLaw:
    if name not in PORT_LAWS:
        raise ValueError(f"no port law {name!r} is registered: {sorted(PORT_LAWS)}")
    return PORT_LAWS[name]


def reference_of(rows: Sequence[RowKind]) -> int:
    """The gauge's reference opening: the first with a pressure row, or the first."""
    return next((index for index, row in enumerate(rows) if row is RowKind.PRESSURE), 0)


#: How near a waveforms' sum must come to zero, relative to their sizes, to be read as zero.
SUM_ZERO = 1e-12


def waveforms_cancel(waveforms: Sequence[Sequence[float]]) -> bool:
    """Whether waveforms ``steady + amplitude cos(2 pi frequency t + phase)`` (each a row of those four) sum to zero at
    every instant: their steady parts, and the phasors ``amplitude exp(i phase)`` of each frequency, cancel."""
    size = math.fsum(abs(w[0]) + abs(w[1]) for w in waveforms)
    if abs(math.fsum(w[0] for w in waveforms)) > SUM_ZERO * size:
        return False
    phasors: dict[float, complex] = {}
    for w in waveforms:
        if w[1] != 0.0:
            key = float(f"{w[2]:.12g}")
            phasors[key] = phasors.get(key, 0j) + w[1] * complex(math.cos(w[3]), math.sin(w[3]))
    return all(abs(value) <= SUM_ZERO * size for value in phasors.values())


def well_posed(rows: Sequence[RowKind], waveforms: Sequence[Sequence[float]] = ()) -> str:
    """Why the openings' rows are not well posed, or ``""``: some pressure row (the gauge's reference row is the
    field's, so with one the system has a solution), or, with none, flux rows covering every opening whose given
    fluxes (``waveforms``, each flux row's parameters) sum to zero at every instant: an incompressible field takes in
    what it lets out."""
    if not rows:
        return "the flow has no opening"
    if any(row is RowKind.PRESSURE for row in rows):
        return ""
    if len(waveforms) == len(rows) and waveforms_cancel(waveforms):
        return ""
    return ("every opening's law gives its flux and none its pressure, and their fluxes do not sum to zero at every "
            "instant: an incompressible flow takes in what it lets out, so the openings' rows have no solution")


class PortSolution(NamedTuple):
    """The port system's answer: each opening's pressure (absolute, Pa), its pressure relative to the reference opening
    (Pa, zero at the reference: the field's own unknowns), that relative pressure's move from the one the step was
    solved at (Pa, zero at the reference), and its outward flux by its row (m^3/s)."""

    pressures: Array
    relative: Array
    moved: Array
    fluxes: Array


def port_system(rows: Sequence[RowKind], a: Array, b: Array, base: Array, response: Array, predicted: Array,
                reference: int, coupling: Sequence[Sequence[bool]] | None = None) -> PortSolution:
    """The openings' pressures and fluxes solving each opening's row with the field's affine response: the fluxes
    ``q = base + R m``, ``m`` the moves of the pressures relative to ``reference`` (``R`` a column for each opening but
    the reference) from ``predicted``, those the step was solved at (relative, zero at the reference); ``b`` the laws'
    blocks placed in one over the openings. Rows: a pressure row ``p_i - (b q)_i = a_i`` with ``p_i = level +
    predicted_i + m_i``, a flux row ``q_i = a_i``; the reference's own
    row is a pressure row where any is (``reference_of``) and gives the level once the moves are solved from the
    others' rows (each other pressure row less the reference's, so the level leaves them); with no pressure row the
    gauge sets the level to zero and the reference's flux row, which the others' imply, is dropped. ``coupling`` names
    which entries of ``b`` a law places (structural: one law's openings with each other; the diagonal by default), so a
    row sums only its own law's terms and a law at one opening forms its row as ``b_i q_i`` alone."""
    count = len(rows)
    others = [index for index in range(count) if index != reference]
    a, b, base, response, predicted = (jnp.asarray(x) for x in (a, b, base, response, predicted))
    dtype = base.dtype
    reduced, right, held, gauge = _reduced(rows, a, b, base, response, predicted, reference, coupling)
    moves = jnp.linalg.solve(reduced, right) if others else jnp.zeros(0, dtype=dtype)
    level = jnp.zeros((), dtype=dtype) if gauge else held[1] - jnp.dot(held[0], moves)
    moved = jnp.stack([moves[others.index(j)] if j != reference else jnp.zeros((), dtype=dtype) for j in range(count)])
    relative = jnp.stack([predicted[j] + moved[j] if j != reference else jnp.zeros((), dtype=dtype)
                          for j in range(count)])
    return PortSolution(level + relative, relative, moved, base + response @ moves)


def _reduced(rows: Sequence[RowKind], a: Array, b: Array, base: Array, response: Array, predicted: Array,
             reference: int, coupling: Sequence[Sequence[bool]] | None
             ) -> tuple[Array, Array, tuple[Array, Array], bool]:
    """The port system's rows over the moves (:func:`port_system`): the matrix and right-hand side of every opening's
    row but the reference's, the reference's own row, and whether the gauge sets the level."""
    count = len(rows)
    others = [index for index in range(count) if index != reference]
    dtype = base.dtype
    if b.ndim == 1:  # one-opening laws only: their blocks on the diagonal
        b = jnp.diag(b)
    couples = coupling if coupling is not None else [[i == j for j in range(count)] for i in range(count)]

    def row_of(i: int) -> tuple[Array, Array]:
        """Opening ``i``'s row over the moves, its coefficients and right-hand side, the level's coefficient one for a
        pressure row and zero for a flux row."""
        if rows[i] is RowKind.FLUX:
            return response[i], a[i] - base[i]
        coupled = [j for j in range(count) if couples[i][j]]
        coefficients = -sum((b[i, j] * response[j] for j in coupled), start=jnp.zeros_like(response[i]))
        if i != reference:
            coefficients = coefficients.at[others.index(i)].add(1.0)
        pushed = sum((b[i, j] * base[j] for j in coupled), start=jnp.zeros((), dtype=dtype))
        return coefficients, a[i] + pushed - predicted[i]

    gauge = rows[reference] is RowKind.FLUX
    held = row_of(reference)
    reduced: list[Array] = []
    right: list[Array] = []
    for i in others:
        coefficients, value = row_of(i)
        if rows[i] is RowKind.PRESSURE and not gauge:
            coefficients, value = coefficients - held[0], value - held[1]
        reduced.append(coefficients)
        right.append(value)
    if not others:
        return jnp.zeros((0, 0), dtype=dtype), jnp.zeros(0, dtype=dtype), held, gauge
    return jnp.stack(reduced), jnp.stack(right), held, gauge


def port_matrix(rows: Sequence[RowKind], b: Array, response: Array, reference: int,
                coupling: Sequence[Sequence[bool]] | None = None) -> Array:
    """The matrix :func:`port_system` solves for the moves (its rows' coefficients, the reference's eliminated): what
    its conditioning is read from (a stiff resistance standing in for a closed valve grows it with the resistance)."""
    response = jnp.asarray(response)
    count = len(rows)
    zeros = jnp.zeros(count, dtype=response.dtype)
    return _reduced(rows, zeros, jnp.asarray(b), zeros, response, zeros, reference, coupling)[0]


def block(law: PortLaw, parameters: Array, dt: Any) -> Array:
    """The block ``B`` of the rows ``law`` gives the solver over a stage of ``dt`` (Pa s/m^3), a square matrix over its
    openings (``1 x 1`` for a one-opening law): :meth:`PortLaw.affine`'s own, at a state of zero (a law's block does not
    depend on its state)."""
    state = jnp.zeros(STATES, dtype=parameters.dtype)
    count = law.openings
    return jnp.reshape(law.affine(parameters, state, jnp.zeros((), dtype=parameters.dtype), dt)[1], (count, count))


def passive(law: PortLaw, parameters: Array, dt: Any) -> Array:
    """Whether the block of the row ``law`` gives the solver (:func:`block`) has a positive semidefinite symmetric part
    at ``parameters`` over a stage of ``dt``, judged by ``definite_at_floor`` (traced: a boolean)."""
    return books.definite_at_floor(block(law, parameters, dt))


__all__ = ["PARAMETERS", "PORT_LAWS", "STATES", "SUM_ZERO", "GivenPressure", "Inflow", "LawBooks", "PortLaw", "PortSolution",
           "RowKind", "SharedDistal", "Valve", "Windkessel", "block", "passive", "port_law_named", "port_matrix", "port_system",
           "reference_of", "waveforms_cancel", "well_posed"]
