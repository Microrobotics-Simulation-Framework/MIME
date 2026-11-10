"""The laws held at a flow's openings (port laws) and the step's port system (F0's B6/B7, decisions 7 to 9 and 11; the
record's "F2's rows, 2", A2's hybrid rows).

**A port law's row.** Each opening's law states one row over the stage, affine in the rest (A2): a **pressure row**,
its pressure affine in its own outward flux, ``p = a + b q`` (a Windkessel, a given pressure), or a **flux row**, its
flux given, ``q = a`` (an inflow; later a closed valve, ``q = 0``). The field calls :meth:`PortLaw.affine` for the
stage, solves its openings' rows together with its own affine response, and calls :meth:`PortLaw.advanced` with the
stage's solved flux and pressure; the law books its own terms (:meth:`PortLaw.books`). A hybrid form is guaranteed only
for standard (power-conserving) relations: the stated limit that stays.

**The port system** (:func:`port_system`). The field sees each opening's pressure relative to one reference opening
(the gauge, decision 9: an incompressible field's effort is defined up to a constant, the laws see absolute pressures);
its fluxes are affine in those differences through its unit fields' response ``R``: ``q = q0 + R (x - x_hat)``. The
rows are solved for the differences and the reference's pressure together. **Well-posed** where some row is a pressure
row; with none, the flux rows (which then cover every opening) sum to zero, so one is dropped for the gauge's reference
row (the reference's pressure set to zero): what the build checks before it builds (:func:`well_posed`).

**Passivity** (CI check (a)): a law's per-stage block ``b`` has a positive semidefinite symmetric part, which keeps the
port system well posed and is invisible to the law's own identity (which books whatever the law does). A source
(a given flux or pressure: it supplies work) is exempt (:attr:`PortLaw.source`)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, override

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
    """A registered port law (the module's interface): its row kind, its parameters (SI, a row an opening), whether it
    stores energy, whether it is a source (exempt from passivity), its class in the ledger, and its numerical terms'
    orders (``mime.testing.term_orders``)."""

    name: ClassVar[str]
    version: ClassVar[str]
    row: ClassVar[RowKind]
    parameters: ClassVar[tuple[str, ...]]
    stores: ClassVar[bool] = False
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
        """The stage's row: ``(a, b)`` with ``p = a + b q`` for a pressure row, ``(q, 0)`` for a flux row (``q``
        outward), at the stage's instant ``time``."""
        raise NotImplementedError

    def advanced(self, parameters: Array, state: Array, flux: Array, pressure: Array, dt: Any) -> Array:
        """The law's own state after the stage, given its solved flux and pressure."""
        return state

    def books(self, parameters: Array, state: Array, ended: Array, flux: Array, pressure: Array, dt: Any) -> LawBooks:
        """The law's own books over the stage: ``dt p q`` it received equals its storage's change, its loss and its
        numerical term."""
        zero = jnp.zeros((), dtype=flux.dtype)
        return LawBooks(zero, zero, zero, zero)

    def block(self, parameters: Array, dt: Any) -> Array:
        """The law's per-stage block ``b`` (Pa s/m^3), a ``1 x 1`` matrix for a one-opening law: what the passivity
        check reads."""
        return jnp.zeros((1, 1), dtype=parameters.dtype)


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

    @override
    def block(self, parameters: Array, dt: Any) -> Array:
        return jnp.reshape(parameters[0] + dt / self._denominator(parameters, dt), (1, 1))


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
    Windkessel.name: Windkessel(), Inflow.name: Inflow(), GivenPressure.name: GivenPressure()})


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


def port_system(rows: Sequence[RowKind], a: Array, b: Array, base: Array, response: Array, predicted: Array,
                reference: int) -> tuple[Array, Array]:
    """The openings' pressures (absolute, Pa) and fluxes (outward, m^3/s) solving each opening's row with the field's
    affine response: the fluxes ``q = base + R (x - x_hat)``, ``x`` the pressures relative to ``reference`` (the
    field's own unknowns, ``R`` a column for each opening but the reference) and ``x_hat`` those the step was solved
    at (``predicted``). Rows: a pressure row ``p_i - b_i q_i = a_i``, a flux row ``q_i = a_i``; with no pressure row the
    reference's row is the gauge's, ``p_r = 0``."""
    count = len(rows)
    others = [index for index in range(count) if index != reference]
    dtype = base.dtype
    gauge_row = all(row is RowKind.FLUX for row in rows)
    shift = jnp.stack([predicted[j] - predicted[reference] for j in others]) if others else jnp.zeros(0, dtype=dtype)
    predicted_flux = base - response @ shift
    matrix_rows: list[Array] = []
    right: list[Array] = []
    for i, row in enumerate(rows):
        if gauge_row and i == reference:
            matrix_rows.append(jnp.zeros(count, dtype=dtype).at[-1].set(1.0))
            right.append(jnp.zeros((), dtype=dtype))
        elif row is RowKind.PRESSURE:
            coefficients = -b[i] * response[i]
            if i != reference:
                coefficients = coefficients.at[others.index(i)].add(1.0)
            matrix_rows.append(jnp.concatenate([coefficients, jnp.ones(1, dtype=dtype)]))
            right.append(a[i] + b[i] * predicted_flux[i])
        else:
            matrix_rows.append(jnp.concatenate([response[i], jnp.zeros(1, dtype=dtype)]))
            right.append(a[i] - predicted_flux[i])
    solved = jnp.linalg.solve(jnp.stack(matrix_rows), jnp.stack(right))
    relative, level = solved[:-1], solved[-1]
    pressures = jnp.stack([level + (relative[others.index(j)] if j != reference else 0.0) for j in range(count)])
    return pressures, predicted_flux + response @ relative


def passive(law: PortLaw, parameters: Array, dt: Any) -> Array:
    """Whether ``law``'s per-stage block at ``parameters`` has a positive semidefinite symmetric part, judged by
    ``definite_at_floor`` (traced: a boolean)."""
    return books.definite_at_floor(law.block(parameters, dt))


__all__ = ["PARAMETERS", "PORT_LAWS", "STATES", "SUM_ZERO", "GivenPressure", "Inflow", "LawBooks", "PortLaw", "RowKind",
           "Windkessel", "passive", "port_law_named", "port_system", "reference_of", "waveforms_cancel", "well_posed"]
