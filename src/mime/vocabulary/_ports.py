"""Port kinds and energy classes: how the energy ledger pairs each delivery a receiver takes, and the class a law is.

A **port kind** names a receiver's flow (an interface its state publishes: a body's twist) and, for each interface it
accepts, what a delivery into it is (:class:`DeliveryKind`): a load read at the step's start, a force the step solves
for, a flow handed in, or a delivered inertia. The receiver imposes its flow; each delivery imposes what its kind says
it is. A kind states no pairing of its own: the pairing is ``mime.core.ledger``'s one rule, keyed by the delivery kind.
Kinds are registered (:func:`register_port_kind`), open to labs; MIME's own are a rigid body's, a chain segment's
joints', a delivered inertia's, a flow-in's, a fluid's and a flow's openings'.

An **energy class** (:class:`EnergyClass`) is one per law, its deliveries and reactions together: dissipative,
potential or workless, each settled by an exact identity on the law alone in its family's suite. There is no source
class: whether a producer is outside the components is worked out from the graph, never declared. A dissipative law
whose operators are definite whatever their inputs says so (:attr:`Definiteness.CONSTRUCTION`), checked once in its
family's suite; one whose definiteness depends on the state (:attr:`Definiteness.STATE`) is checked at run time where
the ledger is attached."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from mime.units import CURRENT, DIMENSIONLESS, FORCE, LENGTH, MASS, TIME, Dimension
from mime.vocabulary._mime import (
    APPLIED_WRENCH,
    BACKGROUND_LOAD,
    CONFINEMENT,
    FLOW_BODY_FORCE,
    FLOW_EXCHANGE_X,
    FLOW_EXCHANGE_Y,
    FLOW_EXCHANGE_Z,
    FLOW_MOMENTUM,
    FLOW_OPENING_FLUX,
    FLOW_OPENING_IMPEDANCE,
    FLOW_OPENING_PRESSURE,
    FLOW_WAVEFORM,
    FRAME_ANGULAR_VELOCITIES,
    JOINT_VELOCITIES,
    LINK_INERTIA,
    LINK_MOMENTUM,
    LINK_WRENCH,
    RESISTANCE,
    TWIST,
)
from mime.vocabulary._vocabulary import ReadKind, TargetKind, lookup


class DeliveryKind(Enum):
    """What a delivery into a port is: all the pairing rule needs to know of it. The values are the keys of
    ``mime.core.ledger``'s pairing, which reads them as strings (``mime.core`` imports nothing above it)."""

    LOAD = "load"
    """A load read at the step's start: paired with the configuration's actual change over the step, after any
    projection its host makes (``x' - x``, not ``dt v'``)."""
    SOLVED = "solved"
    """A force the step solves for (a resistance, a contact, a stop, friction, a noise entering the solve): paired with
    the step's own solution."""
    FLOW_IN = "flow_in"
    """A flow handed in (a spinner, which receives its host frame's angular velocity): paired with the effort it
    returns."""
    INERTIA = "inertia"
    """A delivered inertia (a rotor on a dynamic arm, an added mass): paired by its change, ``-(1/2) v^T dM v``."""


class EnergyClass(Enum):
    """A law's class, its deliveries and reactions together, settled by an exact identity on the law alone in its
    family's suite: a dissipative law's operators positive semidefinite, a potential law's force the negative gradient
    of its family's energy, a workless law's exchange its exact power identity."""

    DISSIPATIVE = "dissipative"
    POTENTIAL = "potential"
    WORKLESS = "workless"


class Definiteness(Enum):
    """Where a dissipative law's operators are shown positive semidefinite: by its construction, whatever its inputs
    (a resistance formed ``L L^T``), checked once in its family's suite; or at the state it reads (a resistance near a
    wall), checked at run time at a stated floor where the ledger is attached."""

    CONSTRUCTION = "construction"
    STATE = "state"


#: Watts and joules, what a flow paired with a delivery gives over a unit of time and over a step.
POWER = FORCE * LENGTH / TIME
ENERGY = FORCE * LENGTH
#: The symbols the vocabulary's units are written in, each a dimension; a radian is a dimensionless count.
_SYMBOLS = {"1": DIMENSIONLESS, "rad": DIMENSIONLESS, "m": LENGTH, "kg": MASS, "s": TIME, "A": CURRENT, "N": FORCE,
            "T": MASS / (CURRENT * TIME**2), "Pa": FORCE / LENGTH**2, "J": ENERGY, "W": POWER}


def _dimension(written: str) -> Dimension:
    """The dimension of units as the vocabulary writes them: symbols separated by spaces, each ``/`` dividing by what
    follows it, each symbol with an optional integer power (``N s/m``, ``m/s^2``, ``kg m^2``)."""
    found = DIMENSIONLESS
    for word in written.split():
        for index, term in enumerate(word.split("/")):
            symbol, _, power = term.partition("^")
            if symbol not in _SYMBOLS or (power and not power.lstrip("-").isdigit()):
                raise ValueError(f"{written!r} names {term!r}, which is not one of {sorted(_SYMBOLS)} with a power")
            factor = _SYMBOLS[symbol] ** (int(power) if power else 1)
            found = found * factor if index == 0 else found / factor
    return found


def _parts(units: str) -> list[Dimension]:
    """The dimension of each part of a composite unit (``N; N m``)."""
    return [_dimension(part.strip()) for part in units.split(";")]


def _paired_dimensions(flow: list[Dimension], delivered: list[Dimension],
                       kind: DeliveryKind) -> list[tuple[Dimension, Dimension]]:
    """Each part of a delivery paired with the flow, and the dimension that product must have."""
    if kind is DeliveryKind.LOAD:
        if len(delivered) != len(flow):
            raise ValueError(f"a load has a part for each of the flow's {len(flow)}, not {len(delivered)}")
        return [(f * d, POWER) for f, d in zip(flow, delivered, strict=True)]
    # An operator's parts are its blocks, upper triangle row by row ("N s/m; N s; N m s"), each against two parts of
    # the flow (one part for every block alike); an inertia is paired by its change, so its product is an energy.
    blocks = [(i, j) for i in range(len(flow)) for j in range(i, len(flow))]
    wanted = ENERGY if kind is DeliveryKind.INERTIA else POWER
    if len(delivered) == 1:
        return [(flow[i] * flow[j] * delivered[0], wanted) for i, j in blocks]
    if len(delivered) != len(blocks):
        raise ValueError(f"an operator has a part for each of the flow's {len(blocks)} blocks, or one, not "
                         f"{len(delivered)}")
    return [(flow[i] * flow[j] * d, wanted) for (i, j), d in zip(blocks, delivered, strict=True)]


@dataclass(frozen=True)
class PortKind:
    """A receiver's port: its ``flow`` (a state it publishes) and, for each interface it accepts, what a delivery into
    it is. Every interface named is a vocabulary entry, and each delivery's units paired with the flow's give watts
    part by part (an operator's blocks against the flow twice, an inertia's change a joule): ``N; N m`` against ``m/s;
    rad/s``. An inequality constraint (a confinement) is held in the receiver's step, not delivered as a value, so its
    units pair with nothing; a delivery in ``mapped`` reaches the flow through a map of the receiver's own (a load on
    a chain's frames, which its Jacobian turns into a generalised force against its joint rates; a drive's waveform,
    which a flow evaluates at its own stage instants into the body force it applies), so its units pair with the flow
    only after that map, which is the receiver's."""

    name: str
    flow: str
    deliveries: Mapping[str, DeliveryKind] = field(default_factory=lambda: MappingProxyType({}))
    description: str = ""
    mapped: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError(f"port kind {self.name!r}: " + "; ".join(problems))

    def problems(self) -> list[str]:
        """What is wrong with the kind, one line each: an interface the vocabulary does not hold, a flow that is not a
        state, or a delivery whose units do not pair with the flow's to watts."""
        flow = lookup(self.flow)
        unknown = [name for name in (self.flow, *self.deliveries) if lookup(name) is None]
        if unknown:
            return [f"names interfaces the vocabulary does not hold: {unknown}"]
        assert flow is not None
        found = [] if flow.kind is ReadKind.STATE else [f"its flow {self.flow!r} is not a state a receiver publishes"]
        found += [f"maps {name!r}, which it does not accept" for name in sorted(self.mapped - set(self.deliveries))]
        try:
            flow_parts = _parts(flow.units)
        except ValueError as error:
            return [*found, f"its flow's units {flow.units!r}: {error}"]
        for name, kind in self.deliveries.items():
            interface = lookup(name)
            assert interface is not None
            if interface.kind is TargetKind.INEQUALITY_CONSTRAINT or name in self.mapped:
                continue
            try:
                pairs = _paired_dimensions(flow_parts, _parts(interface.units), kind)
            except ValueError as error:
                found.append(f"{name}'s units {interface.units!r} do not pair with the flow's {flow.units!r}: {error}")
                continue
            if any(product != wanted for product, wanted in pairs):
                found.append(f"{name} ({interface.units}) paired with {self.flow} ({flow.units}) as a {kind.value} "
                             f"does not give {'joules' if kind is DeliveryKind.INERTIA else 'watts'}")
        return found


_port_kinds: dict[str, PortKind] = {}


def register_port_kind(kind: PortKind) -> PortKind:
    """Register a port kind. The same kind twice changes nothing; a different kind under a name already held is
    refused."""
    held = _port_kinds.get(kind.name)
    if held is not None and held != kind:
        raise ValueError(f"port kind {kind.name!r} is already registered, differently: {held}")
    _port_kinds[kind.name] = kind
    return kind


def port_kind(name: str) -> PortKind | None:
    """The port kind registered as ``name``, or ``None``."""
    return _port_kinds.get(name)


def port_kinds() -> list[str]:
    """The names of every registered port kind, sorted."""
    return sorted(_port_kinds)


RIGID_BODY = register_port_kind(PortKind(
    "rigid_body", TWIST.name, MappingProxyType({
        APPLIED_WRENCH.name: DeliveryKind.LOAD, BACKGROUND_LOAD.name: DeliveryKind.LOAD,
        RESISTANCE.name: DeliveryKind.SOLVED, CONFINEMENT.name: DeliveryKind.SOLVED}),
    "a rigid body's twist about its centre of mass in the world, against the wrenches on it: the loads read at the "
    "step's start, the resistance and a confinement solved for in its step"))
JOINT = register_port_kind(PortKind(
    "joint", JOINT_VELOCITIES.name, MappingProxyType({
        LINK_WRENCH.name: DeliveryKind.LOAD, LINK_MOMENTUM.name: DeliveryKind.FLOW_IN,
        LINK_INERTIA.name: DeliveryKind.INERTIA}),
    "a chain segment's joint rates against the generalised forces of the loads on its frames, read at the step's "
    "start and paired with its joints' change; the momenta of the spinners its frames carry, whose gyroscopic torque "
    "it returns at the rates it applies it at; and the inertia they carry, paired by its change",
    mapped=frozenset({LINK_WRENCH.name})))
SPINNER = register_port_kind(PortKind(
    "spinner", FRAME_ANGULAR_VELOCITIES.name, MappingProxyType({LINK_WRENCH.name: DeliveryKind.LOAD}),
    "a rotor carried by its absolute spin about an axis fixed in its host: the loads on its frames, read at the step's "
    "start, of which the torque on the rotor about its axis is paired with the rotor's turn, and the rest passes to "
    "its host, which pairs it", mapped=frozenset({LINK_WRENCH.name})))
DELIVERED_INERTIA = register_port_kind(PortKind(
    "delivered_inertia", FRAME_ANGULAR_VELOCITIES.name, MappingProxyType({LINK_INERTIA.name: DeliveryKind.INERTIA}),
    "an inertia a frame carries beyond its own, paired by its change"))
FLOW_IN = register_port_kind(PortKind(
    "flow_in", FRAME_ANGULAR_VELOCITIES.name, MappingProxyType({LINK_MOMENTUM.name: DeliveryKind.FLOW_IN}),
    "a spinner's momentum on its host frame, whose angular velocity it receives: paired with the gyroscopic torque "
    "it returns"))
FLUID = register_port_kind(PortKind(
    "fluid", FLOW_MOMENTUM.name, MappingProxyType({
        FLOW_BODY_FORCE.name: DeliveryKind.LOAD, FLOW_WAVEFORM.name: DeliveryKind.LOAD,
        **{exchange.name: DeliveryKind.LOAD for exchange in (FLOW_EXCHANGE_X, FLOW_EXCHANGE_Y, FLOW_EXCHANGE_Z)}}),
    "a medium's fluid's momentum against the uniform body force per unit mass the laws driving it deliver, each "
    "law's paired by the flow's step with its fluid's momentum at the instants it applies it; a drive's waveform "
    "reaches the momentum through the flow's own evaluation of it at its stage instants; an exchange law's force "
    "density on each face set reaches it through the flow's own pairing of it with the velocity there at the step's "
    "solution",
    mapped=frozenset({FLOW_WAVEFORM.name, FLOW_EXCHANGE_X.name, FLOW_EXCHANGE_Y.name, FLOW_EXCHANGE_Z.name})))
OPENING = register_port_kind(PortKind(
    "opening", FLOW_OPENING_FLUX.name, MappingProxyType({
        FLOW_OPENING_PRESSURE.name: DeliveryKind.LOAD, FLOW_OPENING_IMPEDANCE.name: DeliveryKind.SOLVED}),
    "a medium's flow's openings, each its outward volume flux (a row per opening) against its pressure: a law at an "
    "opening states a pressure row, its pressure affine in its flux over the stage (the offset a load, the block solved "
    "with the flow), or a flux row, its flux given and paired with the pressure the flow returns; the laws are held in "
    "the flow's step"))

__all__ = ["DELIVERED_INERTIA", "ENERGY", "FLOW_IN", "FLUID", "JOINT", "OPENING", "POWER", "RIGID_BODY", "SPINNER", "Definiteness",
           "DeliveryKind",
           "EnergyClass", "PortKind", "port_kind", "port_kinds", "register_port_kind"]
