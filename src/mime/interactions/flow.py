"""Laws that drive a medium's flow: ``BodyForce``, the body force that stands in for a pressure drop on a periodic box
(F0's B6 and B7: drives are laws, a body force bound to the medium with no surface); and the laws held at a flow's
openings (decisions 7 and 8; A2's hybrid rows): ``Inflow``, a flux given at an opening; ``Windkessel``, the RCR
whose pressure is affine in the opening's flux over each stage; ``Valve``, a closed valve (no flux); and
``SharedDistal``, two openings draining through their own resistances into one shared distal resistance, a law over
several openings (the record's "F2's rows, 2", item 1); and ``Darcy``, a porous structure's drag on the fluid in it. A law at an opening binds the medium and the opening's surface
(``exp.attach(Windkessel(...), medium=blood, surface=tube.end)``); a law over several binds them all, as a tuple or a
region's openings (``surface=(tree.end_0, tree.end_1)``, ``surface=tube.lumen``), and a law given another count of
openings than it binds is refused. Each builds no node: the flow holds it in its own step, through its
registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, override

from mime.gradients import GradientStatus
from mime.nodes.flow._body_force import BodyForceNode
from mime.nodes.flow.fvm._ports import Inflow as InflowLaw
from mime.nodes.flow.fvm._ports import SharedDistal as SharedDistalLaw
from mime.nodes.flow.fvm._ports import Valve as ValveLaw
from mime.nodes.flow.fvm._ports import Windkessel as WindkesselLaw
from mime.registry import (
    DESIGN,
    PHYSICAL,
    BuildContext,
    Delivery,
    GradientRecord,
    Interaction,
    Plan,
    PlanContext,
    Registration,
    RoleKind,
    SupportRow,
    Value,
    convention,
    register_interaction,
    requires,
)
from mime.units import Acceleration, Compliance, Frequency, HydraulicResistance, Quantity, VolumeFlux
from mime.vocabulary import FLOW_BODY_FORCE, FLOW_OPENING, FLOW_POROUS, FLOW_WAVEFORM, Definiteness, EnergyClass

if TYPE_CHECKING:
    from mime.build import InteractionBuilder


@register_interaction
@dataclass(frozen=True, kw_only=True)
class BodyForce(Interaction):
    """A uniform body force per unit mass on a medium's fluid, along a direction: ``steady``, plus ``amplitude cos(2 pi
    frequency t + phase)`` for a pulsatile drive, ``exp.attach(BodyForce(along=(0, 0, 1), steady=m_per_s2(8)),
    medium=water)``. The flow field applies it to its fluid alone, never its solid, so the flux it drives does not
    depend on how much solid the field's domain holds. The waveform is the drive's; the field kind is what responds."""

    name = "body_force"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    references = ()
    family = None
    builds = {"": (BodyForceNode,)}
    roles = {"medium": RoleKind.MEDIUM}
    support = (SupportRow(None, {"medium": requires(accepts=[FLOW_BODY_FORCE.name])},
                          note="a medium whose flow field takes a body force"),
               SupportRow(None, {"medium": requires(accepts=[FLOW_WAVEFORM.name])},
                          note="a medium whose flow field takes the drive's waveform (RK3's stages, the prescribed "
                               "flow)"))
    delivers = {"force": Delivery("medium", FLOW_BODY_FORCE.name),
                "waveform": Delivery("medium", FLOW_WAVEFORM.name)}
    unmetered = ("a drive declares no class, since a source has none: its own node reads no component's state and it "
                 "delivers loads, so where the flow it drives is metered the build counts it outside the components "
                 "and books its work, paired by the flow with the fluid's momentum, as supplied under its own name; "
                 "a flow the ledger does not meter (the prescribed flow) pairs it with nothing")
    setting_roles = {
        "along": DESIGN, "steady": PHYSICAL,
        "amplitude": convention(PHYSICAL, "a steady drive unless a pulsatile one is declared"),
        "frequency": convention(PHYSICAL, "a steady drive unless a pulsatile one is declared"),
        "phase": convention(PHYSICAL, "the pulsatile part starts at its peak (radians)"),
    }

    along: tuple[float, float, float]
    steady: Annotated[Quantity, Acceleration]
    amplitude: Annotated[Quantity | None, Acceleration] = None
    frequency: Annotated[Quantity | None, Frequency] = None
    phase: float = 0.0

    @override
    def setting_problems(self, settings) -> list[str]:  # type: ignore[no-untyped-def]
        found: list[str] = []
        length = sum(float(c) ** 2 for c in self.along) ** 0.5
        if abs(length - 1.0) > 1e-9:
            found.append(f"along is a unit direction, and {self.along} has length {length:.6g}")
        if (self.amplitude is None) != (self.frequency is None):
            found.append("a pulsatile drive takes both its amplitude and its frequency")
        return found

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        """The force a step and the waveform, each where the medium's field takes it."""
        record = ctx.capabilities.get("medium")
        accepted: set[str] = set(record.accepts) if record is not None else set()
        return Plan(delivers={label: delivery for label, delivery in self.delivers.items()
                              if delivery.target in accepted})

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        node = b.add_response_node(BodyForceNode(
            ctx.node_name(), ctx.timestep, along=self.along, steady=ctx.scalar("steady"),
            amplitude=ctx.scalar("amplitude") if self.amplitude is not None else 0.0,
            frequency=ctx.scalar("frequency") if self.frequency is not None else 0.0,
            phase=self.phase))
        for label in ("force", "waveform"):
            if b.delivers(label):
                b.connect(node[label], into=b.target(label))



#: Where a law at an opening is not yet in a gradient gate (FB15, the gradients with ports, is a later PR's).
_PORT_GRADIENT = GradientRecord(
    where="in its settings, through the flow's step that holds it",
    reason="the gradient gate with ports (its settings, the unit fields and the port system) is not built yet",
    route="F2b's gradient gate with ports: reverse and forward against a central difference that rebuilds")


class _OpeningLaw(Interaction):
    """What a law at an opening shares: its roles (the medium whose flow holds it, the openings' surfaces), how many
    openings it binds (``openings``, its port law's), its id (its name and the openings: ``windkessel:tube.end``,
    ``shared_distal:tree.end_0+tree.end_1``), and its plan, a registration with the medium's flow, which holds it in
    its step as the port law ``port_law`` (``mime.nodes.flow.fvm._ports``)."""

    port_law: ClassVar[str] = ""
    #: How many openings it binds: ``surface=`` names that many (one surface, or a tuple or a region's openings).
    openings: ClassVar[int] = 1
    family = None
    builds = {"": ()}
    roles = {"medium": RoleKind.MEDIUM, "surface": RoleKind.SURFACE}
    id_roles = ("surface",)
    support = (SupportRow(None, {"medium": requires(accepts=[FLOW_OPENING.name])},
                          note="a medium whose flow field holds a law at each of its openings"),)

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=(), registrations=(Registration("medium", FLOW_OPENING.name, self.port_law, "surface"),))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        return None


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Inflow(_OpeningLaw):
    """A volume flux into the flow at an opening, ``steady + amplitude cos(2 pi frequency t + phase)``:
    ``exp.attach(Inflow(steady=mL_per_s(1.0)), medium=blood, surface=tube.start)``. The flow holds the opening's flux
    to it at each step's end (its pressure the unknown that pins it: decision 7's evolved face, a flux row): the
    opening's whole net flux, its lumen's, its band's and its solid's faces together, never the lumen's alone. So the
    lumen carries the inflow less what the solid carries, which runs either way: a few per cent of it with the inflow
    in steady flow at D/h 8; in pulsed flow, where the drive's frequency nears the stiff interior's damping rate, a
    swing with the drive of up to about 30 % either way, the lumen then carrying up to about 1.3 times the inflow (the
    FVM's validity gives the figures). A source: the work it supplies, the pressure the flow returns times its flux, is
    booked under its name."""

    name = "inflow"
    version = "0.1.0"
    port_law = InflowLaw.name
    differentiability = GradientStatus.UNVERIFIED
    gradient = _PORT_GRADIENT
    references = ()
    unmetered = ("a flux given at an opening is a source, which declares no class: the flow holds it in its step and "
                 "books its work, the pressure the flow returns times the flux it gives, under its name")
    setting_roles = {
        "steady": PHYSICAL,
        "amplitude": convention(PHYSICAL, "a steady inflow unless a pulsatile one is declared"),
        "frequency": convention(PHYSICAL, "a steady inflow unless a pulsatile one is declared"),
        "phase": convention(PHYSICAL, "the pulsatile part starts at its peak (radians)"),
    }

    steady: Annotated[Quantity, VolumeFlux]
    amplitude: Annotated[Quantity | None, VolumeFlux] = None
    frequency: Annotated[Quantity | None, Frequency] = None
    phase: float = 0.0

    @override
    def setting_problems(self, settings) -> list[str]:  # type: ignore[no-untyped-def]
        if (self.amplitude is None) != (self.frequency is None):
            return ["a pulsatile inflow takes both its amplitude and its frequency"]
        return []


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Windkessel(_OpeningLaw):
    """The three-element (RCR) Windkessel at an opening: a ``proximal`` resistance, then the ``compliance`` of a
    capacitor drained by a ``distal`` resistance to a distal pressure of zero: ``exp.attach(Windkessel(proximal=...,
    compliance=..., distal=...), medium=blood, surface=tube.end)``. The flow steps it by backward Euler at its own
    instant (decision 8), so the exchange across the opening does no work of its own; its numerical dissipation
    ``N_WK = (C/2)(Pc' - Pc)^2``, first order, is booked under its name with its capacitor's storage and its
    resistances' loss. Dissipative: its per-stage block ``R_p + dt / (C + dt / R_d)`` is positive for positive
    settings (the port-law suite's passivity check)."""

    name = "windkessel"
    version = "0.1.0"
    port_law = WindkesselLaw.name
    differentiability = GradientStatus.UNVERIFIED
    gradient = _PORT_GRADIENT
    references = ("Westerhof2009",)
    energy_class = EnergyClass.DISSIPATIVE
    setting_roles = {"proximal": PHYSICAL, "compliance": PHYSICAL, "distal": PHYSICAL}

    proximal: Annotated[Quantity, HydraulicResistance]
    compliance: Annotated[Quantity, Compliance]
    distal: Annotated[Quantity, HydraulicResistance]

    @override
    def definiteness(self) -> Definiteness | None:
        return Definiteness.CONSTRUCTION

    @override
    def setting_problems(self, settings) -> list[str]:  # type: ignore[no-untyped-def]
        found: list[str] = []
        for name in ("proximal", "compliance", "distal"):
            value = settings[name]
            if not (isinstance(value, float) and value > 0.0):
                found.append(f"{name} is {value!r}, not a value above zero")
        return found


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Valve(_OpeningLaw):
    """A closed valve at an opening: no flux through it, ``exp.attach(Valve(), medium=blood, surface=tree.end_1)``. The
    flow holds its opening's net flux at zero (a flux row, q = 0), not each face's, and solves the pressure behind it;
    workless, since that pressure does no work on a flux of zero. The lumen behind it still carries out what the solid's
    faces on the opening bring back in, the solid's Darcy flow (a few per cent of the vessel's flow at the default stiff
    interior, a hundred times less at ``Band(delta_in=0.01)``: the FVM's validity gives the figures). A valve that opens
    and closes with the pressure across it is not built (decision 8's small Newton)."""

    name = "valve"
    version = "0.1.0"
    port_law = ValveLaw.name
    differentiability = GradientStatus.UNVERIFIED
    gradient = _PORT_GRADIENT
    references = ()
    energy_class = EnergyClass.WORKLESS
    setting_roles = {}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class SharedDistal(_OpeningLaw):
    """Two openings each through its own resistance, ``first`` and ``second`` (in the order ``surface=`` names them),
    into one shared ``distal`` resistance to a distal pressure of zero: ``exp.attach(SharedDistal(first=...,
    second=..., distal=...), medium=blood, surface=(tree.end_0, tree.end_1))``. Its pressures, ``p_k = R_k q_k + R_d
    (q_1 + q_2)``, couple the two openings through the shared resistance (a block with off-diagonals, which the flow
    solves whole each step). Dissipative: its block is symmetric and positive definite for positive settings."""

    name = "shared_distal"
    version = "0.1.0"
    port_law = SharedDistalLaw.name
    openings = SharedDistalLaw.openings
    differentiability = GradientStatus.UNVERIFIED
    gradient = _PORT_GRADIENT
    references = ()
    energy_class = EnergyClass.DISSIPATIVE
    setting_roles = {"first": PHYSICAL, "second": PHYSICAL, "distal": PHYSICAL}

    first: Annotated[Quantity, HydraulicResistance]
    second: Annotated[Quantity, HydraulicResistance]
    distal: Annotated[Quantity, HydraulicResistance]

    @override
    def definiteness(self) -> Definiteness | None:
        return Definiteness.CONSTRUCTION

    @override
    def setting_problems(self, settings) -> list[str]:  # type: ignore[no-untyped-def]
        found: list[str] = []
        for name in ("first", "second", "distal"):
            value = settings[name]
            if not (isinstance(value, float) and value > 0.0):
                found.append(f"{name} is {value!r}, not a value above zero")
        return found


__all__ = ["BodyForce", "Inflow", "SharedDistal", "Valve", "Windkessel"]


#: Where Darcy's law is not in a gradient gate in full: its geometry's derivative carries the band's ripple.
_DARCY_GRADIENT = GradientRecord(
    where="in the structure's geometry (its centre, axis, radius and length), through the flow's step that holds it",
    reason="a geometry derivative through the band carries the band's ripple within a cell (the flow's own geometry "
           "line), and the region's exact distance and its share of the fluid have kinks (its medial set, its axis, "
           "where its band meets the wall's fluid); in the permeability the gate checks it exact",
    route="the flow's geometry derivative through a traced spacing or an offset indicator (the wall's own route)")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Darcy(Interaction):
    """The drag of a porous structure on the fluid that permeates it, Darcy's law in Brinkman's form, a resistivity
    ``mu / K`` on the fluid inside the structure's region: ``exp.attach(Darcy(), medium=blood, structure=clot)``. The
    structure's geometry kind gives the region (a ``Plug``), its ``permeability`` value the isotropic ``K`` (m^2),
    declared with its reason (``exp.declare("clot.permeability", um2(0.01), reason=...)``: no OpenMaterica record
    holds a permeability yet). It builds no node: the medium's flow holds it in its own step, its resistivity added to
    the wall's, on its share of the fluid where it meets the wall's band; its loss is booked under its own id
    (``darcy:clot``). Static: the region and its permeability are the run's, sampled each step from its leaves."""

    name = "darcy"
    version = "0.1.0"
    differentiability = GradientStatus.UNVERIFIED
    gradient = _DARCY_GRADIENT
    references = ("Darcy1856", "Brinkman1947")
    family = None
    builds = {"": ()}
    roles = {"medium": RoleKind.MEDIUM, "structure": RoleKind.STRUCTURE}
    id_roles = ("structure",)
    energy_class = EnergyClass.DISSIPATIVE
    support = (SupportRow(None, {"medium": requires(accepts=[FLOW_POROUS.name])},
                          note="a medium whose flow field holds a porous region in its step"),)
    values = {"permeability": Value("structure", "permeability")}
    setting_roles = {}

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=(), values=dict(self.values),
                    registrations=(Registration("medium", FLOW_POROUS.name, self.name, "structure"),))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        return None
