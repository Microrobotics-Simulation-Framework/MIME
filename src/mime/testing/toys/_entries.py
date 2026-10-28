"""The toy entries, registered under the ``toy`` namespace. They carry no physics: each exists to exercise one part of
the mapping from declarations to a graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any, override

from mime.build import (
    DynamicsBuilder,
    InteractionBuilder,
    ObservationBuilder,
    ProviderBuilder,
    ResolvedRegistration,
    Slot,
)
from mime.gradients import GradientStatus
from mime.registry import (
    BuildContext,
    Delivery,
    Dynamics,
    FieldProvider,
    GroupRequest,
    Interaction,
    Observation,
    Plan,
    PlanContext,
    Port,
    Read,
    Registration,
    RoleKind,
    SupportRow,
    register_dynamics,
    register_field_provider,
    register_interaction,
    register_observation,
    requires,
)
from mime.testing.toys._nodes import (
    ResponseNode,
    ToyAnchoredBodyNode,
    ToyBodyNode,
    ToyDragNode,
    ToyFieldNode,
    ToyHoldNode,
    ToyLiftNode,
    ToyMediumNode,
    ToyRecorderNode,
    ToyRegionProbeNode,
    ToySpringNode,
)
from mime.things import ThingKind
from mime.units import LENGTH, VELOCITY, Force, Frequency, Hz, Length, Mass, N, Quantity, Time, Velocity, kg, m, m_per_s
from mime.vocabulary import Capability, Interface, ReadKind, TargetKind, define, register_namespace

NAMESPACE, VERSION = "toy", "0.1.0"
register_namespace(NAMESPACE, VERSION)

POSITION = define(Interface("toy.body.position", ReadKind.STATE, "m", "world", (3,), magnitude=1.0,
                            description="a toy body's position"))
BODY_VELOCITY = define(Interface("toy.body.velocity", ReadKind.STATE, "m/s", "world", (3,), magnitude=1.0,
                             description="a toy body's velocity"))
LOAD = define(Interface("toy.body.load", TargetKind.LOAD, "N", "world", (3,), sign="on the body", magnitude=1.0,
                        description="the summed load on a toy body"))
HOLD_Z = define(Interface("toy.body.hold_z", TargetKind.EQUALITY_CONSTRAINT, "m/s", "world", (),
                          sign="the body's prescribed velocity along z", magnitude=1.0,
                          description="a prescribed velocity along z, one degree of freedom"))
TORQUE = define(Interface("toy.body.torque", TargetKind.LOAD, "N m", "world", None, sign="on the body",
                          reserved=True, description="reserved, to exercise the refusal of a reserved name"))
MEDIUM_VALUE = define(Interface("toy.medium.value", ReadKind.STATE, "1", None, (3,), magnitude=1.0,
                                description="a toy medium's relaxing value"))
MEDIUM_REGISTRATION = define(Interface("toy.medium.registration", TargetKind.REGISTRATION, "", None, (),
                                       description="a body registered with a toy medium's update"))
PUSH = define(Interface("toy.field.push", ReadKind.FIELD, "1", "world", (3,), magnitude=1.0,
                        description="a toy field, evaluated at points"))

COUPLED_BODY = "toy.coupled_body"


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ToyBody(Dynamics):
    """A point mass that sums loads and takes an exclusive velocity along z."""

    name = "toy.body"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyBodyNode,)}
    applies_to = frozenset({ThingKind.BODY})
    flags = frozenset({Capability.INERTIAL})
    publishes = {POSITION.name: Port("x"), BODY_VELOCITY.name: Port("v")}
    accepts = {LOAD.name: Port("load"), HOLD_Z.name: Port("hold_z")}
    initial_state = {"position": LENGTH, "velocity": VELOCITY}
    mass: Annotated[Quantity, Mass] = kg(1.3)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        b.add_node(ToyBodyNode(ctx.node_name(), ctx.timestep, mass=ctx.scalar("mass"),
                               initial_position=ctx.initial["position"], initial_velocity=ctx.initial["velocity"]))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ToyAnchoredBody(Dynamics):
    """A body that does not move: it claims no flags, so an entry that needs one refuses it."""

    name = "toy.anchored_body"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyAnchoredBodyNode,)}
    applies_to = frozenset({ThingKind.BODY})
    publishes = {POSITION.name: Port("x")}
    accepts = {LOAD.name: Port("load")}
    initial_state = {"position": LENGTH}

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        b.add_node(ToyAnchoredBodyNode(ctx.node_name(), ctx.timestep, initial_position=ctx.initial["position"]))


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class ToyRelaxation(Dynamics):
    """A medium's relaxing value, which takes bodies by registration: it reads each one's position and delivers a
    load into it, under the registering interaction's id."""

    name = "toy.relaxation"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyMediumNode,)}
    applies_to = frozenset({ThingKind.MEDIUM})
    publishes = {MEDIUM_VALUE.name: Port("u")}
    accepts = {MEDIUM_REGISTRATION.name: Port("registrations")}
    registration_kinds = frozenset({COUPLED_BODY})
    relax_rate: Annotated[Quantity, Frequency] = Hz(2.1)
    coupling: Annotated[Quantity, Length**-1] = 1.9 / m(1.0)
    drag_gain: Annotated[Quantity, Force] = N(0.61)
    value: tuple[float, float, float] = (0.1, 0.2, -0.1)

    @override
    def build(self, ctx: BuildContext, b: DynamicsBuilder, registrations: Sequence[ResolvedRegistration]) -> None:
        node = b.add_node(ToyMediumNode(ctx.node_name(), ctx.timestep, bodies=[r.subject for r in registrations],
                                        relax_rate=ctx.scalar("relax_rate"), coupling=ctx.scalar("coupling"),
                                        drag_gain=ctx.scalar("drag_gain"), initial_value=self.value))
        for registration in registrations:
            b.connect(b.read(registration, "position"), into=node[f"pos_{registration.subject}"])
            b.connect(node[f"force_{registration.subject}"], into=b.target(registration, "load"))


@register_field_provider
@dataclass(frozen=True, kw_only=True)
class ToyField(FieldProvider):
    """An evaluated field with a reader slot per reader."""

    name = "toy.field"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyFieldNode,)}
    flags = frozenset({Capability.PROVISION_EVALUATED})
    publishes = {PUSH.name: "push"}
    amplitude: float = 0.4
    wavenumber: Annotated[Quantity, Length**-1] = 1.3 / m(1.0)

    @override
    def build(self, ctx: BuildContext, b: ProviderBuilder, slots: Sequence[Slot]) -> None:
        b.add_node(ToyFieldNode(ctx.node_name(), ctx.timestep, slots=[slot.name for slot in slots],
                                amplitude=self.amplitude, wavenumber=ctx.scalar("wavenumber")))


def _respond(b: InteractionBuilder, node: ResponseNode[Any, Any], read: str, delivered: str) -> None:
    response = b.add_response_node(node)
    b.connect(b.read(read), into=response[node.READS])
    b.connect(response[node.OUTPUT], into=b.target(delivered))


def _together(entry: ToyLift, roles: tuple[str, ...], reason: str) -> tuple[GroupRequest, ...]:
    if not entry.solve_together:
        return ()
    return (GroupRequest(roles, reason, entry.tolerance, entry.max_iterations, own_nodes=True),)


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ToySpring(Interaction):
    """A load pulling the body towards ``x0``."""

    name = "toy.spring"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToySpringNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"x": Read("body", POSITION.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    k: Annotated[Quantity, Force / Length] = N(1.7) / m(1.0)
    x0: Annotated[Quantity, Length] = m((0.05, -0.02, 0.01))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        _respond(b, ToySpringNode(ctx.node_name(), ctx.timestep, k=ctx.scalar("k"), x0=ctx.vector("x0")), "x",
                 "load")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ToyDrag(Interaction):
    """A load against the body's velocity, with two rungs and no default: ``-c v`` (linear) and ``-c v |v| / v1``
    (quadratic, with ``v1`` = 1 m/s, so ``c`` has one dimension in both). The quadratic rung needs an inertial body;
    the linear one takes any body. ``ToyDrag.Quadratic(...)`` is ``ToyDrag(rung="quadratic", ...)``.

    The toy family's member (``mime.testing.toys.TOY_FAMILY``). Version 0.2.0: the quadratic rung takes the
    velocity's norm, so turning the world turns the load; 0.1.0 took each component's magnitude."""

    name = "toy.drag"
    version = "0.2.0"
    differentiability = GradientStatus.EXACT
    family = "toy"
    builds = {"linear": (ToyDragNode,), "quadratic": (ToyDragNode,)}
    roles = {"body": RoleKind.BODY}
    rungs = ("linear", "quadratic")
    support = (SupportRow("quadratic", {"body": requires(Capability.INERTIAL, accepts=[LOAD.name])}),
               SupportRow("linear", {"body": requires(accepts=[LOAD.name])}))
    reads = {"v": Read("body", BODY_VELOCITY.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    rung: str | None = None
    c: Annotated[Quantity, Mass / Time] = N(0.37) / m_per_s(1.0)

    @classmethod
    def Quadratic(cls, **settings: Any) -> ToyDrag:  # noqa: N802 - a rung's alias reads as a class
        return cls(rung="quadratic", **settings)

    @classmethod
    def Linear(cls, **settings: Any) -> ToyDrag:  # noqa: N802
        return cls(rung="linear", **settings)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        # Phase 1 refuses a drag without one of its rungs, so one is set by the time it builds.
        _respond(b, ToyDragNode(ctx.node_name(), ctx.timestep, c=ctx.scalar("c"), rung=str(self.rung)), "v", "load")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ToyLift(Interaction):
    """A load on the body from the medium's value. With ``solve_together`` it asks to be solved with both."""

    name = "toy.lift"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyLiftNode,)}
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    reads = {"u": Read("medium", MEDIUM_VALUE.name)}
    delivers = {"load": Delivery("body", LOAD.name)}
    a: Annotated[Quantity, Force] = N(0.83)
    w: tuple[float, float, float] = (1.0, -0.7, 0.45)
    solve_together: bool = False
    tolerance: float = 1e-12
    max_iterations: int = 50

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(reads=dict(self.reads), delivers=dict(self.delivers),
                    groups=_together(self, ("medium", "body"), "the lift closes a loop through the medium"))

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        _respond(b, ToyLiftNode(ctx.node_name(), ctx.timestep, a=ctx.scalar("a"), w=self.w), "u", "load")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ToyCouple(Interaction):
    """The two-way coupling of a body and a medium, by registration: the medium's own dynamics reads the body and
    loads it, so this entry builds no node. With ``solve_together`` it asks for the pair to be solved together."""

    name = "toy.couple"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": ()}
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    solve_together: bool = False
    tolerance: float = 1e-12
    max_iterations: int = 50

    @override
    def plan(self, ctx: PlanContext) -> Plan:
        registration = Registration("medium", MEDIUM_REGISTRATION.name, COUPLED_BODY, "body",
                                    reads={"position": POSITION.name}, returns={"load": LOAD.name})
        groups = () if not self.solve_together else (
            GroupRequest(("medium", "body"), "a two-way coupling", self.tolerance, self.max_iterations),)
        return Plan(nodes=(), registrations=(registration,), groups=groups)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        return None


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ToyHold(Interaction):
    """Prescribes the body's velocity along z from a field read at its position: an exclusive target, read through
    a provider's reader slot."""

    name = "toy.hold"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyHoldNode,)}
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    reads = {"push": Read("field", PUSH.name, at=Read("body", POSITION.name))}
    delivers = {"hold_z": Delivery("body", HOLD_Z.name)}
    gain: Annotated[Quantity, Velocity] = m_per_s(0.5)

    @override
    def build(self, ctx: BuildContext, b: InteractionBuilder) -> None:
        _respond(b, ToyHoldNode(ctx.node_name(), ctx.timestep, gain=ctx.scalar("gain")), "push", "hold_z")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class ToyTwist(Interaction):
    """Delivers into a reserved name, so that its use is refused."""

    name = "toy.twist"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": ()}
    roles = {"body": RoleKind.BODY}
    delivers = {"torque": Delivery("body", TORQUE.name)}


@register_observation
@dataclass(frozen=True, kw_only=True)
class ToyRecorder(Observation):
    """Records the body's position, with no edge into the physics."""

    name = "toy.recorder"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyRecorderNode,)}
    roles = {"body": RoleKind.BODY}
    reads = {"position": Read("body", POSITION.name)}
    body: Any = None

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        node = b.add_node(ToyRecorderNode(ctx.node_name(), ctx.timestep))
        b.connect(b.read("position"), into=node["position"])


@register_observation
@dataclass(frozen=True, kw_only=True)
class ToyRegionProbe(Observation):
    """Evaluates the region a medium fills at a toy body's position: its signed distance, outward normal and nearest
    point, as a node evaluates a registered shape, so that a step evaluates every shape a geometry kind names."""

    name = "toy.region_probe"
    version = VERSION
    differentiability = GradientStatus.EXACT
    builds = {"": (ToyRegionProbeNode,)}
    roles = {"body": RoleKind.BODY, "medium": RoleKind.MEDIUM}
    id_roles = ("body", "medium")
    reads = {"position": Read("body", POSITION.name)}
    body: Any = None
    medium: Any = None

    @override
    def build(self, ctx: BuildContext, b: ObservationBuilder) -> None:
        region = ctx.region("medium")
        assert region is not None, "the probe's medium fills a region with a shape"
        node = b.add_node(ToyRegionProbeNode(ctx.node_name(), ctx.timestep, shape=region.shape, **region.parameters))
        b.connect(b.read("position"), into=node["position"])
