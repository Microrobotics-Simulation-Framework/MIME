"""Family suites: the cases every law of a family passes, as harnesses a lab calls from its own tests
(``assert_conforms(MyDrag(), family="hydrodynamic", oracle=...)``); passing in the lab's own CI is the claim.

A family states what its laws must satisfy, over the vocabulary's interfaces rather than any one law's labels: its
sign rule, the settings that stand for a material's properties, the pairs a two-way law delivers equal and opposite,
a reference member a swap replaces, and a rig, the smallest experiment that holds one of its laws. Each case is
checked on the law as the rig builds it. The sampled cases (sign, frame, oracle, gradients, equal and opposite,
energy) are
invariants of MADDENING's ``verify_node`` on the law's own node, each drawn and shrunk by Hypothesis; units,
material reach and swap build the rig again; nodes runs MADDENING's whole battery on every node the law builds.

A law that builds no node of its own but confines a body (``Contact``'s projection) is probed through the body's
node: what it reads is the state the body's dynamics reaches without it (a twin of the node built without its
confinement, as a MIME body node takes one: ``surface_shape``, ``surface_*``, ``extent``, ``margin``,
``velocity_rule``), and what it delivers is the state the confined node reaches. The values a law reads (a body's
moment, a medium's density) reach its oracle and its rules beside its settings, by their labels in its plan.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import jax
import jax.numpy
import numpy as np
from jax import Array
from maddening.core.node import SimulationNode
from maddening.testing.verification import VerificationResult

from mime import vocabulary
from mime.build import BuildRefused, EntryPlan, ValueDeclaration
from mime.build._maddening import node_dict, node_update
from mime.build._report import graph_config
from mime.core import ledger, quaternion, shape_named
from mime.experiment import BuiltExperiment, Experiment
from mime.nodes.body._rigid_inertial import CONTACT_ULPS
from mime.registry import Interaction, registration_of, setting_value
from mime.testing._nodes import NodeCase, computing_dtype, node_failures, run_verify_node
from mime.units import TIME, Magnitude, Quantity
from mime.vocabulary import EnergyClass

#: ``jax.numpy``'s hints name a device type from jaxlib's compiled core, which ships no stubs.
jnp: Any = jax.numpy


def _update(node: SimulationNode, state: Mapping[str, Any], inputs: Mapping[str, Any], dt: float,
            params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``node.update``, whose MADDENING annotation takes bare dicts; with ``params`` written over its constants."""
    if params:
        return cast(dict[str, Any], node_update(node)(dict(state), dict(inputs), dt, params=dict(params)))
    return cast(dict[str, Any], node_update(node)(dict(state), dict(inputs), dt))


def _constants(node: SimulationNode) -> dict[str, Any]:
    return dict(cast(dict[str, Any], cast(Any, node).params))


def _initial(node: SimulationNode) -> dict[str, Any]:
    return cast(dict[str, Any], cast(Any, node).initial_state())


#: ``jax.jvp``, whose hints leave its function's arguments unknown.
_jvp: Callable[..., tuple[Any, Any]] = cast(Any, jax).jvp

type At = tuple[str, str]
"""A role and an interface, ``("body", "toy.body.velocity")``: what a family's rules are written over."""

type Oracle = Callable[[Mapping[str, Array], Mapping[str, Magnitude]], Mapping[str, Any]]
"""A law's closed form: from its readings by label and its settings in SI, what it delivers, by label."""


class CaseKind(Enum):
    """The kinds of case a family suite holds."""

    SIGN = "sign"
    FRAME = "frame"
    UNITS = "units"
    MATERIAL_REACH = "material reach"
    ORACLE = "oracle"
    GRADIENTS = "gradients"
    SWAP = "swap"
    EQUAL_AND_OPPOSITE = "equal and opposite"
    NODES = "nodes"
    CLOSED_LOOP = "closed loop"
    ENERGY = "energy"


@dataclass(frozen=True)
class Response:
    """One evaluation of a law: what it read and what it delivered, by label, with its settings and the values it
    reads in SI; ``reading`` and ``delivery`` look them up by role and interface. ``constants`` are its node's
    constructor arguments."""

    read: Mapping[str, Array]
    delivered: Mapping[str, Array]
    settings: Mapping[str, Magnitude]
    reads_at: Mapping[At, str]
    delivers_at: Mapping[At, str]
    constants: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def reading(self, role: str, interface: str) -> Array:
        return self.read[self.reads_at[(role, interface)]]

    def delivery(self, role: str, interface: str) -> Array:
        return self.delivered[self.delivers_at[(role, interface)]]


@dataclass(frozen=True)
class SignRule:
    """A family's sign rule: ``holds(response)`` for every response of every member. ``reads`` and ``delivers``
    are what it is written over, which every member must read and deliver."""

    description: str
    holds: Callable[[Response], bool]
    reads: tuple[At, ...] = ()
    delivers: tuple[At, ...] = ()


def opposes(delivered: At, reading: At, *, rtol: float = 1e-6) -> SignRule:
    """The rule that what is delivered at ``delivered`` never has a component along ``reading``: ``d . r <= 0``,
    within ``rtol`` of ``|d| |r|``. A drag against a velocity, a spring against a displacement."""

    def holds(response: Response) -> bool:
        d, r = response.delivery(*delivered), response.reading(*reading)
        return float(jnp.vdot(d, r)) <= rtol * float(jnp.linalg.norm(d) * jnp.linalg.norm(r))

    return SignRule(f"{delivered[1]} on the {delivered[0]} opposes {reading[1]} of the {reading[0]}", holds,
                    reads=(reading,), delivers=(delivered,))


@dataclass(frozen=True)
class Loop:
    """A closed loop of a body's pose, ``t`` from 0 to ``2 pi``: its position ``centre + across cos t + along sin t``
    and its orientation ``start`` turned by ``t`` about ``axis`` (a unit vector in the world, or zero for none), so
    that its world angular velocity is ``axis`` and its velocity ``-across sin t + along cos t``, each per unit of
    ``t``."""

    centre: tuple[float, float, float]
    across: tuple[float, float, float]
    along: tuple[float, float, float]
    axis: tuple[float, float, float]
    start: tuple[float, float, float, float]

    def poses_and_rates(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The pose at each ``t`` (``(n, 7)``, position then ``(w, x, y, z)``) and its rate (``(n, 6)``, the velocity
        then the angular velocity), in float64."""
        centre, across, along, axis = (np.asarray(v, dtype=np.float64) for v in (self.centre, self.across,
                                                                                    self.along, self.axis))
        cos, sin = np.cos(t)[:, None], np.sin(t)[:, None]
        position = centre + across * cos + along * sin
        velocity = -across * sin + along * cos
        w2, x2, y2, z2 = np.asarray(self.start, dtype=np.float64)
        turn = np.concatenate([np.cos(t / 2)[:, None], np.sin(t / 2)[:, None] * axis], axis=1) \
            if float(np.linalg.norm(axis)) > 0.0 else np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1))
        w1, x1, y1, z1 = turn.T
        orientation = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)
        return np.concatenate([position, orientation], axis=1), np.concatenate([velocity, np.tile(axis, (len(t), 1))],
                                                                               axis=1)


@dataclass(frozen=True)
class WorkLoop:
    """The closed-loop case's setting, for a family whose laws together are the derivative of an energy in a static
    field. ``rig(laws, drawn)`` is the smallest experiment holding ``laws`` (the law under test with the members that
    complete it) on the body named ``body`` in a static field, built from one draw of ``draws()``, a Hypothesis
    strategy; ``loops(drawn)`` are the closed loops of the body's pose that draw drives it round."""

    rig: Callable[[Sequence[Interaction], Any], Experiment]
    draws: Callable[[], Any]
    loops: Callable[[Any], Sequence[Loop]]
    body: str


@dataclass(frozen=True)
class Family:
    """What every law of a family satisfies, and the cases that check it.

    ``rig(law)`` is the smallest experiment holding ``law``. ``material`` names the settings, or the labels of the
    values its laws read (``moment``), that stand for a material's properties, each of which must reach the law's
    output. ``pairs`` are the deliveries a two-way law makes equal and opposite. ``reference()`` is the member a swap
    replaces. ``reading_bounds`` bounds the draws of each read interface (MADDENING's default box is (-1e4, 1e4)),
    and ``constrain`` maps a drawn value of an interface onto what it can be (a pose's quaternion onto unit length).
    ``frame_constants`` names the node constants that are world-frame vectors or points (the gravity a law was built
    with), which turning the world turns too; ``frame_constants_held`` those a member holds only when it reads what
    they describe (a medium's geometry, which a law in an unbounded fluid does not read), turned where it holds them.
    ``uniform_along`` lists displacements of the rig's world along which its setup is uniform (along a straight
    tube's axis): moving every point the law reads by one leaves what it delivers unchanged but for the points it
    delivers, which move with it. ``linear_in`` names the members of ``material`` that what a law delivers is
    proportional to (a resistance to the medium's viscosity): changed by half, the response changes by half. ``loop``
    sets the closed-loop case: the laws that together derive from an energy do no net work round a closed loop of the
    body's pose in a static field. ``potential_scale(response)`` is, for a potential member's energy case, the
    magnitude of the arithmetic its force and its couple are formed by (``|m g| + |F_b|`` and ``|c_b| |F_b|`` for
    gravity), on which the wrench is judged against its energy's negative gradient."""

    name: str
    rig: Callable[[Interaction], Experiment]
    cases: tuple[CaseKind, ...]
    sign: SignRule | None = None
    material: tuple[str, ...] = ()
    pairs: tuple[tuple[At, At], ...] = ()
    reference: Callable[[], Interaction] | None = None
    reading_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict[str, tuple[float, float]])
    constrain: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict[str, Callable[[Any], Any]])
    frame_constants: tuple[str, ...] = ()
    frame_constants_held: tuple[str, ...] = ()
    uniform_along: tuple[tuple[float, float, float], ...] = ()
    linear_in: tuple[str, ...] = ()
    loop: WorkLoop | None = None
    potential_scale: Callable[[Response], tuple[float, float]] | None = None


_families: dict[str, Family] = {}


def register_family(family: Family) -> Family:
    held = _families.get(family.name)
    if held is not None and held is not family and held.rig is not family.rig:
        raise ValueError(f"family {family.name!r} is already registered")
    _families[family.name] = family
    return family


def family_named(name: str) -> Family:
    if name not in _families:
        raise KeyError(f"no family {name!r} is registered; the families are {sorted(_families)}")
    return _families[name]


def registered_families() -> list[Family]:
    """Every registered family, by name."""
    return [_families[name] for name in sorted(_families)]


@dataclass(frozen=True)
class Probe:
    """A law as its family's rig built it: its plan, its one node, and how its labels map onto that node's input
    ports and state fields. For a law that confines a body, ``node`` is the body's node and ``free`` its twin with
    no confinement, whose step is what the law reads (``outputs`` names the fields read and delivered)."""

    built: BuiltExperiment
    plan: EntryPlan
    node: SimulationNode
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    reads_at: Mapping[At, str]
    delivers_at: Mapping[At, str]
    free: SimulationNode | None = None

    def settings(self) -> dict[str, Magnitude]:
        return {**self.plan.settings, **{label: value.value for label, value in self.plan.values.items()}}

    def readings(self, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float,
                 params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.free is not None:
            reached = _update(self.free, state, boundary, dt, params)
            return {label: reached[f] for label, f in self.outputs.items()}
        return {label: boundary[port] for label, port in self.inputs.items() if port in boundary}

    def response(self, boundary: Mapping[str, Any], out: Mapping[str, Any], state: Mapping[str, Any],
                 dt: float) -> Response:
        return Response(self.readings(state, boundary, dt), {label: out[f] for label, f in self.outputs.items()},
                        self.settings(), self.reads_at, self.delivers_at, _constants(self.node))

    def respond(self, read: Mapping[str, Any]) -> dict[str, Array]:
        """What the law delivers, by label, given what it reads, by label, from its node's initial state (for a law
        that confines a body, ``read`` is the state the body is stepped from)."""
        if self.free is not None:
            state = {**_initial(self.node), **{self.outputs[label]: v for label, v in read.items()}}
            stepped = _update(self.node, state, {}, self.built.plans.dt)
        else:
            stepped = _update(self.node, _initial(self.node), {self.inputs[label]: v for label, v in read.items()},
                              self.built.plans.dt)
        return {label: stepped[f] for label, f in self.outputs.items()}


def _planned(built: BuiltExperiment, law: Interaction) -> EntryPlan | None:
    return next((p for p in built.plans.entries if p.entry is law), None) or next(
        (p for p in built.plans.entries if p.entry == law), None)


#: How a MIME body node takes its confinement, as constructor arguments: the twin a confining law is read through is
#: built with these set to no confinement.
_UNCONFINED = {"surface_shape": "", "extent": 0.0, "margin": 0.0, "velocity_rule": ""}


def _confinement_probe(built: BuiltExperiment, plan: EntryPlan, family: Family) -> Probe | str:
    planned = plan.plan
    if planned is None or len(planned.confines) != 1:
        return f"{plan.owner} builds no node and confines no one body, and the sampled cases evaluate a law built " \
               f"as one node or into one body's node"
    (confinement,) = planned.confines
    body = plan.bindings[confinement.role]
    dynamics = next((p for p in built.plans.entries if p.thing == body and p.record is not None), None)
    if dynamics is None or len(dynamics.node_names()) != 1 or dynamics.record is None:
        return f"{plan.owner} confines {body!r}, whose dynamics is not built as one node"
    (name,) = dynamics.node_names()
    node = built.graph.get_node(name)
    saved = node_dict(node)
    held = {key: value for key, value in cast(dict[str, Any], saved["params"]).items()
            if not key.startswith("surface_")}
    try:
        free = type(node)(saved["name"], saved["timestep"], **{**held, **_UNCONFINED})
    except Exception as error:  # noqa: BLE001 - a dynamics that takes no confinement as MIME's nodes do
        return f"{plan.owner} confines {body!r}, whose node does not rebuild without its confinement: {error!r}"
    outputs = {port.field: port.field for port in dynamics.record.publishes.values()}
    at = {(confinement.role, interface): port.field for interface, port in dynamics.record.publishes.items()}
    return Probe(built, plan, node, {}, outputs, at, at, free)


def probe_law(law: Interaction, family: Family) -> Probe | str:
    """``law`` built in ``family``'s rig, or why it cannot be probed: its rig does not build, or it builds other
    than one node (the sampled cases evaluate a law built as one node, or into one body's node)."""
    return _probe_built(family.rig(law), law, family)


def _probe_built(experiment: Experiment, law: Interaction, family: Family) -> Probe | str:
    try:
        built = experiment.build()
    except BuildRefused as refused:
        return f"the {family.name} family's rig does not build with {law.label()}: {refused}"
    plan = _planned(built, law)
    if plan is None:
        return f"the {family.name} family's rig does not hold {law.label()}"
    names = plan.node_names()
    if not names:
        return _confinement_probe(built, plan, family)
    if len(names) != 1:
        return f"{plan.owner} builds {len(names)} nodes, and the sampled cases evaluate a law built as one"
    (name,) = names
    edges = cast(list[dict[str, Any]], graph_config(built.graph)["edges"])
    inputs = {label: e["target_field"] for label, read in plan.reads.items() for e in edges
              if (e["source_node"], e["source_field"], e["target_node"]) == (read.node, read.field, name)}
    outputs = {label: e["source_field"] for label, target in plan.delivers.items() for e in edges
               if (e["target_node"], e["target_field"], e["source_node"]) == (target.node, target.field, name)}
    planned = plan.plan
    reads_at = {(r.role, r.interface): label for label, r in planned.reads.items()} if planned else {}
    delivers_at = {(d.role, d.target): label for label, d in planned.delivers.items()} if planned else {}
    return Probe(built, plan, built.graph.get_node(name), inputs, outputs, reads_at, delivers_at)


def _wide(probe: Probe) -> bool:
    """Whether the law's node computes in float64: its draws are then float64 and its tolerances tighter."""
    return computing_dtype(probe.node).itemsize >= 8


def _interface_of_ports(probe: Probe) -> dict[str, str]:
    """Each input port of the probed node that a vocabulary interface names, by port: a law's reads, or every
    target a confined body's dynamics accepts."""
    if probe.free is None:
        return {probe.inputs[label]: interface for (_, interface), label in probe.reads_at.items()
                if label in probe.inputs}
    record = next((p.record for p in probe.built.plans.entries if probe.node.name in p.node_names()), None)
    return {port.field: interface for interface, port in (record.accepts.items() if record else ())
            if vocabulary.lookup(interface) is not None}


def _interface_of_state(probe: Probe) -> dict[str, str]:
    """Each state field of the probed node that a vocabulary interface names: a confined body's published state."""
    if probe.free is None:
        return {}
    return {probe.outputs[label]: interface for (_, interface), label in probe.reads_at.items()}


def _mapped(by_key: Mapping[str, Callable[[Any], Any]]) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    if not by_key:
        return None

    def apply(drawn: dict[str, Any]) -> dict[str, Any]:
        return {key: (by_key[key](value) if key in by_key else value) for key, value in drawn.items()}

    return apply


def _composed(first: Callable[[dict[str, Any]], dict[str, Any]] | None,
              then: Callable[[dict[str, Any]], dict[str, Any]]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return then if first is None else lambda drawn: then(first(drawn))


def _sampled(probe: Probe, family: Family, name: str,
             predicate: Callable[[Response, Mapping[str, Any], Mapping[str, Any], float], bool],
             max_examples: int, *, also_constrain_state: Callable[[dict[str, Any]], dict[str, Any]] | None = None
             ) -> VerificationResult:
    """``predicate(response, state, boundary, dt)`` as an invariant of the law's node, under ``verify_node``'s
    draws, each drawn value mapped by the family's ``constrain`` (a drawn state then by ``also_constrain_state``)."""
    ports, fields = _interface_of_ports(probe), _interface_of_state(probe)
    bounds = {port: family.reading_bounds[interface] for port, interface in ports.items()
              if interface in family.reading_bounds}
    state_bounds = {f: family.reading_bounds[interface] for f, interface in fields.items()
                    if interface in family.reading_bounds}
    constrain_boundary = _mapped({port: family.constrain[i] for port, i in ports.items() if i in family.constrain})
    constrain_state = _mapped({f: family.constrain[i] for f, i in fields.items() if i in family.constrain})
    if also_constrain_state is not None:
        constrain_state = _composed(constrain_state, also_constrain_state)

    def invariant(state: dict[str, Any], out: dict[str, Any], boundary: dict[str, Any], dt: float) -> bool:
        return predicate(probe.response(boundary, out, state, dt), state, boundary, dt)

    results = run_verify_node(probe.node, state_bounds or None, checks=[], invariants={name: invariant},
                              boundary_bounds=bounds or None, max_examples=max_examples,
                              dtype=computing_dtype(probe.node), constrain_state=constrain_state,
                              constrain_boundary=constrain_boundary)
    return results[name]


def _fail(name: str, detail: str) -> VerificationResult:
    return VerificationResult(name, "FAIL", detail=detail)


type Part = tuple[slice, ...]
"""One part of a value, as an index: a composite's position or orientation, one block of an operator."""

#: How the vocabulary lays out each composite world-frame value, by shape: a pose is a position and a quaternion, a
#: pair of vectors (a wrench, a twist) two vectors, and an operator on pairs four 3 x 3 blocks.
_COMPOSITE_PARTS: dict[tuple[int, ...], tuple[Part, ...]] = {
    (7,): ((slice(0, 3),), (slice(3, 7),)),
    (6,): ((slice(0, 3),), (slice(3, 6),)),
    (6, 6): tuple((rows, columns) for rows in (slice(0, 3), slice(3, 6)) for columns in (slice(0, 3), slice(3, 6))),
}


def parts_of(interface: str | None, shape: tuple[int, ...]) -> tuple[Part, ...]:
    """The parts of a value of ``interface`` that are each judged on their own scale: the parts of a composite, whose
    units the vocabulary separates by ``;`` (a pose's metres and its unit quaternion; an operator's blocks, in N s/m,
    N s and N m s), and otherwise the whole value."""
    found = vocabulary.lookup(interface) if interface is not None else None
    if found is not None and found.shape == shape and ";" in found.units and shape in _COMPOSITE_PARTS:
        return _COMPOSITE_PARTS[shape]
    return (tuple(slice(None) for _ in shape),)


def _per_part(values: np.ndarray, parts: Sequence[Part]) -> np.ndarray:
    """Each component replaced by the largest magnitude in its part."""
    out = np.zeros_like(values)
    for part in parts:
        out[part] = float(np.max(np.abs(values[part]), initial=0.0))
    return out


def _close(found: Any, expected: Any, rtol: float, floor: Any = 0.0, parts: Sequence[Part] | None = None) -> bool:
    """``found`` within ``rtol`` of ``expected``, plus ``floor``, part by part: every component within ``rtol`` of
    the largest component of its part, plus its floor (a number, or an array of ``expected``'s shape). With no parts
    the value is one part. The tolerance has no absolute part of its own: an SI torque of 1e-12 N m is not zero, and
    a rotational resistance of 1e-10 N m s is not judged on the scale of a translational one of 1e-3 N s/m."""
    found, expected = np.asarray(found, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if found.shape != expected.shape:
        return False
    held = parts if parts is not None else (tuple(slice(None) for _ in expected.shape),)
    scale = _per_part(expected, held)
    return bool(np.all(np.abs(found - expected) <= rtol * scale + np.asarray(floor, dtype=np.float64)))


def _nudged(values: Mapping[str, Any], keys: set[str], eps: float, parity: int,
            interfaces: Mapping[str, str]) -> dict[str, Any]:
    """``values`` with each component of ``keys`` moved by 64 units in the last place of the largest magnitude in its
    part (:func:`parts_of`), in alternating directions: arithmetic that mixes a part's components (a rotation, a
    projection onto an axis) rounds a small component on the scale of the large ones."""
    out: dict[str, Any] = {}
    for index, (key, value) in enumerate(values.items()):
        held = np.asarray(value)
        if key not in keys or not np.issubdtype(held.dtype, np.floating):
            out[key] = value
            continue
        signs = np.resize([1.0, -1.0] if (index + parity) % 2 == 0 else [-1.0, 1.0], held.shape)
        magnitude = _per_part(np.abs(held.astype(np.float64)), parts_of(interfaces.get(key), held.shape))
        out[key] = (held + 64 * eps * magnitude * signs).astype(held.dtype)
    return out


def _interface_of_outputs(probe: Probe) -> dict[str, str]:
    """Each output field of the probed node that delivers a vocabulary interface, by field."""
    return {probe.outputs[label]: interface for (_, interface), label in probe.delivers_at.items()
            if label in probe.outputs}


def output_parts(probe: Probe, field: str, shape: tuple[int, ...]) -> tuple[Part, ...]:
    """The parts of the probed node's output ``field`` that are judged each on its own scale."""
    return parts_of(_interface_of_outputs(probe).get(field), shape)


def _rounding(probe: Probe, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> dict[str, np.ndarray]:
    """How far what the law delivers moves when each component it reads moves by 64 units in its last place, in
    alternating directions (a component of a composite by 64 units in the last place of its part's largest
    magnitude): the scale below which two evaluations of the law that order their arithmetic otherwise cannot be told
    apart, cancellation included. Each output field has its own floor, and a composite one a floor per
    part (:func:`parts_of`), each from the part's own movement: a confined position is judged on the rounding of
    metres, not on that of the unit quaternion beside it in the pose, nor of the twist. Never below the dtype's own
    resolution near its smallest normal number (``16 tiny / eps``, about 1e-30 in float32), where XLA flushes
    subnormal results to zero and a turned or reordered evaluation can lose every digit. A confined body reads its
    whole state and every input. Returns an array of each field's shape, by field."""
    info = np.finfo(computing_dtype(probe.node))
    eps, flushed = float(info.eps), 16 * float(info.tiny) / float(info.eps)
    ports = set(boundary) if probe.free is not None else set(probe.inputs.values())
    fields = set(state) if probe.free is not None else set[str]()
    base = _update(probe.node, state, boundary, dt)
    moved = _update(probe.node, _nudged(state, fields, eps, 1, _interface_of_state(probe)),
                    _nudged(boundary, ports, eps, 0, _interface_of_ports(probe)), dt)
    floors: dict[str, np.ndarray] = {}
    for f in probe.outputs.values():
        movement = np.abs(np.asarray(moved[f], dtype=np.float64) - np.asarray(base[f], dtype=np.float64))
        floors[f] = _per_part(movement, output_parts(probe, f, movement.shape)) + flushed
    return floors


def _tolerance(probe: Probe) -> float:
    return 1e-9 if _wide(probe) else 1e-5


def check_sign(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    rule = family.sign
    if rule is None:
        return _fail("sign", f"the {family.name} family lists the sign case and states no sign rule")
    missing = [f"reads no {interface} of its {role}" for role, interface in rule.reads if (role, interface)
               not in probe.reads_at]
    missing += [f"delivers no {interface} into its {role}" for role, interface in rule.delivers
                if (role, interface) not in probe.delivers_at]
    if missing:
        return _fail("sign", f"{probe.plan.owner} {', '.join(missing)}, which the family's sign rule "
                             f"({rule.description}) is written over")
    return _sampled(probe, family, "sign", lambda response, _, __, ___: rule.holds(response), max_examples)


def _rotations() -> list[tuple[np.ndarray, np.ndarray]]:
    """The proper rotations a frame case turns the world by, each as its matrix and its unit quaternion ``(w, x, y,
    z)``: a quarter turn about z, a third of a turn about (1, 1, 1), and one about no particular axis."""
    def about(axis: Sequence[float], angle: float) -> tuple[np.ndarray, np.ndarray]:
        k = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        matrix = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross
        return matrix, np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * k])

    return [about((0.0, 0.0, 1.0), math.pi / 2), about((1.0, 1.0, 1.0), 2 * math.pi / 3),
            about((0.3, -0.5, 0.8), 0.7)]


def _product(p: Any, q: Any) -> Any:
    w1, x1, y1, z1 = p[0], p[1], p[2], p[3]
    w2, x2, y2, z2 = q[0], q[1], q[2], q[3]
    return jnp.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def turned(interface: str, value: Any, rotation: tuple[np.ndarray, np.ndarray]) -> Any | None:
    """``value`` of a world-frame interface in the world turned by ``rotation``, or ``None`` for one turning does
    not act on. By shape, as the vocabulary lays its composites out: a vector ``(3,)``; a pair of vectors ``(6,)``
    (a wrench, a twist); a pose ``(7,)``, its position turned and its orientation turned with it; a tensor ``(3,
    3)``; an operator on pairs ``(6, 6)``."""
    found = vocabulary.lookup(interface)
    if found is None or found.frame != "world":
        return None
    v = jnp.asarray(value)
    r = jnp.asarray(rotation[0], dtype=v.dtype)
    if found.shape == (3,):
        return r @ v
    if found.shape == (6,):
        return jnp.concatenate([r @ v[:3], r @ v[3:]])
    if found.shape == (7,):
        return jnp.concatenate([r @ v[:3], _product(jnp.asarray(rotation[1], dtype=v.dtype), v[3:])])
    if found.shape == (3, 3):
        return r @ v @ r.T
    if found.shape == (6, 6):
        pair = jnp.zeros((6, 6), dtype=v.dtype).at[:3, :3].set(r).at[3:, 3:].set(r)
        return pair @ v @ pair.T
    return None


def _reading_floor(probe: Probe, response: Response, label: str) -> np.ndarray | None:
    """For a law that confines a body, the magnitude of each component of the state it read under ``label``: what
    it delivers is that state with a part taken away (the outward velocity), so a component that cancels to zero is
    judged on the scale of the component it came from, not on its own. ``None`` for a law built as its own node."""
    if probe.free is None or label not in response.read:
        return None
    return np.abs(np.asarray(response.read[label], dtype=np.float64))


def _close_by_component(found: Any, expected: Any, rtol: float, floor: Any, reading: np.ndarray | None,
                        parts: Sequence[Part]) -> bool:
    """:func:`_close`, part by part, for a law built as its own node (``reading`` zero). For a confining law,
    component by component: within ``rtol`` of the larger of the expected component and the component it was read
    from, plus its ``floor``; a pose holds metres beside a unit quaternion, and the field's largest magnitude would
    let a position move by ``rtol`` metres."""
    if reading is None:
        return _close(found, expected, rtol, floor, parts)
    found, expected = np.asarray(found, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if found.shape != expected.shape or reading.shape != expected.shape:
        return False
    return bool(np.all(np.abs(found - expected) <= rtol * np.maximum(np.abs(expected), reading) + floor))


_TURNING_SHAPES = ((3,), (6,), (7,), (3, 3), (6, 6))


def _turns(interface: str) -> bool:
    found = vocabulary.lookup(interface)
    return found is not None and found.frame == "world" and found.shape in _TURNING_SHAPES


def _turned_or_kept(interface: str | None, value: Any, rotation: tuple[np.ndarray, np.ndarray]) -> Any:
    found = turned(interface, value, rotation) if interface is not None else None
    return value if found is None else found


def shifted(interface: str, value: Any, displacement: Sequence[float]) -> Any | None:
    """``value`` of a world-frame interface once everything it locates is moved by ``displacement``, or ``None`` for
    one that locates nothing: a pose's position moves (its orientation does not), and so does a point; a velocity, a
    load or a resistance is unchanged."""
    found = vocabulary.lookup(interface)
    if found is None or found.frame != "world":
        return None
    v = jnp.asarray(value)
    d = jnp.asarray(displacement, dtype=v.dtype)
    if found.shape == (7,) and found.units.split(";")[0].strip() == "m":
        return jnp.concatenate([v[:3] + d, v[3:]])
    if found.shape == (3,) and found.units == "m":
        return v + d
    return None


def _shifted_or_kept(interface: str | None, value: Any, displacement: Sequence[float]) -> Any:
    found = shifted(interface, value, displacement) if interface is not None else None
    return value if found is None else found


def check_frame(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    """Turning the world turns what the law delivers: with every world-frame quantity it reads turned by ``R`` (its
    inputs; for a confined body its state; and the node constants the family names as world-frame), each
    world-frame quantity it delivers is turned by ``R``."""
    ports, fields = _interface_of_ports(probe), _interface_of_state(probe)
    outputs = {probe.outputs[label]: interface for (_, interface), label in probe.delivers_at.items()
               if _turns(interface)}
    if not outputs:
        return _fail("frame", f"{probe.plan.owner} delivers no world-frame quantity, so turning the world tests "
                              f"nothing")
    constants = _constants(probe.node)
    missing = [name for name in family.frame_constants if name not in constants]
    if missing:
        return _fail("frame", f"{probe.plan.owner}'s node has no constants {missing}, which the {family.name} family "
                              f"turns with the world")
    world = (*family.frame_constants, *(name for name in family.frame_constants_held if name in constants))
    rotations, rtol = _rotations(), _tolerance(probe)

    def turns(response: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        out = {probe.outputs[label]: value for label, value in response.delivered.items()}
        floors = _rounding(probe, state, boundary, dt)
        read = [np.abs(np.asarray(v, dtype=np.float64)) for v in response.read.values()] if probe.free else []
        # A confined body delivers its state with a part taken away: a field that cancels is judged on the scale of
        # the state it came from. Turning mixes components, so here the scale is the whole field's.
        cancelled = rtol * max((float(np.max(r, initial=0.0)) for r in read), default=0.0)

        def agrees(stepped: Mapping[str, Any], expected: Callable[[str, Any], Any]) -> bool:
            return all(_close(stepped[f], expected(interface, out[f]), rtol, 4 * floors[f] + cancelled,
                              output_parts(probe, f, np.shape(out[f]))) for f, interface in outputs.items())

        for rotation in rotations:
            given = {port: _turned_or_kept(ports.get(port), value, rotation) for port, value in boundary.items()}
            held = {f: _turned_or_kept(fields.get(f), value, rotation) for f, value in state.items()}
            params = {name: jnp.asarray(rotation[0]) @ jnp.asarray(constants[name]) for name in world}
            stepped = _update(probe.node, held, given, dt, params)
            if not agrees(stepped, lambda interface, value, r=rotation: turned(interface, value, r)):
                return False
        for displacement in family.uniform_along:
            given = {port: _shifted_or_kept(ports.get(port), value, displacement) for port, value in boundary.items()}
            held = {f: _shifted_or_kept(fields.get(f), value, displacement) for f, value in state.items()}
            stepped = _update(probe.node, held, given, dt)
            if not agrees(stepped, lambda interface, value, d=displacement: _shifted_or_kept(interface, value, d)):
                return False
        return True

    return _sampled(probe, family, "frame", turns, max_examples)


def oracle_agrees(probe: Probe, oracle: Oracle) -> Callable[[Response, Mapping[str, Any], Mapping[str, Any], float],
                                                           bool]:
    """The oracle case's predicate on one draw: what the law delivered is the oracle's, within the relative tolerance
    plus the rounding floors."""
    rtol = _tolerance(probe)

    def agrees(response: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        expected = oracle(response.read, response.settings)
        floors = _rounding(probe, state, boundary, dt)
        return set(expected) == set(response.delivered) and all(
            _close_by_component(response.delivered[label], expected[label], rtol, 4 * floors[probe.outputs[label]],
                                _reading_floor(probe, response, label),
                                output_parts(probe, probe.outputs[label], np.shape(response.delivered[label])))
            for label in expected)

    return agrees


#: How many contact bands inside its surface a drawn body is put when it was drawn in the band (the band's own
#: rounding is the node's; a body this far in is free, with room for the draw's float32 rounding of where it stands).
BAND_WIDTHS_INSIDE = 8


def off_the_contact_band(probe: Probe) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    """For a law that confines a body, the map on a drawn state that moves a body drawn within four contact bands of
    its surface (the node's ``CONTACT_ULPS`` units in the last place of ``|x|`` plus the inset, a few times over for
    the rounding of this evaluation against the node's) to ``BAND_WIDTHS_INSIDE`` bands inside it, along the
    surface's normal; ``None`` for any other law.

    A body that starts its step on its surface and moves outward is held there inside the node's solve, along the
    normal at the start of the step. The law's closed form, a free step projected back along the normal at its end,
    reads only that end and cannot know the step began in contact; on a curved surface the two differ at first order
    in the step (on a 1.4 mm shrunk radius, 1 m/s along and across it, 6.6 micrometres and 0.066 m/s in y after a
    step of 0.1 ms).
    That is the node's design (4c3), not a rounding of the closed form, and it is tested on purpose, in the band,
    in ``tests/families/test_family_walk.py``."""
    own = _constants(probe.node)
    if probe.free is None or not own.get("surface_shape"):
        return None
    shape = shape_named(str(own["surface_shape"]))
    parameters = {name: own[f"surface_{name}"] for name in shape.parameter_names}
    inset = float(own["extent"]) + float(own["margin"])
    poses = [f for f, interface in _interface_of_state(probe).items() if interface == vocabulary.POSE.name]

    def moved(drawn: dict[str, Any]) -> dict[str, Any]:
        out = dict(drawn)
        for f in poses:
            held = np.asarray(drawn[f])
            position = jnp.asarray(held[:3])
            beyond = float(shape.signed_distance(parameters, position)) + inset
            resolution = CONTACT_ULPS * float(np.finfo(held.dtype).eps) * (float(np.linalg.norm(held[:3])) + inset)
            if abs(beyond) <= 4 * resolution:
                normal = np.asarray(shape.outward_normal(parameters, position), dtype=np.float64)
                inside = held[:3].astype(np.float64) - (beyond + BAND_WIDTHS_INSIDE * resolution) * normal
                out[f] = np.concatenate([inside, held[3:].astype(np.float64)]).astype(held.dtype)
        return out

    return moved


def check_oracle(probe: Probe, family: Family, oracle: Oracle | None, max_examples: int) -> VerificationResult:
    if oracle is None:
        return _fail("oracle", f"the {family.name} family holds every law to its closed form: pass oracle=, a "
                               f"function from its readings and settings in SI to what it delivers, by label")
    return _sampled(probe, family, "oracle", oracle_agrees(probe, oracle), max_examples,
                    also_constrain_state=off_the_contact_band(probe))


def check_gradients(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    """The derivative of what the law delivers along a direction in what it reads, by forward-mode differentiation,
    is finite and agrees with a central finite difference: within the difference's own truncation error, estimated
    from the difference at half the step (Richardson), so a point where the law is not smooth asks only what a
    finite difference can tell there. The direction moves each part of each input (:func:`parts_of`) in proportion
    to that part's own magnitude, so a pose's millimetres move by a fraction of a millimetre, not by a fraction of its
    unit quaternion."""
    ports = list(probe.inputs.values())
    if not ports:
        return _fail("gradients", f"{probe.plan.owner} reads nothing through its node's inputs, so it has no "
                                  f"derivative to check here")
    step, rtol = (1e-6, 1e-8) if _wide(probe) else (1e-2, 1e-4)
    smallest = float(np.finfo(computing_dtype(probe.node)).tiny) ** 0.5
    interfaces = _interface_of_ports(probe)

    def agrees(response: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        dtype = jnp.asarray(boundary[ports[0]]).dtype
        direction: dict[str, Any] = {}
        for p in ports:
            held = np.asarray(boundary[p], dtype=np.float64)
            scale = np.maximum(_per_part(held, parts_of(interfaces.get(p), held.shape)), smallest)
            direction[p] = jnp.asarray(np.resize([0.6, -0.8, 0.3, 0.5], held.shape) * scale, dtype=dtype)
        h = step

        def delivered(eps: Any) -> Array:
            given = {**boundary, **{p: jnp.asarray(boundary[p], dtype=dtype) + eps * direction[p] for p in ports}}
            stepped = _update(probe.node, state, given, dt)
            return jnp.concatenate([jnp.ravel(stepped[f]) for f in probe.outputs.values()])

        def difference(at: float) -> np.ndarray:
            return np.asarray((delivered(jnp.asarray(at, dtype=dtype)) - delivered(jnp.asarray(-at, dtype=dtype)))
                              / (2 * at), dtype=np.float64)

        _, tangent = _jvp(delivered, (jnp.asarray(0.0, dtype=dtype),), (jnp.asarray(1.0, dtype=dtype),))
        coarse, fine = difference(h), difference(h / 2)
        truncation = float(np.max(np.abs(coarse - fine), initial=0.0))
        floors = _rounding(probe, state, boundary, dt)
        rounding = np.concatenate([np.ravel(floors[f]) for f in probe.outputs.values()])
        floor = 2 * truncation + 4 * rounding / h
        return bool(jnp.all(jnp.isfinite(tangent))) and _close(tangent, fine, rtol, floor)

    return _sampled(probe, family, "gradients", agrees, max_examples)


def check_equal_and_opposite(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    if not family.pairs:
        return _fail("equal and opposite", f"the {family.name} family lists this case and names no pair")
    missing = [at for pair in family.pairs for at in pair if at not in probe.delivers_at]
    if missing:
        return _fail("equal and opposite", f"{probe.plan.owner} delivers no {missing}, which the family pairs")
    rtol = _tolerance(probe)

    def balanced(response: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        floors = _rounding(probe, state, boundary, dt)
        for a, b in family.pairs:
            first, second = np.asarray(response.delivery(*a)), np.asarray(response.delivery(*b))
            floor = 4 * (floors[probe.outputs[probe.delivers_at[a]]] + floors[probe.outputs[probe.delivers_at[b]]])
            if not _close(first + second, np.zeros_like(first), 0.0, rtol * float(np.max(np.abs(first))) + floor):
                return False
        return True

    return _sampled(probe, family, "equal and opposite", balanced, max_examples)


#: How many points the closed-loop case evaluates round each loop, equally spaced in ``t``: the periodic trapezoid
#: rule, whose error on a loop's smooth periodic work falls geometrically with the count (its order is shown in
#: MIME's ``tests/families/test_closed_loop.py``); every other point gives the rule at half the count, whose
#: difference from the whole bounds the whole's quadrature error.
LOOP_POINTS = 64
#: The rounding floor's multiple of how far the work moves when every input of the laws' nodes moves by 64 units in
#: its last place (as the other cases' floors).
LOOP_ROUNDINGS = 4


@dataclass(frozen=True)
class LoopWork:
    """The work done on a body round one closed loop of its pose, by the periodic trapezoid rule at ``LOOP_POINTS``
    points (``work``) and at every other one (``coarse``), with the sum of its terms' magnitudes (``summed``, what the
    work cancels against) and its rounding floor (``floor``: ``LOOP_ROUNDINGS`` times the terms' movement under the
    64-ulp nudge of every input of the laws' nodes, on the scale of those inputs, so a term that cancels to zero is
    judged on the scale it came from). It holds where the work is within the quadrature's error, estimated as the
    whole rule's difference from the half's, plus the floor: no absolute part."""

    work: float
    coarse: float
    summed: float
    floor: float

    @property
    def tolerance(self) -> float:
        return abs(self.work - self.coarse) + self.floor

    @property
    def holds(self) -> bool:
        return abs(self.work) <= self.tolerance

    def describe(self) -> str:
        return (f"work {self.work:.6g} J, {abs(self.work) / self.summed if self.summed else 0.0:.3g} of the summed |W| "
                f"{self.summed:.6g} J, against a tolerance of {self.tolerance:.3g} J (quadrature "
                f"{abs(self.work - self.coarse):.3g}, rounding floor {self.floor:.3g})")


def _nudged_jnp(value: Any, interface: str | None, eps: float, parity: int) -> Any:
    """``value`` with each component moved by 64 units in the last place of the largest magnitude in its part
    (:func:`parts_of`), in alternating directions, in JAX so that it maps over a batch."""
    held = jnp.asarray(value)
    shape = tuple(int(n) for n in held.shape)
    signs = jnp.asarray(np.resize([1.0, -1.0] if parity % 2 == 0 else [-1.0, 1.0], shape), dtype=held.dtype)
    magnitude = jnp.zeros(shape, dtype=held.dtype)
    for part in parts_of(interface, shape):
        magnitude = magnitude.at[part].set(jnp.max(jnp.abs(held[part])))
    return held + 64 * eps * magnitude * signs


def loop_works(built: BuiltExperiment, laws: Sequence[Interaction], body: str, loops: Sequence[Loop], *,
               points: int = LOOP_POINTS) -> list[LoopWork]:
    """The work ``laws``, as ``built`` holds them, do on the body ``body`` round each of ``loops``: the pose
    prescribed, its twist the dynamics' initial one, every node the laws read through (the field's provider) and the
    laws' own nodes evaluated from it as the graph's step does, and the wrenches they deliver into the body's applied
    wrench summed, each term ``W . V`` with ``V`` the loop's rate."""
    plans = [_planned(built, law) for law in laws]
    if any(plan is None for plan in plans):
        raise ValueError(f"the rig does not hold every one of {[law.label() for law in laws]}")
    held_plans = [cast(EntryPlan, plan) for plan in plans]
    own = [name for plan in held_plans for name in plan.node_names()]
    read_from = {read.node for plan in held_plans for read in plan.reads.values() if read.slot is not None}
    targets = {(target.node, target.field) for plan in held_plans for target in plan.delivers.values()
               if target.interface == vocabulary.APPLIED_WRENCH.name and target.thing == body}
    poses = {read.node for plan in held_plans for read in plan.reads.values()
             if read.interface == vocabulary.POSE.name and read.slot is None}
    if len(poses) != 1 or not targets:
        raise ValueError(f"the laws read no one pose of {body!r} or deliver into no applied wrench of it")
    (body_node,) = poses
    graph = built.graph
    order = [name for name in graph.schedule if name in own or name in read_from]
    edges = cast(list[dict[str, Any]], graph_config(graph)["edges"])
    incoming = {name: [(e["source_node"], e["source_field"], e["target_field"]) for e in edges
                       if e["target_node"] == name] for name in order}
    delivered = [(e["source_node"], e["source_field"]) for e in edges
                 if (e["target_node"], e["target_field"]) in targets and e["source_node"] in own]
    interfaces = {(plan_read.node, plan_read.field): plan_read.interface for plan in held_plans
                  for plan_read in plan.reads.values()}
    initial = {name: _initial(graph.get_node(name)) for name in {*order, body_node,
                                                                *(e[0] for name in order for e in incoming[name])}}
    dtype = computing_dtype(graph.get_node(own[0]))
    eps = float(np.finfo(dtype).eps)

    def delivered_wrench(held: Mapping[str, Mapping[str, Any]]) -> Any:
        return sum(jnp.asarray(held[source][field]) for source, field in delivered)

    def evaluated(name: str, held: Mapping[str, Mapping[str, Any]], nudge: bool) -> Mapping[str, Any]:
        node = graph.get_node(name)
        inputs = {port: (held[source] if source in held else initial[source])[field]
                  for source, field, port in incoming[name]}
        if nudge:
            sources = {port: (source, field) for source, field, port in incoming[name]}
            inputs = {port: _nudged_jnp(value, interfaces.get(sources[port]), eps, index)
                      for index, (port, value) in enumerate(inputs.items())}
        step = getattr(node, "fresh_update", None)
        dt = float(node.delta_t)
        return step(dict(initial[name]), inputs, dt) if step is not None else _update(node, initial[name], inputs, dt)

    def wrenches(pose: Any) -> tuple[Any, Any]:
        """What the laws deliver at ``pose``, and the same with every input of their own nodes nudged."""
        plain: dict[str, Mapping[str, Any]] = {body_node: {**initial[body_node], "pose": pose}}
        nudged: dict[str, Mapping[str, Any]] = dict(plain)
        for name in order:
            plain[name] = evaluated(name, plain, False)
            nudged[name] = evaluated(name, nudged, True) if name in own else plain[name]
        return delivered_wrench(plain), delivered_wrench(nudged)

    t = 2.0 * np.pi * np.arange(points) / points
    sampled = [loop.poses_and_rates(t) for loop in loops]
    poses_all = jnp.asarray(np.concatenate([pose for pose, _ in sampled]), dtype=dtype)
    plain_all, nudged_all = jax.vmap(wrenches)(poses_all)
    found, moved = np.asarray(plain_all, dtype=np.float64), np.asarray(nudged_all, dtype=np.float64)
    step = 2.0 * np.pi / points
    works: list[LoopWork] = []
    for index, (_, rate) in enumerate(sampled):
        rows = slice(index * points, (index + 1) * points)
        terms = np.sum(found[rows] * rate, axis=1)
        nudged = np.sum(np.abs(moved[rows] - found[rows]) * np.abs(rate), axis=1)
        works.append(LoopWork(float(np.sum(terms) * step), float(np.sum(terms[::2]) * 2 * step),
                              float(np.sum(np.abs(terms)) * step), LOOP_ROUNDINGS * float(np.sum(nudged) * step)))
    return works


def check_closed_loop(law: Interaction, family: Family, together: Sequence[Interaction],
                      max_examples: int) -> VerificationResult:
    """The law with the members that complete it (``together``) does no net work round a closed loop of the body's
    pose in a static field, on every draw of the family's loop rig and each of its loops: within the loop's quadrature
    error plus its rounding floor (:class:`LoopWork`). A failing draw is reported as drawn, not shrunk: each example
    builds the rig again (about 0.1 s), and Hypothesis shrinking one stopped after five minutes, unfinished."""
    from hypothesis import HealthCheck, Phase, given, settings

    loop = family.loop
    if loop is None:
        return _fail("closed loop", f"the {family.name} family lists the closed-loop case and sets no loop")
    laws = (law, *together)
    last: dict[str, Any] = {}
    count = 0

    @settings(max_examples=max_examples, deadline=None, database=None, phases=(Phase.explicit, Phase.generate),
              suppress_health_check=[HealthCheck.too_slow])
    @given(loop.draws())
    def prop(drawn: Any) -> None:
        nonlocal count
        count += 1
        last.clear()
        last["drawn"] = drawn
        try:
            built = loop.rig(laws, drawn).build()
        except BuildRefused as refused:
            raise AssertionError(f"the {family.name} family's loop rig does not build: {refused}") from None
        found = loop_works(built, laws, loop.body, loop.loops(drawn))
        failed = [f"loop {index}: {work.describe()}" for index, work in enumerate(found) if not work.holds]
        assert not failed, "; ".join(failed)

    try:
        prop()
    except AssertionError as error:
        return VerificationResult("closed loop", "FAIL", detail=str(error), counterexample=dict(last),
                                  n_examples=count)
    return VerificationResult("closed loop", "PASS", n_examples=count)


def _another_dimension(value: object, dimension: Any) -> Quantity:
    magnitude = value.value if isinstance(value, Quantity) else 1.0
    return Quantity(magnitude, dimension * TIME)


def _with_setting(entry: Any, name: str, value: object) -> Any | None:
    """``entry`` with its setting ``name`` (a group's as ``<setting>.<field>``) set to ``value``; ``None`` when the
    group it sits in is left out."""
    head, _, rest = name.partition(".")
    if not rest:
        return dataclasses.replace(entry, **{head: value})
    group = getattr(entry, head)
    return None if group is None else dataclasses.replace(entry, **{head: _with_setting(group, rest, value)})


def check_units(probe: Probe, law: Interaction, family: Family) -> VerificationResult:
    """SI at every port, and every dimensioned setting refused in another dimension: each input of the law's node
    declares the units of the read it takes, as the vocabulary spells them; and a quantity of another dimension for
    each of its dimensioned settings is refused at plan time, naming the setting."""
    problems: list[str] = []
    spec = probe.node.boundary_input_spec()
    for label, port in probe.inputs.items():
        wanted = probe.plan.reads[label].units
        found = getattr(spec.get(port), "expected_units", None)
        if found != wanted:
            problems.append(f"input {port} (read {label!r}) declares units {found!r}, and its read is in {wanted!r}")
    record = registration_of(law)
    for name, declared in (record.settings.items() if record else ()):
        changed = _with_setting(law, name, None)
        if declared.dimension is None or changed is None:
            continue  # a setting in a group the law leaves out
        wrong = _another_dimension(setting_value(law, name), declared.dimension)
        try:
            family.rig(cast(Interaction, _with_setting(law, name, wrong))).plan()
        except BuildRefused as refused:
            if not any(r.code == "setting.units" and f"setting {name!r}" in r.message for r in refused.refusals):
                problems.append(f"setting {name} given {wrong} is refused for other reasons only: {refused.codes()}")
        else:
            problems.append(f"setting {name} given {wrong}, of another dimension than {declared.dimension}, plans")
    if problems:
        return _fail("units", "; ".join(problems))
    return VerificationResult("units", "PASS")


def _probe_reading(probe: Probe) -> dict[str, Array]:
    if probe.free is not None:
        held = _initial(probe.node)
        return {label: jnp.full(np.shape(held[f]), 0.5, dtype=jnp.asarray(held[f]).dtype)
                for label, f in probe.outputs.items()}
    spec = probe.node.boundary_input_spec()
    return {label: jnp.full(tuple(getattr(spec[port], "shape", (3,))), 0.5) for label, port in probe.inputs.items()}


def redeclared(experiment: Experiment, path: str, factor: float) -> Experiment:
    """``experiment`` with its declared value at ``path`` multiplied by ``factor``, and nothing else changed."""
    held = cast(Any, experiment)  # the harness rebuilds the experiment from its declarations, one value changed
    declarations = [ValueDeclaration(d.path, d.value * factor, d.reason)
                    if isinstance(d, ValueDeclaration) and str(d.path) == path and isinstance(d.value, Quantity)
                    else d for d in held._declarations]
    return cast(Experiment, held._declared(held._settings, declarations))


def check_material_reach(probe: Probe, law: Interaction, family: Family) -> VerificationResult:
    """Each setting, or value the law reads, that the family names as a material's property reaches what the law
    delivers: the same reading gives another response once it is changed by half; for one the family names in
    ``linear_in``, a response half as large again, part by part within the dtype's tolerance."""
    problems: list[str] = []
    reading = _probe_reading(probe)
    before = probe.respond(reading)
    for name in family.material:
        if hasattr(law, name):
            value: object = getattr(law, name)
            changed = value * 1.5 if isinstance(value, (Quantity, float, int)) else value
            what = f"setting {name} changed from {value} to {changed}"
            other = probe_law(dataclasses.replace(law, **{name: changed}), family)
        elif name in probe.plan.values:
            path = probe.plan.values[name].path
            what = f"the value {path} it reads as {name!r}, multiplied by 1.5,"
            built = family.rig(law)
            other = _probe_built(redeclared(built, path, 1.5), law, family)
        else:
            problems.append(f"{law.label()} has no setting and reads no value {name!r}, which the {family.name} family "
                            f"names as a material's property")
            continue
        if isinstance(other, str):
            problems.append(other)
            continue
        after = other.respond(reading)
        if all(_close(after[label], before[label], 0.0) for label in before):
            problems.append(f"{what} leaves the law's response unchanged: it does not reach {probe.plan.owner}'s "
                            f"output")
        elif name in family.linear_in and not all(
                _close(after[label], 1.5 * np.asarray(before[label], dtype=np.float64), _tolerance(probe), 0.0,
                       output_parts(probe, probe.outputs[label], np.shape(before[label]))) for label in before):
            problems.append(f"{what} does not change the law's response by the same factor, and the {family.name} "
                            f"family's laws are proportional to it")
    if problems:
        return _fail("material reach", "; ".join(problems))
    return VerificationResult("material reach", "PASS")


def _masked(probe: Probe) -> list[str]:
    text = probe.built.snapshot()
    for name in sorted(probe.plan.node_names(), key=len, reverse=True):
        text = text.replace(name, "<law>")
    return text.replace(f" {type(probe.node).__name__} ", " <law node> ").splitlines()


def check_swap(probe: Probe, family: Family, swap_with: Interaction | None) -> VerificationResult:
    """The law replaces another member of its family in the rig and nothing else changes: the same nodes in the same
    order, the same edges with the same flags and units, the same groups and rates, the law's own node aside."""
    other_law = swap_with if swap_with is not None else (family.reference() if family.reference else None)
    if other_law is None:
        return _fail("swap", f"the {family.name} family names no reference member: pass swap_with=")
    other = probe_law(other_law, family)
    if isinstance(other, str):
        return _fail("swap", other)
    mine, theirs = _masked(probe), _masked(other)
    for line_a, line_b in zip(mine, theirs, strict=False):
        if line_a != line_b:
            return _fail("swap", f"swapping {other_law.label()} for {probe.plan.registered.cls.label()} changes the "
                                 f"rig beyond the law's own node:\n  {other_law.label()}: {line_b}\n  "
                                 f"{probe.plan.registered.cls.label()}: {line_a}")
    if len(mine) != len(theirs):
        return _fail("swap", "the rig's graph has another number of lines once swapped")
    return VerificationResult("swap", "PASS")


def check_nodes(probe: Probe, law: Interaction, node_case: NodeCase | None) -> VerificationResult:
    failures = node_failures(probe.built, law, node_case)
    if failures:
        return _fail("nodes", "\n".join(failures))
    return VerificationResult("nodes", "PASS")


#: The rounding floor the energy case judges against, in units of eps on its scale: the ledger's.
ENERGY_ULPS = ledger.IDENTITY_ULPS
#: How fast a body placed on its confining surface is drawn moving outward, at the least (m/s): enough that a step
#: carries it past the surface by more than the contact's band of rounding, so the law acts on almost every draw (in
#: float32 the real rule's removal at arrival is below one floor on about 5 % of them).
OUTWARD_SPEED = 1e-3


def _eps(dtype: Any) -> float:
    return float(np.finfo(np.dtype(dtype)).eps)


def check_energy(probe: Probe, law: Interaction, family: Family, max_examples: int) -> VerificationResult:
    """``law``'s energy class, settled by an exact identity on the law alone (F0-1 M6.4), judged against rounding on its
    own scale. A law that declares no class states why (``Interaction.unmetered``). A **potential** law's delivered
    wrench is the negative gradient of its family's energy on its node (``MimeNode.energy``), the force by the
    position's gradient and the couple by the gradient on the rotation group (the pose turned by ``exp(theta)`` in the
    world, at ``theta = 0``), within 16 eps of the arithmetic the family says it is formed by
    (``Family.potential_scale``). A **dissipative** law's delivered operators are positive semidefinite at the ledger's
    floor (``mime.core.ledger.definite_at_floor``, the rule its run-time check applies), and a load it delivers does no
    positive work on the flow it reads, or is zero where it reads none (a force read at a state, which does work on some
    motion, is no dissipation). A dissipative law **held in its receiver's step** (a contact) gives no energy through
    the step's solve nor by its rule at arrival, on the receiver's step metered for it alone, from states placed on its
    surface moving outward (:func:`onto_the_surface`), so that it acts on almost every draw (a rule's removal at arrival is below one floor on about 5 % of float32 draws,
    exactly zero on a few). A workless law's identity comes
    with its family."""
    cls = type(law)
    if law.energy_class is None:
        if cls.unmetered:
            return VerificationResult("energy", "PASS", detail=f"unmetered: {cls.unmetered} (a pass by that declaration "
                                                               f"alone: nothing about the law was checked)")
        return _fail("energy", f"{cls.label()} declares no energy class (Interaction.energy_class) and states no reason "
                               f"for being unmetered (Interaction.unmetered)")
    if law.energy_class is EnergyClass.POTENTIAL:
        return _check_potential(probe, family, max_examples)
    if law.energy_class is EnergyClass.DISSIPATIVE:
        if probe.free is not None:
            return _check_held(probe, family, max_examples)
        return _check_dissipative(probe, law, family, max_examples)
    return _fail("energy", f"{cls.label()} is {law.energy_class.value}, and the {family.name} family states no exact "
                           f"identity for that class yet")


def _check_potential(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    node = probe.node
    scale = family.potential_scale
    if scale is None:
        return _fail("energy", f"the {family.name} family states no potential_scale to judge a potential law on")
    if not _states_energy(node):
        return _fail("energy", f"{probe.plan.owner}'s node states no energy (MimeNode.energy), and a potential law's "
                               f"class is its wrench against that energy's negative gradient")
    poses = [label for (_, interface), label in probe.reads_at.items() if interface == vocabulary.POSE.name]
    wrenches = [label for (_, interface), label in probe.delivers_at.items()
                if interface == vocabulary.APPLIED_WRENCH.name]
    if len(poses) != 1 or len(wrenches) != 1:
        return _fail("energy", f"{probe.plan.owner} reads {len(poses)} poses and delivers {len(wrenches)} wrenches, and "
                               f"the potential case judges one wrench against its energy at one pose")
    gradient = _energy_gradient(node)

    def holds(response: Response, *_: Any) -> bool:
        pose = jnp.asarray(response.read[poses[0]])
        constants = {key: jnp.asarray(value, dtype=pose.dtype) for key, value in
                     cast(Any, node)._step_constants(None).items()}
        expected = -np.asarray(gradient(constants, pose), dtype=np.float64)
        found = np.asarray(response.delivered[wrenches[0]], dtype=np.float64)
        force_scale, couple_scale = scale(response)
        floor = ENERGY_ULPS * _eps(pose.dtype)
        apart = (float(np.linalg.norm(found[:3] - expected[:3])), float(np.linalg.norm(found[3:] - expected[3:])))
        return apart[0] <= floor * force_scale and apart[1] <= floor * couple_scale

    return _sampled(probe, family, "energy", holds, max_examples)


def _states_energy(node: SimulationNode) -> bool:
    from mime.core import MimeNode

    return isinstance(node, MimeNode) and type(node).energy is not MimeNode.energy


def _energy_gradient(node: SimulationNode) -> Callable[[Mapping[str, Any], Any], Any]:
    """The gradient of ``node``'s energy at a pose, the position's and the rotation group's (the pose turned by
    ``exp(theta)`` in the world at ``theta = 0``), compiled once."""
    held = cast(Any, node)

    def gradient(p: Mapping[str, Any], pose: Any) -> Any:
        def energy(offset: Any) -> Any:
            moved = jnp.concatenate([pose[:3] + offset[:3],
                                     quaternion.multiply(quaternion.exponential(offset[3:]), pose[3:])])
            return held.energy(p, moved)

        return cast(Any, jax).grad(energy)(jnp.zeros(6, dtype=pose.dtype))

    return cast(Callable[[Mapping[str, Any], Any], Any], cast(Any, jax).jit(gradient))


def _check_dissipative(probe: Probe, law: Interaction, family: Family, max_examples: int) -> VerificationResult:
    operators = {label: interface for (_, interface), label in probe.delivers_at.items()
                 if _shape(interface) == (6, 6)}
    loads = {label: interface for (_, interface), label in probe.delivers_at.items() if _shape(interface) == (6,)}
    flows = {role: label for (role, interface), label in probe.reads_at.items() if interface == vocabulary.TWIST.name}
    roles = {label: role for (role, _), label in probe.delivers_at.items()}
    if not operators and not loads:
        return _fail("energy", f"{probe.plan.owner} delivers no operator and no load, and the dissipative case judges "
                               f"what a law delivers")

    def holds(response: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        for label in operators:
            if not bool(ledger.definite_at_floor(jnp.asarray(response.delivered[label]))):
                return False
        floors = _rounding(probe, state, boundary, dt) if loads else {}
        for label in loads:
            load = np.asarray(response.delivered[label], dtype=np.float64)
            flow_label = flows.get(roles[label])
            if flow_label is None:
                if not bool(np.all(np.abs(load) <= floors[probe.outputs[label]])):
                    return False
                continue
            flow = np.asarray(response.read[flow_label], dtype=np.float64)
            power = float(load @ flow)
            if power > ENERGY_ULPS * _eps(np.asarray(response.delivered[label]).dtype) * float(np.abs(load) @ np.abs(flow)):
                return False
        return True

    del law
    return _sampled(probe, family, "energy", holds, max_examples)


def _shape(interface: str) -> tuple[int, ...] | None:
    found = vocabulary.lookup(interface)
    return tuple(found.shape) if found is not None and found.shape is not None else None


def onto_the_surface(probe: Probe) -> Callable[[dict[str, Any]], dict[str, Any]] | None:
    """For a law that confines a body on a cylinder (its node's ``surface_shape``), a map of drawn states onto the
    shrunk surface (its radius less the body's extent and the margin), in the drawn direction about the axis, with
    the drawn speed along it and at least :data:`OUTWARD_SPEED` outward across it: where the law acts on almost every draw (see :data:`OUTWARD_SPEED`).
    ``None`` for another surface."""
    own = cast(Mapping[str, Any], cast(Any, probe.node).params)
    if own.get("surface_shape") != "cylinder":
        return None
    centre = np.asarray(own["surface_centre"], dtype=np.float64)
    axis = np.asarray(own["surface_axis"], dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    room = float(own["surface_radius"]) - float(own["extent"]) - float(own["margin"])

    def onto(drawn: dict[str, Any]) -> dict[str, Any]:
        pose, twist = np.asarray(drawn["pose"]), np.asarray(drawn["twist"])
        position = pose[:3].astype(np.float64) - centre
        radial = position - (position @ axis) * axis
        if float(np.linalg.norm(radial)) == 0.0:
            radial = np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0])
        normal = radial / np.linalg.norm(radial)
        placed = centre + (position @ axis) * axis + room * normal
        velocity = twist[:3].astype(np.float64)
        along = velocity - (velocity @ normal) * normal
        outward = OUTWARD_SPEED + abs(float(velocity @ normal))
        return {**drawn, "pose": np.concatenate([placed, pose[3:]]).astype(pose.dtype),
                "twist": np.concatenate([along + outward * normal, twist[3:]]).astype(twist.dtype)}

    return onto


def _check_held(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    """A dissipative law held in its receiver's step: the receiver's node metered for it alone, stepped once from
    each drawn state placed on the surface moving outward, counts no step in which the law gave energy
    (``ledger_gift_steps``)."""
    onto = onto_the_surface(probe)
    if onto is None:
        return _fail("energy", f"{probe.plan.owner} confines a body on a surface other than a cylinder, and the case "
                               f"places the body on its surface by the cylinder's closed form")
    node = probe.node
    delivery = ledger.LedgerDelivery("confinement", probe.plan.owner, ledger.DISSIPATIVE, ledger.HOSTED)
    held = cast(Any, node)
    metered = type(node)(node.name, float(held.delta_t), **dict(held.params), ledger=(delivery.encoded(),))

    def holds(_: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        stepped = _update(metered, ledger.with_records(cast(Any, metered), dict(state)), boundary, dt)
        return float(np.asarray(stepped[f"{ledger.RECORDS}gift_steps"])[0]) == 0.0

    return _sampled(probe, family, "energy", holds, max_examples, also_constrain_state=onto)


def conformance(law: Interaction, family: str | Family, *, oracle: Oracle | None = None,
                swap_with: Interaction | None = None, node_case: NodeCase | None = None,
                cases: Sequence[CaseKind] | None = None, max_examples: int = 200,
                together: Sequence[Interaction] = ()) -> dict[CaseKind, VerificationResult]:
    """Every case of ``family`` on ``law``, or the ones named in ``cases``. ``oracle`` is the law's closed form,
    ``swap_with`` the member a swap replaces (by default the family's reference), ``node_case`` how its nodes are
    verified, ``together`` the members that complete it in the closed-loop case (a magnetic torque's force).
    ``max_examples`` is each sampled case's draws, ``verify_node``'s default."""
    held = family_named(family) if isinstance(family, str) else family
    wanted = tuple(held.cases if cases is None else cases)
    unknown = [case for case in wanted if case not in held.cases]
    if unknown:
        raise ValueError(f"the {held.name} family holds no case {[c.value for c in unknown]}")
    probed = probe_law(law, held)
    if isinstance(probed, str):
        return {case: _fail(case.value, probed) for case in wanted}
    run: dict[CaseKind, Callable[[], VerificationResult]] = {
        CaseKind.SIGN: lambda: check_sign(probed, held, max_examples),
        CaseKind.FRAME: lambda: check_frame(probed, held, max_examples),
        CaseKind.UNITS: lambda: check_units(probed, law, held),
        CaseKind.MATERIAL_REACH: lambda: check_material_reach(probed, law, held),
        CaseKind.ORACLE: lambda: check_oracle(probed, held, oracle, max_examples),
        CaseKind.GRADIENTS: lambda: check_gradients(probed, held, max_examples),
        CaseKind.SWAP: lambda: check_swap(probed, held, swap_with),
        CaseKind.EQUAL_AND_OPPOSITE: lambda: check_equal_and_opposite(probed, held, max_examples),
        CaseKind.NODES: lambda: check_nodes(probed, law, node_case),
        CaseKind.CLOSED_LOOP: lambda: check_closed_loop(law, held, together, max_examples),
        CaseKind.ENERGY: lambda: check_energy(probed, law, held, max_examples),
    }
    return {case: run[case]() for case in wanted}



def assert_conforms(law: Interaction, family: str | Family, *, oracle: Oracle | None = None,
                    swap_with: Interaction | None = None, node_case: NodeCase | None = None,
                    cases: Sequence[CaseKind] | None = None, max_examples: int = 200,
                    together: Sequence[Interaction] = ()) -> None:
    """:func:`conformance` as an assertion, listing every case that did not pass with its counterexample."""
    results = conformance(law, family, oracle=oracle, swap_with=swap_with, node_case=node_case, cases=cases,
                          max_examples=max_examples, together=together)
    failed = [f"{case.value}: {result}" for case, result in results.items() if not result.passed]
    name = family if isinstance(family, str) else family.name
    assert not failed, f"{law.label()} does not conform to the {name} family:\n" + "\n".join(failed)
