"""Family suites: the cases every law of a family passes, as harnesses a lab calls from its own tests
(``assert_conforms(MyDrag(), family="hydrodynamic", oracle=...)``); passing in the lab's own CI is the claim.

A family states what its laws must satisfy, over the vocabulary's interfaces rather than any one law's labels: its
sign rule, the settings that stand for a material's properties, the pairs a two-way law delivers equal and opposite,
a reference member a swap replaces, and a rig, the smallest experiment that holds one of its laws. Each case is
checked on the law as the rig builds it. The sampled cases (sign, frame, oracle, gradients, equal and opposite) are
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
from mime.experiment import BuiltExperiment, Experiment
from mime.registry import Interaction, registration_of, setting_value
from mime.testing._nodes import NodeCase, computing_dtype, node_failures, run_verify_node
from mime.units import TIME, Magnitude, Quantity

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
    proportional to (a resistance to the medium's viscosity): changed by half, the response changes by half."""

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


def _sampled(probe: Probe, family: Family, name: str,
             predicate: Callable[[Response, Mapping[str, Any], Mapping[str, Any], float], bool],
             max_examples: int) -> VerificationResult:
    """``predicate(response, state, boundary, dt)`` as an invariant of the law's node, under ``verify_node``'s
    draws, each drawn value mapped by the family's ``constrain``."""
    ports, fields = _interface_of_ports(probe), _interface_of_state(probe)
    bounds = {port: family.reading_bounds[interface] for port, interface in ports.items()
              if interface in family.reading_bounds}
    state_bounds = {f: family.reading_bounds[interface] for f, interface in fields.items()
                    if interface in family.reading_bounds}
    constrain_boundary = _mapped({port: family.constrain[i] for port, i in ports.items() if i in family.constrain})
    constrain_state = _mapped({f: family.constrain[i] for f, i in fields.items() if i in family.constrain})

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


def check_oracle(probe: Probe, family: Family, oracle: Oracle | None, max_examples: int) -> VerificationResult:
    if oracle is None:
        return _fail("oracle", f"the {family.name} family holds every law to its closed form: pass oracle=, a "
                               f"function from its readings and settings in SI to what it delivers, by label")
    rtol = _tolerance(probe)

    def agrees(response: Response, state: Mapping[str, Any], boundary: Mapping[str, Any], dt: float) -> bool:
        expected = oracle(response.read, response.settings)
        floors = _rounding(probe, state, boundary, dt)
        return set(expected) == set(response.delivered) and all(
            _close_by_component(response.delivered[label], expected[label], rtol, 4 * floors[probe.outputs[label]],
                                _reading_floor(probe, response, label),
                                output_parts(probe, probe.outputs[label], np.shape(response.delivered[label])))
            for label in expected)

    return _sampled(probe, family, "oracle", agrees, max_examples)


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


def conformance(law: Interaction, family: str | Family, *, oracle: Oracle | None = None,
                swap_with: Interaction | None = None, node_case: NodeCase | None = None,
                cases: Sequence[CaseKind] | None = None, max_examples: int = 200) -> dict[CaseKind, VerificationResult]:
    """Every case of ``family`` on ``law``, or the ones named in ``cases``. ``oracle`` is the law's closed form,
    ``swap_with`` the member a swap replaces (by default the family's reference), ``node_case`` how its nodes are
    verified. ``max_examples`` is each sampled case's draws, ``verify_node``'s default."""
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
    }
    return {case: run[case]() for case in wanted}


def assert_conforms(law: Interaction, family: str | Family, *, oracle: Oracle | None = None,
                    swap_with: Interaction | None = None, node_case: NodeCase | None = None,
                    cases: Sequence[CaseKind] | None = None, max_examples: int = 200) -> None:
    """:func:`conformance` as an assertion, listing every case that did not pass with its counterexample."""
    results = conformance(law, family, oracle=oracle, swap_with=swap_with, node_case=node_case, cases=cases,
                          max_examples=max_examples)
    failed = [f"{case.value}: {result}" for case, result in results.items() if not result.passed]
    name = family if isinstance(family, str) else family.name
    assert not failed, f"{law.label()} does not conform to the {name} family:\n" + "\n".join(failed)
