"""Family suites: the cases every law of a family passes, as harnesses a lab calls from its own tests
(``assert_conforms(MyDrag(), family="hydrodynamic", oracle=...)``); passing in the lab's own CI is the claim.

A family states what its laws must satisfy, over the vocabulary's interfaces rather than any one law's labels: its
sign rule, the settings that stand for a material's properties, the pairs a two-way law delivers equal and opposite,
a reference member a swap replaces, and a rig, the smallest experiment that holds one of its laws. Each case is
checked on the law as the rig builds it. The sampled cases (sign, frame, oracle, gradients, equal and opposite) are
invariants of MADDENING's ``verify_node`` on the law's own node, each drawn and shrunk by Hypothesis; units,
material reach and swap build the rig again; nodes runs MADDENING's whole battery on every node the law builds.
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
from mime.build import BuildRefused, EntryPlan
from mime.build._maddening import node_update
from mime.build._report import graph_config
from mime.experiment import BuiltExperiment, Experiment
from mime.registry import Interaction, registration_of
from mime.testing._nodes import NodeCase, computing_dtype, node_failures, run_verify_node
from mime.units import TIME, Magnitude, Quantity

#: ``jax.numpy``'s hints name a device type from jaxlib's compiled core, which ships no stubs.
jnp: Any = jax.numpy


def _update(node: SimulationNode, state: Mapping[str, Any], inputs: Mapping[str, Any], dt: float) -> dict[str, Any]:
    """``node.update``, whose MADDENING annotation takes bare dicts."""
    return cast(dict[str, Any], node_update(node)(dict(state), dict(inputs), dt))


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
    """One evaluation of a law: what it read and what it delivered, by label, with its settings in SI; ``reading``
    and ``delivery`` look them up by role and interface."""

    read: Mapping[str, Array]
    delivered: Mapping[str, Array]
    settings: Mapping[str, Magnitude]
    reads_at: Mapping[At, str]
    delivers_at: Mapping[At, str]

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

    ``rig(law)`` is the smallest experiment holding ``law``. ``material`` names the settings that stand for a
    material's properties, each of which must reach the law's output. ``pairs`` are the deliveries a two-way law
    makes equal and opposite. ``reference()`` is the member a swap replaces. ``reading_bounds`` bounds the draws of
    each read interface (MADDENING's default box is (-1e4, 1e4))."""

    name: str
    rig: Callable[[Interaction], Experiment]
    cases: tuple[CaseKind, ...]
    sign: SignRule | None = None
    material: tuple[str, ...] = ()
    pairs: tuple[tuple[At, At], ...] = ()
    reference: Callable[[], Interaction] | None = None
    reading_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict[str, tuple[float, float]])


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


@dataclass(frozen=True)
class Probe:
    """A law as its family's rig built it: its plan, its one node, and how its labels map onto that node's input
    ports and state fields."""

    built: BuiltExperiment
    plan: EntryPlan
    node: SimulationNode
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]
    reads_at: Mapping[At, str]
    delivers_at: Mapping[At, str]

    def response(self, boundary: Mapping[str, Any], state: Mapping[str, Any]) -> Response:
        return Response({label: boundary[port] for label, port in self.inputs.items() if port in boundary},
                        {label: state[f] for label, f in self.outputs.items()}, self.plan.settings, self.reads_at,
                        self.delivers_at)

    def respond(self, read: Mapping[str, Any]) -> dict[str, Array]:
        """What the law delivers, by label, given what it reads, by label, from its node's initial state."""
        state = _update(self.node, _initial(self.node), {self.inputs[label]: v for label, v in read.items()},
                        self.built.plans.dt)
        return {label: state[f] for label, f in self.outputs.items()}


def _planned(built: BuiltExperiment, law: Interaction) -> EntryPlan | None:
    return next((p for p in built.plans.entries if p.entry is law), None) or next(
        (p for p in built.plans.entries if p.entry == law), None)


def probe_law(law: Interaction, family: Family) -> Probe | str:
    """``law`` built in ``family``'s rig, or why it cannot be probed: its rig does not build, or it builds other
    than one node (the sampled cases evaluate a law built as one node)."""
    try:
        built = family.rig(law).build()
    except BuildRefused as refused:
        return f"the {family.name} family's rig does not build with {law.label()}: {refused}"
    plan = _planned(built, law)
    if plan is None:
        return f"the {family.name} family's rig does not hold {law.label()}"
    names = plan.node_names()
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


def _sampled(probe: Probe, family: Family, name: str, predicate: Callable[[Response, Mapping[str, Any], float], bool],
             max_examples: int) -> VerificationResult:
    """``predicate(response, boundary, dt)`` as an invariant of the law's node, under ``verify_node``'s draws."""
    bounds = {probe.inputs[label]: family.reading_bounds[interface] for (_, interface), label in probe.reads_at.items()
              if interface in family.reading_bounds and label in probe.inputs}

    def invariant(state: dict[str, Any], out: dict[str, Any], boundary: dict[str, Any], dt: float) -> bool:
        return predicate(probe.response(boundary, out), boundary, dt)

    results = run_verify_node(probe.node, checks=[], invariants={name: invariant}, boundary_bounds=bounds or None,
                           max_examples=max_examples, dtype=computing_dtype(probe.node))
    return results[name]


def _fail(name: str, detail: str) -> VerificationResult:
    return VerificationResult(name, "FAIL", detail=detail)


def _close(found: Any, expected: Any, rtol: float, floor: float = 0.0) -> bool:
    """``found`` within ``rtol`` of ``expected``'s largest component, plus ``floor``, in the largest component. The
    tolerance has no absolute part of its own: an SI torque of 1e-12 N m is not zero."""
    found, expected = np.asarray(found, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if found.shape != expected.shape:
        return False
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    return float(np.max(np.abs(found - expected), initial=0.0)) <= rtol * scale + floor


def _rounding(probe: Probe, boundary: Mapping[str, Any], dt: float) -> float:
    """How far what the law delivers moves when each component it reads moves by 64 units in its last place, in
    alternating directions: the scale below which two evaluations of the law that order their arithmetic otherwise
    cannot be told apart, cancellation included. Never below the dtype's own resolution near its smallest normal
    number (``16 tiny / eps``, about 1e-30 in float32), where XLA flushes subnormal results to zero and a turned or
    reordered evaluation can lose every digit."""
    info = np.finfo(computing_dtype(probe.node))
    eps, flushed = float(info.eps), 16 * float(info.tiny) / float(info.eps)
    ports = set(probe.inputs.values())
    nudged: dict[str, Any] = {}
    for index, (port, value) in enumerate(boundary.items()):
        held = np.asarray(value)
        signs = np.resize([1.0, -1.0] if index % 2 == 0 else [-1.0, 1.0], held.shape)
        nudged[port] = held + 64 * eps * np.abs(held) * signs if port in ports else held
    initial = _initial(probe.node)
    base, moved = _update(probe.node, initial, boundary, dt), _update(probe.node, initial, nudged, dt)
    return max((float(np.max(np.abs(np.asarray(moved[f]) - np.asarray(base[f])), initial=0.0))
                for f in probe.outputs.values()), default=0.0) + flushed


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
    return _sampled(probe, family, "sign", lambda response, _, __: rule.holds(response), max_examples)


def _rotations() -> list[np.ndarray]:
    """The proper rotations a frame case turns the world by: a quarter turn about z, a third of a turn about
    (1, 1, 1), and one about no particular axis."""
    def about(axis: Sequence[float], angle: float) -> np.ndarray:
        k = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
        cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * cross @ cross

    return [about((0.0, 0.0, 1.0), math.pi / 2), about((1.0, 1.0, 1.0), 2 * math.pi / 3),
            about((0.3, -0.5, 0.8), 0.7)]


def _world_vector(interface: str) -> bool:
    found = vocabulary.lookup(interface)
    return found is not None and found.frame == "world" and found.shape == (3,)


def check_frame(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    """Turning the world turns what the law delivers: for each world-frame vector it reads turned by ``R``, each
    world-frame vector it delivers is turned by ``R``."""
    turned_in = [probe.inputs[label] for (_, interface), label in probe.reads_at.items()
                 if _world_vector(interface) and label in probe.inputs]
    turned_out = [probe.outputs[label] for (_, interface), label in probe.delivers_at.items()
                  if _world_vector(interface)]
    if not turned_out:
        return _fail("frame", f"{probe.plan.owner} delivers no world-frame vector, so turning the world tests nothing")
    rotations, rtol = _rotations(), _tolerance(probe)

    def turns(response: Response, boundary: Mapping[str, Any], dt: float) -> bool:
        out = {probe.outputs[label]: value for label, value in response.delivered.items()}
        floor = 4 * _rounding(probe, boundary, dt)
        for rotation in rotations:
            r = jnp.asarray(rotation, dtype=jnp.asarray(out[turned_out[0]]).dtype)
            given = {port: (r @ value if port in turned_in else value) for port, value in boundary.items()}
            state = _update(probe.node, _initial(probe.node), given, dt)
            if not all(_close(state[f], r @ out[f], rtol, floor) for f in turned_out):
                return False
        return True

    return _sampled(probe, family, "frame", turns, max_examples)


def check_oracle(probe: Probe, family: Family, oracle: Oracle | None, max_examples: int) -> VerificationResult:
    if oracle is None:
        return _fail("oracle", f"the {family.name} family holds every law to its closed form: pass oracle=, a "
                               f"function from its readings and settings in SI to what it delivers, by label")
    rtol = _tolerance(probe)

    def agrees(response: Response, boundary: Mapping[str, Any], dt: float) -> bool:
        expected = oracle(response.read, response.settings)
        floor = 4 * _rounding(probe, boundary, dt)
        return set(expected) == set(response.delivered) and all(
            _close(response.delivered[label], expected[label], rtol, floor) for label in expected)

    return _sampled(probe, family, "oracle", agrees, max_examples)


def check_gradients(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    """The derivative of what the law delivers along a direction in what it reads, by forward-mode differentiation,
    is finite and agrees with a central finite difference: within the difference's own truncation error, estimated
    from the difference at half the step (Richardson), so a point where the law is not smooth asks only what a
    finite difference can tell there."""
    ports = list(probe.inputs.values())
    if not ports:
        return _fail("gradients", f"{probe.plan.owner} reads nothing, so it has no derivative to check")
    step, rtol = (1e-6, 1e-8) if _wide(probe) else (1e-2, 1e-4)
    smallest = float(np.finfo(computing_dtype(probe.node)).tiny) ** 0.5

    def agrees(response: Response, boundary: Mapping[str, Any], dt: float) -> bool:
        dtype = jnp.asarray(boundary[ports[0]]).dtype
        direction = {p: jnp.asarray(np.resize([0.6, -0.8, 0.3, 0.5], np.shape(boundary[p])), dtype=dtype)
                     for p in ports}
        scale = max(float(jnp.max(jnp.abs(jnp.asarray(boundary[p])))) for p in ports)
        h = step * max(scale, smallest)
        initial = _initial(probe.node)

        def delivered(eps: Any) -> Array:
            given = {**boundary, **{p: jnp.asarray(boundary[p], dtype=dtype) + eps * direction[p] for p in ports}}
            state = _update(probe.node, initial, given, dt)
            return jnp.concatenate([jnp.ravel(state[f]) for f in probe.outputs.values()])

        def difference(at: float) -> np.ndarray:
            return np.asarray((delivered(jnp.asarray(at, dtype=dtype)) - delivered(jnp.asarray(-at, dtype=dtype)))
                              / (2 * at), dtype=np.float64)

        _, tangent = _jvp(delivered, (jnp.asarray(0.0, dtype=dtype),), (jnp.asarray(1.0, dtype=dtype),))
        coarse, fine = difference(h), difference(h / 2)
        truncation = float(np.max(np.abs(coarse - fine), initial=0.0))
        floor = 2 * truncation + 4 * _rounding(probe, boundary, dt) / h
        return bool(jnp.all(jnp.isfinite(tangent))) and _close(tangent, fine, rtol, floor)

    return _sampled(probe, family, "gradients", agrees, max_examples)


def check_equal_and_opposite(probe: Probe, family: Family, max_examples: int) -> VerificationResult:
    if not family.pairs:
        return _fail("equal and opposite", f"the {family.name} family lists this case and names no pair")
    missing = [at for pair in family.pairs for at in pair if at not in probe.delivers_at]
    if missing:
        return _fail("equal and opposite", f"{probe.plan.owner} delivers no {missing}, which the family pairs")
    rtol = _tolerance(probe)

    def balanced(response: Response, boundary: Mapping[str, Any], dt: float) -> bool:
        floor = 4 * _rounding(probe, boundary, dt)
        for a, b in family.pairs:
            first, second = np.asarray(response.delivery(*a)), np.asarray(response.delivery(*b))
            if not _close(first + second, np.zeros_like(first), 0.0, rtol * float(np.max(np.abs(first))) + floor):
                return False
        return True

    return _sampled(probe, family, "equal and opposite", balanced, max_examples)


def _another_dimension(value: object, dimension: Any) -> Quantity:
    magnitude = value.value if isinstance(value, Quantity) else 1.0
    return Quantity(magnitude, dimension * TIME)


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
        wrong = _another_dimension(getattr(law, name, None), declared.dimension)
        try:
            family.rig(dataclasses.replace(law, **{name: wrong})).plan()
        except BuildRefused as refused:
            if not any(r.code == "setting.units" and f"setting {name!r}" in r.message for r in refused.refusals):
                problems.append(f"setting {name} given {wrong} is refused for other reasons only: {refused.codes()}")
        else:
            problems.append(f"setting {name} given {wrong}, of another dimension than {declared.dimension}, plans")
    if problems:
        return _fail("units", "; ".join(problems))
    return VerificationResult("units", "PASS")


def _probe_reading(probe: Probe) -> dict[str, Array]:
    spec = probe.node.boundary_input_spec()
    return {label: jnp.full(tuple(getattr(spec[port], "shape", (3,))), 0.5) for label, port in probe.inputs.items()}


def check_material_reach(probe: Probe, law: Interaction, family: Family) -> VerificationResult:
    """Each setting the family names as a material's property reaches what the law delivers: the same reading gives
    another response once the setting is changed by half."""
    problems: list[str] = []
    reading = _probe_reading(probe)
    before = probe.respond(reading)
    for name in family.material:
        if not hasattr(law, name):
            problems.append(f"{law.label()} has no setting {name!r}, which the {family.name} family names as a "
                            f"material's property")
            continue
        value: object = getattr(law, name)
        changed = value * 1.5 if isinstance(value, (Quantity, float, int)) else value
        other = probe_law(dataclasses.replace(law, **{name: changed}), family)
        if isinstance(other, str):
            problems.append(other)
        elif all(_close(other.respond(reading)[label], before[label], 0.0) for label in before):
            problems.append(f"setting {name} changed from {value} to {changed} leaves the law's response unchanged: it "
                            f"does not reach {probe.plan.owner}'s output")
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
