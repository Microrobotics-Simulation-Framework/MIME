"""A command port's numbers as a study's axes (``coils.ports.currents.high``): its bounds whole, one side, or its slew,
and one element by its index where it declares one per element. The stage holds each as a leaf, and so does a
controller inside the compiled step whose allocation reads it (``port_coils__currents__high``), one axis writing both,
so a sweep is one program and a derivative in the bound passes through the stage and the allocation alike.

**The rigs** (``believed_cases``): the millimetre sphere in Navion_2's field, its currents through the coils' port.
``allocating``: :class:`BelievedPD` inside the compiled step, allocating a PD force within the port's bounds, which
at the drawn bound (0.3 to 0.6 A, against the 25 A the case declares) saturates. ``driven``: no controller, the
currents a constant 1 A in the first coil, clipped by the stage at the drawn bound (0.3 to 0.8 A). The objective is the
sphere's final z (the axis its pull moves least, so its own size is near its change: a final x or y is a hundred times
its change, which float32's rounding would swamp). 30 base steps, six controller ticks.

**Gate 1, the derivative against central differences whose own error is read from the step sweep.** Each dtype is
judged on its own: the gradient ``g`` (``Study.gradient`` of the run in that dtype) against the central difference
``D(h) = (J(t (1 + h)) - J(t (1 - h))) / (2 h t)``, ``t`` the bound, at the ladder ``h = 1e-1 .. 1e-7``. The difference's
own error at ``h_k`` is its truncation, estimated by the next rung, ``T_k = |D(h_k) - D(h_k+1)|`` (the truncation falls as
``h^2``, a hundredfold a rung, so the difference holds it to 1 %), plus its rounding, ``R_k = 64 eps |J| / (h_k t)`` (an
evaluation of ``J`` rounded to 64 ``eps`` of itself, the MRS gradient oracle's allowance, over the step). Its floor is
``F = min_k (T_k + R_k)``, at ``h*``, and the line is ``|g - D(h*)| <= 4 F``: four times what the difference itself
cannot resolve. Derived before any run (``predictions_g14a.json``); the line's size against ``|g|`` is the gate's
discriminating range, reported per draw. **The allocator's share** (the derivative through the controller's leaf
alone, the stage's held) is at least a tenth of ``g`` on the allocating rig, so a controller reading a constant is far
outside the line.

**Gate 2, the control that must give zero.** At 200 A no coil reaches its bound (checked on the run's writes), so the
derivative in the bound is exactly zero in both dtypes; at the drawn bound it is not.

**The plants** (``MIME_PLANT_PORT``; each must fail its gate on every run, measured with
``.github/probe/detection_rate.py``, the bound drawn from ``MIME_PORT_SEED``): ``constant``, the allocation reading the
port's declared number, a constant of the law, not the leaf (gate 1, allocating); ``flip_controller`` and ``flip_stage``,
the derivative through the controller's or the stage's leaf of the bound negated, its value kept (gate 1, allocating
and driven); ``one_leaf_controller`` and ``one_leaf_stage``, an axis's derivative and a run's override setting only the
stage's leaf or only the controller's (the test that every path setting a bound sets both, and gate 1)."""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import jax
import numpy as np
import pytest
from believed_cases import REASON, BelievedPD, believed_sphere
from iterative_cases import iterative_coils

import mime.study._fit as fit_module
import mime.study._gradient as gradient_module
import mime.study._override as override_module
from mime.build import ArgumentRefused
from mime.control import _models
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.nodes.control._ports import CommandStageNode
from mime.study import Study, along
from mime.study._axes import PRECISION, STUDY_REASON, AxisKind, base_value, candidates, declared, resolve
from mime.study._point import precision
from mime.things import CommandPort
from mime.units import A, ms, s

AXIS = "coils.ports.currents.high"
STAGE, CONTROLLER = "ports:coils", "controller"
STAGE_LEAF, CONTROLLER_LEAF = "high_currents", "port_coils__currents__high"
DURATION = ms(30)
LADDER = tuple(10.0 ** -k for k in range(1, 8))
#: The rounding of one evaluation of the objective, in eps of its size (the MRS gradient oracle's allowance).
ROUNDING = 64.0
#: The line, in floors of the central difference.
FLOORS = 4.0
#: The allocator's least share of the derivative on the allocating rig.
SHARE = 0.10
UNSATURATED = 200.0
DTYPES = ("float64", "float32")

PLANT = os.environ.get("MIME_PLANT_PORT", "")
SEED = int(os.environ.get("MIME_PORT_SEED", "0"))
_draw = np.random.default_rng(SEED)
#: The draws: the allocating rig's bound and the driven rig's, both well below what the loop asks of a coil.
ALLOCATING_BOUND, DRIVEN_BOUND = float(_draw.uniform(0.3, 0.6)), float(_draw.uniform(0.3, 0.8))


def z(final: Mapping[str, Any]) -> Any:
    return final["sphere"]["pose"][2]


def allocating(bound: float, name: str = "port_allocating", **port: Any) -> Experiment:
    return believed_sphere(name, port=CommandPort(bounds=A(bound), reason=REASON, **port))


def high_only(base: float, high: float, name: str, mode: str = "graph") -> Experiment:
    """The allocating rig as a point of ``.high`` at ``high`` writes it: its low bound the base's, ``-base``."""
    return believed_sphere(name, port=CommandPort(bounds=(A(-base), A(high)), reason=REASON), mode=mode)


def driven(bound: float, name: str = "port_driven", **port: Any) -> Experiment:
    return believed_sphere(name, port=CommandPort(bounds=A(bound), reason=REASON, **port), controller=False)


# -- the plants ---------------------------------------------------------------------------------------------------------


def _flipped(value: Any) -> Any:
    """``value`` itself, its derivative negated."""
    return 2.0 * jax.lax.stop_gradient(value) - value


def _is_controllers(node: str, leaf: str) -> bool:
    return node == CONTROLLER and leaf.startswith("port_")


def _is_stages(node: str, leaf: str) -> bool:
    return node.startswith("ports:") and leaf.startswith(("low_", "high_", "slew_"))


def _one_leaf(monkeypatch: pytest.MonkeyPatch, dropped: Callable[[str, str], bool]) -> None:
    """Every axis's derivative and every run's override set one of the two leaves of a port's number, the other held at
    the build's value."""
    held_written, held_overridden = gradient_module.written, override_module.overridden

    def written(*args: Any, **kwargs: Any) -> Any:
        found = held_written(*args, **kwargs)
        kept = tuple(pair for pair in found.leaves if not dropped(*pair))
        return dataclasses.replace(found, leaves=kept)

    def overridden(built: Any, overrides: Any) -> Any:
        found = held_overridden(built, overrides)
        base = override_module.base_point(built).tree
        nodes = {node: {leaf: (base[node][leaf] if dropped(node, leaf) else value) for leaf, value in leaves.items()}
                 for node, leaves in found.params["nodes"].items()}
        return dataclasses.replace(found, params={**found.params, "nodes": nodes})

    monkeypatch.setattr(gradient_module, "written", written)
    monkeypatch.setattr(fit_module, "written", written)
    monkeypatch.setattr(override_module, "overridden", overridden)


def plant(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    if name == "constant":
        monkeypatch.setattr(_models.ControlModels, "_port_number",
                            lambda self, port, number: np.asarray(getattr(port, number)))
    elif name == "flip_controller":
        held = _models.ControlModels._port_number
        monkeypatch.setattr(_models.ControlModels, "_port_number",
                            lambda self, port, number: _flipped(held(self, port, number)))
    elif name == "flip_stage":
        bound = CommandStageNode._bound

        def flipped(self: Any, key: str, value: Any, p: Mapping[str, Any], *rest: Any) -> Any:
            return bound(self, key, value, {**p, **{f"{side}_{key}": _flipped(p[f"{side}_{key}"])
                                                    for side in ("low", "high") if f"{side}_{key}" in p}}, *rest)

        monkeypatch.setattr(CommandStageNode, "_bound", flipped)
    elif name == "one_leaf_controller":
        _one_leaf(monkeypatch, _is_controllers)
    elif name == "one_leaf_stage":
        _one_leaf(monkeypatch, _is_stages)
    elif name:
        raise ValueError(f"no plant {name!r}")


@pytest.fixture(autouse=True)
def planted(monkeypatch: pytest.MonkeyPatch) -> None:
    plant(monkeypatch, PLANT)


# -- the axis ----------------------------------------------------------------------------------------------------------


def declarations_of(experiment: Experiment) -> Any:
    return experiment.declared_as()[1]


def test_a_ports_declared_numbers_are_axes_listed_and_reached_by_did_you_mean():
    slewed = declarations_of(allocating(0.5, slew=A(4000.0) / s(1.0)))
    offered = [path for path in candidates(slewed) if ".ports." in path]
    assert offered == [f"coils.ports.currents.{number}" for number in ("bounds", "low", "high", "slew")]
    assert [path for path in candidates(declarations_of(allocating(0.5))) if ".ports." in path] == \
        [f"coils.ports.currents.{number}" for number in ("bounds", "low", "high")]
    axis = resolve(AXIS, slewed)
    assert (axis.kind, axis.owner, axis.name, axis.component, axis.built) == \
        (AxisKind.PORT, "coils", "currents.high", None, False)
    with pytest.raises(ArgumentRefused) as raised:
        resolve("coils.ports.curents.high", slewed)
    (refusal,) = raised.value.refusals
    assert refusal.code == "study.unknown-path" and "'coils.ports.currents.high'" in refusal.action


@pytest.mark.parametrize("path", ["coils.ports.currents.high", "coils.ports.currents.slew"])
def test_a_number_the_port_does_not_declare_is_refused_with_the_route_to_declare_it(path):
    """Bounds the receiver states (none here, the coils state none) are not the port's, and a slew it leaves out is a rung
    the stage does not build: neither is an axis."""
    held = declarations_of(believed_sphere("port_latency_only", port=CommandPort(latency=ms(1), reason=REASON)))
    with pytest.raises(ArgumentRefused) as raised:
        resolve(path, held)
    (refusal,) = raised.value.refusals
    assert refusal.code == "study.port-undeclared"
    assert "CommandPort(" in refusal.action and "ports={'currents'" in refusal.action
    assert not [p for p in candidates(held) if ".ports." in p]


def test_one_element_is_an_axis_where_the_port_declares_one_per_element_and_refused_where_it_declares_one_for_all():
    settings, per_element, _ = believed_sphere("port_per_element", port=CommandPort(
        bounds=(A((-0.5, -0.5, -0.5)), A((0.5, 0.6, 0.7))), reason=REASON)).declared_as()
    axis = resolve(f"{AXIS}[1]", per_element)
    assert (axis.kind, axis.component, axis.built) == (AxisKind.PORT, 1, False)
    assert base_value(axis, settings, per_element) == A(0.6)
    written = declared(settings, per_element, {axis: A(0.65)}, STUDY_REASON)
    port = next(d.thing.ports["currents"] for d in written.declarations if getattr(d, "thing", None) is not None
                and d.thing.name == "coils")
    assert port.bounds == (A((-0.5, -0.5, -0.5)), A((0.5, 0.65, 0.7)))
    for path, code in ((f"{AXIS}[1]", "study.component-path"), ("coils.ports.currents.bounds[0]", "study.component-path")):
        with pytest.raises(ArgumentRefused) as raised:
            resolve(path, declarations_of(allocating(0.5)))
        assert [r.code for r in raised.value.refusals] == [code]


def test_a_point_writes_the_number_with_the_ports_reason_and_its_own_and_the_plan_judges_it():
    experiment = allocating(0.5)
    settings, held, _ = experiment.declared_as()
    axis = resolve(AXIS, held)
    written = declared(settings, held, {axis: A(0.4)}, STUDY_REASON, {AXIS: "the amplifier's datasheet"})
    port = next(d.thing.ports["currents"] for d in written.declarations if getattr(d, "thing", None) is not None
                and d.thing.name == "coils")
    assert port.bounds == (-A(0.5), A(0.4))
    assert port.reason == f"{REASON}; {AXIS} = {A(0.4)}: the amplifier's datasheet"
    whole = declared(settings, held, {resolve("coils.ports.currents.bounds", held): A(0.3)}, STUDY_REASON)
    assert next(d.thing.ports["currents"] for d in whole.declarations if getattr(d, "thing", None) is not None
                and d.thing.name == "coils").bounds == A(0.3)
    plan = Study(experiment).sweep(along({AXIS: [A(-0.6), A(0.4)]}), duration=DURATION).plan()
    assert [[r.code for r in point.refusals] for point in plan.points] == [["port.declaration"], []]


def test_a_range_of_a_ports_number_is_declared_as_any_axis_is():
    experiment = allocating(0.5)
    experiment.declare_range(AXIS, A(0.3), A(0.7), reason="the amplifiers a study compares")
    plans = experiment.plan()
    assert AXIS in plans.ranges and plans.ranges[AXIS].kind == "port"
    plan = Study(experiment).sweep(along({AXIS: [A(0.4), A(0.6), A(0.8)]}), duration=DURATION).plan()
    assert not plan.refused and [point.declared.outside for point in plan.points] == [(), (), (AXIS,)]


# -- one program, both leaves ------------------------------------------------------------------------------------------


def test_the_controller_holds_the_bound_it_allocates_within_as_a_leaf_beside_the_stages():
    built = allocating(0.5).build()
    params = built.graph.params["nodes"]
    assert {leaf for leaf in params[CONTROLLER] if leaf.startswith("port_")} == \
        {"port_coils__currents__low", "port_coils__currents__high"}
    for side in ("low", "high"):
        assert float(params[CONTROLLER][f"port_coils__currents__{side}"]) == float(params[STAGE][f"{side}_currents"])
    # a controller called on the host is no node: it makes its law from each run's declared bound
    assert CONTROLLER not in believed_sphere("port_host", mode="callback").build().graph.params["nodes"]


def test_a_sweep_of_the_bound_is_one_group_and_one_compile_and_each_member_is_its_own_build():
    values = [0.3, 0.4, 0.5]
    study = Study(allocating(0.5)).sweep(along({AXIS: [A(v) for v in values], PRECISION: ["float64"] * 3}),
                                         duration=DURATION)
    plan = study.plan()
    assert len(plan.groups) == 1 and plan.compiles == 1, plan.text()
    result = study.run()
    with precision(True):
        for index, value in enumerate(values):
            own = np.asarray(high_only(0.5, value, f"own_{index}").build().run(DURATION).final["sphere"]["pose"])
            found = np.asarray(result.member(index).final["sphere"]["pose"])
            assert np.allclose(found, own, rtol=1e-12, atol=1e-18), (value, found - own)


def test_every_path_that_sets_a_ports_bound_sets_the_stages_leaf_and_the_controllers():
    """A study's point, a run's override, and what an axis writes for a derivative or a fit (``written``): each sets
    both leaves to the value, and the override runs as the experiment built at the value does."""
    value = 0.9 * ALLOCATING_BOUND
    with precision(True):
        experiment = allocating(ALLOCATING_BOUND)
        study = Study(experiment).sweep(along({AXIS: [A(value)]}), duration=DURATION)
        plan = study.plan()
        tree = plan.points[0].tree
        assert (float(tree[STAGE][STAGE_LEAF]), float(tree[CONTROLLER][CONTROLLER_LEAF])) == (value, value)
        built = experiment.build()
        nodes = override_module.overridden(built, {AXIS: A(value)}).params["nodes"]
        assert (float(nodes[STAGE][STAGE_LEAF]), float(nodes[CONTROLLER][CONTROLLER_LEAF])) == (value, value)
        settings, held, _ = experiment.declared_as()
        found = gradient_module.written(resolve(AXIS, held), plan.points[0], settings, held, plan.groups[0].built,
                                        lambda key: None, None, None)
        assert {(STAGE, STAGE_LEAF), (CONTROLLER, CONTROLLER_LEAF)} <= set(found.leaves), found.leaves
        overridden = np.asarray(built.run(DURATION, override={AXIS: A(value)}).final["sphere"]["pose"])
        own = np.asarray(high_only(ALLOCATING_BOUND, value, "port_own").build().run(DURATION).final["sphere"]["pose"])
    assert np.allclose(overridden, own, rtol=1e-12, atol=1e-18), overridden - own


def test_as_a_callback_each_member_makes_its_law_from_its_own_bound():
    """On the host the law is made per member, from the point's declared bound, as built; no leaf holds it."""
    values = [0.3, 0.5]
    with precision(True):
        experiment = believed_sphere("port_callback", port=CommandPort(bounds=A(0.5), reason=REASON), mode="callback")
        result = Study(experiment).sweep(along({AXIS: [A(v) for v in values]}), duration=DURATION).run()
        for index, value in enumerate(values):
            own = high_only(0.5, value, f"port_callback_{index}", "callback").build().run(DURATION).final["sphere"]["pose"]
            assert np.allclose(np.asarray(result.member(index).final["sphere"]["pose"]), np.asarray(own), rtol=1e-12,
                               atol=1e-18)


# -- gate 1: the derivative against central differences ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Gate 1 at one bound in one dtype: the gradient, the central differences of the ladder, the floor's rung, the
    floor and the line, and the objective there."""

    gradient: float
    differences: tuple[float, ...]
    best: int
    floor: float
    line: float
    objective: float

    @property
    def miss(self) -> float:
        return abs(self.gradient - self.differences[self.best])

    @property
    def resolution(self) -> float:
        """The line against the derivative: a relative error in it below this is not seen."""
        return self.line / abs(self.gradient)


def ladder(experiment: Experiment, bound: float, dtype: str) -> Verdict:
    """Gate 1's measurement: the ladder's points as one study (one program), and the gradient at the bound."""
    values = [bound, *(bound * (1 + sign * h) for h in LADDER for sign in (1.0, -1.0))]
    study = Study(experiment).sweep(along({AXIS: [A(v) for v in values], PRECISION: [dtype] * len(values)}),
                                    duration=DURATION)
    result = study.run()
    found = [float(z(result.member(i).final)) for i in range(len(values))]
    gradient = float(np.asarray(study.gradient(z, wrt=[AXIS]).member(0).gradient[AXIS]))
    return verdict(gradient, found, bound, dtype)


def verdict(gradient: float, found: list[float], bound: float, dtype: str) -> Verdict:
    """The floor and the line from the ladder (the module's docstring): ``found`` the objective at the bound, then at
    ``bound (1 + h)`` and ``bound (1 - h)`` for each ``h`` of :data:`LADDER`."""
    differences = tuple((found[1 + 2 * k] - found[2 + 2 * k]) / (2 * h * bound) for k, h in enumerate(LADDER))
    eps = float(np.finfo(np.dtype(dtype)).eps)
    floors = [abs(differences[k] - differences[k + 1]) + ROUNDING * eps * abs(found[0]) / (h * bound)
              for k, h in enumerate(LADDER[:-1])]
    best = int(np.argmin(floors))
    return Verdict(gradient, differences, best, floors[best], FLOORS * floors[best], found[0])


@contextmanager
def leaves_kept(kept: Callable[[str, str], bool]) -> Iterator[None]:
    """The derivative through the leaves ``kept`` alone, the others held: one leaf's share."""
    with pytest.MonkeyPatch.context() as patch:
        held = gradient_module.written

        def written(*args: Any, **kwargs: Any) -> Any:
            found = held(*args, **kwargs)
            return dataclasses.replace(found, leaves=tuple(pair for pair in found.leaves if kept(*pair)))

        patch.setattr(gradient_module, "written", written)
        yield


def share(experiment: Experiment, bound: float, dtype: str, kept: Callable[[str, str], bool]) -> float:
    with leaves_kept(kept):
        study = Study(experiment).sweep(along({AXIS: [A(bound)], PRECISION: [dtype]}), duration=DURATION)
        return float(np.asarray(study.gradient(z, wrt=[AXIS]).member(0).gradient[AXIS]))


def saturated(experiment: Experiment, bound: float, dtype: str, field: tuple[str, str]) -> float:
    """The largest share of the bound a coil is delivered over the run (one where a coil sits at its bound)."""
    with precision(dtype == "float64"):
        history = experiment.build().run(DURATION, history=True).history
    assert history is not None
    return float(np.max(np.abs(np.asarray(history[field[0]][field[1]])))) / bound


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate1_the_derivative_through_the_stage_and_the_allocation_meets_central_differences(dtype):
    bound = ALLOCATING_BOUND
    assert saturated(allocating(bound), bound, dtype, (STAGE, "currents")) >= 1 - 1e-6, "no coil reaches its bound"
    found = ladder(allocating(bound), bound, dtype)
    assert found.miss <= found.line, found
    allocator = share(allocating(bound), bound, dtype, _is_controllers)
    assert abs(allocator) >= SHARE * abs(found.gradient), (allocator, found.gradient)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate1_the_derivative_through_the_stage_and_the_iterative_allocation_meets_central_differences(dtype, monkeypatch):
    """The combination of G14a and AP4a: the allocating rig with the exact path's size patched below its three coils
    (``iterative_coils``), so ``ctx.allocator`` takes the iterative path, whose implicit derivative passes the
    controller's leaf of the bound through ``_bounds``' bound tangents. The line is gate 1's, from the step sweep's own
    floor; the allocator's share is at least a tenth of the derivative, as on the exact path."""
    bound = ALLOCATING_BOUND
    with iterative_coils(monkeypatch):
        assert "iterative" in "\n".join(allocating(bound, "port_iterative_report").build().report.controller)
        assert saturated(allocating(bound, "port_iterative_a"), bound, dtype, (STAGE, "currents")) >= 1 - 1e-6, \
            "no coil reaches its bound"
        found = ladder(allocating(bound, "port_iterative_b"), bound, dtype)
        assert found.miss <= found.line, found
        allocator = share(allocating(bound, "port_iterative_c"), bound, dtype, _is_controllers)
    assert abs(allocator) >= SHARE * abs(found.gradient), (allocator, found.gradient)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate1_the_derivative_through_the_stage_alone_meets_central_differences(dtype):
    bound = DRIVEN_BOUND
    assert saturated(driven(bound), bound, dtype, (STAGE, "currents")) >= 1 - 1e-6, "the stage clips nothing"
    found = ladder(driven(bound), bound, dtype)
    assert found.miss <= found.line, found


@pytest.mark.parametrize("dtype", DTYPES)
def test_gate2_the_derivative_in_a_bound_no_coil_reaches_is_zero_and_not_where_one_does(dtype):
    assert saturated(allocating(UNSATURATED), UNSATURATED, dtype, (STAGE, "currents")) < 0.5, "a coil reached 200 A"
    for bound, zero in ((UNSATURATED, True), (ALLOCATING_BOUND, False)):
        study = Study(allocating(bound)).sweep(along({AXIS: [A(bound)], PRECISION: [dtype]}), duration=DURATION)
        gradient = float(np.asarray(study.gradient(z, wrt=[AXIS]).member(0).gradient[AXIS]))
        assert (gradient == 0.0) is zero, (bound, gradient)


def test_a_trip_declares_the_jump_of_its_derivative_in_the_bound():
    """A port that trips latches its rest from the first step beyond its bound: the derivative in the bound is a jump
    there, declared in the result's path, never a silent zero."""
    experiment = believed_sphere("port_trip", port=CommandPort(bounds=A(0.5), on_bound="trip", reason=REASON),
                                 controller=False)
    result = Study(experiment).sweep(along({AXIS: [A(0.5)]}), duration=DURATION).gradient(z, wrt=[AXIS])
    assert result.path[STAGE] is GradientStatus.PIECEWISE_JUMP and result.status is GradientStatus.PIECEWISE_JUMP
    clipped = Study(driven(0.5)).sweep(along({AXIS: [A(0.5)]}), duration=DURATION).gradient(z, wrt=[AXIS])
    assert clipped.path[STAGE] is GradientStatus.PIECEWISE_KINK


def test_a_slew_is_an_axis_the_stage_and_the_allocations_box_read():
    """The slew's leaf in both: the stage's ``slew_currents`` and the controller's, from which its box is made in the
    step."""
    slew = A(40.0) / s(1.0)
    experiment = believed_sphere("port_slewed", port=CommandPort(bounds=A(25.0), slew=slew, reason=REASON),
                                 controller=BelievedPD(smooth=1.0))
    plan = Study(experiment).sweep(along({"coils.ports.currents.slew": [slew, A(60.0) / s(1.0)]}),
                                   duration=DURATION).plan()
    assert len(plan.groups) == 1 and plan.compiles == 1
    for point, value in zip(plan.points, (40.0, 60.0), strict=True):
        assert (float(point.tree[STAGE]["slew_currents"]), float(point.tree[CONTROLLER]["port_coils__currents__slew"])) \
            == (value, value)

