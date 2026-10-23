"""Lag accounting: every edge is classified as reading this step, the previous step, or the current iterate inside a
coupling group, from MIME's own plan and canonical order, and the build checks the classification against what
MADDENING compiled. Script A's previous-step reads are those of its planned graph (the drag's and the torque's reads of
the body's pose); the build report and the graph snapshot show every edge's lag; an unplanned loop is refused.

The controls: a build that skips the canonical order changes the list, and the test notices; skipping the gathering of
a coupling group's members leaves the full toy with a load read a step late, which is refused; a body registered with
a body that comes after it by name is refused for the same reason; a classification planted wrong is refused by the
cross-check with MADDENING's schedule.
"""

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import pytest
from script_a import script_a

from mime.build import BuildRefused, EdgeLag, Lag, _construct, _plan, _testing, latency_in_steps
from mime.build._builders import COMMAND, DELIVERY, STATE
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.registry import (
    BuildContext,
    Dynamics,
    Interaction,
    Plan,
    PlanContext,
    Port,
    Registration,
    RoleKind,
    register_dynamics,
    register_interaction,
)
from mime.core import node_input
from mime.testing.toys import LOAD, POSITION, ToyBody, ToyBodyNode, coupled_toy, full_toy, toy_experiment
from mime.things import Body, State, ThingKind
from mime.units import LENGTH, VELOCITY, m, m_per_s
from mime.vocabulary import Interface, TargetKind, define, register_namespace

#: Script A's previous-step reads in its planned graph: each law reads the body's pose as it stood at the start of the
#: step.
SCRIPT_A_PREVIOUS_STEP = ["umr.pose -> drag:umr.pose", "umr.pose -> magnetic.torque:umr.pose"]


def test_script_as_previous_step_reads_are_its_planned_graphs():
    report = script_a().build().report
    assert report.previous_step() == SCRIPT_A_PREVIOUS_STEP
    assert {e.text: e.lag for e in report.lags} == {
        "drive:field.amplitude.value -> field.magnetic.amplitude": Lag.THIS_STEP,
        "drive:field.frequency.value -> field.magnetic.frequency": Lag.THIS_STEP,
        "field.magnetic.flux_density_magnetic_torque_umr -> magnetic.torque:umr.flux_density": Lag.THIS_STEP,
        "umr.pose -> magnetic.torque:umr.pose": Lag.PREVIOUS_STEP,
        "umr.pose -> drag:umr.pose": Lag.PREVIOUS_STEP,
        "drag:umr.resistance -> umr.resistance": Lag.THIS_STEP,
        "drag:umr.background_load -> umr.background_load": Lag.THIS_STEP,
        "gravity:umr.wrench -> umr.applied_wrench": Lag.THIS_STEP,
        "magnetic.torque:umr.wrench -> umr.applied_wrench": Lag.THIS_STEP,
        "umr.twist -> swim_speed:umr.twist": Lag.THIS_STEP,
    }
    assert all(edge.planned for edge in report.lags)


def test_the_report_and_the_snapshot_show_every_edges_lag():
    built = script_a().build()
    text = built.report.text()
    assert "a previous-step read is planned when a law, provider, dynamics or observation reads the published state" \
           in text
    assert "umr.pose -> magnetic.torque:umr.pose  previous step, planned" in text
    assert "umr.pose -> magnetic.torque:umr.pose  replacing  [m; 1 -> m; 1]  reads previous step" in built.snapshot()
    assert built.snapshot().count(" reads ") == len(built.graph.edges)


def test_inside_a_coupling_group_an_edge_reads_the_current_iterate():
    lags = {e.text: e.lag for e in full_toy().build().report.lags}
    assert lags["bath.flow.u -> toy.lift:probe.u"] is Lag.CURRENT_ITERATE
    assert lags["probe.x -> bath.flow.pos_probe"] is Lag.CURRENT_ITERATE
    assert lags["probe.x -> toy.spring:probe.x"] is Lag.PREVIOUS_STEP
    assert lags["toy.spring:probe.load -> probe.load"] is Lag.THIS_STEP


def test_without_the_canonical_order_the_list_changes_and_the_test_notices():
    """The control: declared in the script's order, the body is added before the laws, so its loops are closed by
    the torque's and the drag's deliveries, read a step late; the switch lifts the refusal of those unplanned reads to
    show the list."""
    with _testing.declaration_order():
        report = script_a().build().report
    assert report.previous_step() != SCRIPT_A_PREVIOUS_STEP
    assert report.previous_step() == LATE_DELIVERIES
    assert [e.planned for e in report.lags if e.lag is Lag.PREVIOUS_STEP] == [False, False, False]


#: In declaration order, the body comes before the laws, so each law's delivery into it closes its loop a step late.
LATE_DELIVERIES = ["drag:umr.resistance -> umr.resistance", "drag:umr.background_load -> umr.background_load",
                   "magnetic.torque:umr.wrench -> umr.applied_wrench"]


def test_an_unplanned_loop_is_refused_naming_the_edge():
    with _testing.declaration_order(refuse_unplanned=True), pytest.raises(BuildRefused) as refused:
        script_a().build()
    assert refused.value.codes() == ["lag.unplanned"] * 3
    assert "magnetic.torque:umr.wrench -> umr.applied_wrench, built by magnetic.torque:umr, carries a delivery" \
           in refused.value.refusals[2].message
    assert "drag:umr.resistance -> umr.resistance, built by drag:umr, carries a delivery" \
           in refused.value.refusals[0].message
    assert refused.value.report is not None and refused.value.report.previous_step() == LATE_DELIVERIES


def test_without_gathering_a_groups_members_the_full_toy_reads_a_load_a_step_late_and_is_refused(monkeypatch):
    """MADDENING runs a coupling group as one block at its first member's place: with the lift law (a member) added
    before the spring law and the body (a member) after it, the spring would run after the group and the body would
    read its load from the previous step. The canonical order gathers each group at its last member's place."""
    monkeypatch.setattr(_plan, "gathered", lambda order, groups: list(order))
    with pytest.raises(BuildRefused) as refused:
        full_toy().build()
    assert refused.value.codes() == ["lag.unplanned"]
    assert "toy.spring:probe.load -> probe.load" in refused.value.refusals[0].message


def test_a_classification_planted_wrong_is_refused_by_the_cross_check(monkeypatch):
    def flipped(order, edges, groups):
        found = classify(order, edges, groups)
        return tuple(EdgeLag(e.source, e.target, Lag.THIS_STEP, e.carries, e.by) if e.lag is Lag.PREVIOUS_STEP
                     else e for e in found)

    classify = _construct.classify
    monkeypatch.setattr(_construct, "classify", flipped)
    with pytest.raises(BuildRefused) as refused:
        coupled_toy().build()
    codes = set(refused.value.codes())
    assert codes == {"lag.cross-check"}
    messages = [r.message for r in refused.value.refusals]
    assert any("MADDENING reads probe.v -> toy.drag:probe.v from the previous step" in m for m in messages)
    assert any("MADDENING staggers the loop" in m for m in messages)


def test_the_cross_check_passes_on_every_toy_and_script_a():
    for make in (coupled_toy, full_toy, script_a):
        assert make().build().report.refusals == ()


# -- a registration that closes its loop with a delivery ----------------------------------------------------------

register_namespace("lagged", "0.0.1")
PULL = define(Interface("lagged.pull", TargetKind.REGISTRATION, "", None, (),
                        description="a body registered with another body's dynamics"))


class _PullingNode(ToyBodyNode):
    """A toy body that also reads another body's position and loads it towards its own."""

    def __init__(self, name, timestep, other="", **params):
        super().__init__(name, timestep, **params)
        self.params["other"] = other

    def boundary_input_spec(self):
        return {**super().boundary_input_spec(), f"pos_{self.params['other']}": node_input("m", (3,))}

    def initial_state(self):
        return {**super().initial_state(), f"pull_{self.params['other']}": jnp.zeros(3)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        other = self.params["other"]
        own = {k: v for k, v in state.items() if k in ("x", "v")}
        stepped = super().update(own, boundary_inputs, dt, params=params)
        pull = stepped["x"] - boundary_inputs.get(f"pos_{other}", jnp.zeros(3))
        return {**stepped, f"pull_{other}": pull}


@register_dynamics
@dataclass(frozen=True, kw_only=True)
class PullingBody(Dynamics):
    name = "lagged.pulling_body"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    applies_to = frozenset({ThingKind.BODY})
    publishes = {POSITION.name: Port("x")}
    accepts = {LOAD.name: Port("load"), PULL.name: Port("registrations")}
    registration_kinds = frozenset({"lagged.pulled"})
    initial_state = {"position": LENGTH, "velocity": VELOCITY}

    def build(self, ctx: BuildContext, b: Any, registrations: Any) -> None:
        (registration,) = registrations
        node = b.add_node(_PullingNode(ctx.node_name(), ctx.timestep, other=registration.subject,
                                       initial_position=ctx.initial["position"],
                                       initial_velocity=ctx.initial["velocity"]))
        b.connect(b.read(registration, "position"), into=node[f"pos_{registration.subject}"])
        b.connect(node[f"pull_{registration.subject}"], into=b.target(registration, "load"))


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Pull(Interaction):
    """Registers ``body`` with ``puller``'s dynamics, which reads its position and loads it."""

    name = "lagged.pull"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY, "puller": RoleKind.BODY}
    id_roles = ("body", "puller")

    def plan(self, ctx: PlanContext) -> Plan:
        return Plan(nodes=(), registrations=(Registration("puller", PULL.name, "lagged.pulled", "body",
                                                          reads={"position": POSITION.name},
                                                          returns={"load": LOAD.name}),))

    def build(self, ctx: BuildContext, b: Any) -> None:
        return None


def pulled(puller: str, body: str) -> Experiment:
    experiment = toy_experiment("pulled")
    start = State(position=m((0.1, 0.0, 0.0)), velocity=m_per_s((0.0, 0.0, 0.0)))
    experiment.add_body(Body(puller, dynamics=PullingBody(), initial=start))
    experiment.add_body(Body(body, dynamics=ToyBody(), initial=start))
    experiment.attach(Pull(), body=body, puller=puller)
    return experiment


def test_a_registration_that_delivers_to_a_body_added_before_it_is_refused_and_one_after_it_builds():
    """Bodies are added by name. Registered with ``b_puller``, the body ``a_body`` is added first, so the puller
    reads its position this step and it reads the pull a step late: an unplanned loop. Named the other way round,
    the pull is read this step and the puller reads the position at the start of the step: planned."""
    with pytest.raises(BuildRefused) as refused:
        pulled("b_puller", "a_body").build()
    assert refused.value.codes() == ["lag.unplanned"]
    assert "b_puller.pull_a_body -> a_body.load" in refused.value.refusals[0].message
    report = pulled("a_puller", "b_body").build().report
    assert report.previous_step() == ["b_body.x -> a_puller.pos_b_body"]


# -- latency ---------------------------------------------------------------------------------------------------------


def test_latency_counts_the_previous_step_reads_along_a_path():
    """What a controller's realised latency will be computed with in phase 7: a path from a thing's state to a
    command, one base step for each edge on it that reads the previous step."""
    read = EdgeLag(("umr", "pose"), ("observer", "pose"), Lag.THIS_STEP, STATE, "observe:umr")
    decide = EdgeLag(("observer", "estimate"), ("controller", "estimate"), Lag.THIS_STEP, STATE, "controller")
    write = EdgeLag(("controller", "command"), ("arm", "torque"), Lag.PREVIOUS_STEP, COMMAND, "controller")
    assert latency_in_steps([read, decide, write]) == 1
    assert latency_in_steps([read, decide]) == 0
    with pytest.raises(ValueError, match="not consecutive"):
        latency_in_steps([read, write])
    drives = [e for e in script_a().build().report.lags if e.carries == COMMAND]
    assert [latency_in_steps([e]) for e in drives] == [0, 0]
    assert all(e.carries in (STATE, DELIVERY, COMMAND, "field") for e in script_a().build().report.lags)
