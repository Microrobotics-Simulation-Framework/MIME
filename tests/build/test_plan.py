"""Phase 1: declarations resolved and checked with no graph, and every refusal collected, each with its class and
action line."""

from dataclasses import dataclass
from typing import Any

import pytest

from mime.build import BuildRefused, RefusalClass, derive_seed
from mime.experiment import Experiment
from mime.gradients import GradientStatus
from mime.registry import (
    Delivery,
    Interaction,
    Observation,
    Plan,
    Read,
    RoleKind,
    register_interaction,
    register_observation,
)
from mime.testing import graph_snapshot
from mime.testing.toys import (
    HOLD_Z,
    PROBE_START,
    PUSH,
    ToyAnchoredBody,
    ToyBody,
    ToyCouple,
    ToyDrag,
    ToyField,
    ToyHold,
    ToyLift,
    ToyRecorder,
    ToyRelaxation,
    ToySpring,
    ToyTwist,
    coupled_toy,
    toy_experiment,
)
from mime.things import Apparatus, Body, Medium, State
from mime.units import Hz, m, m_per_s2, mm, ms, s, standard_gravity
from mime.vocabulary import register_namespace

register_namespace("planning", "0.0.1")


@register_interaction
@dataclass(frozen=True, kw_only=True)
class PinZ(Interaction):
    """A second delivery into the exclusive target."""

    name = "planning.pin_z"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}
    delivers = {"hold_z": Delivery("body", HOLD_Z.name)}


@register_interaction
@dataclass(frozen=True, kw_only=True)
class Jitter(Interaction):
    """Declares a random stream and builds nothing."""

    name = "planning.jitter"
    version = "0.0.1"
    differentiability = GradientStatus.FIXED_KEY
    roles = {"body": RoleKind.BODY}

    def plan(self, ctx):
        return Plan(nodes=(), streams=("kicks",))

    def build(self, ctx, b):
        return None


@register_observation
@dataclass(frozen=True, kw_only=True)
class FieldProbe(Observation):
    name = "planning.field_probe"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY, "field": RoleKind.FIELD}
    reads = {"push": Read("field", PUSH.name, at=Read("body", "toy.body.position"))}
    body: Any = None
    field: Any = None


@dataclass(frozen=True, kw_only=True)
class Unregistered(Interaction):
    name = "planning.unregistered"
    version = "0.0.1"
    differentiability = GradientStatus.EXACT
    roles = {"body": RoleKind.BODY}


def probe_and_bath(experiment: Experiment, flow=True):
    bath = experiment.add_medium(Medium("bath", flow=ToyRelaxation() if flow else None))
    probe = experiment.add_body(Body("probe", dynamics=ToyBody(), initial=PROBE_START))
    return probe, bath


def refused(experiment: Experiment) -> BuildRefused:
    with pytest.raises(BuildRefused) as caught:
        experiment.plan()
    return caught.value


def refusal(experiment: Experiment, code: str):
    found = [r for r in refused(experiment).refusals if r.code == code]
    assert found, f"no {code} among {refused(experiment).codes()}"
    return found[0]


def test_declaring_and_planning_touch_no_graph(monkeypatch):
    class NoGraph:
        def __init__(self):
            raise AssertionError("a graph was constructed")

    monkeypatch.setattr("mime.build._construct.GraphManager", NoGraph)
    experiment = coupled_toy()
    experiment.plan()
    with pytest.raises(AssertionError, match="a graph was constructed"):
        experiment.build()


def test_every_reason_is_refused_at_once():
    experiment = Experiment("many", up=(0, 0, 1), gravity=standard_gravity())
    probe, bath = probe_and_bath(experiment)
    experiment.attach(ToySpring(), body="prbe")
    experiment.attach(ToyDrag(), body=probe)
    experiment.attach(ToyLift(), body=bath, medium=bath)
    experiment.attach(ToyTwist(), body=probe)
    error = refused(experiment)
    assert {"run.dt-missing", "binding.unknown", "rung.missing", "binding.wrong-kind",
            "vocabulary.reserved"} <= set(error.codes())
    assert all(r.action for r in error.refusals)
    assert error.report is not None and error.report.refusals == error.refusals


def test_a_misspelt_name_suggests_the_candidates():
    experiment = toy_experiment()
    probe_and_bath(experiment)
    experiment.attach(ToySpring(), body="prbe")
    found = refusal(experiment, "binding.unknown")
    assert found.category is RefusalClass.USER
    assert "Did you mean 'probe'?" in found.action and "The candidates are: probe." in found.action


def test_a_misspelt_typed_path_is_an_error_on_its_line():
    experiment = toy_experiment()
    source = experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    assert str(source.push) == "source.push"
    with pytest.raises(AttributeError, match="Did you mean 'push'"):
        source.psh  # noqa: B018


def test_a_misspelt_field_given_as_a_string_suggests_the_candidates():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    experiment.attach(ToyHold(), body=probe, field="source.psh")
    assert "Did you mean 'source.push'?" in refusal(experiment, "binding.unknown").action


def test_nothing_is_bound_by_default_even_with_one_candidate():
    experiment = toy_experiment()
    probe_and_bath(experiment)
    experiment.attach(ToySpring())
    assert refusal(experiment, "binding.missing").action == "bind body= one of: probe"


def test_an_unknown_role_suggests_the_roles():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(ToySpring(), body=probe, bdy=probe)
    assert "Did you mean 'body'?" in refusal(experiment, "binding.unknown-role").action


def test_a_binding_of_the_wrong_kind_is_refused():
    experiment = toy_experiment()
    _, bath = probe_and_bath(experiment)
    experiment.attach(ToySpring(), body=bath)
    assert "'bath' is a medium" in refusal(experiment, "binding.wrong-kind").message


def test_no_rung_is_chosen_by_default():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(ToyDrag(), body=probe)
    assert refusal(experiment, "rung.missing").action == "write the rung: rung= one of 'linear', 'quadratic'"


def test_an_unknown_rung_suggests_the_rungs():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(ToyDrag(rung="quadratc"), body=probe)
    assert "Did you mean 'quadratic'?" in refusal(experiment, "rung.unknown").action


def test_an_alias_is_the_rung_setting():
    assert ToyDrag.Quadratic(c=0.5) == ToyDrag(rung="quadratic", c=0.5)


def test_an_unsupported_construction_is_refused_with_the_supported_ones():
    experiment = toy_experiment()
    probe = experiment.add_body(Body("probe", dynamics=ToyAnchoredBody(), initial=State(position=m((0, 0, 0)))))
    experiment.attach(ToyDrag.Quadratic(), body=probe)
    found = refusal(experiment, "support.unsupported")
    assert "toy.anchored_body 0.1.0 (flags: none)" in found.message
    assert "rung quadratic: body needs flag inertial and accepts toy.body.load (qualifying: toy.body 0.1.0)" \
        in found.action


def test_a_reserved_name_is_refused_as_reserved_not_built():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(ToyTwist(), body=probe)
    found = refusal(experiment, "vocabulary.reserved")
    assert "reserved, not built" in found.message and "load" in found.message


def test_an_unregistered_entry_is_refused():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(Unregistered(), body=probe)
    assert "register_interaction" in refusal(experiment, "entry.unregistered").action


def test_a_read_nothing_publishes_and_a_registration_nothing_accepts_are_refused():
    experiment = toy_experiment()
    probe, bath = probe_and_bath(experiment, flow=False)
    experiment.attach(ToyLift(), body=probe, medium=bath)
    experiment.attach(ToyCouple(), body=probe, medium=bath)
    error = refused(experiment)
    assert {"read.not-published", "registration.not-accepted"} <= set(error.codes())
    assert "toy.relaxation 0.1.0" in refusal(experiment, "registration.not-accepted").action


def test_an_exclusive_target_with_two_deliveries_is_refused_in_planning():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    source = experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    experiment.attach(ToyHold(), body=probe, field=source.push)
    experiment.attach(PinZ(), body=probe)
    assert "planning.pin_z:probe, toy.hold:probe" in refusal(experiment, "target.exclusive").message


def test_ids_are_names_and_bindings_and_a_duplicate_is_refused():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    assert experiment.attach(ToySpring(), body=probe) == "toy.spring:probe"
    assert experiment.attach(ToySpring(k=2.0), body="probe") == "toy.spring:probe"
    assert refusal(experiment, "id.duplicate").category is RefusalClass.USER


def test_a_handle_and_a_name_bind_the_same():
    by_handle = toy_experiment("same")
    probe, bath = probe_and_bath(by_handle)
    by_handle.attach(ToyLift(), body=probe, medium=bath)
    by_name = toy_experiment("same")
    probe_and_bath(by_name)
    by_name.attach(ToyLift(), body="probe", medium="bath")
    assert graph_snapshot(by_handle.build()) == graph_snapshot(by_name.build())


def test_two_things_with_one_name_are_refused():
    experiment = toy_experiment()
    probe_and_bath(experiment)
    experiment.add_medium(Medium("probe"))
    assert refusal(experiment, "thing.duplicate").message.startswith("two things are named 'probe'")


def test_the_initial_state_is_checked_against_the_dynamics():
    experiment = toy_experiment()
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=State(positon=m((0, 0, 0)))))
    assert "Did you mean 'position'?" in refusal(experiment, "state.unknown").action
    assert "position, velocity" in refusal(experiment, "state.missing").message
    experiment = toy_experiment()
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=State(position=s(1.0), velocity=m((0, 0, 0)))))
    assert refused(experiment).codes() == ["state.units", "state.units"]


def test_a_setting_of_the_wrong_dimension_is_refused_at_plan_time_naming_the_setting_and_both_dimensions():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(ToySpring(x0=s((0.05, -0.02, 0.01))), body=probe)
    found = refusal(experiment, "setting.units")
    print(f"\n{found}")
    assert found.category is RefusalClass.USER
    assert found.message == ("interaction toy.spring:probe (toy.spring 0.1.0): setting 'x0' needs units of length (m); "
                             "it was given time (s) ((0.05, -0.02, 0.01) s)")
    assert found.action == "write x0 as a quantity of length (m)"
    # The control: the same setting in metres plans.
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    experiment.attach(ToySpring(x0=m((0.05, -0.02, 0.01))), body=probe)
    experiment.plan()


def test_every_kind_of_entry_has_its_settings_checked_and_a_bare_number_is_refused():
    experiment = toy_experiment()
    experiment.add_medium(Medium("bath", flow=ToyRelaxation(relax_rate=s(2.1))))
    experiment.add_body(Body("probe", dynamics=ToyBody(mass=1.3), initial=PROBE_START))
    experiment.add_apparatus(Apparatus("source", fields={"push": ToyField(wavenumber=m(1.3))}))
    experiment.attach(ToyHold(gain=m(0.5)), body="probe", field="source.push")
    messages = [r.message for r in refused(experiment).refusals if r.code == "setting.units"]
    assert messages == [
        "medium 'bath''s dynamics (toy.relaxation 0.1.0): setting 'relax_rate' needs units of frequency (Hz); it was "
        "given time (s) (2.1 s)",
        "body 'probe''s dynamics (toy.body 0.1.0): setting 'mass' needs units of mass (kg); it was given the bare "
        "value 1.3",
        "field 'push' of apparatus 'source' (toy.field 0.1.0): setting 'wavenumber' needs units of m^-1; it was given "
        "length (m) (1.3 m)",
        "interaction toy.hold:probe (toy.hold 0.1.0): setting 'gain' needs units of velocity (m/s); it was given "
        "length (m) (0.5 m)",
    ]


def test_the_entry_builds_its_nodes_from_its_settings_in_si():
    experiment = coupled_toy()
    experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    experiment.attach(ToyHold(gain=mm(500.0) / s(1.0)), body="probe", field="source.push")
    plans = experiment.plan()
    (hold,) = [p for p in plans.entries if p.owner == "toy.hold:probe"]
    assert hold.settings == {"gain": pytest.approx(0.5)}
    (spring,) = [p for p in plans.entries if p.owner == "toy.spring:probe"]
    assert spring.settings == {"k": 1.7, "x0": (0.05, -0.02, 0.01)}
    params = experiment.build().graph.get_node("toy.hold:probe").params
    assert params["gain"] == pytest.approx(0.5) and isinstance(params["gain"], float)


def test_the_world_frame_has_no_default_axes():
    experiment = Experiment("frame", dt=ms(20))
    assert {"frame.up-missing", "frame.gravity-missing"} <= set(refused(experiment).codes())
    experiment = Experiment("frame", up=(0, 0, 0), gravity=m(9.8), dt=ms(20))
    assert {"frame.up-invalid", "frame.gravity-units"} <= set(refused(experiment).codes())


def test_gravity_is_derived_along_minus_up():
    plans = Experiment("frame", up=(0, 2, 0), gravity=m_per_s2(9.0), dt=ms(20)).plan()
    assert plans.frame.up == (0.0, 1.0, 0.0) and plans.frame.gravity == (-0.0, -9.0, -0.0)


@pytest.mark.parametrize(("dt", "code"), [(None, "run.dt-missing"), (m(1), "run.dt-units"),
                                          (0.02, "run.dt-units"), (s(-1), "run.dt-invalid")])
def test_dt_has_no_default_and_is_a_positive_time(dt, code):
    assert code in refused(Experiment("dt", up=(0, 0, 1), gravity=standard_gravity(), dt=dt)).codes()


def test_a_rate_that_is_not_a_whole_multiple_of_dt_names_the_nearest():
    experiment = coupled_toy()
    experiment.observe(ToyRecorder(body="probe", rate=ms(30)))
    found = refusal(experiment, "run.rate-not-multiple")
    assert found.action.endswith("the nearest are 0.02 s and 0.04 s")


def test_a_rate_may_be_a_period_or_a_frequency():
    for rate in (ms(40), Hz(25)):
        experiment = coupled_toy()
        experiment.observe(ToyRecorder(body="probe", rate=rate))
        built = experiment.build()
        (line,) = [n for n in built.report.nodes if n.owner == "toy.recorder:probe"]
        assert line.divider == 2 and line.timestep == 0.04


def test_a_rate_in_other_units_is_refused():
    experiment = coupled_toy()
    experiment.observe(ToyRecorder(body="probe", rate=m(1)))
    assert "run.rate-units" in refused(experiment).codes()


def test_some_node_runs_at_dt():
    experiment = toy_experiment()
    experiment.add_body(Body("probe", dynamics=ToyBody(rate=ms(40)), initial=PROBE_START))
    assert refusal(experiment, "run.no-node-at-dt").action == "run some entry at dt, or raise dt to 0.04 s"


def test_overlapping_requests_merge_into_one_group_and_must_agree():
    experiment = toy_experiment()
    probe, bath = probe_and_bath(experiment)
    experiment.attach(ToyCouple(solve_together=True), body=probe, medium=bath)
    experiment.attach(ToyLift(solve_together=True), body=probe, medium=bath)
    (group,) = experiment.plan().groups
    assert group.nodes == ("bath.flow", "probe", "toy.lift:probe")
    assert group.tolerance == 1e-12 and group.max_iterations == 50
    experiment = toy_experiment()
    probe, bath = probe_and_bath(experiment)
    experiment.attach(ToyCouple(solve_together=True), body=probe, medium=bath)
    experiment.attach(ToyLift(solve_together=True, tolerance=1e-10), body=probe, medium=bath)
    assert "group.settings" in refused(experiment).codes()


def test_a_group_shares_one_rate():
    experiment = toy_experiment()
    bath = experiment.add_medium(Medium("bath", flow=ToyRelaxation()))
    probe = experiment.add_body(Body("probe", dynamics=ToyBody(rate=ms(40)), initial=PROBE_START))
    experiment.attach(ToyCouple(solve_together=True), body=probe, medium=bath)
    assert "group.rates" in refused(experiment).codes()


def test_an_observation_reading_a_field_is_refused_until_it_has_its_own_provider():
    experiment = toy_experiment()
    probe, _ = probe_and_bath(experiment)
    source = experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    experiment.observe(FieldProbe(body=probe, field=source.push))
    assert "observation.field-read" in refused(experiment).codes()


def test_a_field_nothing_reads_builds_no_node():
    experiment = coupled_toy()
    experiment.add_apparatus(Apparatus("source", fields={"push": ToyField()}))
    assert "source.push" not in graph_snapshot(experiment.build())


def test_seeds_derive_from_the_stream_identity_and_the_experiment_seed():
    experiment = coupled_toy()
    experiment.attach(Jitter(), body="probe")
    seeds = experiment.build().report.seeds
    assert seeds == {"planning.jitter:probe": {"kicks": derive_seed(0, "planning.jitter:probe", {"body": "probe"},
                                                                    "kicks")}}
    alone = toy_experiment()
    probe_and_bath(alone)
    alone.attach(Jitter(), body="probe")
    assert alone.build().report.seeds == seeds
    reseeded = derive_seed(1, "planning.jitter:probe", {"body": "probe"}, "kicks")
    assert reseeded != seeds["planning.jitter:probe"]["kicks"]
