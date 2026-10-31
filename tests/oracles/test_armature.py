"""Joint armature on a dynamic chain (phase 7a5, design #193): a geared joint's drive reflects its motor's rotor onto the
joint, ``N^2 J`` through a reduction ``N``, which its description's links do not hold. Each oracle with a control that
fails on every draw or run; float64, the nodes as the build makes them, stepped as the graph steps them (the host
first, on the motor's reaction and momentum of the step before; then the motor on the host's new frames).

- **Zero is bitwise**: an arm given a zero armature on every joint builds the graph of one given none (its derived
  hash, which holds every node's arrays); the arm's, the motor's and the chain's golden outputs are unchanged
  (``tests/saved/test_golden_outputs.py``). The control: an armature of 1e-12 kg m^2 on one joint changes the hash.
- **The hand model**: a rotor spun up on one joint along its axis, the joint held by nothing, turns it at ``q_dot =
  -J Omega (e . a) / (I_joint + a)`` (the axial momentum ``(I + a) q_dot + J Omega`` stays at zero), written out here
  from the hub's own inertia in its file; the node holds it to the step-late drive reaction, ``dt tau``, at every step
  of a drawn run. The controls: the armature left out of the node's mass matrix, and counted twice.
- **The transmission read**: the hub of ``geared_joint.urdf`` (reduction 20) and the slide of ``geared_slide.urdf`` (a
  lead screw of 5 mm), their actuators' rotors given, reflect ``N^2 J`` (kg m^2 on the hub, kg on the slide); the
  slide pushed by a force accelerates at ``F / (m + N^2 J)``. The control: the reduction used as ``N``, not ``N^2``.
- **The check predicts the kick**: on A0's AR4, held by nothing (no gravity, no joint torques), its motor turned 0.3
  rad on the flange and spun up from rest, every joint's speed over the rotor's spin, measured at the end of a run,
  against the build's ratios, over six armatures (ratios on the wrist from 4.1 down to 0.01). The controls: the check
  reading the bare inertia while the node carries the armature, and the reverse.
- **The build's line, the backstop and the controllers**: the validity line names the joint, the ratio, the speed it
  predicts and ``armature=``, in every profile, never a refusal; the velocity-limit flag after a run; ``ctx.models``
  carries the armature, so a computed torque acts on the joint's whole inertia.

The energy balance with armature is ``test_dynamic_arm.py``'s, over the branched tree's revolute and sliding joints
with their armatures; the spinner's jointed-child reference with armature on its hosts is ``test_spinner_tree.py``'s.
"""

import math
import warnings
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import AR4_HOME, GEARED_JOINT, GEARED_SLIDE, ROTOR_INERTIA, SPIN_JOINT, TOP_SPEED, add_body, \
    armed_hub, dynamic_rpm, experiment, free_ar4, geared_slide
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.build import BuildRefused
from mime.build._kicks import KICK_RATIO
from mime.core.multibody import Transmission, inverse_dynamics, read_urdf
from mime.drives import Constant
from mime.dynamics import Arm
from mime.experiment import Experiment
from mime.scene import asset
from mime.things import Apparatus
from mime.units import kg, kg_m2, ms, rad, s

PROPERTY = settings(deadline=None, max_examples=100)
EPS = float(np.finfo(np.float64).eps)
ROUNDINGS = 256
#: The hubs' inertia about their joints (kg m^2): ``spin_joint.urdf``'s and ``geared_joint.urdf``'s hub, its izz, its
#: centre of mass on the joint.
HUB_INERTIA = 2.43e-6
#: ``geared_joint.urdf``'s reduction; ``geared_slide.urdf``'s, a 5 mm lead screw's 2 pi / 0.005 rad/m, and its
#: carriage's mass (kg).
HUB_REDUCTION, SCREW_REDUCTION, CARRIAGE_MASS = 20.0, 2.0 * math.pi / 0.005, 0.5
#: The registered actuator's rotor inertia, from StepperOnline's drawing of the 17HS19-2004S1: 82 g cm^2.
STEPPER_ROTOR = 82.0e-7
DT = 1e-3

_CACHE: dict[tuple, object] = {}


def built(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.build()


def planned(experiment):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return experiment.plan()


def initial(node):
    key = ("initial", id(node))
    if key not in _CACHE or _CACHE[key][0] is not node:
        _CACHE[key] = (node, node.initial_state())
    return _CACHE[key][1]


def stepped(arm, motor, pa, pm, torque, steps, start):
    """The host and its motor from ``start`` (their states where the build starts them), stepped as the graph steps
    them: each step's joint velocities and the rotor's absolute spin."""
    dofs = len(arm.constants()["dof_names"])

    def step(held, _):
        host, rotor = held
        host = arm.advance(pa, host, {"joint_torques": jnp.zeros(dofs), "link_wrench": rotor["reaction"],
                                      "link_momentum": rotor["momentum"]}, DT)
        rotor = motor.advance(pm, rotor, {"torque": torque, "mount_poses": host["frame_poses"],
                                          "mount_rates": host["frame_angular_velocities"]}, DT)
        return (host, rotor), (host["joint_velocities"], rotor["spin"])

    _, history = jax.lax.scan(step, start, None, length=steps)
    return history


def compiled_run(arm, motor, steps):
    """:func:`stepped` compiled once per pair of nodes and length, its constants and the drive traced: nodes of
    another build of the same tree take it with their own constants."""
    key = ("run", id(arm), id(motor), steps)
    if key not in _CACHE:
        compiled = jax.jit(lambda pa, pm, torque, start: stepped(arm, motor, pa, pm, torque, steps, start))
        _CACHE[key] = compiled
    return _CACHE[key]


def starts(arm, motor):
    return initial(arm), initial(motor)


def nodes(graph, owner):
    return graph.graph.get_node(f"{owner}.arm"), graph.graph.get_node(f"{owner}.motor")


def test_the_descriptions_hold_what_the_oracles_write_out():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hubs = [read_urdf(path).tree for path in (SPIN_JOINT, GEARED_JOINT)]
        slide = read_urdf(GEARED_SLIDE).tree
    for tree in hubs:
        held = tree.body("hub").inertial
        assert held is not None and held.inertia[2][2] == HUB_INERTIA and held.centre_of_mass == (0.0, 0.0, 0.0)
    assert hubs[1].transmissions == (Transmission("turn_transmission", "turn", "hub_motor", HUB_REDUCTION),)
    (screw,) = slide.transmissions
    assert (screw.joint, screw.actuator) == ("slide", "slide_motor")
    assert screw.reduction == pytest.approx(SCREW_REDUCTION, rel=1e-15)
    assert slide.body("carriage").inertial.mass == CARRIAGE_MASS


# -- zero is bitwise ------------------------------------------------------------------------------------------------

def test_a_zero_armature_on_every_joint_builds_the_graph_of_none_and_a_tiny_one_does_not():
    zero = {f"joint_{i}": kg_m2(0.0, reason="none, as the 0.2 line carried") for i in range(1, 7)}
    bare = built(dynamic_rpm("armature_none")).report.derived_hash
    assert built(dynamic_rpm("armature_none", armature=zero)).report.derived_hash == bare
    assert built(dynamic_rpm("armature_none", armature={"joint_6": kg_m2(1e-12)})).report.derived_hash != bare


# -- the hand model ---------------------------------------------------------------------------------------------------

HAND_STEPS = 100


@pytest.fixture(scope="module")
def hub():
    graph = built(armed_hub("hand_hub", armature={"turn": kg_m2(HUB_INERTIA)}))
    return nodes(graph, "hub")


def physical_either(lower: float, upper: float):
    return st.tuples(st.floats(lower, upper), st.sampled_from((-1.0, 1.0))).map(lambda pair: pair[0] * pair[1])


@st.composite
def hand_draws(draw):
    """An armature from 0.05 to 50 times the hub's own inertia (even in its logarithm), a rotor of A0's order (0.5 to 5
    times 1e-5 kg m^2) and a drive of A0's scale (0.1 to 1 mN m either way)."""
    return (HUB_INERTIA * 10.0 ** draw(st.floats(math.log10(0.05), math.log10(50.0))),
            draw(st.floats(0.5, 5.0)) * 1e-5, draw(physical_either(0.1, 1.0)) * 1e-3)


def hand_misfit(arm, motor, armature, plant: float, rotor, torque) -> tuple[float, float]:
    """The largest, over the run, of ``|(I + a) q_dot + J Omega|`` with ``a`` the drawn armature (the hand model's
    axial momentum, zero from rest), the node carrying ``plant`` times it; and the bound ``dt tau`` plus rounding."""
    pa = {**arm._step_constants(None), "armatures": jnp.asarray([plant * armature])}
    pm = {**motor._step_constants(None), "inertia": jnp.asarray(rotor)}
    rates, spins = compiled_run(arm, motor, HAND_STEPS)(pa, pm, jnp.asarray(torque), starts(arm, motor))
    momenta = (HUB_INERTIA + armature) * np.asarray(rates)[:, 0] + rotor * np.asarray(spins)
    scale = abs(torque) * DT * HAND_STEPS
    return float(np.max(np.abs(momenta))), DT * abs(torque) + ROUNDINGS * EPS * scale * HAND_STEPS


@PROPERTY
@given(hand_draws())
def test_a_rotor_spun_up_on_a_free_joint_turns_it_at_the_hand_models_speed_with_its_armature(hub, drawn):
    misfit, bound = hand_misfit(*hub, drawn[0], 1.0, drawn[1], drawn[2])
    assert misfit <= bound, (misfit, bound)


@PROPERTY
@given(hand_draws())
def test_control_the_armature_left_out_of_the_mass_matrix_or_counted_twice_fails_on_every_draw(hub, drawn):
    for plant in (0.0, 2.0):
        misfit, bound = hand_misfit(*hub, drawn[0], plant, drawn[1], drawn[2])
        assert misfit > 2.0 * bound, (plant, misfit, bound)


# -- the transmission read --------------------------------------------------------------------------------------------

def _armatures(graph, owner) -> np.ndarray:
    return np.asarray(graph.graph.get_node(f"{owner}.arm").constants()["armatures"], dtype=np.float64)


def reflected_once(self, rotor_inertia):
    """The planted read: the reduction used as ``N``, not ``N^2``."""
    return self.reduction * rotor_inertia


@pytest.mark.parametrize("planted", [False, True], ids=["read", "planted N not N^2"])
def test_a_transmissions_reduction_reflects_its_actuators_rotor_squared_and_n_alone_is_caught(monkeypatch, planted):
    if planted:
        monkeypatch.setattr(Transmission, "reflected", reflected_once)
    rotor = 2e-7
    cases = (
        (armed_hub("geared_declared", description=GEARED_JOINT, actuators={"hub_motor": kg_m2(rotor)}), "hub",
         HUB_REDUCTION ** 2 * rotor),
        (armed_hub("geared_registered", description=GEARED_JOINT,
                   actuators={"hub_motor": "stepperonline.17hs19-2004s1"}), "hub", HUB_REDUCTION ** 2 * STEPPER_ROTOR),
        (geared_slide("slide_declared", actuators={"slide_motor": kg_m2(rotor)}), "stage",
         SCREW_REDUCTION ** 2 * rotor),
    )
    found = [float(_armatures(built(experiment), owner)[0]) for experiment, owner, _ in cases]
    expected = [armature for _, _, armature in cases]
    matches = [abs(f - e) <= 1e-12 * e for f, e in zip(found, expected, strict=True)]
    assert matches == [not planted] * 3, (found, expected)


@pytest.mark.parametrize("planted", [False, True], ids=["read", "planted N not N^2"])
def test_a_slide_pushed_through_its_lead_screw_accelerates_at_the_force_over_its_mass_and_reflected_rotor(
        monkeypatch, planted):
    if planted:
        monkeypatch.setattr(Transmission, "reflected", reflected_once)
    rotor, force, steps = 1e-7, 1.0, 50
    graph = built(geared_slide("pushed_slide", actuators={"slide_motor": kg_m2(rotor)}, force=force))
    history = graph.run(s(steps * DT), history=True).history
    speed = float(np.asarray(history["stage.arm"]["joint_velocities"])[-1, 0])
    expected = steps * DT * force / (CARRIAGE_MASS + SCREW_REDUCTION ** 2 * rotor)
    assert (abs(speed - expected) <= 1e-12 * expected) is not planted, (speed, expected)


# -- the check predicts the kick ------------------------------------------------------------------------------------

#: The armatures the check is judged at: on the wrist alone, from none to 1e-3 kg m^2, and on every joint.
KICK_CASES = {"none": {}, "wrist 1e-6": {"joint_6": 1e-6}, "wrist 1e-5": {"joint_6": 1e-5},
              "wrist 1e-4": {"joint_6": 1e-4}, "wrist 1e-3": {"joint_6": 1e-3},
              "every joint 1e-4": {f"joint_{i}": 1e-4 for i in range(1, 7)}}
#: The drive (N m) and the run: small enough that the arm's pose barely moves, so the start's mass matrix holds.
KICK_TORQUE, KICK_STEPS = 1e-5, 200
#: How far the measured ratios may lie from the predicted ones, against the largest: the pose moves a little over the
#: run, so the mass matrix and the velocity products are not quite the start's (measured: at most 5.3e-4 over the six
#: cases, the bare wrist's).
KICK_TOLERANCE = 2e-3


def _armed(values: dict[str, float]) -> dict:
    return {joint: kg_m2(value, reason="the check's measured case") for joint, value in values.items()}


@pytest.fixture(scope="module")
def kick_cases():
    """Each case's predicted ratios (the build's, by joint) and measured ones (each joint's speed at the end of the run
    over the rotor's spin a step before), the measured ones run on that case's own built nodes."""
    found = {}
    first = None
    for name, values in KICK_CASES.items():
        experiment = free_ar4(f"kick_{name.replace(' ', '_').replace('-', '_')}", armature=_armed(values))
        plans = planned(experiment)
        (judged,) = plans.kicks
        predicted = {kick.joint: kick.ratio for kick in judged.kicks}
        graph = built(free_ar4(f"kick_{name.replace(' ', '_').replace('-', '_')}", armature=_armed(values)))
        arm, motor = nodes(graph, "rpm")
        first = first or (arm, motor)
        rates, spins = compiled_run(*first, KICK_STEPS)(arm._step_constants(None), motor._step_constants(None),
                                                       jnp.asarray(KICK_TORQUE), starts(arm, motor))
        names = arm.constants()["dof_names"]
        # the drive's reaction reaches the arm a step late, so its joints' momentum at the last step balances the
        # rotor's at the step before
        measured = dict(zip(names, np.abs(np.asarray(rates)[-1]) / abs(float(np.asarray(spins)[-2])), strict=True))
        found[name] = (predicted, measured, np.asarray(arm.constants()["armatures"], dtype=np.float64))
    return found


def kick_misfit(predicted: dict, measured: dict) -> float:
    largest = max(predicted.values())
    return max(abs(measured[joint] - ratio) for joint, ratio in predicted.items()) / largest


def test_the_checks_ratios_predict_each_joints_kick_over_six_armatures(kick_cases):
    print("\ncase: wrist ratio predicted, measured; largest misfit over the joints against the largest ratio")
    for name, (predicted, measured, _) in kick_cases.items():
        misfit = kick_misfit(predicted, measured)
        print(f"  {name}: {predicted['joint_6']:.4g}, {measured['joint_6']:.4g}; {misfit:.2e}")
        assert misfit <= KICK_TOLERANCE, (name, misfit)
    wrist = sorted(predicted["joint_6"] for predicted, _, _ in kick_cases.values())
    assert wrist[0] < 0.02 and wrist[-1] > 3.5, "the cases span the wrist's ratios from a hundredth to four"


def test_control_the_check_reading_the_bare_inertia_while_the_node_carries_the_armature_or_the_reverse_fails(
        kick_cases):
    bare_predicted, bare_measured, _ = kick_cases["none"]
    for name, (predicted, measured, _) in kick_cases.items():
        if name == "none":
            continue
        assert kick_misfit(bare_predicted, measured) > 10 * KICK_TOLERANCE, (name, "bare check, armed node")
        assert kick_misfit(predicted, bare_measured) > 10 * KICK_TOLERANCE, (name, "armed check, bare node")


def test_the_nodes_carry_the_armature_the_build_declared(kick_cases):
    for name, (_, _, armatures) in kick_cases.items():
        expected = np.zeros(6)
        for joint, value in KICK_CASES[name].items():
            expected[int(joint[-1]) - 1] = value
        assert np.array_equal(armatures, expected), name


# -- the build's line -------------------------------------------------------------------------------------------------

def a0_like(name: str, armature=None):
    """:func:`chain_cases.free_ar4` at A0's own top speed, its motor on the flange as A0's."""
    return free_ar4(name, armature=armature or {})


@pytest.mark.parametrize("profile", ["exploratory", "research"])
def test_a_bare_wrist_under_a_spinner_gives_a_validity_line_naming_the_joint_the_ratio_its_speed_and_the_fix(profile):
    experiment = a0_like(f"line_{profile}")
    experiment._settings = replace(experiment._settings, profile=profile)
    try:
        plans = planned(experiment)
        checked = dict(plans.checked)
    except BuildRefused as refused:
        checked = dict(refused.value.report.checked)
    (line,) = checked["rpm.arm"]
    assert line.startswith("joint_6: ") and "a ratio of 3.93" in line and "247 rad/s" in line
    assert f"top speed of {TOP_SPEED:.6g} rad/s" in line and "157 times its speed limit of 1.5708 rad/s" in line
    assert 'armature={"joint_6": kg_m2(...)}' in line


def test_the_line_is_never_a_refusal_and_a_geared_wrist_has_none():
    report = built(a0_like("line_built")).report
    assert not report.refusals and report.checked["rpm.arm"]
    assert any("validity of rpm.arm (computed at build): joint_6:" in line for line in report.summary())
    geared = built(a0_like("line_geared", _armed({"joint_6": 1e-3}))).report
    assert "rpm.arm" not in geared.checked
    (summary,) = [line for line in geared.masses if "turns joint_6 the most" in line]
    assert f"a validity line above {KICK_RATIO:g}" in summary


def test_the_report_says_where_each_joints_armature_comes_from():
    report = built(armed_hub("sourced_hub", description=GEARED_JOINT,
                             actuators={"hub_motor": "stepperonline.17hs19-2004s1"})).report
    text = "\n".join(report.descriptions)
    assert "armature of turn: 0.00328 kg m^2, reflected through transmission 'turn_transmission' (reduction 20" in text
    assert "registered actuator stepperonline.17hs19-2004s1" in text and "82.00 g cm^2" in text
    bare = "\n".join(built(dynamic_rpm("unsourced_arm")).report.descriptions)
    assert "no armature on joint_1, joint_2, joint_3, joint_4, joint_5, joint_6: the URDF's own meaning" in bare


# -- the refusals -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("make", "code", "says"), [
    (lambda: armed_hub("unknown_joint", armature={"elbow": kg_m2(1e-5)}), "chain.armature", "not a moving joint"),
    (lambda: armed_hub("wrong_unit", armature={"turn": kg(1e-5)}), "chain.armature", "a moment of inertia, kg m^2"),
    (lambda: geared_slide("unknown_actuator", actuators={"wrong": kg_m2(1e-7)}), "chain.armature",
     "no simple transmission"),
    (lambda: armed_hub("twice", description=GEARED_JOINT, armature={"turn": kg_m2(1e-5)},
                       actuators={"hub_motor": kg_m2(1e-7)}), "chain.armature", "given twice"),
    (lambda: armed_hub("unregistered", description=GEARED_JOINT, actuators={"hub_motor": "acme.nothing"}),
     "setting.value", "no actuator 'acme.nothing' is registered"),
    (lambda: armed_hub("negative", armature={"turn": kg_m2(-1e-5)}), "setting.value", "of zero or more"),
])
def test_an_armature_the_description_cannot_take_is_refused_by_name(make, code, says):
    with pytest.raises(BuildRefused) as refused:
        planned(make())
    reasons = [str(r) for r in refused.value.refusals if r.code == code]
    assert reasons and says in reasons[0], [str(r) for r in refused.value.refusals]


def test_an_armature_on_the_kinematic_rung_is_refused_since_its_joints_move_as_commanded():
    exp = experiment("kinematic_armature")
    exp.add_apparatus(Apparatus("hub", chain={"arm": Arm.Kinematic(
        description=asset(SPIN_JOINT), initial_positions=rad((0.0,)), armature={"turn": kg_m2(1e-5)})},
        commands={"arm.joint_positions": Constant(rad((0.0,)))}))
    add_body(exp)
    with pytest.raises(BuildRefused) as refused:
        planned(exp)
    assert any("armature is the dynamic rung's" in str(r) for r in refused.value.refusals)


# -- the profile and the saved form -----------------------------------------------------------------------------------

def _listed(experiment, profile: str = "research") -> dict[str, str]:
    experiment._settings = replace(experiment._settings, profile=profile)
    try:
        outcome = planned(experiment).profile
    except BuildRefused as refused:
        outcome = refused.value.report.profile
    return {item.target: item.kind for item in outcome.listed}


def test_research_lists_the_convention_of_none_and_each_armature_without_its_source():
    assert _listed(armed_hub("listed_none"))["hub.arm.armature"] == "a convention, no source"
    sourced = armed_hub("listed_sourced")
    sourced.source("hub.arm.armature", reason="the hub has no gearing")
    assert "hub.arm.armature" not in _listed(sourced)
    declared = _listed(armed_hub("listed_declared", armature={"turn": kg_m2(1e-5)}))
    assert declared["hub.arm.armature.turn"] == "no source" and "hub.arm.armature" not in declared
    reasoned = _listed(armed_hub("listed_reasoned", armature={"turn": kg_m2(1e-5, reason="the drive's datasheet")}))
    assert "hub.arm.armature.turn" not in reasoned
    registered = _listed(armed_hub("listed_registered", description=GEARED_JOINT,
                                   actuators={"hub_motor": "stepperonline.17hs19-2004s1"}))
    assert not [target for target in registered if target.startswith("hub.arm.actuators")], registered


def test_an_armature_and_an_actuator_save_and_load_into_the_graph_they_built(tmp_path):
    experiment = armed_hub("saved_hub", description=GEARED_JOINT,
                           actuators={"hub_motor": kg_m2(2e-7, reason="the drive's datasheet")})
    first = built(experiment)
    first.save(tmp_path / "hub")
    again = built(Experiment.load(tmp_path / "hub"))
    assert again.report.derived_hash == first.report.derived_hash
    assert float(_armatures(again, "hub")[0]) == pytest.approx(HUB_REDUCTION ** 2 * 2e-7, rel=1e-15)


# -- the backstop -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("actuators", "raised"), [((), True), ({"hub_motor": "stepperonline.17hs19-2004s1"}, False)],
                         ids=["bare", "geared"])
def test_a_joint_past_its_speed_limit_raises_the_flag_after_the_run_and_one_within_it_does_not(actuators, raised):
    graph = built(armed_hub("flagged_hub", description=GEARED_JOINT, actuators=actuators))
    path = "hub.arm.beyond_speed_limit"
    for history in (True, False):
        result = graph.run(ms(100), history=history)
        (flag,) = [read for read in result.flags if read.path == path] or [None]
        if raised:
            assert flag is not None and flag.was_raised, (history, flag)
        else:
            assert flag is None or not flag.was_raised, (history, flag)
    if raised:
        (layout,) = [held for held in graph.run(ms(100), history=True).layout.flags if held.path == path]
        assert layout.limits == (5.0,) and "speed limit (turn 5;" in layout.meaning


def test_the_flag_reads_back_from_a_run_file_with_its_limits(tmp_path):
    from mime.results import RunResult

    result = built(armed_hub("read_hub", description=GEARED_JOINT)).run(ms(50), history=True)
    again = RunResult.read(result.to_npz(tmp_path / "run.npz"))
    (flag,) = [held for held in again.layout.flags if held.name == "beyond_speed_limit"]
    assert flag.limits == (5.0,)
    assert [read.text() for read in again.flags] == [read.text() for read in result.flags]


# -- the controllers ------------------------------------------------------------------------------------------------

def test_the_tree_a_controller_reads_carries_the_armature_so_a_computed_torque_acts_on_the_whole_inertia():
    armature = 1e-4
    plans = planned(a0_like("models_armed", _armed({"joint_6": armature})))
    bare = planned(a0_like("models_bare")).models["rpm.arm"]
    tree = plans.models["rpm.arm"]
    assert float(tree.articulation.armatures[5]) == armature
    q, qdd = jnp.asarray(AR4_HOME), jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    gravity = jnp.zeros(3)
    held, without = (np.asarray(inverse_dynamics(model, q, None, qdd, gravity)) for model in (tree, bare))
    assert held[5] - without[5] == pytest.approx(armature, rel=1e-12)


# -- the derivative -------------------------------------------------------------------------------------------------

def test_the_armature_is_differentiable_through_the_step(hub):
    """d(the hub's speed after 20 steps)/d(armature) by reverse mode against the central difference."""
    arm, motor = hub
    pm = {**motor._step_constants(None), "inertia": jnp.asarray(ROTOR_INERTIA)}
    base, start = arm._step_constants(None), starts(arm, motor)

    def speed(armature):
        rates, _ = stepped(arm, motor, {**base, "armatures": jnp.reshape(armature, (1,))}, pm, jnp.asarray(1e-3), 20,
                           start)
        return rates[-1, 0]

    at, step = 1e-5, 1e-9
    derivative = float(jax.grad(speed)(jnp.asarray(at)))
    central = (float(speed(jnp.asarray(at + step))) - float(speed(jnp.asarray(at - step)))) / (2 * step)
    assert derivative == pytest.approx(central, rel=1e-6)
    assert derivative > 0.0, "more armature, a slower hub turning against its rotor"
