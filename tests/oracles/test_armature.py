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
- **The stops and the gyroscopic step**: the planar arm resting on its stop under its weight, with and without
  armature, moves its elbow as the hand value, ``-m g c cos(q1 + q2) / (I + m c^2 + a)`` (the stop's impulse acts on
  the first joint alone, so the elbow's momentum takes none); the control, the stops' impulses through the bare inverse,
  fails on every armed case. A gimbal whose mass matrix is constant (an isotropic inner body), carrying a rotor's
  momentum across both its axes, keeps its kinetic energy, written out by hand with the armature, over every drawn run:
  the Cayley step's account. The control, the armature left out of the Cayley step alone, fails on every draw.
- **The transmission read**: the hub of ``geared_joint.urdf`` (reduction 20) and the slide of ``geared_slide.urdf`` (a
  lead screw of 5 mm), their actuators' rotors given, reflect ``N^2 J`` (kg m^2 on the hub, kg on the slide); the
  slide pushed by a force accelerates at ``F / (m + N^2 J)``. The control: the reduction used as ``N``, not ``N^2``.
  A transmission is judged only where it is used: each form that would be refused there (on a joint the tree lacks,
  on a fixed joint, two on one joint, a reduction of zero or not a number, an element URDF does not define) builds
  where its actuator is not given, on both rungs, the report naming it not used and what is wrong with it, and is
  refused where it is.
- **The check predicts the kick**: on A0's AR4, held by nothing (no gravity, no joint torques), its motor turned 0.3
  rad on the flange and spun up from rest, every joint's speed over the rotor's spin, measured at the end of a run,
  against the build's ratios, over six armatures (ratios on the wrist from 4.1 down to 0.01). The controls: the check
  reading the bare inertia while the node carries the armature, and the reverse. Its line keys on the speed it
  predicts against the joint's limit (7c2's stand-in, 1e-4 kg m^2 on the wrist, 3.73 times it, raises it), and on the
  ratio where the limit or the top speed is not given.
- **The build's line, the backstop and the controllers**: the validity line names the joint, the ratio, the speed it
  predicts and ``armature=``, in every profile, never a refusal; the velocity-limit flag after a run; ``ctx.models``
  carries the armature, so a computed torque acts on the joint's whole inertia.

The energy balance with armature is ``test_dynamic_arm.py``'s, over the branched tree's revolute and sliding joints
with their armatures; the spinner's jointed-child reference with armature on its hosts is ``test_spinner_tree.py``'s.
"""

import json
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

import multibody_cases
from mime.build import BuildRefused
from mime.build._construct import description_lines
from mime.build._kicks import KICK_RATIO
from mime.core.multibody import Body, Inertial, Joint, JointKind, MultibodyTree, Transmission, inverse_dynamics, \
    mass_matrix, read_urdf
from mime.nodes.actuation import DynamicArmNode
from mime.drives import Constant
from mime.dynamics import Arm
from mime.experiment import Experiment
from mime.scene import asset
from mime.things import Apparatus
from mime.units import N_m, kg, kg_m2, ms, rad, rad_per_s, s

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


# -- the stops and the gyroscopic step --------------------------------------------------------------------------------

#: ``multibody_cases.coupled_planar``'s second link, written out from its definition: its mass (kg), its centroid's
#: distance from the elbow (m) and its moment of inertia about its centroid (kg m^2); and the planar arm's stops.
ELBOW_MASS, ELBOW_CENTROID, ELBOW_OWN = 1.5, 0.6, 0.11
PLANAR = multibody_cases.coupled_planar(2.5)
#: The case 7a3's review set: the first joint on its lower stop at 0.3 rad, the elbow at 0.5 rad, in a vertical plane.
RESTING_AT, RESTING_LIMITS = (0.3, 0.5), {"lower": (0.3, -2.5), "upper": (3.0, 2.5)}
PLANAR_GRAVITY = 9.81
#: The armatures on the first joint and the elbow (kg m^2): none; the elbow's own inertia on the elbow; and both.
STOP_ARMATURES = {"none": (0.0, 0.0), "elbow": (0.0, 0.65), "both": (0.3, 0.2)}


def free_node(tree: MultibodyTree) -> DynamicArmNode:
    """``tree``'s dynamic arm with no damping and no friction, at rest in the world's frame, gravity a parameter."""
    a = tree.articulation
    described = DynamicArmNode.described(a, [0.0] * a.dofs, [0.0] * a.dofs)
    return DynamicArmNode("arm", DT, **described, placement=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                          gravity=(0.0, 0.0, 0.0), initial_joint_positions=(0.0,) * a.dofs)


def bare_stops(node: DynamicArmNode) -> DynamicArmNode:
    """The plant: ``node``'s stops and friction given the bare inverse, ``(M + dt D - A)^-1``, the armature left out of
    their impulses alone (the solve keeps it)."""
    held = node.constrained

    def constrained(p, q, free, inverse, dt):
        armature = jnp.diag(jnp.asarray(p["armatures"], dtype=free.dtype))
        return held(p, q, free, jnp.linalg.inv(jnp.linalg.inv(inverse) - armature), dt)

    node.constrained = constrained  # type: ignore[method-assign]
    return node


def resting_elbow(node: DynamicArmNode, armature: tuple[float, float]) -> tuple[np.ndarray, float]:
    """One step from rest on the stop: the joints' velocities, and the elbow's by hand. The stop's impulse acts on the
    first joint alone, so the elbow's momentum ``(A q_dot)_2`` takes only its weight's, ``-dt g_2``; the first joint
    ends at rest, so ``(I + m c^2 + a) q_dot_2 = -dt m g c cos(q1 + q2)``."""
    state = {**node.initial_state(), "joint_positions": jnp.asarray(RESTING_AT),
             "joint_velocities": jnp.zeros(2)}
    params = {"gravity": jnp.asarray([0.0, -PLANAR_GRAVITY, 0.0]), "armatures": jnp.asarray(armature),
              **{name: jnp.asarray(value) for name, value in RESTING_LIMITS.items()}}
    after = np.asarray(node.update(state, {}, DT, params=params)["joint_velocities"])
    weight = ELBOW_MASS * PLANAR_GRAVITY * ELBOW_CENTROID * math.cos(sum(RESTING_AT))
    return after, -DT * weight / (ELBOW_OWN + ELBOW_MASS * ELBOW_CENTROID ** 2 + armature[1])


def test_the_planar_arm_holds_the_elbow_the_hand_value_writes_out():
    held = PLANAR.body("l2").inertial
    assert held is not None and held.mass == ELBOW_MASS and held.centre_of_mass == (ELBOW_CENTROID, 0.0, 0.0)
    assert held.inertia[2][2] == ELBOW_OWN


@pytest.mark.parametrize("case", sorted(STOP_ARMATURES))
def test_a_stop_holding_the_planar_arm_moves_its_elbow_as_the_hand_value_with_its_armature_and_a_bare_inverse_does_not(
        case):
    armature = STOP_ARMATURES[case]
    found, hand = resting_elbow(free_node(PLANAR), armature)
    print(f"\n{case}: the elbow's speed after a step {found[1]:.12g} rad/s, by hand {hand:.12g}")
    assert found[0] == 0.0 and abs(found[1] - hand) <= ROUNDINGS * EPS * abs(hand), (found, hand)
    planted, _ = resting_elbow(bare_stops(free_node(PLANAR)), armature)
    caught = bool(abs(planted[1] - hand) > 1e6 * ROUNDINGS * EPS * abs(hand))
    assert caught == (armature[1] > 0.0), (case, planted, hand)


#: A gimbal whose mass matrix is the same at every pose: its outer body turns about the world's z (its own z inertia,
#: kg m^2), its inner body about the outer's x, isotropic (``GIMBAL_INNER`` about every axis), both centroids at the
#: joints' crossing. So ``M = diag(Iz + i, i)`` with no velocity products, and a step's kinetic energy changes only by
#: what the step's account does to it.
GIMBAL_OUTER, GIMBAL_INNER = (3e-5, 2.5e-5, 2e-5), 1e-5
GIMBAL_STEPS = 100


def constant_gimbal() -> MultibodyTree:
    ox, oy, oz = GIMBAL_OUTER
    outer = Body("outer", Inertial(0.02, (0.0, 0.0, 0.0), ((ox, 0.0, 0.0), (0.0, oy, 0.0), (0.0, 0.0, oz))))
    i = GIMBAL_INNER
    inner = Body("inner", Inertial(0.01, (0.0, 0.0, 0.0), ((i, 0.0, 0.0), (0.0, i, 0.0), (0.0, 0.0, i))))
    return MultibodyTree("constant_gimbal", "base", (Body("base"), outer, inner),
                         (Joint("yaw", JointKind.REVOLUTE, "base", "outer", axis=(0.0, 0.0, 1.0)),
                          Joint("pitch", JointKind.REVOLUTE, "outer", "inner", axis=(1.0, 0.0, 0.0))))


GIMBAL = constant_gimbal()


def gimbal_inertia(armature: float) -> np.ndarray:
    """The gimbal's mass matrix by hand, ``armature`` on its pitch joint."""
    return np.diag([GIMBAL_OUTER[2] + GIMBAL_INNER, GIMBAL_INNER + armature])


def bare_cayley(node: DynamicArmNode) -> DynamicArmNode:
    """The plant: ``node``'s Cayley step given the bare mass matrix, the armature left out of it alone."""
    held = node.spinning

    def spinning(articulation, q, qd, base, inertia, given, dt):
        bare = inertia - jnp.diag(jnp.asarray(articulation.armatures, dtype=inertia.dtype))
        return held(articulation, q, qd, base, bare, given, dt)

    node.spinning = spinning  # type: ignore[method-assign]
    return node


def gimbal_runs(planted: bool):
    """A compiled run of the gimbal's node (``planted``: the bare Cayley step), the rotor's momentum on its inner body
    held across both axes, ``h (sin q1, -cos q1, 0)``, so that ``G`` is the same at every step: each step's joint
    velocities."""
    key = ("gimbal", planted)
    if key not in _CACHE:
        node = free_node(GIMBAL)
        if planted:
            bare_cayley(node)
        base = node._step_constants(None)
        frames = len(GIMBAL.articulation.frame_names)
        inner = GIMBAL.articulation.frame_index("inner")

        def run(armature, momentum, rates):
            p = {**base, "armatures": jnp.stack([jnp.zeros(()), armature])}
            start = {**node.initial_state(), "joint_velocities": rates}

            def step(state, _):
                yaw = state["joint_positions"][0]
                held = momentum * jnp.stack([jnp.sin(yaw), -jnp.cos(yaw), jnp.zeros(())])
                state = node.advance(p, state, {"link_momentum": jnp.zeros((frames, 3)).at[inner].set(held)}, DT)
                return state, state["joint_velocities"]

            return jax.lax.scan(step, start, None, length=GIMBAL_STEPS)[1]

        _CACHE[key] = jax.jit(run)
    return _CACHE[key]


@st.composite
def gimbal_draws(draw):
    """An armature on the pitch joint from half to five times its own inertia, a speed of 0.2 to 2 rad/s in a drawn
    direction, and the rotor's momentum that turns the bare gimbal's velocity through 0.5 to 1.5 rad over the run."""
    armature = GIMBAL_INNER * draw(st.floats(0.5, 5.0))
    turn, direction = draw(st.floats(0.5, 1.5)), draw(st.floats(0.0, 2.0 * math.pi))
    speed = draw(st.floats(0.2, 2.0))
    bare = gimbal_inertia(0.0)
    momentum = turn / (GIMBAL_STEPS * DT) * math.sqrt(bare[0, 0] * bare[1, 1]) * draw(st.sampled_from((-1.0, 1.0)))
    return armature, momentum, speed * np.asarray([math.cos(direction), math.sin(direction)])


def gimbal_energies(planted: bool, drawn) -> tuple[np.ndarray, float, float]:
    """The kinetic energy by hand, with the armature, at each step of the run, its start's, and how far the armed
    gimbal's velocity turned (rad, in the coordinates its energy makes round)."""
    armature, momentum, rates = drawn
    found = np.asarray(gimbal_runs(planted)(jnp.asarray(armature), jnp.asarray(momentum), jnp.asarray(rates)))
    inertia = gimbal_inertia(armature)
    energies = 0.5 * np.einsum("si,ij,sj->s", found, inertia, found)
    root = np.sqrt(np.diag(inertia))
    first, last = rates * root, found[-1] * root
    turned = math.acos(max(-1.0, min(1.0, float(first @ last / (np.linalg.norm(first) * np.linalg.norm(last))))))
    return energies, float(0.5 * rates @ inertia @ rates), turned


def test_the_constant_gimbal_holds_the_mass_matrix_the_oracle_writes_out():
    for q in ((0.0, 0.0), (0.7, -1.2), (2.5, 0.4)):
        found, hand = np.asarray(mass_matrix(GIMBAL, jnp.asarray(q, dtype=jnp.float64))), gimbal_inertia(0.0)
        assert np.max(np.abs(found - hand)) <= ROUNDINGS * EPS * np.max(hand), (q, found)


@PROPERTY
@given(gimbal_draws())
def test_the_gyroscopic_step_keeps_the_gimbals_kinetic_energy_with_its_armature(drawn):
    energies, start, turned = gimbal_energies(False, drawn)
    assert turned > 0.1, ("the rotor's momentum must turn the velocity for the energy to say anything", turned)
    assert np.max(np.abs(energies - start)) <= ROUNDINGS * EPS * GIMBAL_STEPS * start, (energies, start)


@PROPERTY
@given(gimbal_draws())
def test_control_the_armature_left_out_of_the_cayley_step_changes_the_energy_on_every_draw(drawn):
    energies, start, _ = gimbal_energies(True, drawn)
    assert np.max(np.abs(energies - start)) > 1e6 * ROUNDINGS * EPS * GIMBAL_STEPS * start, (energies, start)


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


#: A hub on one continuous joint ``turn`` (``geared_joint.urdf``'s, speed limit 5 rad/s) and a tool welded to it by
#: the fixed joint ``weld``, with one form of simple transmission each, every one of which built on ``v0.3`` as
#: recognised and not used: what is wrong with it, as the report says it.
FORM_HEAD = """<?xml version="1.0"?>
<robot name="forms">
  <link name="base_link"/>
  <joint name="turn" type="continuous"><parent link="base_link"/><child link="hub"/><axis xyz="0 0 1"/>
    <limit effort="1" velocity="5"/></joint>
  <link name="hub"><inertial><mass value="0.1"/><inertia ixx="4e-6" ixy="0" ixz="0" iyy="4e-6" iyz="0" izz="2.43e-6"/>
  </inertial></link>
  <joint name="weld" type="fixed"><parent link="hub"/><child link="tool"/><origin xyz="0.01 0 0"/></joint>
  <link name="tool"/>
"""


def simple(name="t1", joint="turn", actuator="m", reduction="20", extra=""):
    return (f'<transmission name="{name}"><type>transmission_interface/SimpleTransmission</type>'
            f'<joint name="{joint}"><hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>'
            f'</joint><actuator name="{actuator}">{extra}<mechanicalReduction>{reduction}</mechanicalReduction>'
            f'</actuator></transmission>\n')


TRANSMISSION_FORMS = {
    "on a joint the tree lacks": (simple(joint="ghost"), "it drives 'ghost', which is not a joint of forms"),
    "on a fixed joint": (simple(joint="weld"), "it drives 'weld', a fixed joint"),
    "two on one joint": (simple() + simple(name="t2", actuator="m2", reduction="5"),
                         "its joint 'turn' is also driven by transmission 't2'"),
    "a reduction of zero": (simple(reduction="0"), "its reduction is 0, not a finite number other than zero"),
    "a reduction not a number": (simple(reduction="twenty"), "its <mechanicalReduction> 'twenty' is not a number"),
    "an element URDF does not define": (simple(extra="<motorTorqueConstant>0.1</motorTorqueConstant>"),
                                        "it holds transmission[t1]/actuator[m]/motorTorqueConstant, which URDF does "
                                        "not define"),
}


def form_arm(tmp_path, form: str, rung: str, actuators=()) -> Experiment:
    path = tmp_path / f"{form.replace(' ', '_')}.urdf"
    path.write_text(FORM_HEAD + TRANSMISSION_FORMS[form][0] + "</robot>\n")
    exp = experiment(f"form_{rung}")
    settings = {"description": asset(path), "initial_positions": rad((0.0,))}
    if rung == "Kinematic":
        commands = {"arm.joint_positions": Constant(rad((0.0,)))}
    else:
        settings["actuators"] = actuators
        commands = {"arm.joint_torques": Constant(N_m((0.0,)))}
    exp.add_apparatus(Apparatus("hub", chain={"arm": getattr(Arm, rung)(**settings)}, commands=commands))
    add_body(exp)
    return exp


@pytest.mark.parametrize("rung", ["Kinematic", "Dynamic"])
@pytest.mark.parametrize("form", sorted(TRANSMISSION_FORMS))
def test_a_transmission_not_used_builds_on_both_rungs_and_the_report_names_it_not_used_with_what_is_wrong(
        tmp_path, form, rung):
    plans = planned(form_arm(tmp_path, form, rung))
    text = "\n".join(description_lines(plans))
    assert "<transmission> element" in text and "not used (transmission[t1], which would be refused where used: " \
        in text, text
    assert TRANSMISSION_FORMS[form][1] in text, text
    assert "transmission 't1' (on " in text and "recognised and not used" in text, text


@pytest.mark.parametrize("form", sorted(TRANSMISSION_FORMS))
def test_a_transmission_used_with_anything_wrong_with_it_is_refused_by_name(tmp_path, form):
    with pytest.raises(BuildRefused) as refused:
        planned(form_arm(tmp_path, form, "Dynamic", actuators={"m": kg_m2(1e-7, reason="a drawn rotor")}))
    reasons = [str(r) for r in refused.value.refusals if r.code == "chain.armature"]
    assert reasons and "transmission 't1'" in reasons[0] and TRANSMISSION_FORMS[form][1] in reasons[0], \
        [str(r) for r in refused.value.refusals]


def test_a_transmission_used_is_read_and_used_in_the_classification_and_one_not_used_is_not():
    used = planned(armed_hub("used_transmission", description=GEARED_JOINT,
                             actuators={"hub_motor": kg_m2(2e-7, reason="a drawn rotor")}))
    unused = planned(armed_hub("unused_transmission", description=GEARED_JOINT))
    groups = []
    for plans in (used, unused):
        (held,) = [plan.segment.description for plan in plans.entries
                   if plan.segment is not None and plan.segment.description is not None]
        groups.append({e.path: e.group.value for e in held.elements if e.path.startswith("transmission")})
    assert groups[0]["transmission[turn_transmission]"] == "read and used"
    assert groups[0]["transmission[turn_transmission]/actuator[hub_motor]/mechanicalReduction"] == "read and used"
    assert groups[0]["transmission[turn_transmission]/joint[turn]/hardwareInterface"] == "recognised and not used"
    assert groups[1] == {"transmission[turn_transmission]": "recognised and not used"}


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


def kick_cases():
    """Each case's predicted ratios (its build's, by joint), measured ones (each joint's speed at the end of the run
    over the rotor's spin a step before), run on that case's own built nodes, and the armature those nodes hold."""
    found = {}
    first = None
    for name, values in KICK_CASES.items():
        graph = built(free_ar4(f"kick_{name.replace(' ', '_').replace('-', '_')}", armature=_armed(values)))
        (judged,) = graph.plans.kicks
        predicted = {kick.joint: kick.ratio for kick in judged.kicks}
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


def test_the_checks_ratios_predict_each_joints_kick_over_six_armatures_and_its_controls_fail():
    """One test, so that the six builds and runs it reads are made once, not once per worker that runs a test of
    them: each built node carries the armature declared; the prediction holds on every case; and the controls, the
    check reading the bare inertia while the node carries the armature and the reverse, fail on every armed case."""
    cases = kick_cases()
    for name, (_, _, armatures) in cases.items():
        expected = np.zeros(6)
        for joint, value in KICK_CASES[name].items():
            expected[int(joint[-1]) - 1] = value
        assert np.array_equal(armatures, expected), name
    print("\ncase: wrist ratio predicted, measured; largest misfit over the joints against the largest ratio")
    for name, (predicted, measured, _) in cases.items():
        misfit = kick_misfit(predicted, measured)
        print(f"  {name}: {predicted['joint_6']:.4g}, {measured['joint_6']:.4g}; {misfit:.2e}")
        assert misfit <= KICK_TOLERANCE, (name, misfit)
    wrist = sorted(predicted["joint_6"] for predicted, _, _ in cases.values())
    assert wrist[0] < 0.02 and wrist[-1] > 3.5, "the cases span the wrist's ratios from a hundredth to four"
    bare_predicted, bare_measured, _ = cases["none"]
    for name, (predicted, measured, _) in cases.items():
        if name == "none":
            continue
        assert kick_misfit(bare_predicted, measured) > 10 * KICK_TOLERANCE, (name, "bare check, armed node")
        assert kick_misfit(predicted, bare_measured) > 10 * KICK_TOLERANCE, (name, "armed check, bare node")


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
        checked = dict(refused.report.checked)
    (line,) = checked["rpm.arm"]
    assert line.startswith("joint_6: ") and "a ratio of 3.93" in line and "247 rad/s" in line
    assert f"top speed of {TOP_SPEED:.6g} rad/s" in line and "157 times its speed limit of 1.5708 rad/s" in line
    assert "(the line: its speed at the rotor's top speed is above its limit)" in line
    assert 'armature={"joint_6": kg_m2(...)}' in line


def test_the_line_is_never_a_refusal_and_a_geared_wrist_has_none():
    report = built(a0_like("line_built")).report
    assert not report.refusals and report.checked["rpm.arm"]
    assert any("validity of rpm.arm (computed at build): joint_6:" in line for line in report.summary())
    geared = built(a0_like("line_geared", _armed({"joint_6": 1e-3}))).report
    assert "rpm.arm" not in geared.checked
    (summary,) = [line for line in geared.masses if "turns joint_6 the most" in line]
    assert "a validity line where a joint's speed at the rotor's top speed passes its limit" in summary
    assert f"where its ratio passes {KICK_RATIO:g}" in summary


def _checked(experiment) -> dict:
    try:
        return dict(planned(experiment).checked)
    except BuildRefused as refused:
        return dict(refused.report.checked)


def test_the_line_keys_on_the_speed_against_the_limit_and_on_the_ratio_where_either_is_not_given():
    """7c2's stand-in (1e-4 kg m^2 on the wrist): a ratio of 0.093, under 0.1, and 5.86 rad/s at the rotor's top
    speed, 3.73 times the wrist's limit, so a line. A hub limited to 5 rad/s with a ratio of 0.191: no line at 10 rad/s
    of top speed (1.9 rad/s), a line at 100 (19 rad/s), which the ratio alone would not tell apart. Where the limit or
    the top speed is not given, the line keys on the ratio, and says so."""
    (line,) = _checked(a0_like("stand_in", _armed({"joint_6": 1e-4})))["rpm.arm"]
    assert "a ratio of 0.0932" in line and "5.86 rad/s" in line and "3.73 times its speed limit of 1.5708" in line
    assert "rpm.arm" not in _checked(free_ar4("stand_in_no_top", armature=_armed({"joint_6": 1e-4}), top_speed=None))
    (bare,) = _checked(free_ar4("bare_no_top", top_speed=None))["rpm.arm"]
    assert "a ratio of 3.93, above 0.1: the rotor's top speed is not declared, so the line keys on the ratio" in bare
    hub = {"description": GEARED_JOINT, "armature": {"turn": kg_m2(5e-5, reason="a drawn armature")}}
    assert "hub.arm" not in _checked(armed_hub("slow_hub", top_speed=10.0, **hub))
    (fast,) = _checked(armed_hub("fast_hub", top_speed=100.0, **hub))["hub.arm"]
    assert "a ratio of 0.191" in fast and "3.81 times its speed limit of 5 rad/s" in fast
    (free,) = _checked(armed_hub("unlimited_hub", top_speed=10.0))["hub.arm"]
    assert "turn's description gives it no speed limit, so the line keys on the ratio" in free
    assert "its description giving it no speed limit" in free
    assert "hub.arm" not in _checked(armed_hub("geared_unlimited_hub", top_speed=1e4,
                                               armature={"turn": kg_m2(1e-3, reason="a drawn armature")}))


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
    (lambda: armed_hub("twice_in_pairs", armature=(("turn", kg_m2(1e-5)), ("turn", kg_m2(1e-3)))), "chain.armature",
     "armature names 'turn' 2 times"),
    (lambda: armed_hub("actuator_twice", description=GEARED_JOINT,
                       actuators=(("hub_motor", kg_m2(1e-7)), ("hub_motor", "stepperonline.17hs19-2004s1"))),
     "chain.armature", "actuators names 'hub_motor' 2 times"),
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
        outcome = refused.report.profile
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


# -- the backstop -----------------------------------------------------------------------------------------------------

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


TWO_JOINTS = """<?xml version="1.0"?>
<robot name="two_limits">
  <link name="base_link"/>
  <joint name="turn" type="continuous"><parent link="base_link"/><child link="hub"/><axis xyz="0 0 1"/>
    <limit effort="1" velocity="{turn}"/></joint>
  <link name="hub"><inertial><mass value="0.1"/><inertia ixx="4e-6" ixy="0" ixz="0" iyy="4e-6" iyz="0" izz="2.43e-6"/>
  </inertial></link>
  <joint name="tilt" type="revolute"><parent link="hub"/><child link="pan"/><axis xyz="1 0 0"/>
    <origin xyz="0 0 0.02"/><limit lower="-1" upper="1" effort="1" velocity="2"/></joint>
  <link name="pan"><inertial><mass value="0.05"/><inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/>
  </inertial></link>
</robot>
"""


def two_joints(tmp_path, name: str, turn: str, speeds) -> Experiment:
    path = tmp_path / f"{name}.urdf"
    path.write_text(TWO_JOINTS.format(turn=turn))
    exp = experiment(name)
    exp.add_apparatus(Apparatus("pair", chain={"arm": Arm.Dynamic(
        description=asset(path), initial_positions=rad((0.0, 0.0)), initial_velocities=rad_per_s(speeds))},
        commands={"arm.joint_torques": Constant(N_m((0.0, 0.0)))}))
    add_body(exp)
    return exp


def _refuse_constant(name: str):
    raise ValueError(f"the manifest holds {name}, which strict JSON does not")


def test_a_velocity_limit_of_zero_is_no_limit_and_a_joint_with_none_is_null_in_the_manifest(tmp_path):
    """URDF's ``velocity="0"`` read as no limit (MoveIt's reading; :attr:`JointLimits.speed`): the turning joint at 100
    rad/s raises nothing, the tilt limited to 2 rad/s does past it; the manifest writes the joint with no limit as
    ``null``, which strict JSON reads, and reads it back as no limit."""
    from mime.results import RunResult
    from mime.results._result import NPZ_MANIFEST

    held = built(two_joints(tmp_path, "zero_limit", "0", (100.0, 1.0)))
    result = held.run(ms(20), history=True)
    (layout,) = [flag for flag in result.layout.flags if flag.name == "beyond_speed_limit"]
    assert layout.limits == (math.inf, 2.0) and "turn unlimited" in layout.meaning
    assert not [read for read in result.flags if read.path == layout.path and read.was_raised]
    saved = result.to_npz(tmp_path / "run.npz")
    manifest = json.loads(str(np.load(saved)[NPZ_MANIFEST]), parse_constant=_refuse_constant)
    (written,) = [flag for flag in manifest["flags"] if flag.get("name") == "beyond_speed_limit"]
    assert written["limits"] == [None, 2.0]
    (again,) = [flag for flag in RunResult.read(saved).layout.flags if flag.name == "beyond_speed_limit"]
    assert again.limits == (math.inf, 2.0)
    raised = built(two_joints(tmp_path, "zero_limit_tilted", "0", (100.0, 3.0))).run(ms(20), history=True)
    assert [read.was_raised for read in raised.flags if read.path == layout.path] == [True]


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
