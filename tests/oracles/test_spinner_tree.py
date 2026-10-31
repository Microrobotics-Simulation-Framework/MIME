"""The spinner against its rotor as a jointed child (phase 7a4, MIME-ANO-122): the same host with the rotor as one more
revolute joint in its tree, built with ``mime.core.multibody`` and integrated finely here, where every gyroscopic,
reaction and inertia term is exact. The spinner carries the rotor by its absolute spin, its mass and mean inertia
across its axis grafted on the host and its half difference across it handed to the host's mass matrix at each step,
so it converges to the tree at first order: its error at the step halves as the step halves.

Three hosts (``chain_cases.HOSTS``): ``spin_joint.urdf``'s hub, one joint along the spin axis; ``swing.urdf``, one
joint across it, under gravity; and ``gimbal.urdf``, two joints across it, which the gyroscopic term couples. Each with
a rotor symmetric about its axis (A0's magnet turned along it) and one unequal across it (the magnet spun about a
diameter, as A0's is), driven and damped. Each draw starts the host at rest at zero under constant joint torques, the
rotor at a drawn spin; the error is the largest, over a 0.1 s run, of the host's joint rates and the rotor's angular
velocity in the world against their scales, at steps of 1 ms and 0.5 ms, against the tree under RK4 at a quarter of
the smaller (its own error under 1e-7 of the spinner's). Converging is at least first order, ``e(dt/2)`` at most 0.6
of ``e(dt)``: an order of 0.74 or more. A draw whose first-order error vanishes (the swing from rest under a constant
torque, whose joint rate semi-implicit Euler gives exactly) converges faster, and passes.

The controls, in the same draw, each against the same tree, on the host that can see it: the rotor's moment about its
axis counted twice (grafted on the host as well as carried by the spin) and the drive's and bearing's reaction left off
the host, on the hub; the gyroscopic torque's sign flipped, on the gimbal. Each leaves an error the step does not
shrink. The swing sees none of the three (its joint is across the axis, so the rotor's axial moment, its drive's
reaction and its gyroscopic torque all lie along or across the bearing that takes them), and holds the rotor's
inequality across its axis under gravity.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chain_cases import HOSTS, MAGNET_DENSITY, MAGNET_LENGTH, MAGNET_RADIUS, ROTOR_INERTIA, spinning_host
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import quaternion
from mime.core.multibody import IDENTITY, Body, Inertial, Joint, JointKind, MultibodyTree, articulate, \
    forward_dynamics, frame_motions, read_urdf
from mime.nodes.actuation import DynamicMotorNode

PROPERTY = settings(deadline=None, max_examples=100)
HORIZON, STEPS = 0.1, (1e-3, 5e-4)
#: RK4's step for the tree, as a fraction of the spinner's finer step.
FINER = 4
#: The most a halving of the step may leave of the error, for it to converge: an order of 0.74 or more.
HALVED = 0.6
#: The magnet's axis in the rotor's frame, for each host: along the spin axis (symmetric) or across it (unequal).
MAGNETS = {"hub": {"symmetric": (0.0, 0.0, 1.0), "unequal": (1.0, 0.0, 0.0)},
           "swing": {"symmetric": (0.0, 0.0, 1.0), "unequal": (1.0, 0.0, 0.0)},
           "gimbal": {"symmetric": (0.0, 1.0, 0.0), "unequal": (1.0, 0.0, 0.0)}}
#: The joint torques each host is driven by, at most (N m): enough to move it a few tenths of a radian a second.
TORQUES = {"hub": 1e-5, "swing": 1e-4, "gimbal": 1e-4}


class Flipped(DynamicMotorNode):
    """The planted motor: it hands its host its momentum with the sign flipped."""

    def handed(self, p, base, axis, angle, spin, relative, dt):
        spun_up, carried = super().handed(p, base, axis, angle, spin, relative, dt)
        return -spun_up, carried


class Unreacting(DynamicMotorNode):
    """The planted motor: its stator passes its host no reaction of the drive and the bearing."""

    def passed(self, given, base, axis, held, wrenches):
        return super().passed(given, base, axis, jnp.zeros_like(held), wrenches)


def rotor_inertial(axis, magnet_axis) -> Inertial:
    """The rotor's inertial in its own frame, written out here: the declared moment about its axis, and the magnet, a
    uniform cylinder of A0's size and density along ``magnet_axis``, its centroid on the axis."""
    e = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    a = np.asarray(magnet_axis, dtype=float) / np.linalg.norm(magnet_axis)
    mass = MAGNET_DENSITY * math.pi * MAGNET_RADIUS ** 2 * MAGNET_LENGTH
    along, across = 0.5 * mass * MAGNET_RADIUS ** 2, mass * (3.0 * MAGNET_RADIUS ** 2 + MAGNET_LENGTH ** 2) / 12.0
    tensor = ROTOR_INERTIA * np.outer(e, e) + across * np.eye(3) + (along - across) * np.outer(a, a)
    return Inertial(mass, (0.0, 0.0, 0.0), tuple(tuple(float(c) for c in row) for row in tensor))


def jointed(host: str, magnet_axis):
    """The host's tree with the rotor as a jointed child: a revolute joint about the motor's axis at the mount frame's
    origin, its body the rotor's inertial."""
    description, mount, axis, _ = HOSTS[host]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tree = read_urdf(description).tree
    rotor = Body("rotor", rotor_inertial(axis, magnet_axis))
    joint = Joint("rotor_joint", JointKind.REVOLUTE, mount, "rotor", IDENTITY, tuple(float(c) for c in axis))
    return articulate(MultibodyTree(tree.name, tree.root, (*tree.bodies, rotor), (*tree.joints, joint), tree.frames))


_CACHE: dict[tuple, object] = {}


def setup(host: str, kind: str):
    """The spinner's nodes as the build makes them (drive and damping traced), the jointed tree, and the planted
    motors, made once so that each compiles once."""
    key = ("setup", host, kind)
    if key not in _CACHE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph = spinning_host(host, f"tree_{host}_{kind}", magnet_axis=MAGNETS[host][kind],
                                  top_speed=100.0).build().graph
        arm, motor = graph.get_node(f"{host}.arm"), graph.get_node(f"{host}.motor")
        planted = {name: cls(motor.name, motor.delta_t, **motor.constants())
                   for name, cls in (("flipped", Flipped), ("unreacting", Unreacting))}
        _CACHE[key] = (arm, motor, jointed(host, MAGNETS[host][kind]), planted)
    return _CACHE[key]


def spinner_run(arm, motor, pa, pm, torques, drive, spin, dt):
    """The spinner from rest at zero, its rotor at ``spin``, over ``HORIZON`` at ``dt``: each step's joint rates and
    the rotor's angular velocity in the world, stepped as the graph steps them."""
    unequal = bool(np.any(np.asarray(motor.constants()["asymmetry"])))
    host = arm.initial_state()
    rotor = motor.initial_state()
    base = rotor["frame_poses"][0]
    axis = quaternion.rotate(base[3:], jnp.asarray(pm["axis"]))
    momentum, inertia = motor.handed(pm, base, axis, jnp.zeros(()), spin, spin, dt)
    rotor = {**rotor, "spin": spin, "momentum": momentum, "inertia": inertia}

    def step(held, _):
        host, rotor = held
        inputs = {"joint_torques": torques, "link_wrench": rotor["reaction"], "link_momentum": rotor["momentum"],
                  **({"link_inertia": rotor["inertia"]} if unequal else {})}
        host = arm.advance(pa, host, inputs, dt)
        rotor = motor.advance(pm, rotor, {"torque": drive, "mount_poses": host["frame_poses"],
                                          "mount_rates": host["frame_angular_velocities"]}, dt)
        return (host, rotor), (host["joint_velocities"], rotor["frame_angular_velocities"][1])

    _, (rates, turning) = jax.lax.scan(step, (host, rotor), None, length=int(round(HORIZON / dt)))
    return rates, turning


def reference_run(articulation, gravity, host_dofs, rotor_dof, torques, drive, damping, spin, dt):
    """The jointed tree from rest at zero, its rotor joint at ``spin``, under RK4 at ``dt``: at each spinner step of
    ``STEPS[-1]``, the host's joint rates and the rotor body's angular velocity in the world."""
    n = articulation.dofs
    host_index, frame = np.asarray(host_dofs), articulation.frame_index("rotor")

    def forces(qd):
        return jnp.zeros(n).at[host_index].set(torques).at[rotor_dof].set(drive - damping * qd[rotor_dof])

    def rate(y):
        q, qd = y[:n], y[n:]
        return jnp.concatenate([qd, forward_dynamics(articulation, q, qd, forces(qd), gravity)])

    def rk4(y, _):
        k1 = rate(y)
        k2 = rate(y + 0.5 * dt * k1)
        k3 = rate(y + 0.5 * dt * k2)
        k4 = rate(y + dt * k3)
        return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    def sampled(y, _):
        y, _ = jax.lax.scan(rk4, y, None, length=FINER)
        q, qd = y[:n], y[n:]
        return y, (qd[host_index], frame_motions(articulation, q, qd)[1][frame])

    start = jnp.zeros(2 * n).at[n + rotor_dof].set(spin)
    _, (rates, turning) = jax.lax.scan(sampled, start, None, length=int(round(HORIZON / STEPS[-1])))
    return rates, turning


def compiled(name, fn, *static):
    key = (name, *static)
    if key not in _CACHE:
        _CACHE[key] = jax.jit(fn, static_argnums=tuple(range(len(static))))
    return _CACHE[key]


def errors(host: str, kind: str, drawn, plant: str = "") -> tuple[float, float]:
    """The spinner's errors against the tree at the two steps, ``plant`` naming a planted fault or none."""
    torques, drive, damping, spin = drawn
    arm, motor, articulation, planted = setup(host, kind)
    pa = arm._step_constants(None)
    pm = {**motor._step_constants(None), "damping": jnp.asarray(damping)}
    if plant == "twice":
        held = arm.articulation(pa, jnp.float64)
        body = held.frame_bodies[held.frame_index(HOSTS[host][1])]
        e = np.asarray(HOSTS[host][2], dtype=float)
        pa = {**pa, "inertias": jnp.reshape(jnp.reshape(jnp.asarray(pa["inertias"]), (-1, 3, 3)).at[body].add(
            float(pm["inertia"]) * np.outer(e, e)), (-1,))}
    if plant in planted:
        motor = planted[plant]
    gravity = jnp.asarray((0.0, 0.0, -HOSTS[host][3]))
    names = articulation.dof_names
    host_dofs = tuple(names.index(name) for name in arm.constants()["dof_names"])
    tree = compiled(("tree", host, kind), lambda a, g, t, d, b, s: reference_run(
        a, g, host_dofs, names.index("rotor_joint"), t, d, b, s, STEPS[-1] / FINER))(
        articulation, gravity, jnp.asarray(torques), jnp.asarray(drive), jnp.asarray(damping), jnp.asarray(spin))
    found = []
    for dt in STEPS:
        rates, turning = compiled(("spinner", id(arm), id(motor), dt), lambda pa, pm, t, d, s, dt=dt: spinner_run(
            arm, motor, pa, pm, t, d, s, dt))(pa, pm, jnp.asarray(torques), jnp.asarray(drive), jnp.asarray(spin))
        stride = int(round(dt / STEPS[-1]))
        held_rates, held_turning = (np.asarray(x)[stride - 1::stride] for x in tree)
        rate_scale = float(np.max(np.abs(held_rates)))
        turning_scale = float(np.max(np.abs(held_turning)))
        found.append(max(float(np.max(np.abs(np.asarray(rates) - held_rates))) / rate_scale,
                         float(np.max(np.abs(np.asarray(turning) - held_turning))) / turning_scale))
    return found[0], found[1]


def converges(coarse: float, fine: float) -> bool:
    return fine <= HALVED * coarse


@st.composite
def runs(draw, host: str):
    """The host's joint torques (a fifth to all of :data:`TORQUES` either way, per joint), a drive of A0's scale, a
    bearing's damping to 2e-4 N m s, a rotor spin of 10 to 60 rad/s either way, and the rotor's kind."""
    dofs = 2 if host == "gimbal" else 1
    torques = [draw(st.floats(0.2, 1.0)) * TORQUES[host] * draw(st.sampled_from((-1.0, 1.0))) for _ in range(dofs)]
    return (np.asarray(torques), draw(st.floats(0.1, 1.0)) * 1e-3 * draw(st.sampled_from((-1.0, 1.0))),
            draw(st.floats(0.0, 2e-4)), draw(st.floats(10.0, 60.0)) * draw(st.sampled_from((-1.0, 1.0)))), \
        draw(st.sampled_from(("symmetric", "unequal")))


@PROPERTY
@given(runs("hub"))
def test_on_a_joint_along_the_axis_the_spinner_converges_to_the_jointed_rotor_and_its_planted_faults_do_not(drawn):
    run, kind = drawn
    coarse, fine = errors("hub", kind, run)
    assert converges(coarse, fine), (coarse, fine)
    for plant in ("twice", "unreacting"):
        coarse_p, fine_p = errors("hub", kind, run, plant)
        assert not converges(coarse_p, fine_p), (plant, coarse_p, fine_p)


@PROPERTY
@given(runs("swing"))
def test_on_a_joint_across_the_axis_the_spinner_converges_to_the_jointed_rotor(drawn):
    run, kind = drawn
    coarse, fine = errors("swing", kind, run)
    assert converges(coarse, fine), (coarse, fine)


@PROPERTY
@given(runs("gimbal"))
def test_on_a_gimbal_the_spinner_converges_to_the_jointed_rotor_and_its_gyroscopic_sign_flipped_does_not(drawn):
    run, kind = drawn
    coarse, fine = errors("gimbal", kind, run)
    assert converges(coarse, fine), (coarse, fine)
    coarse_p, fine_p = errors("gimbal", kind, run, "flipped")
    assert not converges(coarse_p, fine_p), ("flipped", coarse_p, fine_p)


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_the_jointed_tree_holds_the_rotor_as_one_more_joint(host):
    """The tree the oracle integrates is the host's with one revolute joint more, about the motor's axis."""
    _, _, articulation, _ = setup(host, "unequal")
    assert articulation.dof_names[-1] == "rotor_joint" and articulation.revolute[-1]
    assert np.allclose(np.asarray(articulation.axes)[-1], np.asarray(HOSTS[host][2], dtype=float))
