"""The loops A0's dynamic rungs need, each a controller's ``law(ctx)``: the 0.2 line's computed-torque joint loop on the
targets ``arm_tracking`` reaches, and the rotor's PI velocity loop the 0.2 line's motor ran inside its own step; and the
two together, the one controller A0 declares (an experiment has one controller, and both loops tick every step, so they
are one step at one rate). Registered under the ``oracle`` namespace, in this process only, as the controller tests'
own controllers are.

**What each does, and where it comes from** (the tag's ``ar4_helical_drive/control/controller.py`` and its motor node,
read in phase 7c2). The joint loop writes ``tau = M(q) (Kp (q_t - q) - Kd q_dot) + h(q, q_dot; g)``, which makes each
joint accelerate as ``Kp (q_t - q) - Kd q_dot``: the tag's ``M qdd + (bias - gravity)`` with the gravity the 0.2 arm
node added inside its own step, at that step's ``q``, now carried by the command. ``q_t`` is what ``arm_tracking``'s
step reaches from the joints it reads (the tag's IK seeded from the arm's joints too). ``M`` and ``h`` are those of the
segment's built tree (``ctx.models``, what is mounted on it grafted on) under the experiment's gravity turned into the
arm's base, in single precision, as the tag computed them. The rotor's loop reads the rotor's speed relative to its
stator from the motor's frames (the rotor's angular velocity less the stator's, along the motor's axis turned by the
stator's pose: ``spin`` is the rotor's absolute spin, and no published interface), and writes the drive's torque
``tau = Kp e + Ki I'`` with ``I' = I + e dt``, ``e`` the command less the speed it read: the 0.2 motor's order, which
advanced the integral before using it, on the speed before the step. It computes in the dtype of what it reads, as the
0.2 motor computed in its state's."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from mime.control import ArmTracking
from mime.core import quaternion
from mime.core.multibody import inverse_dynamics
from mime.registry import ControlLaw, Controller, Signal, register_controller
from mime.gradients import GradientStatus
from mime.vocabulary import register_namespace

register_namespace("oracle", "0.0.1")

_F32 = jnp.float32
#: The share of the gravity the joint loop compensates, 1 but for the controls' planted fault (``planted_gravity``):
#: read when a law is made, so that the step it makes holds it as a constant.
_GRAVITY_SHARE = [1.0]


@contextmanager
def planted_gravity(share: float) -> Iterator[None]:
    """Laws made inside compensate ``share`` of the weight, the controls' fault in the joint loop's gravity term."""
    held = _GRAVITY_SHARE[0]
    _GRAVITY_SHARE[0] = share
    try:
        yield
    finally:
        _GRAVITY_SHARE[0] = held


def _base_gravity(ctx: Any, orientation: tuple[float, float, float, float] | None) -> Any:
    """The experiment's gravity in the arm's base frame, where the tree's algorithms take it."""
    gravity = jnp.asarray(ctx.gravity, dtype=_F32)
    return gravity if orientation is None else quaternion.rotate_back(jnp.asarray(orientation, dtype=_F32), gravity)


def joint_loop(tracking: ControlLaw, tree: Any, gravity: Any, stiffness: float, rate: float) -> ControlLaw:
    """The computed-torque joint loop on ``tracking``'s joint targets: its state is ``tracking``'s, it holds the torques
    that balance the weight at the home pose before its first write lands."""
    kp, kd = _F32(stiffness), _F32(rate)

    def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state, targets = tracking.step(state, reads)
        q = jnp.asarray(reads["joint_positions"], dtype=_F32)
        qd = jnp.asarray(reads["joint_velocities"], dtype=_F32)
        accelerations = kp * (targets["joint_positions"] - q) - kd * qd
        return dict(state), {"joint_torques": inverse_dynamics(tree, q, qd, accelerations, gravity)}

    home = tracking.held["joint_positions"]
    held = inverse_dynamics(tree, home, None, None, gravity)
    return ControlLaw(state=tracking.state, held={"joint_torques": held}, step=step)


def rotor_loop(speed: float, axis: tuple[float, float, float], proportional: float, integral: float,
               timestep: float) -> ControlLaw:
    """The rotor's PI velocity loop: its state the error's integral, its held torque zero."""

    def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        rates, poses = jnp.asarray(reads["rotor_rates"]), jnp.asarray(reads["rotor_poses"])
        along = quaternion.rotate(poses[0, 3:], jnp.asarray(axis, dtype=poses.dtype))
        relative = jnp.dot(rates[1] - rates[0], along)
        error = jnp.asarray(speed, dtype=relative.dtype) - relative
        accumulated = state["integral"] + error * timestep
        return {"integral": accumulated}, {"torque": proportional * error + integral * accumulated}

    return ControlLaw(state={"integral": jnp.zeros(())}, held={"torque": jnp.zeros(())}, step=step)


def together(*laws: ControlLaw) -> ControlLaw:
    """Laws that read, write and carry disjoint labels, as one law: one step that runs each in turn."""

    def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        carried: dict[str, Any] = {}
        written: dict[str, Any] = {}
        for law in laws:
            moved, writes = law.step({key: state[key] for key in law.state}, reads)
            carried.update(moved)
            written.update(writes)
        return carried, written

    return ControlLaw(state={key: value for law in laws for key, value in law.state.items()},
                      held={key: value for law in laws for key, value in law.held.items()}, step=step)


_TORQUE_READS = {**ArmTracking.reads, "joint_velocities": Signal("Hz", (-1,))}
_ROTOR_READS = {"rotor_rates": Signal("rad/s", (2, 3)), "rotor_poses": Signal("m; 1", (2, 7))}


@register_controller
@dataclass(frozen=True, kw_only=True)
class ComputedTorqueTracking(ArmTracking):
    """``arm_tracking``'s joint targets, held by the 0.2 line's computed-torque joint loop: ``stiffness`` is ``Kp``
    (1/s^2), ``rate`` is ``Kd`` (1/s), and ``arm`` the segment whose built tree the law reads (``ctx.models``). It
    reads the arm's joint velocities beside ``arm_tracking``'s reads, and writes the joint torques of a dynamic arm."""

    name = "oracle.computed_torque_tracking"
    version = "0.1.0"
    reads = _TORQUE_READS
    writes = {"joint_torques": Signal("N m", (-1,))}
    arm: str
    stiffness: float = 1e4
    rate: float = 200.0

    def joints(self, ctx: Any) -> ControlLaw:
        tree = ctx.models[self.arm]
        gravity = _GRAVITY_SHARE[0] * _base_gravity(ctx, self.orientation)
        return joint_loop(ArmTracking.law(self, ctx), tree, gravity, self.stiffness, self.rate)

    def law(self, ctx: Any) -> ControlLaw:
        return self.joints(ctx)


@register_controller
@dataclass(frozen=True, kw_only=True)
class RotorSpeedPI(Controller):
    """Holds a dynamic motor's rotor at ``speed`` (rad/s) relative to its stator by the 0.2 line's PI velocity loop,
    ``proportional`` (N m s) and ``integral`` (N m) its gains, ``axis`` the motor's axis in its base frame. Reads the
    motor's frames' angular velocities and poses, writes its torque."""

    name = "oracle.rotor_speed_pi"
    version = "0.1.0"
    differentiability = GradientStatus.EXACT
    builds = {"": ()}
    reads = _ROTOR_READS
    writes = {"torque": Signal("N m", ())}
    speed: float
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    proportional: float = 1e-3
    integral: float = 1e-2

    def rotor(self, ctx: Any) -> ControlLaw:
        return rotor_loop(self.speed, self.axis, self.proportional, self.integral, ctx.timestep)

    def law(self, ctx: Any) -> ControlLaw:
        return self.rotor(ctx)


@register_controller
@dataclass(frozen=True, kw_only=True)
class A0Loops(ComputedTorqueTracking):
    """A0's controller on its dynamic rungs: :class:`ComputedTorqueTracking` and :class:`RotorSpeedPI` in one step,
    as the tag's runner called its one controller every step and its motor ran its PI in every step. ``speed``,
    ``motor_axis``, ``proportional`` and ``integral`` are the rotor loop's."""

    name = "oracle.a0_loops"
    version = "0.1.0"
    reads = {**_TORQUE_READS, **_ROTOR_READS}
    writes = {"joint_torques": Signal("N m", (-1,)), "torque": Signal("N m", ())}
    speed: float
    motor_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    proportional: float = 1e-3
    integral: float = 1e-2

    def law(self, ctx: Any) -> ControlLaw:
        return together(self.joints(ctx), rotor_loop(self.speed, self.motor_axis, self.proportional, self.integral,
                                                     ctx.timestep))


__all__ = ["A0Loops", "ComputedTorqueTracking", "RotorSpeedPI", "joint_loop", "planted_gravity", "rotor_loop",
           "together"]
