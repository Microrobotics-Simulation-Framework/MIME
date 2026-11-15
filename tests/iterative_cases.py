"""Test-only controllers that allocate on the iterative path (``ctx.allocator``), registered under the ``oracle``
namespace in this process only, and the experiments they run in.

:class:`ArmFollower` moves A0's AR4 so that its flange stands ``standoff`` above the body it reads: the arm's frame's
position allocated over its six joint positions through the arm's own model (affine in none of them, so the path is
the iterative one), the joints' bounds the description's limits, regularised towards the joints it reads. It runs in
A0's rig (``chain_cases.tracked_rpm``). The coil sphere of ``believed_cases`` takes the iterative path where the exact
path's size is patched below its three coils (:func:`iterative_coils`). :class:`FlaggingPD` is the coil sphere's
controller raising a flag of its own class's declaration, at its step's level or below it, compiled or not."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

import allocation_cases  # noqa: F401 - registers the oracle namespace
from allocation_cases import OFFSET, _start
from believed_cases import BelievedPD
from chain_cases import AR4_HOME, tracked_rpm

from mime.control import allocation_switch, raise_flag
from mime.gradients import GradientStatus
from mime.registry import ControlLaw, Controller, Flag, GradientRecord, Signal, register_controller
from mime.units import m, rad

jnp: Any = jax.numpy
lax: Any = jax.lax

#: The owner the follower declares and passes to its allocation.
FOLLOW = "iterative_cases.active_set"
#: The follower's position scale (m) and its effort's scale (rad).
POSITION_SCALE, EFFORT_SCALE = 1e-3, 20.0


@register_controller
@dataclass(frozen=True, kw_only=True)
class ArmFollower(Controller):
    """Holds the arm's flange (``frame``) ``standoff`` m above the body: each tick, the joint positions that put the
    frame there, by ``ctx.allocator`` on the arm's believed model, from the joints it reads and regularised towards
    them at ``effort`` rad (none where ``effort`` is zero: refused, three rows not fixing six joints). Where ``turn`` is
    above zero the frame is also held at its orientation at the home joints, over that scale (rad): the model's own
    miss, the rotation vector between the two."""

    name = "oracle.arm_follower"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(allocation_switch(FOLLOW, {"joint_positions": -1}),))
    builds = {"": ()}
    reads = {"body_pose": Signal("m; 1", (7,)), "joint_positions": Signal("1", (-1,))}
    writes = {"joint_positions": Signal("1", (-1,))}
    standoff: float = 0.15
    effort: float = EFFORT_SCALE
    frame: str = "link_6"
    turn: float = 0.0

    def law(self, ctx: Any) -> ControlLaw:
        effort, turn = ctx.scalar("effort"), ctx.scalar("turn")
        objective = {f"{self.frame}.position": m(POSITION_SCALE), **({f"{self.frame}.orientation": rad(turn)}
                                                                     if turn > 0.0 else {})}
        solve = ctx.allocator("rpm", commands=("arm.joint_positions",), objective=objective, owner=FOLLOW,
                              effort=rad(effort) if effort > 0.0 else None, toward="last")
        dtype = jnp.zeros(()).dtype
        above = jnp.asarray((0.0, 0.0, 1.0), dtype=dtype) * ctx.scalar("standoff")
        home = jnp.asarray(AR4_HOME, dtype=dtype)
        held = ctx.believed("rpm.arm")({"joint_positions": home}, {})[f"{self.frame}.orientation"]

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            wanted = {f"{self.frame}.position": jnp.asarray(reads["body_pose"][:3], dtype=dtype) + above,
                      **({f"{self.frame}.orientation": held} if turn > 0.0 else {})}
            joints = jnp.asarray(reads["joint_positions"], dtype=dtype)
            commands, _ = solve(wanted, {}, {"joint_positions": joints})
            return {}, {"joint_positions": commands["joint_positions"]}

        return ControlLaw(state={}, held={"joint_positions": home}, step=jax.jit(step))


#: The flag :class:`FlaggingPD` declares.
FAR = Flag("far_from_goal", "the sphere stood more than the controller's level from its goal at its last tick; the "
           "value is the distance (m)", warns=True)


@register_controller
@dataclass(frozen=True, kw_only=True)
class FlaggingPD(BelievedPD):
    """:class:`BelievedPD` raising ``raised`` (by default its declared flag ``far_from_goal``) at each tick the sphere
    stands more than ``level`` m from its goal, the distance its value: at the step's own level (``inside=""``) or
    inside a ``jax.lax`` body (``"cond"``, ``"while_loop"``, ``"scan"``, ``"fori_loop"``); its step returned as it is
    (``compiled=""``), compiled by ``jax.jit`` (``"jit"``) or compiled ahead of time, with no function underneath it
    (``"aot"``)."""

    name = "oracle.flagging_pd"
    raises = (FAR,)
    level: float = 3e-3
    inside: str = ""
    raised: str = FAR.field
    compiled: str = ""

    def law(self, ctx: Any) -> ControlLaw:
        inner = super().law(ctx)
        level = ctx.scalar("level")
        dtype = jnp.zeros(()).dtype
        goal = jnp.asarray(_start(), dtype=dtype) + jnp.asarray(OFFSET, dtype=dtype)
        name, inside = self.raised, self.inside

        def raised(value: Any) -> Any:
            raise_flag(name, value)
            return value

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[Any, Any]:
            after, written = inner.step(state, reads)
            distance = jnp.sqrt(jnp.sum((reads["pose"][:3] - goal) ** 2))
            value = jnp.where(distance > level, distance, 0.0)
            if inside == "cond":
                lax.cond(distance > level, raised, lambda held: held, value)
            elif inside == "while_loop":
                lax.while_loop(lambda held: held[0] < 1, lambda held: (held[0] + 1, raised(held[1])), (0, value))
            elif inside == "scan":
                lax.scan(lambda held, _: (raised(held), None), value, None, length=1)
            elif inside == "fori_loop":
                lax.fori_loop(0, 1, lambda _, held: raised(held), value)
            else:
                raised(value)
            return after, written

        if self.compiled == "jit":
            return ControlLaw(state=inner.state, held=inner.held, step=jax.jit(step))
        if self.compiled == "aot":
            reads = {"pose": jnp.zeros((7,), dtype=dtype)}
            return ControlLaw(state=inner.state, held=inner.held,
                              step=jax.jit(step).lower(dict(inner.state), reads).compile())
        return ControlLaw(state=inner.state, held=inner.held, step=step)


FOLLOWER_READS = {"body_pose": "umr.pose", "joint_positions": "rpm.arm.joint_positions"}


def follower_rpm(name: str = "follower_rpm", *, mode: str = "graph", controller: Controller | None = None,
                 rate: Any = None, latency: Any = None, **settings: Any) -> Any:
    """A0's rig with the arm's joint positions written by :class:`ArmFollower` (``settings`` its own)."""
    from mime.units import ms

    return tracked_rpm(name, controller=controller if controller is not None else ArmFollower(**settings),
                       reads=FOLLOWER_READS, mode=mode, rate=rate if rate is not None else ms(2),
                       latency=latency if latency is not None else ms(1))


@contextlib.contextmanager
def iterative_coils(monkeypatch: Any) -> Iterator[None]:
    """The coil sphere's allocation on the iterative path: the exact path's size patched below Navion_2's three coils,
    for builds made inside. A node's step is traced once (``Node._compiled``), so the patch acts on builds made inside
    it, never on a graph built before it."""
    from mime.control import _models

    with monkeypatch.context() as patch:
        patch.setattr(_models, "EXACT_COMMANDS", 2)
        yield


def final_array(history: Mapping[str, Mapping[str, np.ndarray]], node: str, field: str) -> np.ndarray:
    return np.asarray(history[node][field])


__all__ = ["EFFORT_SCALE", "FAR", "FOLLOW", "FOLLOWER_READS", "POSITION_SCALE", "ArmFollower", "FlaggingPD", "final_array",
           "follower_rpm", "iterative_coils"]
