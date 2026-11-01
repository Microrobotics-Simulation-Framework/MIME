"""A spinner's kick on the segment that carries it, judged at build: how fast a rotor spun up from rest turns the
host's joints, from the host's tree as its dynamics carry it (what is mounted on it grafted on, each joint's armature
on it) at the joints' positions at the start.

A rotor carried by its spin takes angular momentum ``J Omega e`` about its axis ``e`` as it spins up, and its stator
gives the host as much the other way. With nothing holding the host's joints, their momentum balances it, ``M q_dot =
-J Omega S^T e`` (``S`` the angular velocity Jacobian of the frame the rotor rides on, ``S^T e`` each joint's axis
projected on the rotor's), so each joint turns at ``r_j Omega``, its ratio ``r_j = J |(M^-1 S^T e)_j|``: the rotor's
inertia about its axis over the joint's effective inertia for that spin. On one joint along the axis that is ``J /
(I_j + a_j)``, the joint's own inertia with its armature. It is the joint's speed in rad/s per rad/s of the rotor's
spin with every joint free, to first order in the motion (the mass matrix is the start's); a controller holding the
joint takes the momentum back over its own time, and keeps the joint below it (A0's computed-torque loop held the bare
wrist to an eighth of it).

The build reports every spinner's largest ratio with the mounted masses, and gives a validity line, never a refusal
and the same in every profile, for each revolute joint the kick would drive past its speed limit: where the joint's
description gives a speed limit and the rotor's top speed is declared, where the speed the ratio predicts at that top
speed, ``r_j Omega_top``, is above the limit; where either is missing, where the ratio is above :data:`KICK_RATIO`,
which the line says. The line names the joint, the ratio, the speed it predicts against the limit, and ``armature=``
as the fix. A joint's armature declared zero with its reason (a description compared with one that carried none) is a
legitimate choice, which the line states and does not refuse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import jax
import numpy as np
from numpy.typing import NDArray

from mime.core import jnp
from mime.core.multibody import MultibodyTree, forward_kinematics, frame_angular_jacobians, mass_matrix
from mime.core.multibody._tree import rotation_matrix

if TYPE_CHECKING:
    from mime.build._plan import EntryPlan

#: The ratio above which a joint's kick is a validity line where its speed limit or its rotor's top speed is not given,
#: so that no speed can be judged: the joint, held by nothing, turns at a tenth or more of its rotor's speed. A
#: labelled decision, not a measured bound: A0's bare wrist is at 4.1, and the reflected inertia of a geared wrist
#: drive (1e-4 kg m^2 and up, an inference) brings it to 0.1 and below. Where both are given, the line keys on the
#: speed against the limit instead: the kick matters because it passes the limit.
KICK_RATIO = 0.1


@dataclass(frozen=True)
class HostedSpinner:
    """A rotor carried by its spin on a segment whose dynamics carry mass: the spinner's segment (``owner``), the host
    segment and the frame of it the rotor rides on, the rotor's axis in that frame, its moment of inertia about the
    axis (kg m^2: its own declared and its parts'), and its top speed (rad/s) where declared."""

    owner: str
    host: str
    frame: str
    axis: tuple[float, float, float]
    inertia: float
    top_speed: float | None


@dataclass(frozen=True)
class JointKick:
    """One joint's kick: its ratio (rad/s per rad/s of the rotor's spin, held by nothing), its axis projected on the
    rotor's, its armature, and its speed limit where its description gives one."""

    joint: str
    ratio: float
    projection: float
    armature: float
    limit: float | None

    def predicted(self, top: float | None) -> float | None:
        """Its speed (rad/s) at the rotor's top speed ``top``, ``None`` where that is not declared."""
        return None if top is None else self.ratio * top

    def beyond(self, top: float | None) -> bool:
        """Whether it gives a validity line: its predicted speed above its limit where both are known, its ratio above
        :data:`KICK_RATIO` where either is not."""
        speed = self.predicted(top)
        if speed is None or self.limit is None:
            return self.ratio > KICK_RATIO
        return speed > self.limit

    def keyed(self, top: float | None) -> str:
        """What its line keys on, in words."""
        if top is not None and self.limit is not None:
            return "its speed at the rotor's top speed is above its limit"
        missing = "the rotor's top speed is not declared" if top is None else \
            f"{self.joint}'s description gives it no speed limit"
        return f"a ratio of {self.ratio:.3g}, above {KICK_RATIO:g}: {missing}, so the line keys on the ratio"


def joint_kicks(model: MultibodyTree, joints: Sequence[float], frame: str, axis: Sequence[float],
                inertia: float) -> tuple[JointKick, ...]:
    """Each revolute joint's kick from a rotor of ``inertia`` about ``axis`` (in the frame ``frame``), the tree
    ``model`` at the joint positions ``joints``, in float64."""
    articulation = model.articulation
    with cast(Any, jax).enable_x64(True):
        q = jnp.asarray(joints, dtype=jnp.float64)
        held = np.asarray(mass_matrix(model, q), dtype=np.float64)
        index = articulation.frame_index(frame)
        turning = np.asarray(frame_angular_jacobians(model, q), dtype=np.float64)[index]
        pose = np.asarray(forward_kinematics(model, q), dtype=np.float64)[index]
    along: NDArray[np.float64] = rotation_matrix((float(pose[3]), float(pose[4]), float(pose[5]),
                                                  float(pose[6]))) @ np.asarray(axis, dtype=np.float64)
    projected: NDArray[np.float64] = turning.T @ along
    ratios: NDArray[np.float64] = inertia * np.linalg.solve(held, projected)
    armatures = np.asarray(articulation.armatures, dtype=np.float64)
    return tuple(JointKick(name, abs(float(ratios[i])), float(projected[i]), float(armatures[i]),
                           model.joint(name).limits.speed)
                 for i, name in enumerate(articulation.dof_names) if articulation.revolute[i])


def _speed(kick: JointKick, top: float | None) -> str:
    if top is None:
        return (f"{kick.ratio:.3g} rad/s per rad/s of the rotor's spin (its top speed is not declared: "
                f"Motor.Dynamic(top_speed=rad_per_s(...)) gives the joint's speed)")
    speed = kick.ratio * top
    if kick.limit is None:
        limit = ", its description giving it no speed limit"
    elif speed > kick.limit:
        limit = f", {speed / kick.limit:.3g} times its speed limit of {kick.limit:g} rad/s"
    else:
        limit = f", within its speed limit of {kick.limit:g} rad/s"
    return f"{speed:.3g} rad/s at the rotor's top speed of {top:.6g} rad/s{limit}"


@dataclass(frozen=True)
class SpinnerKicks:
    """A hosted spinner's kick on each revolute joint of its host, as the build judged it."""

    spinner: HostedSpinner
    kicks: tuple[JointKick, ...]


@dataclass(frozen=True)
class KickReport:
    """What the build found of every hosted spinner: the mounted masses' lines, the validity lines by the host's id,
    and each spinner's kicks."""

    lines: tuple[str, ...]
    validity: dict[str, tuple[str, ...]]
    judged: tuple[SpinnerKicks, ...]


def kick_lines(hosted: Sequence[HostedSpinner], models: Mapping[str, MultibodyTree],
               segments: Mapping[str, EntryPlan]) -> KickReport:
    """The mounted masses' line for each hosted spinner (its largest kick), and the validity lines of each host whose
    joints' ratios are above :data:`KICK_RATIO`, by the host's id."""
    lines: list[str] = []
    validity: dict[str, list[str]] = {}
    judged: list[SpinnerKicks] = []
    for spinner in hosted:
        model = models.get(spinner.host)
        plan = segments[spinner.host]
        joints = plan.segment.initial_joints if plan.segment is not None else ()
        if model is None or len(joints) != model.articulation.dofs:
            lines.append(f"{spinner.owner}'s rotor on {spinner.host}: its kick on {spinner.host}'s joints is not "
                         f"judged: {spinner.host} gives no tree or no joint positions at its start")
            continue
        kicks = joint_kicks(model, joints, spinner.frame, spinner.axis, spinner.inertia)
        judged.append(SpinnerKicks(spinner, kicks))
        if not kicks:
            continue
        largest = max(kicks, key=lambda kick: kick.ratio)
        lines.append(f"{spinner.owner}'s rotor on {spinner.host}.{spinner.frame}: spun up from rest with "
                     f"{spinner.host}'s joints held by nothing, it turns {largest.joint} the most, at "
                     f"{_speed(largest, spinner.top_speed)} (a ratio of {largest.ratio:.3g}; a validity line where a "
                     f"joint's speed at the rotor's top speed passes its limit, or, with either not given, where its "
                     f"ratio passes {KICK_RATIO:g})")
        for kick in kicks:
            if not kick.beyond(spinner.top_speed):
                continue
            validity.setdefault(spinner.host, []).append(
                f"{kick.joint}: {spinner.owner}'s rotor ({spinner.inertia:.3g} kg m^2 about its axis, which lies "
                f"{kick.projection:.3g} along the joint's) against the joint's inertia for its spin, with its "
                f"armature of {kick.armature:.3g} kg m^2: a ratio of {kick.ratio:.3g}. Spun up from rest with the "
                f"joints held by nothing, the rotor turns {kick.joint} at {_speed(kick, spinner.top_speed)} (the "
                f"line: {kick.keyed(spinner.top_speed)}); a controller holding the joint must take that momentum "
                f"back. A "
                f"geared joint's drive reflects its motor's rotor onto it (N^2 J): give it as "
                f"Arm.Dynamic(armature={{\"{kick.joint}\": kg_m2(...)}}, ...), or through its transmission with "
                f"actuators=; an armature of zero with its reason is a legitimate choice, which this line states")
    return KickReport(tuple(lines), {host: tuple(found) for host, found in validity.items()}, tuple(judged))


__all__ = ["KICK_RATIO", "HostedSpinner", "JointKick", "KickReport", "SpinnerKicks", "joint_kicks", "kick_lines"]
