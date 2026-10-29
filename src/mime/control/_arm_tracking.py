"""A magnet carried by a robot arm, held at a standoff from a body it follows: the controller of the 0.2 line's AR4
rig, ported to the joint positions a kinematic arm takes."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import jax

from mime.core import jnp, quaternion, switch
from mime.core.multibody import DescriptionRefused, DescriptionWarning, InvalidDescription, JointKind, compose, \
    damped_least_squares, frame_jacobian, frame_pose, inverse, read_urdf, relative_turn, shorter_rotation_vector, \
    shorter_way
from mime.gradients import GradientStatus
from mime.registry import BuildContext, ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Asset
from mime.units import DIMENSIONLESS, Length, Magnitude, Quantity, Time, m, s

#: Where the controller computes: in single precision whatever the run's, as the 0.2 line's did.
_F32 = jnp.float32
_IDENTITY = (1.0, 0.0, 0.0, 0.0)
#: Below this length of the home pose's x made perpendicular to the body's z, the two are taken as parallel, and the
#: target's x is the home pose's y instead.
_PARALLEL = 1e-6
_jit: Any = cast(Any, jax).jit


@register_controller
@dataclass(frozen=True, kw_only=True)
class ArmTracking(Controller):
    """Holds the frame ``end_effector`` of an arm (the arm its URDF file ``description`` describes, its base at ``at``
    turned by ``orientation``) at ``standoff`` along ``up`` from where a body is along the line ``axis`` through the
    origin, that distance held within ``reach``: the magnet on the arm's flange follows the robot down a tube.

    Each tick it moves its target a fraction ``smoothing`` of the way to that point (a first-order filter per tick,
    from the end effector's pose at ``home``), and with ``orientation_feedback`` turns the target so that the end
    effector's z follows the body's, projected ``lookahead`` ahead by its angular velocity. It writes the joint
    positions that ``iterations`` damped-least-squares steps (``damping``, the step scaled by ``gain``) reach from the
    arm's present ones: the command of a kinematic arm. It reads the body's pose and twist and the arm's joint
    positions, all physics state: ground truth, not deployable.

    It computes in single precision whatever the run's precision, as the 0.2 line's controller did, so a run in double
    precision with it is mixed. Its step is written in ``jax.numpy``, so it runs as a callback or inside the compiled
    step, where a derivative of the run passes through it; the joint torques of a dynamic arm, which the 0.2 line's
    wrote by computed-torque control, wait for that arm."""

    name = "arm_tracking"
    version = "0.1.0"
    differentiability = GradientStatus.PIECEWISE_JUMP
    gradient = GradientRecord(switches=(
        switch.Declaration(
            owner="arm_tracking.reach.low",
            indicator="the body's distance along the axis less the reach's lower end, m",
            orders={"target_position": 0, "joint_positions": 0},
            route="none: a distance below the reach is held at its end, whose derivative in the distance is zero; a "
                  "smooth envelope would be another controller"),
        switch.Declaration(
            owner="arm_tracking.reach.high",
            indicator="the reach's upper end less the body's distance along the axis, m",
            orders={"target_position": 0, "joint_positions": 0},
            route="none: a distance above the reach is held at its end, as at the lower one"),
        switch.Declaration(
            owner="arm_tracking.half_turn",
            indicator="minus the cosine of half the angle from the end effector's orientation to its target's (the w "
                      "of the turn between them), dimensionless, positive past a half turn",
            orders={"joint_positions": -1},
            route="none: past a half turn the shorter way round is the other way, and each inverse-kinematics step "
                  "turns back; a target within a half turn of the end effector, as a tracking one is, never reaches "
                  "it"),
        switch.Declaration(
            owner="arm_tracking.small_turn",
            indicator="the series' threshold less the squared sine of half the angle from the end effector's "
                      "orientation to its target's, dimensionless, positive where its rotation vector is taken from "
                      "its series",
            orders={"joint_positions": 1},
            route="none needed: the series meets the closed form to rounding in its value and its slope, so the "
                  "derivative on either side is the one function's",
            unmeasured="the series and the closed form agree to single precision's rounding at the threshold, so no "
                       "output moves with the branch by more than its rounding floor"),
        switch.Declaration(
            owner="arm_tracking.parallel",
            indicator="the length of the home pose's x made perpendicular to the body's projected z, less the length "
                      "below which the two are taken as parallel, dimensionless",
            orders={"target_orientation": -1, "joint_positions": -1},
            route="none: with the home pose's x along the body's z the target's x is undefined, and the home pose's "
                  "y stands in for it"),
        switch.Declaration(
            owner="arm_tracking.hemisphere",
            indicator="the length of q (q . held), q the turn's quaternion and held the target's orientation, "
                      "dimensionless: zero at the edge of held's hemisphere",
            orders={"target_orientation": -1, "joint_positions": -1},
            route="none: at the edge q and -q are as near the held orientation, and the filter goes the other way "
                  "round on either side"),
    ))
    references = ("Sciavicco2000", "deJongh2025NonlinDyn")
    builds = {"": ()}
    reads = {"body_pose": Signal("m; 1", (7,)), "body_twist": Signal("m/s; rad/s", (6,)),
             "joint_positions": Signal("1", (-1,))}
    writes = {"joint_positions": Signal("1", (-1,))}
    description: Asset
    end_effector: str
    home: Annotated[Quantity, DIMENSIONLESS]
    axis: tuple[float, float, float]
    up: tuple[float, float, float]
    at: Annotated[Quantity, Length] = m((0.0, 0.0, 0.0))
    orientation: tuple[float, float, float, float] | None = None
    standoff: Annotated[Quantity, Length] = m(0.15)
    reach: Annotated[Quantity, Length] = m((-0.05, 0.05))
    smoothing: float = 0.005
    gain: float = 1.0
    damping: float = 0.05
    iterations: int = 5
    orientation_feedback: bool = False
    lookahead: Annotated[Quantity, Time] = s(0.010)
    acknowledged: tuple[tuple[str, str], ...] = ()

    def read(self) -> Any:
        """The arm's tree, read from its description; ``ValueError`` saying why it cannot be."""
        held = self.description
        content = held.read()
        found = hashlib.sha256(content).hexdigest()
        if held.sha256 is not None and found != held.sha256.lower():
            raise ValueError(f"the description {held.path} has SHA-256 {found}, and it is pinned to {held.sha256}")
        try:
            with warnings.catch_warnings():
                # It reads the kinematics alone; what the file holds beyond them the arm's segment reports.
                warnings.simplefilter("ignore", DescriptionWarning)
                return read_urdf(held.path, sha256=found, acknowledged=dict(self.acknowledged)).tree
        except (DescriptionRefused, InvalidDescription) as error:
            raise ValueError(f"{held.path}: {error}") from error

    @override
    def setting_problems(self, settings: Mapping[str, Magnitude]) -> list[str]:
        if not isinstance(cast(object, self.description), Asset):
            return [f"description={self.description!r} is not a file asset: write description=asset(\"arm.urdf\")"]
        try:
            tree = self.read()
        except (OSError, ValueError) as error:
            return [f"its description cannot be read: {error}"]
        found: list[str] = []
        prismatic = [joint.name for joint in tree.joints if joint.kind is JointKind.PRISMATIC]
        if prismatic:
            found.append(f"the arm has the prismatic joint {', '.join(prismatic)}, and the joint positions this "
                         f"controller reads and writes are angles in radians, in the unit 1 that cannot tell a "
                         f"length in metres from one")
        home = settings["home"]
        if len(home if isinstance(home, tuple) else (home,)) != tree.articulation.dofs:
            found.append(f"home has {len(home if isinstance(home, tuple) else (home,))} positions, and the arm has "
                         f"{tree.articulation.dofs} moving joints")
        if self.end_effector not in tree.frame_names:
            found.append(f"end_effector {self.end_effector!r} is no frame of the arm")
        reach = settings["reach"]
        if not (isinstance(reach, tuple) and len(reach) == 2 and reach[0] < reach[1]):
            found.append(f"reach is {reach!r} m, not a lower and a higher distance along the axis")
        for name in ("axis", "up"):
            held = cast(tuple[float, ...], getattr(self, name))
            if len(held) != 3 or abs(sum(c * c for c in held) - 1.0) > 1e-9:
                found.append(f"{name} is {held!r}, not a unit vector")
        if self.iterations < 1 or not 0.0 < self.smoothing <= 1.0:
            found.append(f"iterations {self.iterations} and smoothing {self.smoothing}: at least one iteration, and "
                         f"a smoothing above 0 and at most 1")
        return found

    @override
    def law(self, ctx: BuildContext) -> ControlLaw:
        tree = self.read()
        frame, iterations = self.end_effector, int(self.iterations)
        base = jnp.asarray((*ctx.vector("at"), *(self.orientation or _IDENTITY)), dtype=_F32)
        to_base = inverse(base)
        home = jnp.asarray(ctx.vector("home"), dtype=_F32)
        start = compose(base, frame_pose(tree, home, frame))
        home_turn = quaternion.rotation_matrix(start[3:])
        axis, up = jnp.asarray(self.axis, dtype=_F32), jnp.asarray(self.up, dtype=_F32)
        low, high = ctx.vector("reach")
        standoff, alpha, lookahead = ctx.scalar("standoff"), self.smoothing, ctx.scalar("lookahead")
        gain, damping, feedback = _F32(self.gain), _F32(self.damping), self.orientation_feedback

        def turned_towards(body: Any, twist: Any, held: Any) -> Any:
            """The target's orientation moved towards one whose z is the body's, projected ahead, and whose x is the
            home pose's x made perpendicular to it, the quaternion of that taken in ``held``'s hemisphere."""
            w = twist[3:] * _F32(lookahead)
            skew = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=_F32)
            ahead = (jnp.eye(3, dtype=_F32) + skew) @ quaternion.rotation_matrix(body[3:])
            # A column of a rotation turned by I + skew, which lengthens every vector: never zero.
            z = ahead[:, 2] / switch.nonzero_norm(ahead[:, 2])
            x = home_turn[:, 0] - jnp.dot(home_turn[:, 0], z) * z
            squared = jnp.sum(x * x)
            apart = squared > _PARALLEL * _PARALLEL
            x = switch.where(apart, x / jnp.sqrt(switch.domain_guard(apart, squared, 1.0)), home_turn[:, 1],
                             owner="arm_tracking.parallel", resolution=lambda: switch.rounding(jnp.ones((), _F32)),
                             indicator=lambda: jnp.sqrt(squared) - _PARALLEL)
            raw = _outer(jnp.stack([x, jnp.cross(z, x), z], axis=1)) @ held
            # raw is q (q . held), q the turn's quaternion: zero where q is a quarter turn from held in four
            # dimensions, the hemisphere's edge, across which its direction flips.
            length = switch.norm(raw, owner="arm_tracking.hemisphere", resolution=lambda: switch.rounding(
                jnp.ones((), _F32)), indicator=lambda: jnp.sqrt(jnp.sum(raw * raw)))
            raw = raw / switch.where(length > 0, length, 1.0, owner="arm_tracking.hemisphere",
                                     resolution=lambda: switch.rounding(jnp.ones((), _F32)), indicator=length)
            blended = alpha * raw + (1.0 - alpha) * held
            # Of two unit quaternions in one hemisphere, mixed by a fraction from 0 to 1: never zero.
            return blended / switch.nonzero_norm(blended)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            body = jnp.asarray(reads["body_pose"], dtype=_F32)
            twist = jnp.asarray(reads["body_twist"], dtype=_F32)
            q = jnp.asarray(reads["joint_positions"], dtype=_F32)
            distance = jnp.dot(body[:3], axis)
            along = switch.clip(distance, low, high, owner="arm_tracking.reach",
                                resolution=lambda: switch.rounding(high - low), indicator=distance)
            position = alpha * (along * axis + standoff * up) + (1.0 - alpha) * state["target_position"]
            held = state["target_orientation"]
            turn = turned_towards(body, twist, held) if feedback else held
            target = compose(to_base, jnp.concatenate([position, turn]))

            joints = q
            for _ in range(iterations):
                reached = frame_pose(tree, joints, frame)
                turned = shorter_way(relative_turn(reached, target), owner="arm_tracking.half_turn")
                error = jnp.concatenate([target[:3] - reached[:3],
                                         shorter_rotation_vector(turned, owner="arm_tracking.small_turn")])
                joints = (joints + damped_least_squares(frame_jacobian(tree, joints, frame), gain * error,
                                                        damping)).astype(_F32)
            return {"target_position": position, "target_orientation": turn}, {"joint_positions": joints}

        compiled = _jit(step)
        return ControlLaw(state={"target_position": start[:3], "target_orientation": start[3:]},
                          held={"joint_positions": home}, step=compiled)


def _outer(r: Any) -> Any:
    """``q q^T`` of the rotation matrix ``r``'s unit quaternion ``q = (w, x, y, z)``: each product of two of its
    components is linear in ``r``, so ``q`` up to its sign is any column of it scaled, with no case to choose."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    return 0.25 * jnp.stack([
        jnp.stack([1.0 + trace, r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]),
        jnp.stack([r[2, 1] - r[1, 2], 1.0 + r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0]]),
        jnp.stack([r[0, 2] - r[2, 0], r[0, 1] + r[1, 0], 1.0 - r[0, 0] + r[1, 1] - r[2, 2], r[1, 2] + r[2, 1]]),
        jnp.stack([r[1, 0] - r[0, 1], r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], 1.0 - r[0, 0] - r[1, 1] + r[2, 2]]),
    ])


__all__ = ["ArmTracking"]
