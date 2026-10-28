"""A magnet carried by a robot arm, held at a standoff from a body it follows: the controller of the 0.2 line's AR4
rig, ported to the joint positions a kinematic arm takes."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, cast, override

import jax

from mime.core import jnp, quaternion
from mime.core.multibody import DescriptionRefused, DescriptionWarning, InvalidDescription, JointKind, compose, \
    damped_least_squares, frame_jacobian, frame_pose, inverse, pose_error, read_urdf
from mime.gradients import GradientStatus
from mime.registry import BuildContext, ControlLaw, Controller, GradientRecord, Signal, register_controller
from mime.scene import Asset
from mime.units import DIMENSIONLESS, Length, Magnitude, Quantity, Time, m, s

#: Where the controller computes: in single precision whatever the run's, as the 0.2 line's did.
_F32 = jnp.float32
_IDENTITY = (1.0, 0.0, 0.0, 0.0)
_fori_loop: Any = cast(Any, jax).lax.fori_loop
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
    precision with it is mixed. It runs as a callback; the joint torques of a dynamic arm, which the 0.2 line's wrote
    by computed-torque control, wait for that arm."""

    name = "arm_tracking"
    version = "0.1.0"
    differentiability = GradientStatus.CUT
    gradient = GradientRecord(
        where="its step: the reach envelope is a clip of the body's distance along the axis, and the inverse "
              "kinematics' iterations hold the switches of mime.core.multibody's damped least squares",
        reason="it runs as a callback only, on the host between compiled chunks, so no derivative of a run passes "
               "through its writes, and its step's switches are not written through mime.core.switch",
        route="run it inside the compiled step (the controller's first mode), with its switches written through "
              "mime.core.switch and declared, so that its status derives from them")
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
            home pose's x made perpendicular to it."""
            w = twist[3:] * _F32(lookahead)
            skew = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]], dtype=_F32)
            ahead = (jnp.eye(3, dtype=_F32) + skew) @ quaternion.rotation_matrix(body[3:])
            z = ahead[:, 2] / jnp.maximum(jnp.linalg.norm(ahead[:, 2]), 1e-12)
            x = home_turn[:, 0] - jnp.dot(home_turn[:, 0], z) * z
            size = jnp.linalg.norm(x)
            x = jnp.where(size > 1e-6, x / jnp.maximum(size, 1e-30), home_turn[:, 1])
            raw = _quaternion_of(jnp.stack([x, jnp.cross(z, x), z], axis=1))
            raw = jnp.where(jnp.dot(held, raw) < 0.0, -1.0, 1.0) * raw
            blended = alpha * raw + (1.0 - alpha) * held
            return blended / jnp.linalg.norm(blended)

        def step(state: Mapping[str, Any], reads: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            body = jnp.asarray(reads["body_pose"], dtype=_F32)
            twist = jnp.asarray(reads["body_twist"], dtype=_F32)
            q = jnp.asarray(reads["joint_positions"], dtype=_F32)
            along = jnp.clip(jnp.dot(body[:3], axis), low, high)
            position = alpha * (along * axis + standoff * up) + (1.0 - alpha) * state["target_position"]
            held = state["target_orientation"]
            turn = turned_towards(body, twist, held) if feedback else held
            target = compose(to_base, jnp.concatenate([position, turn]))

            def solve(_: Any, joints: Any) -> Any:
                error = pose_error(frame_pose(tree, joints, frame), target)
                return (joints + damped_least_squares(frame_jacobian(tree, joints, frame), gain * error,
                                                      damping)).astype(_F32)

            joints = _fori_loop(0, iterations, solve, q)
            return {"target_position": position, "target_orientation": turn}, {"joint_positions": joints}

        compiled = _jit(step)
        return ControlLaw(state={"target_position": start[:3], "target_orientation": start[3:]},
                          held={"joint_positions": home}, step=compiled)


def _quaternion_of(r: Any) -> Any:
    """A rotation matrix's unit quaternion ``(w, x, y, z)``, by the case of its largest diagonal term."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    s0 = jnp.sqrt(trace + 1.0) * 2
    s1 = jnp.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
    s2 = jnp.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
    s3 = jnp.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    by_trace = jnp.stack([0.25 * s0, (r[2, 1] - r[1, 2]) / s0, (r[0, 2] - r[2, 0]) / s0, (r[1, 0] - r[0, 1]) / s0])
    by_x = jnp.stack([(r[2, 1] - r[1, 2]) / s1, 0.25 * s1, (r[0, 1] + r[1, 0]) / s1, (r[0, 2] + r[2, 0]) / s1])
    by_y = jnp.stack([(r[0, 2] - r[2, 0]) / s2, (r[0, 1] + r[1, 0]) / s2, 0.25 * s2, (r[1, 2] + r[2, 1]) / s2])
    by_z = jnp.stack([(r[1, 0] - r[0, 1]) / s3, (r[0, 2] + r[2, 0]) / s3, (r[1, 2] + r[2, 1]) / s3, 0.25 * s3])
    found = jnp.where(trace > 0, by_trace, jnp.where((r[0, 0] > r[1, 1]) & (r[0, 0] > r[2, 2]), by_x,
                                                     jnp.where(r[1, 1] > r[2, 2], by_y, by_z)))
    return found / jnp.maximum(jnp.linalg.norm(found), 1e-30)


__all__ = ["ArmTracking"]
