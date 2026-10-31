"""The nodes of an apparatus chain's kinematic rungs: an arm whose joint positions are commanded, and a motor whose
angle or speed is. Each publishes the world pose and the angular velocity of every frame it names, ``frame_poses``
and ``frame_angular_velocities``, one row per frame, and places its base on a row of the frame poses of the segment it
mounts on (``mount_poses``), or in the world; its frames turn with that row of ``mount_rates``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast, override

import jax
import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import MimeNode, NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, \
    quaternion, switch, typed_view
from mime.core.multibody import Articulation, compose, forward_kinematics, frame_motions

#: The units of a frame's pose, as the vocabulary spells ``apparatus.frame_poses``, and of its angular velocity, as it
#: spells ``apparatus.frame_angular_velocities``.
POSE_UNITS, RATE_UNITS = "m; 1", "rad/s"
#: Where the symbol rows' functions are.
_MOUNTED, _ARM, _MOTOR = (f"mime.nodes.actuation._chain.{name}" for name in ("MountedNode", "KinematicArmNode",
                                                                             "KinematicMotorNode"))


class MountInputs(TypedDict):
    mount_poses: NotRequired[ReadOnly[Float[Array, "mounted 7"]]]
    mount_rates: NotRequired[ReadOnly[Float[Array, "mounted 3"]]]


class ArmState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    frame_angular_velocities: NotRequired[ReadOnly[Float[Array, "frames 3"]]]
    joint_positions: ReadOnly[Float[Array, " dofs"]]


class ArmInputs(MountInputs, closed=True):
    joint_positions: NotRequired[ReadOnly[Float[Array, " revolute"]]]
    joint_displacements: NotRequired[ReadOnly[Float[Array, " prismatic"]]]


class MotorState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    frame_angular_velocities: NotRequired[ReadOnly[Float[Array, "frames 3"]]]
    angle: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape


class MotorInputs(MountInputs, closed=True):
    angle: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    velocity: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape


class MountedNode(MimeNode):
    """A node placed on a frame of a chain segment: its ``placement`` (a translation and a unit quaternion ``(w, x, y,
    z)``) in that frame, the row ``mount_index`` of the ``mount_count`` frame poses it takes as ``mount_poses``
    (``-1``: placed in the world), and the world pose of that frame at the start, ``initial_mount_pose``, which places
    it before its first step. ``mount_poses`` and ``mount_rates`` are declared whether or not an edge feeds them, so
    that every such node has the same inputs; one placed in the world reads neither. ``mount_rates`` holds the angular
    velocity of each of those frames in the world, which the build feeds where the segment mounted on publishes them;
    where it publishes none, the build lets nothing read this node's own rates (their mount's turning unknown).
    ``initial_mount_rate`` is the mount frame's angular velocity at the start."""

    def mounted_inputs(self) -> dict[str, BoundaryInputSpec]:
        count = max(cast(int, self.constants()["mount_count"]), 1)
        return {"mount_poses": node_input(POSE_UNITS, (count, 7)), "mount_rates": node_input(RATE_UNITS, (count, 3))}

    def base_rate(self, given: MountInputs, dtype: Any) -> Array:
        """The angular velocity in the world of the frame the segment's base is placed on: its row of
        ``mount_rates``; zero in the world, and zero where no edge feeds them (a segment mounted on one that publishes
        no rates, whose own the build lets nothing read)."""
        index = cast(int, self.constants()["mount_index"])
        rates = given.get("mount_rates")
        if index < 0 or rates is None:
            return jnp.zeros(3, dtype=dtype)
        return jnp.asarray(rates[index], dtype=dtype)

    def initial_base_rate(self) -> Array:
        """The angular velocity in the world of the frame the segment's base is placed on, at the start."""
        held = self.constants()
        if cast(int, held["mount_index"]) < 0:
            return as_run_array(np.zeros(3))
        return as_run_array(held.get("initial_mount_rate") or (0.0, 0.0, 0.0))

    def base(self, p: Mapping[str, Any], given: MountInputs) -> Array:
        """The world pose of the segment's base: its placement composed onto its mount frame's pose, or its placement
        alone in the world."""
        placement = jnp.asarray(p["placement"])
        index = cast(int, self.constants()["mount_index"])
        if index < 0:
            return placement
        return compose(cast(Array, given.get("mount_poses"))[index], placement)

    def initial_base(self) -> Array:
        """The world pose of the segment's base at the start."""
        held = self.constants()
        placement = as_run_array(held["placement"])
        if cast(int, held["mount_index"]) < 0:
            return placement
        return compose(as_run_array(held["initial_mount_pose"]), placement)



class SegmentNode(MountedNode):
    """A chain segment's node: placed as a :class:`MountedNode` is, and publishing ``frame_poses``."""

    #: The state fields it holds only where the build finds something that reads them (its ``rates``), which a node
    #: built where nothing does leaves out: its symbol table still states them, their units the published interface's.
    held_where_read: ClassVar[tuple[str, ...]] = ("frame_angular_velocities",)

    def publishes_rates(self) -> bool:
        """Whether it computes and publishes its frames' angular velocities (``rates``): a kinematic segment does
        only where the build finds something that reads them, and its step is otherwise its frames' poses alone."""
        return bool(self.constants().get("rates", True))

    def in_world(self, base: Array, local: Array) -> Array:
        """Poses in the segment's base frame (``(k, 7)``), in the world."""
        return jax.vmap(compose, in_axes=(None, 0))(base, local)

    def rates_in_world(self, base: Array, mount: Array, local: Array) -> Array:
        """Angular velocities relative to the segment's base, in its base frame's axes (``(k, 3)``), in the world:
        each turned into the world's axes, with its mount frame's own added."""
        return mount + quaternion.rotate(base[3:], local.T).T


def flattened(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float64).ravel())


ARM_EQUATIONS = (r"\bar{u}_{i} = \min(\max(u_{i}, l_{i}), h_{i}); \quad "
                 r"\theta_{n+1} = \bar{u} + (\theta_{n} - \bar{u}) \exp(-\mathrm{d}t / \tau); \quad "
                 r"\mathbf{P}_{k} = \mathbf{M}_{j} \circ \mathbf{O} \circ \mathrm{FK}_{k}(\theta_{n+1}; \mathbf{a}, "
                 r"\mathbf{J}, \mathbf{K}), \quad "
                 r"\boldsymbol{\Omega}_{k} = \boldsymbol{\omega}_{j} + R(\mathbf{M}_{j} \circ \mathbf{O}) \, "
                 r"\mathrm{W}_{k}(\theta_{n+1}, (\theta_{n+1} - \theta_{n}) / \mathrm{d}t)")


class KinematicArmNode(SegmentNode):
    """A multibody tree whose joint positions are commanded: each commanded position is held within its joint's
    limits where it has them, the joints follow it at once or, with a servo time constant, as a first-order lag held
    exact over the step, and every frame's world pose follows by forward kinematics, and its angular velocity from the
    joints' rates over the step, ``(theta_{n+1} - theta_n) / dt``, with the mount frame's, where ``rates`` holds (the
    build sets it where something reads them: a segment or a spinner mounted on it, a controller, an observation;
    without it the node publishes no ``frame_angular_velocities``, and its step is its poses alone). No mass, no load:
    its motion is prescribed. A revolute joint's position is commanded in ``joint_positions`` (rad) and a prismatic one's
    in ``joint_displacements`` (m), each in the order of its kind's joints; a joint not commanded holds.

    The tree is taken as its articulation's arrays, so that the node is rebuilt from its saved form alone and a study
    can move a joint's axis or origin: ``axes``, ``origin_translations`` and ``origin_rotations`` of its moving joints,
    ``frame_translations`` and ``frame_rotations`` of its frames, flattened; its structure (``parents``, ``revolute``,
    ``frame_bodies``) and names are held as they are. ``lower`` and ``upper`` are the limits of the joints
    ``bounded`` names, which no study fits."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-104",
        description="A kinematic chain segment: commanded joint positions held within their limits, an optional "
                    "first-order servo lag, and every frame's world pose by forward kinematics, placed on the frame it "
                    "mounts on.",
        governing_equations=ARM_EQUATIONS,
        discretization="The servo's first-order lag is solved exactly over the step for a command held through it.",
        assumptions=("Rigid joints and links, with no structural mode (MIME-ANO-101)",
                     "Its motion is prescribed: it takes no load, and its reactions are not applied"),
        references=(Reference("Sciavicco2000", "forward kinematics of a serial chain"),
                    Reference("Featherstone2008", "a kinematic tree and its frames")),
    )
    symbols = (
        Symbol("u", SymbolKind.INPUT, "joint_positions", "1", f"{_ARM}.advance"),
        Symbol("l", SymbolKind.PARAMETER, "lower", "1", f"{_ARM}.held"),
        Symbol("h", SymbolKind.PARAMETER, "upper", "1", f"{_ARM}.held"),
        Symbol(r"\theta", SymbolKind.STATE, "joint_positions", "1",
               f"{_ARM}.advance"),
        Symbol(r"\tau", SymbolKind.PARAMETER, "servo_time_constant", "s",
               f"{_ARM}.advance"),
        Symbol(r"\mathbf{P}", SymbolKind.STATE, "frame_poses", POSE_UNITS,
               f"{_ARM}.advance"),
        Symbol(r"\boldsymbol{\Omega}", SymbolKind.STATE, "frame_angular_velocities", RATE_UNITS, f"{_ARM}.advance"),
        Symbol(r"\boldsymbol{\omega}", SymbolKind.INPUT, "mount_rates", RATE_UNITS, f"{_MOUNTED}.base_rate"),
        Symbol(r"\mathbf{M}", SymbolKind.INPUT, "mount_poses", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{O}", SymbolKind.PARAMETER, "placement", POSE_UNITS,
               f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "axes", "1", f"{_ARM}.frames"),
        Symbol(r"\mathbf{J}", SymbolKind.PARAMETER, "origin_translations", "m",
               f"{_ARM}.frames"),
        Symbol(r"\mathbf{K}", SymbolKind.PARAMETER, "frame_translations", "m",
               f"{_ARM}.frames"),
    )

    def __init__(self, name: str, timestep: float, *, dof_names: Sequence[str], parents: Sequence[int],
                 revolute: Sequence[bool], frame_names: Sequence[str], frame_bodies: Sequence[int],
                 axes: Sequence[float], origin_translations: Sequence[float], origin_rotations: Sequence[float],
                 frame_translations: Sequence[float], frame_rotations: Sequence[float], bounded: Sequence[int],
                 lower: Sequence[float], upper: Sequence[float], placement: Sequence[float], mount_index: int = -1,
                 mount_count: int = 0, initial_mount_pose: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                 initial_mount_rate: Sequence[float] = (0.0, 0.0, 0.0),
                 initial_joint_positions: Sequence[float], servo_time_constant: float | None = None,
                 rates: bool = True) -> None:
        super().__init__(
            name, timestep, dof_names=tuple(dof_names), parents=tuple(int(i) for i in parents),
            revolute=tuple(bool(r) for r in revolute), frame_names=tuple(frame_names),
            frame_bodies=tuple(int(i) for i in frame_bodies), axes=flattened(axes),
            origin_translations=flattened(origin_translations), origin_rotations=flattened(origin_rotations),
            frame_translations=flattened(frame_translations), frame_rotations=flattened(frame_rotations),
            bounded=tuple(int(i) for i in bounded), lower=flattened(lower), upper=flattened(upper),
            placement=flattened(placement), mount_index=int(mount_index), mount_count=int(mount_count),
            initial_mount_pose=flattened(initial_mount_pose), initial_mount_rate=flattened(initial_mount_rate),
            initial_joint_positions=flattened(initial_joint_positions),
            servo_time_constant=None if servo_time_constant is None else float(servo_time_constant), rates=bool(rates))

    @staticmethod
    def described(articulation: Articulation) -> dict[str, Any]:
        """The constructor arguments that hold ``articulation``: its structure and its arrays, flattened, and the
        limits of the joints that have them."""
        held = articulation
        bounded = tuple(i for i in range(held.dofs) if np.isfinite(held.lower[i]) or np.isfinite(held.upper[i]))
        return {"dof_names": held.dof_names, "parents": held.parents, "revolute": held.revolute,
                "frame_names": held.frame_names, "frame_bodies": held.frame_bodies, "axes": flattened(held.axes),
                "origin_translations": flattened(held.origin_translations),
                "origin_rotations": flattened(held.origin_rotations),
                "frame_translations": flattened(held.frame_translations),
                "frame_rotations": flattened(held.frame_rotations), "bounded": bounded,
                "lower": flattened([held.lower[i] for i in bounded]),
                "upper": flattened([held.upper[i] for i in bounded])}

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(),
                "lower": ParamSpec(trainable=False, units="1", description="the bounded joints' lower limits"),
                "upper": ParamSpec(trainable=False, units="1", description="the bounded joints' upper limits"),
                "servo_time_constant": ParamSpec(units="s"), "placement": ParamSpec(units=POSE_UNITS),
                "axes": ParamSpec(units="1"), "origin_translations": ParamSpec(units="m"),
                "frame_translations": ParamSpec(units="m")}

    def _dofs(self) -> int:
        return len(cast(tuple[str, ...], self.constants()["dof_names"]))

    def frames(self, p: Mapping[str, Any], q: Array) -> Array:
        """Every frame's pose in the segment's base frame, by forward kinematics on the articulation the arrays
        ``p`` hold."""
        return forward_kinematics(self.articulation(p, q.dtype), q)

    def kinds(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The indices of its revolute joints and of its prismatic ones."""
        revolute = cast(tuple[bool, ...], self.constants()["revolute"])
        return (tuple(i for i, r in enumerate(revolute) if r), tuple(i for i, r in enumerate(revolute) if not r))

    def articulation(self, p: Mapping[str, Any], dtype: Any) -> Articulation:
        """The articulation the arrays ``p`` hold, massless."""
        held = self.constants()
        n, k = self._dofs(), len(cast(tuple[str, ...], held["frame_names"]))

        def shaped(name: str, *shape: int) -> Array:
            return jnp.reshape(jnp.asarray(p[name], dtype=dtype), shape)

        return Articulation(
            dof_names=cast(tuple[str, ...], held["dof_names"]), parents=cast(tuple[int, ...], held["parents"]),
            revolute=cast(tuple[bool, ...], held["revolute"]), frame_names=cast(tuple[str, ...], held["frame_names"]),
            frame_bodies=cast(tuple[int, ...], held["frame_bodies"]), closures=(),
            axes=shaped("axes", n, 3), origin_rotations=shaped("origin_rotations", n, 4),
            origin_translations=shaped("origin_translations", n, 3), masses=jnp.zeros((n,), dtype=dtype),
            centres_of_mass=jnp.zeros((n, 3), dtype=dtype), inertias=jnp.zeros((n, 3, 3), dtype=dtype),
            armatures=jnp.zeros((n,), dtype=dtype), frame_rotations=shaped("frame_rotations", k, 4),
            frame_translations=shaped("frame_translations", k, 3), lower=jnp.zeros((n,), dtype=dtype),
            upper=jnp.zeros((n,), dtype=dtype))

    def commanded(self, given: ArmInputs, previous: Array) -> Array:
        """The positions commanded: each kind's command at its joints, the joints a command does not reach held."""
        revolute, prismatic = self.kinds()
        if not prismatic:
            return given["joint_positions"] if "joint_positions" in given else previous
        command = previous
        for indices, name in ((revolute, "joint_positions"), (prismatic, "joint_displacements")):
            if indices and name in given:
                command = command.at[np.asarray(indices)].set(jnp.asarray(cast(Any, given)[name], previous.dtype))
        return command

    def held(self, p: Mapping[str, Any], command: Array) -> Array:
        """The command with each bounded joint's position held within its limits."""
        bounded = cast(tuple[int, ...], self.constants()["bounded"])
        if not bounded:
            return command
        index = np.asarray(bounded)
        inside = command[index]
        kept = switch.clip(inside, jnp.asarray(p["lower"], dtype=command.dtype),
                           jnp.asarray(p["upper"], dtype=command.dtype), owner="arm.limit",
                           resolution=lambda: switch.rounding(inside), indicator=inside)
        return command.at[index].set(kept)

    @override
    def initial_state(self) -> NodeFields:
        q = as_run_array(self.constants()["initial_joint_positions"])
        p = {key: as_run_array(value) for key, value in self.constants().items() if key in _ARRAYS and
             value is not None}
        frames = len(cast(tuple[str, ...], self.constants()["frame_names"]))
        poses = self.in_world(self.initial_base(), self.frames(p, q))
        if not self.publishes_rates():
            return node_fields(ArmState(frame_poses=poses, joint_positions=q))
        return node_fields(ArmState(frame_poses=poses, frame_angular_velocities=jnp.tile(self.initial_base_rate(),
                                                                                          (frames, 1)),
                                    joint_positions=q))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        revolute, prismatic = self.kinds()
        commands = {"joint_positions": node_input("1", (len(revolute),))} if revolute else {}
        if prismatic:
            commands["joint_displacements"] = node_input("m", (len(prismatic),))
        return {**commands, **self.mounted_inputs()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(ArmState, state), typed_view(ArmInputs, boundary_inputs)
        previous = held["joint_positions"]
        target = self.held(p, self.commanded(given, previous))
        tau = p.get("servo_time_constant")
        q = target if tau is None else target + (previous - target) * jnp.exp(-dt / jnp.asarray(tau, target.dtype))
        base = self.base(p, given)
        if not self.publishes_rates():
            return node_fields(ArmState(frame_poses=self.in_world(base, self.frames(p, q)), joint_positions=q))
        frames, local = frame_motions(self.articulation(p, q.dtype), q, (q - previous) / dt)
        rates = self.rates_in_world(base, self.base_rate(given, q.dtype), local)
        return node_fields(ArmState(frame_poses=self.in_world(base, frames), frame_angular_velocities=rates,
                                    joint_positions=q))


#: The numeric constants of an arm a step reads: the arrays of its articulation, and its placement.
_ARRAYS = ("axes", "origin_translations", "origin_rotations", "frame_translations", "frame_rotations", "lower", "upper",
           "placement", "servo_time_constant")


MOTOR_EQUATIONS = (r"\phi_{n+1} = a \quad \text{(an angle prescribed)}, \qquad "
                   r"\phi_{n+1} = \phi_{n} + \omega \, \mathrm{d}t \quad \text{(a speed prescribed)}; \quad "
                   r"\mathbf{P}_{s} = \mathbf{M}_{j} \circ \mathbf{O}, \quad "
                   r"\mathbf{P}_{r} = \mathbf{P}_{s} \circ \mathrm{Rot}(\mathbf{e}, \phi_{n+1}); \quad "
                   r"\boldsymbol{\Omega}_{s} = \boldsymbol{\omega}_{j}, \quad "
                   r"\boldsymbol{\Omega}_{r} = \boldsymbol{\omega}_{j} + \dot{\phi} \, R(\mathbf{P}_{s}) \mathbf{e}")


class KinematicMotorNode(SegmentNode):
    """A rotor turning about ``axis`` (a unit vector in its base frame, the frame it mounts on turned by its
    placement) by a prescribed angle, or at a prescribed speed integrated over each step. Its frames are ``base``,
    the stator, and ``rotor``; the stator turns with the frame it mounts on, and the rotor besides at the prescribed
    speed (an angle's change over the step, when the angle is prescribed), each published where ``rates`` holds, as
    the kinematic arm's are. An ideal source of the prescribed motion: no rotor inertia, no load."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-105",
        description="A kinematic motor: a rotor turned by a prescribed angle, or at a prescribed speed integrated "
                    "over each step, about an axis of the frame it mounts on.",
        governing_equations=MOTOR_EQUATIONS,
        discretization="A prescribed speed is integrated by the rectangle rule over the step, exact for a speed held "
                       "through it.",
        assumptions=("An ideal source of the prescribed motion: no cogging, imbalance or thermal terms "
                     "(MIME-ANO-102)", "Its motion is prescribed: it takes no load, and its reaction is not applied"),
        references=(Reference("Sola2017", "a rotation about an axis by the exponential map"),),
    )
    symbols = (
        Symbol(r"\phi", SymbolKind.STATE, "angle", "1", f"{_MOTOR}.advance"),
        Symbol("a", SymbolKind.INPUT, "angle", "1", f"{_MOTOR}.advance"),
        Symbol(r"\omega", SymbolKind.INPUT, "velocity", "Hz", f"{_MOTOR}.advance"),
        Symbol(r"\mathbf{P}", SymbolKind.STATE, "frame_poses", POSE_UNITS,
               f"{_MOTOR}.advance"),
        Symbol(r"\boldsymbol{\Omega}", SymbolKind.STATE, "frame_angular_velocities", RATE_UNITS, f"{_MOTOR}.rates"),
        Symbol(r"\boldsymbol{\omega}", SymbolKind.INPUT, "mount_rates", RATE_UNITS, f"{_MOUNTED}.base_rate"),
        Symbol(r"\mathbf{M}", SymbolKind.INPUT, "mount_poses", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{O}", SymbolKind.PARAMETER, "placement", POSE_UNITS,
               f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{e}", SymbolKind.PARAMETER, "axis", "1", f"{_MOTOR}.poses"),
    )
    FRAMES = ("base", "rotor")

    def __init__(self, name: str, timestep: float, *, axis: Sequence[float], prescribed: str,
                 placement: Sequence[float], mount_index: int = -1, mount_count: int = 0,
                 initial_mount_pose: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                 initial_mount_rate: Sequence[float] = (0.0, 0.0, 0.0), initial_angle: float,
                 rates: bool = True) -> None:
        if prescribed not in ("angle", "velocity"):
            raise ValueError(f"prescribed is 'angle' or 'velocity', not {prescribed!r}")
        super().__init__(name, timestep, axis=flattened(axis), prescribed=prescribed, placement=flattened(placement),
                         mount_index=int(mount_index), mount_count=int(mount_count),
                         initial_mount_pose=flattened(initial_mount_pose),
                         initial_mount_rate=flattened(initial_mount_rate), initial_angle=float(initial_angle),
                         rates=bool(rates))

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "axis": ParamSpec(units="1"),
                "placement": ParamSpec(units=POSE_UNITS)}

    def poses(self, p: Mapping[str, Any], base: Array, angle: Array) -> Array:
        """The stator's and the rotor's world poses: the base, and the base turned by ``angle`` about the axis."""
        axis = jnp.asarray(p["axis"], dtype=angle.dtype)
        turned = jnp.concatenate([jnp.zeros(3, dtype=angle.dtype), quaternion.exponential(angle * axis)])
        return jnp.stack([base, compose(base, turned)])

    def rates(self, p: Mapping[str, Any], base: Array, mount: Array, rate: Array) -> Array:
        """The stator's and the rotor's angular velocities in the world: the mount frame's, and the rotor's ``rate``
        about the axis besides."""
        axis = quaternion.rotate(base[3:], jnp.asarray(p["axis"], dtype=rate.dtype))
        return jnp.stack([mount, mount + rate * axis])

    @override
    def initial_state(self) -> NodeFields:
        angle = as_run_array(self.constants()["initial_angle"])
        p = {"axis": as_run_array(self.constants()["axis"])}
        poses = self.poses(p, self.initial_base(), angle)
        if not self.publishes_rates():
            return node_fields(MotorState(frame_poses=poses, angle=angle))
        mount = self.initial_base_rate()
        return node_fields(MotorState(frame_poses=poses, frame_angular_velocities=jnp.stack([mount, mount]),
                                      angle=angle))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"angle": node_input("1", ()), "velocity": node_input("Hz", ()), **self.mounted_inputs()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        held, given = typed_view(MotorState, state), typed_view(MotorInputs, boundary_inputs)
        previous = held["angle"]
        if self.constants()["prescribed"] == "angle":
            angle = given["angle"] if "angle" in given else previous
            rate = (angle - previous) / dt
        else:
            rate = given["velocity"] if "velocity" in given else as_run_array(0.0)
            angle = previous + rate * dt
        base = self.base(p, given)
        if not self.publishes_rates():
            return node_fields(MotorState(frame_poses=self.poses(p, base, angle), angle=angle))
        return node_fields(MotorState(frame_poses=self.poses(p, base, angle),
                                      frame_angular_velocities=self.rates(p, base, self.base_rate(given, angle.dtype),
                                                                          jnp.asarray(rate, angle.dtype)),
                                      angle=angle))


__all__ = ["ARM_EQUATIONS", "MOTOR_EQUATIONS", "POSE_UNITS", "RATE_UNITS", "ArmInputs", "ArmState", "KinematicArmNode",
           "KinematicMotorNode", "MotorInputs", "MotorState", "MountInputs", "MountedNode", "SegmentNode", "flattened"]
