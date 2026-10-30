"""The nodes of an apparatus chain's dynamic rungs: an arm whose joints are driven by torques, and a motor whose rotor
is. Each takes loads on its frames (``link_wrench``, one row per frame: ``[force; torque about the frame's origin]``
in the world), publishes the world pose of every frame it names as the kinematic rungs do, and integrates its joints
by semi-implicit Euler, first order in time. A motor mounted on another segment publishes the wrench its stator
exerts on that segment's frame (``reaction``), which the build delivers into that segment's loads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast, override

import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import DiscretizationOrder, NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import NotRequired, ReadOnly, TypedDict

from mime.core import NodeFields, Symbol, SymbolKind, as_run_array, jnp, node_fields, node_input, quaternion, \
    switch, typed_view
from mime.core.multibody import Articulation, cholesky_solve, compose, forward_kinematics, frame_forces, \
    inverse_dynamics, mass_matrix
from mime.nodes.actuation._chain import POSE_UNITS, MountInputs, SegmentNode, flattened

#: The units of a frame's load, as the vocabulary spells ``apparatus.link_wrench``.
WRENCH_UNITS = "N; N m"
_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
#: Where the symbol rows' functions are.
_MOUNTED, _ARM, _MOTOR = (f"mime.nodes.actuation.{module}.{name}" for module, name in (
    ("_chain", "MountedNode"), ("_dynamic", "DynamicArmNode"), ("_dynamic", "DynamicMotorNode")))


class DynamicArmState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    joint_positions: ReadOnly[Float[Array, " dofs"]]
    joint_velocities: ReadOnly[Float[Array, " dofs"]]


class DynamicArmInputs(MountInputs, closed=True):
    joint_torques: NotRequired[ReadOnly[Float[Array, " dofs"]]]
    link_wrench: NotRequired[ReadOnly[Float[Array, "frames 6"]]]


class DynamicMotorState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    angle: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    angular_velocity: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    reaction: ReadOnly[Float[Array, "mounted 6"]]


class DynamicMotorInputs(MountInputs, closed=True):
    torque: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    link_wrench: NotRequired[ReadOnly[Float[Array, "frames 6"]]]


ARM_EQUATIONS = (
    r"\mathbf{A} = \mathbf{M}(\theta_{n}) + \mathrm{d}t \, \mathbf{D}, \quad "
    r"\dot{\theta}^{*} = \mathbf{A}^{-1} \left( \mathbf{M}(\theta_{n}) \dot{\theta}_{n} + \mathrm{d}t \, "
    r"(\mathbf{u} - \mathbf{h}(\theta_{n}, \dot{\theta}_{n}; \mathbf{g}, \mathbf{W})) \right); \quad "
    r"c_{j} = \min(\max(A_{jj} \dot{\theta}^{*}_{j} / \mathrm{d}t, -\mu_{j}), \mu_{j}), \quad "
    r"\dot{\theta}_{n+1} = \dot{\theta}^{*} - \mathrm{d}t \, \mathbf{A}^{-1} \mathbf{c}; \quad "
    r"\theta_{n+1} = \min(\max(\theta_{n} + \mathrm{d}t \, \dot{\theta}_{n+1}, l), h), \ "
    r"\dot{\theta}_{n+1} = 0 \text{ where held}; \quad "
    r"\mathbf{P}_{k} = \mathbf{O} \circ \mathrm{FK}_{k}(\theta_{n+1}; \mathbf{a}, \mathbf{J}, \mathbf{K})")


class DynamicArmNode(SegmentNode):
    """A multibody tree whose joints are driven by torques: the mass matrix ``M`` by the composite-rigid-body
    algorithm, and the bias ``h`` (velocity products, gravity and the loads on its frames) by the recursive
    Newton-Euler algorithm, both from ``mime.core.multibody``; each joint's viscous damping ``D`` implicit, and its
    Coulomb friction ``mu`` as the impulse that at most stops it over the step; the joint limits rigid stops that hold
    a joint at its limit and take its velocity there. Its base is fixed in the world at ``placement``; ``gravity`` is
    the world's, and loads on its frames are given in the world, each about its frame's origin.

    The tree is taken as its articulation's arrays, as the kinematic arm's are, with its inertials (``masses``,
    ``centres_of_mass``, ``inertias``, flattened), ``armatures``, ``damping`` and ``friction`` per moving joint."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-109",
        description="A dynamic chain segment: joint torques in, the mass matrix by CRBA and the bias by RNEA with "
                    "gravity and the loads on its frames, implicit joint damping, Coulomb friction as a stopping "
                    "impulse, rigid joint stops, and every frame's world pose by forward kinematics.",
        governing_equations=ARM_EQUATIONS,
        discretization="Semi-implicit Euler, first order in time: the velocity from the forces at the start of the "
                       "step, the joint damping taken at its end; then the position from the new velocity.",
        assumptions=("Rigid joints and links, with no structural mode (MIME-ANO-101)",
                     "Its base is fixed in the world",
                     "Coulomb friction stops each joint on its own share of the mass matrix: exact for one joint, a "
                     "first iterate where joints are coupled",
                     "A joint limit is a rigid, inelastic stop"),
        references=(Reference("Featherstone2008", "the composite-rigid-body and recursive Newton-Euler algorithms"),
                    Reference("Sciavicco2000", "the dynamics of a serial chain")),
    )
    symbols = (
        Symbol(r"\mathbf{u}", SymbolKind.INPUT, "joint_torques", "N m", f"{_ARM}.advance"),
        Symbol(r"\mathbf{W}", SymbolKind.INPUT, "link_wrench", WRENCH_UNITS, f"{_ARM}.loads"),
        Symbol(r"\mathbf{g}", SymbolKind.PARAMETER, "gravity", "m/s^2", f"{_ARM}.advance"),
        Symbol(r"\mathbf{D}", SymbolKind.PARAMETER, "damping", "N m s", f"{_ARM}.advance"),
        Symbol(r"\mu", SymbolKind.PARAMETER, "friction", "N m", f"{_ARM}.stopped"),
        Symbol("l", SymbolKind.PARAMETER, "lower", "1", f"{_ARM}.held"),
        Symbol("h", SymbolKind.PARAMETER, "upper", "1", f"{_ARM}.held"),
        Symbol(r"\theta", SymbolKind.STATE, "joint_positions", "1", f"{_ARM}.advance"),
        Symbol(r"\dot{\theta}", SymbolKind.STATE, "joint_velocities", "Hz", f"{_ARM}.advance"),
        Symbol(r"\mathbf{P}", SymbolKind.STATE, "frame_poses", POSE_UNITS, f"{_ARM}.advance"),
        Symbol(r"\mathbf{O}", SymbolKind.PARAMETER, "placement", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "axes", "1", f"{_ARM}.articulation"),
        Symbol(r"\mathbf{J}", SymbolKind.PARAMETER, "origin_translations", "m", f"{_ARM}.articulation"),
        Symbol(r"\mathbf{K}", SymbolKind.PARAMETER, "frame_translations", "m", f"{_ARM}.articulation"),
    )

    def __init__(self, name: str, timestep: float, *, dof_names: Sequence[str], parents: Sequence[int],
                 revolute: Sequence[bool], frame_names: Sequence[str], frame_bodies: Sequence[int],
                 axes: Sequence[float], origin_translations: Sequence[float], origin_rotations: Sequence[float],
                 frame_translations: Sequence[float], frame_rotations: Sequence[float], masses: Sequence[float],
                 centres_of_mass: Sequence[float], inertias: Sequence[float], armatures: Sequence[float],
                 damping: Sequence[float], friction: Sequence[float], bounded: Sequence[int],
                 lower: Sequence[float], upper: Sequence[float], placement: Sequence[float],
                 gravity: Sequence[float], mount_index: int = -1, mount_count: int = 0,
                 initial_mount_pose: Sequence[float] = _IDENTITY, initial_joint_positions: Sequence[float],
                 initial_joint_velocities: Sequence[float] | None = None) -> None:
        if mount_index != -1:
            raise ValueError("a dynamic arm's base is fixed in the world: it mounts on nothing (mount_index -1)")
        dofs = len(dof_names)
        super().__init__(
            name, timestep, dof_names=tuple(dof_names), parents=tuple(int(i) for i in parents),
            revolute=tuple(bool(r) for r in revolute), frame_names=tuple(frame_names),
            frame_bodies=tuple(int(i) for i in frame_bodies), axes=flattened(axes),
            origin_translations=flattened(origin_translations), origin_rotations=flattened(origin_rotations),
            frame_translations=flattened(frame_translations), frame_rotations=flattened(frame_rotations),
            masses=flattened(masses), centres_of_mass=flattened(centres_of_mass), inertias=flattened(inertias),
            armatures=flattened(armatures), damping=flattened(damping), friction=flattened(friction),
            bounded=tuple(int(i) for i in bounded), lower=flattened(lower), upper=flattened(upper),
            placement=flattened(placement), gravity=flattened(gravity), mount_index=-1,
            mount_count=int(mount_count), initial_mount_pose=flattened(initial_mount_pose),
            initial_joint_positions=flattened(initial_joint_positions),
            initial_joint_velocities=flattened(initial_joint_velocities if initial_joint_velocities is not None
                                               else (0.0,) * dofs))

    @staticmethod
    def described(articulation: Articulation, damping: Sequence[float], friction: Sequence[float]) -> dict[str, Any]:
        """The constructor arguments that hold ``articulation`` with its inertials, and each moving joint's
        ``damping`` and ``friction``, in its order."""
        held = articulation
        bounded = tuple(i for i in range(held.dofs) if np.isfinite(held.lower[i]) or np.isfinite(held.upper[i]))
        return {"dof_names": held.dof_names, "parents": held.parents, "revolute": held.revolute,
                "frame_names": held.frame_names, "frame_bodies": held.frame_bodies, "axes": flattened(held.axes),
                "origin_translations": flattened(held.origin_translations),
                "origin_rotations": flattened(held.origin_rotations),
                "frame_translations": flattened(held.frame_translations),
                "frame_rotations": flattened(held.frame_rotations), "masses": flattened(held.masses),
                "centres_of_mass": flattened(held.centres_of_mass), "inertias": flattened(held.inertias),
                "armatures": flattened(held.armatures), "damping": tuple(float(d) for d in damping),
                "friction": tuple(float(f) for f in friction), "bounded": bounded,
                "lower": flattened([held.lower[i] for i in bounded]),
                "upper": flattened([held.upper[i] for i in bounded])}

    def discretization_order(self) -> DiscretizationOrder:
        return DiscretizationOrder(temporal=1.0)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(),
                "lower": ParamSpec(trainable=False, units="1", description="the bounded joints' lower limits"),
                "upper": ParamSpec(trainable=False, units="1", description="the bounded joints' upper limits"),
                "placement": ParamSpec(units=POSE_UNITS), "gravity": ParamSpec(units="m/s^2"),
                "axes": ParamSpec(units="1"), "origin_translations": ParamSpec(units="m"),
                "frame_translations": ParamSpec(units="m"), "masses": ParamSpec(units="kg", bounds=(0.0, None)),
                "centres_of_mass": ParamSpec(units="m"), "inertias": ParamSpec(units="kg m^2"),
                "armatures": ParamSpec(units="kg m^2", bounds=(0.0, None)),
                "damping": ParamSpec(units="N m s", bounds=(0.0, None)),
                "friction": ParamSpec(units="N m", bounds=(0.0, None))}

    def _counts(self) -> tuple[int, int]:
        held = self.constants()
        return len(cast(tuple[str, ...], held["dof_names"])), len(cast(tuple[str, ...], held["frame_names"]))

    def articulation(self, p: Mapping[str, Any], dtype: Any) -> Articulation:
        """The articulation the arrays ``p`` hold, in ``dtype``."""
        held = self.constants()
        n, k = self._counts()

        def shaped(name: str, *shape: int) -> Array:
            return jnp.reshape(jnp.asarray(p[name], dtype=dtype), shape)

        return Articulation(
            dof_names=cast(tuple[str, ...], held["dof_names"]), parents=cast(tuple[int, ...], held["parents"]),
            revolute=cast(tuple[bool, ...], held["revolute"]), frame_names=cast(tuple[str, ...], held["frame_names"]),
            frame_bodies=cast(tuple[int, ...], held["frame_bodies"]), closures=(),
            axes=shaped("axes", n, 3), origin_rotations=shaped("origin_rotations", n, 4),
            origin_translations=shaped("origin_translations", n, 3), masses=shaped("masses", n),
            centres_of_mass=shaped("centres_of_mass", n, 3), inertias=shaped("inertias", n, 3, 3),
            armatures=shaped("armatures", n), frame_rotations=shaped("frame_rotations", k, 4),
            frame_translations=shaped("frame_translations", k, 3), lower=jnp.zeros((n,), dtype=dtype),
            upper=jnp.zeros((n,), dtype=dtype))

    def loads(self, articulation: Articulation, q: Array, base: Array, wrenches: Array) -> Array:
        """The loads on its frames, given in the world about each frame's origin, as spatial forces on its bodies in
        its base's frame about the base's origin."""
        turned = jnp.concatenate([quaternion.rotate_back(base[3:], wrenches[:, :3].T).T,
                                  quaternion.rotate_back(base[3:], wrenches[:, 3:].T).T], axis=1)
        return frame_forces(articulation, q, turned)

    def stopped(self, p: Mapping[str, Any], system: Array, factor: Array, velocity: Array, dt: float) -> Array:
        """The velocity after each joint's Coulomb friction: the impulse that would stop the joint alone on its own
        diagonal of ``system`` (``M + dt D``, whose Cholesky factor is ``factor``), held within its friction."""
        friction = jnp.asarray(p["friction"], dtype=velocity.dtype)
        needed = jnp.diagonal(system) * velocity / dt
        held = switch.clip(needed, -friction, friction, owner="arm.friction",
                           resolution=lambda: switch.rounding(needed), indicator=needed)
        return velocity - dt * cholesky_solve(factor, held)

    def held(self, p: Mapping[str, Any], q: Array, qd: Array) -> tuple[Array, Array]:
        """Each bounded joint held within its limits, and its velocity taken where it is held."""
        bounded = cast(tuple[int, ...], self.constants()["bounded"])
        if not bounded:
            return q, qd
        index = np.asarray(bounded)
        inside = q[index]
        lower, upper = (jnp.asarray(p[name], dtype=q.dtype) for name in ("lower", "upper"))
        kept = switch.clip(inside, lower, upper, owner="arm.stop", resolution=lambda: switch.rounding(inside),
                           indicator=inside)
        rate = qd[index]
        rate = switch.where(inside > lower, rate, jnp.zeros_like(rate), owner="arm.stop.low",
                            resolution=lambda: switch.rounding(inside), indicator=inside - lower)
        rate = switch.where(inside < upper, rate, jnp.zeros_like(rate), owner="arm.stop.high",
                            resolution=lambda: switch.rounding(inside), indicator=upper - inside)
        return q.at[index].set(kept), qd.at[index].set(rate)

    @override
    def initial_state(self) -> NodeFields:
        held = self.constants()
        q = as_run_array(held["initial_joint_positions"])
        p = {key: as_run_array(value) for key, value in held.items() if key in _ARRAYS}
        frames = forward_kinematics(self.articulation(p, q.dtype), q)
        return node_fields(DynamicArmState(frame_poses=self.in_world(self.initial_base(), frames),
                                           joint_positions=q,
                                           joint_velocities=as_run_array(held["initial_joint_velocities"])))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        dofs, frames = self._counts()
        return {"joint_torques": node_input("N m", (dofs,)),
                "link_wrench": node_input(WRENCH_UNITS, (frames, 6), additive=True), **self.mounted_inputs()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        now, given = typed_view(DynamicArmState, state), typed_view(DynamicArmInputs, boundary_inputs)
        q, qd = now["joint_positions"], now["joint_velocities"]
        dofs, frames = self._counts()
        articulation = self.articulation(p, q.dtype)
        base = self.base(p, given)
        torques = given["joint_torques"] if "joint_torques" in given else jnp.zeros((dofs,), dtype=q.dtype)
        wrenches = given["link_wrench"] if "link_wrench" in given else jnp.zeros((frames, 6), dtype=q.dtype)
        gravity = quaternion.rotate_back(base[3:], jnp.asarray(p["gravity"], dtype=q.dtype))
        bias = inverse_dynamics(articulation, q, qd, None, gravity, self.loads(articulation, q, base, wrenches))
        inertia = mass_matrix(articulation, q)
        system = inertia + dt * jnp.diag(jnp.asarray(p["damping"], dtype=q.dtype))
        factor = jnp.linalg.cholesky(system)
        velocity = self.stopped(p, system, factor, cholesky_solve(factor, inertia @ qd + dt * (torques - bias)), dt)
        q_next, velocity = self.held(p, q + dt * velocity, velocity)
        poses = self.in_world(base, forward_kinematics(articulation, q_next))
        return node_fields(DynamicArmState(frame_poses=poses, joint_positions=q_next, joint_velocities=velocity))


#: The numeric constants of a dynamic arm a step reads.
_ARRAYS = ("axes", "origin_translations", "origin_rotations", "frame_translations", "frame_rotations", "masses",
           "centres_of_mass", "inertias", "armatures", "damping", "friction", "lower", "upper", "placement", "gravity")


MOTOR_EQUATIONS = (
    r"\tau_{l} = \mathbf{e} \cdot \mathbf{T}_{r}, \quad "
    r"\omega_{n+1} = \frac{J \omega_{n} + \mathrm{d}t \, (\tau + \tau_{l})}{J + \mathrm{d}t \, b}, \quad "
    r"\phi_{n+1} = \phi_{n} + \mathrm{d}t \, \omega_{n+1}; \quad "
    r"\mathbf{R} = \begin{bmatrix} \mathbf{F}_{s} + \mathbf{F}_{r} \\ \mathbf{T}_{s} + \mathbf{T}_{r} - "
    r"(\mathbf{e} \cdot \mathbf{T}_{r}) \mathbf{e} - (\tau - b \, \omega_{n+1}) \mathbf{e} + "
    r"(\mathbf{x}_{s} - \mathbf{x}_{m}) \times (\mathbf{F}_{s} + \mathbf{F}_{r}) \end{bmatrix}; \quad "
    r"\mathbf{P}_{s} = \mathbf{M}_{j} \circ \mathbf{O}, \quad "
    r"\mathbf{P}_{r} = \mathbf{P}_{s} \circ \mathrm{Rot}(\mathbf{a}, \phi_{n+1}), \quad \mathbf{e} = R(\mathbf{P}_{s}) "
    r"\mathbf{a}")


class DynamicMotorNode(SegmentNode):
    """A rotor of inertia ``J`` about ``axis`` (a unit vector in its base frame), driven by the torque ``tau`` against
    viscous bearing damping ``b`` and the loads on its rotor about the axis, by semi-implicit Euler with the damping
    implicit. Its frames are ``base``, the stator, and ``rotor``, on one origin.

    Its stator holds the rotor: it takes the reaction of the drive and of the bearing's damping about the axis, and
    the loads on the rotor across it (its force and its torque about any other axis), and with the loads on the
    stator itself passes them to the frame it is mounted on, as ``reaction``: one row per frame of the segment it
    mounts on, that frame's row ``[force; torque about the frame's origin]`` in the world, the others zero."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-110",
        description="A dynamic motor: a rotor driven by a torque against its inertia, viscous bearing damping and the "
                    "loads on it about its axis, and the wrench its stator passes to the frame it is mounted on.",
        governing_equations=MOTOR_EQUATIONS,
        discretization="Semi-implicit Euler, first order in time: the rotor's speed with its damping taken at the "
                       "end of the step, then its angle from the new speed.",
        assumptions=("An ideal torque source: no cogging, imbalance, armature or thermal terms (MIME-ANO-102)",
                     "Rigid bearings, and a rotor whose mass and inertia across its axis the frame it is mounted on "
                     "does not carry",
                     "The rotor's speed relative to its stator, with no term from the stator's own turning"),
        references=(Reference("Sola2017", "a rotation about an axis by the exponential map"),
                    Reference("Featherstone2008", "the reaction of a joint's actuator on its parent")),
    )
    symbols = (
        Symbol(r"\phi", SymbolKind.STATE, "angle", "1", f"{_MOTOR}.advance"),
        Symbol(r"\omega", SymbolKind.STATE, "angular_velocity", "Hz", f"{_MOTOR}.advance"),
        Symbol(r"\tau", SymbolKind.INPUT, "torque", "N m", f"{_MOTOR}.advance"),
        Symbol("J", SymbolKind.PARAMETER, "inertia", "kg m^2", f"{_MOTOR}.advance"),
        Symbol("b", SymbolKind.PARAMETER, "damping", "N m s", f"{_MOTOR}.advance"),
        Symbol(r"\mathbf{T}_{r}", SymbolKind.INPUT, "link_wrench", WRENCH_UNITS, f"{_MOTOR}.advance"),
        Symbol(r"\mathbf{F}_{s}", SymbolKind.INPUT, "link_wrench", WRENCH_UNITS, f"{_MOTOR}.passed"),
        Symbol(r"\mathbf{R}", SymbolKind.STATE, "reaction", WRENCH_UNITS, f"{_MOTOR}.passed"),
        Symbol(r"\mathbf{P}", SymbolKind.STATE, "frame_poses", POSE_UNITS, f"{_MOTOR}.advance"),
        Symbol(r"\mathbf{M}", SymbolKind.INPUT, "mount_poses", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{O}", SymbolKind.PARAMETER, "placement", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "axis", "1", f"{_MOTOR}.poses"),
    )
    FRAMES = ("base", "rotor")

    def __init__(self, name: str, timestep: float, *, axis: Sequence[float], inertia: float, damping: float,
                 placement: Sequence[float], mount_index: int = -1, mount_count: int = 0,
                 initial_mount_pose: Sequence[float] = _IDENTITY, initial_angle: float,
                 initial_angular_velocity: float = 0.0) -> None:
        super().__init__(name, timestep, axis=flattened(axis), inertia=float(inertia), damping=float(damping),
                         placement=flattened(placement), mount_index=int(mount_index), mount_count=int(mount_count),
                         initial_mount_pose=flattened(initial_mount_pose), initial_angle=float(initial_angle),
                         initial_angular_velocity=float(initial_angular_velocity))

    def discretization_order(self) -> DiscretizationOrder:
        return DiscretizationOrder(temporal=1.0)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "axis": ParamSpec(units="1"), "placement": ParamSpec(units=POSE_UNITS),
                "inertia": ParamSpec(units="kg m^2", bounds=(0.0, None)),
                "damping": ParamSpec(units="N m s", bounds=(0.0, None))}

    def _mounted_rows(self) -> int:
        return max(cast(int, self.constants()["mount_count"]), 1)

    def poses(self, p: Mapping[str, Any], base: Array, angle: Array) -> Array:
        """The stator's and the rotor's world poses: the base, and the base turned by ``angle`` about the axis."""
        axis = jnp.asarray(p["axis"], dtype=angle.dtype)
        turned = jnp.concatenate([jnp.zeros(3, dtype=angle.dtype), quaternion.exponential(angle * axis)])
        return jnp.stack([base, compose(base, turned)])

    def passed(self, given: DynamicMotorInputs, base: Array, axis: Array, held: Array, wrenches: Array) -> Array:
        """What the stator passes to the frame it is mounted on: the loads on the stator, those on the rotor but its
        torque about ``axis`` (the world's), and ``-held`` about the axis, the drive's and the damping's reaction;
        moved from the base's origin to the mount frame's, in that frame's row."""
        force = wrenches[0, :3] + wrenches[1, :3]
        across = wrenches[1, 3:] - jnp.dot(axis, wrenches[1, 3:]) * axis
        torque = wrenches[0, 3:] + across - held * axis
        rows = self._mounted_rows()
        index = cast(int, self.constants()["mount_index"])
        if index < 0:
            return jnp.zeros((rows, 6), dtype=base.dtype)
        frame = cast(Array, given.get("mount_poses"))[index]
        moved = torque + jnp.cross(base[:3] - frame[:3], force)
        return jnp.zeros((rows, 6), dtype=base.dtype).at[index].set(jnp.concatenate([force, moved]))

    @override
    def initial_state(self) -> NodeFields:
        held = self.constants()
        angle = as_run_array(held["initial_angle"])
        p = {"axis": as_run_array(held["axis"])}
        return node_fields(DynamicMotorState(frame_poses=self.poses(p, self.initial_base(), angle), angle=angle,
                                             angular_velocity=as_run_array(held["initial_angular_velocity"]),
                                             reaction=as_run_array(np.zeros((self._mounted_rows(), 6)))))

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        return {"torque": node_input("N m", ()), "link_wrench": node_input(WRENCH_UNITS, (2, 6), additive=True),
                **self.mounted_inputs()}

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        now, given = typed_view(DynamicMotorState, state), typed_view(DynamicMotorInputs, boundary_inputs)
        angle, speed = now["angle"], now["angular_velocity"]
        base = self.base(p, given)
        axis = quaternion.rotate(base[3:], jnp.asarray(p["axis"], dtype=angle.dtype))
        torque = given["torque"] if "torque" in given else jnp.zeros((), dtype=angle.dtype)
        wrenches = given["link_wrench"] if "link_wrench" in given else jnp.zeros((2, 6), dtype=angle.dtype)
        inertia, damping = (jnp.asarray(p[name], dtype=angle.dtype) for name in ("inertia", "damping"))
        load = jnp.dot(axis, wrenches[1, 3:])
        speed = (inertia * speed + dt * (torque + load)) / (inertia + dt * damping)
        angle = angle + dt * speed
        return node_fields(DynamicMotorState(
            frame_poses=self.poses(p, base, angle), angle=angle, angular_velocity=speed,
            reaction=self.passed(given, base, axis, torque - damping * speed, wrenches)))


__all__ = ["ARM_EQUATIONS", "MOTOR_EQUATIONS", "WRENCH_UNITS", "DynamicArmInputs", "DynamicArmNode",
           "DynamicArmState", "DynamicMotorInputs", "DynamicMotorNode", "DynamicMotorState"]
