"""The nodes of an apparatus chain's dynamic rungs: an arm whose joints are driven by torques, and a motor whose rotor
is, a spinner (``_spinner``). Each takes loads on its frames (``link_wrench``, one row per frame: ``[force; torque
about the frame's origin]`` in the world), publishes the world pose and angular velocity of every frame it names as the
kinematic rungs do, and integrates its joints by semi-implicit Euler, first order in time. A motor mounted on another
segment publishes the wrench its stator exerts on that segment's frame (``reaction``), which the build delivers into
that segment's loads; and its rotor's momentum about its axis (``momentum``) and what is left of its inertia across the
axis (``inertia``), which a dynamic arm takes on the frame (``link_momentum``, ``link_inertia``): the arm applies the
rotor's gyroscopic torque at its own velocity, and carries its inertia in its mass matrix."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple, cast, override

import numpy as np
from jax import Array
from jaxtyping import Float
from maddening.core.compliance.metadata import DiscretizationOrder, NodeMeta, Reference
from maddening.core.node import BoundaryInputSpec
from maddening.core.params import ParamSpec
from typing_extensions import Annotated, NotRequired, ReadOnly, TypedDict

from mime.core import NodeFields, Symbol, SymbolKind, Units, as_run_array, jnp, node_fields, node_input, quaternion, \
    switch, typed_view
from mime.core import ledger as books
from mime.core.multibody import ROOT, Articulation, cholesky_solve, compose, forward_kinematics, \
    frame_angular_jacobians, frame_forces, frame_motions, inverse_dynamics, mass_matrix
from mime.core.multibody._kinematics import body_poses, constants
from mime.nodes.actuation._chain import POSE_UNITS, RATE_UNITS, MountInputs, SegmentNode, flattened
from mime.nodes.actuation._spinner import cayley, deviator_torque, gyroscopic_matrix, momentum, relative_rate, \
    rotor_rate, spun, turned

#: The units of a frame's load, as the vocabulary spells ``apparatus.link_wrench``; of the spin momentum a frame
#: carries, as it spells ``apparatus.link_momentum``; and of the inertia, ``apparatus.link_inertia``.
WRENCH_UNITS, MOMENTUM_UNITS, INERTIA_UNITS = "N; N m", "N m s", "kg m^2"
_IDENTITY = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
#: Where the symbol rows' functions are.
_MOUNTED, _ARM, _MOTOR = (f"mime.nodes.actuation.{module}.{name}" for module, name in (
    ("_chain", "MountedNode"), ("_dynamic", "DynamicArmNode"), ("_dynamic", "DynamicMotorNode")))


class DynamicArmState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    frame_angular_velocities: ReadOnly[Float[Array, "frames 3"]]
    joint_positions: ReadOnly[Float[Array, " dofs"]]
    joint_velocities: ReadOnly[Float[Array, " dofs"]]


class MeteredDynamicArmState(books.LedgerRecords, closed=True):
    """A metered arm's state: its own fields, its records, and its joints' positions at the run's start, which its
    weight's energy is evaluated at."""

    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    frame_angular_velocities: ReadOnly[Float[Array, "frames 3"]]
    joint_positions: ReadOnly[Float[Array, " dofs"]]
    joint_velocities: ReadOnly[Float[Array, " dofs"]]
    ledger_configuration_start: ReadOnly[Annotated[Float[Array, " dofs"], Units("1")]]


class ArmParts(NamedTuple):
    """What the energy ledger's books read of a dynamic arm's step, besides the state it takes and returns: the
    articulation with what its frames carry at the step's start (``start``) and at its end (``ending``), its base's
    pose, the world's gravity in the base's frame, the joint rates after the Cayley step (``turned``), the mass matrix
    the step starts from (``start_mass``) and the one it solves with (``end_mass``), the system ``A = M_e + dt D``, the
    unconstrained solution ``free``, the share of the impulses through ``A`` that friction took (``friction``, zero on
    every joint without it), and the bias the step solved with (``bias``: the velocity products, the weight and the
    loads on its frames together, as one inverse-dynamics pass gave them)."""

    start: Articulation
    ending: Articulation
    base: Array
    gravity: Array
    turned: Array
    start_mass: Array
    end_mass: Array
    system: Array
    free: Array
    friction: Array
    bias: Array


class DynamicArmInputs(MountInputs, closed=True):
    joint_torques: NotRequired[ReadOnly[Float[Array, " revolute"]]]
    joint_forces: NotRequired[ReadOnly[Float[Array, " prismatic"]]]
    link_wrench: NotRequired[ReadOnly[Float[Array, "frames 6"]]]
    link_momentum: NotRequired[ReadOnly[Float[Array, "frames 3"]]]
    link_inertia: NotRequired[ReadOnly[Float[Array, "frames 2 3 3"]]]


class DynamicMotorState(TypedDict, closed=True):
    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    frame_angular_velocities: ReadOnly[Float[Array, "frames 3"]]
    angle: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    spin: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    reaction: ReadOnly[Float[Array, "mounted 6"]]
    momentum: ReadOnly[Float[Array, "mounted 3"]]
    inertia: ReadOnly[Float[Array, "mounted 2 3 3"]]


class MeteredDynamicMotorState(books.LedgerRecords, closed=True):
    """A metered motor's state: its own fields, its records, and its frames' poses at the run's start, which the energy
    of a field it carries is evaluated at."""

    frame_poses: ReadOnly[Float[Array, "frames 7"]]
    frame_angular_velocities: ReadOnly[Float[Array, "frames 3"]]
    angle: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    spin: ReadOnly[Float[Array, ""]]  # noqa: F722 - a jaxtyping shape
    reaction: ReadOnly[Float[Array, "mounted 6"]]
    momentum: ReadOnly[Float[Array, "mounted 3"]]
    inertia: ReadOnly[Float[Array, "mounted 2 3 3"]]
    ledger_configuration_start: ReadOnly[Annotated[Float[Array, "frames 7"], Units("m; 1")]]


class MotorParts(NamedTuple):
    """What the energy ledger's books read of a dynamic motor's step: its axis in the world, its mount's angular
    velocity, the rotor's speed relative to its stator at the step's end (``relative``) and the torque of its
    inequality across the axis (``unequal``)."""

    axis: Array
    mount: Array
    relative: Array
    unequal: Array


class DynamicMotorInputs(MountInputs, closed=True):
    torque: NotRequired[ReadOnly[Float[Array, ""]]]  # noqa: F722 - a jaxtyping shape
    link_wrench: NotRequired[ReadOnly[Float[Array, "frames 6"]]]


#: The projected Gauss-Seidel sweeps each step makes over its joints' stops and Coulomb friction. One joint is exact in
#: one sweep, and so is every joint with friction where all of them stick (the stick start). Measured through the node
#: over 100 drawn states per tree and kind, as the change a converged solve (2000 sweeps) would still make to the
#: velocity in ``A``'s norm, relative to the unconstrained velocity's: on the AR4, at worst 3e-6 with stops, friction or
#: both; on a strongly coupled planar arm (normalised coupling up to 0.9), exact to rounding with friction alone, and at
#: worst 1.6e-2 with stops, where both of its joints reach a stop in one step (the median exact). There the change
#: falls by about 0.8 a sweep, the square of the coupling (1.6e-4 at 40 sweeps). Forty were tried and not kept: they
#: cost the AR4's node 8 us a step more, and the reverse-mode derivative of its step takes four times as long to
#: compile (a gradient check of MADDENING's battery 198 s against 50 at twenty and 16 with none; one compile alone
#: 2.4 min). The kinetic energy never rises at any sweep, and a bounded joint's position is held within its limits
#: whatever the sweeps leave.
SWEEPS = 20

ARM_EQUATIONS = (
    r"\mathbf{M}_{s} = \mathbf{M}(\theta_{n}; \mathbf{C}_{0}), \quad \mathbf{M}_{e} = \mathbf{M}(\theta_{n}; "
    r"\mathbf{C}_{1}), \quad \mathbf{G} = \sum_{k} \mathbf{S}_{k}^{\top} [\mathbf{H}_{k}]_{\times} "
    r"\mathbf{S}_{k}, \quad "
    r"\dot{\theta}^{\prime} = (\mathbf{M}_{s} - \tfrac{1}{2} \mathrm{d}t \, \mathbf{G})^{-1} (\mathbf{M}_{s} + "
    r"\tfrac{1}{2} \mathrm{d}t \, \mathbf{G}) \, \dot{\theta}_{n}; \quad "
    r"\mathbf{A} = \mathbf{M}_{e} + \mathrm{d}t \, \mathbf{D}, \quad "
    r"\dot{\theta}^{*} = \mathbf{A}^{-1} \left( \mathbf{M}_{s} \dot{\theta}^{\prime} + \mathrm{d}t \, "
    r"(\mathbf{u} - \mathbf{h}(\theta_{n}, \dot{\theta}^{\prime}; \mathbf{g}, \mathbf{W}, \mathbf{C}_{0})) "
    r"\right); \quad "
    r"\dot{\theta}_{n+1} = \arg\min_{v} \tfrac{1}{2} (v - \dot{\theta}^{*})^{\top} \mathbf{A} \, "
    r"(v - \dot{\theta}^{*}) + \mathrm{d}t \sum_{j} \mu_{j} |v_{j}| \ \text{with} \ l \le \theta_{n} + "
    r"\mathrm{d}t \, v \le h, \ \text{by projected Gauss-Seidel sweeps}; \quad "
    r"\theta_{n+1} = \min(\max(\theta_{n} + \mathrm{d}t \, \dot{\theta}_{n+1}, l), h); \quad "
    r"\mathbf{P}_{k} = \mathbf{O} \circ \mathrm{FK}_{k}(\theta_{n+1}; \mathbf{a}, \mathbf{J}, \mathbf{K}), \quad "
    r"\boldsymbol{\Omega}_{k} = R(\mathbf{O}) \, \mathrm{W}_{k}(\theta_{n+1}, \dot{\theta}_{n+1})")


class DynamicArmNode(SegmentNode):
    """A multibody tree whose joints are driven by torques: the mass matrix ``M`` by the composite-rigid-body
    algorithm, and the bias ``h`` (velocity products, gravity and the loads on its frames) by the recursive
    Newton-Euler algorithm, both from ``mime.core.multibody``; each joint's viscous damping ``D`` implicit; and the
    joint limits as rigid, inelastic stops and each joint's Coulomb friction ``mu`` as impulses through the step's
    system ``A = M + dt D``, so that what holds one joint moves every joint coupled to it.

    **The impulses.** The step's velocity is the one nearest the unconstrained velocity in ``A``'s norm, less what
    friction dissipates (``dt sum mu_j |v_j|``), among the velocities that keep each bounded joint within its limits
    over the step: a convex problem, whose solution never holds more kinetic energy than the unconstrained step. It is
    solved by projected Gauss-Seidel over the joints with a stop or friction, ``SWEEPS`` sweeps, each joint's update
    exact for its own impulse with the others' held: its inertia with the others free, ``1 / (A^-1)_jj``; its
    friction torque held within ``mu_j`` (the impulse that would stop it over the step, clipped to its friction); then
    its velocity held within its stop's gap, ``(l - theta) / dt`` to ``(h - theta) / dt``, so that a joint driven
    past a limit arrives at it at the end of the step and no further (its stop's impulse one-sided, released where it
    would pull the joint off). The two are one update: the exact minimiser of the joint's own problem, friction and
    gap together. Every update lowers the problem's dual, so the kinetic energy in ``A``'s norm never rises at any
    sweep. Before the sweeps, the joints with friction start held still (the impulses that hold every one of them still
    with the others free), scaled back towards zero as far as their friction needs: an arm whose friction can hold it
    starts, and stays, exactly still. A bounded joint's position is then held within its limits, which keeps what the
    sweeps leave unconverged from carrying it past one.

    Its base is fixed in the world at ``placement``; ``gravity`` is the world's, and loads on its frames are given in
    the world, each about its frame's origin. A revolute joint is driven by its torque in ``joint_torques`` (N m) and a
    prismatic one by its force in ``joint_forces`` (N), each in the order of its kind's joints. The tree is taken as
    its articulation's arrays, as the kinematic arm's are, with its inertials (``masses``, ``centres_of_mass``,
    ``inertias``, flattened: the description's, with what is mounted on its links grafted on at build), ``armatures``
    and ``damping`` per moving joint, ``friction`` per joint that has it (``rubbing``), and the limits of each joint
    that has them (``bounded``). Every frame's angular velocity follows from the step's joint rates.

    **What spinners mounted on its frames hand it**, a step late (a dynamic motor's, read from its state over the step
    before). ``link_momentum``: the momentum about its axis of each rotor a frame carries, ``h = J Omega e`` in the
    world, whose gyroscopic torque ``h x omega`` on the frame enters as the skew joint-space matrix ``G = sum_k S_k^T
    [h_k]x S_k`` (``S_k`` the frame's angular velocity Jacobian), applied at the step's own velocity by Crank-Nicolson,
    the Cayley transform of ``M^-1 G``: it keeps ``q_dot M q_dot`` to rounding at every step size, as the term does no
    work. Before it, the term was applied from the start-of-step velocity, forward Euler on a skew coupling, which grew
    the kinetic energy by ``1 + (nu dt)^2`` a step at every step size. ``link_inertia``: what is left of each rotor's
    inertia across its axis once its mean is grafted on (its half difference, which turns with the rotor), at the
    step's start and its end, in the world: added to its frame's body in the mass matrix and the bias at the start, and
    in the system the step solves at its end, so that the rotor's turning moves the joints' momentum ``M q_dot`` as
    its inertia changes. The Cayley step comes first, and the rest of the step starts from the velocity it gives, its
    velocity products among it: a splitting of the step into the gyroscopic flow, which keeps the kinetic energy to
    rounding, and the arm's own step, so that the coupling adds nothing to the arm's own scheme's energy error at any
    ``nu dt`` (measured over 1e4 steps on A0 at ``nu dt`` 0.01 and 0.1: 3.3e-5 of the energy either way, the arm's own
    first-order drift). It keeps the system the step solves symmetric and positive definite, its impulses' problem the
    same convex one."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-109",
        description="A dynamic chain segment: joint torques in, the mass matrix by CRBA and the bias by RNEA with "
                    "gravity and the loads on its frames, implicit joint damping, rigid joint stops and Coulomb "
                    "friction as impulses through the step's system by projected Gauss-Seidel, and every frame's "
                    "world pose by forward kinematics.",
        governing_equations=ARM_EQUATIONS,
        discretization="Semi-implicit Euler, first order in time, after the gyroscopic torque of the spin momenta "
                       "on its frames by a Cayley step (Crank-Nicolson on a skew matrix, which does no work): the "
                       "velocity from the forces at the start of the step at the velocity that step gives, the joint "
                       "damping taken at its end, the inertia its frames carry at the start and at the end, the "
                       "stops' and friction's impulses on the step's system; then the position from the new velocity.",
        assumptions=("Rigid joints and links, with no structural mode (MIME-ANO-101)",
                     "Its base is fixed in the world",
                     "A joint limit is a rigid, inelastic stop: a joint driven past it arrives at it at the end of the "
                     "step, and the stop takes the rest of its velocity on the next",
                     "The stops' and friction's impulses are solved by a fixed count of projected Gauss-Seidel sweeps: "
                     "exact for one joint and where every joint with friction sticks, converged on coupled joints "
                     "as SWEEPS states, and never adding kinetic energy"),
        references=(Reference("Featherstone2008", "the composite-rigid-body and recursive Newton-Euler algorithms"),
                    Reference("Sciavicco2000", "the dynamics of a serial chain")),
    )
    symbols = (
        Symbol(r"\mathbf{u}", SymbolKind.INPUT, "joint_torques", "N m", f"{_ARM}.driving"),
        Symbol(r"\mathbf{W}", SymbolKind.INPUT, "link_wrench", WRENCH_UNITS, f"{_ARM}.loads"),
        Symbol(r"\mathbf{H}", SymbolKind.INPUT, "link_momentum", MOMENTUM_UNITS, f"{_ARM}.spinning"),
        Symbol(r"\mathbf{C}", SymbolKind.INPUT, "link_inertia", INERTIA_UNITS, f"{_ARM}.carrying"),
        Symbol(r"\mathbf{g}", SymbolKind.PARAMETER, "gravity", "m/s^2", f"{_ARM}.weight"),
        Symbol(r"\mathbf{D}", SymbolKind.PARAMETER, "damping", "N m s", f"{_ARM}.advance"),
        Symbol(r"\mu", SymbolKind.PARAMETER, "friction", "N m", f"{_ARM}.constrained"),
        Symbol("l", SymbolKind.PARAMETER, "lower", "1", f"{_ARM}.constrained"),
        Symbol("h", SymbolKind.PARAMETER, "upper", "1", f"{_ARM}.constrained"),
        Symbol(r"\theta", SymbolKind.STATE, "joint_positions", "1", f"{_ARM}.advance"),
        Symbol(r"\dot{\theta}", SymbolKind.STATE, "joint_velocities", "rad/s", f"{_ARM}.advance"),
        Symbol(r"\mathbf{P}", SymbolKind.STATE, "frame_poses", POSE_UNITS, f"{_ARM}.advance"),
        Symbol(r"\boldsymbol{\Omega}", SymbolKind.STATE, "frame_angular_velocities", RATE_UNITS, f"{_ARM}.advance"),
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
                 damping: Sequence[float], rubbing: Sequence[int], friction: Sequence[float],
                 bounded: Sequence[int], lower: Sequence[float], upper: Sequence[float], placement: Sequence[float],
                 gravity: Sequence[float], mount_index: int = -1, mount_count: int = 0,
                 initial_mount_pose: Sequence[float] = _IDENTITY, initial_joint_positions: Sequence[float],
                 initial_joint_velocities: Sequence[float] | None = None, ledger: Sequence[str] = ()) -> None:
        if mount_index != -1:
            raise ValueError("a dynamic arm's base is fixed in the world: it mounts on nothing (mount_index -1)")
        if len(rubbing) != len(friction) or not all(math.isfinite(f) and f > 0.0 for f in friction):
            raise ValueError(f"friction {tuple(friction)!r} is not one level above zero per joint of rubbing "
                             f"{tuple(rubbing)!r}")
        dofs = len(dof_names)
        super().__init__(
            name, timestep, dof_names=tuple(dof_names), parents=tuple(int(i) for i in parents),
            revolute=tuple(bool(r) for r in revolute), frame_names=tuple(frame_names),
            frame_bodies=tuple(int(i) for i in frame_bodies), axes=flattened(axes),
            origin_translations=flattened(origin_translations), origin_rotations=flattened(origin_rotations),
            frame_translations=flattened(frame_translations), frame_rotations=flattened(frame_rotations),
            masses=flattened(masses), centres_of_mass=flattened(centres_of_mass), inertias=flattened(inertias),
            armatures=flattened(armatures), damping=flattened(damping), rubbing=tuple(int(i) for i in rubbing),
            friction=flattened(friction), bounded=tuple(int(i) for i in bounded), lower=flattened(lower), upper=flattened(upper),
            placement=flattened(placement), gravity=flattened(gravity), mount_index=-1,
            mount_count=int(mount_count), initial_mount_pose=flattened(initial_mount_pose),
            initial_joint_positions=flattened(initial_joint_positions),
            initial_joint_velocities=flattened(initial_joint_velocities if initial_joint_velocities is not None
                                               else (0.0,) * dofs), ledger=ledger)
        held = books.deliveries(self.constants())
        unknown = sorted({d.field for d in held} - {*_ARM_PAIRED, *(role for role, _, _ in self.hosted_laws())})
        if unknown:
            raise ValueError(f"the ledger delivers into {unknown}, which a dynamic arm pairs nothing for: it pairs "
                             f"{sorted(_ARM_PAIRED)} and hosts {[role for role, _, _ in self.hosted_laws()]}")

    @staticmethod
    def described(articulation: Articulation, damping: Sequence[float], friction: Sequence[float]) -> dict[str, Any]:
        """The constructor arguments that hold ``articulation`` with its inertials, and each moving joint's
        ``damping`` and ``friction``, in its order: the joints with friction above zero are ``rubbing``, and those with
        a finite limit ``bounded``."""
        held = articulation
        bounded = tuple(i for i in range(held.dofs) if np.isfinite(held.lower[i]) or np.isfinite(held.upper[i]))
        rubbing = tuple(i for i, level in enumerate(friction) if level > 0.0)
        return {"dof_names": held.dof_names, "parents": held.parents, "revolute": held.revolute,
                "frame_names": held.frame_names, "frame_bodies": held.frame_bodies, "axes": flattened(held.axes),
                "origin_translations": flattened(held.origin_translations),
                "origin_rotations": flattened(held.origin_rotations),
                "frame_translations": flattened(held.frame_translations),
                "frame_rotations": flattened(held.frame_rotations), "masses": flattened(held.masses),
                "centres_of_mass": flattened(held.centres_of_mass), "inertias": flattened(held.inertias),
                "armatures": flattened(held.armatures), "damping": tuple(float(d) for d in damping),
                "rubbing": rubbing, "friction": tuple(float(friction[i]) for i in rubbing), "bounded": bounded,
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
                "friction": ParamSpec(trainable=False, units="N m",
                                      description="the Coulomb friction of each joint that has it (rubbing)")}

    def _counts(self) -> tuple[int, int]:
        held = self.constants()
        return len(cast(tuple[str, ...], held["dof_names"])), len(cast(tuple[str, ...], held["frame_names"]))

    def kinds(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The indices of its revolute joints and of its prismatic ones."""
        revolute = cast(tuple[bool, ...], self.constants()["revolute"])
        return (tuple(i for i, r in enumerate(revolute) if r), tuple(i for i, r in enumerate(revolute) if not r))

    def driving(self, given: DynamicArmInputs, dtype: Any) -> Array:
        """The joint forces commanded: each revolute joint's torque and each prismatic joint's force, zero where
        none is given."""
        dofs, _ = self._counts()
        revolute, prismatic = self.kinds()
        if not prismatic:
            return given["joint_torques"] if "joint_torques" in given else jnp.zeros((dofs,), dtype=dtype)
        forces = jnp.zeros((dofs,), dtype=dtype)
        for indices, name in ((revolute, "joint_torques"), (prismatic, "joint_forces")):
            if indices and name in given:
                forces = forces.at[np.asarray(indices)].set(jnp.asarray(cast(Any, given)[name], dtype))
        return forces

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

    def weight(self, p: Mapping[str, Any], base: Array) -> Array:
        """The world's gravity in its base's frame, where the tree's algorithms take it."""
        return quaternion.rotate_back(base[3:], jnp.asarray(p["gravity"], dtype=base.dtype))

    def loads(self, articulation: Articulation, q: Array, base: Array, wrenches: Array) -> Array:
        """The loads on its frames, given in the world about each frame's origin, as spatial forces on its bodies in
        its base's frame about the base's origin."""
        turned = jnp.concatenate([quaternion.rotate_back(base[3:], wrenches[:, :3].T).T,
                                  quaternion.rotate_back(base[3:], wrenches[:, 3:].T).T], axis=1)
        return frame_forces(articulation, q, turned)

    def constrained(self, p: Mapping[str, Any], q: Array, free: Array, inverse: Array,
                    dt: Any) -> tuple[Array, Array]:
        """The joints' positions and velocities at the end of the step from the unconstrained velocity ``free``, with
        their stops and Coulomb friction as impulses through ``inverse``, ``(M + dt D)^-1``: the class's account."""
        positions, velocity, _ = self.constrained_parts(p, q, free, inverse, dt, split=False)
        return positions, velocity

    def constrained_parts(self, p: Mapping[str, Any], q: Array, free: Array, inverse: Array, dt: Any, *,
                          split: bool) -> tuple[Array, Array, Array]:
        """:meth:`constrained`, and with ``split`` friction's share of the impulses (each joint with friction, its
        last update's ``-dt`` times its friction torque; zero on every other joint), which the energy ledger books
        apart from the stops'. The positions and velocities are the same operations either way."""
        held = self.constants()
        rubbing, bounded = (cast(tuple[int, ...], held[name]) for name in ("rubbing", "bounded"))
        dtype = free.dtype
        friction = jnp.asarray(p["friction"], dtype=dtype)
        lower, upper = (jnp.asarray(p[name], dtype=dtype) for name in ("lower", "upper"))
        own = jnp.diagonal(inverse)
        impulse = jnp.zeros_like(free)
        rubbed = jnp.zeros_like(free)
        if rubbing:
            # The start: the impulses that hold every joint with friction still with the others free, scaled back
            # towards zero until each is within its joint's friction over the step. They lie on the line from no
            # impulse to the problem's least without friction's bound on those joints, so they add no kinetic
            # energy; where every such joint can stick, they are its solution.
            index = np.asarray(rubbing)
            still = -cholesky_solve(jnp.linalg.cholesky(inverse[index[:, None], index[None, :]]), free[index])
            largest = jnp.ones((), dtype=dtype)
            for share in (still / (dt * friction)) ** 2:
                largest = switch.maximum(largest, share, owner="arm.stick",
                                         resolution=lambda: switch.rounding(largest), indicator=lambda: largest - share)
            impulse = impulse.at[index].set(still / largest)
        velocity = free + inverse @ impulse
        ends = list(q[np.asarray(bounded, dtype=int)])
        for _ in range(SWEEPS):
            for joint in sorted({*rubbing, *bounded}):
                rest = velocity[joint] - own[joint] * impulse[joint]
                moved = rest
                if joint in rubbing:
                    needed = rest / (own[joint] * dt)
                    level = friction[rubbing.index(joint)]
                    torque = switch.clip(needed, -level, level, owner="arm.friction",
                                         resolution=lambda: switch.rounding(needed), indicator=needed)
                    moved = rest - own[joint] * dt * torque
                    if split:
                        rubbed = rubbed.at[joint].set(-dt * torque)
                if joint in bounded:
                    k = bounded.index(joint)
                    end: Array = q[joint] + dt * moved
                    moved = switch.maximum(moved, (lower[k] - q[joint]) / dt, owner="arm.stop.low",
                                           resolution=lambda: switch.rounding(end), indicator=lambda: end - lower[k])
                    moved = switch.minimum(moved, (upper[k] - q[joint]) / dt, owner="arm.stop.high",
                                           resolution=lambda: switch.rounding(end), indicator=lambda: upper[k] - end)
                    ends[k] = end
                change = (moved - velocity[joint]) / own[joint]
                velocity = velocity + inverse[:, joint] * change
                impulse = impulse.at[joint].add(change)
        positions = q + dt * velocity
        if not bounded:
            return positions, velocity, rubbed
        index = np.asarray(bounded)
        reached = jnp.stack(ends)
        kept = switch.clip(positions[index], lower, upper, owner="arm.stop",
                           resolution=lambda: switch.rounding(reached), indicator=reached)
        return positions.at[index].set(kept), velocity, rubbed

    @override
    def initial_state(self) -> NodeFields:
        held = self.constants()
        q = as_run_array(held["initial_joint_positions"])
        qd = as_run_array(held["initial_joint_velocities"])
        p = {key: as_run_array(value) for key, value in held.items() if key in _ARRAYS}
        articulation = self.articulation(p, q.dtype)
        frames = forward_kinematics(articulation, q)
        base = self.initial_base()
        _, local = frame_motions(articulation, q, qd)
        return books.with_records(self, node_fields(DynamicArmState(
            frame_poses=self.in_world(base, frames),
            frame_angular_velocities=self.rates_in_world(base, jnp.zeros_like(base[:3]), local),
            joint_positions=q, joint_velocities=qd)))

    def own_inputs(self) -> dict[str, BoundaryInputSpec]:
        """The arm's inputs without the ledger: its commands, the loads, momenta and inertias its frames take, and its
        mount's."""
        _, frames = self._counts()
        revolute, prismatic = self.kinds()
        commands = {"joint_torques": node_input("N m", (len(revolute),))} if revolute else {}
        if prismatic:
            commands["joint_forces"] = node_input("N", (len(prismatic),))
        return {**commands, "link_wrench": node_input(WRENCH_UNITS, (frames, 6), additive=True),
                "link_momentum": node_input(MOMENTUM_UNITS, (frames, 3), additive=True),
                "link_inertia": node_input(INERTIA_UNITS, (frames, 2, 3, 3), additive=True), **self.mounted_inputs()}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        """:meth:`own_inputs`; with the ledger attached, each delivery's own beside the input it would sum into."""
        return books.with_law_inputs(self.own_inputs(), books.deliveries(self.constants()))

    def carrying(self, articulation: Articulation, poses: Array, given: DynamicArmInputs) -> tuple[Articulation,
                                                                                                   Articulation]:
        """The articulation with the inertia its frames carry beyond its own (``link_inertia``) added to their bodies,
        at the step's start and at its end: each given in the world, turned into its body's axes by the frame poses
        ``poses`` at the step's start. A frame fixed to the base carries nothing that moves."""
        if "link_inertia" not in given:
            return articulation, articulation
        carried = given["link_inertia"]
        rotations = cast(Array, articulation.frame_rotations)
        start, end = jnp.zeros_like(articulation.inertias), jnp.zeros_like(articulation.inertias)
        for k, body in enumerate(articulation.frame_bodies):
            if body == ROOT:
                continue
            onto = quaternion.rotation_matrix(poses[k, 3:]) @ quaternion.rotation_matrix(rotations[k]).T
            start = start.at[body].add(onto.T @ carried[k, 0] @ onto)
            end = end.at[body].add(onto.T @ carried[k, 1] @ onto)
        return (dataclasses.replace(articulation, inertias=articulation.inertias + start),
                dataclasses.replace(articulation, inertias=articulation.inertias + end))

    def spinning(self, articulation: Articulation, q: Array, qd: Array, base: Array, inertia: Array,
                 given: DynamicArmInputs, dt: Any) -> Array:
        """The joint rates after the gyroscopic torques of the spin momenta its frames carry (``link_momentum``,
        in the world) over the step: the Cayley step on ``G``, in the base's axes, which keeps ``q_dot M q_dot``."""
        if "link_momentum" not in given:
            return qd
        momenta = quaternion.rotate_back(base[3:], given["link_momentum"].T).T
        return cayley(inertia, gyroscopic_matrix(frame_angular_jacobians(articulation, q), momenta), qd, dt)

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        """The step (:meth:`stepped_with_parts`); with the ledger attached, its books recorded beside it
        (``mime.core.ledger.metered_step``), decided while the step is traced."""
        if not self.metered:
            return self.stepped_with_parts(p, state, boundary_inputs, dt)[0]
        return books.metered_step(self, self.own_inputs(), p, state, boundary_inputs, dt, self.stepped_with_parts)

    def stepped_with_parts(self, p: Mapping[str, Any], state: Mapping[str, Array],
                           boundary_inputs: Mapping[str, Array], dt: float) -> tuple[NodeFields, ArmParts]:
        """The step, and what the ledger's books read of it (:class:`ArmParts`): the step's arithmetic is the same
        whether the ledger is attached or not, so a program without it keeps every bit."""
        now, given = typed_view(DynamicArmState, state), typed_view(DynamicArmInputs, boundary_inputs)
        q, qd = now["joint_positions"], now["joint_velocities"]
        dofs, frames = self._counts()
        own = self.articulation(p, q.dtype)
        articulation, ending = self.carrying(own, now["frame_poses"], given)
        base = self.base(p, given)
        torques = self.driving(given, q.dtype)
        wrenches = given["link_wrench"] if "link_wrench" in given else jnp.zeros((frames, 6), dtype=q.dtype)
        inertia = mass_matrix(articulation, q)
        turned_qd = self.spinning(articulation, q, qd, base, inertia, given, dt)
        bias = inverse_dynamics(articulation, q, turned_qd, None, self.weight(p, base),
                                self.loads(articulation, q, base, wrenches))
        system = inertia if ending is articulation else mass_matrix(ending, q)
        factor = jnp.linalg.cholesky(system + dt * jnp.diag(jnp.asarray(p["damping"], dtype=q.dtype)))
        free = cholesky_solve(factor, inertia @ turned_qd + dt * (torques - bias))
        inverse = cholesky_solve(factor, jnp.eye(dofs, dtype=q.dtype))
        if self.metered:
            q_next, velocity, rubbed = self.constrained_parts(p, q, free, inverse, dt, split=True)
        else:
            # Through ``constrained``, which a test may replace on one node to plant a rule.
            q_next, velocity = self.constrained(p, q, free, inverse, dt)
            rubbed = jnp.zeros_like(free)
        stepped = system + dt * jnp.diag(jnp.asarray(p["damping"], dtype=q.dtype))
        frames, local = frame_motions(own, q_next, velocity)
        poses = self.in_world(base, frames)
        rates = self.rates_in_world(base, jnp.zeros_like(base[:3]), local)
        return (node_fields(DynamicArmState(frame_poses=poses, frame_angular_velocities=rates, joint_positions=q_next,
                                            joint_velocities=velocity)),
                ArmParts(articulation, ending, base, self.weight(p, base), turned_qd, inertia, system, stepped, free,
                         rubbed, bias))

    # ------------------------------------------------------------------------------------------- the energy ledger

    #: The arm's own scheme's numerical terms, in the order its records keep them: semi-implicit Euler's ``(1/2) dv^T
    #: M_s dv`` (``dv = v' - v_c``, ``v_c`` the rates after the Cayley step); the Cayley step's departure from doing no
    #: work (its spinners' gyroscopic works less the change it made in ``(1/2) v^T M_s v``, zero to rounding); the
    #: velocity products' pairing against the mass matrix's change with the joints, ``dt v' . C(q, v_c) v_c - (K(q',
    #: v') - K(q, v'))`` (zero in the continuum, first order in the step); and the loads' work on the joints' change
    #: beyond the step's solution, ``F . (dq - dt v')``, where a stop's clip moved a joint (rounding elsewhere).
    LEDGER_TERMS = ("backward_euler", "cayley", "velocity_products", "configuration")
    #: The commands an arm pairs as loads read at the step's start, with its joints' change.
    PAIRED_COMMANDS = ("joint_torques", "joint_forces")

    @override
    def hosted_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The laws the arm holds in its own step, each as ``(role, law, class)``, its law's id its node's name and its
        role: its weight (a potential, the experiment's gravity on its links), its joints' viscous damping, and, where
        it has them, its stops' and its Coulomb friction's impulses (dissipative), each booked under its own name."""
        held = self.constants()
        found = [("weight", f"{self.name}.weight", books.POTENTIAL),
                 ("damping", f"{self.name}.damping", books.DISSIPATIVE)]
        if held["bounded"]:
            found.append(("stops", f"{self.name}.stops", books.DISSIPATIVE))
        if held["rubbing"]:
            found.append(("friction", f"{self.name}.friction", books.DISSIPATIVE))
        return tuple(found)

    @override
    def configuration(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """The arm's joint positions, what its weight's energy reads."""
        return jnp.asarray(state["joint_positions"])

    def frame_poses_at(self, p: Mapping[str, Any], configuration: Array) -> Array:
        """Its frames' world poses at the joint positions ``configuration``: what a field it carries is placed by."""
        q = jnp.asarray(configuration)
        base = jnp.asarray(p["placement"], dtype=q.dtype)
        return self.in_world(base, forward_kinematics(self.articulation(p, q.dtype), q))

    @override
    def storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """``(1/2) v^T M(q) v``, its joints' kinetic energy with its own inertia (what its frames carry beyond it is
        read at each step, and the jump between steps is recorded apart)."""
        q, v = jnp.asarray(state["joint_positions"]), jnp.asarray(state["joint_velocities"])
        return 0.5 * v @ mass_matrix(self.articulation(p, q.dtype), q) @ v

    @override
    def energy(self, p: Mapping[str, Any], configuration: Array) -> Array:
        """Its weight's energy at the joints' positions ``configuration``: ``-sum_i m_i g . c_i(q)``, each link's mass
        at its centre of mass, in its base's frame, where its gravity is taken."""
        q = jnp.asarray(configuration)
        articulation = self.articulation(p, q.dtype)
        held = constants(articulation, q.dtype)
        bodies = body_poses(articulation, q, held)
        gravity = self.weight(p, jnp.asarray(p["placement"], dtype=q.dtype))
        total = jnp.zeros((), dtype=q.dtype)
        for i, body in enumerate(bodies):
            centre = body[:3] + quaternion.rotate(body[3:], held["centres_of_mass"][i])
            total = total - held["masses"][i] * jnp.dot(gravity, centre)
        return total

    def _generalised(self, start: Articulation, q: Array, base: Array, wrenches: Array) -> tuple[Array, Array]:
        """The joint forces of loads on its frames (given in the world about each frame's origin), and the magnitude
        of that arithmetic."""
        zero = jnp.zeros(3, dtype=q.dtype)
        found = -inverse_dynamics(start, q, None, None, zero, self.loads(start, q, base, wrenches))
        return found, books.magnitude(found)

    def _carried_change(self, start: Articulation, q: Array, poses: Array, carried: Array) -> Array:
        """The change in the mass matrix of an inertia its frames carry, ``carried`` (``(frames, 2, 3, 3)``, at the
        step's start and end in the world): the mass matrix of the tree holding that change alone, which is linear in
        each body's inertia."""
        rotations = cast(Array, start.frame_rotations)
        change = jnp.zeros_like(start.inertias)
        for k, body in enumerate(start.frame_bodies):
            if body == ROOT:
                continue
            onto = quaternion.rotation_matrix(poses[k, 3:]) @ quaternion.rotation_matrix(rotations[k]).T
            change = change.at[body].add(onto.T @ (carried[k, 1] - carried[k, 0]) @ onto)
        bare = dataclasses.replace(start, inertias=change, masses=jnp.zeros_like(start.masses),
                                   centres_of_mass=jnp.zeros_like(start.centres_of_mass),
                                   armatures=jnp.zeros_like(start.armatures))
        return mass_matrix(bare, q)

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: ArmParts,
                       dt: float) -> books.StepAccount:
        """The step's identity, ``K' - K = sum_law W_law + W_unattributed - N``, each delivery paired by the one rule: a
        load on its frames and a command with the joints' actual change ``dq`` (the loads by the joint forces the tree
        turns them into), a spinner's momentum with the rates its gyroscopic torque is applied at (the Cayley step's
        midpoint), an inertia its frames carry by its change at the step's solution, ``-(1/2) v'^T dM v'``; its own
        weight with ``dq``, its damping and its impulses (stops, friction) with the step's solution. ``K`` is ``(1/2)
        v^T M v`` with what its frames carry at the step's start, ``K'`` with what they carry at its end, at the
        joints' end positions. From ``A v' = M_s v_c + dt (u - h) + p`` with ``A = M_e + dt D`` and ``p = A (v' -
        v_free)`` the impulses; the step's scale is each product's arithmetic in magnitude, the solve's included."""
        q, v = state["joint_positions"], state["joint_velocities"]
        dtype = q.dtype
        vn, qn = ended["joint_velocities"], ended["joint_positions"]
        start, ending, base = parts.start, parts.ending, parts.base
        own = self.own_inputs()
        given = {**{name: jnp.asarray(entry.default, dtype=dtype) for name, entry in own.items()},
                 **books.summed(own, held, boundary_inputs)}
        dq = qn - q
        beyond = dq - dt * vn
        zero = jnp.zeros(3, dtype=dtype)
        turned, mass_start, mass_end = parts.turned, parts.start_mass, parts.end_mass
        middle = 0.5 * (v + turned)
        jacobians = frame_angular_jacobians(start, q)
        poses = state["frame_poses"]
        damping = jnp.diag(jnp.asarray(p["damping"], dtype=dtype))

        def quadratic(matrix: Array, x: Array) -> tuple[Array, Array]:
            return 0.5 * x @ matrix @ x, 0.5 * books.magnitude(x) @ books.magnitude(matrix) @ books.magnitude(x)

        loads_total = jnp.zeros_like(q)

        def paired(field: str, value: Array) -> tuple[Array, Array]:
            """A delivery into ``field`` paired by the one rule, and the magnitude of that arithmetic."""
            nonlocal loads_total
            if field == "link_wrench":
                force, size = self._generalised(start, q, base, value)
                loads_total = loads_total + force
                return jnp.dot(force, dq), size @ books.magnitude(dq)
            if field == "link_momentum":
                momenta = quaternion.rotate_back(base[3:], value.T).T
                gyroscopic = gyroscopic_matrix(jacobians, momenta)
                return dt * middle @ gyroscopic @ middle, \
                    dt * books.magnitude(middle) @ books.magnitude(gyroscopic) @ books.magnitude(middle)
            if field == "link_inertia":
                change = self._carried_change(start, q, poses, value)
                work, size = quadratic(change, vn)
                return -work, size
            command = self.driving(cast(DynamicArmInputs, {field: value}), dtype)
            return jnp.dot(command, dq), books.magnitude(command) @ books.magnitude(dq)

        rows = books.laws(held)
        work = jnp.zeros(len(rows), dtype=dtype)
        hosted = jnp.zeros((len(rows), 2), dtype=dtype)
        size = jnp.zeros((), dtype=dtype)
        momentum_work = jnp.zeros((), dtype=dtype)
        carried_work = jnp.zeros((), dtype=dtype)
        impulse = parts.system @ (vn - parts.free)
        weight_row: int | None = None
        for index, delivery in enumerate(held):
            row = rows.index(delivery.law)
            if delivery.kind == books.HOSTED:
                if delivery.field == "weight":
                    weight_row = row
                    continue
                if delivery.field == "damping":
                    found, magnitude = quadratic(damping, vn)
                    found, magnitude = -2.0 * dt * found, 2.0 * dt * magnitude
                elif delivery.field == "friction":
                    found = jnp.dot(parts.friction, vn)
                    magnitude = books.magnitude(parts.friction) @ books.magnitude(vn)
                else:
                    stops = impulse - parts.friction
                    found, magnitude = jnp.dot(stops, vn), books.magnitude(stops) @ books.magnitude(vn)
                work, size = work.at[row].add(found), size + magnitude
                continue
            found, magnitude = paired(delivery.field, boundary_inputs[books.law_input(delivery.field, index)])
            if delivery.field == "link_momentum":
                momentum_work = momentum_work + found
            if delivery.field == "link_inertia":
                carried_work = carried_work + found
            work, size = work.at[row].add(found), size + magnitude
        unattributed = jnp.zeros((), dtype=dtype)
        for field in _ARM_PAIRED:
            if field not in own:
                continue
            found, magnitude = paired(field, boundary_inputs.get(field, jnp.asarray(own[field].default, dtype=dtype)))
            if field == "link_momentum":
                momentum_work = momentum_work + found
            if field == "link_inertia":
                carried_work = carried_work + found
            unattributed, size = unattributed + found, size + magnitude
        # The carried inertia's change as the step itself used it, ``M_e - M_s``: each delivery's own change is the
        # mass matrix of its change alone (the mass matrix is linear in a body's inertia), and what that leaves of the
        # step's (the rounding of two mass matrices' difference) is booked under the first law that delivers one.
        carrying = next((rows.index(d.law) for d in held if d.kind == books.INERTIA), None)
        if carrying is not None:
            step_change, _ = quadratic(mass_end - mass_start, vn)
            work = work.at[carrying].add(-step_change - carried_work)
        commands = self.driving(cast(DynamicArmInputs, given), dtype)
        products = inverse_dynamics(start, q, turned, None, zero, None)
        # The weight's joint forces as the step itself used them: its bias, less the velocity products and every
        # load's own joint forces (the inverse dynamics is linear in each), so that what one pass of it rounds and
        # the passes taken apart do not is the weight's, a potential's, and nothing else's.
        weight = -(parts.bias - products) - loads_total
        if weight_row is not None:
            work = work.at[weight_row].add(jnp.dot(weight, dq))
            size = size + books.magnitude(weight) @ books.magnitude(dq)
        applied = commands - parts.bias + products
        load_size = books.magnitude(loads_total)
        next_end = mass_matrix(ending, qn)
        start_storage, start_size = quadratic(mass_start, v)
        turned_storage, turned_size = quadratic(mass_start, turned)
        moved = vn - turned
        backward_euler, euler_size = quadratic(mass_start, moved)
        end_here, end_here_size = quadratic(mass_end, vn)
        end_storage, end_size = quadratic(next_end, vn)
        cayley = momentum_work - (turned_storage - start_storage)
        velocity_products = dt * jnp.dot(vn, products) - (end_storage - end_here)
        configuration = jnp.dot(applied, beyond)
        speed, products_size = books.magnitude(vn), books.magnitude(products)
        commands_size, weight_size = books.magnitude(commands), books.magnitude(weight)
        size = size + start_size + turned_size + euler_size + end_here_size + end_size \
            + dt * speed @ products_size + (commands_size + load_size + weight_size) @ books.magnitude(beyond) \
            + speed @ books.magnitude(parts.system) @ (speed + books.magnitude(parts.free)) \
            + speed @ (books.magnitude(mass_start) @ books.magnitude(turned) + dt * (commands_size + load_size
                                                                                      + weight_size + products_size))
        return books.StepAccount(
            work=work, hosted=hosted, unattributed=unattributed,
            terms=jnp.stack([backward_euler, cayley, velocity_products, configuration]),
            storage_start=start_storage, storage_end=end_storage, scale=size, operators={})


#: The inputs a dynamic arm pairs, by what each holds: the loads on its frames, its spinners' momenta, the inertia its
#: frames carry and its commands.
_ARM_PAIRED = ("link_wrench", "link_momentum", "link_inertia", "joint_torques", "joint_forces")


#: The numeric constants of a dynamic arm a step reads.
_ARRAYS = ("axes", "origin_translations", "origin_rotations", "frame_translations", "frame_rotations", "masses",
           "centres_of_mass", "inertias", "armatures", "damping", "friction", "lower", "upper", "placement", "gravity")


MOTOR_EQUATIONS = (
    r"\mathbf{e} = R(\mathbf{P}_{s}) \mathbf{a}, \quad \tau_{l} = \mathbf{e} \cdot \mathbf{T}_{r}, \quad "
    r"\mathbf{Q}_{n} = R_{n} \mathbf{Q} R_{n}^{\top}, \ R_{n} = R(\mathbf{P}_{s}) \, "
    r"\mathrm{Rot}(\mathbf{a}, \phi_{n}), \quad "
    r"\tau_{q} = (\boldsymbol{\omega} \times \mathbf{e}) \cdot \mathbf{Q}_{n} \boldsymbol{\omega}, \quad "
    r"\Omega_{n+1} = \frac{J \Omega_{n} + \mathrm{d}t \, (\tau + \tau_{l} + \tau_{q} + b \, \mathbf{e} \cdot "
    r"\boldsymbol{\omega})}{J + \mathrm{d}t \, b}, \quad "
    r"\phi_{n+1} = \phi_{n} + \mathrm{d}t \, (\Omega_{n+1} - \mathbf{e} \cdot \boldsymbol{\omega}); \quad "
    r"\mathbf{R} = \begin{bmatrix} \mathbf{F}_{s} + \mathbf{F}_{r} \\ \mathbf{T}_{s} + \mathbf{T}_{r} - "
    r"(\mathbf{e} \cdot \mathbf{T}_{r}) \mathbf{e} - (\tau - b \, (\Omega_{n+1} - \mathbf{e} \cdot "
    r"\boldsymbol{\omega}) + \tau_{q}) \mathbf{e} + "
    r"(\mathbf{x}_{s} - \mathbf{x}_{m}) \times (\mathbf{F}_{s} + \mathbf{F}_{r}) \end{bmatrix}, \quad "
    r"\mathbf{H} = J \Omega_{n+1} \mathbf{e}, \quad "
    r"\mathbf{C} = (\mathbf{Q}_{n+1}, \mathbf{Q}_{n+1}^{+}), \ \mathbf{Q}_{n+1}^{+} \text{ at } \phi_{n+1} + "
    r"\mathrm{d}t \, (\Omega_{n+1} - \mathbf{e} \cdot \boldsymbol{\omega}); \quad "
    r"\mathbf{P}_{s} = \mathbf{M}_{j} \circ \mathbf{O}, \quad "
    r"\mathbf{P}_{r} = \mathbf{P}_{s} \circ \mathrm{Rot}(\mathbf{a}, \phi_{n+1}); \quad "
    r"\boldsymbol{\Omega}_{s} = \boldsymbol{\omega}, \quad \boldsymbol{\Omega}_{r} = \boldsymbol{\omega} + "
    r"(\Omega_{n+1} - \mathbf{e} \cdot \boldsymbol{\omega}) \mathbf{e}")


class DynamicMotorNode(SegmentNode):
    """A rotor of inertia ``J`` about ``axis`` (a unit vector in its base frame), a spinner: it carries the rotor's
    absolute spin about the axis ``Omega``, driven by the torque ``tau`` against viscous bearing damping ``b`` on its
    speed relative to its stator and the loads on its rotor about the axis, by semi-implicit Euler with the damping
    implicit; its angle ``phi`` is relative to its stator, turned by ``Omega - e . omega``. ``omega`` is the angular
    velocity of the frame it mounts on (``mount_rates``), and its pose ``mount_poses``, both this step's: the segment
    it mounts on steps first, and the motor reads its new state. ``J Omega_dot`` is the sum of the axial torques and,
    for a rotor unequal across its axis, ``(omega x e) . Q omega``, ``Q`` the half difference of its two moments
    across the axis (``asymmetry``, a 3 x 3 matrix in its base frame at angle zero, with ``Q a = 0``), turned with the
    rotor: exact for a rotor whose centre of mass is on its axis and whose inertia is not turned off it, whose mass and
    mean inertia across the axis ride on the link it is mounted on, grafted there at build. Its frames are ``base``,
    the stator, and ``rotor``, on one origin.

    Its stator holds the rotor: it takes the reaction of the drive and of the bearing's damping about the axis, of
    ``Q``'s torque on the spin, and the loads on the rotor across it (its force and its torque about any other axis),
    and with the loads on the stator itself passes them to the frame it is mounted on, as ``reaction``: one row per
    frame of the segment it mounts on, that frame's row ``[force; torque about the frame's origin]`` in the world, the
    others zero. What the frame needs of the spinning rotor beyond that it reads from ``momentum``, the rotor's
    ``J Omega e`` in that frame's row, whose gyroscopic torque the segment applies at its own velocity, and
    ``inertia``, the rotor's ``Q`` in the world at its angle now and a step on, which the segment carries in its mass
    matrix: a dynamic arm takes both on its next step. ``initial_angular_velocity`` is the rotor's speed relative to
    its stator at the start, so that its spin starts at that plus the mount's turning about the axis."""

    meta = NodeMeta(
        algorithm_id="MIME-NODE-110",
        description="A dynamic motor: a rotor carried by its absolute spin about its axis, driven by a torque against "
                    "its inertia, viscous bearing damping on its speed relative to its stator and the loads on it "
                    "about its axis, and the wrench its stator passes to the frame it is mounted on, its gyroscopic "
                    "torque among it.",
        governing_equations=MOTOR_EQUATIONS,
        discretization="Semi-implicit Euler, first order in time: the rotor's spin with its damping taken at the end "
                       "of the step, then its angle from the new spin; the mount's pose and angular velocity are this "
                       "step's, read after the segment it is mounted on has stepped. What it passes that segment "
                       "(its reaction, momentum and inertia) that segment reads on its next step.",
        assumptions=("An ideal torque source: no cogging, imbalance, armature or thermal terms (MIME-ANO-102)",
                     "Rigid bearings, and a rotor with its centre of mass on its axis and its inertia not turned off "
                     "it: its mass and mean inertia across the axis ride on the link it is mounted on, grafted there "
                     "at build, its inertia about the axis is carried by its absolute spin, and its inequality across "
                     "the axis turns with it (MIME-ANO-122)",
                     "What crosses between the rotor and the frame it is mounted on: the mount's pose and angular "
                     "velocity in, this step's; the reaction, the rotor's spin momentum and its inequality across the "
                     "axis out, which that frame's segment reads on its next step"),
        references=(Reference("Sola2017", "a rotation about an axis by the exponential map"),
                    Reference("Featherstone2008", "the reaction of a joint's actuator on its parent")),
    )
    symbols = (
        Symbol(r"\phi", SymbolKind.STATE, "angle", "1", f"{_MOTOR}.advance"),
        Symbol(r"\Omega", SymbolKind.STATE, "spin", "rad/s", f"{_MOTOR}.advance"),
        Symbol(r"\tau", SymbolKind.INPUT, "torque", "N m", f"{_MOTOR}.advance"),
        Symbol("J", SymbolKind.PARAMETER, "inertia", "kg m^2", f"{_MOTOR}.advance"),
        Symbol("b", SymbolKind.PARAMETER, "damping", "N m s", f"{_MOTOR}.advance"),
        Symbol(r"\mathbf{T}_{r}", SymbolKind.INPUT, "link_wrench", WRENCH_UNITS, f"{_MOTOR}.advance"),
        Symbol(r"\mathbf{F}_{s}", SymbolKind.INPUT, "link_wrench", WRENCH_UNITS, f"{_MOTOR}.passed"),
        Symbol(r"\mathbf{R}", SymbolKind.STATE, "reaction", WRENCH_UNITS, f"{_MOTOR}.passed"),
        Symbol(r"\mathbf{H}", SymbolKind.STATE, "momentum", MOMENTUM_UNITS, f"{_MOTOR}.handed"),
        Symbol(r"\mathbf{C}", SymbolKind.STATE, "inertia", INERTIA_UNITS, f"{_MOTOR}.handed"),
        Symbol(r"\mathbf{Q}", SymbolKind.PARAMETER, "asymmetry", INERTIA_UNITS, f"{_MOTOR}.unequal"),
        Symbol(r"\mathbf{P}", SymbolKind.STATE, "frame_poses", POSE_UNITS, f"{_MOTOR}.advance"),
        Symbol(r"\boldsymbol{\Omega}", SymbolKind.STATE, "frame_angular_velocities", RATE_UNITS, f"{_MOTOR}.advance"),
        Symbol(r"\boldsymbol{\omega}", SymbolKind.INPUT, "mount_rates", RATE_UNITS, f"{_MOUNTED}.base_rate"),
        Symbol(r"\mathbf{M}", SymbolKind.INPUT, "mount_poses", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{O}", SymbolKind.PARAMETER, "placement", POSE_UNITS, f"{_MOUNTED}.base"),
        Symbol(r"\mathbf{a}", SymbolKind.PARAMETER, "axis", "1", f"{_MOTOR}.poses"),
    )
    FRAMES = ("base", "rotor")

    def __init__(self, name: str, timestep: float, *, axis: Sequence[float], inertia: float, damping: float,
                 placement: Sequence[float], asymmetry: Sequence[float] = (0.0,) * 9, mount_index: int = -1,
                 mount_count: int = 0, initial_mount_pose: Sequence[float] = _IDENTITY,
                 initial_mount_rate: Sequence[float] = (0.0, 0.0, 0.0), initial_angle: float,
                 initial_angular_velocity: float = 0.0, ledger: Sequence[str] = ()) -> None:
        if len(asymmetry) != 9 and np.asarray(asymmetry).shape != (3, 3):
            raise ValueError(f"asymmetry {asymmetry!r} is not a 3 x 3 matrix")
        super().__init__(name, timestep, axis=flattened(axis), inertia=float(inertia), damping=float(damping),
                         placement=flattened(placement), asymmetry=flattened(asymmetry), mount_index=int(mount_index),
                         mount_count=int(mount_count), initial_mount_pose=flattened(initial_mount_pose),
                         initial_mount_rate=flattened(initial_mount_rate), initial_angle=float(initial_angle),
                         initial_angular_velocity=float(initial_angular_velocity), ledger=ledger)
        held = books.deliveries(self.constants())
        unknown = sorted({d.field for d in held} - {*_MOTOR_PAIRED, *(role for role, _, _ in self.hosted_laws())})
        if unknown:
            raise ValueError(f"the ledger delivers into {unknown}, which a dynamic motor pairs nothing for: it pairs "
                             f"{sorted(_MOTOR_PAIRED)} and hosts {[role for role, _, _ in self.hosted_laws()]}")

    def discretization_order(self) -> DiscretizationOrder:
        return DiscretizationOrder(temporal=1.0)

    @override
    def param_specs(self) -> dict[str, ParamSpec]:
        return {**super().param_specs(), "axis": ParamSpec(units="1"), "placement": ParamSpec(units=POSE_UNITS),
                "inertia": ParamSpec(units="kg m^2", bounds=(0.0, None)),
                "damping": ParamSpec(units="N m s", bounds=(0.0, None)),
                "asymmetry": ParamSpec(units=INERTIA_UNITS,
                                       description="the rotor's half difference across its axis, in its base frame "
                                                   "at angle zero")}

    def _mounted_rows(self) -> int:
        return max(cast(int, self.constants()["mount_count"]), 1)

    def poses(self, p: Mapping[str, Any], base: Array, angle: Array) -> Array:
        """The stator's and the rotor's world poses: the base, and the base turned by ``angle`` about the axis."""
        axis = jnp.asarray(p["axis"], dtype=angle.dtype)
        turned = jnp.concatenate([jnp.zeros(3, dtype=angle.dtype), quaternion.exponential(angle * axis)])
        return jnp.stack([base, compose(base, turned)])

    def passed(self, given: DynamicMotorInputs, base: Array, axis: Array, held: Array, wrenches: Array) -> Array:
        """What the stator passes to the frame it is mounted on: the loads on the stator, those on the rotor but its
        torque about ``axis`` (the world's), and ``-held`` about the axis, the reaction of every other torque on the
        spin (the drive's, the damping's and the inequality's); moved from the base's origin to the mount frame's, in
        that frame's row."""
        force = wrenches[0, :3] + wrenches[1, :3]
        across = wrenches[1, 3:] - jnp.dot(axis, wrenches[1, 3:]) * axis
        torque = wrenches[0, 3:] + across - held * axis
        index = cast(int, self.constants()["mount_index"])
        if index < 0:
            return self.in_row(jnp.zeros(6, dtype=base.dtype))
        frame = cast(Array, given.get("mount_poses"))[index]
        moved = torque + jnp.cross(base[:3] - frame[:3], force)
        return self.in_row(jnp.concatenate([force, moved]))

    def in_row(self, value: Array) -> Array:
        """``value`` in the mount frame's row of one row per frame of the segment it is mounted on, the others zero
        (all zero when it is placed in the world)."""
        rows, index = self._mounted_rows(), cast(int, self.constants()["mount_index"])
        held = jnp.zeros((rows, *value.shape), dtype=value.dtype)
        return held if index < 0 else held.at[index].set(value)

    def unequal(self, p: Mapping[str, Any], base: Array, angle: Array) -> Array:
        """The rotor's half difference across its axis, ``asymmetry`` turned with it, in the world at ``angle``."""
        rotor = self.poses(p, base, angle)[1]
        return turned(jnp.reshape(jnp.asarray(p["asymmetry"], dtype=angle.dtype), (3, 3)),
                      quaternion.rotation_matrix(rotor[3:]))

    def handed(self, p: Mapping[str, Any], base: Array, axis: Array, angle: Array, spin: Array, relative: Array,
               dt: Any) -> tuple[Array, Array]:
        """What the frame it is mounted on carries of the rotor, in its row: its spin momentum ``J Omega e``, and its
        half difference across the axis at ``angle`` and a step on at the speed ``relative``."""
        inertia = jnp.asarray(p["inertia"], dtype=angle.dtype)
        now, ahead = self.unequal(p, base, angle), self.unequal(p, base, angle + dt * relative)
        return self.in_row(momentum(inertia, spin, axis)), self.in_row(jnp.stack([now, ahead]))

    @override
    def initial_state(self) -> NodeFields:
        held = self.constants()
        angle = as_run_array(held["initial_angle"])
        p = {"axis": as_run_array(held["axis"])}
        base = self.initial_base()
        mount = self.initial_base_rate()
        axis = quaternion.rotate(base[3:], p["axis"])
        relative = as_run_array(held["initial_angular_velocity"])
        spin = relative + jnp.dot(axis, mount)
        p.update({name: as_run_array(held[name]) for name in ("inertia", "asymmetry")})
        spun_up, carried = self.handed(p, base, axis, angle, spin, relative, self.delta_t)
        return books.with_records(self, node_fields(DynamicMotorState(
            frame_poses=self.poses(p, base, angle), frame_angular_velocities=jnp.stack([mount, rotor_rate(spin, axis,
                                                                                                         mount)]),
            angle=angle, spin=spin, reaction=as_run_array(np.zeros((self._mounted_rows(), 6))), momentum=spun_up,
            inertia=carried)))

    def own_inputs(self) -> dict[str, BoundaryInputSpec]:
        """The motor's inputs without the ledger: its torque, the loads on its frames and its mount's."""
        return {"torque": node_input("N m", ()), "link_wrench": node_input(WRENCH_UNITS, (2, 6), additive=True),
                **self.mounted_inputs()}

    @override
    def boundary_input_spec(self) -> dict[str, BoundaryInputSpec]:
        """:meth:`own_inputs`; with the ledger attached, each delivery's own beside the input it would sum into."""
        return books.with_law_inputs(self.own_inputs(), books.deliveries(self.constants()))

    @override
    def advance(self, p: Mapping[str, Any], state: NodeFields, boundary_inputs: Mapping[str, Array],
                dt: float) -> NodeFields:
        """The step (:meth:`stepped_with_parts`); with the ledger attached, its books recorded beside it."""
        if not self.metered:
            return self.stepped_with_parts(p, state, boundary_inputs, dt)[0]
        return books.metered_step(self, self.own_inputs(), p, state, boundary_inputs, dt, self.stepped_with_parts)

    def stepped_with_parts(self, p: Mapping[str, Any], state: Mapping[str, Array],
                           boundary_inputs: Mapping[str, Array], dt: float) -> tuple[NodeFields, MotorParts]:
        """The step, and what the ledger's books read of it (:class:`MotorParts`): the same operations either way."""
        now, given = typed_view(DynamicMotorState, state), typed_view(DynamicMotorInputs, boundary_inputs)
        angle, spin = now["angle"], now["spin"]
        base = self.base(p, given)
        mount = self.base_rate(given, angle.dtype)
        axis = quaternion.rotate(base[3:], jnp.asarray(p["axis"], dtype=angle.dtype))
        torque = given["torque"] if "torque" in given else jnp.zeros((), dtype=angle.dtype)
        wrenches = given["link_wrench"] if "link_wrench" in given else jnp.zeros((2, 6), dtype=angle.dtype)
        inertia, damping = (jnp.asarray(p[name], dtype=angle.dtype) for name in ("inertia", "damping"))
        unequal = deviator_torque(self.unequal(p, base, angle), axis, mount)
        spin = spun(inertia, damping, spin, torque + jnp.dot(axis, wrenches[1, 3:]) + unequal, axis, mount, dt)
        relative = relative_rate(spin, axis, mount)
        angle = angle + dt * relative
        spun_up, carried = self.handed(p, base, axis, angle, spin, relative, dt)
        return (node_fields(DynamicMotorState(
            frame_poses=self.poses(p, base, angle),
            frame_angular_velocities=jnp.stack([mount, rotor_rate(spin, axis, mount)]), angle=angle, spin=spin,
            reaction=self.passed(given, base, axis, torque - damping * relative + unequal, wrenches),
            momentum=spun_up, inertia=carried)), MotorParts(axis, mount, relative, unequal))

    # ------------------------------------------------------------------------------------------- the energy ledger

    #: The motor's own scheme's numerical terms: semi-implicit Euler's ``(1/2) J (Omega' - Omega)^2``, and the torques'
    #: work on the angle's change beyond the step's solution, ``X (d phi - dt (Omega' - e . omega))`` (rounding).
    LEDGER_TERMS = ("backward_euler", "configuration")
    #: The command a motor pairs, with its angle's change on its stator: the drive's net work.
    PAIRED_COMMANDS = ("torque",)
    #: The class of what it hands the segment it is mounted on, by output: its rotor's momentum (its gyroscopic torque,
    #: which the host applies by a Cayley step, does no work) and its inequality across the axis (the inertia the host
    #: carries, whose change is paid by the inequality's torque on the spin: one exchange, ``<motor>.inertia`` on both
    #: receivers) are workless; its stator's reaction carries the torques it holds and the loads on its frames it
    #: passes on, its half of the reaction ``<motor>.reaction`` on its own books.
    DELIVERY_CLASSES = MappingProxyType({"momentum": books.WORKLESS, "inertia": books.WORKLESS,
                                         "reaction": books.UNMETERED})
    #: The hosted laws whose other half is on the segment it is mounted on: supplied by that segment where it is
    #: outside (a kinematic one, whose motion is prescribed).
    EXCHANGED = ("inertia", "reaction")

    @override
    def hosted_laws(self) -> tuple[tuple[str, str, str], ...]:
        """The laws the motor holds in its step, each ``(role, law, class)``: its bearing's damping on the rotor's turn
        relative to its stator (dissipative); its rotor's inequality across its axis, whose torque on the spin is the
        rotor's half of the exchange whose other half is the inertia its host carries (``<motor>.inertia``, the same
        law on both receivers, workless); and its stator's half of the reaction its host takes (``<motor>.reaction``):
        the torques it holds (the drive's, the bearing's, the inequality's) on the rotor's share of the host's turn,
        less what the loads it passes on did on its motion, which each law's row books as that law's."""
        return (("bearing", f"{self.name}.bearing", books.DISSIPATIVE),
                ("inertia", f"{self.name}.inertia", books.WORKLESS),
                ("reaction", f"{self.name}.reaction", books.UNMETERED))

    @override
    def configuration(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """Its frames' world poses: where a field it carries is placed, which the field's energy reads."""
        return jnp.asarray(state["frame_poses"])

    def frame_poses_at(self, p: Mapping[str, Any], configuration: Array) -> Array:
        """Its frames' world poses at ``configuration``, which is them."""
        return jnp.asarray(configuration)

    @override
    def storage(self, p: Mapping[str, Any], state: Mapping[str, Array]) -> Array:
        """``(1/2) J Omega^2``, the rotor's kinetic energy about its axis (its mass and its inertia across the axis ride
        on its host)."""
        spin = jnp.asarray(state["spin"])
        return 0.5 * jnp.asarray(p["inertia"], dtype=spin.dtype) * spin * spin

    @override
    def identity_terms(self, p: Mapping[str, Any], held: Sequence[books.LedgerDelivery], state: Mapping[str, Array],
                       boundary_inputs: Mapping[str, Array], ended: Mapping[str, Array], parts: MotorParts,
                       dt: float) -> books.StepAccount:
        """The step's identity, ``(1/2) J Omega'^2 - (1/2) J Omega^2 = sum_law W_law + W_unattributed - N``, from ``J
        (Omega' - Omega) = dt X``, ``X = tau + T + tau_q - b r`` (``r = Omega' - e . omega``, the rotor's speed on its
        stator): the drive with the angle's change ``d phi``; a load on the rotor about the axis ``T`` with the rotor's
        own turn, ``d phi + dt e . omega``; the bearing ``-b r d phi``; the inequality's torque on the spin ``tau_q d
        phi``; and the stator's torques on its host's share, ``(tau + tau_q - b r) dt e . omega``. A load on its frames
        across the axis passes to its host, which pairs it."""
        spin, angle = state["spin"], state["angle"]
        dtype = spin.dtype
        spun, turned_to = ended["spin"], ended["angle"]
        inertia, damping = (jnp.asarray(p[name], dtype=dtype) for name in ("inertia", "damping"))
        axis, mount, relative, unequal = parts
        own = self.own_inputs()
        given = {**{name: jnp.asarray(entry.default, dtype=dtype) for name, entry in own.items()},
                 **books.summed(own, held, boundary_inputs)}
        turn = turned_to - angle
        host = dt * jnp.dot(axis, mount)
        bearing = -damping * relative
        # The stator's motion over the step, its base from where the step before left it to where it is now: what the
        # loads it passes on do work on, the host's (its turn in its small-angle form, a step's turn being small).
        before, after = state["frame_poses"][0], ended["frame_poses"][0]
        moved_by = after[:3] - before[:3]
        turned_by = 2.0 * quaternion.multiply(after[3:], quaternion.conjugate(before[3:]))[1:]
        passed = jnp.zeros((), dtype=dtype)

        def paired(field: str, value: Array) -> tuple[Array, Array]:
            """A delivery into ``field`` paired by the one rule, and the magnitude of that arithmetic: the torque with
            the angle's change; a load on its frames by its torque on the rotor about the axis with the rotor's turn,
            and by what it passes on to the host with the stator's motion (the same work the stator's reaction then
            carries away, booked against it under ``<motor>.reaction``)."""
            nonlocal passed
            if field == "torque":
                return value * turn, books.magnitude(value) * books.magnitude(turn)
            axial = jnp.dot(axis, value[1, 3:])
            force = value[0, :3] + value[1, :3]
            torque = value[0, 3:] + value[1, 3:] - axial * axis
            handed = jnp.dot(force, moved_by) + jnp.dot(torque, turned_by)
            passed = passed + handed
            return axial * (turn + host) + handed, (
                books.magnitude(axial) * (books.magnitude(turn) + books.magnitude(host))
                + books.magnitude(force) @ books.magnitude(moved_by)
                + books.magnitude(torque) @ books.magnitude(turned_by))

        rows = books.laws(held)
        work = jnp.zeros(len(rows), dtype=dtype)
        hosted = jnp.zeros((len(rows), 2), dtype=dtype)
        size = jnp.zeros((), dtype=dtype)
        torque = given["torque"]
        held_torque = torque + unequal + bearing
        reaction_row: int | None = None
        for index, delivery in enumerate(held):
            row = rows.index(delivery.law)
            if delivery.kind == books.HOSTED:
                if delivery.field == "reaction":
                    reaction_row = row
                    continue
                found = {"bearing": bearing * turn, "inertia": unequal * turn}[delivery.field]
                magnitude = {"bearing": books.magnitude(bearing) * books.magnitude(turn),
                             "inertia": books.magnitude(unequal) * books.magnitude(turn)}[delivery.field]
            else:
                found, magnitude = paired(delivery.field, boundary_inputs[books.law_input(delivery.field, index)])
            work, size = work.at[row].add(found), size + magnitude
        unattributed = jnp.zeros((), dtype=dtype)
        for field in _MOTOR_PAIRED:
            found, magnitude = paired(field, boundary_inputs.get(field, jnp.asarray(own[field].default, dtype=dtype)))
            unattributed, size = unattributed + found, size + magnitude
        if reaction_row is not None:
            # The stator's torques on the host's share of the rotor's turn, less what the loads it passes on did on its
            # motion: its half of the reaction its host takes.
            work = work.at[reaction_row].add(held_torque * host - passed)
            size = size + (books.magnitude(torque) + books.magnitude(unequal) + books.magnitude(bearing)) \
                * books.magnitude(host) + books.magnitude(passed)
        axial = jnp.dot(axis, given["link_wrench"][1, 3:])
        total = torque + axial + unequal + bearing
        moved = spun - spin
        backward_euler = 0.5 * inertia * moved * moved
        configuration = total * (turn - dt * relative)
        torque_size, axial_size, spun_size = books.magnitude(torque), books.magnitude(axial), books.magnitude(spun)
        unequal_size, bearing_size = books.magnitude(unequal), books.magnitude(bearing)
        size = size + 0.5 * inertia * (spin * spin + spun * spun + moved * moved) \
            + (torque_size + axial_size + unequal_size + bearing_size) * (books.magnitude(turn)
                                                                          + dt * books.magnitude(relative)) \
            + spun_size * (inertia * (books.magnitude(spin) + spun_size) + dt * (
                torque_size + axial_size + unequal_size + damping * (spun_size + books.magnitude(host) / dt)))
        return books.StepAccount(
            work=work, hosted=hosted, unattributed=unattributed, terms=jnp.stack([backward_euler, configuration]),
            storage_start=self.storage(p, state), storage_end=self.storage(p, ended), scale=size, operators={})


#: The inputs a dynamic motor pairs: its torque and the loads on its frames.
_MOTOR_PAIRED = ("torque", "link_wrench")


__all__ = ["ARM_EQUATIONS", "INERTIA_UNITS", "MOMENTUM_UNITS", "MOTOR_EQUATIONS", "SWEEPS", "WRENCH_UNITS", "ArmParts",
           "DynamicArmInputs", "DynamicArmNode", "DynamicArmState", "DynamicMotorInputs", "DynamicMotorNode",
           "DynamicMotorState", "MeteredDynamicArmState", "MeteredDynamicMotorState", "MotorParts"]
