"""The energy ledger's cases on the rigid body's node, as the build will meter it: ``rigid_inertial`` 0.4.0's node with
a ledger of a load law and a resistance law (and, held, its contact hosted in its step), one step from drawn states.

The draws are at FL-9's scales (script A's body), in the dtype the node computes in: a mass about 8.5e-5 kg, principal
moments about 1e-10 kg m^2 within the triangle inequality about drawn axes, speeds to a centimetre a second, rates to
100 rad/s, loads to 1e-5 N and 1e-8 N m, a resistance of FL-9's in a tube (4e-5 N s/m, 8e-10 N m s, a coupling up to
0.3 of their geometric mean), steps from 0.1 to 1 ms (four, each compiled once). Every component is drawn 0.1 to 1 of its scale with its own sign,
so none underflows and none sits on an axis plane."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from mime.core import ledger as books
from mime.nodes.body import HULL, RigidInertialNode, RigidInertialTensorNode
from mime.scene import Cylinder
from mime.scene._sampling import quadrature_arrays, quadrature_counts
from mime.units import mm

MASS = 8.491116020725203e-05
TUBE_RADIUS, MARGIN = 3.175e-3, 1e-4
#: The laws a case's body is metered for: a load (unmetered, as a field's torque is) and a resistance (dissipative,
#: its definiteness the state's in the checked arm), and, held, the contact hosted in its step.
LOAD_LAW = books.LedgerDelivery("applied_wrench", "load:case", books.UNMETERED, books.LOAD)
DRAG_LAW = books.LedgerDelivery("resistance", "drag:case", books.DISSIPATIVE, books.SOLVED)
CHECKED_DRAG = books.LedgerDelivery("resistance", "drag:case", books.DISSIPATIVE, books.SOLVED, books.CHECK_DEFINITE)
CONTACT_LAW = books.LedgerDelivery("confinement", "contact:case", books.DISSIPATIVE, books.HOSTED)
#: The hull arm's body: a rod 1 mm in radius and 6 mm long, its hull's points at 0.4 mm (280 of them), as the hull
#: contact's own tests hold it.
ROD = (Cylinder(radius=mm(1.0), length=mm(6.0), axis=(1.0, 0.0, 0.0)),
       {"radius": 1e-3, "length": 6e-3, "axis": (1.0, 0.0, 0.0)})
ROD_SPACING = 4e-4
#: The steps drawn from, each compiled once: a tenth of script A's to twice it.
STEPS = (1e-4, 2.5e-4, 5e-4, 1e-3)


@contextmanager
def in_dtype(name: str) -> Iterator[Any]:
    """The run's dtype ``name`` (``float32`` or ``float64``) while the block runs: the dtype the node computes in."""
    with jax.enable_x64(name == "float64"):
        yield jnp.dtype(name)


def unit(draw: Callable[..., Any], size: int) -> np.ndarray:
    """A unit vector of ``size`` components, each 0.1 to 1 in size with its own sign, normalised."""
    magnitudes = draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size))
    signs = draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size))
    held = np.asarray(magnitudes, dtype=np.float64) * np.asarray(signs)
    return held / np.linalg.norm(held)


def vector(draw: Callable[..., Any], least: float, most: float) -> np.ndarray:
    """A vector of three components whose length is drawn from ``least`` to ``most``, in a drawn direction."""
    return draw(st.floats(least, most)) * unit(draw, 3)


def rotation_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tensor(draw: Callable[..., Any]) -> np.ndarray:
    """A body's inertia tensor about its centre of mass: principal moments 0.5 to 3 e-10 kg m^2, the largest at most
    0.975 of the other two's sum, about axes turned by a drawn rotation."""
    least = draw(st.floats(0.5e-10, 1.5e-10))
    middle = least * (1.0 + draw(st.floats(0.0, 0.6)))
    largest = min(middle * (1.0 + draw(st.floats(0.0, 0.6))), 0.975 * (least + middle))
    axes = rotation_of(unit(draw, 4))
    return axes @ np.diag([least, middle, largest]) @ axes.T


def resistance(draw: Callable[..., Any], stiffness: float = 1.0) -> np.ndarray:
    """A symmetric positive definite resistance at FL-9's scales in a tube, turned into the world by a drawn
    rotation, times ``stiffness``."""
    translation = np.diag(4e-5 * np.asarray(draw(st.lists(st.floats(1.0, 2.0), min_size=3, max_size=3))))
    turning = np.diag(8e-10 * np.asarray(draw(st.lists(st.floats(1.0, 3.0), min_size=3, max_size=3))))
    entries = np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=9, max_size=9)), dtype=np.float64)
    coupling = 0.3 * np.sqrt(4e-5 * 8e-10) * entries.reshape(3, 3) / 3.0
    turn = rotation_of(unit(draw, 4))
    both = np.zeros((6, 6))
    both[:3, :3] = both[3:, 3:] = turn
    return stiffness * (both @ np.block([[translation, coupling], [coupling.T, turning]]) @ both.T)


@st.composite
def free_states(draw: Callable[..., Any], *, stiff: bool = False, pushing: bool = False) -> dict[str, np.ndarray]:
    """A free body's state and what the laws deliver to it. ``stiff``: the resistance 1 to 10^6 times FL-9's, so that
    ``dt |R| / m`` reaches several hundred. ``pushing``: the load along the body's motion, its force a share 0.1 to 1
    of its length along the velocity and its torque along the angular velocity, so that it does work on the step."""
    velocity, rate = vector(draw, 1e-4, 1e-2), vector(draw, 0.1, 100.0)
    force, torque = vector(draw, 1e-8, 1e-5), vector(draw, 1e-12, 1e-8)
    if pushing:
        along = draw(st.floats(0.1, 1.0))
        force = np.linalg.norm(force) * (along * velocity / np.linalg.norm(velocity) + (1.0 - along) * unit(draw, 3))
        torque = np.linalg.norm(torque) * (along * rate / np.linalg.norm(rate) + (1.0 - along) * unit(draw, 3))
    stiffness = 10.0 ** draw(st.floats(0.0, 6.0)) if stiff else 1.0
    return {"mass": np.asarray(MASS * draw(st.floats(0.5, 2.0))), "tensor": tensor(draw),
            "pose": np.concatenate([vector(draw, 1e-5, 1e-2), unit(draw, 4)]),
            "twist": np.concatenate([velocity, rate]), "load": np.concatenate([force, torque]),
            "resistance": resistance(draw, stiffness), "dt": np.asarray(draw(st.sampled_from(STEPS)))}


@st.composite
def centre_held_states(draw: Callable[..., Any]) -> dict[str, np.ndarray]:
    """A body held by its centre in script A's tube (along x through the origin), on the tube's wall shrunk by its
    extent (1 to 2 mm) and the margin, moving outward across it: the state the hold acts on in the solve, and the
    projection and the velocity rule after it."""
    drawn = draw(free_states())
    extent = draw(st.floats(1e-3, 2e-3))
    around = draw(st.floats(0.0, 2.0 * math.pi))
    outward = np.array([0.0, math.cos(around), math.sin(around)])
    room = TUBE_RADIUS - extent - MARGIN
    velocity = drawn["twist"][:3]
    speed = draw(st.floats(1e-4, 1e-2))
    along = velocity - outward * np.dot(outward, velocity)
    position = np.concatenate([[draw(st.floats(-1e-2, 1e-2))], room * outward[1:]])
    return {**drawn, "extent": np.asarray(extent), "pose": np.concatenate([position, drawn["pose"][3:]]),
            "twist": np.concatenate([along + speed * outward, drawn["twist"][3:]])}


@cache
def rod_points() -> np.ndarray:
    kind, settings = ROD
    patches = tuple(kind.patches)
    points, _, _ = quadrature_arrays(kind, settings, patches, quadrature_counts(kind, settings, ROD_SPACING, patches))
    return np.asarray(points, dtype=np.float64)


def rod_least_gap(position: np.ndarray, orientation: np.ndarray) -> float:
    placed = position + rod_points() @ rotation_of(orientation).T
    return float(np.min(TUBE_RADIUS - MARGIN - np.hypot(placed[:, 1], placed[:, 2])))


@st.composite
def hull_held_states(draw: Callable[..., Any]) -> dict[str, np.ndarray]:
    """The rod held by its hull in script A's tube: its axis along the tube's or tilted up to 0.2 rad, spun about its
    own axis, its centre placed so that its nearest point stands 0 to 1 um inside the wall less the margin, moving
    outward towards it, at script A's step (the hull's step compiled once); the rest as a free body's."""
    drawn = draw(free_states())
    spin, tilt = draw(st.floats(0.0, 2.0 * math.pi)), draw(st.floats(0.0, 0.2))
    about, around = draw(st.floats(0.0, 2.0 * math.pi)), draw(st.floats(0.0, 2.0 * math.pi))
    across = np.array([0.0, math.cos(about), math.sin(about)])
    turn = np.concatenate([[math.cos(tilt / 2)], math.sin(tilt / 2) * across])
    spun = np.array([math.cos(spin / 2), math.sin(spin / 2), 0.0, 0.0])
    w1, x1, y1, z1 = turn
    w2, x2, y2, z2 = spun
    orientation = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    outward = np.array([0.0, math.cos(around), math.sin(around)])
    gap = draw(st.floats(0.0, 1e-6))
    low, high = 0.0, TUBE_RADIUS
    for _ in range(80):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if rod_least_gap(middle * outward, orientation) > gap else (low, middle)
    velocity = drawn["twist"][:3]
    speed = draw(st.floats(1e-4, 1e-2))
    along = velocity - outward * np.dot(outward, velocity)
    return {**drawn, "pose": np.concatenate([low * outward, orientation]), "dt": np.asarray(5e-4),
            "twist": np.concatenate([along + speed * outward, drawn["twist"][3:]])}


#: The defects a :class:`PlantedBody` can carry, each sized in the step's own floors (``IDENTITY_ULPS`` units of eps on
#: the step's scale, from the unplanted step), so that a plant clears the floor on every draw by construction:
#: ``ANGULAR_GIFT`` and ``LINEAR_GIFT`` raise the end of the step's storage by that many floors by scaling the ended
#: twist's angular or translational part, unbooked; ``CONTACT_GIFT`` makes the contact's net work over the step that
#: many floors, positive (a contact that gives energy, honestly accounted: the angular gift booked under the contact's
#: name, which the identity cannot see); the rest add that many floors
#: (either sign) to a booked term, a misbooking in the step's account: the hosted contact's loss and its projection's
#: shift, and the three terms of the scheme.
ANGULAR_GIFT, LINEAR_GIFT, CONTACT_GIFT, HOSTED_LOSS, HOSTED_SHIFT, EULER, GYROSCOPIC, CONFIGURATION = range(8)
#: Two more that are not a number of floors: ``REVERSED_HOLD`` takes the contact's correction the wrong way that many
#: times over (the physical form of a contact that gives energy), ``HALVED_TERMS`` books the scheme's terms at half
#: (a flag, 1 for halved).
REVERSED_HOLD, HALVED_TERMS = 8, 9
PLANTS = ("angular_gift", "linear_gift", "contact_gift", "hosted_loss", "hosted_shift", "backward_euler", "gyroscopic",
          "configuration", "reversed_hold", "halved_terms")


class PlantedBody(RigidInertialTensorNode):
    """A plant, one node for every defect so that each arm compiles once per dtype: ``rigid_inertial`` 0.4.0's node
    whose step is the unplanted step, then the defects of ``p["plant"]`` (floors each, zero for none, in the order of
    :data:`PLANTS`; the last two not floors), the account formed from the planted end of the step. With every size
    zero it is the unplanted node, bit for bit."""

    def held(self, p, position, system, moved, dt):
        corrected = super().held(p, position, system, moved, dt)
        return jnp.where(p["plant"][REVERSED_HOLD] != 0.0, moved - p["plant"][REVERSED_HOLD] * (corrected - moved),
                         corrected)

    def hull_held(self, p, body, pose, system, moved, dt):
        found = super().hull_held(p, body, pose, system, moved, dt)
        reversed_hold = moved - p["plant"][REVERSED_HOLD] * (found[0] - moved)
        return (jnp.where(p["plant"][REVERSED_HOLD] != 0.0, reversed_hold, found[0]), *found[1:])

    def advance(self, p, state, boundary_inputs, dt):
        held = books.deliveries(self.constants())
        own = books.unrecorded(state)
        ended, parts = self.stepped_with_parts(p, own, books.summed(self.own_inputs(), held, boundary_inputs), dt)
        clean = self.identity_terms(p, held, own, boundary_inputs, ended, parts, dt)
        dtype = ended["twist"].dtype
        floor = books.IDENTITY_ULPS * jnp.finfo(dtype).eps * clean.scale
        plant = p["plant"]
        twist = ended["twist"]
        inertia = self._world_inertia(p, ended)
        linear = 0.5 * p["mass"] * jnp.dot(twist[:3], twist[:3])
        angular = 0.5 * jnp.dot(twist[3:], inertia @ twist[3:])
        rows = books.laws(held)
        contacts = [rows.index(d.law) for d in held if d.kind == books.HOSTED]
        # The contact gives energy: its net work over the step ends at ``CONTACT_GIFT`` floors, whatever it dissipated
        # in the unplanted step, the step's storage raised by the same, so that the identity closes.
        planted_contact = plant[CONTACT_GIFT] != 0.0
        gifts = [jnp.where(planted_contact, plant[CONTACT_GIFT] * floor - clean.work[row], 0.0) for row in contacts]
        given = sum(gifts, jnp.zeros((), dtype=dtype))
        raise_by = jnp.stack([plant[LINEAR_GIFT] * floor, plant[ANGULAR_GIFT] * floor + given])
        share = jnp.sqrt(1.0 + raise_by / jnp.stack([linear, angular]))
        planted = {**ended, "twist": jnp.concatenate([share[0] * twist[:3], share[1] * twist[3:]])}
        account = self.identity_terms(p, held, own, boundary_inputs, planted, parts, dt)
        work, hosted = account.work, account.hosted
        for row, gift in zip(contacts, gifts, strict=True):
            work = work.at[row].add(gift)
            hosted = hosted.at[row].add(floor * jnp.stack([plant[HOSTED_LOSS], plant[HOSTED_SHIFT]]))
        halved = jnp.where(plant[HALVED_TERMS] != 0.0, 0.5, 1.0)
        account = dataclasses.replace(account, work=work, hosted=hosted,
                                      terms=halved * account.terms + floor * plant[EULER:CONFIGURATION + 1])
        return {**planted, **books.recorded(state, account, held)}


def metered_body(node_class: type[RigidInertialNode] = RigidInertialTensorNode, *, held: str = "",
                 extent: float = 1.5e-3, laws: tuple[books.LedgerDelivery, ...] = (LOAD_LAW, DRAG_LAW),
                 dt: float = 5e-4, loops: bool = False) -> RigidInertialNode:
    """``node_class`` metered for ``laws`` (and the contact, where ``held`` is ``"centre"`` or ``"hull"``): free, its
    centre held in script A's tube by ``extent`` plus the margin, or its hull's points (the rod's) held, its sweeps
    and search XLA loops (the contact 0.5.0, the build's default) where ``loops``, unrolled (0.4.0) otherwise."""
    confinement: dict[str, Any] = {}
    if held:
        laws = (*laws, CONTACT_LAW)
        confinement = {"surface_shape": "cylinder", "margin": MARGIN, "velocity_rule": "remove_outward",
                       "surface_centre": (0.0, 0.0, 0.0), "surface_axis": (1.0, 0.0, 0.0),
                       "surface_radius": TUBE_RADIUS}
    if held == "centre":
        confinement["extent"] = extent
    if held == "hull":
        kind, settings = ROD
        patches = tuple(kind.patches)
        confinement.update(hull_kind=kind.name, hull_kind_version=kind.version, hull_patches=patches,
                           hull_counts=quadrature_counts(kind, settings, ROD_SPACING, patches),
                           **{f"{HULL}{name}": value for name, value in settings.items()})
        if loops:
            confinement["contact_loops"] = True
    nine = issubclass(node_class, RigidInertialTensorNode)
    return node_class("umr", dt, mass=MASS, inertia=tuple(np.eye(3).reshape(9) * 1e-10) if nine else (1e-10,) * 3,
                      initial_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), initial_twist=(0.0,) * 6,
                      ledger=tuple(d.encoded() for d in laws), **confinement)


def law_inputs(node: RigidInertialNode, load: Any, resistance: Any, dtype: Any,
               unattributed: Any = None) -> dict[str, jax.Array]:
    """The node's inputs: each law's own (the load's and the resistance's), and the summed ones, which take
    ``unattributed`` into the applied wrench where it is given and nothing otherwise."""
    held = books.deliveries(node.constants())
    found = {name: jnp.zeros(entry.shape, dtype=dtype) for name, entry in node.own_inputs().items()}
    if unattributed is not None:
        found["applied_wrench"] = jnp.asarray(unattributed, dtype=dtype)
    for index, delivery in enumerate(held):
        if delivery.kind == books.HOSTED:
            continue
        found[books.law_input(delivery.field, index)] = jnp.asarray(
            load if delivery.field == "applied_wrench" else resistance, dtype=dtype)
    return found


def one_step(node: RigidInertialNode, drawn: Mapping[str, np.ndarray], dtype: Any,
             unattributed_share: float = 0.0, plant: Mapping[int, float] | None = None) -> dict[str, jax.Array]:
    """The node's state after one metered step from ``drawn``, its records included; ``unattributed_share`` of the
    load delivered into the summed input too, an edge no law owns. ``plant``, for a :class:`PlantedBody`, maps a
    defect's index to its size in the step's own floors. A node of an earlier version, which takes three
    moments about the body's axes, takes the drawn tensor's diagonal."""
    planted = np.zeros(len(PLANTS))
    for index, floors in (plant or {}).items():
        planted[index] = floors
    inertia = drawn["tensor"] if isinstance(node, RigidInertialTensorNode) else np.diag(drawn["tensor"])
    return stepper(node, dtype, unattributed_share)(
        *(jnp.asarray(drawn[k], dtype=dtype) for k in ("mass", "pose", "twist", "load", "resistance")),
        jnp.asarray(inertia, dtype=dtype), jnp.asarray(drawn.get("extent", node.constants()["extent"]), dtype=dtype),
        jnp.asarray(planted, dtype=dtype), float(drawn["dt"]))


_steppers: dict[tuple[int, str, float], Callable[..., dict[str, jax.Array]]] = {}


def stepper(node: RigidInertialNode, dtype: Any, share: float) -> Callable[..., dict[str, jax.Array]]:
    """The node's metered step compiled once per node, dtype and share, the step's length static."""
    key = (id(node), str(jnp.dtype(dtype)), share)
    if key not in _steppers:
        def step(mass: Any, pose: Any, twist: Any, load: Any, resistance: Any, inertia: Any, extent: Any,
                 planted: Any, dt: float) -> dict[str, jax.Array]:
            p = {**{k: jnp.asarray(v, dtype=dtype) for k, v in node._step_constants(None).items()}, "mass": mass,
                 "inertia": jnp.reshape(inertia, (-1,)), "extent": extent, "plant": planted}
            state = {**{k: jnp.asarray(v, dtype=dtype) for k, v in node.initial_state().items()},
                     "pose": pose, "twist": twist}
            inputs = law_inputs(node, load, resistance, dtype, share * load if share else None)
            return node.advance(p, state, inputs, dt)

        _steppers[key] = jax.jit(step, static_argnums=8)
    return _steppers[key]


def identity_over_floor(found: Mapping[str, jax.Array]) -> float:
    """The step identity's residual in units of its floor (above one fails)."""
    eps = float(jnp.finfo(found["ledger_step_identity"].dtype).eps)
    return float(found["ledger_step_identity"]) / (books.IDENTITY_ULPS * eps)


def swinging_rotor(dt: float) -> Any:
    """dev1's swing under gravity, started 0.3 rad out, carrying a dynamic motor whose rotor carries A0's magnet across
    its axis (its half difference across it non-zero), spun from 10 Hz by 1 mN m; the magnet's field read by a body
    30 cm below, both magnetic laws attached, so that their reactions reach the rotor and the motor is metered too."""
    from chain_cases import BODY_INERTIA, BODY_MASS, BODY_MOMENT, MAGNET_LENGTH, MAGNET_MOMENT, MAGNET_RADIUS, \
        ROTOR_INERTIA, SWING
    from mime.drives import Constant
    from mime.dynamics import Arm, Motor, RigidInertial
    from mime.experiment import Experiment
    from mime.fields import PointDipole
    from mime.interactions import Magnetic
    from mime.observations import EnergyLedger
    from mime.scene import Cylinder, Scene, Z, asset
    from mime.things import Apparatus, Body, Part, PermanentMagnet, State
    from mime.units import A_m2, N_m, N_m_s, kg, kg_m2, kg_per_m3, m, m_per_s, ms, rad, rad_per_s, standard_gravity

    #: Sintered NdFeB's typical density (an inference), A0's magnet's.
    magnet_density = 7500.0

    exp = Experiment("swinging_rotor", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(dt * 1e3))
    exp.add_apparatus(Apparatus("swing", chain={
        "arm": Arm.Dynamic(description=asset(SWING), initial_positions=rad((0.3,)),
                           initial_velocities=rad_per_s((0.0,))),
        "motor": Motor.Dynamic(axis=Z, inertia=kg_m2(ROTOR_INERTIA), damping=N_m_s(0.0), initial_angle=rad(0.0),
                               mount="arm", initial_velocity=rad_per_s(2 * np.pi * 10.0),
                               top_speed=rad_per_s(2 * np.pi * 20.0))},
        parts={"magnet": Part(Cylinder(radius=m(MAGNET_RADIUS), length=m(MAGNET_LENGTH), axis=(1.0, 0.0, 0.0)),
                              mount="motor.rotor")},
        magnets={"magnet": PermanentMagnet(direction=(1.0, 0.0, 0.0))}, fields={"magnetic": PointDipole(part="magnet")},
        commands={"arm.joint_torques": Constant(N_m((0.0,))), "motor.torque": Constant(N_m(1e-3))}))
    exp.declare("swing.parts.magnet.density", kg_per_m3(magnet_density), reason="sintered NdFeB, its typical value")
    exp.declare("swing.magnetic_moment", A_m2(MAGNET_MOMENT), reason="A0's rotor magnet")
    exp.add_body(Body("umr", dynamics=RigidInertial(), initial=State(
        position=m((0.0, 0.0, -0.3)), orientation=(1.0, 0.0, 0.0, 0.0), velocity=m_per_s((0.0, 0.0, 0.0)),
        angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    for path, value in (("umr.mass", kg(BODY_MASS)), ("umr.inertia", kg_m2(BODY_INERTIA)),
                        ("umr.magnetic_moment", A_m2(BODY_MOMENT)), ("umr.magnet_centre", m((0.0, 0.0, 0.0)))):
        exp.declare(path, value, reason="a millimetre body, round numbers of its order")
    for law in (Magnetic.Torque(), Magnetic.Force()):
        exp.attach(law, body="umr", field="swing.magnetic")
    exp.observe(EnergyLedger())
    return exp

__all__ = ["ANGULAR_GIFT", "CHECKED_DRAG", "CONFIGURATION", "CONTACT_GIFT", "CONTACT_LAW", "DRAG_LAW", "EULER",
           "GYROSCOPIC", "HALVED_TERMS", "HOSTED_LOSS", "HOSTED_SHIFT", "LINEAR_GIFT", "LOAD_LAW", "MASS", "PLANTS",
           "PlantedBody", "REVERSED_HOLD",
           "centre_held_states", "free_states", "hull_held_states", "identity_over_floor", "in_dtype", "law_inputs",
           "metered_body", "one_step", "swinging_rotor", "unit", "vector"]
