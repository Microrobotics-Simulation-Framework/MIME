"""``rigid_inertial`` 0.4.0's node carries a body's full inertia tensor (F0-3): its oracles, in float64.

**The principal-frame equivalence.** A body whose tensor ``J`` has products moves as the same body turned into its
principal frame, ``J = Q D Q^T``, with the diagonal ``D`` on 0.3.0's node (the released one, which takes three
moments), its torque and its coupled resistance turned with it: its attitude, turned back by ``Q``, is the principal
body's, to rounding, at every step of the run. Drawn: the tensor (principal moments at FL-9's scale, within the
triangle inequality, about drawn axes; or FL-9's own tensor, which its parts give), the starting pose and rate, the
body's moment and its resistance in its own frame; driven (a 1.2 mT field turning at 10 Hz, a drag, 0.5 ms steps) or
free (200 rad/s, 0.1 ms steps). The equivalence holds within the draw's own floor over :data:`STEPS` steps: the larger of
:data:`FLOOR` and :data:`RESOLUTION_FACTOR` times what the principal body itself moves under a one-ulp change of its
moments or of its start (a driven draw amplifies rounding about a thousandfold per thousand steps, so an absolute floor
has no relation to a draw's own sensitivity). **The plants**, each caught on every draw, more than ten times that
floor, so that a draw whose floor nears a plant fails and is never filtered away: a product's sign flipped in one place of the
tensor, in both, the world inertia formed ``Q^T J Q``, the step's Jacobian transposed. The drawn tensors' principal
moments are distinct and their axes turned 20 to 70 degrees from the body's: a body of equal moments, or one whose
principal axes are a permutation of the body's, has no products, and on such draws (4.5 % of a first strategy's, the
moments drawn equal or the axes a quarter or third turn about a diagonal) no product plant could move anything. In float64 only: in float32 one
threshold missed the screw's sign flip on most driven draws (F0's spike S6), where its products, a thousandth of its
moments, move the attitude by less than float32's rounding of it.

**Gradients through equal moments.** The derivative of the final attitude in the tensor's six entries, by reverse
mode through the steps, agrees with central differences to their own resolution at FL-9's tensor and with its two
transverse moments exactly equal, the axisymmetric body every screw of one start is. The control: the same body
carried as principal moments and axes (an eigen-decomposition of the tensor) has no finite derivative there.

**Torque-free conservation**, a second check and not a control (S6: it catches a transposed Jacobian on fewer than
half the draws): a body with products keeps its kinetic energy and the magnitude of its angular momentum to second
order in the step, with no drift."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.core import quaternion
from mime.nodes.body import (
    RigidInertialMidpointNode,
    RigidInertialNode,
    RigidInertialTensorNode,
    gyroscopic_jacobian,
    split_gyroscopic_jacobian,
)

MASS = 8.491116020725203e-05
#: Steps of each run, and the two operating points: script A's step, driven by its field and resisted by a drag; and
#: a tenth of it, free and spinning fast.
STEPS = 2000
DRIVEN = {"dt": 5e-4, "rate": 5.0, "field": 1.2e-3, "frequency": 10.0}
FREE = {"dt": 1e-4, "rate": 200.0, "field": 0.0, "frequency": 0.0}
#: How far, in radians, the body's attitude may stand from the principal body's over a run at least, and ten times the
#: draw's floor the least a plant must move it. The unplanted arm's resolution as first measured, 200 draws a set
#: through this file's own strategies and runs (float64): at most 1.5e-12, 1.1e-11, 5.2e-11 and 2.0e-12 radians in the
#: four sets (drawn tensors driven and free, FL-9's driven and free). That sample missed the tail: on CI one stress run
#: in four had an FL-9 driven draw at 1.1e-9 to 2.2e-8, because the principal body itself, run twice with its start
#: changed by one ulp, stands 1.3e-8 apart (error growth about 10^3 per thousand steps). So the floor is the larger of
#: this constant and :data:`RESOLUTION_FACTOR` times the principal body's own divergence on the draw (:func:`resolution`).
#: Measured over 1500 fresh draws a set (float64): the largest floor reached 2.4e-7 radians (FL-9 driven; 63 of its draws
#: had a floor above this constant, 3 of the drawn tensors driven), the unplanted arm at most 0.14 of its floor, and the
#: least plant 8.4e-4 radians, 8.4e5 times the floor of its draw, every plant clearing PLANT_FACTOR times it.
FLOOR = 1e-9
RESOLUTION_FACTOR = 100.0
PLANT_FACTOR = 10.0
TAU = 2.0 * np.pi


def unit(draw: Callable[..., Any], size: int) -> np.ndarray:
    """A unit vector of ``size`` components, each 0.1 to 1 in size with its own sign, normalised: no component near an
    axis plane, none that underflows."""
    magnitudes = draw(st.lists(st.floats(0.1, 1.0), min_size=size, max_size=size))
    signs = draw(st.lists(st.sampled_from((-1.0, 1.0)), min_size=size, max_size=size))
    held = np.asarray(magnitudes, dtype=np.float64) * np.asarray(signs)
    return held / np.linalg.norm(held)


def rotation_of(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of a unit quaternion ``(w, x, y, z)``, body to world, in float64 (``mime.core.quaternion``'s
    formula)."""
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quaternion_of(rotation: np.ndarray) -> np.ndarray:
    """A unit quaternion ``(w, x, y, z)`` of a proper rotation matrix, from the largest of its four squares
    (Shepperd's method)."""
    r = np.asarray(rotation, dtype=np.float64)
    squares = np.array([1 + r[0, 0] + r[1, 1] + r[2, 2], 1 + r[0, 0] - r[1, 1] - r[2, 2],
                        1 - r[0, 0] + r[1, 1] - r[2, 2], 1 - r[0, 0] - r[1, 1] + r[2, 2]])
    k = int(np.argmax(squares))
    s = 0.5 * np.sqrt(squares[k])
    w_x, w_y, w_z = r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]
    x_y, x_z, y_z = r[0, 1] + r[1, 0], r[0, 2] + r[2, 0], r[1, 2] + r[2, 1]
    q = [np.array([4 * s * s, w_x, w_y, w_z]), np.array([w_x, 4 * s * s, x_y, x_z]),
         np.array([w_y, x_y, 4 * s * s, y_z]), np.array([w_z, x_z, y_z, 4 * s * s])][k] / (4 * s)
    return q / np.linalg.norm(q)


def product(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The Hamilton product ``p q`` of quaternions along their last axis, in float64 (``mime.core.quaternion``'s
    takes one pair)."""
    w1, x1, y1, z1 = np.moveaxis(np.asarray(p, dtype=np.float64), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def attitude_angle(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The angle between unit quaternions ``p`` and ``q`` (radians), accurate near zero."""
    relative = product(np.asarray(p, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0]), q)
    return 2.0 * np.arctan2(np.linalg.norm(relative[..., 1:], axis=-1), np.abs(relative[..., 0]))


@cache
def fl9_tensor() -> np.ndarray:
    """FL-9's inertia tensor about its centre of mass in its frame, as its parts give it (the shell at its mean
    density, through the part resolver)."""
    from profile_cases import parts_screw

    built = parts_screw("exploratory").build()
    return np.asarray(built.graph.get_node("umr").constants()["inertia"], dtype=np.float64).reshape(3, 3)


@st.composite
def principal_moments(draw: Callable[..., Any]) -> np.ndarray:
    """Three principal moments at FL-9's scale, in a drawn order: the least 0.5 to 1.5 e-10 kg m^2, each next 10 to
    60 % above the one before, and the largest at most 0.975 of the sum of the other two (the triangle inequality, with
    room). A body with distinct moments: equal ones give a tensor with no products about any axes, which no plant
    could move."""
    least = draw(st.floats(0.5e-10, 1.5e-10))
    middle = least * (1.0 + draw(st.floats(0.1, 0.6)))
    largest = min(middle * (1.0 + draw(st.floats(0.1, 0.6))), 0.975 * (least + middle))
    order = draw(st.permutations((0, 1, 2)))
    return np.asarray((least, middle, largest), dtype=np.float64)[list(order)]


@st.composite
def principal_turn(draw: Callable[..., Any]) -> np.ndarray:
    """The turn from the body's axes to its principal axes, a unit quaternion: 20 to 70 degrees about a drawn axis.
    No turn there is a signed permutation of the axes (those turn by 0, 90, 120 or 180 degrees), which would leave a
    body of distinct moments with no products for a plant to move."""
    angle = np.radians(draw(st.floats(20.0, 70.0)))
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * unit(draw, 3)])


def body_resistance(draw: Callable[..., Any]) -> np.ndarray:
    """A symmetric positive definite resistance in the body's frame at FL-9's scales (4e-5 N s/m, 8e-10 N m s), its
    translation-rotation block a drawn share of the geometric mean of the two, up to 0.3."""
    translation = np.diag(4e-5 * np.array([1.0, 2.0, 2.0]))
    rotation = np.diag(8e-10 * np.array([1.0, 3.0, 3.0]))
    entries = np.asarray(draw(st.lists(st.floats(-1.0, 1.0), min_size=9, max_size=9)), dtype=np.float64)
    coupling = 0.3 * np.sqrt(4e-5 * 8e-10) * entries.reshape(3, 3) / 3.0
    return np.block([[translation, coupling], [coupling.T, rotation]])


@st.composite
def draws(draw: Callable[..., Any], operating: Mapping[str, float], screw: bool) -> dict[str, np.ndarray]:
    """One body and its run: the tensor with products and its principal frame ``Q`` (``J = Q D Q^T``), the starting
    pose and rate, and the body's moment and resistance in its own frame, with the same turned into the principal
    frame."""
    if screw:
        tensor = fl9_tensor()
        moments, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0.0:
            axes[:, 2] = -axes[:, 2]
    else:
        moments, axes = draw(principal_moments()), rotation_of(draw(principal_turn()))
        tensor = axes @ np.diag(moments) @ axes.T
    start = unit(draw, 4)
    turn = quaternion_of(axes)
    driven = operating["field"] > 0.0
    moment = 1.68e-3 * unit(draw, 3) if driven else np.zeros(3)
    resistance = body_resistance(draw) if driven else np.zeros((6, 6))
    both = np.zeros((6, 6))
    both[:3, :3] = both[3:, 3:] = axes
    return {"tensor": tensor, "moments": moments, "turn": turn, "start": start,
            "principal_start": product(start, turn),
            "twist": np.concatenate([1e-3 * unit(draw, 3), operating["rate"] * unit(draw, 3)]),
            "moment": moment, "principal_moment": axes.T @ moment, "resistance": resistance,
            "principal_resistance": both.T @ resistance @ both}


def forcing(pose: jax.Array, t: jax.Array, moment: jax.Array, resistance: jax.Array,
            operating: Mapping[str, float]) -> dict[str, jax.Array]:
    """The field's torque on the body's moment (the field turning about the world's x) and its resistance turned into
    the world, at the step's start."""
    rotation = quaternion.rotation_matrix(pose[3:])
    phase = TAU * operating["frequency"] * t
    field = operating["field"] * jnp.stack([jnp.zeros_like(phase), jnp.cos(phase), jnp.sin(phase)])
    turn = jnp.zeros((6, 6), dtype=pose.dtype).at[:3, :3].set(rotation).at[3:, 3:].set(rotation)
    return {"applied_wrench": jnp.concatenate([jnp.zeros(3, dtype=pose.dtype), jnp.cross(rotation @ moment, field)]),
            "resistance": turn @ resistance @ turn.T}


class WorldInertiaTransposed(RigidInertialTensorNode):
    """A plant: the world inertia formed ``Q^T J Q``, the body's rotation taken the wrong way."""

    def stepped(self, p: Mapping[str, Any], pose: jax.Array, twist: jax.Array, given: Mapping[str, jax.Array],
                dt: float) -> tuple[jax.Array, jax.Array, jax.Array]:
        backwards = jnp.concatenate([pose[:3], quaternion.conjugate(pose[3:])])
        rotation = quaternion.rotation_matrix(pose[3:])
        mass_matrix, _, _ = super().stepped(p, backwards, twist, given, dt)
        omega = twist[3:]
        inertia = mass_matrix[3:, 3:]
        none = jnp.zeros_like(twist)
        load = (given.get("applied_wrench", none) + given.get("background_load", none)
                + jnp.concatenate([jnp.zeros_like(omega), -jnp.cross(omega, inertia @ omega)]))
        resistance = given.get("resistance", jnp.zeros((6, 6), dtype=twist.dtype))
        stepping = self.stepping_mass(p, mass_matrix, rotation, omega, dt)
        return mass_matrix, stepping / dt + resistance, self.velocities(stepping, resistance, twist, load, dt)


class JacobianTransposed(RigidInertialTensorNode):
    """A plant: the gyroscopic Jacobian transposed in the step's matrix."""

    def gyroscopic_jacobian(self, rotation: jax.Array, moments: jax.Array, omega: jax.Array) -> jax.Array:
        return super().gyroscopic_jacobian(rotation, moments, omega).T


def flipped(tensor: np.ndarray, both: bool) -> np.ndarray:
    """``tensor`` with its largest product's sign flipped below the diagonal, or in both places."""
    held = tensor.copy()
    a, b = max(((0, 1), (0, 2), (1, 2)), key=lambda ij: abs(held[ij]))
    held[b, a] = -held[b, a]
    if both:
        held[a, b] = -held[a, b]
    return held


@cache
def runner(node_class: type[RigidInertialNode], dt: float, driven: bool) -> Callable[..., jax.Array]:
    """The compiled run of ``node_class`` for :data:`STEPS` steps of ``dt``, its quaternion after each: the node is
    built once, and the tensor, the start and the forcing are the run's arguments."""
    nine = node_class is not RigidInertialMidpointNode
    node = node_class("probe", dt, mass=MASS, inertia=tuple(np.eye(3).reshape(9) if nine else np.ones(3)),
                      initial_pose=(0.0,) * 7, initial_twist=(0.0,) * 6)
    operating = DRIVEN if driven else FREE

    def run(inertia: jax.Array, pose: jax.Array, twist: jax.Array, moment: jax.Array,
            resistance: jax.Array) -> jax.Array:
        p = {"mass": jnp.asarray(MASS, dtype=pose.dtype), "inertia": inertia}

        def one(state: dict[str, jax.Array], k: jax.Array) -> tuple[dict[str, jax.Array], jax.Array]:
            given = forcing(state["pose"], k * dt, moment, resistance, operating) if driven else {}
            new = node.advance(p, state, given, dt)
            return {"pose": new["pose"], "twist": new["twist"]}, new["pose"][3:]

        _, found = jax.lax.scan(one, {"pose": pose, "twist": twist}, jnp.arange(STEPS, dtype=pose.dtype))
        return found

    return jax.jit(run)


def principal_attitude(drawn: Mapping[str, np.ndarray], driven: bool, moments: np.ndarray,
                       start: np.ndarray) -> np.ndarray:
    """The principal body's quaternion after each step, on 0.3.0's node, from ``moments`` and ``start``."""
    dt = (DRIVEN if driven else FREE)["dt"]
    return np.asarray(runner(RigidInertialMidpointNode, dt, driven)(
        jnp.asarray(moments), jnp.asarray(np.r_[np.zeros(3), start]), jnp.asarray(drawn["twist"]),
        jnp.asarray(drawn["principal_moment"]), jnp.asarray(drawn["principal_resistance"])))


def one_ulp_up(values: np.ndarray) -> np.ndarray:
    """``values`` with each entry one unit in the last place further from zero."""
    return np.nextafter(values, np.where(values >= 0.0, np.inf, -np.inf))


def resolution(drawn: Mapping[str, np.ndarray], driven: bool) -> float:
    """The draw's own resolution: the most the principal body's attitude moves over the run under a one-ulp change of
    its moments, and under a one-ulp change of its start (radians; infinite where a run is not finite). A comparison of
    two arms that differ by rounding cannot be finer than this."""
    reference = principal_attitude(drawn, driven, drawn["moments"], drawn["principal_start"])
    moved = (principal_attitude(drawn, driven, one_ulp_up(drawn["moments"]), drawn["principal_start"]),
             principal_attitude(drawn, driven, drawn["moments"], one_ulp_up(drawn["principal_start"])))
    spread = [attitude_angle(reference, other) for other in moved]
    return float(max(np.max(angle) for angle in spread)) if all(np.all(np.isfinite(a)) for a in spread) \
        else float("inf")


def apart(drawn: Mapping[str, np.ndarray], node_class: type[RigidInertialNode], tensor: np.ndarray,
          driven: bool) -> float:
    """The most the body's attitude, turned back by ``Q``, stands from the principal body's over the run (radians;
    infinite where either is not finite)."""
    dt = (DRIVEN if driven else FREE)["dt"]
    as_given = runner(node_class, dt, driven)(
        jnp.asarray(tensor.reshape(9)), jnp.asarray(np.r_[np.zeros(3), drawn["start"]]), jnp.asarray(drawn["twist"]),
        jnp.asarray(drawn["moment"]), jnp.asarray(drawn["resistance"]))
    principal = principal_attitude(drawn, driven, drawn["moments"], drawn["principal_start"])
    angle = attitude_angle(product(np.asarray(as_given), drawn["turn"]), principal)
    return float(np.max(angle)) if np.all(np.isfinite(angle)) else float("inf")


def judged(drawn: Mapping[str, np.ndarray], driven: bool) -> dict[str, float]:
    """The draw's resolution, the unplanted arm's distance from the principal body and each plant's."""
    tensor = drawn["tensor"]
    return {"resolution": resolution(drawn, driven),
            "unplanted": apart(drawn, RigidInertialTensorNode, tensor, driven),
            "sign in one place": apart(drawn, RigidInertialTensorNode, flipped(tensor, False), driven),
            "sign in both places": apart(drawn, RigidInertialTensorNode, flipped(tensor, True), driven),
            "world inertia Q^T J Q": apart(drawn, WorldInertiaTransposed, tensor, driven),
            "Jacobian transposed": apart(drawn, JacobianTransposed, tensor, driven)}


def floor_of(found: dict[str, float]) -> float:
    """The draw's floor from its judged distances (taking ``resolution`` out of them): the larger of :data:`FLOOR` and
    :data:`RESOLUTION_FACTOR` times the resolution."""
    return max(FLOOR, RESOLUTION_FACTOR * found.pop("resolution"))


SETS = {"drawn tensors, driven": (DRIVEN, False), "drawn tensors, free": (FREE, False),
        "FL-9's tensor, driven": (DRIVEN, True), "FL-9's tensor, free": (FREE, True)}


@pytest.fixture
def float64():
    with jax.enable_x64(True):
        yield


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize("name", SETS)
@settings(deadline=None)
@given(data=st.data())
def test_a_body_with_products_moves_as_its_principal_body_and_every_plant_is_caught(name, data):
    operating, screw = SETS[name]
    drawn = data.draw(draws(operating, screw))
    found = judged(drawn, operating is DRIVEN)
    floor = floor_of(found)
    assert np.isfinite(floor), "the principal body's own run is not finite: the draw has no resolution"
    assert found.pop("unplanted") <= floor, floor
    missed = {plant: distance for plant, distance in found.items() if not distance > PLANT_FACTOR * floor}
    assert not missed, (floor, missed)


#: The gradient's run: FL-9 driven from a fixed start for a fifth of a second at script A's step.
GRADIENT_STEPS = 400


def fl9_driven_final_attitude(steps: int) -> Callable[[jax.Array], jax.Array]:
    """The final quaternion of FL-9 driven by the field against a coupled resistance from a fixed start, as a function
    of its inertia tensor (3 x 3, body frame), through 0.4.0's node."""
    dt = DRIVEN["dt"]
    node = RigidInertialTensorNode("probe", dt, mass=MASS, inertia=tuple(fl9_tensor().reshape(9)),
                                   initial_pose=(0.0,) * 7, initial_twist=(0.0,) * 6)
    draws = np.random.default_rng(3)
    start = draws.normal(size=4)
    pose = jnp.asarray(np.r_[np.zeros(3), start / np.linalg.norm(start)])
    twist = jnp.asarray(np.r_[np.zeros(3), 5.0 * draws.normal(size=3)])
    moment = jnp.asarray((0.0, 1.68e-3, 0.0))
    translation, rotation = np.diag(4e-5 * np.array([1.0, 2.0, 2.0])), np.diag(8e-10 * np.array([1.0, 3.0, 3.0]))
    coupling = 0.1 * np.sqrt(4e-5 * 8e-10) * draws.normal(size=(3, 3))
    resistance = jnp.asarray(np.block([[translation, coupling], [coupling.T, rotation]]))

    def final(tensor: jax.Array) -> jax.Array:
        p = {"mass": jnp.asarray(MASS), "inertia": jnp.reshape(tensor, (9,))}

        def one(state: dict[str, jax.Array], k: jax.Array) -> tuple[dict[str, jax.Array], None]:
            new = node.advance(p, state, forcing(state["pose"], k * dt, moment, resistance, DRIVEN), dt)
            return {"pose": new["pose"], "twist": new["twist"]}, None

        state, _ = jax.lax.scan(one, {"pose": pose, "twist": twist}, jnp.arange(steps, dtype=pose.dtype))
        return state["pose"][3:]

    return final


def symmetric(entries: jax.Array) -> jax.Array:
    """The symmetric 3 x 3 of its six entries ``(xx, yy, zz, xy, xz, yz)``, each product in both places."""
    xx, yy, zz, xy, xz, yz = entries
    return jnp.stack([jnp.stack([xx, xy, xz]), jnp.stack([xy, yy, yz]), jnp.stack([xz, yz, zz])])


def equal_transverse_moments() -> np.ndarray:
    """FL-9's tensor with its products removed and its two transverse moments set exactly equal, as six entries: the
    axisymmetric body, at which an eigen-decomposition has no derivative."""
    tensor = fl9_tensor()
    transverse = 0.5 * (tensor[1, 1] + tensor[2, 2])
    return np.array([tensor[0, 0], transverse, transverse, 0.0, 0.0, 0.0])


def central_differences(function: Callable[[jax.Array], jax.Array], at: np.ndarray, step: float) -> np.ndarray:
    compiled = jax.jit(function)
    return np.stack([(np.asarray(compiled(jnp.asarray(at + step * e))) - np.asarray(compiled(jnp.asarray(at - step * e))))
                     / (2.0 * step) for e in np.eye(len(at))], axis=1)


def fl9_entries() -> np.ndarray:
    """FL-9's own tensor, with its product, as six entries."""
    tensor = fl9_tensor()
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])


@pytest.mark.usefixtures("float64")
@pytest.mark.parametrize("point", ["equal transverse moments", "FL-9's tensor"])
def test_the_derivative_through_exactly_equal_moments_is_finite_and_agrees_with_central_differences(point):
    """Reverse mode through :data:`GRADIENT_STEPS` steps against central differences at two steps, a millionth and a
    hundred-thousandth of the transverse moment: the derivative within the differences' own resolution, how far the
    two differ; at the axisymmetric body and at FL-9's own tensor."""
    final = fl9_driven_final_attitude(GRADIENT_STEPS)
    at = equal_transverse_moments() if point == "equal transverse moments" else fl9_entries()

    def of_entries(entries: jax.Array) -> jax.Array:
        return final(symmetric(entries))

    found = np.asarray(jax.jit(jax.jacrev(of_entries))(jnp.asarray(at)))
    fine, coarse = (central_differences(of_entries, at, share * at[1]) for share in (1e-6, 1e-5))
    resolution = float(np.linalg.norm(fine - coarse))
    error = float(np.linalg.norm(found - fine))
    print(f"\nderivative's distance from central differences {error / np.linalg.norm(fine):.2e} of it, the "
          f"differences' resolution {resolution / np.linalg.norm(fine):.2e}")
    assert np.all(np.isfinite(found)) and np.linalg.norm(fine) > 0.0
    assert error <= resolution


@pytest.mark.usefixtures("float64")
def test_control_the_eigen_route_has_no_derivative_at_equal_moments():
    """The same body carried as principal moments and axes, ``J = Q diag(D) Q^T`` from an eigen-decomposition: the
    decomposition's derivative divides by the moments' differences, so reverse mode through it is not finite where two
    are equal, on every run (the point is fixed)."""
    final = fl9_driven_final_attitude(GRADIENT_STEPS)

    def by_eigen(entries: jax.Array) -> jax.Array:
        moments, axes = jnp.linalg.eigh(symmetric(entries))
        return final(axes @ jnp.diag(moments) @ axes.T)

    derivative = jax.jit(jax.jacrev(by_eigen))
    assert not np.all(np.isfinite(np.asarray(derivative(jnp.asarray(equal_transverse_moments())))))
    assert np.all(np.isfinite(np.asarray(derivative(jnp.asarray(fl9_entries()))))), \
        "at FL-9's own, distinct moments the eigen route has a derivative: its failure above is the equal moments'"


def torque_free_invariants(tensor: np.ndarray, spin: np.ndarray, dt: float, seconds: float) -> tuple[np.ndarray, ...]:
    """The torque-free body's kinetic energy and its angular momentum's magnitude after each step, as ratios of their
    starts, through 0.4.0's node from the identity pose, ``spin`` its angular velocity in the world."""
    node = RigidInertialTensorNode("probe", dt, mass=MASS, inertia=tuple(tensor.reshape(9)),
                                   initial_pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                                   initial_twist=(0.0, 0.0, 0.0, *spin))
    held = jnp.asarray(tensor)

    def step(state: Any, _: None) -> tuple[Any, tuple[jax.Array, jax.Array]]:
        state = node.update(state, {}, dt)
        turn = quaternion.rotation_matrix(state["pose"][3:])
        in_body = turn.T @ state["twist"][3:]
        return state, (0.5 * in_body @ held @ in_body, jnp.linalg.norm(held @ in_body))

    _, (energy, momentum) = jax.lax.scan(step, node.initial_state(), None, length=round(seconds / dt))
    start = tensor @ spin
    return (np.asarray(energy) / (0.5 * float(spin @ start)), np.asarray(momentum) / float(np.linalg.norm(start)))


@pytest.mark.usefixtures("float64")
def test_a_torque_free_body_with_products_keeps_its_energy_and_momentum_to_second_order_with_no_drift():
    """A second check, not a control: the moments (1.0, 1.2, 0.8) e-10 of the existing oracle about axes turned 40
    degrees from the body's, spinning at 206 rad/s about no principal axis. Each invariant's largest error over a second
    is within 3e-4 at 1 ms (as for the principal body), falls by four, within a fifth, when the step halves, and is no
    larger over the second half second than over the first."""
    axes = rotation_of(np.concatenate([[np.cos(np.radians(20.0))], np.sin(np.radians(20.0)) *
                                       np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])]))
    tensor = axes @ np.diag([1.0e-10, 1.2e-10, 0.8e-10]) @ axes.T
    spin = axes @ np.array([150.0, 100.0, 100.0])
    errors = {}
    for dt in (1e-3, 5e-4):
        for name, ratio in zip(("energy", "momentum"), torque_free_invariants(tensor, spin, dt, 1.0), strict=True):
            half = len(ratio) // 2
            errors[name, dt] = (float(np.max(np.abs(ratio - 1.0))), float(np.max(np.abs(ratio[:half] - 1.0))),
                                float(np.max(np.abs(ratio[half:] - 1.0))))
    print("\n" + "; ".join(f"{name} at {dt * 1e3:g} ms: {held[0]:.2e}" for (name, dt), held in errors.items()))
    for name in ("energy", "momentum"):
        whole, first, second = errors[name, 1e-3]
        assert whole < 3e-4 and second <= first * 1.0001
        assert 3.2 < whole / errors[name, 5e-4][0] < 4.8


def drawn_rotations_and_rates(count: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotation matrices and world rates of a physical scale (rad/s), fixed draws."""
    draws = np.random.default_rng(20261107)
    turns = draws.normal(size=(count, 4))
    return (np.stack([rotation_of(q / np.linalg.norm(q)) for q in turns]), draws.uniform(-300.0, 300.0, size=(count, 3)))


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_the_split_jacobian_is_the_derivative_of_the_gyroscopic_term_for_a_full_tensor(x64):
    draws = np.random.default_rng(7)
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        worst = 0.0
        for rotation, omega in zip(*drawn_rotations_and_rates(20), strict=True):
            moments = draws.uniform(0.5e-10, 1.5e-10, size=3)
            axes = np.linalg.qr(draws.normal(size=(3, 3)))[0]
            tensor = jnp.asarray(axes @ np.diag(moments) @ axes.T, dtype=dtype)
            turn, rate = jnp.asarray(rotation, dtype=dtype), jnp.asarray(omega, dtype=dtype)
            inertia = turn @ tensor @ turn.T
            expected = jax.jacfwd(lambda w, inertia=inertia: -jnp.cross(w, inertia @ w))(rate)
            found = split_gyroscopic_jacobian(turn, tensor, rate)
            worst = max(worst, float(jnp.max(jnp.abs(found - expected)) / jnp.max(jnp.abs(expected))))
    print(f"\nlargest difference from the derivative, relative to its largest entry: {worst:.1e}")
    assert worst < (1e-14 if x64 else 1e-5)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_for_a_tensor_with_no_products_the_split_jacobian_is_0_3_0s_bit_for_bit(x64):
    """On the node's functions alone, at 200 drawn poses and rates, three unequal moments and three equal ones: the
    products' part is exactly zero and the sum exactly the diagonal's. (Per compiled program, bit for bit is gate 1's,
    measured on the released programs.)"""
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        for moments in ((1.0e-10, 1.2e-10, 0.8e-10), (1.7e-10,) * 3):
            held = jnp.asarray(moments, dtype=dtype)
            for rotation, omega in zip(*drawn_rotations_and_rates(200), strict=True):
                turn, rate = jnp.asarray(rotation, dtype=dtype), jnp.asarray(omega, dtype=dtype)
                split = split_gyroscopic_jacobian(turn, jnp.diag(held), rate)
                assert np.asarray(split).tobytes() == np.asarray(gyroscopic_jacobian(turn, held, rate)).tobytes()
