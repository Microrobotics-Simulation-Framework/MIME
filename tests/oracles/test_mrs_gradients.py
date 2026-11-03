"""``Drag.MRS``'s derivatives in free space: DEFINITION 5.3's three checks, the differentiation proposals of the design
(the forward mode reuses the factorisation; the linear-solve rule needs no derivative through it; the matrix is
symmetric, so the adjoint solves with it again; the reverse mode is checkpointed), and their controls.

**Check 3, the probe protocol, written before the first run.** Float64. A trainable's derivative is compared with
a central difference by rebuilding: the experiment written again with the setting at ``theta (1 +- h)``, planned,
built and run, ``h = 1e-4``, and again at ``h / 2``. The difference's truncation is estimated as ``|D(h) - D(h/2)|``
and its rounding as ``64 eps |loss| / (h theta)``; the derivative passes within ``1e-6 |D(h/2)|`` plus four times
the two. A pose is moved on the manifold, ``(p + dx, exp(dtheta) q)``, and its derivative read in the six components
``(dx, dtheta)``, by the same difference at ``h = 1e-6`` (m and rad).

**Check 1, the forward response per input.** The node's one input is the pose. Moved by it, the resistance is the one
a node constructed fresh gives with the body's part placed there instead: turned, the part turned by the same
rotation; moved, the part moved, its resistance then taken about the moved centre. Control: a node that reads the
pose's position and keeps the identity's orientation, as a node baked at its starting pose would, fails.

**Check 2, a difference by rebuilding, per trainable**, on two experiments: a sphere pulled through still water by a
constant force and torque (the gradient-pulled body the generality rule names), its position and orientation after
five steps, against the sphere's radius and the water's viscosity; and de Jongh's FL-9 turned by a torque about its
axis (a swimmer), its displacement along the axis after five steps, against its wavenumber. Control: a node whose blob
was fixed when it was built, from the radius it was built with, rather than from the samples it makes at each step (a
tied quantity left out of the tie): the derivative through ``params`` misses the blob's share and fails.

**The proposals.** The forward mode's program factorises once (one ``lu``) for the value and its tangent. The
reverse mode keeps no ``3N x 3N`` array between the forward and backward passes, only the step's inputs: its
checkpoint is what does it, since without it the factors are kept. The point-force matrix is symmetric (measured in
``test_port_mrs.py``), and the adjoint is the solve's own transpose. Controls: an adjoint that takes the 0.2 line's
traction matrix for symmetric fails the reverse mode's check, while its forward mode passes; a derivative through the
factorisation (``lu_factor`` and ``lu_solve`` differentiated) agrees with the linear-solve rule to rounding, so no
check of values can tell them apart: what it costs is the difference, and it keeps the factors' derivatives."""

import contextlib
import dataclasses
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import pytest
from mrs_cases import BALL, PILL, drag_node, free_space, mrs, point_forces, resistance_at
from rigid_oracles import ConstantWrench
from script_a import fl9

import mime.interactions.hydrodynamic as hydrodynamic
import mime.nodes.hydro.mrs._node as mrs_node
import mime.nodes.hydro.mrs._solve as mrs_solve
import mime.scene._sampling as scene_sampling
from mime.build._maddening import node_state, parameter_leaves
from mime.core import quaternion
from mime.nodes.hydro.mrs import MRSDragNode, blob_squared
from mime.scene import Sphere
from mime.units import Quantity, kg, kg_m2, mm, mPa_s

H, POSE_STEP, STEPS = 1e-4, 1e-6, 5
RELATIVE = 1e-6
SPACING = mm(0.6)
#: FL-9 sampled coarsely: its derivatives are checked here, not its accuracy.
COARSE_SCREW = mm(1.5)


def difference(loss, theta: float, h: float = H) -> tuple[float, float]:
    """The central difference at ``h`` and ``h / 2`` (relative steps of ``theta``)."""
    def at(step: float) -> float:
        return (loss(theta * (1 + step)) - loss(theta * (1 - step))) / (2 * step * theta)

    return at(h), at(h / 2)


def agrees(found: float, coarse: float, fine: float, value: float, theta: float, h: float = H) -> bool:
    rounding = 64 * np.finfo(float).eps * abs(value) / (h * abs(theta))
    return bool(abs(found - fine) <= RELATIVE * abs(fine) + 4 * (abs(coarse - fine) + rounding))


# -- check 1 -----------------------------------------------------------------------------------------------------

def turned(axis_angle) -> np.ndarray:
    return np.asarray(quaternion.exponential(jnp.asarray(axis_angle, dtype=jnp.float64)))


def shifted(resistance: np.ndarray, d: np.ndarray) -> np.ndarray:
    """``resistance`` about a point ``d`` from its own: ``T^T R T``, ``T = [[I, [d]x], [0, I]]``... with ``d`` from
    the new point to the old."""
    cross = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    move = np.eye(6)
    move[:3, 3:] = cross
    return move.T @ resistance @ move


def fresh(node: MRSDragNode, rotation: np.ndarray, offset: np.ndarray, cls: type = MRSDragNode) -> MRSDragNode:
    """``node`` constructed again with its part turned by ``rotation`` and moved by ``offset``."""
    held = dict(node.constants())
    part = np.asarray(held.pop("part_rotation")).reshape(3, 3)
    centre = np.asarray(held.pop("part_centre"))
    return cls(node.name, node.delta_t, part_rotation=tuple((rotation @ part).ravel()),
               part_centre=tuple(rotation @ centre + offset), **held)


def at(node: MRSDragNode, pose) -> np.ndarray:
    out = node.update(node.initial_state(), {"pose": jnp.asarray(pose, dtype=jnp.float64)}, 1e-3)
    return np.asarray(out["resistance"])


class _IdentityOrientationNode(MRSDragNode):
    """Reads the pose's position and keeps the identity's orientation, as a node baked at that pose would."""

    def advance(self, p, state, boundary_inputs, dt):
        pose = boundary_inputs["pose"]
        held = jnp.concatenate([pose[:3], jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=pose.dtype)])
        return super().advance(p, state, {**boundary_inputs, "pose": held}, dt)


def response_misses(cls: type = MRSDragNode) -> list[float]:
    built = free_space("check_1", {"screw": fl9()}, mrs(COARSE_SCREW)).build()
    node = built.graph.get_node("drag:screw")
    node = fresh(node, np.eye(3), np.zeros(3), cls)
    q = turned((0.4, -0.7, 0.2))
    rotation = np.asarray(quaternion.rotation_matrix(jnp.asarray(q)))
    offset = np.array([2e-3, -1e-3, 3e-3])
    identity = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    turned_input = at(node, (0.0, 0.0, 0.0, *q))
    turned_fresh = at(fresh(node, rotation, np.zeros(3), cls), identity)
    moved_input = at(node, (*offset, 1.0, 0.0, 0.0, 0.0))
    moved_fresh = shifted(at(fresh(node, np.eye(3), offset, cls), identity), offset)
    scale = np.sqrt(np.outer(np.diag(turned_fresh), np.diag(turned_fresh)))
    return [float(np.abs((turned_input - turned_fresh) / scale).max()),
            float(np.abs((moved_input - moved_fresh) / scale).max())]


def test_check_1_the_pose_moves_the_resistance_as_a_part_placed_there_at_construction():
    misses = response_misses()
    print(f"\nturned, moved: {misses}")
    assert max(misses) <= 1e-11


def test_control_a_node_that_keeps_the_identitys_orientation_fails_check_1():
    misses = response_misses(_IdentityOrientationNode)
    print(f"\nturned, moved: {misses}")
    assert misses[0] > 1e-3


# -- check 2 -----------------------------------------------------------------------------------------------------

#: The pulled sphere: light, so that its step's mass term (m / dt) is of its resistance's order and the motion feels
#: the resistance at once; pulled along a direction and turned about another.
PULL = (2e-6, -1e-6, 3e-6, 4e-14, 2e-14, -3e-14)


def pulled(radius: float, viscosity: float):
    experiment = free_space("pulled", {"ball": Sphere(radius=mm(radius * 1e3))}, mrs(SPACING),
                            viscosity=mPa_s(viscosity * 1e3), mass=kg(2e-8), inertia=kg_m2((2e-17, 2e-17, 2e-17)))
    experiment.attach(ConstantWrench(value=PULL), body="ball")
    return experiment.build()


def swimmer(wavenumber: float):
    experiment = free_space("swimmer", {"swimmer": fl9(nu=wavenumber)}, mrs(COARSE_SCREW), mass=kg(1e-7),
                            inertia=kg_m2((1e-15, 1e-15, 1e-15)))
    experiment.attach(ConstantWrench(value=(0.0, 0.0, 0.0, 3e-11, 0.0, 0.0)), body="swimmer")
    return experiment.build()


WEIGHTS = np.array([0.3, -0.5, 0.7]), np.array([0.2, 0.9, -0.4])


def pulled_loss(state) -> jax.Array:
    """The ball's position (in mm) and its quaternion's vector part, each against fixed weights."""
    pose = state["ball"]["pose"]
    return 1e3 * jnp.dot(pose[:3], WEIGHTS[0]) + jnp.dot(pose[4:], WEIGHTS[1])


def swimmer_loss(state) -> jax.Array:
    """The screw's displacement from the origin, where it starts unturned, along its axis (x in its frame), in mm."""
    pose = state["swimmer"]["pose"]
    return 1e3 * jnp.dot(pose[:3], quaternion.rotate(pose[3:], jnp.asarray((1.0, 0.0, 0.0))))


def run(built, params=None) -> dict:
    graph = built.graph
    graph.reset_state()
    final = graph.run_scan(STEPS, params=params)
    return final


def through_params(built, node: str, leaf: str, loss) -> float:
    graph = built.graph
    base = graph.params

    def of(value):
        held = jax.tree.map(lambda x: x, base)
        held["nodes"][node][leaf] = value
        return loss(run(built, held))

    found = float(jax.grad(of)(jnp.asarray(base["nodes"][node][leaf])))
    graph.reset_state()
    return found


@pytest.mark.parametrize(("trainable", "leaf", "theta"), [("radius", "shape_radius", 1e-3),
                                                          ("viscosity", "viscosity", 1e-3)])
def test_check_2_the_pulled_spheres_derivative_equals_the_difference_by_rebuilding(trainable, leaf, theta):
    def rebuilt(value: float) -> float:
        held = pulled(value, 1e-3) if trainable == "radius" else pulled(1e-3, value)
        return float(pulled_loss(run(held)))

    built = pulled(1e-3, 1e-3)
    found = through_params(built, "drag:ball", leaf, pulled_loss)
    coarse, fine = difference(rebuilt, theta)
    print(f"\n{trainable}: through params {found:.10e}; rebuilt, h {coarse:.10e}, h/2 {fine:.10e}")
    assert agrees(found, coarse, fine, rebuilt(theta), theta)


def test_check_2_the_swimmers_derivative_in_its_wavenumber_equals_the_difference_by_rebuilding():
    """Rebuilt with the sampling's choice held, as the counts are (the check below)."""
    built = swimmer(2.33)
    first = drag_node(built, "swimmer")

    def rebuilt(value: float) -> float:
        with held_choice(first):
            return float(swimmer_loss(run(swimmer(value))))

    value = float(swimmer_loss(run(built)))
    found = through_params(built, "drag:swimmer", "shape_wavenumber", swimmer_loss)
    coarse, fine = difference(rebuilt, 2.33)
    print(f"\nwavenumber: through params {found:.10e}; rebuilt, h {coarse:.10e}, h/2 {fine:.10e}; swim "
          f"{value:.6e} mm in {STEPS} steps")
    assert value != 0.0 and agrees(found, coarse, fine, value, 2.33)


class _BakedBlobNode(MRSDragNode):
    """Its blob fixed at construction, from the samples of the radius it was built with: on a separated quadrature in
    its body (whose derivative rule is traced after the step), on point forces in the step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        built = self.samples({key: jnp.asarray(value) for key, value in self.constants().items()
                              if key.startswith(("shape_", "part_")) or key == "viscosity"})
        self._baked = float(blob_squared(built.weights, self.constants()["blob"]))

    def body(self, theta):
        return dataclasses.replace(super().body(theta), eps_squared=jnp.asarray(self._baked, dtype=theta["blob"].dtype))

    def advance(self, p, state, boundary_inputs, dt):
        from mime.nodes.hydro.mrs import _node

        held = _node.blob_squared
        _node.blob_squared = lambda weights, blob: jnp.asarray(self._baked, dtype=weights.dtype)
        try:
            return super().advance(p, state, boundary_inputs, dt)
        finally:
            _node.blob_squared = held


def test_control_a_blob_fixed_at_construction_fails_check_2(monkeypatch):
    monkeypatch.setattr(hydrodynamic, "MRSDragNode", _BakedBlobNode)

    def rebuilt(value: float) -> float:
        return float(pulled_loss(run(pulled(value, 1e-3))))

    found = through_params(pulled(1e-3, 1e-3), "drag:ball", "shape_radius", pulled_loss)
    coarse, fine = difference(rebuilt, 1e-3)
    print(f"\nthe blob baked: through params {found:.10e}; rebuilt {fine:.10e}")
    assert not agrees(found, coarse, fine, rebuilt(1e-3), 1e-3)


# -- check 2 on every trainable leaf of the node -------------------------------------------------------------------

#: Each kind the rung serves, with the spacing it is sampled at here.
LEAF_KINDS = {"ball": (BALL, SPACING), "pill": (PILL, SPACING), "screw": (fl9(), COARSE_SCREW)}
#: Every trainable leaf of the node on each kind, as the node declares them; the test below takes each.
LEAVES = {"ball": ("blob", "shape_radius", "viscosity"),
          "pill": ("blob", "shape_height", "shape_radius", "viscosity"),
          "screw": ("blob", "shape_length", "shape_modulation", "shape_radius", "shape_wavenumber", "viscosity")}


@contextlib.contextmanager
def held_choice(node: MRSDragNode):
    """Every sampling made inside it is ``node``'s choice: what a build at a moved setting takes when its
    discretisation is held, as its counts are."""
    held = node._sampling  # noqa: SLF001 - the node's own choice, as the rung made it
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scene_sampling, "_MADE", {})
        patch.setattr(scene_sampling, "_quasi_uniform", lambda *args: held)
        yield


def written_with(body: str, leaf: str, value: float):
    """The experiment of one body of ``body``'s kind with the setting ``leaf`` is made from written as ``value`` (SI):
    the medium's viscosity, the rung's blob, or a setting of the body's kind."""
    kind, spacing = LEAF_KINDS[body]
    if leaf.startswith("shape_"):
        name = leaf.removeprefix("shape_")
        kind = dataclasses.replace(kind, **{name: Quantity(value, getattr(kind, name).dimension)})
    law = mrs(spacing, blob=value) if leaf == "blob" else mrs(spacing)
    viscosity = mPa_s(value * 1e3) if leaf == "viscosity" else mPa_s(1.0)
    return free_space(f"leaf_{body}", {body: kind}, law, viscosity=viscosity)


#: Fixed weights of the 36 entries of a resistance, in the loss each leaf is differentiated through.
WEIGHTS_6 = jnp.asarray(np.random.default_rng(20261028).normal(size=(6, 6)))


def resistance_through(built, body: str, params) -> jax.Array:
    """The node's resistance with ``params`` (its leaves), the body at its starting pose (moved and turned)."""
    node = drag_node(built, body)
    pose = jnp.asarray(node_state(built.graph, body)["pose"], dtype=jnp.float64)
    return node.advance(params, node.initial_state(), {"pose": pose}, 1e-3)["resistance"]


def stopped_at(leaf: str) -> type[MRSDragNode]:
    class Stopped(MRSDragNode):
        """The node with ``leaf``'s derivative stopped where it enters the step."""

        def advance(self, p, state, boundary_inputs, dt):
            return super().advance({**p, leaf: jax.lax.stop_gradient(p[leaf])}, state, boundary_inputs, dt)

    return Stopped


def test_every_trainable_leaf_of_the_node_is_taken_by_check_2():
    for body, leaves in LEAVES.items():
        node = drag_node(written_with(body, "blob", 0.5).build(), body)
        assert tuple(sorted(k for k, spec in node.param_specs().items() if spec.trainable)) == leaves, body


@pytest.mark.parametrize(("body", "leaf"), [(body, leaf) for body, leaves in LEAVES.items() for leaf in leaves],
                         ids=[f"{body}-{leaf}" for body, leaves in LEAVES.items() for leaf in leaves])
def test_check_2_each_leafs_derivative_equals_the_difference_by_rebuilding_and_a_stopped_leaf_fails(monkeypatch,
                                                                                                    body, leaf):
    """The node's derivative in each trainable leaf, through ``params``, of a weighted sum of its resistance (each entry
    scaled by its size as first built), against the central difference of the same sum with the setting written
    again, the experiment rebuilt at each step, the samples' counts unchanged by it and their choice held: the
    sampling's choice is the discretisation, as the counts are, and the choice made again at a moved capsule's height
    or screw's twist is another discretisation, whose difference is no derivative (the node takes its choice as built,
    as it takes its counts). The control, in the same experiment: the leaf behind a ``stop_gradient`` where it enters
    the step fails."""
    built = written_with(body, "blob", 0.5).build()
    leaves = parameter_leaves(built.graph)[f"drag:{body}"]
    first = np.abs(np.diag(np.asarray(resistance_through(built, body, leaves))))
    scale = jnp.asarray(np.sqrt(np.outer(first, first)))
    counts = drag_node(built, body).constants()["counts"]

    def loss(built, x):
        held = parameter_leaves(built.graph)[f"drag:{body}"]
        return jnp.sum(WEIGHTS_6 * resistance_through(built, body, {**held, leaf: x}) / scale)

    first_choice = drag_node(built, body).constants()["sampling_chosen"]

    def rebuilt(value: float) -> float:
        with held_choice(drag_node(built, body)):
            again = written_with(body, leaf, value).build()
        assert drag_node(again, body).constants()["counts"] == counts, "the step crossed a change of count"
        assert np.array_equal(drag_node(again, body).constants()["sampling_chosen"], first_choice)
        return float(jnp.sum(WEIGHTS_6 * resistance_at(again, body) / scale))

    theta = float(leaves[leaf])
    found = float(jax.grad(lambda x: loss(built, x))(leaves[leaf]))
    coarse, fine = difference(rebuilt, theta)
    value = rebuilt(theta)
    monkeypatch.setattr(hydrodynamic, "MRSDragNode", stopped_at(leaf))
    planted = written_with(body, leaf, theta).build()
    stopped = float(jax.grad(lambda x: loss(planted, x))(leaves[leaf]))
    print(f"\n{body} {leaf}: through params {found:.10e}; rebuilt, h {coarse:.10e}, h/2 {fine:.10e}; the leaf "
          f"stopped {stopped:.3e}")
    assert agrees(found, coarse, fine, value, theta)
    assert not agrees(stopped, coarse, fine, value, theta)


# -- check 3's pose on the manifold --------------------------------------------------------------------------------

def test_the_pose_derivative_on_the_manifold_equals_its_difference():
    built = free_space("manifold", {"screw": fl9()}, mrs(COARSE_SCREW)).build()
    node = built.graph.get_node("drag:screw")
    pose = jnp.asarray(node_state(built.graph, "screw")["pose"], dtype=jnp.float64)
    leaves = parameter_leaves(built.graph)["drag:screw"]
    rng = np.random.default_rng(20261028)
    weights = jnp.asarray(rng.normal(size=(6, 6)))

    def loss(tangent):
        moved = jnp.concatenate([pose[:3] + tangent[:3], quaternion.multiply(quaternion.exponential(tangent[3:]),
                                                                           pose[3:])])
        out = node.update(node_state(built.graph, "drag:screw"), {"pose": moved}, 1e-3, params=leaves)
        held = out["resistance"]
        return jnp.sum(weights * held / jnp.sqrt(jnp.outer(jnp.diag(held), jnp.diag(held))))

    found = np.asarray(jax.grad(loss)(jnp.zeros(6)))
    value = float(loss(jnp.zeros(6)))
    for k in range(6):
        step = POSE_STEP * (1e-3 if k < 3 else 1.0)
        unit = jnp.zeros(6).at[k].set(1.0)
        coarse = float(loss(step * unit) - loss(-step * unit)) / (2 * step)
        fine = float(loss(step / 2 * unit) - loss(-step / 2 * unit)) / step
        assert agrees(float(found[k]), coarse, fine, value, 1.0, step), (k, found[k], coarse, fine)


# -- the proposals ---------------------------------------------------------------------------------------------------

def primitives(jaxpr, name: str) -> int:
    """How many equations of ``name`` a program holds, inside its nested programs too."""
    count = 0
    for equation in jaxpr.eqns:
        count += equation.primitive.name == name
        for value in equation.params.values():
            for inner in value if isinstance(value, tuple | list) else (value,):
                held = getattr(inner, "jaxpr", inner)
                if hasattr(held, "eqns"):
                    count += primitives(held, name)
    return count


def ball_loss():
    """A weighted sum of the ball's resistance through the shipped node at its starting pose (turned), each entry scaled
    by its size at the built radius, as a function of the radius leaf; and that radius. A solve planted in the shipped
    one's place (``mime.nodes.hydro.mrs._solve.point_forces``, which ``resistance`` reads by name) is the node's. The
    rung is drag 0.3.0's point forces, which solve at every step, as drag 0.4.0's separated quadrature does with a wall
    (``test_mrs_separated.py`` checks its derivative there); in free space 0.4.0 holds the resistance and solves none."""
    built = free_space("proposals", {"ball": BALL}, point_forces(SPACING)).build()
    node = drag_node(built, "ball")
    leaves = parameter_leaves(built.graph)["drag:ball"]
    pose = jnp.asarray(node_state(built.graph, "ball")["pose"], dtype=jnp.float64)
    state = node.initial_state()
    held = np.asarray(node.advance(leaves, state, {"pose": pose}, 1e-3)["resistance"])
    scale = jnp.asarray(np.sqrt(np.outer(np.abs(np.diag(held)), np.abs(np.diag(held)))))
    weights = jnp.asarray(np.random.default_rng(20261028).normal(size=(6, 6)))

    def loss(radius):
        found = node.advance({**leaves, "shape_radius": radius}, state, {"pose": pose}, 1e-3)["resistance"]
        return jnp.sum(weights * found / scale)

    return loss, float(leaves["shape_radius"])


def symmetric_traction_solve(matrix, motions):
    """A traction matrix, ``A W`` with ``W`` a non-uniform diagonal (spread 3 to 1, as the helix's weights), solved with
    its adjoint taken as if it were symmetric: its value is the shipped solve's."""
    spread = jnp.repeat(1.0 + 0.5 * jnp.cos(jnp.arange(matrix.shape[0] // 3, dtype=matrix.dtype)), 3)
    held = matrix * spread[None, :]
    return spread[:, None] * jax.lax.custom_linear_solve(
        lambda x: jnp.matmul(held, x, precision="highest"), motions,
        solve=lambda _, b: jnp.linalg.solve(held, b), symmetric=True)


def through_factors(matrix, motions):
    return jax.scipy.linalg.lu_solve(jax.scipy.linalg.lu_factor(matrix), motions)


@jax.custom_jvp
def without_the_matrix_term(matrix, motions):
    """The linear-solve rule with its ``dA x`` term dropped: the tangent ``A^-1 db`` alone."""
    return jnp.linalg.solve(matrix, motions)


@without_the_matrix_term.defjvp
def _without_the_matrix_term(primals, tangents):
    matrix, motions = primals
    return jnp.linalg.solve(matrix, motions), jnp.linalg.solve(matrix, tangents[1])


#: Each solve in the shipped node's place: whether its forward and its reverse mode pass the check.
SOLVES = {"the linear-solve rule, as shipped": (None, True, True),
          "an adjoint taking a traction matrix for symmetric": (symmetric_traction_solve, True, False),
          "a derivative through the factors": (through_factors, True, True),
          "the linear-solve rule without its dA x term": (without_the_matrix_term, False, False)}


@pytest.mark.parametrize("label", ["the linear-solve rule, as shipped", "the linear-solve rule without its dA x term"])
def test_the_shipped_forward_mode_factorises_once_for_the_value_and_its_tangent(monkeypatch, label):
    """The node's program, and the program of its value and tangent, each hold one ``lu``; the control, a tangent
    solved apart from the value, holds a second."""
    solve, *_ = SOLVES[label]
    if solve is not None:
        monkeypatch.setattr(mrs_solve, "point_forces", solve)
    loss, radius = ball_loss()
    value = jax.make_jaxpr(loss)(radius).jaxpr
    tangent = jax.make_jaxpr(lambda r: jax.jvp(loss, (r,), (1.0,)))(radius).jaxpr
    counts = (primitives(value, "lu"), primitives(tangent, "lu"))
    print(f"\n{label}: lu in the value's program, and in the value's and tangent's: {counts}")
    assert (counts == (1, 1)) is (solve is None)


@pytest.mark.parametrize("label", list(SOLVES))
def test_the_shipped_solves_forward_and_reverse_modes_against_the_difference_and_the_planted_solves(monkeypatch,
                                                                                                     label):
    """The node's derivative in the ball's radius, forward (``jax.jvp``) and reverse (``jax.grad``), against the
    central difference of its value, with each solve in turn in the shipped one's place: the planted adjoint fails in
    reverse mode alone, since its forward solve is right; the derivative through the factors passes and equals the
    shipped rule's; the rule without its ``dA x`` term fails in both."""
    solve, forward_passes, reverse_passes = SOLVES[label]
    loss, radius = ball_loss()
    shipped = float(jax.grad(loss)(radius))
    if solve is not None:
        monkeypatch.setattr(mrs_solve, "point_forces", solve)
    value = float(loss(radius))
    coarse, fine = difference(lambda r: float(loss(r)), radius)
    reverse = float(jax.grad(loss)(radius))
    forward = float(jax.jvp(loss, (radius,), (1.0,))[1])
    print(f"\n{label}: reverse {reverse:.10e}, forward {forward:.10e}, difference {fine:.10e}")
    assert agrees(forward, coarse, fine, value, radius) is forward_passes
    assert agrees(reverse, coarse, fine, value, radius) is reverse_passes
    if solve is through_factors:
        assert math.isclose(reverse, shipped, rel_tol=1e-12)


def residual_sizes(checkpointed: bool) -> list[int]:
    built = free_space("residuals", {"ball": BALL}, point_forces(SPACING)).build()
    node = built.graph.get_node("drag:ball")
    leaves = parameter_leaves(built.graph)["drag:ball"]
    state = node.initial_state()
    pose = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0))

    def step(radius):
        return node.advance({**leaves, "shape_radius": radius}, state, {"pose": pose}, 1e-3)["resistance"]

    held = mrs_node.checkpoint
    if not checkpointed:
        mrs_node.checkpoint = lambda f, *a, **k: f
    try:
        _, pullback = jax.vjp(step, leaves["shape_radius"])
    finally:
        mrs_node.checkpoint = held
    return [int(np.size(leaf)) for leaf in jax.tree_util.tree_leaves(pullback)]


def test_the_reverse_mode_keeps_no_matrix_between_its_passes_and_its_checkpoint_is_what_does_it():
    built = free_space("count", {"ball": BALL}, point_forces(SPACING)).build()
    count = drag_node(built, "ball").sample_count
    matrix = (3 * count) ** 2
    with_checkpoint, without = residual_sizes(True), residual_sizes(False)
    print(f"\n{count} samples, the matrix {matrix} numbers: kept with the checkpoint, largest "
          f"{max(with_checkpoint)}; without, largest {max(without)}")
    assert max(with_checkpoint) < matrix <= max(without)


def test_the_resistance_scales_with_the_viscosity_and_its_sphere_with_the_radius_cubed_in_rotation():
    found = resistance_at(free_space("scale", {"ball": BALL}, mrs(SPACING)).build(), "ball", (0, 0, 0, 1, 0, 0, 0))
    doubled = resistance_at(free_space("scale", {"ball": Sphere(radius=mm(2.0))}, mrs(SPACING * 2)).build(), "ball",
                            (0, 0, 0, 1, 0, 0, 0))
    assert np.allclose(np.diag(doubled) / np.diag(found), [2, 2, 2, 8, 8, 8], rtol=1e-12)
