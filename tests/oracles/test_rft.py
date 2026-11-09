"""The rft rung against R6, the exact prolate spheroid and its own definiteness check (phase R's gates 1 and 2).

- **R6 through MIME's node** (gate 1): the coupling spike's helix (``rft_cases``), its centreline read at R6's 70
  points, gives R6's swimming speed U / Omega and the 6 x 6's eigenvalues in free space and with dev1's tube table, each
  within ``R6_LINE`` of the spike's, in float64. The line is the node path's rounding with room: phase R's first step
  reproduced the four numbers to 7e-15 relative with the spike's formulas against MIME's kernel and tube image, and the
  node computes the same sums in another order. **Control:** the coefficients swapped (``xi_par`` for ``xi_perp``)
  turns the speed's sign, caught at any line below 2.2.
- **The slender spheroid's coefficients** reproduce a prolate spheroid's exact translational resistance (Kim & Karrila's
  closed form, ``mrs_cases.prolate``) to leading order: within ``SPHEROID_LINES`` at aspect ratios 10, 30 and 100 (the
  first step measured 0.91 %, 0.10 % and 0.01 %), the miss falling with the aspect ratio as ``1 / ln`` squared does.
- **The check catches RFT's wall limit and nothing else** (gate 2): R6's helix with the tube table is refused at build
  (``drag.rft-indefinite``) and stopped at a step, and in free space it is neither; a straight rod, whose spin about
  its own axis meets no drag, passes the floored check at every one of 50 turned poses in both dtypes, where dev1's
  strict check, which reads the spin's rounding of either sign, refuses some (27 and 23 of the 50, float64 and
  float32, when this was written): a coin toss on a correct body. **Control:** the check aimed at the free block instead of the resistance with the wall
  passes R6's helix in the tube, so the gate's refusal assertion fails with it."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mrs_cases import prolate
from rft_cases import (
    D,
    R6_EIGENVALUES_FREE,
    R6_EIGENVALUES_TUBE,
    R6_HELIX,
    R6_SETTINGS,
    R6_SPACING,
    R6_SPEED_FREE,
    R6_SPEED_TUBE,
    R6_TUBE,
    about_the_axis,
    centroid,
    dimensionless,
    helix_body,
    r6_node,
    rft,
    swimming_speed,
)

from mime.build import BuildRefused
from mime.core import quaternion
from mime.nodes.hydro._rft_drag import SEMIDEFINITE_KINDS, drag_coefficients
from mime.nodes.hydro.mrs._quadrature import (
    DIAGONAL_FLOOR,
    floored_least_eigenvalue,
    scaled_least_eigenvalue,
    semidefinite_floor,
)
from mime.nodes.hydro.mrs._solve import rigid_motions
from mime.scene import Capsule
from mime.units import mm

#: Gate 1's line, relative: the node path's rounding with room (the first step's reproduction: 7e-15).
R6_LINE = 1e-9
#: The slender spheroid's coefficients against the exact prolate translation, by aspect ratio: the first step's misses
#: (0.91 %, 0.10 %, 0.01 % along; 0.07 %, 0.01 %, 0.00 % across) with room, falling with the aspect ratio.
SPHEROID_LINES = {10.0: 0.012, 30.0: 0.002, 100.0: 0.0003}


def r6_resistance(*, tube: bool, swapped: bool = False) -> np.ndarray:
    """R6's helix's resistance from the node, about its axis's middle, in D units."""
    node = r6_node(tube=tube)
    params = {**node.constants(), **node.params_pytree()}
    p = {key: jnp.asarray(value) if isinstance(value, float) else value for key, value in params.items()}
    p = {**node._step_constants(p), **p}
    if swapped:
        original = drag_coefficients

        def flipped(*args):
            par, perp = original(*args)
            return perp, par

        import mime.nodes.hydro._rft_drag as module

        module.drag_coefficients = flipped
        try:
            found, accepted = node.resistance_at(p, jnp.asarray([*centroid(R6_SETTINGS), 1.0, 0.0, 0.0, 0.0]))
        finally:
            module.drag_coefficients = original
    else:
        found, accepted = node.resistance_at(p, jnp.asarray([*centroid(R6_SETTINGS), 1.0, 0.0, 0.0, 0.0]))
    assert bool(accepted)
    return dimensionless(about_the_axis(np.asarray(found, dtype=np.float64), centroid(R6_SETTINGS)))


@pytest.mark.parametrize("tube, speed, eigenvalues", [(False, R6_SPEED_FREE, R6_EIGENVALUES_FREE),
                                                      (True, R6_SPEED_TUBE, R6_EIGENVALUES_TUBE)])
def test_r6_swimming_speed_and_spectrum_through_the_node(tube, speed, eigenvalues):
    resistance = r6_resistance(tube=tube)
    assert swimming_speed(resistance) == pytest.approx(speed, rel=R6_LINE)
    assert np.linalg.eigvalsh(resistance) == pytest.approx(np.asarray(eigenvalues), rel=R6_LINE)


def test_r6_through_the_built_experiment_in_free_space():
    built = helix_body("r6_free", R6_HELIX, R6_SETTINGS, rft(spacing=R6_SPACING, wall_kind="free_space")).build()
    from mime.build._maddening import node_state, parameter_leaves

    node = built.graph.get_node("drag:helix")
    assert node.point_count == 70
    out = node.update(node_state(built.graph, "drag:helix"), {"pose": node_state(built.graph, "helix")["pose"]},
                      1e-3, params=parameter_leaves(built.graph)["drag:helix"])
    resistance = about_the_axis(np.asarray(out["resistance"], dtype=np.float64), centroid(R6_SETTINGS))
    assert swimming_speed(resistance) / D == pytest.approx(R6_SPEED_FREE, rel=R6_LINE)


def test_control_the_coefficients_swapped_turn_the_speed():
    held = swimming_speed(r6_resistance(tube=False, swapped=True))
    assert abs(held - R6_SPEED_FREE) / R6_SPEED_FREE > 1.0
    assert held < 0.0


def _rod_resistance(points: np.ndarray, tangent: np.ndarray, ds: float, xpar: float, xperp: float,
                    centre: np.ndarray) -> np.ndarray:
    m = np.asarray(rigid_motions(jnp.asarray(points), jnp.asarray(centre)), dtype=np.float64)
    along = np.outer(tangent, tangent)
    block = (xpar * along + xperp * (np.eye(3) - along)) * ds
    return m.T @ np.kron(np.eye(len(points)), block) @ m


@pytest.mark.parametrize("ratio", sorted(SPHEROID_LINES))
def test_the_slender_spheroid_coefficients_carry_a_prolate_spheroids_translation(ratio):
    from mime.scene import Ellipsoid
    from mime.testing import centreline_problems
    from mime.units import mm

    a, b = 1e-3, 1e-3 / ratio
    kind = Ellipsoid(semi_axes=mm((1.0, 1.0 / ratio, 1.0 / ratio)))
    settings = {"semi_axes": (a, b, b)}
    assert centreline_problems(kind, settings) == []
    segment = kind.centreline(settings).segments[0]
    count = 400
    s = jnp.asarray((np.arange(count) + 0.5) * 2 * a / count)
    frame = segment.frame(settings, s)
    xpar, xperp = drag_coefficients("spheroid", 1.0, float(segment.filament_radius(settings)), None,
                                    float(segment.length(settings)))
    found = _rod_resistance(np.asarray(frame.point), np.asarray(frame.tangent)[0], 2 * a / count, float(xpar),
                            float(xperp), np.zeros(3))
    exact = prolate(a, b)
    assert abs(found[0, 0] / exact[0] - 1.0) < SPHEROID_LINES[ratio]
    assert abs(found[1, 1] / exact[1] - 1.0) < SPHEROID_LINES[ratio]


def test_r6_with_the_tube_table_is_refused_at_build():
    experiment = helix_body("r6_tube", R6_HELIX, R6_SETTINGS, rft(spacing=R6_SPACING, wall_kind="tube"), tube=R6_TUBE)
    with pytest.raises(BuildRefused) as refused:
        experiment.build()
    codes = [r.code for r in refused.value.refusals]
    assert codes == ["drag.rft-indefinite"]


def test_r6_in_free_space_is_not_refused_and_its_step_does_not_stop():
    built = helix_body("r6_free", R6_HELIX, R6_SETTINGS, rft(spacing=R6_SPACING, wall_kind="free_space")).build()
    from mime.build._maddening import node_state, parameter_leaves

    node = built.graph.get_node("drag:helix")
    out = node.update(node_state(built.graph, "drag:helix"), {"pose": node_state(built.graph, "helix")["pose"]},
                      1e-3, params=parameter_leaves(built.graph)["drag:helix"])
    assert float(out["indefinite"]) == 0.0
    assert np.all(np.isfinite(np.asarray(out["resistance"])))


@pytest.mark.parametrize("on_indefinite", ["stop", "flag"])
def test_r6_with_the_tube_table_stops_or_flags_at_a_step(on_indefinite):
    node = r6_node(tube=True, on_indefinite=on_indefinite)
    state = node.initial_state()
    params = {**node.params_pytree()}
    pose = jnp.asarray([*centroid(R6_SETTINGS), 1.0, 0.0, 0.0, 0.0])
    out = node.update(state, {"pose": pose}, 1e-3, params=params)
    assert float(out["indefinite"]) == 1.0
    assert float(out["stopped_at"]) == 1.0
    stopped = bool(np.all(np.isnan(np.asarray(out["resistance"]))))
    assert stopped == (on_indefinite == "stop")
    problem = node.stop_problem(out)
    assert (problem is not None and problem.code == "drag.rft-indefinite") == (on_indefinite == "stop")


def _turned_rod(dtype, key):
    """A straight rod's RFT resistance at a random turn and centre, in ``dtype``."""
    k1, k2 = jax.random.split(key)
    q = np.asarray(jax.random.normal(k1, (4,)), dtype=np.float64)
    q = q / np.linalg.norm(q)
    centre = np.asarray(jax.random.uniform(k2, (3,), minval=-1e-3, maxval=1e-3), dtype=np.float64)
    count, length, radius = 60, 1e-3, 1e-5
    s = (np.arange(count) + 0.5) / count * length - length / 2
    turn = np.asarray(quaternion.rotation_matrix(jnp.asarray(q)), dtype=np.float64)
    points = (np.asarray(centre) + np.stack([s, np.zeros(count), np.zeros(count)], 1) @ turn.T).astype(dtype)
    tangent = turn[:, 0].astype(dtype)
    xpar, xperp = (float(v) for v in drag_coefficients("spheroid", 1e-3, radius, None, length))
    m = np.asarray(rigid_motions(jnp.asarray(points), jnp.asarray(centre, dtype=dtype)))
    along = np.outer(tangent, tangent)
    block = ((xpar * along + xperp * (np.eye(3) - along)) * length / count).astype(dtype)
    return jnp.asarray(m.T @ np.kron(np.eye(count, dtype=dtype), block) @ m, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_straight_rod_passes_the_floored_check_and_fails_the_strict_one(dtype):
    refused = 0
    with jax.enable_x64(dtype == np.float64):
        for i in range(50):
            found = _turned_rod(dtype, jax.random.PRNGKey(i))
            assert float(floored_least_eigenvalue(found)) > -semidefinite_floor(dtype)
            refused += not float(scaled_least_eigenvalue(found)) > 0.0
    assert refused >= 1


def test_control_the_check_on_the_free_block_passes_r6_in_the_tube():
    free = r6_resistance(tube=False)
    with_wall = r6_resistance(tube=True)
    assert float(floored_least_eigenvalue(jnp.asarray(free))) > -semidefinite_floor(np.float64)
    assert not float(floored_least_eigenvalue(jnp.asarray(with_wall))) > -semidefinite_floor(np.float64)
    assert math.isfinite(float(np.min(np.linalg.eigvalsh(with_wall))))


def _planted(node, pose, dtype, c):
    """The node's resistance at ``pose`` with its least scaled eigenvalue planted at ``-c`` (its least eigenvector
    shifted, in the scaling the kind's check uses), and the least eigenvalue the node's check reads there."""
    leaves = {key: jnp.asarray(value, dtype=jnp.float64) if isinstance(value, (float, tuple, np.ndarray)) else value
              for key, value in node.constants().items()}
    found = np.asarray(node.resistance_at({**leaves, **node._wall_arguments()}, jnp.asarray(pose, dtype=jnp.float64))[0])
    symmetric = (found + found.T) / 2.0
    diagonal = np.abs(np.diag(symmetric))
    if node.constants()["kind"] in SEMIDEFINITE_KINDS:
        largest = np.repeat([diagonal[:3].max(), diagonal[3:].max()], 3)
        scale = np.sqrt(np.maximum(diagonal, DIAGONAL_FLOOR * largest))
    else:
        scale = np.sqrt(diagonal)
    scaled = symmetric / np.outer(scale, scale)
    values, vectors = np.linalg.eigh(scaled)
    shift = -c - values[0]
    planted = np.outer(scale, scale) * (scaled + shift * np.outer(vectors[:, 0], vectors[:, 0]))
    least, floor = node.judged(jnp.asarray(planted, dtype=dtype), dtype)
    return float(least), floor


def _random_poses(count):
    rng = np.random.default_rng(20261109)
    quaternions = rng.normal(size=(count, 4))
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    return np.concatenate([rng.uniform(-1e-4, 1e-4, size=(count, 3)), quaternions], axis=1)


def _node_of(kind, settings):
    built = helix_body("rft_plant", kind, settings, rft(wall_kind="free_space")).build()
    return built.graph.get_node("drag:helix")


PLANT_KINDS = {"capsule": (Capsule(radius=mm(0.03), height=mm(1.0), axis=(1.0, 0.0, 0.0)), {}),
               "helix": (R6_HELIX, R6_SETTINGS)}


@pytest.mark.parametrize("kind", ["capsule", "helix"])
@pytest.mark.parametrize("dtype, amounts", [(np.float64, (1e-3, 1e-8, 1e-12)), (np.float32, (1e-3, 1e-4))])
def test_a_planted_indefinite_resistance_is_refused_at_every_pose_above_the_kinds_floor(kind, dtype, amounts):
    """The plant: the node's own resistance at 100 random poses with its least scaled eigenvalue moved to ``-c``. The
    capsule (semidefinite) is refused where ``c`` is above its floor, 64 units in the last place; the helix (definite),
    judged against zero, where ``c`` is above rounding. Every pose, not most: the amounts sit decades above the
    thresholds (1.4e-14 and 7.6e-6 at the floor)."""
    node = _node_of(*PLANT_KINDS[kind])
    with jax.enable_x64(True):
        for c in amounts:
            reads = [_planted(node, pose, dtype, c) for pose in _random_poses(100)]
            assert all(not least > -floor for least, floor in reads), (kind, dtype, c, max(l for l, _ in reads))


def test_control_below_the_floor_the_semidefinite_kind_passes_and_the_definite_one_is_still_refused():
    """A planted ``-1e-16`` in float64 is under the capsule's floor (it passes: the spin's rounding is not refused) and
    over the helix's zero (it is refused): the split between the kinds is what is tested."""
    with jax.enable_x64(True):
        capsule, helix = _node_of(*PLANT_KINDS["capsule"]), _node_of(*PLANT_KINDS["helix"])
        poses = _random_poses(50)
        assert all(not l > -f for l, f in (_planted(helix, p, np.float64, 1e-13) for p in poses))
        assert all(l > -f for l, f in (_planted(capsule, p, np.float64, 1e-16) for p in poses))
