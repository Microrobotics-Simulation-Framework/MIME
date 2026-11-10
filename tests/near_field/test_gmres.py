"""The near field's iterative route: the joint system past the takeover line solved by GMRES, each body's held free
part's inverse its preconditioner block, every solve stopped on its true residual and reported.

- **Gate 2 on the iterative route**: the grand resistance against the dense route's on the same joint matrix, pairs and
  a triangle of coarse spheres, to the solve's tolerance (float64, 1e-10), and in float32 at its default tolerance
  (1e-5) against float64's dense route (gate 11's arm). The transposed preconditioner in the primal is a cost check
  (it moves the iterations, never the answer), not an accuracy control (S3's 183 of 183).
- **The stop**: a converged column's true residual is at most its tolerance in every column (the control: a cap of two
  iterations leaves columns above it, each counted at the cap); a tolerance below the dtype's floor stops stagnated
  before the cap, flagged; a zero right-hand side is accepted at zero, a NaN one returns NaN with its own count.
- **The report block** (F1a's design, the FVM's fields) carries the primal's, the tangent's and the transpose solve's
  own report in every mode (primal, ``jvp``, ``grad``, ``vmap``, ``checkpoint``), and a cap of one iteration planted is
  seen at the cap in every one of them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from near_field_cases import NODE, bodies, entrywise, joint_node, mrs, precision, rebuilt, spheres, start_inputs

from mime.build._maddening import parameter_leaves
from mime.interactions import NearField
from mime.nodes.hydro.near_field import joint_matrix
from mime.nodes.hydro.near_field._contract import joint_motions, offsets
from mime.nodes.hydro.near_field._gmres import STAGNATION_WINDOW, REPORT_FIELDS, Stopping, gmres, reported_solve

SPACING = 0.4e-3
CONFIGURATIONS = [(0.5e-3, 2), (0.2e-3, 2), (0.3e-3, 3)]
ITERATIONS, RESIDUAL = REPORT_FIELDS.index("iterations"), REPORT_FIELDS.index("residual")
AT_CAP, STAGNATED = REPORT_FIELDS.index("at_cap"), REPORT_FIELDS.index("stagnated")
NOT_FINITE, AT_ZERO = REPORT_FIELDS.index("not_finite"), REPORT_FIELDS.index("accepted_at_zero")
#: Gate 2's lines on the iterative route, each entry over its diagonal scale: float64 against the dense route on the same
#: matrix at the tolerance 1e-10, and float32 at its 1e-5 against float64's dense route.
LINE, FLOAT32_LINE = 1e-8, 2e-4


def cluster(gap: float, count: int, **settings):
    return bodies("cluster", spheres(gap, count), mrs(SPACING, **settings)).build()


def stepped_resistance(built) -> tuple[np.ndarray, dict]:
    node = joint_node(built)
    out = node.update(node.initial_state(), start_inputs(built), node.delta_t,
                      params=parameter_leaves(built.graph)[NODE])
    return np.asarray(out["resistance"], dtype=np.float64), {key: np.asarray(value) for key, value in out.items()}


@pytest.mark.parametrize(("gap", "count"), CONFIGURATIONS)
def test_past_the_line_the_grand_resistance_is_the_dense_routes_to_the_solves_tolerance(gap, count):
    with precision("float64"):
        dense, _ = stepped_resistance(cluster(gap, count))
        iterative, out = stepped_resistance(cluster(gap, count, near_field=NearField(solve="gmres")))
    with precision("float32"):
        single, single_out = stepped_resistance(cluster(gap, count, near_field=NearField(solve="gmres")))
    found, float32 = entrywise(iterative, dense), entrywise(single, dense)
    print(f"\n{count} spheres {gap * 1e3:g} mm apart: iterative against dense {found:.3g} (line {LINE:g}, report "
          f"{out['solve_report']}); float32's {float32:.3g} (line {FLOAT32_LINE:g}, report {single_out['solve_report']})")
    assert found <= LINE and float32 <= FLOAT32_LINE
    for report in (out["solve_report"], single_out["solve_report"]):
        assert report[AT_CAP] == report[STAGNATED] == report[NOT_FINITE] == 0.0
    assert out["solve_flagged"] == single_out["solve_flagged"] == 0.0


def test_the_route_is_the_declared_one_and_the_build_report_says_which():
    with precision("float64"):
        dense, iterative = cluster(0.5e-3, 2), cluster(0.5e-3, 2, near_field=NearField(solve="gmres"))
    small, large = joint_node(dense), joint_node(iterative)
    assert not small.iterative and large.iterative and small.unknowns == large.unknowns > 0
    assert "solve_report" not in small.initial_state() and "solve_report" in large.initial_state()
    assert "tolerance" not in small.constants() and large.constants()["tolerance"] == 1e-10
    said = " ".join(large.built_lines(None, None))
    assert "by GMRES" in said and "NearField(solve='gmres')" in said and "1e-10" in said
    assert "dense LU" in " ".join(small.built_lines(None, None))


# -- the solve on a joint system ------------------------------------------------------------------------------------


def system(dtype: str, gap: float = 0.2e-3, count: int = 3):
    """The joint matrix, its motions and each body's preconditioner block (and its transpose) of ``count`` coarse spheres
    ``gap`` apart, as the node makes them in a ``dtype`` run past the line."""
    with precision(dtype):
        built = cluster(gap, count, near_field=NearField(solve="gmres"))
        node = joint_node(built)
        p = node.constants(parameter_leaves(built.graph)[NODE])
        inputs = start_inputs(built)
        poses = jnp.stack([jnp.asarray(inputs[f"pose_{i}"]) for i in range(count)])
        joint = joint_matrix(node._operators, [node._leaves(i, p) for i in range(count)],  # noqa: SLF001
                             [node._held_value(i, p) for i in range(count)], poses, p["viscosity"],  # noqa: SLF001
                             node._wall(), {}, p["reach"])  # noqa: SLF001
        at = offsets([q.count for q in joint.placed])
        inverses = [node._held_inverse_value(i, p) for i in range(count)]  # noqa: SLF001
        ops = node._operators  # noqa: SLF001

        def joined(transpose: bool):
            parts = [op.preconditioner(inverses[i], poses[i], p["viscosity"], transpose) for i, op in enumerate(ops)]
            return lambda v: jnp.concatenate([part(v[at[i]:at[i + 1]]) for i, part in enumerate(parts)])

        return joint.matrix, joint_motions(joint), joined(False), joined(True)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_every_converged_column_holds_its_tolerance_by_its_true_residual_and_a_short_one_is_counted(dtype):
    """The stop on the true residual: every column of a converged solve is at most its tolerance, recomputed in float64
    from the answer. The control: a cap of two iterations planted leaves columns above the tolerance, and the report
    counts each of them at the cap, never silent."""
    tolerance = 1e-10 if dtype == "float64" else 1e-5
    with precision(dtype):
        matrix, motions, precondition, _ = system(dtype)
        solved = [jax.jit(lambda a, b, c=cap: gmres(lambda v: a @ v, b, precondition, Stopping(tolerance, cap=c)))(
            matrix, motions) for cap in (600, 2)]
    a, b = np.asarray(matrix, np.float64), np.asarray(motions, np.float64)
    columns = np.linalg.norm(b, axis=0) > 0.0

    def relative(x):
        return np.linalg.norm(a @ np.asarray(x, np.float64) - b, axis=0)[columns] / np.linalg.norm(b, axis=0)[columns]

    (ours, report), (short, capped) = ((relative(x), r) for x, r in solved)
    # float32's own evaluation of the residual differs from float64's by its rounding, a few eps of the answer's scale.
    slack = 1.0 if dtype == "float64" else 1.5
    above = int(np.sum(short > tolerance))
    print(f"\n{dtype}: worst column {ours.max():.3g} (report {float(report[RESIDUAL]):.3g}, tolerance {tolerance:g}); "
          f"capped at two, {above} of {short.size} columns above it, {float(capped[AT_CAP]):g} counted at the cap")
    assert report[AT_CAP] == report[STAGNATED] == 0.0 and ours.max() <= slack * tolerance
    assert above >= 1 and float(capped[AT_CAP]) >= above, "the control: a short stop is above and counted"


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_tolerance_below_the_dtypes_floor_stops_stagnated_before_its_cap_and_returns_its_best(dtype):
    with precision(dtype):
        matrix, motions, precondition, _ = system(dtype)
        converged = gmres(lambda v: matrix @ v, motions, precondition, Stopping(1e-10 if dtype == "float64" else 1e-5))
        x, report = gmres(lambda v: matrix @ v, motions, precondition, Stopping(1e-30, cap=600))
    print(f"\n{dtype}: a tolerance of 1e-30 stops at {report}")
    assert report[STAGNATED] == motions.shape[1] and report[AT_CAP] == 0.0 and report[ITERATIONS] < 600
    assert 0.0 < float(report[RESIDUAL]) <= float(converged[1][RESIDUAL])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_each_stall_restart_begins_the_window_again_and_none_traces_the_solve_as_before(dtype):
    """``Stopping.stall_restarts`` (a recheck raises them, design #337): at a tolerance under the floor every column
    stagnates whatever they are, and each restart begins its window again, so the solve takes at least a window of
    restarts more each (``STAGNATION_WINDOW`` restarts of at least one iteration), and never reaches the tolerance. (That
    none leaves a run's GMRES as it was is the bits probe's, base against head.)"""
    with precision(dtype):
        matrix, motions, precondition, _ = system(dtype)
        found = {restarts: jax.jit(lambda a, b, k=restarts: gmres(lambda v: a @ v, b, precondition,
                                                                  Stopping(1e-30, cap=6000, stall_restarts=k)))(
            matrix, motions) for restarts in (0, 1, 3)}
    iterations = {k: float(report[ITERATIONS]) for k, (_, report) in found.items()}
    print(f"\n{dtype}: iterations by stall restarts {iterations}")
    for _, report in found.values():
        assert report[STAGNATED] == motions.shape[1] and report[AT_CAP] == 0.0
    assert iterations[0] + STAGNATION_WINDOW <= iterations[1] and iterations[1] + 2 * STAGNATION_WINDOW <= iterations[3]


def test_the_rechecks_stopping_lifts_the_cap_and_raises_the_stall_restarts_as_the_cgs_does():
    """``recheck_stopping`` for GMRES, by the CG's constants: the cap to the larger of ten times itself and the default,
    the stall restarts raised by ``STALL_RESTARTS``, everything else, the tolerance first, as it was."""
    import dataclasses

    from mime.nodes.flow.fvm._solve import DEFAULT_CAP, RECHECK_CAP_FACTOR, STALL_RESTARTS
    from mime.nodes.hydro.near_field._gmres import CAP, recheck_stopping

    for cap, restarts in [(1, 0), (CAP, 0), (CAP, 2), (4 * DEFAULT_CAP, 1)]:
        held = Stopping(1e-10, restart=17, cap=cap, window=4, stagnation=0.25, stall_restarts=restarts)
        assert recheck_stopping(held) == dataclasses.replace(
            held, cap=max(cap * RECHECK_CAP_FACTOR, DEFAULT_CAP), stall_restarts=restarts + STALL_RESTARTS)


def test_a_zero_column_is_accepted_at_zero_and_a_nan_one_returns_nan_with_its_count():
    with precision("float64"):
        matrix, motions, precondition, _ = system("float64", count=2)
        held = motions.at[:, 1].set(0.0).at[0, 2].set(jnp.nan)
        x, report = gmres(lambda v: matrix @ v, held, precondition, Stopping(1e-10))
    x = np.asarray(x)
    assert np.all(x[:, 1] == 0.0) and np.all(np.isnan(x[:, 2])) and np.all(np.isfinite(np.delete(x, 2, axis=1)))
    assert report[AT_ZERO] == 1.0 and report[NOT_FINITE] == 1.0 and report[RESIDUAL] == -1.0


# -- the report block ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [600, 1])
def test_the_report_block_carries_each_solves_report_in_every_mode(cap):
    """The primal's report as the answer's block, the tangent's as its tangent, the transpose solve's as the gradient of
    the block's input (the ``solve_report`` leaf), under ``vmap`` and ``checkpoint`` too; a cap of one iteration planted
    is at the cap in every one, every run."""
    with precision("float64"):
        matrix, motions, precondition, transposed = system("float64", count=2)
        stopping = Stopping(1e-10, cap=cap)

        def solve(theta, tap):
            return reported_solve(theta * matrix, motions, tap, precondition, transposed, stopping)

        def loss(theta, tap):
            x, _ = solve(theta, tap)
            return jnp.sum(jnp.sin(1e3 * x))

        tap = jnp.zeros(6)
        _, primal = solve(1.0, tap)
        _, (_, tangent) = jax.jvp(lambda th: solve(th, tap), (1.0,), (1.0,))
        _, transpose = jax.grad(loss, argnums=(0, 1))(1.0, tap)
        batched = jax.vmap(jax.grad(loss, argnums=1), in_axes=(0, None))(jnp.asarray([0.9, 1.1]), tap)
        checkpointed = jax.grad(jax.checkpoint(loss), argnums=1)(1.0, tap)
    columns = motions.shape[1]
    for report in (primal, tangent, transpose, batched[0], batched[1], checkpointed):
        report = np.asarray(report)
        if cap == 1:
            assert report[ITERATIONS] == 1.0 and report[AT_CAP] >= 1.0, report
        else:
            assert report[ITERATIONS] >= 1.0 and report[AT_CAP] == report[STAGNATED] == 0.0, report
            assert 0.0 < report[RESIDUAL] <= 1e-10
        assert report[AT_CAP] + report[STAGNATED] + report[NOT_FINITE] + report[AT_ZERO] <= columns


def test_each_solve_takes_its_tolerance_from_its_own_right_hand_side():
    """The derivative solves start cold and stop relative to their own right-hand sides: the transpose solve's iterations
    and the gradient over the loss's scale are the same at loss scales 1e-12, 1 and 1e6."""
    with precision("float64"):
        matrix, motions, precondition, transposed = system("float64", count=2)

        def loss(theta, tap, scale):
            x, _ = reported_solve(theta * matrix, motions, tap, precondition, transposed, Stopping(1e-10))
            return scale * jnp.sum(jnp.sin(1e3 * x))

        rows = [jax.grad(loss, argnums=(0, 1))(1.0, jnp.zeros(6), scale) for scale in (1e-12, 1.0, 1e6)]
    gradients = [float(g) / scale for (g, _), scale in zip(rows, (1e-12, 1.0, 1e6), strict=True)]
    iterations = [float(t[ITERATIONS]) for _, t in rows]
    print(f"\ngradient over the scale {gradients}; the transpose solve's iterations {iterations}")
    assert max(abs(g - gradients[1]) for g in gradients) <= 1e-9 * abs(gradients[1])
    assert len(set(iterations)) == 1
