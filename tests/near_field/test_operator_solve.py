"""The shared GMRES's operator form (``reported_operator_solve``, F5's Schwarz products): the matrix given as its
product, which runs solves of its own and reports them.

- Its answers are the matrix form's (``reported_solve``) to the solve's tolerance, on a random system and on columns
  that are zero; its first report block is GMRES's own, its second the products' reports reduced (counts summed, the
  residual the worst), one product counted for every application of the operator.
- Its derivative (forward and reverse, through ``custom_linear_solve``) in a parameter of the operator is the matrix
  form's, and the transpose solve returns zeros in the products' block.
- A product reporting a stop short (a count of one in ``at_cap``) shows in the second block on every solve; the control
  (the products' reports dropped) shows none.

float64, small dense systems: the operator's own solves are planted reports, not solves (F5's flow node runs the real
ones)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mime.nodes.hydro.near_field._gmres import REPORT_SIZE, Stopping, reduced, reported_operator_solve, reported_solve

SIZE = 24
TOLERANCE = 1e-11


def system(seed: int):
    rng = np.random.default_rng(seed)
    base = np.eye(SIZE) * 3.0 + 0.4 * rng.standard_normal((SIZE, SIZE)) / np.sqrt(SIZE)
    bend = rng.standard_normal((SIZE, SIZE)) / np.sqrt(SIZE)
    columns = rng.standard_normal((SIZE, 3))
    columns[:, 1] = 0.0
    return jnp.asarray(base), jnp.asarray(bend), jnp.asarray(columns)


def identity(v):
    return v


def solved_both(theta, base, bend, b, short: bool = False, reports: bool = True):
    matrix = base + theta * bend
    stopping = Stopping(TOLERANCE)
    dense, _ = reported_solve(matrix, b, jnp.zeros(REPORT_SIZE), identity, identity, stopping)
    planted = jnp.zeros(REPORT_SIZE).at[0].set(1.0).at[2].set(1.0 if short else 0.0).at[1].set(1e-12)

    def apply(x):
        y = matrix @ x
        return y, planted if reports else jnp.zeros(REPORT_SIZE)

    found, report = reported_operator_solve(apply, b, jnp.zeros(2 * REPORT_SIZE), identity, identity, stopping)
    return dense, found, report


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_operator_form_answers_as_the_matrix_form_and_counts_its_products(seed):
    base, bend, b = system(seed)
    dense, found, report = jax.jit(solved_both)(0.3, base, bend, b)
    assert float(jnp.max(jnp.abs(found - dense)) / jnp.max(jnp.abs(dense))) < 100 * TOLERANCE
    assert float(jnp.max(jnp.abs(found[:, 1]))) == 0.0
    own, inner = report[:REPORT_SIZE], report[REPORT_SIZE:]
    assert float(own[1]) <= TOLERANCE and float(own[2] + own[3] + own[4]) == 0.0
    # Every application of the operator is a product: the cycle's Arnoldi steps and one true residual a restart.
    assert float(inner[0]) >= float(own[0]) + 1.0
    assert float(inner[1]) == pytest.approx(1e-12)
    assert float(inner[2]) == 0.0


def test_a_product_stopped_short_shows_on_every_solve_and_the_control_shows_none():
    base, bend, b = system(3)
    caught = [float(jax.jit(solved_both, static_argnames=("short", "reports"))(0.1 * k, base, bend, b,
                                                                                short=True)[2][REPORT_SIZE + 2]) > 0
              for k in range(5)]
    assert all(caught)
    quiet = [float(jax.jit(solved_both, static_argnames=("short", "reports"))(0.1 * k, base, bend, b, short=True,
                                                                               reports=False)[2][REPORT_SIZE + 2])
             for k in range(5)]
    assert quiet == [0.0] * 5


def test_its_derivative_is_the_matrix_forms_forward_and_reverse():
    base, bend, b = system(4)
    weights = jnp.asarray(np.random.default_rng(5).standard_normal((SIZE, 3)))

    def loss(theta, operator_form: bool):
        dense, found, _ = solved_both(theta, base, bend, b)
        return jnp.sum(weights * (found if operator_form else dense))

    for direction in ("forward", "reverse"):
        if direction == "forward":
            ours = jax.jvp(lambda t: loss(t, True), (0.2,), (1.0,))[1]
            theirs = jax.jvp(lambda t: loss(t, False), (0.2,), (1.0,))[1]
        else:
            ours = jax.grad(lambda t: loss(t, True))(0.2)
            theirs = jax.grad(lambda t: loss(t, False))(0.2)
        assert abs(float(ours) - float(theirs)) <= 1e-8 * abs(float(theirs))


def test_the_transpose_solve_reports_zeros_for_its_products():
    base, bend, b = system(6)

    def blocks(tap):
        matrix = base + 0.2 * bend
        _, report = reported_operator_solve(lambda x: (matrix @ x, jnp.ones(REPORT_SIZE)), b, tap, identity,
                                            identity, Stopping(TOLERANCE))
        return report

    def through(tap):
        matrix = base + 0.2 * bend
        found, _ = reported_operator_solve(lambda x: (matrix @ x, jnp.ones(REPORT_SIZE)), b, tap, identity,
                                           identity, Stopping(TOLERANCE))
        return jnp.sum(found)

    cotangent = jax.grad(through)(jnp.zeros(2 * REPORT_SIZE))
    assert float(jnp.max(jnp.abs(cotangent[REPORT_SIZE:]))) == 0.0
    assert float(cotangent[0]) >= 1.0 and float(cotangent[1]) <= TOLERANCE
    assert float(blocks(jnp.zeros(2 * REPORT_SIZE))[REPORT_SIZE]) >= 1.0


def test_reports_reduce_by_summing_counts_and_keeping_the_worst_residual():
    held = jnp.asarray([3.0, 1e-9, 0.0, 1.0, 0.0, 0.0])
    report = jnp.asarray([5.0, 1e-7, 1.0, 0.0, 0.0, 2.0])
    assert np.asarray(reduced(held, report)).tolist() == [8.0, 1e-7, 1.0, 1.0, 0.0, 2.0]
