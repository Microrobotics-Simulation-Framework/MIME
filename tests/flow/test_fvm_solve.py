"""The solve: stopping never silent (gate 6), the preconditioner's cost check (gate 7), at least one iteration and the
zero exception (gate 8a), the report block in every mode, and starts that scale with their right-hand side."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import eps, in_dtype, tube_node
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.nodes.flow.fvm import Jacobi, Stopping, guarded_cg, projected_solve, scaling_start
from mime.nodes.flow.fvm._solve import REPORT_FIELDS

AT_CAP, STAGNATED, NOT_FINITE, AT_ZERO = (REPORT_FIELDS.index(name) for name in
                                          ("at_cap", "stagnated", "not_finite", "accepted_at_zero"))


@functools.cache
def problem_in(dtype: str) -> "Problem":
    """One problem per dtype, built and compiled once (a draw changes only the right-hand side)."""
    with in_dtype(dtype):
        return Problem(dtype)


class Problem:
    """The projected implicit operator of the empty tube at D/h 8 (19 x 19 x 2 cells), at a backward-Euler shift."""

    def __init__(self, dtype: str, shift: float = 50.0):
        self.node = tube_node(8, length_cells=2)
        p = self.node._step_constants(None)  # noqa: SLF001
        self.p = p
        self.grid = self.node.grid(p)
        fields = self.node.fields(p, self.grid)
        self.sigma = self.node.resistivity(p, fields)
        self.shift = jnp.asarray(shift, dtype=dtype)
        self.apply = self.node.operator_of(p, self.grid, self.sigma)(self.shift)
        self.diagonal = self.node.diagonal_of(p, self.grid, self.sigma, self.shift)
        self.dtype = dtype
        self.compiled: dict = {}

    def projected(self, x):
        return self.grid.project(self.apply(self.grid.project(x)))

    def precondition(self, kind: str = "jacobi"):
        """Jacobi; "wrong": Jacobi made with ten times the viscosity (still SPD); "one_sided": not symmetric."""
        diagonal = self.diagonal
        if kind == "wrong":
            diagonal = self.node.diagonal_of({**self.p, "viscosity": 10.0 * self.p["viscosity"]}, self.grid,
                                             self.sigma, self.shift)
        raw = Jacobi().made(diagonal)
        if kind == "one_sided":
            def apply(r):
                z = raw(self.grid.project(r))
                return self.grid.project((z[0] + 0.9 * jnp.roll(z[0], 1, axis=0), z[1], z[2]))
            return apply
        return lambda r: self.grid.project(raw(self.grid.project(r)))

    def rhs(self, seed: int, amplitude: float = 1.0):
        rng = np.random.default_rng(seed)
        return self.grid.project(tuple(jnp.asarray(amplitude * rng.standard_normal(self.grid.cells),
                                                   dtype=self.dtype) for _ in range(3)))

    def true_residual(self, x, b):
        r = tuple(bb - aa for bb, aa in zip(b, self.projected(x), strict=True))
        return float(jnp.sqrt(self.grid.inner(r, r)) / jnp.sqrt(self.grid.inner(b, b)))

    def cg(self, b, start=None, precondition=None, **stopping):
        start = tuple(jnp.zeros_like(c) for c in b) if start is None else start
        key = (tuple(sorted(stopping.items())), precondition)
        if key not in self.compiled:
            settings_ = {"tolerance": 1e-8 if self.dtype == "float64" else 1e-5, **stopping}
            chosen = self.precondition("jacobi" if precondition is None else precondition)
            self.compiled[key] = jax.jit(lambda bb, ss: guarded_cg(self.projected, bb, chosen, ss,
                                                                   Stopping(**settings_), self.grid.inner))
        return self.compiled[key](b, start)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 2**31 - 1), amplitude=st.floats(1e-3, 1e3))
def test_control_an_early_stop_is_read_by_the_true_residual_on_every_draw(seed, amplitude):
    """Gate 6: the CG stopped at a hundredth of its count; the true residual, recomputed outside the solver from the
    returned iterate, is above the tolerance, and the report says the cap stopped it."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            problem = problem_in(dtype)
            b = problem.rhs(seed, amplitude)
            _, full = problem.cg(b)
            cap = max(1, int(full[0]) // 100)
            x, report = problem.cg(b, cap=cap)
            tolerance = 1e-8 if dtype == "float64" else 1e-5
            assert problem.true_residual(x, b) > tolerance
            assert float(report[AT_CAP]) == 1.0
            assert int(report[0]) == cap


def test_the_converged_answer_holds_its_tolerance_by_the_true_residual(dtype):
    problem = Problem(dtype)
    b = problem.rhs(7)
    x, report = problem.cg(b)
    tolerance = 1e-8 if dtype == "float64" else 1e-5
    assert problem.true_residual(x, b) <= tolerance
    assert float(report[1]) <= tolerance
    assert float(report[AT_CAP] + report[STAGNATED] + report[NOT_FINITE] + report[AT_ZERO]) == 0.0


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_a_wrong_preconditioner_moves_only_the_cost(seed):
    """Gate 7, a cost check: Jacobi made with ten times the viscosity is still symmetric positive definite, so it
    changes the count and not the answer beyond the tolerance."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            problem = problem_in(dtype)
            b = problem.rhs(seed)
            right, report = problem.cg(b)
            other, other_report = problem.cg(b, precondition="wrong")
            tolerance = 1e-8 if dtype == "float64" else 1e-5
            assert problem.true_residual(other, b) <= tolerance
            assert float(other_report[0]) != float(report[0])
            difference = tuple(a - c for a, c in zip(other, right, strict=True))
            size = float(jnp.sqrt(problem.grid.inner(right, right)))
            assert float(jnp.sqrt(problem.grid.inner(difference, difference))) <= 100 * tolerance * size


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_control_a_preconditioner_that_is_not_symmetric_fails_the_solve(seed):
    """Gate 7's planted error: a one-sided preconditioner under CG, which the solve's stop must not pass as converged."""
    for dtype in ("float32", "float64"):
        with in_dtype(dtype):
            problem = problem_in(dtype)
            b = problem.rhs(seed)
            tolerance = 1e-8 if dtype == "float64" else 1e-5
            x, report = problem.cg(b, precondition="one_sided", cap=400)
            stopped = float(report[AT_CAP] + report[STAGNATED] + report[NOT_FINITE])
            assert stopped == 1.0 or problem.true_residual(x, b) > tolerance


def test_a_warm_start_already_within_the_tolerance_takes_an_iteration(dtype):
    """Gate 8a (option 1b): never accept the warm start unsolved; the control, no minimum, accepts it with none."""
    problem = Problem(dtype)
    b = problem.rhs(11)
    x, _ = problem.cg(b)
    _, report = problem.cg(b, start=x)
    assert int(report[0]) >= 1
    _, control = problem.cg(b, start=x, least_iterations=0)
    assert int(control[0]) == 0


def test_a_tolerance_below_rounding_stops_stagnated_before_its_cap(dtype):
    """The planted stall: asked for a residual the dtype cannot reach, the solve reaches its rounding floor and stops
    there as stagnated, the window's checks no longer falling under half the best before them, well before its cap;
    its iterate stays finite (a breakdown at the floor takes no step)."""
    problem = Problem(dtype)
    for seed in range(4):
        x, report = problem.cg(problem.rhs(seed), tolerance=1e-30)
        assert float(report[STAGNATED]) == 1.0 and float(report[AT_CAP]) == 0.0, report
        assert int(report[0]) < 5000 and bool(jnp.all(jnp.isfinite(jnp.stack([jnp.max(jnp.abs(c)) for c in x]))))


def test_an_exactly_zero_residual_is_accepted_unsolved(dtype):
    problem = Problem(dtype)
    zero = tuple(jnp.zeros(problem.grid.cells, dtype=dtype) for _ in range(3))
    x, report = problem.cg(zero)
    assert int(report[0]) == 0 and float(report[AT_ZERO]) == 1.0
    assert all(bool(jnp.all(c == 0.0)) for c in x)


def test_a_residual_that_is_not_finite_has_its_own_flag(dtype):
    problem = Problem(dtype)
    b = problem.rhs(13)
    b = (b[0].at[0, 0, 0].set(jnp.nan), b[1], b[2])
    _, report = problem.cg(b)
    assert float(report[NOT_FINITE]) == 1.0
    assert float(report[1]) == -1.0
    assert bool(jnp.all(jnp.isfinite(report)))


@pytest.mark.parametrize("scale", [1e-12, 1e6])
def test_the_start_scales_with_its_right_hand_side(float64, scale):
    problem = Problem("float64")
    b = problem.rhs(17)
    guess = problem.rhs(19)
    start = scaling_start(guess, b, problem.projected, problem.grid.inner)
    scaled = scaling_start(guess, tuple(scale * c for c in b), problem.projected, problem.grid.inner)
    for a, c in zip(start, scaled, strict=True):
        assert float(jnp.max(jnp.abs(c - scale * a))) <= 64 * eps("float64") * scale * float(jnp.max(jnp.abs(a)))


# -- the report block (BACKWARD_DIAGNOSTICS's design) ------------------------------------------------------------------

def _solved(problem, cap):
    stopping = Stopping(tolerance=jnp.asarray(1e-10, dtype=problem.dtype), cap=cap)

    def solve(theta, tap):
        rhs = tuple(theta * c for c in problem.rhs(23))
        u, report = projected_solve(problem.grid, problem.apply, problem.diagonal, rhs, None, stopping, tap, Jacobi())
        return u, report

    return solve


@pytest.mark.parametrize("cap", [5000, 2])
def test_the_report_block_carries_each_solves_report_in_every_mode(float64, cap):
    problem = Problem("float64")
    solve = _solved(problem, cap)
    tap = jnp.zeros(6)
    _, primal = solve(1.0, tap)
    _, (_, tangent) = jax.jvp(lambda th: solve(th, tap), (1.0,), (1.0,))

    def loss(theta, t):
        u, _ = solve(theta, t)
        return problem.grid.inner(u, u)

    _, transposed = jax.grad(loss, argnums=(0, 1))(1.0, tap)
    batched = jax.vmap(jax.grad(loss, argnums=1), in_axes=(0, None))(jnp.asarray([0.5, 2.0]), tap)
    checkpointed = jax.grad(jax.checkpoint(loss), argnums=1)(1.0, tap)
    for report in (primal, tangent, transposed, batched[0], batched[1], checkpointed):
        assert float(report[AT_CAP]) == (1.0 if cap == 2 else 0.0)
        assert float(report[0]) >= 1.0
        if cap == 2:
            assert float(report[0]) == 2.0
    assert float(transposed[1]) <= (1e-10 if cap == 5000 else np.inf)


def test_control_the_primals_plain_warm_start_does_not_scale(float64):
    """The plant the gradient gate's 1e-30 arm catches, seen here at any scale: the start ignores its right-hand side."""
    problem = Problem("float64")
    b, guess = problem.rhs(17), problem.rhs(19)

    def plain(g, rhs, operator, inner):
        return g

    start, scaled = plain(guess, b, problem.projected, problem.grid.inner), plain(guess, tuple(1e-12 * c for c in b),
                                                                                 problem.projected, problem.grid.inner)
    assert any(float(jnp.max(jnp.abs(c - 1e-12 * a))) > 1e-3 * float(jnp.max(jnp.abs(a)))
               for a, c in zip(start, scaled, strict=True))
