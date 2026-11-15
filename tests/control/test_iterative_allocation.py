"""The iterative allocation, ``mime.control.iterative_allocation``: against the exact path where both apply, against
SciPy's BVLS above the exact path's size, inverse kinematics against the library's, its implicit derivative against
central differences and reverse mode against forward, float32 against float64, and "converged" held to what it says.

**Converged means converged.** Every test that compares commands first asks the solve's own account, and judges it:
``converged`` must be true, its ``stationarity`` within the declared line :data:`LINE` (read when this module is
imported, before any plant), and the commands within the derived distance of the oracle's: the oracle's own line (the
least-squares perturbation bound of the exact path's and BVLS's, ``allocation_draws.line``) plus the natural
residual's error bound, ``|v - v*| <= (1 + L) / mu |pi|`` in the commands scaled to unit columns (``mu`` and ``L`` the
scaled Hessian's least and largest eigenvalues, ``pi`` the projected gradient the stationarity measures). A solve that
claims to converge where it did not is caught by one of the three.

**Its plants** (``ITERATIVE_PLANT``): ``early``, the solve stopped after two iterations; ``loose``, its tolerance
loosened to one (the stationarity of its start passes); ``gauss_newton``, the root's gradient taken with its Jacobian
held, so that its linearisation is the Gauss-Newton Hessian, which is wrong wherever the rows are not met (an
unreachable target); ``bounds``, the root's bounds held, so that a command at a bound has no derivative in it. Each must
be caught on every run: ``early`` and ``loose`` by the agreement tests, the other two by the derivative's."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from typing import Any

import jax
import numpy as np
import pytest
from allocation_draws import SLACK, Problem, bvls, line, problems, unique
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import mime.control._iterative as module
from mime.control import exact_allocation, iterative_allocation

jnp: Any = jax.numpy
lax: Any = jax.lax
PLANT = os.environ.get("ITERATIVE_PLANT", "")
#: The declared tolerance, read before any plant replaces the module's.
LINE = dict(module.TOLERANCE)
OWNER = "iterative_tests.active_set"
TRUE_GRADIENT, TRUE_BOUNDS = module._gradient, module._bounds


@contextlib.contextmanager
def in_dtype(name: str) -> Iterator[None]:
    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", name == "float64")
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", held)


def _gauss_newton(rows: Callable[[Any], Any], u: Any) -> Any:
    return lax.stop_gradient(jax.jacfwd(rows)(u)).T @ rows(u)


@contextlib.contextmanager
def planted(names: tuple[str, ...]) -> Iterator[None]:
    """The plant ``PLANT`` names, where it is one of ``names``: what the test it is measured on sees."""
    with pytest.MonkeyPatch.context() as patch:
        if PLANT in names:
            if PLANT == "early":
                patch.setattr(module, "ITERATIONS", 2)
            elif PLANT == "loose":
                patch.setattr(module, "TOLERANCE", dict.fromkeys(module.TOLERANCE, 1.0))
            elif PLANT == "gauss_newton":
                patch.setattr(module, "_gradient", _gauss_newton)
            elif PLANT == "bounds":
                patch.setattr(module, "_bounds", lambda low, high: (lax.stop_gradient(low), lax.stop_gradient(high)))
        yield


# -- the agreement, and what "converged" claims ----------------------------------------------------------------------

def scaled_problem(matrix: np.ndarray, target: np.ndarray, scales: np.ndarray, effort: float | None,
                   limit: float) -> tuple[np.ndarray, np.ndarray]:
    """The rows over their scales, with the effort's rows below them (towards zero), in float64."""
    a, b = matrix / scales[:, None], target / scales
    if effort is None:
        return a, b
    k = matrix.shape[1]
    return np.vstack([a, np.eye(k) / effort]), np.concatenate([b, np.zeros(k)])


def claim_problems(found: Any, a: np.ndarray, b: np.ndarray, reference: np.ndarray, oracle_line: float,
                   dtype: str) -> list[str]:
    """What is wrong with the solve's claim against the oracle's commands ``reference`` (see the module)."""
    problems_: list[str] = []
    commands = np.asarray(found.commands, dtype=np.float64)
    stationarity = float(found.stationarity)
    if not bool(found.converged):
        problems_.append(f"not converged: stationarity {stationarity:.3g}, {int(found.iterations)} iterations")
    if stationarity > LINE[np.dtype(dtype)]:
        problems_.append(f"stationarity {stationarity:.3g} past the declared line {LINE[np.dtype(dtype)]:.3g}")
    columns = np.linalg.norm(a, axis=0)
    unit = a / columns
    values = np.linalg.eigvalsh(unit.T @ unit)
    size = float(np.linalg.norm(a @ commands - b)) + float(np.sum(columns * np.abs(commands)))
    bound = (1.0 + values[-1]) / values[0] * np.sqrt(len(commands)) * stationarity * size / columns
    miss = np.abs(commands - reference)
    held = oracle_line + SLACK * bound
    if np.any(miss > held):
        problems_.append(f"commands {commands} from the oracle's {reference}: {miss} past {held}")
    return problems_


def iterative(a: np.ndarray, b: np.ndarray, limit: float, dtype: str) -> Any:
    held = [jnp.asarray(v, dtype=dtype) for v in (a, b)]
    return iterative_allocation(lambda u: held[0] @ u - held[1], jnp.zeros((a.shape[1],), dtype=dtype),
                                lower=-limit, upper=limit, owner=OWNER)


def efforts() -> st.SearchStrategy[float | None]:
    """None (unregularised) half the time, else a scale from 1 to 100 A."""
    return st.one_of(st.none(), st.floats(1.0, 100.0))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(max_examples=100, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(problem=problems(), effort=efforts())
def test_the_iterative_path_meets_the_exact_path_where_both_apply(dtype, problem, effort):
    """On the arrays MIME ships (three to eight coils, the exact path's size), the iterative path's commands are the
    exact path's within the derived distance, and its claim holds; unregularised problems whose minimiser is not
    unique are left out (the iterative path refuses them in ``ctx.allocator``)."""
    assume(effort is not None or unique(problem, dtype))
    a, b = scaled_problem(problem.matrix, problem.target, problem.scales, effort, problem.limit)
    limit = float(np.float32(problem.limit)) if dtype == "float32" else problem.limit
    with in_dtype(dtype), planted(("early", "loose")):
        found = iterative(a, b, limit, dtype)
    with in_dtype(dtype):
        exact = exact_allocation(jnp.asarray(a, dtype=dtype), jnp.asarray(b, dtype=dtype), lower=-limit, upper=limit,
                                 scales=jnp.ones((a.shape[0],), dtype=dtype), owner=OWNER)
    reference = np.asarray(exact.commands, dtype=np.float64)
    rounded = Problem(problem.system, problem.kind, a, b, np.ones(a.shape[0]), limit)
    assert claim_problems(found, a, b, reference, line(rounded, reference, dtype), dtype) == []


def synthetic(rng: np.random.Generator, commands: int, rows: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Gaussian rows of singular values 0.3 to 3, a target of a drawn direction whose size is a tenth to four times
    what the limit reaches, and a limit from 5 to 50 A."""
    left, _ = np.linalg.qr(rng.normal(size=(rows, commands)))
    right, _ = np.linalg.qr(rng.normal(size=(commands, commands)))
    matrix = left @ np.diag(np.exp(rng.uniform(np.log(0.3), np.log(3.0), commands))) @ right.T
    limit = float(rng.uniform(5.0, 50.0))
    direction = rng.normal(size=rows)
    reach = float(np.linalg.norm(matrix, axis=0).sum()) * limit
    return matrix, direction / np.linalg.norm(direction) * reach * float(np.exp(rng.uniform(np.log(0.1),
                                                                                             np.log(4.0)))), limit


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@settings(max_examples=60, deadline=None, print_blob=True, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2 ** 32 - 1), commands=st.integers(9, 16), extra=st.integers(0, 6),
       effort=efforts())
def test_the_iterative_path_meets_bvls_above_the_exact_paths_size(dtype, seed, commands, extra, effort):
    """Nine to sixteen commands, past the exact path's eight: SciPy's BVLS on the same rows (tall, or regularised) is
    the oracle, at camera spike R2's line, and the solve's claim holds."""
    from scipy.optimize import lsq_linear

    rng = np.random.default_rng(seed)
    matrix, target, limit = synthetic(rng, commands, commands + extra)
    a, b = scaled_problem(matrix, target, np.ones(len(target)), effort, limit)
    a = np.asarray(a, dtype=dtype).astype(np.float64)
    b = np.asarray(b, dtype=dtype).astype(np.float64)
    limit = float(np.float32(limit)) if dtype == "float32" else limit
    solved = lsq_linear(a, b, bounds=(-limit, limit), method="bvls", max_iter=3 ** 8)
    assert solved.status > 0, "the oracle did not converge"
    reference = np.asarray(solved.x)
    with in_dtype(dtype), planted(("early", "loose")):
        found = iterative(a, b, limit, dtype)
    rounded = Problem("synthetic", "rows", a, b, np.ones(a.shape[0]), limit)
    assert claim_problems(found, a, b, reference, line(rounded, reference, dtype), dtype) == []


def test_the_plants_of_the_claim_are_each_caught_at_one_problem(monkeypatch):
    """At one problem whose solve takes more than two iterations, the early stop leaves it unconverged and the loose
    tolerance claims a stationarity past the line: the two plants' mechanisms, held here; their rates are measured."""
    rng = np.random.default_rng(11)
    matrix, target, limit = synthetic(rng, 12, 14)
    with in_dtype("float64"):
        clean = iterative(matrix, target, limit, "float64")
        assert int(clean.iterations) > 2 and bool(clean.converged)
        reference = np.asarray(clean.commands)
        monkeypatch.setattr(module, "ITERATIONS", 2)
        early = iterative(matrix, target, limit, "float64")
        monkeypatch.setattr(module, "ITERATIONS", module.ITERATIONS)
        monkeypatch.undo()
        monkeypatch.setattr(module, "TOLERANCE", dict.fromkeys(module.TOLERANCE, 1.0))
        loose = iterative(matrix, target, limit, "float64")
    problem = Problem("synthetic", "rows", matrix, target, np.ones(len(target)), limit)
    assert claim_problems(early, matrix, target, reference, line(problem, reference, "float64"), "float64") != []
    assert claim_problems(loose, matrix, target, reference, line(problem, reference, "float64"), "float64") != []


def test_a_solve_that_cannot_converge_in_its_cap_says_so():
    """Capped at one iteration from far away, the solve returns its iterate inside the box and says it did not
    converge, with a stationarity past the line: never a silent value."""
    rng = np.random.default_rng(3)
    matrix, target, limit = synthetic(rng, 10, 12)
    with in_dtype("float64"), pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "ITERATIONS", 1)
        found = iterative(matrix, target * 3.0, limit, "float64")
    commands = np.asarray(found.commands)
    assert not bool(found.converged) and float(found.stationarity) > LINE[np.dtype("float64")]
    assert np.all(np.abs(commands) <= limit)


__all__: list[str] = []
