"""Draws of allocation problems on the arrays MIME ships, and SciPy's BVLS as their oracle.

A problem is a point uniform in the central half of an array's declared region (``Workspace_Dimensions``), the rows a
caller asks there (the field's three, the gradient's five, both, or the force on a moment, ``F_j = m_i dB_i/dx_j``),
each over its scale (10 mT, 1 T/m, and for a force the force the moment feels per unit current, so that its rows are of
order one), and a wanted value of a drawn direction whose size is log-uniform from a tenth to four times what the
limit reaches (about half the problems hold a coil at its limit). The limit is drawn from 5 to 50 A, the moment from
0.1 to 1 mA m^2.

**The oracle** is SciPy's BVLS on the scaled rows, normalised to a largest entry of one: its tolerance is absolute (a
KKT residual of 1e-10), so on rows of order one it reaches the optimum, where on SI force rows (N/A, about 1e-6) it
stops short. Its iterations are capped at ``3^K``, the active sets there are: SciPy's default, as many iterations as
commands, stops it short on eight coils asked for a field and a gradient (status 0, a residual above the optimum's, on
about one draw in 300 of CardioMag's and OctoMag's), and an oracle that does not converge is refused. In float32 it
solves the float32 problem (the rows and target rounded to float32 as the law rounds them), in float64.

**The line** (camera spike R2's): the two are the exact least squares of one active set, so where the minimiser is
unique they differ by rounding, which the least-squares perturbation bound puts at ``eps (2 kappa + kappa^2 |r| / (|A|
|x|))`` of the free commands (``kappa`` the free columns' condition after unit scaling), plus the limit's own rounding;
the test allows ``SLACK`` times that. **Uniqueness** is the full column rank of the scaled rows, read from their
singular values: the least over the largest above ``UNIQUE``. Where it is not unique the two may return different
minimisers; and where it is, the law still cannot tell apart two faces whose residuals are within its tie window, which
float32 resolves more finely than the window does (a gap of 85 units in the last place of the target's norm, against a
computed residual good to a third of one), so the two can differ by whole commands. The law ties a candidate's residual
to the best's within its window, ``TIE_ULPS`` eps times the target's norm and what the candidate makes
(``mime.control._allocation``), the best's window and the candidate's added, and takes the least effort among those: its
residual norm may exceed BVLS's by both windows (at its commands and at BVLS's) and ``SLACK`` times the residual's
rounding, and no more. Where BVLS's residual is no worse than the law's past that rounding, the law's effort, ``|u|^2``,
is no more than BVLS's past the line's share of it: each command within the line of BVLS's, ``|u|^2`` within
``line (2 |u_BVLS|_1 + K line)`` (the line on the free columns, whose solve the rounding is of)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from hypothesis import strategies as st
from scipy.optimize import lsq_linear

from mime.data import asset
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import calibration_of
from mime.units import m

SYSTEMS = ("navion_2", "cardiomag", "octomag")
KINDS = ("field", "gradient", "both", "force")
#: The line's multiple of the perturbation bound (the spike's).
SLACK = 16.0
#: The scaled rows' least singular value over their largest above which the minimiser is unique.
UNIQUE = 1e-6
#: The oracle's own copy of the law's tie window, in units in the last place: a literal, never imported from
#: ``mime.control._allocation``, so that a change to the law's constant is seen by a test (``test_exact_allocation``'s
#: pin) and does not move the oracle with it. Changing it is a contract change that needs the maintainer's word.
DECLARED_TIE_ULPS = 64
FIELD_SCALE, GRADIENT_SCALE = 1e-2, 1.0


@cache
def provider(system: str) -> CalibratedMultipole:
    return CalibratedMultipole(calibration=asset(system))


@cache
def region(system: str) -> tuple[np.ndarray, np.ndarray]:
    """The declared region's centre and half its half-widths (the central half)."""
    held = calibration_of(asset(system)).region
    return held.mean(axis=1), 0.25 * (held[:, 1] - held[:, 0])


def five(gradient: np.ndarray) -> np.ndarray:
    return np.stack([gradient[0, 0], gradient[0, 1], gradient[0, 2], gradient[1, 1], gradient[1, 2]])


@dataclass(frozen=True)
class Problem:
    """``matrix`` ``(R, K)`` in SI per ampere, ``target`` ``(R,)`` (wanted less the background), ``scales`` ``(R,)``,
    ``limit`` (A, every coil), and what it was drawn as."""

    system: str
    kind: str
    matrix: np.ndarray
    target: np.ndarray
    scales: np.ndarray
    limit: float

    def scaled(self, dtype: str) -> tuple[np.ndarray, np.ndarray]:
        """The rows and target over their scales, rounded as the law rounds them in ``dtype``, in float64."""
        a, b, s = (np.asarray(v, dtype=dtype) for v in (self.matrix, self.target, self.scales))
        return (a / s[:, None]).astype(np.float64), (b / s).astype(np.float64)


@st.composite
def problems(draw: st.DrawFn, systems: tuple[str, ...] = SYSTEMS, kinds: tuple[str, ...] = KINDS,
             size: tuple[float, float] = (0.1, 4.0)) -> Problem:
    unit = st.floats(-1.0, 1.0, allow_nan=False)
    system, kind = draw(st.sampled_from(systems)), draw(st.sampled_from(kinds))
    centre, half = region(system)
    point = centre + np.asarray([draw(unit) for _ in range(3)]) * half
    actuation = provider(system).actuation(m(tuple(point)))
    limit = draw(st.floats(5.0, 50.0))
    if kind == "force":
        moment = np.asarray([draw(st.floats(-1.0, 1.0)) for _ in range(3)])
        if np.linalg.norm(moment) < 1e-3:
            moment[2] = 1.0
        moment = moment / np.linalg.norm(moment) * draw(st.floats(1e-4, 1e-3))
        matrix = np.einsum("i,ijk->jk", moment, actuation.gradient)
        scales = np.full(3, float(np.max(np.linalg.norm(matrix, axis=0))) or 1.0)
    else:
        rows = {"field": actuation.field, "gradient": five(actuation.gradient), "both": actuation.matrix()}[kind]
        matrix = rows
        scales = np.asarray({"field": [FIELD_SCALE] * 3, "gradient": [GRADIENT_SCALE] * 5,
                             "both": [FIELD_SCALE] * 3 + [GRADIENT_SCALE] * 5}[kind])
    direction = np.asarray([draw(st.floats(-1.0, 1.0)) for _ in range(len(matrix))])
    if np.linalg.norm(direction) < 1e-3:  # a direction is never zero: a wanted value of zero is its own case
        direction[0] = 1.0
    direction = direction / np.linalg.norm(direction)
    reach = reach_of(matrix, limit)
    wanted = direction * reach * float(np.exp(draw(st.floats(np.log(size[0]), np.log(size[1])))))
    return Problem(system, kind, np.asarray(matrix, dtype=np.float64), wanted, scales, limit)


def reach_of(matrix: np.ndarray, limit: float) -> float:
    """What every coil at ``limit`` reaches at most, by the rows' norm: the size wanted values are drawn against."""
    return float(np.linalg.norm(matrix, axis=0).sum()) * limit


def bvls(problem: Problem, dtype: str = "float64") -> np.ndarray:
    """SciPy's BVLS on the scaled rows, normalised to a largest entry of one (see the module)."""
    a, b = problem.scaled(dtype)
    size = float(np.max(np.abs(a))) or 1.0
    limit = float(np.float32(problem.limit)) if dtype == "float32" else problem.limit
    solved = lsq_linear(a / size, b / size, bounds=(-limit, limit), method="bvls", max_iter=3 ** a.shape[1])
    if solved.status <= 0:
        raise AssertionError(f"the oracle did not converge (BVLS status {solved.status}): {problem}")
    return np.asarray(solved.x, dtype=np.float64)


def unique(problem: Problem, dtype: str = "float64") -> bool:
    a, _ = problem.scaled(dtype)
    if a.shape[0] < a.shape[1]:
        return False
    singular = np.linalg.svd(a / np.linalg.norm(a, axis=0), compute_uv=False)
    return bool(singular[-1] > UNIQUE * singular[0])


def line(problem: Problem, reference: np.ndarray, dtype: str) -> float:
    """The swap line on the commands, in amperes: ``SLACK`` times the perturbation bound of ``reference``'s free
    commands, plus the limit's rounding (see the module)."""
    eps = float(np.finfo(dtype).eps)
    a, b = problem.scaled(dtype)
    free = np.abs(reference) < problem.limit * (1 - 1e-9)
    floor = eps * problem.limit
    if not free.any():
        return SLACK * floor
    norms = np.linalg.norm(a[:, free], axis=0)
    singular = np.linalg.svd(a[:, free] / norms, compute_uv=False)
    kappa = singular[0] / singular[-1]
    residual = float(np.linalg.norm(a @ reference - b))
    made = float(np.linalg.norm(reference[free] * norms))
    return SLACK * (eps * (2 * kappa + kappa ** 2 * residual / max(made, 1e-300)) * max(np.max(np.abs(reference)),
                                                                                          problem.limit) + floor)


def residual_line(problem: Problem, dtype: str) -> float:
    """The residual norm's rounding line: ``SLACK`` times eps times the rows' size at the limit plus the target's."""
    eps = float(np.finfo(dtype).eps)
    a, b = problem.scaled(dtype)
    return SLACK * eps * (np.linalg.norm(a, 2) * problem.limit * np.sqrt(a.shape[1]) + np.linalg.norm(b))


def effort_line(problem: Problem, reference: np.ndarray, dtype: str) -> float:
    """The effort's line among tied minimisers, in A^2: each command within :func:`line` of ``reference``'s,
    ``|u|^2`` within ``line (2 |reference|_1 + K line)``."""
    held = line(problem, reference, dtype)
    return held * (2 * float(np.sum(np.abs(reference))) + len(reference) * held)


def tie_window(problem: Problem, commands: np.ndarray, dtype: str) -> float:
    """One side of the law's tie window at ``commands``: ``DECLARED_TIE_ULPS`` eps times the scaled target's norm and
    what the commands make (``mime.control._allocation._selection``'s ``width``)."""
    a, b = problem.scaled(dtype)
    return DECLARED_TIE_ULPS * float(np.finfo(dtype).eps) * (float(np.linalg.norm(b)) +
                                                             float(np.linalg.norm(a @ commands)))


def two_sided_window(problem: Problem, best: np.ndarray, candidate: np.ndarray, dtype: str) -> float:
    """The law's tie rule, both sides: a candidate is tied to the best where its residual is within the best's window
    and its own added (``residual <= residual[best] + width[best] + width``)."""
    return tie_window(problem, best, dtype) + tie_window(problem, candidate, dtype)


def verdict(problem: Problem, commands: np.ndarray, dtype: str) -> list[str]:
    """What is wrong with ``commands`` against the oracle: outside the limits, past the line on a unique problem (unless
    the law's residual is within its tie windows of BVLS's: the law's own rule for faces it cannot tell apart), or, on one
    that is not, a residual above BVLS's past its rounding, or more effort than BVLS's among tied minimisers."""
    found: list[str] = []
    limit = float(np.float32(problem.limit)) if dtype == "float32" else problem.limit
    if not np.all(np.isfinite(commands)):
        return [f"commands not finite: {commands}"]
    if np.any(np.abs(commands) > limit):
        found.append(f"a command past its limit {limit}: {commands}")
    reference = bvls(problem, dtype)
    a, b = problem.scaled(dtype)
    mine, theirs = float(np.linalg.norm(a @ commands - b)), float(np.linalg.norm(a @ reference - b))
    held = residual_line(problem, dtype)
    tied = held + two_sided_window(problem, reference, commands, dtype)
    if unique(problem, dtype):
        miss, near = float(np.max(np.abs(commands - reference))), line(problem, reference, dtype)
        # A unique minimiser binds the law to BVLS's commands only where the law can tell its face from another: faces
        # whose residuals are within the law's tie window are tied by its own rule.
        if miss > near and mine - theirs > tied:
            found.append(f"unique: |law - BVLS| = {miss:.3g} A over the line {near:.3g} A, and the law's residual "
                         f"{mine:.6g} over BVLS's {theirs:.6g} past the tie window {tied:.3g} ({problem.system} "
                         f"{problem.kind}, BVLS {reference}, law {commands})")
    else:
        if mine - theirs > tied:
            found.append(f"not unique: the law's residual {mine:.6g} over BVLS's {theirs:.6g} past {tied:.3g}")
        elif theirs - mine <= held and float(np.sum(commands ** 2)) - float(np.sum(reference ** 2)) > \
                effort_line(problem, reference, dtype):
            found.append(f"not unique: the law's effort {np.sum(commands ** 2):.6g} over BVLS's "
                         f"{np.sum(reference ** 2):.6g} among tied minimisers")
    return found


# -- draws over every input of the law -----------------------------------------------------------------------------

INPUTS = ("matrix", "wanted", "lower", "upper", "offset", "reference", "reference_scales", "scales")


@dataclass(frozen=True)
class General:
    """Every input of ``exact_allocation`` for the derivative test, in float64: ``matrix`` ``(R, K)`` with singular
    values 0.5 to 2, ``wanted``, ``offset`` and ``scales`` ``(R,)``, ``lower``, ``upper``, ``reference`` and (where
    regularised) ``reference_scales`` ``(K,)``, and the seed its tangents are drawn from."""

    matrix: np.ndarray
    wanted: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    offset: np.ndarray
    reference: np.ndarray
    reference_scales: np.ndarray | None
    scales: np.ndarray
    seed: int

    def inputs(self) -> dict[str, np.ndarray]:
        held = {name: getattr(self, name) for name in INPUTS}
        return {name: value for name, value in held.items() if value is not None}

    @property
    def form(self) -> str:
        rows, commands = self.matrix.shape
        regularised = self.reference_scales is not None
        return f"K={commands} R={rows} {'regularised' if regularised else 'plain'} " \
               f"{'wide' if commands > rows and not regularised else 'tall'}"


#: The shapes drawn, (commands, rows, regularised): wide (more commands than rows, unregularised) at 3, 5 and 8
#: commands, tall at 3, 6 and 8, regularised at 4, 6 and 8. Fixed, so that each compiles once.
SHAPES = ((3, 2, False), (5, 3, False), (8, 3, False), (3, 5, False), (6, 6, False), (8, 8, False), (4, 2, True),
          (6, 4, True), (8, 3, True))


def general_at(commands: int, rows: int, regularised: bool, seed: int) -> General:
    """The ``General`` that ``generals`` draws at this shape and seed (so a failing draw is rebuilt from the two)."""
    rng = np.random.default_rng(seed)
    size = min(rows, commands)
    left = np.linalg.qr(rng.normal(size=(rows, size)))[0]
    right = np.linalg.qr(rng.normal(size=(commands, size)))[0]
    matrix = (left * rng.uniform(0.5, 2.0, size)) @ right.T
    return General(matrix=matrix, wanted=rng.normal(size=rows) * float(np.exp(rng.uniform(np.log(0.3), np.log(5.0)))),
                   lower=-rng.uniform(0.5, 2.0, commands), upper=rng.uniform(0.5, 2.0, commands),
                   offset=rng.normal(size=rows) * 0.1, reference=rng.uniform(-0.3, 0.3, commands),
                   reference_scales=rng.uniform(1.0, 5.0, commands) if regularised else None,
                   scales=rng.uniform(0.5, 2.0, rows), seed=seed)


@st.composite
def generals(draw: st.DrawFn) -> General:
    """A problem of one of ``SHAPES``, 3 to 8 commands, tall or wide or regularised, the matrix of condition at most 4,
    wanted values of 0.3 to 5 times the limits' size (some commands at a bound, some free), limits 0.5 to 2 either
    side. Drawn from one seed: the shape is what Hypothesis varies, and no draw is near-singular or at a numerical
    extreme."""
    commands, rows, regularised = draw(st.sampled_from(SHAPES))
    return general_at(commands, rows, regularised, draw(st.integers(0, 2 ** 32 - 1)))


def column_oracle(matrix: np.ndarray, wanted: np.ndarray, limit: float) -> float:
    """The least ``|matrix u - wanted|^2`` over ``|u| <= limit``, by SciPy's BVLS on the columns normalised to unit
    length (``v = u * norm``, bounds ``limit * norm``), so its absolute tolerance judges every column alike, run to
    ``3^K`` iterations and refused if it does not converge."""
    norms = np.linalg.norm(matrix, axis=0)
    solved = lsq_linear(matrix / norms, wanted, bounds=(-limit * norms, limit * norms), method="bvls", tol=1e-15,
                        max_iter=3 ** matrix.shape[1])
    if solved.status <= 0:
        raise AssertionError(f"the oracle did not converge (BVLS status {solved.status})")
    return float(np.sum((matrix @ (solved.x / norms) - wanted) ** 2))
