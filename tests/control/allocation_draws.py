"""Draws of allocation problems on the arrays MIME ships, and SciPy's BVLS as their oracle.

A problem is a point uniform in the central half of an array's declared region (``Workspace_Dimensions``), the rows a
caller asks there (the field's three, the gradient's five, both, or the force on a moment, ``F_j = m_i dB_i/dx_j``),
each over its scale (10 mT, 1 T/m, and for a force the force the moment feels per unit current, so that its rows are of
order one), and a wanted value of a drawn direction whose size is log-uniform from a tenth to four times what the
limit reaches (about half the problems hold a coil at its limit). The limit is drawn from 5 to 50 A, the moment from
0.1 to 1 mA m^2.

**The oracle** is SciPy's BVLS on the scaled rows, normalised to a largest entry of one: its tolerance is absolute (a
KKT residual of 1e-10), so on rows of order one it reaches the optimum, where on SI force rows (N/A, about 1e-6) it
stops short. In float32 it solves the float32 problem (the rows and target rounded to float32 as the law rounds them),
in float64.

**The line** (camera spike R2's): the two are the exact least squares of one active set, so where the minimiser is
unique they differ by rounding, which the least-squares perturbation bound puts at ``eps (2 kappa + kappa^2 |r| / (|A|
|x|))`` of the free commands (``kappa`` the free columns' condition after unit scaling), plus the limit's own rounding;
the test allows ``SLACK`` times that. **Uniqueness** is the full column rank of the scaled rows, read from their
singular values: the least over the largest above ``UNIQUE``. Where it is not unique the two may return different
minimisers, and their residual norms are compared at ``SLACK`` times the residual's rounding, and among tied minimisers
the law's effort, ``|u|^2``, is no more than BVLS's."""

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
    return np.asarray(lsq_linear(a / size, b / size, bounds=(-limit, limit), method="bvls").x, dtype=np.float64)


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


def effort_line(problem: Problem, dtype: str) -> float:
    """The effort's rounding line among tied minimisers, in A^2: ``SLACK`` times twice eps times the scaled rows'
    condition over their nonzero singular values, times the largest effort the limits allow (``K`` limits squared)."""
    eps = float(np.finfo(dtype).eps)
    a, _ = problem.scaled(dtype)
    singular = np.linalg.svd(a / np.linalg.norm(a, axis=0), compute_uv=False)
    kappa = singular[0] / singular[singular > UNIQUE * singular[0]][-1]
    return SLACK * 2 * eps * kappa * a.shape[1] * problem.limit ** 2


def verdict(problem: Problem, commands: np.ndarray, dtype: str) -> list[str]:
    """What is wrong with ``commands`` against the oracle: outside the limits, past the line on a unique problem, or,
    on one that is not, a residual above BVLS's past its rounding, or more effort than BVLS's among tied minimisers."""
    found: list[str] = []
    limit = float(np.float32(problem.limit)) if dtype == "float32" else problem.limit
    if not np.all(np.isfinite(commands)):
        return [f"commands not finite: {commands}"]
    if np.any(np.abs(commands) > limit):
        found.append(f"a command past its limit {limit}: {commands}")
    reference = bvls(problem, dtype)
    a, b = problem.scaled(dtype)
    if unique(problem, dtype):
        miss, held = float(np.max(np.abs(commands - reference))), line(problem, reference, dtype)
        if miss > held:
            found.append(f"unique: |law - BVLS| = {miss:.3g} A over the line {held:.3g} A ({problem.system} "
                         f"{problem.kind}, BVLS {reference}, law {commands})")
    else:
        mine, theirs = float(np.linalg.norm(a @ commands - b)), float(np.linalg.norm(a @ reference - b))
        held = residual_line(problem, dtype)
        if mine - theirs > held:
            found.append(f"not unique: the law's residual {mine:.6g} over BVLS's {theirs:.6g} past {held:.3g}")
        elif theirs - mine <= held and float(np.sum(commands ** 2)) - float(np.sum(reference ** 2)) > \
                effort_line(problem, dtype):
            found.append(f"not unique: the law's effort {np.sum(commands ** 2):.6g} over BVLS's "
                         f"{np.sum(reference ** 2):.6g} among tied minimisers")
    return found
