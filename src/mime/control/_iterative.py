"""The iterative allocation: commands within explicit bounds that minimise a sum of squares of rows a model makes
from them, by projected Newton, with the derivative of the converged solution by the implicit function theorem.

**The problem.** ``min 1/2 |r(u)|^2`` over ``lower <= u <= upper``, ``r`` the caller's rows (each a miss over its scale,
and any regularisation's rows), in any model: an affine one above the exact path's size, a nonlinear one (an arm's
frames from its joint positions). The bounds may be infinite.

**The method** (Bertsekas's projected Newton, 1982). In commands scaled to unit columns of the Jacobian ``J`` (``v = c
u``, ``c`` the columns' lengths, taken again each iteration), each iteration: an epsilon-active set, the commands within
a band of a bound with the gradient pushing them out (the band the lesser of :data:`ACTIVE_BAND` of the problem's size
and the stationarity's length); on the free commands Newton's step on the exact Hessian where it is positive definite
there (its least eigenvalue above ``RCOND`` of its largest), which converges quadratically near a minimiser whatever the
rows' miss, and the Gauss-Newton step ``(Jf^T Jf + floor) d = -Jf^T r`` where it is not (far from one), with a floor of
:data:`FLOOR` on the scaled normal matrix (a Levenberg term at the dtype's rank test, which keeps the step defined where
the rows do not fix the free commands); a scaled gradient step on the held ones; the box as a projection, and Armijo's
test on the projection arc (:data:`ARMIJO`, at most :data:`BACKTRACKS` halvings), nonmonotone for Newton's step on a
positive definite Hessian that no bound cuts: against the largest objective of the last :data:`_MEMORY` points, not the
current one's alone, since in a curved valley that step first raises the objective (the Maratos effect). Where a step
had to be cut, the next is damped (Levenberg's term, :data:`_DAMPING`, raised by :data:`_DAMPING_FACTOR` at each further
cut and lowered at each step taken whole), so that in a curved valley's floor the steps shorten and turn towards the
gradient instead of crawling by halvings. The step's path bends with the rows (Transtrum and Sethna's geodesic
acceleration): it is ``u + alpha step + alpha^2 bend / 2``, ``bend`` the least-squares fit of the rows' second derivative
along the step, so that the step follows a curved valley to second order instead of leaving it; zero on an affine
model. Where the bend is long beside the step (``2 |bend| > ACCELERATION |step|`` in the scaled commands, Transtrum
and Sethna's acceptance), the rows' second-order model does not hold that far, and the bend is left out: the step is
the straight one, under its line search. The decrease is measured as a difference of squares, ``sum (r - r') (r + r') / 2``, and a decrease within what
the objective resolves passes. What it resolves is judged on the scale the rows came from: a row is a difference of
terms (a frame's position and its goal, a field and its target) whose rounding is the terms', not the row's, which
cancels to far less, so each row's rounding is taken as eps of the problem's size (below), which bounds its terms
(exactly so on an affine model). It starts from the caller's start projected into the box. On an affine model the two
steps are one, Newton's, so the iteration is exact once its active set is the solution's. Unregularised, near a
singular pose of an arm, Newton's step along the direction the rows barely fix is long, and the point reached can
be far from the start (the AR4's wrist: a joint's limit 1.8 rad away, where the library's damped solve met the pose
0.1 rad away); a slew box around the start bounds how far the solve can go.

**Its local phase.** From the first point within the tolerance on, the solve is in Newton's local phase: full steps,
with no line search, while a step is longer than the commands' rounding (four units in the last place of the
problem's size, in the scaled commands), at most :data:`_POLISHING_STEPS`; what is returned is the point within the
tolerance whose step was the shortest. The tolerance bounds the stationarity, not the distance to the minimiser, at
which the derivative is taken; on an ill-conditioned problem (a weak effort, whose rows set the scaled Hessian's least
eigenvalues near 1e-8 of its largest) the two part far, and there the objective cannot judge the step: along a soft
direction its decrease is below its rounding, and in a curved valley (the redundant joints of an arm whose frame meets
its target) the full step can raise it past even the nonmonotone test's allowance while it converges quadratically.

**The columns.** A column's length is floored at the rows' curvature in its command, ``|J_k|^2 >= rho |S_kk|`` (``S``
the exact Hessian's second-order part, ``sum_i r_i grad^2 r_i``, and ``rho`` :data:`_CURVATURE`, four units in the last
place over the tolerance): where a column vanishes at a minimiser that the rows' curvature holds (``r = sin u - 2`` at
``pi / 2``), the gradient is the curvature times the command's distance, and in the column's own length the
stationarity reads the rows' miss whatever that distance, so the solve never met its tolerance there. Floored, it reads
the distance in the curvature's units, at most ``eps / rho`` (a quarter of the tolerance) at the command's rounding. The
floor moves no column whose length is above it, so on an affine model (``S`` zero) the solve is the one it was.

**Its guards, declared** (MIME's rule for single precision: a tolerance at or above the dtype's reachable floor, with a
stagnation stop). It stops at :data:`ITERATIONS` unless the point it has reached is within the tolerance (whose local
phase then ends by its own count), at the end of the local phase, or where it stagnates: no step passes Armijo's test,
or :data:`STALLS` steps in a row leave the commands within their rounding (the local phase's line), lower the
objective by no more than it resolves, and do not halve the stationarity. A step that moves the commands further is
progress the objective cannot show, not stagnation: in float32, on an unreachable target (an objective of 1e6 whose
steps lowered it by less than one), steps of a thousand times the commands' rounding were counted as stalls and stopped
two solves of 1202 with their stationarity 30 times the tolerance; counted only where the commands stand still, every
one converged, in at most 11 iterations. The stationarity is the projected gradient's length in the scaled commands, ``|v - P(v - grad_v)|_inf``,
over the problem's size, ``|r| + sum_k c_k |u_k|`` (the rows' miss and what the commands make, so its rounding floor is
the dtype's units in the last place of it whatever the rows cancel); the solve has met its tolerance where it is within
:data:`TOLERANCE`. ``converged`` is read again at the commands returned, never from the loop's own account, and claims a
first-order stationary point within the box, isolated: the stationarity within its tolerance there, and the free
columns of full rank or the derivative's tangent system resolved (below). The rank is judged from the rows' singular
values, not their squares, the normal matrix's eigenvalues: a singular value at or below :data:`RANK` times ``max(M,
K)`` units in the last place of the largest is not resolved from zero (``M`` rows, ``K`` commands). It does not claim a
local minimiser, and makes no second-order check: a strict maximum of the objective, where the gradient is zero, is a
first-order stationary point (a planar arm started there claims to converge), and the exact Hessian's least eigenvalue
is not resolved where it would tell (at the follower's problem in float32 it read -4e-5 of the largest, noise about a
true 2e-8). Where it is not converged, the commands returned are the last iterate, inside the box, and the caller says
so: ``ctx.allocator`` raises its controller's flag ``allocation_unconverged``, carrying the stationarity.

**The derivative.** ``jax.lax.custom_root`` on ``F(u) = c (u - P(u - grad(u) / c^2))``, ``c`` held (a positive scaling
changes the root of no component): at the root, a free command's row is the gradient's and a held one's ``u - bound``.
The tangent solve is ``F``'s linearisation at the root, taken by autodiff of the exact gradient, so its free rows are
the exact Hessian's, ``J^T J + sum_i r_i grad^2 r_i``, never the Gauss-Newton one (which is wrong wherever the rows are
not met, as an unreachable target's); a held command's derivative is its bound's (a port's, or the slew box's around
the last command). Each closed-over value of the rows (a wanted value, a model's parameters, a point) and of the bounds
carries its tangent through ``F``. The tangent system, ``K x K``, is solved by one dense LU (``mime.core.solves``),
its rows and then its columns equilibrated (a row's scale is its command's column length, which moves no root). It is
the derivative of the converged commands: where the solve did not converge, it is the implicit function's at the
iterate, which solves nothing, and the step's flag says so.

**Its resolution.** The tangent system is the exact Hessian's, whose condition is the square of the rows' (a weak
effort's: the follower's is 4e6 to 1e8). Its error over its size is bounded by its condition times ``M + 3 K`` units
in the last place (the assembly's sums over the ``M`` rows and the LU's ``3 K``, Higham's bound for a solve, growth
aside), whatever the forward solve's accuracy: ``resolution`` is that bound, read at the commands returned on the
system as the derivative factors it (rows and columns equilibrated), and ``resolved`` holds where it is at most
:data:`RESOLVED`. Measured at the follower's problem in float32, efforts 0.05 to 20 rad (56 draws), the derivative's
error against float64's was 0.005 to 0.7 of the bound; at its own effort the derivative was 2.3 times its largest
entry off (median of 60 draws) while the joints were 2e-6 rad from float64's. Where it is not resolved,
``ctx.allocator`` raises the flag ``allocation_derivative_unresolved``, carrying the bound: a derivative through that
step is not resolved in the run's dtype (in float32 at the follower's problem, at every step; float64 resolves it).
The derivative is left as the LU gives it. Neither a float64 tangent system at the float32 commands (its error a median
0.6 of the largest entry, at most 46, over 20 draws: the derivative at the commands float32 reaches is not the
minimiser's, the condition times their 2e-6 rad) nor a pseudo-inverse on the resolved directions (median 3.3, which
drops the response along the others) resolves it. What did, at float64's commands, is the rows' own factorisation (the
Gauss-Newton part by QR of ``J``, median 4e-4), which leaves the second-order part out: a route, not built here.

**Its status.** On an affine model the problem is convex: the minimiser is continuous (regularised, or of full column
rank), and its derivative has kinks where a command reaches a bound or leaves it (``piecewise_kink``). On a nonlinear
one the point is the stationary one reached from the start, which can change branch as what is wanted moves
(``piecewise_jump``). The switch is its caller's owner (:func:`mime.control.allocation_switch`), evaluated once at the
solution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import numpy as np

from mime.control._allocation import RCOND
from mime.core import switch
from mime.core.solves import dense_solve

jnp: Any = jax.numpy
lax: Any = jax.lax
#: JAX's transforms, whose hints a strict checker reads as partly unknown.
_jax: Any = jax

#: The most iterations the solve takes before its local phase, which ends by its own count (:data:`_POLISHING_STEPS`).
ITERATIONS = 50
#: The most halvings of a step Armijo's test takes before the solve counts the iteration as stagnated.
BACKTRACKS = 30
#: Armijo's fraction of the predicted decrease a step must make.
ARMIJO = 1e-4
#: The stationarity, over the problem's size, at which the solve has converged, by dtype: at or above the reachable floor.
TOLERANCE = {np.dtype(np.float64): 1e-10, np.dtype(np.float32): 1e-5}
#: The Levenberg floor on the scaled normal matrix (whose diagonal is one), by dtype: the square of the exact path's
#: rank test's, which keeps the Gauss-Newton step defined where the rows do not fix the free commands.
FLOOR = {dtype: value * value for dtype, value in RCOND.items()}
#: The numerical rank's line, in units in the last place of an ``M x K`` matrix's largest singular value times
#: ``max(M, K)``: a singular value at or below it is not resolved from zero (``jnp.linalg.pinv``'s default; a backward
#: stable decomposition's singular values are those of the matrix moved by a few ``max(M, K)`` units of its largest,
#: and the matrix's own rounding, a Jacobian's by autodiff, adds as much again). The free columns' rank is judged at
#: it. Measured: rows with an exact null vector read at most 1e-2 of the line (200 of 200, float64), the AR4's wrist
#: singularity at most 3e-2 of it, and the float32 follower's least singular value (its rows' at 1 to 5 cm, the
#: review's draws) 8 to 40 times it.
RANK = 10.0
#: The largest error bound over its size at which the derivative is called resolved (``resolution``, see the module):
#: a tenth, declared.
RESOLVED = 0.1
#: Transtrum and Sethna's acceptance of the bend (geodesic acceleration, 2012), their standard value: a bend longer than
#: ``ACCELERATION / 2`` of its step, in the scaled commands, is left out of it. A default the maintainer may revise.
ACCELERATION = 0.75
#: The widest band, as a share of the problem's size in the scaled commands, within which a command at a bound is held.
ACTIVE_BAND = 1e-3
#: The steps in a row that leave the commands within their rounding and lower the objective by no more than its
#: rounding before the solve stops as stagnated.
STALLS = 2
#: The points whose largest objective Newton's step is judged against (Grippo, Lampariello and Lucidi's nonmonotone
#: line search, 1986), where the Hessian on the free commands is positive definite and no bound cuts the step: in a
#: curved valley (an arm's redundant joints, a weak effort holding them) the full step first raises the objective and
#: then lowers it past where it began, the Maratos effect, and a test against the current point alone halved those
#: steps at every other iteration: a warm start 0.1 mrad from the follower's minimiser took the 50 iterations of the
#: cap. A step the box cuts, or a Gauss-Newton one, is judged against the current point alone: with an allowance, a
#: float32 OctoMag draw cycled between two corners of the box for the whole cap, and Gauss-Newton's long steps near
#: the AR4's wrist singularity left the inverse kinematics' start for a joint limit three times as often. Over 2220
#: of the follower's warm starts, the most iterations went from 47 to 35 (to 15 with the damping below, which without
#: the allowance left 3 of them at the cap).
_MEMORY = 4
#: Levenberg's damping on the scaled matrix (whose diagonal is about one) after a step that had to be cut, and the
#: factor it is raised by at each further cut and lowered by at each step taken whole (to none below the first value).
#: A warm start crawling by halvings in the indefinite region of a curved valley met the cap 0.027 rad from the
#: minimiser (the warm-start test's seed 1210936); damped, it converged in 20.
_DAMPING, _DAMPING_FACTOR = 1e-3, 10.0
#: The most full steps of the local phase (see the module), from the first point within the tolerance on. A reference
#: solve of the follower's derivative test met its tolerance 2e-3 rad from the minimiser (its objective 37 % above the
#: minimum; the scaled Hessian's least eigenvalue 1.3e-7 of its largest), and its full steps went 0.24, 7e-4, 1e-3,
#: 4e-8 and 2e-12 (scaled), the first raising the objective, the sixth within the commands' rounding: five steps.
#: Over 2220 of the test's solves (60 draws), every one ended within the rounding with eight.
_POLISHING_STEPS = 8
#: The share ``rho`` of the rows' curvature at which a column's length is floored, by dtype: four units in the last place
#: over the tolerance, so that at a minimiser where the column vanishes the stationarity's rounding is a quarter of the
#: tolerance (see the module).
_CURVATURE = {dtype: 4.0 * float(np.finfo(dtype).eps) / value for dtype, value in TOLERANCE.items()}

type Rows = Callable[[Any], Any]


class IterativeAllocation(NamedTuple):
    """The commands ``(K,)``; each command's place, ``active`` ``(K,)``, ``-1`` held at its lower bound, ``0`` free,
    ``+1`` at its upper one; the ``objective`` there, the sum of the squares of the rows; whether it ``converged`` (read
    at the commands returned: see the module for what it claims); its ``stationarity`` there, over the problem's size;
    the ``iterations`` it took; whether its derivative is ``resolved`` in the dtype, and its ``resolution``, the tangent
    system's condition times ``M + 3 K`` units in the last place, a bound on its error over its size (see the
    module)."""

    commands: Any
    active: Any
    objective: Any
    converged: Any
    stationarity: Any
    iterations: Any
    resolved: Any
    resolution: Any


class _At(NamedTuple):
    """The rows, their Jacobian, its columns' lengths (floored at the rows' curvature), the gradient, the
    stationarity's vector, the size and the exact Hessian, at a point."""

    rows: Any
    jacobian: Any
    columns: Any
    gradient: Any
    projected: Any
    size: Any
    hessian: Any


def _objective(rows: Rows) -> Callable[[Any], Any]:
    def objective(u: Any) -> Any:
        made = rows(u)
        return 0.5 * jnp.sum(made * made)

    return objective


def _at(rows: Rows, u: Any, lower: Any, upper: Any) -> _At:
    """The solve's view of the problem at ``u``: none of it is differentiated (the derivative is the root's)."""
    def twice(v: Any) -> tuple[Any, Any]:
        made = rows(v)
        return made, made

    jacobian, made = _jax.jacfwd(twice, has_aux=True)(u)
    hessian = _jax.hessian(_objective(rows))(u)
    squared = jnp.sum(jacobian * jacobian, axis=0)
    # The column's length floored at the rows' curvature in its command (see the module): |S_kk|, the Hessian's
    # diagonal less the column's square.
    squared = jnp.maximum(squared, _CURVATURE[np.dtype(u.dtype)] * jnp.abs(jnp.diagonal(hessian) - squared))
    columns = jnp.sqrt(jnp.where(squared > jnp.finfo(u.dtype).tiny, squared, 1.0))
    gradient = jacobian.T @ made
    projected = columns * (u - jnp.clip(u - gradient / (columns * columns), lower, upper))
    size = jnp.sqrt(jnp.sum(made * made)) + jnp.sum(columns * jnp.abs(u)) + jnp.finfo(u.dtype).tiny
    return _At(made, jacobian, columns, gradient, projected, size, hessian)


def _held(at: _At, u: Any, lower: Any, upper: Any, band: Any) -> tuple[Any, Any]:
    """The commands held at their lower and upper bounds: within ``band`` of one (scaled), the gradient pushing out."""
    scaled = at.gradient / at.columns
    return ((at.columns * (u - lower) <= band) & (scaled > 0.0),
            (at.columns * (upper - u) <= band) & (scaled < 0.0))


def _singular(matrix: Any) -> tuple[Any, Any]:
    """The least and the largest singular values of ``matrix`` (``M x K``, ``M >= K``), as the eigenvalues of the
    symmetric ``[[0, A], [A^T, 0]]``, which are ``+-`` each singular value and ``M - K`` zeros (Jordan and Wielandt): a
    symmetric eigensolver's rounding is units in the last place of the largest, as a singular value decomposition's,
    never their square, as the normal matrix's ``A^T A`` is. A rank test uses the matrix and solves nothing."""
    size, count = matrix.shape
    joined = jnp.concatenate([jnp.concatenate([jnp.zeros((size, size), dtype=matrix.dtype), matrix], axis=1),
                              jnp.concatenate([matrix.T, jnp.zeros((count, count), dtype=matrix.dtype)], axis=1)])
    values = jnp.sort(jnp.abs(jnp.linalg.eigvalsh(joined)))
    return values[size - count], values[-1]


def _verdict(rows: Rows, at: _At, u: Any, free: Any, lower: Any, upper: Any) -> tuple[Any, Any]:
    """At the commands returned: whether the point is isolated as its rows see it (the free columns of full rank at
    the rank line, :data:`RANK`, scaled, a held command's column its own unit column), and the derivative's
    ``resolution``, the tangent system's condition times ``M + 3 K`` units in the last place, the system as
    :func:`_tangent` factors it."""
    dtype = u.dtype
    eps, tiny = float(jnp.finfo(dtype).eps), jnp.finfo(dtype).tiny
    mask = free.astype(dtype)
    least, largest = _singular(jnp.concatenate([at.jacobian / at.columns * mask[None, :], jnp.diag(1.0 - mask)]))
    full = least > RANK * max(at.rows.shape[0], u.shape[0]) * eps * largest
    least, largest = _singular(_equilibrated(_jax.jacfwd(_root(rows, lower, upper))(u))[0])
    return full, (at.rows.shape[0] + 3 * u.shape[0]) * eps * largest / jnp.maximum(least, tiny)


def _bend(rows: Rows, u: Any, moving: Any, matrix: Any, mask: Any, scaled: Any, columns: Any) -> Any:
    """The step's bend (Transtrum and Sethna's geodesic acceleration): the least-squares fit, on the free commands
    (``mask``), of ``J bend = -r''``, ``r''`` the rows' second derivative along ``moving``, in the scaled commands
    (``scaled`` the Jacobian over ``columns``, ``matrix`` the step's normal matrix with its floor and damping), so that
    along ``u + alpha moving + alpha^2 bend / 2`` the rows' linear model holds to second order."""
    def along(v: Any) -> Any:
        return _jax.jvp(rows, (v,), (moving,))[1]

    curvature = _jax.jvp(along, (u,), (moving,))[1]
    return -dense_solve(matrix, mask * (scaled.T @ curvature)) / columns


def _solved(rows: Rows, start: Any, lower: Any, upper: Any, dtype: Any) -> tuple[Any, tuple[Any, ...]]:
    """The forward solve, with no derivative: the commands, and ``(converged, stationarity, iterations, active,
    resolved, resolution)`` read at them."""
    held = np.dtype(dtype)
    tolerance, floor, rcond = TOLERANCE[held], FLOOR[held], RCOND[held]
    eps = float(np.finfo(held).eps)
    count = start.shape[0]
    eye = jnp.eye(count, dtype=dtype)

    def iteration(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        u, taken, stalls, _, before, quiet, polished, best, shortest, recent, damping = carry
        at = _at(rows, u, lower, upper)
        measure = jnp.max(jnp.abs(at.projected)) / at.size
        stationary = measure <= tolerance
        # The objectives of the last _MEMORY points, this one's last: a step is judged against the largest.
        recent = jnp.concatenate([recent[1:], (0.5 * jnp.sum(at.rows * at.rows))[None]])
        allowance = jnp.max(recent) - recent[-1]
        # Newton's local phase, from the first point within the tolerance on (see _POLISHING_STEPS).
        polishing = stationary | (polished > 0)
        # A stall: the last step quiet (its decrease within what the objective resolves, the commands within their
        # rounding), and the stationarity not halved by it.
        stalls = jnp.where(quiet & (measure >= 0.5 * before), stalls + 1, 0)
        band = jnp.minimum(ACTIVE_BAND * at.size, jnp.sqrt(jnp.sum(at.projected * at.projected)))
        low, high = _held(at, u, lower, upper, band)
        free = ~(low | high)
        mask = free.astype(dtype)
        pair = mask[:, None] * mask[None, :]
        outside = jnp.diag(1.0 - mask)
        scaled = at.jacobian / at.columns
        # Newton's step on the free commands where the exact Hessian is positive definite there (near a minimiser,
        # where it converges quadratically whatever the rows' miss), the Gauss-Newton one where it is not.
        exact = at.hessian / (at.columns[:, None] * at.columns[None, :]) * pair + outside
        values = jnp.linalg.eigvalsh(exact)
        definite = values[0] > rcond * jnp.maximum(values[-1], 1.0)
        normal = (scaled.T @ scaled) * pair + outside + floor * eye
        # Levenberg's damping on the free commands, raised where the last step had to be cut (the model is not to
        # be trusted that far: a curved valley's floor) and lowered where it was taken whole; none in the local phase.
        damped = jnp.where(polishing, 0.0, damping) * jnp.diag(mask)
        newton = -dense_solve(jnp.where(definite, exact, normal) + damped, mask * at.gradient / at.columns)
        step = jnp.where(free, newton, -at.gradient / at.columns) / at.columns
        # The step's path bends with the rows (geodesic acceleration, Transtrum and Sethna): along ``u + alpha step +
        # alpha^2 bend / 2``, ``bend`` the least-squares fit of ``J bend = -r''``, the rows' second derivative along
        # the free step, so that the rows' linear model holds to second order along the path. In a curved valley's
        # floor the straight step leaves the valley (by its rows' curvature times its square) and is cut, and the
        # steps after it zigzag: two warm starts of the follower's derivative test took the whole cap, 3 of 7474
        # solves; with the bend none of the same 7474 did, and the most iterations went from 50 to 15. On an affine
        # model ``r''`` is zero, and so is the bend.
        moving = jnp.where(free, step, 0.0)
        bend = _bend(rows, u, moving, normal + damped, mask, scaled, at.columns)
        # Transtrum and Sethna's acceptance: where the bend is long beside the step, the rows' second-order model does
        # not hold that far, and the bend is left out: the step is the straight one, under its line search. Every bend
        # accepted, on Rosenbrock's valley in six commands the step and its bend cycled with the bend 5 to 20 times the
        # step, at a step of 1/32: 8 of 300 solves met the cap, against none with no bend. Their own rule rejects the
        # whole step and raises the damping; here, where the damping only follows the line search, that crawled: 171
        # of the same 300 met the cap.
        long = 2.0 * jnp.sqrt(jnp.sum((at.columns * bend) ** 2)) > ACCELERATION * jnp.sqrt(
            jnp.sum((at.columns * moving) ** 2))
        bend = jnp.where(long, 0.0, bend)
        # What the objective resolves, judged on the scale the rows came from: a row is a difference of terms (a
        # frame's position and its goal, a field and its target) whose rounding is eps of the terms, not of the row,
        # which cancels to far less (A0's follower: rows of 4e-4 from terms of 600). Each row's terms are bounded by
        # the problem's size (exactly so on an affine model, ``|A_i u| + |b_i| <= |r| + sum_k c_k |u_k|``), so its
        # rounding by eps of the size, and the decrease, ``sum (r - r') (r + r') / 2``, is uncertain by that times
        # ``sum |r + r'|``; this covers the rows' sum's rounding and the subtraction's.
        resolved = 4.0 * eps * at.size

        def trial(alpha: Any) -> tuple[Any, Any, Any]:
            free_move = u + alpha * step + 0.5 * alpha * alpha * bend
            moved = jnp.clip(free_move, lower, upper)
            predicted = alpha * jnp.sum(jnp.where(free, -at.gradient * step, 0.0)) + \
                jnp.sum(jnp.where(free, 0.0, at.gradient * (u - moved)))
            after = rows(moved)
            # The decrease as a difference of squares, whose subtraction's rounding is the rows' change's.
            change = at.rows - after
            decrease = 0.5 * jnp.sum(change * (at.rows + after))
            noise = resolved * jnp.sum(jnp.abs(at.rows) + jnp.abs(after))
            # A decrease within its noise passes too: near the floor a Newton step lowers the stationarity by more
            # than the objective can show, and the stall count above stops a step that does not. Newton's step on a
            # positive definite Hessian that no bound cuts may raise the objective above this point's, within the
            # largest of the last _MEMORY points' (a projected or Gauss-Newton step is judged against this point's
            # alone: one the projection cuts can cycle between two corners of the box on an allowance).
            allowed = jnp.where(definite & jnp.all(moved == free_move), allowance, 0.0)
            return moved, jnp.abs(decrease) <= noise, decrease + allowed + noise >= ARMIJO * predicted

        def halve(search: tuple[Any, Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any, Any]:
            alpha, tries, _, _, _ = search
            return alpha * 0.5, tries + 1, *trial(alpha * 0.5)

        def searching(search: tuple[Any, Any, Any, Any, Any]) -> Any:
            return (~search[4]) & (search[1] < BACKTRACKS) & ~polishing

        first = trial(jnp.ones((), dtype=dtype))
        _, _, moved, small, passed = lax.while_loop(searching, halve,
                                                    (jnp.ones((), dtype=dtype), jnp.zeros((), dtype=np.int32), *first))
        # The local phase: full steps, no line search, while a step is longer than the commands' rounding at the
        # problem's size, at most _POLISHING_STEPS; what is returned is the point within the tolerance whose step was
        # the shortest, the nearest the minimiser that the solve saw.
        length = jnp.max(at.columns * jnp.abs(first[0] - u))
        nearer = stationary & (length < shortest)
        best, shortest = jnp.where(nearer, u, best), jnp.where(nearer, length, shortest)
        polish = polishing & (length > 4.0 * eps * at.size) & (polished < _POLISHING_STEPS)
        # The cap is read here, at a point already weighed: one within the tolerance goes on to its local phase,
        # which ends by its own count.
        stop = jnp.where(polishing, ~polish, ~passed | (stalls >= STALLS) | (taken >= ITERATIONS))
        after = jnp.where(polish, first[0], jnp.where(stop, jnp.where(polishing, best, u), moved))
        damping = jnp.where(first[2], jnp.where(damping > _DAMPING, damping / _DAMPING_FACTOR, 0.0),
                            jnp.maximum(damping * _DAMPING_FACTOR, _DAMPING))
        # A quiet step, for the stall count: its decrease within what the objective resolves and the commands left
        # within their rounding (a step that moves them further is progress the objective cannot show).
        still = small & ~polishing & (jnp.max(at.columns * jnp.abs(moved - u)) <= 4.0 * eps * at.size)
        return (after, taken + jnp.where(stop, 0, 1).astype(np.int32), stalls, stop, measure, still,
                polished + polish.astype(np.int32), best, shortest, recent, damping)

    first = (start, jnp.zeros((), dtype=np.int32), jnp.zeros((), dtype=np.int32), jnp.zeros((), dtype=bool),
             jnp.asarray(np.inf, dtype=dtype), jnp.zeros((), dtype=bool), jnp.zeros((), dtype=np.int32), start,
             jnp.asarray(np.inf, dtype=dtype), jnp.full((_MEMORY,), -np.inf, dtype=dtype), jnp.zeros((), dtype=dtype))

    def going(carry: tuple[Any, ...]) -> Any:
        return ~carry[3]

    u, taken, *_ = lax.while_loop(going, iteration, first)
    at = _at(rows, u, lower, upper)
    stationarity = jnp.max(jnp.abs(at.projected)) / at.size
    low, high = _held(at, u, lower, upper, jnp.minimum(ACTIVE_BAND * at.size,
                                                       jnp.sqrt(jnp.sum(at.projected * at.projected))))
    full, resolution = _verdict(rows, at, u, ~(low | high), lower, upper)
    resolved = resolution <= RESOLVED
    converged = (stationarity <= tolerance) & (full | resolved)
    active = jnp.where(low, -1, jnp.where(high, 1, 0)).astype(np.int8)
    return u, (converged, stationarity, taken, active, resolved, resolution)


def _gradient(rows: Rows, u: Any) -> Any:
    """The gradient of ``1/2 |rows(u)|^2``, exact: its linearisation is the exact Hessian's."""
    return _jax.grad(_objective(rows))(u)


def _bounds(lower: Any, upper: Any) -> tuple[Any, Any]:
    """The bounds as the root reads them, each carrying its tangent: a held command's derivative is its bound's."""
    return lower, upper


def _root(rows: Rows, lower: Any, upper: Any) -> Callable[[Any], Any]:
    """``F(u) = c (u - P(u - grad / c^2))``, zero exactly at a stationary point in the box, ``c`` the columns' lengths
    held (a scaling that moves no root)."""

    def residual(u: Any) -> Any:
        jacobian = lax.stop_gradient(_jax.jacfwd(rows)(u))
        squared = jnp.sum(jacobian * jacobian, axis=0)
        columns = jnp.sqrt(jnp.where(squared > jnp.finfo(u.dtype).tiny, squared, 1.0))
        low, high = _bounds(lower, upper)
        return columns * (u - jnp.clip(u - _gradient(rows, u) / (columns * columns), low, high))

    return residual


def _equilibrated(matrix: Any) -> tuple[Any, Any, Any]:
    """``matrix`` with its rows and then its columns scaled to a largest entry of one, and the two scales:
    ``matrix = diag(rows) scaled diag(columns)``."""
    tiny = jnp.finfo(matrix.dtype).tiny
    largest = jnp.max(jnp.abs(matrix), axis=1)
    rows = jnp.where(largest > tiny, largest, 1.0)
    held = matrix / rows[:, None]
    largest = jnp.max(jnp.abs(held), axis=0)
    columns = jnp.where(largest > tiny, largest, 1.0)
    return held / columns[None, :], rows, columns


def _tangent(linear: Callable[[Any], Any], rhs: Any) -> Any:
    """The tangent system ``F'(u*) x = rhs``, ``K x K``, by one dense LU of its matrix, its rows and then its
    columns equilibrated (:func:`_equilibrated`)."""
    scaled, rows, columns = _equilibrated(_jax.jacfwd(linear)(rhs))
    return dense_solve(scaled, rhs / rows) / columns


def iterative_allocation(rows: Rows, start: Any, *, lower: Any, upper: Any, owner: str) -> IterativeAllocation:
    """The commands within ``[lower, upper]`` that minimise ``1/2 |rows(u)|^2``, by projected Newton from ``start``
    (see the module), with the derivative of the converged commands by the implicit function theorem. ``rows`` maps
    commands ``(K,)`` to rows ``(M,)`` in their dtype, each a miss over its scale, and may close over traced values,
    each of which carries its derivative; ``lower`` and ``upper`` ``(K,)`` or one value, possibly infinite, ``lower <=
    upper`` the caller's. ``owner`` is the switch's owner, which the calling entry declares
    (:func:`mime.control.allocation_switch`): the active set, evaluated once at the solution."""
    first = jnp.asarray(start)
    dtype = first.dtype if jnp.issubdtype(first.dtype, jnp.floating) else jnp.zeros(()).dtype
    first = first.astype(dtype)
    low = jnp.broadcast_to(jnp.asarray(lower, dtype=dtype), first.shape)
    high = jnp.broadcast_to(jnp.asarray(upper, dtype=dtype), first.shape)
    guess = jnp.clip(first, low, high)
    def solve(_: Any, initial: Any) -> tuple[Any, tuple[Any, Any, Any, Any]]:
        return _solved(rows, initial, low, high, dtype)

    with switch.not_collected():
        commands, (converged, stationarity, taken, active, resolved, resolution) = lax.custom_root(
            _root(rows, low, high), guess, solve, _tangent, has_aux=True)
    inside = jnp.minimum(commands - low, high - commands)
    switch.predicate(active == 0, owner=owner, resolution=lambda: switch.rounding(jnp.maximum(jnp.abs(low), jnp.abs(high))),
                     indicator=lambda: inside)
    made = rows(commands)
    return IterativeAllocation(commands, active, jnp.sum(made * made), converged, stationarity, taken, resolved,
                               resolution)


__all__ = ["ACCELERATION", "ACTIVE_BAND", "ARMIJO", "BACKTRACKS", "FLOOR", "ITERATIONS", "RANK", "RESOLVED", "STALLS",
           "TOLERANCE", "IterativeAllocation", "iterative_allocation"]
