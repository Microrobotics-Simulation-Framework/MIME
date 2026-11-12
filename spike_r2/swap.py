"""The swap test: the JAX bounded allocation law (mime.control._allocation) against allocate() (SciPy's BVLS, float64,
on the host) across Navion_2's operating range, limits inactive and active, in the run's dtype (JAX_ENABLE_X64).

Draws (seeded by DETECTION_SEED): points uniform in the central half of the declared region (Workspace_Dimensions);
for each, three problems allocate() states (field only, gradient only, both) and the loop's own (a force on a moment,
compared with SciPy's BVLS directly, the solver allocate() calls): wanted magnitudes log-uniform from a tenth to four
times what 25 A in every coil reaches, so that about half the draws hold a coil at its limit.

The tolerance, derived from the solve: both are the exact least squares of one active set's free columns, so they differ
by rounding, which the perturbation bound for least squares puts at ``eps (2 kappa + kappa^2 |r| / (|A| |x|))`` of the
free currents (``kappa`` the free columns' condition after unit scaling), plus the input's rounding into the dtype; the
test allows ``SLACK`` times that, in units of the limit, and the residual's excess over SciPy's at the same relative size.
Near an active-set boundary the two may pick neighbouring sets; the solution is continuous there, so the same bound holds.

SPIKE_PLANT=limits: the law is handed limits 1 % looser (a planted error the test must catch on every run)."""

from __future__ import annotations

import json
import os
import sys

import jax
import numpy as np
from scipy.optimize import lsq_linear

from mime.control._allocation import bounded_lsq, force_rows, gradient5
from mime.data import asset
from mime.fields import CalibratedMultipole, allocate
from mime.fields._coil_array import calibration_of
from mime.units import A, T, T_per_m, m

jnp = jax.numpy
SEED = int(os.environ.get("DETECTION_SEED", "0") or 0)
PLANT = os.environ.get("SPIKE_PLANT", "")
DRAWS = int(os.environ.get("SWAP_DRAWS", "300"))
LIMIT = 25.0
SLACK = 16.0
DTYPE = np.dtype(jnp.zeros(()).dtype)
EPS = float(np.finfo(DTYPE).eps)


def symmetric_traceless(rng: np.random.Generator) -> np.ndarray:
    g = rng.normal(size=(3, 3))
    g = 0.5 * (g + g.T)
    return g - np.trace(g) / 3.0 * np.eye(3)


def bound_of(matrix: np.ndarray, x: np.ndarray, r: float) -> float:
    """The perturbation bound for the free columns of ``x`` (those within the limit), in units of the limit."""
    free = np.abs(x) < LIMIT * (1 - 1e-9)
    if not free.any():
        return EPS * 4
    cols = matrix[:, free] / np.linalg.norm(matrix[:, free], axis=0)
    s = np.linalg.svd(cols, compute_uv=False)
    kappa = s[0] / s[-1]
    xs = np.linalg.norm(x[free] * np.linalg.norm(matrix[:, free], axis=0))
    return EPS * (2 * kappa + kappa ** 2 * r / max(xs, 1e-300)) * max(np.max(np.abs(x)), LIMIT) / LIMIT + EPS


law = jax.jit(lambda a, b, l: bounded_lsq(a, b, l))


def check(seed: int, draws: int, loosen: float) -> tuple[float, bool, dict, list]:
    """The swap test's worst ratio over ``draws`` draws of ``seed``, the law handed limits ``1 + loosen`` times the
    declared ones; whether every current was finite; the summary and the rows."""
    rng = np.random.default_rng(1000 + seed)
    provider = CalibratedMultipole(calibration=asset("navion_2"))
    region = calibration_of(asset("navion_2")).region
    centre, half = region.mean(axis=1), 0.25 * (region[:, 1] - region[:, 0])
    worst, rows, unscaled = 0.0, [], []
    active = 0
    for _ in range(draws):
        point = centre + rng.uniform(-1, 1, 3) * half
        act = provider.actuation(m(tuple(point)))
        reach_b = np.linalg.norm(act.field, axis=0).sum() * LIMIT
        reach_g = np.linalg.norm(act.matrix()[3:], axis=0).sum() * LIMIT
        b = rng.normal(size=3); b *= reach_b * np.exp(rng.uniform(np.log(0.1), np.log(4.0))) / np.linalg.norm(b)
        g = symmetric_traceless(rng); g *= reach_g * np.exp(rng.uniform(np.log(0.1), np.log(4.0))) / np.linalg.norm(g)
        moment = rng.normal(size=3); moment *= 5e-4 / np.linalg.norm(moment)
        problems = []
        for kind, kw in (("field", {"field": T(tuple(b))}), ("gradient", {"gradient": T_per_m(tuple(g.ravel()))}),
                         ("both", {"field": T(tuple(b)), "gradient": T_per_m(tuple(g.ravel()))})):
            held = allocate(act, limits=A(LIMIT), **kw)
            x_ref = np.asarray(held.currents.value, dtype=np.float64)
            matrix = {"field": act.field, "gradient": act.matrix()[3:], "both": act.matrix()}[kind]
            target = {"field": b - act.background, "gradient": act.matrix()[3:] @ np.zeros(3) * 0 + np.array(
                [g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2]]), "both": np.concatenate(
                [b - act.background, [g[0, 0], g[0, 1], g[0, 2], g[1, 1], g[1, 2]]])}[kind]
            problems.append((kind, matrix, target, x_ref, held.at_limit))
        mf = np.einsum("i,ijk->jk", moment, act.gradient)
        reach_f = np.linalg.norm(mf, axis=0).sum() * LIMIT
        f = rng.normal(size=3); f *= reach_f * np.exp(rng.uniform(np.log(0.1), np.log(4.0))) / np.linalg.norm(f)
        # SciPy's BVLS judges optimality by an absolute tolerance (1e-10) on the gradient, which force rows in N/A
        # (about 1e-6) reach long before the optimum: it is handed the rows scaled to order one, as allocate()'s own
        # rows in T and T/m are. The unscaled call's miss is recorded beside (FORCE_UNSCALED).
        size = float(np.max(np.abs(mf)))
        solved = lsq_linear(mf / size, f / size, bounds=(-LIMIT, LIMIT), method="bvls")
        raw = lsq_linear(mf, f, bounds=(-LIMIT, LIMIT), method="bvls")
        unscaled.append(float(np.linalg.norm(mf @ raw.x - f) / max(np.linalg.norm(mf @ solved.x - f), 1e-300)))
        problems.append(("force", mf, f, np.asarray(solved.x), tuple(np.abs(solved.x) >= LIMIT * (1 - 1e-9))))
        for kind, matrix, target, x_ref, at_limit in problems:
            limits = np.full(3, LIMIT * (1.0 + loosen))
            got = law(jnp.asarray(matrix, dtype=DTYPE), jnp.asarray(target, dtype=DTYPE), jnp.asarray(limits, dtype=DTYPE))
            x = np.asarray(got.currents, dtype=np.float64)
            r_ref = float(np.linalg.norm(matrix @ x_ref - target))
            bound = SLACK * bound_of(matrix, x_ref, r_ref)
            miss = float(np.max(np.abs(x - x_ref))) / LIMIT
            ratio = miss / bound
            r_got = float(np.linalg.norm(matrix @ x - target))
            r_ratio = (r_got - r_ref) / (SLACK * EPS * (np.linalg.norm(matrix) * LIMIT * 3 + np.linalg.norm(target)))
            active += any(at_limit)
            worst = max(worst, ratio, r_ratio)
            rows.append({"kind": kind, "miss_over_limit": miss, "bound": bound, "ratio": ratio, "residual_excess_ratio": r_ratio,
                         "at_limit": [bool(a) for a in at_limit], "finite": bool(np.all(np.isfinite(x)))})
    finite = all(r["finite"] for r in rows)
    summary = {"dtype": str(DTYPE), "seed": seed, "plant": loosen, "problems": len(rows), "with_a_coil_at_limit": active,
               "worst_ratio": worst, "finite": finite,
               "scipy_unscaled_force_residual_over_scaled_max": max(unscaled),
               "scipy_unscaled_force_short_share": float(np.mean(np.asarray(unscaled) > 1 + 1e-6)),
               "worst_by_kind": {k: max(r["ratio"] for r in rows if r["kind"] == k) for k in ("field", "gradient", "both", "force")},
               "largest_miss_over_limit": {k: max(r["miss_over_limit"] for r in rows if r["kind"] == k)
                                           for k in ("field", "gradient", "both", "force")}}
    return worst, finite, summary, rows


def limits_oracle(size: float, seed: int) -> bool:
    """discriminating_range.py's oracle: whether the swap test (30 draws) flags limits loosened by ``size``."""
    worst, finite, _, _ = check(seed, 30, size)
    return not finite or worst > 1.0


def main() -> int:
    worst, finite, summary, rows = check(SEED, DRAWS, 0.01 if PLANT == "limits" else 0.0)
    print(json.dumps(summary))
    if not finite or worst > 1.0:
        for row in sorted(rows, key=lambda r: -max(r["ratio"], r["residual_excess_ratio"]))[:3]:
            print(json.dumps(row))
    out = os.environ.get("SWAP_OUT")
    if out:
        with open(out, "w") as handle:
            json.dump({"summary": summary, "rows": rows}, handle)
    return 0 if finite and worst <= 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
