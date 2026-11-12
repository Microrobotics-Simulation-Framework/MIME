"""The single-``where`` control at the allocation (camera spike R2): a rank-deficient array, Navion_2's three coils at a
drawn point with a drawn moment and a fourth coil that duplicates one of them scaled (two coils wired to one supply:
dependent columns), asked for a force at an active-set boundary (one coil a thousandth inside its limit in the wanted
currents, three hundredths in float32). The 81 active sets include those with both duplicates free, whose triangle is singular.

The check: the derivative of the made force ``A i(b)`` (unique, whichever minimiser is picked) in the wanted force
``b``, by reverse mode, finite and equal to central differences (``h`` a millionth of ``|b|`` in float64, a thousandth in
float32; the line ``3 tau + 3 rho`` of ``gradcheck.py``, here with ``rho = 3 eps |A i| / h``, the made force's rounding).
SPIKE_VARIANT=single_where drops the first ``where``: the singular sets' NaN reaches the derivative (``0 * NaN``), and the
check fails. Exit 1 caught, 0 not."""

from __future__ import annotations

import json
import os
import sys

import jax
import numpy as np

from mime.control._allocation import bounded_lsq
from mime.data import asset
from mime.fields import CalibratedMultipole
from mime.fields._coil_array import calibration_of
from mime.units import m

jnp = jax.numpy
SEED = int(os.environ.get("DETECTION_SEED", "0") or 0)
VARIANT = os.environ.get("SPIKE_VARIANT", "double_where")
DTYPE = np.dtype(jnp.zeros(()).dtype)
EPS = float(np.finfo(DTYPE).eps)
LIMIT = 25.0
#: How far inside its limit the coil at the boundary is wanted, and the relative step of the central differences.
MARGIN = {np.dtype(np.float64): 1e-3, np.dtype(np.float32): 3e-2}
STEP = {np.dtype(np.float64): 1e-6, np.dtype(np.float32): 1e-3}


def main() -> int:
    rng = np.random.default_rng(9000 + SEED)
    region = calibration_of(asset("navion_2")).region
    centre, half = region.mean(axis=1), 0.25 * (region[:, 1] - region[:, 0])
    act = CalibratedMultipole(calibration=asset("navion_2")).actuation(m(tuple(centre + rng.uniform(-1, 1, 3) * half)))
    moment = rng.normal(size=3)
    moment *= 5e-4 / np.linalg.norm(moment)
    rows = np.einsum("i,ijk->jk", moment, act.gradient)
    twin = int(rng.integers(3))
    # Scaled by a power of two, so that the duplicate stays exactly dependent in either dtype: a scale that rounds leaves
    # the scaled columns a rounding apart, the triangle's diagonal tiny but not zero and the singular sets' solve finite
    # (measured in float32 with a drawn scale: two draws of three gave no NaN, the single where then harmless).
    matrix = np.concatenate([rows, rows[:, twin:twin + 1] * float(rng.choice([0.5, 1.0, 2.0]))], axis=1)
    at_limit = int(rng.choice([k for k in range(3) if k != twin]))
    wanted = rng.uniform(-0.6, 0.6, 4) * LIMIT
    # Just inside the boundary, by more than the central difference reaches: exactly on it the central difference is
    # the mean of the two one-sided slopes and reverse mode one of them (measured: a ratio of 1e8 in float64).
    wanted[at_limit] = LIMIT * rng.choice([-1.0, 1.0]) * (1.0 - MARGIN[DTYPE])
    target = matrix @ wanted
    a, b = jnp.asarray(matrix, dtype=DTYPE), jnp.asarray(target, dtype=DTYPE)
    limits = jnp.full((4,), LIMIT, dtype=DTYPE)

    def made(b_: jax.Array) -> jax.Array:
        return a @ bounded_lsq(a, b_, limits, variant=VARIANT).currents

    jac = np.asarray(jax.jacrev(made)(b), dtype=np.float64)
    size = float(np.linalg.norm(target))
    h = STEP[DTYPE] * size
    cd = np.zeros((3, 3))
    tau = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1.0

        def diff(step: float) -> np.ndarray:
            up = np.asarray(made(jnp.asarray(target + step * e, dtype=DTYPE)), dtype=np.float64)
            down = np.asarray(made(jnp.asarray(target - step * e, dtype=DTYPE)), dtype=np.float64)
            return (up - down) / (2 * step)

        c1, c2 = diff(h), diff(h / 2)
        cd[:, j], tau[:, j] = c2, np.abs(c1 - c2) / 3
    rho = 3 * EPS * size / (h / 2)
    line = 3 * tau + 3 * rho
    finite = bool(np.all(np.isfinite(jac)))
    ok = finite and bool(np.all(np.abs(jac - cd) <= line))
    print(json.dumps({"seed": SEED, "dtype": str(DTYPE), "variant": VARIANT, "finite": finite, "ok": ok,
                      "worst_ratio": float(np.max(np.abs(jac - cd) / line)) if finite else None,
                      "selected_signs": np.asarray(bounded_lsq(a, b, limits).signs).tolist(), "twin": twin,
                      "at_limit": at_limit}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
