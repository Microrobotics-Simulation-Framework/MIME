"""allocate()'s currents on a seeded set of requests, to an NPZ: ``python probe/allocate_dump.py <out.npz>``, run with
the tree under test's ``src`` first on ``PYTHONPATH`` (the base's allocate(), SciPy's BVLS on unweighted rows; or the
head's, the exact allocation). Per array, ``COUNT`` points uniform in the central half of the declared region, each
with allocate()'s three requests (a field alone, a gradient alone, both) of a uniform direction and a size log-uniform
from 0.1 to 4 times what the limit reaches, at a limit uniform from 5 to 50 A; and ``SMALL`` fields alone of 0.05 to 0.2
uT (where BVLS's absolute tolerance stops it short). Only NumPy draws the requests, so both trees ask the same; the rows
are written too, and the comparison checks they match."""

from __future__ import annotations

import sys

import numpy as np

import mime
from mime.data import asset
from mime.fields import CalibratedMultipole, allocate
from mime.fields._coil_array import calibration_of
from mime.units import A, T, T_per_m, m

SYSTEMS = ("navion_2", "cardiomag", "octomag")
COUNT, SMALL = 1000, 200
#: The gradient's five independent entries' places in the 3 x 3 (``mime.fields._coil_array._gradient5``).
FIVE = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2))


def gradient_of(five: np.ndarray) -> np.ndarray:
    g = np.zeros((3, 3))
    for value, (i, j) in zip(five, FIVE, strict=True):
        g[i, j] = g[j, i] = value
    g[2, 2] = -g[0, 0] - g[1, 1]
    return g


def unit(rng: np.random.Generator, size: int) -> np.ndarray:
    held = rng.uniform(-1.0, 1.0, size)
    return held / np.linalg.norm(held)


def main(out: str, count: int = COUNT, small_count: int = SMALL) -> None:
    arrays: dict[str, np.ndarray] = {}
    for index, system in enumerate(SYSTEMS):
        provider = CalibratedMultipole(calibration=asset(system))
        region = np.asarray(calibration_of(asset(system)).region, dtype=np.float64)
        centre, half = region.mean(axis=1), 0.25 * (region[:, 1] - region[:, 0])
        rng = np.random.default_rng(1000 + index)
        currents, kinds, limits, matrices, targets = [], [], [], [], []
        requests = [(kind, False) for _ in range(count) for kind in ("field", "gradient", "both")] + \
            [("field", True)] * small_count
        for kind, small in requests:
            actuation = provider.actuation(m(tuple(centre + rng.uniform(-1.0, 1.0, 3) * half)))
            limit = float(rng.uniform(5.0, 50.0))
            field_rows = np.asarray(actuation.field)
            five_rows = np.stack([np.asarray(actuation.gradient)[i, j] for i, j in FIVE])
            rows = {"field": field_rows, "gradient": five_rows, "both": np.concatenate([field_rows, five_rows])}[kind]
            size = float(rng.uniform(0.05e-6, 0.2e-6)) if small else \
                float(np.linalg.norm(rows, axis=0).sum()) * limit * float(np.exp(rng.uniform(np.log(0.1), np.log(4.0))))
            target = unit(rng, len(rows)) * size
            asked: dict[str, object] = {}
            if kind in ("field", "both"):
                asked["field"] = T(tuple(target[:3] + np.asarray(actuation.background)))
            if kind in ("gradient", "both"):
                asked["gradient"] = T_per_m(tuple(gradient_of(target[-5:]).reshape(-1)))
            found = allocate(actuation, limits=A(limit), **asked)
            currents.append(np.asarray(found.currents.value, dtype=np.float64))
            kinds.append("small" if small else kind)
            limits.append(limit)
            matrices.append(np.pad(rows, ((0, 8 - len(rows)), (0, 0))))
            targets.append(np.pad(target, (0, 8 - len(target))))
        arrays[f"{system}/currents"] = np.asarray(currents)
        arrays[f"{system}/kinds"] = np.asarray(kinds)
        arrays[f"{system}/limits"] = np.asarray(limits)
        arrays[f"{system}/matrices"] = np.asarray(matrices)
        arrays[f"{system}/targets"] = np.asarray(targets)
    np.savez(out, **arrays)
    print(f"{out}: mime from {mime.__file__}")


if __name__ == "__main__":
    main(sys.argv[1], *(int(v) for v in sys.argv[2:4]))
