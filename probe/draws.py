"""The swap test's draws (``tests/control/allocation_draws.problems``) by a seeded NumPy generator, for the probe's
measurements: a kind uniform among the four, a point uniform in the central half of the declared region, a limit
uniform from 5 to 50 A, a moment of 0.1 to 1 mA m^2 in a uniform direction for a force, and a wanted value of a uniform
direction whose size is log-uniform from 0.1 to 4 times what the limit reaches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "control"))

from allocation_draws import (FIELD_SCALE, GRADIENT_SCALE, KINDS, Problem, five, provider, reach_of,  # noqa: E402
                              region)
from mime.units import m  # noqa: E402


def _direction(rng: np.random.Generator, size: int) -> np.ndarray:
    held = rng.uniform(-1.0, 1.0, size)
    return held / np.linalg.norm(held)


def draw(system: str, rng: np.random.Generator, kind: str | None = None) -> Problem:
    kind = kind if kind is not None else str(rng.choice(KINDS))
    centre, half = region(system)
    actuation = provider(system).actuation(m(tuple(centre + rng.uniform(-1.0, 1.0, 3) * half)))
    limit = float(rng.uniform(5.0, 50.0))
    if kind == "force":
        moment = _direction(rng, 3) * float(rng.uniform(1e-4, 1e-3))
        matrix = np.einsum("i,ijk->jk", moment, actuation.gradient)
        scales = np.full(3, float(np.max(np.linalg.norm(matrix, axis=0))))
    else:
        matrix = {"field": actuation.field, "gradient": five(actuation.gradient), "both": actuation.matrix()}[kind]
        scales = np.asarray({"field": [FIELD_SCALE] * 3, "gradient": [GRADIENT_SCALE] * 5,
                             "both": [FIELD_SCALE] * 3 + [GRADIENT_SCALE] * 5}[kind])
    wanted = _direction(rng, len(matrix)) * reach_of(matrix, limit) * float(np.exp(rng.uniform(np.log(0.1), np.log(4.0))))
    return Problem(system, kind, np.asarray(matrix, dtype=np.float64), wanted, scales, limit)
