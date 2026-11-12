"""G13, calibration fitted in MIME from a sphere grid: ``Study.fit`` recovers a camera's focal lengths, principal point and
two radial coefficients from the in-loop camera's own live render of a 2 x 2 grid of spheres on a tilted plane (48 x 36 pixels), with the
detector's numbers as the node's leaves; the fitted camera is labelled fitted, with its fit.

This **estimates** a model's parameters from the data that model generated (noiseless, so the optimum is the truth): it
bounds nothing about a real camera, and a real calibration's error is its own. The lines are committed first in
``predictions_camera_in_loop.json`` (``gate_13_calibration_fit``).

**Controls**: five starts, each a wrong declaration (the focal lengths 10 % off, the centre 4 px off, the distortion off)
are each recovered; and a frozen leaf (``CAMERA_PLANT=frozen-leaf``: ``fx`` left out of the fit, at its wrong start) leaves
a focal error a thousand times the line, caught every run."""

from __future__ import annotations

import numpy as np
import pytest
from calibration_cases import PATHS, TRUTH, detector, grid
from in_loop_cases import planted

from mime.study import Study
from mime.units import ms

NOISE = 0.29  # a quantisation step over root twelve: the declared noise of a noiseless synthetic image, in counts
LINES = {"float64": {"focal": 1e-4, "centre": 0.01, "distortion": 1e-4},
         "float32": {"focal": 1e-2, "centre": 0.5, "distortion": 1e-2}}
STARTS = [
    {"fx": 51.0, "fy": 41.0, "cx": 28.3, "cy": 14.3, "k1": -0.05, "k2": 0.0},
    {"fx": 42.0, "fy": 50.0, "cx": 20.3, "cy": 21.3, "k1": -0.2, "k2": 0.06},
    {"fx": 50.0, "fy": 50.0, "cx": 26.3, "cy": 20.3, "k1": -0.02, "k2": 0.05},
    {"fx": 42.0, "fy": 41.0, "cx": 21.3, "cy": 15.3, "k1": -0.2, "k2": 0.0},
    {"fx": 46.5, "fy": 45.75, "cx": 24.3, "cy": 17.3, "k1": 0.0, "k2": 0.0},
]


def target() -> np.ndarray:
    built = grid(detector()).build()
    built.run(ms(1))
    return np.asarray(built.current_state()["camera:"]["expected"], dtype=np.float64)


def test_the_grid_recovers_the_cameras_numbers_from_a_wrong_start(dtype):
    held = target()
    lines = LINES[dtype]
    for start in STARTS:
        exp = grid(detector(**start))
        names = [n for n in start if not (planted("frozen-leaf") and n == "fx")]
        fit = Study(exp).fit(lambda final: (final["camera:"]["expected"] - held).ravel(),
                             start={PATHS[n]: start[n] for n in names}, noise=NOISE, duration=ms(1), iterations=40,
                             acknowledged={"camera:": "a synthetic calibration, whose derivative the fit checks"})
        found = {n: float(np.asarray(fit.values[PATHS[n]]).reshape(-1)[0]) for n in names}
        found = {**{n: start[n] for n in start if n not in found}, **found}
        assert abs(found["fx"] - TRUTH["fx"]) <= lines["focal"] * TRUTH["fx"], (start, found, fit.ended)
        assert abs(found["fy"] - TRUTH["fy"]) <= lines["focal"] * TRUTH["fy"], (start, found, fit.ended)
        assert abs(found["cx"] - TRUTH["cx"]) <= lines["centre"], (start, found, fit.ended)
        assert abs(found["cy"] - TRUTH["cy"]) <= lines["centre"], (start, found, fit.ended)
        assert abs(found["k1"] - TRUTH["k1"]) <= lines["distortion"], (start, found, fit.ended)
        assert abs(found["k2"] - TRUTH["k2"]) <= lines["distortion"], (start, found, fit.ended)
        label = detector(**start).fitted(found, f"Study.fit of a sphere grid, ended {fit.ended}, {len(fit.taken)} "
                                                f"iterates, loss {fit.loss:.3g}")
        assert label.source.startswith("fitted: Study.fit of a sphere grid, ended ")
        assert detector(**start).source == "declared"
