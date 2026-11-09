"""CI check (b) (row #301, R-19): each numerical term the flow's step identity books falls at its order, so a physical
loss written into an ``N`` formula, which the identity books whatever it is called and closes all the same, is caught.

On a smooth swirl in the periodic tube with its Poiseuille drive (IMEX Euler, convection on, D/h 8, a fixed time of
0.05 in units of D and nu), each term's magnitude summed over the run:

- ``N_BE`` and ``N_conv`` at ``dt`` and ``dt / 2``: first order, their ratio within :data:`FIRST_ORDER`;
- ``N_Krylov`` at two tolerances a hundred apart: it falls with the tolerance, its ratio at least
  :data:`KRYLOV_FALL`.

The projection's term (``N_proj``) is not booked on the periodic box: ``tests/flow/test_fvm_ledger.py`` shows it zero to
rounding at two tolerances. The Windkessels' ``N_WK`` comes with them.

Every arm starts from one warmed state (the swirl and its drive after a short run at a quarter of the step), so the
stiff interior's start, damped in a step whatever the step, is not in the sums.

**The control, each failing every run**: the viscous loss moved into the term (written into its formula, the viscous
row emptied by as much): the identity still closes, and the term's ratio leaves its band, a dissipation per unit time
not falling with the step or the tolerance. Over-budget cases go on the duration allowlist, never under ``slow``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fvm_cases import METERED, eps, in_dtype, run, tube_node

from mime.core import ledger as books
from mime.nodes.flow.fvm import ImplicitFlowNode

#: The fixed time each term is summed over (D = 1, nu = 1: about a fifth of the swirl's decay time).
TIME = 0.05
#: The band a first-order term's sum at dt over its sum at dt / 2 lies in: R-19's, as the rotor's exchange test.
FIRST_ORDER = (1.6, 2.4)
#: The least ratio of ``N_Krylov``'s summed magnitude at a tolerance to its sum at a hundredth of it.
KRYLOV_FALL = 10.0
#: Each term's index in the records (``ImplicitFlowNode.LEDGER_TERMS``).
TERMS = {name: index for index, name in enumerate(ImplicitFlowNode.LEDGER_TERMS)}


def cellular(node: ImplicitFlowNode, seed: int) -> dict[str, Any]:
    """A smooth swirl across the tube, uniform along it (a cellular flow of the box's width with a drawn phase in
    each component), projected: convection does work at mixed time levels, and every mode is resolved in time."""
    rng = np.random.default_rng(seed)
    p = node._step_constants(None)  # noqa: SLF001
    grid = node.grid(p)
    a, b, c = rng.uniform(0.0, 2.0 * np.pi, 3)
    width = grid.cells[0] * grid.spacing
    points = [np.asarray(grid.face_points(k, "float64")) for k in range(3)]
    u = np.sin(2.0 * np.pi * points[0][..., 1] / width + a)
    v = np.sin(2.0 * np.pi * points[1][..., 0] / width + b)
    w = np.sin(2.0 * np.pi * (points[2][..., 0] + points[2][..., 1]) / width + c)
    swirl = grid.project((u, v, w))
    return {f"initial_velocity_{k}": np.asarray(swirl[i]) for i, k in enumerate("xyz")}


#: The step of the dt arms and of the tolerance arms, and the warm-up's: a twentieth of the coarse arm's run at a
#: quarter of its step, which the stiff interior and the drive's start settle in (the penalty damps the solid's start in
#: a step whatever the step is, which no order describes).
STEP, WARM_STEP, WARM_STEPS = 0.0025, 0.000625, 20


def warmed(dtype: str, seed: int) -> dict[str, Any]:
    """The swirl and its drive after the warm-up: the start of every arm."""
    with in_dtype(dtype):
        start = cellular(tube_node(8, length_cells=4, dt=WARM_STEP), seed)
        state = run(tube_node(8, length_cells=4, dt=WARM_STEP, convection=True, **start), WARM_STEPS)
    return {f"initial_velocity_{c}": np.asarray(state[f"velocity_{c}"]) for c in "xyz"}


def summed(dtype: str, dt: float, start: dict[str, Any], **settings: Any) -> dict[str, float]:
    """Each term's summed magnitude over :data:`TIME` (``N_BE`` is never negative; ``N_conv``, ``dt <C(u), u - u'>``
    with ``<C(u), u>`` zero to rounding on a divergence-free field, and ``N_Krylov`` change sign), the viscous loss's
    sum, and the identity's worst step over its floor."""
    with in_dtype(dtype):
        node = tube_node(8, length_cells=4, dt=dt, convection=True, ledger=METERED, **start, **settings)
        state, records = run(node, int(round(TIME / dt)), record=lambda s: (s[f"{books.RECORDS}step_terms"],
                                                                             s[f"{books.RECORDS}step_work"]))
    terms = np.abs(np.asarray(records[0], dtype=np.float64))
    work = np.asarray(records[1], dtype=np.float64)
    return {**{name: float(terms[:, index].sum()) for name, index in TERMS.items()},
            "viscous": float(-work[:, 1].sum()),
            "worst": float(state[f"{books.RECORDS}identity_worst"]) / (books.IDENTITY_ULPS * eps(dtype))}


def moved_into(term: str):
    """The plant: the viscous loss written into ``term``'s formula, the viscous row emptied by as much."""
    real = ImplicitFlowNode.energy_terms

    def planted(self, *args: Any) -> dict[str, Any]:
        terms = dict(real(self, *args))
        terms[term] = terms[term] + terms["viscous"]
        terms["viscous"] = 0.0 * terms["viscous"]
        return terms

    return planted


#: The tolerances of the Krylov arms, a hundred apart, each dtype's: float32's above its rounding floor.
TOLERANCES = {"float64": (1e-6, 1e-8), "float32": (1e-3, 1e-5)}


def dt_ratios(dtype: str, seed: int) -> tuple[dict[str, float], dict[str, float]]:
    start = warmed(dtype, seed)
    coarse, fine = summed(dtype, STEP, start), summed(dtype, STEP / 2, start)
    return {name: coarse[name] / fine[name] for name in ("backward_euler", "convection")}, coarse


def krylov_ratio(dtype: str, seed: int) -> tuple[float, dict[str, float]]:
    start = warmed(dtype, seed)
    loose, tight = (summed(dtype, STEP, start, tolerance=t) for t in TOLERANCES[dtype])
    return loose["krylov"] / tight["krylov"], loose


@pytest.mark.parametrize("seed", range(3))
def test_n_be_and_n_conv_fall_at_first_order_in_dt(dtype, seed):
    ratios, coarse = dt_ratios(dtype, seed)
    assert coarse["worst"] <= 1.0
    for name, ratio in ratios.items():
        assert FIRST_ORDER[0] <= ratio <= FIRST_ORDER[1], (name, ratio)


@pytest.mark.parametrize("seed", range(3))
def test_n_krylov_falls_with_the_tolerance(dtype, seed):
    ratio, loose = krylov_ratio(dtype, seed)
    assert loose["worst"] <= 1.0
    assert ratio >= KRYLOV_FALL, ratio


@pytest.mark.parametrize("term", ["backward_euler", "convection"])
@pytest.mark.parametrize("seed", range(3))
def test_control_the_viscous_loss_written_into_a_dt_term_leaves_its_band(dtype, seed, term, monkeypatch):
    """The identity still closes with the loss moved, and the term's ratio leaves the first-order band."""
    monkeypatch.setattr(ImplicitFlowNode, "energy_terms", moved_into(term))
    ratios, coarse = dt_ratios(dtype, seed)
    assert coarse["worst"] <= 1.0
    assert not FIRST_ORDER[0] <= ratios[term] <= FIRST_ORDER[1], ratios[term]


@pytest.mark.parametrize("seed", range(3))
def test_control_the_viscous_loss_written_into_n_krylov_stops_it_falling(dtype, seed, monkeypatch):
    monkeypatch.setattr(ImplicitFlowNode, "energy_terms", moved_into("krylov"))
    ratio, loose = krylov_ratio(dtype, seed)
    assert loose["worst"] <= 1.0
    assert ratio < KRYLOV_FALL, ratio
