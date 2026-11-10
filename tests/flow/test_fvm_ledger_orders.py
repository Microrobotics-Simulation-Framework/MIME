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
not falling with the step or the tolerance. Over-budget cases go on the duration allowlist, never under ``slow``.

**The resolution** (measured on this case, three seeds; ``N_BE`` and ``N_conv`` alike in both dtypes): a physical loss
of ``k`` times the term's own coarse sum, moved from the viscous row into the term, leaves the band on one seed of the
three from ``k = 0.1`` (``N_BE`` at 1.58, ``N_conv`` at 1.45), which fails the test, and on all three from about 0.6
for ``N_BE`` (``N_conv`` keeps one seed in band there); ``N_Krylov`` stays in band at 0.03 of its looser sum and leaves
on all three at 0.3. Against the viscous loss these are about 0.8 % of it moved into ``N_BE``, 2.5e-6 of it into
``N_conv``, and 1.4e-7 (float64) or 3.3e-4 (float32) into ``N_Krylov``; the control moves all of it. **A gift a term's
own formula books** (convection's, at mixed time levels) is booked under that term, where the step identity cannot see
it: these tests, on this case, are the only check of it, at these resolutions."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fvm_cases import METERED, eps, in_dtype, tube_node

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
#: The tolerances of the Krylov arms, a hundred apart, each dtype's: float32's above its rounding floor; the dt arms take
#: the tighter.
TOLERANCES = {"float64": (1e-6, 1e-8), "float32": (1e-3, 1e-5)}
#: The drive, constant: the tube's Poiseuille body force.
DRIVE = 8.0


def moved_into(term: str):
    """The plant: the viscous loss written into ``term``'s formula, the viscous row emptied by as much."""
    real = ImplicitFlowNode.energy_terms

    def planted(self, *args: Any) -> dict[str, Any]:
        terms = dict(real(self, *args))
        terms[term] = terms[term] + terms["viscous"]
        terms["viscous"] = 0.0 * terms["viscous"]
        return terms

    return planted


def arms(dtype: str, seeds: range) -> list[dict[str, float]]:
    """For each seed, every arm from its warmed state: the terms' summed magnitudes (``N_BE`` is never negative;
    ``N_conv``, ``dt <C(u), u - u'>`` with ``<C(u), u>`` zero to rounding on a divergence-free field, and ``N_Krylov``
    change sign) at ``dt`` and ``dt / 2``, ``N_Krylov``'s at the two tolerances, the viscous loss, and the identity's worst
    step over its floor. One program for every arm: the step, the run's length and the tolerance are its arguments."""
    loose, tight = TOLERANCES[dtype]
    with in_dtype(dtype):
        node = tube_node(8, length_cells=4, dt=STEP, convection=True, tolerance=tight, ledger=METERED)
        p0 = node._step_constants(None)  # noqa: SLF001
        held = books.deliveries(node.constants())
        given = {books.law_input(d.field, index): jnp.asarray([0.0, 0.0, DRIVE]) if d.field == "body_force"
                 else jnp.zeros(7) for index, d in enumerate(held) if d.kind == books.LOAD}

        def arm(state, dt, steps, tolerance):
            p = {**p0, "tolerance": tolerance}

            def body(_, carry):
                s, sums = carry
                s = node.advance(p, s, given, dt)
                terms = jnp.abs(s[f"{books.RECORDS}step_terms"])
                return s, sums + jnp.concatenate([terms, -s[f"{books.RECORDS}step_work"][1:2]])

            ended, sums = jax.lax.fori_loop(0, steps, body, (state, jnp.zeros(len(TERMS) + 1, dtype=dt.dtype)))
            return ended, sums

        compiled = jax.jit(arm)
        dtype_of = jnp.asarray(0.0).dtype
        found = []
        for seed in seeds:
            start = node.initial_state()
            swirl = cellular(node, seed)
            begin = {**start, **{f"velocity_{c}": jnp.asarray(swirl[f"initial_velocity_{c}"], dtype=dtype_of)
                                 for c in "xyz"}}
            begin = books.with_records(node, books.unrecorded(begin))
            warm, _ = compiled(begin, jnp.asarray(WARM_STEP, dtype=dtype_of), WARM_STEPS,
                               jnp.asarray(tight, dtype=dtype_of))
            fresh = books.with_records(node, books.unrecorded({**start, **{f"velocity_{c}": warm[f"velocity_{c}"]
                                                                            for c in "xyz"}}))
            sums, worst = {}, 0.0
            for label, dt, tolerance in (("coarse", STEP, tight), ("fine", STEP / 2, tight), ("loose", STEP, loose)):
                ended, summed = compiled(fresh, jnp.asarray(dt, dtype=dtype_of), int(round(TIME / dt)),
                                         jnp.asarray(tolerance, dtype=dtype_of))
                sums[label] = np.asarray(summed, dtype=np.float64)
                worst = max(worst, float(ended[f"{books.RECORDS}identity_worst"]) / (books.IDENTITY_ULPS * eps(dtype)))
            found.append({**{f"{name}_ratio": float(sums["coarse"][i] / sums["fine"][i])
                             for name, i in TERMS.items() if name != "krylov"},
                          "krylov_ratio": float(sums["loose"][TERMS["krylov"]] / sums["coarse"][TERMS["krylov"]]),
                          "viscous": float(sums["coarse"][-1]), "worst": worst})
        return found


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_each_numerical_term_falls_at_its_order(dtype):
    for found in arms(dtype, range(3)):
        assert found["worst"] <= 1.0
        for name in ("backward_euler", "convection"):
            assert FIRST_ORDER[0] <= found[f"{name}_ratio"] <= FIRST_ORDER[1], (name, found)
        assert found["krylov_ratio"] >= KRYLOV_FALL, found


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("term", ["backward_euler", "convection", "krylov"])
def test_control_the_viscous_loss_written_into_a_term_leaves_its_order(dtype, term, monkeypatch):
    """The identity still closes with the loss moved, and the term's ratio leaves its band on every seed."""
    monkeypatch.setattr(ImplicitFlowNode, "energy_terms", moved_into(term))
    for found in arms(dtype, range(3)):
        assert found["worst"] <= 1.0
        if term == "krylov":
            assert found["krylov_ratio"] < KRYLOV_FALL, found
        else:
            assert not FIRST_ORDER[0] <= found[f"{term}_ratio"] <= FIRST_ORDER[1], found
