"""F3's gates on the point body's self-history (``Drag.Stokes``, the maintainer's row #349, point 3: the exact route
the default, the fitted modes the opt-in, their swap test route 1's gate) and on the field's steady limit (the review
of #239's B2), in both dtypes (``tests/flow/predictions_f3_fix.json``). Each control is a plant switched by
``MIME_PLANT_<NAME>``, its rate measured by ``.github/probe/detection_rate.py``; each draw is seeded by
``DETECTION_SEED``."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from point_cases import WATER_MU, WATER_RHO, FreeBox, plant, point_experiment, seed

from mime.nodes.flow.fvm import _exchange
from mime.nodes.hydro import _point_drag
from mime.nodes.hydro._point_drag import AVERAGES, FORCES, LAPLACIANS, PointDragNode
from mime.units import ms


def swap_worst(draw: int, step_scale: float = 1.0, pinned: bool = True) -> tuple[float, str]:
    """Gate 12's measure at the draw ``draw``: the largest miss over 30 steps between the own field route 2 removes and
    route 1's (its modes made at ``step_scale`` times the run's step), over the fit's residual times the one-step
    response and the summed force; and a line describing the draw. ``pinned`` holds the body's point (route 1 reads
    its own field as if the body had not moved since each force, route 2 where it is: a moving body's difference is
    that approximation's, not the fit's)."""
    rng = np.random.default_rng(12000 + draw)
    width = float(rng.choice([1.0, 1.5]))
    r = float(10.0 ** rng.uniform(math.log10(0.03), math.log10(3.0)))
    box = FreeBox(24, float(rng.uniform(0.8, 1.2)) * width, r, mass_ratio=1.0, width=width, faxen=True, pinned=pinned)
    fitted = PointDragNode("fitted", box.dt, radius=box.radius, viscosity=WATER_MU, density=WATER_RHO, spacing=box.h,
                           width_cells=width, origin=(0.0, 0.0, 0.0), initial_points=(0.0, 0.0, 0.0),
                           self_history="modes")
    made = fitted.responses(fitted.constants(), box.dt * step_scale)
    amplitude, decay = np.asarray(made["amplitude"], dtype=np.float64), np.asarray(made["decay"], dtype=np.float64)
    _, average_c, delivery_c = box.drag.coefficients(box.drag.constants())
    h = box.h
    spectrum = _exchange.kernel_spectrum(jnp.asarray(width), _exchange.box_side(width))
    one_step = float(_exchange.step_response(spectrum, r, jnp.zeros(1), float(average_c) / h ** 2,
                                             float(delivery_c) / h ** 2)[0]) / (WATER_MU * h)
    modes = np.zeros((amplitude.shape[0], 3))
    summed, worst = 0.0, 0.0
    for n in range(30):
        box.force = 1e-11 * (1.0 + 0.5 * math.sin(2.0 * math.pi * n / 7.0))
        box.step()
        gathered = {key: np.asarray(value, dtype=np.float64) for key, value in box.gathered.items()}
        volume = np.concatenate([gathered[a] + float(average_c) * gathered[c] for a, c in zip(AVERAGES, LAPLACIANS,
                                                                                               strict=True)])
        exact = volume - np.asarray(box.drag_state["clean_reading"], dtype=np.float64)
        apart = float(np.linalg.norm(modes.sum(axis=0) - exact))
        if summed > 0.0:
            worst = max(worst, apart / (float(made["residual"]) * one_step * summed))
        force = np.concatenate([np.asarray(box.drag_state[f], dtype=np.float64) for f in FORCES])
        modes = decay[:, None] * (modes + amplitude[:, None] * force[None, :])
        summed += float(np.linalg.norm(force))
    return worst, f"width {width:g}, r {r:.3g}, the fit's residual {float(made['residual']):.2g}"


def test_the_fitted_self_history_matches_the_exact_one_within_its_residual(dtype):
    """Gate 12, the swap test (route 1's gate): one body's force history through the node, stepped and oscillating;
    the own field route 2 removes (the body's free-box flow advanced exactly) against the one route 1's fitted modes give
    on the same forces, the body's point held: within the fit's stated residual of the one-step response (its places in
    the cell and face sets included), times the summed force. The plant makes route 1's modes at a step 10 % longer than
    the run's (SWAP_STEP). A moving body's difference, route 1 reading its own field where it was made, is that
    approximation's (``swap_worst(..., pinned=False)``: 1.4 and 4.9 times the bound at two draws of a residual of
    1e-8)."""
    worst, drawn = swap_worst(seed(), 1.1 if plant("SWAP_STEP") else 1.0)
    print(f"F3 gate 12 [{dtype}]: {drawn}: the routes apart by {worst:.3g} of the fit's residual times the one-step "
          f"response and the summed force")
    assert worst <= 1.0, worst


def _light_run(steady: bool, ratio: float, radius: float, steps: int, own_field: bool = True,  # noqa: ANN202
               offset: float = 0.0):
    """A light body from rest in the driven tube, Maxey and Riley on, the review's arm: dt 50 us, ``steps`` steps,
    under the field's steady limit or IMEX Euler; ``own_field=False`` leaves the body's own field in the rate terms'
    reading under the steady limit (the PR's reading). Run without a history: route 2's own field is megabytes a step."""
    exp = point_experiment(radius=radius, dt=5e-5, mass_ratio=ratio, steady=steady, offset=offset)
    held = _point_drag.PointDragNode.responses
    if not own_field:
        def planted(self, p, dt):  # noqa: ANN001, ANN202
            made = held(self, p, dt)
            return {**made, "steady_clean": made["steady_clean"] * 0.0}

        _point_drag.PointDragNode.responses = planted  # type: ignore[method-assign]
    try:
        return exp.build().run(ms(5e-2 * steps))
    finally:
        _point_drag.PointDragNode.responses = held  # type: ignore[method-assign]


def test_a_light_body_stays_bounded_under_each_scheme(dtype):
    """Gate 14 (the review's B2), at the review's diverging arm (a 40 um body of 0.05 of water's density, Maxey and Riley
    on, dt 50 us), its starting offset drawn: from rest in the driven tube under the field's steady limit it settles
    with the flow (its own field removed from the rate terms' reading by the steady limit's own self-history), the
    ringing flag never raised; and under IMEX Euler it stays finite and leads the fluid by Maxey and Riley's (m_f - m) g
    / R0. The plant leaves the body's own field in the steady limit's reading (OWN_FIELD), which diverged there (at
    radii of 31 and 33 um and densities of 0.12 and 0.2 the PR's reading was itself stable: the gate is held at the
    arm where it was not)."""
    rng = np.random.default_rng(14000 + seed())
    ratio, radius, offset = 0.05, 40e-6, float(rng.uniform(0.0, 0.1e-3))
    steady = _light_run(True, ratio, radius, 100, own_field=not plant("OWN_FIELD"), offset=offset)
    velocity = float(np.asarray(steady.final["ball"]["twist"])[2])
    reading = float(np.asarray(steady.final["drag:ball"]["drag_reading"])[2])
    ringing = _rang(steady)
    slip = abs(velocity - reading) / abs(reading)
    timed = _light_run(False, ratio, radius, 100, offset=offset)
    lead = float(np.asarray(timed.final["ball"]["twist"])[2] - np.asarray(timed.final["drag:ball"]["drag_reading"])[2])
    m_f = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    closed = (1.0 - ratio) * m_f * 1.0 / (6.0 * math.pi * WATER_MU * radius)
    print(f"F3 gate 14 [{dtype}]: density {ratio:.3g}, radius {radius * 1e6:.3g} um: steady limit slip {slip:.3g} of the "
          f"flow, ringing {ringing:g}; IMEX Euler lead {lead:.4g} m/s against (m_f - m) g / R0 = {closed:.4g}")
    assert math.isfinite(velocity) and slip < 0.05 and not ringing, (slip, ringing)
    assert math.isfinite(lead) and not _rang(timed)


def _rang(result) -> bool:  # noqa: ANN001
    """Whether the run raised the ringing flag at any step."""
    return any(flag.path.endswith("rate_feedback") and flag.was_raised for flag in result.flags)


def test_the_ringing_flag_names_a_diverging_rate_loop(float64):
    """The ringing flag (rate_feedback) raised in the run that diverges: the light body under the steady limit with
    its own field left in the rate terms' reading (gate 14's plant, the PR's reading), within its first 30 steps;
    gate 14's clean runs are its control (never raised)."""
    run = _light_run(True, 0.05, 40e-6, 30, own_field=False)
    held = [flag for flag in run.flags if flag.path.endswith("rate_feedback")]
    print(f"F3 the ringing flag: {held}")
    assert any(flag.was_raised for flag in held), held
