"""F3's gates on the point body's self-history (``Drag.Stokes``, the maintainer's row #349, point 3: the exact route
the default, the fitted modes the opt-in, their swap test route 1's gate) and on the field's steady limit (the review
of #239's B2), in both dtypes (``tests/flow/predictions_f3_fix.json``). Each control is a plant switched by
``MIME_PLANT_<NAME>``, its rate measured by ``.github/probe/detection_rate.py``; each draw is seeded by
``DETECTION_SEED``."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from point_cases import WATER_MU, WATER_RHO, FreeBox, plant, point_experiment, seed, with_second_body

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
               offset: float = 0.0, coupling: str | None = None, apart: float = 0.0):
    """A light body from rest in the driven tube, Maxey and Riley on, the review's arm: dt 50 us, ``steps`` steps,
    under the field's steady limit or IMEX Euler; ``own_field=False`` leaves the body's own field in the rate terms'
    reading under the steady limit (the PR's reading); ``coupling`` the drag's (one pass by default); ``apart`` (in
    radii), a second such body that far along the axis. Run without a history: route 2's own field is megabytes a
    step."""
    exp = point_experiment(radius=radius, dt=5e-5, mass_ratio=ratio, steady=steady, offset=offset, coupling=coupling)
    if apart:
        exp = with_second_body(exp, (offset, 0.0, apart * radius), radius=radius, coupling=coupling, mass_ratio=ratio)
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


#: Gate 14's steady-limit arm runs past the one-pass ringing the refuting review of #270 found from step 115 on a body
#: at rest in its flow (its M3, raised to step 1500), which the rate load's share of the drag's scale now holds quiet.
SETTLED_STEPS = 1500


@pytest.mark.parametrize("coupling", [None, "lagged"])
def test_a_light_body_stays_bounded_under_each_scheme(dtype, coupling):
    """Gate 14 (the review's B2), one pass and lagged, at the review's diverging arm (a 40 um body of 0.05 of water's
    density, Maxey and Riley on, dt 50 us), its starting offset drawn: from rest in the driven tube under the field's
    steady limit it settles with the flow (its own field removed from the rate terms' reading by the steady limit's own
    self-history), the ringing flag never raised over SETTLED_STEPS steps (held one pass the reading rings at its solve's
    tolerance from about step 110, under RATE_SHARE of the drag's scale: the plant RATE_SHARE sets the share to zero,
    the flag's reading before #270's fix round, which raises it there); and under IMEX Euler it stays finite and leads
    the fluid by Maxey and Riley's (m_f - m) g / R0. The plant leaves the body's own field in the steady limit's reading (OWN_FIELD), which diverged there (at
    radii of 31 and 33 um and densities of 0.12 and 0.2 the PR's reading was itself stable: the gate is held at the
    arm where it was not). The plant diverges under both schemes: lagged it rings and grows; one pass it runs away
    without the sign reversals the flag counts (the body at about -1e3 m/s after 100 steps at every one of 20 draws,
    the flag raised at none of them), so there the slip catches it."""
    rng = np.random.default_rng(14000 + seed())
    ratio, radius, offset = 0.05, 40e-6, float(rng.uniform(0.0, 0.1e-3))
    held_share = _point_drag.RATE_SHARE
    if plant("RATE_SHARE"):
        _point_drag.RATE_SHARE = 0.0
    try:
        steady = _light_run(True, ratio, radius, SETTLED_STEPS, own_field=not plant("OWN_FIELD"), offset=offset,
                            coupling=coupling)
    finally:
        _point_drag.RATE_SHARE = held_share
    velocity = float(np.asarray(steady.final["ball"]["twist"])[2])
    reading = float(np.asarray(steady.final["drag:ball"]["drag_reading"])[2])
    ringing = _rang(steady)
    slip = abs(velocity - reading) / abs(reading)
    timed = _light_run(False, ratio, radius, 100, offset=offset, coupling=coupling)
    lead = float(np.asarray(timed.final["ball"]["twist"])[2] - np.asarray(timed.final["drag:ball"]["drag_reading"])[2])
    m_f = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    closed = (1.0 - ratio) * m_f * 1.0 / (6.0 * math.pi * WATER_MU * radius)
    print(f"F3 gate 14 [{dtype}, {coupling or 'one pass'}]: density {ratio:.3g}, radius {radius * 1e6:.3g} um: steady "
          f"limit slip {slip:.3g} of the flow, ringing {ringing:g}; IMEX Euler lead {lead:.4g} m/s against (m_f - m) g "
          f"/ R0 = {closed:.4g}")
    assert math.isfinite(velocity) and slip < 0.05 and not ringing, (slip, ringing)
    assert math.isfinite(lead) and not _rang(timed)


#: The refuting review of #270's three one-pass runaways under the steady limit (M2): a light body (0.05 of water's
#: density) of 150 um on the axis, of 40 um 0.4 mm off it, and two of 40 um 2.5 radii apart along it; each run to a
#: step past its flag's onset and before it goes non-finite (measured: the flag at 24, 28 and 18 in float64, 16, 28 and
#: 18 in float32; non-finite at 39 and 111 in float64, 32 and 123 in float32; the pair bounded at about 7 m/s).
RUNAWAYS = {"light150": dict(radius=150e-6, steps=28), "near_wall": dict(offset=0.4e-3, steps=50),
            "pair": dict(apart=2.5, steps=30)}


@pytest.mark.parametrize("case", sorted(RUNAWAYS))
def test_the_ringing_flag_names_each_one_pass_runaway_and_lagged_holds_it(dtype, case):
    """Each of the review's one-pass runaways under the steady limit raises rate_feedback (its rate load reversing and
    growing above RATE_SHARE of the drag's scale) before it goes non-finite, and the route the flag names holds it:
    coupling='lagged' runs the same steps bounded (under 0.5 m/s, the flow's speed some 0.06) with the flag quiet.
    Gate 14's settled arm is the flag's control (one pass, quiet to SETTLED_STEPS)."""
    settings = RUNAWAYS[case]
    steps = int(settings["steps"])
    kwargs = {k: v for k, v in settings.items() if k != "steps"}
    radius = float(kwargs.pop("radius", 40e-6))
    one_pass = _light_run(True, 0.05, radius, steps, **kwargs)
    lagged = _light_run(True, 0.05, radius, steps, coupling="lagged", **kwargs)
    bodies = ["ball", "bob"] if "apart" in kwargs else ["ball"]
    fastest = max(float(np.max(np.abs(np.asarray(lagged.final[b]["twist"])[:3]))) for b in bodies)
    print(f"F3 PR 2 the runaway flag [{dtype}, {case}]: one pass raised {_rang(one_pass)}; lagged raised "
          f"{_rang(lagged)}, fastest {fastest:.3g} m/s")
    assert _rang(one_pass) and not _rang(lagged) and math.isfinite(fastest) and fastest < 0.5, (case, fastest)


def _rang(result) -> bool:  # noqa: ANN001
    """Whether the run raised the ringing flag at any step."""
    return any(flag.path.endswith("rate_feedback") and flag.was_raised for flag in result.flags)


def test_the_ringing_flag_names_a_diverging_rate_loop(float64):
    """The ringing flag (rate_feedback) raised in the run that diverges: the light body under the steady limit with
    its own field left in the rate terms' reading (gate 14's plant, the PR's reading), within its first 30 steps,
    lagged (declared): the flag counts the lagged loop's signature, a load that reverses its sign and grows. One pass,
    the same plant runs away without those reversals (F3's second PR's probe: the flag raised at none of 20 draws within
    100 steps, at 7 of 20 within 300, while the body reached about -1e3 m/s), which the flag does not name: gate 14's
    slip catches it there. Gate 14's clean runs are its control (never raised)."""
    run = _light_run(True, 0.05, 40e-6, 30, own_field=False, coupling="lagged")
    held = [flag for flag in run.flags if flag.path.endswith("rate_feedback")]
    print(f"F3 the ringing flag: {held}")
    assert any(flag.was_raised for flag in held), held
