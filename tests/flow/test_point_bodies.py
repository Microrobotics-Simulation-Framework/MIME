"""F3's gates on the point body (``Drag.Stokes``) and the exchange law, in both dtypes (``tests/flow/predictions_f3.json``
holds their predictions, committed before they ran). Each control is a plant switched by ``MIME_PLANT_<NAME>``
(``point_cases.plant``), its rate measured by ``.github/probe/detection_rate.py``; each draw is seeded by
``DETECTION_SEED``."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from point_cases import (TUBE_RADIUS, WATER_MU, WATER_RHO, FreeBox, axis_profile, plant, point_experiment, seed,
                         with_second_body)

from mime.build import BuildRefused
from mime.core import run_dtype
from mime.mappings import gaussian_mac
from mime.nodes.flow.fvm import MACGrid, periodic_tube_layout
from mime.nodes.flow.fvm._exchange import HASIMOTO
from mime.nodes.hydro._point_drag import FAXEN_FORCES, FORCES
from mime.units import ms

#: Each dtype's line for the adjoint identity, relative to the products' magnitudes (a few units of rounding).
ADJOINT_LINE = {"float32": 1e-5, "float64": 1e-13}


def _drawn_pair(rng: np.random.Generator, dtype: str) -> tuple[float, float]:
    """One draw of the adjoint identity through the registered mapping kind: a box, points, forces and a field; the
    largest relative miss of ``<S f, u> = f . G u`` over the face sets and the two modes, and the scale it is
    judged on. The plant (``MIME_PLANT_EXCHANGE``) changes the gather alone."""
    cells = int(rng.integers(12, 24))
    h = float(rng.choice([1.0, 0.25, 1.0 / 32.0]))
    count = int(rng.integers(1, 6))
    width = float(rng.uniform(1.0, 2.0))
    grid = MACGrid((cells, cells, cells), h)
    points = jnp.asarray(rng.uniform(0.0, cells * h, (count, 3)), dtype=dtype)
    forces = jnp.asarray(rng.normal(size=count), dtype=dtype)
    worst = 0.0
    planted = plant("EXCHANGE")
    for component in range(3):
        field = jnp.asarray(rng.normal(size=grid.cells), dtype=dtype)
        for mode in ("average", "laplacian"):
            def made(direction: str, wider: float = 0.0) -> object:
                return gaussian_mac(nx=cells, ny=cells, nz=cells, spacing=h, width_cells=width * (1.0 + wider),
                                    component=component, mode=mode, direction=direction, count=count)

            gather_points = points
            gather = made("gather", 1e-9 if planted == "wider" else 0.0)
            if planted == "offset":
                gather_points = points + 0.5 * h
            scatter = made("scatter")
            density = scatter.apply(forces, geom=points)  # type: ignore[attr-defined]
            read = gather.apply(field, geom=gather_points)  # type: ignore[attr-defined]
            if planted == "onesided" and mode == "laplacian":
                read = read + 0.1 * made("gather").apply(field, geom=points)  # type: ignore[attr-defined]
            left = grid.inner((density, density, density), (field, field, field)) / 3.0
            right = jnp.sum(forces * read)
            scale = grid.inner((jnp.abs(density),) * 3, (jnp.abs(field),) * 3) / 3.0 + jnp.sum(jnp.abs(forces * read))
            worst = max(worst, float(jnp.abs(left - right) / scale))
    return worst, ADJOINT_LINE[dtype]


def test_the_exchange_law_is_a_transpose_pair_in_the_grids_inner_product(dtype):
    """Gate 1: over 60 draws, gather and scatter, both modes, every face set, through the registered mapping kind."""
    rng = np.random.default_rng(1000 + seed())
    misses = [_drawn_pair(rng, dtype) for _ in range(60)]
    print(f"F3 gate 1 [{dtype}]: largest relative miss {max(m for m, _ in misses):.3g} over 60 draws")
    assert all(miss <= line for miss, line in misses), max(miss for miss, _ in misses)


def test_one_body_in_still_fluid_coupled_moves_as_it_would_uncoupled(dtype):
    """Gate 2: a point body pulled and turned in a free box (no wall), two-way, with the Faxén pair the rung builds, at
    steady state: at its uncoupled Stokes speed less Hasimoto's image term, and at its uncoupled rotation (the review's
    M3: the pair, untested at the node before). The plants: the constant correction left out (CORRECTION); the Faxén
    delivery at the volume average's coefficient, the review's one-sided mutant (ONESIDED); the couple's correction
    left out (COUPLE_CORRECTION)."""
    rng = np.random.default_rng(2000 + seed())
    box = FreeBox(24, float(rng.uniform(0.8, 1.2)), 50.0, mass_ratio=1e-3, force=1e-11, torque=1e-16)
    if plant("CORRECTION") or plant("COUPLE_CORRECTION"):
        box.drag.responses = _without(box.drag.responses,  # type: ignore[method-assign]
                                      "steady" if plant("CORRECTION") else "couple_steady")
    if plant("ONESIDED"):
        box.drag.stepped = _one_sided(box.drag, box.drag.stepped)  # type: ignore[method-assign]
    for _ in range(40):
        box.step()
    mobility = 1.0 / (6.0 * math.pi * WATER_MU * box.radius)
    expected = mobility * box.force - HASIMOTO * box.force / (6.0 * math.pi * WATER_MU * box.cells * box.h)
    turning = box.torque / (8.0 * math.pi * WATER_MU * box.radius ** 3)
    moved = abs(float(box.velocity[2]) - expected) / (mobility * box.force)
    turned = abs(float(box.spin[2]) - turning) / turning
    print(f"F3 gate 2 [{dtype}]: |V - V_expected|/(M0 F) = {moved:.3g}, |W - W_expected|/W_expected = {turned:.3g}")
    assert moved < 2e-3 and turned < 2e-3, (moved, turned)


def _without(responses, key: str):  # noqa: ANN001, ANN202 - the plant's wrapper
    def planted(p, dt):  # noqa: ANN001, ANN202
        made = responses(p, dt)
        return {**made, key: made[key] * 0.0}
    return planted


def _one_sided(node, stepped):  # noqa: ANN001, ANN202 - the review's mutant: the Faxén delivery at the average's coefficient
    def planted(p, state, inputs, dt):  # noqa: ANN001, ANN202
        ended = dict(stepped(p, state, inputs, dt))
        _, average, _ = node.coefficients(p)
        for force, faxen in zip(FORCES, FAXEN_FORCES, strict=True):
            ended[faxen] = average * ended[force]
        return ended
    return planted


def _final(result, node: str, field: str) -> np.ndarray:  # noqa: ANN001
    return np.asarray(result.final[node][field])


#: Each dtype's line for a body's motion against its law replayed on the run's own readings, relative to the motion's
#: scale: a few units of rounding of the recurrence's terms over the run's steps.
REPLAY_LINE = {"float64": 1e-9, "float32": 1e-5}


def _law_replayed(result, mass_ratio: float, radius: float, dt: float) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """The body's axial velocity after each step, and the law's (Maxey and Riley's, as the node states it) from rest
    on the run's own readings: ``(m + m_a)(V' - V)/dt = R0 (u_d - V') + (m_f + m_a) (u_c - u_c,prev)/dt``, the rate
    zero on the first step."""
    velocity = np.asarray(result.history["ball"]["twist"])[:, 2]
    drag = np.asarray(result.history["drag:ball"]["drag_reading"])[:, 2]
    clean = np.asarray(result.history["drag:ball"]["clean_reading"])[:, 2]
    m_f = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    m, m_a, r0 = mass_ratio * m_f, 0.5 * m_f, 6.0 * math.pi * WATER_MU * radius
    law, v = [], 0.0
    for n in range(len(velocity)):
        rate = (clean[n] - clean[n - 1]) / dt if n >= 1 else 0.0
        v = ((m + m_a) / dt * v + r0 * drag[n] + (m_f + m_a) * rate) / ((m + m_a) / dt + r0)
        law.append(v)
    return velocity, np.asarray(law)


def test_a_neutral_body_in_a_uniformly_driven_flow_moves_with_the_fluid(dtype):
    """Gate 3: a neutrally buoyant point body in the driven tube, two-way, offsets up to 0.2 mm (the original range),
    moves with the fluid up to Maxey and Riley's own slip, judged against that slip: the law's recurrence on the run's
    own readings (the drag's and the flow without the body's), which carries the Faxén slip (a^2/6 - a^2/10) lap u
    that the wall's viscous front brings (the review's m1). The plant switches the Maxey-Riley terms off."""
    rng = np.random.default_rng(3000 + seed())
    radius, dt = float(rng.uniform(30e-6, 50e-6)), 2e-4
    exp = point_experiment(radius=radius, offset=float(rng.uniform(0.0, 0.2e-3)), dt=dt)
    if plant("MAXEY_RILEY"):
        _patch_node(maxey_riley=False)
    try:
        result = exp.build().run(ms(8.0), history=True)
    finally:
        _patch_node()
    velocity, law = _law_replayed(result, 1.0, radius, dt)
    reading = np.asarray(result.history["drag:ball"]["drag_reading"])[:, 2]
    missed = float(np.max(np.abs(velocity - law)) / np.max(np.abs(velocity)))
    print(f"F3 gate 3 [{dtype}]: |V - V_law|/|V| = {missed:.3g}; the law's own slip at the end "
          f"{abs(law[-1] - reading[-1]) / abs(velocity[-1]):.3g} of V")
    assert missed < REPLAY_LINE[dtype], missed


_ORIGINAL_INIT = None


def _patch_node(**forced: object) -> None:
    """Force constructor arguments of every point body's node built in this process (a plant), or restore it."""
    from mime.nodes.hydro import _point_drag

    global _ORIGINAL_INIT  # noqa: PLW0603 - the plant's own switch
    if _ORIGINAL_INIT is None:
        _ORIGINAL_INIT = _point_drag.PointDragNode.__init__
    original = _ORIGINAL_INIT
    if not forced:
        _point_drag.PointDragNode.__init__ = original  # type: ignore[method-assign]
        return

    def planted(self, name, timestep, **kwargs):  # noqa: ANN001, ANN202
        original(self, name, timestep, **{**kwargs, **forced})

    _point_drag.PointDragNode.__init__ = planted  # type: ignore[method-assign]


def test_a_force_free_body_in_poiseuille_flow_moves_at_faxens_closed_form(dtype):
    """Gate 4: steady Stokes flow in the tube (its steady limit), a neutral body a third of the radius off the axis,
    against ``u(b) + (a^2/6) lap u`` on a Poiseuille profile fitted to the grid's own row through the axis. At a =
    sigma = h the Faxén term is 1.0 % of the centreline speed and the bare kernel's own 3.1 %. The plant switches the
    Faxén pair off."""
    h = 2.0 * TUBE_RADIUS / 16.0
    offset = float(np.random.default_rng(4000 + seed()).uniform(0.2, 0.45)) * TUBE_RADIUS
    exp = point_experiment(radius=h, offset=offset, dt=5e-4, steady=True, kernel_width=1.0)
    if plant("FAXEN"):
        _patch_node(faxen=False)
    try:
        built = exp.build()
        result = built.run(ms(10.0))
    finally:
        _patch_node()
    flow = result.final["water.flow"]
    layout = periodic_tube_layout(TUBE_RADIUS, 1e-3, cells_across=16)
    x, row = axis_profile(flow, layout.centre, layout.spacing)
    inside = np.abs(x) < 0.7 * TUBE_RADIUS
    fit = np.polyfit(x[inside] ** 2, row[inside], 1)
    curvature, centre_speed = -fit[0], fit[1]
    expected = centre_speed - curvature * (offset ** 2 + (2.0 / 3.0) * h ** 2)
    velocity = _final(result, "ball", "twist")[2]
    print(f"F3 gate 4 [{dtype}]: |V - V_Faxen|/U = {abs(velocity - expected) / centre_speed:.3g}, Faxen term "
          f"{curvature * (2.0 / 3.0) * h ** 2 / centre_speed:.3g} of U")
    assert abs(velocity - expected) / centre_speed < 3e-3, (velocity, expected, centre_speed)


def test_one_way_a_heavy_body_slips_with_the_added_masss_closed_form(dtype):
    """Gate 5: one-way (the fluid untouched), a body 1.5 to 3 times the fluid's density in the uniformly accelerating
    flow follows the backward-Euler recurrence of ``(m + m_f/2) dw/dt = (m_f - m) g - R0 w`` on the run's own readings
    (the review's m2: the flow's reading lags g (n+1) dt by up to 4e-5 of its speed, the solve's, which a closed form in
    g alone mistook for the law's). The plant switches the added mass off (the fluid-acceleration force kept)."""
    rng = np.random.default_rng(5000 + seed())
    radius, dt, ratio = float(rng.uniform(30e-6, 50e-6)), 1e-4, float(rng.uniform(1.5, 3.0))
    exp = point_experiment(radius=radius, dt=dt, mass_ratio=ratio, one_way=True, drive=1.0)
    if plant("ADDED_MASS"):
        _patch_node(added_mass=False)
    try:
        result = exp.build().run(ms(3.0), history=True)
    finally:
        _patch_node()
    velocity, law = _law_replayed(result, ratio, radius, dt)
    m_f = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    terminal = abs((ratio - 1.0) * m_f * 1.0 / (6.0 * math.pi * WATER_MU * radius))
    missed = float(np.max(np.abs(velocity - law)) / terminal)
    print(f"F3 gate 5 [{dtype}]: largest |V - V_law|/|w_inf| = {missed:.3g}")
    assert missed < REPLAY_LINE[dtype], missed


def test_the_rate_terms_read_the_flow_without_the_body_and_stay_finite(dtype):
    """Gate 6, at the floor sweep's NaN arm: lagged, a = 1.33 sigma, r = 0.03, a near-massless body under a step force
    in a free box. The plant reads the rate terms on the constant correction's reading instead, which diverges."""
    rate = _constant_rate if plant("RATE_READING") else None
    r = float(np.random.default_rng(6000 + seed()).uniform(0.02, 0.05))
    box = FreeBox(24, 1.33, r, mass_ratio=1e-3, force=1e-9, faxen=False, rate_reading=rate)
    for _ in range(200):
        box.step()
        if not bool(jnp.all(jnp.isfinite(box.velocity))):
            break
    terminal = box.force / (6.0 * math.pi * WATER_MU * box.radius)
    speed = float(jnp.max(jnp.abs(box.velocity)))
    print(f"F3 gate 6 [{dtype}]: the body's speed after 200 steps over its terminal speed {speed / terminal:.3g}")
    assert math.isfinite(speed) and speed < 2.0 * terminal, speed / terminal


def _constant_rate(box: FreeBox, gathered: dict) -> dict:  # noqa: ANN001
    """The plant: the drag reading (the constant correction's) put where the rate terms read, and the step's load
    made again from it."""
    state = dict(box.drag_state)
    drag = jnp.asarray(state["drag_reading"])
    previous = getattr(box, "_planted_previous", drag)
    box._planted_previous = drag  # type: ignore[attr-defined]  # noqa: SLF001
    radius = box.radius
    m_f = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    rate = (drag - previous) / box.dt
    clean = jnp.asarray(state["clean_reading"])
    load = jnp.asarray(state["added_load"])
    shifted = load.at[:3].add(1.5 * m_f * (rate - (clean - getattr(box, "_planted_clean", clean)) / box.dt))
    box._planted_clean = clean  # type: ignore[attr-defined]  # noqa: SLF001
    del gathered
    return {**state, "added_load": shifted}


def _refusal(exp, code: str):  # noqa: ANN001, ANN202
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    found = [r for r in refused.value.refusals if r.code == code]
    assert found, {r.code for r in refused.value.refusals}
    return found[0]


def _named(text: str, pattern: str) -> float:
    import re

    held = re.search(pattern, text)
    assert held is not None, text
    return float(held.group(1))


def test_the_build_refuses_a_body_no_width_lifts_above_the_floor(dtype):
    """Gate 7 (the review's M5: the remedies named are the ones that work, each built): a sphere of 8 cells' radius at
    32 cells across is past what the widest kernel admits (drag.stokes-size), and the coarser grid its refusal names
    plans it; a sphere of 6 cells' radius at a small step is under the floor at every allowed width
    (drag.stokes-floor), and the larger step its refusal names plans it; a sphere of a cell's radius plans."""
    h = 2.0 * TUBE_RADIUS / 32.0
    size = _refusal(point_experiment(radius=8.0 * h, dt=1e-4, cells_across=32), "drag.stokes-size")
    cells = int(_named(str(size), r"FVM\(cells_across=(\d+)\) plans it"))
    point_experiment(radius=8.0 * h, dt=1e-4, cells_across=cells).build()
    floor = _refusal(point_experiment(radius=6.0 * h, dt=1e-5, cells_across=32), "drag.stokes-floor")
    larger = _named(str(floor), r"dt=([0-9.e+-]+) s clears it")
    point_experiment(radius=6.0 * h, dt=larger, cells_across=32).build()
    print(f"F3 gate 7 [{dtype}]: a = 8h plans at {cells} cells across; a = 6h plans at dt {larger:.3g} s")
    point_experiment(radius=h, dt=1e-4, cells_across=32).build()


def test_the_run_dtype_is_the_fixtures(dtype):
    assert str(jnp.dtype(run_dtype())) == dtype


def _metered(radius: float, dt: float, ratio: float):  # noqa: ANN202
    from mime.observations import EnergyLedger

    exp = point_experiment(radius=radius, dt=dt, mass_ratio=ratio)
    exp.observe(EnergyLedger())
    return exp.build().run(ms(4.0))


def test_the_books_close_with_the_exchange_booked_under_the_drag(dtype):
    """Gate 8: metered, a heavy point body slipping in the driven tube, two-way: every receiver's books close at every
    step (#233's ``ledger_closes``), the exchange's fluid side booked under the drag's id and the Maxey-Riley part
    under its own row; and the drag's sign judged by its node (the review's B1), its gift count read onto both of the
    drag's rows, none. The plant is the review's energy gift, +0.5 R0 V on the drag's load to the body (GIFT)."""
    from mime.testing import ledger_closes

    if plant("GIFT"):
        _wrap_stepped(_gift)
    try:
        result = _metered(40e-6, 2e-4, 2.0)
    finally:
        _wrap_stepped(None)
    verdict = ledger_closes(result)
    assert verdict.status == "PASS", verdict.detail
    assert result.energy is not None
    rows = {(law.law, law.receiver): law for law in result.energy.laws}
    assert {("drag:ball", "water.flow"), ("drag:ball", "ball"), ("drag:ball.maxey_riley", "ball")} <= set(rows), rows
    gifts = {receiver: rows[("drag:ball", receiver)].gift_steps for receiver in ("ball", "water.flow")}
    print(f"F3 gate 8 [{dtype}]: the drag's gift steps {gifts}")
    assert gifts == {"ball": 0, "water.flow": 0}, gifts


def _gift(stepped):  # noqa: ANN001, ANN202 - the review's energy gift: +0.5 R0 V on the drag's load to the body
    def planted(self, p, state, inputs, dt):  # noqa: ANN001, ANN202
        ended = dict(stepped(self, p, state, inputs, dt))
        r0 = 6.0 * math.pi * p["viscosity"] * p["radius"]
        ended["background_load"] = ended["background_load"].at[:3].add(0.5 * r0 * inputs["twist"][:3])
        return ended
    return planted


def test_the_lagged_exchanges_unpaired_work_falls_at_its_order(dtype):
    """Gate 13 (the review's M1): with the law's own dissipation booked (the fluid's viscous dissipation of the
    kernel's own field, M_inf f . f, taken out of it, which the fluid's row holds), the lagged exchange's unpaired work,
    both of the drag's rows plus that dissipation, summed over a fixed time, falls as the step does: ``term_orders`` at
    its declared order (2 a step). The plant books the dissipation as R0 |V - u_d|^2 (the PR's, DISSIPATION), whose
    unpaired work does not fall."""
    from mime.core.ledger import TermOrder
    from mime.testing import term_orders

    rng = np.random.default_rng(13000 + seed())
    radius, ratio = float(rng.uniform(30e-6, 50e-6)), float(rng.uniform(1.5, 3.0))

    def sums(dt_scale: float, tolerance_scale: float) -> dict[str, float]:
        del tolerance_scale
        if plant("DISSIPATION"):
            _wrap_stepped(_drag_dissipation)
        try:
            result = _metered(radius, 2e-4 * dt_scale, ratio)
        finally:
            _wrap_stepped(None)
        assert result.energy is not None
        work = sum(float(law.work) for law in result.energy.laws if law.law == "drag:ball")
        exchanged = abs(float(next(law.work for law in result.energy.laws
                                   if (law.law, law.receiver) == ("drag:ball", "water.flow"))))
        unpaired = work + float(np.asarray(result.final["drag:ball"]["dissipated"]))
        print(f"F3 gate 13 [{dtype}] dt {2e-4 * dt_scale:.3g}: unpaired {unpaired:.4g} J, {unpaired / exchanged:.4g} of "
              f"the work exchanged with the fluid")
        return {"unpaired": abs(unpaired)}

    report = term_orders(sums, ["unpaired"], {"unpaired": TermOrder(dt=2)})
    print(f"F3 gate 13 [{dtype}]: {report.checks}")
    assert report.passed, report.checks


def _drag_dissipation(stepped):  # noqa: ANN001, ANN202 - the PR's booking: R0 |V - u_d|^2, the kernel's own field in it
    def planted(self, p, state, inputs, dt):  # noqa: ANN001, ANN202
        ended = dict(stepped(self, p, state, inputs, dt))
        before = jnp.asarray(state["drag_reading"])
        felt = 6.0 * math.pi * p["viscosity"] * p["radius"] * (inputs["twist"][:3] - before)
        moved = dt * jnp.sum(felt * (jnp.asarray(state["raw_reading"]) - before))
        ended["dissipated"] = ended["dissipated"] + jnp.where(jnp.asarray(state["steps"]) > 0.0, moved, 0.0)
        return ended
    return planted


def test_a_derivative_through_the_exchange_law_matches_a_central_difference(float64):
    """Gate 9: the body's axial position after 10 steps in the fluid's viscosity (the flow's speed and the drag both
    move with it), forward mode by ``Study.gradient``, against a central difference of two runs. The plant holds the
    gathered flow out of the derivative (``stop_gradient`` on the reading), which must fail it."""
    import jax

    from mime.study import Study, along
    from mime.units import Pa_s

    def position(final):  # noqa: ANN001, ANN202
        return final["ball"]["pose"][2]

    if plant("GRADIENT"):
        _hold_reading(jax.lax.stop_gradient)
    try:
        exp = point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0)
        result = Study(exp).sweep(along({"water.viscosity": [Pa_s(1e-3)]}), duration=ms(2.0)).gradient(
            position, wrt=["water.viscosity"], mode="forward",
            acknowledged={"water.flow": "the derivative is in the viscosity, which the FVM's gradient gate checks "
                                        "exact; its unverified geometry leaves are not moved"})
    finally:
        _hold_reading(None)
    derivative = float(np.asarray(result.member(0).gradient["water.viscosity"]))
    step = 1e-3 * WATER_MU
    ends = []
    for mu in (WATER_MU + step, WATER_MU - step):
        moved = point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0, viscosity=mu)
        ends.append(float(moved.build().run(ms(2.0)).final["ball"]["pose"][2]))
    difference = (ends[0] - ends[1]) / (2.0 * step)
    print(f"F3 gate 9: derivative {derivative:.10g}, central difference {difference:.10g}, relative "
          f"{abs(derivative - difference) / abs(difference):.3g}")
    assert abs(derivative - difference) <= 1e-5 * abs(difference), (derivative, difference)


_ORIGINAL_STEPPED = None


def _wrap_stepped(wrap) -> None:  # noqa: ANN001
    """Wrap every point body's node's step in ``wrap`` (a plant: a function of the step returning its replacement),
    or restore it."""
    from mime.nodes.hydro import _point_drag

    global _ORIGINAL_STEPPED  # noqa: PLW0603 - the plant's own switch
    if _ORIGINAL_STEPPED is None:
        _ORIGINAL_STEPPED = _point_drag.PointDragNode.stepped
    _point_drag.PointDragNode.stepped = _ORIGINAL_STEPPED if wrap is None else wrap(  # type: ignore[method-assign]
        _ORIGINAL_STEPPED)


def _hold_reading(wrap) -> None:  # noqa: ANN001
    """Wrap the gathered readings a point body's node takes in ``wrap`` (a plant), or restore it."""
    from mime.nodes.hydro import _point_drag

    if wrap is None:
        _wrap_stepped(None)
        return

    def held(stepped):  # noqa: ANN001, ANN202
        def step(self, p, state, inputs, dt):  # noqa: ANN001, ANN202
            readings = {k: (wrap(v) if k in _point_drag.AVERAGES + _point_drag.LAPLACIANS + _point_drag.ROTATIONS
                            else v) for k, v in inputs.items()}
            return stepped(self, p, state, readings, dt)
        return step

    _wrap_stepped(held)


def _refused(exp) -> set[str]:  # noqa: ANN001
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    return {r.code for r in refused.value.refusals}


def test_the_stokes_rung_refuses_a_body_that_is_no_sphere_a_medium_with_no_grid_and_rk3(float64):
    """The plan's refusals: a body whose hull is no sphere (drag.stokes-sphere), a medium with no flow on a grid (the
    rung's support row, support.unsupported), a flow stepped by RK3 (drag.stokes-scheme) and an open vessel's flow
    (flow.open-exchange); the control, gate 3's experiment, plans."""
    assert "drag.stokes-sphere" in _refused(point_experiment(hull="capsule"))
    assert "support.unsupported" in _refused(point_experiment(flow=False))
    assert "drag.stokes-scheme" in _refused(point_experiment(dt=1e-5, scheme="rk3"))
    assert "flow.open-exchange" in _refused(_open_vessel())
    point_experiment().build()


def _open_vessel():  # noqa: ANN202
    """Gate 3's experiment with the flow's vessel open at its ends (an inflow and a Windkessel at them)."""
    return point_experiment(opened=True)


def _shared_lines(one_way: bool) -> tuple[dict[str, str], float]:
    """Each drag's shared-medium line with a second point body 0.3 mm along the axis from the first, both of radius
    40 um; and the two spheres' volume over the grid's box."""
    built = with_second_body(point_experiment(), (0.0, 0.0, 0.3e-3), one_way=one_way).build()
    lines = {owner: next(line for line in built.report.checked[owner] if "solved flow" in line)
             for owner in ("drag:ball", "drag:bob")}
    held = built.graph.get_node("water.flow").constants()
    box = held["cells_x"] * held["cells_y"] * held["cells_z"] * float(held["spacing"]) ** 3
    return lines, 2.0 * 4.0 / 3.0 * math.pi * 40e-6 ** 3 / box


def test_two_point_bodies_in_one_medium_are_reported_interacting_through_the_flow(float64):
    """Two two-way point bodies in one medium: each one's line names the other as scattering into the flow it reads,
    their starting separation in radii (0.3 mm over 40 um, 7.5) and their volume fraction in the grid's box. The
    control: with the second body one-way, the first's line names no body scattering into its flow, and the
    second's still names the first."""
    lines, fraction = _shared_lines(one_way=False)
    print("\n" + "\n".join(lines.values()))
    for owner, other in (("drag:ball", "'bob'"), ("drag:bob", "'ball'")):
        assert f"into which {other} scatters its reaction" in lines[owner]
        assert "starts 7.5 radii from it" in lines[owner] and f"fill {fraction:.3g} of the grid's box" in lines[owner]
    control, _ = _shared_lines(one_way=True)
    assert "no other body in it scatters a reaction" in control["drag:ball"]
    assert "into which 'ball' scatters its reaction" in control["drag:bob"]
