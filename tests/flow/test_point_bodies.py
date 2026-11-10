"""F3's gates on the point body (``Drag.Stokes``) and the exchange law, in both dtypes (``tests/flow/predictions_f3.json``
holds their predictions, committed before they ran). Each control is a plant switched by ``MIME_PLANT_<NAME>``
(``point_cases.plant``), its rate measured by ``.github/probe/detection_rate.py``; each draw is seeded by
``DETECTION_SEED``."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from point_cases import TUBE_RADIUS, WATER_MU, WATER_RHO, FreeBox, axis_profile, plant, point_experiment, seed

from mime.build import BuildRefused
from mime.core import run_dtype
from mime.mappings import gaussian_mac
from mime.nodes.flow.fvm import MACGrid, periodic_tube_layout
from mime.nodes.flow.fvm._exchange import HASIMOTO
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
    assert all(miss <= line for miss, line in misses), max(miss for miss, _ in misses)


def test_one_body_in_still_fluid_coupled_moves_as_it_would_uncoupled(dtype):
    """Gate 2: a point body pulled through a free box (no wall), two-way, at steady state, at its uncoupled Stokes
    speed less Hasimoto's image term. The plant leaves the constant correction out."""
    rng = np.random.default_rng(2000 + seed())
    box = FreeBox(24, float(rng.uniform(0.8, 1.2)), 50.0, mass_ratio=1e-3, force=1e-11, faxen=False)
    if plant("CORRECTION"):
        box.drag.responses = _without_correction(box.drag.responses)  # type: ignore[method-assign]
    for _ in range(40):
        box.step()
    mobility = 1.0 / (6.0 * math.pi * WATER_MU * box.radius)
    expected = mobility * box.force - HASIMOTO * box.force / (6.0 * math.pi * WATER_MU * box.cells * box.h)
    found = float(box.velocity[2])
    assert abs(found - expected) / (mobility * box.force) < 2e-3, (found, expected)


def _without_correction(responses):  # noqa: ANN001, ANN202 - the plant's wrapper
    def planted(p, dt):  # noqa: ANN001, ANN202
        made = responses(p, dt)
        return {**made, "steady": made["steady"] * 0.0}
    return planted


def _final(result, node: str, field: str) -> np.ndarray:  # noqa: ANN001
    return np.asarray(result.final[node][field])


def test_a_neutral_body_in_a_uniformly_driven_flow_moves_with_the_fluid(dtype):
    """Gate 3: on the tube's axis, before the wall's viscous front reaches it, the fluid accelerates uniformly and a
    neutrally buoyant point body with it. The plant switches the Maxey-Riley terms off."""
    rng = np.random.default_rng(3000 + seed())
    exp = point_experiment(radius=float(rng.uniform(30e-6, 50e-6)), offset=float(rng.uniform(0.0, 0.2e-3)), dt=2e-4)
    if plant("MAXEY_RILEY"):
        _patch_node(maxey_riley=False)
    try:
        result = exp.build().run(ms(8.0))
    finally:
        _patch_node()
    velocity = _final(result, "ball", "twist")[2]
    reading = _final(result, "drag:ball", "drag_reading")[2]
    assert abs(velocity - reading) / abs(velocity) < 1e-4, (velocity, reading)


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
    exp = point_experiment(radius=h, offset=offset, dt=5e-4, steady=True, width=1.0)
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
    assert abs(velocity - expected) / centre_speed < 3e-3, (velocity, expected, centre_speed)


def test_one_way_a_heavy_body_slips_with_the_added_masss_closed_form(dtype):
    """Gate 5: one-way (the fluid untouched), a body twice the fluid's density on the axis of a uniformly accelerating
    flow slips by the backward-Euler recurrence of ``(m + m_f/2) dw/dt = (m_f - m) g - R0 w`` (the reading lagged a
    step). The plant switches the added mass off (the fluid-acceleration force kept)."""
    rng = np.random.default_rng(5000 + seed())
    radius, dt, g, ratio = float(rng.uniform(30e-6, 50e-6)), 1e-4, 1.0, float(rng.uniform(1.5, 3.0))
    exp = point_experiment(radius=radius, dt=dt, mass_ratio=ratio, one_way=True, drive=g)
    if plant("ADDED_MASS"):
        _patch_node(added_mass=False)
    try:
        result = exp.build().run(ms(3.0), history=True)
    finally:
        _patch_node()
    velocity = np.asarray(result.history["ball"]["twist"])[:, 2]
    m_f = WATER_RHO * 4.0 / 3.0 * math.pi * radius ** 3
    m, m_a, r0 = ratio * m_f, 0.5 * m_f, 6.0 * math.pi * WATER_MU * radius
    # The fluid on the axis accelerates at g from rest (u^n = g n dt) until the wall's viscous front arrives (R^2/nu =
    # 0.25 s). One-way, the body's law takes nothing back, so the graph steps the flow first and the body reads the
    # field its step has just made, u^(n+1); the rate of its reading from the step after.
    v, predicted = 0.0, []
    for n in range(len(velocity)):
        rate = g if n >= 1 else 0.0
        v = ((m + m_a) / dt * v + r0 * g * (n + 1) * dt + (m_f + m_a) * rate) / ((m + m_a) / dt + r0)
        predicted.append(v)
    terminal = abs((m_f - m) * g / r0)
    assert np.max(np.abs(velocity - np.asarray(predicted))) / terminal < 1e-3


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


def test_the_build_refuses_a_body_no_width_lifts_above_the_floor(dtype):
    """Gate 7: a sphere of 8 cells' radius (the widest kernel, 4 cells, leaves it at a/sigma 2, past a_eff) is refused
    with the floor's code; one of a cell's radius plans."""
    h = 2.0 * TUBE_RADIUS / 32.0
    with pytest.raises(BuildRefused) as refused:
        point_experiment(radius=8.0 * h, dt=1e-4, cells_across=32).build()
    assert "drag.stokes-floor" in {r.code for r in refused.value.refusals}
    point_experiment(radius=h, dt=1e-4, cells_across=32).build()


def test_the_run_dtype_is_the_fixtures(dtype):
    assert str(jnp.dtype(run_dtype())) == dtype


def test_the_books_close_with_the_exchange_booked_under_the_drag(dtype):
    """Gate 8: metered, a heavy point body slipping in the driven tube, two-way: every receiver's books close at every
    step (#233's ``ledger_closes``), the exchange's fluid side booked under the drag's id and the Maxey-Riley part
    under its own row."""
    from mime.observations import EnergyLedger
    from mime.testing import ledger_closes

    exp = point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0)
    exp.observe(EnergyLedger())
    result = exp.build().run(ms(4.0))
    verdict = ledger_closes(result)
    assert verdict.status == "PASS", verdict.detail
    rows = {(law.law, law.receiver) for law in result.energy.laws} if result.energy is not None else set()
    assert {("drag:ball", "water.flow"), ("drag:ball", "ball"), ("drag:ball.maxey_riley", "ball")} <= rows, rows


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
    assert abs(derivative - difference) <= 1e-5 * abs(difference), (derivative, difference)


_HELD_STEPPED = None


def _hold_reading(wrap) -> None:  # noqa: ANN001
    """Wrap the gathered readings a point body's node takes in ``wrap`` (a plant), or restore it."""
    from mime.nodes.hydro import _point_drag

    global _HELD_STEPPED  # noqa: PLW0603 - the plant's own switch
    if _HELD_STEPPED is None:
        _HELD_STEPPED = _point_drag.PointDragNode.stepped
    original = _HELD_STEPPED
    if wrap is None:
        _point_drag.PointDragNode.stepped = original  # type: ignore[method-assign]
        return

    def held(self, p, state, inputs, dt):  # noqa: ANN001, ANN202
        readings = {k: (wrap(v) if k in _point_drag.AVERAGES + _point_drag.LAPLACIANS else v) for k, v in inputs.items()}
        return original(self, p, state, readings, dt)

    _point_drag.PointDragNode.stepped = held  # type: ignore[method-assign]


def _refused(exp) -> set[str]:  # noqa: ANN001
    with pytest.raises(BuildRefused) as refused:
        exp.build()
    return {r.code for r in refused.value.refusals}


def test_the_stokes_rung_refuses_a_body_that_is_no_sphere_a_medium_with_no_grid_and_rk3(float64):
    """The plan's refusals: a body whose hull is no sphere (drag.stokes-sphere), a medium whose flow is not solved on
    a grid (drag.stokes-no-grid), and a flow stepped by RK3 (drag.stokes-scheme); the control, gate 3's experiment,
    plans."""
    assert "drag.stokes-sphere" in _refused(point_experiment(hull="capsule"))
    assert "drag.stokes-no-grid" in _refused(point_experiment(flow=False))
    assert "drag.stokes-scheme" in _refused(point_experiment(dt=1e-5, scheme="rk3"))
    point_experiment().build()
