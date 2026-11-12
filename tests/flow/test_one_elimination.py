"""F3's second PR: the one elimination over a flow's exchange partners, point bodies held in the flow's step beside its
openings (one pass, the default wherever a body's own step is linear).

**The return-path gate** (gate 1): a drawn step of the flow node holding its partners, against the monolithic
reference (``elimination_cases.monolithic_step``: the whole step, every face, each opening's pressure, each body's
force, couple and twist, as one dense system solved by one LU in float64, independent of the unit fields, their warm
starts, the elimination's small system and the conjugate gradient). Each draw has two partners or more, so that the
cross terms carry: a periodic box with two bodies anywhere in it that two spheres can be (their centres at least the
sum of their radii apart), or an open vessel with its two openings and a body. The draw is fresh each run
(``point_cases.fresh_seed``). **Its line is derived, per draw and per quantity** (``elimination_cases.float_lines``):
the step's own solves' achieved residual and the dtype's rounding carried through the monolithic step's inverse,
each quantity in its own units (a face's velocity, an opening's pressure and flux, and each body's force, couple,
velocity and spin, each vector against its own components' largest line), so a force is judged against force and a
spin against spin, never one scale for quantities some 1e5 apart. Its controls (``MIME_PLANT_ELIMINATION``): ``drop``, the last partner dropped from the elimination (its
deliveries held at zero); ``cross``, every partner's rows read only its own deliveries (the cross terms zeroed);
``crossx<f>``, the cross terms scaled by ``f`` (the discriminating range's); ``twistsign``, the closed velocity's sign
flipped in the solution the flow writes (the refuting review's mutant, which one scale for twist and spin missed in
float32 at every periodic draw).

**The ledger** (gate 2): a body coupled one pass, metered, closes its books at every step with the drag's two rows
summing to the law's own dissipation, ``-dt f . (M0 - M_inf) f``, as the drag's node books it: no lagged residual."""

from __future__ import annotations

import math
import os
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from elimination_cases import WATER_MU, WATER_RHO, Body, body_inputs, float_lines, monolithic_step, sphere_mass
from point_cases import fresh_seed, point_experiment, seed
from point_cases import plant as plant_named

import mime.core.elimination as elimination
import mime.nodes.flow.fvm._partners as partners
from mime.core import run_dtype
from mime.nodes.flow.fvm import ImplicitFlowNode, OpenFlowNode
from mime.units import ms, um

def plant() -> str:
    return os.environ.get("MIME_PLANT_ELIMINATION", "")


def _planted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control's plant in the elimination the flow's step calls (``_partners.exchange_system``)."""
    held = elimination.exchange_system
    kind = plant()
    if not kind:
        return
    if kind == "twistsign":
        import mime.nodes.flow.fvm._node as flow_node
        import mime.nodes.flow.fvm._open_node as open_node

        stepped = partners.partnered_step

        def flipped(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            found = stepped(*args, **kwargs)
            return found._replace(twists=found.twists.at[:, :3].multiply(-1.0))

        monkeypatch.setattr(flow_node, "partnered_step", flipped)
        monkeypatch.setattr(open_node, "partnered_step", flipped)
        return
    factor = float(kind[len("crossx"):]) if kind.startswith("crossx") else None

    def planted(rows):  # noqa: ANN001, ANN202
        counts = [int(r.matrix.shape[0]) for r in rows]
        starts = np.cumsum([0, *counts])
        changed = []
        for k, r in enumerate(rows):
            own = slice(int(starts[k]), int(starts[k + 1]))
            if kind == "drop" and k == len(rows) - 1:
                matrix = jnp.zeros_like(r.matrix).at[:, own].set(jnp.eye(counts[k], dtype=r.matrix.dtype))
                free = None if r.free is None else jnp.zeros_like(r.free)
                changed.append(elimination.PartnerRows(matrix, jnp.zeros_like(r.right), free))
            elif kind == "cross" or factor is not None:
                mask = jnp.zeros_like(r.matrix).at[:, own].set(1.0)
                scale = mask if factor is None else mask + factor * (1.0 - mask)
                changed.append(elimination.PartnerRows(r.matrix * scale, r.right, r.free))
            else:
                changed.append(r)
        return held(changed)

    monkeypatch.setattr(partners, "exchange_system", planted)


def _periodic_draw(rng: np.random.Generator) -> tuple[ImplicitFlowNode, list[Body], float]:
    """A periodic box of 8 cells a side with two bodies drawn anywhere in it that two spheres can be (redrawn until
    their centres' least distance across the box's images is at least the sum of their radii): radius 0.8 to 1.5
    cells, kernel one or 1.5 cells wide (its reach wraps the box: the gate is the algebra's, which a wrapped kernel
    holds as any), density 0.5 to 3 times the water's, a twist of the body's own scales and a load on each."""
    cells, h = 8, 1e-5
    side = cells * h
    dt = float(rng.uniform(0.05, 1.0)) * h * h * WATER_RHO / WATER_MU
    radii = [float(rng.uniform(0.8, 1.5)) * h for _ in range(2)]
    while True:
        points = rng.uniform(0.0, side, (2, 3))
        apart = points[1] - points[0]
        apart -= side * np.round(apart / side)
        if float(np.linalg.norm(apart)) >= radii[0] + radii[1]:
            break
    bodies = []
    for at, a in zip(points, radii, strict=True):
        mass, moment = sphere_mass(a, float(rng.uniform(0.5, 3.0)))
        point = tuple(float(c) for c in at)
        twist = (*(rng.normal(0.0, 1e-3, 3)), *(rng.normal(0.0, 1e-3 / a, 3)))
        loads = (*(rng.normal(0.0, 6 * math.pi * WATER_MU * a * 1e-3, 3)),
                 *(rng.normal(0.0, 8 * math.pi * WATER_MU * a ** 3 * 1e-3 / a, 3)))
        bodies.append(Body(a, point, tuple(float(t) for t in twist), mass, moment, float(rng.choice([1.0, 1.5])),
                           tuple(float(f) for f in loads)))
    node = ImplicitFlowNode("flow", dt, cells_x=cells, cells_y=cells, cells_z=cells, spacing=h, viscosity=WATER_MU,
                            density=WATER_RHO, tube_centre=(side / 2, side / 2, 0.0), tube_axis=(0.0, 0.0, 1.0),
                            tube_radius=0.35 * side, scheme="imex_euler", convection=False,
                            tolerance=1e-12 if run_dtype() == jnp.float64 else 1e-5,
                            partner_radius=[b.radius for b in bodies], partner_widths=[b.width for b in bodies],
                            partner_ids=["drag:a", "drag:b"])
    return node, bodies, dt


def _open_draw(rng: np.random.Generator) -> tuple[OpenFlowNode, list[Body], float]:
    """An open vessel (8 x 8 x 16 cells, nondimensional: diameter, viscosity and density one), an inflow at its low
    plane and a Windkessel at its high one, and a body drawn in its middle third, its kernel one cell wide (its
    truncated reach meets the planes, where its weights are under 1e-5 of the peak: the gate is the algebra's, and the
    monolithic step reads and delivers through the same kernel as the node)."""
    h = 1.0 / 8.0
    length = 16 * h
    dt = float(rng.uniform(0.005, 0.05))
    resistance = 8.0 * length / (math.pi * 0.35 ** 4)
    params = [[float(rng.uniform(0.05, 0.2)) * math.pi / 4.0, 0.0, 0.0, 0.0],
              [float(rng.uniform(0.1, 1.0)) * resistance, 1.0 / resistance, 3.0 * resistance, 0.0]]
    a = float(rng.uniform(0.8, 1.5)) * h
    mass = 4.0 / 3.0 * math.pi * a ** 3 * float(rng.uniform(0.5, 3.0))
    moment = 0.4 * mass * a * a
    centre = (0.5, 0.5)
    point = (centre[0] + float(rng.uniform(-0.1, 0.1)), centre[1] + float(rng.uniform(-0.1, 0.1)),
             float(rng.uniform(length / 3, 2 * length / 3)))
    twist = (*(rng.normal(0.0, 0.03, 3)), *(rng.normal(0.0, 0.03 / a, 3)))
    loads = (*(rng.normal(0.0, 6 * math.pi * a * 0.03, 3)), *(rng.normal(0.0, 8 * math.pi * a ** 2 * 0.03, 3)))
    body = Body(a, point, tuple(float(t) for t in twist), mass, moment, 1.0, tuple(float(f) for f in loads))
    node = OpenFlowNode("flow", dt, cells_x=8, cells_y=8, cells_z=16, spacing=h, viscosity=1.0, density=1.0,
                        tube_centre=(0.5, 0.5, 0.0), tube_axis=(0.0, 0.0, 1.0), tube_radius=0.35,
                        openings=("tube.start", "tube.end"), opening_laws=("inflow", "windkessel"),
                        opening_parameters=params, convection=False,
                        tolerance=1e-12 if run_dtype() == jnp.float64 else 1e-5,
                        partner_radius=[a], partner_widths=[body.width], partner_ids=["drag:ball"])
    return node, [body], dt


def gate_one(drawn: int) -> dict[str, float]:
    """Gate 1 at the draw of ``drawn``: each quantity's largest difference from the monolithic step over its derived
    line (a ratio; the gate holds at most one), by block: the field, the bodies' force, couple, velocity and spin, and
    an open vessel's pressures and fluxes. Even seeds draw a periodic box with two bodies, odd ones an open vessel with
    a body beside its openings."""
    rng = np.random.default_rng(drawn)
    node, bodies, dt = (_periodic_draw if drawn % 2 == 0 else _open_draw)(rng)
    state = dict(node.initial_state())
    scale = 1e-3 if isinstance(node, ImplicitFlowNode) and not isinstance(node, OpenFlowNode) else 0.03
    for x in "xyz":
        state[f"velocity_{x}"] = jnp.asarray(rng.normal(0.0, scale, np.shape(state[f"velocity_{x}"])),
                                             dtype=run_dtype())
    inputs = {name: value for b, body in enumerate(bodies) for name, value in body_inputs(body, b, dt,
                                                                                          run_dtype()).items()}
    if not isinstance(node, OpenFlowNode):
        inputs["body_force"] = jnp.asarray(rng.normal(0.0, 0.5, 3), dtype=run_dtype())
    ended = node.update(dict(state), inputs, dt)
    p = node._step_constants(None)
    reference = monolithic_step(node, p, state, inputs, dt)
    lines = float_lines(node, p, state, inputs, dt, reference, ended)
    solution = np.asarray(ended["partner_solution"], dtype=np.float64)

    def over(found: Any, wanted: Any, line: Any) -> float:
        """The largest difference over the line: of each face, each opening's value, or each body's vector (its
        force, couple, velocity or spin, a row of three) against the largest line of its components, a vector's
        rounding being its own size's, not each component's."""
        miss = np.abs(np.asarray(found, dtype=np.float64) - wanted)
        if miss.ndim == 2:
            return float(np.max(np.max(miss, axis=1) / np.max(line, axis=1)))
        return float(np.max(miss / line))

    field = np.concatenate([np.ravel(np.asarray(ended[f"velocity_{x}"], dtype=np.float64)) for x in "xyz"])
    wanted = np.concatenate([np.ravel(v) for v in reference.velocity])
    found = {"field": over(field, wanted, lines.field),
             "force": over(solution[:, :3], reference.forces[:, :3], lines.forces[:, :3]),
             "couple": over(solution[:, 3:6], reference.forces[:, 3:], lines.forces[:, 3:]),
             "velocity": over(solution[:, 6:9], reference.twists[:, :3], lines.twists[:, :3]),
             "spin": over(solution[:, 9:12], reference.twists[:, 3:], lines.twists[:, 3:])}
    if isinstance(node, OpenFlowNode):
        found["pressure"] = over(ended["opening_pressure"], reference.pressures, lines.pressures)
        found["flux"] = over(ended["opening_flux"], reference.fluxes, lines.fluxes)
    sharpest = {"force": float(np.max(lines.forces[:, :3]) / np.max(np.abs(reference.forces[:, :3]))),
                "spin": float(np.max(lines.twists[:, 3:]) / np.max(np.abs(reference.twists[:, 3:])))}
    print(f"F3 PR 2 gate 1 [{jnp.dtype(run_dtype()).name}, seed {drawn}, "
          f"{type(node).__name__}]: over the line " + ", ".join(f"{key} {value:.3g}" for key, value in found.items()) +
          f"; the line against the force's and the spin's own scales {sharpest['force']:.2g}, {sharpest['spin']:.2g}; "
          f"the solves' reported residual {lines.parts['achieved']:.3g}")
    return found


def test_the_coupled_step_is_the_monolithic_step_within_its_derived_line(dtype, monkeypatch):
    """Gate 1, the return path: one freshly drawn step of the flow holding its partners against the monolithic step
    (the module states both), its field, each body's force, couple, velocity and spin, and the openings' pressures and
    fluxes, each within its line derived from the step's own solves and the dtype's rounding (``float_lines``)."""
    _planted(monkeypatch)
    drawn = fresh_seed()
    found = gate_one(drawn)
    assert max(found.values()) <= 1.0, (f"DETECTION_SEED={drawn}", found)


def _metered(coupling: str | None):  # noqa: ANN202
    from mime.observations import EnergyLedger

    exp = point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0, coupling=coupling)
    exp.observe(EnergyLedger())
    built = exp.build()
    return built, built.run(ms(4.0))


def _last_step_books(built, result) -> float:  # noqa: ANN001
    """The law's own dissipation in the run's last step, which the drag's node books the step after (one pass): the
    force the flow solved against the slip from its kernel's reading of the solved field, ``dt f . (V - G_F u')``."""
    node = built.graph.get_node("drag:ball")
    drag, _, _ = node.coefficients(node._step_constants(None))
    row = np.asarray(result.final["water.flow"]["partner_solution"])[0]
    kernel = row[12:]
    reading = np.concatenate([kernel[:3] + float(drag) * kernel[3:6], kernel[6:9]])
    twist = np.asarray(result.final["ball"]["twist"])
    return float(node.delta_t) * float(np.dot(row[:6], twist - reading))


def test_one_pass_the_books_close_with_the_drags_rows_summing_to_its_own_dissipation(dtype, monkeypatch):
    """Gate 2, the ledger in the affine scheme: metered, a heavy point body slipping in the driven tube, one pass:
    every receiver's books close at every step, the drag's rows on the body and on the fluid both judged by its node
    (none a gift), and their sum over the run is minus the law's own dissipation as its node books it (``dissipated``,
    the force against the slip from the reading it was solved with; its last step's, which the node books the step
    after, added from the run's final state), to the books' rounding: the exchange is workless, with no lagged
    residual. The lagged route's sum differs from its node's dissipation by the exchange's unpaired work, first order
    in the step: its share is printed beside. The plant (``MIME_PLANT_ELIMINATION=handed``) adds a hundredth of the
    handed load's size to it along the body's motion: the exchange then works on the body, and the sum misses."""
    from mime.nodes.flow.fvm import ImplicitFlowNode as Flow
    from mime.testing import ledger_closes

    if plant() == "handed":
        ended = Flow.partner_ended

        def planted(self, grid, bodies, partnered, held_units):  # noqa: ANN001, ANN202
            found = ended(self, grid, bodies, partnered, held_units)
            load = found["partner_0_load"]
            return {**found, "partner_0_load": load + 0.01 * jnp.sqrt(jnp.sum(load * load))}

        monkeypatch.setattr(Flow, "partner_ended", planted)
    shares = {}
    for coupling in (None, "lagged"):
        built, result = _metered(coupling)
        verdict = ledger_closes(result)
        assert verdict.status == "PASS", verdict.detail
        assert result.energy is not None
        rows = {(law.law, law.receiver): law for law in result.energy.laws}
        gifts = {receiver: rows[("drag:ball", receiver)].gift_steps for receiver in ("ball", "water.flow")}
        assert gifts == {"ball": 0, "water.flow": 0}, gifts
        paired = float(rows[("drag:ball", "ball")].work) + float(rows[("drag:ball", "water.flow")].work)
        books = float(np.asarray(result.final["drag:ball"]["dissipated"]))
        if coupling is None:
            books += _last_step_books(built, result)
        shares[coupling or "one_pass"] = abs(paired + books) / abs(books)
    print(f"F3 PR 2 gate 2 [{dtype}]: the drag's paired work against its own dissipation, relative: {shares}")
    assert shares["one_pass"] <= (1e-9 if dtype == "float64" else 1e-3), shares


def _kinds(exp) -> set[str]:  # noqa: ANN001
    """The registration kinds the flow takes from the experiment's plan."""
    plans = exp.plan()
    return {r.kind for plan in plans.entries for r in plan.registrations}


def _refused(exp) -> set[str]:  # noqa: ANN001
    from mime.build import BuildRefused

    with pytest.raises(BuildRefused) as refused:
        exp.plan()
    return {r.code for r in refused.value.refusals}


def _contact(exp, *, body: str = "ball"):  # noqa: ANN001, ANN202
    from mime.interactions import Contact

    exp.attach(Contact(rung="projection", margin=um(5.0)), body=body, surface="tube.inner_surface")
    return exp


def test_the_plan_holds_a_free_body_one_pass_and_lags_one_whose_step_is_not_linear_with_its_reason(float64):
    """The scheme (the maintainer's rule, item 2 (a)): a free body is held one pass; one with a contact (its step
    projected) or with coupling='lagged' written is lagged, never refused, and its node's build line says why;
    coupling='one_pass' written on the contact's body is refused by name (drag.stokes-one-pass), and a coupling the
    rung does not know is refused. What the experiment observes never chooses it: with the energy ledger attached the
    free body is still held one pass. The controls: each lagged plan's kind is the lagged one."""
    from mime.observations import EnergyLedger
    from mime.vocabulary import EXCHANGE_PARTNER, ONE_PASS_PARTNER

    assert _kinds(point_experiment()) == {ONE_PASS_PARTNER}
    metered = point_experiment()
    metered.observe(EnergyLedger())
    assert _kinds(metered) == {ONE_PASS_PARTNER}
    assert _kinds(point_experiment(coupling="lagged")) == {EXCHANGE_PARTNER}
    assert _kinds(_contact(point_experiment())) == {EXCHANGE_PARTNER}
    assert _kinds(point_experiment(one_way=True)) == {EXCHANGE_PARTNER}
    assert _refused(_contact(point_experiment(coupling="one_pass"))) == {"drag.stokes-one-pass"}
    assert _refused(point_experiment(coupling="sometimes"))
    built = _contact(point_experiment(dt=1e-4)).build()
    line = next(line for line in built.report.checked["drag:ball"] if line.startswith("lagged"))
    assert "not one the flow's one elimination can hold" in line, line


def test_the_build_report_names_lagged_the_cheaper_route_from_the_measured_body_count(float64):
    """The build report's line for the bodies a flow holds one pass names coupling='lagged' as the cheaper route from
    HELD_SUGGESTS_LAGGED bodies on (the maintainer's item 2 (a)), the count set from the cost measured through the node.
    The control: one body fewer, the line states the cost and suggests nothing."""
    from mime.nodes.flow.fvm._node import HELD_SUGGESTS_LAGGED

    def line(count: int) -> str:
        h = 1e-5
        node = ImplicitFlowNode("flow", 1e-4, cells_x=8, cells_y=8, cells_z=8, spacing=h, viscosity=WATER_MU,
                                density=WATER_RHO, tube_centre=(4 * h, 4 * h, 0.0), tube_axis=(0.0, 0.0, 1.0),
                                tube_radius=2.8 * h, scheme="imex_euler", convection=False,
                                partner_radius=[h] * count, partner_widths=[1.0] * count,
                                partner_ids=[f"drag:b{i}" for i in range(count)])
        (held,) = node.held_lines()
        return held

    assert "coupling='lagged'" in line(HELD_SUGGESTS_LAGGED)
    fewer = line(HELD_SUGGESTS_LAGGED - 1)
    assert f"{6 * (HELD_SUGGESTS_LAGGED - 1)} unit fields" in fewer and "lagged" not in fewer, fewer


def test_an_open_vessels_flow_holds_a_body_one_pass_and_refuses_a_lagged_one_by_name(float64):
    """An open vessel's flow holds its partners in its one elimination only: a free body plans one pass beside the
    openings; a one-way body, a contact's body (lagged) and one with coupling='lagged' are refused by name
    (flow.open-exchange), with the route. The control: the same free body in the open vessel plans (an open vessel is
    laid along +z, flow.open-axis, so a pose is carried into its grid by a translation)."""
    from mime.vocabulary import ONE_PASS_PARTNER

    assert ONE_PASS_PARTNER in _kinds(point_experiment(opened=True))
    for exp in (point_experiment(opened=True, one_way=True), point_experiment(opened=True, coupling="lagged"),
                _contact(point_experiment(opened=True))):
        assert _refused(exp) == {"flow.open-exchange"}


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_a_derivative_through_the_one_elimination_matches_a_central_difference(float64, monkeypatch, mode):
    """Gate 3: one pass, the body's axial position after 10 steps in the fluid's viscosity, forward and reverse mode by
    ``Study.gradient`` (reverse is what its auto mode takes past two numbers), against a central difference of two runs
    (gate 9's case, lagged there): the derivative passes through the unit fields' solves and the elimination's LUs. The
    plant (``MIME_PLANT_ELIMINATION=held``) holds the elimination's solution out of the derivative, which must fail
    it."""
    import jax

    from mime.study import Study, along
    from mime.units import Pa_s

    if plant() == "held":
        held = elimination.exchange_system

        def stopped(rows):  # noqa: ANN001, ANN202
            solved = held(rows)
            return elimination.Eliminated(*(None if v is None else jax.lax.stop_gradient(v) for v in solved))

        monkeypatch.setattr(partners, "exchange_system", stopped)

    def position(final):  # noqa: ANN001, ANN202
        return final["ball"]["pose"][2]

    exp = point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0)
    result = Study(exp).sweep(along({"water.viscosity": [Pa_s(1e-3)]}), duration=ms(2.0)).gradient(
        position, wrt=["water.viscosity"], mode=mode,
        acknowledged={"water.flow": "the derivative is in the viscosity, which the FVM's gradient gate checks exact; "
                                    "its unverified geometry leaves are not moved"})
    derivative = float(np.asarray(result.member(0).gradient["water.viscosity"]))
    step = 1e-3 * WATER_MU
    ends = [float(point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0, viscosity=mu).build().run(ms(2.0))
                  .final["ball"]["pose"][2]) for mu in (WATER_MU + step, WATER_MU - step)]
    difference = (ends[0] - ends[1]) / (2.0 * step)
    print(f"F3 PR 2 gate 3 [{mode}]: derivative {derivative:.10g}, central difference {difference:.10g}, relative "
          f"{abs(derivative - difference) / abs(difference):.3g}")
    assert abs(derivative - difference) <= 1e-5 * abs(difference), (derivative, difference)


def test_the_body_steps_with_the_row_the_flow_closed_and_the_flag_sees_a_load_the_row_missed(dtype, monkeypatch):
    """Gate 4, the build's mirror: one pass, a heavy body in the driven tube; its every other load and resistance (the
    rate terms' and the added mass's) reaches the flow's row as it reaches the body, so the body's twist is the one the
    flow closed, to rounding: ``row_mismatch`` under its line and ``row_mismatch_flagged`` never raised over the run.
    The plant (``MIME_PLANT_ELIMINATION=mirror``) drops the mirror: the body then steps with loads its row did not hold,
    and the flag is raised."""
    import mime.build._construct as construct

    if plant() == "mirror":
        monkeypatch.setattr(construct, "held_rows", lambda built, edges: [])
    rng = np.random.default_rng(4000 + seed())
    radius, ratio = float(rng.uniform(30e-6, 50e-6)), float(rng.uniform(1.5, 3.0))
    exp = point_experiment(radius=radius, dt=2e-4, mass_ratio=ratio)
    result = exp.build().run(ms(4.0), history=True)
    mismatch = np.asarray(result.history["drag:ball"]["row_mismatch"])
    flagged = np.asarray(result.history["drag:ball"]["row_mismatch_flagged"])
    print(f"F3 PR 2 gate 4 [{dtype}] a={radius:.3g} ratio {ratio:.2f}: worst row mismatch {mismatch.max():.3g}, "
          f"flagged in {int(flagged.sum())} of {flagged.size} steps")
    assert flagged.sum() == 0.0, mismatch.max()


#: The pair's line, relative to its products' magnitudes: a few units of rounding (the periodic box's own gate,
#: ``tests/flow/test_point_bodies.py``'s ``ADJOINT_LINE``).
PAIR_LINE = {"float32": 1e-5, "float64": 1e-13}
#: The open grid's draws: this many a run, at 0.8 to 4.8 cells from a drawn plane, the first within 1.6 (where the old
#: scatter, the control, missed by 6.9e-2 and more), and this many on the periodic box.
PAIR_DRAWS = {"open": 40, "periodic": 20}


def _pair_miss(rng: np.random.Generator, grid: Any, dtype: str, near: float | None) -> float:
    """``<S_F f, u>_W`` against ``f . G_F u`` (the held body's delivery and reading, ``_partners.delivered`` and
    ``read``) at one drawn point, force and couple, Faxén coefficient and field, relative to the products'
    magnitudes; ``near``, the point that many cells from a drawn plane of the open grid (``None``, anywhere in the
    periodic box)."""
    from mime.nodes.flow.fvm._exchange import faxen
    from mime.nodes.flow.fvm._partners import delivered, read, stencils

    h = float(grid.spacing)
    cells = grid.cells
    width = float(rng.choice([1.0, 1.5]))
    z = float(rng.uniform(0.0, cells[2] * h)) if near is None else (
        near * h if rng.random() < 0.5 else cells[2] * h - near * h)
    point = jnp.asarray([*(rng.uniform(0.0, cells[0] * h, 2)), z], dtype=dtype)
    coefficient = faxen(float(rng.uniform(0.8, 1.5)) * h, width * h, 1.0 / 6.0)
    values = jnp.asarray(rng.normal(size=6), dtype=dtype)
    field = tuple(jnp.asarray(rng.normal(size=grid.shape(k)), dtype=dtype) for k in range(3))
    held = stencils(grid, point, width)
    density = delivered(grid, held, values, coefficient)
    reading = read(grid, held, field, coefficient)
    left, right = grid.inner(density, field), jnp.sum(values * reading)
    scale = grid.inner_magnitude(density, field) + jnp.sum(jnp.abs(values * reading))
    return float(jnp.abs(left - right) / scale)


def test_the_held_bodys_delivery_and_reading_are_a_transpose_pair_up_to_an_open_plane(dtype, monkeypatch):
    """The exchange a flow holds a body by is a transpose pair in its grid's inner product, ``<S_F f, u>_W = f . G_F u``,
    to rounding: on the open grid at 0.8 to 4.8 cells from either plane (the kernel folded onto the plane's faces,
    which carry half a cell), and on the periodic box, fresh draws each run. The control (``MIME_PLANT_PAIR=whole``)
    puts back the scatter that delivered a whole cell's density on the planes' faces (#270's review, M4: the exchange
    then did work the books did not pair), and fails every run: its first open draw lies within 1.6 cells."""
    from mime.nodes.flow.fvm import MACGrid

    if plant_named("PAIR") == "whole":
        held_scatter = partners.scatter

        def whole(grid, component, values, stencil, mode="average"):  # noqa: ANN001, ANN202
            density = held_scatter(grid, component, values, stencil, mode)
            weighed = getattr(grid, "weights", None)
            return density if weighed is None else density * weighed(values.dtype)[component]

        monkeypatch.setattr(partners, "scatter", whole)
    drawn = fresh_seed()
    rng = np.random.default_rng(drawn)
    node, _, _ = _open_draw(np.random.default_rng(drawn))
    opened = node.grid(node._step_constants(None))
    periodic = MACGrid((8, 8, 8), 1e-5)
    misses = [_pair_miss(rng, opened, dtype, float(rng.uniform(0.8, 1.6 if k == 0 else 4.8)))
              for k in range(PAIR_DRAWS["open"])]
    misses += [_pair_miss(rng, periodic, dtype, None) for _ in range(PAIR_DRAWS["periodic"])]
    print(f"F3 PR 2 the held pair [{dtype}, seed {drawn}]: largest relative miss {max(misses):.3g} (open grid "
          f"{max(misses[:PAIR_DRAWS['open']]):.3g}, periodic {max(misses[PAIR_DRAWS['open']:]):.3g})")
    assert max(misses) <= PAIR_LINE[dtype], (f"DETECTION_SEED={drawn}", max(misses))
