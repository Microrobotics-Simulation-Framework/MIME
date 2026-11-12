"""F3's second PR: the one elimination over a flow's exchange partners, point bodies held in the flow's step beside its
openings (one pass, the default wherever a body's own step is linear).

**The return-path gate** (gate 1): a drawn step of the flow node holding its partners, against the monolithic
reference (``elimination_cases.monolithic_step``: the whole step, every face, each opening's pressure, each body's
force, couple and twist, as one dense system solved by one LU in float64, independent of the unit fields, their warm
starts, the elimination's small system and the conjugate gradient). Each draw has two partners or more, so that the
cross terms carry: a periodic box with two bodies 1 to 2.5 cells apart, or an open vessel with its two openings and a
body. (The box's bodies were first drawn anywhere in it: at 4.8 and 6.2 cells apart the cross terms moved a body's
force by 1.7e-3 and 1.9e-3, under float32's line, and the cross control missed 2 of 30 float32 draws.) Its controls
(``MIME_PLANT_ELIMINATION``): ``drop``, the last partner dropped from the elimination (its deliveries held at zero);
``cross``, every partner's rows read only its own deliveries (the cross terms zeroed). The line is the field's solve's:
the conjugate gradient stops at its tolerance on the field, which the elimination carries into every partner.

**The ledger** (gate 2): a body coupled one pass, metered, closes its books at every step with the drag's two rows
summing to the law's own dissipation, ``-dt f . (M0 - M_inf) f``, as the drag's node books it: no lagged residual."""

from __future__ import annotations

import math
import os

import jax.numpy as jnp
import numpy as np
import pytest
from elimination_cases import WATER_MU, WATER_RHO, Body, body_inputs, monolithic_step, sphere_mass
from point_cases import point_experiment, seed

import mime.core.elimination as elimination
import mime.nodes.flow.fvm._partners as partners
from mime.core import run_dtype
from mime.nodes.flow.fvm import ImplicitFlowNode, OpenFlowNode
from mime.units import ms, um

#: The return-path gate's lines, by dtype: the largest difference from the monolithic step of the field (on the field's
#: largest magnitude), of each body's force and couple (on their largest) and of its twist (on its largest), and of the
#: openings' pressures and fluxes (on theirs). The field's solve stops at its tolerance (1e-12 in float64, 1e-5 in
#: float32), which every partner's unknowns carry through the elimination.
LINES = {"float64": 1e-8, "float32": 2e-3}


def plant() -> str:
    return os.environ.get("MIME_PLANT_ELIMINATION", "")


def _planted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control's plant in the elimination the flow's step calls (``_partners.exchange_system``)."""
    held = elimination.exchange_system
    kind = plant()
    if not kind:
        return

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
            elif kind == "cross":
                mask = jnp.zeros_like(r.matrix).at[:, own].set(1.0)
                changed.append(elimination.PartnerRows(r.matrix * mask, r.right, r.free))
            else:
                changed.append(r)
        return held(changed)

    monkeypatch.setattr(partners, "exchange_system", planted)


def _periodic_draw(rng: np.random.Generator) -> tuple[ImplicitFlowNode, list[Body], float]:
    """A periodic box of 8 cells a side with two bodies drawn in it, the second 1 to 2.5 cells from the first in a drawn
    direction (so that each reads the other's field above float32's line): radius 0.8 to 1.5 cells, kernel one or 1.5
    cells wide (its reach wraps the box: the gate is the algebra's, which a wrapped kernel holds as any), density 0.5 to
    3 times the water's, a twist of the body's own scales and a load on each."""
    cells, h = 8, 1e-5
    side = cells * h
    dt = float(rng.uniform(0.05, 1.0)) * h * h * WATER_RHO / WATER_MU
    bodies = []
    first = rng.uniform(0.0, side, 3)
    direction = rng.normal(0.0, 1.0, 3)
    second = np.mod(first + float(rng.uniform(1.0, 2.5)) * h * direction / np.linalg.norm(direction), side)
    for at in (first, second):
        a = float(rng.uniform(0.8, 1.5)) * h
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


def _relative(found: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.max(np.abs(reference)))
    return float(np.max(np.abs(np.asarray(found) - reference))) / scale if scale > 0.0 else 0.0


def test_the_coupled_step_is_the_monolithic_step_within_the_solves_line(dtype, monkeypatch):
    """Gate 1, the return path: one drawn step of the flow holding its partners against the monolithic step (the module
    states both), its field, each body's force, couple and twist, and the openings' pressures and fluxes, each within
    the line of the field's solve. Even seeds draw a periodic box with two bodies, odd ones an open vessel with a body
    beside its openings."""
    _planted(monkeypatch)
    rng = np.random.default_rng(seed())
    node, bodies, dt = (_periodic_draw if seed() % 2 == 0 else _open_draw)(rng)
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
    reference = monolithic_step(node, node._step_constants(None), state, inputs, dt)
    solution = np.asarray(ended["partner_solution"])
    found = {"field": max(_relative(np.asarray(ended[f"velocity_{x}"]), reference.velocity[k])
                          for k, x in enumerate("xyz")),
             "force": max(_relative(solution[b, :6], reference.forces[b]) for b in range(len(bodies))),
             "twist": max(_relative(solution[b, 6:12], reference.twists[b]) for b in range(len(bodies)))}
    if isinstance(node, OpenFlowNode):
        found["pressure"] = _relative(np.asarray(ended["opening_pressure"]), reference.pressures)
        found["flux"] = _relative(np.asarray(ended["opening_flux"]), reference.fluxes)
    print(f"F3 PR 2 gate 1 [{dtype}, seed {seed()}, {type(node).__name__}]: " +
          ", ".join(f"{key} {value:.3g}" for key, value in found.items()))
    assert max(found.values()) <= LINES[dtype], found


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


def test_a_derivative_through_the_one_elimination_matches_a_central_difference(float64, monkeypatch):
    """Gate 3: one pass, the body's axial position after 10 steps in the fluid's viscosity, forward mode by
    ``Study.gradient``, against a central difference of two runs (gate 9's case, lagged there): the derivative passes
    through the unit fields' solves and the elimination's LU. The plant (``MIME_PLANT_ELIMINATION=held``) holds the
    elimination's solution out of the derivative, which must fail it."""
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
        position, wrt=["water.viscosity"], mode="forward",
        acknowledged={"water.flow": "the derivative is in the viscosity, which the FVM's gradient gate checks exact; "
                                    "its unverified geometry leaves are not moved"})
    derivative = float(np.asarray(result.member(0).gradient["water.viscosity"]))
    step = 1e-3 * WATER_MU
    ends = [float(point_experiment(radius=40e-6, dt=2e-4, mass_ratio=2.0, viscosity=mu).build().run(ms(2.0))
                  .final["ball"]["pose"][2]) for mu in (WATER_MU + step, WATER_MU - step)]
    difference = (ends[0] - ends[1]) / (2.0 * step)
    print(f"F3 PR 2 gate 3: derivative {derivative:.10g}, central difference {difference:.10g}, relative "
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
