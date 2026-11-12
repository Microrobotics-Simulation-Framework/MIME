"""F5's Schwarz coupling through the flow node and the build.

- **Gate 3 through the node**: the flow node holding an MRS sphere (its initial preparation, then one step composed
  from the body's twist) against the monolithic step (``schwarz_cases.monolithic_step``), float64, IMEX Euler, with a
  drive: the field, the tractions' wrench (the load the body took) and the composed residual. Control: the composition
  dropped (the field left at its base, the return path's plant), which misses the monolithic field past the line.
- **What is frozen** (gate 6): the next step's columns are prepared at the pose the body reached, and the step composed
  at the pose the columns were prepared at, never the live one. Control: the composition read at the body's new pose
  misses the monolithic field.
- **The return path and the ledger** (gate 5): three node steps' stored field and kinetic energy against the
  monolithic step from the node's own state (control: the composition dropped); metered, every receiver's books
  close, the hand-over is workless and the drag's two rows sum to the near field's own dissipation less the free box's
  (control: a hand-over gift).
- **A derivative through the first step**: the first preparation is made again in the derivative's program
  (``MimeNode.started``), so the body's position after two steps has the central difference's derivative in the
  viscosity (control: the preparation held as the initial state's).
- **The build**: ``Drag.MRS`` in a medium the FVM fills is a Schwarz partner (no drag node; the body added before its
  medium's flow; the response a prepared delivery, read the step before and planned), and the refusals with their
  routes: a near field with a wall kind of its own, a second body in the flow, and the exchange's opt-ins written where
  the medium has no flow on a grid."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import decayed, draw, monolithic_step, pieces, plant, schwarz_flow, sphere_member
from schwarz_experiments import schwarz_sphere as experiment

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.nodes.flow.fvm._schwarz import RESIDUAL_LINE, image_terms, stencils

SPACING, CELLS, LENGTH, TUBE, RADIUS, DT = 1e-4, 10, 8, 4e-4, 1.5e-4, 1e-3
BOX = 32
LINE = 1e-7


@contextlib.contextmanager
def indefinite_planted(on: bool) -> Iterator[None]:
    """Gate 9's plant: the response the flow prepares with its sign turned (``MIME_PLANT_SCHWARZ=indefinite``), which
    the step's definiteness check stops the run at, its state then held."""
    import mime.nodes.flow.fvm._node as flow_node

    original = flow_node.prepared

    def turned(*args: Any, **kwargs: Any) -> Any:
        made = original(*args, **kwargs)
        return made._replace(resistance=-made.resistance)

    if on:
        flow_node.prepared = turned
    try:
        yield
    finally:
        flow_node.prepared = original


def node_step(drop_composition: bool = False, live_pose: bool = False) -> dict[str, float]:
    drop_composition = drop_composition or plant() == "drop"
    live_pose = live_pose or plant() == "live"
    with jax.enable_x64(True):
        member = sphere_member(RADIUS, 1e-4)
        rng = draw()
        pose = np.array([*(np.array([0.5 * CELLS, 0.5 * CELLS, 0.5 * LENGTH]) + rng.uniform(-0.5, 0.5, 3)) * SPACING,
                         1.0, 0.0, 0.0, 0.0])
        node = schwarz_flow(member, pose, CELLS, LENGTH, SPACING, TUBE, DT, 2 * RADIUS, drive=(0.0, 0.0, 2.0),
                            schwarz_box=BOX, schwarz_tolerance=1e-11)
        state = node.initial_state()
        mass = np.diag([3e-8] * 3 + [5e-16] * 3) / DT
        twist0 = np.zeros(6)
        loads = rng.uniform(-1.0, 1.0, 6) * np.array([2e-9] * 3 + [1e-16] * 3)
        right = mass @ twist0 + loads
        resistance, load = np.asarray(state["schwarz_0_resistance"]), np.asarray(state["schwarz_0_load"])
        twist = np.linalg.solve(mass + resistance, right + load)
        moved = pose.copy()
        moved[:3] += DT * twist[:3]
        p = node._step_constants(node.params_pytree())  # noqa: SLF001 - the step's own constants, as a run gives them
        given = dict(state)
        if live_pose:
            given["schwarz_0_at"] = jnp.asarray(moved)
        with indefinite_planted(plant() == "indefinite"):
            ended = node.advance(p, given, {"schwarz_0_twist": jnp.asarray(twist),
                                            "schwarz_0_pose": jnp.asarray(moved),
                                            "waveform": jnp.asarray([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0])}, DT)
        at = pieces(node, DT)
        box = node.schwarz_box()
        host = node.schwarz
        body = host.body(p, jnp.asarray(pose))
        held = stencils(at.grid, box, body.placed.force, 1.0)
        fields = node.fields(p, at.grid)
        drive = node.drive(fields, jnp.asarray([0.0, 0.0, 2.0]))
        before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
        rhs = tuple(at.shift * before[k] + drive[k] for k in range(3))
        zero = jnp.zeros((BOX, BOX, BOX))
        reference = monolithic_step(at, box, held, body, rhs, (zero, zero, zero), jnp.zeros(3), mass, right, None)
        field = (ended["velocity_x"], ended["velocity_y"], ended["velocity_z"]) if not drop_composition else \
            (state["schwarz_base_x"], state["schwarz_base_y"], state["schwarz_base_z"])
        found = np.concatenate([np.ravel(c) for c in field])
        wanted = np.concatenate([np.ravel(c) for c in reference.velocity])
        wrench = np.asarray(body.placed.wrench()).T @ reference.tractions
        took = resistance @ twist - load
        return dict(field=float(np.linalg.norm(found - wanted) / np.linalg.norm(wanted)),
                    twist=float(np.linalg.norm(twist - reference.twist) / np.linalg.norm(reference.twist)),
                    wrench=float(np.linalg.norm(took - wrench) / np.linalg.norm(wrench)),
                    residual=float(ended["schwarz_residual"]), flagged=float(ended["schwarz_flagged"]),
                    at=float(np.max(np.abs(np.asarray(ended["schwarz_0_at"]) - moved))))


def test_the_flow_node_steps_as_the_monolithic_step():
    found = node_step()
    assert found["field"] < LINE and found["twist"] < LINE and found["wrench"] < LINE, found
    assert found["residual"] < RESIDUAL_LINE * 1e-11 and found["flagged"] == 0.0, found
    assert found["at"] == 0.0, found


def test_the_composition_dropped_misses_the_monolithic_field():
    assert node_step(drop_composition=True)["field"] > LINE


def test_a_position_read_from_the_live_pose_misses_the_monolithic_field_and_is_flagged():
    found = node_step(live_pose=True)
    assert found["field"] > LINE and found["flagged"] == 1.0, found


def refusals(exp: Experiment) -> set[str]:
    with pytest.raises(BuildRefused) as caught:
        exp.build()
    return {refusal.code for refusal in caught.value.refusals}


def test_an_mrs_drag_in_a_flowing_medium_is_a_schwarz_partner_its_body_before_the_flow():
    built = experiment().build()
    order = [line.node for line in built.report.nodes]
    assert "drag:ball" not in order
    assert order.index("ball") < order.index("water.flow")
    handed = [lag for lag in built.report.lags if lag.source[0] == "water.flow"]
    assert {lag.target for lag in handed} == {("ball", "resistance"), ("ball", "background_load")}
    assert all(lag.prepared and lag.planned and lag.lag.value == "previous step" for lag in handed)
    read = {lag.target[1] for lag in built.report.lags if lag.source[0] == "ball" and lag.target[0] == "water.flow"}
    assert read == {"schwarz_0_pose", "schwarz_0_twist"}


@pytest.mark.parametrize(("case", "code"), [
    ({"wall_kind": "tube"}, "drag.schwarz-wall-kind"),
    ({"second": "bob"}, "flow.schwarz-bodies"),
    ({"second": "pebble"}, "flow.schwarz-bodies"),
    ({"settings": {"one_way": True}, "flowing": False}, "drag.schwarz-scheme"),
    ({"settings": {"coupling": "lagged"}, "flowing": False}, "drag.schwarz-scheme"),
])
def test_what_the_schwarz_coupling_does_not_take_is_refused_by_name(case, code):
    assert code in refusals(experiment(**case))


def test_a_planted_indefinite_response_stops_the_run_by_name(monkeypatch):
    """Gate 9's control: the response the flow prepares planted indefinite (its sign turned) is caught at the step that
    prepares it, the run stopped there and refused as ``flow.schwarz-indefinite``; the unplanted step is not."""
    import mime.nodes.flow.fvm._node as flow_node

    original = flow_node.prepared

    def turned(*args, **kwargs):
        made = original(*args, **kwargs)
        return made._replace(resistance=-made.resistance)

    with jax.enable_x64(True):
        member = sphere_member(RADIUS, 1e-4)
        pose = np.array([0.5 * CELLS * SPACING, 0.5 * CELLS * SPACING, 0.5 * LENGTH * SPACING, 1.0, 0.0, 0.0, 0.0])
        node = schwarz_flow(member, pose, CELLS, LENGTH, SPACING, TUBE, DT, 2 * RADIUS, schwarz_box=BOX)
        state = node.initial_state()
        assert float(state["schwarz_0_margin"]) > 0.0
        p = node._step_constants(node.params_pytree())  # noqa: SLF001 - the step's own constants
        inputs = {"schwarz_0_twist": jnp.zeros(6), "schwarz_0_pose": jnp.asarray(pose)}
        monkeypatch.setattr(flow_node, "prepared", turned)
        ended = {key: np.asarray(value) for key, value in node.advance(p, state, inputs, DT).items()}
    assert float(ended["schwarz_indefinite"]) == 1.0 and int(ended["stopped_at"]) == 1
    problem = node.stop_problem(ended)
    assert problem is not None and problem.code == "flow.schwarz-indefinite"


def node_steps(count: int, drop_composition: bool = False) -> list[dict[str, float]]:
    """Gate 5's return path: ``count`` steps of the flow node holding the sphere, each against the monolithic step from
    the node's state at that step's start (its field, the box's carried field and image term): the stored field's
    and its kinetic energy's relative errors a step. The plant drops the composition (the field left at its base)."""
    drop_composition = drop_composition or plant() == "drop"
    found: list[dict[str, float]] = []
    with jax.enable_x64(True):
        member = sphere_member(RADIUS, 1e-4)
        rng = draw()
        pose = np.array([*(np.array([0.5 * CELLS, 0.5 * CELLS, 0.5 * LENGTH]) + rng.uniform(-0.5, 0.5, 3)) * SPACING,
                         1.0, 0.0, 0.0, 0.0])
        node = schwarz_flow(member, pose, CELLS, LENGTH, SPACING, TUBE, DT, 2 * RADIUS, drive=(0.0, 0.0, 2.0),
                            schwarz_box=BOX, schwarz_tolerance=1e-11)
        p = node._step_constants(node.params_pytree())  # noqa: SLF001 - the step's own constants, as a run gives them
        state = dict(node.initial_state())
        at = pieces(node, DT)
        box, host = node.schwarz_box(), node.schwarz
        drive = node.drive(node.fields(p, at.grid), jnp.asarray([0.0, 0.0, 2.0]))
        _, decay = image_terms(box.side, box.spacing, at.shift, at.nu, at.mu)
        mass = np.diag([3e-8] * 3 + [5e-16] * 3) / DT
        twist0 = np.zeros(6)
        for _ in range(count):
            loads = rng.uniform(-1.0, 1.0, 6) * np.array([2e-9] * 3 + [1e-16] * 3)
            right = mass @ twist0 + loads
            resistance, load = np.asarray(state["schwarz_0_resistance"]), np.asarray(state["schwarz_0_load"])
            twist = np.linalg.solve(mass + resistance, right + load)
            prepared_at = np.asarray(state["schwarz_0_at"])
            moved = prepared_at.copy()
            moved[:3] += DT * twist[:3]
            ended = dict(node.advance(p, state, {"schwarz_0_twist": jnp.asarray(twist),
                                                 "schwarz_0_pose": jnp.asarray(moved),
                                                 "waveform": jnp.asarray([0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0])}, DT))
            body = host.body(p, jnp.asarray(prepared_at))
            held = stencils(at.grid, box, body.placed.force, 1.0)
            before = (state["velocity_x"], state["velocity_y"], state["velocity_z"])
            rhs = tuple(at.shift * before[k] + drive[k] for k in range(3))
            carried = decayed(box, (state["schwarz_own_x"], state["schwarz_own_y"], state["schwarz_own_z"]), at)
            reference = monolithic_step(at, box, held, body, rhs, carried, decay * state["schwarz_image"], mass, right,
                                        None)
            field = (ended["velocity_x"], ended["velocity_y"], ended["velocity_z"]) if not drop_composition else \
                (state["schwarz_base_x"], state["schwarz_base_y"], state["schwarz_base_z"])
            stored = node.storage(p, dict(zip(("velocity_x", "velocity_y", "velocity_z"), field, strict=True)))
            wanted = node.storage(p, dict(zip(("velocity_x", "velocity_y", "velocity_z"), reference.velocity,
                                              strict=True)))
            flat = np.concatenate([np.ravel(c) for c in field])
            exact = np.concatenate([np.ravel(c) for c in reference.velocity])
            found.append(dict(field=float(np.linalg.norm(flat - exact) / np.linalg.norm(exact)),
                              energy=float(abs(stored - wanted) / abs(wanted)),
                              twist=float(np.linalg.norm(twist - reference.twist) / np.linalg.norm(reference.twist))))
            state, twist0 = ended, twist
    return found


def test_the_stored_field_and_its_energy_are_the_monolithic_steps_each_step():
    """Gate 5's return path: three steps, each composed field and its kinetic energy against the monolithic step from
    the node's own state at the step's start (the box's carried field and image term from the second step on)."""
    found = node_steps(3)
    print(f"F5 gate 5 [float64]: per step {found}")
    assert all(step["field"] < LINE and step["energy"] < LINE and step["twist"] < LINE for step in found), found


def test_the_return_path_dropped_misses_the_monolithic_energy_every_step():
    found = node_steps(3, drop_composition=True)
    assert all(step["field"] > LINE and step["energy"] > LINE for step in found), found


#: Gate 5's lines by dtype: the exchange's two rows (the body's and the flow's exchange) and the flow's row against the
#: composed field's work at the force points, relative to the larger of the two (measured at about 3e-16 in float64).
#: float32's is the body's pairing's own: its load paired with its position's change, about 500 eps of cancellation
#: on a 0.05 mm offset moving a tenth of a micrometre a step.
WORKLESS = {"float64": 1e-9, "float32": 1e-3}


def _metered_books(gift: float = 0.0) -> dict[str, Any]:
    """The metered sphere, driven, three steps: the ledger's verdict, the drag's two rows and gift counts, and the
    flow's books of the law summed. ``gift`` plants a hand-over gift: the load handed to the body scaled by
    ``1 + gift`` (``MIME_PLANT_SCHWARZ=gift``)."""
    import mime.nodes.flow.fvm._node as flow_node
    from mime.observations import EnergyLedger
    from mime.testing import ledger_closes
    from mime.units import ms

    gift = gift or (0.5 if plant() == "gift" else 0.0)
    original = flow_node.prepared

    def given(*args: Any, **kwargs: Any) -> Any:
        made = original(*args, **kwargs)
        return made._replace(load=(1.0 + gift) * made.load)

    flow_node.prepared = given if gift else original
    try:
        exp = experiment()
        exp.observe(EnergyLedger())
        result = exp.build().run(ms(3.0))
    finally:
        flow_node.prepared = original
    assert result.energy is not None
    rows = {(law.law, law.receiver): law for law in result.energy.laws}
    flow = result.final["water.flow"]
    return dict(verdict=ledger_closes(result), body=rows[("drag:ball", "ball")],
                fluid=rows[("drag:ball", "water.flow")], books=np.asarray(flow["schwarz_0_books_sum"]),
                gifts=float(flow["schwarz_0_gift_steps"]))


def test_the_books_close_with_the_drag_held_in_the_flows_step_and_the_exchange_workless(dtype):
    """Gate 5 (the ledger): metered, the driven sphere, every receiver's books close at every step (``ledger_closes``);
    the drag's row on the body is the flow's exchange with its sign turned (the hand-over workless), its row on the
    flow the composed field's work at the force points (the scatter and gather a transpose pair), and their sum the
    near field's own dissipation less the free box's, the quadrature and the composed residual's beside it, which the
    flow judges each step (none given)."""
    books = _metered_books()
    assert books["verdict"].status == "PASS", books["verdict"].detail
    exchange, near, boxed, quadrature, residual = books["books"]
    body, fluid = books["body"].work, books["fluid"].work
    grid_read = exchange - near + boxed - quadrature - residual
    print(f"F5 gate 5 [{dtype}]: body {body:.6g}, exchange {exchange:.6g}, fluid {fluid:.6g}, grid read {grid_read:.6g}, "
          f"near {near:.6g}, box {boxed:.6g}, quadrature {quadrature:.3g}, residual {residual:.3g}")
    line = WORKLESS[dtype]
    assert abs(body + exchange) <= line * max(abs(body), abs(exchange)), (body, exchange)
    assert abs(fluid - grid_read) <= line * max(abs(fluid), abs(grid_read)), (fluid, grid_read)
    assert fluid + body < 0.0 and near > boxed, books
    assert books["gifts"] == 0.0 and books["body"].gift_steps == 0 and books["fluid"].gift_steps == 0, books


def test_a_planted_hand_over_gift_is_not_workless(dtype):
    """Gate 5's control: the load handed to the body planted half again its size, which the body's row books and the
    flow's exchange does not."""
    books = _metered_books(gift=0.5)
    body, exchange = books["body"].work, books["books"][0]
    assert abs(body + exchange) > WORKLESS[dtype] * max(abs(body), abs(exchange)), (body, exchange)


def _started_held(on: bool) -> Any:
    """Gate 11-early's plant (``MIME_PLANT_SCHWARZ=started``): the flow's first preparation held as the initial state's,
    ``started`` making nothing in the derivative's program (a silent zero derivative through the first step)."""
    from mime.nodes.flow.fvm._node import ImplicitFlowNode

    original = ImplicitFlowNode.started
    if on:
        ImplicitFlowNode.started = lambda self, params, state: {}  # type: ignore[method-assign]
    return original


def test_a_derivative_through_the_first_step_matches_a_central_difference(float64):
    """The derivative of the body's axial position after two steps (the first step's preparation, made in the
    flow's initial state, included) in the water's viscosity, forward mode by ``Study.gradient``, against a central
    difference of two runs built at the moved viscosity. The plant holds the first preparation as the initial state's
    in the derivative's program, which must fail it."""
    from mime.nodes.flow.fvm._node import ImplicitFlowNode
    from mime.study import Study, along
    from mime.units import Pa_s, ms

    def position(final):  # noqa: ANN001, ANN202
        return final["ball"]["pose"][2]

    original = _started_held(plant() == "started")
    try:
        result = Study(experiment()).sweep(along({"water.viscosity": [Pa_s(1e-3)]}), duration=ms(2.0)).gradient(
            position, wrt=["water.viscosity"], mode="forward",
            acknowledged={"water.flow": "the derivative is in the viscosity; the flow's geometry leaves are not moved"})
    finally:
        ImplicitFlowNode.started = original  # type: ignore[method-assign]
    derivative = float(np.asarray(result.member(0).gradient["water.viscosity"]))
    step = 1e-3 * 1e-3
    ends = [float(experiment(viscosity=mu).build().run(ms(2.0)).final["ball"]["pose"][2])
            for mu in (1e-3 + step, 1e-3 - step)]
    difference = (ends[0] - ends[1]) / (2.0 * step)
    print(f"F5 gate 11-early: derivative {derivative:.10g}, central difference {difference:.10g}, relative "
          f"{abs(derivative - difference) / abs(difference):.3g}")
    assert abs(derivative - difference) <= DERIVATIVE_LINE * abs(difference), (derivative, difference)


#: The derivative's line against the central difference: the solves' tolerances (the columns' 1e-8, float64) move the
#: difference's two runs by about that share of the position's change, and the difference's own truncation is about
#: the step squared (1e-6).
DERIVATIVE_LINE = 1e-4
