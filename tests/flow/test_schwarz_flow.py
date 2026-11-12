"""F5's Schwarz coupling through the flow node and the build.

- **Gate 3 through the node**: the flow node holding an MRS sphere (its initial preparation, then one step composed
  from the body's twist) against the monolithic step (``schwarz_cases.monolithic_step``), float64, IMEX Euler, with a
  drive: the field, the tractions' wrench (the load the body took) and the composed residual. Control: the composition
  dropped (the field left at its base, the return path's plant), which misses the monolithic field past the line.
- **What is frozen** (gate 6): the next step's columns are prepared at the pose the body reached, and the step composed
  at the pose the columns were prepared at, never the live one. Control: the composition read at the body's new pose
  misses the monolithic field.
- **The build**: ``Drag.MRS`` in a medium the FVM fills is a Schwarz partner (no drag node; the body added before its
  medium's flow; the response a prepared delivery, read the step before and planned), and the refusals with their
  routes: a near field with a wall kind of its own, a second body in the flow, an open vessel, and the opt-ins not
  built on the mrs rung."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from schwarz_cases import draw, monolithic_step, pieces, plant, schwarz_flow, sphere_member
from schwarz_experiments import schwarz_sphere as experiment

from mime.build import BuildRefused
from mime.experiment import Experiment
from mime.nodes.flow.fvm._schwarz import RESIDUAL_LINE, stencils

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
    ({"opened": True}, "flow.schwarz-open"),
    ({"settings": {"one_way": True}}, "drag.schwarz-scheme"),
    ({"settings": {"coupling": "lagged"}}, "drag.schwarz-scheme"),
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


def test_the_energy_ledger_on_a_schwarz_flow_is_refused_by_name_until_its_rows_are_booked():
    from mime.observations import EnergyLedger

    exp = experiment()
    exp.observe(EnergyLedger())
    assert "flow.schwarz-ledger" in refusals(exp)
