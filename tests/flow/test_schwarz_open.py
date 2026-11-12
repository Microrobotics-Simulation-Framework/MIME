"""An MRS body coupled by Schwarz in an open vessel (F5): the openings are eliminated inside each product and in the
composition, by the unit fields of the step's base, so the composed step meets every opening's law.

- **The case** (float64, the sphere of ``tests/schwarz_experiments.py`` in the vessel open at its ends, a steady inflow
  at its start and a Windkessel at its end, three steps): the inflow's flux holds at the composed field, the
  body's forces moving no fluid through it, the near field's equations hold at the composed field (the products and
  the composition solve the same walled problem, openings included), nothing is flagged, and the body moves.
- **Control** (``MIME_PLANT_SCHWARZ=cross``): the body-opening cross terms zeroed (no unit field added for the body's
  forces), which moves fluid through the inflow's opening."""

from __future__ import annotations

from typing import Any

import numpy as np
from schwarz_cases import plant
from schwarz_experiments import schwarz_sphere as experiment

from mime.nodes.flow.fvm._schwarz import RESIDUAL_LINE, SCHWARZ_TOLERANCE
from mime.units import ms

#: The inflow's outward flux at the start opening (m^3/s): 1e-3 mL/s in.
INFLOW = -1e-9
#: The inflow's flux at the composed field against its law: the face flux is linear in the field, so a rounding line.
FLUX_LINE = 1e-9


def test_the_composed_step_meets_each_openings_law_and_the_near_fields_equations(float64):
    from mime.nodes.flow.fvm import OpenFlowNode

    original = OpenFlowNode.homogeneous
    if plant() == "cross":
        def crossless(self: Any, grid: Any, rows: Any, blocks: Any, units: Any, response: Any, velocity: Any) -> Any:
            return velocity, original(self, grid, rows, blocks, units, response, velocity)[1]

        OpenFlowNode.homogeneous = crossless  # type: ignore[method-assign]
    try:
        result = experiment(opened=True).build().run(ms(3.0))
    finally:
        OpenFlowNode.homogeneous = original  # type: ignore[method-assign]
    flow = result.final["water.flow"]
    flux = np.asarray(flow["opening_flux"])
    residual = float(flow["schwarz_residual"])
    moved = float(np.linalg.norm(np.asarray(result.final["ball"]["pose"][:3]) - np.asarray((0.05e-3, 0.0, 0.0))))
    print(f"F5 open: inflow's flux {flux[0]:.12g} m^3/s against {INFLOW:g} (off {abs(flux[0] - INFLOW) / abs(INFLOW):.3g}), "
          f"the composed residual {residual:.3g}, the body moved {moved:.3g} m, flagged {float(flow['schwarz_flagged'])}")
    assert abs(flux[0] - INFLOW) <= FLUX_LINE * abs(INFLOW), flux
    assert abs(flux[0] + flux[1]) <= FLUX_LINE * abs(INFLOW), flux
    assert residual <= RESIDUAL_LINE * SCHWARZ_TOLERANCE["float64"], residual
    assert float(flow["schwarz_flagged"]) == 0.0 and moved > 0.0


#: Gate 3 with openings against the dense step: every unknown a linear function of solves stopped at their tolerances
#: (the columns' GMRES at 1e-11, the walled CG at 1e-12, its products' at the inner share), as the periodic gate 3's.
MONOLITHIC_LINE = 1e-7
#: The open node remade at these tolerances for gate 3 (the experiment's own are the dtype's defaults).
TIGHT_SOLVES = {"tolerance": 1e-12, "schwarz_tolerance": 1e-11}


def open_node_step() -> dict[str, float]:
    """One step of the open vessel's flow node (1 mm long, float64) from its initial preparation, the body's twist from
    the handed response and a drawn load, against the dense step with the openings' rows: the field, the openings'
    pressures, the twist and the wrench the body took. Plant: ``cross`` (the body-opening cross terms zeroed)."""
    import jax
    import jax.numpy as jnp
    from schwarz_cases import draw, open_monolithic_step

    from mime.nodes.flow.fvm import OpenFlowNode

    with jax.enable_x64(True):
        made: Any = experiment(opened=True, length=1.0).build().graph.get_node("water.flow")
        node: Any = type(made)(made.name, float(made.delta_t), **{**dict(made.params), **TIGHT_SOLVES})
        original = OpenFlowNode.homogeneous
        if plant() == "cross":
            def crossless(self: Any, grid: Any, rows: Any, blocks: Any, units: Any, response: Any, velocity: Any) -> Any:
                return velocity, original(self, grid, rows, blocks, units, response, velocity)[1]

            OpenFlowNode.homogeneous = crossless  # type: ignore[method-assign]
        try:
            state = node.initial_state()
            p = node._step_constants(node.params_pytree())  # noqa: SLF001 - the step's own constants, as a run gives them
            dt = float(node.delta_t)
            rng = draw()
            mass = np.diag([3e-8] * 3 + [5e-16] * 3) / dt
            right = rng.uniform(-1.0, 1.0, 6) * np.array([2e-9] * 3 + [1e-16] * 3)
            resistance, load = np.asarray(state["schwarz_0_resistance"]), np.asarray(state["schwarz_0_load"])
            twist = np.linalg.solve(mass + resistance, right + load)
            pose = np.asarray(state["schwarz_0_at"])
            moved = pose.copy()
            moved[:3] += dt * twist[:3]
            ended = node.advance(p, dict(state), {"schwarz_0_twist": jnp.asarray(twist),
                                                  "schwarz_0_pose": jnp.asarray(moved)}, dt)
        finally:
            OpenFlowNode.homogeneous = original  # type: ignore[method-assign]
        body = node.schwarz.body(p, jnp.asarray(pose))
        reference, pressures = open_monolithic_step(node, p, state, body, mass, right, dt)
    found = np.concatenate([np.ravel(np.asarray(ended[f"velocity_{x}"])) for x in "xyz"])
    wanted = np.concatenate([np.ravel(c) for c in reference.velocity])
    wrench = np.asarray(body.placed.wrench()).T @ reference.tractions
    took = resistance @ twist - load
    pressure = np.asarray(ended["opening_pressure"])
    return dict(field=float(np.linalg.norm(found - wanted) / np.linalg.norm(wanted)),
                pressures=float(np.max(np.abs(pressure - pressures)) / np.max(np.abs(pressures))),
                twist=float(np.linalg.norm(twist - reference.twist) / np.linalg.norm(reference.twist)),
                wrench=float(np.linalg.norm(took - wrench) / np.linalg.norm(wrench)),
                residual=float(ended["schwarz_residual"]), flagged=float(ended["schwarz_flagged"]))


def test_the_open_flow_node_steps_as_the_dense_step_with_the_openings_rows(float64):
    """Gate 3 with openings: the open node's step (the openings eliminated in each product and in the composition) is
    the dense step whose unknowns are the field, the openings' pressures, the tractions and the twist, within the
    solves' line, unflagged."""
    found = open_node_step()
    print(f"F5 gate 3, openings: {found}")
    assert max(found["field"], found["pressures"], found["twist"], found["wrench"]) < MONOLITHIC_LINE, found
    assert found["residual"] < RESIDUAL_LINE * TIGHT_SOLVES["schwarz_tolerance"] and found["flagged"] == 0.0, found
