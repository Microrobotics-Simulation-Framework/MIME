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
