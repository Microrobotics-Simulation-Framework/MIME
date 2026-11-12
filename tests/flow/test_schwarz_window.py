"""The near-contact window below its line is flagged, never refused (F5; decided on the window spike): a Schwarz body
nearer the wall than two cells of the grid that resolves the gap, or within a tenth of its local radius where that
radius is under twenty of them, raises ``schwarz_0_near_contact_window`` at the step, its gap in cells reported
beside it (``schwarz_0_wall_gap_cells``), and the run goes on.

- **The line**, on draws of the local radius and the gap in cells (each run draws its own): the flag on every draw below
  the line as decided, on none above it. Control: the check removed (``MIME_PLANT_SCHWARZ=window``), which misses
  every draw below the line.
- **In a run**: the sphere 1.2 cells from the wall raises the flag and runs on; 2.4 cells from it, with its radius 1.2
  cells, raises none."""

from __future__ import annotations

import numpy as np
from schwarz_cases import draw, plant
from schwarz_experiments import schwarz_sphere as experiment

from mime.nodes.flow.fvm import _schwarz_host as host
from mime.units import ms

DRAWS = 200


def _decided(gap: float, radius: float) -> bool:
    """The window's line as decided: the local radius at least 20 cells, and the gap at least 2 cells and at most a
    tenth of the radius; a body nearer than either bound is near contact, and near contact below the line is flagged."""
    near = gap < 2.0 or gap <= 0.1 * radius
    resolved = radius >= 20.0 and 2.0 <= gap
    return near and not resolved


def test_the_flag_is_raised_on_every_draw_below_the_line_and_on_none_above():
    rng = draw()
    radius = np.exp(rng.uniform(np.log(0.5), np.log(80.0), DRAWS))
    gap = np.exp(rng.uniform(np.log(0.05), np.log(30.0), DRAWS))
    flagged = (lambda g, r: np.zeros_like(g, dtype=bool)) if plant() == "window" else host.window
    found = np.asarray(flagged(gap, radius), dtype=bool)
    expected = np.asarray([_decided(float(g), float(r)) for g, r in zip(gap, radius, strict=True)])
    below = int(expected.sum())
    print(f"F5 window: {below} of {DRAWS} draws below the line; flagged {int(found[expected].sum())} of them and "
          f"{int(found[~expected].sum())} above it")
    assert below > 0 and below < DRAWS
    assert found[expected].all() and not found[~expected].any()


def test_a_body_near_the_wall_raises_the_flag_and_the_run_goes_on():
    near = experiment(offset=0.2).build().run(ms(1.0))
    far = experiment().build().run(ms(1.0))
    raised = {run: {read.flag.field: read.was_raised for read in result.flags}
              for run, result in (("near", near), ("far", far))}
    gaps = [float(result.final["water.flow"]["schwarz_0_wall_gap_cells"]) for result in (near, far)]
    print(f"F5 window in a run: gaps {gaps[0]:.3g} and {gaps[1]:.3g} cells; flags {raised}")
    assert raised["near"]["schwarz_0_near_contact_window"] is True
    assert raised["far"].get("schwarz_0_near_contact_window", False) is False  # a flag no step raised is not listed
    assert near.stopped is None and gaps[0] < 2.0 <= gaps[1]
