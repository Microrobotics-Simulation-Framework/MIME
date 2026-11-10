"""FA2 and FA4: the band on walls no tube tree gives exactly, each through the node's wall seam (``wall_cases``'s
solids, the node's own step otherwise), steady Stokes at 16 cells across the narrowest diameter, turned and moved by a
sub-cell offset at random:

- **FA2**, a helical tube (``kappa a`` 0.15, torsion a fifth of it): Dean's first-order profile fitted with a free
  radius, amplitude and centre in the normal plane; the centre's offset along the principal normal (the lumen's
  displacement, which a flux cannot see) within ``DISPLACEMENT_LINE``, and the free radius's shift within
  ``SHIFT_LINE`` of the planar calibration. Control: the solver's centreline a tenth of a cell outward of the oracle's
  (``MIME_PLANT_WALL=centreline``). The oracle's own check: without Dean's term the fit puts the centre ``3 kappa a^2
  / 8`` towards the axis, about seven tenths of a cell;
- **FA4**, an elliptic tube (semi-axes 0.5 and 0.3 D): the quadratic fitted to the interior's profile, its zero
  level's semi-axes, each one's shift within ``SECTION_LINE`` of a round tube's by the same estimator at the same
  offset. Control: the solver's major semi-axis a quarter cell longer (``MIME_PLANT_WALL=wall``).

The predictions are ``predictions_general_wall.json``'s, committed before these ran."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fvm_cases import in_dtype, run
from test_fvm_tube import planar_calibration
from wall_cases import EllipticTube, Helix, dean_fit, draw, ellipse_fit, plant, seeds, solid_node

from mime.nodes.flow.fvm._box import _odd
from mime.scene._sampling import counted

#: FA2's line on the lumen's displacement along the principal normal, in cells.
DISPLACEMENT_LINE = 0.05
#: The wall's shift against the planar calibration, in cells (S1's line).
SHIFT_LINE = 0.15
#: FA4's line on each semi-axis's shift against the round tube's, in cells.
SECTION_LINE = 0.1
#: The helix: the tube's radius, its curvature times it, and its pitch over two pi (D = 1).
TUBE, CURVED, PITCH = 0.5, 0.15, 0.35
CENTRELINE_PLANT = 0.1


def helix_case(seed: int, dh: int = 16, planted: bool = False):
    """The helical tube in its box at a drawn phase and sub-cell offset: its node, the oracle's helix and the box."""
    kappa = CURVED / TUBE
    radius = (1.0 + math.sqrt(1.0 - 4.0 * kappa * kappa * PITCH * PITCH)) / (2.0 * kappa)
    phase, offset = draw(seed)
    h_target = 1.0 / dh
    along = max(1, counted(2.0 * math.pi * PITCH / h_target))
    h = 2.0 * math.pi * PITCH / along
    across = _odd(counted(2.0 * (radius + TUBE) / h + 3 + 8))
    cells = (across, across, along)
    centre = ((across / 2 + offset[0]) * h, (across / 2 + offset[1]) * h)
    oracle = Helix(centre[0], centre[1], radius, PITCH, TUBE, phase + offset[2] * h / PITCH)
    given = Helix(centre[0], centre[1], radius + (CENTRELINE_PLANT * h if planted else 0.0), PITCH, TUBE,
                  oracle.phase)
    return solid_node(given, cells, h, TUBE, tolerance=1e-10), oracle, cells, h


@pytest.mark.parametrize("seed", seeds(1201, 1202))
def test_fa2_the_curved_tubes_lumen_is_not_displaced_against_deans_profile(seed):
    with in_dtype("float64"):
        node, helix, cells, h = helix_case(seed, planted=plant("centreline") is not None)
        state = run(node, 1)
        fitted = dean_fit(helix, state, cells, h, 2.0 * h)
    displacement = fitted["normal"] / h
    shift = (fitted["radius"] - TUBE) / h
    planar = planar_calibration(16, "float64")
    print(f"\nseed {seed}: displacement {displacement:+.4f} cells along the normal, {fitted['binormal'] / h:+.4f} "
          f"along the binormal; shift {shift:+.4f} (planar {planar:+.4f}); amplitude {fitted['amplitude']:.4f} of "
          f"G cos(pitch)/4; {fitted['points']:.0f} cells; report {np.asarray(state['solve_report'])}")
    assert abs(displacement) <= DISPLACEMENT_LINE
    assert abs(shift - planar) < SHIFT_LINE


def ellipse_case(seed: int, a: float, b: float, dh: int = 16, grown: float = 0.0):
    angle, offset = draw(seed)
    h = 1.0 / dh
    across = _odd(counted(2.0 * max(a, b) / h + 3 + 8))
    cells = (across, across, 4)
    centre = ((across / 2 + offset[0]) * h, (across / 2 + offset[1]) * h)
    oracle = EllipticTube(centre[0], centre[1], a, b, angle)
    given = EllipticTube(centre[0], centre[1], a + grown * h, b, angle)
    return solid_node(given, cells, h, min(a, b), tolerance=1e-10), oracle, cells, h


@pytest.mark.parametrize("seed", seeds(1301, 1302))
def test_fa4_the_elliptic_sections_semi_axes_shift_as_a_round_tubes(seed):
    with in_dtype("float64"):
        node, tube, cells, h = ellipse_case(seed, 0.5, 0.3, grown=0.25 if plant("wall") else 0.0)
        a, b = ellipse_fit(tube, run(node, 1), cells, h, 2.0 * h)
        round_node, round_tube, round_cells, _ = ellipse_case(seed, 0.5, 0.5)
        r, r_other = ellipse_fit(round_tube, run(round_node, 1), round_cells, h, 2.0 * h)
    round_shift = 0.5 * ((r - 0.5) + (r_other - 0.5)) / h
    shifts = ((a - 0.5) / h, (b - 0.3) / h)
    print(f"\nseed {seed}: semi-axes' shifts {shifts[0]:+.4f}, {shifts[1]:+.4f} cells; the round tube's "
          f"{round_shift:+.4f} ({(r - 0.5) / h:+.4f}, {(r_other - 0.5) / h:+.4f})")
    assert all(abs(shift - round_shift) <= SECTION_LINE for shift in shifts)
