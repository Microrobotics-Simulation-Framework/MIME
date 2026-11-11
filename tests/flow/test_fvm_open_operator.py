"""The open box's operator at its planes (the review of F2b's PR 2, B2; prediction in ``predictions_open_ports.json``,
committed before it ran): the open grid's Laplacian and convection against closed forms on drawn fields, exact to
rounding on every face, the planes' half cells included (``open_cases.operator_errors``). No other gate sees these
stencils: FB1's flow is developed, so its ``w`` has no gradient along z at the planes; FB4 compares two solves of one
operator; the books book whatever the operator is. Its controls: each open plane's viscous half cell at a full cell
(the review's mutant, which moved a drawn flow by up to 37 % and passed every other gate) and each plane's convective
half cell at a full cell. Their rates over fresh draws are measured on Actions (the PR's evidence table)."""

from __future__ import annotations

import pytest
from open_cases import PLANTS, operator_draw, operator_errors

#: The draws CI runs (the probes run 200 fresh ones a control).
SEEDS = range(12)


def test_the_open_operator_is_its_closed_form_on_every_face(dtype):
    for seed in SEEDS:
        found = operator_errors(operator_draw(seed), dtype)
        assert max(found.values()) <= 1.0, (seed, found)


@pytest.mark.parametrize(("plant", "part"), [("laplacian_full_cell", "viscous_planes"),
                                             ("end_full_cell", "convective_planes"),
                                             ("start_full_cell", "convective_planes")])
def test_control_a_planes_half_cell_at_a_full_cell_fails_every_draw(dtype, monkeypatch, plant, part):
    PLANTS[plant](monkeypatch.setattr)
    for seed in SEEDS:
        found = operator_errors(operator_draw(seed), dtype)
        assert found[part] > 1.0, (seed, found)
