"""The FVM's declaration at the build: the build report's line for the box (its cells, spacing and resolution, and
the step's memory estimate, the cells times XLA's measured bytes a cell, against the device's memory), and the memory
refusal (gate 13): a box whose step will not fit the device is refused when planned, with its route."""

from __future__ import annotations

import pytest
from test_fvm_experiment import experiment

import mime.dynamics.flow._fvm as fvm
from mime.build import BuildRefused
from mime.units import mm


def _box_line(built: object) -> str:
    lines = [line for held in built.report.checked.values() for line in held]  # type: ignore[attr-defined]
    return next(line for line in lines if line.startswith("the box:"))


def test_the_build_reports_the_box_and_its_memory(float64):
    built = experiment().build()
    node = built.graph.get_node("water.flow")
    c = node.constants()
    cells = c["cells_x"] * c["cells_y"] * c["cells_z"]
    line = _box_line(built)
    assert f"{c['cells_x']} x {c['cells_y']} x {c['cells_z']} = {cells} cells" in line
    assert f"{fvm.memory_estimate('steady', cells, metered=False) / 2 ** 20:.4g} MiB" in line
    assert "16 across the lumen" in line


@pytest.mark.parametrize("planted", [False, True])
def test_a_box_whose_step_will_not_fit_the_device_is_refused_with_its_route(float64, planted):
    """The default box against a planted device a byte under its estimate is refused (``flow.memory``), and a byte
    over it builds (the control); a box of some eight billion cells is refused against the host's own memory."""
    if not planted:
        with pytest.raises(BuildRefused) as refused:
            experiment(cells_across=2000, length=mm(1.0)).plan()
        assert "flow.memory" in [each.code for each in refused.value.refusals]
        return
    cells = experiment().build().graph.get_node("water.flow").constants()
    needed = fvm.memory_estimate("steady", cells["cells_x"] * cells["cells_y"] * cells["cells_z"], metered=False)
    for room, fits in ((int(needed) - 1, False), (int(needed) + 1, True)):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fvm, "device_memory", lambda room=room: ("a planted device", room))
            if fits:
                experiment().plan()
                continue
            with pytest.raises(BuildRefused) as refused:
                experiment().plan()
        found = [each for each in refused.value.refusals if each.code == "flow.memory"]
        assert len(found) == 1 and "a planted device" in found[0].message and "cells_across" in found[0].action
