"""Every drag rung assumes its medium at rest (it delivers no load from the medium's flow), so until a rung reads a
provided flow each is refused, coded ``drag.medium-flows`` and naming the field, in a medium that carries a flow
providing the velocity (``field.flow.velocity``, SF-2's quantity rule): the surrogate, mrs and rft rungs, on the
current drag and on the version kept registered. The flow is the prescribed one (Womersley), and the FVM for the mrs
rung. Each rung's control is the same experiment with the medium at rest, which plans without a refusal."""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import cast

import pytest
from mrs_cases import mrs_tube
from rft_cases import rft_tube
from script_a import script_a

from mime.build import BuildRefused
from mime.build._declarations import ValueDeclaration
from mime.dynamics import FVM, Womersley
from mime.experiment import Experiment
from mime.interactions.hydrodynamic import PointForceDrag
from mime.scene import PeriodicBox
from mime.things import Medium
from mime.units import kg_per_m3, m_per_s, mm

CODE = "drag.medium-flows"


def surrogate() -> Experiment:
    return script_a(velocity=m_per_s((1e-3, 0.0, 0.0)))


def kept_mrs() -> Experiment:
    experiment = mrs_tube()
    experiment.remove("drag:ball")
    experiment.attach(PointForceDrag.MRS(spacing=mm(0.3), blob=0.5, wall_kind="tube"), body="ball", medium="water")
    return experiment


def flowing(case: object, flow: object) -> Experiment:
    """``case``'s experiment with its medium ``water`` carrying ``flow`` (the case modules' ``Medium`` given the field
    for the call), and the water's density, which a flow reads."""
    modules = [sys.modules[name] for name in ("script_a", "mrs_cases", "rft_cases")]
    with pytest.MonkeyPatch.context() as patch:
        for module in modules:
            patch.setattr(module, "Medium", functools.partial(Medium, fields={"flow": flow}))
        experiment = cast(Callable[[], Experiment], case)()
    _, declarations, _ = experiment.declared_as()
    if not any(isinstance(d, ValueDeclaration) and str(d.path) == "water.density" for d in declarations):
        experiment.declare("water.density", kg_per_m3(1000.0), reason="water")
    return experiment


def refused_codes(experiment: Experiment) -> list[str]:
    try:
        experiment.plan()
    except BuildRefused as refused:
        return [each.code for each in refused.refusals]
    return []


@pytest.mark.parametrize(("case", "flow"), [
    (surrogate, Womersley()),
    (mrs_tube, Womersley()),
    (kept_mrs, Womersley()),
    (rft_tube, Womersley()),
    (mrs_tube, FVM(domain=PeriodicBox(length=mm(1.0)))),
], ids=["surrogate", "mrs", "mrs-0.3.0", "rft", "mrs-fvm"])
def test_a_drag_rung_is_refused_in_a_medium_whose_flow_provides_the_velocity(case, flow):
    assert refused_codes(case()) == []  # the control: the medium at rest
    experiment = flowing(case, flow)
    with pytest.raises(BuildRefused) as refused:
        experiment.plan()
    codes = [each.code for each in refused.value.refusals]
    assert codes == [CODE], codes
    assert "water.flow" in refused.value.refusals[0].message and "at rest" in refused.value.refusals[0].message
