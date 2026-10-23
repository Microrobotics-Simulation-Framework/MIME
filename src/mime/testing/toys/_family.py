"""The toy family, the control for the family suites: laws that resist a toy body's motion. Each reads the body's
velocity and delivers a load into it that never has a component along that velocity; its coefficient stands for a
medium's viscosity, a material's property that must reach the load. Its member is :class:`ToyDrag`, both rungs."""

from __future__ import annotations

from mime.experiment import Experiment
from mime.registry import Interaction
from mime.testing._families import CaseKind, Family, opposes, register_family
from mime.testing.toys._entries import BODY_VELOCITY, LOAD, ToyBody, ToyDrag
from mime.testing.toys._experiments import PROBE_START, toy_experiment
from mime.things import Body


def resistance_rig(law: Interaction) -> Experiment:
    """A toy body with ``law`` attached to it: the smallest experiment that holds a toy law."""
    experiment = toy_experiment("toy_family_rig")
    experiment.add_body(Body("probe", dynamics=ToyBody(), initial=PROBE_START))
    experiment.attach(law, body="probe")
    return experiment


TOY_FAMILY = register_family(Family(
    name="toy",
    rig=resistance_rig,
    cases=(CaseKind.SIGN, CaseKind.FRAME, CaseKind.UNITS, CaseKind.MATERIAL_REACH, CaseKind.ORACLE,
           CaseKind.GRADIENTS, CaseKind.SWAP, CaseKind.NODES),
    sign=opposes(("body", LOAD.name), ("body", BODY_VELOCITY.name)),
    material=("c",),
    reference=ToyDrag.Linear,
))
