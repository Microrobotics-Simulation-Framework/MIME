"""Many runs of one experiment: a sweep over its points, each the experiment with its values declared, grouped by the
compiled program each needs and batched within a group, its commands held from rest or ramped (its protocol), and the
step-out read from it by the quantity it is; a contact's friction identified from a run's data with its verdict
(``fit_friction``). Variant loops and sensitivities come later."""

from mime.study._axes import DT, PRECISION, Axis, AxisKind, RunPath
from mime.study._friction import FRICTION_RANGE, FrictionFit, fit_friction
from mime.study._protocol import Crossing, FromRest, Ramp, StepOut, ramp_crossings, step_out
from mime.study._study import Group, Points, Study, StudyPlan, along, grid

__all__ = ["DT", "FRICTION_RANGE", "PRECISION", "Axis", "AxisKind", "Crossing", "FrictionFit", "FromRest", "Group",
           "Points", "Ramp", "RunPath", "StepOut", "Study", "StudyPlan", "along", "fit_friction", "grid",
           "ramp_crossings", "step_out"]
