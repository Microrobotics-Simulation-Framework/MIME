"""Many runs of one experiment: a sweep over its points, each the experiment with its values declared, grouped by the
compiled program each needs and batched within a group, its commands held from rest or ramped (its protocol), and the
step-out read from it by the quantity it is. Variant loops, fits and sensitivities come later."""

from mime.study._axes import DT, PRECISION, Axis, AxisKind, RunPath
from mime.study._protocol import Crossing, FromRest, Ramp, StepOut, ramp_crossings, step_out
from mime.study._study import Group, Points, Study, StudyPlan, along, grid

__all__ = ["DT", "PRECISION", "Axis", "AxisKind", "Crossing", "FromRest", "Group", "Points", "Ramp", "RunPath",
           "StepOut", "Study", "StudyPlan", "along", "grid", "ramp_crossings", "step_out"]
