"""Many runs of one experiment: a sweep over its points, each the experiment with its values declared, grouped by the
compiled program each needs and batched within a group. Variant loops, fits and sensitivities come later."""

from mime.study._axes import DT, PRECISION, Axis, AxisKind, RunPath
from mime.study._study import Group, Points, Study, StudyPlan, along, grid

__all__ = ["DT", "PRECISION", "Axis", "AxisKind", "Group", "Points", "RunPath", "Study", "StudyPlan", "along", "grid"]
