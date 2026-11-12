"""The shape suite's points-side property on the tube tree (``test_shape_suite.py`` holds every other shape): autodiff
equals the float64 difference on the point's side, for the tube tree's signed distance, normal and nearest point.

Split out so that it takes a shard of its own: each of the three compiles the tube tree's union for its probes' dtypes
(100 to 220 s on CI). The test is the one in ``test_shape_suite.py`` over the other shapes' pairs, with the same body
(``shape_suite_support.points_side_holds``) and the same settings."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shape_suite_support import SEPARATE, points_side_holds

from mime.testing import SHAPE_CASES

PAIRS = [(shape, name) for shape in SEPARATE for name in SHAPE_CASES[shape].capabilities]


@pytest.mark.parametrize(("shape", "name"), PAIRS)
@settings(deadline=None)  # a draw's first evaluation compiles the shape's checks for its dtype and starts
@given(data=st.data())
def test_autodiff_equals_the_difference_on_the_points_side(shape, name, data):
    points_side_holds(shape, name, data)
