"""The shape suite on the tube tree (``test_shape_suite.py`` holds every other shape): the points-side property for its
signed distance, normal and nearest point; and its fixed examples, a probe near each declared set, where autodiff equals
the difference at each and the planted wrong tangent fails at each.

Split out of ``test_shape_suite.py`` so that it takes a shard of its own: the tube tree's union compiles for each of
these tests (about 60 s a dtype on a runner), seven tests that were 72 % of that file's measured seconds. One file and
not two, so that one shard holds the four ``memory_heavy`` tests (which take turns on the run's lock) beside the probes
that share the union's compiles with them. The examples are made once per shape and dtype in a process
(``shape_suite_support.examples_of``): the planted-tangent control reads what the property made. The property is the one
in ``test_shape_suite.py`` over the other shapes' pairs, with the same body (``shape_suite_support.points_side_holds``)
and the same settings."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from shape_suite_support import DTYPES, SEPARATE, examples_pass, planted_tangent_fails, points_side_holds

from mime.testing import SHAPE_CASES

PAIRS = [(shape, name) for shape in SEPARATE for name in SHAPE_CASES[shape].capabilities]


@pytest.mark.parametrize(("shape", "name"), PAIRS)
@settings(deadline=None)  # a draw's first evaluation compiles the shape's checks for its dtype and starts
@given(data=st.data())
def test_autodiff_equals_the_difference_on_the_points_side(shape, name, data):
    points_side_holds(shape, name, data)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.memory_heavy
def test_autodiff_equals_the_difference_at_an_example_near_each_of_the_tube_trees_sets(dtype):
    """The property at fixed points, at the tube tree's examples (:data:`SEPARATE`)."""
    examples_pass(dtype, SEPARATE)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.memory_heavy
def test_control_a_planted_wrong_tangent_fails_at_every_example_near_the_tube_trees_sets(dtype):
    """The same control at the tube tree's examples (:data:`SEPARATE`)."""
    planted_tangent_fails(dtype, SEPARATE)
