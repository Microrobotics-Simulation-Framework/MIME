"""The shape suite's fixed examples on the tube tree (``test_shape_suite.py`` holds every other shape's): a probe near
each declared set, autodiff equals the difference at each, and the planted wrong tangent fails at each.

Split out so that it takes a shard of its own: the tube tree's union compiles for each of its examples (about 60 s a
dtype on a runner). The four tests stay in one file because they read the same examples, made once per dtype in a
process (``shape_suite_support.examples_of``): the planted-tangent control reads what the property made."""

import pytest
from shape_suite_support import DTYPES, SEPARATE, examples_pass, planted_tangent_fails


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.memory_heavy
def test_autodiff_equals_the_difference_at_an_example_near_each_of_the_tube_trees_sets(dtype):
    """The same at the tube tree's examples (:data:`SEPARATE`)."""
    examples_pass(dtype, SEPARATE)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.memory_heavy
def test_control_a_planted_wrong_tangent_fails_at_every_example_near_the_tube_trees_sets(dtype):
    """The same control at the tube tree's examples (:data:`SEPARATE`)."""
    planted_tangent_fails(dtype, SEPARATE)
