"""The union's ``unconverged`` count (``mime.scene.ExactUnion``): the planned points that failed the check and whose own
seam was never reached, which keep the plain minimum rather than another candidate's depth. The gates never make a
point fail to converge, so the count itself was unchecked: a union that never counted one passed every gate.

The wedge of two half-spaces (gate 2's draw) has one seam, so every failing point's own seam is it. The seam is made
unreachable for the points on one side of a plane (the test's mask), through the real union: its other candidates stay
as they are, so the depth the seam would have given is there and is not used."""

import numpy as np
import pytest
from test_exact_union import CLOSED_LINE, drawn_wedge, wedge_union
from union_oracles import seed_list

import mime.scene._union as union_module
from mime.core import jnp
from mime.scene import ExactUnion
from mime.scene._union import Seam


@pytest.fixture(autouse=True)
def fresh_programs():
    """A union's program is compiled once for its plan and traced with the patches in force then: no program from
    another test, and none left behind."""
    union_module._compiled.cache_clear()
    yield
    union_module._compiled.cache_clear()


def seam_unreached_where(monkeypatch, unreached) -> None:
    """The seam not reached at the points ``unreached`` (a function of the union's local points) picks; its distance
    stays whatever the union found."""
    held = ExactUnion._seam

    def seam(self, parameters, points, i, j, radius):
        found = held(self, parameters, points, i, j, radius)
        return Seam(found.distance, found.converged, found.reached_seam & ~unreached(points))

    monkeypatch.setattr(ExactUnion, "_seam", seam)


@pytest.mark.parametrize("seed", seed_list((21, 22, 23)))
def test_an_unconverged_point_is_counted_and_keeps_the_minimum(seed, monkeypatch):
    normals, origin, fillet, points = drawn_wedge(seed)
    union = wedge_union(normals, origin, lambda p: p["fillet"])
    parameters = {"fillet": jnp.asarray(fillet)}
    at = jnp.asarray(points)
    clean = union.signed_distance(parameters, at)
    failing = np.asarray(clean.failing)
    local = np.asarray(union._local(parameters, at))
    middle = np.median(local[failing, 0])
    side = local[:, 0] > middle
    expected = int(np.sum(failing & side))
    assert 0 < expected < int(failing.sum()), "the mask must leave failing points on both sides to discriminate"
    assert int(clean.unconverged) == 0 and int(clean.unrefined) == 0

    seam_unreached_where(monkeypatch, lambda local_points: local_points[..., 0] > middle)
    union_module._compiled.cache_clear()
    found = union.signed_distance(parameters, at)
    count = int(found.unconverged)
    print(f"\nseed {seed}: {int(failing.sum())} failing, {expected} unconverged expected, {count} counted")

    minimum = np.asarray(union.minimum(parameters, at))
    distance = np.asarray(found.distance)
    refined = np.asarray(clean.distance)
    kept = failing & side
    assert count == expected
    assert int(found.unrefined) == 0 and np.array_equal(np.asarray(found.failing), failing)
    assert np.max(np.abs(distance[kept] - minimum[kept])) <= CLOSED_LINE["float64"]
    assert np.max(np.abs(distance[~kept] - refined[~kept])) <= CLOSED_LINE["float64"]
    assert np.max(np.abs(refined[kept] - minimum[kept])) > 1e-6, "the refined depth must differ to discriminate"


@pytest.mark.parametrize("seed", seed_list((21,)))
def test_a_seam_no_projection_reaches_leaves_every_planned_failing_point_unconverged(seed, monkeypatch):
    """The same by the union's own loops: with no projection steps no start reaches the seam, so every planned failing
    point counts, and the failing points outside the plan count as unrefined, not as unconverged."""
    normals, origin, fillet, points = drawn_wedge(seed)
    union = wedge_union(normals, origin, lambda p: p["fillet"])
    parameters = {"fillet": jnp.asarray(fillet)}
    at = jnp.asarray(points)
    failing = np.asarray(union.signed_distance(parameters, at).failing)
    plan = np.nonzero(failing)[0][::2]
    outside = int(failing.sum()) - plan.size
    assert plan.size > 0 and outside > 0
    union_module._compiled.cache_clear()
    monkeypatch.setattr(union_module, "PROJECTIONS", 0)
    found = union.signed_distance(parameters, at, plan)
    assert int(found.unconverged) == plan.size
    assert int(found.unrefined) == outside
    minimum = np.asarray(union.minimum(parameters, at))
    assert np.max(np.abs(np.asarray(found.distance)[failing] - minimum[failing])) <= CLOSED_LINE["float64"]
