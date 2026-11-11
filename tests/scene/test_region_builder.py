"""The region builder (``mime.scene._region``): a region cut from a vessel's centreline by planes, each cut continued by
its extension and bent onto a face of the grid's axis, built as a tube tree. Geometry only: the flow on it is
``tests/flow/test_fvm_region.py``'s."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.scene import TubeTree
from mime.scene._region import RegionRefused, Vessel, build_region, tube_vessel

Z = np.asarray([0.0, 0.0, 1.0])
#: A Y: a parent of radius 0.5 from (0, 0, -3) to the junction at the origin, daughters of 0.4 and 0.35 up and out.
Y_POINTS = np.asarray([[0.0, 0.0, -3.0], [0.0, 0.0, 0.0], [1.2, 0.0, 2.6], [-1.0, 0.0, 2.8]]) * 1e-3
Y_SEGMENTS = ((0, 1), (1, 2), (1, 3))
Y_RADII = np.asarray([[0.5, 0.5], [0.4, 0.4], [0.35, 0.35]]) * 1e-3


def y_vessel(points: np.ndarray = Y_POINTS) -> Vessel:
    return Vessel(points, Y_SEGMENTS, Y_RADII, 0.1, (("root", 0), ("end_0", 2), ("end_1", 3)))


def _distance_to_tree(region, point) -> float:
    return float(np.asarray(region.signed_distance(np.asarray([point]))).reshape(-1)[0])


def test_the_whole_vessel_keeps_its_ends_names_and_turns_each_cut_to_a_face():
    region = build_region(y_vessel(), around=None, reach=None, extension=2.0, bend=2.0)
    assert [o.name for o in region.openings] == ["root", "end_0", "end_1"]
    assert [o.face for o in region.openings] == [-1, 1, 1]
    assert [o.upstream for o in region.openings] == [True, False, False]
    low, high = region.faces
    for opening in region.openings:
        assert opening.centre[2] == pytest.approx(low if opening.face < 0 else high, abs=1e-15)
        assert opening.extension == pytest.approx(4.0 * opening.radius)
        # The opening's centre is on the lumen's axis: its distance is minus its radius (the run is straight there).
        assert _distance_to_tree(region, opening.centre) == pytest.approx(-opening.radius, rel=1e-9)
    # The daughters turn by their angle from z; the root's branch is along z already.
    daughter = math.degrees(math.acos(2.6 / math.hypot(1.2, 2.6)))
    assert region.openings[0].turn == pytest.approx(0.0, abs=1e-9)
    assert region.openings[1].turn == pytest.approx(daughter, rel=1e-12)
    assert region.openings[1].bend == pytest.approx(4.0 * 0.4e-3 * math.radians(daughter), rel=1e-12)


def test_the_fillet_radius_is_the_vessels():
    region = build_region(y_vessel(), around=None, reach=None, extension=1.0, bend=2.0)
    made = TubeTree.si(region.points, region.segments, region.radii, region.fillet).fillet_radius_value()
    given = TubeTree.si(Y_POINTS, Y_SEGMENTS, Y_RADII, 0.1).fillet_radius_value()
    assert made == pytest.approx(given, rel=1e-12)


def test_each_path_is_the_cuts_section_from_the_cut_through_its_bend_to_past_its_face():
    """Along each opening's path (its extension, the bend's midpoint, the run's end at the face) the distance is minus
    the cut's radius; a radius off the path's axis it is zero; and the vessel's own part ends at the cuts."""
    region = build_region(y_vessel(), around=1, reach=1.5e-3, extension=2.0, bend=2.0)
    for row, opening in zip(region.paths, region.openings, strict=True):
        cut, tangent, extension = row[0:3], row[3:6], row[6]
        centre, start, normal, curve, angle = row[7:10], row[10:13], row[13:16], row[16], row[17]
        middle = centre + curve * (start * math.cos(angle / 2) + np.cross(normal, start) * math.sin(angle / 2)) \
            if angle > 0 else cut + extension * tangent
        for point in (cut + 0.5 * extension * tangent, middle, np.asarray(opening.centre)):
            assert _distance_to_tree(region, point) == pytest.approx(-opening.radius, rel=1e-9)
        side = np.cross(tangent, [0.3, 0.5, 0.8])
        side = side / np.linalg.norm(side)
        assert _distance_to_tree(region, cut + 0.5 * extension * tangent + opening.radius * side) == \
            pytest.approx(0.0, abs=1e-12)


def test_a_cut_around_a_junction_takes_the_end_beyond_it_and_its_section():
    region = build_region(y_vessel(), around=1, reach=1.5e-3, extension=0.0, bend=2.0)
    cuts = {o.name: np.asarray(o.cut) for o in region.openings}
    np.testing.assert_allclose(cuts["root"], [0.0, 0.0, -1.5e-3], atol=1e-15)
    direction = Y_POINTS[2] / np.linalg.norm(Y_POINTS[2])
    np.testing.assert_allclose(cuts["end_0"], 1.5e-3 * direction, atol=1e-15)
    assert {o.name: o.radius for o in region.openings} == pytest.approx({"root": 0.5e-3, "end_0": 0.4e-3,
                                                                         "end_1": 0.35e-3})


def test_a_reach_past_an_end_cuts_at_the_end_itself():
    region = build_region(y_vessel(), around=1, reach=10e-3, extension=0.0, bend=2.0)
    whole = build_region(y_vessel(), around=None, reach=None, extension=0.0, bend=2.0)
    for cut, own in zip(region.openings, whole.openings, strict=True):
        np.testing.assert_allclose(cut.cut, own.cut, atol=1e-15)


def test_a_cut_with_several_ends_beyond_it_is_refused():
    """A tree dividing twice: a cut on the parent below its first junction leads to three ends."""
    points = np.asarray([[0, 0, -4], [0, 0, -2], [0, 0, 0], [1, 0, 2], [-1, 0, 2], [-2, 0, 1.0]], dtype=float) * 1e-3
    segments = ((0, 1), (1, 2), (2, 3), (2, 4), (1, 5))
    radii = np.full((5, 2), 0.3e-3)
    vessel = Vessel(points, segments, radii, 0.1, (("root", 0), ("end_0", 3), ("end_1", 4), ("end_2", 5)))
    passed = build_region(vessel, around=2, reach=2.5e-3, extension=0.0, bend=2.0)  # past j1: one end a cut
    assert [o.name for o in passed.openings] == ["root", "end_0", "end_1", "end_2"]
    with pytest.raises(RegionRefused) as refused:  # between j2 and j1: the root and end_2 beyond one cut
        build_region(vessel, around=2, reach=1.0e-3, extension=0.0, bend=2.0)
    assert refused.value.kind == "cut" and "ends beyond it" in refused.value.message


def test_a_cut_square_to_the_axis_and_openings_on_one_face_are_refused():
    across = tube_vessel((0.0, 0.0, 0.0), (2e-3, 0.0, 0.0), 0.3e-3, 0.1)
    with pytest.raises(RegionRefused) as refused:
        build_region(across, around=None, reach=None, extension=0.0, bend=2.0)
    assert refused.value.kind == "face"
    hairpin = Vessel(np.asarray([[0, 0, 0], [0, 0, 2], [1.5, 0, 2.2], [1.5, 0, 0]], dtype=float) * 1e-3,
                     ((0, 1), (1, 2), (2, 3)), np.full((3, 2), 0.3e-3), 0.1, (("start", 0), ("end", 3)))
    with pytest.raises(RegionRefused) as one_face:
        build_region(hairpin, around=None, reach=None, extension=0.0, bend=2.0)
    assert one_face.value.kind == "face" and "both faces" in one_face.value.message


def test_openings_whose_lumens_cross_the_line_between_them_are_refused():
    close = y_vessel(np.asarray([[0.0, 0.0, -3.0], [0.0, 0.0, 0.0], [0.3, 0.0, 2.6], [-0.3, 0.0, 2.8]]) * 1e-3)
    with pytest.raises(RegionRefused) as refused:
        build_region(close, around=None, reach=None, extension=0.0, bend=2.0)
    assert "midway" in refused.value.message


@settings(max_examples=60, deadline=None)
@given(tilt=st.floats(0.5, 55.0), azimuth=st.floats(0.0, 2.0 * math.pi), downward=st.booleans(),
       offset=st.tuples(*[st.floats(-2e-4, 2e-4)] * 3), extension=st.floats(0.0, 4.0), bend=st.floats(1.0, 4.0))
def test_a_tilted_tube_at_any_rotation_reaches_both_faces_along_its_own_section(tilt, azimuth, downward, offset,
                                                                                 extension, bend):
    """A straight tube tilted from z by up to 55 degrees at any azimuth (never on a symmetry plane by construction),
    either way along z, and moved by part of a cell: each cut turns onto its face by the tube's own tilt, the plane
    meets a straight run of the cut's radius, and the region keeps the tube's own section between its cuts."""
    theta = math.radians(tilt)
    direction = np.asarray([math.sin(theta) * math.cos(azimuth), math.sin(theta) * math.sin(azimuth),
                            (-1.0 if downward else 1.0) * math.cos(theta)])
    start = np.asarray(offset)
    end = start + 3e-3 * direction
    region = build_region(tube_vessel(start, end, 0.4e-3, 0.1), around=None, reach=None, extension=extension, bend=bend)
    for opening in region.openings:
        assert opening.turn == pytest.approx(tilt, rel=1e-9)
        assert _distance_to_tree(region, opening.centre) == pytest.approx(-0.4e-3, rel=1e-9)
    assert _distance_to_tree(region, 0.5 * (start + end)) == pytest.approx(-0.4e-3, rel=1e-9)
    assert {o.face for o in region.openings} == {-1, 1}
    assert region.openings[0].face == (1 if downward else -1)
