"""The centreline contract (``mime.registry.Centreline``) on every kind that declares one, and its controls.

Every kind implementing ``centreline`` is held by ``mime.testing.centreline_problems``: the tangent the unit derivative
of the point in arc length and the frame's normal rotation-minimising, to rounding, on its own scale; the helix over
physical geometries Hypothesis draws (radius 10 um to 1 mm, pitch one to eight radii of either hand, a half to five
turns, the filament a fiftieth to a quarter of the radius, settings the kind refuses left out, as no body can have
them). Its derivatives in the parameters agree with central differences. A polyline's junctions meet and its frame is
carried round each corner; the guide's two routes (a lab's kind declaring a straight segment, a mesh's skeleton by
points) keep the contract.

**Controls**, each measured to fail on every draw: a tangent scaled by ``1 + 1e-6`` (no unit vector), and a helix
read by its angle rather than its arc length (its tangent no derivative of its point in arc length)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mime.build import ArgumentRefused
from mime.registry import Centreline, CentrelineFrame, CentrelineSegment, Junction, implements, registered
from mime.registry import EntryKind
from mime.scene import Capsule, Ellipsoid, Helix, Tube, TubeTree, polyline, straight
from mime.scene._centreline import transported
from mime.scene._helix import HELIX_CENTRELINE
from mime.testing import centreline_problems
from mime.units import DIMENSIONLESS, Quantity, m, mm

KINDS = {
    "helix": (Helix(radius=mm(0.15), pitch=mm(0.6), turns=2.5, filament_radius=mm(0.03)),
              {"radius": 0.15e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.03e-3}),
    "capsule": (Capsule(radius=mm(0.05), height=mm(2.0), axis=(0.0, 0.6, 0.8)),
                {"radius": 0.05e-3, "height": 2.0e-3, "axis": (0.0, 0.6, 0.8)}),
    "ellipsoid": (Ellipsoid(semi_axes=mm((0.1, 1.0, 0.1))), {"semi_axes": (0.1e-3, 1.0e-3, 0.1e-3)}),
    "tube": (Tube(axis=(0.0, 0.0, 1.0), inner_radius=mm(0.5), wall=mm(0.1), length=mm(20.0)),
             {"axis": (0.0, 0.0, 1.0), "inner_radius": 0.5e-3, "wall": 0.1e-3, "length": 20e-3,
              "centre": (0.0, 0.0, 0.0)}),
    "tube_tree": (TubeTree(points=mm((0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 1.5, 0.0, 2.6, -1.5, 0.0, 2.6)),
                           segments=((0, 1), (1, 2), (1, 3)), radii=mm((0.5, 0.5, 0.5, 0.4, 0.5, 0.3))),
                  {"points": (0.0, 0.0, -3e-3, 0.0, 0.0, 0.0, 1.5e-3, 0.0, 2.6e-3, -1.5e-3, 0.0, 2.6e-3),
                   "radii": (0.5e-3, 0.5e-3, 0.5e-3, 0.4e-3, 0.5e-3, 0.3e-3)}),
}

#: The property tests' settings: the profile's examples, no deadline (a draw compiles nothing new, but the first does).
PROPERTY = settings(deadline=None)


def helix_settings(radius: float, pitch: float, turns: float, filament: float) -> dict[str, float]:
    return {"radius": radius, "pitch": pitch, "turns": turns, "filament_radius": filament}


def helix_of(settings: dict[str, float]) -> Helix:
    return Helix(radius=m(settings["radius"]), pitch=m(settings["pitch"]),
                 turns=Quantity(settings["turns"], DIMENSIONLESS), filament_radius=m(settings["filament_radius"]))


@st.composite
def helices(draw: st.DrawFn) -> dict[str, float]:
    radius = draw(st.floats(1e-5, 1e-3))
    pitch = draw(st.floats(1.0, 8.0)) * radius * draw(st.sampled_from((1.0, -1.0)))
    settings = helix_settings(radius, pitch, draw(st.floats(0.5, 5.0)), draw(st.floats(0.02, 0.25)) * radius)
    held = helix_of(settings)
    from hypothesis import assume

    assume(not held.setting_problems(settings))
    return settings


def test_every_kind_that_declares_a_centreline_is_held_here():
    declared = {record.name for record in registered(EntryKind.GEOMETRY) if implements(record.cls, "centreline")}
    assert declared <= set(KINDS) | {"labx.bead"}, declared - set(KINDS)
    assert {"helix", "capsule", "ellipsoid", "tube"} <= declared


@pytest.mark.parametrize("name", sorted(KINDS))
def test_each_kind_keeps_the_contract(name):
    kind, settings = KINDS[name]
    assert centreline_problems(kind, settings) == []


@PROPERTY
@given(helices())
def test_the_helix_keeps_the_contract_over_physical_geometries(settings):
    assert centreline_problems(helix_of(settings), settings) == []


def test_the_tubes_radius_is_its_lumens_from_the_signed_distance():
    kind, settings = KINDS["tube"]
    segment = kind.centreline(settings).segments[0]
    radii = np.asarray(segment.radius(settings, jnp.asarray([0.0, 1e-3, 19e-3])))
    assert radii == pytest.approx(0.5e-3, rel=1e-15)


def test_the_helix_centreline_derivatives_agree_with_central_differences():
    kind, settings = KINDS["helix"]
    segment = kind.centreline(settings).segments[0]
    s = jnp.asarray([0.3e-3, 1.4e-3, 2.5e-3])

    def at(theta: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return segment.frame(theta, s)[0]

    theta = {key: jnp.asarray(value) for key, value in settings.items()}
    for name in ("radius", "pitch", "turns", "filament_radius"):
        exact = jax.jvp(lambda x, name=name: at({**theta, name: x}), (theta[name],), (jnp.asarray(1.0),))[1]
        h = 1e-6 * settings[name]
        up, down = (at({**theta, name: theta[name] + d}) for d in (h, -h))
        difference = (up - down) / (2.0 * h)
        assert np.asarray(exact) == pytest.approx(np.asarray(difference), rel=1e-6, abs=1e-9)


def test_a_polyline_meets_at_its_junctions_and_carries_its_frame():
    line = polyline([(0.0, 0.0, 0.0), (1e-3, 0.0, 0.0), (1e-3, 1e-3, 0.0), (2e-3, 1e-3, 1e-3)],
                    radii=[1e-5, 2e-5, 2e-5, 1e-5])
    assert [junction.name for junction in line.junctions] == ["junction_0", "junction_1"]
    assert centreline_problems(KINDS["helix"][0], {}, line) == []
    assert float(line.segments[0].filament_radius({})) == pytest.approx(2e-5)


@pytest.mark.parametrize("points, radii, said", [
    ([(0.0, 0.0, 0.0), (1e-3, 0.0, 0.0), (0.0, 0.0, 0.0)], None, "back on itself"),
    ([(0.0, 0.0, 0.0)], None, "two points or more"),
    ([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], None, "distinct and finite"),
    ([(0.0, 0.0, 0.0), (1e-3, 0.0, 0.0)], [1e-5, -1e-5], "above zero a point")],
    ids=["a_corner_turning_back", "one_point", "a_repeated_point", "a_radius_below_zero"])
def test_a_polyline_given_bad_points_is_refused_with_its_code(points, radii, said):
    with pytest.raises(ArgumentRefused) as refused:
        polyline(points, radii)
    (refusal,) = refused.value.refusals
    assert refusal.code == "call.polyline" and said in refusal.message


def _branch(daughter_normals):
    """A Y of three straight segments: a trunk along x, and two daughters leaving its end at 45 degrees up and down in
    the x-y plane, each with the given normal at its start."""
    trunk_normal = np.array([0.0, 0.0, 1.0])
    trunk = straight(lambda p: (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), lambda p: 1e-3, normal=trunk_normal)
    up, down = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0), np.array([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    daughters = [straight(lambda p: (1e-3, 0.0, 0.0), direction, lambda p: 1e-3, normal=normal)
                 for direction, normal in zip((up, down), daughter_normals, strict=True)]
    return Centreline((trunk, *daughters), (Junction("branch", ((0, 1), (1, 0), (2, 0))),)), trunk_normal, (up, down)


def test_a_branched_centrelines_frame_is_continuous_at_the_branch_point():
    """Each daughter's normal at the branch is the trunk's carried across the corner; a daughter whose normal is turned
    a quarter turn from that is reported, on every run (the control: the same Y, one daughter's normal turned)."""
    _, trunk_normal, (up, down) = _branch([np.array([0.0, 0.0, 1.0])] * 2)
    carried = [transported(trunk_normal, np.array([1.0, 0.0, 0.0]), d) for d in (up, down)]
    good, _, _ = _branch(carried)
    assert centreline_problems(KINDS["helix"][0], {}, good) == []
    turned = np.cross(up, carried[0])
    bad, _, _ = _branch([turned, carried[1]])
    (problem,) = centreline_problems(KINDS["helix"][0], {}, bad)
    assert "junction 'branch'" in problem and "segment 1's normal" in problem and "carried across the corner" in problem
    both, _, _ = _branch([turned, np.cross(down, carried[1])])
    assert len(centreline_problems(KINDS["helix"][0], {}, both)) == 2


def test_the_guides_straight_rod_keeps_the_contract():
    rod = Centreline((straight(lambda p: (-p["length"] / 2.0, 0.0, 0.0), (1.0, 0.0, 0.0), lambda p: p["length"],
                               filament_radius=lambda p: p["radius"]),))
    assert centreline_problems(KINDS["helix"][0], {"length": 1e-3, "radius": 1e-5}, rod) == []


def test_an_ellipsoid_that_is_no_prolate_spheroid_has_none():
    with pytest.raises(ValueError, match="prolate spheroid"):
        Ellipsoid(semi_axes=mm((0.1, 1.0, 0.2))).centreline({"semi_axes": (0.1e-3, 1.0e-3, 0.2e-3)})


def _helix_frame(parameters, s):
    return HELIX_CENTRELINE.segments[0].frame(parameters, s)


def _scaled_frame(parameters, s):
    held = _helix_frame(parameters, s)
    return CentrelineFrame(held.point, held.tangent * (1.0 + 1e-6), held.normal)


def _w(parameters):
    return jnp.sqrt(parameters["radius"] ** 2 + (parameters["pitch"] / (2.0 * math.pi)) ** 2)


def _by_angle_frame(parameters, s):
    return _helix_frame(parameters, jnp.asarray(s) * _w(parameters))  # s read as the angle: the point moves w a unit


#: The planted centrelines, module-level so that the harness compiles each once for every draw.
SCALED = Centreline((CentrelineSegment(length=HELIX_CENTRELINE.segments[0].length, frame=_scaled_frame,
                                       filament_radius=HELIX_CENTRELINE.segments[0].filament_radius),))
BY_ANGLE = Centreline((CentrelineSegment(length=lambda p: HELIX_CENTRELINE.segments[0].length(p) / _w(p),
                                         frame=_by_angle_frame,
                                         filament_radius=HELIX_CENTRELINE.segments[0].filament_radius),))


@PROPERTY
@given(helices())
def test_control_a_scaled_tangent_is_caught(settings):
    found = centreline_problems(helix_of(settings), settings, SCALED)
    assert any("no unit vector" in problem for problem in found), found


@PROPERTY
@given(helices())
def test_control_a_helix_read_by_its_angle_is_caught(settings):
    found = centreline_problems(helix_of(settings), settings, BY_ANGLE)
    assert any("not its point's derivative" in problem for problem in found), found
