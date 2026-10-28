"""Region primitives for OpenMaterica: each region of a structure's kind is one of the scene manifest's analytic
primitives exactly, or refused with its reason. The oracle is OpenMaterica's own inside test, as its specification
states it (``geometry.md`` section 3.1, read on its ``main``: a sphere, a cylinder and a slab; a point on a surface is
inside; squared distances in double precision, in the order written). Its cases SC-006 and SC-007 have no case files
yet (its scene-documents suite has none), so their statements are taken as data here: inside, outside and on each
surface; a cylinder half its length either side of its centre, a slab half its thickness.

For every primitive MIME gives, its inside test agrees with MIME's own membership of the region at drawn points off
the boundary, the region read as the kind registers it with nothing added. The tube's lumen is the one primitive that
is the region clipped: the region is the unbounded cylinder a law reads, and its primitive is the finite cylinder of
the tube's length, so the two agree within the tube's ends, and past an end the primitive is outside where the region
is inside (both asserted, each on its own points). On the boundary, at points the primitive's own fields place there
exactly, OpenMaterica's test says inside.

Every handed region names the prim the stage writer wrote it as, read back from the stage written from the same plan: a
sphere's solid is the structure's own Sphere prim, the tube's lumen a Cylinder below it, and a slab has none.

The control: a primitive whose direction or normal is not a unit vector, drawn off unit length by 1e-9 to a tenth, is
flagged by MIME's own check of what it hands over (at 64 units in the last place of a float64; OpenMaterica states no
tolerance and has no code or case for the check yet) on every draw, and a kind giving one is refused at hand-off; the
same check passes every primitive MIME's kinds give."""

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st
from pxr import Sdf, Usd, UsdGeom
from script_a import ALL_RECORDS, fl9, script_a

from mime import Experiment, Scene, Structure
from mime.build import (
    UNIT_TOLERANCE,
    BuildRefused,
    manifest_geometry,
    primitive_problems,
    region_handoff,
    stage_spec,
)
from mime.core import jnp, shape_named
from mime.registry import RegionPrimitive
from mime.scene import DeJonghHelix, SolidBox, Sphere, Tube, Z, placed, stage_text
from mime.units import m, mm, ms, standard_gravity

PROPERTY = settings(deadline=None)
EPS = float(np.finfo(np.float64).eps)
BAND_ULPS = 64.0


def materica_inside(geometry: dict, point: tuple[float, float, float]) -> bool:
    """OpenMaterica's inside test, ``geometry.md`` section 3.1, as written: ``d = p - centre``; a sphere ``d.d <=
    radius^2``; a cylinder, with ``t = d.a``, ``abs(t) <= length / 2`` and ``d.d - t^2 <= radius^2``; a slab
    ``abs((p - point).n) <= thickness / 2``."""
    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    if geometry["primitive"] == "slab":
        d = [p - c for p, c in zip(point, geometry["point"], strict=True)]
        return abs(dot(d, geometry["normal"])) <= geometry["thickness"] / 2
    d = [p - c for p, c in zip(point, geometry["centre"], strict=True)]
    if geometry["primitive"] == "sphere":
        return dot(d, d) <= geometry["radius"] ** 2
    t = dot(d, geometry["direction"])
    return abs(t) <= geometry["length"] / 2 and dot(d, d) - t ** 2 <= geometry["radius"] ** 2


def handed(geometry, name="held"):
    experiment = Experiment("regions", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1))
    experiment.add_structure(Structure(name, geometry=geometry))
    plans = experiment.plan()
    return region_handoff(plans), plans


def mime_inside(geometry, settings_, region: str, point) -> tuple[bool, float]:
    """MIME's own membership of the region as the kind registers it, and how far the point is from its boundary: the
    region's shape, and nothing added."""
    parameters = {k: jnp.asarray(v) for k, v in geometry.region_parameters(region, settings_).items()}
    distance = float(shape_named(geometry.region_shapes[region]).signed_distance(parameters, jnp.asarray(point)))
    return distance <= 0.0, distance


def beyond_an_end(geometry, settings_, point) -> float:
    """How far past the tube's end planes the point is along its axis (negative within them): where the lumen's
    primitive, which is clipped there, and the lumen, which has no ends, part."""
    along = abs(sum((p - c) * a for p, c, a in zip(point, settings_["centre"], settings_["axis"], strict=True)))
    return along - settings_["length"] / 2


@pytest.fixture(autouse=True)
def x64():
    import jax

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


lengths = st.floats(1e-5, 1e-2)
units_vectors = st.tuples(*(st.floats(-1.0, 1.0),) * 3).filter(lambda v: math.sqrt(sum(c * c for c in v)) >= 0.1)
offsets = st.tuples(*(st.floats(-2.0, 2.0),) * 3)


def kinds():
    return st.one_of(
        st.builds(lambda r: Sphere(radius=m(r)), lengths),
        st.builds(lambda r, w, length, a, c: Tube(axis=a, inner_radius=m(r), wall=m(w), length=m(length),
                                                  centre=m(c)),
                  lengths, lengths, lengths, units_vectors, st.tuples(*(st.floats(-1e-2, 1e-2),) * 3)),
        st.builds(lambda r, length: DeJonghHelix(radius=m(r), length=m(length), wavenumber=fl9().wavenumber,
                                                 modulation=fl9().modulation), lengths, lengths))


def probe(geometry: dict, offset) -> tuple[float, float, float]:
    """A point in the primitive's own box, ``offset`` (each component in [-2, 2]) in units of its half sizes: a
    sphere's radius; a cylinder's half length along its direction and its radius across; a slab's half thickness
    along its normal and its thickness across. About half the points fall inside."""
    if geometry["primitive"] == "sphere":
        return tuple(c + o * geometry["radius"] for c, o in zip(geometry["centre"], offset, strict=True))
    along, half, across = ((geometry["direction"], geometry["length"] / 2, geometry["radius"])
                           if geometry["primitive"] == "cylinder" else
                           (geometry["normal"], geometry["thickness"] / 2, geometry["thickness"]))
    a = np.asarray(along)
    u = np.cross(a, (1.0, 0.0, 0.0) if abs(a[0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    start = np.asarray(geometry.get("centre", geometry.get("point")))
    point = start + offset[0] * half * a + offset[1] * across * u + offset[2] * across * v
    return tuple(float(c) for c in point)


@PROPERTY
@given(kinds(), offsets)
def test_each_primitive_mime_gives_agrees_with_its_region_off_the_boundary(geometry, offset):
    handoff, plans = handed(geometry)
    settings_ = plans.geometries["held"].settings
    assert handoff.regions, "every kind drawn has a region with a primitive"
    size = max(abs(v) for value in settings_.values() for v in (value if isinstance(value, tuple) else (value,)))
    for region in handoff.regions:
        assert primitive_problems(region.primitive) == []
        geometry_ = manifest_geometry(region.primitive)
        point = probe(geometry_, offset)
        inside, distance = mime_inside(geometry, settings_, region.region.partition(".")[2], point)
        assume(abs(distance) > BAND_ULPS * EPS * size)
        if isinstance(geometry, Tube):
            past = beyond_an_end(geometry, settings_, point)
            assume(abs(past) > BAND_ULPS * EPS * size)
            if past > 0.0:
                event("lumen past an end")
                assert not materica_inside(geometry_, point), (region, point, past)
                continue
        event(f"{geometry_['primitive']} {'inside' if inside else 'outside'}")
        assert materica_inside(geometry_, point) is inside, (region, point, distance)


def surface_points(geometry: dict) -> list[tuple[float, float, float]]:
    """Points the primitive's own fields place on its surface: SC-006's on each surface and SC-007's extents."""
    if geometry["primitive"] == "sphere":
        c, r = geometry["centre"], geometry["radius"]
        return [(c[0] + r, c[1], c[2]), (c[0], c[1] - r, c[2])]
    if geometry["primitive"] == "slab":
        p, n, half = geometry["point"], geometry["normal"], geometry["thickness"] / 2
        return [tuple(a + s * half * b for a, b in zip(p, n, strict=True)) for s in (-1.0, 1.0)]
    c, a, half = geometry["centre"], geometry["direction"], geometry["length"] / 2
    return [tuple(x + s * half * y for x, y in zip(c, a, strict=True)) for s in (-1.0, 1.0)]


def test_sc_006_and_sc_007_as_data_on_the_primitives_script_as_tube_and_the_probed_shapes_give():
    """Script A's lumen, a ball's solid and FL-9's span: inside at the centre, outside well beyond, inside on each
    surface the fields place exactly (a cylinder's ends half its length either side of its centre, a slab's faces half
    its thickness either side of its point), and outside one step past an end along an axis of the frame."""
    geometries = [manifest_geometry(r.primitive) for r in region_handoff(script_a().plan()).regions]
    geometries += [manifest_geometry(handed(kind)[0].regions[0].primitive)
                   for kind in (Sphere(radius=mm(1.0)), fl9())]
    assert [g["primitive"] for g in geometries] == ["cylinder", "sphere", "slab"]
    for geometry in geometries:
        centre = geometry.get("centre", geometry.get("point"))
        assert materica_inside(geometry, tuple(centre))
        assert not materica_inside(geometry, tuple(c + 1.0 for c in centre))
        for point in surface_points(geometry):
            assert materica_inside(geometry, point), (geometry, point)
    lumen = geometries[0]
    assert lumen["direction"] == [1.0, 0.0, 0.0] and lumen["length"] == 0.1
    assert materica_inside(lumen, (0.05, 0.0, 0.0)) and not materica_inside(lumen, (math.nextafter(0.05, 1.0), 0.0,
                                                                                    0.0))


def test_script_as_handoff_is_the_lumen_with_its_material_and_the_wall_refused():
    handoff = region_handoff(script_a(records=ALL_RECORDS).plan())
    [lumen] = handoff.regions
    assert lumen.manifest_region() == {
        "name": "tube_lumen", "material": lumen.material,
        "geometry": {"primitive": "cylinder", "centre": [0.0, 0.0, 0.0], "direction": [1.0, 0.0, 0.0],
                     "radius": 0.003175, "length": 0.1}}
    assert lumen.material["id"] == "example/water" and lumen.prim == "/World/tube/lumen" and lumen.medium == "water"
    [wall] = handoff.refused
    assert wall.region == "tube.wall" and "annulus" in wall.reason
    declared = region_handoff(script_a().plan()).regions[0]
    assert declared.material is None and "declared" in declared.material_note
    with pytest.raises(ValueError, match="no material reference"):
        declared.manifest_region()


def test_a_capsule_and_the_screw_are_refused_with_their_reasons():
    from mime.scene import Capsule

    capsule, _ = handed(Capsule(radius=mm(0.5), height=mm(0.4), axis=(0.0, 0.6, 0.8)))
    screw, _ = handed(fl9())
    assert [(r.region, "hemispheres" in r.reason) for r in capsule.refused] == [("held.solid", True)]
    assert [r.region for r in screw.refused] == ["held.screw"] and [r.region for r in screw.regions] == ["held.span"]
    for refused in (*capsule.refused, *screw.refused):
        assert "label volume" in refused.reason and "scene document" in refused.reason, refused


def test_a_structure_whose_name_cannot_make_a_region_name_is_refused_when_planned_not_at_hand_off():
    """``Tube1`` names its regions ``Tube1_lumen``, which the manifest does not read: planning refuses it, naming the
    name to use, and the same structure named in lowercase plans. A structure whose kind gives no region primitives
    is not held to the manifest's names."""
    def plan(name, geometry):
        experiment = Experiment("regions", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1))
        experiment.add_structure(Structure(name, geometry=geometry))
        return experiment.plan()

    tube = Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(1.0), wall=mm(0.5), length=mm(10.0))
    with pytest.raises(BuildRefused) as refused:
        plan("Tube1", tube)
    [found] = refused.value.refusals
    assert found.code == "thing.name" and "Tube1_<region>" in found.message and "'tube1'" in found.action
    assert [r.name for r in region_handoff(plan("tube1", tube)).regions] == ["tube1_lumen"]
    plan("Box1", SolidBox(size=mm((1.0, 2.0, 3.0))))


def test_the_lumens_primitive_is_the_region_clipped_at_the_tubes_length_and_the_region_has_no_ends():
    """The registered lumen is the unbounded cylinder; its primitive the finite one of the tube's length. On the axis
    one length past an end the region says inside and the primitive outside, and within the ends they agree."""
    tube = Tube(axis=(0.0, 0.6, 0.8), inner_radius=mm(1.0), wall=mm(0.5), length=mm(10.0), centre=mm((1.0, 2.0, 3.0)))
    handoff, plans = handed(tube)
    settings_ = plans.geometries["held"].settings
    [lumen] = handoff.regions
    geometry_ = manifest_geometry(lumen.primitive)
    centre, axis = np.asarray(settings_["centre"]), np.asarray(settings_["axis"])
    for along, clipped in ((0.0, False), (4e-3, False), (14e-3, True), (-14e-3, True)):
        point = tuple(float(c) for c in centre + along * axis)
        inside, _ = mime_inside(tube, settings_, "lumen", point)
        assert inside, "the registered region has no ends"
        assert materica_inside(geometry_, point) is not clipped, (along, clipped)
        assert (beyond_an_end(tube, settings_, point) > 0.0) is clipped


def test_every_handed_region_names_the_prim_the_writer_wrote_read_back_from_the_stage():
    """Sphere: its solid is the structure's own Sphere prim. Tube: the lumen a Cylinder below it. Helix: the span, a
    slab, has no prim, and none is there. Each prim found is the primitive handed: its type, radius and length, and
    its centre and axis from its composed transform, in the stage's units."""
    cases = [Sphere(radius=mm(1.5)),
             Tube(axis=(0.0, 0.6, 0.8), inner_radius=mm(1.0), wall=mm(0.5), length=mm(10.0),
                  centre=mm((1.0, 2.0, 3.0))),
             fl9()]
    named = []
    for geometry in cases:
        experiment = Experiment("regions", scene=Scene.new(up=Z, units="mm"), gravity=standard_gravity(), dt=ms(1))
        experiment.add_structure(Structure("held", geometry=geometry))
        plans = experiment.plan()
        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(stage_text(stage_spec(plans)))
        stage = Usd.Stage.Open(layer)
        for region in region_handoff(plans).regions:
            primitive = region.primitive
            if primitive.primitive == "slab":
                assert region.prim is None
                assert not stage.GetPrimAtPath(f"/World/held/{region.region.partition('.')[2]}").IsValid()
                named.append((type(geometry).__name__, None))
                continue
            prim = stage.GetPrimAtPath(region.prim)
            assert prim.IsValid(), (region.prim, "is not on the stage")
            assert prim.GetTypeName() == primitive.primitive.capitalize()
            scale = 0.001  # the stage's metres per unit
            assert prim.GetAttribute("radius").Get() * scale == pytest.approx(primitive.fields["radius"], rel=1e-12)
            world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            assert np.allclose(np.asarray(world.ExtractTranslation()) * scale, primitive.fields["centre"], atol=1e-12)
            if primitive.primitive == "cylinder":
                assert prim.GetAttribute("height").Get() * scale == pytest.approx(primitive.fields["length"], rel=1e-12)
                axis = np.asarray(world.TransformDir((0.0, 0.0, 1.0)))
                assert np.allclose(axis, primitive.fields["direction"], atol=1e-12)
            named.append((type(geometry).__name__, region.prim))
    assert named == [("Sphere", "/World/held"), ("Tube", "/World/held/lumen"), ("DeJonghHelix", None)]


def off_unit(random) -> RegionPrimitive:
    """A drawn primitive of each kind with a direction, its direction or normal off unit length by a drawn amount."""
    vector = random.normal(size=3)
    vector = vector / np.linalg.norm(vector) * (1.0 + float(random.choice((-1.0, 1.0))) * 10 ** random.uniform(-9, -1))
    direction = tuple(float(c) for c in vector)
    if random.integers(2):
        return RegionPrimitive("cylinder", {"centre": (0.0, 0.0, 0.0), "direction": direction, "radius": 1e-3,
                                            "length": 1e-2})
    return RegionPrimitive("slab", {"point": (0.0, 0.0, 0.0), "normal": direction, "thickness": 1e-3})


def test_control_a_region_primitive_with_a_non_unit_direction_is_flagged_on_every_draw():
    random = np.random.default_rng(5)
    planted = [off_unit(random) for _ in range(500)]
    caught = sum(any("not one within" in p for p in primitive_problems(primitive)) for primitive in planted)
    print(f"\nflagged {caught} of {len(planted)}")
    assert caught == len(planted)
    assert primitive_problems(RegionPrimitive("cylinder", {"centre": (0.0, 0.0, 0.0), "direction": (1.0, 0.0, 0.0),
                                                           "radius": 0.0, "length": 1.0})) == [
        "its radius is 0.0, not a length above zero"]
    assert primitive_problems(RegionPrimitive("box", {})) != []


def test_the_directions_mime_hands_over_are_unit_to_a_few_ulps_far_inside_the_tolerance():
    """The tolerance is 64 units in the last place of a float64; the direction of the tube's lumen, a unit axis a kind
    has turned, is within 4 of one over 3,000 random axes of any scale of component, and the planted errors start at
    1e-9, 70,000 times the tolerance."""
    random = np.random.default_rng(2)
    worst = 0.0
    for _ in range(3000):
        raw = random.normal(size=3) * 10.0 ** random.uniform(-3, 3)
        axis = tuple(float(c) for c in raw / np.linalg.norm(raw))
        tube = Tube(axis=axis, inner_radius=mm(1.0), wall=mm(0.5), length=mm(10.0))
        settings_ = {"axis": axis, "inner_radius": 1e-3, "wall": 5e-4, "length": 1e-2, "centre": (0.0, 0.0, 0.0)}
        direction = placed(tube.region_primitive("lumen", settings_), tube, settings_).fields["direction"]
        worst = max(worst, abs(math.sqrt(math.fsum(c * c for c in direction)) - 1.0))
    print(f"\nworst deviation {worst / EPS:.2f} ulps; tolerance {UNIT_TOLERANCE / EPS:.0f}")
    assert worst <= 4.0 * EPS and UNIT_TOLERANCE == 64.0 * EPS


def test_control_a_kind_giving_a_non_unit_direction_is_refused_at_hand_off_never_handed_over(monkeypatch):
    """The tube's lumen planted a part in 10^6 off unit length. The control: unplanted, it is handed over."""
    experiment = Experiment("regions", scene=Scene.new(up=Z, units="m"), gravity=standard_gravity(), dt=ms(1))
    experiment.add_structure(Structure("held", geometry=Tube(axis=(1.0, 0.0, 0.0), inner_radius=mm(1.0),
                                                             wall=mm(0.5), length=mm(10.0))))
    plans = experiment.plan()
    assert [r.name for r in region_handoff(plans).regions] == ["held_lumen"]
    given_by_tube = Tube.region_primitive

    def off_unit_lumen(self, region, settings_):
        held = given_by_tube(self, region, settings_)
        if region != "lumen":
            return held
        return replace(held, fields={**held.fields, "direction": (0.0, 0.0, 1.0 + 1e-6)})

    monkeypatch.setattr(Tube, "region_primitive", off_unit_lumen)
    with pytest.raises(BuildRefused) as refused:
        region_handoff(plans)
    [found] = refused.value.refusals
    assert found.code == "entry.defect" and "direction" in found.message


def test_a_primitive_is_carried_into_the_scene_frame_by_its_kinds_placement():
    tube = Tube(axis=(0.0, 0.6, 0.8), inner_radius=mm(1.0), wall=mm(0.5), length=mm(10.0), centre=mm((1.0, 2.0, 3.0)))
    settings_ = {"axis": (0.0, 0.6, 0.8), "inner_radius": 1e-3, "wall": 5e-4, "length": 1e-2,
                 "centre": (1e-3, 2e-3, 3e-3)}
    moved = placed(tube.region_primitive("lumen", settings_), tube, settings_)
    assert moved.fields["direction"] == (0.0, 0.6, 0.8) and moved.fields["centre"] == (1e-3, 2e-3, 3e-3)
