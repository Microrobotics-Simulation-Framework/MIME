"""Region primitives for OpenMaterica: each region of a structure's kind is one of the scene manifest's analytic
primitives exactly, or refused with its reason. The oracle is OpenMaterica's own inside test, as its specification
states it (``geometry.md`` section 3.1, read on its ``main``: a sphere, a cylinder and a slab; a point on a surface is
inside; squared distances in double precision, in the order written). Its cases SC-006 and SC-007 have no case files
yet (its scene-documents suite has none), so their statements are taken as data here: inside, outside and on each
surface; a cylinder half its length either side of its centre, a slab half its thickness.

For every primitive MIME gives, its inside test agrees with MIME's own membership of the region at drawn points off
the boundary (the lumen as its finite prim: the cylinder within the tube's length), and on the boundary, at points
the primitive's own fields place there exactly, OpenMaterica's test says inside.

The control: a primitive whose direction or normal is not a unit vector, drawn off unit length by 1e-9 to a tenth, is
flagged by MIME's check of what it hands over (OpenMaterica's record check RC-092) on every draw, and a kind giving one
is refused at hand-off; the same check passes every primitive MIME's kinds give."""

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import assume, event, given, settings
from hypothesis import strategies as st
from script_a import ALL_RECORDS, fl9, script_a

from mime import Experiment, Scene, Structure
from mime.build import BuildRefused, manifest_geometry, primitive_problems, region_handoff
from mime.core import jnp, shape_named
from mime.registry import RegionPrimitive
from mime.scene import DeJonghHelix, Sphere, Tube, Z, placed
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
    """MIME's own membership of the region, and how far the point is from its boundary: the region's shape, and for
    the tube's lumen, whose prim is finite, the tube's length too."""
    parameters = {k: jnp.asarray(v) for k, v in geometry.region_parameters(region, settings_).items()}
    distance = float(shape_named(geometry.region_shapes[region]).signed_distance(parameters, jnp.asarray(point)))
    if isinstance(geometry, Tube):
        axis, centre = settings_["axis"], settings_["centre"]
        along = abs(sum((p - c) * a for p, c, a in zip(point, centre, axis, strict=True))) - settings_["length"] / 2
        distance = max(distance, along)
    return distance <= 0.0, distance


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
