"""Geometry kinds on the stage: a kind with a native form is written as USD's own primitive, with no copy of its
parameters and no centre or axis attribute, and read back to the same parameters; USD's own definition of the
primitive (its extent, computed by USD from the attributes written) lies on the kind's surface. A kind without one is
MIME's schema, its drawing derived on its ``proxy`` child. Every prim that is a kind carries ``MimeGeometryAPI``, and
``mime:geometry`` without it, or it without a kind, is refused. Taking a kind from a stage is stage-first, and refused.

The controls, each failing on every draw (their detection rates printed): a capsule written with its height including
its caps, and native lengths written in metres into a stage of another unit (in metres the two are the same number,
so the control draws the other units: a test that cannot discriminate is no evidence)."""

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pxr import Sdf, Usd, UsdGeom
from robot_asset import bound_robot, write_robot

from mime import Experiment, Scene, Structure
from mime.build import BuildRefused, stage_spec
from mime.core import jnp, shape_named
from mime.registry import implements
from mime.scene import (
    GEOMETRY_API,
    Capsule,
    DeJonghHelix,
    SolidBox,
    SolidCylinder,
    SolidEllipsoid,
    Sphere,
    Tube,
    X,
    Z,
    check_stage,
    geometry_schema_problems,
    kind_frame,
    read_kind,
    stage_text,
)
from mime.scene import _testing as stage_switches
from mime.scene._kinds import rotate
from mime.units import dimensionless, m, mm, ms, standard_gravity

PROPERTY = settings(deadline=None)
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "um": 1e-6}
EPS = float(np.finfo(np.float64).eps)
#: USD computes a primitive's extent in float32: the oracle resolves a length to about this share of it.
EXTENT_ROUNDING = 4.0 * float(np.finfo(np.float32).eps)

lengths = st.floats(1e-6, 1e-2)
axes = st.tuples(*(st.floats(-1.0, 1.0),) * 3).filter(lambda v: math.sqrt(sum(c * c for c in v)) >= 0.1)


def native_kinds():
    return st.one_of(
        st.builds(lambda r: Sphere(radius=m(r)), lengths),
        st.builds(lambda r, h, a: Capsule(radius=m(r), height=m(h), axis=a), lengths, st.floats(0.0, 2e-2), axes),
        st.builds(lambda r, h, a: SolidCylinder(radius=m(r), length=m(h), axis=a), lengths, lengths, axes))


def staged(geometry, units: str = "m"):
    """``geometry`` as a structure in a scene of ``units``: the stage written from the plan, and the plan."""
    experiment = Experiment("kinds", scene=Scene.new(up=Z, units=units), gravity=standard_gravity(), dt=ms(1))
    experiment.add_structure(Structure("held", geometry=geometry))
    plans = experiment.plan()
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(stage_text(stage_spec(plans)))
    return Usd.Stage.Open(layer), plans


def round_trip_problems(geometry, units: str) -> list[str]:
    """Every way the kind written into a stage of ``units`` reads back other than as the plan's settings, or is not
    written as its native primitive, or carries a copy of a parameter."""
    stage, plans = staged(geometry, units)
    prim = stage.GetPrimAtPath("/World/held")
    native = type(geometry).native
    found: list[str] = []
    if prim.GetTypeName() != native.primitive:
        found.append(f"written as a {prim.GetTypeName()}, not a {native.primitive}")
    copies = [a.GetName() for a in prim.GetAttributes() if a.GetName().startswith("mime:geometry:")
              and a.GetName() != "mime:geometry:version"]
    if copies:
        found.append(f"carries {copies} beside its native attributes")
    read, problems = read_kind(prim, UNITS[units])
    if read is None:
        return [*found, *(str(p) for p in problems)]
    expected = plans.geometries["held"].settings
    for name, value in expected.items():
        back = read.settings.get(name)
        if isinstance(value, tuple):
            if back is None or max(abs(a - b) for a, b in zip(back, value, strict=True)) > 8.0 * EPS:
                found.append(f"{name} wrote {value} and read back {back}")
        elif back is None or abs(back - value) > 4.0 * EPS * abs(value):
            found.append(f"{name} wrote {value!r} and read back {back!r}")
    return found


def oracle_problems(geometry, units: str) -> list[str]:
    """USD's own definition of the written primitive against the kind: the extent USD computes from the attributes,
    in metres, holds points of the kind's surface on each of its own axes. For a kind with a run-time shape each such
    point is on the shape's surface; a cylinder's are at its radius and half its length."""
    stage, plans = staged(geometry, units)
    prim = stage.GetPrimAtPath("/World/held")
    low, high = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
    scale = UNITS[units]
    settings_ = plans.geometries["held"].settings
    tips = [tuple(scale * c * (1.0 if i == j else 0.0) for j in range(3)) for i in range(3) for c in (low[i], high[i])]
    size = max(abs(c) for tip in tips for c in tip)
    found: list[str] = []
    if isinstance(geometry, SolidCylinder):
        expected = [settings_["radius"], settings_["radius"], settings_["length"] / 2.0]
        for i in range(3):
            if max(abs(abs(scale * low[i]) - expected[i]), abs(scale * high[i] - expected[i])) > EXTENT_ROUNDING * size:
                found.append(f"USD's extent along its own axis {i} is {scale * low[i]!r} to {scale * high[i]!r} m, "
                             f"and the cylinder's {expected[i]!r} either side")
        return found
    (region,) = geometry.regions
    shape = shape_named(geometry.region_shapes[region])
    parameters = {k: jnp.asarray(v) for k, v in geometry.region_parameters(region, settings_).items()}
    translation, orientation = kind_frame(geometry, settings_)
    for tip in tips:
        point = np.add(rotate(orientation, tip), translation)
        distance = float(shape.signed_distance(parameters, jnp.asarray(point)))
        if abs(distance) > EXTENT_ROUNDING * size:
            found.append(f"USD's extent reaches {tip} in the kind's frame, {distance!r} m from its surface")
    return found


@pytest.fixture(autouse=True)
def x64():
    import jax

    held = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", held)


@PROPERTY
@given(native_kinds(), st.sampled_from(sorted(UNITS)))
def test_a_native_kind_is_its_usd_primitive_and_reads_back_to_its_parameters(geometry, units):
    assert round_trip_problems(geometry, units) == []


@PROPERTY
@given(native_kinds(), st.sampled_from(sorted(UNITS)))
def test_usds_own_definition_of_the_written_primitive_lies_on_the_kinds_surface(geometry, units):
    assert oracle_problems(geometry, units) == []


def detection(draws, check) -> tuple[int, int]:
    caught = sum(bool(check(*draw)) for draw in draws)
    return caught, len(draws)


def drawn(strategy, count: int, seed: int) -> list:
    """``count`` draws of ``strategy``, from a seeded generator: the controls' detection rates are measured over
    these, each control asserted on every one."""
    random = np.random.default_rng(seed)
    found = []
    for _ in range(count):
        found.append(strategy(random))
    return found


def random_capsule(random) -> Capsule:
    axis = random.normal(size=3)
    return Capsule(radius=m(float(10 ** random.uniform(-6, -2))), height=m(float(random.uniform(0.0, 2e-2))),
                   axis=tuple(float(c) for c in axis / np.linalg.norm(axis)))


def test_control_a_capsule_written_with_its_height_including_its_caps_fails_both_checks_on_every_draw():
    """USD's ``Capsule.height`` excludes the caps (its extent along the axis is half the height plus the radius):
    written as the whole length, the capsule reads back taller and USD draws it past the kind's surface."""
    draws = [(capsule, units) for capsule, units in zip(drawn(random_capsule, 200, 7), ["m", "cm", "mm", "um"] * 50,
                                                        strict=True)]
    with stage_switches.capsule_height_with_caps():
        round_trip = detection(draws, round_trip_problems)
        oracle = detection(draws, oracle_problems)
    print(f"\ncaught by the round trip {round_trip[0]} of {round_trip[1]}, by USD's definition {oracle[0]} of "
          f"{oracle[1]}")
    assert round_trip == oracle == (200, 200)
    assert detection(draws[:20], round_trip_problems) == (0, 20)  # unplanted, the same draws pass


def random_native(random):
    axis = random.normal(size=3)
    axis = tuple(float(c) for c in axis / np.linalg.norm(axis))
    radius, length = (m(float(10 ** random.uniform(-6, -2))) for _ in range(2))
    return [Sphere(radius=radius), Capsule(radius=radius, height=length, axis=axis),
            SolidCylinder(radius=radius, length=length, axis=axis)][int(random.integers(3))]


def test_control_native_lengths_written_in_metres_into_a_stage_of_other_units_fail_both_checks_on_every_draw():
    draws = [(kind, units) for kind, units in zip(drawn(random_native, 150, 11), ["cm", "mm", "um"] * 50,
                                                  strict=True)]
    with stage_switches.native_lengths_in_metres():
        round_trip = detection(draws, round_trip_problems)
        oracle = detection(draws, oracle_problems)
        in_metres = detection([(kind, "m") for kind, _ in draws[:20]], round_trip_problems)
    print(f"\ncaught by the round trip {round_trip[0]} of {round_trip[1]}, by USD's definition {oracle[0]} of "
          f"{oracle[1]}; in metres, where it is the same number, {in_metres[0]} of {in_metres[1]}")
    assert round_trip == oracle == (150, 150)
    assert in_metres == (0, 20)


def test_a_kind_without_a_native_form_is_mimes_schema_and_its_drawing_a_derived_proxy():
    helix = DeJonghHelix(radius=mm(1.56), length=mm(7.47), wavenumber=dimensionless(2.33),
                         modulation=dimensionless(0.33))
    for geometry in (helix, SolidBox(size=mm((1.0, 2.0, 3.0))), SolidEllipsoid(semi_axes=mm((1.0, 2.0, 3.0))),
                     Tube(axis=+X, inner_radius=mm(3.0), wall=mm(1.0), length=mm(9.0))):
        stage, plans = staged(geometry, "mm")
        prim = stage.GetPrimAtPath("/World/held")
        assert prim.GetTypeName() == "Xform" and GEOMETRY_API in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
        proxy = stage.GetPrimAtPath("/World/held/proxy")
        assert UsdGeom.Imageable(proxy).GetPurposeAttr().Get() == "proxy"
        assert proxy.GetAttribute("mime:derived_from").Get() and proxy.GetAttribute("mime:derived_hash").Get()
        read, problems = read_kind(prim, 0.001)
        assert problems == [] and read is not None
        for name, value in plans.geometries["held"].settings.items():
            back = read.settings[name]
            assert np.allclose(back, value, rtol=8 * EPS, atol=8 * EPS), (name, back, value)
    assert read.settings["axis"] == pytest.approx((1.0, 0.0, 0.0), abs=4 * EPS)  # the tube's, from its transform


def test_every_kind_on_one_stage_passes_usd_cores_validators(tmp_path):
    stage, _ = staged(Capsule(radius=mm(0.5), height=mm(0.4), axis=(0.0, 0.6, 0.8)), "mm")
    path = tmp_path / "kinds.usda"
    stage.GetRootLayer().Export(str(path))
    findings = check_stage(path)
    print("\n".join(f"{f.severity} {f.validator}: {f.message}" for f in findings) or "no findings")
    assert [f for f in findings if f.severity == "error"] == []


def test_mime_geometry_without_its_schema_or_the_schema_without_a_kind_is_refused():
    """``MimeGeometryAPI`` is read through ``GetPrimTypeInfo``, as ``MimeMagnetAPI`` is. The controls: the prim as
    written reads."""
    stage, _ = staged(Sphere(radius=mm(1.0)))
    prim = stage.GetPrimAtPath("/World/held")
    assert geometry_schema_problems(prim) == []
    prim.RemoveAppliedSchema(GEOMETRY_API)
    [(code, _, _)] = geometry_schema_problems(prim)
    assert code == "geometry.schema-not-applied"
    prim.AddAppliedSchema(GEOMETRY_API)
    prim.RemoveProperty("mime:geometry")
    [(code, _, _)] = geometry_schema_problems(prim)
    assert code == "geometry.schema-incomplete"


def test_scene_open_refuses_stage_first_authoring_naming_the_python_first_way():
    with pytest.raises(NotImplementedError) as refused:
        Scene.open("robot.usda")
    for words in ("stage-first authoring", "Scene.new(up=..., units=...)", "scene.reference(asset(path), at=...)"):
        assert words in str(refused.value)


def test_an_asset_authoring_a_geometry_kind_is_refused_as_stage_first(tmp_path):
    """A kind on an asset's prim would be a second authority for a shape the experiment declares in Python. The
    control: the same asset without it builds."""
    robot = write_robot(tmp_path / "robot.usda")
    bound_robot(robot=robot).plan()
    layer = Sdf.Layer.FindOrOpen(str(robot))
    stage = Usd.Stage.Open(layer)
    shell = stage.GetPrimAtPath("/UMR/Shell")
    shell.AddAppliedSchema(GEOMETRY_API)
    shell.CreateAttribute("mime:geometry", Sdf.ValueTypeNames.Token).Set("sphere")
    layer.Save()
    with pytest.raises(BuildRefused) as refused:
        bound_robot(robot=Path(robot)).plan()
    [found] = [r for r in refused.value.refusals if r.code == "scene.stage-first"]
    assert "/UMR/Shell" in found.message and "Python" in found.action


def test_every_mime_kind_with_a_native_form_is_one_of_usds_primitives():
    for kind in (Sphere, Capsule, SolidCylinder):
        assert implements(kind, "native") and hasattr(UsdGeom, kind.native.primitive)
    for kind in (DeJonghHelix, Tube, SolidBox, SolidEllipsoid):
        assert not implements(kind, "native") and implements(kind, "drawing")
