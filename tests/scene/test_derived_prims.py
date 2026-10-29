"""Derived prims: a kind's drawn mesh on its ``proxy`` child and a region's primitive (the tube's lumen) are output
only, written with the hash of what they hold and of the parameters they come from. The stage round trip reports an
edited derived prim and loads (``research``: a ``strict`` profile, when profiles arrive, refuses it); any other edit
of the stage is refused; the rebuilt stage is the one first written. A reader with only the stage finds a derived prim
edited through a stronger layer, and one left stale by an edit of its source.

The control: edits planted in the proxy (a vertex moved, a face turned over, the proxy removed, the mesh and its
recorded hash rewritten together) are each reported on every draw, by the comparison the load makes; the hash a
reader with only the stage recomputes catches every one but the removed proxy and the rewritten hash, which it
cannot see, and says so. The form of a derived prim beyond its content (a proxy's purpose, orientation, subdivision,
visibility or scale op; a lumen's purpose, visibility, extent or scale op) is found by that reader and by the load, each
edit planted once and found on every run; what no check covers is stated in the scene README."""

import shutil

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, Vt
from script_a import script_a

from mime.experiment import Experiment
from mime.saved import LoadRefused
from mime.scene import (
    DERIVED_HASH,
    STAGE_FILE,
    derived_findings,
    stage_derived_findings,
    stage_edits,
)
from mime.scene._derived import mesh_hash

PROXY = "/World/tube/proxy"


@pytest.fixture(scope="module")
def saved_a(tmp_path_factory):
    folder = tmp_path_factory.mktemp("saved_a")
    script_a().build().save(folder)
    return folder


def edited(text: str, edit) -> str:
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(text)
    edit(layer)
    return layer.ExportToString()


def moved_vertex(index: int, by: float):
    def edit(layer):
        points = list(layer.GetAttributeAtPath(f"{PROXY}.points").default)
        p = points[index]
        points[index] = Gf.Vec3f(p[0] * (1.0 + by), p[1], p[2])
        layer.GetAttributeAtPath(f"{PROXY}.points").default = Vt.Vec3fArray(points)
    return edit


def turned_face(face: int):
    def edit(layer):
        indices = list(layer.GetAttributeAtPath(f"{PROXY}.faceVertexIndices").default)
        indices[4 * face:4 * face + 4] = indices[4 * face:4 * face + 4][::-1]
        layer.GetAttributeAtPath(f"{PROXY}.faceVertexIndices").default = Vt.IntArray(indices)
    return edit


def removed(layer):
    spec = layer.GetPrimAtPath(PROXY)
    del spec.nameParent.nameChildren[spec.name]


def rehashed(index: int, by: float):
    """A vertex moved and the recorded hash rewritten to match: invisible to a reader with only the stage."""
    move = moved_vertex(index, by)

    def edit(layer):
        move(layer)
        held = [layer.GetAttributeAtPath(f"{PROXY}.{name}").default for name in
                ("points", "faceVertexCounts", "faceVertexIndices")]
        layer.GetAttributeAtPath(f"{PROXY}.{DERIVED_HASH}").default = mesh_hash(*held)
    return edit


def test_the_round_trip_reports_an_edited_derived_mesh_and_loads_rebuilding_the_stage_first_written(saved_a, tmp_path):
    folder = tmp_path / "edited"
    shutil.copytree(saved_a, folder)
    written = (folder / STAGE_FILE).read_text()
    (folder / STAGE_FILE).write_text(edited(written, moved_vertex(5, 1e-3)))
    loaded = Experiment.load(folder)
    report = loaded.load_report
    print(report.text())
    [line] = report.derived_prims_edited
    assert line.startswith(f"{PROXY} was edited after it was written: it hashes to sha256:")
    assert "derived prims edited after it was written (reported, not refused)" in report.text()
    assert loaded.build().stage_text == written
    # The control: the folder as saved loads with nothing to report.
    assert Experiment.load(saved_a).load_report.derived_prims_edited == ()


@pytest.mark.parametrize("edit", ["inner_radius", "lumen"])
def test_an_edit_beyond_the_derived_prims_is_refused(saved_a, tmp_path, edit):
    """The tube's own parameter, and the lumen moved out of its kind's frame along with an edit of its recorded
    hash: the lumen is derived, so moving it is reported, but a parameter of the kind is the stage's own."""
    folder = tmp_path / edit
    shutil.copytree(saved_a, folder)
    written = (folder / STAGE_FILE).read_text()

    def parameter(layer):
        layer.GetAttributeAtPath("/World/tube.mime:geometry:inner_radius").default = 0.004

    def lumen(layer):
        layer.GetAttributeAtPath("/World/tube/lumen.radius").default = 0.004

    (folder / STAGE_FILE).write_text(edited(written, parameter if edit == "inner_radius" else lumen))
    if edit == "lumen":
        assert len(Experiment.load(folder).load_report.derived_prims_edited) == 1
        return
    with pytest.raises(LoadRefused) as refused:
        Experiment.load(folder)
    assert refused.value.codes() == ["load.stage-changed"]


def test_a_reader_with_only_the_stage_finds_a_proxy_edited_through_a_stronger_layer_and_a_stale_one(saved_a):
    """The composed stage: an over in a stronger layer moves a vertex of the proxy (edited), or changes the tube's
    radius (the proxy and the lumen are stale: their source moved and they were not regenerated). The control: the
    stage as saved has no finding."""
    assert derived_findings(Usd.Stage.Open(str(saved_a / STAGE_FILE))) == []
    for over, expected in ((f"{PROXY}.points", {(PROXY, "edited")}),
                           ("/World/tube.mime:geometry:inner_radius", {(PROXY, "stale"),
                                                                       ("/World/tube/lumen", "stale")})):
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths.append(str(saved_a / STAGE_FILE))
        attribute = stage.GetAttributeAtPath(over)
        value = attribute.Get()
        attribute.Set(Vt.Vec3fArray([Gf.Vec3f(p[0] * 1.01, p[1], p[2]) for p in value])
                      if over.endswith("points") else value * 1.01)
        found = derived_findings(stage)
        print("\n".join(f.message for f in found))
        assert {(f.path, f.finding) for f in found} == expected


def test_control_planted_proxy_edits_are_reported_on_every_draw(saved_a):
    """Drawn edits of each kind; each must be found by the comparison the load makes, and the hash check must find
    all but the rewritten hash. Rates printed."""
    written = (saved_a / STAGE_FILE).read_text()
    stage = Usd.Stage.Open(str(saved_a / STAGE_FILE))
    vertices = len(stage.GetAttributeAtPath(f"{PROXY}.points").Get())
    random = np.random.default_rng(3)
    plants = [("moved", moved_vertex(int(random.integers(vertices)), float(10 ** random.uniform(-5, -1))))
              for _ in range(40)]
    plants += [("turned", turned_face(int(random.integers(4 * 64)))) for _ in range(40)]
    plants += [("removed", removed)]
    plants += [("rehashed", rehashed(int(random.integers(vertices)), float(10 ** random.uniform(-5, -1))))
               for _ in range(40)]
    by_load = {name: 0 for name, _ in plants}
    by_hash = dict(by_load)
    counts = dict(by_load)
    for name, plant in plants:
        text = edited(written, plant)
        counts[name] += 1
        edits = stage_edits(text, written)
        by_load[name] += int(not edits.outside and edits.derived == (PROXY,))
        by_hash[name] += int(any(f.path == PROXY for f in stage_derived_findings(text)))
    print(f"\nby the load's comparison {by_load}, by the stage's own hash {by_hash}, of {counts}")
    assert by_load == counts
    assert by_hash == {**counts, "removed": 0, "rehashed": 0}


LUMEN = "/World/tube/lumen"


def authored(path: str, name: str, type_name, value):
    """An edit that authors ``name`` on the prim at ``path`` with ``value`` (the attribute made if it is not there)."""
    def edit(layer):
        spec = layer.GetPrimAtPath(path)
        attribute = layer.GetAttributeAtPath(f"{path}.{name}")
        if attribute is None:
            attribute = Sdf.AttributeSpec(spec, name, type_name)
        attribute.default = value
    return edit


def with_scale_op(path: str):
    """A scale op on the prim: its ``xformOpOrder`` authored beside it, as USD requires, so the op composes."""
    def edit(layer):
        spec = layer.GetPrimAtPath(path)
        authored(path, "xformOp:scale", Sdf.ValueTypeNames.Double3, Gf.Vec3d(2.0, 2.0, 2.0))(layer)
        order = layer.GetAttributeAtPath(f"{path}.xformOpOrder")
        if order is None:
            order = Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)
            order.default = Vt.TokenArray(["xformOp:scale"])
        else:
            order.default = Vt.TokenArray([*order.default, "xformOp:scale"])
    return edit


#: What the stage's reader can see of a derived prim beyond its content: the form a proxy mesh is written in, and the
#: lumen's purpose, visibility and extent. Each is an edit the hash of the points, counts and indices cannot see.
FORM_EDITS = {
    "proxy purpose": (PROXY, authored(PROXY, "purpose", Sdf.ValueTypeNames.Token, "default")),
    "proxy orientation": (PROXY, authored(PROXY, "orientation", Sdf.ValueTypeNames.Token, "leftHanded")),
    "proxy subdivision": (PROXY, authored(PROXY, "subdivisionScheme", Sdf.ValueTypeNames.Token, "catmullClark")),
    "proxy visibility": (PROXY, authored(PROXY, "visibility", Sdf.ValueTypeNames.Token, "invisible")),
    "proxy scale op": (PROXY, with_scale_op(PROXY)),
    "lumen purpose": (LUMEN, authored(LUMEN, "purpose", Sdf.ValueTypeNames.Token, "default")),
    "lumen visibility": (LUMEN, authored(LUMEN, "visibility", Sdf.ValueTypeNames.Token, "invisible")),
    "lumen extent": (LUMEN, authored(LUMEN, "extent", Sdf.ValueTypeNames.Float3Array,
                                     Vt.Vec3fArray([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)]))),
    "lumen scale op": (LUMEN, with_scale_op(LUMEN)),
}


@pytest.mark.parametrize("name", sorted(FORM_EDITS))
def test_control_an_edit_of_a_derived_prims_form_is_found_by_the_stages_own_check_and_by_the_load(saved_a, name):
    """The proxy's purpose, orientation, subdivision, visibility and a scale op of its own, and the lumen's purpose,
    visibility, extent and scale op, none of which the mesh's points, counts and indices carry: each is found on the
    prim by a reader with only the stage, and by the load's comparison, on every run (the planted edits are fixed). The
    stage as saved has no finding."""
    path, plant = FORM_EDITS[name]
    written = (saved_a / STAGE_FILE).read_text()
    assert stage_derived_findings(written) == []
    text = edited(written, plant)
    assert text != written
    found = stage_derived_findings(text)
    print(f"\n{name}: {[f.message for f in found]}")
    assert [(f.path, f.finding) for f in found] == [(path, "edited")]
    edits = stage_edits(text, written)
    assert not edits.outside and edits.derived == (path,)
