"""Script A's stage as a GLB with no recording: the screw's and the tube's proxy meshes (USD's ``proxy`` purpose,
which a view draws) and the lumen's cylinder, tessellated, each translucent part blended and two-sided as its
``displayOpacity`` says; no animation. The control: the same stage converted drawing the default purpose alone has
neither proxy."""

import warnings

import pytest
from gltf_cases import glb_parts, node_by_usd
from script_a import script_a

from mime.viz import gltf
from mime.viz._gltf import Settings, convert
from mime.scene._pxr import Usd


@pytest.fixture(scope="module")
def stage_file(tmp_path_factory):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = script_a().build()
    path = tmp_path_factory.mktemp("script_a") / "scene.usda"
    path.write_text(built.stage_text)
    return path


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


def material_of(document, node):
    return document["materials"][document["meshes"][node["mesh"]]["primitives"][0]["material"]]


def test_the_proxies_and_the_lumen_are_drawn_translucent_where_the_stage_says(stage_file):
    result = gltf(stage_file)
    document, _ = glb_parts(result.glb)
    assert "animations" not in document and result.samples == 0
    for path, opacity in (("/World/tube/proxy", 0.3), ("/World/tube/lumen", 0.15), ("/World/umr/shell/proxy", 1.0)):
        _, node = node_by_usd(document, path)
        material = material_of(document, node)
        assert material["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(opacity), path
        assert (material.get("alphaMode"), material.get("doubleSided")) == \
            (("BLEND", True) if opacity < 1.0 else (None, None)), path
    assert result.skipped == {} and result.triangles > 2000


def test_control_the_default_purpose_alone_draws_no_proxy(stage_file):
    converted = convert(Usd.Stage.Open(str(stage_file)), Settings(purposes=("default",)))
    document, _ = glb_parts(converted.glb)
    paths = {node.get("extras", {}).get("usd") for node in document["nodes"]}
    assert "/World/tube/lumen" in paths and "/World/tube/proxy" not in paths and "/World/umr/shell/proxy" not in paths
