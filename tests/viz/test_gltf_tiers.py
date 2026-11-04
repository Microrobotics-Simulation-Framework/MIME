"""A shareable view never holds a mesh whose terms do not let it travel: A0's stage, its eighteen AR4
meshes planted in a temporary asset cache (``visual_cases.planted``: CI never fetches the AR4), converted by
``mime.viz.gltf``.

- **Not stated, and referenced**: with no record stating the meshes' tier, and with every record stating
  ``referenced`` (the AR4's own), the GLB holds no mesh under the eighteen visual prims and none of the planted
  meshes' bytes, the same nodes as with nothing baked, and says ``withheld == {"ar4_mk5": 18}``: the bones are drawn.
- **Open (the control, which a guard that lets everything through, or keeps everything out, fails)**: the same plant
  with its records stating ``open``: the GLB holds the eighteen meshes, their bytes, and withholds nothing.
- **The second guard alone**: the view layer written for a USD viewer (``mime visuals view``, every baked mesh drawn)
  named as the target is refused (``visuals.not-shareable``), the referenced plant and the unstated one alike; the same
  layer over the open plant converts.

And A0's recording, written from samples: the arm's frames and the body animate, and the field's arrow, zero at the
first sample, takes a scale of nothing there (glTF has no visibility over time)."""

import warnings

import numpy as np
import pytest
from chain_cases import kinematic_rpm
from gltf_cases import accessor, glb_parts, node_by_usd, recorded
from visual_cases import holds_any, mesh_bytes, planted

from mime.build import ArgumentRefused
from mime.viz import gltf, view
from mime.viz._gltf._xform import HIDDEN_SCALE

SET = "ar4_mk5"


@pytest.fixture(scope="module")
def a0():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kinematic_rpm().build()


@pytest.fixture(scope="module")
def stage_file(a0, tmp_path_factory):
    path = tmp_path_factory.mktemp("a0") / "scene.usda"
    path.write_text(a0.stage_text)
    return path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def visual_meshes(document) -> list[str]:
    """The USD paths of the GLB's mesh nodes below a prim that names a visual mesh."""
    return [node["extras"]["usd"] for node in document["nodes"] if "mesh" in node and
            node.get("extras", {}).get("usd", "").endswith("/mesh")]


@pytest.mark.parametrize("tier", [None, "referenced"])
def test_a_mesh_whose_tier_is_not_shareable_is_left_out_and_said(cache, stage_file, tier):
    bare = glb_parts(gltf(stage_file).glb)[0]
    layers = planted(tier=tier)
    result = gltf(stage_file)
    document, _ = glb_parts(result.glb)
    assert visual_meshes(document) == [] and result.withheld == {SET: 18} and result.drawn == 0
    assert [n.get("extras", {}).get("usd") for n in document["nodes"]] == \
        [n.get("extras", {}).get("usd") for n in bare["nodes"]]
    assert not holds_any(result.glb, mesh_bytes(layers))
    assert any(line.startswith(f"18 meshes of {SET} are not in this view") for line in result.lines())
    assert document["asset"]["extras"]["mime"]["withheld"] == {SET: 18}


def test_control_the_same_plant_stated_open_is_drawn(cache, stage_file):
    layers = planted(tier="open")
    result = gltf(stage_file)
    document, _ = glb_parts(result.glb)
    assert len(visual_meshes(document)) == 18 and result.withheld == {} and result.drawn == 18
    assert holds_any(result.glb, mesh_bytes(layers))
    assert "Annin Robotics" in document["asset"]["copyright"]


@pytest.mark.parametrize("tier", [None, "referenced"])
def test_the_second_guard_refuses_a_view_layer_that_draws_them(cache, stage_file, tier):
    planted(tier=tier)
    path, found = view(stage_file)
    assert found.drawn == 18, "the USD viewer's layer draws every baked mesh"
    with pytest.raises(ArgumentRefused, match="visuals.not-shareable"):
        gltf(path)


def test_control_the_second_guard_passes_the_same_layer_over_an_open_plant(cache, stage_file):
    planted(tier="open")
    path, _ = view(stage_file)
    assert len(visual_meshes(glb_parts(gltf(path).glb)[0])) == 18


@pytest.fixture(scope="module")
def a0_run(a0, tmp_path_factory):
    folder = tmp_path_factory.mktemp("a0run") / "run"
    field_key = next(e for e in a0.binding().entries if e.kind == "vector")

    def state_at(k, start):
        held = {node: dict(fields) for node, fields in start.items()}
        poses = np.array(held["rpm.arm"]["frame_poses"], dtype=np.float64, copy=True)
        turn = 0.1 * k
        poses[5, 3:7] = [np.cos(turn / 2), 0.0, 0.0, np.sin(turn / 2)]
        held["rpm.arm"]["frame_poses"] = poses
        body = np.array(held["umr"]["pose"], dtype=np.float64, copy=True)
        body[2] += 1e-4 * k
        held["umr"]["pose"] = body
        held[field_key.node][field_key.field] = np.array([0.0, 0.0, 0.0]) if k == 0 else np.array([0.0, 1e-3 * k, 0.0])
        return held

    return recorded(a0, folder, 5, state_at)


def test_a0s_recording_animates_the_arms_frames_the_body_and_the_fields_arrow(cache, a0_run):
    document, binary = glb_parts(gltf(a0_run).glb)
    (animation,) = document["animations"]
    animated = {(c["target"]["node"], c["target"]["path"]): animation["samplers"][c["sampler"]]
                for c in animation["channels"]}
    link, _ = node_by_usd(document, "/World/rpm/arm/link_1")
    body, _ = node_by_usd(document, "/World/umr")
    arrow, _ = node_by_usd(document, "/World/rpm/magnetic/magnetic_torque_umr")
    assert (link, "rotation") in animated and (body, "translation") in animated
    scales = accessor(document, binary, animated[(arrow, "scale")]["output"])
    assert np.all(scales[0] == np.float32(HIDDEN_SCALE)) and np.all(scales[1:] > HIDDEN_SCALE)

