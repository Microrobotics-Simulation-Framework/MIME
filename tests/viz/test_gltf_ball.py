"""The settling ball's recording as a GLB (``mime.viz.gltf``), its recording written from chosen samples
(``gltf_cases.recorded``): one mesh for the ball's sphere, one animation of its translation and rotation over the
recording's eleven times, slowed to about eight seconds, its last translation the recorded one to float32, its rotations
of unit length and continuous in sign though the recording plants antipodal pairs, under a root node that turns +Z
to glTF's +Y and scales metres to a glTF unit chosen from the scene's size. The control: with the sign left as
recorded, the same file holds the planted flips."""

import math

import numpy as np
import pytest
from gltf_cases import BALL_RADIUS, accessor, ball, glb_parts, node_by_usd, recorded, with_pose

import mime.viz._gltf._xform as xform
from mime.viz import gltf
from mime.viz._primitives import SEGMENTS

SAMPLES, FALL = 11, 4.2e-6


def pose(k: int) -> np.ndarray:
    """The ball at sample ``k``: fallen ``k`` tenths of FALL and turned 0.3 rad a sample about x, every other
    quaternion written with its sign flipped (the same turn)."""
    half = 0.15 * k
    sign = -1.0 if k % 2 else 1.0
    return np.array([0.0, 0.0, -FALL * k / (SAMPLES - 1),
                     sign * math.cos(half), sign * math.sin(half), 0.0, 0.0])


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ball") / "run"
    recorded(ball().build(), folder, SAMPLES, lambda k, start: with_pose(start, "ball", pose(k)), every=3)
    return folder


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


def test_the_ball_is_one_mesh_moved_by_one_animation(run):
    result = gltf(run)
    document, binary = glb_parts(result.glb)
    meshes = [node for node in document["nodes"] if "mesh" in node and node["name"] != "frame"]
    assert [node["extras"]["usd"] for node in meshes] == ["/World/ball/steel"]
    (animation,) = document["animations"]
    ball_node, _ = node_by_usd(document, "/World/ball")
    assert sorted(c["target"]["path"] for c in animation["channels"]) == ["rotation", "translation"]
    assert {c["target"]["node"] for c in animation["channels"]} == {ball_node}
    shape = document["meshes"][meshes[0]["mesh"]]["primitives"][0]
    assert document["accessors"][shape["indices"]]["count"] == 3 * 2 * SEGMENTS * (SEGMENTS // 2 - 1)
    points = accessor(document, binary, shape["attributes"]["POSITION"])
    assert np.linalg.norm(points, axis=1).max() == pytest.approx(BALL_RADIUS, rel=1e-6)


def test_its_times_are_the_recordings_slowed_and_its_last_translation_the_recorded_one(run):
    result = gltf(run)
    document, binary = glb_parts(result.glb)
    (animation,) = document["animations"]
    sampler = {c["target"]["path"]: animation["samplers"][c["sampler"]] for c in animation["channels"]}
    times = accessor(document, binary, sampler["translation"]["input"])
    simulated = (SAMPLES - 1) * 3 * 1e-4
    assert result.simulated_s == pytest.approx(simulated) and result.samples == SAMPLES
    assert result.slow == float(f"{8.0 / simulated:.2g}")
    assert np.allclose(times, np.arange(SAMPLES) * 3e-4 * result.slow, rtol=1e-6)
    moved = accessor(document, binary, sampler["translation"]["output"])
    assert moved[-1] == pytest.approx(np.float32([0.0, 0.0, -FALL]), abs=float(np.spacing(np.float32(FALL))))
    extras = document["asset"]["extras"]["mime"]
    assert extras["slow"] == result.slow and extras["unit_m"] == result.unit_m
    root = document["nodes"][0]
    assert root["rotation"] == pytest.approx([-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
    assert root["scale"] == pytest.approx([1.0 / result.unit_m] * 3) and 1.0 <= result.across_m / result.unit_m < 10


def test_its_rotations_are_unit_and_continuous_where_the_recording_flips_their_sign(run):
    document, binary = glb_parts(gltf(run).glb)
    (animation,) = document["animations"]
    sampler = {c["target"]["path"]: animation["samplers"][c["sampler"]] for c in animation["channels"]}
    turns = accessor(document, binary, sampler["rotation"]["output"]).astype(np.float64)
    assert np.allclose(np.linalg.norm(turns, axis=1), 1.0, atol=1e-6)
    assert np.all(np.einsum("ij,ij->i", turns[1:], turns[:-1]) > 0.0)
    expected = np.array([[math.sin(0.15 * k), 0.0, 0.0, math.cos(0.15 * k)] for k in range(SAMPLES)])
    assert np.allclose(turns, expected, atol=1e-6)


def test_control_with_the_sign_left_as_recorded_the_flips_are_in_the_file(run, monkeypatch):
    monkeypatch.setattr(xform, "continuous", lambda rows: rows / np.linalg.norm(rows, axis=1, keepdims=True))
    document, binary = glb_parts(gltf(run).glb)
    (animation,) = document["animations"]
    sampler = {c["target"]["path"]: animation["samplers"][c["sampler"]] for c in animation["channels"]}
    turns = accessor(document, binary, sampler["rotation"]["output"]).astype(np.float64)
    assert np.sum(np.einsum("ij,ij->i", turns[1:], turns[:-1]) < 0.0) == SAMPLES - 1


def test_the_frame_holds_the_whole_fall(run):
    document, binary = glb_parts(gltf(run).glb)
    _, frame = next((i, n) for i, n in enumerate(document["nodes"]) if n["name"] == "frame")
    corners = accessor(document, binary, document["meshes"][frame["mesh"]]["primitives"][0]["attributes"]["POSITION"])
    # The bounds are USD's: each sample's box of the sphere's extent, turned with it, so a turned ball's box reaches
    # up to sqrt(2) of its radius from its centre; the frame holds the whole fall within that.
    low, high = corners.min(axis=0), corners.max(axis=0)
    assert -FALL - math.sqrt(2) * BALL_RADIUS * 1.001 <= low[2] <= -FALL - BALL_RADIUS * 0.999
    assert BALL_RADIUS * 0.999 <= high[2] <= math.sqrt(2) * BALL_RADIUS * 1.001
    assert np.allclose(low[:2], -high[:2], rtol=1e-6) and high[0] == pytest.approx(BALL_RADIUS, rel=1e-6)
