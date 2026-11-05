"""A vector's arrow in the view is drawn at most a quarter of the scene's diagonal long (``ARROW_CAP_FRACTION``), in the
glTF and the page alone:

- **Capped**: A0's recording with the field at 0.1 T, which its stage draws at 8.3 m per tesla, 0.83 m beside a scene
  0.7 m across: the arrow's scale in the GLB is the cap, in every sample that is longer, and the page's notes say so,
  naming the length it would have; the directions are the recording's, untouched.
- **Framing**: the viewer frames what the GLB holds, so a capped arrow does not set the frame (the diagonal is the scene
  without its arrow, plus at most the cap).
- **Nothing recorded changes**: every file of the run folder has the digest it had before the view was made, and the
  recording's arrow scale still reads the full length through usd-core.
- **Controls**: with the cap lifted (``ARROW_CAP_FRACTION`` infinite) the same recording's arrow is drawn at its full
  length, which is above the cap (so the test can fail), and the notes say nothing; a field of 1 mT, whose arrow is
  within the cap, gives the same GLB with or without the cap and no note.

The operating point is A0's stage (an arm 0.7 m across, its bodies at the origin), not a millimetre scene: the cap is
a fraction of what the scene holds, so the test needs only that the arrow is longer than a quarter of it."""

import hashlib
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from chain_cases import kinematic_rpm
from gltf_cases import accessor, glb_parts, node_by_usd, recorded
from pxr import Sdf

import mime.viz._export as export_module
from mime.viz import RECORDING_FILE, RUN_STAGE, gltf, page
from mime.viz._gltf._xform import HIDDEN_SCALE

ARROW = "/World/rpm/magnetic/magnetic_torque_umr"
SCALE = f"{ARROW}.xformOp:scale"


@pytest.fixture(scope="module")
def a0():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return kinematic_rpm().build()


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


def run_with_field(a0, folder: Path, tesla: float) -> Path:
    """A0's recording, five samples, the field zero at the first and ``tesla * k`` along y at sample ``k``."""
    key = next(e for e in a0.binding().entries if e.kind == "vector")

    def state_at(k, start):
        held = {node: dict(fields) for node, fields in start.items()}
        held[key.node][key.field] = np.array([0.0, 0.0, 0.0]) if k == 0 else np.array([0.0, tesla * k, 0.0])
        return held

    return recorded(a0, folder, 5, state_at)


def arrow_tracks(result) -> dict[str, np.ndarray]:
    document, binary = glb_parts(result.glb)
    index, _ = node_by_usd(document, ARROW)
    (animation,) = document["animations"]
    return {c["target"]["path"]: accessor(document, binary, animation["samplers"][c["sampler"]]["output"])
            for c in animation["channels"] if c["target"]["node"] == index}


def digests(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def notes_of_capping(result) -> list[str]:
    return [line for line in result.lines() if "capped" in line]


@pytest.fixture(scope="module")
def strong(a0, tmp_path_factory):
    return run_with_field(a0, tmp_path_factory.mktemp("strong") / "run", 0.025)  # 0.1 T at the last sample


def test_a_longer_arrow_is_drawn_at_the_cap_and_the_notes_say_so(strong):
    result = gltf(strong)
    scales = arrow_tracks(result)["scale"]
    cap, would_be = result.arrow_cap_m, result.capped["magnetic_torque_umr"]
    assert would_be > 1.5 * cap > 0.0, "the arrow is longer than the cap, so the cap acts"
    assert scales[-1, 0] == pytest.approx(cap, rel=1e-6), "drawn at the cap"
    assert np.all(scales[0] == np.float32(HIDDEN_SCALE)), "a zero field stays hidden"
    assert np.all(scales <= np.float32(cap) * (1 + 1e-6))
    (note,) = notes_of_capping(result)
    assert "magnetic_torque_umr" in note and "the stage and the recording hold its full length" in note
    assert note in page(result).replace("&#x27;", "'")


def test_the_cap_is_a_quarter_of_the_scene_without_the_arrow(a0, strong, tmp_path):
    """The oracle is the same scene recorded with a zero field, whose arrow is a point at the body: its diagonal is
    the scene's without any arrow, found by no code that leaves a prim out."""
    bare = gltf(run_with_field(a0, tmp_path / "bare", 0.0))
    assert not notes_of_capping(bare)
    assert gltf(strong).arrow_cap_m == pytest.approx(export_module.ARROW_CAP_FRACTION * bare.across_m, rel=1e-6)


def test_a_capped_arrow_keeps_its_direction_and_does_not_set_the_frame(strong, monkeypatch):
    capped = gltf(strong)
    monkeypatch.setattr(export_module, "ARROW_CAP_FRACTION", math.inf)
    full = gltf(strong)
    assert np.array_equal(arrow_tracks(capped)["rotation"], arrow_tracks(full)["rotation"])
    assert capped.across_m < full.across_m, "the frame is the scene and a capped arrow, not the arrow at its length"


def test_control_with_the_cap_lifted_the_arrow_is_drawn_at_its_full_length(strong, monkeypatch):
    capped = gltf(strong)
    monkeypatch.setattr(export_module, "ARROW_CAP_FRACTION", math.inf)
    full = gltf(strong)
    scale = arrow_tracks(full)["scale"][-1, 0]
    assert scale > 1.5 * capped.arrow_cap_m, "a test of the cap fails when the cap is gone"
    assert scale == pytest.approx(capped.capped["magnetic_torque_umr"], rel=1e-5)
    assert full.capped == {} and not notes_of_capping(full)


def test_an_arrow_within_the_cap_is_drawn_as_before_and_unmentioned(a0, tmp_path, monkeypatch):
    weak = run_with_field(a0, tmp_path / "weak", 0.001 / 4)
    held = gltf(weak)
    monkeypatch.setattr(export_module, "ARROW_CAP_FRACTION", math.inf)
    lifted = gltf(weak)
    assert held.capped == {} and not notes_of_capping(held)
    assert held.glb == lifted.glb and held.across_m == lifted.across_m


def test_no_file_of_the_run_changes_and_the_recording_keeps_the_full_length(a0, strong):
    folder = strong.parent
    before = digests(folder)
    assert RECORDING_FILE in "".join(before) and RUN_STAGE in before
    assert before[RUN_STAGE] == hashlib.sha256(a0.stage_text.encode("utf-8")).hexdigest(), "the stage as built"
    result = gltf(strong)
    page(result)
    assert digests(folder) == before
    layer = Sdf.Layer.FindOrOpen(str(strong))
    drawn = layer.QueryTimeSample(Sdf.Path(SCALE), 4)
    assert drawn[0] == pytest.approx(result.capped["magnetic_torque_umr"], rel=1e-9) and drawn[0] > result.arrow_cap_m


def test_control_the_digests_would_show_a_changed_file(strong, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / RECORDING_FILE).write_bytes(strong.read_bytes())
    before = digests(copy)
    (copy / RECORDING_FILE).write_bytes(strong.read_bytes() + b"\0")
    assert digests(copy) != before
