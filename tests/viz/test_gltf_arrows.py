"""A vector's arrow in the view is capped, in the glTF and the page alone, at the lesser of two lengths
(``ARROW_BODY_FACTOR`` times the largest extent of the body it is drawn at, read from that body's geometry in the stage,
and ``ARROW_CAP_FRACTION`` of the diagonal of the scene without its arrows); an arrow at no body has the scene's alone.

**The operating point is A0's.** The scene is A0's (the AR4, the 35 x 20 mm rotor magnet, FL-9 in the tube), an arm
0.7 m across around a body 7.47 mm long, and the arrow is drawn at 8.2992 m per tesla. The field at the body is
``A0_FIELD_T``, the value a short closed-loop run of A0 records (below), and 10 mT, the one the maintainer saw drawn
8 cm long. The scene's quarter is 0.177 m, so the scene's rule alone leaves both as they were: it is the body's rule
(2 x 7.47 mm = 14.9 mm) that acts, at 10 mT and not at A0's own field.

- **A0's own field**: the arrow is 8.2 mm, within its cap (14.9 mm), so the GLB is byte for byte the one with the caps
  lifted, and the notes say nothing.
- **10 mT**: the arrow is drawn at twice the helix's extent, in every sample, along the recording's direction; the notes
  name it by its full prim path, the length it is drawn at and the length it would be.
- **Controls**: with the caps lifted (``ARROW_BODY_FACTOR`` and ``ARROW_CAP_FRACTION`` infinite) the same recording draws
  the arrow at 83 mm, which is above the cap (so the test can fail); a zero field stays hidden.
- **The scene's rule**, on a stage written for it (a body, a far body that makes the scene large, arrows given lengths
  as time samples): an arrow at no body is capped at a quarter of the scene without arrows; two apparatus whose arrows
  have one name and one body are two capped arrows, each by its prim path.
- **Nothing recorded changes**: every file of the run folder has the digest it had when the recording was written, taken
  in the fixture before any view code runs; the control plants a file written on the first call only, and the digests
  see it.

``A0_FIELD_T`` is the flux density at the body that ``run_to`` of A0 (``tests/script_a0.py``'s ``a0()`` as
``examples/tutorial_a0.py`` closes it: the arm at its home pose, the rotor turning) recorded at its first two samples,
0.9895 and 0.9932 mT [measured, 2026-11-05, CPU, ``jax_disable_jit``, 2 steps of 0.5 ms]. The review's estimate from the
dipole formula alone, 0.55 mT, was 1.8 times lower than the run's number; why was not investigated.
The test is an acceptance test for how the arrow is drawn at that field, not a statement about the field's accuracy."""

import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from gltf_cases import accessor, glb_parts, node_by_usd, recorded
from pxr import Gf, Sdf, Usd, UsdGeom
from script_a0 import a0 as script_a0

import mime.viz._export as export_module
from mime.scene import StageSpec, ThingPrim, VectorPrim
from mime.scene._stage import stage_text
from mime.viz import RECORDING_FILE, RUN_STAGE, gltf, page
from mime.viz._gltf._xform import HIDDEN_SCALE

ARROW = "/World/rpm/magnetic/magnetic_torque_umr"
SCALE = f"{ARROW}.xformOp:scale"
A0_FIELD_T = 0.00099
HELIX_LENGTH_M = 7.47e-3
CAP_M = 2 * HELIX_LENGTH_M
PER_TESLA = 8.2992


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def a0():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return script_a0().build()


def digests(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


@dataclass(frozen=True)
class Run:
    """A run folder of A0 and the digests of every file in it, taken right after the recording was written."""

    recording: Path
    baseline: dict[str, str]

    @property
    def folder(self) -> Path:
        return self.recording.parent

    def unchanged(self) -> bool:
        return digests(self.folder) == self.baseline


def record(built, folder: Path, tesla: float) -> Run:
    """A0's recording over five samples: the field zero at the first and ``tesla`` along y at the others. The digests
    are taken here, before any view of it is made."""
    key = next(e for e in built.binding().entries if e.kind == "vector")

    def state_at(k, start):
        held = {node: dict(fields) for node, fields in start.items()}
        held[key.node][key.field] = np.zeros(3) if k == 0 else np.array([0.0, tesla, 0.0])
        return held

    recording = recorded(built, folder, 5, state_at)
    return Run(recording, digests(folder))


def run_fixture(name: str, tesla: float):
    @pytest.fixture(scope="module", name=name)
    def make(a0, tmp_path_factory):
        run = record(a0, tmp_path_factory.mktemp(name) / "run", tesla)
        yield run
        assert run.unchanged(), "a view of the run changed a file of its folder"

    return make


own = run_fixture("own", A0_FIELD_T)
ten_mt = run_fixture("ten_mt", 0.010)
none = run_fixture("none", 0.0)


def arrow_tracks(result, prim: str = ARROW) -> dict[str, np.ndarray]:
    document, binary = glb_parts(result.glb)
    index, _ = node_by_usd(document, prim)
    (animation,) = document["animations"]
    return {c["target"]["path"]: accessor(document, binary, animation["samplers"][c["sampler"]]["output"])
            for c in animation["channels"] if c["target"]["node"] == index}


def notes_of_capping(result) -> list[str]:
    return [line for line in result.lines() if "capped" in line]


@pytest.fixture
def lifted(monkeypatch):
    """Both caps gone: every arrow as its prim says."""
    monkeypatch.setattr(export_module, "ARROW_CAP_FRACTION", math.inf)
    monkeypatch.setattr(export_module, "ARROW_BODY_FACTOR", math.inf)


def test_the_arrow_at_a0s_own_field_is_within_its_cap_and_drawn_byte_for_byte_as_before(own, monkeypatch):
    held = gltf(own.recording)
    arrow = A0_FIELD_T * PER_TESLA
    assert arrow < 0.6 * CAP_M, "the arrow is well under its cap, so the cap has nothing to do"
    assert held.capped == {} and not notes_of_capping(held)
    monkeypatch.setattr(export_module, "ARROW_CAP_FRACTION", math.inf)
    monkeypatch.setattr(export_module, "ARROW_BODY_FACTOR", math.inf)
    free = gltf(own.recording)
    assert held.glb == free.glb and held.across_m == free.across_m
    assert arrow_tracks(held)["scale"][-1, 0] == pytest.approx(arrow, rel=1e-5)


def test_at_10_mT_the_arrow_is_drawn_at_twice_the_helixs_extent_and_the_notes_say_so(ten_mt, none):
    result = gltf(ten_mt.recording)
    (cut,) = result.capped.values()
    assert cut.rule == "body" and cut.drawn_m == pytest.approx(CAP_M, rel=1e-5)
    assert cut.would_be_m == pytest.approx(0.010 * PER_TESLA, rel=1e-9)
    assert cut.drawn_m < 0.25 * gltf(none.recording).across_m / 5, "the scene's quarter is far above it: the body's rule acts"
    scales = arrow_tracks(result)["scale"]
    assert np.all(scales[0] == np.float32(HIDDEN_SCALE)), "a zero field stays hidden"
    assert scales[1:].shape == (4, 3) and np.all(scales[1:] == np.float32(cut.drawn_m))
    (note,) = notes_of_capping(result)
    assert "rpm/magnetic/magnetic_torque_umr drawn at 14.9 mm, not 83 mm" in note
    assert "the stage and the recording hold its full length" in note
    assert note in page(result).replace("&#x27;", "'")


def test_control_with_the_caps_lifted_the_arrow_is_drawn_at_its_full_length(ten_mt, lifted):
    full = gltf(ten_mt.recording)
    scale = arrow_tracks(full)["scale"][-1, 0]
    assert scale == pytest.approx(0.010 * PER_TESLA, rel=1e-5)
    assert scale > 5 * CAP_M, "above the cap, so a test of the cap fails when the cap is gone"
    assert full.capped == {} and not notes_of_capping(full)


def test_the_capped_arrow_points_where_the_uncapped_one_does_and_does_not_widen_the_frame(ten_mt, monkeypatch):
    capped = gltf(ten_mt.recording)
    monkeypatch.setattr(export_module, "ARROW_CAP_FRACTION", math.inf)
    monkeypatch.setattr(export_module, "ARROW_BODY_FACTOR", math.inf)
    full = gltf(ten_mt.recording)
    assert np.array_equal(arrow_tracks(capped)["rotation"], arrow_tracks(full)["rotation"])
    assert capped.across_m <= full.across_m


def test_the_run_folder_keeps_every_digest_it_was_written_with_and_the_recording_the_full_length(ten_mt):
    assert RECORDING_FILE in "".join(ten_mt.baseline) and RUN_STAGE in ten_mt.baseline
    page(gltf(ten_mt.recording))
    assert ten_mt.unchanged()
    drawn = Sdf.Layer.FindOrOpen(str(ten_mt.recording)).QueryTimeSample(Sdf.Path(SCALE), 4)
    assert drawn[0] == pytest.approx(0.010 * PER_TESLA, rel=1e-9) and drawn[0] > 5 * CAP_M


def test_control_the_digests_see_a_file_written_on_the_first_call_only(a0, tmp_path, monkeypatch):
    """The reviewer's mutant: the view writes a file into the run folder if it is not there yet. The baseline is taken
    when the recording is written, before any view code runs, so the first call's write is a difference; a baseline
    taken after earlier calls would already hold the file and see nothing."""
    run = record(a0, tmp_path / "fresh", 0.010)
    real = export_module.arrows_of

    def writing(stage, purposes):
        planted = run.folder / "viewed.cache"
        if not planted.exists():
            planted.write_bytes(b"first call")
        return real(stage, purposes)

    monkeypatch.setattr(export_module, "arrows_of", writing)
    gltf(run.recording)
    assert not run.unchanged(), "the digests did not see a file written once"


def test_control_the_digests_would_show_a_changed_file(ten_mt, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / RECORDING_FILE).write_bytes(ten_mt.recording.read_bytes())
    before = digests(copy)
    (copy / RECORDING_FILE).write_bytes(ten_mt.recording.read_bytes() + b"\0")
    assert digests(copy) != before


# The scene's rule and the prim path as the key, on stages written for them: A0 has one vector prim, at one body.

NEAR_RADIUS_M, FAR_RADIUS_M = 2e-3, 50e-3
NEAR_CAP_M = 2 * 2 * NEAR_RADIUS_M


def stage_with_arrows(folder: Path, arrows: dict[str, float], *, at_body: bool = True) -> Path:
    """A stage of a body 4 mm across (``near``, at the origin) and one 0.1 m across 0.5 m away that makes the scene
    large, with a vector prim at each path of ``arrows`` (at ``near``, or 1 cm off it where ``at_body`` is false), drawn
    from its second time sample at the length given: the stage as a recording leaves it, written without one."""
    folder.mkdir(parents=True, exist_ok=True)
    spec = StageSpec(
        (0.0, 0.0, 1.0), 1.0, (),
        (ThingPrim("/World/near", "near", "body", translation=(0.0, 0.0, 0.0), extent=NEAR_RADIUS_M),
         ThingPrim("/World/far", "far", "body", translation=(0.5, 0.0, 0.0), extent=FAR_RADIUS_M)),
        vectors=tuple(VectorPrim(path, (0.0, 0.0, 0.0) if at_body else (0.0, 0.01, 0.0), "T", path, 1.0)
                      for path in arrows))
    path = folder / "arrows.usda"
    path.write_text(stage_text(spec), encoding="utf-8")
    stage = Usd.Stage.Open(str(path))
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(1.0)
    stage.SetTimeCodesPerSecond(1.0)
    for prim_path, length in arrows.items():
        prim = stage.GetPrimAtPath(prim_path)
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        scale = prim.GetAttribute("xformOp:scale")
        scale.Set(Gf.Vec3d(1e-9, 1e-9, 1e-9), 0.0)
        scale.Set(Gf.Vec3d(length, length, length), 1.0)
    stage.GetRootLayer().Save()
    return path


def test_an_arrow_at_no_body_is_capped_at_the_scenes_quarter_alone(tmp_path):
    result = gltf(stage_with_arrows(tmp_path / "long", {"/World/stray/field/b": 5.0}, at_body=False))
    bare = gltf(stage_with_arrows(tmp_path / "bare", {"/World/stray/field/b": 1e-6}, at_body=False))
    ((prim, cut),) = result.capped.items()
    assert prim == "/World/stray/field/b" and cut.rule == "scene" and cut.would_be_m == pytest.approx(5.0)
    assert cut.drawn_m == pytest.approx(0.25 * bare.across_m, rel=1e-3), "a quarter of the scene without the arrow"
    assert cut.drawn_m > 10 * NEAR_CAP_M, "the body's rule would have made it far shorter, and it is not applied here"


def test_an_arrow_at_a_body_is_capped_by_that_body_not_the_scene(tmp_path):
    ((_, cut),) = gltf(stage_with_arrows(tmp_path, {"/World/stand/field/b": 5.0})).capped.items()
    assert cut.rule == "body" and cut.drawn_m == pytest.approx(NEAR_CAP_M, rel=1e-5)


def test_two_apparatus_with_one_field_name_and_one_body_are_two_capped_arrows(tmp_path):
    lengths = {"/World/a/magnetic/torque_umr": 1.0, "/World/b/magnetic/torque_umr": 2.0}
    result = gltf(stage_with_arrows(tmp_path, lengths))
    assert sorted(result.capped) == sorted(lengths), "each by its prim path, not by the name they share"
    assert {p: c.would_be_m for p, c in result.capped.items()} == pytest.approx(lengths)
    assert all(c.drawn_m == pytest.approx(NEAR_CAP_M, rel=1e-5) for c in result.capped.values())
    (note,) = notes_of_capping(result)
    assert note.startswith("2 arrows are drawn capped")
    assert "a/magnetic/torque_umr drawn at 8 mm, not 1 m" in note and "b/magnetic/torque_umr drawn at 8 mm, not 2 m" in note


def test_control_the_leaf_name_as_the_key_would_have_counted_one(tmp_path):
    """The old key was the prim's last path component: the same name for the two arrows of the test above."""
    result = gltf(stage_with_arrows(tmp_path, {"/World/a/magnetic/torque_umr": 1.0, "/World/b/magnetic/torque_umr": 2.0}))
    assert len({prim.rsplit("/", 1)[-1] for prim in result.capped}) == 1 and len(result.capped) == 2
