"""A description's visuals in the stage: the AR4's eighteen meshes named on their frames by the visual set's name and
each file's sha256, never their bytes, with bones from its frames drawn in their place; the asset cache's overlay
drawing the baked meshes (a synthetic stand-in, ``visual_cases.planted``: CI never fetches the AR4) and hiding the
bones; and, with no bake in the cache, the stage drawing the bones and saying so. The recipe itself: every visual mesh
of MIME's AR4 description is one of its files, pinned to one upstream commit, with its credit, licence and tier."""

import re
import warnings
from pathlib import Path

import pytest
from chain_cases import AR4, kinematic_rpm
from pxr import Usd, UsdGeom
from visual_cases import planted

from mime.core.multibody import read_urdf
from mime.scene import VISUAL_ASSET, VISUAL_ROLE, VISUAL_SHA256, mesh_path, mesh_source, visual_sets
from mime.viz import overlay, view


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MIME_ASSETS_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture(scope="module")
def stage_file(tmp_path_factory) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = kinematic_rpm().build()
    path = tmp_path_factory.mktemp("stage") / "scene.usda"
    path.write_text(built.stage_text)
    return path


def ar4_meshes() -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [s.mesh or "" for s in read_urdf(AR4).shapes if s.role == "visual" and s.kind == "mesh"]


def test_every_visual_mesh_of_the_ar4_is_a_file_of_its_pinned_set():
    held = visual_sets()["ar4_mk5"]
    assert sorted(mesh_path(m) for m in ar4_meshes()) == sorted(f.path for f in held.files) and len(held.files) == 18
    commit = held.upstream["commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    for source in held.files:
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256) and re.fullmatch(r"[0-9a-f]{40}", source.git_blob)
        assert source.location == (f"https://raw.githubusercontent.com/Annin-Robotics/ar4_ros_driver/{commit}/"
                                   f"annin_ar4_description/{source.path}")
    assert (held.licence, held.tier, held.role) == ("LicenseRef-Annin-Robotics-NonCommercial", "no-redistribution",
                                                    "presentation")
    assert "Annin Robotics" in held.credit and "non-commercial" in held.notice.lower()
    assert mesh_source("package://annin_ar4_description/meshes/ar4_mk5/Link_1_Aluminum.STL") is not None
    assert mesh_source("meshes/elsewhere/Link_1_Aluminum.STL") is None


def test_the_stage_names_each_mesh_by_its_set_and_sha256_and_holds_none_of_its_bytes(stage_file):
    stage = Usd.Stage.Open(str(stage_file))
    named = [(str(prim.GetAttribute(VISUAL_ASSET).Get()), str(prim.GetAttribute(VISUAL_SHA256).Get()))
             for prim in stage.Traverse() if prim.GetAttribute(VISUAL_ASSET)]
    held = visual_sets()["ar4_mk5"]
    assert sorted(named) == sorted(("ar4_mk5", f.sha256) for f in held.files)
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert all("/visuals/" not in str(prim.GetPath()) for prim in meshes)
    bones = [prim for prim in stage.Traverse() if prim.GetAttribute(VISUAL_ROLE) and
             prim.GetAttribute(VISUAL_ROLE).Get() == "frames"]
    assert len(bones) >= 6 and all(prim.IsA(UsdGeom.Cylinder) for prim in bones)


def test_with_nothing_baked_the_stage_draws_its_bones_and_says_so(cache, stage_file):
    found = overlay(stage_file)
    assert found.path is None and found.drawn == 0 and found.missing == {"ar4_mk5": 18}
    assert "drawn as simple shapes" in found.text() and "mime visuals fetch ar4_mk5" in found.text()
    assert view(stage_file)[0] == stage_file


def test_the_overlay_draws_the_baked_meshes_and_hides_the_bones(cache, stage_file):
    planted()
    found = overlay(stage_file)
    assert found.drawn == 18 and found.missing == {} and found.path is not None
    assert found.path.is_relative_to(cache) and "Annin Robotics" in found.text()
    path, _ = view(stage_file)
    assert path.is_relative_to(cache)
    stage = Usd.Stage.Open(str(path))
    drawn = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and "/visuals/" in str(prim.GetPath())]
    assert len(drawn) == 18 and all(len(UsdGeom.Mesh(p).GetPointsAttr().Get()) > 0 for p in drawn)
    hidden = [prim for prim in stage.Traverse() if prim.GetAttribute(VISUAL_ROLE) and
              prim.GetAttribute(VISUAL_ROLE).Get() == "frames"]
    assert hidden and all(UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible for p in hidden)
    # The control: the stage opened alone draws no mesh of the set, whatever the cache holds.
    alone = Usd.Stage.Open(str(stage_file))
    assert not [p for p in alone.Traverse() if p.IsA(UsdGeom.Mesh) and "/visuals/" in str(p.GetPath())]


def test_a_description_with_no_visuals_draws_bones_from_its_frames():
    from chain_cases import C_ARM

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reading = read_urdf(C_ARM)
    from mime.dynamics.apparatus._arm import segment_visuals

    import numpy as np

    drawn = segment_visuals(reading, np.zeros((len(reading.tree.frame_names), 7)) + [0.3, 0, 0, 1, 0, 0, 0])
    assert drawn and {v.role for v in drawn} == {"frames"} and all(v.kind == "bone" for v in drawn)


def test_a_set_it_does_not_know_and_a_folder_with_no_stage_are_refused(tmp_path):
    from mime.build import ArgumentRefused
    from mime.viz import fetch

    with pytest.raises(ArgumentRefused, match="visuals.unknown-set"):
        fetch("ar5_mk9", say=lambda line: None)
    with pytest.raises(ArgumentRefused, match="visuals.no-stage"):
        view(tmp_path)


def test_a_source_that_is_no_mesh_is_refused_by_name(cache, monkeypatch):
    """A file placed in the cache under its sha256 (so nothing is downloaded) whose bytes are no STL file."""
    import hashlib

    from mime.build import BuildRefused
    from mime.scene import VisualFile, VisualSet
    from mime.viz import _visuals, fetch

    data = b"solid nothing\nendsolid nothing\n"
    digest = hashlib.sha256(data).hexdigest()
    cache.mkdir(parents=True)
    (cache / digest).write_bytes(data)
    planted_set = VisualSet("planted", "a planted set", {}, "CC0-1.0", "https://example.org", "open", "presentation",
                            "nobody", "none", (), "never", (VisualFile("meshes/x.STL", digest, len(data), "0" * 40,
                                                                       "https://example.org/x.STL"),))
    monkeypatch.setattr(_visuals, "visual_sets", lambda: {"planted": planted_set})
    monkeypatch.setenv("MIME_OFFLINE", "1")
    with pytest.raises(BuildRefused, match="visuals.convert"):
        fetch("planted", say=lambda line: None)
