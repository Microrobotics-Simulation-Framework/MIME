"""The static mesh in the camera's scene: what the build notes say of it (its triangles and look, a drawing's facets and
sagitta, a decimation's distance in pixels at this camera), a body inside a closed attenuating mesh taking it as its
medium, the offline chords (``mime.sensing.mesh_chords``), and a scene with no mesh writing what it wrote before.

**Controls** (``CAMERA_PLANT``, code patches with no draw):

- ``no-medium``: the mesh a body lies in not found (the displaced-medium rule dropped): the bead's pixel is off by the
  medium's absorption along the bead's chord;
- ``chords-unsigned``: every crossing's ``t`` summed without its sign: the chords are off by twice the entry's ``t``."""

from __future__ import annotations

import math

import jax
import numpy as np
import pytest
from in_loop_cases import DETECTOR, LOOK_ACROSS, ballistic, planted
from mesh_cases import VESSEL, box, rotation, tube, vessel_layer, with_vessel
from test_mesh_swap import tube_bounds

import mime.sensing._mesh as mesh_module
from mime import Structure
from mime.scene import Box, Ellipsoid
from mime.sensing import CameraPose, mesh_chords, ray_table
from mime.sensing import _scene as scene
from mime.sensing._mesh import KEYS, Frozen
from mime.units import ms, um

if planted("no-medium"):
    scene._inside = lambda roster, subjects, meshes: {}  # type: ignore[assignment]

if planted("chords-unsigned"):
    original_crossings = mesh_module._crossings  # noqa: SLF001

    def unsigned(*args):
        crossed, t, _ = original_crossings(*args)
        return crossed, t, jax.numpy.ones_like(crossed)

    mesh_module._crossings = unsigned  # noqa: SLF001

ALONG_X = rotation((2 ** -0.5, 0.0, 2 ** -0.5, 0.0))


def text_of(exp) -> str:
    return exp.build().report.text()


def test_the_build_notes_name_each_mesh_with_its_triangles_look_and_sub_rays():
    points, faces = tube(60e-6, 70e-6, 480e-6, 32)
    text = text_of(with_vessel(ballistic(hide=("tube",), absorbers=((VESSEL, 3000.0),)), points @ ALONG_X.T, faces))
    assert f"traced static meshes (a bounding volume hierarchy, once a run, 4 x 4 sub-rays a pixel): {VESSEL} " \
           f"({len(faces)} triangles, attenuating)" in text
    assert f"decimation distance not recorded for: {VESSEL}" in text
    assert "made once a run, and again at each study point that varies the detector's numbers" in text


def test_a_meshs_constants_are_not_arrays_the_graph_compares_byte_for_byte_at_every_chunk():
    """MADDENING takes an ``ndarray`` among a node's constants for one that can change in place and compares its bytes
    at each chunk of a run: a 100 000-triangle mesh's megabytes became 8 ms a chunk (forty times the steps' own time)
    before the camera held its triangles as :class:`Frozen`, which the derived graph records all the same."""
    points, faces = tube(60e-6, 70e-6, 480e-6, 32)
    node = with_vessel(ballistic(hide=("tube",), absorbers=((VESSEL, 3000.0),)), points @ ALONG_X.T,
                       faces).build().graph.get_node("camera:")
    held = node.constants()
    assert isinstance(held["mesh_triangles"], Frozen) and isinstance(held["mesh_owners"], Frozen)
    assert not [key for key, value in node.params.items() if isinstance(value, np.ndarray)]
    assert np.asarray(held["mesh_triangles"]).shape == (len(faces), 3, 3)


def test_a_recorded_decimation_is_given_in_pixels_at_this_camera():
    points, faces = tube(60e-6, 70e-6, 480e-6, 32)
    exp = ballistic(hide=("tube",), absorbers=((VESSEL, 3000.0),))
    exp._settings.scene.reference(mesh_module_asset(points @ ALONG_X.T, faces, 2e-6), at=VESSEL)  # noqa: SLF001
    text = text_of(exp)
    # 2 um at the mesh's nearest corner, 930 um from the camera at 750 px: 1.61 px
    assert f"decimation distance at this camera: {VESSEL} 1.61 px (2e-06 m)" in text
    assert "decimation distance not recorded" not in text


def mesh_module_asset(points, faces, distance):
    from mime.scene import asset

    return asset(vessel_layer(points, faces, decimation=distance))


def test_write_mesh_records_a_decimation_only_where_given(tmp_path):
    from mime.viz import Mesh, write_mesh

    points, faces = box((1e-3, 1e-3, 1e-3))
    plain = write_mesh(tmp_path / "plain.usda", Mesh(points, faces), "Vessel").read_bytes()
    again = write_mesh(tmp_path / "again.usda", Mesh(points, faces), "Vessel", decimation_distance=None).read_bytes()
    given = write_mesh(tmp_path / "given.usda", Mesh(points, faces), "Vessel", decimation_distance=3e-5).read_text()
    assert plain == again and b"customData" not in plain
    # USD keeps a namespaced key as a dictionary of dictionaries: mime, then decimation_distance_m
    assert "dictionary mime = {" in given and "double decimation_distance_m = 0.00003" in given


def test_structures_of_drawn_kinds_are_imaged_by_their_drawings_with_their_sagitta():
    exp = ballistic(hide=("tube",), absorbers=())
    exp.add_structure(Structure("block", geometry=Box(size=um((30.0, 30.0, 30.0)))))
    exp.add_structure(Structure("egg", geometry=Ellipsoid(semi_axes=um((20.0, 15.0, 10.0)))))
    built = exp.build()
    text = built.report.text()
    assert "structures imaged by their drawing: block (box, exact), egg (ellipsoid, " in text
    sagitta = 20e-6 * (1.0 - math.cos(math.sqrt(2.0) * math.pi / 48))
    assert f"48 facets a turn, sagitta about {sagitta:.3g} m" in text
    assert "appearance defaulted for: ball, block, egg" in text
    built.run(ms(3))
    image = np.asarray(built.current_state()["camera:"]["expected"])
    assert image[60, 80] == 0.0  # the opaque block and egg at the origin, in front of the ball's path


def bead_in_a_box(medium: float) -> tuple[float, float]:
    """The bead's centre pixel inside a closed box of ``medium`` per metre, and its truth: the box's absorption along
    the box's chord less the bead's, and the bead's own along its chord (the displaced-medium rule)."""
    points, faces = box((150e-6, 150e-6, 150e-6))
    exp = with_vessel(ballistic(hide=("tube",), absorbers=((VESSEL, medium), ("ball", 5000.0)),
                                detector=DETECTOR), points, faces)
    built = exp.build()
    text = built.report.text()
    assert planted("no-medium") or f"ball in {VESSEL}" in text
    built.run(ms(1))
    image = np.asarray(built.current_state()["camera:"]["expected"], dtype=np.float64)
    pixel = image[60, 80 - int(round(50e-6 * DETECTOR.fx / 1e-3))]
    truth = 1000.0 * math.exp(-medium * (300e-6 - 40e-6) - 5000.0 * 40e-6)
    return float(pixel), truth


def test_a_body_inside_a_closed_attenuating_mesh_displaces_its_medium(float64):
    pixel, truth = bead_in_a_box(2000.0)
    # the pixel's ray through the bead's centre: its chord 40 um, the box's 300 um; within the box edge's pixel share
    assert pixel == pytest.approx(truth, rel=2e-3), (pixel, truth)


def test_mesh_chords_are_the_wall_chords_at_each_pixel_centre(float64):
    points, faces = tube(60e-6, 70e-6, 480e-6, 64)
    pose = CameraPose(np.array([0.0, -1e-3, 0.0]), np.asarray(LOOK_ACROSS))
    found = mesh_chords({"wall": (points @ ALONG_X.T, faces)}, DETECTOR.pinhole(), pose)
    assert found.names == ("wall",) and found.chords.shape == (1, 1, 120, 160)
    chords = np.asarray(found.chords)[0, 0]
    rays = ray_table(DETECTOR.pinhole(), pose)
    directions = np.stack([np.asarray(c, dtype=np.float64) for c in rays.direction], axis=-1)
    _, low, high = tube_bounds(directions, 64)
    slack = 64.0 * np.finfo(np.float64).eps * 1e-3
    assert np.all(chords >= low - slack) and np.all(chords <= high + slack)
    # the bracket's width through the axis: 2 (70 + 60 um) (1 - cos(pi / 64)) = 0.31 um of the 20 um wall (wider at a
    # grazing silhouette), so the check would see a chord off by a crossing's t (a millimetre) or by the wall's thickness
    assert float((high - low)[60, 80]) < 0.4e-6
    assert int(np.asarray(found.crossings)[0, 0, 60, 80]) == 4
    assert int(np.asarray(found.crossings)[0, 0, 0, 80]) == 0


def test_a_scene_with_no_mesh_writes_the_spec_and_leaves_it_wrote_before():
    node = ballistic().build().graph.get_node("camera:")
    spec = node._spec()  # noqa: SLF001
    assert "meshes" not in spec and "mesh_samples" not in spec
    assert all("around" not in each for each in spec["subjects"])
    assert not set(KEYS) & set(node.params_pytree()) and not node.traces_meshes()
    assert "mesh_triangles" not in node.constants()
    assert node.derivative_leaves_problem(frozenset({"fx", "fy"})) is None
