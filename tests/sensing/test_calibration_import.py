"""``Pinhole.from_calibration``: calibration files OpenCV 4.10 wrote (``cv2.FileStorage``, YAML and XML, generated once by
``make_calibration_files.py`` and pinned by their sha256) are read to the camera built by hand from the same numbers,
labelled fitted with the file's name and sha256; ``project`` with the imported numbers matches ``cv2.projectPoints``
within 64 units in the last place of the pixel's scale (the camera core's G1 line), in both dtypes. A model beyond the five
coefficients (the rational model's eight), a camera matrix with a skew or another last row, and an image size other than
the camera's are refused with their codes.

**The controls** swap p1 and p2 on import (``CAMERA_PLANT=p-swap``) and drop k3 (``CAMERA_PLANT=k3-drop``): each fails the
equality and the projection, every run (the files carry all five coefficients distinct and far from zero)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from in_loop_cases import planted

import mime.sensing._calibration as calibration
from mime.build import ArgumentRefused
from mime.sensing import BrownConrady, CameraPose, Pinhole, project

FILES = Path(__file__).with_name("calibration_files")
GOLDENS = json.loads(Path(__file__).with_name("calibration_goldens.json").read_text(encoding="utf-8"))
ULPS = 64.0

if planted("p-swap") or planted("k3-drop"):
    held = calibration.read_calibration

    def planted_read(path):
        found = held(path)
        k1, k2, p1, p2, k3 = found.distortion
        changed = (k1, k2, p2, p1, k3) if planted("p-swap") else (k1, k2, p1, p2, 0.0)
        return type(found)(found.fx, found.fy, found.cx, found.cy, changed, found.width, found.height, found.name,
                           found.sha256)

    calibration.read_calibration = planted_read


@pytest.mark.parametrize("name", sorted(GOLDENS["sha256"]))
def test_the_files_are_the_pinned_ones(name):
    assert hashlib.sha256((FILES / name).read_bytes()).hexdigest() == GOLDENS["sha256"][name]


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda c: c["file"])
def test_an_import_equals_the_camera_built_by_hand_and_projects_as_opencv_does(dtype, case):
    imported = Pinhole.from_calibration(FILES / case["file"])
    fx, fy, cx, cy = case["matrix"]
    k1, k2, p1, p2, k3 = case["coefficients"]
    by_hand = Pinhole(case["width"], case["height"], fx, fy, cx, cy, BrownConrady(k1, k2, p1, p2, k3))
    assert (imported.width, imported.height) == (by_hand.width, by_hand.height)
    for name in ("fx", "fy", "cx", "cy"):
        assert getattr(imported, name) == pytest.approx(getattr(by_hand, name), rel=1e-15, abs=0)
    for name in ("k1", "k2", "p1", "p2", "k3"):
        assert getattr(imported.distortion, name) == pytest.approx(getattr(by_hand.distortion, name), rel=1e-15, abs=0)
    assert imported.distortion.source == f"fitted: {case['file']}, sha256 {GOLDENS['sha256'][case['file']]}"
    pose = CameraPose(jnp.asarray(case["position"]), jnp.asarray(case["orientation"]))
    pixels, _ = project(imported, pose, jnp.asarray(case["points"]))
    expected = np.asarray(case["pixels"])
    scale = fx + np.abs(expected)
    error = float(np.max(np.abs(np.asarray(pixels, dtype=np.float64) - expected) / scale))
    assert error <= ULPS * float(np.finfo(dtype).eps), error


def test_a_model_beyond_the_five_coefficients_is_refused():
    with pytest.raises(ArgumentRefused) as error:
        Pinhole.from_calibration(FILES / "rational.yml")
    assert "camera.calibration-model" in str(error.value)


@pytest.mark.parametrize(("old", "new"), [("911.8516936884148, 0.,", "911.8516936884148, 0.12,"),
                                          ("0., 0., 1. ]", "0., 0.001, 1. ]")], ids=["skew", "last-row"])
def test_a_camera_matrix_with_a_skew_or_another_last_row_is_refused(tmp_path, old, new):
    text = (FILES / "camera_a.yml").read_text()
    assert text.count(old) == 1
    path = tmp_path / "skewed.yml"
    path.write_text(text.replace(old, new))
    with pytest.raises(ArgumentRefused) as error:
        Pinhole.from_calibration(path)
    assert "camera.calibration-model" in str(error.value)
    assert Pinhole.from_calibration(FILES / "camera_a.yml").fx == 911.8516936884148


def test_an_image_size_other_than_the_cameras_is_refused():
    with pytest.raises(ArgumentRefused) as error:
        Pinhole.from_calibration(FILES / "camera_a.yml", width=320, height=240)
    assert "camera.calibration-size" in str(error.value) and "640 by 480" in str(error.value)
    assert Pinhole.from_calibration(FILES / "camera_a.yml", width=640, height=480).width == 640


def test_a_file_that_is_no_calibration_is_refused(tmp_path):
    (tmp_path / "x.yml").write_text("%YAML:1.0\n---\nimage_width: 640\n", encoding="utf-8")
    with pytest.raises(ArgumentRefused) as error:
        Pinhole.from_calibration(tmp_path / "x.yml")
    assert "camera.calibration-file" in str(error.value)
