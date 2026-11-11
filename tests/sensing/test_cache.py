"""The static cache: rendered once per camera, refused when any number it was made from has moved (each of the
camera's, its pose's, a static subject's setting and appearance), and a cached frame equal to the uncached one (the
same arithmetic), in both dtypes; drawn moves of every size from a part in a thousand to a tenth.

**The control** (``CAMERA_PLANT=key-without-cx``) leaves the principal point's u out of the key: a moved ``cx`` is then
not refused, every run (the parametrised test moves each number)."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import TUBE, across, planted
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.sensing._render as render_module
from mime.build import ArgumentRefused
from mime.sensing import Appearance, BrownConrady, CameraPose, Pinhole, Subject, prepare, render

NUMBERS = ["fx", "fy", "cx", "cy", "k1", "position", "orientation", "inner_radius", "wall", "length", "centre",
           "wall_attenuation", "water_attenuation"]


def scene(moved: str = "", factor: float = 1.0) -> tuple[list[Subject], Pinhole, CameraPose]:
    camera, pose = across(32, BrownConrady(k1=-0.05))
    values = {name: getattr(camera, name) for name in ("fx", "fy", "cx", "cy")}
    if moved in values:
        values[moved] = values[moved] * factor
    distortion = BrownConrady(k1=-0.05 * (factor if moved == "k1" else 1.0))
    camera = Pinhole(camera.width, camera.height, values["fx"], values["fy"], values["cx"], values["cy"], distortion)
    if moved == "position":
        pose = CameraPose(pose.position * jnp.asarray([1.0, factor, 1.0]), pose.orientation)
    if moved == "orientation":
        turned = pose.orientation + jnp.asarray([0.0, 0.0, 0.0, factor - 1.0])
        pose = CameraPose(pose.position, turned / jnp.linalg.norm(turned))
    tube = dict(TUBE)
    for name in ("inner_radius", "wall", "length"):
        if moved == name:
            tube[name] = tube[name] * factor
    if moved == "centre":
        tube["centre"] = (0.0, 0.0, 1e-3 * (factor - 1.0))
    wall = Appearance.attenuating(200.0 * (factor if moved == "wall_attenuation" else 1.0))
    water = Appearance.attenuating(20.0 * (factor if moved == "water_attenuation" else 1.0))
    held = [Subject("tube", "tube", tube, "wall", wall), Subject("water", "tube", tube, "lumen", water, parent="tube"),
            Subject("umr", "sphere", {"radius": 0.7e-3}, appearance=Appearance.opaque(), parent="water", moving=True)]
    return held, camera, pose


POSES = {"umr": (jnp.asarray((0.2e-3, 0.0, -2.4e-3)), jnp.asarray((1.0, 0.0, 0.0, 0.0)))}


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("key-without-cx"):
        original = render_module.cache_key

        def without_cx(subjects, camera, pose, dtype):
            return original(subjects, dataclasses.replace(camera, cx=0.0), pose, dtype)

        monkeypatch.setattr(render_module, "cache_key", without_cx)


@pytest.mark.parametrize("moved", NUMBERS)
def test_a_moved_number_makes_the_cache_stale(dtype, keyed, moved):
    held, camera, pose = scene()
    cache = prepare(held, camera, pose)
    after, camera, pose = scene(moved, 1.001)
    with pytest.raises(ArgumentRefused) as error:
        render(after, camera, pose, POSES, cache=cache)
    assert {refusal.code for refusal in error.value.refusals} == {"camera.cache-stale"}


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(moved=st.sampled_from(NUMBERS), size=st.floats(1e-3, 0.1), sign=st.sampled_from([-1.0, 1.0]))
def test_moves_of_every_size_make_it_stale(dtype, keyed, moved, size, sign):
    held, camera, pose = scene()
    cache = prepare(held, camera, pose)
    after, camera, pose = scene(moved, 1.0 + sign * size)
    with pytest.raises(ArgumentRefused):
        render(after, camera, pose, POSES, cache=cache)


def test_a_cached_frame_is_the_uncached_one(dtype):
    held, camera, pose = scene()
    cached = render(held, camera, pose, POSES, cache=prepare(held, camera, pose)).transmission
    uncached = render(held, camera, pose, POSES).transmission
    tolerance = 1e-12 if dtype == "float64" else 1e-6
    assert float(np.max(np.abs(np.asarray(cached) - np.asarray(uncached)))) <= tolerance
