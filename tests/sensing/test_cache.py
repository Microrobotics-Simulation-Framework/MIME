"""The static cache: rendered once per camera, refused when any input it was made from has moved, and a cached frame
equal to the uncached one (the same arithmetic), in both dtypes; drawn moves of every size from a part in a thousand to
a tenth. **The walk** changes every field of every input (each static subject's, its appearance's, the camera's and its
distortion's, the pose's, the stage's display primvars, the corner sub-samples) and requires a new key wherever the
static image or its flags moved; a field or a ``render`` parameter it does not know fails it.

**The controls:** ``CAMERA_PLANT=key-without-cx`` leaves the principal point's u out of the key, ``key-omits-window``,
``key-omits-tolerance`` and ``key-omits-corner-samples`` leave out the fields the review found missing: each a moved
input not refused, every run."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math

import jax.numpy as jnp
import numpy as np
import pytest
from camera_cases import PLANT, TUBE, across, planted
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import mime.sensing._render as render_module
from mime.build import ArgumentRefused
from mime.sensing import Appearance, BrownConrady, CameraPose, Display, Look, Pinhole, Subject, render, static_cache

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


@functools.cache
def prepared(dtype: str) -> object:
    """The unmoved scene's cache, made once per dtype: a moved number's refusal comes from the key, before any render."""
    held, camera, pose = scene()
    return static_cache(held, camera, pose)


@pytest.fixture
def keyed(monkeypatch: pytest.MonkeyPatch) -> None:
    if planted("key-without-cx"):
        original = render_module.cache_key

        def without_cx(subjects, camera, pose, dtype, *rest):
            return original(subjects, dataclasses.replace(camera, cx=0.0), pose, dtype, *rest)

        monkeypatch.setattr(render_module, "cache_key", without_cx)


@pytest.mark.parametrize("moved", NUMBERS)
def test_a_moved_number_makes_the_cache_stale(dtype, keyed, moved):
    cache = prepared(dtype)
    after, camera, pose = scene(moved, 1.001)
    with pytest.raises(ArgumentRefused) as error:
        render(after, camera, pose, POSES, cache=cache)
    assert {refusal.code for refusal in error.value.refusals} == {"camera.cache-stale"}


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(moved=st.sampled_from(NUMBERS), size=st.floats(1e-3, 0.1), sign=st.sampled_from([-1.0, 1.0]))
def test_moves_of_every_size_make_it_stale(dtype, keyed, moved, size, sign):
    cache = prepared(dtype)
    after, camera, pose = scene(moved, 1.0 + sign * size)
    with pytest.raises(ArgumentRefused):
        render(after, camera, pose, POSES, cache=cache)


def test_a_cached_frame_is_the_uncached_one(dtype):
    held, camera, pose = scene()
    cached = render(held, camera, pose, POSES, cache=static_cache(held, camera, pose)).transmission
    uncached = render(held, camera, pose, POSES).transmission
    tolerance = 1e-12 if dtype == "float64" else 1e-6
    assert float(np.max(np.abs(np.asarray(cached) - np.asarray(uncached)))) <= tolerance


# The walk: every input the render takes, each changed, the static image and flags made again, and wherever they
# moved the key must have moved too. A field added to any of these dataclasses, or a parameter added to ``render``,
# fails the walk until it is given a change here (or named as not reaching the static image, with its reason).

#: ``render``'s parameters that do not reach the static subjects' image, and why.
NOT_STATIC = {
    "poses": "the moving subjects' poses",
    "exposure": "the moving subjects' sub-frames",
    "response": "applied after the static image",
    "key": "the response's noise",
    "cache": "the cache itself",
}


def walk_scene() -> dict[str, object]:
    """A 48 x 36 camera across a short tube with its end in view (corners), a static bead in a declared window too small
    for it (window), poking out of the back of a static drop it sits in, its chord outside the drop ranging over the
    tolerance's limit across the window (tolerance, the parent's appearance), and a moving robot; the drop's appearance
    from the stage."""
    s = math.sqrt(0.5)
    camera = Pinhole(48, 36, 900.0, 900.0, 23.5, 17.5, BrownConrady(k1=-0.05, k2=0.01))
    pose = CameraPose(jnp.asarray([0.0, -0.03, 0.0]), jnp.asarray([s, -s, 0.0, 0.0]))
    pipe = {"axis": (1.0, 0.0, 0.0), "inner_radius": 0.2e-3, "wall": 0.1e-3, "length": 1.2e-3, "centre": (-0.5e-3, 0.0, 0.0)}
    subjects = [
        Subject("pipe", "tube", pipe, "wall", Appearance.attenuating(2000.0)),
        Subject("drop", "sphere", {"radius": 0.2e-3}, pose=((0.45e-3, 0.0, 0.2e-3), (1.0, 0.0, 0.0, 0.0))),
        Subject("bead", "sphere", {"radius": 0.12e-3}, appearance=Appearance.attenuating(5000.0), parent="drop",
                pose=((0.45e-3, 0.15e-3, 0.2e-3), (1.0, 0.0, 0.0, 0.0)), window=2, tolerance=0.03),
        Subject("robot", "sphere", {"radius": 0.1e-3}, appearance=Appearance.opaque(), moving=True),
    ]
    return {"subjects": subjects, "camera": camera, "pose": pose, "stage": {"drop": Display(opacity=0.4)},
            "corner_samples": 4}


def _scaled(value: object, factor: float) -> object:
    if isinstance(value, (tuple, list)):
        return tuple(_scaled(v, factor) for v in value)
    return float(value) * factor + (factor - 1.0) * 1e-5


def _turned(q: object) -> object:
    held = np.asarray(q, dtype=np.float64) + np.array([0.0, 0.0, 0.0, 1e-3])
    return tuple(float(c) for c in held / np.linalg.norm(held))


#: How each field is changed: a number scaled (a part in a hundred, plus 1e-5 so that a zero moves), the rest named.
FIELD_CHANGES = {
    (Subject, "name"): lambda s: dataclasses.replace(s, name=s.name + "2"),
    (Subject, "kind"): lambda s: (dataclasses.replace(s, kind="capsule", coverage="box_edge",
                                                      settings={"radius": s.settings["radius"],
                                                                "height": s.settings["radius"]})
                                  if s.kind == "sphere" else None),
    (Subject, "settings"): None,
    (Subject, "region"): lambda s: dataclasses.replace(s, region="lumen") if s.kind == "tube" else None,
    (Subject, "appearance"): None,
    (Subject, "parent"): lambda s: dataclasses.replace(s, parent=None) if s.parent else None,
    (Subject, "moving"): lambda s: dataclasses.replace(s, moving=not s.moving),
    (Subject, "pose"): lambda s: dataclasses.replace(s, pose=(_scaled(s.pose[0], 1.01), _turned(s.pose[1]))),
    (Subject, "coverage"): lambda s: (dataclasses.replace(s, coverage="box_edge" if s.coverage == "disc" else "disc")
                                      if s.kind == "sphere" else None),
    (Subject, "window"): lambda s: dataclasses.replace(s, window=(s.window or 8) + 1),
    (Subject, "route"): lambda s: dataclasses.replace(s, route="closed form"),
    (Subject, "tolerance"): lambda s: dataclasses.replace(s, tolerance=s.tolerance * 4),
    (Appearance, "look"): None, (Appearance, "attenuation"): None, (Appearance, "blocks"): None,
    (Appearance, "colour"): None,
    (Pinhole, "width"): lambda c: dataclasses.replace(c, width=c.width + 2),
    (Pinhole, "height"): lambda c: dataclasses.replace(c, height=c.height + 2),
    (Pinhole, "fx"): lambda c: dataclasses.replace(c, fx=c.fx * 1.01),
    (Pinhole, "fy"): lambda c: dataclasses.replace(c, fy=c.fy * 1.01),
    (Pinhole, "cx"): lambda c: dataclasses.replace(c, cx=c.cx + 0.3),
    (Pinhole, "cy"): lambda c: dataclasses.replace(c, cy=c.cy + 0.3),
    (Pinhole, "distortion"): None,
    (CameraPose, "position"): lambda p: dataclasses.replace(p, position=p.position + jnp.asarray([1e-5, 0.0, 0.0])),
    (CameraPose, "orientation"): lambda p: dataclasses.replace(p, orientation=jnp.asarray(_turned(p.orientation))),
    (Display, "colour"): lambda d: dataclasses.replace(d, colour=(0.2, 0.3, 0.4)),
    (Display, "opacity"): lambda d: dataclasses.replace(d, opacity=0.6),
}


def _appearance_changes(a: Appearance) -> list[tuple[str, Appearance]]:
    other = Look.OPAQUE if a.look is not Look.OPAQUE else Look.ATTENUATING
    return [("look", dataclasses.replace(a, look=other, attenuation=a.attenuation or 100.0)),
            ("attenuation", dataclasses.replace(a, attenuation=float(a.attenuation) * 1.01 + 1.0)),
            ("blocks", dataclasses.replace(a, blocks=float(a.blocks) + 0.1)),
            ("colour", dataclasses.replace(a, colour=(0.1, 0.2, 0.3)))]


def walk_changes(scene: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    """Every change the walk makes, each a named copy of the scene with one input changed."""
    out: list[tuple[str, dict[str, object]]] = []
    subjects: list[Subject] = scene["subjects"]  # type: ignore[assignment]
    for i, subject in enumerate(subjects):
        def with_subject(new: Subject, i: int = i) -> dict[str, object]:
            held = list(subjects)
            held[i] = new
            if new.name != subjects[i].name:
                held = [dataclasses.replace(s, parent=new.name) if s.parent == subjects[i].name else s for s in held]
            return {**scene, "subjects": held}
        for f in dataclasses.fields(Subject):
            if f.name in render_module.DERIVED_FIELDS:
                continue
            change = FIELD_CHANGES[(Subject, f.name)]
            if f.name == "settings":
                for name, value in subject.settings.items():
                    settings = {**subject.settings, name: _scaled(value, 1.01)}
                    out.append((f"{subject.name}.settings.{name}", with_subject(dataclasses.replace(subject, settings=settings))))
            elif f.name == "appearance":
                if subject.appearance is not None:
                    for name, new in _appearance_changes(subject.appearance):
                        out.append((f"{subject.name}.appearance.{name}", with_subject(dataclasses.replace(subject, appearance=new))))
            elif change is not None and (new := change(subject)) is not None:
                out.append((f"{subject.name}.{f.name}", with_subject(new)))
    camera: Pinhole = scene["camera"]  # type: ignore[assignment]
    for f in dataclasses.fields(Pinhole):
        if f.name == "distortion":
            for g in dataclasses.fields(BrownConrady):
                value = getattr(camera.distortion, g.name)
                new = "a fit's file" if g.name == "source" else float(value) + 2e-3
                out.append((f"camera.distortion.{g.name}",
                            {**scene, "camera": dataclasses.replace(camera, distortion=dataclasses.replace(
                                camera.distortion, **{g.name: new}))}))
        else:
            out.append((f"camera.{f.name}", {**scene, "camera": FIELD_CHANGES[(Pinhole, f.name)](camera)}))
    for f in dataclasses.fields(CameraPose):
        out.append((f"pose.{f.name}", {**scene, "pose": FIELD_CHANGES[(CameraPose, f.name)](scene["pose"])}))
    stage: dict[str, Display] = scene["stage"]  # type: ignore[assignment]
    for f in dataclasses.fields(Display):
        out.append((f"stage.drop.{f.name}", {**scene, "stage": {"drop": FIELD_CHANGES[(Display, f.name)](stage["drop"])}}))
    out.append(("corner_samples", {**scene, "corner_samples": 2}))
    out.append(("stage", {**scene, "stage": None}))
    return out


def _made(scene: dict[str, object]) -> tuple[str, np.ndarray, dict[str, object]]:
    cache = static_cache(scene["subjects"], scene["camera"], scene["pose"], stage=scene["stage"],
                         corner_samples=scene["corner_samples"])
    return (cache.key, np.asarray(cache.transmission),
            {k: np.asarray(v).tolist() for k, v in cache.flags.items()})


@pytest.fixture
def omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: a field left out of the key (``CAMERA_PLANT=key-omits-window``, ``-tolerance``,
    ``-corner-samples``)."""
    if PLANT.startswith("key-omits-"):
        field = PLANT.removeprefix("key-omits-")
        if field == "corner-samples":
            original = render_module.cache_key
            monkeypatch.setattr(render_module, "cache_key",
                                lambda subjects, camera, pose, dtype, corner_samples=4: original(subjects, camera,
                                                                                                 pose, dtype))
        else:
            monkeypatch.setattr(render_module, "DERIVED_FIELDS", render_module.DERIVED_FIELDS | {field})


def test_the_key_walks_every_input(dtype, omitted):
    """Each change that moves the static image or its flags moves the key; and the walk covers every field of every
    input's dataclass and every parameter of ``render`` (a new one fails here until it is walked)."""
    walked = {name for name in inspect.signature(render).parameters} - set(NOT_STATIC)
    assert walked == {"subjects", "camera", "pose", "stage", "corner_samples"}, walked
    for cls in (Subject, Appearance, Pinhole, CameraPose, Display):
        missing = [f.name for f in dataclasses.fields(cls) if (cls, f.name) not in FIELD_CHANGES
                   and not (cls is Subject and f.name in render_module.DERIVED_FIELDS)]
        assert not missing, (cls.__name__, missing)
    scene = walk_scene()
    key, image, flags = _made(scene)
    moved = []
    for name, changed in walk_changes(scene):
        new_key, new_image, new_flags = _made(changed)
        if new_image.shape != image.shape or not np.array_equal(new_image, image) or new_flags != flags:
            moved.append(name)
            assert new_key != key, f"{name} moved the static image but not the key"
    # The walk's own premise: the review's omissions move the image here.
    assert {"bead.window", "bead.tolerance", "corner_samples"} <= set(moved), moved
