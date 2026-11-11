"""G11, refusals: the kinds without a closed form (the ellipsoid and the box, which have no run-time shape yet; the screw
and the helix, which need the sampled route) are refused for the camera with their reasons, as is a lab's kind with no
closed form registered; a route override naming a route not built; a subject written against its kind; subjects named
twice, a parent no subject is, a cycle of parents, a moving subject with no pose; a stale static cache and one given
while its numbers are traced.

**The control** (``CAMERA_PLANT=screw-form``) registers a closed form for the screw: its refusal then fails to come,
every run."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from camera_cases import TUBE, across, planted, tube_subjects

import mime.sensing._subjects as subjects_module
from mime.build import ArgumentRefused
from mime.sensing import Appearance, Pinhole, Subject, prepare, render


def code(error: pytest.ExceptionInfo[ArgumentRefused]) -> set[str]:
    return {refusal.code for refusal in error.value.refusals}


@pytest.mark.parametrize("kind", ["ellipsoid", "box", "screw", "helix", "labx.bead"])
def test_a_kind_without_a_closed_form_is_refused_with_its_reason(kind, monkeypatch):
    if planted("screw-form"):
        monkeypatch.setitem(subjects_module._closed_forms, "screw", subjects_module._closed_forms["sphere"])
    with pytest.raises(ArgumentRefused) as error:
        Subject("thing", kind, {"radius": 1e-3})
    assert code(error) == {"camera.no-closed-form"}
    message = str(error.value)
    assert kind in message and ("sampled route" in message or "run-time shape" in message
                                or "register the kind's closed form" in message)


def test_a_route_not_built_is_refused():
    with pytest.raises(ArgumentRefused) as error:
        Subject("ball", "sphere", {"radius": 1e-3}, route="sampled")
    assert code(error) == {"camera.route"}


@pytest.mark.parametrize("written", [{"region": "lumen"}, {"coverage": "disc", "kind": "capsule"},
                                     {"window": 0}, {"coverage": "soft"}])
def test_a_subject_written_against_its_kind_is_refused(written):
    arguments = {"name": "thing", "kind": "sphere", "settings": {"radius": 1e-3, "height": 1e-3}, **written}
    with pytest.raises(ArgumentRefused) as error:
        Subject(**arguments)
    assert code(error) == {"camera.subject"}


def test_names_parents_and_poses_are_checked():
    camera, pose = across(32)
    ball = Subject("ball", "sphere", {"radius": 1e-3}, appearance=Appearance.opaque(), moving=True)
    for held in ([ball, ball], [Subject("ball", "sphere", {"radius": 1e-3}, parent="nowhere")],
                 [Subject("a", "sphere", {"radius": 1e-3}, parent="b"),
                  Subject("b", "sphere", {"radius": 1e-3}, parent="a")], [ball]):
        with pytest.raises(ArgumentRefused) as error:
            render(held, camera, pose, {})
        assert code(error) == {"camera.subject"}


def test_a_stale_cache_and_a_traced_one_are_refused():
    camera, pose = across(32)
    held = tube_subjects(Appearance.attenuating(200.0), Appearance.attenuating(20.0))
    cache = prepare(held, camera, pose)
    moved = Pinhole(camera.width, camera.height, camera.fx * 1.001, camera.fy, camera.cx, camera.cy)
    with pytest.raises(ArgumentRefused) as error:
        render(held, moved, pose, cache=cache)
    assert code(error) == {"camera.cache-stale"}
    wider = [Subject("tube", "tube", {**TUBE, "wall": TUBE["wall"] * 1.01}, "wall", Appearance.attenuating(200.0)),
             held[1]]
    with pytest.raises(ArgumentRefused) as error:
        render(wider, camera, pose, cache=cache)
    assert code(error) == {"camera.cache-stale"}

    def through(focal):
        traced = Pinhole(camera.width, camera.height, focal, focal, camera.cx, camera.cy)
        return jnp.sum(render(held, traced, pose, cache=cache).transmission)

    with pytest.raises(ArgumentRefused) as error:
        jax.grad(through)(jnp.asarray(camera.fx))
    assert code(error) == {"camera.cache-traced"}


def test_exact_disc_coverage_is_a_spheres_default_and_windowed(dtype):
    """A sphere takes exact disc coverage unless the box edge is declared (which the note names); its window is sized
    from its silhouette where none is declared, and a traced pose with no declared window is refused."""
    from mime.sensing import appearance_sources

    camera, pose = across(32)
    ball = Subject("ball", "sphere", {"radius": 1e-3}, appearance=Appearance.opaque(), moving=True)
    boxed = Subject("other", "sphere", {"radius": 1e-3}, appearance=Appearance.opaque(), coverage="box")
    assert ball.coverage == "disc" and boxed.coverage == "box"
    assert appearance_sources([boxed])[1] == ("the box edge in place of exact disc coverage (opt-in) for: other",)
    frame = render([ball], camera, pose, {"ball": (jnp.zeros(3), jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
    assert not bool(frame.flags["window_overflow:ball"]) and "small_silhouette:ball" not in frame.flags
    with pytest.raises(ArgumentRefused) as error:
        jax.grad(lambda x: jnp.sum(render([ball], camera, pose, {"ball": (jnp.stack([x, 0.0, 0.0]),
                                                                          jnp.asarray((1.0, 0.0, 0.0, 0.0)))}
                                          ).transmission))(jnp.asarray(0.0))
    assert code(error) == {"camera.subject"}
