"""Appearance's sources: a subject's declared appearance wins over the stage's display primvars for it, which win over
the default (opaque, mid-grey); each subject's source is reported and the defaulted ones named in the build note; the
backlit camera reads the stage's display opacity as a flat absorber (the share of the backlight it blocks where it
covers a pixel), in both dtypes.

**The control** (``CAMERA_PLANT=stage-first``) reads the stage before the declaration: the declared subject's source
and its pixels then differ, every run."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from camera_cases import across, planted

import mime.sensing._render as render_module
from mime.sensing import Appearance, Display, Source, Subject, appearance_sources, render

STAGE = {"declared": Display(opacity=0.9), "staged": Display(colour=(0.2, 0.6, 0.8), opacity=0.4)}


def subjects() -> list[Subject]:
    def ball(name: str, x: float, appearance: Appearance | None) -> Subject:
        return Subject(name, "sphere", {"radius": 0.8e-3}, appearance=appearance, pose=((x, 0.0, 0.0),
                                                                                         (1.0, 0.0, 0.0, 0.0)))

    return [ball("declared", -3e-3, Appearance.attenuating(0.0)), ball("staged", 0.0, None),
            ball("bare", 3e-3, None)]


def plant(monkeypatch) -> None:
    if planted("stage-first"):
        original = render_module.resolve

        def stage_first(held, stage):
            return original([dataclasses.replace(s, appearance=None) if stage and s.name in stage else s
                             for s in held], stage)

        monkeypatch.setattr(render_module, "resolve", stage_first)


def test_each_subjects_source_and_the_note(monkeypatch):
    plant(monkeypatch)
    sources, notes = appearance_sources(subjects(), STAGE)
    assert sources == {"declared": Source.DECLARED, "staged": Source.STAGE, "bare": Source.DEFAULT}
    assert notes == ("appearance defaulted for: bare",)


def test_the_pixels_follow_the_sources(dtype, monkeypatch):
    plant(monkeypatch)
    camera, pose = across(64)
    image = np.asarray(render(subjects(), camera, pose, stage=STAGE).transmission, dtype=np.float64)
    row = image.shape[0] // 2
    centres = [int(round((x / 13.3e-3) * 64 + 31.5)) for x in (-3e-3, 0.0, 3e-3)]
    declared, staged, bare = (image[row, c] for c in centres)
    assert declared == 1.0
    assert abs(staged - 0.6) <= 64 * float(jnp.finfo(dtype).eps)
    assert bare == 0.0
