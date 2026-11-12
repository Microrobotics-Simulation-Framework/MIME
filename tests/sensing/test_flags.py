"""Each of the camera's flags fires where its condition holds and not where it does not, in both dtypes:

- ``window_overflow``: a declared window of half-size 5 on a 15 px sphere, against its derived window;
- ``small_silhouette``: a box-edged sphere at 3 px, against one at 15 px (exact disc coverage has none);
- ``behind``: a sphere 5 mm behind the camera, against one in front;
- ``ghosting``: two sub-frames 4 px apart, against 0.2 px;
- ``beyond_distortion``: a strong barrel (``k1 = -0.9`` over a wide field) folding back inside the image, against no
  distortion.

(``outside_parent`` is ``test_media``'s; ``thin_cap`` ``test_corners``'s.) **The control** (``CAMERA_PLANT=flag-off``)
silences the flag under test at its source (the overflow's test, the small-silhouette radius, the behind test, the
ghosting distance, the table's flag): its firing case fails, every run."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest
from camera_cases import planted

import mime.sensing._render as render_module
from mime.sensing import Appearance, BrownConrady, CameraPose, Exposure, Pinhole, Rays, Subject, render

IDENTITY = (1.0, 0.0, 0.0, 0.0)


def _camera(focal: float = 900.0, distortion: BrownConrady | None = None) -> tuple[Pinhole, CameraPose]:
    s = math.sqrt(0.5)
    return (Pinhole(160, 120, focal, focal, 79.5, 59.5, distortion or BrownConrady()),
            CameraPose(jnp.asarray([0.0, -0.03, 0.0]), jnp.asarray([s, -s, 0.0, 0.0])))


def _ball(radius: float, **settings: object) -> Subject:
    return Subject("b", "sphere", {"radius": radius}, appearance=Appearance.opaque(), moving=True, **settings)


def _at(x: float, y: float = 0.0) -> dict[str, object]:
    return {"b": (jnp.asarray((x, y, 0.0)), jnp.asarray(IDENTITY))}


def _without_beyond(original):
    def table(camera, pose, **kw):
        rays = original(camera, pose, **kw)
        return Rays(rays.origin, rays.direction, rays.along_u, rays.along_v, jnp.zeros_like(rays.beyond))
    return table


def _silence(flag: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if not planted("flag-off"):
        return
    if flag == "window_overflow":
        monkeypatch.setattr(render_module, "_overflows", lambda *args: jnp.zeros((), dtype=bool))
    elif flag == "small_silhouette":
        monkeypatch.setattr(render_module, "SMALL_SILHOUETTE_PX", 0.0)
    elif flag == "behind":
        monkeypatch.setattr(render_module, "_behind", lambda depth, subject: jnp.zeros((), dtype=bool))
    elif flag == "ghosting":
        monkeypatch.setattr(render_module, "GHOSTING_PX", math.inf)
    else:
        monkeypatch.setattr(render_module, "ray_table", _without_beyond(render_module.ray_table))


def _two_frames(step: float) -> dict[str, object]:
    return {"b": (jnp.asarray([[0.0, 0.0, 0.0], [step, 0.0, 0.0]]), jnp.asarray([IDENTITY, IDENTITY]))}


#: Per flag: the render that should raise it and the one that should not, each ``(subject, camera, poses, exposure)``.
CASES = {
    "window_overflow": (lambda: (_ball(0.5e-3, window=5), _camera(), _at(0.0), Exposure()),
                        lambda: (_ball(0.5e-3), _camera(), _at(0.0), Exposure())),
    "small_silhouette": (lambda: (_ball(0.1e-3, coverage="box_edge"), _camera(), _at(0.0), Exposure()),
                         lambda: (_ball(0.5e-3, coverage="box_edge"), _camera(), _at(0.0), Exposure())),
    "behind": (lambda: (_ball(0.5e-3), _camera(), _at(0.0, -0.035), Exposure()),
               lambda: (_ball(0.5e-3), _camera(), _at(0.0), Exposure())),
    "ghosting": (lambda: (_ball(0.5e-3), _camera(), _two_frames(0.133e-3), Exposure(2)),
                 lambda: (_ball(0.5e-3), _camera(), _two_frames(0.0067e-3), Exposure(2))),
    "beyond_distortion": (lambda: (_ball(0.5e-3), _camera(96.0, BrownConrady(k1=-0.9)), _at(0.0), Exposure()),
                          lambda: (_ball(0.5e-3), _camera(96.0), _at(0.0), Exposure())),
}


def _raised(flag: str, case: tuple[Subject, tuple[Pinhole, CameraPose], dict[str, object], Exposure]) -> bool:
    subject, (camera, pose), poses, exposure = case
    flags = render([subject], camera, pose, poses, exposure=exposure).flags
    value = flags["beyond_distortion"] if flag == "beyond_distortion" else flags[f"{flag}:b"]
    return bool(jnp.asarray(value).astype(bool))


@pytest.mark.parametrize("flag", list(CASES))
def test_each_flag_fires_where_its_condition_holds(dtype, flag, monkeypatch):
    _silence(flag, monkeypatch)
    raise_it, leave_it = CASES[flag]
    assert _raised(flag, raise_it())
    assert not _raised(flag, leave_it())
