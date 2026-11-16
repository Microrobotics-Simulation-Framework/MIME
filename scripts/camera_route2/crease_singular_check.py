"""Is the march term's singular pixel the scheme's error or the geometry's? ``python
scripts/camera_route2/crease_singular_check.py OUT.json``, from the repository's root (numpy; the renderer's arm needs
JAX, float64).

``crease_march_line.py`` found the held march's derivative error small (at most 9.8e-5 px a px) at 357 of its 360 poses
and large at three (0.50, 0.030, 0.024), all of it at one pixel at the rod's end, where a sub-ray passes a few
thousandths of a pixel from the side's axis. Two cases are possible: the scheme's own error there (its derivative far
from the exact image's), or a near-singularity of the geometry itself (the exact image's derivative as steep, the
scheme following it). This tells them apart at those three poses, along x and z:

- **the singular pixel**: the band pixel where the held and live derivatives (``crease_march_line.View``) differ most;
- **the scheme's derivative** there, held (the jvp's) and live (the rendered image's on its branch), central
  differences at ``crease_march_line.EPSILON``; and the renderer's own jvp and its central difference at that step, in
  float64 (``renderer``), so that the re-implementation is checked at the pixel too;
- **the exact geometry's derivative** there: the exact image's (``hull_oracle``) central difference at ``STEPS``, 1e-4
  to 1e-1 px, its value at the smallest step taken where it has settled (within ``SETTLED`` of the next step's);
- **the line**, ``LINE``, stated before the run (``predictions_camera_pr5b.json``, ``singular_check``): the scheme
  agrees with the geometry at a pixel when its derivative lies within ``LINE`` px^2 a px of the body's motion of the
  exact one;
- **the positive control**: every other band pixel of the same pose and direction, the scheme against the geometry
  (the check passes where nothing is singular, or the line is too tight to tell);
- **the plant**: the near-axis sub-ray's span line moved an extra ``PLANT`` px a px of the body's motion in the held
  arm (an error in the near-axis term the size of the body's own speed), which must cross the line;
- **the pixel's own error** at the body moved 1e-4 to 1e-1 px (``pixel_error_moved``), the scheme's image made afresh
  against the exact one, beside the pixel line.

``LINE`` (0.1) is declared: a tenth of a straight edge crossing a whole pixel at the body's own speed. A body translated
rigidly moves its outline's images at most at the depth ratio (about 1.04 px a px here), so the exact derivative of a
pixel's share is at most the outline's length within the pixel (at most 4) times that."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sensing"))
import crease_line as C  # noqa: E402
import crease_march_line as M  # noqa: E402
import hull_oracle  # noqa: E402

POSES = ((6.0, 75.0, (0.3, 0.8)), (3.0, 0.0, (0.0, 0.0)), (2.0, 0.0, (0.0, 0.0)))
STEPS = (1e-4, 1e-3, 1e-2, 1e-1)
SETTLED = 1e-3
LINE = 0.1
PLANT = 1.0


def exact_image(view: M.View, shift: np.ndarray) -> np.ndarray:
    centre = view.centre + shift
    return hull_oracle.shares(hull_oracle.cylinder_outline(centre, view.axis, C.RADIUS, C.LENGTH), C.FOCAL,
                              C.DISTANCE, C.WIDTH, C.HEIGHT)


def near_axis(view: M.View, pixel: int) -> tuple[int, float]:
    """The sub-ray of the band pixel ``pixel`` (its index in ``view.pick``) nearest the side's axis, and its distance
    from it in pixels."""
    rod = C.Rod(view.axis, view.centre)
    a = rod.axis
    w = rod.o - (rod.o @ a) * a
    found = []
    for sub in view.subs:
        d = sub[pixel]
        across = d - (d @ a) * a
        t = -(w @ across) / (across @ across)
        found.append(float(np.linalg.norm(w + t * across)) / C.PIXEL)
    best = int(np.argmin(found))
    return best, found[best]


class Planted(M.View):
    """``crease_march_line.View`` with the near-axis sub-ray's span line moved an extra ``PLANT`` px a px of the body's
    motion along ``direction`` in the held arm."""

    def __init__(self, view: M.View, pixel: int, sub: int, direction: int):
        self.__dict__.update(view.__dict__)
        self.planted = (pixel, sub, direction)

    def image(self, shift: np.ndarray, arm: str) -> np.ndarray:
        pixel, planted_sub, direction = self.planted
        rod = C.Rod(self.axis, self.centre + shift)
        held, h, interior, _ = self.march
        share = np.zeros(len(self.pick))
        size = 1.0 / C.CORNER_SAMPLES
        for index, (sub, (place, _, slope0, curvature)) in enumerate(zip(self.subs, self.held, strict=True)):
            found = M.lines(rod, sub, self.du, self.dv, place, slope0, curvature)
            g, nu, nv = found[0]
            found[0] = (np.where(interior, g, 1e9), nu, nv)
            if index == planted_sub:
                g, nu, nv = found[1]
                g = g.copy()
                g[pixel] += PLANT * shift[direction] / C.PIXEL
                found[1] = (g, nu, nv)
            share += C.composed(found, C.TERMS, size) / C.CORNER_SAMPLES ** 2
        covered = self.outside.copy()
        covered.ravel()[self.pick] = share
        return covered


def renderer(view: M.View, flat: int, direction: int) -> dict:
    """The renderer's jvp of the pixel's share in the body's place and its central difference at ``EPSILON``
    (float64)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

    camera = Pinhole(C.WIDTH, C.HEIGHT, C.FOCAL, C.FOCAL, (C.WIDTH - 1) / 2, (C.HEIGHT - 1) / 2)
    pose = CameraPose(jnp.asarray([0.0, -C.DISTANCE, 0.0]), jnp.asarray([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]))
    body = Subject("rod", "cylinder", {"radius": C.RADIUS, "length": C.LENGTH,
                                       "axis": tuple(float(a) for a in view.axis)},
                   appearance=Appearance.opaque(), moving=True, route="sampled")

    def share(at):
        found = render([body], camera, pose, {"rod": (at, jnp.asarray((1.0, 0.0, 0.0, 0.0)))})
        return (1.0 - found.transmission).reshape(-1)[flat]

    e = np.zeros(3)
    e[direction] = C.PIXEL
    at = jnp.asarray(view.centre)
    tangent = float(jax.jvp(share, (at,), (jnp.asarray(e),))[1])
    values = jax.jit(share)
    difference = float((values(at + M.EPSILON * jnp.asarray(e)) - values(at - M.EPSILON * jnp.asarray(e)))
                       / (2 * M.EPSILON))
    return {"jvp": tangent, "difference": difference}


def moved_error(pose: tuple, direction: int, step: float, v: int, u: int) -> float:
    """The scheme's image (its march made afresh) less the exact image at the pixel, the body moved ``step`` px."""
    turn, roll, (du, dv) = pose
    moved = M.View(turn, roll, (du + step, dv) if direction == 0 else (du, dv - step))
    return float(abs(moved.image(np.zeros(3), "live") - exact_image(moved, np.zeros(3)))[v, u])


def one(pose: tuple, with_renderer: bool) -> dict:
    view = M.View(*pose)
    row: dict = {"turn": pose[0], "roll": pose[1], "offset": list(pose[2])}
    for name, direction in (("x", 0), ("z", 2)):
        e = np.zeros(3)
        e[direction] = M.EPSILON * C.PIXEL
        scheme = {arm: (view.image(e, arm) - view.image(-e, arm)) / (2 * M.EPSILON) for arm in ("held", "live")}
        exact = {}
        for step in STEPS:
            s = np.zeros(3)
            s[direction] = step * C.PIXEL
            exact[step] = (exact_image(view, s) - exact_image(view, -s)) / (2 * step)
        settled = np.abs(exact[STEPS[0]] - exact[STEPS[1]]) <= SETTLED
        band = np.zeros(C.HEIGHT * C.WIDTH, dtype=bool)
        band[view.pick] = True
        band = band.reshape(C.HEIGHT, C.WIDTH)
        singular = int(np.argmax(np.where(band, np.abs(scheme["held"] - scheme["live"]), -1.0)))
        v, u = divmod(singular, C.WIDTH)
        pixel = int(np.nonzero(view.pick == singular)[0][0])
        sub, distance = near_axis(view, pixel)
        planted = Planted(view, pixel, sub, direction)
        plant = ((planted.image(e, "held") - planted.image(-e, "held")) / (2 * M.EPSILON))[v, u]
        others = band.copy()
        others[v, u] = False
        apart = {arm: np.abs(scheme[arm] - exact[STEPS[0]]) for arm in ("held", "live")}
        worst_other = {arm: float(np.max(apart[arm][others])) for arm in ("held", "live")}
        where_other = {arm: list(map(int, divmod(int(np.argmax(np.where(others, apart[arm], -1.0))), C.WIDTH)))
                       for arm in ("held", "live")}
        found = {
            "pixel": {"u": u, "v": v, "near_axis_sub_ray": sub, "its_distance_from_the_axis_px": distance},
            "scheme": {arm: float(scheme[arm][v, u]) for arm in ("held", "live")},
            "exact": {str(step): float(exact[step][v, u]) for step in STEPS},
            "exact_settled": bool(settled[v, u]),
            "apart": {arm: float(apart[arm][v, u]) for arm in ("held", "live")},
            "agrees": {arm: bool(apart[arm][v, u] <= LINE) for arm in ("held", "live")},
            "plant": {"held": float(plant), "apart": float(abs(plant - exact[STEPS[0]][v, u])),
                      "moved_by": float(abs(plant - scheme["held"][v, u])),
                      "crosses": bool(abs(plant - scheme["held"][v, u]) > LINE)},
            "other_pixels": {"largest_apart": worst_other, "at_v_u": where_other,
                             "unsettled": int(np.sum(others & ~settled)),
                             "agree": {arm: bool(worst_other[arm] <= LINE) for arm in ("held", "live")}},
            "pixel_error_moved": {str(step): moved_error(pose, direction, step, v, u) for step in STEPS},
        }
        if with_renderer:
            found["renderer"] = renderer(view, singular, direction)
        row[name] = found
        print(json.dumps({"pose": pose, name: found}), flush=True)
    return row


def main() -> None:
    with_renderer = "--no-renderer" not in sys.argv
    rows = [one(pose, with_renderer) for pose in POSES]
    report = {"line": LINE, "plant_px_a_px": PLANT, "steps_px": STEPS, "settled": SETTLED, "rows": rows}
    out = [a for a in sys.argv[1:] if not a.startswith("--")]
    if out:
        Path(out[0]).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
