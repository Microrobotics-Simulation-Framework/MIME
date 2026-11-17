"""Does the jvp now equal the rendered image's derivative at the singular pixels? ``python
scripts/camera_route2/crease_held_point_check.py OUT.json``, from the repository's root (the renderer, float64).

**Why.** ``crease_singular_check.py`` found the held march's derivative error concentrated at one pixel at three poses,
where a sub-ray passes a few thousandths of a pixel from the cylinder side's axis at the rod's end: there the jvp moved
the span's place by an implicit step whose curvature (a second difference a march spacing wide) is hundreds of times
below the profile's, while the rendered image moves it with its own held point. ``_sampled._place`` now gives each
branch's read point the derivative of the image's own held point (the march's points and spacing as they move, the
quartic's least through its five live values, its steps and branches as they fell), and ``_moving`` reads the implicit
step's slope and curvature at that moving place: a change of derivatives only (the images are bit for bit the same).

**What it reports**, at each pose and along the body's x and z, at the singular pixel and on the test's aggregates
(the darkness's equivalent radius and centroid on the pixels about the caps' rims, ``crease``):

- the renderer's **jvp**;
- the **rendered image's central difference** at each step of ``STEPS`` (a quarter apart);
- the **exact image's central difference** (``hull_oracle``) at ``EXACT_STEPS``.

**The line**, stated before the run (``predictions_camera_pr5b.json``, ``held_point_check``): the jvp equals the rendered
image's derivative when it lies within the sweep's own error of the last step's difference: twice the change between
the last two steps (a quarter apart: where the error is second order in the step, thirty times the last step's error)
plus the rounding there, ``FLOOR_ULPS`` units in the last place of the share over the step (G17's
``derivative_line``, at the pixel).

**The controls**:

- **the plant** (``near-axis``): every sub-ray within ``NEAR`` px of the side's axis has its read point's place moved an
  extra ``PLANT`` of the body's motion along x and z, in the jvp only (its value is the place's); at roll 0 four
  sub-rays of a row pass the axis within a thousandth of a pixel and only one of them reaches its sub-cell, so the plant
  takes all of them. It must cross the line at all three poses;
- **the positive control**: every other pixel of the band (within two of a partial pixel of the exact image), the jvp
  against the rendered difference by the same line, at each pose and direction: it must hold at all six.

The exact image's difference is reported beside them, never judged against the jvp: where the rendered image's
derivative departs from the exact image's, that is the image's error (its share in one sub-cell), not the jvp's."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sensing"))
import crease_line as C  # noqa: E402
import hull_oracle  # noqa: E402

#: The three poses (turn, roll, offset in pixels) and their singular pixel (u, v), from ``crease_singular_check_pr5b.json``.
POSES = (((6.0, 75.0, (0.3, 0.8)), (43, 59)), ((3.0, 0.0, (0.0, 0.0)), (83, 29)), ((2.0, 0.0, (0.0, 0.0)), (81, 29)))
#: The sweep's steps, in pixels of the body's motion: a quarter apart, 1/16384 to 1/16777216 px. The rendered share at
#: the first pose's pixel turns within about a thousandth of a pixel (its spurious sliver), so the sweep starts well
#: below that.
STEPS = tuple(4.0 ** -k for k in range(7, 13))
EXACT_STEPS = (1e-4, 1e-3, 1e-2, 1e-1)
#: The rounding allowed at the last step, in units in the last place of the share (G17's test takes 64).
FLOOR_ULPS = 64
#: The plant: sub-rays within ``NEAR`` px of the side's axis, their place moved ``PLANT`` of the body's motion.
NEAR = 0.01
PLANT = 1.0
#: The sub-rays' distance from the axis the plant measures, in metres a pixel at the body.
PIXEL = C.PIXEL


def crease_mask(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """``test_crease.crease_mask``: the pixels within two of the caps' rims' images."""
    mask = np.zeros((C.HEIGHT, C.WIDTH), dtype=bool)
    for rim in hull_oracle.cylinder_outline(centre, axis, C.RADIUS, C.LENGTH):
        projected = hull_oracle.project(rim, C.FOCAL, C.DISTANCE, C.WIDTH, C.HEIGHT)
        for u, v in np.round(projected).astype(int)[::20]:
            mask[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = True
    return mask


def aggregates(dark: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v, u = np.indices(dark.shape)
    covered = dark * mask
    total = covered.sum()
    return np.array([math.sqrt(total / math.pi), (covered * u).sum() / total, (covered * v).sum() / total])


def band(exact: np.ndarray) -> np.ndarray:
    partial = (exact > 0) & (exact < 1)
    found = partial.copy()
    for dj in range(-2, 3):
        for di in range(-2, 3):
            found |= np.roll(np.roll(partial, dj, 0), di, 1)
    return found


def plant_near_axis() -> None:
    """``_sampled._place`` with the plant added: on sub-rays within ``NEAR`` px of the side's axis, the place moves an
    extra ``PLANT`` of the body's motion along x and z (the rays' origin moves against the body), its value unchanged."""
    import jax
    import jax.numpy as jnp

    import mime.sensing._sampled as sampled_route

    found = sampled_route._place

    def planted(sampled, p, o, d, held, read, made):
        place = found(sampled, p, o, d, held, read, made)
        a = jax.lax.stop_gradient(jnp.asarray(p["axis"]))
        a = a / jnp.linalg.norm(a)
        h_o, h_d = (jax.lax.stop_gradient(c) for c in (o, d))
        cross = (h_d[1] * a[2] - h_d[2] * a[1], h_d[2] * a[0] - h_d[0] * a[2], h_d[0] * a[1] - h_d[1] * a[0])
        length = jnp.sqrt(sum(c * c for c in cross))
        apart = jnp.abs(sum(h_o[i] * cross[i] for i in range(3))) / length
        motion = -((o[0] - h_o[0]) + (o[2] - h_o[2]))
        return place + jnp.where(apart < NEAR * PIXEL, PLANT * motion, 0.0)

    sampled_route._place = planted


def image_error(tangent: np.ndarray, exact: np.ndarray, judged: np.ndarray, centre: np.ndarray,
                axis: np.ndarray) -> dict:
    """The image's error at each band pixel (the jvp, the rendered image's derivative on its branch, less the exact
    image's at 1e-4 px), by ``crease_image_line``'s classes: its largest outside them and in each, and the five
    largest pixels with their sub-rays' least distance from the side's axis."""
    import crease_image_line as I

    error = np.abs(tangent - exact)
    near = I.axis_distance(centre, axis)
    classes = I.classes(judged, near, I.flagged(centre, axis))
    found: dict = {}
    for name, mask in classes.items():
        if mask.any():
            at = np.unravel_index(int(np.argmax(np.where(mask, error, -1.0))), error.shape)
            found[name] = {"error": float(error[at]), "at_v_u": [int(at[0]), int(at[1])], "near_px": float(near[at])}
    order = np.argsort(np.where(judged, error, -1.0).ravel())[::-1][:5]
    found["largest"] = [{"at_v_u": [int(k // error.shape[1]), int(k % error.shape[1])], "error": float(error.ravel()[k]),
                         "near_px": float(near.ravel()[k])} for k in order]
    return found


def renderer_arm(pose: tuple, pixel: tuple[int, int]) -> dict:
    """The jvp and the rendered difference sweep, at the pixel and on the band and the crease aggregates."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mime.sensing import Appearance, CameraPose, Pinhole, Subject, render

    turn, roll, (du, dv) = pose
    axis = C.axis_of(turn, roll)
    centre = C.CENTRE + np.array([du, 0.0, -dv]) * C.PIXEL
    camera = Pinhole(C.WIDTH, C.HEIGHT, C.FOCAL, C.FOCAL, (C.WIDTH - 1) / 2, (C.HEIGHT - 1) / 2)
    camera_pose = CameraPose(jnp.asarray([0.0, -C.DISTANCE, 0.0]),
                             jnp.asarray([math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0]))
    body = Subject("rod", "cylinder", {"radius": C.RADIUS, "length": C.LENGTH, "axis": tuple(float(a) for a in axis)},
                   appearance=Appearance.opaque(), moving=True, route="sampled")

    def dark(at):
        return 1.0 - render([body], camera, camera_pose, {"rod": (at, jnp.asarray((1.0, 0.0, 0.0, 0.0)))}).transmission

    values = jax.jit(dark)
    tangents = jax.jit(lambda at, e: jax.jvp(dark, (at,), (e,))[1])
    at = jnp.asarray(centre)
    exact0 = hull_oracle.shares(hull_oracle.cylinder_outline(centre, axis, C.RADIUS, C.LENGTH), C.FOCAL, C.DISTANCE,
                                C.WIDTH, C.HEIGHT)
    judged = band(exact0)
    mask = crease_mask(centre, axis)
    u, v = pixel
    row: dict = {}
    for name, index in (("x", 0), ("z", 2)):
        e = np.zeros(3)
        e[index] = C.PIXEL
        tangent = np.asarray(tangents(at, jnp.asarray(e)))
        differences = [(np.asarray(values(at + s * jnp.asarray(e))) - np.asarray(values(at - s * jnp.asarray(e))))
                       / (2 * s) for s in STEPS]
        line = 2.0 * np.abs(differences[-1] - differences[-2]) + FLOOR_ULPS * np.finfo(np.float64).eps / STEPS[-1]
        apart = np.abs(tangent - differences[-1])
        others = judged.copy()
        others[v, u] = False
        # A pixel whose jvp is not finite fails the control (the line is never zero: its floor is positive).
        ratio = np.where(others, np.where(np.isfinite(apart), apart / line, np.inf), 0.0)
        worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        dark0 = np.asarray(values(at))
        exact = {}
        for s in EXACT_STEPS:
            moved = [hull_oracle.shares(hull_oracle.cylinder_outline(centre + sign * s * e, axis, C.RADIUS, C.LENGTH),
                                        C.FOCAL, C.DISTANCE, C.WIDTH, C.HEIGHT) for sign in (1.0, -1.0)]
            exact[s] = (moved[0] - moved[1]) / (2 * s), (aggregates(moved[0], mask) - aggregates(moved[1], mask)) / (2 * s)
        rendered_aggregate = [(aggregates(np.asarray(values(at + s * jnp.asarray(e))), mask)
                               - aggregates(np.asarray(values(at - s * jnp.asarray(e))), mask)) / (2 * s)
                              for s in STEPS[-2:]]
        total = (dark0 * mask).sum()
        _, cu, cv = aggregates(dark0, mask)
        vv, uu = np.indices(dark0.shape)
        weights = np.stack([np.full(dark0.shape, 1.0 / (2.0 * math.pi * math.sqrt(total / math.pi))),
                            (uu - cu) / total, (vv - cv) / total]) * mask
        jvp_aggregate = np.sum(weights * tangent, axis=(1, 2))
        row[name] = {
            "pixel": {"jvp": float(tangent[v, u]), "rendered": [float(x[v, u]) for x in differences],
                      "line": float(line[v, u]), "apart": float(apart[v, u]),
                      "within": bool(apart[v, u] <= line[v, u]),
                      "exact": {str(s): float(exact[s][0][v, u]) for s in EXACT_STEPS}},
            "aggregates": {"jvp": jvp_aggregate.tolist(), "rendered": rendered_aggregate[-1].tolist(),
                           "rendered_line": float(np.linalg.norm(2.0 * (rendered_aggregate[-1] - rendered_aggregate[-2]))),
                           "exact": {str(s): exact[s][1].tolist() for s in EXACT_STEPS},
                           "rendered_less_exact": float(np.linalg.norm(rendered_aggregate[-1] - exact[EXACT_STEPS[0]][1]))},
            "other_pixels": {"largest_share_of_line": float(ratio[worst]), "at_v_u": [int(worst[0]), int(worst[1])],
                             "within": bool(ratio[worst] <= 1.0), "count": int(others.sum())},
            "nonfinite_jvp_pixels": int((~np.isfinite(tangent)).sum()),
            "image": image_error(tangent, exact[EXACT_STEPS[0]][0], judged, centre, axis),
        }
    return row


def main() -> None:
    planted = "--plant" in sys.argv
    if planted:
        plant_near_axis()
    rows = []
    for pose, pixel in POSES:
        row = {"turn": pose[0], "roll": pose[1], "offset": list(pose[2]), "pixel_u_v": list(pixel),
               **renderer_arm(pose, pixel)}
        print(json.dumps(row), flush=True)
        rows.append(row)
    report = {"plant": {"near_px": NEAR, "size": PLANT} if planted else None, "steps_px": STEPS,
              "exact_steps_px": EXACT_STEPS, "floor_ulps": FLOOR_ULPS, "rows": rows}
    out = [a for a in sys.argv[1:] if not a.startswith("--")]
    if out:
        Path(out[0]).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
