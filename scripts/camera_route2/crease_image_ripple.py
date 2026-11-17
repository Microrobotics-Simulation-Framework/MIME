"""Where the creased image's own derivative error comes from, at the pixels ``crease_image_line.py``'s scan found worst:
``python scripts/camera_route2/crease_image_ripple.py OUT.json`` (numpy only, the scheme's re-implementation).

- **The oracle's steadiness**: the exact image's central difference at each pixel over steps 1e-6 to 1e-2 px, and the
  scheme's live one over 1e-5 to 1e-3 px, so that a departure between them is the scheme's, not the oracle's.
- **The ripple**: the scheme's share at the pixel as the body moves 0 to 0.1 px (every choice recomputed at each
  place) against the exact share, with each sub-ray's least as the scheme holds it and **exact** (the ray's closest
  approach to the side's axis, which a refined least converges to: option (iii)'s and (ii)'s limit). Where the exact
  least removes the departure, it is the held least's; where it leaves it, it is not."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "sensing"))
import crease_line as C  # noqa: E402
import crease_march_line as M  # noqa: E402
import hull_oracle  # noqa: E402

#: (pose, (v, u), direction index): the scan's worst rest pixels, a thin cap's, and the first singular pose's corner.
STEADY = [((0.0, 20.0, (0.3, 0.8)), (49, 25), 0), ((4.0, 30.0, (0.0, 0.0)), (37, 19), 0),
          ((0.0, 90.0, (0.3, 0.8)), (61, 47), 2), ((0.25, 10.0, (0.3, 0.8)), (34, 83), 0),
          ((1.5, 0.0, (0.3, 0.8)), (39, 20), 0), ((6.0, 75.0, (0.3, 0.8)), (58, 35), 2)]
#: The ripple's pixels (moved along x): the worst rest pixel, the second, and the first singular pixel.
RIPPLE = {"rest_turn0": ((0.0, 20.0, (0.3, 0.8)), (49, 25)), "rest_turn4": ((4.0, 30.0, (0.0, 0.0)), (37, 19)),
          "singular": ((6.0, 75.0, (0.3, 0.8)), (59, 43))}
_held_least = M.least


def exact_least(rod, d, start, h, choice=None):
    _, made = _held_least(rod, d, start, h, choice)
    a, o = rod.axis, rod.o
    w = o - (o @ a) * a
    e = d - (d @ a)[..., None] * a
    return -(e @ w) / np.sum(e * e, axis=-1), made


def shares(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return hull_oracle.shares(hull_oracle.cylinder_outline(centre, axis, C.RADIUS, C.LENGTH), C.FOCAL, C.DISTANCE,
                              C.WIDTH, C.HEIGHT)


def steady(pose: tuple, pixel: tuple[int, int], index: int) -> dict:
    v, u = pixel
    view = M.View(*pose)
    e = np.zeros(3)
    e[index] = C.PIXEL
    exact = {str(s): float((shares(view.centre + s * e, view.axis) - shares(view.centre - s * e, view.axis))[v, u]
                           / (2 * s)) for s in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)}
    live = {str(s): float((view.image(s * e, "live") - view.image(-s * e, "live"))[v, u] / (2 * s))
            for s in (1e-5, 1e-4, 1e-3)}
    return {"pose": list(pose), "v_u": [v, u], "direction": "xz"[index // 2],
            "exact_share": float(shares(view.centre, view.axis)[v, u]),
            "scheme_share": float(view.image(np.zeros(3), "held")[v, u]), "exact_cd": exact, "scheme_cd": live}


def ripple(pose: tuple, pixel: tuple[int, int]) -> dict:
    (turn, roll, (du, dv)), (v, u) = pose, pixel
    found = {}
    for arm in ("held", "exact"):
        M.least = _held_least if arm == "held" else exact_least
        rows = []
        for k in range(21):
            view = M.View(turn, roll, (du + 0.005 * k, dv))
            rows.append((float(view.image(np.zeros(3), "held")[v, u]), float(shares(view.centre, view.axis)[v, u])))
        slopes = [((b[0] - a[0]) / 0.005, (b[1] - a[1]) / 0.005) for a, b in zip(rows, rows[1:])]
        found[arm] = {"value_error": [s - x for s, x in rows], "slope_scheme": [s for s, _ in slopes],
                      "slope_exact": [x for _, x in slopes],
                      "slope_error_largest": max(abs(s - x) for s, x in slopes)}
    M.least = _held_least
    return {"pose": list(pose), "v_u": [v, u], "step_px": 0.005, **found}


def main() -> None:
    report = {"steady": [steady(*case) for case in STEADY],
              "ripple": {name: ripple(*case) for name, case in RIPPLE.items()}}
    for name, row in report["ripple"].items():
        print(name, {arm: round(row[arm]["slope_error_largest"], 4) for arm in ("held", "exact")}, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
