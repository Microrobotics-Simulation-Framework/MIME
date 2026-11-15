"""The corner gate's line, derived from the fix's own error terms before any draw of its confirmation: geometry and
arithmetic only, never the renderer or the truth. ``python scripts/camera_route2/corner_line.py [OUT.json]`` prints each
coil's notches over the drawn poses and the line at each width and dtype; ``tests/sensing/test_sampled.py`` holds the
same formula (``corner_line``) with this script's geometric inputs.

**The scheme the line bounds** (camera PR 5a's fix round): at a pixel where two branches of the outline cross (both
pieces' lines within the corner reach), the 4 x 4 corner sub-samples, each the exact share of its square inside the
union of the two branches' half-planes (``coverage.union_covered``), each branch's line read at the sub-ray's own least
(the held point moved to it on the sub-ray, ``_sampled.pieces``).

**Its terms**, each a bound on one two-branch pixel's error in a pixel's share, against the exact coverage:

- *the union*: exact for two straight branches; zero.
- *the cell term*, each branch's departure from its line in each sub-cell. The line is the branch's tangent at the foot of
  the sub-cell's centre (the piece is the image's signed distance to the branch, to the first order). Every point of the
  branch inside a sub-cell of side ``s`` lies within ``a = s / sqrt(2)`` of its centre, so within ``a`` of the foot along
  the branch, where an arc of radius ``rho >= 2 a`` stands off its tangent by at most ``x^2 / (rho (1 + sqrt(3)/2))``.
  A sub-cell's error is at most that over ``[-a, a]``, ``0.1263 s^3 / rho``; a branch crosses at most ``2 / s + 1 = 9``
  of a pixel's sub-cells; two branches: ``2.2736 s^3 / rho = 0.035525 / rho_px`` at ``s = 1/4``. **Assumption:** every
  branch at a two-branch pixel has a radius of curvature of at least the coil's least notch branch's, ``rho_min``
  (below); a notch's hidden continuation lies inside the other lobe and its half-plane.
- *the refinement's residual*: the sub-ray's least found from the held point (the parabola's least through the
  piece's values a march spacing either side, then the quartic's through five values half a spacing apart, by two
  Newton steps), on a straight filament at the angle ``gamma`` its arm makes with the line of sight at the notch
  (``gamma`` from the geometry, the sub-ray at most 0.53 px from its pixel's centre): its largest value error,
  ``refined_residual``, times ``2 sqrt(2)`` (two branches, each at most a diagonal long in the pixel); under 1e-4, taken
  as 1e-4. (Written first for the parabola's step alone; the quartic was added for the derivative before any draw of
  the confirmation, its residual smaller, and the term, 1e-4, unchanged.) **Amended after the confirmation failed
  (run 43642088699), the line unchanged:** the scheme broke this term's premise. At a notch a sub-ray's least lies up
  to several spacings from its pixel's held point (the branch's foot runs ``(rho + r) / rho`` as fast along the
  filament as the sub-ray across the image, and the held point is the march's parabola, up to 0.66 spacing off the
  centre ray's least), and between the two branches' basins the five values are not convex, so the parabola and the
  quartic did not move: seeds 2052, 2129 (128 px) and 2049 (256 px) read a branch 0.10 to 0.17 px too far out. The
  scheme now descends at the march's spacing to a sample lower than both its neighbours (``_sampled.LEAST_REACH``
  moves; a branch whose descent runs its reach towards the other has merged into the other's basin and takes its
  piece), then takes the quartic about it, or the least of the inner three where lower than the quartic's own value.
  On the straight filament the profile is symmetric about its least, so the descent's lowest sample is within half
  a spacing of it: ``refined_residual`` scans the held point there; under 1.1e-5 px, the term stays 1e-4. The gate
  counts the judged pixels' sub-ray branches whose descent ran its reach away from the other branch (its premise:
  none). **Not covered:** a branch whose least lies at a kink of its profile, where the arc's nearest point passes
  its end onto the cap (seed 2152, coil2 at 512 px: the quartic stepped the wrong way, 0.04 px off): the scan of
  notches below is of an endless helix's apexes, and this term's straight filament is smooth. The guard reads the
  inner three's least there, within half a spacing of the least, so a value error up to ``f'' h^2 / 8``
  (``f'' ~ 1 / r`` on the cap), which this script does not bound.
- *the truth's floor*: the sign-only truth at ``side^2`` jittered samples a pixel is a sum of independent terms, one a
  cell its outline crosses, each of range ``1 / side^2``; at most ``2 (sqrt(2) side pi/2 + 1)`` cells (two branches,
  each at most a quarter circle long); Hoeffding's bound at a probability of ``4e-8`` a pixel (a union bound over about
  2.5e5 judged pixels, one per cent in all): ``sqrt(ln(2 / 4e-8) N / 2) / side^2``, 0.0123 at ``side = 64``. Its sign
  samples (512, the least refined by a parabola) misplace the outline by under 1e-4 px: 1e-4.
- *float32's rounding*: a sub-ray's point is the camera's offset from the body (0.1 m), rotated into the body's frame
  and stepped along the ray: about five roundings at that scale, ``5 u D`` with ``u = 2^-24``, each branch's distance
  wrong by at most that over the pixel's size, along at most a diagonal: ``2 sqrt(2) 5 u D / p``. Float64's is
  below 1e-10 and taken as zero.

**The geometry.** Under orthographic projection the coil's image is the union of discs of its filament's radius about its
projected centreline; roll about its axis and turns in the image only move and turn that image, so only the turn in
depth, ``beta``, shapes it: each turn about an apex is ``X = B u + s A sin u, Z = R cos u`` (``B = b cos beta``, ``A = R
sin beta``, ``s = +1`` at one kind of apex and ``-1`` at the other), symmetric about the apex's vertical, where its
concave notch lies: the first point below the apex the arms' tube leaves. The drawn poses turn the coil by at most 10
degrees in depth; the perspective across the coil adds 0.6 (one millimetre at 0.1 m), so ``beta`` is scanned over +-10.6
degrees in steps of 0.2."""

from __future__ import annotations

import json
import math
import sys

import numpy as np

#: The coils the gate draws: the PR's (``test_sampled.COIL``) and the confirmation's second geometry.
COILS = {"coil": {"radius": 0.4e-3, "pitch": 0.6e-3, "turns": 2.5, "filament_radius": 0.08e-3},
         "coil2": {"radius": 0.4e-3, "pitch": 0.52e-3, "turns": 3.0, "filament_radius": 0.08e-3}}
#: The march's samples a ray and enclosing radius (m) for each, as ``samples_for`` and the kind give them.
MARCH = {"coil": (48, 0.9314753298e-3), "coil2": (49, 0.9541036959e-3)}
FIELD, DISTANCE, WIDTHS = 3e-3, 0.1, (128, 256, 512)
CORNER_SAMPLES = 4
TRUTH_SIDE, TRUTH_SIGNS, PER_PIXEL_PROBABILITY = 64, 512, 4e-8
BETA_DEGREES = np.round(np.arange(-10.6, 10.61, 0.2), 2)


def notches(radius: float, pitch: float, filament: float, betas=BETA_DEGREES) -> list[dict]:
    """Each apex kind's notch at each turn in depth: the outside wedge's angle (degrees), the branch's radius of
    curvature (m) and the arm's angle to the line of sight there (degrees)."""
    b = pitch / (2.0 * math.pi)
    length = math.hypot(b, radius)
    u = np.linspace(0.0, math.pi, 100001)
    found = []
    for s in (1.0, -1.0):
        for beta_degrees in betas:
            beta = math.radians(float(beta_degrees))
            big, small = b * math.cos(beta), radius * math.sin(beta)
            x, z = big * u + s * small * np.sin(u), radius * np.cos(u)
            xp, zp = big + s * small * np.cos(u), -radius * np.sin(u)
            xpp, zpp = -s * small * np.sin(u), -radius * np.cos(u)
            rho = (xp ** 2 + zp ** 2) ** 1.5 / np.maximum(np.abs(xp * zpp - zp * xpp), 1e-300)

            def gap(height: float) -> tuple[float, int]:
                d = np.hypot(x, z - height)
                k = int(np.argmin(d))
                return float(d[k] - filament), k

            heights = np.linspace(radius, -radius, 801)
            foot = None
            for i in range(1, len(heights)):
                if gap(heights[i])[0] > 0.0:
                    low, high = heights[i - 1], heights[i]
                    for _ in range(50):
                        middle = 0.5 * (low + high)
                        low, high = (low, middle) if gap(middle)[0] > 0.0 else (middle, high)
                    foot = gap(high)[1]
                    break
            if foot is None or foot == 0:
                continue
            cos_gamma = abs(b * math.sin(beta) - s * radius * math.cos(u[foot]) * math.cos(beta)) / length
            found.append({"beta": float(beta_degrees), "apex": int(s),
                          "wedge": 2.0 * math.degrees(math.atan2(abs(xp[foot]), abs(zp[foot]))),
                          "branch_radius": float(rho[foot] - filament),
                          "gamma": math.degrees(math.acos(min(cos_gamma, 1.0)))})
    return found


def refined_residual(gamma_degrees: float, filament: float, pixel: float, spacing: float) -> float:
    """The largest value error (m) at a sub-ray's refined point, on a straight filament at ``gamma`` to the ray, the
    edge within a sub-cell's reach; the refinement as ``_sampled.pieces`` takes it: the descent's lowest sample, within
    half a spacing of the least on this symmetric profile, then the parabola's least through the outer three of five
    values half a spacing apart about it and two Newton steps on their quartic, each at most half a spacing, or the
    inner three's least where it is lower than the quartic's own value. ``gamma`` is kept for the profile's shape."""
    gamma = math.radians(gamma_degrees)
    sine = math.sin(gamma)
    worst = 0.0
    for b in filament + np.linspace(-0.18, 0.18, 7) * pixel:
        def f(t: float) -> float:
            return math.sqrt(b * b + (sine * t) ** 2) - filament
        for held in np.linspace(-0.5 * spacing, 0.5 * spacing, 401):
            fl2, fl1, f0, fh1, fh2 = (f(held + 0.5 * k * spacing) for k in (-2, -1, 0, 1, 2))
            curvature = fl2 - 2.0 * f0 + fh2
            y = min(max((fl2 - fh2) / curvature, -2.0), 2.0) if curvature > 0.0 else 0.0
            c1 = (fl2 - 8 * fl1 + 8 * fh1 - fh2) / 12.0
            c2 = (-fl2 + 16 * fl1 - 30 * f0 + 16 * fh1 - fh2) / 24.0
            c3 = (-fl2 + 2 * fl1 - 2 * fh1 + fh2) / 12.0
            c4 = (fl2 - 4 * fl1 + 6 * f0 - 4 * fh1 + fh2) / 24.0
            for _ in range(2):
                slope = c1 + y * (2 * c2 + y * (3 * c3 + y * 4 * c4))
                turn = 2 * c2 + y * (6 * c3 + y * 12 * c4)
                if turn > 0.0:
                    y = min(max(y + min(max(-slope / turn, -1.0), 1.0), -2.0), 2.0)
            if f0 + y * (c1 + y * (c2 + y * (c3 + y * c4))) > min(f0, fl1, fh1):
                y = 0.0 if f0 <= min(fl1, fh1) else (-1.0 if fl1 < fh1 else 1.0)
            worst = max(worst, f(held + 0.5 * y * spacing) - f(0.0))
    return worst


def cell_term(rho_px: float, s: float = 1.0 / CORNER_SAMPLES) -> float:
    a = s / math.sqrt(2.0)
    per_cell = (2.0 / 3.0) * a ** 3 / (rho_px * (1.0 + math.sqrt(3.0) / 2.0))
    return 2 * (round(2.0 / s) + 1) * per_cell


def truth_floor(side: int = TRUTH_SIDE, probability: float = PER_PIXEL_PROBABILITY) -> float:
    cells = 2.0 * (math.sqrt(2.0) * side * math.pi / 2.0 + 1.0)
    return math.sqrt(math.log(2.0 / probability) * cells / 2.0) / side ** 2


def rounding_term(dtype: str, pixel: float) -> float:
    unit = 2.0 ** -24 if dtype == "float32" else 0.0
    return 2.0 * math.sqrt(2.0) * 5.0 * unit * DISTANCE / pixel


def main() -> None:
    report: dict = {}
    for name, coil in COILS.items():
        found = notches(coil["radius"], coil["pitch"], coil["filament_radius"])
        rho_min = min(n["branch_radius"] for n in found)
        samples, enclosing = MARCH[name]
        entry: dict = {"notches": len(found), "rho_min_m": rho_min,
                       "wedge_deg": [min(n["wedge"] for n in found), max(n["wedge"] for n in found)],
                       "gamma_min_deg": min(n["gamma"] for n in found), "widths": {}}
        for width in WIDTHS:
            pixel = FIELD / width
            spacing = 2.0 * (enclosing + 2.0 * pixel) / (samples - 1)
            residual = max(refined_residual(n["gamma"], coil["filament_radius"], pixel, spacing) for n in found)
            terms = {"cell": cell_term(rho_min / pixel), "refined": max(2.0 * math.sqrt(2.0) * residual / pixel, 1e-4),
                     "truth": truth_floor(), "truth_signs": 1e-4}
            entry["widths"][width] = {"rho_px": rho_min / pixel, "refined_residual_px": residual / pixel,
                                      **{f"term_{k}": v for k, v in terms.items()},
                                      **{f"line_{dtype}": sum(terms.values()) + rounding_term(dtype, pixel)
                                         for dtype in ("float64", "float32")}}
        report[name] = entry
    print(json.dumps(report, indent=1))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as handle:
            json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
