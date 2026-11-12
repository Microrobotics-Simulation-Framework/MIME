"""Gate 5 at stage 1's operating point (the fix round, the review's M4; the review's script with the snap read where the
snapped corner flips, 1e-6 fine cells either side, beside the review's mid-interval pair and a jump from the sweep's
two sides extrapolated to the flip): the CI test's sweep (17 poses an eighth of a fine cell apart, the sphere moved
away from a plane wall) at rule 2's fine cell (20 across the radius, a = 0.7 mm, h 35 um), the patch the sphere plus 8
cells a side (56^3), the surface 0.1 a (2 fine cells, the hand-over gap) above the wall at the sweep's start; the plane's
exact steady given values (MIME's regularised image above, the own Stokeslets' negative below); m = sum F . v, F the free
normal translation's forces. PROBE_ARM g5-<regime>-<read>. Under a shift the given values stay the steady image's: the
sweep then tests smoothness only (the lattice moving through the band), not accuracy."""
import json, math, os, sys, time
sys.path.insert(0, "probe")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from common_op import A, REGIMES, sphere
from patch_cases import PlaneSolid, image_at, regularised_free
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_stopping, placed, snapped_origin
from mime.nodes.flow.fvm._patch_transfers import Quadrature
from mime.nodes.flow.fvm._solve import REPORT_SIZE

arm = os.environ.get("PROBE_ARM", "g5-steady-cell")
_, regime, read = arm.split("-")
mu, shift = REGIMES[regime]
cells, margin, gap = 20.0, 8, 0.1
h = A / cells
n = int(math.ceil(2 * (cells + margin)))
patch = RefinementPatch((n, n, n), h, read=read)
grid = patch.grid()
s = sphere(mu)
forces = jnp.asarray(s["forces"])
stopping = patch_stopping(patch, jnp.float64, tolerance=1e-11)


@jax.jit
def solve(placement, given, forces):
    return patch_reflection(patch, placement, given, forces, None, stopping, jnp.zeros(REPORT_SIZE))


def m(centre):
    centre = np.asarray(centre)
    origin = snapped_origin(patch, jnp.asarray(centre))
    sources = s["force"] + centre
    quadrature = Quadrature(jnp.asarray(s["points"] + centre), jnp.asarray(s["weights"]),
                            jnp.asarray(s["owners"], dtype=jnp.int32), jnp.asarray(s["area"]))
    placement = placed(patch, [PlaneSolid(0.0)], origin, 0, jnp.asarray(sources), jnp.asarray(s["eps2"]), mu,
                       shift=shift, quadrature=quadrature)
    corner = np.asarray(origin) * h
    given = []
    for c in range(3):
        at = np.asarray(grid.given_points(c, jnp.float64)) + corner
        above = at[:, 2] > 0.0
        values = np.zeros(at.shape[0])
        values[above] = image_at(at[above], sources, s["eps2"], s["forces"], 0.0, mu)[:, c]
        values[~above] = -regularised_free(at[~above], sources, s["eps2"], s["forces"], mu)[:, c]
        given.append(jnp.asarray(values))
    out = solve(placement, tuple(given), forces)
    return float(jnp.sum(forces * out.values)), float(out.report[0]), int(out.outside), tuple(np.asarray(origin).tolist())


def pose(up):
    return np.array([0.37 * h, 0.21 * h, A * (1.0 + gap) + up * h])


def period_one_amplitude(values, step, keep=None):
    derivative = np.diff(values) / step
    x = (np.arange(derivative.size) + 0.5) * step
    if keep is not None:
        derivative, x = derivative[keep], x[keep]
    basis = np.stack([np.ones_like(x), x, x * x, np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis, derivative, rcond=None)
    return float(np.hypot(coefficients[3], coefficients[4]) / np.mean(np.abs(derivative)))


t0 = time.time()
step = 0.125
offsets = np.arange(17) * step
rows = [m(pose(u)) for u in offsets]
values = np.array([r[0] for r in rows])
origins = [r[3] for r in rows]
snaps = [k for k in range(1, len(offsets)) if origins[k] != origins[k - 1]]
keep = np.asarray([k + 1 not in snaps for k in range(len(offsets) - 1)])
wobble = period_one_amplitude(values, step, keep)
jumps = []
for k in snaps:
    low, high = offsets[k - 1], offsets[k]
    start = m(pose(low))[3]
    for _ in range(50):
        middle = 0.5 * (low + high)
        held = tuple(np.asarray(snapped_origin(patch, jnp.asarray(pose(middle)))).tolist())
        low, high = (middle, high) if held == start else (low, middle)
    flip = 0.5 * (low + high)
    below = m(pose(flip - 1e-6))[0]
    above = m(pose(flip + 1e-6))[0]
    mid = pose(offsets[k] - 0.5 * step)
    pair = abs(m(mid + np.array([0.0, 0.0, 1e-3 * h]))[0] - m(mid - np.array([0.0, 0.0, 1e-3 * h]))[0])
    # The sweep's two sides, each a quadratic through its three poses nearest the flip, extrapolated to it.
    left = np.polyval(np.polyfit(offsets[k - 3:k], values[k - 3:k], 2), flip)
    right = np.polyval(np.polyfit(offsets[k:k + 3], values[k:k + 3], 2), flip)
    jumps.append(dict(flip=float(flip), straddled=abs(above - below) / abs(below), mid_pair=pair / abs(below),
                      extrapolated=abs(right - left) / abs(below)))
result = dict(arm=arm, regime=regime, read=read, viscosity=mu, shift=shift, cells=cells, n=n, h=h, gap=gap,
              wobble=wobble, snap_jumps=jumps, snaps=snaps, values=values.tolist(), iterations=[r[1] for r in rows],
              outside=[r[2] for r in rows], seconds=time.time() - t0)
json.dump(result, open(f"probe/out/gate5_{arm}.json", "w"), indent=1)
print(arm, "wobble", wobble, "snap jumps", jumps, "snaps", snaps, "iterations", result["iterations"][:3], "s", result["seconds"])
