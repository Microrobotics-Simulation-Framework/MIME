"""The solid depth in the tube at stage 1's box (the fix round, after the cost probe, run 43239921793): there the depth
8 moved the reflection's metric by -7.9 to -10.7 % against the whole solid's, where the plane oracle read -0.05 %. That
probe's given values were the steady regularised Stokeslets' negative, while under a shift the own field is the
Brinkman one: its given values are not consistent with the solid's v = -u_own, which the depth leaves to them.
PROBE_ARM tube<n>-<regime>-<given>: the patch n^3 in the 5 mm tube, the sphere 0.1 a from the wall (338 points, the cell
read), given values ``stokes`` (the cost probe's) or ``own`` (the own response's closed form at the given faces, the
Brinkman difference included: consistent); the metric m = sum F . v and the values at the read points, depth 8 with
its planned capacity against the whole solid on every face."""
import dataclasses, json, math, os, sys, time
sys.path.insert(0, "probe")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from common_op import A, REGIMES, sphere
from patch_cases import regularised_free
from mime.nodes.flow.fvm import _patch
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_stopping, placed, planned_capacity, \
    snapped_origin
from mime.nodes.flow.fvm._patch_transfers import Quadrature
from mime.nodes.flow.fvm._solve import REPORT_SIZE

arm = os.environ.get("PROBE_ARM", "tube107-l497-own")
size, regime, kind = arm[4:].split("-")
n = int(size)
mu, shift = REGIMES[regime]
h = A / 20.0
R = 2.5e-3


class Tube:
    region = "wall"

    def signed_distance(self, points):
        return jnp.sqrt(points[..., 1] ** 2 + points[..., 2] ** 2) - R


s = sphere(mu)
centre = np.array([0.0, 0.0, -(R - A * 1.1)])
sources = s["force"] + centre
quadrature = Quadrature(jnp.asarray(s["points"] + centre), jnp.asarray(s["weights"]),
                        jnp.asarray(s["owners"], dtype=jnp.int32), jnp.asarray(s["area"]))
forces = jnp.asarray(s["forces"])
base = RefinementPatch((n, n, n), h, read="cell")
origin = snapped_origin(base, jnp.asarray(centre))
corner = np.asarray(origin) * h
grid = base.grid()
given = []
for c in range(3):
    at = np.asarray(grid.given_points(c, jnp.float64)) + corner
    if kind == "stokes":
        given.append(jnp.asarray(-regularised_free(at, sources, s["eps2"], s["forces"], mu)[:, c]))
    else:
        place = _patch._OwnPlace(jnp.asarray(at), jnp.full(at.shape[0], c, jnp.int32), jnp.asarray(sources),
                                 jnp.asarray(s["eps2"]), jnp.asarray(mu), jnp.asarray(shift))
        given.append(-_patch._own_values(place, forces))
given = tuple(given)
stopping = patch_stopping(base, jnp.float64)
tap = jnp.zeros(REPORT_SIZE)
result = dict(arm=arm, n=n, viscosity=mu, shift=shift, given=kind)
values = {}
t0 = time.time()
for label, patch in (("whole", dataclasses.replace(base, shell_capacity=0, solid_depth=math.inf)),
                     ("depth8", dataclasses.replace(base, shell_capacity=planned_capacity(base, [Tube()], [origin])))):
    placement = placed(patch, [Tube()], origin, jnp.asarray(0), jnp.asarray(sources), jnp.asarray(s["eps2"]), mu,
                       shift=shift, quadrature=quadrature)
    t = time.time()
    solved = jax.block_until_ready(jax.jit(lambda p, b, f, patch=patch: patch_reflection(
        patch, p, b, f, None, stopping, tap))(placement, given, forces))
    values[label] = np.asarray(solved.values)
    result[label] = dict(seconds=time.time() - t, iterations=float(solved.report[0]), residual=float(solved.report[1]),
                         capacity=patch.capacity(), needed=int(placement.shell_needed),
                         metric=float(np.sum(np.asarray(forces) * values[label])))
    print(label, json.dumps(result[label]), flush=True)
result["values_rel"] = float(np.max(np.abs(values["depth8"] - values["whole"])) / np.max(np.abs(values["whole"])))
result["metric_rel"] = result["depth8"]["metric"] / result["whole"]["metric"] - 1.0
result["seconds"] = time.time() - t0
os.makedirs("probe/out", exist_ok=True)
json.dump(result, open(f"probe/out/tube_{arm}.json", "w"), indent=1)
print(arm, "values_rel", result["values_rel"], "metric_rel", result["metric_rel"])
