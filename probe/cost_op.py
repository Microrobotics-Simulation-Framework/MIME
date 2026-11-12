"""The fix round's product at stage 1's size (the review's B1 and M2): Landers' 1.4 mm sphere (338 points, the cell
read) 0.1 a from the wall of the 5 mm straight tube, rule 2's fine cell (35 um), the patch n^3 centred on it, under the
step operator at a stage-1 corner. PROBE_ARM cost<n>-<regime>: a forward product, timed on the runner (median of two
after the compile), for three placements: ``every`` (every interior face, the penalty on the whole solid: the PR's
head before this round, the reference values), ``planned`` (the new default: the penalty and the drive kept 8 cells
into the solid, the capacity planned at the pose) and ``planned_whole`` (the planned capacity, the whole solid: values
identical to ``every``); the values' departures from ``every``. PROBE_ARM rev<n>-<regime>: reverse mode at the new
default only (so the arm's measured peak is reverse mode's): the compiled temporaries of the forward, the transpose,
the vjp in the forces and given values and the vjp in the force points; then the transpose and the vjp run, timed,
the dot-product identity <R x, c> = <x, R^T c> at the operating point and the vjp against the transpose."""
import dataclasses, json, math, os, sys, time
sys.path.insert(0, "probe")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from common_op import A, REGIMES, sphere
from patch_cases import regularised_free
from mime.nodes.flow.fvm._patch import RefinementPatch, own_response, patch_reflection, patch_reflection_transpose, \
    patch_stopping, placed, planned_capacity, snapped_origin
from mime.nodes.flow.fvm._patch_transfers import Quadrature
from mime.nodes.flow.fvm._solve import REPORT_SIZE

arm = os.environ.get("PROBE_ARM", "cost107-l497")
kind = arm[:3] if arm.startswith("rev") else "cost"
size, regime = arm[len(kind):].split("-")
n = int(size)
mu, shift = REGIMES[regime]
h = A / 20.0
R = 2.5e-3
out_dir = "probe/out"
os.makedirs(out_dir, exist_ok=True)


class Tube:
    """The straight tube's wall along x, radius R, its axis at (y, z) = (0, 0); signed distance positive in the solid."""

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
given = tuple(jnp.asarray(-regularised_free(np.asarray(grid.given_points(c, jnp.float64)) + corner, sources,
                                            s["eps2"], s["forces"], mu)[:, c]) for c in range(3))
stopping = patch_stopping(base, jnp.float64)
tap = jnp.zeros(REPORT_SIZE)


def timed(f, *args, repeats=2):
    t = time.time(); out = jax.block_until_ready(f(*args)); first = time.time() - t
    times = []
    for _ in range(repeats):
        t = time.time(); out = jax.block_until_ready(f(*args)); times.append(time.time() - t)
    return out, first, float(np.median(times))


def placement_of(patch):
    return jax.block_until_ready(placed(patch, [Tube()], origin, jnp.asarray(0), jnp.asarray(sources),
                                        jnp.asarray(s["eps2"]), mu, shift=shift, quadrature=quadrature))


result = dict(arm=arm, n=n, cells=n ** 3, h=h, viscosity=mu, shift=shift, devices=str(jax.devices()),
              cpu_count=os.cpu_count(), points=int(sources.shape[0]))
t = time.time()
planned = planned_capacity(base, [Tube()], [origin])
result["plan_seconds"] = time.time() - t
result["planned_capacity"] = planned
default = dataclasses.replace(base, shell_capacity=planned)
if kind == "cost":
    variants = {"every": dataclasses.replace(base, shell_capacity=0, solid_depth=math.inf),
                "planned_whole": dataclasses.replace(base, shell_capacity=planned_capacity(
                    dataclasses.replace(base, solid_depth=math.inf), [Tube()], [origin]), solid_depth=math.inf),
                "planned": default}
    reference = None
    for label, patch in variants.items():
        t = time.time(); placement = placement_of(patch); t_place = time.time() - t
        own = jax.jit(lambda p, f: own_response(p, f))
        _, own_first, own_time = timed(own, placement, forces)
        forward = jax.jit(lambda p, b, f, patch=patch: patch_reflection(patch, p, b, f, None, stopping, tap))
        solved, fwd_first, fwd_time = timed(forward, placement, given, forces)
        row = dict(capacity=patch.capacity(), solid_depth=patch.solid_depth, face_count=patch.face_count(),
                   shell_needed=int(placement.shell_needed), pairs=patch.capacity() * int(sources.shape[0]),
                   place_seconds=t_place, own_seconds=own_time, own_first=own_first, forward_seconds=fwd_time,
                   forward_first=fwd_first, iterations=float(solved.report[0]), residual=float(solved.report[1]),
                   report=np.asarray(solved.report).tolist(), outside=int(solved.outside),
                   flux_share=float(solved.flux_share))
        values = np.asarray(solved.values)
        if reference is None:
            reference = values
        else:
            row["values_rel_vs_every"] = float(np.max(np.abs(values - reference)) / np.max(np.abs(reference)))
            m_ref = float(np.sum(np.asarray(forces) * reference))
            row["metric_rel_vs_every"] = float(np.sum(np.asarray(forces) * values)) / m_ref - 1.0
        result[label] = row
        print(label, json.dumps(row), flush=True)
else:
    placement = placement_of(default)
    rng = np.random.default_rng(0)
    cot = jnp.asarray(rng.normal(size=(sources.shape[0], 3)))

    def forward(p, b, f):
        return patch_reflection(default, p, b, f, None, stopping, tap).values

    def transpose(p, c):
        return patch_reflection_transpose(default, p, c, None, stopping, tap)

    def reverse(p, b, f, c):
        _, pull = jax.vjp(lambda bb, ff: forward(p, bb, ff), b, f)
        return pull(c)

    def reverse_points(p, b, f, c):
        _, pull = jax.vjp(lambda q: forward(p._replace(points=q), b, f), p.points)
        return pull(c)

    compiled = {}
    for name, fn, args in (("forward", forward, (placement, given, forces)), ("transpose", transpose, (placement, cot)),
                           ("vjp", reverse, (placement, given, forces, cot)),
                           ("vjp_points", reverse_points, (placement, given, forces, cot))):
        analysis = jax.jit(fn).lower(*args).compile().memory_analysis()
        compiled[name] = dict(temp_gib=analysis.temp_size_in_bytes / 2 ** 30,
                              argument_gib=analysis.argument_size_in_bytes / 2 ** 30,
                              output_gib=analysis.output_size_in_bytes / 2 ** 30)
        print(name, json.dumps(compiled[name]), flush=True)
    result["compiled"] = compiled
    result["capacity"] = default.capacity()
    values, fwd_first, fwd_time = timed(jax.jit(forward), placement, given, forces, repeats=1)
    (back_given, back_forces, report), tr_first, tr_time = timed(jax.jit(transpose), placement, cot, repeats=1)
    (vjp_given, vjp_forces), vjp_first, vjp_time = timed(jax.jit(reverse), placement, given, forces, cot, repeats=1)
    left = float(jnp.vdot(values, cot))
    right = sum(float(jnp.vdot(g, b)) for g, b in zip(given, back_given)) + float(jnp.vdot(forces, back_forces))
    scale = float(jnp.linalg.norm(values) * jnp.linalg.norm(cot)) + math.sqrt(
        sum(float(jnp.vdot(g, g)) for g in given) + float(jnp.vdot(forces, forces))) * math.sqrt(
        sum(float(jnp.vdot(b, b)) for b in back_given) + float(jnp.vdot(back_forces, back_forces)))
    agree = max(float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) for a, b in
                zip(list(vjp_given) + [vjp_forces], list(back_given) + [back_forces]))
    result.update(forward_seconds=fwd_time, forward_first=fwd_first, transpose_seconds=tr_time, transpose_first=tr_first,
                  vjp_seconds=vjp_time, vjp_first=vjp_first, transpose_iterations=float(report[0]),
                  identity_relative=abs(left - right) / scale, vjp_vs_transpose=agree)
    print("reverse", json.dumps({k: result[k] for k in ("transpose_seconds", "vjp_seconds", "identity_relative",
                                                        "vjp_vs_transpose", "transpose_iterations")}), flush=True)
json.dump(result, open(f"{out_dir}/{kind}_{arm}.json", "w"), indent=1)
