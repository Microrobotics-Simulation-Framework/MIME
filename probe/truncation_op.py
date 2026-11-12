"""The fix round's re-run of the review's truncation probe on the implemented solid depth (RefinementPatch.solid_depth,
the penalty and the drive kept together d cells into the solid, its capacity planned by planned_capacity): the own
field on fewer faces, with its error: the plane oracle (the PR's patch_cases: the plane's exact given values,
the image above the wall and the own Stokeslets' negative below, which is consistent with v = -u_own in the solid) with
a solid 14 fine cells deep in the box (10 cells across the radius, the surface 0.2 a = 2 cells above the wall, margin 16:
52^3), the cell read; the coupled resistance's normal and along-wall columns (GMRES at 1e-7 over the patch CG at 1e-8).
Variants: every interior face (the default, the reference); a planned capacity (the faces with 1/K > 0: must agree to
the solves' tolerance); the own field's drive kept only within d fine cells of the wall's surface, the penalty kept deeper
(the PR's rejected shell, here with consistent given values), d = 4, 8; the drive and the penalty both kept only within
d cells (the deep solid left as unpenalised fluid, whose consistent given values make v = -u_own there in the limit),
d = 4, 8. Each variant's evaluated faces counted."""
import dataclasses, json, math, os, sys, time
sys.path[:0] = ["tests", "tests/patch"]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from patch_cases import Case, PlaneSolid, built_patch, image_at, normal_k, regularised_free, resistance
from plane_oracles import brenner
from lubrication_cases import VISCOSITY
from mime.nodes.flow.fvm._patch import RefinementPatch, patch_reflection, patch_stopping, placed, planned_capacity
from mime.nodes.flow.fvm._solve import REPORT_SIZE

case = Case(spacing=0.25, cells=10.0, gap=0.2, margin=16, read="cell")
patch0, placement0, centre, sphere = built_patch(case)
h = patch0.spacing
grid = patch0.grid()
sources = sphere.force + centre
corner = np.asarray(placement0.origin) * h
given_points = [np.asarray(grid.given_points(c, jnp.float64)) + corner for c in range(3)]
face_z = [np.asarray(grid.face_points(c, jnp.float64))[..., 2] + corner[2] for c in range(3)]


def variant(name):
    """The placement of a variant, and its patch: ``every`` the whole solid on every interior face (the reference);
    ``planned`` the whole solid at a planned capacity; ``depth_<d>`` the implemented solid depth with its planned
    capacity; ``drive_<d>`` the whole solid's penalty with the drive cut at d cells (the PR's rejected shell)."""
    whole = dataclasses.replace(patch0, solid_depth=math.inf, shell_capacity=0)
    if name == "every":
        patch = whole
    elif name == "planned":
        patch = dataclasses.replace(whole, shell_capacity=planned_capacity(whole, [PlaneSolid(0.0)], [placement0.origin]))
    elif name.startswith("depth_"):
        cut = dataclasses.replace(patch0, solid_depth=float(name.split("_")[1]))
        patch = dataclasses.replace(cut, shell_capacity=planned_capacity(cut, [PlaneSolid(0.0)], [placement0.origin]))
    else:
        patch = whole
    placement = placed(patch, [PlaneSolid(0.0)], placement0.origin, 0, placement0.points, placement0.blob,
                       placement0.viscosity, quadrature=placement0.quadrature)
    if name.startswith("drive_"):
        depth = float(name.split("_")[1])
        kept = jnp.asarray(-np.asarray(placement.shell_points[:, 2]) <= depth * h)
        placement = placement._replace(shell_weight=jnp.where(kept, placement.shell_weight, 0.0))
    return patch, placement


def reader(patch, placement):
    stopping = patch_stopping(patch, jnp.float64, tolerance=case.tolerance)
    solver = jax.jit(lambda p, b, f: patch_reflection(patch, p, b, f, None, stopping, jnp.zeros(REPORT_SIZE)))
    record = {"iterations": []}

    def reflection(forces):
        given = []
        for c in range(3):
            at = given_points[c]
            above = at[:, 2] > 0.0
            values = np.zeros(at.shape[0])
            values[above] = image_at(at[above], sources, sphere.eps_squared, forces, 0.0, VISCOSITY)[:, c]
            values[~above] = -regularised_free(at[~above], sources, sphere.eps_squared, forces, VISCOSITY)[:, c]
            given.append(jnp.asarray(values))
        out = solver(placement, tuple(given), jnp.asarray(forces))
        record["iterations"].append(float(out.report[0]))
        return np.asarray(out.values)

    return reflection, record


rows = {}
t0 = time.time()
for name in ("every", "planned", "depth_4", "depth_8", "drive_4", "drive_8"):
    t = time.time()
    patch, placement = variant(name)
    reflection, record = reader(patch, placement)
    found = resistance(case, reflection, ("Uz", "Ux"), rtol=1e-7)
    evaluated = int(np.count_nonzero(np.asarray(placement.shell_weight) > 0))
    rows[name] = dict(normal=normal_k(found), along=found["Ux"]["force"][0], products=[found[k]["products"] for k in ("Uz", "Ux")],
                      residuals=[found[k].get("true_residual") for k in ("Uz", "Ux")], faces_evaluated=evaluated,
                      capacity=patch.capacity(), mean_iterations=float(np.mean(record["iterations"])), seconds=time.time() - t)
    print(name, json.dumps(rows[name]), flush=True)
ref = rows["every"]
for name, row in rows.items():
    row["normal_rel"] = row["normal"] / ref["normal"] - 1.0
    row["along_rel"] = row["along"] / ref["along"] - 1.0
out = dict(case=dict(cells=case.cells, gap=case.gap, margin=case.margin, n=patch0.cells, read=case.read, h=h,
                     solid_depth_cells=float((-corner[2]) / h), shell_needed=int(placement0.shell_needed),
                     face_count=patch0.face_count(), brenner=brenner(case.gap)), rows=rows, seconds=time.time() - t0)
json.dump(out, open("probe/out/truncation.json", "w"), indent=1)
print(json.dumps({k: (v["normal_rel"], v["along_rel"], v["faces_evaluated"]) for k, v in rows.items()}))
