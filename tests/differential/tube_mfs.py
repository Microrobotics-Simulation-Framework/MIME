"""The method of fundamental solutions for a sphere in a straight tube, at unit viscosity and unit sphere radius, sharing
no code with MIME (the singular Stokeslet of tests/plane_oracles, written there again): singular Stokeslets a few local
steps inside the sphere and outside the tube's wall, collocated in least squares on a denser set.

**The tube** is a cylinder of radius R = 1 / ratio about the z axis, closed by hemispherical caps beyond |z| = L (L =
``half`` R, 2.5 by default): a closed vessel, so the net flux through any section is zero, as in the infinite tube (Liron and Shahar's
image has zero flux too); the disturbance decays exponentially along the axis, and the caps' distance is checked.

**The sphere** is centred at (-(R - 1 - gap), 0, 0): its contact with the wall is at -x. The geometry has two mirror
planes, y = 0 and z = 0, so each rigid motion has a parity under each (U_x even, even; U_y and Omega_z odd in y; U_z
and Omega_y odd in z; Omega_x odd in both), and each parity class is solved alone on the quadrant y > 0, z > 0, its
kernel the sum over the four images ``chi_m G(x, M_m s) M_m``: a sixteenth of the full matrix each.

**Both surfaces are graded** towards the contact as the first probe's MFS: the local step is

    sigma = min(most, beta sqrt(d), lam d, CURVATURE_SHARE r_curv),   d the point's distance to the other surface,

on the sphere, and on the wall the same but growing by GROWTH per unit distance past one radius from the sphere, up to
``FAR_SHARE`` R: the wall far along the tube carries a disturbance that decays over about R / 4. The points are chosen by
a greedy Poisson-disk pass over candidates from an adaptive grid in each patch's own parameters; points within half a
disk of a mirror plane are left out (their images stand for them). Sources sit ``DEPTH`` local steps out of the fluid.

Outputs: the 6 x 6 about the sphere's centre (force and torque on the fluid per unit motion, world frame), the
residual per class, the counts.

The near-contact gates' oracle for a sphere in a tube (``tests/oracles/test_near_contact.py``), first written for the
MIME-ANO-154 follow-up probe and kept here as it ran there. Not collected; one ratio's rows at a level and gaps:

    python tests/differential/tube_mfs.py 0.22 C 2.5 <out.json> 0.1 0.01

``generate_near_contact_references.py`` writes ``references/sphere_tube_mfs.json`` from such rows."""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/, where plane_oracles lives
from plane_oracles import free_stokeslet  # noqa: E402

KAPPA = 0.85
DEPTH = 1.5
CURVATURE_SHARE = 0.4
GROWTH = 0.25
#: The far wall's step, a share of R. Measured on the axis against Bohlin's wall factor (a/R 0.34, level A): 0.3 with
#: the cylinder's half-length 4 R read -1.2 % on the axial translation (the pressure-driven mode leaks through a coarse
#: far wall; a longer vessel leaked more, -2.4 % at 6 R), 0.15 at 2.5 R -9e-4, 0.1 at 2.5 R -7.5e-5. The other entries
#: moved under 3e-5.
FAR_SHARE = 0.1
LEVELS = {"A": {"beta": 0.25, "most": 0.12, "lam": 3.0, "far": 0.1},
          "B": {"beta": 0.18, "most": 0.1, "lam": 2.0, "far": 0.1},
          "C": {"beta": 0.13, "most": 0.085, "lam": 1.5, "far": 0.1},
          "Bfar": {"beta": 0.18, "most": 0.1, "lam": 2.0, "far": 0.07}}
MIRRORS = [np.diag([1.0, 1.0, 1.0]), np.diag([1.0, -1.0, 1.0]), np.diag([1.0, 1.0, -1.0]), np.diag([1.0, -1.0, -1.0])]
#: The parity classes (sign under y -> -y, sign under z -> -z) and the rigid motions in each (U_x..U_z, Omega_x..).
CLASSES = {(1, 1): (0,), (-1, 1): (1, 5), (1, -1): (2, 4), (-1, -1): (3,)}


class Tube:
    def __init__(self, ratio: float, gap: float, half: float):
        self.R = 1.0 / ratio
        self.L = half * self.R
        self.centre = np.array([-(self.R - 1.0 - gap), 0.0, 0.0])

    # -- patches: parameters -> points, normals into the fluid, and each parameter's arc-length scale ---------------
    def sphere(self, u, v):
        """u: the polar angle from the contact direction (-x), v: the azimuth from +y towards +z."""
        p = self.centre + np.stack([-np.cos(u), np.sin(u) * np.cos(v), np.sin(u) * np.sin(v)], -1)
        return p, p - self.centre

    def cylinder(self, u, v):
        """u: the angle round the axis from the contact side (-x), v: z."""
        p = np.stack([-self.R * np.cos(u), self.R * np.sin(u), v], -1)
        n = np.stack([np.cos(u), -np.sin(u), np.zeros_like(u)], -1)
        return p, n

    def cap(self, u, v):
        """u: the angle from the cap's pole (+z), v: the angle round the axis from -x."""
        p = np.stack([-self.R * np.sin(u) * np.cos(v), self.R * np.sin(u) * np.sin(v), self.L + self.R * np.cos(u)], -1)
        return p, -(p - np.array([0.0, 0.0, self.L]))

    # -- distances to the other surface ---------------------------------------------------------------------------
    def to_wall(self, p):
        """A sphere point's distance to the cylinder (the caps are far)."""
        return self.R - np.hypot(p[..., 0], p[..., 1])

    def to_sphere(self, p):
        return np.linalg.norm(p - self.centre, axis=-1) - 1.0

    def sigma(self, p, on_sphere, beta, most, lam):
        """The local step: on the sphere min(most, beta sqrt(d), lam d, CURVATURE_SHARE); on the wall min(lam d,
        max(min(most, beta sqrt(d)), GROWTH (d - 1)), FAR_SHARE R, CURVATURE_SHARE R), d the distance to the other
        surface."""
        if on_sphere:
            d = np.maximum(self.to_wall(p), 1e-12)
            return np.minimum.reduce([np.full(d.shape, most), beta * np.sqrt(d), lam * d,
                                      np.full(d.shape, CURVATURE_SHARE)])
        d = np.maximum(self.to_sphere(p), 1e-12)
        near = np.minimum(most, beta * np.sqrt(d))
        return np.minimum.reduce([lam * d, np.maximum(near, GROWTH * (d - 1.0)), np.full(d.shape, FAR_SHARE * self.R),
                                  np.full(d.shape, CURVATURE_SHARE * self.R)])


def _arc(patch, tube, u0, u1, v0, v1):
    """Each cell's larger arc length (its two parameters' extents on the surface)."""
    if patch == "sphere":
        s = np.maximum(np.sin(u0), np.sin(u1))
        s = np.where((u0 < math.pi / 2) & (u1 > math.pi / 2), 1.0, s)
        return u1 - u0, s * (v1 - v0)
    if patch == "cylinder":
        return tube.R * (u1 - u0), v1 - v0
    s = np.where((u0 < math.pi / 2) & (u1 > math.pi / 2), 1.0, np.maximum(np.sin(u0), np.sin(u1)))
    return tube.R * (u1 - u0), tube.R * s * (v1 - v0)


def candidates(patch, tube, box, step_of, start):
    """Candidate points on a patch's parameter ``box`` from an adaptive grid: cells split (each parameter whose arc
    exceeds half the cell's larger) until the cell's larger arc is at most a third of the local step at its centre and
    corners; the leaves' centres, with the step there."""
    surface = getattr(tube, patch)
    (a0, a1), (b0, b1) = box
    arc_u, arc_v = _arc(patch, tube, np.array([a0]), np.array([a1]), np.array([b0]), np.array([b1]))
    if patch == "sphere":
        arc_v = np.array([b1 - b0])
    na, nb = max(1, int(math.ceil(arc_u[0] / start))), max(1, int(math.ceil(arc_v[0] / start)))
    ua, ub = np.linspace(a0, a1, na + 1), np.linspace(b0, b1, nb + 1)
    U0, V0 = np.meshgrid(ua[:-1], ub[:-1], indexing="ij")
    U1, V1 = np.meshgrid(ua[1:], ub[1:], indexing="ij")
    cells = [c.ravel() for c in (U0, U1, V0, V1)]
    leaves = []
    while len(cells[0]):
        u0, u1, v0, v1 = cells
        du, dv = _arc(patch, tube, u0, u1, v0, v1)
        big = np.maximum(du, dv)
        probe_u = np.stack([0.5 * (u0 + u1), u0, u0, u1, u1], 1)
        probe_v = np.stack([0.5 * (v0 + v1), v0, v1, v0, v1], 1)
        p, _ = surface(probe_u, probe_v)
        local = step_of(p.reshape(-1, 3)).reshape(-1, 5).min(axis=1)
        done = big <= local / 3.0
        leaves.append((0.5 * (u0[done] + u1[done]), 0.5 * (v0[done] + v1[done])))
        keep = ~done
        u0, u1, v0, v1, du, dv, big = (x[keep] for x in (u0, u1, v0, v1, du, dv, big))
        split_u = du > 0.5 * big
        split_v = dv > 0.5 * big
        nu = np.where(split_u, 2, 1)
        nv = np.where(split_v, 2, 1)
        out = [[], [], [], []]
        for iu in (0, 1):
            for iv in (0, 1):
                take = (iu < nu) & (iv < nv)
                wu = (u1 - u0) / nu
                wv = (v1 - v0) / nv
                out[0].append((u0 + iu * wu)[take]); out[1].append((u0 + (iu + 1) * wu)[take])
                out[2].append((v0 + iv * wv)[take]); out[3].append((v0 + (iv + 1) * wv)[take])
        cells = [np.concatenate(o) for o in out]
    u = np.concatenate([l[0] for l in leaves])
    v = np.concatenate([l[1] for l in leaves])
    p, n = surface(u, v)
    return p, n / np.linalg.norm(n, axis=1)[:, None], step_of(p)


def poisson(x, sigma, seed):
    """The greedy Poisson-disk choice (the first probe's mfs_body.chosen's): in bands of the step, smallest first, a
    point kept where no kept point lies within KAPPA times its step."""
    order = np.lexsort((np.random.default_rng(seed).random(len(x)), sigma))
    x, sigma = x[order], sigma[order]
    accepted: list = []
    bands = np.floor(np.log2(sigma / sigma.min()) * 2).astype(int)
    tree = None
    for band in np.unique(bands):
        idx = np.nonzero(bands == band)[0]
        if tree is not None:
            d, _ = tree.query(x[idx], k=1)
            idx = idx[d >= KAPPA * sigma[idx]]
        if not len(idx):
            continue
        cell = KAPPA * sigma[idx].max()
        keys = np.floor(x[idx] / cell).astype(int)
        radius2 = (KAPPA * sigma[idx]) ** 2
        grid: dict = {}
        mine = []
        for k in range(len(idx)):
            kx, ky, kz = keys[k]
            p = x[idx[k]]
            ok = True
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for q in grid.get((kx + dx, ky + dy, kz + dz), ()):
                            d = p - q
                            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < radius2[k]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                grid.setdefault((kx, ky, kz), []).append(p)
                mine.append(idx[k])
        accepted += mine
        tree = cKDTree(x[np.array(accepted)])
    return order[np.array(accepted)]


def points(tube, level, scale, seed):
    """Each surface's points on the quadrant (y, z > 0, half a disk clear of each mirror plane), their normals into
    the fluid, their steps over ``scale`` and whether each is the sphere's."""
    beta, most, lam = level["beta"], level["most"], level["lam"]
    out = []
    for patch, box, on_sphere, start in (
            ("sphere", ((0.0, math.pi), (0.0, math.pi / 2)), True, most),
            ("cylinder", ((0.0, math.pi), (0.0, tube.L)), False, FAR_SHARE * tube.R),
            ("cap", ((0.0, math.pi / 2), (0.0, math.pi)), False, FAR_SHARE * tube.R)):
        def step_of(p, on_sphere=on_sphere):
            return tube.sigma(p, on_sphere, beta, most, lam) / scale
        x, n, s = candidates(patch, tube, box, step_of, start / scale)
        clear = (x[:, 1] >= 0.5 * KAPPA * s) & (x[:, 2] >= 0.5 * KAPPA * s)
        x, n, s = x[clear], n[clear], s[clear]
        keep = poisson(x, s, seed + len(out))
        out.append((x[keep], n[keep], s[keep], np.full(len(keep), on_sphere)))
    return [np.concatenate(c) for c in zip(*out)]


def kernel(colloc, sources, parity, chunk=300):
    """The quadrant's kernel for a parity class: the velocity at each collocation point per unit force at each source
    and at its three images, ``sum_m chi_m G(x, M_m s) M_m``, (3T, 3S)."""
    sy, sz = parity
    chis = (1.0, sy, sz, sy * sz)
    out = np.zeros((3 * len(colloc), 3 * len(sources)))
    for start in range(0, len(colloc), chunk):
        part = colloc[start:start + chunk]
        block = np.zeros((len(part), len(sources), 3, 3))
        for M, chi in zip(MIRRORS, chis):
            block += chi * free_stokeslet(part, sources @ M, 0.0) @ M
        out[3 * start:3 * (start + len(part))] = block.transpose(0, 2, 1, 3).reshape(3 * len(part), 3 * len(sources))
    return out


def rigid(x, centre):
    arm = x - centre
    eye = np.eye(3)
    cols = [np.broadcast_to(eye[i], arm.shape) for i in range(3)] + [np.cross(eye[i], arm) for i in range(3)]
    return np.stack(cols, -1)  # (T, 3, 6)


def mfs(ratio, gap, level, half=2.5, oversample=1.25):
    global FAR_SHARE
    tube = Tube(ratio, gap, half)
    lv = LEVELS[level]
    FAR_SHARE = lv["far"]
    xs, ns, ss, sph_s = points(tube, lv, 1.0, 1)
    xc, _, _, sph_c = points(tube, lv, oversample, 101)
    sources = xs - DEPTH * ss[:, None] * ns
    resistance = np.zeros((6, 6))
    residual = {}
    motions = rigid(xc, tube.centre) * sph_c[:, None, None]
    for parity, cols in CLASSES.items():
        rhs = motions[:, :, cols].reshape(3 * len(xc), len(cols))
        K = kernel(xc, sources, parity)
        fitted, *_ = scipy.linalg.lstsq(K, rhs, lapack_driver="gelsy", check_finite=False, overwrite_a=True)
        del K
        worst = np.zeros(len(cols))
        for start in range(0, len(xc), 600):
            part = kernel(xc[start:start + 600], sources, parity)
            worst = np.maximum(worst, np.abs(part @ fitted - rhs[3 * start:3 * (start + 600)]).max(axis=0))
        residual[f"{parity[0]:+d}{parity[1]:+d}"] = worst.tolist()
        f = fitted.reshape(len(sources), 3, len(cols))[sph_s]
        s = sources[sph_s]
        chis = (1.0, parity[0], parity[1], parity[0] * parity[1])
        for j, col in enumerate(cols):
            force, torque = np.zeros(3), np.zeros(3)
            for M, chi in zip(MIRRORS, chis):
                g = chi * f[:, :, j] @ M
                force += g.sum(axis=0)
                torque += np.cross(s @ M - tube.centre, g).sum(axis=0)
            resistance[:3, col], resistance[3:, col] = force, torque
    return {"resistance": resistance.tolist(), "residual": residual, "sources": int(len(xs)),
            "sphere_sources": int(sph_s.sum()), "collocation": int(len(xc)), "R": tube.R, "half": half,
            "centre": tube.centre.tolist()}


if __name__ == "__main__":
    ratio, level, half = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
    out = Path(sys.argv[4])
    gaps = [1.0 / ratio - 1.0 if g == "axis" else float(g) for g in sys.argv[5:]]
    rows = json.loads(out.read_text()) if out.exists() else {}
    for g in gaps:
        start = time.perf_counter()
        row = mfs(ratio, g, level, half)
        row["seconds"] = time.perf_counter() - start
        rows[f"{ratio!r} {level} {half!r} {g!r}"] = row
        out.write_text(json.dumps(rows))
        print(ratio, level, half, g, row["sources"], row["sphere_sources"], row["collocation"],
              f"{row['seconds']:.1f}s", f"res {max(max(v) for v in row['residual'].values()):.2e}", flush=True)
