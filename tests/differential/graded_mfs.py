"""The method of fundamental solutions for an ellipsoid above the plane z = 0 at unit viscosity (the sphere's ring MFS of
tests/plane_oracles generalised; its kernels reused, nothing of MIME's): singular Stokeslets with Blake's image at
sources ``depth`` local steps inside the surface, collocated in least squares on a denser set. Both point sets are chosen
on the surface by a greedy Poisson-disk pass over dense candidates (rings in the polar angle about the body's x axis,
each ring as many points as its circumference holds at the candidate spacing), with the local step

    sigma(x) = min(most, beta sqrt(z(x)), CURVATURE_SHARE r_min(x)),

z(x) the surface point's height above the wall (the sphere's ``beta sqrt(gap + 1 - cos theta)``: the lubrication's
scale near the contact) and r_min its smaller principal radius of curvature (so a spheroid's tips are resolved).
Units: lengths in the minor semi-axis. Outputs as mfs_sphere.mfs: the 6 x 6 about the centre (force and torque on the
fluid per unit motion, world frame), the loads on the held body in u = (z, 0, 0) and u = (1, 0, 0).

The near-contact gates' oracle for a sphere above a plane (``tests/oracles/test_near_contact.py``), first written for the
MIME-ANO-154 probe and kept here as it ran there. Not collected; one sphere row at a level and gaps:

    python tests/differential/graded_mfs.py 1.0 0.0 C <out.json> 0.3 0.1 0.01

``generate_near_contact_references.py`` writes ``references/sphere_plane_graded_mfs.json`` from such rows."""
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from plane_oracles import blake, free_stokeslet  # noqa: E402

CURVATURE_SHARE = 0.4
KAPPA = 0.85
#: The spheroids' three levels (L1 to L3) and the sphere table's (A to C). ``lam`` holds the step under ``lam`` times the
#: local height above the wall, so that the sources near the contact sit at most 1.5 lam local gaps deep (measured on the
#: sphere at h/a 0.01: sources 2.7 gaps deep, -6e-4 on the normal against Brenner; 3.75 gaps, -2.1e-3; 5.3, -3.7e-2).
LEVELS = {"L1": {"beta": 0.35, "most": 0.15, "depth": 1.5, "lam": 3.0},
          "L2": {"beta": 0.3, "most": 0.12, "depth": 1.5, "lam": 3.0},
          "L3": {"beta": 0.25, "most": 0.1, "depth": 1.5, "lam": 3.0},
          "A": {"beta": 0.25, "most": 0.12, "depth": 1.5, "lam": 3.0},
          "B": {"beta": 0.18, "most": 0.1, "depth": 1.5, "lam": 2.0},
          "C": {"beta": 0.13, "most": 0.085, "depth": 1.5, "lam": 1.5}}


def tilt(degrees: float) -> np.ndarray:
    """World from body: the body turned about y by ``degrees`` (its x axis raised towards +z)."""
    t = math.radians(degrees)
    return np.array([[math.cos(t), 0.0, -math.sin(t)], [0.0, 1.0, 0.0], [math.sin(t), 0.0, math.cos(t)]])


def support(axes: np.ndarray, n: np.ndarray):
    """Points with outward unit normals ``n`` (body frame) and their principal radii (smaller, larger)."""
    a2 = axes ** 2
    h = np.sqrt(np.sum(a2 * n * n, axis=-1))
    points = a2 * n / h[:, None]
    # radii: the support's Hessian A^2/h - (A^2 n)(A^2 n)^T / h^3 on the tangent plane
    hessian = np.diag(a2)[None] / h[:, None, None] \
        - np.einsum("ni,nj->nij", a2 * n, a2 * n) / h[:, None, None] ** 3
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(n, helper); t1 /= np.linalg.norm(t1, axis=1)[:, None]
    t2 = np.cross(n, t1)
    frame = np.stack([t1, t2], axis=1)
    radii = np.linalg.eigvalsh(np.einsum("nai,nij,nbj->nab", frame, hessian, frame))
    return points, radii


def candidates(axes, spacing, theta_range=(0.0, math.pi), phi_range=None):
    """Surface points (and outward normals) about ``spacing`` apart: rings in the polar angle about x (theta), within
    ``theta_range`` and, where given, the azimuth ``phi_range`` (phi from y towards z)."""
    c, b = axes[0], axes[1]
    out = []
    theta = theta_range[0]
    if theta <= 0.0:
        out.append((0.0, 0.0))
    while True:
        ds = math.sqrt((c * math.sin(theta)) ** 2 + (b * math.cos(theta)) ** 2)
        theta += spacing / max(ds, 1e-12)
        if theta >= min(theta_range[1], math.pi - 0.5 * spacing / c):
            break
        count = max(6, int(math.ceil(2 * math.pi * b * math.sin(theta) / spacing)))
        step = 2 * math.pi / count
        phase = np.random.default_rng(len(out)).uniform(0, step)
        if phi_range is None:
            phis = phase + step * np.arange(count)
        else:
            phis = np.arange(phi_range[0] + phase, phi_range[1], step)
        out += [(theta, p) for p in phis]
    if theta_range[1] >= math.pi:
        out.append((math.pi, 0.0))
    angles = np.array(out).reshape(-1, 2)
    th, ph = angles[:, 0], angles[:, 1]
    x = np.stack([axes[0] * np.cos(th), axes[1] * np.sin(th) * np.cos(ph), axes[2] * np.sin(th) * np.sin(ph)], 1)
    n = x / axes ** 2
    return x, n / np.linalg.norm(n, axis=1)[:, None]


def region_of(axes, x, pad):
    """The (theta, phi) box holding the points ``x``, padded by ``pad`` of arc length; phi in full where the box nears a
    pole or wraps."""
    th = np.arccos(np.clip(x[:, 0] / axes[0], -1.0, 1.0))
    ph = np.arctan2(x[:, 2] / axes[2], x[:, 1] / axes[1])
    pad_theta = pad / axes[1]
    theta_range = (max(0.0, th.min() - pad_theta), min(math.pi, th.max() + pad_theta))
    least_sine = min(math.sin(theta_range[0]), math.sin(theta_range[1])) if theta_range[0] > 0 and \
        theta_range[1] < math.pi else 0.0
    if theta_range[0] < math.pi / 2 < theta_range[1]:
        least_sine = min(least_sine if least_sine else 1.0, math.sin(theta_range[0]), math.sin(theta_range[1]))
    if least_sine < 0.2 or ph.max() - ph.min() > 1.5 * math.pi:
        return theta_range, None
    pad_phi = pad / (axes[1] * least_sine)
    return theta_range, (ph.min() - pad_phi, ph.max() + pad_phi)


def chosen(axes, rotation, centre, beta, most, scale, seed, lam=None):
    """The greedy Poisson-disk choice at the local step sigma(x) / scale: (body points, normals, sigma). Candidates come
    in levels, each half the last's spacing and generated only over the region whose step needs it, each kept where
    it is at most a third of the local step."""
    sig_all, pts, nrm = [], [], []
    level = most / scale / 3.0
    region = ((0.0, math.pi), None)
    upper = np.inf
    while True:
        x, n = candidates(axes, level, *region)
        _, radii = support(axes, n)
        z = (x @ rotation.T + centre)[:, 2]
        sigma = np.minimum.reduce([np.full(len(x), most), beta * np.sqrt(np.maximum(z, 1e-12)),
                                   CURVATURE_SHARE * radii[:, 0],
                                   np.full(len(x), np.inf) if lam is None else lam * np.maximum(z, 1e-12)]) / scale
        keep = (sigma >= 3.0 * level) & (sigma < upper)
        pts.append(x[keep]); nrm.append(n[keep]); sig_all.append(sigma[keep])
        finer = sigma < 3.0 * level
        if not finer.any():
            break
        region = region_of(axes, x[finer], 4.0 * level)
        upper = 3.0 * level
        level /= 2.0
    x, n, sigma = np.concatenate(pts), np.concatenate(nrm), np.concatenate(sig_all)
    order = np.lexsort((np.random.default_rng(seed).random(len(x)), sigma))
    x, n, sigma = x[order], n[order], sigma[order]
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
        points = x[idx]
        for k in range(len(idx)):
            kx, ky, kz = keys[k]
            p = points[k]
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
    accepted_index = np.array(accepted)
    return x[accepted_index], n[accepted_index], sigma[accepted_index]


def assembled(colloc, sources, chunk=400):
    """The singular Stokeslet plus Blake's image, every collocation point against every source, in row chunks."""
    kernel = np.empty((3 * len(colloc), 3 * len(sources)))
    for start in range(0, len(colloc), chunk):
        part = colloc[start:start + chunk]
        kernel[3 * start:3 * (start + len(part))] = (free_stokeslet(part, sources, 0.0) + blake(part, sources)) \
            .transpose(0, 2, 1, 3).reshape(3 * len(part), 3 * len(sources))
    return kernel


def motions(x, centre):
    arm = x - centre
    eye = np.eye(3)
    cols = [np.broadcast_to(eye[i], arm.shape) for i in range(3)] + [np.cross(eye[i], arm) for i in range(3)]
    return np.stack(cols, -1).reshape(-1, 6)


def mfs(axes, tilt_degrees, gap, beta, most, depth, oversample=1.25, lam=None):
    axes = np.asarray(axes, dtype=float)
    rotation = tilt(tilt_degrees)
    down = rotation.T @ np.array([0.0, 0.0, -1.0])
    lowest = math.sqrt(np.sum(axes ** 2 * down ** 2))
    centre = np.array([0.0, 0.0, lowest + gap])
    xs, ns, ss = chosen(axes, rotation, centre, beta, most, 1.0, 1, lam)
    xc, _, _ = chosen(axes, rotation, centre, beta, most, oversample, 2, lam)
    sources = (xs - depth * ss[:, None] * ns) @ rotation.T + centre
    colloc = xc @ rotation.T + centre
    kernel = assembled(colloc, sources)
    shear = np.zeros((len(colloc), 3)); shear[:, 0] = colloc[:, 2]
    uniform = np.zeros((len(colloc), 3)); uniform[:, 0] = 1.0
    rhs = np.concatenate([motions(colloc, centre), shear.reshape(-1, 1), uniform.reshape(-1, 1)], axis=1)
    fitted, *_ = scipy.linalg.lstsq(kernel, rhs, lapack_driver="gelsy", check_finite=False, overwrite_a=True,
                                    overwrite_b=False)
    del kernel
    residual = np.abs(assembled(colloc, sources) @ fitted - rhs).max(axis=0)
    wrench = motions(sources, centre).T @ fitted
    return {"resistance": wrench[:, :6].tolist(), "shear_load": wrench[:, 6].tolist(),
            "uniform_load": wrench[:, 7].tolist(), "residual": residual.tolist(), "sources": len(sources),
            "collocation": len(colloc), "centre": centre.tolist(), "lowest": lowest}


if __name__ == "__main__":
    import json
    import os
    aspect, tilt_degrees, level = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    out = Path(sys.argv[4])
    gaps = [float(g) for g in sys.argv[5:]]
    rows = json.loads(out.read_text()) if out.exists() else {}
    for g in gaps:
        start = time.perf_counter()
        row = mfs((aspect, 1.0, 1.0), tilt_degrees, g, **LEVELS[level])
        row["seconds"] = time.perf_counter() - start
        rows[f"{aspect!r} {tilt_degrees!r} {level} {g!r}"] = row
        out.write_text(json.dumps(rows))
        print(aspect, tilt_degrees, level, g, row["sources"], row["collocation"], f"{row['seconds']:.1f}s",
              f"res {max(row['residual'][:6]):.2e}", flush=True)
    _ = os
