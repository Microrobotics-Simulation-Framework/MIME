"""The BEM's error budget at A (a diagnostic, not a register entry): the pieces the study's scripts share.

**What is measured.** The regularised-Stokeslet BEM (``mime.nodes.hydro.bem``) puts point forces at the samples, which
are also its quadrature points, with the blob ``eps = blob x h`` (``h`` the samples' mean spacing, the square root of
their area over their count), and adds the tube's image system, read from a table for every pair and *not*
regularised. Two errors are separated here:

- **regularisation:** the continuous regularised problem (exact quadrature at a fixed ``eps``) differs from the Stokes
  problem by a term in ``eps`` (and, near the wall, by the image's own regularisation, which the node leaves out);
- **quadrature:** the discrete sum differs from the continuous regularised integral by a term that depends on ``h``
  and on ``eps / h``.

**The grid** (``grid_level``): at each spacing, the samples and the wall's correction ``G`` are assembled once per pose
and solved at each blob fraction ``BLOBS`` (the node's own assembly and solve, ``a0f_solves.Level``'s form), in the
tube and in free space; A at the floored pose over the eight phases of the half turn, and the two controls, a sphere in
free space and a sphere on the tube's axis (A's tube, the shipped table).

**The sphere's regularisation alone** (``sphere_regularised_drag``): a sphere's drag with the regularised Stokeslet
and exact quadrature, from the rotational symmetry. The operator maps the translation's two poloidal fields of degree
one (``e_z`` and ``n_z n``) onto themselves, so the continuous first-kind equation is a 2 x 2 system whose entries are
the velocity of each density at the pole and at the equator, each a smooth surface integral evaluated here to
quadrature convergence. Independent of the BEM's code; its own controls are the limit ``eps -> 0`` (Stokes' ``6 pi mu
a``) and a discrete solve at ``h << eps``.

**Blake's plane-wall image** (``blake``): the image of a point force at height ``h`` above a no-slip plane (Blake
1971), and the same with every image singularity regularised by the free part's blob (the image Stokeslet ``S_eps``,
the source dipole from the blob-smoothed ``1/R``, the Stokeslet doublet from ``S_eps``; the coefficients ``h`` and
``h^2`` of the source's height kept): **one regularised form, not the unique one**. Its no-slip residual on the wall is
reported, since it is not zero (the exact smoothing of Blake's solution differs from it at ``O(eps^2 / h^2)``).

No measured speed enters anything here."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

#: The grid (the brief's): nominal spacings (mm), coarse to fine, and the blob fractions.
SPACINGS_MM = (0.2, 0.16, 0.128, 0.1024)
BLOBS = (0.25, 0.35, 0.5, 0.7, 1.0)
FROZEN_SPACING_MM, FROZEN_BLOB = 0.16, 0.5
#: A's tube (the 1/4 inch tube's inner radius) and the controls' sphere, 0.3 of it (Haberman & Sayre's formula agrees
#: with Bohlin's series there to about 1e-5, ``haberman_sayre_check``).
TUBE_RADIUS_M = 3.175e-3
SPHERE_RATIO = 0.3
VISCOSITY = 1e-3


# ---- assembly and solves ---------------------------------------------------------------------------------------------
class Case:
    """A built experiment's BEM at one discretisation: for a pose, the samples, their mean spacing and the wall's
    correction assembled once (``parts``), then the resistance at any blob fraction, with the wall or without it."""

    def __init__(self, built: Any, drag_node: str) -> None:
        import jax
        import jax.numpy as jnp

        from mime.build._maddening import parameter_leaves
        from mime.core import quaternion
        from mime.nodes.hydro.bem import wall_correction_named
        from mime.nodes.hydro.bem._solve import PRECISION, point_forces, rigid_motions, stokeslet_matrix

        self.built, self.drag_node = built, drag_node
        self.node = built.graph.get_node(drag_node)
        self.drag = parameter_leaves(built.graph)[drag_node]
        constants = self.node.constants()
        self.wall = wall_correction_named(constants["wall_kind"], constants["wall_kind_version"])
        node, wall = self.node, self.wall

        def assemble(pose: Any, drag: Any, with_wall: bool) -> dict[str, Any]:
            p = node._step_constants(drag)  # noqa: SLF001 - what the node's step reads
            held = node.samples(p)
            points = pose[:3] + jnp.matmul(held.points, quaternion.rotation_matrix(pose[3:]).T, precision=PRECISION)
            out = {"points": points, "area_per_sample": jnp.sum(held.weights) / held.weights.shape[0],
                   "viscosity": p["viscosity"], "centre": pose[:3]}
            if with_wall and wall is not None:
                corrected = wall.correction(points, {name: p[f"wall_{name}"] for name in wall.parameters},
                                            p["viscosity"])
                out["G"], out["accepted"] = corrected.matrix, corrected.accepted
            return out

        def solve(points: Any, eps2: Any, viscosity: Any, centre: Any, wall_matrix: Any) -> Any:
            matrix = stokeslet_matrix(points, eps2, viscosity)
            if wall_matrix is not None:
                matrix = matrix + wall_matrix
            motions = rigid_motions(points, centre)
            found = jnp.matmul(motions.T, point_forces(matrix, motions), precision=PRECISION)
            return (found + found.T) / 2.0

        self._assemble = jax.jit(assemble, static_argnums=2)
        self._solve = jax.jit(solve)
        self._solve_free = jax.jit(lambda points, eps2, viscosity, centre: solve(points, eps2, viscosity, centre, None))

    def parts(self, pose: Any, with_wall: bool = True) -> dict[str, Any]:
        import jax

        return jax.block_until_ready(self._assemble(pose, self.drag, with_wall))

    def solve(self, parts: dict[str, Any], blob: float, with_wall: bool, wall_matrix: Any = None) -> np.ndarray:
        """The resistance (SI) at ``blob``; ``wall_matrix`` replaces the assembled ``G`` where given."""
        import jax

        eps2 = blob * blob * parts["area_per_sample"]
        if with_wall:
            held = parts["G"] if wall_matrix is None else wall_matrix
            found = self._solve(parts["points"], eps2, parts["viscosity"], parts["centre"], held)
        else:
            found = self._solve_free(parts["points"], eps2, parts["viscosity"], parts["centre"])
        return np.asarray(jax.block_until_ready(found), dtype=np.float64)


def describe(parts: dict[str, Any]) -> dict[str, Any]:
    area = float(np.asarray(parts["area_per_sample"]))
    return {"samples": int(parts["points"].shape[0]), "mean_spacing_m": math.sqrt(area),
            "eps_m": {f"{b}": b * math.sqrt(area) for b in BLOBS}}


def smallest_eigenvalue_scaled(r: np.ndarray, length: float) -> float:
    """``r`` scaled by ``mu L``, ``mu L^2``, ``mu L^3`` (``bem.nondimensional``'s form at ``length``), its smallest
    eigenvalue."""
    scale = np.diag([math.sqrt(VISCOSITY * length)] * 3 + [math.sqrt(VISCOSITY * length ** 3)] * 3)
    inverse = np.linalg.inv(scale)
    held = inverse @ r @ inverse
    return float(np.linalg.eigvalsh((held + held.T) / 2.0).min())


# ---- the experiments -------------------------------------------------------------------------------------------------
def a_rung(spacing_mm: float) -> Any:
    """A's closed rung (``a0f_solves.Floored``: FL-9 floored in the 1/4 inch tube, margin 0.1 mm) at ``spacing_mm``;
    its blob leaf is unused here (each solve takes its own)."""
    import a0f_solves

    return a0f_solves.Floored(spacing_mm, 0.5)


def sphere_experiment(spacing_mm: float, tube: bool, ratio: float = SPHERE_RATIO) -> Any:
    """A sphere of ``ratio`` of A's tube radius at rest on the axis of A's tube (along z, the tube wall kind on the
    shipped table) or, ``tube`` false, in unbounded water; ``Drag.BEM`` at ``spacing_mm``, nothing else."""
    from mime.dynamics import RigidInertial
    from mime.experiment import Experiment
    from mime.interactions import Drag
    from mime.scene import Scene, Sphere, Tube, Z
    from mime.things import Body, Medium, Part, State, Structure
    from mime.units import kg, kg_m2, m, m_per_s, m_per_s2, mm, mPa_s, ms, rad_per_s

    experiment = Experiment("bem_budget_sphere", scene=Scene.new(up=Z, units="m"), gravity=m_per_s2(0.0), dt=ms(1))
    if tube:
        experiment.add_structure(Structure("tube", geometry=Tube(axis=(0.0, 0.0, 1.0), inner_radius=m(TUBE_RADIUS_M),
                                                                 wall=mm(0.5), length=mm(200.0))))
        experiment.add_medium(Medium("water", fills="tube.lumen"))
    else:
        experiment.add_medium(Medium("water"))
    experiment.declare("water.viscosity", mPa_s(1.0), reason="water")
    experiment.add_body(Body("ball", dynamics=RigidInertial(),
                             parts={"shell": Part(Sphere(radius=m(ratio * TUBE_RADIUS_M)))},
                             initial=State(position=m((0.0, 0.0, 0.0)), orientation=(1.0, 0.0, 0.0, 0.0),
                                           velocity=m_per_s((0.0, 0.0, 0.0)),
                                           angular_velocity=rad_per_s((0.0, 0.0, 0.0)))))
    experiment.declare("ball.mass", kg(1e-5), reason="a millimetre body's order")
    experiment.declare("ball.inertia", kg_m2((1e-11, 1e-11, 1e-11)), reason="a millimetre body's order")
    experiment.attach(Drag.BEM(spacing=mm(spacing_mm), blob=0.5, wall_kind="tube" if tube else "free_space"),
                      body="ball", medium="water")
    return experiment.build()


def sphere_pose(built: Any) -> Any:
    from mime.build._maddening import node_state

    return node_state(built.graph, "ball")["pose"]


def haberman_sayre(ratio: float) -> float:
    """``tests/tube_cases.haberman_sayre`` (the form commonly quoted from Haberman & Sayre 1958)."""
    return (1 - 0.75857 * ratio ** 5) / (1 - 2.1050 * ratio + 2.0865 * ratio ** 3 - 1.7068 * ratio ** 5
                                          + 0.72603 * ratio ** 6)


def bohlin(ratio: float) -> float:
    """Bohlin's (1960) series for the same factor, as commonly quoted: a second source for the control's value."""
    l = ratio
    return 1.0 / (1 - 2.10444 * l + 2.08877 * l ** 3 - 0.94813 * l ** 5 - 1.372 * l ** 6 + 3.87 * l ** 8
                  - 4.19 * l ** 10)


# ---- the sphere's regularisation alone -------------------------------------------------------------------------------
def _panels(eps: float, per_panel: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0, pi] in the polar angle about the target, graded at the scale ``eps``."""
    edges = [0.0]
    step = eps / 8.0
    while edges[-1] + step < math.pi:
        edges.append(edges[-1] + step)
        step *= 1.6
    edges.append(math.pi)
    x, w = np.polynomial.legendre.leggauss(per_panel)
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        nodes.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
        weights.append(0.5 * (hi - lo) * w)
    return np.concatenate(nodes), np.concatenate(weights)


def _velocity(target: np.ndarray, density: Any, eps: float, per_panel: int, around: int) -> np.ndarray:
    """``8 pi mu u`` at ``target`` on the unit sphere, from ``density(y)`` (``(P, 3)`` from ``(P, 3)`` points), with the
    regularised Stokeslet at ``eps``: the polar angle measured from the target, graded panels, and the trapezoid rule
    round it."""
    theta, w_theta = _panels(eps, per_panel)
    psi = 2.0 * math.pi * np.arange(around) / around
    axis = target / np.linalg.norm(target)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    y = (st[..., None] * (np.cos(psi)[None, :, None] * e1 + np.sin(psi)[None, :, None] * e2)
         + ct[..., None] * axis).reshape(-1, 3)
    weight = (w_theta[:, None] * np.sin(theta)[:, None] * np.full((1, around), 2.0 * math.pi / around)).reshape(-1)
    r = target - y
    r2 = np.sum(r * r, axis=1)
    t = density(y)
    denom = (r2 + eps * eps) ** 1.5
    u = ((r2 + 2 * eps * eps)[:, None] * t + r * np.sum(r * t, axis=1)[:, None]) / denom[:, None]
    return np.sum(u * weight[:, None], axis=0)


def sphere_regularised_drag(eps_over_a: float, per_panel: int = 48, around: int = 32) -> float:
    """The drag of a translating sphere with the regularised Stokeslet at ``eps`` and exact quadrature, over Stokes'
    ``6 pi mu a U``. See the module's docstring."""
    pole, equator = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    densities = (lambda y: np.broadcast_to(np.array([0.0, 0.0, 1.0]), y.shape),
                 lambda y: y[:, 2:3] * y)
    matrix = np.empty((2, 2))
    for column, density in enumerate(densities):
        at_pole = _velocity(pole, density, eps_over_a, per_panel, around)[2]
        at_equator = _velocity(equator, density, eps_over_a, per_panel, around)
        a = at_equator[2]
        matrix[0, column], matrix[1, column] = a, at_pole - a
    # 8 pi mu u = matrix @ (alpha, beta): u = e_z everywhere on the sphere (A = 8 pi, B = 0 at mu = 1)
    alpha, beta = np.linalg.solve(matrix, np.array([8.0 * math.pi, 0.0]))
    force = alpha * 4.0 * math.pi + beta * 4.0 * math.pi / 3.0
    return float(force / (6.0 * math.pi))


# ---- Blake's plane-wall image ----------------------------------------------------------------------------------------
def _stokeslet(r: Any, eps2: Any) -> Any:
    import jax.numpy as jnp

    r2 = jnp.dot(r, r)
    return ((r2 + 2.0 * eps2) * jnp.eye(3) + jnp.outer(r, r)) / (r2 + eps2) ** 1.5


def _potential(r: Any, eps2: Any) -> Any:
    """``1/R`` smoothed by the Cortez blob (``(2 R^2 + 3 eps^2) / (2 (R^2 + eps^2)^{3/2})``, ``-4 pi G_eps``); ``1/R``
    at ``eps = 0``. Its Laplacian is checked against ``-4 pi phi_eps`` in ``blake_checks``."""
    import jax.numpy as jnp

    r2 = jnp.dot(r, r)
    return (2.0 * r2 + 3.0 * eps2) / (2.0 * (r2 + eps2) ** 1.5)


def blake_image(x: Any, y: Any, eps2: Any) -> Any:
    """``8 pi mu`` times the image part of Blake's (1971) system for a point force at ``y`` above the no-slip plane
    ``z = 0`` (fluid ``z > 0``), every singularity regularised at ``eps2`` (``eps2 = 0``: Blake's own): ``-S(R) + 2 h
    D_jk d/dR_k [h R_i / R^3 - S_i3(R)]``, ``R = x - y*``, ``D = diag(1, 1, -1)``. Finite at ``x = y``."""
    import jax
    import jax.numpy as jnp

    h = y[2]
    star = y * jnp.array([1.0, 1.0, -1.0])
    big = x - star

    def bracket(rr: Any) -> Any:
        return -h * jax.grad(_potential)(rr, eps2) - _stokeslet(rr, eps2)[:, 2]

    jac = jax.jacfwd(bracket)(big)  # [i, k]
    d = jnp.diag(jnp.array([1.0, 1.0, -1.0]))
    return -_stokeslet(big, eps2) + 2.0 * h * jac @ d


def blake(x: Any, y: Any, eps2: Any) -> Any:
    """``8 pi mu G(x, y)``: the free Stokeslet at ``eps2`` and ``blake_image``."""
    return _stokeslet(x - y, eps2) + blake_image(x, y, eps2)


def blake_checks(seed: int = 7) -> dict[str, Any]:
    """Blake's system against what defines it, before it is used: no slip on the wall (point form: zero to rounding;
    regularised: its residual, reported), divergence-free in ``x``, reciprocity ``G(x, y) = G(y, x)^T``; the smoothed
    potential's Laplacian against ``-4 pi phi_eps``. Controls: a sign of the doublet flipped fails no slip."""
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    sources = rng.uniform([-1, -1, 0.05], [1, 1, 1.0], size=(20, 3))
    walls = np.concatenate([rng.uniform(-2, 2, size=(20, 2)), np.zeros((20, 1))], axis=1)
    point = jax.jit(lambda x, y: blake(x, y, 0.0))
    residual = [float(np.max(np.abs(point(jnp.asarray(w), jnp.asarray(s)))) /
                     np.max(np.abs(_stokeslet(jnp.asarray(w - s), 0.0)))) for w, s in zip(walls, sources, strict=True)]
    out["point_no_slip_relative_max"] = max(residual)
    for eps in (0.05, 0.1, 0.2):
        held = jax.jit(lambda x, y, e=eps * eps: blake(x, y, e))
        res = [float(np.max(np.abs(held(jnp.asarray(w), jnp.asarray(s)))) /
                     np.max(np.abs(_stokeslet(jnp.asarray(w - s), 0.0)))) for w, s in zip(walls, sources, strict=True)]
        heights = sources[:, 2]
        out[f"regularised_no_slip_eps_{eps}"] = {"relative_max": max(res),
                                                 "at_source_height": float(heights[int(np.argmax(res))])}
    targets = rng.uniform([-1, -1, 0.05], [1, 1, 1.0], size=(20, 3))
    div = jax.jit(lambda x, y: jnp.trace(jax.jacfwd(lambda xx: blake(xx, y, 0.0))(x)) * 0.0
                  + jnp.einsum("iji->j", jax.jacfwd(lambda xx: blake(xx, y, 0.0))(x)))
    out["point_divergence_max"] = float(max(np.max(np.abs(div(jnp.asarray(t), jnp.asarray(s))))
                                            for t, s in zip(targets, sources, strict=True)))
    out["point_reciprocity_max"] = float(max(np.max(np.abs(point(jnp.asarray(t), jnp.asarray(s))
                                                           - point(jnp.asarray(s), jnp.asarray(t)).T))
                                             for t, s in zip(targets, sources, strict=True)))

    def flipped(x: Any, y: Any) -> Any:
        h = y[2]
        star = y * jnp.array([1.0, 1.0, -1.0])
        big = x - star
        jac = jax.jacfwd(lambda rr: -h * jax.grad(_potential)(rr, 0.0) - _stokeslet(rr, 0.0)[:, 2])(big)
        return _stokeslet(x - y, 0.0) - _stokeslet(big, 0.0) - 2.0 * h * jac @ jnp.diag(jnp.array([1.0, 1.0, -1.0]))

    fl = jax.jit(flipped)
    out["control_flipped_doublet_no_slip_relative_max"] = max(
        float(np.max(np.abs(fl(jnp.asarray(w), jnp.asarray(s)))) / np.max(np.abs(_stokeslet(jnp.asarray(w - s), 0.0))))
        for w, s in zip(walls, sources, strict=True))
    # the smoothed potential: lap Phi = -4 pi phi_eps, phi_eps = 15 eps^4 / (8 pi (r^2 + eps^2)^{7/2})
    eps = 0.1
    lap = jax.jit(lambda r: jnp.trace(jax.hessian(_potential)(r, eps * eps)))
    worst = 0.0
    for radius in (0.01, 0.05, 0.1, 0.3, 1.0):
        r = jnp.array([radius, 0.3 * radius, -0.2 * radius])
        r2 = float(jnp.dot(r, r))
        expected = -4.0 * math.pi * 15.0 * eps ** 4 / (8.0 * math.pi * (r2 + eps * eps) ** 3.5)
        worst = max(worst, abs(float(lap(r)) - expected) / abs(expected))
    out["potential_laplacian_relative_max"] = worst
    out["passes"] = bool(out["point_no_slip_relative_max"] < 1e-12 and out["point_divergence_max"] < 1e-9
                         and out["point_reciprocity_max"] < 1e-12 and out["potential_laplacian_relative_max"] < 1e-9
                         and out["control_flipped_doublet_no_slip_relative_max"] > 1e-2)
    return out


def tangent_plane(local_source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a source in the unit tube's frame (axis z), the wall's tangent plane at the foot of its radius: the plane's
    frame (rows: along the axis, across, the wall's normal into the fluid) and its origin on the wall."""
    radial = np.array([local_source[0], local_source[1], 0.0])
    radial /= np.linalg.norm(radial)
    normal = -radial
    along = np.array([0.0, 0.0, 1.0])
    return np.stack([along, np.cross(normal, along), normal]), radial


def now() -> float:
    return time.time()


# ---- the near-wall image, regularised (the lever measured in the grid) ----------------------------------------------
#: Samples nearer the wall than this (m) take the regularised image's correction, every pair of them, in the tangent
#: plane at the source's foot. The correction falls as ``eps^2 / R^3`` past the blob, so a pair's beyond it is small.
NEAR_WALL_M = 0.6e-3


def near_wall(points: np.ndarray, frame: np.ndarray, centre: np.ndarray, radius: float,
              near_m: float = NEAR_WALL_M) -> np.ndarray:
    """The indices of the samples nearer the tube's wall than ``near_m``."""
    local = (np.asarray(points) - centre) @ frame.T / radius
    height = 1.0 - np.hypot(local[:, 0], local[:, 1])
    return np.flatnonzero(height * radius < near_m)


def image_correction_blocks(points: Any, index: np.ndarray, frame: Any, centre: Any, radius: float, viscosity: Any,
                            eps2_si: Any) -> Any:
    """``(n, n, 3, 3)`` world blocks ``[target, source]`` (in ``corrected.matrix``'s units) of the regularised plane-wall
    image less the point one, for every pair of the samples ``index``, each in the tangent plane at its source's
    foot. Jittable in ``points`` and ``eps2_si``; ``index`` is static."""
    import jax
    import jax.numpy as jnp

    frame = jnp.asarray(frame)
    local = jnp.matmul(points[index] - jnp.asarray(centre), frame.T, precision="highest") / radius
    eps2 = eps2_si / (radius * radius)
    radial = local[:, :2] / jnp.linalg.norm(local[:, :2], axis=1, keepdims=True)
    normal = jnp.concatenate([-radial, jnp.zeros((len(index), 1))], axis=1)
    along = jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), normal.shape)
    planes = jnp.stack([along, jnp.cross(normal, along), normal], axis=1)  # (n, 3, 3) rows
    origins = jnp.concatenate([radial, jnp.zeros((len(index), 1))], axis=1)

    def for_source(s: Any) -> Any:
        q, o = planes[s], origins[s]
        y = jnp.matmul(q, local[s] - o, precision="highest")
        xs = jnp.matmul(local - o, q.T, precision="highest")

        def one(x: Any) -> Any:
            return blake_image(x, y, eps2) - blake_image(x, y, 0.0)

        blocks = jax.vmap(one)(xs)  # (n, 3, 3) plane frame
        tube = jnp.einsum("ai,tab,bj->tij", q, blocks, q, precision="highest") / (8.0 * math.pi)
        return jnp.einsum("ai,tab,bj->tij", frame, tube, frame, precision="highest") / (radius * viscosity)

    by_source = jax.lax.map(for_source, jnp.arange(len(index)))  # [source, target, 3, 3]
    return jnp.swapaxes(by_source, 0, 1)


def scatter_rows(index: np.ndarray) -> np.ndarray:
    return (3 * np.asarray(index)[:, None] + np.arange(3)[None, :]).ravel()


def no_slip_residuals(eps_values: tuple[float, ...] = (0.0364, 0.0729, 0.1457), heights: tuple[float, ...] = (0.1, 0.2, 0.4),
                      reach: float = 1.0) -> list[dict[str, float]]:
    """The control on the regularised image (added after the grid was read): the velocity on the wall over the point
    free Stokeslet's there, at wall points within ``reach`` of a source at each height (lengths in mm), for this
    study's regularised system (``blake`` at eps) and the node's (the free Stokeslet at eps, the image at a point)."""
    import jax
    import jax.numpy as jnp

    mine = jax.jit(lambda x, y, e2: blake(x, y, e2))
    node = jax.jit(lambda x, y, e2: _stokeslet(x - y, e2) + blake_image(x, y, 0.0))
    out = []
    for eps in eps_values:
        for h in heights:
            y = jnp.array([0.0, 0.0, h])
            a, b = [], []
            for r in np.linspace(0.0, reach, 41):
                for angle in (0.0, 0.7, 1.6):
                    x = jnp.array([r * math.cos(angle), r * math.sin(angle), 0.0])
                    ref = float(jnp.max(jnp.abs(_stokeslet(x - y, 0.0))))
                    a.append(float(jnp.max(jnp.abs(mine(x, y, eps * eps)))) / ref)
                    b.append(float(jnp.max(jnp.abs(node(x, y, eps * eps)))) / ref)
            out.append({"eps_mm": eps, "height_mm": h, "this_study_worst": max(a), "this_study_mean": float(np.mean(a)),
                        "node_worst": max(b), "node_mean": float(np.mean(b))})
    return out
