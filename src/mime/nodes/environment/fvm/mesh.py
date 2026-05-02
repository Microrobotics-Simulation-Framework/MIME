"""FVMMesh — face-graph topology + precomputed geometry.

Every cell-centred operator in this solver is expressed as a
gather-compute-scatter triple over a face graph:

    phi_owner = phi[owner]      # gather
    phi_neigh = phi[neighbour]
    flux_f    = compute(phi_owner, phi_neigh, geom_f)
    res       = segment_sum(flux_f, owner, N_cells)
              - segment_sum(flux_f, neighbour, N_cells)

This module owns the topology and geometry that every operator gathers
against. Nothing here is recomputed inside the time-stepping loop.

The structured Cartesian builders here construct a fully populated
``FVMMesh`` (interior face graph + boundary patches) for a 2D or 3D
brick of cells with uniform spacing. The only thing the solver later
sees that is structured-Cartesian-specific is the FFT pressure path —
all other operators are mesh-agnostic by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Boundary patch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryPatch:
    """A labelled, geometrically homogeneous set of boundary faces.

    Each entry corresponds to one mesh face on the domain boundary.

    Attributes
    ----------
    name : str
        Human-readable label (``"wall"``, ``"inlet"``, ``"top_lid"``).
    owner : jnp.ndarray, shape ``[N_bf]``
        Cell index that owns each boundary face.
    Sf : jnp.ndarray, shape ``[N_bf, dim]``
        Outward face area vector (magnitude = face area, direction =
        outward normal).
    n : jnp.ndarray, shape ``[N_bf, dim]``
        Unit outward normal.
    area : jnp.ndarray, shape ``[N_bf]``
        Face area magnitude.
    d : jnp.ndarray, shape ``[N_bf, dim]``
        Vector from owner cell centroid to face centroid.
    face_x : jnp.ndarray, shape ``[N_bf, dim]``
        Face centroid position.
    """
    name: str
    owner: jnp.ndarray
    Sf: jnp.ndarray
    n: jnp.ndarray
    area: jnp.ndarray
    d: jnp.ndarray
    face_x: jnp.ndarray


# ---------------------------------------------------------------------------
# FVMMesh
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FVMMesh:
    """Face-graph mesh: interior face graph + boundary patches + cell geom.

    The interior face arrays describe a *directed* face graph. Each face
    has an ``owner`` and a ``neighbour`` cell. The face area vector
    ``Sf`` points from owner toward neighbour. ``w`` is the linear
    interpolation weight such that

        phi_f = w * phi_owner + (1 - w) * phi_neighbour.

    For uniform Cartesian meshes ``w = 0.5``; the field is kept generic
    to support stretched / unstructured meshes later without operator
    changes.

    Boundary patches are stored separately because they carry different
    physics (Dirichlet/Neumann/inlet/outlet) — but the face data layout
    is identical to interior faces, so the same gather-compute-scatter
    primitives apply.

    Cartesian metadata is stored optionally (``shape``, ``spacing``,
    ``origin``) for the FFT pressure solver and for visualisation /
    IBM mask generation. Operators do not consume these.

    Notes
    -----
    All arrays are JAX arrays so the mesh is a valid pytree leaf
    structure — vmap/grad over mesh perturbations is therefore
    straightforward (relevant for shape optimisation).

    The shape of every array is static at JIT time because the mesh is
    constructed once and passed as a closed-over pytree. The solver
    sees ``mesh`` as a regular pytree input.
    """

    # Interior face graph
    owner: jnp.ndarray         # [N_faces]   int32
    neighbour: jnp.ndarray     # [N_faces]   int32
    Sf: jnp.ndarray            # [N_faces, dim] float32
    n: jnp.ndarray             # [N_faces, dim] float32 — unit normal
    area: jnp.ndarray          # [N_faces]   float32
    d: jnp.ndarray             # [N_faces, dim] — owner -> neighbour centroid
    d_mag: jnp.ndarray         # [N_faces]   |d|
    w: jnp.ndarray             # [N_faces]   linear interpolation weight

    # Cell-centred geometry
    V: jnp.ndarray             # [N_cells]    cell volumes
    x: jnp.ndarray             # [N_cells, dim] cell centroids

    # Boundary patches
    patches: Tuple[BoundaryPatch, ...] = ()

    # Bookkeeping (Python ints; static at JIT time)
    N_cells: int = 0
    N_faces: int = 0
    dim: int = 2

    # Optional Cartesian metadata
    cartesian_shape: Tuple[int, ...] | None = None  # (Nx, Ny[, Nz])
    cartesian_spacing: Tuple[float, ...] | None = None  # (dx, dy[, dz])
    cartesian_origin: Tuple[float, ...] | None = None

    def patch(self, name: str) -> BoundaryPatch:
        """Look up a boundary patch by name."""
        for p in self.patches:
            if p.name == name:
                return p
        raise KeyError(
            f"Boundary patch {name!r} not found "
            f"(have: {[p.name for p in self.patches]})"
        )

    def reshape_cartesian(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Reshape a flat ``[N_cells, ...]`` array to Cartesian layout.

        Used by the FFT pressure solver and visualisation. Operators do
        not call this; they only see the flat layout.
        """
        if self.cartesian_shape is None:
            raise ValueError("mesh is not Cartesian-structured")
        return phi.reshape(self.cartesian_shape + phi.shape[1:])

    def flatten_cartesian(self, phi: jnp.ndarray) -> jnp.ndarray:
        """Inverse of :meth:`reshape_cartesian`."""
        if self.cartesian_shape is None:
            raise ValueError("mesh is not Cartesian-structured")
        nd = len(self.cartesian_shape)
        trailing = phi.shape[nd:]
        return phi.reshape((self.N_cells,) + trailing)


# Register as a pytree so jax can flatten / vmap mesh-bearing functions.
def _mesh_flatten(m: FVMMesh):
    children = (
        m.owner, m.neighbour, m.Sf, m.n, m.area, m.d, m.d_mag, m.w,
        m.V, m.x,
        tuple(p.owner for p in m.patches),
        tuple(p.Sf for p in m.patches),
        tuple(p.n for p in m.patches),
        tuple(p.area for p in m.patches),
        tuple(p.d for p in m.patches),
        tuple(p.face_x for p in m.patches),
    )
    aux = (
        tuple(p.name for p in m.patches),
        m.N_cells, m.N_faces, m.dim,
        m.cartesian_shape, m.cartesian_spacing, m.cartesian_origin,
    )
    return children, aux


def _mesh_unflatten(aux, children):
    (owner, neighbour, Sf, n, area, d, d_mag, w, V, x,
     p_owner, p_Sf, p_n, p_area, p_d, p_fx) = children
    (names, N_cells, N_faces, dim,
     cshape, cspacing, corigin) = aux
    patches = tuple(
        BoundaryPatch(
            name=names[i],
            owner=p_owner[i], Sf=p_Sf[i], n=p_n[i],
            area=p_area[i], d=p_d[i], face_x=p_fx[i],
        )
        for i in range(len(names))
    )
    return FVMMesh(
        owner=owner, neighbour=neighbour, Sf=Sf, n=n, area=area,
        d=d, d_mag=d_mag, w=w, V=V, x=x, patches=patches,
        N_cells=N_cells, N_faces=N_faces, dim=dim,
        cartesian_shape=cshape, cartesian_spacing=cspacing,
        cartesian_origin=corigin,
    )


jax.tree_util.register_pytree_node(FVMMesh, _mesh_flatten, _mesh_unflatten)


# ---------------------------------------------------------------------------
# Cartesian builders
# ---------------------------------------------------------------------------

def make_cartesian_mesh_2d(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
    *,
    origin: Tuple[float, float] = (0.0, 0.0),
    dtype=jnp.float32,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> FVMMesh:
    """Construct a 2D structured Cartesian face-graph mesh.

    Cells are indexed in C-order: ``cell_id(i, j) = i * Ny + j``,
    ``i ∈ [0, Nx)`` (x-direction), ``j ∈ [0, Ny)`` (y-direction).

    Interior face ordering: all x-faces first, then all y-faces. An
    x-face at ``(i, j)`` separates cell ``(i, j)`` (owner) and cell
    ``(i+1, j)`` (neighbour). A y-face at ``(i, j)`` separates cell
    ``(i, j)`` (owner) and cell ``(i, j+1)`` (neighbour).

    Boundary patches: ``"x_min"``, ``"x_max"``, ``"y_min"``, ``"y_max"``.
    They are not assigned BC types here — that is the solver's concern.
    """
    dx = Lx / Nx
    dy = Ly / Ny
    N_cells = Nx * Ny

    # Cell centroids: (i+0.5)*dx, (j+0.5)*dy in C-order.
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    x = np.stack(
        [origin[0] + (ii + 0.5) * dx, origin[1] + (jj + 0.5) * dy],
        axis=-1,
    ).reshape(N_cells, 2)
    V = np.full((N_cells,), dx * dy, dtype=np.float64)

    # ---- Interior x-faces: between (i, j) and (i+1, j) ----
    # If periodic_x, also include the wrap face (Nx-1, j) -> (0, j).
    if periodic_x:
        i_lo = np.arange(Nx)
        i_hi = (i_lo + 1) % Nx
    else:
        i_lo = np.arange(Nx - 1)
        i_hi = i_lo + 1
    iix, jjx = np.meshgrid(i_lo, np.arange(Ny), indexing="ij")
    iix_n, _ = np.meshgrid(i_hi, np.arange(Ny), indexing="ij")
    own_x = (iix * Ny + jjx).reshape(-1)
    nei_x = (iix_n * Ny + jjx).reshape(-1)
    Nf_x = own_x.size
    Sf_x = np.zeros((Nf_x, 2)); Sf_x[:, 0] = dy   # area = dy*1, normal = +x
    n_x  = np.zeros((Nf_x, 2)); n_x[:, 0] = 1.0
    d_x  = np.zeros((Nf_x, 2)); d_x[:, 0] = dx
    area_x = np.full((Nf_x,), dy)

    # ---- Interior y-faces: between (i, j) and (i, j+1) ----
    if periodic_y:
        j_lo = np.arange(Ny)
        j_hi = (j_lo + 1) % Ny
    else:
        j_lo = np.arange(Ny - 1)
        j_hi = j_lo + 1
    iiy, jjy = np.meshgrid(np.arange(Nx), j_lo, indexing="ij")
    _, jjy_n = np.meshgrid(np.arange(Nx), j_hi, indexing="ij")
    own_y = (iiy * Ny + jjy).reshape(-1)
    nei_y = (iiy * Ny + jjy_n).reshape(-1)
    Nf_y = own_y.size
    Sf_y = np.zeros((Nf_y, 2)); Sf_y[:, 1] = dx
    n_y  = np.zeros((Nf_y, 2)); n_y[:, 1] = 1.0
    d_y  = np.zeros((Nf_y, 2)); d_y[:, 1] = dy
    area_y = np.full((Nf_y,), dx)

    owner = np.concatenate([own_x, own_y])
    neighbour = np.concatenate([nei_x, nei_y])
    Sf = np.concatenate([Sf_x, Sf_y], axis=0)
    n = np.concatenate([n_x, n_y], axis=0)
    area = np.concatenate([area_x, area_y])
    d = np.concatenate([d_x, d_y], axis=0)
    d_mag = np.linalg.norm(d, axis=1)
    w = np.full((d.shape[0],), 0.5)
    N_faces = owner.size

    # ---- Boundary patches ----
    def _patch(name, owner_cells, normal, area_val, half_step):
        N_bf = owner_cells.size
        n_arr = np.zeros((N_bf, 2)); n_arr[:] = normal
        Sf_arr = n_arr * area_val
        area_arr = np.full((N_bf,), area_val)
        d_arr = n_arr * half_step
        face_x = x[owner_cells] + d_arr
        return BoundaryPatch(
            name=name,
            owner=jnp.asarray(owner_cells, dtype=jnp.int32),
            Sf=jnp.asarray(Sf_arr, dtype=dtype),
            n=jnp.asarray(n_arr, dtype=dtype),
            area=jnp.asarray(area_arr, dtype=dtype),
            d=jnp.asarray(d_arr, dtype=dtype),
            face_x=jnp.asarray(face_x, dtype=dtype),
        )

    patches_list = []
    if not periodic_x:
        x_min_owner = (0 * Ny + np.arange(Ny))            # i = 0
        x_max_owner = ((Nx - 1) * Ny + np.arange(Ny))     # i = Nx - 1
        patches_list.append(_patch(
            "x_min", x_min_owner, np.array([-1.0, 0.0]), dy, dx / 2,
        ))
        patches_list.append(_patch(
            "x_max", x_max_owner, np.array([+1.0, 0.0]), dy, dx / 2,
        ))
    if not periodic_y:
        y_min_owner = (np.arange(Nx) * Ny + 0)            # j = 0
        y_max_owner = (np.arange(Nx) * Ny + (Ny - 1))     # j = Ny - 1
        patches_list.append(_patch(
            "y_min", y_min_owner, np.array([0.0, -1.0]), dx, dy / 2,
        ))
        patches_list.append(_patch(
            "y_max", y_max_owner, np.array([0.0, +1.0]), dx, dy / 2,
        ))
    patches = tuple(patches_list)

    return FVMMesh(
        owner=jnp.asarray(owner, dtype=jnp.int32),
        neighbour=jnp.asarray(neighbour, dtype=jnp.int32),
        Sf=jnp.asarray(Sf, dtype=dtype),
        n=jnp.asarray(n, dtype=dtype),
        area=jnp.asarray(area, dtype=dtype),
        d=jnp.asarray(d, dtype=dtype),
        d_mag=jnp.asarray(d_mag, dtype=dtype),
        w=jnp.asarray(w, dtype=dtype),
        V=jnp.asarray(V, dtype=dtype),
        x=jnp.asarray(x, dtype=dtype),
        patches=patches,
        N_cells=int(N_cells),
        N_faces=int(N_faces),
        dim=2,
        cartesian_shape=(Nx, Ny),
        cartesian_spacing=(float(dx), float(dy)),
        cartesian_origin=(float(origin[0]), float(origin[1])),
    )


def make_cartesian_mesh_3d(
    Nx: int,
    Ny: int,
    Nz: int,
    Lx: float,
    Ly: float,
    Lz: float,
    *,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    dtype=jnp.float32,
    periodic_x: bool = False,
    periodic_y: bool = False,
    periodic_z: bool = False,
) -> FVMMesh:
    """Construct a 3D structured Cartesian face-graph mesh.

    Cells indexed in C-order: ``cell_id(i, j, k) = (i*Ny + j)*Nz + k``.
    Interior faces ordered ``[x-faces, y-faces, z-faces]``.
    Boundary patches: ``"x_min"``, ``"x_max"``, ``"y_min"``, ``"y_max"``,
    ``"z_min"``, ``"z_max"``.
    """
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz
    N_cells = Nx * Ny * Nz

    ii, jj, kk = np.meshgrid(
        np.arange(Nx), np.arange(Ny), np.arange(Nz), indexing="ij",
    )
    x = np.stack(
        [origin[0] + (ii + 0.5) * dx,
         origin[1] + (jj + 0.5) * dy,
         origin[2] + (kk + 0.5) * dz],
        axis=-1,
    ).reshape(N_cells, 3)
    V = np.full((N_cells,), dx * dy * dz, dtype=np.float64)

    def _cell(i, j, k):
        return (i * Ny + j) * Nz + k

    # x-faces
    if periodic_x:
        i_lo_x = np.arange(Nx); i_hi_x = (i_lo_x + 1) % Nx
    else:
        i_lo_x = np.arange(Nx - 1); i_hi_x = i_lo_x + 1
    iix, jjx, kkx = np.meshgrid(i_lo_x, np.arange(Ny), np.arange(Nz), indexing="ij")
    iix_n, _, _ = np.meshgrid(i_hi_x, np.arange(Ny), np.arange(Nz), indexing="ij")
    own_x = _cell(iix, jjx, kkx).reshape(-1)
    nei_x = _cell(iix_n, jjx, kkx).reshape(-1)
    Nf_x = own_x.size
    Sf_x = np.zeros((Nf_x, 3)); Sf_x[:, 0] = dy * dz
    n_x = np.zeros((Nf_x, 3)); n_x[:, 0] = 1.0
    d_x = np.zeros((Nf_x, 3)); d_x[:, 0] = dx
    area_x = np.full((Nf_x,), dy * dz)

    # y-faces
    if periodic_y:
        j_lo_y = np.arange(Ny); j_hi_y = (j_lo_y + 1) % Ny
    else:
        j_lo_y = np.arange(Ny - 1); j_hi_y = j_lo_y + 1
    iiy, jjy, kky = np.meshgrid(np.arange(Nx), j_lo_y, np.arange(Nz), indexing="ij")
    _, jjy_n, _ = np.meshgrid(np.arange(Nx), j_hi_y, np.arange(Nz), indexing="ij")
    own_y = _cell(iiy, jjy, kky).reshape(-1)
    nei_y = _cell(iiy, jjy_n, kky).reshape(-1)
    Nf_y = own_y.size
    Sf_y = np.zeros((Nf_y, 3)); Sf_y[:, 1] = dx * dz
    n_y = np.zeros((Nf_y, 3)); n_y[:, 1] = 1.0
    d_y = np.zeros((Nf_y, 3)); d_y[:, 1] = dy
    area_y = np.full((Nf_y,), dx * dz)

    # z-faces
    if periodic_z:
        k_lo_z = np.arange(Nz); k_hi_z = (k_lo_z + 1) % Nz
    else:
        k_lo_z = np.arange(Nz - 1); k_hi_z = k_lo_z + 1
    iiz, jjz, kkz = np.meshgrid(np.arange(Nx), np.arange(Ny), k_lo_z, indexing="ij")
    _, _, kkz_n = np.meshgrid(np.arange(Nx), np.arange(Ny), k_hi_z, indexing="ij")
    own_z = _cell(iiz, jjz, kkz).reshape(-1)
    nei_z = _cell(iiz, jjz, kkz_n).reshape(-1)
    Nf_z = own_z.size
    Sf_z = np.zeros((Nf_z, 3)); Sf_z[:, 2] = dx * dy
    n_z = np.zeros((Nf_z, 3)); n_z[:, 2] = 1.0
    d_z = np.zeros((Nf_z, 3)); d_z[:, 2] = dz
    area_z = np.full((Nf_z,), dx * dy)

    owner = np.concatenate([own_x, own_y, own_z])
    neighbour = np.concatenate([nei_x, nei_y, nei_z])
    Sf = np.concatenate([Sf_x, Sf_y, Sf_z], axis=0)
    n = np.concatenate([n_x, n_y, n_z], axis=0)
    area = np.concatenate([area_x, area_y, area_z])
    d = np.concatenate([d_x, d_y, d_z], axis=0)
    d_mag = np.linalg.norm(d, axis=1)
    w = np.full((d.shape[0],), 0.5)
    N_faces = owner.size

    def _patch(name, owner_cells, normal, area_val, half_step):
        N_bf = owner_cells.size
        n_arr = np.zeros((N_bf, 3)); n_arr[:] = normal
        Sf_arr = n_arr * area_val
        area_arr = np.full((N_bf,), area_val)
        d_arr = n_arr * half_step
        face_x = x[owner_cells] + d_arr
        return BoundaryPatch(
            name=name,
            owner=jnp.asarray(owner_cells, dtype=jnp.int32),
            Sf=jnp.asarray(Sf_arr, dtype=dtype),
            n=jnp.asarray(n_arr, dtype=dtype),
            area=jnp.asarray(area_arr, dtype=dtype),
            d=jnp.asarray(d_arr, dtype=dtype),
            face_x=jnp.asarray(face_x, dtype=dtype),
        )

    patches_list = []
    if not periodic_x:
        jj_, kk_ = np.meshgrid(np.arange(Ny), np.arange(Nz), indexing="ij")
        patches_list.append(_patch("x_min", _cell(0, jj_, kk_).reshape(-1),
                                    np.array([-1.0, 0.0, 0.0]), dy * dz, dx / 2))
        patches_list.append(_patch("x_max", _cell(Nx - 1, jj_, kk_).reshape(-1),
                                    np.array([+1.0, 0.0, 0.0]), dy * dz, dx / 2))
    if not periodic_y:
        ii_, kk_ = np.meshgrid(np.arange(Nx), np.arange(Nz), indexing="ij")
        patches_list.append(_patch("y_min", _cell(ii_, 0, kk_).reshape(-1),
                                    np.array([0.0, -1.0, 0.0]), dx * dz, dy / 2))
        patches_list.append(_patch("y_max", _cell(ii_, Ny - 1, kk_).reshape(-1),
                                    np.array([0.0, +1.0, 0.0]), dx * dz, dy / 2))
    if not periodic_z:
        ii_, jj_ = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
        patches_list.append(_patch("z_min", _cell(ii_, jj_, 0).reshape(-1),
                                    np.array([0.0, 0.0, -1.0]), dx * dy, dz / 2))
        patches_list.append(_patch("z_max", _cell(ii_, jj_, Nz - 1).reshape(-1),
                                    np.array([0.0, 0.0, +1.0]), dx * dy, dz / 2))
    patches = tuple(patches_list)

    return FVMMesh(
        owner=jnp.asarray(owner, dtype=jnp.int32),
        neighbour=jnp.asarray(neighbour, dtype=jnp.int32),
        Sf=jnp.asarray(Sf, dtype=dtype),
        n=jnp.asarray(n, dtype=dtype),
        area=jnp.asarray(area, dtype=dtype),
        d=jnp.asarray(d, dtype=dtype),
        d_mag=jnp.asarray(d_mag, dtype=dtype),
        w=jnp.asarray(w, dtype=dtype),
        V=jnp.asarray(V, dtype=dtype),
        x=jnp.asarray(x, dtype=dtype),
        patches=patches,
        N_cells=int(N_cells),
        N_faces=int(N_faces),
        dim=3,
        cartesian_shape=(Nx, Ny, Nz),
        cartesian_spacing=(float(dx), float(dy), float(dz)),
        cartesian_origin=tuple(float(o) for o in origin),
    )
