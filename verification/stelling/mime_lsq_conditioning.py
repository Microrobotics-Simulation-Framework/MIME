"""The stelling conditioning contract attached to the REAL MIME
least-squares gradient — MIME's stelling SOCKET.

This file lives MIME-SIDE, in ``MIME/verification/stelling/``, per
stelling's ARCHITECTURE.md Rule 1: a socket is a living verification
surface encoding maintainer knowledge (admissible envelopes, failure
geometry, which gates catch what) and belongs in the library's repo, run
by the library's CI — stelling ships the instruments, MIME owns this
socket. Built against stelling 0.1.0 (the pin line printed at startup
records the live versions); its known-answer STATUSES are the record in
stelling's ``design/la-attachment.md`` and must not drift.

Target: mime-engine @ 7ce1efb4311b, ``src/mime/nodes/environment/fvm/``
(operators.py + mesh.py — the only two sim files this script reads or
imports from). The call site is ``grad_least_squares(phi, mesh,
boundary_face_values=None, *, reg=1e-30)`` (operators.py:119-172): per
cell P it assembles the normal matrix ``M_P = Σ_f d_f ⊗ d_f`` over the
face stencil plus ``reg·I`` and solves via ``jnp.linalg.inv`` + einsum.
The contract is :func:`stelling.contracts.conditioning_2x2_field` — the
requires face (M well-conditioned, per cell, closed form) mechanized
through the ONE stelling pipeline; the ensures face (norm-sensitivity of
the solve output) DECLARED, never checked.

Structure, in order:

* **seam** — what actually happens when ``jax.ops.segment_sum`` (the
  real assembly's reduction) meets stelling's propagation: measured, not
  assumed.
* **fidelity check F (mandatory, first)** — the unrolled transcription
  (M, MIME's rhs recomputed the same unrolled way, ``jnp.linalg.inv`` +
  einsum exactly as operators.py:167-171) against the REAL imported
  ``grad_least_squares``, random phi (fixed seed), in BOTH
  configurations (boundary-starved ``None`` / boundary-fed all-patches),
  on TWO meshes: the 2x1 the KAs are about AND a 3x2 whose cell stencils
  are non-congruent (on 2x1 alone, cell-permuting/role-swapping
  mutations of the assembly produce the correct values — degenerate
  evidence). Layered with exact M cross-checks in both configurations
  and the reg signature pin, because each layer has a measured blind
  spot (a 10x reg error is invisible to F alone: absorbed in fed,
  nullspace-aligned in starved). A transcription that cannot be tied to
  the real function this way is not an attachment.
* **KA-A** — the scar: the DEFAULT call (``boundary_face_values=None``)
  is boundary-starved; on the smallest interior-face mesh each cell's M
  is the rank-1 one-face stencil + reg. Expected REFUTED on intervals
  alone, per cell.
* **KA-B** — the same mesh boundary-fed (all four patches present):
  healthy M, VERIFIED, with the vacuity instrument's inert line recorded
  verbatim (all-point envelope).
* **KA-C** — declared boxes on the delta components: the requires over a
  FAMILY of axis-aligned geometries; interval straddle quoted, solver
  VERIFIED (QF_NRA) on the narrow region, solver REFUTED with a
  replay-confirmed witness on the widened region, the witness's
  condition number re-derived exactly in-script.

Everything asserted below is a hard gate: the script is its own check.
MIME is exercised, never edited. Run (any cwd, a python with stelling +
mime + jax importable):

    python MIME/verification/stelling/mime_lsq_conditioning.py

The companion ``battery.py`` gauges this socket's fidelity gate stack
through ``stelling.fidelity.gauge`` (the L21 loop); run it alongside.

Precision, recorded: ``jax_enable_x64=True`` before any tracing, and the
mesh is constructed with ``dtype=jnp.float64`` (the constructor DEFAULTS
to ``jnp.float32`` — mesh.py:229; the probe arithmetic this attaches to
is binary64). The actual mesh array dtypes are printed and asserted.
"""

import inspect
import math
from fractions import Fraction
from importlib.metadata import version as _pkg_version

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_2d  # noqa: E402
from mime.nodes.environment.fvm.operators import grad_least_squares  # noqa: E402

import stelling  # noqa: E402
from stelling._jax_compat import jax_version, trace  # noqa: E402
from stelling.contracts import check_contract, conditioning_2x2_field  # noqa: E402
from stelling.harness import any_array, assert_  # noqa: E402
from stelling.preconditions import check  # noqa: E402
from stelling.propagate import propagate  # noqa: E402
from stelling.verdict import SolverStamp  # noqa: E402

try:
    MIME_VERSION = _pkg_version("mime-engine")
except Exception:  # pragma: no cover - version report only
    MIME_VERSION = "unversioned"

print(
    f"pinned: mime-engine {MIME_VERSION} @ 7ce1efb4311b | "
    f"jax {jax_version()} | stelling {stelling.__version__}"
)

KAPPA = 8.0
TIMEOUT_MS = 20000
VMODE = "inputs-only"

# reg is read from the REAL function's signature default, not hardcoded
REG = inspect.signature(grad_least_squares).parameters["reg"].default
assert REG == 1e-30, f"operators.py reg default moved: {REG!r}"

# --- the real mesh, constructed OUTSIDE any trace ----------------------------
#
# make_cartesian_mesh_2d (mesh.py:222). 1x1 has N_faces=0 (no interior
# face graph — nothing for the gradient stencil to see); 2x1 is the
# smallest mesh with an interior face. dtype=jnp.float64 OVERRIDES the
# constructor's float32 default (mesh.py:229) so the attachment runs in
# the binary64 the probe arithmetic was done in.

MESH = make_cartesian_mesh_2d(2, 1, 1.0, 1.0, dtype=jnp.float64)
N_CELLS = MESH.N_cells
N_FACES = MESH.N_faces
OWNER = np.asarray(MESH.owner)
NEIGH = np.asarray(MESH.neighbour)
D_INT = np.asarray(MESH.d)  # [N_faces, 2], x_N - x_O   (operators.py:141)
PATCH_NAMES = tuple(p.name for p in MESH.patches)
PATCH_OWNER = {p.name: np.asarray(p.owner) for p in MESH.patches}
PATCH_D = {p.name: np.asarray(p.d) for p in MESH.patches}  # owner→face centroid

# the connectivity and geometry this attachment is about, asserted
assert N_CELLS == 2 and N_FACES == 1
assert OWNER.tolist() == [0] and NEIGH.tolist() == [1]
assert PATCH_NAMES == ("x_min", "x_max", "y_min", "y_max")
assert PATCH_OWNER["x_min"].tolist() == [0] and PATCH_OWNER["x_max"].tolist() == [1]
assert PATCH_OWNER["y_min"].tolist() == [0, 1] and PATCH_OWNER["y_max"].tolist() == [0, 1]
assert D_INT.tolist() == [[0.5, 0.0]]
assert PATCH_D["x_min"].tolist() == [[-0.25, 0.0]]
assert PATCH_D["x_max"].tolist() == [[0.25, 0.0]]
assert PATCH_D["y_min"].tolist() == [[0.0, -0.5], [0.0, -0.5]]
assert PATCH_D["y_max"].tolist() == [[0.0, 0.5], [0.0, 0.5]]
# actual dtypes, recorded and asserted (float64 requested and delivered
# under x64; index arrays stay int32 by the constructor's own choice)
assert D_INT.dtype == np.float64 and all(
    PATCH_D[n].dtype == np.float64 for n in PATCH_NAMES
)
assert OWNER.dtype == np.int32 and NEIGH.dtype == np.int32 and all(
    PATCH_OWNER[n].dtype == np.int32 for n in PATCH_NAMES
)
print(
    f"real mesh: make_cartesian_mesh_2d(2, 1, 1.0, 1.0, dtype=jnp.float64) -> "
    f"N_cells={N_CELLS}, N_faces={N_FACES} (interior, owner {OWNER.tolist()} -> "
    f"neighbour {NEIGH.tolist()}), d={D_INT.tolist()}; patches "
    f"{list(PATCH_NAMES)} with d "
    f"{ {n: PATCH_D[n].tolist() for n in PATCH_NAMES} }"
)
print(
    f"  dtypes as constructed: d/patch.d {D_INT.dtype} (float64 requested — "
    f"the constructor default is float32, mesh.py:229), owner/patch.owner "
    f"{OWNER.dtype}"
)

# --- seam: jax.ops.segment_sum under stelling's propagation ------------------
#
# The real assembly reduces with jax.ops.segment_sum (operators.py:143).
# TRANSITION, recorded: this section measured stelling's scatter-add WALL
# until stelling commit 3f78fdd added censused transfer/emission rows for
# scatter-add (and stack). Before: the traced `scatter-add` equation had
# no transfer row — the summed M went to ⊤, the obligation was UNKNOWN,
# and escalation declined naming the primitive ("primitive 'scatter-add'
# is outside the supported emission set"); that wall is WHY the
# transcription below unrolls the segment sums over static connectivity,
# and the unrolled transcription STAYS (it is the fidelity-gauged
# artifact — see battery.py; re-transcribing on segment_sum would be new
# work with its own battery). Now: this section pins the SUPPORTED
# behavior measured under current stelling — full coverage, a sound
# scatter-add transfer, the obligation interval-discharged, no solver
# needed. A future stelling regression that reintroduces the wall must
# fail these asserts loudly. (The living-socket function, working: the
# socket catches the verifier's capability growth, and its pins move by
# measurement, not by hope.)

print("\n==== seam: segment_sum under the tracer (measured)")


def _cat_scalars(*scalars):
    """Family assembly via concatenate over reshaped single-element
    pieces. (Historical note: chosen when jnp.stack's `stack` primitive
    had no transfer row; stelling 3f78fdd added stack rows, measured
    supported now, but this assembly stays — the KA traces and query
    hashes are built on it.)"""
    return jnp.concatenate([jnp.reshape(s, (1,)) for s in scalars])


def seg_probe():
    d0 = any_array((), "float64", (0.5, 0.5))
    d1 = any_array((), "float64", (0.0, 0.0))
    # dd = d ⊗ d for the one interior face, shaped [1, 2, 2] as at
    # operators.py:142, then the real reduction of operators.py:143
    dd = jnp.reshape(_cat_scalars(d0 * d0, d0 * d1, d1 * d0, d1 * d1), (1, 2, 2))
    M = jax.ops.segment_sum(dd, jnp.asarray(OWNER), num_segments=N_CELLS)
    return (assert_(M[..., 0, 0] >= 0.0),)


_seg_cj = trace(seg_probe)
_seg_p = propagate(_seg_cj)
_seg_unknown_prims = [name for name, _ in _seg_p.coverage.unknown_primitives]
_seg_transfers = dict(_seg_p.transfers_used)
print(f"  propagation coverage: {_seg_p.coverage.summary()}")
print(f"  primitives with no transfer: {_seg_unknown_prims}")
print(f"  obligation statuses: {[o.status for o in _seg_p.obligations]}")
print(
    f"  scatter-add transfer tier: {_seg_transfers.get('scatter-add')!r} "
    f"(supported since stelling 3f78fdd; this section asserted the WALL "
    f"before that commit)"
)
assert _seg_p.coverage.summary() == "20 eqns: 20 known (100%)"
assert _seg_unknown_prims == []
assert [o.status for o in _seg_p.obligations] == ["discharged"]
assert "definitely true for all 2 element(s)" in _seg_p.obligations[0].detail
assert _seg_transfers.get("scatter-add") == "sound"
_seg_v = check(seg_probe, vacuity_mode=VMODE, solver_timeout_ms=TIMEOUT_MS)
assert _seg_v.status == "VERIFIED"
assert isinstance(_seg_v.stamp.solver, SolverStamp)
assert not _seg_v.stamp.solver.invoked  # intervals decided it; no escalation
print(f"  check(): {_seg_v.status}; solver stamp, verbatim: "
      f"{_seg_v.stamp.solver.reason}")

# --- the transcription: M's assembly UNROLLED over static connectivity ------
#
# Hand transcription of operators.py:141-167, with the segment sums
# unrolled over a real mesh's STATIC connectivity (owner / neighbour /
# patch-owner index arrays are concrete numpy data from the real mesh
# object):
#
#   :141  d = mesh.d                                  [N_faces, dim]
#   :142  dd = d[:, :, None] * d[:, None, :]          — the outer product;
#         element (0,1) is dx*dy and element (1,0) is dy*dx, kept as
#         DISTINCT products here so the posed symmetry obligations judge
#         the transcription's actual off-diagonals
#   :143  M  = segment_sum(dd, mesh.owner)            — cell P receives
#         dd_f for every interior face f with owner[f] == P
#   :144     + segment_sum(dd, mesh.neighbour)        — and for every f
#         with neighbour[f] == P ((−d)⊗(−d) = d⊗d)
#   :154-160  for each patch IN boundary_face_values (presence, not
#         value: the guard is `if patch.name not in bvals: continue`),
#         db = patch.d; M += segment_sum(db ⊗ db, patch.owner)
#   :167  M = M + reg * eye(dim)
#
# The per-cell accumulation keeps ONE SUBTOTAL PER SEGMENT_SUM and
# combines the subtotals in the real code's order — float addition is
# not associative, and on a mesh where a cell meets several faces per
# segment (the 3x2 fidelity mesh below) a merged flat sum would be a
# different association than the code being transcribed.
#
# ONE assembler serves every consumer below — the fidelity checks run it
# on concrete float64 scalars (both meshes), the KA transforms run it on
# traced declarations (the 2x1 mesh); there is no second assembly path.


def mesh_static(mesh):
    """A real mesh object's static connectivity, as concrete numpy."""
    return {
        "n_cells": mesh.N_cells,
        "n_faces": mesh.N_faces,
        "owner": np.asarray(mesh.owner),
        "neigh": np.asarray(mesh.neighbour),
        "patch_names": tuple(p.name for p in mesh.patches),
        "patch_owner": {p.name: np.asarray(p.owner) for p in mesh.patches},
    }


def assemble_M_family(ms, d_int, patch_d, fed):
    """[N_cells, 2, 2] normal-matrix family; entries traced or concrete.

    ``ms``: :func:`mesh_static` of the target mesh; ``d_int``: per-
    interior-face ``(dx, dy)``; ``patch_d``: per-patch list of per-face
    ``(dx, dy)`` (needed only for patches in ``fed``); ``fed``: the
    patch names present in ``boundary_face_values``.
    """
    rows = []
    for P in range(ms["n_cells"]):
        groups = []  # one entry per segment_sum of the real code, in order
        g = [d_int[f] for f in range(ms["n_faces"]) if ms["owner"][f] == P]
        if g:
            groups.append(g)  # operators.py:143 — owner segment
        g = [d_int[f] for f in range(ms["n_faces"]) if ms["neigh"][f] == P]
        if g:
            groups.append(g)  # operators.py:144 — neighbour segment
        for name in ms["patch_names"]:  # operators.py:154-160 — fed only
            if name not in fed:
                continue
            po = ms["patch_owner"][name]
            g = [patch_d[name][k] for k in range(po.size) if po[k] == P]
            if g:
                groups.append(g)
        assert groups, f"cell {P} has an empty stencil on this mesh"
        m00 = m01 = m10 = m11 = None
        for g in groups:
            s00 = s01 = s10 = s11 = None  # the segment's own subtotal
            for (dx, dy) in g:
                t00, t01, t10, t11 = dx * dx, dx * dy, dy * dx, dy * dy  # :142
                s00 = t00 if s00 is None else s00 + t00
                s01 = t01 if s01 is None else s01 + t01
                s10 = t10 if s10 is None else s10 + t10
                s11 = t11 if s11 is None else s11 + t11
            m00 = s00 if m00 is None else m00 + s00
            m01 = s01 if m01 is None else m01 + s01
            m10 = s10 if m10 is None else m10 + s10
            m11 = s11 if m11 is None else m11 + s11
        m00 = m00 + REG  # operators.py:167 — reg * eye(2), diagonal only
        m11 = m11 + REG
        rows.append(_cat_scalars(m00, m01, m10, m11))
    return jnp.reshape(jnp.concatenate(rows), (ms["n_cells"], 2, 2))


def assemble_rhs(ms, phi, d_int, patch_d, fed, bvals):
    """MIME's rhs, recomputed the same unrolled way (operators.py:146-165).

    :146  dphi = phi[neighbour] - phi[owner]
    :148-152  rhs_P += d_f * dphi_f for owner AND neighbour cells
              ((−d)(−dphi) = d·dphi — the operator's own comment, :147)
    :161-165  fed patches: dphi_b = bvals[name] - phi[patch.owner];
              rhs_P += db * dphi_b. Fidelity-only: the CONTRACT is about
              M (the values feed rhs, never M). Same subtotal-per-segment
              association as the M assembler.
    """
    rows = []
    for P in range(ms["n_cells"]):
        groups = []
        for role in ("owner", "neigh"):  # :150-152 — the two segment_sums
            seg = ms[role]
            g = []
            for f in range(ms["n_faces"]):
                if seg[f] == P:
                    dphi = phi[int(ms["neigh"][f])] - phi[int(ms["owner"][f])]
                    g.append((dphi * d_int[f][0], dphi * d_int[f][1]))
            if g:
                groups.append(g)
        for name in ms["patch_names"]:  # :161-165 — fed patches only
            if name not in fed:
                continue
            po = ms["patch_owner"][name]
            g = []
            for k in range(po.size):
                if po[k] == P:
                    dphi_b = bvals[name][k] - phi[int(po[k])]
                    g.append(
                        (dphi_b * patch_d[name][k][0], dphi_b * patch_d[name][k][1])
                    )
            if g:
                groups.append(g)
        r0 = r1 = None
        for g in groups:
            s0 = s1 = None
            for (c0, c1) in g:
                s0 = c0 if s0 is None else s0 + c0
                s1 = c1 if s1 is None else s1 + c1
            r0 = s0 if r0 is None else r0 + s0
            r1 = s1 if r1 is None else r1 + s1
        rows.append(_cat_scalars(r0, r1))
    return jnp.reshape(jnp.concatenate(rows), (ms["n_cells"], 2))


def concrete_deltas(mesh):
    ms = mesh_static(mesh)
    d = np.asarray(mesh.d)
    pd = {p.name: np.asarray(p.d) for p in mesh.patches}
    d_int = [(jnp.float64(d[f, 0]), jnp.float64(d[f, 1])) for f in range(ms["n_faces"])]
    patch_d = {
        name: [
            (jnp.float64(pd[name][k, 0]), jnp.float64(pd[name][k, 1]))
            for k in range(ms["patch_owner"][name].size)
        ]
        for name in ms["patch_names"]
    }
    return d_int, patch_d


def transcribed_grad(mesh, ms, phi, fed, bvals):
    d_int, patch_d = concrete_deltas(mesh)
    M = assemble_M_family(ms, d_int, patch_d, fed)
    rhs = assemble_rhs(ms, phi, d_int, patch_d, fed, bvals)
    Minv = jnp.linalg.inv(M)  # operators.py:170
    return jnp.einsum("cij,c...j->c...i", Minv, rhs)  # operators.py:171


def M_via_segment_sum(mesh, fed):
    """The real assembly formula run literally (jax.ops.segment_sum on
    the mesh's own arrays) — the value oracle for the M cross-checks."""
    ms = mesh_static(mesh)
    d = np.asarray(mesh.d)
    dd = d[:, :, None] * d[:, None, :]
    M = jax.ops.segment_sum(
        jnp.asarray(dd), jnp.asarray(ms["owner"]), num_segments=ms["n_cells"]
    ) + jax.ops.segment_sum(
        jnp.asarray(dd), jnp.asarray(ms["neigh"]), num_segments=ms["n_cells"]
    )
    for name in ms["patch_names"]:
        if name not in fed:
            continue
        db = np.asarray(dict((p.name, p.d) for p in mesh.patches)[name])
        M = M + jax.ops.segment_sum(
            jnp.asarray(db[:, :, None] * db[:, None, :]),
            jnp.asarray(ms["patch_owner"][name]),
            num_segments=ms["n_cells"],
        )
    return M + REG * jnp.eye(2, dtype=M.dtype)[None]


# --- fidelity check F (mandatory, first) -------------------------------------
#
# The gate stack, layered because each layer has a MEASURED blind spot:
#
# * F (output comparison, both configs) pins the float-visible
#   assembly-to-output path — rhs, inv, einsum included;
# * the exact M cross-checks (transcribed M == segment_sum rebuild, BOTH
#   configurations) pin the assembled VALUES;
# * reg is ALSO pinned by reading and asserting the real signature
#   default (top of this script) — because a 10x reg error (1e-29 for
#   1e-30) is invisible to F alone in BOTH configurations: absorbed in
#   fed (0.3125 + 1e-29 == 0.3125 in binary64, so even exact == on the
#   fed M passes) and nullspace-aligned in starved (rhs = [c, 0.0]
#   exactly on this mesh, so F observes only the first column of M⁻¹ and
#   never touches the reg-only second diagonal). The STARVED M
#   cross-check catches it by value (1e-29 != 1e-30 where the entry IS
#   reg).
#
# Adoption warning: a reader copying this pattern onto a solver whose
# regulariser (or any absorbed parameter) is NOT signature-readable must
# pin it by construction — F cannot see that class of error.
#
# The 2x1 mesh alone is DEGENERATE fidelity evidence: both cells'
# stencils are congruent, so cell-permuting / role-swapping / sign-
# flipping mutations of the assembly produce the CORRECT M values there.
# The full stack therefore also runs on a 3x2 mesh, whose cell stencils
# are non-congruent (corner/edge cells differ in the fed config;
# interior-face counts differ per cell in the starved config). Measured
# residual even there: mutations value-identical BY ALGEBRA on every
# Cartesian box — transposing the symmetric M, swapping the
# owner/neighbour roles in its sum, sign flips under d⊗d, and the 180°
# cell permutation (the constructor's boxes are centro-symmetric) —
# are undetectable by value comparison on ANY such mesh.


def run_fidelity(label, mesh):
    ms = mesh_static(mesh)
    all_patches = ms["patch_names"]
    rng = np.random.RandomState(20260721)
    max_diff = {"starved": 0.0, "fed": 0.0}
    for _trial in range(5):
        phi = jnp.asarray(rng.uniform(-1.0, 1.0, size=(ms["n_cells"],)))
        bvals = {
            name: jnp.asarray(
                rng.uniform(-1.0, 1.0, size=(ms["patch_owner"][name].size,))
            )
            for name in all_patches
        }
        real_starved = grad_least_squares(phi, mesh, None)  # THE DEFAULT
        mine_starved = transcribed_grad(mesh, ms, phi, (), {})
        real_fed = grad_least_squares(phi, mesh, bvals)
        mine_fed = transcribed_grad(mesh, ms, phi, all_patches, bvals)
        max_diff["starved"] = max(
            max_diff["starved"],
            float(np.max(np.abs(np.asarray(real_starved) - np.asarray(mine_starved)))),
        )
        max_diff["fed"] = max(
            max_diff["fed"],
            float(np.max(np.abs(np.asarray(real_fed) - np.asarray(mine_fed)))),
        )
    print(f"  [{label}] max abs diff, boundary-starved (None):         {max_diff['starved']!r}")
    print(f"  [{label}] max abs diff, boundary-fed (all patches):      {max_diff['fed']!r}")
    assert max_diff["starved"] == 0.0, f"{label}: not bit-identical (starved)"
    assert max_diff["fed"] == 0.0, f"{label}: not bit-identical (fed)"
    d_int, patch_d = concrete_deltas(mesh)
    for fed_cfg, cfg_name in (((), "starved"), (all_patches, "fed")):
        M_mine = np.asarray(assemble_M_family(ms, d_int, patch_d, fed_cfg))
        M_oracle = np.asarray(M_via_segment_sum(mesh, fed_cfg))
        identical = bool(np.array_equal(M_mine, M_oracle))
        print(f"  [{label}] exact M cross-check, {cfg_name}: identical={identical}")
        assert identical, f"{label}: transcribed M != segment_sum M ({cfg_name})"
    return max_diff


print("\n==== fidelity check F: transcription vs the REAL grad_least_squares")
print("  5 random phi fields per mesh (seed 20260721), full pipeline both sides")
run_fidelity("2x1", MESH)

# the M values and condition numbers the KAs are about (2x1)
_STATIC = mesh_static(MESH)
_d_int_c, _patch_d_c = concrete_deltas(MESH)
_M_mine_fed = assemble_M_family(_STATIC, _d_int_c, _patch_d_c, PATCH_NAMES)
_M_mine_starved = assemble_M_family(_STATIC, _d_int_c, _patch_d_c, ())
print(f"  M boundary-fed (both cells, measured):     {np.asarray(_M_mine_fed)[0].tolist()}")
print(f"  M boundary-starved (both cells, measured): {np.asarray(_M_mine_starved)[0].tolist()}")
_cond_fed = [float(np.linalg.cond(np.asarray(_M_mine_fed)[i], 2)) for i in range(N_CELLS)]
_cond_starved = [float(np.linalg.cond(np.asarray(_M_mine_starved)[i], 2)) for i in range(N_CELLS)]
print(f"  cond_2 per cell, fed:     {_cond_fed}")
print(f"  cond_2 per cell, starved: {_cond_starved}")
assert all(c < 2.0 for c in _cond_fed)
assert all(c > 1e29 for c in _cond_starved)

# the non-congruent-stencil mesh: same full stack, bit-identity asserted
MESH3 = make_cartesian_mesh_2d(3, 2, 1.0, 1.0, dtype=jnp.float64)
assert mesh_static(MESH3)["n_cells"] == 6 and mesh_static(MESH3)["n_faces"] == 7
run_fidelity("3x2", MESH3)

print("  what pins what: F pins the float-visible assembly-to-output path; the")
print("  exact M cross-checks pin the assembled values in both configurations;")
print("  reg is pinned by the signature read+assert (a 10x reg error is invisible")
print("  to F alone in both configurations: absorbed in fed, nullspace-aligned in")
print("  starved — the starved M cross-check catches it by value).")
print("  residual limit: mutations that produce value-identical assemblies on every")
print("  checked mesh (transpose of the symmetric M, owner/neighbour role swap,")
print("  sign flips under d⊗d, the 180° cell permutation of a centro-symmetric")
print("  Cartesian box) are undetectable by value comparison, which is why the")
print("  transcription also cites and mirrors the source line-by-line.")

# --- KA-A: the scar — boundary-starved IS the default ------------------------
#
# grad_least_squares' DEFAULT is boundary_face_values=None: NO boundary
# patch contributes to M (the operators.py:155-157 guard skips every
# patch), so on this mesh each cell's M is the rank-1 one-face stencil
# d⊗d + reg·I — the probe's Part 1 matrix. Mesh data enter as POINT
# declarations at their real values: theta is the interior x-delta
# d[0,0] = 0.5, declared by the template; the remaining component is its
# own point declaration inside the transform.
#
# WHY declarations and not transcribed consts: the contract's claim is
# posed over a declared envelope, and a point envelope is still an
# envelope — this is the same code path an adopter will use with real
# ranges (KA-C widens exactly these declarations into boxes). The
# fidelity check above deliberately used concrete arrays instead,
# because it is a measurement of the transcription, not a claim over a
# declared set.

print("\n==== KA-A: boundary-starved (the real default), point declarations")


def _declared_point(v):
    return any_array((), "float64", (float(v), float(v)))


def transform_starved(theta):
    d_int = [(theta, _declared_point(D_INT[0, 1]))]
    return assemble_M_family(_STATIC, d_int, {}, fed=())


ka_a = check_contract(
    conditioning_2x2_field(
        (), "float64", (float(D_INT[0, 0]), float(D_INT[0, 0])), KAPPA,
        transform_starved,
    ),
    vacuity_mode=VMODE,
)
print(ka_a.render())
assert ka_a.requires_status == "REFUTED"
assert [o.status for o in ka_a.requires.obligations] == [
    "discharged", "discharged", "discharged", "violated-over-set",
    "unknown", "unknown",  # symmetry pair straddles at outward-rounded ±0
]
assert "definitely false for 2/2 element(s)" in ka_a.requires.obligations[3].detail
assert isinstance(ka_a.requires.stamp.solver, SolverStamp)
assert not ka_a.requires.stamp.solver.invoked
assert "no solver invoked" in ka_a.requires.stamp.solver.reason
assert ka_a.ensures_status == "DECLARED"

# --- KA-B: boundary-fed, same mesh -------------------------------------------
#
# All four patches given boundary values. The VALUES are irrelevant to
# M — operators.py:155-160 adds patch.d ⊗ patch.d for every patch whose
# NAME is present in boundary_face_values; the values themselves touch
# only the rhs (:161-165). So the contract's family is a function of
# patch PRESENCE alone, and this transform feeds all four.

print("\n==== KA-B: boundary-fed (all four patches present), point declarations")


def transform_fed(theta):
    d_int = [(theta, _declared_point(D_INT[0, 1]))]
    patch_d = {
        name: [
            (_declared_point(PATCH_D[name][k, 0]), _declared_point(PATCH_D[name][k, 1]))
            for k in range(PATCH_OWNER[name].size)
        ]
        for name in PATCH_NAMES
    }
    return assemble_M_family(_STATIC, d_int, patch_d, fed=PATCH_NAMES)


ka_b = check_contract(
    conditioning_2x2_field(
        (), "float64", (float(D_INT[0, 0]), float(D_INT[0, 0])), KAPPA,
        transform_fed,
    ),
    vacuity_mode=VMODE,
    solver_timeout_ms=TIMEOUT_MS,  # the posed symmetry pair straddles at
    # outward-rounded ±0 even on this all-point envelope and needs the
    # trivial solver discharge; the four conditioning conjuncts are
    # interval-decided with wide margins
)
print(ka_b.render())
assert ka_b.requires_status == "VERIFIED"
for i in range(4):
    ob = ka_b.requires.obligations[i]
    assert ob.status == "discharged" and "definitely true" in ob.detail
for i in (4, 5):
    ob = ka_b.requires.obligations[i]
    assert ob.status == "discharged" and "solver escalation" in ob.detail
_inert = [
    a for a in ka_b.requires.stamp.assumptions
    if "vacuity instrument inert (mode=inputs-only)" in a
]
assert len(_inert) == 1, "the committed inert line is missing — a finding"
print(f"\n  KA-B vacuity line, verbatim: {_inert[0]}")
assert "every declared input is a point interval" in _inert[0]
# the inert path runs no widen re-run: nothing may be tagged as one
_stamps_b = ka_b.requires.stamp.solver
assert isinstance(_stamps_b, tuple) and _stamps_b
assert not any(s.reason.startswith("vacuity widen re-check: ") for s in _stamps_b)

# --- KA-C: the geometric region — a FAMILY of geometries ---------------------
#
# Declared boxes on the delta components through the SAME transcribed
# assembly: theta[0] is the interior x-delta magnitude, theta[1] the y
# boundary delta magnitude, both over one closed box; the x boundary
# deltas are tied to the interior by the Cartesian constructor's own
# relation patch.d = n * (dx/2) (mesh.py:309, :326-330), and the zero
# components are transcribed consts (the axis-aligned family). Every
# member IS a realizable Cartesian 2x1 geometry (dx = theta[0],
# dy = 2*theta[1]); the real mesh is the member theta = (0.5, 0.5).
#
# Arithmetic of the region (derived here, MEASURED below): per cell
#   M00 = t0^2 + (t0/2)^2 + reg = 1.25*t0^2 + reg,  M11 = 2*t1^2 + reg,
#   off-diagonals exactly 0
# so cond_2 = max(M00, M11)/min(M00, M11). Over [0.4, 0.6]^2 the worst
# corner is (0.4, 0.6): cond = 0.72/0.2 = 3.6 <= kappa = 8 — the narrow
# region satisfies the requires everywhere. Over the widened [0.4, 1.6]^2
# the corner (0.4, 1.6) gives cond = 5.12/0.2 = 25.6 > 8 — violated
# somewhere.

print("\n==== KA-C: declared boxes on delta components (axis-aligned family)")


# Scale note (measured, the budget is a committed decision and stands):
# this 2-cell family is far inside the ONE per-obligation emission budget
# (stelling.obligation.ELEMENT_BUDGET = 512 element terms). With this
# per-cell assembly shape (~11 element terms per matrix) the conditioning
# conjunct's solver escalation runs up to N = 46 matrices and declines
# loudly at N = 47 ("517 element terms ... over the per-obligation
# emission budget of 512", both numbers quoted); interval-decidable
# obligations keep working at any N. On a real mesh, pose deliberate
# sub-families — a region, a sampled subset, a known-worst cell class —
# rather than the whole mesh.


def axis_aligned_contract(theta_range):
    def transform(theta):
        t0 = theta[0]
        t1 = theta[1]
        neg_t1 = -t1
        d_int = [(t0, 0.0)]
        xb = t0 * 0.5  # mesh.py:309 — patch d = normal * (dx/2)
        patch_d = {
            "x_min": [(-xb, 0.0)],
            "x_max": [(xb, 0.0)],
            "y_min": [(0.0, neg_t1), (0.0, neg_t1)],
            "y_max": [(0.0, t1), (0.0, t1)],
        }
        return assemble_M_family(_STATIC, d_int, patch_d, fed=PATCH_NAMES)

    return conditioning_2x2_field((2,), "float64", theta_range, KAPPA, transform)


# narrow region, interval-only: the reduction is dependency-shaped
ka_c_iv = check_contract(axis_aligned_contract((0.4, 0.6)), vacuity_mode=VMODE)
assert ka_c_iv.requires_status == "UNKNOWN"
assert ka_c_iv.requires.obligations[3].status == "unknown"
_straddle3 = [
    n for n in ka_c_iv.requires.notes
    if "straddles" in n and "obligation #3" in n
]
assert len(_straddle3) == 1
print(f"  narrow [0.4, 0.6], interval path: {ka_c_iv.requires_status}")
print(f"  straddle, verbatim: {_straddle3[0]}")

# narrow region, escalated: VERIFIED through QF_NRA
ka_c_v = check_contract(
    axis_aligned_contract((0.4, 0.6)), vacuity_mode=VMODE,
    solver_timeout_ms=TIMEOUT_MS,
)
print(ka_c_v.render())
assert ka_c_v.requires_status == "VERIFIED"
_ratio = ka_c_v.requires.obligations[3]
assert _ratio.status == "discharged"
assert "solver escalation (QF_NRA)" in _ratio.detail and "unsat" in _ratio.detail
_vac = [
    a for a in ka_c_v.requires.stamp.assumptions
    if a.startswith("vacuity checked (mode=inputs-only)")
]
assert len(_vac) == 1
print(f"\n  KA-C narrow vacuity line, verbatim: {_vac[0]}")
# the envelope must be load-bearing exactly for the conditioning
# reduction: obligation 3 may NOT discharge with the bounds widened
# (positivity/PSD/symmetry of this family are construction theorems and
# legitimately survive widening; the cond bound is the geometric claim)
assert not any(
    n.startswith("obligation #3: discharges with all declared bounds widened")
    for n in ka_c_v.requires.notes
)

# widened region, escalated: REFUTED with a replay-confirmed witness
ka_c_r = check_contract(
    axis_aligned_contract((0.4, 1.6)), vacuity_mode=VMODE,
    solver_timeout_ms=TIMEOUT_MS,
)
print(ka_c_r.render())
assert ka_c_r.requires_status == "REFUTED"
assert ka_c_r.requires.witnesses
for w in ka_c_r.requires.witnesses:
    assert w.obligation_index == 3  # the conditioning reduction, and only it
    assert "exact-rational replay" in w.replay
    vals = {name: Fraction(text) for name, text in w.values}
    t0, t1 = vals["x0_0"], vals["x0_1"]
    # realizable geometry inside the declared boxes (exact dyadic bounds)
    assert Fraction(0.4) <= t0 <= Fraction(1.6)
    assert Fraction(0.4) <= t1 <= Fraction(1.6)
    # the witness's condition number, re-derived by hand in EXACT
    # arithmetic from the transcription's own entry expressions
    m00 = t0 * t0 + (t0 * Fraction(1, 2)) ** 2 + Fraction(REG)
    m11 = t1 * t1 + t1 * t1 + Fraction(REG)
    hi, lo_ = max(m00, m11), min(m00, m11)
    assert hi > Fraction(KAPPA) * lo_, "witness cond_2 must exceed kappa"
    print(
        f"  witness t0={float(t0)!r}, t1={float(t1)!r}: a Cartesian 2x1 "
        f"geometry with dx={float(t0)!r}, dy={float(2 * t1)!r}; "
        f"cond_2 = {float(hi / lo_)!r} > {KAPPA}"
    )

# --- summary -----------------------------------------------------------------

print("\n==== summary")
print(
    "KA-A boundary-starved (the real default): requires REFUTED — interval "
    "path alone, no solver invoked; the ratio conjunct definitely false for "
    "both cells (the probe's Part 1 matrix)."
)
print(
    "KA-B boundary-fed (all four patches PRESENT — values never enter M): "
    "requires VERIFIED — conditioning conjuncts interval-decided, posed "
    "symmetry pair solver-discharged; vacuity instrument inert (all-point "
    "envelope); ensures face DECLARED, never checked."
)
print(
    "KA-C geometric region: narrow [0.4, 0.6] interval UNKNOWN with the "
    "straddle quoted, then solver VERIFIED (QF_NRA); widened [0.4, 1.6] "
    "solver REFUTED with a replay-confirmed witness whose cond_2 exceeds "
    "kappa, re-derived exactly in-script."
)
