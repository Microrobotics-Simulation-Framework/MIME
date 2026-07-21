"""The MIME socket's mutation battery, gauged by stelling.fidelity.

MIME-side per stelling's ARCHITECTURE.md Rule 1, companion to
``mime_lsq_conditioning.py`` (the socket). This file measures the
socket's fidelity gate stack the L21 way — a fidelity check pins only
what it can distinguish, and distinguishing power is measured with
deliberately WRONG assemblies, never asserted. The loop itself is
``stelling.fidelity.gauge``, which refuses to bless the stack if the
battery is empty, the baseline fails a gate, a survivor is unexplained,
or a residual explanation is measurably stale.

Subjects: per-mutation transcribed assemblers (M and rhs), numpy, with
one subtotal per segment_sum in the real code's association order (float
addition is non-associative; a flat merged sum would differ bitwise on
the 3x2 mesh for reasons that are not mutations). Gates, 8 = 2 meshes x
(output-fidelity F in both boundary configurations + exact M cross-check
in both configurations):

* ``F-*``: the mutated pipeline (mutated M, rhs, jnp.linalg.inv +
  einsum) vs the REAL ``grad_least_squares`` import, 5 seeded random phi
  fields, bit-identity;
* ``Mcheck-*``: the mutated M values vs the real assembly formula run
  literally with ``jax.ops.segment_sum`` on the mesh's own arrays.

Residual class: the five mutations that are value-identical BY ALGEBRA
on every Cartesian box; each explanation below is a claim the gauge
re-measures every run (a residual mutation that becomes catchable makes
the run FAIL as a stale claim).

Hard-asserted here beyond the gauge's own refusals, because they are the
load-bearing facts of the round-2 audit (stelling design/la-attachment.md):
the baseline passes every gate, and ``reg_10x`` is caught — by exactly
the starved M cross-check on the 2x1 mesh (the gate added when
measurement showed F alone is blind to it: absorbed in the fed
configuration, nullspace-aligned in the starved one).

Run (any cwd, a python with stelling + mime + jax importable):

    python MIME/verification/stelling/battery.py
"""

from importlib.metadata import version as _pkg_version

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from mime.nodes.environment.fvm.mesh import make_cartesian_mesh_2d  # noqa: E402
from mime.nodes.environment.fvm.operators import grad_least_squares  # noqa: E402

import stelling  # noqa: E402
from stelling.fidelity import gauge  # noqa: E402

REG = 1e-30
SEED = 20260721
N_TRIALS = 5

try:
    MIME_VERSION = _pkg_version("mime-engine")
except Exception:  # pragma: no cover - version report only
    MIME_VERSION = "unversioned"

print(
    f"pinned: mime-engine {MIME_VERSION} @ 7ce1efb4311b | "
    f"jax {jax.__version__} | stelling {stelling.__version__}"
)

MUTATIONS = (
    "drop_neigh", "swap_x_patch_d", "swap_x_patch_d_shared",
    "reg_10x", "reg_zero", "reg_offdiag", "swap_cells_M", "transpose_M",
    "drop_y_max_in_M", "owner_neigh_swapped_in_M", "d_sign_flip_in_M",
)

RESIDUAL = {
    "transpose_M": (
        "M is symmetric by construction: transposing it is the identity "
        "on its values"
    ),
    "owner_neigh_swapped_in_M": (
        "M's owner+neighbour sum is role-symmetric: swapping the segments "
        "reorders nothing value-visible"
    ),
    "d_sign_flip_in_M": "(-d)(x)(-d) = d(x)d: the sign is squared away",
    "swap_x_patch_d": (
        "x_min/x_max patch deltas differ only in sign on a Cartesian box "
        "and enter M as d(x)d (sign squared away); the *_shared variant "
        "flows into rhs and IS caught"
    ),
    "swap_cells_M": (
        "the 180-degree cell permutation is a symmetry of every Cartesian "
        "box (the constructor's boxes are centro-symmetric): the permuted "
        "M is value-identical per cell"
    ),
}


def statics(mesh):
    return dict(
        n_cells=mesh.N_cells, n_faces=mesh.N_faces,
        owner=np.asarray(mesh.owner), neigh=np.asarray(mesh.neighbour),
        d_int=np.asarray(mesh.d),
        patch_names=tuple(p.name for p in mesh.patches),
        patch_owner={p.name: np.asarray(p.owner) for p in mesh.patches},
        patch_d={p.name: np.asarray(p.d) for p in mesh.patches},
    )


def build(st, mut):
    """(assemble_M, assemble_rhs) with mutation `mut` applied; 'none' is
    the faithful transcription (the baseline)."""
    own, nei = st["owner"], st["neigh"]
    d_int = st["d_int"].copy()
    patch_d = {k: v.copy() for k, v in st["patch_d"].items()}
    reg = REG
    drop_neigh = mut == "drop_neigh"
    drop_patch = mut == "drop_y_max_in_M"
    reg_offdiag = mut == "reg_offdiag"
    swap_cells = mut == "swap_cells_M"
    transpose = mut == "transpose_M"
    if mut in ("swap_x_patch_d", "swap_x_patch_d_shared"):
        patch_d["x_min"], patch_d["x_max"] = patch_d["x_max"], patch_d["x_min"]
    shared_rhs = mut == "swap_x_patch_d_shared"
    if mut == "reg_10x":
        reg = 1e-29
    if mut == "reg_zero":
        reg = 0.0
    if mut == "owner_neigh_swapped_in_M":
        own, nei = st["neigh"], st["owner"]
    if mut == "d_sign_flip_in_M":
        d_int = -d_int

    def assemble_M(fed):
        M = np.zeros((st["n_cells"], 2, 2))
        for P in range(st["n_cells"]):
            groups = []
            g = [np.outer(d_int[f], d_int[f])
                 for f in range(st["n_faces"]) if own[f] == P]
            if g:
                groups.append(g)
            if not drop_neigh:
                g = [np.outer(d_int[f], d_int[f])
                     for f in range(st["n_faces"]) if nei[f] == P]
                if g:
                    groups.append(g)
            for name in st["patch_names"]:
                if name not in fed or (drop_patch and name == "y_max"):
                    continue
                po = st["patch_owner"][name]
                g = [np.outer(patch_d[name][k], patch_d[name][k])
                     for k in range(po.size) if po[k] == P]
                if g:
                    groups.append(g)
            acc = None
            for g in groups:
                s = None
                for t in g:
                    s = t if s is None else s + t
                acc = s if acc is None else acc + s
            M[P] = acc
        M = M + (reg * np.ones((2, 2))[None] if reg_offdiag
                 else reg * np.eye(2)[None])
        if swap_cells:
            M = M[::-1]
        if transpose:
            M = np.transpose(M, (0, 2, 1))
        return M

    def assemble_rhs(phi, fed, bvals):
        # rhs stays the CORRECT transcription (mutations target M only),
        # except shared-data mutations, which flow into rhs naturally
        rhs_pd = patch_d if shared_rhs else st["patch_d"]
        r = np.zeros((st["n_cells"], 2))
        for P in range(st["n_cells"]):
            groups = []
            for seg in (st["owner"], st["neigh"]):
                g = []
                for f in range(st["n_faces"]):
                    if seg[f] == P:
                        dphi = phi[int(st["neigh"][f])] - phi[int(st["owner"][f])]
                        g.append(dphi * st["d_int"][f])
                if g:
                    groups.append(g)
            for name in st["patch_names"]:
                if name not in fed:
                    continue
                po = st["patch_owner"][name]
                g = []
                for k in range(po.size):
                    if po[k] == P:
                        g.append((bvals[name][k] - phi[int(po[k])]) * rhs_pd[name][k])
                if g:
                    groups.append(g)
            acc = None
            for g in groups:
                s = None
                for t in g:
                    s = t if s is None else s + t
                acc = s if acc is None else acc + s
            r[P] = acc
        return r

    return assemble_M, assemble_rhs


def M_oracle(st, fed):
    """The real assembly formula run literally (jax.ops.segment_sum on the
    mesh's own arrays) — the value oracle for the Mcheck gates."""
    d = st["d_int"]
    dd = d[:, :, None] * d[:, None, :]
    M = jax.ops.segment_sum(jnp.asarray(dd), jnp.asarray(st["owner"]),
                            num_segments=st["n_cells"]) + \
        jax.ops.segment_sum(jnp.asarray(dd), jnp.asarray(st["neigh"]),
                            num_segments=st["n_cells"])
    for name in st["patch_names"]:
        if name not in fed:
            continue
        db = st["patch_d"][name]
        M = M + jax.ops.segment_sum(
            jnp.asarray(db[:, :, None] * db[:, None, :]),
            jnp.asarray(st["patch_owner"][name]),
            num_segments=st["n_cells"])
    return np.asarray(M + REG * jnp.eye(2, dtype=M.dtype)[None])


MESHES = {
    "2x1": make_cartesian_mesh_2d(2, 1, 1.0, 1.0, dtype=jnp.float64),
    "3x2": make_cartesian_mesh_2d(3, 2, 1.0, 1.0, dtype=jnp.float64),
}
STATICS = {label: statics(mesh) for label, mesh in MESHES.items()}
ORACLES = {
    (label, cfg): M_oracle(st, st["patch_names"] if cfg == "fed" else ())
    for label, st in STATICS.items()
    for cfg in ("starved", "fed")
}


def subject_for(mut):
    """A subject = the mutation's per-mesh assembler pair; opaque to the
    gauge, unpacked by the gates."""
    return {label: build(st, mut) for label, st in STATICS.items()}


def f_gate(label, cfg):
    """PASS = the mutated pipeline is bit-identical to the real
    grad_least_squares on `label`'s mesh in configuration `cfg`, over
    N_TRIALS seeded random fields."""
    mesh, st = MESHES[label], STATICS[label]

    def gate(subject):
        aM, arhs = subject[label]
        rng = np.random.RandomState(SEED)
        for _ in range(N_TRIALS):
            phi = rng.uniform(-1.0, 1.0, size=(st["n_cells"],))
            bvals = {n: rng.uniform(-1.0, 1.0, size=(st["patch_owner"][n].size,))
                     for n in st["patch_names"]}
            fed = st["patch_names"] if cfg == "fed" else ()
            bv = bvals if cfg == "fed" else {}
            M = aM(fed)
            mine = np.asarray(jnp.einsum(
                "cij,c...j->c...i", jnp.linalg.inv(jnp.asarray(M)),
                jnp.asarray(arhs(phi, fed, bv))))
            real = np.asarray(grad_least_squares(
                jnp.asarray(phi), mesh, bv if cfg == "fed" else None))
            if not np.array_equal(real, mine, equal_nan=False):
                return False
        return True

    return gate


def m_gate(label, cfg):
    """PASS = the mutated M equals the segment_sum oracle exactly."""
    st = STATICS[label]

    def gate(subject):
        aM, _ = subject[label]
        fed = st["patch_names"] if cfg == "fed" else ()
        return bool(np.array_equal(aM(fed), ORACLES[(label, cfg)]))

    return gate


GATES = {}
for _label in MESHES:
    for _cfg in ("starved", "fed"):
        GATES[f"{_label}:F-{_cfg}"] = f_gate(_label, _cfg)
        GATES[f"{_label}:Mcheck-{_cfg}"] = m_gate(_label, _cfg)

report = gauge(
    subject_for("none"),
    GATES,
    {mut: subject_for(mut) for mut in MUTATIONS},
    residual=RESIDUAL,
)
print(report.render())

# the two load-bearing facts, hard-asserted beyond the gauge's refusals
assert report.gates == tuple(GATES), "baseline did not pass the full stack"
reg_10x_catchers = dict(report.caught_by)["reg_10x"]
assert reg_10x_catchers, "reg_10x must be caught"
assert "2x1:Mcheck-starved" in reg_10x_catchers, (
    "reg_10x must be caught by the starved M cross-check on 2x1 — the "
    "gate that exists because F alone is measurably blind to it"
)
print(
    "\nload-bearing facts asserted: baseline passes every gate; reg_10x "
    f"caught by {list(reg_10x_catchers)} (the starved M cross-check among "
    "them)."
)
