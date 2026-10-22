# MIME's stelling socket

This directory is the **living stelling verification socket for MIME**,
per stelling's `ARCHITECTURE.md` Rule 1: a socket (harnesses and
envelopes, known answers, fidelity gates and mutation battery,
consequence declarations) encodes maintainer knowledge and lives with
the library it verifies — MIME's repo, MIME's CI. stelling ships the
instruments (`stelling.contracts`, `stelling.fidelity`); MIME owns this
surface. The frozen record of the pass that produced it is stelling's
`design/la-attachment.md`; the KA **statuses** below must match that
table across maintenance (query hashes may drift; statuses may not).

## Files

- `mime_lsq_conditioning.py` — the socket. Attaches
  `stelling.contracts.conditioning_2x2_field` to the real
  `grad_least_squares` normal-matrix assembly
  (`src/mime/nodes/environment/fvm/operators.py:119-172`), with the
  segment sums unrolled over the real mesh's static connectivity.
  Sections, each hard-asserted so the script is its own gate: the
  `segment_sum`-under-the-tracer seam (measured `scatter-add` decline);
  fidelity check F (transcribed pipeline vs the real import,
  bit-identity, both boundary configurations, on the 2x1 KA mesh AND a
  non-congruent 3x2 mesh) layered with exact M cross-checks in both
  configurations and the `reg` signature pin; then the known answers:
  - **KA-A** boundary-starved (the real `boundary_face_values=None`
    default): requires **REFUTED** by intervals alone, both cells, no
    solver — the rank-1 stencil `d⊗d + 1e-30·I`, cond ≈ 2.5e29;
  - **KA-B** all four patches fed (values never enter M — presence
    does): **VERIFIED**, symmetry pair trivially solver-discharged,
    inert-vacuity line recorded verbatim;
  - **KA-C** delta boxes through the same assembly: narrow [0.4, 0.6]
    interval **UNKNOWN** (straddle quoted) then solver **VERIFIED**
    (QF_NRA); widened [0.4, 1.6] solver **REFUTED** with a
    replay-confirmed witness whose condition number is re-derived
    exactly in-script (> kappa = 8).
- `battery.py` — the twelve-assembly mutation battery, rebuilt on
  `stelling.fidelity.gauge`. Eight gates (2 meshes x output-fidelity F
  and exact M cross-check in both configurations) against eleven
  deliberately wrong assemblies plus the faithful baseline. Asserts the
  two load-bearing facts: the baseline passes every gate, and `reg_10x`
  is caught by exactly the starved M cross-check on 2x1 (F alone is
  measurably blind to a 10x `reg` error: absorbed when boundary-fed,
  nullspace-aligned when starved). The five value-identical survivors
  (transpose of the symmetric M, owner/neighbour role swap, d sign
  flips squared away by `d⊗d`, the x-patch delta swap, the 180° cell
  permutation of a centro-symmetric box) are declared as the residual
  class with their algebraic explanations — and the gauge re-measures
  those explanations every run, refusing on a stale claim.

## Running

Any working directory; a Python with `mime`, `stelling`, `jax` (0.11.x,
x64), `numpy`, and for the socket's solver steps `z3`/`cvc5`:

    python MIME/verification/stelling/mime_lsq_conditioning.py
    python MIME/verification/stelling/battery.py

Both exit 0 with every assert green, or die loudly at the first
falsified claim.

## Version pins

- mime-engine @ `7ce1efb4311b` (the commit the transcription cites
  line-by-line; `operators.py` + `mesh.py` are the only MIME files
  read or imported).
- stelling `0.1.0` **at or after commit `3f78fdd`** (jax 0.11.0 under
  `jax_enable_x64=True`); both files print the live pin line at
  startup. The commit hash matters because the version number did not
  move: `3f78fdd` added censused scatter-add (and stack) rows to
  stelling, and the socket's seam probe flipped at that boundary from
  measuring the old wall (obligation ⊤/UNKNOWN, escalation declining by
  name) to pinning the supported behavior (full coverage, sound
  scatter-add transfer, interval-discharged) — on pre-`3f78fdd`
  stelling the seam section fails its asserts by design.

## Why a battery ships next to the socket (L21)

A fidelity check pins only what it can distinguish, and its
discriminating power is a thing you **measure** with deliberately wrong
variants — not a thing you assert. This battery is that measurement,
and `stelling.fidelity.gauge` is the shipped loop that refuses to bless
an unmeasured stack: no mutations, a failing baseline, an unexplained
survivor, or a stale residual explanation each abort the run instead of
producing a report. If maintenance changes the assembly or the gates,
the battery — not anyone's memory — says what the stack can still see.
