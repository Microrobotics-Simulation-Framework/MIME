# MIME — working notes for Claude

## Reasoning discipline

A recurring failure mode: mechanistically plausible explanations asserted before the
operating point was checked. Five wrong claims followed this pattern in a single session
(cancellation-ratio "finding", planar-array push-away, fit-validity box, EP-0c data loss,
gate misconfiguration), plus one still in doubt (cpr=4 sphere "broken"). Every one was
caught by a control arm, not by the primary analysis.

Rules that follow:

- **Before asserting a mechanism, check the operating point.** "Plausible" is not
  "confirmed." The 0.2 line's gate experiment ran at Re≈19 (cell-Pe ≈ 0.4–3); a 22.5% error
  measured at cell-Pe ≈ 39 said nothing about it. Read the parameters before reading the comment.

- **A test that cannot discriminate is not a null result.** If both arms NaN, or if the
  metric is invariant under the thing being tested, the test produced no evidence. Say so
  explicitly rather than reading absence of difference as confirmation.

- **A scalar cannot rotate a vector.** Check that the correction being applied is
  structurally capable of testing what it is cited as ruling out — before running it.
  Every previous "not per-sensor gain" / "not per-coil gain" conclusion about a *direction*
  error was uninformative for exactly this reason.

- **Controls are load-bearing.** The Re→0 arm (γ provably irrelevant with no convection),
  the golden oracle, the subharmonicity identity, and the subagent instructed to *refute*
  rather than confirm all did more work than the primary analyses. Attach a control to
  anything consequential.

- **Prefer the metric that predicts.** A quantity that merely correlates with the failure
  invites the wrong fix. `field_cancellation_ratio` (field-space, 1/r³-weighted) predicts
  measured float32 error to ~1.4× across a 3600× spread; its parametric moment-space
  predecessor ranked the same models in the *wrong order*. If a metric cannot predict the
  number you care about, it is not the metric.

- **Distinguish "unvalidated" from "wrong", and "declared" from "fitted".** An accuracy
  envelope that was never measured is *unknown*, not *bounded*. A number in a config file
  is a declaration until something fits it.

## Domain gotchas

- **Precision is per rig, not global.** Some coupled rigs need `JAX_ENABLE_X64=1` and, on a
  GPU, `JAX_DEFAULT_MATMUL_PRECISION=highest`: the 0.2 line's Schwarz gate experiment went NaN
  in float32, and its ar4 run was 15% off on the GPU under reduced-precision (TF32) matmuls in
  the MLP drag. Standalone rigs run fine in float32, where blanket x64 costs 2× for nothing.
  If unsure, run both and compare.

- **Feed every external input.** MADDENING 0.4 zero-fills an external input nobody feeds,
  which is how the 0.2 line's ar4 graph went NaN (a zero quaternion). In 0.3 a graph's
  external inputs come only from a controller's declaration, and the run feeds each one;
  code that drives MADDENING directly (guard tests, spikes) must do the same. `step({})` is
  almost never right.

- **What CI does not run.** `@pytest.mark.slow` and `@pytest.mark.gpu` tests do not run in
  CI's default lanes (`addopts` has `-m "not slow"`), and `testpaths = ["tests"]`, so
  `scripts/` is never collected. A physics assertion behind either marker is documentation,
  not a test. A test over CI's time budget goes on `tests/duration_allowlist.txt` with its
  reason, not under `slow`; Hypothesis example counts are never shrunk for speed.

- **A skipped test looks like a pass.** `gpu`-marked tests skip silently without a CUDA
  jaxlib: check with `python -c "import jax; print(jax.devices())"`, and `CpuDevice` means
  they did not run, whatever the summary says (in the 0.2 line this left a CPU-only
  environment with no executable test of the FVM wall path). pytest also skips any folder
  named `build` by default: `pyproject.toml` overrides `norecursedirs`, and
  `tests/test_collection.py` fails if a test file goes uncollected.

- **Run pyright as CI does**: inside the environment, or with `--pythonpath
  .venv/bin/python`. Otherwise it cannot resolve MADDENING and reports hundreds of spurious
  errors. `src/mime` is strict at zero errors; jaxlib ships no stubs, so calls whose types
  read as unknown go through `mime.build._maddening`.

- **`@verification_benchmark`'s signature** is `(benchmark_id, description, node_type,
  benchmark_type, acceptance_criteria, references=())`, from
  `maddening.core.compliance.validation`. The `new-node` and `new-verification-test` skills
  give it correctly; older notes and the 0.2 line's skills had it wrong, so don't copy a
  call from those.

- **eMNS calibrations** (they return with the coil-array provider): `Source_Position` is a
  fitted basis function, not hardware (OctoMag has sources 245 km out).
  `Workspace_Dimensions` is declared, not fitted. Upstream `getGradient()` returns the
  *field*. The 0.2 notes are at the tag `pre-0.3`, in
  `src/mime/data/emns_calibrations/README.md`.

## Property tests

Most of phase 4's agent time went into flakes, and every one so far had one of four causes. Check them before
changing a tolerance.

- **Draw what a body can have.** Hypothesis shrinks towards extremes: quaternions whose squares underflow, a torque
  as large as a force on a millimetre lever (10^8 rad in one step), subnormal components. Strategies stay at
  physical scales; numerical extremes are tested on purpose, separately.
- **Judge a result on the scale it came from.** Where terms cancel (projection removing the outward velocity),
  compare against the inputs' scale, not the result's. Tolerances have no absolute part; their floor comes from the
  law's own rounding.
- **Draw in the dtype the node computes in.** Float32 draws under x64 fail the structure checks.
- **A control must fail every run, not most.** Measure a planted error's detection rate before relying on it: the
  contact oracle missed a 1 µm error about one run in eight, because the harness's rounding floor is one number for
  the whole state, about 15 µm on a typical float32 draw (`src/mime/testing/README.md`). Plant above the oracle's
  resolution, or sharpen the resolution.

When one flakes: capture the failing example first (Hypothesis prints a reproduction blob with `print_blob=True`),
classify it by the four causes, then fix. Stress-run a suite in a loop before calling it stable. Never `xfail` or
derandomize a property test to make it pass, and never shrink its example count for speed.

## Claim hygiene

When a validation compares against measurements, state what it **bounds** versus what it
**estimates**. The Navion campaign bounds model error from *above* at ~10% NRMSE; at least
half that residual is provably the measuring apparatus (per-sensor triad misalignment
~3.9°, 97.2% reproducible, absorbing 48% of mean-square under cross-validation). The model
may be considerably better and that dataset cannot tell us. Its narrow genuine value is
grounding **sign and frame against physical hardware** — which the software oracle
(`mag_manip`, 4e-14) structurally cannot do. That is an **acceptance test, not an accuracy
claim**, and it must be written in those terms wherever it appears.

Corollary: do not tighten such a test's threshold by baking in your own error model (e.g.
the per-sensor rotation correction). That is fitting the residual into a "verification"
test.
