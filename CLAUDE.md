# MIME — working notes for Claude

## Reasoning discipline

A recurring failure mode: mechanistically plausible explanations asserted before the
operating point was checked. Five wrong claims followed this pattern in a single session
(cancellation-ratio "finding", planar-array push-away, fit-validity box, EP-0c data loss,
gate misconfiguration), plus one still in doubt (cpr=4 sphere "broken"). Every one was
caught by a control arm, not by the primary analysis.

Rules that follow:

- **Before asserting a mechanism, check the operating point.** "Plausible" is not
  "confirmed." The gate experiment runs at Re≈19 (cell-Pe ≈ 0.4–3); a 22.5% error measured
  at cell-Pe ≈ 39 says nothing about it. Read the parameters before reading the comment.

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

- **Precision is per-rig, not global.** The schwarz gate experiment needs
  `JAX_ENABLE_X64=1` (float32 → NaN) and `JAX_DEFAULT_MATMUL_PRECISION=highest` on GPU.
  Standalone sphere/FVM rigs run fine in float32 — blanket-enabling x64 costs 2× for
  nothing. If unsure, run both and compare.

- **`gm.step({})` is almost never right.** Check a graph's external inputs and feed them.
  The schwarz graph needs four (FVM sample points, FVM forcing points, motor
  `commanded_velocity`, body `spin_rate`), normally supplied by its controller.

- **`@pytest.mark.slow` and `@pytest.mark.gpu` mean "does not run in CI"**
  (`addopts = "-m 'not slow'"`). `testpaths = ["tests"]`, so `scripts/` is never collected.
  A physics assertion behind either marker is documentation, not a test.

- **The `new-node` skill documents the wrong `@verification_benchmark` signature.** The
  real one is `(benchmark_id, description, node_type, benchmark_type, acceptance_criteria,
  references=())` from `maddening.core.compliance.validation`.

- **eMNS calibrations**: `Source_Position` is a fitted basis function, not hardware
  (OctoMag has sources 245 km out). `Workspace_Dimensions` is declared, not fitted.
  Upstream `getGradient()` returns the *field*. See
  `src/mime/data/emns_calibrations/README.md`.

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
