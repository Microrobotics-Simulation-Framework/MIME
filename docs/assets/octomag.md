# OctoMag's calibration

**What it is.** The calibrated multipole model (MPEM, order one) of OctoMag, eight coils about a small workspace: each
coil's sources, each a fitted position, direction and coefficient, and the system's declared region of interest.
Packaged as `asset("octomag")`, read by the field provider `CalibratedMultipole`.

**The file.** `src/mime/data/emns_calibrations/octomag.yaml`, byte for byte its upstream's, which every read checks
(`mime.data.asset_bytes`), its sha256:

`d0d92541ea3669bb1c4d4daf4acd36b9d3100f728e6270e2aebac53906cc0ad0`

**Its upstream.** [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public),
`mag_control/mpem/cal/OctoMag_Calibration.yaml` at `5ef021b0f4c0b91d1d8f3aedf353fe24bf793e84`.

**Its terms.** Apache-2.0, by the file's own header, retained; credit: Multi-Scale Robotics Lab, ETH Zurich. Its tier is
open: it ships in the wheel.

**Each value's basis.**
- `Source_Position`, `Source_Direction`, `B_Coeff`: fitted by the system's makers. A source is a basis function, not
  hardware; its position need not be a coil's.
- `Workspace_Dimensions`: declared, never fitted (the upstream fitter never writes it). Being inside it is no evidence
  of accuracy, and being outside it none of extrapolation; the fit's own domain is not published.

**Its conditioning** (`Calibration.cancellation` and `floor`, over the declared region): cancellation 6.9e-5; float64
rounding 3.2e-12 and float32 1.7e-3 of the field, above the guard: refused in float32; least distance from a source to
the region 20 um, at a corner; furthest source 245 km.

**Its accuracy, by use.** Against ETH's `mag_manip`, MIME's evaluation agrees to a tolerance derived from the field's
cancellation at the compared points (`tests/oracles/test_calibrated_multipole.py`): a software oracle, which pins sign
and frame to mag_manip's convention and cannot ground them against hardware. Its fit places sources far out with large
coefficients, whose fields nearly cancel in its small region: the answer is the fit's, and computing it costs digits,
which is why float32 is refused. Its cancellation is not converged on the grid: finer grids find it deeper.
