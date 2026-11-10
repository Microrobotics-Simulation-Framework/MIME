# OctoMag's calibration

**What it is.** ETH MSRL's published calibration of OctoMag (`OctoMag_Calibration.yaml`, undated in its name), evaluated by MIME: a calibrated multipole model (MPEM, order one) of eight coils about a small workspace, each coil's sources, each a fitted position, direction and coefficient, and the system's declared region of interest.
Packaged as `asset("octomag")`, read by the field provider `CalibratedMultipole`.

**The file.** `src/mime/data/emns_calibrations/octomag.yaml`, byte for byte its upstream's, which every read checks
(`mime.data.asset_bytes`), its sha256:

`d0d92541ea3669bb1c4d4daf4acd36b9d3100f728e6270e2aebac53906cc0ad0`

**Its upstream.** [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public),
`mag_control/mpem/cal/OctoMag_Calibration.yaml` at `5ef021b0f4c0b91d1d8f3aedf353fe24bf793e84`.

**Its terms.** Apache-2.0, by the file's own header, retained; credit: Multi-Scale Robotics Lab, ETH Zurich. Its tier is
open: it ships in the wheel.

**Each value's basis.**
- `Source_Position`, `Source_Direction`, `B_Coeff`: fitted by ETH MSRL. A source is a basis function, not
  hardware; its position need not be a coil's.
- `Workspace_Dimensions`: declared, never fitted (the upstream fitter never writes it). Being inside it is no evidence
  of accuracy, and being outside it none of extrapolation; the fit's own domain is not published.

**Its conditioning** (`Calibration.cancellation` and `floor`, over the declared region): cancellation 6.9e-5; float64
rounding 3.2e-12 and float32 1.7e-3 of the field, above the guard: refused in float32; least distance from a source to
the region 20 um, outside its -x face (0.8 and 0.6 mm from the face's edges, not at a corner; the nearest of the
region's corners is 1 mm from it); furthest source 245 km. Its guard (`near_source`, a coil's gradient per unit current
above 30 times its median over the region): the medians are 0.067 to 0.087 T/(m A); 19 of 60 000 drawn points of the
region reach it, a pocket about 1.2 mm across around that source, where the source's own term takes the gradient to
10^8 times the median at 20 um and 86 times at the nearest grid corner; elsewhere its smooth gradient stays within 5.5
times the median at the 99.9th percentile.

**Its accuracy, by use.** Against ETH's `mag_manip`, MIME's evaluation agrees to a tolerance derived from the field's
cancellation at the compared points (`tests/oracles/test_calibrated_multipole.py`): a software oracle, which pins sign
and frame to mag_manip's convention and cannot ground them against hardware. Its fit places sources far out with large
coefficients, whose fields nearly cancel in its small region: the answer is the fit's, and computing it costs digits,
which is why float32 is refused. Its cancellation is not converged on the grid: finer grids find it deeper.
