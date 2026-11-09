# CardioMag's calibration

**What it is.** The calibrated multipole model (MPEM, order one) of CardioMag, eight coils, the variant with no offset
sources: each coil's sources, each a fitted position, direction and coefficient, and the system's declared region of
interest. Packaged as `asset("cardiomag")`, read by the field provider `CalibratedMultipole`.

**The file.** `src/mime/data/emns_calibrations/cardiomag.yaml`, byte for byte its upstream's, which every read checks
(`mime.data.asset_bytes`), its sha256:

`31bd9b234586497275c8abff6ff292ece7ed029be8b6077b027264870facd51c`

**Its upstream.** [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public),
`mag_control/mpem/cal/C_Mag_Calibration_06-25-2015_no_offset.yaml` at `5ef021b0f4c0b91d1d8f3aedf353fe24bf793e84`.

**Its terms.** Apache-2.0, by the file's own header, retained; credit: Multi-Scale Robotics Lab, ETH Zurich. Its tier is
open: it ships in the wheel.

**Each value's basis.**
- `Source_Position`, `Source_Direction`, `B_Coeff`: fitted by the system's makers. A source is a basis function, not
  hardware; its position need not be a coil's.
- `Workspace_Dimensions`: declared, never fitted (the upstream fitter never writes it). Being inside it is no evidence
  of accuracy, and being outside it none of extrapolation; the fit's own domain is not published.

**Its conditioning** (`Calibration.cancellation` and `floor`, over the declared region): cancellation 0.25; float64
rounding 8.9e-16 and float32 4.8e-7 of the field; least distance from a source to the region 0.163 m.

**Its accuracy, by use.** Against ETH's `mag_manip`, MIME's evaluation agrees to a tolerance derived from the field's
cancellation at the compared points (`tests/oracles/test_calibrated_multipole.py`): a software oracle, which pins sign
and frame to mag_manip's convention and cannot ground them against hardware. The upstream variant with offset sources
(current-independent sources for remanent and ambient field) is not shipped: here zero current gives zero field. Its
file mixes CRLF and LF line ends, which the repository keeps as they are (`.gitattributes`), so its sha256 is its
upstream's.
