# CardioMag's calibration

**What it is.** ETH MSRL's published calibration of CardioMag (`mag_control/mpem/cal/C_Mag_Calibration_06-25-2015_no_offset.yaml`), evaluated by MIME: a calibrated multipole model (MPEM, order one) of eight coils, the variant whose `Offset` block is not enabled, each coil's sources, each a fitted position, direction and coefficient, and the system's declared region of
interest. Packaged as `asset("cardiomag")`, read by the field provider `CalibratedMultipole`.

**The file.** `src/mime/data/emns_calibrations/cardiomag.yaml`, byte for byte its upstream's, which every read checks
(`mime.data.asset_bytes`), its sha256:

`31bd9b234586497275c8abff6ff292ece7ed029be8b6077b027264870facd51c`

**Its upstream.** [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public),
`mag_control/mpem/cal/C_Mag_Calibration_06-25-2015_no_offset.yaml` at `5ef021b0f4c0b91d1d8f3aedf353fe24bf793e84`.

**Its terms.** Apache-2.0, by the file's own header, retained; credit: Multi-Scale Robotics Lab, ETH Zurich. Its tier is
open: it ships in the wheel.

**Each value's basis.**
- `Source_Position`, `Source_Direction`, `B_Coeff`: fitted by ETH MSRL. A source is a basis function, not
  hardware; its position need not be a coil's.
- `Workspace_Dimensions`: declared, never fitted (the upstream fitter never writes it). Being inside it is no evidence
  of accuracy, and being outside it none of extrapolation; the fit's own domain is not published.

**Its conditioning** (`Calibration.cancellation` and `floor`, over the declared region): cancellation 0.25; float64
rounding 8.9e-16 and float32 4.8e-7 of the field; least distance from a source to the region 0.163 m. Its guard
(`near_source`, a coil's gradient per unit current above 30 times its median over the region): the medians are 0.016 to
0.027 T/(m A), and none of 60 000 drawn points of the region reaches it (its smooth gradient reaches 2.4 times the
median).

**Its current limits.** None ships (`allocate()` takes the user's). A limit is MIME's declaration with its source, never a maker's specification. The figure published for CardioMag: "the maximum
current in each electromagnet is 35 A, and the maximum power in the whole system is 15 kW" (Yu et al. 2019, Sec. III-A,
the eight-electromagnet system with a 10 cm cubic workspace). Nothing states that the unit measured is the one this
calibration was fitted on, and the 15 kW cap is not modelled here (the coils' resistance is unknown), so 35 A in every coil
at once is an upper bound. The previous line's 20 A had no source and is dropped.

**Its accuracy, by use.** Against ETH's `mag_manip`, MIME's evaluation agrees to a tolerance derived from the field's
cancellation at the compared points (`tests/oracles/test_calibrated_multipole.py`): a software oracle, which pins sign
and frame to mag_manip's convention and cannot ground them against hardware. **The file carries an `Offset` block
of eight current-independent sources, worth about 1.5 mT at the median over the region (1.3 to 1.6 mT, 200 draws), which
neither mag_manip nor MIME evaluates**: upstream loads that block as the offset only when `Coil_List` names it, and this
file's list does not (so the file is named `no_offset`; its content is not). Against a 5 to 50 mT field that is 3 to
30 %. A real CardioMag has a field at zero current of that order that this evaluation drops; `background` is not it, the
sources being position dependent. A `Coil_List` that names the block is refused by the reader. Its
file mixes CRLF and LF line ends, which the repository keeps as they are (`.gitattributes`), so its sha256 is its
upstream's.
