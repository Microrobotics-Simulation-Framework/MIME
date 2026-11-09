# Navion_2's calibration

**What it is.** The calibrated multipole model (MPEM, order one) of Navion_2, a planar array of three coils: each coil's
sources, each a fitted position, direction and coefficient, and the system's declared region of interest. Packaged as
`asset("navion_2")`, read by the field provider `CalibratedMultipole`.

**The file.** `src/mime/data/emns_calibrations/navion_2.yaml`, byte for byte its upstream's, which every read checks
(`mime.data.asset_bytes`), its sha256:

`ea6eeb803b48d73551648e4df08116373a03fed4e871eb2b5c13706f57796b56`

**Its upstream.** [ethz-msrl/Workspace_eMNS](https://github.com/ethz-msrl/Workspace_eMNS),
`models/Navion_2_Calibration_24-02-2020.yaml` at `524f85199d092259b80608718bf9ac0e7414ee49`.

**Its terms.** Apache-2.0, by the repository's `LICENSE` (the file holds no header); credit: Multi-Scale Robotics Lab,
ETH Zurich. Its tier is open: it ships in the wheel.

**Each value's basis.**
- `Source_Position`, `Source_Direction`, `B_Coeff`: fitted by the system's makers. A source is a basis function, not
  hardware; its position need not be a coil's.
- `Workspace_Dimensions`: declared, never fitted (the upstream fitter never writes it). Being inside it is no evidence
  of accuracy, and being outside it none of extrapolation; the fit's own domain is not published.

**Its conditioning** (`Calibration.cancellation` and `floor`, over the declared region): cancellation 8.0e-3; float64
rounding 2.8e-14 and float32 1.5e-5 of the field; least distance from a source to the region 0.116 m.

**Its accuracy, by use.** Against ETH's `mag_manip`, MIME's evaluation agrees to a tolerance derived from the field's
cancellation at the compared points (`tests/oracles/test_calibrated_multipole.py`): a software oracle, which pins sign
and frame to mag_manip's convention and cannot ground them against hardware. Its sources' directions are not of unit
length in the file; the reader normalises them, which the oracle discriminates. The same file is in Tesla_core_public,
numerically identical; that repository's top-level licence is a BSD licence with an advertising clause, so the copy
shipped is Workspace_eMNS's. Against ETH's measured Navion campaign the previous line found a global sign between model
and measurement (MIME-ANO-153) and an error that bounds the model's from above, at least half of it the measuring
apparatus's: an acceptance test of sign and frame against hardware, not an accuracy claim, and not run here.
