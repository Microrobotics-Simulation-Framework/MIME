# `mime.data/emns_calibrations`

**ETH MSRL's published calibrations of three electromagnetic navigation systems**, each a calibrated multipole model (MPEM, order one), evaluated by MIME (`mime.fields.CalibratedMultipole`). They are a third party's files, shipped byte for byte;
each file's provenance, sha256 and conditioning are on its card in `docs/assets/` (`navion_2.md`, `octomag.md`,
`cardiomag.md`).

| File | Calibration of | Upstream |
|---|---|---|
| `navion_2.yaml` | Navion_2 | [ethz-msrl/Workspace_eMNS](https://github.com/ethz-msrl/Workspace_eMNS), its `models/` folder, `models/Navion_2_Calibration_24-02-2020.yaml` |
| `octomag.yaml` | OctoMag | [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public), its `mag_control/mpem/cal/` folder, `OctoMag_Calibration.yaml` (undated in its name) |
| `cardiomag.yaml` | CardioMag, the variant whose `Offset` block is not enabled | the same repository and folder, `mag_control/mpem/cal/C_Mag_Calibration_06-25-2015_no_offset.yaml` |

Copyright Multi-Scale Robotics Lab, ETH Zurich (Prof. Bradley J. Nelson), under the Apache License, MIME's own (the repository's `LICENSE`); the attribution is in `NOTICE`. MIME is not affiliated with or endorsed by the makers of these systems.

## The licences, as read

Each licence was read from its source, at the commit its card names, before the file entered the repository.
Workspace_eMNS's `LICENSE` is the Apache License, and its Navion_2 file holds no header of its own. Tesla_core_public's
top-level `LICENSE` is a BSD licence with an advertising clause; its two files carry their own Apache License headers,
which govern them. The Navion_2 file is in Tesla_core_public too, numerically identical, and the copy shipped is
Workspace_eMNS's. No source code of either repository is copied: MIME's evaluator is its own, from the published
formulation (Petruska et al.).

## What is changed

Nothing. The previous line appended a newline to the Navion_2 file; this one ships the upstream bytes, so each file's
sha256 is its upstream's. CardioMag's file mixes CRLF and LF line ends, and `.gitattributes` keeps every file here as
it is.

## Each value's basis

`Source_Position`, `Source_Direction` and `B_Coeff` are fitted: a source is a basis function, not hardware.
`Source_Direction` is not a unit vector in the files, and is normalised when read. `Workspace_Dimensions` is declared,
never fitted (the upstream fitter never writes it). Every shipped calibration is of order one and holds no interior
terms (`A_Coeff: ~`). **CardioMag's file carries an `Offset` block** (eight current-independent sources, about 1.5 mT at
the median over its region) that upstream's `mag_manip` does not evaluate, because it loads that block only when
`Coil_List` names it, and this file's does not; MIME does not evaluate it either (a `Coil_List` that names it is
refused). Navion_2's and OctoMag's files carry no `Offset` block.

## Current limits

None of the files states a current limit, and MIME ships none (`allocate()` takes the user's); a limit is MIME's declaration with its source, never a maker's specification. The figures published for Navion_2, none a coil rating: 25 A a coil as one lab's operating restriction (Zughaibi et al. 2025, Sec. III-B), the figure MIME declares where it needs one; 45 A as what that lab says the hardware supports (the same section, and
Zughaibi et al. 2026, Sec. II-C); drivers rated 50 A continuous and 100 A peak (Gervasoni et al. 2024, Experimental
Section and Fig. 3). No source says that the unit measured is the one the calibration was fitted on. For CardioMag: 35 A in each electromagnet and 15 kW in the whole system (Yu et al. 2019, Sec. III-A; the 15 kW cap is not modelled), again with no statement that its unit is the
one the calibration was fitted on; the previous line's unsourced 20 A is dropped.

## The rule

These files are never edited here. A newer calibration is a new file with its own pins, taken from its source.
