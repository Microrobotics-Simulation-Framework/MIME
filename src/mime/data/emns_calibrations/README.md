# `mime.data/emns_calibrations`

**Three electromagnetic navigation systems' calibrations**, each a calibrated multipole model (MPEM, order one) as its
makers publish it, read by `mime.fields.CalibratedMultipole`. They are third parties' files, shipped byte for byte;
each file's provenance, sha256 and conditioning are on its card in `docs/assets/` (`navion_2.md`, `octomag.md`,
`cardiomag.md`).

| File | System | Upstream |
|---|---|---|
| `navion_2.yaml` | Navion_2 | [ethz-msrl/Workspace_eMNS](https://github.com/ethz-msrl/Workspace_eMNS), its `models/` folder |
| `octomag.yaml` | OctoMag | [ethz-msrl/Tesla_core_public](https://github.com/ethz-msrl/Tesla_core_public), its `mag_control/mpem/cal/` folder |
| `cardiomag.yaml` | CardioMag, the variant whose `Offset` block is not enabled | the same repository and folder |

Copyright Multi-Scale Robotics Lab, ETH Zurich (Prof. Bradley J. Nelson), under the Apache License, MIME's own (the
repository's `LICENSE`); the attribution is in `NOTICE`.

## The licences, as read

Each licence was read from its source, at the commit its card names, before the file entered the repository.
Workspace_eMNS's `LICENSE` is the Apache License, and its Navion file holds no header of its own. Tesla_core_public's
top-level `LICENSE` is a BSD licence with an advertising clause; its two files carry their own Apache License headers,
which govern them. The Navion file is in Tesla_core_public too, numerically identical, and the copy shipped is
Workspace_eMNS's. No source code of either repository is copied: MIME's evaluator is its own, from the published
formulation (Petruska et al.).

## What is changed

Nothing. The previous line appended a newline to the Navion file; this one ships the upstream bytes, so each file's
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

None of the files states a current limit, and MIME ships none (`allocate()` takes the user's). The figures published for
the Navion system, none a coil rating: 25 A a coil as one lab's operating restriction (Zughaibi et al. 2025, Sec. III-B),
the figure MIME uses where it needs one; 45 A as what that lab says the hardware supports (the same section, and
Zughaibi et al. 2026, Sec. II-C); drivers rated 50 A continuous and 100 A peak (Gervasoni et al. 2024, Experimental
Section and Fig. 3). No source says that the unit measured is the one the calibration was fitted on. For the CardioMag: 35 A in each
electromagnet and 15 kW in the whole system (Yu et al. 2019, Sec. III-A), again with no statement that its unit is the
one the calibration was fitted on; the previous line's unsourced 20 A is dropped.

## The rule

These files are never edited here. A newer calibration is a new file with its own pins, taken from its source.
