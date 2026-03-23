---
bibliography: ../../bibliography.bib
---

# de Boer et al. (2025) — Extracted Parameters for Tier 1 Replication

**Paper**: "Wireless mechanical and hybrid thrombus fragmentation of ex vivo
endovascular thrombosis model in the iliac artery"
Marcus C. J. de Boer et al., Appl. Phys. Rev. 12, 011416 (2025)
DOI: 10.1063/5.0233677

**Target**: Figure 12 — swimming speed vs. time at step-out frequency for 6 UMR configurations.

---

## UMR Geometry

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| Cylinder body diameter | 1.74 | mm | §VI.F p.15 |
| Conical tip start diameter | 1.74 | mm | §VI.F p.15 |
| Conical tip end diameter | 0.51 | mm | §VI.F p.15 |
| Conical tip length | 1.9 | mm | §VI.F p.15 |
| Total UMR length | 6.0 | mm | §VI.F p.15 |
| External diameter (body + fins) | 2.84 | mm | §VI.F p.15, Table S2 |
| Number of fin sets | 2 | — | §VI.F p.15 ("Two sets of three propeller fins") |
| Fins per set | 3 | — | §VI.F p.15 |
| Fin length along UMR | 2.03 | mm | §VI.F p.15 |
| Fin width | 0.55 | mm | §VI.F p.15 |
| Fin thickness | 0.15 | mm | §VI.F p.15 |
| Fin type | Discontinuous helix | — | §II.B p.4, §VI.F p.15 |

## Magnetic Properties

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| Magnet material | NdBFe Grade N45 | — | §VI.H p.17 |
| Magnet size | 1 × 1 × 1 | mm³ | §VI.H p.17 |
| Magnets per UMR | 1, 2, or 3 | — | Fig. 12 |
| Magnetic moment per magnet | 1.07 × 10⁻³ | A·m² | §VI.H p.17 |
| Moment orientation | Perpendicular to long axis | — | §VI.E p.15 |
| RPM permanent magnet | NdBFe Grade N45, 35mm dia, 20mm height | — | §VI.I p.17 |
| RPM magnetic moment | 20.67 | A·m² | §VI.I p.17 |
| RPM-UMR gap | 150 | mm | §VI.H p.17 |
| Field strength at UMR | ~1.4 (measured), 3 (simulation) | mT | §VI.H p.17, Fig. 12 |

## Swimming Characterisation

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| Swimming tube ID | 9.5 | mm | §VI.I p.17 |
| RPM placement | 100mm (1-mag), 150mm (2,3-mag) above tube | mm | §VI.I p.17 |
| Max RPM frequency | 42 | Hz | §VI.I p.17 |
| Fluid | Water | — | Fig. 4a-c |
| Fluid viscosity (water, 37°C) | ~0.69 | mPa·s | Standard |

## Figure 12 Data Points (from paper)

Simulation at 3 mT field strength, Newton's second law with Euler's method.

| Diameter (mm) | Magnets | Step-out freq (Hz) | Peak speed (m/s) |
|---------------|---------|-------------------|-----------------|
| 2.8 | 1 | 128 | ~0.4 |
| 2.8 | 2 | 181 | ~0.7 |
| 2.8 | 3 | 222 | ~0.85 |
| 2.1 | 1 | 144 | ~0.5 |
| 2.1 | 2 | 204 | ~0.8 |
| 2.1 | 3 | 250 | ~1.1 |

Note: 2.1mm UMR is described as "75% of baseline size" (p.16).

## Swimming Model (Eq. 1)

$$
U \propto R_{\text{cyl}} \omega \varepsilon_{\text{cyl}}^2 f(De, \beta)
$$

where:
- R_cyl = average UMR radius
- ε_cyl = helical amplitude normalised by R_cyl
- De = τω = Deborah number (τ = fluid relaxation timescale)
- β = η_s / η = serum viscosity / total viscosity
- f(De, β) = unspecified function of viscoelastic parameters

**Critical note**: The paper does NOT tabulate drag coefficients. Eq. 1 is a scaling
relation, not a closed-form drag model. The actual simulation uses "Newton's
second law with Euler's method" (§VI.E p.16), which implies a full force-balance
ODE with the specific drag model embedded in the code but not published as
explicit coefficients. See ADD-1 in UMR_REPLICATION_PLAN.md.

## Drag Torque Comparison (Fig. 4d)

OpenFOAM CFD simulation comparing continuous vs. discontinuous helix:
- Discontinuous helix has lower drag torque than continuous
- The difference increases with actuation frequency
- At 200 Hz: continuous ~4×10⁻⁴ N·m, discontinuous ~3×10⁻⁴ N·m
- "Our frequency range" marked on figure: ~100-300 Hz

## Rheological Data

| Property | Blood | Blood Clot | Source |
|----------|-------|------------|--------|
| Deborah number (De) | 2 | 3 | §VI.C p.15 |
| Relaxation timescale (τ) | 20 s | 30 s | §VI.C p.15 |
| Viscosity at 0.1 s⁻¹ | 34 Pa·s | — | Fig. 4f |
| Viscosity at 100 s⁻¹ | ~60 mPa·s | — | Fig. 4f |
| G' at 1 Hz | 68-206 kPa (blood), 205-532 kPa (clot) | — | §VI.C p.15 |
| Predominantly elastic | Yes (G' > G'') | Yes | §VI.C p.15 |

## Vessel Dimensions

| Vessel | Inner Diameter | Source |
|--------|---------------|--------|
| Iliac artery | 4.7–9.4 mm | §VI.F p.16 |
| Swimming tube (characterisation) | 9.5 mm | §VI.I p.17 |
| Vessel diameter for flow calc | 8 mm | §VI.E p.16 |

## Wear Model (Reye-Archard-Khrushchov, Eqs. 3-6)

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| K_g (wear coefficient) | 7.1 | μN·s | §VI.G p.17 |
| W (normal load) | ~U (swimming speed) | — | §VI.G p.16 |
| L (sliding distance) | ~f (frequency) | — | §VI.G p.16 |
| H (hardness) | ~G' (storage modulus) | — | §VI.G p.16 |
| K_l (lysis rate) | 0.51 | mm³/min | §VI.G p.17 |
| M (interaction parameter) | 0.7 | — | §VI.G p.17 |
