---
bibliography: ../bibliography.bib
---

# B0 Experimental Dataset: Rodenborn et al. (2013)

**Status**: Confirmed as primary B0 dataset.

## Citation

[@Rodenborn2013] Rodenborn, B., Chen, C.-H., Swinney, H.L., Liu, B., & Zhang, H.P. (2013). Propulsion of microorganisms by a helical flagellum. *PNAS*, 110(5), E338–E347.

## Why This Dataset

Rodenborn et al. provides:
1. Experimental measurements of thrust, torque, and drag for helical swimmers
2. Published simulation codes (regularised Stokeslet method) for comparison
3. Parameter documentation sufficient for simulation reproduction
4. Definitive evidence that RFT is insufficient (helix radius ~ wavelength)
5. Data across the parameter ranges relevant to microrobotics

## Robot Parameters

| Parameter | Value | Units | Source |
|-----------|-------|-------|--------|
| Helix wavelength (lambda) | 2.08–6.25 | mm | Table 1 |
| Helix amplitude (A) | 0.5–2.5 | mm | Table 1 |
| Filament radius (r_f) | 0.4 | mm | Methods |
| Number of wavelengths | 1–4 | - | Table 1 |
| Rotation rate | 0.01–1.0 | rev/s | Methods |

Note: These are macro-scale experiments (mm-scale) in high-viscosity silicone oil, designed to match the low-Re regime of microrobots. The Reynolds number is Re << 1, matching our target regime.

## Fluid Properties

| Parameter | Value | Units |
|-----------|-------|-------|
| Fluid | Silicone oil | - |
| Viscosity | 1.0 | Pa.s |
| Density | 971 | kg/m^3 |

## Validation Approach for B0

1. Configure RigidBodyNode with helical resistance tensor matching one of Rodenborn's geometries
2. Apply known rotation rate via ExternalMagneticFieldNode
3. Compare simulated thrust and drag against Table 1 measurements
4. Pass criterion: position RMSE < 15% of channel diameter, velocity RMSE < 20% of mean velocity

## Simulation Codes

The Rodenborn et al. regularised Stokeslet simulation codes are publicly available on MATLAB File Exchange ("Helical Swimming Simulator"). These serve as the high-fidelity reference for B1 (step-out frequency within ±5%).

## Scaling to Microrobot Parameters

The experimental data is in the low-Re regime and thus dynamically similar to a microrobot in CSF. To map to MIME's default parameters:
- Scale lengths by the ratio of characteristic sizes
- Viscosity and density use CSF values (mu = 8.5e-4 Pa.s, rho = 1002 kg/m^3)
- The Re matching ensures the same Oberbeck-Stechert coefficients apply
