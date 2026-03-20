# Context of Use (COU) Template — MIME

This template helps downstream users define their Context of Use per ASME V&V 40, adapted for microrobotics simulation.

## 1. Question of Interest

*What does the model predict?*

Example: "Predict the trajectory and drug release profile of a helical microrobot navigating from the lateral ventricle to the third ventricle via the foramen of Monro."

## 2. Quantities of Interest

*Which specific output variables?*

Example: robot position (x,y,z) vs. time, drug concentration at target site vs. time, step-out frequency for the specific robot geometry.

## 3. Model Influence

*How much does the simulation drive the decision?*

| Level | Description |
|-------|-------------|
| Low | Simulation is one of several information sources |
| Medium | Simulation is the primary information source but clinical judgement dominates |
| High | Simulation output directly determines the clinical decision |

## 4. Decision Consequence

*What is the severity of harm if the simulation is wrong?*

| Level | Description |
|-------|-------------|
| Low | No injury; financial or time cost only |
| Medium | Non-serious injury is possible |
| High | Death or serious injury is possible |

## 5. Risk Assessment

*Model influence x Decision consequence = Required evidence level.*

## 6. Credibility Goals

*What verification, validation, and UQ evidence is needed?*

- Code verification: which MIME benchmarks (B0–B5) are relevant?
- Calculation verification: mesh convergence, solver settings for the specific COU
- Validation: experimental comparisons for the specific anatomy and robot design
- Uncertainty quantification: which parameters are uncertain, what propagation method?

## 7. Evidence Plan

*How will the required evidence be generated?*
