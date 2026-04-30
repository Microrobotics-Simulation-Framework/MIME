---
bibliography: ../../bibliography.bib
---

# Permanent Magnet Node

**Module**: `mime.nodes.actuation.permanent_magnet`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-101`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)

## Summary

Computes the magnetic field $\mathbf{B}$ and its spatial gradient
$\nabla\mathbf{B}$ produced by a *finite* permanent magnet at an
arbitrary target point in world coordinates. Three field models are
available (`point_dipole`, `current_loop`, `coulombian_poles`) plus an
unconditional uniform Earth-field background. The gradient
$\nabla\mathbf{B}$ is computed by `jax.jacrev` of the same function used
for $\mathbf{B}$ — a single code path guarantees the analytical
gradient is consistent with the analytical field for every model.

## Governing Equations

Let the magnet sit at world position $\mathbf{p}_m$ with orientation
quaternion $\mathbf{q}_m$ and body-frame magnetisation axis
$\hat{\mathbf{a}}_b$. The world-frame moment is

$$
\mathbf{m}_{\text{world}} = R(\mathbf{q}_m)\,(|\mathbf{m}| \, \hat{\mathbf{a}}_b)
$$

with $|\mathbf{m}|$ the configured `dipole_moment_a_m2`.

For a target point $\mathbf{x}_T$, define
$\mathbf{r} = \mathbf{x}_T - \mathbf{p}_m$, $r = |\mathbf{r}|$,
$\hat{\mathbf{r}} = \mathbf{r}/r$.

**Point dipole** (`point_dipole`):

$$
\mathbf{B}_{\text{dipole}}(\mathbf{r}) =
\frac{\mu_0}{4\pi} \cdot
\frac{3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}}{r^3}.
$$

**Current loop**, on-axis ($\rho = 0$, $z$ measured along $\hat{\mathbf{m}}$):

$$
B_z^{\text{loop}}(z) =
\frac{\mu_0 I R^2}{2 (R^2 + z^2)^{3/2}},
\qquad I_{\text{eff}} = \frac{|\mathbf{m}|}{\pi R^2}.
$$

Off-axis: dipole field with a near-field correction
$(1 - (R/r)^2)$ clipped to $[0, 1]$.

**Coulombian poles** (uniformly magnetised cylinder), on-axis:

$$
B_z^{\text{coul}}(z) =
\frac{\mu_0 M}{2}
\left[
\frac{z + L/2}{\sqrt{R^2 + (z + L/2)^2}}
-
\frac{z - L/2}{\sqrt{R^2 + (z - L/2)^2}}
\right],
\qquad M = \frac{|\mathbf{m}|}{\pi R^2 L}.
$$

Off-axis: dipole field (closed form involves elliptic integrals; not yet
implemented).

**Total field with amplitude scale and Earth background:**

$$
\mathbf{B}_{\text{total}}(\mathbf{x}_T) =
\alpha\,\mathbf{B}_{\text{model}}(\mathbf{x}_T)
+ \mathbf{B}_{\text{earth}}.
$$

**Spatial gradient (single code path):**

$$
\bigl[\nabla \mathbf{B}\bigr]_{ij} =
\frac{\partial B_i}{\partial x_j}
=
\bigl[\mathrm{jacrev}(\mathbf{B}_{\text{total}}, \mathbf{x}_T)\bigr]_{ij}.
$$

## Discretization

Analytical for all three field models on the magnet axis. AGM-based
complete elliptic integrals $K(m)$, $E(m)$ are provided as a JAX-
traceable library (12 iterations $\Rightarrow \approx 7$–$9$ digit
agreement with `scipy.special.ellipk`/`ellipe`); they are not used in
the v1 off-axis path but are available for future upgrades. Off-axis
`current_loop` and `coulombian_poles` fall back to dipole + finite-size
correction.

The gradient $\nabla\mathbf{B}$ is reverse-mode automatic differentiation
of the same function returning $\mathbf{B}$ — machine precision, no
hand-coded gradient drift.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{m}_{\text{world}} = R(\mathbf{q}_m)\,\mathbf{m}_{\text{body}}$ | `mime.nodes.actuation.permanent_magnet.PermanentMagnetNode._m_world` | `quat_to_rotation_matrix(q) @ m_body` |
| $\mathbf{B}_{\text{dipole}} = (\mu_0/4\pi)[3(\mathbf{m}\cdot\hat{\mathbf{r}})\hat{\mathbf{r}} - \mathbf{m}]/r^3$ | `mime.nodes.actuation.permanent_magnet._b_point_dipole` | Closed form |
| $B_z^{\text{loop}} = \mu_0 I R^2 / (2(R^2+z^2)^{3/2})$ on axis; dipole $\times(1-(R/r)^2)$ off axis | `mime.nodes.actuation.permanent_magnet._b_current_loop` | `jnp.where(on_axis, ...)` |
| $B_z^{\text{coul}}$ closed form on axis; dipole off axis | `mime.nodes.actuation.permanent_magnet._b_coulombian_poles` | Two opposite-sign discs |
| $\mathbf{B}_{\text{total}} = \alpha\,\mathbf{B}_{\text{model}} + \mathbf{B}_{\text{earth}}$ | `mime.nodes.actuation.permanent_magnet._b_total_world` | Dispatch via `_FIELD_MODELS` |
| $\nabla\mathbf{B}$ | `mime.nodes.actuation.permanent_magnet.PermanentMagnetNode.update` | `jax.jacrev(_field_fn)(target)` — single code path for all three models |
| AGM $K(m)$, $E(m)$ library | `mime.nodes.actuation.permanent_magnet._ellipk_agm` / `_ellipe_agm` | JAX-traceable, 12 iterations |

## Assumptions and Simplifications

1. Rigid permanent moment — no demagnetisation, no temperature drift.
2. Magnet pose supplied externally — no internal dynamics for the magnet.
3. Earth field is uniform and static over the workspace.
4. `current_loop` and `coulombian_poles` use closed form on-axis; off-axis falls back to dipole $\times$ near-field correction. Full elliptic-integral off-axis form is *not* implemented in v1.
5. No eddy currents or shielding from biological tissue.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| $|r|/R_{\text{magnet}}$ | 5--100 | `point_dipole` far-field validity envelope |
| $|\mathbf{m}|$ | $10^{-6}$--$10$ A·m² | Sub-mm to cm permanent magnets |
| `amplitude_scale` | 0--1 | Linear scaling of magnet contribution |

## Known Limitations and Failure Modes

1. `point_dipole` only valid at $z \gtrsim 5\,R_{\text{magnet}}$ — $\approx 4\%$ error at $z = 3\,R_{\text{magnet}}$ (e.g. 3 mm from a 1 mm magnet) and $< 1\%$ at $z = 10\,R_{\text{magnet}}$. See `docs/deliverables/dejongh_benchmark_summary.md` lines 393--413 for the calibrating numbers.
2. Off-axis `current_loop` and `coulombian_poles` fall back to dipole + $(1 - (R/r)^2)$ correction — for high-fidelity off-axis fields use a magnetostatic FEM solver.
3. Coulombian model assumes a uniformly magnetised cylinder; non-uniform magnetisation is not represented.
4. No saturation, hysteresis, or B-H curve — the moment is constant and rigid.

## Stability Conditions

Unconditionally stable — analytical evaluation, no time integration.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| field_vector | (3,) | T | Last-computed B at the target |
| field_gradient | (3,3) | T/m | Last-computed $\partial B_i / \partial x_j$ at the target |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| dipole_moment_a_m2 | float | required | A·m² | Magnitude of the magnetic moment |
| magnetization_axis_in_body | tuple | (0,0,1) | -- | Moment axis in magnet body frame (will be normalised) |
| magnet_geometry | str | "cylinder" | -- | Free-form geometry tag (documentation only) |
| magnet_radius_m | float | required | m | Cylinder radius |
| magnet_length_m | float | required | m | Cylinder length |
| field_model | str | "point_dipole" | -- | One of `point_dipole`, `current_loop`, `coulombian_poles` |
| earth_field_world_t | tuple | (2e-5, 0, -4.5e-5) | T | Static Earth-field background (mid-latitude rough) |

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| magnet_pose_world | (7,) | identity at origin | replacive | Magnet pose [x,y,z, qw,qx,qy,qz] |
| target_position_world | (3,) | zeros | replacive | Point at which to evaluate $\mathbf{B}$, $\nabla\mathbf{B}$ |
| amplitude_scale | () | 1.0 | replacive | Commandable multiplier on magnet contribution |

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| field_vector | (3,) | T | $\mathbf{B}$ at target (for downstream PermanentMagnetResponse) |
| field_gradient | (3,3) | T/m | $\nabla\mathbf{B}$ at target |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | pH Range | Temp Range | Viscosity Range |
|-------------|------------|----------|----------|------------|----------------|
| CSF | stagnant or pulsatile | n/a (external) | n/a | n/a | n/a |
| Blood | poiseuille / pulsatile | n/a (external) | n/a | n/a | n/a |

The magnet is an **external apparatus** — it does not sit in the
fluid. The field it produces does, however, drive the magnetised UMR
inside CSF or blood; the operating regime is therefore inherited from
the downstream `PermanentMagnetResponseNode` and `RigidBodyNode`.

### Mode 2 Independent Verification

Three benchmarks accompany this node (full reports under
`docs/validation/benchmark_reports/`):

- **MIME-VER-110** (`mime_ver_110_dipole_field.md`) — far-field point-dipole formula vs analytical; on-axis `current_loop` and `coulombian_poles` against their own closed-form expressions.
- **MIME-VER-111** (`mime_ver_111_dipole_gradient.md`) — `jax.jacrev` of the dipole field against the analytical $\nabla\mathbf{B}$ formula; verifies the single-code-path gradient claim.
- **MIME-VER-112** (`mime_ver_112_earth_field_superposition.md`) — Earth-field superposition is exact; flipping the Earth field flips the residual by exactly $2\,\mathbf{B}_{\text{earth}}$.

In addition, the test suite contains JAX-traceability tests
(`jit`, `grad`, `vmap`) and a `GraphManager` integration test that runs
the node for one step inside a graph.

## References

- [@Jackson1998] Jackson, J.D. (1998). *Classical Electrodynamics*, Sec. 5.6 — magnetic dipole field.
- [@Abbott2009] Abbott, J.J. et al. (2009). *How Should Microrobots Swim?* — magnetic actuation of microrobots.
- [@deJongh2024] de Jongh, S. et al. (2024). Confined-swimming benchmark; calibrating numbers for the near-field validity envelope.
- [@Furlani2001] Furlani, E.P. (2001). *Permanent Magnet and Electromechanical Devices* — closed-form Coulombian-pole and current-loop expressions for cylindrical bar magnets, used by the `current_loop` and `coulombian_poles` field models.

## Verification Evidence

- MIME-VER-110: Dipole field vs analytical
- MIME-VER-111: Dipole gradient (`jax.jacrev`) vs analytical
- MIME-VER-112: Earth-field superposition exactness
- Test file: `tests/verification/test_permanent_magnet.py`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-04-30 | Initial implementation — three field models + AGM elliptic library + jacrev gradient |
