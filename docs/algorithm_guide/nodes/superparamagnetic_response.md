---
bibliography: ../../bibliography.bib
---

# Superparamagnetic Response Node

**Module**: `mime.nodes.robot.superparamagnetic_response`
**Stability**: experimental
**Algorithm ID**: `MIME-NODE-009`
**Version**: 1.0.0
**Verification Mode**: Mode 2 (Independent)
**Upstream Node**: — (not a wrapper)
**MADDENING Version Pin**: —

## Summary

Computes the gradient-pulling force on a **superparamagnetic** body — a bead or
capsule loaded with single-domain magnetic nanoparticles — whose moment is
**field-aligned** and whose magnitude comes from a **measured magnetization
curve** `m(|B|)`. It is the body-side counterpart of `CoilArrayFieldNode`
(MIME-NODE-103): that node emits `(field_vector, field_gradient)`; this one turns
them into `(magnetic_force, magnetic_torque)` for a `RigidBodyNode`.

The torque vanishes identically (the moment is parallel to **B** by
construction), so the gradient force is the entire physics — which is exactly why
gradient pulling, not field steering, is this device class's actuation mode.

## Governing Equations

The moment is field-aligned, with a magnitude set by the curve and scaled exactly
by volume:

$$
\mathbf{m}(\mathbf{B}) = m\!\left(|\mathbf{B}|\right) \cdot
  \frac{V}{V_\text{ref}} \cdot \hat{\mathbf{B}},
\qquad \hat{\mathbf{B}} = \frac{\mathbf{B}}{|\mathbf{B}|}
$$

$$
\boldsymbol{\tau} = \mathbf{m} \times \mathbf{B} \equiv \mathbf{0}
\qquad\text{(structural: } \mathbf{m} \parallel \mathbf{B}\text{)}
$$

$$
\mathbf{F} = (\mathbf{m} \cdot \nabla)\mathbf{B} = \nabla\!\mathbf{B} \cdot \mathbf{m},
\qquad F_i = m_j \frac{\partial B_i}{\partial x_j}
$$

**The force is exact for an induced moment; the energy is not.** For a *constant*
moment $\mathbf{F} = \nabla(\mathbf{m}\cdot\mathbf{B})$. For an *induced* moment
the stored energy carries a factor of one half, $U = -\tfrac{1}{2}\mathbf{m}\cdot\mathbf{B}$,
and it is a standard error to propagate that half into the force. It does not
belong there: the force on each elementary dipole is $(\mathbf{m}\cdot\nabla)\mathbf{B}$
with the *instantaneous* moment, however that moment arose.

For a curl-free field ($\nabla\mathbf{B}$ symmetric) this is identically

$$
\mathbf{F} = m(|\mathbf{B}|)\,\nabla|\mathbf{B}|,
\qquad \nabla|\mathbf{B}| = (\nabla\mathbf{B})^{\!\top}\hat{\mathbf{B}}
$$

so the force is **conservative** with potential
$\Phi(|\mathbf{B}|) = \int_0^{|\mathbf{B}|} m(b)\,\mathrm{d}b$, reducing to
$\chi_\text{eff} V \nabla|\mathbf{B}|^2 / 2\mu_0$ in the linear regime — half and
all. The equivalence is pinned by MIME-VER-146.

### Primary path — measured curve (Langevin)

$$
m(|\mathbf{B}|) = m_\text{sat}\,L\!\left(\frac{|\mathbf{B}|}{b_0}\right),
\qquad L(x) = \coth x - \frac{1}{x}
$$

with initial susceptibility $\mathrm{d}m/\mathrm{d}B|_0 = m_\text{sat}/(3b_0)$ and
high-field asymptote $m \to m_\text{sat}(1 - b_0/|\mathbf{B}|)$.

### Secondary path — first-principles (gated)

$$
\chi_\text{eff} = \frac{\chi}{1 + N\chi},
\qquad
m(|\mathbf{B}|) = \min\!\left(\chi_\text{eff}\,\frac{|\mathbf{B}|}{\mu_0}\,V,\;
m_\text{sat}\right)
$$

$\chi$ is the **intrinsic** susceptibility; $N$ is the demagnetization factor
($1/3$ for a sphere, [@Osborn1945]). Bounded above by $\chi_\text{eff} \to 1/N = 3$.

## Discretization

Analytical — no spatial or temporal discretization. Two numerical constructs
carry the accuracy claims:

**Langevin evaluation.** $\coth x - 1/x$ evaluates to `inf - inf` at $x=0$ and
cancels catastrophically nearby (both terms $\sim 1/x$, difference $\sim x/3$, so
the relative error goes as $\sim 3\varepsilon/x^2$ — already $4\times10^{-8}$ in
float64 at $x=10^{-4}$). Below $|x| = 0.1$ the implementation therefore uses the
4-term Maclaurin series

$$
L(x) = \frac{x}{3} - \frac{x^3}{45} + \frac{2x^5}{945} - \frac{x^7}{4725} + O(x^9)
$$

whose truncation is $\sim x^8/15592$, and the direct form above. At the $x=0.1$
crossover both branches are accurate to $\sim 10^{-13}$ relative in float64.
Branch selection uses the **double-where** idiom: reverse-mode AD evaluates both
branches, so a single `jnp.where` would still propagate NaN from the untaken one.
Verified against an independent Lambert continued-fraction oracle (MIME-VER-145).

**Safe norm.** $|\mathbf{B}|$ uses the same double-where guard so that
$\mathbf{B} = \mathbf{0}$ yields $0$ with a $0$ gradient rather than NaN. The
gradient of $|\mathbf{B}|$ at the origin is genuinely undefined (the norm has a
cone point); returning $0$ is a *choice* that keeps the model traceable through a
field null, not a claim about the physics.

**Curve fitting** (`LangevinCurve.from_saturation_and_point`) runs on the host in
**float64 numpy**, deliberately not through `jnp`: a fit runs once at
construction, and its result must not depend on whether the caller has
`JAX_ENABLE_X64` set. Bisection, 200 iterations, on the strictly monotone $L$.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{m} = m(\lvert\mathbf{B}\rvert)\,(V/V_\text{ref})\,\hat{\mathbf{B}}$ | `mime.nodes.robot.superparamagnetic_response.SuperparamagneticResponseNode.update` | Field-aligned; volume scale from params |
| $\mathbf{F} = \nabla\mathbf{B}\cdot\mathbf{m}$ | `mime.nodes.robot.superparamagnetic_response.SuperparamagneticResponseNode.update` | `grad_B @ m_vec`; jacrev index convention |
| $\boldsymbol{\tau} = \mathbf{m}\times\mathbf{B} \equiv 0$ | `mime.nodes.robot.superparamagnetic_response.SuperparamagneticResponseNode.update` | Emitted as exact `jnp.zeros(3)`; see below |
| $L(x) = \coth x - 1/x$ | `langevin` | Series below $\lvert x\rvert=0.1$, double-where |
| $L(x)$ host float64 (fitting only) | `_langevin_np` | Dtype-independent fits |
| $m = m_\text{sat}L(\lvert\mathbf{B}\rvert/b_0)$ | `mime.nodes.robot.superparamagnetic_response.LangevinCurve.moment_am2` | Primary path |
| $b_0 = m_\text{sat}/(3\,\text{slope})$ | `mime.nodes.robot.superparamagnetic_response.LangevinCurve.from_saturation_and_slope` | Closed form |
| $m_\text{sat}L(b/b_0) = m_\text{meas}$ solved for $b_0$ | `mime.nodes.robot.superparamagnetic_response.LangevinCurve.from_saturation_and_point` | Bisection, float64 |
| $m = \min(\text{slope}\cdot\lvert\mathbf{B}\rvert, m_\text{sat})$ | `mime.nodes.robot.superparamagnetic_response.LinearSaturationCurve.moment_am2` | Crude; kink at the knee |
| piecewise-linear table | `mime.nodes.robot.superparamagnetic_response.InterpolatedCurve.moment_am2` | `jnp.interp`; real VSM data. Table **must** start at $(0,0)$ — enforced |
| $\chi_\text{eff} = \chi/(1+N\chi)$ | `mime.nodes.robot.superparamagnetic_response.DemagLimitedLinearCurve.effective_susceptibility` | **Gated** secondary path |
| $\delta\lvert B\rvert/\lvert B\rvert = \lvert(\nabla\mathbf{B})^{\!\top}\hat{\mathbf{B}}\rvert d/\lvert B\rvert$ | `small_body_field_variation_ratio` | Advisory diagnostic |
| $\lvert v\rvert$ with 0 gradient at $v=0$ | `_safe_norm` | Double-where |
| $V = \pi d^3/6$ | `sphere_volume_m3` | |

## Assumptions and Simplifications

1. **Field-aligned moment**: $\mathbf{m} \parallel \mathbf{B}$ — no easy axis, no
   remanence, no hysteresis. True for a superparamagnet above its blocking
   temperature; false for a ferromagnetic or shape-anisotropic body.
2. **Small body**: the field at the body *centre* represents the field across the
   whole body. This is what lets a VSM curve (measured in a **uniform** field)
   transfer to a gradient-pulling context at all. See below.
3. **Measured path takes the APPLIED field** and the curve already contains the
   body's self-demagnetization. No demag correction is applied, and none must be.
4. **Volume scaling** $m \propto V$ is exact for spheres ($N = 1/3$ is
   size-independent) at fixed material and shape.
5. **Quasi-static**: the moment follows $|\mathbf{B}|$ instantaneously — no Néel
   or Brownian relaxation lag.
6. **Dilute limit**: the body does not perturb the applied field; no bead–bead
   dipolar interaction.
7. **Isothermal**: $b_0 = k_BT/\mu_\text{particle}$ is fixed at construction.

### The demagnetization double-count — the likeliest misuse

A VSM applies a known field to the *real object* and measures the moment that
results. The object's self-demagnetization is therefore **already baked into the
measured curve**: $m_\text{measured}(|\mathbf{B}_\text{applied}|)$ *is*
$\chi_\text{eff}$, not $\chi$. The field `CoilArrayFieldNode` emits at the bead is
likewise the *applied* field. Correcting again **double-counts** and
under-predicts the moment, hence the force.

Both paths consume the applied field, for different reasons:

| path | constitutive law | where demag lives |
|------|------------------|-------------------|
| measured (primary) | $m = m_\text{curve}(\lvert\mathbf{B}_\text{app}\rvert)$ | already inside the curve |
| first-principles (gated) | $m = \frac{\chi}{1+N\chi}\frac{\lvert\mathbf{B}_\text{app}\rvert}{\mu_0}V$ | the $1/(1+N\chi)$ factor |

The error is **stacking** them. Pick one row and stop.

### The small-body assumption, quantified

$$
\frac{\delta|\mathbf{B}|}{|\mathbf{B}|} \sim
\frac{\left|\nabla|\mathbf{B}|\right| \, d}{|\mathbf{B}|}
$$

For the Landers capsule ($d = 1.69$ mm, $|\mathbf{B}| = 30$ mT) the number depends
on which gradient the eMNS is running, and the spread is a factor of 4:

| $\lvert\nabla\mathbf{B}\rvert$ | $\delta\lvert\mathbf{B}\rvert$ | ratio | regime |
|---|---|---|---|
| 0.25 T/m (Navion single) | 0.42 mT | **1.4 %** | comfortable |
| 0.5 T/m | 0.845 mT | **2.8 %** | good |
| 1.0 T/m (Navion double) | 1.69 mT | **5.6 %** | **at the edge** |

The commonly-quoted "good to ~3 %" is the **0.5 T/m** figure. At the double-gradient
setting — precisely the mode Landers uses to pull *against flow*, i.e. the anchor's
most demanding configuration — the assumption is only good to ~6 %, above the ~0.05
rule of thumb, and a 2 mm body there reaches 6.7 %. **Quote the number for the
gradient you are running, not the headline one.**

It is **linear in $d$ and in $|\nabla\mathbf{B}|$, and inverse in $|\mathbf{B}|$**,
so it degrades exactly where gradient pulling gets interesting: larger bodies,
tighter gradients, weak-field regions. The last is sharpest — the ratio *diverges*
as $|\mathbf{B}| \to 0$ regardless of body size, so **a field null breaks this
model for any body, however small**. It is the gradient of the *magnitude* that
matters: a gradient that merely rotates $\mathbf{B}$ across the body does not
break the assumption.

`small_body_field_variation_ratio()` computes it, and the node exposes
`small_body_ratio()` against its own diameter. **It is not checked at runtime** —
the check is not JAX-traceable. Below ~0.05 the curve transfers cleanly; by ~0.2
the centre field is not representative and the moment is a fiction.

It returns **`inf` at a field null**, not 0. That is the honest value rather than a
guard: the ratio genuinely diverges as $|\mathbf{B}| \to 0$, so returning 0 — the
most reassuring value a caller could read — would report "transfers cleanly" at
exactly the voxel that must be flagged. An eMNS null can be *exactly* zero by
symmetry in a computed field, and sweeping this diagnostic across a workspace to
locate the breakdown is its intended use. `update()` never calls it, so the `inf`
cannot reach the force.

## Validated Physical Regimes

**These four are not independently attainable.** The small-body ratio is *derived*
from the other three ($\sim |\nabla\mathbf{B}| d / |\mathbf{B}|$), so it couples
them: its ceiling is what the $(|\nabla\mathbf{B}|, d)$ corners actually reach at
the 30 mT operating point, not a free dimension.

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| $\lvert\mathbf{B}\rvert$ | 0 – 0.1 T | eMNS band; Landers/OctoMag ~30 mT. **Verified at the single measured point 30 mT** (MIME-VER-144) plus the analytic high-field asymptote — the rest of the band is the declared domain of use, not validated data points. There is only one VSM datum in the source |
| $\lvert\nabla\mathbf{B}\rvert$ | 0 – 1 T/m | Navion ~0.25 T/m single, ~1 T/m double [@Landers2025]. The double-gradient mode is what forces the ratio ceiling to 0.07 |
| body diameter | 0 – 2 mm | Landers capsule 1.4–1.69 mm. Bound is the small-body assumption, not the algebra |
| small-body ratio | 0 – 0.07 | **Derived, not independent.** 0.07 is what the other three corners reach: 2 mm at 1 T/m and 30 mT gives 0.067; the 1.69 mm anchor at its own double-gradient setting gives 0.056. It is 0.014 at 0.25 T/m and 0.028 at 0.5 T/m — the anchor is comfortable single-gradient and **at the edge** in double |

## Known Limitations and Failure Modes

1. **Torque is identically zero — this node cannot reorient a body.** If it is a
   body's only magnetic effect, that body's orientation is *unactuated* and will
   drift under drag and contact alone. Correct for an ideal superparamagnetic
   sphere; a real capsule has some shape anisotropy and a small real aligning
   torque, modelled here as exactly zero.

   The zero is stronger than the point-dipole argument suggests. The
   *distributed* force over a body in a gradient might be expected to exert a net
   torque about the centre of mass; for a sphere with uniform field-aligned
   magnetization it does not:
   $\boldsymbol{\tau} = \int \mathbf{r}\times(\mathbf{M}\cdot\nabla)\mathbf{B}\,\mathrm{d}V
   = \tfrac{VR^2}{5}\,\varepsilon_{ijk}M_l\,\partial_l\partial_j B_k = 0$,
   because $\partial_jB_k$ is symmetric in $(j,k)$ for a curl-free field while
   $\varepsilon_{ijk}$ is antisymmetric in $(j,k)$. So the exact zero follows from
   Maxwell plus sphericity, not merely from the point-dipole idealisation — though
   it does assume uniform $\mathbf{M}$, i.e. the small-body assumption again.
2. **The small-body assumption is not checked at runtime.** Nothing warns you; the
   moment is simply wrong. Diverges as $|\mathbf{B}| \to 0$.
3. **No relaxation dynamics.** The Landers nanocubes are 18.4 nm, whose Néel time
   is nanoseconds against millisecond timesteps, so the quasi-static limit is safe
   *there* — but that is a property of the device, not of the node.
4. **A single Langevin is monodisperse and non-interacting.** A real polydisperse,
   dipolar-coupled loading has a broader knee. This surfaces as a ~6 %
   inconsistency in the Landers numbers themselves (below). Prefer
   `InterpolatedCurve` with real data.
5. **No temperature dependence.** A curve measured at room temperature is used
   unchanged at 37 °C (~5 % in $b_0$ via $T$ alone).
6. **`grad_B` is consumed as given** — not symmetrised, Maxwell consistency not
   enforced. A non-symmetric gradient silently yields a non-conservative force
   that does net work around a closed loop.
7. **Volume scaling is exact for a sphere, not a licence to change shape.**
   Rescaling a sphere-measured curve onto a rod or disc is wrong: $N$ depends on
   aspect ratio.

### The Landers data is internally inconsistent — by ~6 %

[@Landers2025] quotes three magnetic numbers: $m(30\,\text{mT}) = 72$ memu,
$m_\text{sat} \approx 225$ memu, and a low-field slope of 2.4 memu/mT. A Langevin
has **two** parameters, so three constraints over-determine it — and they are also
mutually inconsistent, provably:

$$
72\ \text{memu} \;=\; 2.4\ \text{memu/mT} \times 30\ \text{mT} \quad\textbf{exactly}
$$

so the quoted slope is the **chord** through the 30 mT point, not the tangent at
the origin. But $L$ is strictly **concave** on $x>0$, so for any finite $b_0$,
$m(30\,\text{mT}) < \text{slope} \times 30\,\text{mT} = 72$ memu, with 72 attained
only in the $b_0 \to \infty$ (pure-linear, $m_\text{sat}\to\infty$) limit. **No
Langevin with finite saturation fits both.**

| fit | $b_0$ [mT] | $m$(30 mT) | tangent [memu/mT] | $m_\text{sat}$ |
|-----|-----------|-----------|-------------------|----------------|
| pin ($m_\text{sat}$, chord slope) | 31.25 | 67.93 memu (**−5.7 %**) | 2.4 (exact) | 225 (exact) |
| pin ($m_\text{sat}$, 30 mT point) | 29.26 | 72.0 memu (exact) | 2.563 (**+6.8 %**) | 225 (exact) |

`LangevinCurve.landers_capsule()` returns **the second**: 30 mT is the operating
point, and 72 memu is a direct VSM reading whereas 2.4 memu/mT is a derived chord.
Use `from_saturation_and_slope()` for the other end of the trade. Both pinned by
MIME-VER-145.

### The Landers capsule size is ambiguous in the source — a 43 % force swing

The main text reports a mean diameter of **1.69 mm**; the paper's **own CFD**
models **1.4 mm** at 3187.74 kg/m³. The paper never reconciles them. The volume
ratio is $(1.4/1.69)^3 = 0.57$ — **a 43 % swing in the moment and therefore in the
force**, which dwarfs every other error here. This node takes no position:
`LANDERS_CAPSULE_DIAMETER_M` and `LANDERS_CFD_DIAMETER_M` are both exported,
deliberately without a default preference. Exact volume scaling makes them one
constructor argument apart — sweep both.

## Stability Conditions

None — the node is algebraic and stateless (its `update()` is a pure function of
its boundary inputs; state is output-only). It introduces no timestep restriction.

The *coupled* system does: the gradient force is a stiff, position-dependent load
on the downstream `RigidBodyNode`, and $\nabla|\mathbf{B}|^2$ is subharmonic
(Earnshaw, MIME-VER-143/149), so there is **no stable equilibrium** to relax
toward. Any apparent trap in a simulation is a numerical artefact or the work of
another effect (drag, contact, walls).

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `magnetization` | (3,) | A·m² | Field-aligned magnetic **moment** in the lab frame — see the units warning below |
| `magnetic_torque` | (3,) | N·m | Always exactly zero (structural) |
| `magnetic_force` | (3,) | N | Gradient-pull force, lab frame |

> **`magnetization` is a MOMENT in A·m², not a magnetization in A/m** (MIME-ANO-107).
> This matches `PermanentMagnetResponseNode` (MIME-NODE-008), the majority
> convention — but `MagneticResponseNode` (MIME-NODE-002) emits an output *of the
> same name* in **A/m**, multiplying by volume inside its own force/torque. For the
> Landers capsule the two differ by $1/V = 4.0\times10^8$. The inconsistency is
> pre-existing in MIME-NODE-002 and is not this node's to fix, but **anything
> reading `magnetization` generically across the response-node family — the effects
> layer above all — must not treat the field as interchangeable.** MIME-VER-148
> compensates explicitly (`m_legacy = norm(legacy["magnetization"]) * V`).

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `curve` | `MagnetizationCurve` | *required, keyword-only* | — | Constitutive law $\lvert B\rvert \to m$ plus its reference volume |
| `volume_m3` | float | `curve.reference_volume_m3` | m³ | This body's volume; rescales the curve exactly |
| `reference_volume_m3` | float | *derived* | m³ | Recorded from the curve |
| `volume_scale` | float | *derived* | — | $V/V_\text{ref}$ |
| `curve_kind` | str | *derived* | — | Provenance tag for `graph.json` |
| `curve_description` | str | *derived* | — | Human-readable curve summary |

Heavy curve data (e.g. an `InterpolatedCurve` table) lives **outside** `params`,
which feed `graph.json` serialisation — the same split `CoilArrayFieldNode` uses
for its calibration.

## Boundary Inputs

| Field | Shape | Default | Coupling Type | Description |
|-------|-------|---------|---------------|-------------|
| `field_vector` | (3,) | zeros | explicit | **APPLIED** B at the body centre [T], lab frame. Not demag-corrected |
| `field_gradient` | (3,3) | zeros | explicit | $\partial B_i/\partial x_j$ [T/m], jacrev convention |

**There is deliberately no `orientation` input.** Both sibling response nodes take
one; this node does not, because a field-aligned isotropic moment is *structurally*
orientation-independent — $\mathbf{m} = m(|\mathbf{B}|)\hat{\mathbf{B}}$ contains
no body-frame quantity, so there is no frame to rotate into and nothing for a
quaternion to do. Declaring it and ignoring it would advertise a dependence that
does not exist, and this repo has already been bitten by a dead boundary input.

This is safe against existing wiring: `_MagneticEffect.build()` unconditionally
adds a `body.orientation → response.orientation` edge, and
`GraphManager.validate()` skips edges whose target field is absent from the
target's `boundary_input_spec()` (`if e.target_field not in bi_spec: continue`).
The hardwired edge is therefore a silent no-op rather than an error — pinned by
`test_orientation_edge_is_a_harmless_no_op`.

## Boundary Fluxes (outputs)

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `magnetic_force` | (3,) | N | To `RigidBodyNode` on an **additive** edge |
| `magnetic_torque` | (3,) | N·m | To `RigidBodyNode` on an **additive** edge; always zero |

## MIME-Specific Sections

### Anatomical Operating Context

| Compartment | Flow Regime | Re Range | pH Range | Temp Range | Viscosity Range |
|-------------|------------|----------|----------|------------|----------------|
| blood | poiseuille | — | — | — | — |

Declared as context only: the node itself carries **no** `AnatomicalRegimeMeta`,
because its physics is purely magnetic and has no fluid, pH or temperature
coupling. The anatomical context belongs to the fluid and body nodes it is wired
to. The anchor experiment ([@Landers2025]) gradient-pulls the capsule against
blood flow in vasculature; **that fluid coupling is currently blocked on an
unrelated CFD problem**, so this node is first used in magnetics-only and
Stokes-regime experiments.

### Biocompatibility Context (robot_body nodes only)

Materials describe the [@Landers2025] capsule: **gelatin** matrix,
**zinc-substituted iron-oxide nanocubes** (18.4 nm edge), **tantalum** (radiopaque
marker). ISO 10993 class: `NOT_ASSESSED`.

This is a technical descriptor, **not a safety claim**. Specific hazards recorded
in metadata:

- The node's physics is **material-agnostic** — it takes whatever curve it is
  given. These materials describe the *default anchor device only* and are wrong
  for any other body; update them alongside the curve.
- Zinc-substituted iron oxide is **not** the same material as the magnetite whose
  biocompatibility literature is often cited by analogy. Zinc leaching is a
  distinct question and is not addressed by a magnetite assessment.
- The manufacturer must perform their own ISO 10993 evaluation.

### Clinical Relevance

Gradient pulling is the actuation mode for untethered, non-swimming therapeutic
carriers: a capsule too small to swim and with no easy axis to steer can still be
*dragged* along $\nabla|\mathbf{B}|$ toward a target and held against flow. The
[@Landers2025] device is the clinically-oriented instance — a radiopaque,
drug-loadable capsule navigated by a clinical eMNS. This node supplies the force
that competes with haemodynamic drag in that scenario, and the ability to sweep
body size exactly is what lets the 1.69-vs-1.4 mm ambiguity be *bounded* rather
than guessed.

Not a clinical claim — context for understanding the model's role.

### Mode 2 Independent Verification

| ID | Type | Claim |
|----|------|-------|
| MIME-VER-144 | analytical | Reproduces the measured Landers VSM datum (72 memu at 30 mT, $d=1.69$ mm) to 1e-9 relative; exact volume scaling to the 1.4 mm CFD variant to 1e-12; high-field asymptote $m \to m_\text{sat}(1 - b_0/B)$ |
| MIME-VER-145 | analytical | $L(x)$ against an **independent Lambert continued-fraction oracle** to 1e-12 over $x \in [10^{-8}, 10]$, spanning both branches and the crossover; low-field deviation follows $x^2/15$; the series branch is shown *load-bearing* (the direct form is 4e-8 wrong at $x=10^{-4}$); the ±6 % Landers tension asserted, not hidden |
| MIME-VER-146 | analytical | Force identity $\nabla\mathbf{B}\cdot\mathbf{m} = m(\lvert B\rvert)\nabla\lvert B\rvert$ to <1e-12 over 200 random symmetric-traceless gradients; **shown to fail** for an asymmetric gradient (the check has teeth); jacrev index convention pinned by hand computation; linear-regime textbook $\chi V\nabla\lvert B\rvert^2/2\mu_0$ cross-check |
| MIME-VER-147 | analytical | Torque is **exactly** (bitwise) zero for all inputs, and the emitted moment is field-parallel to <1e-14, so the zero is earned rather than covering a discrepancy |
| MIME-VER-148 | cross-code | Two-path consistency: intrinsic $\chi = 1.98$ through $\chi_\text{eff} = \chi/(1+N\chi)$ reproduces the measurement to 1e-6; `MagneticResponseNode` overshoots by 2.51×; detuning its $n$ factors is shown to invent a spurious torque |
| MIME-VER-149 | regression | Earnshaw consistency: $\nabla\cdot\mathbf{F} = k\lVert\nabla\mathbf{B}\rVert_F^2 \geq 0$ over 100 random Maxwell-consistent gradients — no stable trap |

Plus unit coverage of jit/vmap/grad traceability, NaN-freedom at $\mathbf{B}=0$
(value *and* gradient), dtype-independent fitting, and `GraphManager` coupling.

### Why the two existing response nodes do not fit

`MagneticResponseNode` (MIME-NODE-002) *is* a soft-magnetic node and *does* handle
demagnetization — but it hardwires the susceptibility to the demagnetization
**ceiling**, $\chi_\text{diag} = [1/n_\text{axi}, 1/n_\text{rad}, 1/n_\text{rad}]$,
i.e. the $\chi\to\infty$ limit of $\chi_\text{eff} = \chi/(1+N\chi)$. For a sphere
that pins $\chi_\text{eff} = 3$ and predicts $m = 3B/\mu_0 \cdot V = 1.81\times10^{-4}$
A·m² at 30 mT against the **measured** $7.2\times10^{-5}$ — **2.5× too high**. The
susceptibility cannot be lowered without violating $n_\text{axi} + 2n_\text{rad} = 1$,
and detuning the $n$ factors invents spurious anisotropy and hence spurious torque
on what is physically an isotropic bead (MIME-VER-148 asserts both halves).

`PermanentMagnetResponseNode` (MIME-NODE-008) assumes a body-fixed moment — the
wrong physics for a thermally-free superparamagnet.

For the record the two paths here *agree*: an intrinsic $\chi = 1.98$ gives
$\chi_\text{eff} = 1.19$ and reproduces the measured $7.2\times10^{-5}$ exactly — an
ordinary value for a dense iron-oxide loading, comfortably below the ceiling of 3.
The measurement is consistent with a real material; it is MIME-NODE-002's
*ceiling* assumption that is wrong here, not the data.

## References

- [@Landers2025] Landers et al. (2025). *Clinically ready magnetic microrobots for
  targeted therapies*, Science 390(6774). — **The anchor.** Source of the capsule
  geometry, the VSM curve (72 memu @ 30 mT, sat ~225 memu), and both the 1.69 mm
  main-text diameter and the irreconcilable 1.4 mm CFD diameter.
- [@Bean1959] Bean & Livingston (1959). *Superparamagnetism*, J. Appl. Phys.
  30(4). — The canonical Langevin form for an ensemble of thermally-free
  single-domain moments; basis for `LangevinCurve`, and the source of its
  monodisperse/non-interacting idealisation.
- [@Osborn1945] Osborn (1945). *Demagnetizing factors of the general ellipsoid*,
  Phys. Rev. 67. — $N = 1/3$ for a sphere, **independent of size**: what makes the
  volume scaling exact, and what bounds $\chi_\text{eff}$ by $1/N$.
- [@Abbott2020] Abbott, Diller & Petruska (2020). *Magnetic methods in robotics*,
  Annu. Rev. Control Robot. Auton. Syst. 3. — $\mathbf{F} = (\mathbf{m}\cdot\nabla)\mathbf{B}$
  for an induced moment, and the one-half that belongs in the energy but not the
  force.

## Verification Evidence

- Verification: `tests/verification/test_superparamagnetic_response.py`
  (MIME-VER-144 … MIME-VER-149)
- Unit: `tests/nodes/test_superparamagnetic_response.py`
- Related: `tests/verification/test_coil_array_field.py` (MIME-VER-140…143), the
  upstream field source; MIME-VER-143 establishes the subharmonicity of
  $\lvert B\rvert^2$ that MIME-VER-149 relies on.

Volume scaling — the node's headline feature and the mechanism carrying the 43 %
MIME-ANO-105 sweep — is pinned in MIME-VER-144 **through `update()`**, on the
emitted force and moment at both diameters, not only through the `moment_am2()`
helper. The two apply the scale via separate expressions, so the helper's coverage
does not protect the path the graph actually consumes.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.1 | 2026-07-15 | Review fixes. `InterpolatedCurve` now **rejects** tables not anchored at $(0,0)$ (`jnp.interp`'s low-end clamp silently manufactured remanence and a force discontinuity at a field null). `small_body_field_variation_ratio()` returns `inf`, not 0, at $\lvert B\rvert = 0$. Small-body ratio quantified at all three Navion gradients (1.4 / 2.8 / 5.6 %) rather than only 0.5 T/m; `small_body_ratio` regime ceiling 0.05 → 0.07 so the declared box is self-consistent with its own corners. Added volume-scaled `initial_susceptibility_am2_per_t()` to the node. Documented the `magnetization` unit clash with MIME-NODE-002 (MIME-ANO-107). Test gaps closed: volume scaling through `update()`, `m_sat_am2()` scaling, `diameter_m` sourcing, `DemagLimitedLinearCurve` saturation clamp. |
| 1.0.0 | 2026-07-15 | Initial implementation |
